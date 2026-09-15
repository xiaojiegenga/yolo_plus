# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Candidate export, validation, and prediction for local crop refinement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch import nn

from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.models.yolo.segment.refine import (
    apply_local_refinement,
    assign_candidate_labels,
    crop_and_pad_candidates,
    low_overlap_high_confidence_counts,
)
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.utils import LOGGER
from ultralytics.utils.metrics import box_iou


class CandidateCropValidator(SegmentationValidator):
    """Generate labeled Train-split crops from frozen segmentation predictions."""

    def __init__(
        self,
        *args: Any,
        output_dir: str | Path,
        crop_size: int = 96,
        positive_iou: float = 0.5,
        background_iou: float = 0.1,
        background_class: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        if not 0 <= background_iou < positive_iou <= 1:
            raise ValueError("IoU thresholds must satisfy 0 <= background_iou < positive_iou <= 1")
        self.output_dir = Path(output_dir)
        self.crop_size = crop_size
        self.positive_iou = positive_iou
        self.background_iou = background_iou
        self.background_class = background_class
        self.records: list[dict[str, Any]] = []
        self.class_counts = [0] * (background_class + 1)
        self.ignored_count = 0

    def init_metrics(self, model: nn.Module) -> None:
        """Initialize postprocessing state and create empty class directories."""
        super().init_metrics(model)
        if self.output_dir.exists():
            raise FileExistsError(f"candidate output already exists: {self.output_dir}")
        for class_index in range(self.background_class + 1):
            (self.output_dir / "crops" / str(class_index)).mkdir(parents=True, exist_ok=False)
        self.records.clear()
        self.class_counts = [0] * (self.background_class + 1)
        self.ignored_count = 0

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        """Assign candidate labels and save non-ignored crops."""
        for image_index, prediction in enumerate(preds):
            prepared = self._prepare_batch(image_index, batch)
            labels = assign_candidate_labels(
                prediction["bboxes"],
                prediction["cls"],
                prepared["bboxes"],
                prepared["cls"],
                positive_iou=self.positive_iou,
                background_iou=self.background_iou,
                background_class=self.background_class,
            )
            self.ignored_count += int((labels < 0).sum())
            selected_indices = torch.where(labels >= 0)[0]
            if not selected_indices.shape[0]:
                continue

            boxes = prediction["bboxes"].index_select(0, selected_indices)
            image_indices = torch.full(
                (boxes.shape[0],),
                image_index,
                dtype=torch.long,
                device=boxes.device,
            )
            crops = crop_and_pad_candidates(
                batch["img"],
                boxes,
                image_indices,
                output_size=self.crop_size,
            )
            if prepared["bboxes"].shape[0]:
                max_ious = box_iou(boxes, prepared["bboxes"]).max(dim=1).values
            else:
                max_ious = torch.zeros(boxes.shape[0], device=boxes.device)

            for crop, selected_index, max_iou in zip(crops, selected_indices, max_ious):
                label = int(labels[selected_index])
                record_index = len(self.records)
                relative_path = Path("crops") / str(label) / f"{record_index:07d}.png"
                pixels = (
                    crop.detach()
                    .float()
                    .clamp_(0, 1)
                    .mul_(255)
                    .round_()
                    .byte()
                    .permute(1, 2, 0)
                    .cpu()
                    .numpy()
                )
                Image.fromarray(pixels).save(self.output_dir / relative_path, format="PNG")
                record = {
                    "path": relative_path.as_posix(),
                    "label": label,
                    "source_image": str(prepared["im_file"]),
                    "predicted_class": int(prediction["cls"][selected_index]),
                    "base_score": float(prediction["conf"][selected_index]),
                    "bbox_xyxy": [float(value) for value in prediction["bboxes"][selected_index]],
                    "max_gt_iou": float(max_iou),
                    "candidate_index": int(selected_index),
                }
                self.records.append(record)
                self.class_counts[label] += 1

    def gather_stats(self) -> None:
        """Candidate generation is intentionally single-process."""

    def get_stats(self) -> dict[str, int]:
        """Return candidate counts for the validation loop."""
        return {
            **{f"candidates/class_{index}": count for index, count in enumerate(self.class_counts)},
            "candidates/ignored": self.ignored_count,
        }

    def finalize_metrics(self) -> None:
        """Write the candidate manifest and summary after generation completes."""
        manifest_path = self.output_dir / "manifest.jsonl"
        with manifest_path.open("w", encoding="utf-8", newline="\n") as file:
            for record in self.records:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")
        summary = {
            "class_counts": {str(index): count for index, count in enumerate(self.class_counts)},
            "ignored": self.ignored_count,
            "saved": len(self.records),
            "crop_size": self.crop_size,
            "positive_iou": self.positive_iou,
            "background_iou": self.background_iou,
        }
        with (self.output_dir / "summary.json").open("w", encoding="utf-8") as file:
            json.dump(summary, file, ensure_ascii=False, indent=2)

    def print_results(self) -> None:
        """Log candidate counts without producing detection metrics."""
        LOGGER.info(
            "Candidate crops saved: "
            + ", ".join(f"class {index}={count}" for index, count in enumerate(self.class_counts))
            + f", ignored={self.ignored_count}"
        )


class LocalRefinementValidator(SegmentationValidator):
    """Apply a local crop refiner before accumulating standard segmentation metrics."""

    def __init__(
        self,
        *args: Any,
        refiner: nn.Module,
        crop_size: int = 96,
        score_batch_size: int = 128,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.refiner = refiner
        self.crop_size = crop_size
        self.score_batch_size = score_batch_size
        self.low_overlap_high_confidence = 0
        self.total_high_confidence = 0
        self._batch_images: torch.Tensor | None = None

    def preprocess(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Retain the normalized detector input used to crop refinement candidates."""
        batch = super().preprocess(batch)
        self._batch_images = batch["img"]
        return batch

    def init_metrics(self, model: nn.Module) -> None:
        """Initialize standard metrics and place the refiner on the validation device."""
        super().init_metrics(model)
        dtype = torch.float16 if self.args.quantize == 16 else torch.float32
        self.refiner.to(device=self.device, dtype=dtype).eval()
        self.low_overlap_high_confidence = 0
        self.total_high_confidence = 0

    def postprocess(self, preds: list[torch.Tensor]) -> list[dict[str, torch.Tensor]]:
        """Run standard segmentation postprocessing, then update confidence only."""
        detections = super().postprocess(preds)
        if self._batch_images is None:
            raise RuntimeError("preprocess must run before postprocess")
        return apply_local_refinement(
            self._batch_images,
            detections,
            self.refiner,
            crop_size=self.crop_size,
            batch_size=self.score_batch_size,
        )

    def update_metrics(self, preds: list[dict[str, torch.Tensor]], batch: dict[str, Any]) -> None:
        """Record the H mechanism diagnostic before standard metric accumulation."""
        low_overlap, high_confidence = low_overlap_high_confidence_counts(preds, batch, self)
        self.low_overlap_high_confidence += low_overlap
        self.total_high_confidence += high_confidence
        super().update_metrics(preds, batch)

    @property
    def low_overlap_high_confidence_rate(self) -> float:
        """Return the fraction of high-confidence predictions having IoU below 0.1 to all GT boxes."""
        if not self.total_high_confidence:
            return 0.0
        return self.low_overlap_high_confidence / self.total_high_confidence


class LocalRefinementPredictor(SegmentationPredictor):
    """Use the same crop score refinement in independent prediction."""

    def __init__(
        self,
        *args: Any,
        refiner: nn.Module,
        crop_size: int = 96,
        score_batch_size: int = 128,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.refiner = refiner
        self.crop_size = crop_size
        self.score_batch_size = score_batch_size

    def setup_model(self, *args: Any, **kwargs: Any) -> None:
        """Set up the base predictor and move the refiner to its device and precision."""
        super().setup_model(*args, **kwargs)
        dtype = torch.float16 if self.model.fp16 else torch.float32
        self.refiner.to(device=self.device, dtype=dtype).eval()

    def construct_results(
        self,
        preds: list[torch.Tensor],
        img: torch.Tensor,
        orig_imgs: list[Any],
        protos: torch.Tensor,
    ) -> list[Any]:
        """Refine NMS scores before boxes and masks are mapped back to original images."""
        detections = [
            {"bboxes": pred[:, :4], "conf": pred[:, 4], "cls": pred[:, 5]}
            for pred in preds
        ]
        refined = apply_local_refinement(
            img,
            detections,
            self.refiner,
            crop_size=self.crop_size,
            batch_size=self.score_batch_size,
        )
        refined_preds = []
        for pred, detection in zip(preds, refined):
            updated = pred.clone()
            updated[:, 4] = detection["conf"]
            refined_preds.append(updated)
        return super().construct_results(refined_preds, img, orig_imgs, protos)
