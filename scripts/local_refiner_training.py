"""Training and validation helpers for the H local crop classifier."""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from local_refiner_config import write_json


class CandidateCropDataset(Dataset[tuple[torch.Tensor, int]]):
    """Load generated 96x96 crops with only horizontal and vertical flips."""

    def __init__(self, candidate_dir: Path, flipud: float, fliplr: float) -> None:
        manifest_path = candidate_dir / "manifest.jsonl"
        self.candidate_dir = candidate_dir
        self.records = [
            json.loads(line)
            for line in manifest_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        self.flipud = flipud
        self.fliplr = fliplr

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self.records[index]
        with Image.open(self.candidate_dir / record["path"]) as image:
            pixels = np.asarray(image.convert("RGB"), dtype=np.float32).copy()
        crop = torch.from_numpy(pixels).permute(2, 0, 1).div_(255)
        if torch.rand(()) < self.flipud:
            crop = crop.flip(1)
        if torch.rand(()) < self.fliplr:
            crop = crop.flip(2)
        return crop, int(record["label"])


def balanced_epoch_indices(records: list[dict[str, Any]], seed: int, epoch: int) -> list[int]:
    """Return a shuffled epoch list with exactly equal counts for classes 0, 1, and 2."""
    by_class: dict[int, list[int]] = defaultdict(list)
    for index, record in enumerate(records):
        by_class[int(record["label"])].append(index)
    missing = [class_index for class_index in range(3) if not by_class[class_index]]
    if missing:
        raise RuntimeError(f"cannot build 1:1:1 sampling; empty classes: {missing}")

    random_source = random.Random(seed + epoch)
    target_count = max(len(by_class[class_index]) for class_index in range(3))
    indices: list[int] = []
    for class_index in range(3):
        source = by_class[class_index]
        if len(source) == target_count:
            selected = source.copy()
            random_source.shuffle(selected)
        else:
            selected = [random_source.choice(source) for _ in range(target_count)]
        indices.extend(selected)
    random_source.shuffle(indices)
    return indices


def validation_args(runtime: dict[str, Any], split: str, conf: float, plots: bool = False) -> dict[str, Any]:
    """Build the fixed validation protocol shared by calibration and refined Val."""
    train = runtime["train"]
    return {
        "task": "segment",
        "mode": "val",
        "model": str(runtime["model"]),
        "data": str(runtime["data"]),
        "split": split,
        "imgsz": int(train["imgsz"]),
        "batch": int(train.get("val_batch", 16)),
        "device": str(train["device"]),
        "workers": int(train.get("workers", 8)),
        "conf": conf,
        "iou": float(train.get("val_iou", 0.7)),
        "max_det": int(train.get("max_det", 300)),
        "rect": True,
        "plots": plots,
        "save_json": False,
        "save_txt": False,
        "quantize": None,
        "verbose": True,
    }


def metric_snapshot(metrics: Any) -> dict[str, Any]:
    """Capture overall, per-class, and AP75 segmentation metrics."""
    return {
        "results": dict(metrics.results_dict),
        "mask_ap75": float(metrics.seg.map75),
        "mask_per_class_ap50_95": [float(value) for value in metrics.seg.maps],
        "box_ap75": float(metrics.box.map75),
        "box_per_class_ap50_95": [float(value) for value in metrics.box.maps],
    }


def run_base_calibration(runtime: dict[str, Any], base_model: nn.Module) -> dict[str, Any]:
    """Reproduce the fixed D1 Val metric before generating candidates."""
    from ultralytics.models.yolo.segment import SegmentationValidator

    validator = SegmentationValidator(
        save_dir=runtime["run_dir"] / "base_calibration",
        args=validation_args(runtime, split="val", conf=0.001),
    )
    validator(model=base_model)
    snapshot = metric_snapshot(validator.metrics)
    expected = float(runtime["refiner"].get("d1_mask_map_reference", 0.37327))
    tolerance = float(runtime["refiner"].get("d1_calibration_tolerance", 0.002))
    actual = float(snapshot["results"]["metrics/mAP50-95(M)"])
    snapshot["expected_mask_map"] = expected
    snapshot["absolute_error"] = abs(actual - expected)
    snapshot["tolerance"] = tolerance
    write_json(runtime["run_dir"] / "base_calibration.json", snapshot)
    if abs(actual - expected) > tolerance:
        raise RuntimeError(
            f"D1 calibration failed: expected {expected:.5f} +/- {tolerance:.5f}, got {actual:.5f}"
        )
    return snapshot


def generate_candidates(runtime: dict[str, Any], base_model: nn.Module) -> dict[str, Any]:
    """Generate and label frozen D1 candidates from Train images only."""
    from ultralytics.models.yolo.segment import CandidateCropValidator

    refiner_config = runtime["refiner"]
    validator = CandidateCropValidator(
        save_dir=runtime["run_dir"] / "candidate_generation",
        output_dir=runtime["run_dir"] / "candidates",
        crop_size=int(refiner_config["crop_size"]),
        positive_iou=float(refiner_config["positive_iou"]),
        background_iou=float(refiner_config["background_iou"]),
        background_class=2,
        args=validation_args(
            runtime,
            split="train",
            conf=float(refiner_config["candidate_conf"]),
        ),
    )
    stats = validator(model=base_model)
    if any(count == 0 for count in validator.class_counts):
        raise RuntimeError(f"candidate generation produced an empty class: {validator.class_counts}")
    return {"stats": stats, "class_counts": validator.class_counts, "ignored": validator.ignored_count}


def train_one_epoch(
    refiner: nn.Module,
    dataset: CandidateCropDataset,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    batch_size: int,
    workers: int,
    seed: int,
    epoch: int,
    amp_enabled: bool,
) -> tuple[float, float, int]:
    """Train one class-balanced refiner epoch."""
    from ultralytics.data.build import seed_worker
    from ultralytics.utils.torch_utils import autocast

    indices = balanced_epoch_indices(dataset.records, seed, epoch)
    generator = torch.Generator()
    generator.manual_seed(seed + epoch)
    loader = DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )
    refiner.train()
    loss_sum = 0.0
    correct = 0
    seen = 0
    for crops, labels in loader:
        crops = crops.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast(amp_enabled, device=device.type):
            logits = refiner(crops)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        batch_count = labels.shape[0]
        loss_sum += float(loss.detach()) * batch_count
        correct += int((logits.argmax(dim=1) == labels).sum())
        seen += batch_count
    return loss_sum / seen, correct / seen, seen


def validate_refiner(
    runtime: dict[str, Any],
    base_model: nn.Module,
    refiner: nn.Module,
    save_dir: Path,
    plots: bool = False,
) -> tuple[dict[str, Any], float]:
    """Run the shared refined validator and return metrics plus the H diagnostic."""
    from ultralytics.models.yolo.segment import LocalRefinementValidator

    refiner_config = runtime["refiner"]
    validator = LocalRefinementValidator(
        save_dir=save_dir,
        refiner=refiner,
        crop_size=int(refiner_config["crop_size"]),
        score_batch_size=int(refiner_config["score_batch_size"]),
        args=validation_args(runtime, split="val", conf=float(runtime["train"].get("val_conf", 0.001)), plots=plots),
    )
    validator(model=base_model)
    return metric_snapshot(validator.metrics), validator.low_overlap_high_confidence_rate


def append_results(path: Path, record: dict[str, Any]) -> None:
    """Append a stable epoch row to results.csv."""
    write_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(record))
        if write_header:
            writer.writeheader()
        writer.writerow(record)


def checkpoint_metadata(
    runtime: dict[str, Any],
    epoch: int,
    train_loss: float,
    train_accuracy: float,
    metrics: dict[str, Any],
    diagnostic_rate: float,
) -> dict[str, Any]:
    """Build metadata stored with both the base model and refiner."""
    return {
        "method": "D1 + local candidate classification refinement",
        "run_id": runtime["run_name"],
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "metrics": metrics,
        "low_overlap_high_confidence_rate": diagnostic_rate,
        "train_args": {
            "task": "segment",
            "model": str(runtime["model"]),
            "data": str(runtime["data"]),
            "imgsz": int(runtime["train"]["imgsz"]),
        },
        "refiner_config": runtime["refiner"],
    }


def save_epoch_checkpoint(
    path: Path,
    base_model: nn.Module,
    refiner: nn.Module,
    metadata: dict[str, Any],
) -> None:
    """Save a composite H checkpoint."""
    from ultralytics.models.yolo.segment import save_local_refinement_checkpoint

    path.parent.mkdir(parents=True, exist_ok=True)
    save_local_refinement_checkpoint(path, base_model, refiner, metadata)
