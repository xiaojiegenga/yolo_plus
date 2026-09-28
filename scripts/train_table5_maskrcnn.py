"""Train the Table 5 Mask R-CNN baseline on data-v2 COCO annotations."""

from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as nn_functional
import yaml
from PIL import Image
from pycocotools import mask as mask_utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_Weights,
    maskrcnn_resnet50_fpn,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms.functional import pil_to_tensor


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "experiments/data-v2-table5-maskrcnn-r50-b4-s42.yaml"
METRIC_COLUMNS = (
    "epoch",
    "train_loss",
    "box_map50",
    "box_map50_95",
    "mask_map50",
    "mask_map50_95",
    "lr",
)


class RicePestCocoDataset(Dataset):
    """Load images resized to the training scale and rasterize each YOLO-derived COCO polygon."""

    def __init__(
        self,
        image_root: Path,
        annotation_file: Path,
        *,
        fliplr: float = 0.0,
        flipud: float = 0.0,
        image_size: int = 640,
    ) -> None:
        self.image_root = image_root
        self.coco = COCO(str(annotation_file))
        self.image_ids = sorted(self.coco.getImgIds())
        self.fliplr = fliplr
        self.flipud = flipud
        self.image_size = image_size
        self.annotations: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for annotation in self.coco.dataset["annotations"]:
            # COCOeval rewrites GT segmentations in place; keep private polygon copies.
            record = dict(annotation)
            record["segmentation"] = [list(polygon) for polygon in annotation["segmentation"]]
            self.annotations[record["image_id"]].append(record)

    def __len__(self) -> int:
        return len(self.image_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        image_id = self.image_ids[index]
        info = self.coco.loadImgs(image_id)[0]
        image_path = self.image_root / info["file_name"]
        with Image.open(image_path) as source:
            image = source.convert("RGB")
        orig_width, orig_height = image.size
        # Pin the longer side to image_size so the model's internal resize stays an identity.
        if orig_width >= orig_height:
            width = self.image_size
            height = max(1, round(orig_height * self.image_size / orig_width))
        else:
            height = self.image_size
            width = max(1, round(orig_width * self.image_size / orig_height))
        image = image.resize((width, height), Image.BILINEAR)
        scale_x = width / orig_width
        scale_y = height / orig_height

        boxes: list[list[float]] = []
        labels: list[int] = []
        areas: list[float] = []
        masks: list[torch.Tensor] = []
        for annotation in self.annotations[image_id]:
            polygon = annotation["segmentation"][0]
            scaled_polygon = [
                coordinate * (scale_x if position % 2 == 0 else scale_y)
                for position, coordinate in enumerate(polygon)
            ]
            rle = mask_utils.merge(mask_utils.frPyObjects([scaled_polygon], height, width))
            mask = torch.from_numpy(
                np.ascontiguousarray(mask_utils.decode(rle), dtype=np.uint8)
            )
            if not mask.any():
                continue
            x, y, box_width, box_height = annotation["bbox"]
            boxes.append(
                [
                    x * scale_x,
                    y * scale_y,
                    (x + box_width) * scale_x,
                    (y + box_height) * scale_y,
                ]
            )
            labels.append(annotation["category_id"])
            areas.append(annotation["area"])
            masks.append(mask)

        target: dict[str, torch.Tensor] = {
            "boxes": torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.tensor(labels, dtype=torch.int64),
            "masks": (
                torch.stack(masks) if masks else torch.zeros((0, height, width), dtype=torch.uint8)
            ),
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "orig_size": torch.tensor([orig_height, orig_width], dtype=torch.int64),
            "area": torch.tensor(areas, dtype=torch.float32),
            "iscrowd": torch.zeros(len(labels), dtype=torch.int64),
        }
        image_tensor = pil_to_tensor(image).float().div_(255)

        if random.random() < self.fliplr:
            image_tensor = image_tensor.flip(-1)
            target["masks"] = target["masks"].flip(-1)
            target["boxes"][:, [0, 2]] = width - target["boxes"][:, [2, 0]]
        if random.random() < self.flipud:
            image_tensor = image_tensor.flip(-2)
            target["masks"] = target["masks"].flip(-2)
            target["boxes"][:, [1, 3]] = height - target["boxes"][:, [3, 1]]
        return image_tensor, target


def collate_batch(batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]):
    return tuple(zip(*batch))


def build_model(image_size: int, pretrained: str) -> torch.nn.Module:
    weights_name = pretrained.removeprefix("MaskRCNN_ResNet50_FPN_Weights.")
    weights = MaskRCNN_ResNet50_FPN_Weights[weights_name]
    model = maskrcnn_resnet50_fpn(
        weights=weights,
        min_size=image_size,
        max_size=image_size,
    )
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, 3)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, 3)
    return model


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    coco_ground_truth: COCO,
    device: torch.device,
    use_amp: bool,
) -> dict[str, float]:
    model.eval()
    box_results: list[dict[str, Any]] = []
    mask_results: list[dict[str, Any]] = []
    with torch.inference_mode():
        for images, targets in loader:
            with torch.autocast("cuda", enabled=use_amp):
                outputs = model([image.to(device) for image in images])
            for image, target, output in zip(images, targets, outputs):
                image_id = int(target["image_id"])
                orig_height, orig_width = target["orig_size"].tolist()
                scale_x = orig_width / image.shape[-1]
                scale_y = orig_height / image.shape[-2]
                scores = output["scores"].cpu()
                labels = output["labels"].cpu()
                boxes = output["boxes"].cpu()
                boxes[:, [0, 2]] *= scale_x
                boxes[:, [1, 3]] *= scale_y
                if len(output["masks"]):
                    # Map masks back to the original image scale before COCO submission.
                    masks = nn_functional.interpolate(
                        output["masks"].float(),
                        size=(orig_height, orig_width),
                        mode="bilinear",
                        align_corners=False,
                    )
                    masks = (masks[:, 0] >= 0.5).to(torch.uint8).cpu()
                else:
                    masks = output["masks"].cpu()
                for prediction_index in range(min(len(scores), 100)):
                    score = float(scores[prediction_index])
                    if score < 0.001:
                        break
                    x0, y0, x1, y1 = boxes[prediction_index].tolist()
                    common = {
                        "image_id": image_id,
                        "category_id": int(labels[prediction_index]),
                        "score": score,
                    }
                    box_results.append(
                        {**common, "bbox": [x0, y0, x1 - x0, y1 - y0]}
                    )
                    binary_mask = np.asfortranarray(masks[prediction_index].numpy())
                    segmentation = mask_utils.encode(binary_mask)
                    segmentation["counts"] = segmentation["counts"].decode("ascii")
                    mask_results.append({**common, "segmentation": segmentation})

    metrics: dict[str, float] = {}
    for kind, predictions in (("box", box_results), ("mask", mask_results)):
        if not predictions:
            metrics[f"{kind}_map50"] = 0.0
            metrics[f"{kind}_map50_95"] = 0.0
            continue
        detections = coco_ground_truth.loadRes(predictions)
        evaluator = COCOeval(coco_ground_truth, detections, "bbox" if kind == "box" else "segm")
        evaluator.params.imgIds = loader.dataset.image_ids
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        metrics[f"{kind}_map50_95"] = float(evaluator.stats[0])
        metrics[f"{kind}_map50"] = float(evaluator.stats[1])
    return metrics


def train_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    accumulate: int,
    use_amp: bool,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_total = 0.0
    for batch_index, (images, targets) in enumerate(loader, start=1):
        images = [image.to(device) for image in images]
        targets = [
            {key: value.to(device) for key, value in target.items()}
            for target in targets
        ]
        with torch.autocast("cuda", enabled=use_amp):
            losses = model(images, targets)
            loss = sum(losses.values())
        group_start = ((batch_index - 1) // accumulate) * accumulate
        group_size = min(accumulate, len(loader) - group_start)
        scaler.scale(loss / group_size).backward()
        if batch_index % accumulate == 0 or batch_index == len(loader):
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        loss_total += float(loss.detach())
    return loss_total / len(loader)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-name")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Validate data/model and cache COCO weights on CPU without starting training.",
    )
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    train_config = config["train"]
    if config["model"] != "maskrcnn_resnet50_fpn":
        raise ValueError("This entry point requires maskrcnn_resnet50_fpn")

    data_root = Path(config["data_root"])
    annotation_root = REPO_ROOT / config["annotations"]
    train_dataset = RicePestCocoDataset(
        data_root / "images",
        annotation_root / "train.json",
        fliplr=float(train_config["fliplr"]),
        flipud=float(train_config["flipud"]),
        image_size=int(train_config["imgsz"]),
    )
    val_dataset = RicePestCocoDataset(
        data_root / "images",
        annotation_root / "val.json",
        image_size=int(train_config["imgsz"]),
    )
    if len(train_dataset) != 938 or len(val_dataset) != 117:
        raise ValueError("Expected data-v2 train/val image counts: 938/117")
    torch.manual_seed(int(train_config["seed"]))
    random.seed(int(train_config["seed"]))
    np.random.seed(int(train_config["seed"]))
    model = build_model(int(train_config["imgsz"]), config["pretrained"])

    if args.prepare_only:
        train_image, train_target = train_dataset[0]
        val_image, _ = val_dataset[0]
        model.roi_heads.detections_per_img = 10
        model.eval()
        with torch.inference_mode():
            model([val_image])
        print(f"[READY] {len(train_dataset)} train / {len(val_dataset)} val images")
        print(f"[READY] Sample: {tuple(train_image.shape)}, {len(train_target['labels'])} instances")
        print("[READY] Mask R-CNN pretrained weights and CPU inference")
        print("[NO GPU TRAINING STARTED]")
        return

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run --prepare-only in no-card mode")
    run_name = args.run_name or config["experiment"]
    run_dir = REPO_ROOT / train_config["project"] / run_name
    if run_dir.exists():
        raise FileExistsError(f"Run already exists: {run_dir}")
    run_dir.mkdir(parents=True)
    (run_dir / "weights").mkdir()
    (run_dir / "args.yaml").write_text(
        yaml.safe_dump({**config, "run_name": run_name}, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    batch_size = int(train_config["batch"])
    workers = int(train_config["workers"])
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=workers,
        collate_fn=collate_batch,
    )
    device = torch.device("cuda:0")
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_config["lr0"]),
        weight_decay=float(train_config["weight_decay"]),
    )
    epochs = int(train_config["epochs"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=float(train_config["lr0"]) * float(train_config["lrf"]),
    )
    use_amp = bool(train_config["amp"])
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    results_path = run_dir / "results.csv"
    best_score = -1.0
    best_epoch = 0
    print(f"[START] {run_dir} | Mask R-CNN R50-FPN | {epochs} epochs")
    with results_path.open("w", newline="", encoding="utf-8") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=METRIC_COLUMNS)
        writer.writeheader()
        for epoch in range(1, epochs + 1):
            loss = train_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                device,
                int(train_config["accumulate"]),
                use_amp,
            )
            metrics = evaluate(model, val_loader, val_dataset.coco, device, use_amp)
            row = {
                "epoch": epoch,
                "train_loss": loss,
                **metrics,
                "lr": optimizer.param_groups[0]["lr"],
            }
            writer.writerow(row)
            results_file.flush()
            score = metrics["mask_map50_95"]
            if score > best_score:
                best_score = score
                best_epoch = epoch
                torch.save(model.state_dict(), run_dir / "weights/best.pt")
            torch.save(model.state_dict(), run_dir / "weights/last.pt")
            print(
                f"epoch {epoch}/{epochs} loss={loss:.4f} "
                f"mask_mAP50={metrics['mask_map50']:.4f} "
                f"mask_mAP50-95={score:.4f} best={best_score:.4f}@{best_epoch}",
                flush=True,
            )
            scheduler.step()
            if epoch - best_epoch >= int(train_config["patience"]):
                print(f"[EARLY STOP] No Mask mAP50-95 improvement for {train_config['patience']} epochs")
                break
    print(f"[DONE] {run_dir}")
    print(f"python scripts/transfer_run.py pack --run-id {run_name}")


if __name__ == "__main__":
    main()
