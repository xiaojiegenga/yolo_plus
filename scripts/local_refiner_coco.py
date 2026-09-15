"""COCO conversion and metrics for original-resolution local-refiner evaluation."""

from __future__ import annotations

from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def resolve_data_paths(data_yaml: Path, split: str = "val") -> tuple[list[Path], Path, list[str]]:
    """Resolve a directory-based YOLO split and its matching label directory."""
    with data_yaml.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, dict):
        raise ValueError(f"data YAML must contain a mapping: {data_yaml}")

    root = Path(data.get("path") or data_yaml.parent)
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    split_value = data.get(split)
    if not isinstance(split_value, str):
        raise ValueError(f"data YAML {split!r} must be a directory path")
    image_dir = Path(split_value)
    if not image_dir.is_absolute():
        image_dir = root / image_dir
    image_dir = image_dir.resolve()
    if not image_dir.is_dir():
        raise FileNotFoundError(f"image directory not found: {image_dir}")

    try:
        images_index = image_dir.parts.index("images")
    except ValueError as error:
        raise ValueError(f"Val path must contain an 'images' component: {image_dir}") from error
    label_parts = list(image_dir.parts)
    label_parts[images_index] = "labels"
    label_dir = Path(*label_parts)
    if not label_dir.is_dir():
        raise FileNotFoundError(f"label directory not found: {label_dir}")

    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        raise FileNotFoundError(f"no images found in: {image_dir}")
    names_value = data.get("names")
    if isinstance(names_value, dict):
        names = [str(names_value[index]) for index in sorted(names_value)]
    elif isinstance(names_value, list):
        names = [str(name) for name in names_value]
    else:
        raise ValueError("data YAML must provide class names")
    return images, label_dir, names


def build_ground_truth(
    images: list[Path],
    label_dir: Path,
    names: list[str],
    imgsz: int,
) -> tuple[dict[str, Any], dict[str, int], dict[int, list[list[float]]], dict[int, float]]:
    """Build a COCO ground-truth mapping from YOLO polygon labels."""
    import numpy as np
    from PIL import Image

    ground_truth: dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": [{"id": index + 1, "name": name} for index, name in enumerate(names)],
    }
    path_to_id: dict[str, int] = {}
    boxes_by_image: dict[int, list[list[float]]] = defaultdict(list)
    resize_gains: dict[int, float] = {}
    for image_id, image_path in enumerate(images, start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        path_to_id[str(image_path.resolve())] = image_id
        resize_gains[image_id] = imgsz / max(width, height)
        ground_truth["images"].append(
            {"id": image_id, "file_name": image_path.name, "width": width, "height": height}
        )
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            values = [float(value) for value in line.split()]
            if len(values) < 7 or (len(values) - 1) % 2:
                raise ValueError(f"invalid polygon label in {label_path}: {line}")
            category_id = int(values[0]) + 1
            polygon = np.asarray(values[1:], dtype=np.float64).reshape(-1, 2) * [width, height]
            lower = polygon.min(axis=0)
            upper = polygon.max(axis=0)
            box_width, box_height = upper - lower
            bbox = [float(lower[0]), float(lower[1]), float(box_width), float(box_height)]
            boxes_by_image[image_id].append(
                [float(lower[0]), float(lower[1]), float(upper[0]), float(upper[1])]
            )
            ground_truth["annotations"].append(
                {
                    "id": len(ground_truth["annotations"]) + 1,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "segmentation": [polygon.reshape(-1).tolist()],
                    "area": float(box_width * box_height * resize_gains[image_id] ** 2),
                    "iscrowd": 0,
                }
            )
    return ground_truth, path_to_id, boxes_by_image, resize_gains


def _mean_valid(values: Any) -> float:
    """Average COCO metric entries while excluding its -1 sentinel values."""
    selected = values[values >= 0]
    return float(selected.mean()) if selected.size else 0.0


def evaluate_coco(
    ground_truth: dict[str, Any],
    predictions: list[dict[str, Any]],
    resize_gains: dict[int, float],
    kind: str,
) -> dict[str, Any]:
    """Evaluate overall and per-class AP with the established original-image protocol."""
    import numpy as np
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco = COCO()
    coco.dataset = deepcopy(ground_truth)
    coco.createIndex()
    detections = coco.loadRes(deepcopy(predictions))
    for annotation in detections.anns.values():
        width, height = annotation["bbox"][2:]
        annotation["area"] = width * height * resize_gains[annotation["image_id"]] ** 2

    evaluator = COCOeval(coco, detections, kind)
    evaluator.params.maxDets = [1, 10, 300]
    evaluator.params.areaRng = [
        [0, 1e10],
        [0, np.nextafter(1024.0, 0.0)],
        [1024.0, 1e10],
    ]
    evaluator.params.areaRngLbl = ["all", "small", "non_small"]
    evaluator.evaluate()
    evaluator.accumulate()

    precision = evaluator.eval["precision"]
    output: dict[str, Any] = {"overall": {}, "per_class": {}}
    for area_index, area_name in enumerate(evaluator.params.areaRngLbl):
        area_precision = precision[:, :, :, area_index, -1]
        output["overall"][area_name] = {
            "AP50": _mean_valid(area_precision[0]),
            "AP75": _mean_valid(area_precision[5]),
            "AP50_95": _mean_valid(area_precision),
        }
    for class_offset, category_id in enumerate(evaluator.params.catIds):
        output["per_class"][str(category_id - 1)] = {}
        for area_index, area_name in enumerate(evaluator.params.areaRngLbl):
            class_precision = precision[:, :, class_offset, area_index, -1]
            output["per_class"][str(category_id - 1)][area_name] = {
                "AP50": _mean_valid(class_precision[0]),
                "AP75": _mean_valid(class_precision[5]),
                "AP50_95": _mean_valid(class_precision),
            }
    return output


def low_overlap_diagnostic(
    predictions: list[dict[str, Any]],
    boxes_by_image: dict[int, list[list[float]]],
    confidence: float = 0.5,
    overlap_iou: float = 0.1,
) -> dict[str, Any]:
    """Measure high-confidence prediction boxes having little overlap with every GT box."""
    import numpy as np

    selected = [prediction for prediction in predictions if prediction["score"] >= confidence]
    low_overlap = 0
    for prediction in selected:
        x, y, width, height = prediction["bbox"]
        candidate = np.asarray([x, y, x + width, y + height], dtype=np.float64)
        targets = np.asarray(boxes_by_image.get(prediction["image_id"], []), dtype=np.float64)
        if not targets.size:
            low_overlap += 1
            continue
        upper_left = np.maximum(candidate[:2], targets[:, :2])
        lower_right = np.minimum(candidate[2:], targets[:, 2:])
        intersection = np.clip(lower_right - upper_left, 0, None).prod(axis=1)
        candidate_area = np.prod(candidate[2:] - candidate[:2])
        target_area = np.prod(targets[:, 2:] - targets[:, :2], axis=1)
        iou = intersection / np.maximum(candidate_area + target_area - intersection, 1e-7)
        low_overlap += int(iou.max() < overlap_iou)
    total = len(selected)
    return {
        "confidence": confidence,
        "overlap_iou": overlap_iou,
        "low_overlap": low_overlap,
        "total_high_confidence": total,
        "rate": low_overlap / total if total else 0.0,
    }
