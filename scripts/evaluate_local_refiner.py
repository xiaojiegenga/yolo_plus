"""Evaluate a local-refinement checkpoint on Val at original image resolution."""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any

from local_refiner_coco import (
    build_ground_truth,
    evaluate_coco,
    low_overlap_diagnostic,
    resolve_data_paths,
)
from local_refiner_config import SOURCE_ROOT


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--score-batch-size", type=int, default=128)
    return parser.parse_args()


def _predict(
    checkpoint: Path,
    images: list[Path],
    path_to_id: dict[str, int],
    device: str,
    imgsz: int,
    crop_size: int,
    score_batch_size: int,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Run original-size predictions through the shared refinement predictor."""
    import numpy as np
    from pycocotools import mask as mask_utils
    from ultralytics import YOLO
    from ultralytics.models.yolo.segment import LocalRefinementPredictor, load_local_refinement_checkpoint

    _, refiner, _ = load_local_refinement_checkpoint(checkpoint)
    model = YOLO(checkpoint)
    predictor = partial(
        LocalRefinementPredictor,
        refiner=refiner,
        crop_size=crop_size,
        score_batch_size=score_batch_size,
    )
    predictions: list[dict[str, Any]] = []
    latencies: list[float] = []
    results = model.predict(
        source=[str(path) for path in images],
        stream=True,
        predictor=predictor,
        imgsz=imgsz,
        rect=False,
        batch=1,
        conf=0.001,
        iou=0.7,
        max_det=300,
        retina_masks=True,
        quantize=None,
        device=device,
        save=False,
        verbose=False,
    )
    for result in results:
        image_id = path_to_id[str(Path(result.path).resolve())]
        latencies.append(float(sum(result.speed.values())))
        masks = result.masks.data.cpu().numpy() if result.masks is not None else []
        for xyxy, score, class_index, mask in zip(
            result.boxes.xyxy.cpu().numpy(),
            result.boxes.conf.cpu().numpy(),
            result.boxes.cls.cpu().numpy(),
            masks,
        ):
            x1, y1, x2, y2 = (float(value) for value in xyxy)
            encoded = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
            encoded["counts"] = encoded["counts"].decode("ascii")
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": int(class_index) + 1,
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(score),
                    "segmentation": encoded,
                }
            )
    return predictions, latencies


def evaluate_original_images(
    checkpoint: str | Path,
    data_yaml: str | Path,
    output_dir: str | Path,
    device: str = "0",
    imgsz: int = 640,
    crop_size: int = 96,
    score_batch_size: int = 128,
) -> dict[str, Any]:
    """Run and save the complete original-image Val evaluation."""
    checkpoint = Path(checkpoint).resolve()
    data_yaml = Path(data_yaml).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    images, label_dir, names = resolve_data_paths(data_yaml)
    ground_truth, path_to_id, boxes_by_image, resize_gains = build_ground_truth(
        images, label_dir, names, imgsz
    )
    predictions, latencies = _predict(
        checkpoint,
        images,
        path_to_id,
        device,
        imgsz,
        crop_size,
        score_batch_size,
    )
    prediction_path = output_dir / "original_image_predictions.json"
    with prediction_path.open("w", encoding="utf-8") as file:
        json.dump(predictions, file, ensure_ascii=False)

    result = {
        "protocol": {
            "split": "val",
            "images": len(images),
            "instances": len(ground_truth["annotations"]),
            "imgsz": imgsz,
            "rect": False,
            "batch": 1,
            "half": False,
            "conf": 0.001,
            "iou": 0.7,
            "max_det": 300,
            "retina_masks": True,
            "area_policy": "bbox area after scale to 640; AP gate uses the all-area group",
        },
        "class_names": {str(index): name for index, name in enumerate(names)},
        "prediction_count": len(predictions),
        "mean_total_latency_ms": sum(latencies) / len(latencies),
        "metrics": {
            "bbox": evaluate_coco(ground_truth, predictions, resize_gains, "bbox"),
            "segm": evaluate_coco(ground_truth, predictions, resize_gains, "segm"),
        },
        "low_overlap_high_confidence": low_overlap_diagnostic(predictions, boxes_by_image),
    }
    with (output_dir / "original_image_eval.json").open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
    return result


def main() -> None:
    """Run the command-line evaluator."""
    args = parse_args()
    sys.path.insert(0, str(SOURCE_ROOT))
    result = evaluate_original_images(
        checkpoint=args.checkpoint,
        data_yaml=args.data,
        output_dir=args.output_dir,
        device=args.device,
        imgsz=args.imgsz,
        crop_size=args.crop_size,
        score_batch_size=args.score_batch_size,
    )
    print(json.dumps(result["metrics"]["segm"]["overall"]["all"], indent=2))


if __name__ == "__main__":
    main()
