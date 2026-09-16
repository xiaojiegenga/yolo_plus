"""CPU paired evaluation on Val: small-object COCO AP + high-confidence low-IoU rate.

Pairs the stage-0 baseline 000 with a candidate run on the same CPU, same protocol, so the
comparison is internally consistent regardless of the original GPU evaluation hardware.
Protocol is identical to the old project's evaluate_c_small_val.py: imgsz=640, rect=False,
batch=1, conf=0.001, iou=0.7, max_det=300, retina_masks=True, fp32. The 640-area criterion
(GT and unmatched prediction bbox area * (640/max(h,w))^2 < 1024) defines the small group.

Extra diagnostics vs the old script: predictions with score >= {0.5, 0.25} whose maximum IoU
with any same-image GT box is < 0.1 (the background-confusion metric SFCM targets), plus the
background-image false-positive rate. These use original-image coordinates.

Run (pycocotools is provided via PYTHONPATH from the cached cp310 wheel directory):

    PYTHONPATH=/path/to/val-error-deps python scripts/evaluate_cpu_pair.py \
        --data-root /path/to/datasets \
        --candidate-weights runs/data-v2-cmp1-dss-b16-s42/weights/best.pt \
        --candidate-label DSS
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import platform
import sys

ROOT = Path(__file__).resolve().parents[1]  # new repo root


def box_iou_xywh(a: list[float], b: list[float]) -> float:
    """IoU of two [x, y, w, h] boxes in the same coordinate frame."""
    ax2, ay2 = a[0] + a[2], a[1] + a[3]
    bx2, by2 = b[0] + b[2], b[1] + b[3]
    ix = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    iy = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = ix * iy
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def negative_image_metrics(predictions: list[dict], background_ids: set[int]) -> dict:
    """Count predictions at confidence 0.25 on images with no annotated objects."""
    selected = [p for p in predictions if p["image_id"] in background_ids and p["score"] >= 0.25]
    images_with_fp = len({p["image_id"] for p in selected})
    return {
        "conf": 0.25,
        "images": len(background_ids),
        "images_with_fp": images_with_fp,
        "image_fp_rate": images_with_fp / len(background_ids),
        "FP": len(selected),
        "FP_per_image": len(selected) / len(background_ids),
        "max_conf": max((p["score"] for p in selected), default=0.0),
    }


def low_iou_stats(predictions: list[dict], gt_boxes_by_image: dict[int, list], confs=(0.5, 0.25)) -> dict:
    """High-confidence predictions whose max IoU with any same-image GT box is < 0.1."""
    out = {}
    for conf in confs:
        total, low = 0, 0
        per_cat = {1: [0, 0], 2: [0, 0]}
        for p in predictions:
            if p["score"] < conf:
                continue
            total += 1
            cat = p["category_id"]
            per_cat[cat][0] += 1
            miou = max((box_iou_xywh(p["bbox"], b) for b in gt_boxes_by_image.get(p["image_id"], [])), default=0.0)
            if miou < 0.1:
                low += 1
                per_cat[cat][1] += 1
        out[str(conf)] = {
            "predictions": total,
            "max_any_gt_box_iou_lt_0.1": low,
            "rate": low / total if total else 0.0,
            "per_category": {str(c): {"predictions": v[0], "low_iou": v[1]} for c, v in per_cat.items()},
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "ultralytics-main")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--baseline-weights",
        type=Path,
        default=Path(r"E:\study\graduate_sec\论文撰写\模型训练\runs\data-v2-abl-000-y26m-b16-s42\weights\best.pt"),
    )
    parser.add_argument("--candidate-weights", type=Path, required=True)
    parser.add_argument("--candidate-label", default="DSS")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--reuse-cache",
        action="store_true",
        help="Reuse an existing cached predictions JSON for a label instead of re-running inference",
    )
    parser.add_argument(
        "--mask-chunk",
        type=int,
        default=32,
        help="Detections per process_mask_native chunk; lower it when RAM is tight",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "experiment_records/evaluations/data-v2-cmp1-cpu-pair.json")
    args = parser.parse_args()

    sys.path.insert(0, str(args.source_root.resolve()))
    import gc

    import numpy as np
    import torch
    from PIL import Image
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from ultralytics import YOLO, __version__
    from ultralytics.utils import ops as ultralytics_ops

    torch.set_num_threads(args.threads)

    # retina_masks=True synthesizes every detection mask at ORIGINAL image resolution inside one
    # process_mask_native call; on this CPU (15 GB RAM) a ~200-detection 2157x1618 image needs a
    # ~6 GB transient peak and OOMs. Chunk the call over detections: scale_masks + crop_mask are
    # per-detection elementwise ops, so chunking is mathematically identical, only memory-bounded.
    _process_mask_native_orig = ultralytics_ops.process_mask_native

    def _process_mask_native_chunked(protos, masks_in, bboxes, shape, _chunk=args.mask_chunk):
        n = masks_in.shape[0]
        if n <= _chunk:
            return _process_mask_native_orig(protos, masks_in, bboxes, shape)
        return torch.cat(
            [
                _process_mask_native_orig(protos, masks_in[i : i + _chunk], bboxes[i : i + _chunk], shape)
                for i in range(0, n, _chunk)
            ],
            dim=0,
        )

    ultralytics_ops.process_mask_native = _process_mask_native_chunked

    images = sorted(
        p for p in (args.data_root / "images/val").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    gt = {
        "images": [],
        "annotations": [],
        "categories": [{"id": 1, "name": "Rice leaffolder"}, {"id": 2, "name": "Rice stemborers"}],
    }
    gains: dict[int, float] = {}
    counts: Counter = Counter()
    paths: dict[str, int] = {}
    gt_boxes_by_image: dict[int, list] = {}
    for image_id, path in enumerate(images, 1):
        with Image.open(path) as image:
            w, h = image.size
        gains[image_id] = 640 / max(w, h)
        paths[str(path.resolve())] = image_id
        gt["images"].append({"id": image_id, "file_name": path.name, "width": w, "height": h})
        label = args.data_root / "labels/val" / (path.stem + ".txt")
        for line in label.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            values = list(map(float, line.split()))
            category = int(values[0]) + 1
            polygon = np.array(values[1:], dtype=np.float64).reshape(-1, 2) * [w, h]
            lo, hi = polygon.min(0), polygon.max(0)
            box = [*lo.tolist(), *(hi - lo).tolist()]
            area640 = float(np.prod(hi - lo) * gains[image_id] ** 2)
            counts[(category, "small" if area640 < 1024 else "non_small")] += 1
            gt["annotations"].append(
                {
                    "id": len(gt["annotations"]) + 1,
                    "image_id": image_id,
                    "category_id": category,
                    "bbox": box,
                    "segmentation": [polygon.reshape(-1).tolist()],
                    "area": area640,
                    "iscrowd": 0,
                }
            )
            gt_boxes_by_image.setdefault(image_id, []).append(box)
    assert len(images) == 117 and len(gt["annotations"]) == 557
    assert sum(v for (cat, _), v in counts.items() if cat == 1) == 462
    assert counts[(1, "small")] == 174, counts
    background_ids = {item["id"] for item in gt["images"]} - {item["image_id"] for item in gt["annotations"]}

    cache = ROOT / ".cache"
    cache.mkdir(exist_ok=True)
    image_list = cache / "cpupair-val-images.txt"
    image_list.write_text("\n".join(str(p.resolve()) for p in images), encoding="utf-8")

    result = {
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "protocol": {
            "split": "val",
            "images": len(images),
            "instances": len(gt["annotations"]),
            "data_root": str(args.data_root.resolve()),
            "source_root": str(args.source_root.resolve()),
            "ultralytics": __version__,
            "torch": torch.__version__,
            "device": f"cpu ({platform.processor()}), threads={args.threads}",
            "imgsz": 640,
            "rect": False,
            "batch": 1,
            "half": False,
            "conf": 0.001,
            "iou": 0.7,
            "max_det": 300,
            "retina_masks": True,
            "small": "GT and unmatched prediction bbox area * (640/max(original_h,original_w))^2 < 1024",
            "metric": "COCOeval 101-point AP at IoU .50:.05:.95; R50_025 uses IoU=.50 and conf>=.25",
            "area_policy": "BBox area at 640 for both bbox and segm; retain out-of-range GT as ignored matching objects",
            "mask_policy": "COCO polygon rasterization for GT; full-resolution predicted masks",
            "counts": {str(cat): {area: counts[(cat, area)] for area in ["small", "non_small"]} for cat in [1, 2]},
        },
        "models": {},
    }

    runs = {"000": args.baseline_weights, args.candidate_label: args.candidate_weights}
    for label, weights in runs.items():
        print(f"[LOAD] {label}: {weights}", flush=True)
        pred_file = cache / f"cpupair-{label}-predictions.json"
        if args.reuse_cache and pred_file.exists():
            predictions = json.loads(pred_file.read_text(encoding="utf-8"))
            print(f"[CACHE] {label}: reusing {len(predictions)} cached predictions from {pred_file}", flush=True)
        else:
            model = YOLO(str(weights))
            predictions = []
            for index, pred in enumerate(
                model.predict(
                    source=str(image_list),
                    stream=True,
                    imgsz=640,
                    rect=False,
                    batch=1,
                    conf=0.001,
                    iou=0.7,
                    max_det=300,
                    retina_masks=True,
                    half=False,
                    device="cpu",
                    save=False,
                    verbose=False,
                ),
                1,
            ):
                image_id = paths[str(Path(pred.path).resolve())]
                masks = pred.masks.data.cpu().numpy() if pred.masks is not None else []
                for xyxy, score, cls, mask in zip(
                    pred.boxes.xyxy.cpu().numpy(), pred.boxes.conf.cpu().numpy(), pred.boxes.cls.cpu().numpy(), masks
                ):
                    x1, y1, x2, y2 = map(float, xyxy)
                    rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
                    rle["counts"] = rle["counts"].decode("ascii")
                    predictions.append(
                        {
                            "image_id": image_id,
                            "category_id": int(cls) + 1,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": float(score),
                            "segmentation": rle,
                        }
                    )
                if index % 20 == 0:
                    print(f"[PREDICT] {label}: {index}/117", flush=True)
                del pred, masks
                gc.collect()
            pred_file.write_text(json.dumps(predictions), encoding="utf-8")
            del model
            gc.collect()

        result["models"][label] = {
            "weights": str(weights),
            "metrics": {},
            "negative_images_conf025": negative_image_metrics(predictions, background_ids),
            "high_conf_low_iou": low_iou_stats(predictions, gt_boxes_by_image),
        }
        for kind in ["bbox", "segm"]:
            coco = COCO()
            coco.dataset = deepcopy(gt)
            coco.createIndex()
            detections = coco.loadRes(deepcopy(predictions))
            # loadRes computes native bbox area; use the same fixed 640-area criterion as GT.
            for item in detections.anns.values():
                item["area"] = item["bbox"][2] * item["bbox"][3] * gains[item["image_id"]] ** 2
            evaluator = COCOeval(coco, detections, kind)
            evaluator.params.maxDets = [1, 10, 300]
            evaluator.params.areaRng = [[0, 1e10], [0, np.nextafter(1024.0, 0.0)], [1024.0, 1e10]]
            evaluator.params.areaRngLbl = ["all", "small", "non_small"]
            evaluator.evaluate()
            evaluator.accumulate()
            per_kind = {}
            for k, cat in enumerate(evaluator.params.catIds):
                per_kind[str(cat)] = {}
                for a, group in enumerate(evaluator.params.areaRngLbl):
                    p = evaluator.eval["precision"][:, :, k, a, -1]
                    recall = evaluator.eval["recall"][:, k, a, -1]
                    selected = [
                        e
                        for e in evaluator.evalImgs
                        if e is not None and e["category_id"] == cat and list(e["aRng"]) == evaluator.params.areaRng[a]
                    ]
                    denom = sum(int((~e["gtIgnore"].astype(bool)).sum()) for e in selected)
                    tp = sum(
                        int(
                            (
                                (e["dtMatches"][0] > 0)
                                & ~e["dtIgnore"][0]
                                & (np.array(e["dtScores"]) >= 0.25)
                            ).sum()
                        )
                        for e in selected
                    )
                    fp = sum(
                        int(
                            (
                                (e["dtMatches"][0] == 0)
                                & ~e["dtIgnore"][0]
                                & (np.array(e["dtScores"]) >= 0.25)
                            ).sum()
                        )
                        for e in selected
                    )
                    per_kind[str(cat)][group] = {
                        "n_gt": denom,
                        "AP50": float(p[0][p[0] >= 0].mean()),
                        "AP50_95": float(p[p >= 0].mean()),
                        "AR50_95_max300": float(recall[recall >= 0].mean()),
                        "R50_max300_conf0001": float(recall[0]),
                        "R50_conf025": tp / denom,
                        "P50_conf025": tp / (tp + fp) if tp + fp else 0.0,
                        "TP50_conf025": tp,
                        "FP50_conf025": fp,
                    }
            result["models"][label]["metrics"][kind] = per_kind
        print(f"[DONE] {label}: small leaffolder segm = {json.dumps(result['models'][label]['metrics']['segm']['1']['small'])}", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVED] {args.output}", flush=True)


if __name__ == "__main__":
    main()
