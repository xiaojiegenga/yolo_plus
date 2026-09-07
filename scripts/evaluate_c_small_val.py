"""Compare Baseline and C on Val using bbox area after letterboxing to 640 to define small objects.

This separate COCO-style evaluation does not replace the training CSV metrics.
Run with the C implementation on the import path via --source-root.
"""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
RUNS = {"000": "data-v2-abl-000-y26m-b16-s42", "C": "data-v2-abl-001-p2head-b16-s42"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="C checkout's ultralytics-main directory")
    parser.add_argument("--data-root", type=Path, required=True, help="Dataset containing images/val and labels/val")
    parser.add_argument("--output", type=Path, default=ROOT / "experiment_records/evaluations/data-v2-c-small-val.json")
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    import numpy as np
    import torch
    from PIL import Image
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from ultralytics import YOLO, __version__

    torch.set_num_threads(2)
    images = sorted(p for p in (args.data_root / "images/val").iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    gt = {"images": [], "annotations": [], "categories": [{"id": 1, "name": "Rice leaffolder"}, {"id": 2, "name": "Rice stemborers"}]}
    gains, counts, paths = {}, Counter(), {}
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
            gt["annotations"].append({"id": len(gt["annotations"]) + 1, "image_id": image_id,
                "category_id": category, "bbox": box, "segmentation": [polygon.reshape(-1).tolist()],
                "area": area640, "iscrowd": 0})
    assert len(images) == 117 and len(gt["annotations"]) == 557
    assert sum(v for (cat, _), v in counts.items() if cat == 1) == 462
    assert counts[(1, "small")] == 174, counts
    image_list = ROOT / ".cache/c-small-val-images.txt"
    image_list.parent.mkdir(parents=True, exist_ok=True)
    image_list.write_text("\n".join(str(p.resolve()) for p in images), encoding="utf-8")

    result = {"created_at_local": datetime.now().isoformat(timespec="seconds"),
        "protocol": {"split": "val", "images": len(images), "instances": len(gt["annotations"]),
            "data_root": str(args.data_root.resolve()), "source_root": str(args.source_root.resolve()),
            "ultralytics": __version__, "torch": torch.__version__, "device": torch.cuda.get_device_name(0),
            "imgsz": 640, "rect": False, "batch": 1, "half": False, "conf": 0.001, "iou": 0.7, "max_det": 300,
            "retina_masks": True, "small": "GT and unmatched prediction bbox area * (640/max(original_h,original_w))^2 < 1024",
            "metric": "COCOeval 101-point AP at IoU .50:.05:.95; R50_025 uses IoU=.50 and conf>=.25",
            "area_policy": "BBox area at 640 for both bbox and segm; retain out-of-range GT as ignored matching objects",
            "mask_policy": "COCO polygon rasterization for GT; full-resolution predicted masks",
            "counts": {str(cat): {area: counts[(cat, area)] for area in ["small", "non_small"]} for cat in [1, 2]}},
        "models": {}}
    for label, run in RUNS.items():
        model = YOLO(ROOT / "runs" / run / "weights/best.pt")
        predictions = []
        for index, pred in enumerate(model.predict(source=str(image_list), stream=True, imgsz=640, rect=False,
                batch=1, conf=0.001, iou=0.7, max_det=300, retina_masks=True, half=False, device=0,
                save=False, verbose=False), 1):
            image_id = paths[str(Path(pred.path).resolve())]
            for xyxy, score, cls, mask in zip(pred.boxes.xyxy.cpu().numpy(), pred.boxes.conf.cpu().numpy(),
                    pred.boxes.cls.cpu().numpy(), pred.masks.data.cpu().numpy() if pred.masks is not None else []):
                x1, y1, x2, y2 = map(float, xyxy)
                rle = mask_utils.encode(np.asfortranarray(mask.astype(np.uint8)))
                rle["counts"] = rle["counts"].decode("ascii")
                predictions.append({"image_id": image_id, "category_id": int(cls) + 1,
                    "bbox": [x1, y1, x2-x1, y2-y1], "score": float(score), "segmentation": rle})
            if index % 20 == 0:
                print(f"[PREDICT] {label}: {index}/117", flush=True)
        pred_file = ROOT / ".cache" / f"c-small-{label}-predictions.json"
        pred_file.write_text(json.dumps(predictions), encoding="utf-8")
        result["models"][label] = {"run_id": run, "metrics": {}}
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
            evaluator.params.areaRng = [[0, 1e10], [0, np.nextafter(1024., 0.)], [1024., 1e10]]
            evaluator.params.areaRngLbl = ["all", "small", "non_small"]
            evaluator.evaluate()
            evaluator.accumulate()
            per_kind = {}
            for k, cat in enumerate(evaluator.params.catIds):
                per_kind[str(cat)] = {}
                for a, group in enumerate(evaluator.params.areaRngLbl):
                    p = evaluator.eval["precision"][:, :, k, a, -1]
                    recall = evaluator.eval["recall"][:, k, a, -1]
                    selected = [e for e in evaluator.evalImgs if e is not None and e["category_id"] == cat
                        and list(e["aRng"]) == evaluator.params.areaRng[a]]
                    denom = sum(int((~e["gtIgnore"].astype(bool)).sum()) for e in selected)
                    tp = sum(int(((e["dtMatches"][0] > 0) & ~e["dtIgnore"][0]
                        & (np.array(e["dtScores"]) >= .25)).sum()) for e in selected)
                    fp = sum(int(((e["dtMatches"][0] == 0) & ~e["dtIgnore"][0]
                        & (np.array(e["dtScores"]) >= .25)).sum()) for e in selected)
                    per_kind[str(cat)][group] = {"n_gt": denom, "AP50": float(p[0][p[0]>=0].mean()),
                        "AP50_95": float(p[p>=0].mean()), "AR50_95_max300": float(recall[recall>=0].mean()),
                        "R50_max300_conf0001": float(recall[0]), "R50_conf025": tp/denom,
                        "P50_conf025": tp/(tp+fp) if tp+fp else 0., "TP50_conf025": tp, "FP50_conf025": fp}
            result["models"][label]["metrics"][kind] = per_kind
        print(f"[DONE] {label}", json.dumps(result["models"][label]["metrics"]["segm"]["1"]), flush=True)
        del model
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[SAVED] {args.output}")


if __name__ == "__main__":
    main()
