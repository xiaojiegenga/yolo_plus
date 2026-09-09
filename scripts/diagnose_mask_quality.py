"""Measure Val object geometry and score/MaskIoU alignment from saved predictions.

Uses the native-mask predictions produced by evaluate_c_small_val.py. Oracle
scores use Val ground truth for diagnosis only and are not model performance.
"""

import argparse
from collections import Counter
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools import mask as mu
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from scipy.stats import spearmanr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    images = sorted(p for p in (args.data_root / "images/val").iterdir()
                    if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    gt = {"images": [], "annotations": [], "categories": [
        {"id": 1, "name": "Rice leaffolder"}, {"id": 2, "name": "Rice stemborers"}]}
    geometry, gains = {1: [], 2: []}, {}
    for iid, path in enumerate(images, 1):
        with Image.open(path) as im:
            w, h = im.size
        gains[iid] = 640 / max(w, h)
        gt["images"].append(dict(id=iid, file_name=path.name, width=w, height=h))
        for line in (args.data_root / "labels/val" / (path.stem + ".txt")).read_text().splitlines():
            if not line.strip():
                continue
            vals = list(map(float, line.split()))
            cat = int(vals[0]) + 1
            poly = np.array(vals[1:]).reshape(-1, 2) * [w, h]
            lo, hi = poly.min(0), poly.max(0)
            bw, bh = hi - lo
            area640 = float(bw * bh * gains[iid] ** 2)
            sides = sorted(cv2.minAreaRect(poly.astype(np.float32))[1])
            geometry[cat].append(dict(image=path.name, area640=area640,
                short_side640=float(sides[0] * gains[iid]),
                oriented_aspect=float(sides[1] / max(sides[0], 1e-8)),
                polygon_fill=float(cv2.contourArea(poly.astype(np.float32)) / (bw * bh))))
            gt["annotations"].append(dict(id=len(gt["annotations"]) + 1,
                image_id=iid, category_id=cat, bbox=[*lo.tolist(), float(bw), float(bh)],
                segmentation=[poly.reshape(-1).tolist()], area=area640, iscrowd=0))
    assert len(images) == 117 and len(gt["annotations"]) == 557
    with redirect_stdout(io.StringIO()):
        coco = COCO()
        coco.dataset = gt
        coco.createIndex()
    gt_rle = {a["id"]: coco.annToRLE(a) for a in gt["annotations"]}
    result = {"protocol": {"split": "val", "data_root": str(args.data_root),
        "images": len(images), "instances": len(gt["annotations"]),
        "predictions": "Fixed best.pt; FP32, imgsz=640, rect=False, conf=.001, max_det=300, retina_masks=True",
        "metrics": "COCO 101-point AP; same protocol as data-v2-d1-val.json",
        "oracle": "score multiplied by maximum same-class GT MaskIoU; GT-assisted diagnostic, not achievable-model AP",
        "pairing": "Per prediction maximum same-class IoU; diagnostic counts are not one-to-one precision or recall"},
        "geometry": {}, "models": {}}
    for cat, records in geometry.items():
        result["geometry"][str(cat)] = {"instances": len(records),
            "small_bbox640_lt1024": sum(r["area640"] < 1024 for r in records),
            "oriented_aspect_ge3": sum(r["oriented_aspect"] >= 3 for r in records),
            "short_side640_lt8": sum(r["short_side640"] < 8 for r in records)}
        for key in ("area640", "short_side640", "oriented_aspect", "polygon_fill"):
            result["geometry"][str(cat)][key + "_q10_q50_q90"] = np.quantile(
                [r[key] for r in records], [.1, .5, .9]).tolist()
    counts = Counter(a["image_id"] for a in gt["annotations"])
    result["geometry"]["instances_per_image_q50_q90_max"] = [
        float(np.quantile([counts[i] for i in gains], .5)),
        float(np.quantile([counts[i] for i in gains], .9)), max(counts.values())]
    result["geometry"]["image_sizes"] = dict(Counter(f"{im['width']}x{im['height']}" for im in gt["images"]))

    def evaluate(preds):
        with redirect_stdout(io.StringIO()):
            dt = coco.loadRes(deepcopy(preds))
            for a in dt.anns.values():
                a["area"] = a["bbox"][2] * a["bbox"][3] * gains[a["image_id"]] ** 2
            ev = COCOeval(coco, dt, "segm")
            ev.params.maxDets = [1, 10, 300]
            ev.params.areaRng = [[0, 1e10], [0, np.nextafter(1024., 0.)], [1024., 1e10]]
            ev.params.areaRngLbl = ["all", "small", "non_small"]
            ev.evaluate()
            ev.accumulate()
        out = {}
        for k, cat in enumerate(ev.params.catIds):
            out[str(cat)] = {}
            for ai, group in enumerate(ev.params.areaRngLbl):
                p = ev.eval["precision"][:, :, k, ai, -1]
                out[str(cat)][group] = {"AP50": float(p[0][p[0] >= 0].mean()),
                    "AP50_95": float(p[p >= 0].mean())}
        out["overall"] = {key: float(np.mean([out[str(c)]["all"][key] for c in (1, 2)]))
            for key in ("AP50", "AP50_95")}
        return out

    for pp in args.predictions:
        preds = json.loads(pp.read_text())
        max_mask, max_box, mask_at_box = np.zeros(len(preds)), np.zeros(len(preds)), np.zeros(len(preds))
        max_any_box = np.zeros(len(preds))
        for iid in gains:
            image_ids = [j for j, p in enumerate(preds) if p["image_id"] == iid]
            image_gt = [a for a in gt["annotations"] if a["image_id"] == iid]
            if image_ids and image_gt:
                max_any_box[image_ids] = mu.iou([preds[j]["bbox"] for j in image_ids],
                    [a["bbox"] for a in image_gt], [0]*len(image_gt)).max(1)
            for cat in (1, 2):
                ids = [j for j, p in enumerate(preds) if p["image_id"] == iid and p["category_id"] == cat]
                anns = [a for a in gt["annotations"] if a["image_id"] == iid and a["category_id"] == cat]
                if not ids or not anns:
                    continue
                for j in ids:
                    im = gt["images"][iid-1]
                    assert preds[j]["segmentation"]["size"] == [im["height"], im["width"]]
                mi = mu.iou([preds[j]["segmentation"] for j in ids], [gt_rle[a["id"]] for a in anns], [0]*len(anns))
                bi = mu.iou([preds[j]["bbox"] for j in ids], [a["bbox"] for a in anns], [0]*len(anns))
                max_mask[ids], max_box[ids] = mi.max(1), bi.max(1)
                mask_at_box[ids] = mi[np.arange(len(ids)), bi.argmax(1)]
        scores = np.array([p["score"] for p in preds])
        cats = np.array([p["category_id"] for p in preds])
        error_groups = {}
        for threshold in (.25, .5):
            sel = scores >= threshold
            error_groups[str(threshold)] = dict(
                predictions=int(sel.sum()),
                same_class_box_iou_ge05=int((sel & (max_box >= .5)).sum()),
                wrong_class_box_iou_ge05=int((sel & (max_box < .5) & (max_any_box >= .5)).sum()),
                any_box_iou_01_to05=int((sel & (max_any_box >= .1) & (max_any_box < .5)).sum()),
                any_box_iou_lt01=int((sel & (max_any_box < .1)).sum()))
            assert sum(v for k, v in error_groups[str(threshold)].items() if k != "predictions") == int(sel.sum())
        stats = {}
        for cat in (1, 2):
            sel = (cats == cat) & (scores >= .25)
            located = sel & (max_box >= .5)
            stats[str(cat)] = {"predictions_conf025": int(sel.sum()),
                "max_mask_iou_lt05_conf025": int((sel & (max_mask < .5)).sum()),
                "max_box_iou_ge05_conf025": int(located.sum()),
                "mask_iou_lt05_at_best_box_conf025": int((located & (mask_at_box < .5)).sum()),
                "spearman_score_maskiou_conf025": float(spearmanr(scores[sel], max_mask[sel]).statistic)}
        oracle = deepcopy(preds)
        for j, p in enumerate(oracle):
            p["score"] *= float(max_mask[j])
        overlap_oracle = deepcopy(preds)
        for j, p in enumerate(overlap_oracle):
            if max_mask[j] > 0:
                p["score"] *= float(max_mask[j])
        background_oracle = deepcopy(preds)
        for j, p in enumerate(background_oracle):
            if max_any_box[j] < .1:
                p["score"] = 0.
        candidates = sorted([j for j in range(len(preds)) if scores[j] >= .5 and max_mask[j] < .5],
                            key=lambda j: -scores[j])[:6]
        examples = [dict(image=gt["images"][preds[j]["image_id"]-1]["file_name"],
            category=int(cats[j]), score=float(scores[j]), mask_iou=float(max_mask[j]),
            box_iou=float(max_box[j])) for j in candidates]
        result["models"][pp.stem] = {"prediction_source": str(pp), "predictions": len(preds),
            "box_overlap_groups": error_groups,
            "alignment": stats, "original": evaluate(preds), "oracle_score_times_iou": evaluate(oracle),
            "oracle_overlap_only_unmatched_scores_unchanged": evaluate(overlap_oracle),
            "oracle_suppress_any_box_iou_lt01": evaluate(background_oracle),
            "high_score_low_mask_iou_examples": examples}
        print(pp.stem, json.dumps(result["models"][pp.stem], ensure_ascii=False), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Geometry:", json.dumps(result["geometry"], ensure_ascii=False))


if __name__ == "__main__":
    main()
