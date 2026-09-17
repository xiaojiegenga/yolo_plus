"""Final split evaluation for a frozen run (Test 终评).

Evaluates a frozen checkpoint on the val and/or test split with the official Ultralytics
validation protocol (imgsz 640, batch 16, conf 0.001, iou 0.7, max_det 300) and records
overall plus per-class Box/Mask metrics. Running the val split first serves as a protocol
reproduction check against the numbers recorded during training (cmp1 best epoch:
Mask mAP50 0.71687 / mAP50-95 0.38929); the test split is the deliverable for the paper.

Cloud (from the repo root, RTX 5090):

    python scripts/eval_test_split.py \
        --weights runs/data-v2-cmp1-dss-b16-s42/weights/best.pt \
        --data experiments/yolo_data_v2_cloud.yaml \
        --device 0

Local smoke test (CPU, same protocol): pass a local data yaml and --device cpu.
Outputs: exports/test-eval.json + exports/test-eval.md; Ultralytics plots land in
runs/eval-cmp1-<split>/.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]

# Recorded cmp1 best-epoch Val numbers (official protocol) for the reproduction check.
RECORDED_VAL = {"metrics/mAP50(M)": 0.71687, "metrics/mAP50-95(M)": 0.38929}


def fmt(value: float) -> str:
    return f"{value:.5f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "experiments/yolo_data_v2_cloud.yaml")
    parser.add_argument("--source-root", type=Path, default=ROOT / "ultralytics-main")
    parser.add_argument("--splits", default="val,test", help="comma-separated splits, e.g. 'val' or 'val,test'")
    parser.add_argument("--device", default="0")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-prefix", type=Path, default=ROOT / "exports/test-eval")
    args = parser.parse_args()

    sys.path.insert(0, str(args.source_root.resolve()))
    from ultralytics import YOLO, __version__  # noqa: E402
    import torch  # noqa: E402

    results = {
        "created_at_local": datetime.now().isoformat(timespec="seconds"),
        "weights": str(args.weights.resolve()),
        "data": str(args.data.resolve()),
        "ultralytics": __version__,
        "torch": torch.__version__,
        "protocol": {
            "imgsz": 640,
            "batch": args.batch,
            "conf": 0.001,
            "iou": 0.7,
            "max_det": 300,
            "half": False,
            "device": str(args.device),
        },
        "splits": {},
    }

    model = YOLO(str(args.weights))
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        print(f"\n[VAL] split={split}", flush=True)
        metrics = model.val(
            data=str(args.data),
            split=split,
            imgsz=640,
            batch=args.batch,
            conf=0.001,
            iou=0.7,
            max_det=300,
            plots=True,
            save_json=False,
            device=args.device,
            workers=args.workers,
            half=False,
            project="runs",
            name=f"eval-cmp1-{split}",
            verbose=True,
        )
        entry = {"results_dict": {k: float(v) for k, v in metrics.results_dict.items()}}
        entry["save_dir"] = str(getattr(metrics, "save_dir", ""))
        box = getattr(metrics, "box", None)
        mask = getattr(metrics, "mask", None) or getattr(metrics, "seg", None)
        per_class = {}
        if box is not None and mask is not None:
            names = metrics.names
            for i, cls_idx in enumerate(box.ap_class_index):
                name = names[int(cls_idx)] if hasattr(names, "__getitem__") else str(cls_idx)
                per_class[str(name)] = {
                    "box P/R/mAP50/mAP50-95": [round(float(x), 5) for x in box.class_result(i)],
                    "mask P/R/mAP50/mAP50-95": [round(float(x), 5) for x in mask.class_result(i)],
                }
        entry["per_class"] = per_class
        if split == "val":
            entry["reproduction_check"] = {
                key: {"recorded": value, "this_run": float(metrics.results_dict.get(key, float("nan")))}
                for key, value in RECORDED_VAL.items()
            }
        results["splits"][split] = entry
        print(
            f"[DONE] {split}: Mask mAP50={metrics.results_dict.get('metrics/mAP50(M)')} "
            f"mAP50-95={metrics.results_dict.get('metrics/mAP50-95(M)')}",
            flush=True,
        )

    args.out_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.out_prefix.with_suffix(".json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    lines = [
        "# Test 终评记录（官方协议）",
        "",
        f"- 权重：`{results['weights']}`",
        f"- 协议：imgsz 640 / batch {args.batch} / conf 0.001 / iou 0.7 / max_det 300 / device {args.device}",
        f"- 时间：{results['created_at_local']}",
        "",
        "| split | Mask P | Mask R | Mask mAP50 | Mask mAP50-95 | Box mAP50 | Box mAP50-95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for split, entry in results["splits"].items():
        rd = entry["results_dict"]
        lines.append(
            f"| {split} | {fmt(rd['metrics/precision(M)'])} | {fmt(rd['metrics/recall(M)'])} "
            f"| {fmt(rd['metrics/mAP50(M)'])} | {fmt(rd['metrics/mAP50-95(M)'])} "
            f"| {fmt(rd['metrics/mAP50(B)'])} | {fmt(rd['metrics/mAP50-95(B)'])} |"
        )
    for split, entry in results["splits"].items():
        if not entry["per_class"]:
            continue
        lines += ["", f"## {split} 分类型", "", "| 类别 | Mask P | R | mAP50 | mAP50-95 | Box P | R | mAP50 | mAP50-95 |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for name, pc in entry["per_class"].items():
            b = pc["box P/R/mAP50/mAP50-95"]
            m = pc["mask P/R/mAP50/mAP50-95"]
            lines.append(
                f"| {name} | {fmt(m[0])} | {fmt(m[1])} | {fmt(m[2])} | {fmt(m[3])} "
                f"| {fmt(b[0])} | {fmt(b[1])} | {fmt(b[2])} | {fmt(b[3])} |"
            )
    if "val" in results["splits"]:
        check = results["splits"]["val"].get("reproduction_check", {})
        lines += ["", "## val 复现校验（对照训练期 best epoch）", ""]
        for key, item in check.items():
            lines.append(f"- {key}: 记录 {item['recorded']:.5f} vs 本次 {item['this_run']:.5f}")
    args.out_prefix.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[SAVED] {args.out_prefix.with_suffix('.json')}")
    print(f"[SAVED] {args.out_prefix.with_suffix('.md')}")


if __name__ == "__main__":
    main()
