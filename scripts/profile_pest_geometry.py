"""统计训练集和验证集的虫害掩膜几何特征，按模型输入尺寸等比缩放。

示例：python scripts/profile_pest_geometry.py --data-root E:/datasets
      --output knowledge/data/dataset_geometry_train_val_640-r2.json --imgsz 640
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


QUANTILES = [0, 10, 25, 50, 75, 90, 100]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="包含 images/ 和 labels/ 的数据集根目录",
    )
    parser.add_argument("--output", type=Path, required=True, help="输出 JSON 路径")
    parser.add_argument(
        "--imgsz", type=int, default=640, help="缩放后图像最长边，默认 640"
    )
    return parser.parse_args()


def distribution(values):
    return {
        str(q): round(float(value), 3)
        for q, value in zip(QUANTILES, np.percentile(values, QUANTILES))
    }


def summarize(items):
    if not items:
        return {"instances": 0}
    result = {"instances": len(items)}
    for key in [
        "bbox_area",
        "bbox_short",
        "bbox_long",
        "bbox_aspect",
        "polygon_area",
        "polygon_fill",
        "rotated_short",
        "rotated_long",
        "rotated_aspect",
        "equivalent_thickness",
    ]:
        result[key + "_quantiles"] = distribution([item[key] for item in items])
    for key, limits in [
        ("bbox_area", [32**2, 96**2]),
        ("bbox_short", [4, 8, 16, 32]),
        ("rotated_short", [4, 8, 16, 32]),
        ("equivalent_thickness", [2, 4, 8, 16]),
        ("polygon_area", [32**2, 96**2]),
    ]:
        below = {}
        for limit in limits:
            count = sum(item[key] < limit for item in items)
            below[str(limit)] = {
                "count": count,
                "percent": round(100 * count / len(items), 2),
            }
        result[key + "_below"] = below
    above = {}
    for limit in [3, 5, 10]:
        count = sum(item["rotated_aspect"] > limit for item in items)
        above[str(limit)] = {
            "count": count,
            "percent": round(100 * count / len(items), 2),
        }
    result["rotated_aspect_above"] = above
    return result


def profile_split(data_root, split, imgsz):
    images = sorted(
        path
        for path in (data_root / "images" / split).iterdir()
        if path.suffix.lower()
        in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    )
    labels = {
        path.stem: path for path in (data_root / "labels" / split).glob("*.txt")
    }
    rows = []
    problems = []
    sizes = Counter()
    images_by_class = Counter()
    object_counts = []
    empty_labels = 0
    for path in images:
        with Image.open(path) as image:
            width, height = image.size
        sizes[f"{width}x{height}"] += 1
        ratio = imgsz / max(width, height)
        label = labels.get(path.stem)
        if label is None:
            problems.append(f"Missing label: {path}")
            continue
        content = [
            line
            for line in label.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        ]
        object_counts.append(len(content))
        empty_labels += not content
        present = set()
        for line_num, line in enumerate(content, 1):
            values = np.array([float(value) for value in line.split()])
            cls = int(values[0])
            coords = values[1:]
            if len(coords) < 6 or len(coords) % 2:
                problems.append(f"Nonpolygon: {label}:{line_num}")
                continue
            present.add(cls)
            points = coords.reshape(-1, 2) * np.array([width, height])
            scaled = (points * ratio).astype(np.float32)
            bbox_min = scaled.min(axis=0)
            bbox_max = scaled.max(axis=0)
            bbox_wh = bbox_max - bbox_min
            short, long = sorted(bbox_wh)
            area = abs(float(cv2.contourArea(scaled)))
            rotated = cv2.minAreaRect(scaled)
            rshort, rlong = sorted(rotated[1])
            item = {
                "split": split,
                "image": str(path),
                "label_line": line_num,
                "class": cls,
                "image_size": [width, height],
                "original_bbox_xyxy": [
                    round(float(value), 2)
                    for value in np.r_[points.min(axis=0), points.max(axis=0)]
                ],
                "bbox_area": float(short * long),
                "bbox_short": float(short),
                "bbox_long": float(long),
                "bbox_aspect": float(long / max(short, 1e-8)),
                "polygon_area": area,
                "polygon_fill": area / max(float(short * long), 1e-8),
                "rotated_short": rshort,
                "rotated_long": rlong,
                "rotated_aspect": rlong / max(rshort, 1e-8),
                "equivalent_thickness": area / max(rlong, 1e-8),
            }
            rows.append(item)
        images_by_class.update(present)
    unpaired = set(labels) - {path.stem for path in images}
    if unpaired:
        problems.append(f"{split}: {len(unpaired)} labels without image")
    stats = {
        "images": len(images),
        "labels": len(labels),
        "empty_labels": empty_labels,
        "image_sizes": dict(sizes),
        "images_by_class": dict(images_by_class),
        "instances_per_image_quantiles": distribution(object_counts),
        "all": summarize(rows),
        "classes": {
            str(cls): summarize([item for item in rows if item["class"] == cls])
            for cls in [0, 1]
        },
    }
    return rows, stats, problems


def representative_examples(rows):
    examples = []
    for cls in [0, 1]:
        own = [item for item in rows if item["class"] == cls]
        for metric in ["bbox_area", "rotated_aspect"]:
            median = np.median([item[metric] for item in own])
            candidates = sorted(own, key=lambda item: abs(item[metric] - median))
            row = next(
                item
                for item in candidates
                if item["image"] not in {example["image"] for example in examples}
            )
            examples.append(dict(row, representative_metric=metric))
    return examples


def main():
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"输出文件已存在：{args.output}")

    rows = []
    split_stats = {}
    problems = []
    for split in ["train", "val"]:
        split_rows, stats, split_problems = profile_split(
            args.data_root, split, args.imgsz
        )
        rows.extend(split_rows)
        split_stats[split] = stats
        problems.extend(split_problems)

    stats = {
        "root": str(args.data_root),
        "imgsz": args.imgsz,
        "measurement": (
            "Pixels after proportional resize with longest image side "
            f"{args.imgsz}, before translation padding; train/val only. "
            "Bbox is axis-aligned; rotated box via OpenCV minAreaRect. "
            "Polygon area is continuous contour area, not rasterized mask area. "
            "Equivalent thickness = polygon area / rotated-box long side."
        ),
        "splits": split_stats,
        "combined": summarize(rows),
        "combined_classes": {
            str(cls): summarize([item for item in rows if item["class"] == cls])
            for cls in [0, 1]
        },
        "examples": representative_examples(rows),
        "problems": problems,
    }
    args.output.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        f"Written {args.output}: "
        f"train={split_stats['train']['images']}, "
        f"val={split_stats['val']['images']}, instances={len(rows)}"
    )


if __name__ == "__main__":
    main()
