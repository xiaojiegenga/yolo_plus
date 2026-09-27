"""Convert the data-v2 YOLO segmentation labels to COCO instance annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


REPO_ROOT = Path(__file__).resolve().parents[1]
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CATEGORIES = [
    {"id": 1, "name": "Rice leaffolder"},
    {"id": 2, "name": "Rice stemborers"},
]
EXPECTED_IMAGES = {"train": 938, "val": 117}


def polygon_area(points: list[tuple[float, float]]) -> float:
    return abs(
        sum(
            x0 * y1 - x1 * y0
            for (x0, y0), (x1, y1) in zip(points, points[1:] + points[:1])
        )
    ) / 2


def parse_label(line: str, width: int, height: int) -> tuple[int, list[float]]:
    fields = line.split()
    if len(fields) < 7 or len(fields) % 2 != 1:
        raise ValueError(f"分割标签需要类别和至少 3 个坐标点：{line}")

    category_id = int(fields[0]) + 1
    if category_id not in (1, 2):
        raise ValueError(f"标签类别必须为 0 或 1：{line}")

    values = [float(value) for value in fields[1:]]
    if any(value < 0 or value > 1 for value in values):
        raise ValueError(f"标签坐标必须归一化到 0—1：{line}")

    polygon = [
        coordinate * (width if index % 2 == 0 else height)
        for index, coordinate in enumerate(values)
    ]
    return category_id, polygon


def convert_split(data_root: Path, split: str) -> tuple[dict, int]:
    image_root = data_root / "images" / split
    label_root = data_root / "labels" / split
    if not image_root.is_dir() or not label_root.is_dir():
        raise FileNotFoundError(f"缺少 {split} 图像或标签目录：{data_root}")

    paths = sorted(
        path for path in image_root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if len(paths) != EXPECTED_IMAGES[split]:
        raise ValueError(
            f"{split} 应有 {EXPECTED_IMAGES[split]} 张图像，实际为 {len(paths)}"
        )

    images: list[dict] = []
    annotations: list[dict] = []
    duplicates_removed = 0
    for image_id, image_path in enumerate(paths, start=1):
        with Image.open(image_path) as image:
            width, height = image.size
        relative_path = image_path.relative_to(data_root / "images")
        images.append(
            {
                "id": image_id,
                "file_name": relative_path.as_posix(),
                "width": width,
                "height": height,
            }
        )

        label_path = label_root / image_path.relative_to(image_root).with_suffix(".txt")
        if not label_path.exists():
            continue
        seen_labels: set[tuple[str, ...]] = set()
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            label_fields = tuple(line.split())
            if label_fields in seen_labels:
                duplicates_removed += 1
                continue
            seen_labels.add(label_fields)
            category_id, polygon = parse_label(line, width, height)
            points = list(zip(polygon[::2], polygon[1::2]))
            xs, ys = zip(*points)
            if max(xs) <= min(xs) or max(ys) <= min(ys):
                raise ValueError(f"无效分割框：{label_path}")
            annotations.append(
                {
                    "id": len(annotations) + 1,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [polygon],
                    "bbox": [min(xs), min(ys), max(xs) - min(xs), max(ys) - min(ys)],
                    "area": polygon_area(points),
                    "iscrowd": 0,
                }
            )

    if split == "val" and len(annotations) != 557:
        raise ValueError(f"Val 应有 557 个实例，实际为 {len(annotations)}")
    return {
        "info": {"description": f"rice-pest-data-v2 {split}"},
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": CATEGORIES,
    }, duplicates_removed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/root/yolo_data"))
    parser.add_argument(
        "--output-dir", type=Path, default=REPO_ROOT / "data" / "table5" / "annotations"
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val"):
        result, duplicates_removed = convert_split(args.data_root, split)
        output_path = args.output_dir / f"{split}.json"
        output_path.write_text(
            json.dumps(result, ensure_ascii=False), encoding="utf-8"
        )
        print(
            f"[READY] {split}: {len(result['images'])} images, "
            f"{len(result['annotations'])} instances, "
            f"{duplicates_removed} duplicate labels removed -> {output_path}"
        )


if __name__ == "__main__":
    main()
