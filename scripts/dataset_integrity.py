"""YOLO instance-segmentation dataset integrity checks and content fingerprinting.

The dataset itself is intentionally stored outside this Git repository.  This
module creates a deterministic fingerprint from every image and matching label
file so a formal experiment can prove which local dataset revision it used.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import yaml


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
PARENT_SUFFIX = re.compile(r"_r\d+c\d+$", re.IGNORECASE)


def sha256_file(path: Path) -> str:
    """Return an uppercase SHA-256 digest without loading the whole file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _dataset_root(data_yaml: Path, data: dict[str, Any]) -> Path:
    root = Path(data["path"]).expanduser()
    if not root.is_absolute():
        root = data_yaml.parent / root
    return root.resolve()


def _label_dir(image_dir: Path) -> Path:
    """Map .../images/<split> to .../labels/<split>."""
    parts = list(image_dir.parts)
    candidates = [index for index, part in enumerate(parts) if part.lower() == "images"]
    if not candidates:
        raise ValueError(f"无法从图片目录推导标签目录（缺少 images 层级）：{image_dir}")
    parts[candidates[-1]] = "labels"
    return Path(*parts)


def audit_yolo_seg_dataset(data_yaml: str | Path) -> dict[str, Any]:
    """Validate a YOLO polygon dataset and return a reproducible summary."""
    data_yaml = Path(data_yaml).resolve()
    with data_yaml.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    required = {"path", "train", "val", "test", "nc", "names"}
    missing = sorted(required.difference(data or {}))
    if missing:
        raise ValueError(f"数据集 YAML 缺少字段：{', '.join(missing)}")

    nc = int(data["nc"])
    root = _dataset_root(data_yaml, data)
    issues: list[dict[str, Any]] = []
    images_by_split: dict[str, int] = {}
    labels_by_split: dict[str, int] = {}
    empty_labels_by_split: dict[str, int] = {}
    objects_by_split_and_class: dict[str, dict[str, int]] = {}
    parent_splits: dict[str, set[str]] = defaultdict(set)
    image_hash_splits: dict[str, set[str]] = defaultdict(set)
    file_digests: list[tuple[str, str]] = []

    for split in ("train", "val", "test"):
        image_dir = (root / str(data[split])).resolve()
        label_dir = _label_dir(image_dir)
        if not image_dir.is_dir():
            issues.append({"type": "missing_image_dir", "split": split, "path": str(image_dir)})
            continue
        if not label_dir.is_dir():
            issues.append({"type": "missing_label_dir", "split": split, "path": str(label_dir)})
            continue

        images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        labels = sorted(label_dir.glob("*.txt"))
        images_by_stem = {path.stem: path for path in images}
        labels_by_stem = {path.stem: path for path in labels}
        images_by_split[split] = len(images)
        labels_by_split[split] = len(labels)
        empty_labels_by_split[split] = 0
        class_counts = Counter()

        for stem, image_path in images_by_stem.items():
            if stem not in labels_by_stem:
                issues.append({"type": "missing_label", "split": split, "file": image_path.name})
        for stem, label_path in labels_by_stem.items():
            if stem not in images_by_stem:
                issues.append({"type": "orphan_label", "split": split, "file": label_path.name})

        for image_path in images:
            parent_key = PARENT_SUFFIX.sub("", image_path.stem)
            parent_splits[parent_key].add(split)
            image_digest = sha256_file(image_path)
            image_hash_splits[image_digest].add(split)
            file_digests.append((image_path.relative_to(root).as_posix(), image_digest))

            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.is_file():
                continue
            label_digest = sha256_file(label_path)
            file_digests.append((label_path.relative_to(root).as_posix(), label_digest))
            lines = [line.strip() for line in label_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
            if not lines:
                empty_labels_by_split[split] += 1
            duplicate_count = len(lines) - len(set(lines))
            if duplicate_count:
                issues.append(
                    {
                        "type": "duplicate_label_line",
                        "split": split,
                        "file": label_path.name,
                        "count": duplicate_count,
                    }
                )

            for line_number, line in enumerate(lines, start=1):
                tokens = line.split()
                if len(tokens) < 7 or (len(tokens) - 1) % 2:
                    issues.append(
                        {
                            "type": "malformed_polygon",
                            "split": split,
                            "file": label_path.name,
                            "line": line_number,
                        }
                    )
                    continue
                try:
                    class_id = int(tokens[0])
                    coords = [float(value) for value in tokens[1:]]
                except ValueError:
                    issues.append(
                        {
                            "type": "non_numeric_label",
                            "split": split,
                            "file": label_path.name,
                            "line": line_number,
                        }
                    )
                    continue
                if not 0 <= class_id < nc:
                    issues.append(
                        {
                            "type": "invalid_class",
                            "split": split,
                            "file": label_path.name,
                            "line": line_number,
                        }
                    )
                    continue
                if any(not math.isfinite(value) or not 0 <= value <= 1 for value in coords):
                    issues.append(
                        {
                            "type": "invalid_coordinate",
                            "split": split,
                            "file": label_path.name,
                            "line": line_number,
                        }
                    )
                    continue
                class_counts[class_id] += 1

        objects_by_split_and_class[split] = {
            str(class_id): class_counts[class_id] for class_id in range(nc)
        }

    content_digest = hashlib.sha256()
    for relative_path, file_digest in sorted(file_digests):
        content_digest.update(relative_path.encode("utf-8"))
        content_digest.update(b"\0")
        content_digest.update(bytes.fromhex(file_digest))
        content_digest.update(b"\0")

    crossing_parents = {
        parent: sorted(splits) for parent, splits in parent_splits.items() if len(splits) > 1
    }
    crossing_hashes = {
        digest: sorted(splits) for digest, splits in image_hash_splits.items() if len(splits) > 1
    }
    issue_types = Counter(issue["type"] for issue in issues)
    return {
        "schema_version": 1,
        "dataset_id": "rice-pest-data-v1",
        "dataset_root": str(root),
        "dataset_yaml": str(data_yaml),
        "dataset_yaml_sha256": sha256_file(data_yaml),
        "dataset_content_sha256": content_digest.hexdigest().upper(),
        "images_by_split": images_by_split,
        "labels_by_split": labels_by_split,
        "empty_labels_by_split": empty_labels_by_split,
        "objects_by_split_and_class": objects_by_split_and_class,
        "parent_groups": len(parent_splits),
        "parent_groups_crossing_splits": len(crossing_parents),
        "exact_image_hashes_crossing_splits": len(crossing_hashes),
        "issue_count": len(issues),
        "issue_types": dict(sorted(issue_types.items())),
        "issue_samples": issues[:20],
        "crossing_parent_samples": dict(list(sorted(crossing_parents.items()))[:20]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="校验并生成 YOLO 分割数据集内容指纹。")
    parser.add_argument("--data", required=True, help="YOLO 数据集 YAML 路径。")
    parser.add_argument("--output", default=None, help="可选 JSON 输出路径。")
    args = parser.parse_args()
    report = audit_yolo_seg_dataset(args.data)
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
        print(f"[INFO] Dataset manifest: {output}")
    print(payload)


if __name__ == "__main__":
    main()
