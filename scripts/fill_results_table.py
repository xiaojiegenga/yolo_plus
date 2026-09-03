"""从 Ultralytics results.csv 更新实验汇总表。"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TABLE = REPO_ROOT / "experiment_records" / "comparison.csv"
DEFAULT_FIELDS = [
    "version",
    "data",
    "status",
    "split",
    "mask_p",
    "mask_r",
    "mask_map50",
    "mask_map50_95",
    "leaffolder_p",
    "leaffolder_r",
    "leaffolder_ap50",
    "stemborers_p",
    "stemborers_r",
    "stemborers_ap50",
    "git_branch",
    "metric_source",
    "notes",
]
METRIC_FIELDS = {
    "mask_p": "metrics/precision(M)",
    "mask_r": "metrics/recall(M)",
    "mask_map50": "metrics/mAP50(M)",
    "mask_map50_95": "metrics/mAP50-95(M)",
}
FITNESS_FIELDS = (
    "metrics/mAP50-95(B)",
    "metrics/mAP50-95(M)",
)
GENERATED_NOTE_PREFIXES = (
    "results.csv 中 Mask mAP50-95 最高行为 epoch ",
    "results.csv 中 official fitness 最高行为 epoch ",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", help="汇总表中的名称，默认使用目录名")
    parser.add_argument(
        "--data",
        default="data-v2",
        help="数据版本，默认使用当前项目的 data-v2",
    )
    parser.add_argument("--status", default="completed")
    parser.add_argument("--split", default="val")
    parser.add_argument("--branch", default="")
    parser.add_argument("--notes", default="")
    parser.add_argument("--table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="显示将写入的行，不修改 CSV",
    )
    return parser.parse_args()


def resolve_path(value: Path, must_exist: bool = True) -> Path:
    if value.is_absolute():
        if must_exist and not value.exists():
            raise FileNotFoundError(f"找不到路径：{value}")
        return value.resolve()

    candidates = [Path.cwd() / value, REPO_ROOT / value]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    if not must_exist:
        return (REPO_ROOT / value).resolve()
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"找不到路径；已检查：{searched}")


def parse_float(value: str, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"字段 {field} 不是有效数字：{value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"字段 {field} 不是有限数字：{value!r}")
    return number


def official_fitness(row: dict[str, str]) -> float:
    return sum(parse_float(row[field], field) for field in FITNESS_FIELDS)


def read_best_fitness_row(results_path: Path) -> dict[str, str]:
    with results_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        rows = [
            {
                str(key).strip(): str(value).strip()
                for key, value in row.items()
                if key is not None and value is not None
            }
            for row in reader
        ]

    if not rows:
        raise ValueError(f"results.csv 没有数据行：{results_path}")

    required = list(
        dict.fromkeys(["epoch", *METRIC_FIELDS.values(), *FITNESS_FIELDS])
    )
    missing = [field for field in required if field not in rows[0]]
    if missing:
        raise ValueError(f"results.csv 缺少字段：{missing}")

    return max(rows, key=official_fitness)


def format_number(value: str, field: str) -> str:
    return f"{parse_float(value, field):.6f}".rstrip("0").rstrip(".")


def relative_source(path: Path) -> str:
    try:
        return path.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def read_table(table_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    if not table_path.exists():
        return list(DEFAULT_FIELDS), []

    with table_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fields = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]

    if "version" not in fields:
        raise ValueError(f"汇总表缺少 version 列：{table_path}")
    for field in DEFAULT_FIELDS:
        if field not in fields:
            fields.append(field)
    return fields, rows


def build_summary(
    cli_args: argparse.Namespace,
    results_path: Path,
    best_row: dict[str, str],
    previous: dict[str, str] | None,
) -> dict[str, str]:
    summary: dict[str, str] = dict(previous or {})
    run_id = cli_args.run_id or results_path.parent.name
    epoch = best_row["epoch"]

    summary.update(
        {
            "version": run_id,
            "data": cli_args.data,
            "status": cli_args.status,
            "split": cli_args.split,
            "metric_source": relative_source(results_path),
        }
    )
    for output_field, source_field in METRIC_FIELDS.items():
        summary[output_field] = format_number(best_row[source_field], source_field)

    if cli_args.branch:
        summary["git_branch"] = cli_args.branch

    fitness = official_fitness(best_row)
    generated_note = (
        f"results.csv 中 official fitness 最高行为 epoch {epoch}（{fitness:.5f}）"
    )
    existing_notes = [summary.get("notes", ""), cli_args.notes]
    note_parts = [
        part.strip()
        for note in existing_notes
        for part in note.split("；")
        if part.strip() and not part.strip().startswith(GENERATED_NOTE_PREFIXES)
    ]
    note_parts.append(generated_note)
    summary["notes"] = "；".join(dict.fromkeys(note_parts))
    return summary


def write_table(
    table_path: Path,
    fields: list[str],
    rows: list[dict[str, Any]],
) -> None:
    table_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = table_path.with_suffix(f"{table_path.suffix}.tmp")
    with temporary_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(table_path)


def main() -> None:
    cli_args = parse_args()
    run_dir = resolve_path(cli_args.run_dir)
    results_path = run_dir / "results.csv"
    if not results_path.is_file():
        raise FileNotFoundError(f"Run 中没有 results.csv：{results_path}")

    best_row = read_best_fitness_row(results_path)
    table_path = resolve_path(cli_args.table, must_exist=False)
    fields, rows = read_table(table_path)
    run_id = cli_args.run_id or run_dir.name
    previous = next((row for row in rows if row.get("version") == run_id), None)
    summary = build_summary(cli_args, results_path, best_row, previous)

    if previous is None:
        rows.append(summary)
    else:
        rows[rows.index(previous)] = summary

    print(
        "| {version} | {status} | {epoch} | {fitness:.5f} | {mask_p} | {mask_r} | "
        "{mask_map50} | {mask_map50_95} |".format(
            epoch=best_row["epoch"],
            fitness=official_fitness(best_row),
            **summary,
        )
    )
    if cli_args.dry_run:
        print("[DRY RUN] 未修改汇总表。")
        return

    write_table(table_path, fields, rows)
    print(f"[DONE] 已更新：{table_path}")


if __name__ == "__main__":
    main()
