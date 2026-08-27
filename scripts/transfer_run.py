"""在云服务器和本地电脑之间打包或解包完整 Run。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_ROOT = REPO_ROOT / "runs"
EXPORTS_ROOT = REPO_ROOT / "exports"
RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    pack_parser = subparsers.add_parser("pack", help="把 runs/<run-id> 打成 ZIP")
    pack_parser.add_argument("--run-id", required=True)
    pack_parser.add_argument(
        "--archive",
        type=Path,
        help="输出 ZIP；默认 exports/<run-id>.zip",
    )

    unpack_parser = subparsers.add_parser(
        "unpack",
        help="把回传 ZIP 解包到 runs/<run-id>",
    )
    unpack_parser.add_argument("--archive", type=Path, required=True)
    return parser.parse_args()


def validate_run_id(run_id: str) -> None:
    if not RUN_ID_PATTERN.fullmatch(run_id):
        raise ValueError(
            "run-id 只能包含英文字母、数字、点、下划线和连字符"
        )


def resolve_input_path(value: Path) -> Path:
    path = value.expanduser()
    candidates = (
        [path]
        if path.is_absolute()
        else [Path.cwd() / path, REPO_ROOT / path]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"找不到文件；已检查：{searched}")


def resolve_output_path(value: Path) -> Path:
    path = value.expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def pack_run(run_id: str, archive_value: Path | None) -> Path:
    validate_run_id(run_id)
    run_dir = (RUNS_ROOT / run_id).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(f"找不到 Run：{run_dir}")

    archive_path = resolve_output_path(
        archive_value or EXPORTS_ROOT / f"{run_id}.zip"
    )
    if archive_path.exists():
        raise FileExistsError(f"归档已存在，请先更名或移走：{archive_path}")
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    files = sorted(path for path in run_dir.rglob("*") if path.is_file())
    if not files:
        raise ValueError(f"Run 目录中没有文件：{run_dir}")

    with ZipFile(
        archive_path,
        mode="w",
        compression=ZIP_DEFLATED,
        allowZip64=True,
    ) as archive:
        for file_path in files:
            relative_path = file_path.relative_to(run_dir)
            archive.write(file_path, arcname=(Path(run_id) / relative_path).as_posix())

    print(f"[PACKED] {len(files)} 个文件")
    print(f"[ARCHIVE] {archive_path}")
    return archive_path


def validate_members(archive: ZipFile) -> str:
    members = archive.infolist()
    file_members = [member for member in members if not member.is_dir()]
    if not file_members:
        raise ValueError("归档中没有文件")

    roots: set[str] = set()
    for member in members:
        if "\\" in member.filename:
            raise ValueError(f"归档包含不安全路径：{member.filename}")
        path = PurePosixPath(member.filename)
        if path.is_absolute() or ".." in path.parts or not path.parts:
            raise ValueError(f"归档包含不安全路径：{member.filename}")
        if not member.is_dir():
            roots.add(path.parts[0])

    if len(roots) != 1:
        raise ValueError(f"归档必须只包含一个 Run 根目录，当前为：{sorted(roots)}")
    run_id = next(iter(roots))
    validate_run_id(run_id)
    return run_id


def unpack_run(archive_value: Path) -> Path:
    archive_path = resolve_input_path(archive_value)
    with ZipFile(archive_path, mode="r") as archive:
        run_id = validate_members(archive)
        run_dir = (RUNS_ROOT / run_id).resolve()
        if run_dir.exists():
            raise FileExistsError(f"本地 Run 已存在，不覆盖：{run_dir}")

        RUNS_ROOT.mkdir(parents=True, exist_ok=True)
        archive.extractall(RUNS_ROOT)

    print(f"[UNPACKED] {run_dir}")
    print(
        "[LOCAL NEXT] 运行 python scripts/fill_results_table.py "
        f"--run-dir runs/{run_id} --run-id {run_id}"
    )
    return run_dir


def main() -> None:
    cli_args = parse_args()
    if cli_args.command == "pack":
        pack_run(cli_args.run_id, cli_args.archive)
        return
    unpack_run(cli_args.archive)


if __name__ == "__main__":
    main()
