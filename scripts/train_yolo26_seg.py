"""使用简化 YAML 配置启动 YOLO 实例分割训练。"""

from __future__ import annotations

import argparse
import re
import sys
from datetime import datetime
from pathlib import Path, PureWindowsPath
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "ultralytics-main"
DEFAULT_CONFIG = REPO_ROOT / "experiments" / "yolo26m_seg_5090.yaml"
RUN_NAME_PATTERN = re.compile(r"^[A-Za-z0-9._-]+$")
SCP_HOST = "gtm-adl-westc-connect.seetacloud.com"
LOCAL_EXPORTS_ROOT = PureWindowsPath(
    r"E:\study\graduate_sec\论文撰写\模型训练\exports"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="实验参数 YAML，默认使用 experiments/yolo26m_seg_5090.yaml",
    )
    parser.add_argument("--experiment", help="覆盖配置中的 experiment")
    parser.add_argument("--model", help="覆盖配置中的 model")
    parser.add_argument("--pretrained", help="自定义模型 YAML 对应的预训练权重")
    parser.add_argument("--data", help="覆盖配置中的 data YAML")
    parser.add_argument("--run-name", help="输出目录名；不填时自动附加时间")
    parser.add_argument(
        "--preflight10",
        action="store_true",
        help="把 epochs 临时改为 10，用于短预检",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示最终参数，不导入 Ultralytics 或启动训练",
    )
    return parser.parse_args()


def resolve_existing_path(value: str | Path, base: Path = REPO_ROOT) -> Path:
    path = Path(value).expanduser()
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, base / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    searched = ", ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"找不到文件：{value}；已检查：{searched}")


def load_config(path_value: Path) -> tuple[Path, dict[str, Any]]:
    config_path = resolve_existing_path(path_value)
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"配置必须是 YAML 映射：{config_path}")
    if not isinstance(config.get("train"), dict):
        raise ValueError(f"配置缺少 train 映射：{config_path}")
    return config_path, config


def validate_data_yaml(path: Path) -> None:
    with path.open("r", encoding="utf-8") as file:
        data_config = yaml.safe_load(file)

    if not isinstance(data_config, dict):
        raise ValueError(f"数据 YAML 格式错误：{path}")
    if data_config.get("nc") != 2:
        raise ValueError(f"数据 YAML 的 nc 应为 2，当前为 {data_config.get('nc')!r}")

    names = data_config.get("names")
    if not isinstance(names, (dict, list)) or len(names) != 2:
        raise ValueError("数据 YAML 必须定义两个类别名称")


def resolve_model(value: str) -> str:
    path = Path(value).expanduser()
    if path.is_absolute() and path.exists():
        return str(path.resolve())

    local_path = REPO_ROOT / path
    if local_path.exists():
        return str(local_path.resolve())

    # 官方模型名由 Ultralytics 自行解析或下载。
    return value


def create_run_name(
    requested_name: str | None,
    experiment: str,
    preflight: bool,
) -> str:
    if requested_name:
        run_name = requested_name
    else:
        suffix = "-preflight" if preflight else ""
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{experiment}{suffix}-{timestamp}"

    if not RUN_NAME_PATTERN.fullmatch(run_name):
        raise ValueError(
            "run-name 只能包含英文字母、数字、点、下划线和连字符"
        )
    return run_name


def build_runtime(
    cli_args: argparse.Namespace,
) -> tuple[Path, str, str | None, dict[str, Any]]:
    config_path, config = load_config(cli_args.config)

    experiment = cli_args.experiment or config.get("experiment")
    model = cli_args.model or config.get("model")
    data = cli_args.data or config.get("data")
    if not isinstance(experiment, str) or not experiment:
        raise ValueError("配置必须提供非空的 experiment")
    if not isinstance(model, str) or not model:
        raise ValueError("配置必须提供非空的 model")
    if not isinstance(data, str) or not data:
        raise ValueError("配置必须提供非空的 data")

    data_path = resolve_existing_path(data)
    validate_data_yaml(data_path)

    runtime = dict(config["train"])
    project_value = runtime.get("project", "runs")
    project_path = Path(str(project_value)).expanduser()
    if not project_path.is_absolute():
        project_path = REPO_ROOT / project_path
    project_path = project_path.resolve()

    run_name = create_run_name(cli_args.run_name, experiment, cli_args.preflight10)
    if cli_args.preflight10:
        runtime["epochs"] = 10

    # 顶层字段明确映射到 Ultralytics 参数，避免把元数据误传给 train()。
    runtime["data"] = str(data_path)
    runtime["project"] = str(project_path)
    runtime["name"] = run_name
    runtime["exist_ok"] = False

    run_path = project_path / run_name
    if run_path.exists():
        raise FileExistsError(f"Run 已存在，请更换 run-name：{run_path}")

    pretrained = cli_args.pretrained
    if pretrained:
        pretrained = str(resolve_existing_path(pretrained))

    print(f"[CONFIG] {config_path}")
    print(f"[MODEL] {resolve_model(model)}")
    print(f"[OUTPUT] {run_path}")
    print("[TRAIN ARGS]")
    print(yaml.safe_dump(runtime, allow_unicode=True, sort_keys=False).rstrip())
    return run_path, resolve_model(model), pretrained, runtime


def run_training(
    model_source: str,
    pretrained: str | None,
    runtime: dict[str, Any],
) -> Path:
    if not SOURCE_ROOT.is_dir():
        raise FileNotFoundError(f"找不到 Ultralytics 源码目录：{SOURCE_ROOT}")
    sys.path.insert(0, str(SOURCE_ROOT))

    from ultralytics import YOLO, __version__

    print(f"[ULTRALYTICS] {__version__}")
    model = YOLO(model_source)
    if pretrained:
        model.load(pretrained)

    result = model.train(**runtime)
    save_dir = getattr(result, "save_dir", None)
    return Path(save_dir) if save_dir else Path(runtime["project"]) / runtime["name"]


def print_transfer_commands(save_dir: Path) -> None:
    """打印云端打包、本地 SCP 下载和解包命令。"""
    run_id = save_dir.name
    archive_name = f"{run_id}.zip"
    remote_archive = (REPO_ROOT / "exports" / archive_name).as_posix()
    local_archive = LOCAL_EXPORTS_ROOT / archive_name

    print("[NEXT 1/3 - CLOUD PACK]")
    print(f"python scripts/transfer_run.py pack --run-id {run_id}")
    print("[NEXT 2/3 - LOCAL SCP]")
    print("在本地 PowerShell 运行，并把 SCP_PORT 替换为实例 SSH 端口：")
    print(
        f'scp -P SCP_PORT root@{SCP_HOST}:{remote_archive} '
        f'"{local_archive}"'
    )
    print("[NEXT 3/3 - LOCAL UNPACK]")
    print(
        "python scripts/transfer_run.py unpack "
        f'--archive "exports/{archive_name}"'
    )


def main() -> None:
    cli_args = parse_args()
    expected_run_path, model_source, pretrained, runtime = build_runtime(cli_args)
    if cli_args.dry_run:
        print("[DRY RUN] 参数读取成功，未启动训练。")
        return

    save_dir = run_training(model_source, pretrained, runtime)
    print(f"[DONE] 原始结果：{save_dir.resolve()}")
    print_transfer_commands(save_dir.resolve())
    if save_dir.resolve() != expected_run_path.resolve():
        print(f"[NOTE] 预期输出目录为：{expected_run_path}")


if __name__ == "__main__":
    main()
