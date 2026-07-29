"""YOLO26 水稻害虫实例分割统一训练入口。

设计目标：
1. Baseline 和所有单项改进共用同一份训练参数；
2. 改进实验只允许改变模型源码/模型 YAML 和实验名称；
3. 在正式训练前检查源码路径、数据、权重、Git 状态和配置指纹；
4. 不在本脚本中混入 CBAM、P2、Dice 等版本专用实现。

正式训练示例：
    python scripts/train_yolo26_seg.py --experiment baseline

只检查、不训练：
    python scripts/train_yolo26_seg.py --experiment baseline --dry-run

手动预检（非正式实验，只允许 1 或 10 epoch）：
    python scripts/train_yolo26_seg.py --experiment v2-p2 ... --preflight-epochs 1

未来自定义模型 YAML 示例：
    python scripts/train_yolo26_seg.py ^
        --experiment v2-p2 ^
        --model ultralytics-main/ultralytics/cfg/models/26/yolo26m-p2-seg.yaml ^
        --pretrained ultralytics-main/yolo26m-seg.pt ^
        --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "ultralytics-main"
PROFILE_PATH = REPO_ROOT / "experiments" / "yolo26m_seg_baseline_train.yaml"
DEFAULT_WEIGHTS = SOURCE_ROOT / "yolo26m-seg.pt"
DEFAULT_DATA = REPO_ROOT.parent / "code" / "yolo_data.yaml"

# 该值由 profile 中 train 字典按 JSON key 排序后计算。
# 它用于阻止训练参数被无意修改；若以后确实要研究超参数，应创建独立实验，
# 不要直接改动 Baseline 公平对比配置。
EXPECTED_TRAIN_ARGS_SHA256 = "FA16F5C3748A9B978E62EDC50E85A5F1FA014CCBA1A3382AA1378030F4F26926"
EXPECTED_PROFILE_ID = "yolo26m-seg-baseline-20260628"
EXPECTED_ULTRALYTICS_VERSION = "8.4.80"
EXPECTED_BASELINE_WEIGHT_SHA256 = "16B636F04E8FB6A325B3370F22DC5E5535FF473E384F4D041FD28D788F6EE9F5"
EXPECTED_DATASET_YAML_SHA256 = "75996638EB9BBAED8B80D0413FFD57B374C0024B2C9F9EF5689AD90B5ADF78AF"


def parse_args() -> argparse.Namespace:
    """解析只影响模型身份和文件路径的参数，不开放训练超参数覆盖。"""
    parser = argparse.ArgumentParser(
        description="使用锁定的 Baseline 参数训练 YOLO26m-seg 或其单项改进模型。"
    )
    parser.add_argument(
        "--experiment",
        default="baseline",
        help="实验标识，例如 baseline、v2-p2、v3-dice；只用于记录和运行目录命名。",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_WEIGHTS),
        help="Baseline .pt 或改进模型 YAML。相对路径按项目根目录解析。",
    )
    parser.add_argument(
        "--pretrained",
        default=None,
        help="自定义模型 YAML 使用的预训练权重；YAML 模型默认要求提供。",
    )
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA),
        help="数据集 YAML；内容必须与 Baseline 数据配置哈希一致。",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="可选的输出目录名；不填写时根据实验标识和时间自动生成。",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="完成全部检查并构建模型，但不启动训练。",
    )
    parser.add_argument(
        "--preflight-epochs",
        type=int,
        choices=(1, 10),
        default=None,
        help=(
            "仅供用户手动启动的非正式预检训练，可选 1 或 10；除 epochs 外仍使用锁定 Baseline 参数。"
            "不填写时为 400 epoch 正式实验。"
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """分块计算文件 SHA-256，避免一次读入大型权重。"""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def canonical_hash(data: dict[str, Any]) -> str:
    """计算与 YAML 排版和换行无关的参数指纹。"""
    payload = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def resolve_path(value: str | Path) -> Path:
    """将相对路径统一按项目根目录解析。"""
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = REPO_ROOT / path
    return path.resolve()


def load_baseline_profile() -> tuple[dict[str, Any], dict[str, Any], str]:
    """读取并校验唯一的 Baseline 训练参数配置。"""
    if not PROFILE_PATH.is_file():
        raise FileNotFoundError(f"Baseline profile 不存在：{PROFILE_PATH}")

    with PROFILE_PATH.open("r", encoding="utf-8") as file:
        profile = yaml.safe_load(file)

    if not isinstance(profile, dict) or not isinstance(profile.get("train"), dict):
        raise ValueError("Baseline profile 必须包含 train 字典。")

    expected_metadata = {
        "profile_id": EXPECTED_PROFILE_ID,
        "ultralytics_version": EXPECTED_ULTRALYTICS_VERSION,
        "baseline_weight_sha256": EXPECTED_BASELINE_WEIGHT_SHA256,
        "dataset_yaml_sha256": EXPECTED_DATASET_YAML_SHA256,
    }
    metadata_diff = {
        key: (expected, profile.get(key))
        for key, expected in expected_metadata.items()
        if profile.get(key) != expected
    }
    if metadata_diff:
        raise RuntimeError(f"Baseline profile 元数据不一致：{metadata_diff}")

    train_args = profile["train"]
    train_hash = canonical_hash(train_args)
    if train_hash != EXPECTED_TRAIN_ARGS_SHA256:
        raise RuntimeError(
            "Baseline 训练参数指纹不一致，已拒绝启动。\n"
            f"期望：{EXPECTED_TRAIN_ARGS_SHA256}\n"
            f"实际：{train_hash}\n"
            "如果是有意研究超参数，请新建独立实验配置，不要修改公平对比基准。"
        )
    return profile, train_args, train_hash


def git_output(*args: str) -> str:
    """读取当前项目 Git 信息。"""
    result = subprocess.run(
        ["git", *args],
        cwd=REPO_ROOT,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def verify_git_state(dry_run: bool) -> tuple[str, str]:
    """正式训练必须对应一个干净、可回溯的源码提交。"""
    branch = git_output("branch", "--show-current")
    commit = git_output("rev-parse", "HEAD")
    # ignored 的本地笔记不会出现，但未跟踪的源码/YAML 必须被发现，
    # 否则可能训练出无法用某个 commit 复现的模型。
    status = git_output("status", "--porcelain", "--untracked-files=all")

    if status and not dry_run:
        raise RuntimeError(
            "Git 中存在未提交的受跟踪改动，正式训练已拒绝启动。\n"
            "请先检查、提交源码和模型配置，再重新训练。\n"
            f"当前状态：\n{status}"
        )
    if status and dry_run:
        print("[WARN] dry-run 检测到未提交改动；正式训练时必须先提交。")
    return branch or "detached", commit


def verify_ultralytics_source() -> tuple[Any, str, Path]:
    """确保训练导入的是本项目可编辑安装源码，而不是其他环境中的副本。"""
    import ultralytics
    from ultralytics import YOLO

    package_file = Path(ultralytics.__file__).resolve()
    expected_package = (SOURCE_ROOT / "ultralytics").resolve()
    if expected_package != package_file.parent and expected_package not in package_file.parents:
        raise RuntimeError(
            "当前 Python 没有导入本项目源码，已拒绝训练。\n"
            f"实际：{package_file}\n"
            f"期望：{expected_package}\n"
            f"请在 yolo26 环境中执行：cd {SOURCE_ROOT} && pip install -e ."
        )
    return YOLO, str(ultralytics.__version__), package_file


def verify_data_yaml(path: Path, expected_hash: str) -> dict[str, Any]:
    """检查数据配置没有在不同实验之间发生变化。"""
    if not path.is_file():
        raise FileNotFoundError(f"数据集 YAML 不存在：{path}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            "数据集 YAML 与 Baseline 不一致，已拒绝启动。\n"
            f"期望：{expected_hash}\n"
            f"实际：{actual_hash}"
        )
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    required = {"path", "train", "val", "test", "nc", "names"}
    missing = sorted(required.difference(data or {}))
    if missing:
        raise ValueError(f"数据集 YAML 缺少字段：{', '.join(missing)}")
    if data["nc"] != 2:
        raise ValueError(f"Baseline 数据集应为 2 类，实际 nc={data['nc']}")
    return data


def verify_weight(path: Path, expected_hash: str) -> str:
    """检查所有对比实验从同一份 YOLO26m-seg 预训练权重出发。"""
    if not path.is_file():
        raise FileNotFoundError(f"预训练权重不存在：{path}")
    actual_hash = sha256_file(path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            "预训练权重与 Baseline 不一致，已拒绝启动。\n"
            f"期望：{expected_hash}\n"
            f"实际：{actual_hash}"
        )
    return actual_hash


def build_model(
    yolo_class: Any,
    model_path: Path,
    pretrained_path: Path | None,
    baseline_weight_hash: str,
) -> tuple[Any, str, dict[str, Any] | None]:
    """构建 Baseline 或自定义 YAML 模型，不包含任何改进专用代码。"""
    if not model_path.is_file():
        raise FileNotFoundError(f"模型文件不存在：{model_path}")

    suffix = model_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if pretrained_path is None:
            raise ValueError("自定义模型 YAML 必须通过 --pretrained 指定 Baseline 预训练权重。")
        verify_weight(pretrained_path, baseline_weight_hash)
        model = yolo_class(str(model_path))
        model.load(str(pretrained_path))
        mode = "custom-yaml + baseline-pretrained"
    elif suffix == ".pt":
        if pretrained_path is not None:
            raise ValueError(".pt 模型不应再同时传入 --pretrained。")
        verify_weight(model_path, baseline_weight_hash)
        model = yolo_class(str(model_path))
        mode = "baseline-pretrained-pt"
    else:
        raise ValueError("模型文件必须是 .pt、.yaml 或 .yml。")

    if getattr(model, "task", None) != "segment":
        raise RuntimeError(f"模型任务必须是 segment，实际为：{getattr(model, 'task', None)}")

    head = model.model.model[-1]
    transfer_report = getattr(head, "pretrained_transfer_report", None)
    if getattr(head, "requires_pretrained_head_transfer", False) and not transfer_report:
        raise RuntimeError(
            f"{head.__class__.__name__} 要求语义化 Head 权重迁移，但没有生成迁移报告，已拒绝继续。"
        )
    return model, mode, transfer_report


def make_run_name(experiment: str, custom_name: str | None, preflight_epochs: int | None = None) -> str:
    """生成可读、可追溯且不会覆盖旧实验的目录名。"""
    if custom_name:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", custom_name):
            raise ValueError("--run-name 只能包含字母、数字、点、下划线和连字符。")
        return custom_name

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", experiment):
        raise ValueError("--experiment 只能使用小写字母、数字、下划线和连字符。")
    tag = "" if experiment == "baseline" else f"_{experiment.replace('-', '_')}"
    gate_tag = "" if preflight_epochs is None else f"_preflight{preflight_epochs}"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"yolo26m{tag}{gate_tag}_seg_{timestamp}"


def print_summary(
    *,
    experiment: str,
    run_name: str,
    model_path: Path,
    pretrained_path: Path | None,
    model_mode: str,
    data_path: Path,
    profile: dict[str, Any],
    train_args: dict[str, Any],
    train_hash: str,
    branch: str,
    commit: str,
    package_file: Path,
    transfer_report: dict[str, Any] | None,
    run_kind: str,
    profile_epochs: int,
) -> None:
    """在真正训练前完整显示公平对比信息。"""
    print("\n========== YOLO26 Fair Training Config ==========")
    print(f"Experiment       : {experiment}")
    print(f"Run kind         : {run_kind}")
    print(f"Run name         : {run_name}")
    print(f"Model            : {model_path}")
    print(f"Model mode       : {model_mode}")
    print(f"Pretrained       : {pretrained_path or model_path}")
    print(f"Data             : {data_path}")
    print(f"Profile          : {profile['profile_id']}")
    print(f"Profile SHA256   : {train_hash}")
    print(f"Git branch       : {branch}")
    print(f"Git commit       : {commit}")
    print(f"Ultralytics path : {package_file}")
    if run_kind != "formal":
        print(
            f"[NOTICE] 非正式预检：锁定 profile 的 epochs={profile_epochs}，"
            f"本次仅运行 epochs={train_args['epochs']}，结果不得写入正式对比表。"
        )
    print("-------------------------------------------------")
    for key in (
        "imgsz",
        "epochs",
        "batch",
        "optimizer",
        "seed",
        "deterministic",
        "device",
        "workers",
        "mask_ratio",
        "mosaic",
        "mixup",
        "copy_paste",
    ):
        print(f"{key:17}: {train_args[key]}")
    if transfer_report:
        print("-------------------------------------------------")
        print("Pretrained head transfer:")
        print(json.dumps(transfer_report, ensure_ascii=False, indent=2))
    print("=================================================\n")


def write_manifest(
    save_dir: Path,
    *,
    experiment: str,
    run_name: str,
    model_path: Path,
    pretrained_path: Path | None,
    data_path: Path,
    profile: dict[str, Any],
    train_hash: str,
    branch: str,
    commit: str,
    transfer_report: dict[str, Any] | None,
    run_kind: str,
    profile_epochs: int,
    effective_epochs: int,
) -> None:
    """将可复现信息写入原始 run；该目录由用户后续自行备份。"""
    manifest = {
        "experiment": experiment,
        "run_name": run_name,
        "git_branch": branch,
        "git_commit": commit,
        "model": str(model_path),
        "pretrained": str(pretrained_path or model_path),
        "pretrained_sha256": profile["baseline_weight_sha256"],
        "data": str(data_path),
        "dataset_yaml_sha256": profile["dataset_yaml_sha256"],
        "baseline_profile": profile["profile_id"],
        "train_args_sha256": train_hash,
        "run_kind": run_kind,
        "formal_comparison_eligible": run_kind == "formal",
        "profile_epochs": profile_epochs,
        "effective_epochs": effective_epochs,
        "runtime_overrides": {} if run_kind == "formal" else {"epochs": effective_epochs},
        "pretrained_head_transfer": transfer_report,
    }
    path = save_dir / "experiment_manifest.json"
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Experiment manifest: {path}")


def main() -> None:
    args = parse_args()
    model_path = resolve_path(args.model)
    pretrained_path = resolve_path(args.pretrained) if args.pretrained else None
    data_path = resolve_path(args.data)

    profile, train_args, train_hash = load_baseline_profile()
    effective_train_args = dict(train_args)
    run_kind = "formal"
    if args.preflight_epochs is not None:
        effective_train_args["epochs"] = args.preflight_epochs
        run_kind = f"preflight-{args.preflight_epochs}-epoch"
    run_name = make_run_name(args.experiment, args.run_name, args.preflight_epochs)

    branch, commit = verify_git_state(args.dry_run)
    yolo_class, version, package_file = verify_ultralytics_source()

    expected_version = str(profile["ultralytics_version"])
    if version != expected_version:
        raise RuntimeError(
            f"Ultralytics 版本不一致：期望 {expected_version}，实际 {version}。"
        )

    verify_data_yaml(data_path, str(profile["dataset_yaml_sha256"]))
    model, model_mode, transfer_report = build_model(
        yolo_class,
        model_path,
        pretrained_path,
        str(profile["baseline_weight_sha256"]),
    )

    print_summary(
        experiment=args.experiment,
        run_name=run_name,
        model_path=model_path,
        pretrained_path=pretrained_path,
        model_mode=model_mode,
        data_path=data_path,
        profile=profile,
        train_args=effective_train_args,
        train_hash=train_hash,
        branch=branch,
        commit=commit,
        package_file=package_file,
        transfer_report=transfer_report,
        run_kind=run_kind,
        profile_epochs=int(train_args["epochs"]),
    )

    if args.dry_run:
        model.info(verbose=False)
        print("[DRY-RUN] 所有检查通过，未启动训练。")
        return

    # 复制后再加入运行时路径参数，避免修改从 profile 读取的锁定字典。
    runtime_args = dict(effective_train_args)
    runtime_args.update(data=str(data_path), name=run_name)
    model.train(**runtime_args)
    if model.trainer is None or not getattr(model.trainer, "save_dir", None):
        raise RuntimeError("训练结束后没有取得 trainer.save_dir，无法写入实验清单。")
    save_dir = Path(model.trainer.save_dir).resolve()

    write_manifest(
        save_dir,
        experiment=args.experiment,
        run_name=run_name,
        model_path=model_path,
        pretrained_path=pretrained_path,
        data_path=data_path,
        profile=profile,
        train_hash=train_hash,
        branch=branch,
        commit=commit,
        transfer_report=transfer_report,
        run_kind=run_kind,
        profile_epochs=int(train_args["epochs"]),
        effective_epochs=int(effective_train_args["epochs"]),
    )

    best_path = save_dir / "weights" / "best.pt"
    print("\n[INFO] Training complete.")
    print(f"[INFO] Artifacts: {save_dir}")
    if run_kind != "formal":
        print("[NOTICE] 本次是非正式预检，只用于排错，不能写入 Baseline/V2 正式指标对比。")
    print("[INFO] 正式对比请使用 best.pt 单独执行 split=val：")
    print(f"  yolo segment val model=\"{best_path}\" data=\"{data_path}\" split=val")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        raise SystemExit(1) from error
