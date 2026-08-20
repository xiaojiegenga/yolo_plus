"""YOLO26 水稻害虫实例分割统一训练入口。

设计目标：
1. Baseline 和所有单项改进共用同一份训练参数；
2. 改进实验默认只允许改变模型源码/模型 YAML 和实验名称；
3. 默认只执行轻量必要检查，云端拉取代码和下载数据后即可启动；
4. 不在本脚本中混入 CBAM、P2、Dice 等版本专用实现。
5. data-v2 Baseline 将 batch=8 直接锁进独立 profile，不再作为临时覆盖参数；
6. 默认只校验类别、目录和文件数量；全量内容指纹作为可选严格模式。

data-v2 Baseline-b8 正式训练（400 epoch）：
    python scripts/train_yolo26_seg.py --experiment data-v2-baseline-b8

10 epoch 预检：
    python scripts/train_yolo26_seg.py --experiment data-v2-baseline-b8 --preflight10

不提供 1 epoch 训练模式。需要完整内容哈希时可额外传入 --strict-checks，
默认云端流程不执行耗时的全量文件哈希。

未来自定义模型 YAML 示例：
    python scripts/train_yolo26_seg.py ^
        --experiment v2-p2 ^
        --model ultralytics-main/ultralytics/cfg/models/26/yolo26m-p2-seg.yaml ^
        --pretrained ultralytics-main/yolo26m-seg.pt ^
        --preflight10
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

from dataset_integrity import IMAGE_SUFFIXES, audit_yolo_seg_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "ultralytics-main"
PROFILE_PATH = REPO_ROOT / "experiments" / "yolo26m_seg_data_v2_baseline_b8.yaml"
LOCAL_WEIGHTS = SOURCE_ROOT / "yolo26m-seg.pt"
DEFAULT_WEIGHTS = LOCAL_WEIGHTS if LOCAL_WEIGHTS.is_file() else REPO_ROOT / "yolo26m-seg.pt"
LOCAL_DATA = REPO_ROOT.parent / "code" / "yolo_data.yaml"
CLOUD_DATA = REPO_ROOT / "experiments" / "yolo_data_v2_cloud.yaml"
DEFAULT_DATA = LOCAL_DATA if LOCAL_DATA.is_file() else CLOUD_DATA

# 该值由 profile 中 train 字典按 JSON key 排序后计算。
# 它用于阻止训练参数被无意修改；若以后确实要研究超参数，应创建独立实验，
# 不要直接改动 Baseline 公平对比配置。
EXPECTED_TRAIN_ARGS_SHA256 = "FA16F5C3748A9B978E62EDC50E85A5F1FA014CCBA1A3382AA1378030F4F26926"
EXPECTED_PROFILE_ID = "yolo26m-seg-data-v2-baseline-b8-20260820"
EXPECTED_ULTRALYTICS_VERSION = "8.4.80"
EXPECTED_BASELINE_WEIGHT_SHA256 = "16B636F04E8FB6A325B3370F22DC5E5535FF473E384F4D041FD28D788F6EE9F5"
EXPECTED_DATASET_ID = "rice-pest-data-v2"
EXPECTED_DATASET_CONTENT_SHA256 = "02B9A2475D45CE5C88D933E0B7338235AD1622DFEB3266C2E7356EE874538C49"


def parse_args() -> argparse.Namespace:
    """解析模型身份和路径；正式训练参数全部由 data-v2 profile 锁定。"""
    parser = argparse.ArgumentParser(
        description="使用锁定的 Baseline 参数训练 YOLO26m-seg 或其单项改进模型。"
    )
    parser.add_argument(
        "--experiment",
        default="data-v2-baseline-b8",
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
        help="数据集 YAML；Windows 默认本地配置，Linux 默认 /root/yolo_data 配置。",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="可选的输出目录名；不填写时根据实验标识和时间自动生成。",
    )
    parser.add_argument(
        "--preflight10",
        action="store_true",
        help="只运行10 epoch短跑；不填写时执行profile锁定的400 epoch正式训练。",
    )
    parser.add_argument(
        "--strict-checks",
        action="store_true",
        help="可选：训练前计算全部数据和权重SHA-256；默认关闭以简化云端流程。",
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
        "dataset_id": EXPECTED_DATASET_ID,
        "dataset_content_sha256": EXPECTED_DATASET_CONTENT_SHA256,
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


def read_git_state() -> tuple[str, str]:
    """记录 Git 身份；轻量云端流程不再因工作区状态阻止训练。"""
    branch = git_output("branch", "--show-current")
    commit = git_output("rev-parse", "HEAD")
    status = git_output("status", "--porcelain", "--untracked-files=all")
    if status:
        print("[WARN] Git 工作区存在改动；训练仍会继续，commit 信息仅供记录。")
    return branch or "detached", commit


def verify_ultralytics_source() -> tuple[Any, str, Path]:
    """直接优先导入仓库源码，不要求云端重复执行 pip install -e。"""
    source_path = str(SOURCE_ROOT)
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    import ultralytics
    from ultralytics import YOLO

    package_file = Path(ultralytics.__file__).resolve()
    expected_package = (SOURCE_ROOT / "ultralytics").resolve()
    if expected_package != package_file.parent and expected_package not in package_file.parents:
        raise RuntimeError(
            "当前 Python 没有导入本项目源码，已拒绝训练。\n"
            f"实际：{package_file}\n"
            f"期望：{expected_package}\n"
            "请确认从 Git 仓库根目录执行 scripts/train_yolo26_seg.py。"
        )
    return YOLO, str(ultralytics.__version__), package_file


def verify_data_yaml(path: Path) -> tuple[dict[str, Any], str]:
    """只检查训练所需字段和类别；允许 Windows/Linux 使用不同根路径。"""
    if not path.is_file():
        raise FileNotFoundError(f"数据集 YAML 不存在：{path}")
    actual_hash = sha256_file(path)
    with path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file)
    required = {"path", "train", "val", "test", "nc", "names"}
    missing = sorted(required.difference(data or {}))
    if missing:
        raise ValueError(f"数据集 YAML 缺少字段：{', '.join(missing)}")
    if data["nc"] != 2:
        raise ValueError(f"Baseline 数据集应为 2 类，实际 nc={data['nc']}")
    return data, actual_hash


def _dataset_root(data_yaml: Path, data: dict[str, Any]) -> Path:
    """解析数据根目录，同时兼容 Windows 和 Linux YAML。"""
    root = Path(str(data["path"])).expanduser()
    if not root.is_absolute():
        root = data_yaml.parent / root
    return root.resolve()


def verify_dataset_quick(
    path: Path,
    data: dict[str, Any],
    profile: dict[str, Any],
) -> dict[str, Any]:
    """默认轻量检查：目录存在且图片/标签数量符合 data-v2 profile。"""
    root = _dataset_root(path, data)
    images_by_split: dict[str, int] = {}
    labels_by_split: dict[str, int] = {}
    for split in ("train", "val", "test"):
        image_dir = (root / str(data[split])).resolve()
        label_dir = root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(
                f"data-v2 缺少 {split} 目录：images={image_dir}, labels={label_dir}"
            )
        images_by_split[split] = sum(
            1 for item in image_dir.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES
        )
        labels_by_split[split] = sum(1 for item in label_dir.glob("*.txt") if item.is_file())

    expected = profile["dataset_summary"]
    if images_by_split != expected["images"] or labels_by_split != expected["labels"]:
        raise RuntimeError(
            "data-v2 文件数量与 profile 不一致。\n"
            f"期望图片：{expected['images']}，实际：{images_by_split}\n"
            f"期望标签：{expected['labels']}，实际：{labels_by_split}"
        )
    return {
        "dataset_id": profile["dataset_id"],
        "dataset_root": str(root),
        "dataset_content_sha256": "NOT_CHECKED_QUICK_MODE",
        "images_by_split": images_by_split,
        "labels_by_split": labels_by_split,
        "empty_labels_by_split": {},
        "objects_by_split_and_class": {},
        "parent_groups": None,
        "parent_groups_crossing_splits": None,
        "exact_image_hashes_crossing_splits": None,
        "issue_count": None,
    }


def verify_dataset_strict(path: Path, profile: dict[str, Any]) -> dict[str, Any]:
    """可选严格检查：验证完整内容哈希、标签结构和跨 split 泄漏。"""
    print("[INFO] strict-checks：正在计算 data-v2 全量内容指纹...")
    report = audit_yolo_seg_dataset(path, dataset_id=str(profile["dataset_id"]))
    expected_hash = str(profile["dataset_content_sha256"])
    if report["dataset_content_sha256"] != expected_hash:
        raise RuntimeError(
            "data-v2 内容指纹与锁定 profile 不一致，已拒绝启动。\n"
            f"期望：{expected_hash}\n"
            f"实际：{report['dataset_content_sha256']}"
        )
    if report["issue_count"]:
        raise RuntimeError(
            f"data-v2 存在 {report['issue_count']} 项结构问题，已拒绝启动："
            f"{report['issue_types']}"
        )
    if report["parent_groups_crossing_splits"]:
        raise RuntimeError(
            "data-v2 仍存在原图跨 train/val/test，已拒绝启动："
            f"{report['parent_groups_crossing_splits']} 个原图组"
        )
    if report["exact_image_hashes_crossing_splits"]:
        raise RuntimeError(
            "data-v2 存在完全相同图片跨 split，已拒绝启动："
            f"{report['exact_image_hashes_crossing_splits']} 组"
        )
    return report


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


def ensure_weight(path: Path) -> Path:
    """本地没有官方 m-seg 权重时自动下载，避免云端手动上传权重。"""
    if path.is_file():
        return path
    if path.name != "yolo26m-seg.pt":
        raise FileNotFoundError(f"模型权重不存在：{path}")
    from ultralytics.utils.downloads import attempt_download_asset

    print(f"[INFO] 未找到 {path.name}，正在从 Ultralytics 官方资产自动下载...")
    downloaded = Path(attempt_download_asset(path)).resolve()
    if not downloaded.is_file():
        raise FileNotFoundError(
            f"自动下载 {path.name} 失败，请检查云服务器网络：{downloaded}"
        )
    return downloaded


def build_model(
    yolo_class: Any,
    model_path: Path,
    pretrained_path: Path | None,
    baseline_weight_hash: str,
    strict_checks: bool,
) -> tuple[Any, str, Path, Path | None]:
    """构建 Baseline 或自定义 YAML 模型，不包含任何改进专用代码。"""
    suffix = model_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        if not model_path.is_file():
            raise FileNotFoundError(f"模型 YAML 不存在：{model_path}")
        if pretrained_path is None:
            raise ValueError("自定义模型 YAML 必须通过 --pretrained 指定 Baseline 预训练权重。")
        pretrained_path = ensure_weight(pretrained_path)
        if strict_checks:
            verify_weight(pretrained_path, baseline_weight_hash)
        model = yolo_class(str(model_path))
        model.load(str(pretrained_path))
        mode = "custom-yaml + baseline-pretrained"
    elif suffix == ".pt":
        if pretrained_path is not None:
            raise ValueError(".pt 模型不应再同时传入 --pretrained。")
        model_path = ensure_weight(model_path)
        if strict_checks:
            verify_weight(model_path, baseline_weight_hash)
        model = yolo_class(str(model_path))
        mode = "baseline-pretrained-pt"
    else:
        raise ValueError("模型文件必须是 .pt、.yaml 或 .yml。")

    if getattr(model, "task", None) != "segment":
        raise RuntimeError(f"模型任务必须是 segment，实际为：{getattr(model, 'task', None)}")
    return model, mode, model_path, pretrained_path


def make_run_name(experiment: str, custom_name: str | None, preflight10: bool) -> str:
    """生成可读、可追溯且不会覆盖旧实验的目录名。"""
    if custom_name:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", custom_name):
            raise ValueError("--run-name 只能包含字母、数字、点、下划线和连字符。")
        return custom_name

    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", experiment):
        raise ValueError("--experiment 只能使用小写字母、数字、下划线和连字符。")
    tag = "" if experiment == "baseline" else f"_{experiment.replace('-', '_')}"
    preflight_tag = "_preflight10" if preflight10 else ""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return f"yolo26m{tag}{preflight_tag}_seg_{timestamp}"


def print_summary(
    *,
    experiment: str,
    run_name: str,
    model_path: Path,
    pretrained_path: Path | None,
    model_mode: str,
    data_path: Path,
    data_yaml_hash: str,
    profile: dict[str, Any],
    train_args: dict[str, Any],
    train_hash: str,
    dataset_report: dict[str, Any],
    run_kind: str,
    checks_mode: str,
    branch: str,
    commit: str,
    package_file: Path,
) -> None:
    """在真正训练前完整显示公平对比信息。"""
    print("\n========== YOLO26 Fair Training Config ==========")
    print(f"Experiment       : {experiment}")
    print(f"Run kind         : {run_kind}")
    print(f"Checks           : {checks_mode}")
    print(f"Run name         : {run_name}")
    print(f"Model            : {model_path}")
    print(f"Model mode       : {model_mode}")
    print(f"Pretrained       : {pretrained_path or model_path}")
    print(f"Data             : {data_path}")
    print(f"Data YAML SHA256 : {data_yaml_hash}")
    print(f"Profile          : {profile['profile_id']}")
    print(f"Profile SHA256   : {train_hash}")
    print(f"Dataset ID       : {profile['dataset_id']}")
    print(f"Dataset SHA256   : {dataset_report['dataset_content_sha256']}")
    if checks_mode == "quick":
        print(f"Expected SHA256  : {profile['dataset_content_sha256']} (not recomputed)")
    print(f"Dataset images   : {dataset_report['images_by_split']}")
    print(f"Parent leakage   : {dataset_report['parent_groups_crossing_splits']}")
    print(f"Git branch       : {branch}")
    print(f"Git commit       : {commit}")
    print(f"Ultralytics path : {package_file}")
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
    print("=================================================\n")


def write_manifest(
    save_dir: Path,
    *,
    experiment: str,
    run_name: str,
    model_path: Path,
    pretrained_path: Path | None,
    data_path: Path,
    data_yaml_hash: str,
    profile: dict[str, Any],
    train_hash: str,
    effective_train_args: dict[str, Any],
    dataset_report: dict[str, Any],
    run_kind: str,
    checks_mode: str,
    branch: str,
    commit: str,
) -> None:
    """将可复现信息写入原始 run；该目录由用户后续自行备份。"""
    manifest = {
        "experiment": experiment,
        "run_name": run_name,
        "git_branch": branch,
        "git_commit": commit,
        "model": str(model_path),
        "pretrained": str(pretrained_path or model_path),
        "pretrained_sha256": (
            profile["baseline_weight_sha256"]
            if checks_mode == "strict"
            else "NOT_CHECKED_QUICK_MODE"
        ),
        "expected_pretrained_sha256": profile["baseline_weight_sha256"],
        "data": str(data_path),
        "dataset_id": profile["dataset_id"],
        "dataset_yaml_sha256": data_yaml_hash,
        "dataset_content_sha256": dataset_report["dataset_content_sha256"],
        "expected_dataset_content_sha256": profile["dataset_content_sha256"],
        "checks_mode": checks_mode,
        "dataset_summary": {
            "images_by_split": dataset_report["images_by_split"],
            "labels_by_split": dataset_report["labels_by_split"],
            "empty_labels_by_split": dataset_report["empty_labels_by_split"],
            "objects_by_split_and_class": dataset_report["objects_by_split_and_class"],
            "parent_groups": dataset_report["parent_groups"],
            "parent_groups_crossing_splits": dataset_report["parent_groups_crossing_splits"],
            "exact_image_hashes_crossing_splits": dataset_report["exact_image_hashes_crossing_splits"],
            "issue_count": dataset_report["issue_count"],
        },
        "baseline_profile": profile["profile_id"],
        "train_args_sha256": train_hash,
        "effective_train_args_sha256": canonical_hash(effective_train_args),
        "run_kind": run_kind,
        "formal_comparison_eligible": run_kind == "formal",
        "paired_comparison_group": profile["paired_comparison_group"],
        "profile_epochs": int(profile["train"]["epochs"]),
        "effective_epochs": int(effective_train_args["epochs"]),
        "profile_batch": int(profile["train"]["batch"]),
        "effective_batch": int(effective_train_args["batch"]),
        "runtime_overrides": {"epochs": 10} if run_kind == "preflight-10-epoch" else {},
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
    run_kind = "preflight-10-epoch" if args.preflight10 else "formal"
    checks_mode = "strict" if args.strict_checks else "quick"
    effective_train_args = dict(train_args)
    if args.preflight10:
        effective_train_args["epochs"] = 10
    run_name = make_run_name(args.experiment, args.run_name, args.preflight10)

    branch, commit = read_git_state()
    yolo_class, version, package_file = verify_ultralytics_source()

    expected_version = str(profile["ultralytics_version"])
    if version != expected_version:
        raise RuntimeError(
            f"Ultralytics 版本不一致：期望 {expected_version}，实际 {version}。"
        )

    data, data_yaml_hash = verify_data_yaml(data_path)
    dataset_report = (
        verify_dataset_strict(data_path, profile)
        if args.strict_checks
        else verify_dataset_quick(data_path, data, profile)
    )
    model, model_mode, model_path, pretrained_path = build_model(
        yolo_class,
        model_path,
        pretrained_path,
        str(profile["baseline_weight_sha256"]),
        args.strict_checks,
    )

    print_summary(
        experiment=args.experiment,
        run_name=run_name,
        model_path=model_path,
        pretrained_path=pretrained_path,
        model_mode=model_mode,
        data_path=data_path,
        data_yaml_hash=data_yaml_hash,
        profile=profile,
        train_args=effective_train_args,
        train_hash=train_hash,
        dataset_report=dataset_report,
        run_kind=run_kind,
        checks_mode=checks_mode,
        branch=branch,
        commit=commit,
        package_file=package_file,
    )

    runtime_args = dict(effective_train_args)
    runtime_args.update(data=str(data_path), name=run_name)
    results = model.train(**runtime_args)
    save_dir = Path(results.save_dir).resolve()

    write_manifest(
        save_dir,
        experiment=args.experiment,
        run_name=run_name,
        model_path=model_path,
        pretrained_path=pretrained_path,
        data_path=data_path,
        data_yaml_hash=data_yaml_hash,
        profile=profile,
        train_hash=train_hash,
        effective_train_args=effective_train_args,
        dataset_report=dataset_report,
        run_kind=run_kind,
        checks_mode=checks_mode,
        branch=branch,
        commit=commit,
    )

    best_path = save_dir / "weights" / "best.pt"
    print("\n[INFO] Training complete.")
    print(f"[INFO] Artifacts: {save_dir}")
    if args.preflight10:
        print("[NOTICE] 本次是10 epoch预检，只用于确认云端流程，不写入正式指标表。")
    else:
        print("[NOTICE] 本次是 data-v2 / batch=8 正式 Baseline。")
    print("[INFO] 正式对比请使用 best.pt 单独执行 split=val：")
    print(f"  yolo segment val model=\"{best_path}\" data=\"{data_path}\" split=val")


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"[ERROR] {error}", file=sys.stderr)
        raise SystemExit(1) from error
