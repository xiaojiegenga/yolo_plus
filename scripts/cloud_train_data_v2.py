"""data-v2 云服务器启动入口。

示例：
    python scripts/cloud_train_data_v2.py --preflight10
    python scripts/cloud_train_data_v2.py --run-name data-v2-5090-baseline

脚本保留镜像自带的 CUDA PyTorch；仅在其他运行依赖缺失时安装仓库内
Ultralytics，然后调用统一训练入口。它不执行数据或文件哈希检查。
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "ultralytics-main"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_yolo26_seg.py"
LOCAL_REFINER_SCRIPT = REPO_ROOT / "scripts" / "train_local_refiner.py"


def ensure_cloud_runtime(require_coco: bool = False) -> None:
    """保留镜像自带的 CUDA PyTorch，只补齐其余项目依赖。"""
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError(
            "当前镜像没有 PyTorch。请使用预装 CUDA 版 PyTorch 的 GPU 镜像。"
        )

    required_modules = (
        "yaml",
        "cv2",
        "numpy",
        "matplotlib",
        "PIL",
        "psutil",
        "requests",
        "scipy",
        "torchvision",
        "polars",
        "thop",
    )
    missing = [
        name for name in required_modules if importlib.util.find_spec(name) is None
    ]
    if missing:
        print(f"[SETUP] 缺少依赖 {missing}，安装仓库内 Ultralytics...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-e", str(SOURCE_ROOT)],
            cwd=REPO_ROOT,
            check=True,
        )

    if require_coco and importlib.util.find_spec("pycocotools") is None:
        print("[SETUP] 安装原图 Val 依赖 pycocotools==2.0.11...")
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "pycocotools==2.0.11"],
            cwd=REPO_ROOT,
            check=True,
        )


def main() -> None:
    arguments = sys.argv[1:]
    local_refiner = "--local-refiner" in arguments
    if local_refiner:
        arguments = [argument for argument in arguments if argument != "--local-refiner"]
    ensure_cloud_runtime(require_coco=local_refiner)
    script = LOCAL_REFINER_SCRIPT if local_refiner else TRAIN_SCRIPT
    command = [sys.executable, str(script), *arguments]
    exit_code = subprocess.call(command, cwd=REPO_ROOT)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
