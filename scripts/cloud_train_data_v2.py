"""data-v2 云服务器简化启动入口。

使用方式：
    # 10 epoch预检
    python scripts/cloud_train_data_v2.py --preflight10

    # 400 epoch正式训练
    python scripts/cloud_train_data_v2.py

公共GPU镜像必须预装可用的CUDA版PyTorch。本脚本只在其余Ultralytics依赖缺失时，
自动执行一次 ``pip install -e ultralytics-main``，然后调用统一训练入口。
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "ultralytics-main"
TRAIN_SCRIPT = REPO_ROOT / "scripts" / "train_yolo26_seg.py"


def ensure_cloud_runtime() -> None:
    """保留镜像自带CUDA PyTorch，只补齐其余项目依赖。"""
    if importlib.util.find_spec("torch") is None:
        raise RuntimeError(
            "当前镜像没有PyTorch。请重新选择预装CUDA版PyTorch的GPU镜像，"
            "不要让本脚本自动安装可能不匹配的torch。"
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
    )
    missing = [name for name in required_modules if importlib.util.find_spec(name) is None]
    if not missing:
        return

    print(f"[SETUP] 缺少依赖 {missing}，正在安装仓库内Ultralytics及其依赖...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-e", str(SOURCE_ROOT)],
        cwd=REPO_ROOT,
        check=True,
    )


def main() -> None:
    ensure_cloud_runtime()
    command = [sys.executable, str(TRAIN_SCRIPT), *sys.argv[1:]]
    raise SystemExit(subprocess.call(command, cwd=REPO_ROOT))


if __name__ == "__main__":
    main()
