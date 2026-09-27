"""Check the RTMDet-Ins-m dataset, CUDA ops, config and weights on CPU."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "experiments/data-v2-table5-rtmdet-ins-m.py"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()

    import mmcv
    import mmdet
    from mmcv.ops import nms
    from mmdet.registry import DATASETS, MODELS
    from mmdet.utils import register_all_modules
    from mmengine.config import Config
    from mmengine.runner import load_checkpoint

    register_all_modules(init_default_scope=True)
    config = Config.fromfile(str(args.config.resolve()))
    checkpoint = Path(config.load_from)
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing RTMDet checkpoint: {checkpoint}")

    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    scores = torch.tensor([0.9])
    _, keep = nms(boxes, scores, 0.5)
    if len(keep) != 1:
        raise RuntimeError("MMCV NMS operator did not produce one box")

    train_dataset = DATASETS.build(config.train_dataloader.dataset)
    val_dataset = DATASETS.build(config.val_dataloader.dataset)
    if len(train_dataset) != 938 or len(val_dataset) != 117:
        raise ValueError("Expected data-v2 train/val image counts: 938/117")
    sample = val_dataset[0]

    model = MODELS.build(config.model)
    load_checkpoint(model, str(checkpoint), map_location="cpu", strict=False)
    model.eval()
    with torch.inference_mode():
        output = model.test_step(
            {
                "inputs": [sample["inputs"]],
                "data_samples": [sample["data_samples"]],
            }
        )
    print(f"[READY] MMCV {mmcv.__version__}, MMDetection {mmdet.__version__}")
    print(f"[READY] {len(train_dataset)} train / {len(val_dataset)} val images")
    print(f"[READY] RTMDet-Ins-m CPU inference: {len(output[0].pred_instances)} predictions")
    print("[NO GPU TRAINING STARTED]")


if __name__ == "__main__":
    main()
