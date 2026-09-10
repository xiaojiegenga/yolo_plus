"""用 CPU 合成输入检查 D1 结构、训练入口和官方预训练权重迁移。"""

from __future__ import annotations

import argparse
import runpy
import sys
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "ultralytics-main"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="yolo26m-seg.pt")
    args = parser.parse_args()

    import torch
    from ultralytics import YOLO
    from ultralytics.nn.tasks import SegmentationModel
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    torch.set_num_threads(4)
    scope = runpy.run_path(str(REPO_ROOT / "ultralytics-main/tests/test_p2proto_model.py"))
    for name, test in scope.items():
        if name.startswith("test_"):
            test()
            print(f"PASS {name}", flush=True)

    config = REPO_ROOT / "ultralytics-main/ultralytics/cfg/models/26/yolo26m-p2proto-seg.yaml"
    candidate = YOLO(str(config)).load(args.weights)
    source = candidate.ckpt["model"].float().state_dict()
    # Match the trainer's 80-to-2 class reconstruction and loading path.
    model = SegmentationModel(deepcopy(candidate.model.yaml), nc=2, verbose=False)
    model.load(candidate.model)
    state = model.state_dict()
    matched = {key for key in source if key in state and source[key].shape == state[key].shape}
    for key in matched:
        torch.testing.assert_close(state[key], source[key], rtol=0, atol=0)
    changed = {key for key in source if key in state and source[key].shape != state[key].shape}
    assert "model.23.proto.cv3.conv.weight" in changed
    assert all(".proto.semseg.2." in key or ".cv3." in key or ".one2one_cv3." in key for key in changed)
    added = set(state) - set(source)
    assert all(key.startswith(tuple(f"model.23.proto.{name}." for name in ("p2_proj", "hi_up", "hi_cv"))) for key in added)
    assert not set(source) - set(state)
    print(f"Official weights: {len(matched)}/{len(state)}; shape changes={len(changed)}; added={len(added)}")
    model.eval().fuse(verbose=False)
    print(f"D1 fused Params={get_num_params(model):,}; GFLOPs@640={get_flops(model, 640):.6f}")
    print("PASS official weights, 2-class reconstruction and fusion")


if __name__ == "__main__":
    main()
