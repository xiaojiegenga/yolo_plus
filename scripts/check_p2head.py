"""Check P2Head transfer, shapes, loss gradients and model cost without dataset training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", help="Optional official checkpoint path/name; downloads if an official name is absent")
    args = parser.parse_args()
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops, get_num_params

    torch.set_num_threads(2)
    tests = runpy.run_path(str(ROOT / "ultralytics-main/tests/test_p2head_model.py"))
    count = 0
    for name, function in tests.items():
        if name.startswith("test_"):
            function()
            count += 1
            print(f"[PASS] {name}", flush=True)
    if args.weights:
        source = YOLO(args.weights).model
        target = tests["build_model"]()
        target.load(source)
        actual = target.state_dict()
        matched = 0
        for key, value in source.float().state_dict().items():
            dest = tests["target_key"](key)
            if actual[dest].shape == value.shape:
                assert torch.equal(actual[dest], value), dest
                matched += 1
        print(f"[PASS] official checkpoint transfer: {matched} tensors", flush=True)
    report = {"tests_passed": count, "nc": 2, "imgsz": 640, "models": {}}
    for name, is_p2 in (("baseline", False), ("p2head", True)):
        model = tests["build_model"](is_p2).eval().fuse(verbose=False)
        report["models"][name] = {"fused_params": get_num_params(model), "gflops": get_flops(model, imgsz=640)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
