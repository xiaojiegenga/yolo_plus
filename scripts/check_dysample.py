"""用合成输入检查 E 结构、官方权重迁移；--cuda 额外检查 AMP 与 GPU 反向。"""

from copy import deepcopy
from pathlib import Path
import argparse
import runpy
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ultralytics-main"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", default="yolo26m-seg.pt")
    parser.add_argument("--cuda", action="store_true", help="同时检查实际 CUDA 的 AMP、FP16 推理和分割损失反向")
    args = parser.parse_args()
    import torch
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.nn.tasks import SegmentationModel
    from ultralytics.utils.torch_utils import get_flops, get_num_params, init_seeds

    torch.set_num_threads(4)
    scope = runpy.run_path(str(ROOT / "ultralytics-main/tests/test_dysample_model.py"))
    for name, test in scope.items():
        if name.startswith("test_"):
            test()
            print(f"PASS {name}", flush=True)

    init_seeds(42, deterministic=True)
    candidate = YOLO(str(scope["MODEL"])).load(args.weights)
    source = candidate.ckpt["model"].float().state_dict()
    initial_offsets = {k: v.clone() for k, v in candidate.model.state_dict().items()
        if k.startswith(("model.11.", "model.14."))}
    model = SegmentationModel(deepcopy(candidate.model.yaml), nc=2, verbose=False)
    model.load(candidate.model)
    state = model.state_dict()
    matched = {k for k in source if k in state and source[k].shape == state[k].shape}
    changed = {k for k in source if k in state and source[k].shape != state[k].shape}
    added = set(state) - set(source)
    assert len(matched) == 890 and len(changed) == 14 and len(added) == 6
    assert not set(source) - set(state)
    assert all(".cv3." in k or ".one2one_cv3." in k or ".proto.semseg.2." in k for k in changed)
    for key in matched:
        torch.testing.assert_close(state[key], source[key], rtol=0, atol=0)
    for key, value in initial_offsets.items():
        torch.testing.assert_close(state[key], value, rtol=0, atol=0)
    print(f"PASS official weights: {len(matched)}/{len(state)}; class shape changes={len(changed)}; added={len(added)}")

    if args.cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("--cuda 需要可用的 CUDA GPU")
        model = model.cuda().train()
        model.args = get_cfg(overrides={"task": "segment", "mask_ratio": 2})
        image = torch.rand(1, 3, 160, 160, device="cuda")
        masks = torch.zeros(1, 80, 80, device="cuda")
        masks[0, 20:60, 28:40] = 1
        batch = {"img": image, "batch_idx": torch.zeros(1, device="cuda"),
            "cls": torch.zeros(1, 1, device="cuda"),
            "bboxes": torch.tensor([[.425, .5, .15, .5]], device="cuda"),
            "masks": masks, "sem_masks": (masks > 0).float()}
        with torch.autocast("cuda", dtype=torch.float16):
            loss, items = model.loss(batch)
        loss.sum().backward()
        assert torch.isfinite(loss).all() and torch.isfinite(items).all()
        for i in (11, 14):
            grad = model.model[i].offset.weight.grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        with torch.no_grad():
            pred = model.eval().half()(image.half())[0]
        assert all(torch.isfinite(t).all() for t in pred)
        print(f"PASS CUDA AMP loss/backward and FP16 prediction: {torch.cuda.get_device_name(0)}")
        model = model.float().cpu()

    model.eval().fuse(verbose=False)
    print(f"E fused Params={get_num_params(model):,}; THOP GFLOPs@640={get_flops(model, 640):.6f}")
    print("THOP 不统计 grid_sample 等函数算子；该 GFLOPs 不是完整计算量。")


if __name__ == "__main__":
    main()
