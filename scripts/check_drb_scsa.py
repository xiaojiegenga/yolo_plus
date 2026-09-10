"""检查 DRB/SCSA 的结构、迁移、融合和分割反向；不启动数据集训练。"""

from __future__ import annotations

import argparse
import io
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "experiments/data-v2-abl-drb-scsa-p34-b16-s42.yaml"
MODEL = ROOT / "ultralytics-main/ultralytics/cfg/models/26/yolo26m-drb-scsa-seg.yaml"
CACHE = ROOT / ".cache/ultralytics"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ["YOLO_CONFIG_DIR"] = str(CACHE)
sys.path.insert(0, str(ROOT / "ultralytics-main"))

import torch
import yaml

from ultralytics import YOLO
from ultralytics.nn.modules import C3k2DRB, DilatedReparamBlock, ZeroInitResidualSCSA
from ultralytics.nn.modules.block import Bottleneck, C2PSA, PSABlock
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT
from ultralytics.utils.torch_utils import get_flops

DEVICE = "cpu"
WEIGHTS = None


def make_model(nc: int = 2) -> SegmentationModel:
    return SegmentationModel(str(MODEL), nc=nc, verbose=False)


def tensor_leaves(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from tensor_leaves(item)


class DRBSCSATests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)

    def test_drb_fusion(self) -> None:
        block = DilatedReparamBlock(16).to(DEVICE)
        for _ in range(3):
            block(torch.randn(2, 16, 20, 20, device=DEVICE))
        block.eval()
        x = torch.randn(2, 16, 20, 20, device=DEVICE)
        with torch.no_grad():
            expected = block(x)
            block.fuse()
            torch.testing.assert_close(block(x), expected, atol=2e-5, rtol=2e-5)

    def test_attention_identity_and_gradients(self) -> None:
        block = ZeroInitResidualSCSA(32).to(DEVICE)
        x = torch.randn(2, 32, 16, 16, device=DEVICE, requires_grad=True)
        y = block(x)
        torch.testing.assert_close(x, y, atol=0, rtol=0)
        y.square().mean().backward()
        self.assertGreater(block.beta.grad.abs().item(), 0)
        self.assertEqual(block.scsa.q.weight.grad.abs().sum().item(), 0)
        block.zero_grad(set_to_none=True)
        with torch.no_grad():
            block.beta.fill_(0.1)
        block(x).square().mean().backward()
        self.assertGreater(block.scsa.q.weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in block.parameters()))

    def test_structure_and_640_shapes(self) -> None:
        model = make_model().to(DEVICE).eval()
        stages = [2, 4, 7, 10, 15, 18, 21, 24]
        self.assertEqual([i for i, m in enumerate(model.model) if isinstance(m, C3k2DRB)], stages)
        self.assertEqual([i for i, m in enumerate(model.model) if isinstance(m, ZeroInitResidualSCSA)], [5, 8])
        for i in stages:
            self.assertFalse(any(isinstance(m, Bottleneck) for m in model.model[i].modules()))
        self.assertEqual(sum(isinstance(m, DilatedReparamBlock) for m in model.modules()), 15)
        self.assertIsInstance(model.model[12], C2PSA)
        self.assertIsInstance(model.model[24].m[0][1], PSABlock)
        self.assertEqual(model.model[14].f, [-1, 8])
        self.assertEqual(model.model[17].f, [-1, 5])
        self.assertEqual(model.model[-1].f, [18, 21, 24])
        self.assertEqual(model.stride.tolist(), [8, 16, 32])
        shapes = {}
        handles = []
        for i in (5, 8, 18, 21, 24):
            def capture(module, inputs, output, index=i):
                shapes[index] = list(output.shape)
            handles.append(model.model[i].register_forward_hook(capture))
        with torch.no_grad():
            output = model(torch.randn(1, 3, 640, 640, device=DEVICE))
        for handle in handles:
            handle.remove()
        self.assertEqual(shapes, {5: [1, 512, 80, 80], 8: [1, 512, 40, 40],
                                  18: [1, 256, 80, 80], 21: [1, 512, 40, 40], 24: [1, 512, 20, 20]})
        self.assertEqual(list(output[0][1].shape), [1, 32, 160, 160])
        self.assertTrue(all(torch.isfinite(t).all() for t in tensor_leaves(output[0])))

    def test_pretrained_and_rebuild(self) -> None:
        source = YOLO(WEIGHTS).model if WEIGHTS else SegmentationModel("yolo26m-seg.yaml", verbose=False)
        target = make_model(nc=80)
        target.load(source, verbose=True)
        layer_map = target.yaml["baseline_layer_map"]
        state = target.state_dict()
        matched = 0
        for key, tensor in source.float().state_dict().items():
            _, layer, suffix = key.split(".", 2)
            destination = f"model.{layer_map[int(layer)]}.{suffix}"
            if destination in state and state[destination].shape == tensor.shape:
                torch.testing.assert_close(state[destination], tensor, atol=0, rtol=0)
                matched += 1
        print(f"[TRANSFER] matched tensors: {matched}; official weights: {bool(WEIGHTS)}")
        # Trainer reconstructs nc=2 from the already-remapped 80-class model.
        rebuilt = make_model(nc=2)
        rebuilt.load(target, verbose=False)
        for key, tensor in target.state_dict().items():
            if key in rebuilt.state_dict() and rebuilt.state_dict()[key].shape == tensor.shape:
                torch.testing.assert_close(rebuilt.state_dict()[key], tensor, atol=0, rtol=0)
        torch.testing.assert_close(rebuilt.model[12].cv1.conv.weight, source.model[10].cv1.conv.weight)
        torch.testing.assert_close(rebuilt.model[24].m[0][1].attn.qkv.conv.weight,
                                   source.model[22].m[0][1].attn.qkv.conv.weight)

    def test_segmentation_backward(self) -> None:
        model = make_model().to(DEVICE).train()
        with CONFIG.open(encoding="utf-8") as file:
            recipe = yaml.safe_load(file)["train"]
        model.args = SimpleNamespace(**(DEFAULT_CFG_DICT | recipe))
        masks = torch.zeros(2, 128, 128, device=DEVICE)
        masks[:, 32:96, 32:96] = 1
        sem_masks = torch.zeros((2, 128, 128), device=DEVICE, dtype=torch.long)
        sem_masks[0, 32:96, 32:96] = 0
        sem_masks[1, 32:96, 32:96] = 1
        batch = {"img": torch.rand(2, 3, 256, 256, device=DEVICE),
                 "batch_idx": torch.tensor([0., 1.], device=DEVICE),
                 "cls": torch.tensor([[0.], [1.]], device=DEVICE),
                 "bboxes": torch.tensor([[.5, .5, .5, .5]] * 2, device=DEVICE),
                 "masks": masks, "sem_masks": sem_masks}
        for nonempty in (True, False):
            model.zero_grad(set_to_none=True)
            if not nonempty:
                batch.update(batch_idx=batch["batch_idx"][:0], cls=batch["cls"][:0],
                             bboxes=batch["bboxes"][:0], masks=torch.zeros_like(masks),
                             sem_masks=torch.zeros_like(sem_masks))
            with torch.autocast(device_type=DEVICE, enabled=DEVICE == "cuda"):
                loss, items = model(batch)
            self.assertTrue(torch.isfinite(loss).all())
            if nonempty:
                self.assertGreater(items[1].item(), 0)
            loss.sum().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
            for i in (5, 8):
                self.assertIsNotNone(model.model[i].beta.grad)
            print(f"[BACKWARD] device={DEVICE}, positive={nonempty}, loss={items.tolist()}")

    def test_checkpoint_and_fused_inference(self) -> None:
        model = make_model().to(DEVICE).eval()
        x = torch.rand(1, 3, 256, 256, device=DEVICE)
        with torch.no_grad():
            expected = model(x)[0]
        stream = io.BytesIO()
        torch.save(model, stream)
        stream.seek(0)
        fused = torch.load(stream, weights_only=False).fuse(verbose=False)
        self.assertFalse(any(isinstance(m, torch.nn.BatchNorm2d) for m in fused.modules()))
        with torch.no_grad():
            actual = fused(x)[0]
        for a, b in zip(tensor_leaves(actual), tensor_leaves(expected)):
            torch.testing.assert_close(a, b, rtol=2e-3, atol=2e-3)
        print(f"[COST] fused params={sum(p.numel() for p in fused.parameters())}; GFLOPs@640={get_flops(fused, 640):.6f}")
        if DEVICE == "cuda":
            with torch.no_grad():
                half_output = fused.half()(x.half())[0]
            self.assertTrue(all(torch.isfinite(t).all() for t in tensor_leaves(half_output)))

    def test_recipe(self) -> None:
        with CONFIG.open(encoding="utf-8") as file:
            candidate = yaml.safe_load(file)
        with (ROOT / "experiments/data-v2-abl-000-y26m-b16-s42.yaml").open(encoding="utf-8") as file:
            baseline = yaml.safe_load(file)
        self.assertEqual(candidate["train"], baseline["train"])
        self.assertEqual(candidate["data"], baseline["data"])
        self.assertEqual(candidate["pretrained"], baseline["model"])


def main() -> None:
    global DEVICE, WEIGHTS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda", action="store_true", help="包含 CUDA AMP 反向和 FP16 推理")
    parser.add_argument("--weights", help="官方 yolo26m-seg.pt 路径或下载名称")
    args = parser.parse_args()
    DEVICE = "cuda" if args.cuda else "cpu"
    WEIGHTS = args.weights
    if args.cuda and not torch.cuda.is_available():
        raise RuntimeError("当前环境没有可用 CUDA，请先激活云端 GPU 环境")
    torch.set_num_threads(4)
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(DRBSCSATests))
    raise SystemExit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
