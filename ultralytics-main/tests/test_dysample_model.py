"""Focused checks for the independent E Neck upsampling ablation."""

import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from torch.nn import functional as F

from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import DySample, Segment26
from ultralytics.nn.tasks import SegmentationModel, load_checkpoint
from ultralytics.utils import YAML

ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "ultralytics-main/ultralytics/cfg/models/26"
MODEL = MODEL_ROOT / "yolo26m-dysample-seg.yaml"


def build_model():
    cfg = YAML.load(MODEL)
    cfg["scale"] = "m"
    return SegmentationModel(cfg, nc=2, verbose=False)


def test_zero_offsets_equal_bilinear_on_rectangular_features():
    module = DySample(16)
    torch.nn.init.zeros_(module.offset.weight)
    torch.nn.init.zeros_(module.offset.bias)
    image = torch.randn(2, 16, 7, 11, requires_grad=True)
    actual = module(image)
    expected = F.interpolate(image, scale_factor=2, mode="bilinear", align_corners=False)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=2e-6)
    actual.square().mean().backward()
    assert image.grad is not None and torch.isfinite(image.grad).all()
    assert torch.isfinite(module.offset.weight.grad).all() and module.offset.weight.grad.abs().sum() > 0


def test_offsets_move_sampling_horizontally_without_mixing_groups():
    module = DySample(4, groups=4)
    torch.nn.init.zeros_(module.offset.weight)
    torch.nn.init.zeros_(module.offset.bias)
    with torch.no_grad():
        module.offset.bias[:16] = 4.0  # factor 0.25 makes this one input pixel along x
    ramp = torch.arange(9).float().reshape(1, 1, 1, 9).expand(1, 4, 5, 9)
    ramp = ramp + torch.arange(4).reshape(1, 4, 1, 1) * 100
    original = F.interpolate(ramp, scale_factor=2, mode="bilinear", align_corners=False)
    shifted = module(ramp)
    torch.testing.assert_close(shifted[..., 1:-3], original[..., 3:-1], rtol=1e-5, atol=1e-5)


def test_only_two_neck_layers_change_and_baseline_weights_keep_shapes():
    base = YAML.load(MODEL_ROOT / "yolo26-seg.yaml")
    candidate = YAML.load(MODEL)
    assert candidate["backbone"] == base["backbone"]
    assert [i + 11 for i, (a, b) in enumerate(zip(base["head"], candidate["head"])) if a != b] == [11, 14]
    base["scale"] = "m"
    baseline = SegmentationModel(base, nc=2, verbose=False)
    model = build_model()
    assert type(model.model[-1]) is Segment26
    assert model.stride.tolist() == [8, 16, 32]
    assert [i for i, layer in enumerate(model.model) if isinstance(layer, DySample)] == [11, 14]
    state, original = model.state_dict(), baseline.state_dict()
    assert all(key in state and state[key].shape == value.shape for key, value in original.items())
    assert set(state) - set(original) == {
        f"model.{i}.{name}" for i in (11, 14) for name in ("offset.weight", "offset.bias", "init_pos")
    }
    assert sum(p.numel() for p in model.parameters()) - sum(p.numel() for p in baseline.parameters()) == 32832


def test_model_emits_standard_prototypes_for_square_and_rectangular_inputs():
    model = build_model().eval()
    for height, width in ((640, 640), (128, 160)):
        with torch.no_grad():
            detections, prototypes = model(torch.rand(1, 3, height, width))[0]
        assert detections.shape == (1, 300, 38)
        assert prototypes.shape == (1, 32, height // 4, width // 4)
        assert torch.isfinite(detections).all() and torch.isfinite(prototypes).all()


def test_segmentation_loss_reaches_both_upsamplers_with_and_without_targets():
    model = build_model().train()
    model.args = get_cfg(overrides={"task": "segment", "mask_ratio": 2})
    masks = torch.zeros(1, 80, 80)
    masks[0, 20:60, 28:40] = 1
    for populated in (True, False):
        batch = {
            "img": torch.rand(1, 3, 160, 160),
            "batch_idx": torch.zeros(1 if populated else 0),
            "cls": torch.zeros(1 if populated else 0, 1),
            "bboxes": torch.tensor([[.425, .5, .15, .5]]) if populated else torch.zeros(0, 4),
            "masks": masks if populated else torch.zeros_like(masks),
            "sem_masks": (masks > 0).float() if populated else torch.zeros_like(masks),
        }
        model.zero_grad(set_to_none=True)
        loss, items = model.loss(batch)
        loss.sum().backward()
        assert torch.isfinite(loss).all() and torch.isfinite(items).all()
        for index in (11, 14):
            grad = model.model[index].offset.weight.grad
            assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
        if populated:
            assert items[1] > 0


def test_checkpoint_fusion_preserves_detection_and_mask_outputs():
    model = build_model().eval()
    image = torch.rand(1, 3, 128, 160)
    with torch.no_grad():
        expected = model(image)[0]
    with TemporaryDirectory() as directory:
        path = Path(directory) / "e.pt"
        torch.save({"model": model, "train_args": {"task": "segment"}}, path)
        loaded, _ = load_checkpoint(path, fuse=True)
    with torch.no_grad():
        actual = loaded(image)[0]
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-4)


def test_recipe_and_training_entry_preserve_frozen_parameters_and_seed():
    baseline = YAML.load(ROOT / "experiments/data-v2-abl-000-y26m-b16-s42.yaml")
    candidate = YAML.load(ROOT / "experiments/data-v2-abl-e-dysample-b16-s42.yaml")
    assert candidate["train"] == baseline["train"]
    assert candidate["data"] == baseline["data"]
    assert candidate["pretrained"] == baseline["model"]
    spec = importlib.util.spec_from_file_location("e_train_entry", ROOT / "scripts/train_yolo26_seg.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    snapshots = []

    def capture_train(wrapper, **kwargs):
        snapshots.append([wrapper.model.model[i].offset.weight.detach().clone() for i in (11, 14)])

    with patch("ultralytics.YOLO.train", capture_train):
        for initial_seed in (7, 99):
            torch.manual_seed(initial_seed)
            entry.run_training(str(MODEL), None,
                {"seed": 42, "deterministic": True, "project": "runs", "name": "e-entry-check"})
    for a, b in zip(*snapshots):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
