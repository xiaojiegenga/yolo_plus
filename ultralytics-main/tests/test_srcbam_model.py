"""Focused tests for the data-v2 SR-CBAM ablation models."""

from copy import deepcopy
from pathlib import Path

import torch

from ultralytics.nn.modules import C3k2SRCBAM, C3k2ZRCBAM, ResidualCBAM, ZeroInitResidualCBAM
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import YAML


MODEL_ROOT = Path(__file__).parents[1] / "ultralytics" / "cfg" / "models" / "26"


def _build_model(yaml_name: str) -> SegmentationModel:
    cfg = YAML.load(MODEL_ROOT / yaml_name)
    cfg["scale"] = "m"
    return SegmentationModel(deepcopy(cfg), nc=2, verbose=False)


def test_residual_cbam_shape_mix_and_backward():
    module = ResidualCBAM(32, reduction=16, kernel_size=7, init_mix=0.1)
    inputs = torch.randn(2, 32, 20, 20, requires_grad=True)

    outputs = module(inputs)
    outputs.mean().backward()

    assert outputs.shape == inputs.shape
    assert torch.allclose(torch.sigmoid(module.mix_logit), torch.tensor(0.1))
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()
    assert all(parameter.grad is not None for parameter in module.parameters())


def test_srcbam_model_keeps_baseline_indexes_and_weight_keys():
    baseline = _build_model("yolo26-seg.yaml")
    srcbam = _build_model("yolo26m-srcbam-seg.yaml")

    assert isinstance(srcbam.model[4], C3k2SRCBAM)
    assert isinstance(srcbam.model[6], C3k2SRCBAM)
    assert sum(isinstance(layer, C3k2SRCBAM) for layer in srcbam.model) == 2
    assert srcbam.model[-1].f == [16, 19, 22]

    baseline_state = baseline.state_dict()
    srcbam_state = srcbam.state_dict()
    assert set(baseline_state).issubset(srcbam_state)
    assert all(baseline_state[key].shape == srcbam_state[key].shape for key in baseline_state)
    extra_keys = set(srcbam_state).difference(baseline_state)
    assert len(extra_keys) == 8
    assert all(".srcbam." in key for key in extra_keys)
    assert sum(parameter.numel() for parameter in srcbam.parameters()) - sum(
        parameter.numel() for parameter in baseline.parameters()
    ) == 65_734

    srcbam.eval()
    with torch.no_grad():
        outputs = srcbam(torch.zeros(1, 3, 64, 64))
    assert outputs is not None


def test_zero_init_residual_cbam_starts_as_identity_and_learns_scale():
    module = ZeroInitResidualCBAM(32, reduction=16, kernel_size=7)
    inputs = torch.randn(2, 32, 20, 20, requires_grad=True)

    outputs = module(inputs)
    outputs.mean().backward()

    assert torch.equal(outputs, inputs)
    assert module.residual_scale.item() == 0.0
    assert module.residual_scale.grad is not None
    assert torch.isfinite(module.residual_scale.grad)
    assert inputs.grad is not None and torch.isfinite(inputs.grad).all()


def test_a2_model_uses_zero_init_cbam_only_at_p3_and_keeps_weight_keys():
    baseline = _build_model("yolo26-seg.yaml")
    a2_model = _build_model("yolo26m-p3-zrcbam-seg.yaml")

    assert isinstance(a2_model.model[4], C3k2ZRCBAM)
    assert not isinstance(a2_model.model[6], (C3k2SRCBAM, C3k2ZRCBAM))
    assert sum(isinstance(layer, C3k2ZRCBAM) for layer in a2_model.model) == 1
    assert a2_model.model[-1].f == [16, 19, 22]

    baseline_state = baseline.state_dict()
    a2_state = a2_model.state_dict()
    assert set(baseline_state).issubset(a2_state)
    assert all(baseline_state[key].shape == a2_state[key].shape for key in baseline_state)
    extra_keys = set(a2_state).difference(baseline_state)
    assert len(extra_keys) == 4
    assert all(".zrcbam." in key for key in extra_keys)
    assert sum(parameter.numel() for parameter in a2_model.parameters()) - sum(
        parameter.numel() for parameter in baseline.parameters()
    ) == 32_867

    a2_model.eval()
    with torch.no_grad():
        outputs = a2_model(torch.zeros(1, 3, 64, 64))
    assert outputs is not None
