"""Focused checks for the data-v2 P3 strip-context ablations F and D1+F."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import C3k2StripContext, StripContext
from ultralytics.nn.tasks import SegmentationModel, load_checkpoint
from ultralytics.utils import YAML


MODEL_ROOT = Path(__file__).parents[1] / "ultralytics" / "cfg" / "models" / "26"
MODEL_PAIRS = (
    ("yolo26-seg.yaml", "yolo26m-p3-strip-seg.yaml", 160),
    ("yolo26m-p2proto-seg.yaml", "yolo26m-p2proto-p3-strip-seg.yaml", 320),
)


def _build_model(yaml_name: str) -> SegmentationModel:
    config = YAML.load(MODEL_ROOT / yaml_name)
    config["scale"] = "m"
    return SegmentationModel(deepcopy(config), nc=2, verbose=False)


def _assert_nonzero_finite_gradients(module: torch.nn.Module):
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name


def test_strip_context_identity_and_two_step_learning():
    """Zero projection preserves the source features while every context branch can learn."""
    torch.manual_seed(42)
    module = StripContext(32)
    inputs = torch.randn(2, 32, 12, 16)
    target = torch.randn_like(inputs)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)

    assert torch.equal(module(inputs), inputs)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        outputs = module(inputs)
        (outputs - target).square().mean().backward()
        _assert_nonzero_finite_gradients(module.project)
        if step == 1:
            for branch in (module.reduce, module.local, module.horizontal, module.vertical):
                _assert_nonzero_finite_gradients(branch)
        optimizer.step()

    assert not torch.equal(module(inputs), inputs)


def test_strip_context_forward_split_preserves_learned_context():
    """The split path must retain the context residual after its projection becomes nonzero."""
    torch.manual_seed(42)
    module = C3k2StripContext(32, 32, n=2).eval()
    torch.nn.init.normal_(module.strip_context.project.weight, std=0.02)
    inputs = torch.randn(1, 32, 16, 20)
    with torch.no_grad():
        torch.testing.assert_close(module(inputs), module.forward_split(inputs), rtol=0, atol=0)


def test_strip_models_transfer_source_weights_and_keep_prediction_geometry():
    """Both candidates start with their source predictions and preserve P3/P4/P5 detection."""
    torch.manual_seed(42)
    image = torch.rand(1, 3, 640, 640)
    for source_yaml, candidate_yaml, proto_size in MODEL_PAIRS:
        source = _build_model(source_yaml).eval()
        candidate = _build_model(candidate_yaml).eval()
        source_state, candidate_state = source.state_dict(), candidate.state_dict()
        assert set(source_state).issubset(candidate_state)
        assert all(source_state[key].shape == candidate_state[key].shape for key in source_state)
        extra_keys = set(candidate_state) - set(source_state)
        assert extra_keys and all(key.startswith("model.16.strip_context.") for key in extra_keys)
        assert [i for i, layer in enumerate(candidate.model) if isinstance(layer, C3k2StripContext)] == [16]
        assert len(source.model) == len(candidate.model)
        assert source.stride.tolist() == candidate.stride.tolist() == [8.0, 16.0, 32.0]
        assert source.model[-1].f == candidate.model[-1].f

        candidate.load(source, verbose=False)
        for key, value in source_state.items():
            torch.testing.assert_close(candidate.state_dict()[key], value, rtol=0, atol=0)
        with torch.no_grad():
            expected = source(image)[0]
            actual = candidate(image)[0]
        assert actual[1].shape == (1, 32, proto_size, proto_size)
        torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
        torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def _segmentation_batch(with_targets: bool):
    """Use two elongated instances from different classes and the frozen mask_ratio=2."""
    masks = torch.zeros(1, 64, 64)
    semantic_masks = torch.zeros_like(masks)
    if with_targets:
        masks[0, 8:32, 12:20] = 1
        masks[0, 40:48, 32:56] = 2
        semantic_masks[0, 40:48, 32:56] = 1
    return {
        "img": torch.rand(1, 3, 128, 128),
        "batch_idx": torch.zeros(2 if with_targets else 0),
        "cls": torch.tensor([[0.0], [1.0]]) if with_targets else torch.zeros(0, 1),
        "bboxes": torch.tensor([[0.25, 0.3125, 0.125, 0.375], [0.6875, 0.6875, 0.375, 0.125]])
        if with_targets
        else torch.zeros(0, 4),
        "masks": masks,
        "sem_masks": semantic_masks,
    }


def test_strip_models_segmentation_loss_with_targets_and_background():
    """Both source heads must backpropagate through the new projection for real and empty targets."""
    for _, yaml_name, _ in MODEL_PAIRS:
        torch.manual_seed(42)
        model = _build_model(yaml_name).train()
        model.args = get_cfg(overrides={"task": "segment", "mask_ratio": 2})
        for with_targets in (True, False):
            model.zero_grad(set_to_none=True)
            loss, items = model.loss(_segmentation_batch(with_targets))
            assert torch.isfinite(loss).all() and torch.isfinite(items).all()
            loss.sum().backward()
            _assert_nonzero_finite_gradients(model.model[16].strip_context.project)
            gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
            assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
            if with_targets:
                assert items[1] > 0
                proto_gradients = [p.grad for p in model.model[-1].proto.parameters() if p.grad is not None]
                assert proto_gradients and any(torch.count_nonzero(gradient) > 0 for gradient in proto_gradients)


def test_strip_models_checkpoint_fusion_preserves_learned_predictions():
    """Saving and fusing must preserve active strip branches, including their BatchNorm statistics."""
    for _, yaml_name, _ in MODEL_PAIRS:
        torch.manual_seed(42)
        model = _build_model(yaml_name).eval()
        context = model.model[16].strip_context
        torch.nn.init.normal_(context.project.weight, std=0.02)
        torch.nn.init.normal_(context.project.bias, std=0.01)
        for branch in (context.reduce, context.local, context.horizontal, context.vertical):
            branch.bn.running_mean.uniform_(-0.2, 0.2)
            branch.bn.running_var.uniform_(0.5, 1.5)
        image = torch.rand(1, 3, 128, 160)
        with torch.no_grad():
            expected = model(image)[0]
        with TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "strip.pt"
            torch.save({"model": model, "train_args": {"task": "segment"}}, checkpoint)
            loaded, _ = load_checkpoint(checkpoint, fuse=True)
        with torch.no_grad():
            actual = loaded(image)[0]
        assert not hasattr(loaded.model[16].strip_context.horizontal, "bn")
        assert not hasattr(loaded.model[16].strip_context.vertical, "bn")
        torch.testing.assert_close(actual[0], expected[0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(actual[1], expected[1], rtol=1e-4, atol=1e-4)


def test_strip_recipes_match_baseline():
    root = MODEL_ROOT.parents[4]
    baseline = YAML.load(root / "experiments/data-v2-abl-000-y26m-b16-s42.yaml")
    for config_name in ("data-v2-abl-f-p3strip-b16-s42.yaml", "data-v2-abl-d1f-p2proto-p3strip-b16-s42.yaml"):
        candidate = YAML.load(root / "experiments" / config_name)
        assert candidate["train"] == baseline["train"]
        assert candidate["data"] == baseline["data"]
        assert candidate["pretrained"] == baseline["model"]
