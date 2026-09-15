"""Focused checks for the data-v2 P3 oriented strip-context ablation G."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import C3k2OrientedStrip, Conv, DiagonalConv, OrientedStripContext
from ultralytics.nn.tasks import SegmentationModel, load_checkpoint
from ultralytics.utils import YAML
from ultralytics.utils.torch_utils import fuse_conv_and_bn


MODEL_ROOT = Path(__file__).parents[1] / "ultralytics" / "cfg" / "models" / "26"
SOURCE_YAML = "yolo26-seg.yaml"
CANDIDATE_YAML = "yolo26m-p3-oristrip-seg.yaml"
CONTEXT_BRANCHES = ("reduce", "local", "horizontal", "vertical", "diagonal", "antidiagonal")


def _build_model(yaml_name: str) -> SegmentationModel:
    config = YAML.load(MODEL_ROOT / yaml_name)
    config["scale"] = "m"
    return SegmentationModel(deepcopy(config), nc=2, verbose=False)


def _assert_nonzero_finite_gradients(module: torch.nn.Module):
    for name, parameter in module.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert torch.count_nonzero(parameter.grad) > 0, name


def test_diagonal_conv_samples_only_its_own_diagonal():
    """Each diagonal branch keeps one kernel diagonal and leaves the other taps unused."""
    torch.manual_seed(42)
    module = DiagonalConv(4, 5)
    antidiagonal = DiagonalConv(4, 5, anti=True)

    assert torch.count_nonzero(module.conv.weight * module.diagonal) == 4 * 5
    assert torch.count_nonzero(antidiagonal.conv.weight * antidiagonal.diagonal) == 4 * 5
    assert torch.count_nonzero(module.diagonal * antidiagonal.diagonal) == 1  # centre tap only

    module.zero_grad(set_to_none=True)
    module._diagonal_conv(torch.randn(1, 4, 9, 9)).square().mean().backward()
    off_diagonal = module.conv.weight.grad * (1 - module.diagonal)
    assert torch.count_nonzero(off_diagonal) == 0
    assert torch.count_nonzero(module.conv.weight.grad) == 4 * 5


def test_diagonal_conv_prefers_its_orientation():
    """A unit diagonal kernel responds to a matching image diagonal and not to the crossing one."""
    main = DiagonalConv(1, 5)
    anti = DiagonalConv(1, 5, anti=True)
    main.conv.weight.data.fill_(1.0)
    anti.conv.weight.data.fill_(1.0)

    inputs = torch.zeros(1, 1, 9, 9)
    inputs[0, 0, torch.arange(9), torch.arange(9)] = 1.0
    with torch.no_grad():
        centre = (0, 0, 4, 4)
        assert main._diagonal_conv(inputs)[centre].item() == 5.0
        assert anti._diagonal_conv(inputs)[centre].item() == 1.0


def test_oriented_strip_identity_and_two_step_learning():
    """Zero projection preserves the source features while every context branch can learn."""
    torch.manual_seed(42)
    module = OrientedStripContext(32)
    inputs = torch.randn(2, 32, 24, 28)
    target = torch.randn_like(inputs)
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)

    assert torch.equal(module(inputs), inputs)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        outputs = module(inputs)
        (outputs - target).square().mean().backward()
        _assert_nonzero_finite_gradients(module.project)
        if step == 1:
            for branch in CONTEXT_BRANCHES:
                _assert_nonzero_finite_gradients(getattr(module, branch))
        optimizer.step()

    assert not torch.equal(module(inputs), inputs)


def test_oriented_strip_forward_split_preserves_learned_context():
    """The split path must retain the context residual after its projection becomes nonzero."""
    torch.manual_seed(42)
    module = C3k2OrientedStrip(32, 32, n=2).eval()
    torch.nn.init.normal_(module.oriented_context.project.weight, std=0.02)
    inputs = torch.randn(1, 32, 16, 20)
    with torch.no_grad():
        torch.testing.assert_close(module(inputs), module.forward_split(inputs), rtol=0, atol=0)


def test_oriented_strip_model_transfers_source_weights_and_keeps_prediction_geometry():
    """The candidate starts with its source predictions and preserves P3/P4/P5 detection."""
    torch.manual_seed(42)
    image = torch.rand(1, 3, 640, 640)
    source = _build_model(SOURCE_YAML).eval()
    candidate = _build_model(CANDIDATE_YAML).eval()
    source_state, candidate_state = source.state_dict(), candidate.state_dict()
    assert set(source_state).issubset(candidate_state)
    assert all(source_state[key].shape == candidate_state[key].shape for key in source_state)
    extra_keys = set(candidate_state) - set(source_state)
    assert extra_keys and all(key.startswith("model.16.oriented_context.") for key in extra_keys)
    assert [i for i, layer in enumerate(candidate.model) if isinstance(layer, C3k2OrientedStrip)] == [16]
    assert len(source.model) == len(candidate.model)
    assert source.stride.tolist() == candidate.stride.tolist() == [8.0, 16.0, 32.0]
    assert source.model[-1].f == candidate.model[-1].f

    candidate.load(source, verbose=False)
    for key, value in source_state.items():
        torch.testing.assert_close(candidate.state_dict()[key], value, rtol=0, atol=0)
    with torch.no_grad():
        expected = source(image)[0]
        actual = candidate(image)[0]
    assert actual[1].shape == (1, 32, 160, 160)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def _run_and_capture(model: SegmentationModel, image: torch.Tensor):
    """Return the P3 neck output and the prediction tuple of one inference pass."""
    captured = []
    handle = model.model[16].register_forward_hook(
        lambda _module, _inputs, output: captured.append(output.detach().clone())
    )
    try:
        with torch.no_grad():
            outputs = model(image)
    finally:
        handle.remove()
    return captured[-1], outputs[0]


def test_oriented_strip_context_survives_conv_batchnorm_fusion():
    """Folding each branch BatchNorm must preserve the context output and the diagonal masks."""
    torch.manual_seed(42)
    module = OrientedStripContext(32).eval()
    torch.nn.init.normal_(module.project.weight, std=0.02)
    torch.nn.init.normal_(module.project.bias, std=0.01)
    with torch.no_grad():
        for branch in CONTEXT_BRANCHES:
            submodule = getattr(module, branch)
            submodule.bn.running_mean.uniform_(-0.2, 0.2)
            submodule.bn.running_var.uniform_(0.5, 1.5)
            submodule.bn.weight.uniform_(0.5, 1.5)
            submodule.bn.bias.uniform_(-0.2, 0.2)
    inputs = torch.randn(1, 32, 24, 28)
    with torch.no_grad():
        expected = module(inputs)

    fused = deepcopy(module)
    for submodule in fused.modules():
        if isinstance(submodule, Conv) and hasattr(submodule, "bn"):
            submodule.conv = fuse_conv_and_bn(submodule.conv, submodule.bn)
            delattr(submodule, "bn")
            submodule.forward = submodule.forward_fuse
    with torch.no_grad():
        actual = fused(inputs)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-4)
    for branch in ("diagonal", "antidiagonal"):
        submodule = getattr(fused, branch)
        taps = submodule.conv.out_channels * submodule.conv.kernel_size[0]
        assert torch.count_nonzero(submodule.conv.weight * submodule.diagonal) == taps


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


def test_oriented_strip_model_segmentation_loss_with_targets_and_background():
    """The source head must backpropagate through the new projection for real and empty targets."""
    torch.manual_seed(42)
    model = _build_model(CANDIDATE_YAML).train()
    model.args = get_cfg(overrides={"task": "segment", "mask_ratio": 2})
    for with_targets in (True, False):
        model.zero_grad(set_to_none=True)
        loss, items = model.loss(_segmentation_batch(with_targets))
        assert torch.isfinite(loss).all() and torch.isfinite(items).all()
        loss.sum().backward()
        _assert_nonzero_finite_gradients(model.model[16].oriented_context.project)
        gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
        assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)
        if with_targets:
            assert items[1] > 0
            proto_gradients = [p.grad for p in model.model[-1].proto.parameters() if p.grad is not None]
            assert proto_gradients and any(torch.count_nonzero(gradient) > 0 for gradient in proto_gradients)


def test_oriented_strip_model_checkpoint_fusion_preserves_learned_predictions():
    """Saving and fusing must preserve active strip branches, including their diagonal masks."""
    torch.manual_seed(42)
    model = _build_model(CANDIDATE_YAML).eval()
    context = model.model[16].oriented_context
    torch.nn.init.normal_(context.project.weight, std=0.02)
    torch.nn.init.normal_(context.project.bias, std=0.01)
    with torch.no_grad():
        for branch in CONTEXT_BRANCHES:
            module = getattr(context, branch)
            module.bn.running_mean.uniform_(-0.2, 0.2)
            module.bn.running_var.uniform_(0.5, 1.5)
            module.bn.weight.uniform_(0.5, 1.5)
            module.bn.bias.uniform_(-0.2, 0.2)
    image = torch.rand(1, 3, 128, 160)
    expected_p3, expected = _run_and_capture(model, image)
    with TemporaryDirectory() as directory:
        checkpoint = Path(directory) / "oristrip.pt"
        torch.save({"model": model, "train_args": {"task": "segment"}}, checkpoint)
        loaded, _ = load_checkpoint(checkpoint, fuse=True)
    actual_p3, actual = _run_and_capture(loaded, image)

    loaded_context = loaded.model[16].oriented_context
    for branch in CONTEXT_BRANCHES:
        assert not hasattr(getattr(loaded_context, branch), "bn")
    for branch in ("diagonal", "antidiagonal"):
        submodule = getattr(loaded_context, branch)
        taps = submodule.conv.out_channels * submodule.conv.kernel_size[0]
        assert torch.count_nonzero(submodule.conv.weight * submodule.diagonal) == taps

    torch.testing.assert_close(actual_p3, expected_p3, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-4, atol=1e-4)


def test_oriented_strip_yaml_differs_from_source_only_at_p3():
    """The candidate graph must repeat the source everywhere except the P3 neck output."""
    source = YAML.load(MODEL_ROOT / SOURCE_YAML)
    candidate = YAML.load(MODEL_ROOT / CANDIDATE_YAML)
    source_layers = source["backbone"] + source["head"]
    candidate_layers = candidate["backbone"] + candidate["head"]
    assert len(source_layers) == len(candidate_layers) == 24

    for field in ("nc", "end2end", "reg_max", "scales"):
        assert candidate[field] == source[field], field
    assert candidate["scale"] == "m"

    changed = [index for index, (before, after) in enumerate(zip(source_layers, candidate_layers)) if before != after]
    assert changed == [16], changed
    assert source_layers[16] == [-1, 2, "C3k2", [256, True]]
    assert candidate_layers[16] == [-1, 2, "C3k2OrientedStrip", [256, True]]


def test_oriented_strip_recipe_matches_baseline():
    root = MODEL_ROOT.parents[4]
    baseline = YAML.load(root / "experiments/data-v2-abl-000-y26m-b16-s42.yaml")
    candidate = YAML.load(root / "experiments/data-v2-abl-g-oristrip-b16-s42.yaml")
    assert candidate["train"] == baseline["train"]
    assert candidate["data"] == baseline["data"]
    assert candidate["pretrained"] == baseline["model"]
