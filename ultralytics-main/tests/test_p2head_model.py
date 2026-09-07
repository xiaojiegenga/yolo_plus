"""Focused checks for the independent data-v2 P2Head ablation."""

from copy import deepcopy
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import torch

from ultralytics.nn.modules import Segment26P2Lite
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, YAML

ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = ROOT / "ultralytics-main/ultralytics/cfg/models/26"
BRANCHES = ("cv2", "cv3", "cv4", "one2one_cv2", "one2one_cv3", "one2one_cv4")


def build_model(p2=True, nc=2):
    """Build a model without downloading weights or reading a dataset."""
    torch.manual_seed(42)
    cfg = YAML.load(MODEL_ROOT / ("yolo26m-p2lite-seg.yaml" if p2 else "yolo26-seg.yaml"))
    cfg["scale"] = "m"
    return SegmentationModel(deepcopy(cfg), nc=nc, verbose=False)


def target_key(key):
    """Describe the expected semantic map independently of the model loader."""
    if not key.startswith("model.23."):
        return key
    parts = key.split(".")
    parts[1] = "28"
    if parts[2] in BRANCHES:
        parts[3] = str(int(parts[3]) + 1)
    return ".".join(parts)


def test_baseline_transfer_preserves_original_predictions_and_new_parameters():
    source, target = build_model(False), build_model()
    before = {k: v.clone() for k, v in target.state_dict().items()}
    target.load(source, verbose=False)
    expected = {target_key(k): v for k, v in source.state_dict().items()}
    actual = target.state_dict()
    for key, value in expected.items():
        assert torch.equal(actual[key], value), key
    fresh = set(actual) - set(expected)
    assert fresh
    for key in fresh:
        assert torch.equal(actual[key], before[key]), key
    source.eval()
    target.eval()
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        baseline = source(image)
        p2 = target(image)
    for mode in ("one2many", "one2one"):
        for field in ("boxes", "scores", "mask_coefficient"):
            assert torch.equal(baseline[1][mode][field], p2[1][mode][field][..., 256:]), (mode, field)
    assert torch.equal(baseline[0][1], p2[0][1])


def test_coco_class_adaptation_and_same_layout_reload():
    source, target = build_model(False, 80), build_model(nc=2)
    target.load(source, verbose=False)
    state = target.state_dict()
    skipped = []
    for key, value in source.state_dict().items():
        dest = target_key(key)
        if state[dest].shape == value.shape:
            assert torch.equal(state[dest], value), dest
        else:
            skipped.append(dest)
            assert ("cv3." in dest and ".2." in dest) or ".proto.semseg.2." in dest, dest
    assert len(skipped) == 14  # Six class predictors and one semantic predictor: weight + bias.
    clone = build_model()
    clone.load(target, verbose=False)
    assert all(torch.equal(v, clone.state_dict()[k]) for k, v in state.items())
    # YOLO.train rebuilds the loaded C model with the dataset's class count.
    loaded_coco = build_model(nc=80)
    loaded_coco.load(source, verbose=False)
    rebuilt = build_model(nc=2)
    rebuilt.load(loaded_coco, verbose=False)
    for key, value in loaded_coco.state_dict().items():
        if rebuilt.state_dict()[key].shape == value.shape:
            assert torch.equal(rebuilt.state_dict()[key], value), key


def test_four_scales_proto_resolution_and_fused_checkpoint():
    model = build_model().eval()
    head = model.model[-1]
    assert isinstance(head, Segment26P2Lite)
    assert head.f == [27, 16, 19, 22]
    assert model.stride.tolist() == [4, 8, 16, 32]
    with torch.no_grad():
        output = model(torch.zeros(1, 3, 640, 640))
    preds = output[1]["one2one"]
    assert [tuple(x.shape[2:]) for x in preds["feats"]] == [(160, 160), (80, 80), (40, 40), (20, 20)]
    assert preds["boxes"].shape == (1, 4, 34000)
    assert preds["scores"].shape == (1, 2, 34000)
    assert preds["mask_coefficient"].shape == (1, 32, 34000)
    assert output[0][0].shape == (1, 300, 38)
    assert output[0][1].shape == (1, 32, 160, 160)
    stream = BytesIO()
    torch.save(model, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=False).eval().fuse(verbose=False)
    image = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        original = model(image)
        fused = restored(image)
    assert fused[0][0].shape == original[0][0].shape
    assert fused[0][1].shape == (1, 32, 16, 16)
    assert torch.isfinite(fused[0][0]).all()
    assert torch.allclose(original[0][1], fused[0][1], rtol=1e-4, atol=1e-5)
    for field in ("boxes", "scores", "mask_coefficient"):
        assert torch.allclose(original[1]["one2one"][field], fused[1]["one2one"][field], rtol=1e-4, atol=1e-5)


def test_real_segmentation_loss_backward_with_positive_and_empty_images():
    model = build_model().train()
    model.args = SimpleNamespace(**{**DEFAULT_CFG_DICT, "mask_ratio": 2, "overlap_mask": True})
    masks = torch.zeros(2, 32, 32)
    masks[0, 12:20, 12:20] = 1
    batch = {
        "img": torch.rand(2, 3, 64, 64),
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.25, 0.25]]),
        "masks": masks,
        "sem_masks": torch.zeros(2, 32, 32, dtype=torch.long),
    }
    output = model(batch["img"])
    assert output["one2many"]["proto"][0].shape == (2, 32, 16, 16)
    assert output["one2many"]["proto"][1].shape == (2, 2, 8, 8)
    losses, _ = model.loss(batch, output)
    assert torch.isfinite(losses).all()
    assert losses[1] > 0  # Exercise positive mask assignment, not only the empty-target path.
    losses.sum().backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()
    for index in (23, 25, 27):
        assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.model[index].parameters())
    for name in BRANCHES:
        assert all(p.grad is not None for p in getattr(model.model[-1], name)[0].parameters()), name
    model.zero_grad(set_to_none=True)
    batch.update(batch_idx=torch.empty(0), cls=torch.empty(0, 1), bboxes=torch.empty(0, 4), masks=torch.zeros_like(masks))
    losses, _ = model.loss(batch)
    assert torch.isfinite(losses).all()
    losses.sum().backward()


def test_training_recipe_matches_formal_baseline():
    base = YAML.load(ROOT / "experiments/data-v2-abl-000-y26m-b16-s42.yaml")
    p2 = YAML.load(ROOT / "experiments/data-v2-abl-001-p2head-b16-s42.yaml")
    assert p2["train"] == base["train"]
    assert p2["data"] == base["data"]
    assert p2["pretrained"] == base["model"] == "yolo26m-seg.pt"
