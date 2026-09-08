"""Focused tests for the data-v2 P2-refined mask prototype ablation (D1)."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import importlib.util
from types import SimpleNamespace

import torch

from ultralytics.nn.modules import Segment26P2
from ultralytics.nn.modules.block import Proto26P2
from ultralytics.cfg import get_cfg
from ultralytics.nn.tasks import SegmentationModel, load_checkpoint
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.utils import YAML
from ultralytics.utils import ops


MODEL_ROOT = Path(__file__).parents[1] / "ultralytics" / "cfg" / "models" / "26"
IMGSZ = 640
MASK_RATIO = 2  # frozen data-v2 recipe: ground-truth masks are stored at imgsz / mask_ratio


def _build_model(yaml_name: str) -> SegmentationModel:
    cfg = YAML.load(MODEL_ROOT / yaml_name)
    cfg["scale"] = "m"
    return SegmentationModel(deepcopy(cfg), nc=2, verbose=False)


def _protos(model: SegmentationModel, imgsz: int = IMGSZ) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 3, imgsz, imgsz))
    proto = output[0][1] if isinstance(output[0], (tuple, list)) else output[1]
    return proto[0] if isinstance(proto, tuple) else proto


def test_proto26p2_doubles_prototype_resolution_and_backward():
    channels = (256, 256, 512, 512)
    proto = Proto26P2(channels, 256, 32, 2)
    features = [
        torch.randn(1, 256, 160, 160, requires_grad=True),
        torch.randn(1, 256, 80, 80),
        torch.randn(1, 512, 40, 40),
        torch.randn(1, 512, 20, 20),
    ]

    proto.eval()
    outputs = proto(features)
    outputs.mean().backward()

    assert outputs.shape == (1, 32, 320, 320)  # stride 2 for a 640 input
    assert features[0].grad is not None and torch.isfinite(features[0].grad).all()
    assert all(parameter.grad is not None for parameter in proto.p2_proj.parameters())


def test_p2proto_model_emits_stride2_prototypes():
    baseline = _protos(_build_model("yolo26-seg.yaml"))
    p2proto = _protos(_build_model("yolo26m-p2proto-seg.yaml"))

    assert baseline.shape[-2:] == (IMGSZ // 4, IMGSZ // 4)
    assert p2proto.shape[-2:] == (IMGSZ // 2, IMGSZ // 2)
    assert p2proto.shape[1] == baseline.shape[1] == 32  # prototype count unchanged


def test_p2proto_keeps_detection_pyramid_and_baseline_weight_keys():
    baseline = _build_model("yolo26-seg.yaml")
    p2proto = _build_model("yolo26m-p2proto-seg.yaml")

    assert isinstance(p2proto.model[-1], Segment26P2)
    assert len(p2proto.model) == len(baseline.model)  # layer indexes unchanged
    assert p2proto.stride.tolist() == baseline.stride.tolist() == [8.0, 16.0, 32.0]

    baseline_keys, p2proto_keys = set(baseline.state_dict()), set(p2proto.state_dict())
    assert not baseline_keys - p2proto_keys  # nothing from the baseline is dropped
    added = {key for key in p2proto_keys - baseline_keys if "num_batches_tracked" not in key}
    assert {key.split(".")[3] for key in added} == {"p2_proj", "hi_up", "hi_cv"}


def test_p2proto_prototypes_align_with_frozen_mask_ratio():
    """mask_ratio=2 stores masks at 320x320, which the stride-2 prototypes match without interpolation."""
    model = _build_model("yolo26m-p2proto-seg.yaml")
    mask_size = IMGSZ // MASK_RATIO

    assert _protos(model).shape[-2:] == (mask_size, mask_size)


def test_p2proto_segmentation_loss_is_finite_with_and_without_targets():
    model = _build_model("yolo26m-p2proto-seg.yaml")
    model.args = get_cfg(overrides={"task": "segment", "mask_ratio": MASK_RATIO})
    model.train()
    mask_size = IMGSZ // MASK_RATIO
    image = torch.rand(1, 3, IMGSZ, IMGSZ)

    masks = torch.zeros(1, mask_size, mask_size)
    masks[0, 40:120, 60:96] = 1.0  # one thin elongated instance, encoded as instance index 1
    populated = {
        "img": image,
        "batch_idx": torch.zeros(1),
        "cls": torch.zeros(1, 1),
        "bboxes": torch.tensor([[0.24, 0.25, 0.11, 0.25]]),
        "masks": masks,
        "sem_masks": (masks > 0).float(),  # auxiliary semantic target, same resolution as masks
    }
    empty = {
        "img": image,
        "batch_idx": torch.zeros(0),
        "cls": torch.zeros(0, 1),
        "bboxes": torch.zeros(0, 4),
        "masks": torch.zeros(1, mask_size, mask_size),
        "sem_masks": torch.zeros(1, mask_size, mask_size),
    }

    for batch in (populated, empty):
        model.zero_grad(set_to_none=True)
        loss, items = model.loss(deepcopy(batch))
        loss.sum().backward()

        assert torch.isfinite(loss).all() and torch.isfinite(items).all()
        grads = [p.grad for p in model.model[-1].proto.hi_cv.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads)
        if batch is populated:
            assert items[1] > 0
            assert any(torch.count_nonzero(g) > 0 for g in grads)


def test_p2proto_checkpoint_fusion_preserves_predictions():
    model = _build_model("yolo26m-p2proto-seg.yaml").eval()
    image = torch.rand(1, 3, 128, 160)
    with torch.no_grad():
        expected = model(image)[0]
    with TemporaryDirectory() as directory:
        path = Path(directory) / "d1.pt"
        torch.save({"model": model, "train_args": {"task": "segment"}}, path)
        loaded, _ = load_checkpoint(path, fuse=True)
    with torch.no_grad():
        actual = loaded(image)[0]
    assert loaded.model[-1].proto.semseg is None
    assert actual[1].shape == (1, 32, 64, 80)
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-4, atol=1e-4)


def test_p2proto_recipe_matches_baseline():
    root = MODEL_ROOT.parents[4]
    baseline = YAML.load(root / "experiments/data-v2-abl-000-y26m-b16-s42.yaml")
    candidate = YAML.load(root / "experiments/data-v2-abl-d1-p2proto-r2-b16-s42.yaml")
    assert candidate["train"] == baseline["train"]
    assert candidate["data"] == baseline["data"]
    assert candidate["pretrained"] == baseline["model"]


def test_training_entry_seeds_new_prototype_parameters():
    root = MODEL_ROOT.parents[4]
    spec = importlib.util.spec_from_file_location("d1_train_entry", root / "scripts/train_yolo26_seg.py")
    entry = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(entry)
    snapshots = []

    def capture_train(wrapper, **kwargs):
        snapshots.append(wrapper.model.model[-1].proto.hi_up.weight.detach().clone())

    # Exercise the real construction path; replace only training so no dataset or Run is created.
    with patch("ultralytics.YOLO.train", capture_train):
        for initial_seed in (7, 99):
            torch.manual_seed(initial_seed)
            entry.run_training(
                str(MODEL_ROOT / "yolo26m-p2proto-seg.yaml"),
                None,
                {"seed": 42, "deterministic": True, "project": "runs", "name": "d1-entry-check"},
            )
    torch.testing.assert_close(snapshots[0], snapshots[1], rtol=0, atol=0)


def _validation_batch(height, width):
    masks = torch.zeros(2, height // 2, width // 2)
    masks[0, height // 8 : height // 4, width // 8 : width // 4] = 1
    return {
        "img": torch.zeros(2, 3, height, width, dtype=torch.uint8),
        "batch_idx": torch.tensor([0.0]),
        "cls": torch.tensor([[0.0]]),
        "bboxes": torch.tensor([[0.375, 0.375, 0.25, 0.25]]),
        "masks": masks,
        "ori_shape": [(height, width)] * 2,
        "ratio_pad": [((1.0, 1.0), (0, 0))] * 2,
        "im_file": ["positive.jpg", "background.jpg"],
    }


def test_validator_mask_grid_and_box_crop_for_both_proto_strides():
    with TemporaryDirectory() as directory:
        validator = SegmentationValidator(save_dir=Path(directory), args={"plots": False, "conf": 0.25})
        validator.device = torch.device("cpu")
        validator.nc = 2
        validator.end2end = True
        for height, width in ((512, 672), (160, 128)):
            batch = validator.preprocess(_validation_batch(height, width))
            box = torch.tensor([[width / 4, height / 4, width / 2, height / 2]])
            detections = torch.zeros(2, 1, 7)
            detections[0, 0] = torch.cat((box[0], torch.tensor([0.9, 0.0, 1.0])))
            for process in (ops.process_mask, ops.process_mask_native):
                validator.process = process
                for stride in (4, 2):
                    proto = torch.ones(2, 1, height // stride, width // stride)
                    predictions = validator.postprocess(((detections, proto), {}))
                    prepared = validator._prepare_batch(0, batch)
                    expected_shape = (height // 4, width // 4) if process is ops.process_mask else (height, width)
                    assert predictions[0]["masks"].shape == (1, *expected_shape)
                    assert predictions[1]["masks"].shape == (0, *expected_shape)
                    assert torch.equal(predictions[0]["masks"].float(), prepared["masks"])
                    assert validator._process_batch(predictions[0], prepared)["tp_m"].all()
                    empty = validator._prepare_batch(1, batch)
                    assert validator._process_batch(predictions[1], empty)["tp_m"].shape == (0, 10)
                # The baseline's stride-4 masks match its original postprocessing exactly.
                proto = torch.randn(2, 1, height // 4, width // 4)
                actual = validator.postprocess(((detections, proto), {}))[0]["masks"]
                expected = process(proto[0], torch.ones(1, 1), box, (height, width))
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_p2proto_model_outputs_pass_validator_metrics():
    model = _build_model("yolo26m-p2proto-seg.yaml").eval()
    with TemporaryDirectory() as directory:
        validator = SegmentationValidator(
            save_dir=Path(directory), args={"plots": False, "conf": 1e-9, "max_det": 20}
        )
        validator.device = torch.device("cpu")
        validator.data = {"val": "synthetic-val"}
        validator.init_metrics(SimpleNamespace(names={0: "Rice leaffolder", 1: "Rice stemborers"}, end2end=True))
        for height, width in ((128, 160), (160, 128)):
            batch = validator.preprocess(_validation_batch(height, width))
            with torch.no_grad():
                outputs = model(batch["img"])
            predictions = validator.postprocess(outputs)
            assert predictions[0]["cls"].numel() > 0
            validator.update_metrics(predictions, batch)
        assert validator.seen == 4
