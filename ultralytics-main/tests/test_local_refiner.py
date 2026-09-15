"""CPU tests for the H local candidate refinement implementation."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

from ultralytics import YOLO
from ultralytics.models.yolo.segment.predict import SegmentationPredictor
from ultralytics.models.yolo.segment import (
    LocalRefinementPredictor,
    LocalRefinementValidator,
    load_local_refinement_checkpoint,
    save_local_refinement_checkpoint,
)
from ultralytics.models.yolo.segment.refine import (
    apply_local_refinement,
    assign_candidate_labels,
    crop_and_pad_candidates,
)
from ultralytics.models.yolo.segment.val import SegmentationValidator
from ultralytics.nn.modules import LocalCropRefiner


class FixedRefiner(nn.Module):
    """Return fixed logits for deterministic score tests."""

    def __init__(self, logits: torch.Tensor) -> None:
        super().__init__()
        self.register_buffer("logits", logits)

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        return self.logits[: crops.shape[0]].to(crops.device)


def sample_detection() -> dict[str, torch.Tensor]:
    """Build one prediction containing a mask and two classes."""
    return {
        "bboxes": torch.tensor([[1.0, 1.0, 6.0, 5.0], [2.0, 2.0, 7.0, 7.0]]),
        "conf": torch.tensor([0.8, 0.6]),
        "cls": torch.tensor([0.0, 1.0]),
        "masks": torch.randint(0, 2, (2, 8, 8), dtype=torch.uint8),
    }


def test_local_crop_refiner_uses_exact_yolo26_first_five_layers() -> None:
    model_yaml = Path(__file__).parents[1] / "ultralytics/cfg/models/26/yolo26m-p2proto-seg.yaml"
    model = YOLO(model_yaml)
    refiner = LocalCropRefiner(model.model.model[:5])

    assert sum(parameter.numel() for parameter in refiner.parameters()) == 1_224_515
    assert refiner(torch.rand(2, 3, 96, 96)).shape == (2, 3)


def test_crop_and_pad_candidates_preserves_aspect_ratio() -> None:
    images = torch.ones(1, 3, 8, 8)
    boxes = torch.tensor([[2.0, 3.0, 6.0, 5.0]])
    crops = crop_and_pad_candidates(images, boxes, torch.tensor([0]), output_size=8)

    assert crops.shape == (1, 3, 8, 8)
    assert torch.allclose(crops[:, :, :2], torch.full_like(crops[:, :, :2], 114 / 255))
    assert torch.allclose(crops[:, :, 2:6], torch.ones_like(crops[:, :, 2:6]))
    assert torch.allclose(crops[:, :, 6:], torch.full_like(crops[:, :, 6:], 114 / 255))


def test_candidate_assignment_uses_same_class_and_background_rules() -> None:
    candidates = torch.tensor(
        [
            [0.0, 0.0, 10.0, 10.0],
            [20.0, 20.0, 30.0, 30.0],
            [40.0, 40.0, 50.0, 50.0],
            [0.0, 0.0, 10.0, 10.0],
        ]
    )
    candidate_classes = torch.tensor([0.0, 1.0, 0.0, 1.0])
    targets = torch.tensor([[0.0, 0.0, 10.0, 10.0], [21.0, 21.0, 31.0, 31.0]])
    target_classes = torch.tensor([0.0, 1.0])

    labels = assign_candidate_labels(candidates, candidate_classes, targets, target_classes)

    assert labels.tolist() == [0, 1, 2, -1]


def test_probability_one_is_identity_and_prediction_geometry_is_unchanged() -> None:
    images = torch.rand(1, 3, 8, 8)
    detection = sample_detection()
    original = {key: value.clone() for key, value in detection.items()}
    refiner = FixedRefiner(torch.tensor([[100.0, -100.0, -100.0], [-100.0, 100.0, -100.0]]))

    refined = apply_local_refinement(images, [detection], refiner)[0]

    assert torch.equal(refined["conf"], original["conf"])
    assert torch.equal(refined["bboxes"], original["bboxes"])
    assert torch.equal(refined["cls"], original["cls"])
    assert torch.equal(refined["masks"], original["masks"])
    assert torch.equal(detection["conf"], original["conf"])


def test_empty_candidates_are_supported() -> None:
    detection = {
        "bboxes": torch.empty(0, 4),
        "conf": torch.empty(0),
        "cls": torch.empty(0),
        "masks": torch.empty(0, 8, 8),
    }
    refined = apply_local_refinement(
        torch.rand(1, 3, 8, 8),
        [detection],
        FixedRefiner(torch.empty(0, 3)),
    )
    assert refined[0]["conf"].numel() == 0


def test_validator_and_predictor_apply_identical_scores(monkeypatch: pytest.MonkeyPatch) -> None:
    images = torch.rand(1, 3, 8, 8)
    detection = sample_detection()
    logits = torch.tensor([[1.0, 0.0, -1.0], [0.0, 2.0, -1.0]])

    validator = object.__new__(LocalRefinementValidator)
    validator.refiner = FixedRefiner(logits)
    validator.crop_size = 96
    validator.score_batch_size = 128
    validator._batch_images = images
    monkeypatch.setattr(SegmentationValidator, "postprocess", lambda self, preds: [sample_detection()])
    validator_output = validator.postprocess([])[0]["conf"]

    predictor = object.__new__(LocalRefinementPredictor)
    predictor.refiner = FixedRefiner(logits)
    predictor.crop_size = 96
    predictor.score_batch_size = 128
    monkeypatch.setattr(
        SegmentationPredictor,
        "construct_results",
        lambda self, preds, img, orig_imgs, protos: preds,
    )
    raw_prediction = torch.cat(
        (
            detection["bboxes"],
            detection["conf"].unsqueeze(1),
            detection["cls"].unsqueeze(1),
            torch.zeros(2, 32),
        ),
        dim=1,
    )
    predictor_output = predictor.construct_results(
        [raw_prediction],
        images,
        [torch.empty(0)],
        torch.empty(1, 32, 2, 2),
    )[0][:, 4]

    assert torch.allclose(validator_output, predictor_output)


def test_composite_checkpoint_round_trip(tmp_path: Path) -> None:
    base = nn.Sequential(nn.Conv2d(3, 4, 1), nn.ReLU())
    refiner = nn.Sequential(nn.Flatten(), nn.Linear(12, 3))
    checkpoint = tmp_path / "best.pt"
    metadata = {"epoch": 4, "train_args": {"task": "segment"}}

    save_local_refinement_checkpoint(checkpoint, base, refiner, metadata)
    loaded_base, loaded_refiner, loaded_metadata = load_local_refinement_checkpoint(checkpoint)

    assert loaded_metadata["epoch"] == 4
    assert torch.allclose(loaded_base[0].weight, base[0].weight, rtol=1e-3, atol=5e-4)
    assert torch.allclose(loaded_refiner[1].weight, refiner[1].weight, rtol=1e-3, atol=5e-4)
