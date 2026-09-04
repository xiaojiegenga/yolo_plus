"""Focused tests for the B BCE plus soft Dice instance-mask loss."""

import torch
import torch.nn.functional as F

from ultralytics.cfg import get_cfg
from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.ops import crop_mask


def _square_target(size: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """Return one square mask and its xyxy box in mask coordinates."""
    target = torch.zeros(1, size, size)
    target[:, 2:6, 2:6] = 1.0
    box = torch.tensor([[2.0, 2.0, 6.0, 6.0]])
    return target, box


def test_instance_dice_rewards_overlap_and_ignores_outside_box() -> None:
    """Perfect in-box overlap should beat disjoint overlap and ignore logits outside the target box."""
    target, box = _square_target()
    perfect_logits = torch.where(target.bool(), torch.tensor(10.0), torch.tensor(-10.0))
    perfect_with_outside_noise = perfect_logits.clone()
    perfect_with_outside_noise[:, :2] = 10.0
    disjoint_logits = torch.full_like(target, -10.0)
    disjoint_logits[:, 2:4, 2:6] = 10.0

    perfect_loss = v8SegmentationLoss.instance_mask_dice_loss(perfect_logits, target, box)
    noisy_loss = v8SegmentationLoss.instance_mask_dice_loss(perfect_with_outside_noise, target, box)
    disjoint_loss = v8SegmentationLoss.instance_mask_dice_loss(disjoint_logits, target, box)

    torch.testing.assert_close(perfect_loss, noisy_loss)
    assert perfect_loss.item() < disjoint_loss.item()
    assert 0.0 <= perfect_loss.item() <= 1.0
    assert 0.0 <= disjoint_loss.item() <= 1.0


def test_zero_dice_gain_matches_original_bce_exactly() -> None:
    """A zero Dice gain must reproduce the original area-normalized BCE path."""
    torch.manual_seed(7)
    target, box = _square_target(size=8)
    coefficients = torch.randn(1, 4)
    proto = torch.randn(4, 8, 8)
    area = torch.tensor([16.0 / 64.0])
    pred_mask = torch.einsum("in,nhw->ihw", coefficients, proto)
    expected = (
        crop_mask(F.binary_cross_entropy_with_logits(pred_mask, target, reduction="none"), box)
        .mean(dim=(1, 2))
        .div(area)
        .sum()
    )

    actual = v8SegmentationLoss.single_mask_loss(
        target, coefficients, proto, box, area, dice_gain=0.0, dice_smooth=1.0
    )

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_combined_mask_loss_has_finite_gradients_and_preserves_target() -> None:
    """The combined loss must backpropagate finite gradients without modifying ground-truth masks."""
    torch.manual_seed(11)
    target = torch.zeros(2, 12, 12)
    target[0, 2:8, 1:7] = 1.0
    target[1, 5:11, 6:10] = 1.0
    target_before = target.clone()
    boxes = torch.tensor([[1.0, 2.0, 7.0, 8.0], [6.0, 5.0, 10.0, 11.0]])
    areas = torch.tensor([36.0 / 144.0, 24.0 / 144.0])
    coefficients = torch.randn(2, 8, requires_grad=True)
    proto = torch.randn(8, 12, 12, requires_grad=True)

    loss = v8SegmentationLoss.single_mask_loss(
        target,
        coefficients,
        proto,
        boxes,
        areas,
        dice_gain=0.5,
        dice_smooth=1.0,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert coefficients.grad is not None and torch.isfinite(coefficients.grad).all()
    assert proto.grad is not None and torch.isfinite(proto.grad).all()
    torch.testing.assert_close(target, target_before)


def test_instance_dice_parameters_are_accepted_by_config() -> None:
    """The training config must retain the frozen B hyperparameters."""
    cfg = get_cfg(overrides={"instance_dice_gain": 0.5, "instance_dice_smooth": 1.0})

    assert cfg.instance_dice_gain == 0.5
    assert cfg.instance_dice_smooth == 1.0
