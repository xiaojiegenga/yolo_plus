# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Tensor operations shared by local-refinement validation and prediction."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ultralytics.utils.metrics import box_iou


def _candidate_layout(
    detections: list[dict[str, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    """Flatten candidate boxes and retain their image/local indices."""
    boxes: list[torch.Tensor] = []
    batch_indices: list[torch.Tensor] = []
    locations: list[tuple[int, int]] = []
    for image_index, detection in enumerate(detections):
        image_boxes = detection["bboxes"]
        if image_boxes.numel() == 0:
            continue
        boxes.append(image_boxes)
        batch_indices.append(
            torch.full(
                (image_boxes.shape[0],),
                image_index,
                dtype=torch.long,
                device=image_boxes.device,
            )
        )
        locations.extend((image_index, local_index) for local_index in range(image_boxes.shape[0]))

    if not boxes:
        device = detections[0]["bboxes"].device if detections else torch.device("cpu")
        return (
            torch.empty((0, 4), device=device),
            torch.empty((0,), dtype=torch.long, device=device),
            locations,
        )
    return torch.cat(boxes), torch.cat(batch_indices), locations


def crop_and_pad_candidates(
    images: torch.Tensor,
    boxes: torch.Tensor,
    batch_indices: torch.Tensor,
    output_size: int = 96,
    fill: float = 114 / 255,
) -> torch.Tensor:
    """Crop boxes without distortion, pad the short side, and resize to a square tensor."""
    if images.ndim != 4 or images.shape[1] != 3:
        raise ValueError("images must have shape [B, 3, H, W]")
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape [N, 4]")
    if boxes.shape[0] != batch_indices.shape[0]:
        raise ValueError("boxes and batch_indices must have the same length")
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    if not boxes.shape[0]:
        return images.new_empty((0, 3, output_size, output_size))

    height, width = images.shape[-2:]
    boxes = boxes.to(device=images.device, dtype=torch.float32).clone()
    batch_indices = batch_indices.to(device=images.device, dtype=torch.long)
    boxes[:, [0, 2]].clamp_(0, width)
    boxes[:, [1, 3]].clamp_(0, height)
    box_width = (boxes[:, 2] - boxes[:, 0]).clamp_min(1.0)
    box_height = (boxes[:, 3] - boxes[:, 1]).clamp_min(1.0)
    side = torch.maximum(box_width, box_height)
    center_x = (boxes[:, 0] + boxes[:, 2]) / 2
    center_y = (boxes[:, 1] + boxes[:, 3]) / 2

    axis = (torch.arange(output_size, device=images.device, dtype=torch.float32) + 0.5) / output_size - 0.5
    offset_x = axis.view(1, 1, output_size) * side.view(-1, 1, 1)
    offset_y = axis.view(1, output_size, 1) * side.view(-1, 1, 1)
    sample_x = center_x.view(-1, 1, 1) + offset_x
    sample_y = center_y.view(-1, 1, 1) + offset_y
    sample_x = sample_x.expand(-1, output_size, -1)
    sample_y = sample_y.expand(-1, -1, output_size)

    grid_x = (2 * sample_x + 1) / width - 1
    grid_y = (2 * sample_y + 1) / height - 1
    grid = torch.stack((grid_x, grid_y), dim=-1).to(dtype=images.dtype)
    selected_images = images.index_select(0, batch_indices)
    crops = F.grid_sample(
        selected_images - fill,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ) + fill

    content = (offset_x.abs() <= box_width.view(-1, 1, 1) / 2) & (
        offset_y.abs() <= box_height.view(-1, 1, 1) / 2
    )
    fill_value = torch.as_tensor(fill, device=images.device, dtype=crops.dtype)
    return torch.where(content.unsqueeze(1), crops, fill_value)


def assign_candidate_labels(
    candidate_boxes: torch.Tensor,
    candidate_classes: torch.Tensor,
    target_boxes: torch.Tensor,
    target_classes: torch.Tensor,
    positive_iou: float = 0.5,
    background_iou: float = 0.1,
    background_class: int = 2,
) -> torch.Tensor:
    """Assign a foreground class, background, or -1 ignore label to each candidate."""
    labels = torch.full(
        (candidate_boxes.shape[0],),
        -1,
        dtype=torch.long,
        device=candidate_boxes.device,
    )
    if not candidate_boxes.shape[0]:
        return labels
    if not target_boxes.shape[0]:
        labels.fill_(background_class)
        return labels

    target_classes = target_classes.to(candidate_classes.device).long()
    ious = box_iou(candidate_boxes, target_boxes)
    max_all_iou = ious.max(dim=1).values
    labels[max_all_iou < background_iou] = background_class

    for index in range(candidate_boxes.shape[0]):
        same_class = target_classes == candidate_classes[index].long()
        if same_class.any() and ious[index, same_class].max() >= positive_iou:
            labels[index] = candidate_classes[index].long()
    return labels


@torch.inference_mode()
def apply_local_refinement(
    images: torch.Tensor,
    detections: list[dict[str, torch.Tensor]],
    refiner: nn.Module,
    crop_size: int = 96,
    batch_size: int = 128,
) -> list[dict[str, torch.Tensor]]:
    """Multiply each YOLO score by the refiner probability for its predicted class."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    refined = [{**detection, "conf": detection["conf"].clone()} for detection in detections]
    boxes, batch_indices, locations = _candidate_layout(refined)
    if not locations:
        return refined

    crops = crop_and_pad_candidates(images, boxes, batch_indices, output_size=crop_size)
    probabilities: list[torch.Tensor] = []
    for start in range(0, crops.shape[0], batch_size):
        logits = refiner(crops[start : start + batch_size])
        probabilities.append(logits.float().softmax(dim=1))
    probabilities_tensor = torch.cat(probabilities)

    for flat_index, (image_index, local_index) in enumerate(locations):
        predicted_class = refined[image_index]["cls"][local_index].long()
        foreground_probability = probabilities_tensor[flat_index, predicted_class]
        refined[image_index]["conf"][local_index] *= foreground_probability
    return refined


def low_overlap_high_confidence_counts(
    detections: list[dict[str, torch.Tensor]],
    batch: dict[str, Any],
    validator: Any,
    confidence: float = 0.5,
    overlap_iou: float = 0.1,
) -> tuple[int, int]:
    """Count high-confidence candidates and those with low overlap to every ground-truth box."""
    low_overlap = 0
    high_confidence = 0
    for image_index, detection in enumerate(detections):
        selected = detection["conf"] >= confidence
        count = int(selected.sum())
        high_confidence += count
        if not count:
            continue
        target_boxes = validator._prepare_batch(image_index, batch)["bboxes"]
        if not target_boxes.shape[0]:
            low_overlap += count
            continue
        overlaps = box_iou(detection["bboxes"][selected], target_boxes).max(dim=1).values
        low_overlap += int((overlaps < overlap_iou).sum())
    return low_overlap, high_confidence
