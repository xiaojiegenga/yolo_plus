# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Composite checkpoints for a frozen segmentation model and local crop refiner."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

from torch import nn

from ultralytics.utils.patches import torch_load, torch_save


def save_local_refinement_checkpoint(
    path: str | Path,
    base_model: nn.Module,
    refiner: nn.Module,
    metadata: dict[str, Any],
) -> None:
    """Save a standard-loadable base model and its local refiner in one checkpoint."""
    checkpoint = {
        "model": deepcopy(base_model).half().cpu(),
        "ema": None,
        "refiner": deepcopy(refiner).half().cpu(),
        "train_args": metadata.get("train_args", {}),
        "h_metadata": metadata,
    }
    torch_save(checkpoint, Path(path))


def load_local_refinement_checkpoint(path: str | Path) -> tuple[nn.Module, nn.Module, dict[str, Any]]:
    """Load the base model, refiner, and H metadata from a composite checkpoint."""
    checkpoint = torch_load(Path(path), map_location="cpu")
    base_model = checkpoint.get("ema") or checkpoint.get("model")
    refiner = checkpoint.get("refiner")
    if not isinstance(base_model, nn.Module):
        raise TypeError("local refinement checkpoint does not contain a base model")
    if not isinstance(refiner, nn.Module):
        raise TypeError("local refinement checkpoint does not contain a refiner")
    metadata = checkpoint.get("h_metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError("h_metadata must be a mapping")
    return base_model.float(), refiner.float(), metadata
