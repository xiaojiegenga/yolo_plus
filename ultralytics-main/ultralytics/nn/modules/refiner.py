# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Second-stage refinement modules for YOLO instance segmentation."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy

import torch
from torch import nn


class LocalCropRefiner(nn.Module):
    """Classify YOLO candidate crops as either a foreground class or background."""

    def __init__(
        self,
        backbone_layers: Iterable[nn.Module],
        in_channels: int = 512,
        num_classes: int = 3,
    ) -> None:
        """Build an independent backbone copy followed by global pooling and a linear head."""
        super().__init__()
        layers = list(backbone_layers)
        if not layers:
            raise ValueError("backbone_layers must contain at least one module")
        if in_channels <= 0:
            raise ValueError("in_channels must be positive")
        if num_classes <= 1:
            raise ValueError("num_classes must be greater than one")

        self.backbone = nn.Sequential(*deepcopy(layers))
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(in_channels, num_classes)
        self.num_classes = num_classes

    def forward(self, crops: torch.Tensor) -> torch.Tensor:
        """Return class logits for normalized candidate crops in BCHW format."""
        features = self.backbone(crops)
        return self.classifier(self.pool(features).flatten(1))
