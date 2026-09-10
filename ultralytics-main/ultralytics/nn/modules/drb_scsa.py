# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""C3k2-DRB and standalone residual SCSA for the data-v2 structural experiment.

Adapted from AILab-CVC/UniRepLKNet and HZAI-ZJNU/SCSA (Apache-2.0).
Changes: fixed K=9 DRB, PyTorch-only SCSA, FP32 attention reduction, CSP integration.
Upstream license: licenses/DRB-SCSA-Apache-2.0.txt.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import Bottleneck, C3k2
from ultralytics.utils.torch_utils import fuse_conv_and_bn


class DilatedReparamBlock(nn.Module):
    """Depthwise 9x9 convolution plus four dilated Conv-BN branches."""

    branch_specs = ((5, 1), (5, 2), (3, 3), (3, 4))

    def __init__(self, channels: int):
        super().__init__()
        self.large = nn.Conv2d(channels, channels, 9, padding=4, groups=channels, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.branches = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(channels, channels, k, padding=d * (k // 2), dilation=d,
                          groups=channels, bias=False),
                nn.BatchNorm2d(channels),
            )
            for k, d in self.branch_specs
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "bn"):
            return self.large(x)
        y = self.bn(self.large(x))
        for branch in self.branches:
            y = y + branch(x)
        return y

    @torch.no_grad()
    def fuse(self) -> None:
        """Merge running-statistic BN and dilated kernels for evaluation."""
        if not hasattr(self, "bn"):
            return
        merged = fuse_conv_and_bn(self.large, self.bn)
        for branch, (k, d) in zip(self.branches, self.branch_specs):
            conv = fuse_conv_and_bn(branch[0], branch[1])
            size = (k - 1) * d + 1
            kernel = conv.weight.new_zeros((*conv.weight.shape[:2], size, size))
            kernel[:, :, ::d, ::d] = conv.weight
            pad = (9 - size) // 2
            merged.weight.add_(F.pad(kernel, (pad, pad, pad, pad)))
            merged.bias.add_(conv.bias)
        self.large = merged
        del self.bn, self.branches


class C3k2DRB(C3k2):
    """Replace internal Bottlenecks with DRB, preserving CSP projections and PSA."""

    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = False,
                 e: float = 0.5, attn: bool = False, g: int = 1, shortcut: bool = True):
        super().__init__(c1, c2, n, c3k, e, attn, g, shortcut)
        self._replace_bottlenecks(self.m)

    @classmethod
    def _replace_bottlenecks(cls, module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            if isinstance(child, Bottleneck):
                channels = child.cv1.conv.in_channels
                if channels != child.cv2.conv.out_channels:
                    raise ValueError("C3k2-DRB requires equal Bottleneck input/output channels")
                setattr(module, name, DilatedReparamBlock(channels))
            else:
                cls._replace_bottlenecks(child)


class SCSA(nn.Module):
    """Four-group spatial gates followed by single-head channel self-attention."""

    def __init__(self, channels: int):
        super().__init__()
        if channels % 4:
            raise ValueError("SCSA channels must be divisible by four")
        group = channels // 4
        self.spatial_convs = nn.ModuleList(
            nn.Conv1d(group, group, k, padding=k // 2, groups=group)
            for k in (3, 5, 7, 9)
        )
        self.norm_h = nn.GroupNorm(4, channels)
        self.norm_w = nn.GroupNorm(4, channels)
        self.pool = nn.AvgPool2d(7, stride=7)
        self.norm = nn.GroupNorm(1, channels)
        self.q = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)
        self.k = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)
        self.v = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)
        self.scale = channels ** -0.5

    def _spatial(self, x: torch.Tensor, norm: nn.Module) -> torch.Tensor:
        features = [conv(part) for conv, part in zip(self.spatial_convs, x.chunk(4, dim=1))]
        return norm(torch.cat(features, dim=1)).sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self._spatial(x.mean(dim=3), self.norm_h).unsqueeze(3)
        w = self._spatial(x.mean(dim=2), self.norm_w).unsqueeze(2)
        spatial = x * h * w
        y = self.norm(self.pool(spatial))
        q, k, v = (layer(y).flatten(2) for layer in (self.q, self.k, self.v))
        # Keep attention reductions in FP32 under CUDA AMP and half-precision inference.
        with torch.autocast(device_type=x.device.type, enabled=False):
            scores = (q.float() @ k.float().transpose(1, 2)) * self.scale
            gate = (scores.softmax(dim=-1) @ v.float()).mean(dim=2).sigmoid()
        return spatial * gate.to(spatial.dtype).unsqueeze(2).unsqueeze(3)


class ZeroInitResidualSCSA(nn.Module):
    """Standalone Y = X + beta*SCSA(X), with one learned zero-initialized beta."""

    def __init__(self, channels: int):
        super().__init__()
        self.scsa = SCSA(channels)
        self.beta = nn.Parameter(torch.zeros(()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.beta.to(x.dtype) * self.scsa(x)
