# MIT License
# Copyright (c) 2023 Wenze Liu
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""DySample LP upsampling, adapted from https://github.com/tiny-smart/dysample.

E uses four sampling groups and the static 0.25 offset factor. Coordinate
construction and bilinear sampling run in FP32, including under AMP.
"""

import torch
from torch import nn
from torch.nn import functional as F


class DySample(nn.Module):
    """Learn content-dependent sampling positions while preserving feature channels.

    Args:
        c1 (int): Input channels, divisible by groups.
        scale (int): Integer spatial upsampling factor.
        groups (int): Channel groups sharing sampling positions.
    """

    def __init__(self, c1: int, scale: int = 2, groups: int = 4):
        super().__init__()
        if c1 % groups:
            raise ValueError(f"DySample channels ({c1}) must be divisible by groups ({groups})")
        self.scale = scale
        self.groups = groups
        self.offset = nn.Conv2d(c1, 2 * groups * scale**2, 1)
        nn.init.normal_(self.offset.weight, std=0.001)
        nn.init.zeros_(self.offset.bias)
        positions = (torch.arange(scale, dtype=torch.float32) - (scale - 1) / 2) / scale
        yy, xx = torch.meshgrid(positions, positions, indexing="ij")
        self.register_buffer("init_pos", torch.stack((xx, yy)).repeat(1, groups, 1).reshape(1, -1, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Sample grouped features at learned offsets and return (B, C, scale*H, scale*W)."""
        batch, channels, height, width = x.shape
        offsets = self.offset(x).float() * 0.25 + self.init_pos.float()
        offsets = offsets.reshape(batch, 2, -1, height, width)
        yy, xx = torch.meshgrid(
            torch.arange(height, device=x.device, dtype=torch.float32) + 0.5,
            torch.arange(width, device=x.device, dtype=torch.float32) + 0.5,
            indexing="ij",
        )
        centers = torch.stack((xx, yy)).reshape(1, 2, 1, height, width)
        normalizer = x.new_tensor((width, height), dtype=torch.float32).reshape(1, 2, 1, 1, 1)
        grid = 2 * (centers + offsets) / normalizer - 1
        grid = F.pixel_shuffle(grid.reshape(batch, -1, height, width), self.scale)
        grid = grid.reshape(batch, 2, self.groups, self.scale * height, self.scale * width)
        grid = grid.permute(0, 2, 3, 4, 1).flatten(0, 1).contiguous()
        sampled = F.grid_sample(
            x.float().reshape(batch * self.groups, channels // self.groups, height, width),
            grid, mode="bilinear", padding_mode="border", align_corners=False,
        )
        return sampled.reshape(batch, channels, self.scale * height, self.scale * width).to(x.dtype)
