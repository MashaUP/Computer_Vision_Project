"""LeJEPA projectors applied to the deepest encoder map (used only during pre-training)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import MLP


class ClassicProjector(nn.Module):
    """Global-average-pooled feature -> MLP with BatchNorm (official LeJEPA: 2048-2048-512)."""

    def __init__(self, in_channels: int, dims: tuple[int, ...] = (2048, 2048, 512)):
        super().__init__()
        self.mlp = MLP(in_channels, list(dims), norm_layer=nn.BatchNorm1d)
        self.out_dim = dims[-1]

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:  # (B, C, h, w) -> (B, D)
        return self.mlp(fmap.mean(dim=(-2, -1)))


class DenseProjector(nn.Module):
    """Pool to a ``grid`` x ``grid`` token map -> per-token MLP with LayerNorm (METER: 256-512-64)."""

    def __init__(self, in_channels: int, dims: tuple[int, ...] = (256, 512, 64), grid: int = 4):
        super().__init__()
        self.mlp = MLP(in_channels, list(dims), norm_layer=nn.LayerNorm)
        self.grid = grid
        self.out_dim = dims[-1]

    def forward(self, fmap: torch.Tensor) -> torch.Tensor:  # (B, C, h, w) -> (B, grid*grid, D)
        tokens = F.adaptive_avg_pool2d(fmap, self.grid).flatten(2).transpose(1, 2)
        return self.mlp(tokens)
