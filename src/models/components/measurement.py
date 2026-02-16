"""Measurement models (linear by default)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn


@dataclass
class LinearMeasurementConfig:
    state_dim: int
    obs_dim: int


class LinearMeasurement(nn.Module):
    """Linear measurement y = H x with fixed H."""

    def __init__(self, config: LinearMeasurementConfig, matrix: Optional[torch.Tensor] = None) -> None:
        super().__init__()
        if matrix is None:
            matrix = torch.eye(config.obs_dim, config.state_dim)
        self.H = nn.Parameter(matrix, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.H.transpose(-1, -2)


class RangeMeasurement(nn.Module):
    """Nonlinear range measurement to fixed anchors (standard x - s form)."""

    def __init__(self, anchors: torch.Tensor) -> None:
        super().__init__()
        # anchors: (2, M)
        self.register_buffer("anchors", anchors, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, state_dim)
        pos = x[..., :2]  # (B, 2)
        dx = pos[..., 0:1] - self.anchors[0, :]  # (B, M)
        dy = pos[..., 1:2] - self.anchors[1, :]  # (B, M)
        return torch.sqrt(dx * dx + dy * dy + 1e-8)


__all__ = ["LinearMeasurement", "LinearMeasurementConfig", "RangeMeasurement"]
