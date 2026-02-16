"""PSD parameterisations for process/measurement covariances."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn


@dataclass
class PSDConfig:
    dim: int
    init_scale: float = 1.0
    min_diag: float = 1e-5


class PSDParameter(nn.Module):
    """Lower-triangular PSD factor using softplus for the diagonal."""

    def __init__(self, config: PSDConfig) -> None:
        super().__init__()
        self.config = config
        tril_indices = torch.tril_indices(config.dim, config.dim)
        self.register_buffer("tril_rows", tril_indices[0], persistent=False)
        self.register_buffer("tril_cols", tril_indices[1], persistent=False)
        num_params = config.dim * (config.dim + 1) // 2
        init = torch.zeros(num_params)
        with torch.no_grad():
            init[self.tril_rows == self.tril_cols] = torch.log(
                torch.exp(torch.tensor(config.init_scale)) - 1
            )
        self.raw = nn.Parameter(init)
        self.softplus = nn.Softplus()

    def lower_triangle(self) -> torch.Tensor:
        dim = self.config.dim
        tril_rows = self.tril_rows
        tril_cols = self.tril_cols
        L = torch.zeros(dim, dim, device=self.raw.device, dtype=self.raw.dtype)
        L[tril_rows, tril_cols] = self.raw
        diag_mask = tril_rows == tril_cols
        if diag_mask.any():
            diag_raw = self.raw[diag_mask]
            diag_vals = self.softplus(diag_raw) + self.config.min_diag
            diag_rows = tril_rows[diag_mask]
            diag_cols = tril_cols[diag_mask]
            L[diag_rows, diag_cols] = diag_vals
        return L

    def matrix(self) -> torch.Tensor:
        L = self.lower_triangle()
        return L @ L.transpose(-1, -2)

    def log_det(self) -> torch.Tensor:
        diag = torch.diagonal(self.lower_triangle(), dim1=-2, dim2=-1)
        return 2.0 * torch.log(diag).sum()


class ScalarPSDParameter(nn.Module):
    """PSD matrix constrained to a scalar scale times a fixed SPD base."""

    def __init__(
        self,
        base_matrix: torch.Tensor,
        clamp_range: Tuple[float, float] = (-3.0, 3.0),
        trainable: bool = True,
        jitter: float = 1e-6,
    ) -> None:
        super().__init__()
        base = 0.5 * (base_matrix + base_matrix.transpose(-1, -2))
        dim = base.size(-1)
        eye = torch.eye(dim, device=base.device, dtype=base.dtype)
        self.register_buffer("base_matrix", base, persistent=True)
        chol = torch.linalg.cholesky(base + jitter * eye)
        self.register_buffer("base_chol", chol, persistent=True)
        self.register_buffer("base_logdet", torch.logdet(base + jitter * eye), persistent=True)
        self.clamp_range = clamp_range
        beta = nn.Parameter(torch.zeros((), device=base.device, dtype=base.dtype), requires_grad=trainable)
        self.register_parameter("beta", beta)

    def set_base(self, new_base: torch.Tensor, jitter: float = 1e-6) -> None:
        base = 0.5 * (new_base + new_base.transpose(-1, -2))
        dim = base.size(-1)
        eye = torch.eye(dim, device=base.device, dtype=base.dtype)
        chol = torch.linalg.cholesky(base + jitter * eye)
        self.base_matrix = base
        self.base_chol = chol
        self.base_logdet = torch.logdet(base + jitter * eye)

    def _scale(self) -> torch.Tensor:
        return torch.exp(self.beta)

    @torch.no_grad()
    def clamp_parameter(self) -> None:
        self.beta.data.clamp_(self.clamp_range[0], self.clamp_range[1])

    def lower_triangle(self) -> torch.Tensor:
        scale = torch.sqrt(self._scale())
        return scale * self.base_chol

    def matrix(self) -> torch.Tensor:
        return self._scale() * self.base_matrix

    def log_det(self) -> torch.Tensor:
        dim = self.base_matrix.size(-1)
        return dim * torch.log(self._scale()) + self.base_logdet


__all__ = ["PSDConfig", "PSDParameter", "ScalarPSDParameter"]
