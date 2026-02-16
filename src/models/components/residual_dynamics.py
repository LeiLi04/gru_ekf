"""Residual dynamics model (GRU/MLP) for mean correction and process-noise scaling.

This module implements a residual correction to the nominal dynamics:

    x^-_k = f_known(x_{k-1}) + delta_k

where `delta_k` is predicted by a lightweight GRU (or MLP) using filter-internal
signals (e.g., the previous filtered state and previous innovation).

Note:
- The GRU provides a bounded mean correction `delta_k` for the state prediction.
- Optionally, the GRU can also output a scalar `beta_k > 0` that scales the
  process noise term in the covariance prediction: Sigma^- = F Sigma F^T + beta Q0.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Optional

import torch
from torch import nn

from src.models.components.nn_blocks import ResidualMLP  # type: ignore

StateFn = Callable[[torch.Tensor], torch.Tensor]


@dataclass
class DynamicsConfig:
    state_dim: int
    input_dim: Optional[int] = None
    hidden_dim: int = 128
    depth: int = 3
    cov_rank: int = 0
    cov_factor_scale: float = 1.0
    use_gru: bool = True
    dt: float = 0.02
    tanh_scale: float = 0.1
    residual_init_std: float = 1e-3
    max_delta: Optional[float] = None
    scale_a_min: float = 1.0
    scale_a_max: float = 1.0
    use_beta_head: bool = False
    beta_min: float = 0.1
    beta_max: float = 10.0
    beta_init: float = 1.0
    feature_mode: str = "basic"


class ResidualDynamics(nn.Module):
    """Residual dynamics f(x) = f_known(x) + delta_theta(x, extra)."""

    def __init__(
        self,
        config: DynamicsConfig,
        f_known: Optional[StateFn] = None,
        phys_derivative: Optional[StateFn] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.state_dim = config.state_dim
        self.dt = config.dt
        self.f_known = f_known or (lambda x: x)
        self.phys_derivative_fn = phys_derivative
        self.use_gru = bool(config.use_gru)
        self.input_dim = config.input_dim if config.input_dim is not None else config.state_dim
        self.extra_dim = max(0, self.input_dim - self.state_dim)
        self.max_delta = float(config.max_delta if config.max_delta is not None else config.tanh_scale)
        self.a_min = float(config.scale_a_min)
        self.a_max = float(config.scale_a_max)
        self.use_beta_head = bool(config.use_beta_head)
        self.beta_min = float(config.beta_min)
        self.beta_max = float(config.beta_max)
        self.beta_init = float(config.beta_init)
        self.cov_rank = max(0, int(config.cov_rank))
        self.cov_factor_scale = float(config.cov_factor_scale)

        if self.use_gru:
            self.residual_net = nn.GRU(
                input_size=self.input_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.depth,
                batch_first=True,
            )
            self.fc_delta = nn.Linear(config.hidden_dim, self.state_dim)
            self.fc_beta = nn.Linear(config.hidden_dim, 1)
            self.fc_scale = None
            self.fc_cov = None
        else:
            self.residual_net = ResidualMLP(
                in_features=self.input_dim,
                hidden_features=config.hidden_dim,
                out_features=self.state_dim,
                depth=config.depth,
                gated=False,
            )
            self.fc_beta = None
        self.tanh_scale = float(config.tanh_scale)
        self._zero_init_residual()

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
        extra: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        known = self.f_known(x)
        if extra is None and self.extra_dim > 0:
            extra = torch.zeros(x.size(0), self.extra_dim, device=x.device, dtype=x.dtype)
        residual_input = torch.cat([x, extra], dim=-1) if extra is not None else x

        if self.use_gru:
            seq = residual_input.unsqueeze(1)
            gru_out, h_next = self.residual_net(seq, hidden)
            h_last = gru_out[:, -1, :]
            delta_raw = self.fc_delta(h_last)
            delta = self.max_delta * torch.tanh(delta_raw)
            mixed = known + delta
            if self.use_beta_head:
                log_beta_raw = self.fc_beta(h_last)
                log_beta = torch.clamp(
                    log_beta_raw,
                    min=math.log(self.beta_min),
                    max=math.log(self.beta_max),
                )
                beta_t = torch.exp(log_beta)
            else:
                beta_t = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
            return mixed, beta_t, None, delta, h_next

        residual = torch.tanh(self.residual_net(residual_input)) * self.tanh_scale
        mixed = known + residual
        beta_t = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        delta = residual
        return mixed, beta_t, None, delta, None

    def phys_derivative(self, x: torch.Tensor) -> torch.Tensor:
        if self.phys_derivative_fn is not None:
            return self.phys_derivative_fn(x)
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)
            known_next = self.f_known(x)
        return (known_next - x) / self.dt

    def _zero_init_residual(self) -> None:
        if self.use_gru:
            with torch.no_grad():
                self.fc_delta.weight.normal_(mean=0.0, std=self.config.residual_init_std)
                self.fc_delta.bias.zero_()
                if hasattr(self, "fc_beta") and self.fc_beta is not None:
                    self.fc_beta.weight.zero_()
                    self.fc_beta.bias.fill_(math.log(max(self.beta_init, 1e-12)))
        else:
            with torch.no_grad():
                if hasattr(self.residual_net, "output_proj"):
                    self.residual_net.output_proj.weight.normal_(
                        mean=0.0, std=self.config.residual_init_std
                    )
                    self.residual_net.output_proj.bias.zero_()

    def reset_hidden(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Optional[torch.Tensor]:
        if not self.use_gru:
            return None
        return torch.zeros(self.config.depth, batch_size, self.config.hidden_dim, device=device, dtype=dtype)


def build_residual_dynamics(
    config: DynamicsConfig,
    f_known: Optional[StateFn] = None,
    phys_derivative: Optional[StateFn] = None,
) -> ResidualDynamics:
    return ResidualDynamics(config=config, f_known=f_known, phys_derivative=phys_derivative)


__all__ = ["ResidualDynamics", "DynamicsConfig", "build_residual_dynamics"]
