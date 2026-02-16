"""Differentiable EKF with Joseph-form covariance update."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from src.utils.linear_algebra import chol_logdet, chol_solve, safe_cholesky
from .psd import PSDParameter
from .residual_dynamics import ResidualDynamics
from .measurement import LinearMeasurement
from src.models.components.nn_blocks import MLP, MLPConfig  # noqa: F401
from src.utils.jacobians import batch_jacobian


@dataclass
class EKFConfig:
    state_dim: int
    obs_dim: int
    dt: float
    jitter: float = 1e-6


class DifferentiableEKF(nn.Module):
    """Differentiable EKF supporting batched sequences."""

    def __init__(
        self,
        config: EKFConfig,
        dynamics: ResidualDynamics,
        measurement: nn.Module,
        q_param: PSDParameter,
        r_param: PSDParameter,
    ) -> None:
        super().__init__()
        self.config = config
        self.dynamics = dynamics
        self.measurement = measurement
        self.q_param = q_param
        self.r_param = r_param
        self.state_dim = config.state_dim
        self.obs_dim = config.obs_dim
        self.eye = torch.eye(self.state_dim)

    def _q0_matrix(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        base = self.q_param.matrix()
        base = 0.5 * (base + base.transpose(-1, -2))
        return base.to(device=device, dtype=dtype)

    def forward(
        self,
        observations: torch.Tensor,
        x0: torch.Tensor,
        Sigma0: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_jacobians: bool = False,
        hidden: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        B, T, _ = observations.shape
        device = observations.device
        eye = self.eye.to(device)
        obs_eye = torch.eye(self.obs_dim, device=device, dtype=observations.dtype)
        uses_gru = bool(getattr(self.dynamics, "use_gru", False))

        def ensure_batch(t: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
            if t.dim() == len(target_shape) - 1:
                t = t.unsqueeze(0).expand(target_shape)
            return t

        x_filt = ensure_batch(x0, (B, self.state_dim))
        Sigma_filt = ensure_batch(Sigma0, (B, self.state_dim, self.state_dim))
        if uses_gru:
            if hidden is None:
                hidden = self.dynamics.reset_hidden(B, device=device, dtype=observations.dtype)
            else:
                if hidden.size(1) != B:
                    raise ValueError(f"Hidden state batch dim {hidden.size(1)} != observations batch {B}")

        Q0 = self._q0_matrix(device=device, dtype=observations.dtype).unsqueeze(0).expand(B, -1, -1)
        R = self.r_param.matrix().to(device=device, dtype=observations.dtype).unsqueeze(0).expand(B, -1, -1)

        x_filt_history = []
        x_pred_history = []
        Sigma_filt_history = []
        Sigma_pred_history = []
        innovations = []
        S_mats = []
        logdet_S = []
        whitened_innov = []
        delta_history = []
        beta_history = []
        F_history = [] if return_jacobians else None
        H_history = [] if return_jacobians else None

        feature_mode = str(getattr(self.dynamics.config, "feature_mode", "basic")).lower()
        if feature_mode not in {"basic", "advanced"}:
            feature_mode = "basic"

        prev_innov = torch.zeros(B, self.obs_dim, device=device, dtype=observations.dtype)
        prev_dx_update = torch.zeros(B, self.state_dim, device=device, dtype=observations.dtype)
        prev_dx_evolve = torch.zeros(B, self.state_dim, device=device, dtype=observations.dtype)
        prev_dy = torch.zeros(B, self.obs_dim, device=device, dtype=observations.dtype)
        prev_f5 = torch.zeros(B, self.obs_dim, device=device, dtype=observations.dtype)
        prev_H_flat = torch.zeros(
            B, self.obs_dim * self.state_dim, device=device, dtype=observations.dtype
        )
        prev_obs = observations[:, 0].detach().clone()
        phys_jac_fn = getattr(self.dynamics, "f_known_jacobian", None)
        for t in range(T):
            if feature_mode == "basic":
                extra = prev_innov
            else:
                extra = torch.cat(
                    [
                        prev_innov,
                        prev_dx_update,
                        prev_dx_evolve,
                        prev_dy,
                        prev_f5,
                        prev_H_flat,
                    ],
                    dim=-1,
                )

            x_filt_prev = x_filt
            hidden_in = hidden if uses_gru else None
            if uses_gru:
                x_pred, beta_t, _, delta_t, hidden_out = self.dynamics(x_filt, hidden_in, extra=extra)
            else:
                x_pred, beta_t, _, delta_t, _ = self.dynamics(x_filt)

                hidden_out = None

            # Covariance prediction must not depend on the NN mean correction `delta_t`.
            # Use the Jacobian of the nominal/known dynamics f_known.
            if phys_jac_fn is not None:
                F_jac = phys_jac_fn(x_filt.detach())
            else:
                F_jac = batch_jacobian(lambda inp: self.dynamics.f_known(inp), x_filt.detach())
            hidden = hidden_out
            Sigma_filt_phys = Sigma_filt.detach()
            beta_scale = beta_t.reshape(B, 1, 1)
            Sigma_pred = F_jac @ Sigma_filt_phys @ F_jac.transpose(-1, -2) + beta_scale * Q0
            Sigma_pred = 0.5 * (Sigma_pred + Sigma_pred.transpose(-1, -2))
            if return_jacobians and F_history is not None:
                F_history.append(F_jac)

            y_pred = self.measurement(x_pred)
            if hasattr(self.measurement, "H"):
                H_mat = self.measurement.H
                H = H_mat.unsqueeze(0).expand(B, -1, -1)
            else:
                H = batch_jacobian(lambda inp: self.measurement(inp), x_pred)
            S = H @ Sigma_pred @ H.transpose(-1, -2) + R
            if self.config.jitter > 0.0:
                jitter = self.config.jitter
                jitter_eye = obs_eye.unsqueeze(0).expand(B, self.obs_dim, self.obs_dim)
                S = S + jitter * jitter_eye
            S = 0.5 * (S + S.transpose(-1, -2))
            if return_jacobians and H_history is not None:
                H_history.append(H)

            # F5: linearization error proxy h(x) - H x at the linearization point.
            Hx = (H @ x_pred.unsqueeze(-1)).squeeze(-1)
            f5 = y_pred - Hx

            chol_S, _ = safe_cholesky(S, jitter=self.config.jitter)
            logdet = chol_logdet(chol_S)

            obs_t = observations[:, t]
            delta_y = obs_t - y_pred
            if mask is not None:
                delta_y = delta_y.masked_fill(mask[:, t].unsqueeze(-1), 0.0)

            Sigma_HT = Sigma_pred @ H.transpose(-1, -2)
            K = chol_solve(chol_S, Sigma_HT.transpose(-1, -2)).transpose(-1, -2)
            correction = (K @ delta_y.unsqueeze(-1)).squeeze(-1)
            x_filt = x_pred + correction

            KH = K @ H
            I_minus_KH = eye.unsqueeze(0) - KH
            Sigma_filt = (
                I_minus_KH @ Sigma_pred @ I_minus_KH.transpose(-1, -2) + K @ R @ K.transpose(-1, -2)
            )
            Sigma_filt = 0.5 * (Sigma_filt + Sigma_filt.transpose(-1, -2))

            whitened = torch.linalg.solve_triangular(chol_S, delta_y.unsqueeze(-1), upper=False).squeeze(-1)

            x_pred_history.append(x_pred)
            x_filt_history.append(x_filt)
            Sigma_pred_history.append(Sigma_pred)
            Sigma_filt_history.append(Sigma_filt)
            innovations.append(delta_y)
            S_mats.append(S)
            logdet_S.append(logdet)
            whitened_innov.append(whitened)
            delta_history.append(delta_t)
            beta_history.append(beta_t)
            prev_innov = delta_y.detach()
            prev_dx_update = correction.detach()
            prev_dx_evolve = (x_filt - x_filt_prev).detach()
            obs_t_detached = obs_t.detach()
            prev_dy = (obs_t_detached - prev_obs).detach()
            prev_obs = obs_t_detached
            prev_f5 = f5.detach()
            prev_H_flat = H.detach().reshape(B, -1)

        result = {
            "x_pred": torch.stack(x_pred_history, dim=1),
            "x_filt": torch.stack(x_filt_history, dim=1),
            "Sigma_pred": torch.stack(Sigma_pred_history, dim=1),
            "Sigma_filt": torch.stack(Sigma_filt_history, dim=1),
            "innovations": torch.stack(innovations, dim=1),
            "S": torch.stack(S_mats, dim=1),
            "logdet_S": torch.stack(logdet_S, dim=1),
            "whitened": torch.stack(whitened_innov, dim=1),
            "delta": torch.stack(delta_history, dim=1),
            "beta": torch.stack(beta_history, dim=1),
        }
        if uses_gru:
            result["hidden_last"] = hidden
        if return_jacobians and F_history is not None and H_history is not None:
            result["F"] = torch.stack(F_history, dim=1)
            result["H"] = torch.stack(H_history, dim=1)
        if mask is not None:
            result["mask"] = mask
        return result


__all__ = ["DifferentiableEKF", "EKFConfig"]
