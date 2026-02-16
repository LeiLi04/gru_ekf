"""LightningModule wrapping the GRU-augmented EKF with Hydra-configurable components."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pytorch_lightning as pl
import torch
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig

from src.models.components.losses.innov_nll import innovation_nll
from src.eval.metrics.nis_nees import compute_nis
from src.models.components import (
    DifferentiableEKF,
    DynamicsConfig,
    EKFConfig,
    PSDConfig,
    PSDParameter,
    RangeMeasurement,
    ResidualDynamics,
    build_residual_dynamics,
)
from src.train.warmstart_q0 import covariance_matching_warm_start, set_psd_parameter_from_matrix
from src.utils.masking import masked_mean


class GruAugmentedEkfLitModule(pl.LightningModule):
    """Innovation-NLL training for GRU-augmented EKF using Lightning."""

    def __init__(
        self,
        state_dim: int = 4,
        obs_dim: int = 4,
        dt: float = 0.01,
        cov_factor_scale: float = 1.0,
        cov_rank: int = 0,
        dynamics_hidden: int = 128,
        dynamics_depth: int = 3,
        dynamics_use_gru: bool = True,
        dynamics_tanh_scale: float = 0.1,
        dynamics_residual_init_std: float = 1e-3,
        dynamics_scale_a_min: float = 1.0,
        dynamics_scale_a_max: float = 1.0,
        max_delta: Optional[float] = None,
        dynamics_use_beta_head: bool = False,
        dynamics_beta_min: float = 0.1,
        dynamics_beta_max: float = 10.0,
        dynamics_beta_init: float = 1.0,
        dynamics_feature_mode: str = "basic",
        q_init: float = 0.05,
        r_init: float = 0.1,
        train_q: bool = False,
        train_r: bool = False,
        train_stage: str = "joint",
        tbptt_steps: int = 50,
        sigma0_scale: float = 1.0,
        lambda_nis: float = 0.0,
        lambda_delta: float = 1e-4,
        lambda_beta: float = 0.0,
        lambda_beta_smooth: float = 0.0,
        lambda_L: float = 0.0,
        amp: bool = True,
        optimizer: Optional[DictConfig | Dict[str, Any]] = None,
        scheduler: Optional[DictConfig | Dict[str, Any]] = None,
        warmup: Optional[DictConfig | Dict[str, Any]] = None,
        dataset_path: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.optimizer_cfg = optimizer
        self.scheduler_cfg = scheduler
        self.warmup_cfg = warmup or {}

        anchors, q_base, r_base, dt_from_data = self._load_dataset_metadata(dataset_path)
        if dt_from_data is not None:
            dt = float(dt_from_data)

        self.dynamics, self.ekf = self._build_model_components(
            state_dim=state_dim,
            obs_dim=obs_dim,
            dt=dt,
            cov_rank=cov_rank,
            cov_factor_scale=cov_factor_scale,
            dynamics_hidden=dynamics_hidden,
            dynamics_depth=dynamics_depth,
            dynamics_use_gru=dynamics_use_gru,
            dynamics_tanh_scale=dynamics_tanh_scale,
            dynamics_residual_init_std=dynamics_residual_init_std,
            dynamics_scale_a_min=dynamics_scale_a_min,
            dynamics_scale_a_max=dynamics_scale_a_max,
            max_delta=max_delta,
            dynamics_use_beta_head=dynamics_use_beta_head,
            dynamics_beta_min=dynamics_beta_min,
            dynamics_beta_max=dynamics_beta_max,
            dynamics_beta_init=dynamics_beta_init,
            dynamics_feature_mode=dynamics_feature_mode,
            q_init=q_init,
            r_init=r_init,
            train_q=train_q,
            train_r=train_r,
            anchors=anchors,
            q_base=q_base,
            r_base=r_base,
        )
        self.tbptt_steps = int(tbptt_steps) if tbptt_steps else 0
        self.sigma0_scale = float(sigma0_scale)
        self.lambda_nis = float(lambda_nis)
        self.lambda_delta = float(lambda_delta)
        self.lambda_beta = float(lambda_beta)
        self.lambda_beta_smooth = float(lambda_beta_smooth)
        # `lambda_L` historically regularized a low-rank covariance injection term.
        # The current paper draft does not use this term, so it is kept only for
        # backward-compatible configs and logging.
        self.lambda_L = float(lambda_L)
        self.train_q = bool(train_q)
        self.train_r = bool(train_r)
        self.train_stage = str(train_stage)
        if self.train_stage not in {"joint", "delta", "beta"}:
            raise ValueError("train_stage must be one of {'joint', 'delta', 'beta'}.")
        self.dynamics_feature_mode = str(dynamics_feature_mode)
        if self.dynamics_feature_mode not in {"basic", "advanced"}:
            raise ValueError("dynamics_feature_mode must be one of {'basic', 'advanced'}.")
        self.state_dim = int(state_dim)
        self.obs_dim = int(obs_dim)

    # ------------------- build helpers ------------------- #
    def _load_dataset_metadata(
        self, dataset_path: Optional[str]
    ) -> Tuple[torch.Tensor, Optional[np.ndarray], Optional[np.ndarray], Optional[float]]:
        default_anchors = torch.tensor(
            [
                [-1.5, 1.0, -1.0, 1.5],
                [0.5, 1.0, -1.0, -0.5],
            ],
            dtype=torch.float32,
        )
        if not dataset_path:
            return default_anchors, None, None, None
        path = Path(to_absolute_path(dataset_path))
        if not path.exists():
            return default_anchors, None, None, None
        payload = np.load(path, allow_pickle=True)
        anchors_np = payload.get("anchors", None)
        q_base = payload.get("Q", None)
        r_base = payload.get("R", None)
        sigma_r = payload.get("sigma_r", None)
        dt_val = payload.get("dt", None)
        anchors = torch.as_tensor(anchors_np, dtype=torch.float32) if anchors_np is not None else default_anchors
        if q_base is None:
            raise ValueError(f"[dataset] Missing Q in dataset: {path}")
        if r_base is None:
            if sigma_r is None:
                raise ValueError(f"[dataset] Missing R/sigma_r in dataset: {path}")
            sigma_val = float(np.asarray(sigma_r).item())
            obs_dim = int(anchors.shape[1])
            r_base = np.eye(obs_dim, dtype=float) * (sigma_val ** 2)
        return anchors, q_base, r_base, float(dt_val) if dt_val is not None else None

    def _build_model_components(
        self,
        *,
        state_dim: int,
        obs_dim: int,
        dt: float,
        cov_rank: int,
        cov_factor_scale: float,
        dynamics_hidden: int,
        dynamics_depth: int,
        dynamics_use_gru: bool,
        dynamics_tanh_scale: float,
        dynamics_residual_init_std: float,
        dynamics_scale_a_min: float,
        dynamics_scale_a_max: float,
        max_delta: Optional[float],
        dynamics_use_beta_head: bool,
        dynamics_beta_min: float,
        dynamics_beta_max: float,
        dynamics_beta_init: float,
        dynamics_feature_mode: str,
        q_init: float,
        r_init: float,
        train_q: bool,
        train_r: bool,
        anchors: torch.Tensor,
        q_base: Optional[np.ndarray],
        r_base: Optional[np.ndarray],
    ) -> Tuple[ResidualDynamics, DifferentiableEKF]:
        def f_known(x: torch.Tensor) -> torch.Tensor:
            px, py, vx, vy = x[..., 0], x[..., 1], x[..., 2], x[..., 3]
            out = torch.stack([px + vx * dt, py + vy * dt, vx, vy], dim=-1)
            return out

        F_template = torch.tensor(
                [
                    [1.0, 0.0, dt, 0.0],
                    [0.0, 1.0, 0.0, dt],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
                dtype=torch.float32,
            )

        def f_known_jac(x: torch.Tensor) -> torch.Tensor:
            return F_template.to(device=x.device, dtype=x.dtype).expand(x.size(0), -1, -1)

        # The paper's base variant uses a GRU mean correction and keeps the KF
        # covariance recursion intact (no low-rank covariance injection). We
        # therefore force the legacy covariance injection settings off.
        cov_rank = 0
        cov_factor_scale = 1.0
        dynamics_scale_a_min = 1.0
        dynamics_scale_a_max = 1.0

        feature_mode = str(dynamics_feature_mode)
        if feature_mode == "basic":
            input_dim = state_dim + obs_dim
        elif feature_mode == "advanced":
            # Filter-internal signals (prev step): innov, dx_update, dx_evolve, dy_diff,
            # plus optional nonlinear measurement cues: F5 and flattened Jacobian H.
            extra_dim = obs_dim + state_dim + state_dim + obs_dim + obs_dim + (obs_dim * state_dim)
            input_dim = state_dim + extra_dim
        else:
            raise ValueError(f"Unknown dynamics_feature_mode: {feature_mode}")

        dyn_cfg = DynamicsConfig(
            state_dim=state_dim,
            input_dim=input_dim,
            hidden_dim=dynamics_hidden,
            depth=dynamics_depth,
            cov_rank=cov_rank,
            cov_factor_scale=cov_factor_scale,
            use_gru=dynamics_use_gru,
            dt=dt,
            tanh_scale=dynamics_tanh_scale,
            residual_init_std=dynamics_residual_init_std,
            max_delta=max_delta,
            scale_a_min=dynamics_scale_a_min,
            scale_a_max=dynamics_scale_a_max,
            use_beta_head=dynamics_use_beta_head,
            beta_min=dynamics_beta_min,
            beta_max=dynamics_beta_max,
            beta_init=dynamics_beta_init,
            feature_mode=feature_mode,
        )
        dynamics = build_residual_dynamics(dyn_cfg, f_known=f_known, phys_derivative=None)
        dynamics.f_known_jacobian = lambda x: f_known_jac(x)  # type: ignore[attr-defined]

        if q_base is not None:
            q_base_t = torch.as_tensor(q_base, dtype=torch.float32)
            q_init_scale = float(torch.mean(torch.diagonal(q_base_t))) ** 0.5
        else:
            q_base_t = None
            q_init_scale = float(q_init) ** 0.5

        if r_base is not None:
            r_base_t = torch.as_tensor(r_base, dtype=torch.float32)
            r_init_scale = float(torch.mean(torch.diagonal(r_base_t))) ** 0.5
        else:
            r_base_t = None
            r_init_scale = float(r_init) ** 0.5

        q_param = PSDParameter(PSDConfig(state_dim, init_scale=q_init_scale))
        r_param = PSDParameter(PSDConfig(obs_dim, init_scale=r_init_scale))
        if q_base_t is not None:
            set_psd_parameter_from_matrix(q_param, q_base_t, jitter=1e-6)
        else:
            set_psd_parameter_from_matrix(q_param, torch.eye(state_dim) * q_init, jitter=1e-6)
        if r_base_t is not None:
            set_psd_parameter_from_matrix(r_param, r_base_t, jitter=1e-5)
        else:
            set_psd_parameter_from_matrix(r_param, torch.eye(obs_dim) * r_init, jitter=1e-5)
        for p in q_param.parameters():
            p.requires_grad_(train_q)
        for p in r_param.parameters():
            p.requires_grad_(train_r)

        measurement = RangeMeasurement(anchors)
        ekf_cfg = EKFConfig(state_dim=state_dim, obs_dim=obs_dim, dt=dt)
        ekf = DifferentiableEKF(ekf_cfg, dynamics, measurement, q_param, r_param)
        return dynamics, ekf

    # ------------------- Lightning hooks ------------------- #
    def _apply_train_stage(self) -> None:
        stage = self.train_stage

        if stage == "joint":
            for p in self.dynamics.parameters():
                p.requires_grad_(True)
            return

        if stage == "delta":
            if hasattr(self.dynamics, "use_beta_head"):
                self.dynamics.use_beta_head = False  # type: ignore[attr-defined]
            for name, p in self.dynamics.named_parameters():
                p.requires_grad_(not name.startswith("fc_beta"))
            return

        if stage == "beta":
            if hasattr(self.dynamics, "use_beta_head"):
                self.dynamics.use_beta_head = True  # type: ignore[attr-defined]
            for p in self.dynamics.parameters():
                p.requires_grad_(False)
            fc_beta = getattr(self.dynamics, "fc_beta", None)
            if fc_beta is None:
                raise ValueError("train_stage='beta' requires GRU dynamics with fc_beta.")
            for p in fc_beta.parameters():
                p.requires_grad_(True)
            return

        raise ValueError(f"Unknown train_stage: {stage}")

    def on_fit_start(self) -> None:
        self._apply_train_stage()
        if not self.warmup_cfg or not self.warmup_cfg.get("use", False):
            return
        if self.trainer is None or self.trainer.datamodule is None:
            return
        warm_loader = getattr(self.trainer.datamodule, "warm_dataloader", lambda: None)()
        if warm_loader is None:
            return
        eps = float(self.warmup_cfg.get("eps", 1e-6))
        for p in self.ekf.q_param.parameters():
            p.requires_grad_(True)
        q0_star = covariance_matching_warm_start(self.ekf, warm_loader, device=self.device, eps=eps)
        for p in self.ekf.q_param.parameters():
            p.requires_grad_(self.train_q)
        # Lightning does not allow `self.log()` inside `on_fit_start`.
        # Store for logging in `on_train_start`.
        self._warmup_q0_star = float(q0_star)

    def on_train_start(self) -> None:
        q0_star = getattr(self, "_warmup_q0_star", None)
        if q0_star is None:
            return
        # Log once at the beginning of training.
        self.log("warmup/q0", float(q0_star), prog_bar=False, on_step=False, on_epoch=True)
        delattr(self, "_warmup_q0_star")

    def _initial_state(self, batch_size: int, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
        x0 = torch.zeros(batch_size, self.state_dim, device=self.device, dtype=dtype)
        Sigma0 = (
            torch.eye(self.state_dim, device=self.device, dtype=dtype)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            * self.sigma0_scale
        )
        return x0, Sigma0

    def _step(self, batch, stage: str) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        obs, mask = batch
        obs = obs.to(self.device)
        mask = mask.to(self.device) if mask is not None else None
        B, T, _ = obs.shape
        chunk_size = self.tbptt_steps if self.tbptt_steps else T
        chunk_size = max(int(chunk_size), 1)
        x0, Sigma0 = self._initial_state(B, obs.dtype)
        hidden = (
            self.dynamics.reset_hidden(B, device=self.device, dtype=obs.dtype)
            if getattr(self.dynamics, "use_gru", False)
            else None
        )

        total_loss = torch.zeros((), device=self.device)
        nis_accum = torch.zeros((), device=self.device)
        reg_delta_accum = torch.zeros((), device=self.device)
        reg_beta_accum = torch.zeros((), device=self.device)
        reg_beta_smooth_accum = torch.zeros((), device=self.device)
        beta_mean_accum = torch.zeros((), device=self.device)
        steps = 0

        for start_t in range(0, T, chunk_size):
            end_t = min(start_t + chunk_size, T)
            obs_chunk = obs[:, start_t:end_t, :]
            mask_chunk = mask[:, start_t:end_t] if mask is not None else None
            outputs = self.ekf(obs_chunk, x0, Sigma0, mask=mask_chunk, hidden=hidden)

            delta_y = outputs["innovations"]
            S = outputs["S"]
            logdet_S = outputs["logdet_S"]
            loss = innovation_nll(delta_y, S, logdet_S, mask_chunk)

            nis_vals, _ = compute_nis(delta_y, S, mask_chunk)
            nis_mean = masked_mean(nis_vals.unsqueeze(-1), mask_chunk).mean()
            if self.lambda_nis > 0.0:
                nis_penalty = (nis_mean - self.obs_dim) ** 2
                loss = loss + self.lambda_nis * nis_penalty
            if self.lambda_delta > 0.0 and "delta" in outputs:
                delta = outputs["delta"]
                if mask_chunk is not None:
                    valid = (~mask_chunk).unsqueeze(-1).to(delta.dtype)
                    denom = valid.sum().clamp(min=1.0)
                    delta_reg = (delta.pow(2) * valid).sum() / denom
                else:
                    delta_reg = delta.pow(2).mean()
                loss = loss + self.lambda_delta * delta_reg
                reg_delta_accum += delta_reg.detach()

            if (self.lambda_beta > 0.0 or self.lambda_beta_smooth > 0.0) and "beta" in outputs:
                beta = outputs["beta"]
                log_beta = torch.log(beta.clamp_min(1e-12))
                if mask_chunk is not None:
                    valid = (~mask_chunk).unsqueeze(-1).to(log_beta.dtype)
                    denom = valid.sum().clamp(min=1.0)
                    beta_reg = (log_beta.pow(2) * valid).sum() / denom
                else:
                    beta_reg = log_beta.pow(2).mean()

                beta_smooth = torch.zeros((), device=self.device)
                if log_beta.size(1) > 1:
                    d_log_beta = log_beta[:, 1:, :] - log_beta[:, :-1, :]
                    if mask_chunk is not None:
                        diff_mask = mask_chunk[:, 1:] | mask_chunk[:, :-1]
                        valid_diff = (~diff_mask).unsqueeze(-1).to(d_log_beta.dtype)
                        denom_diff = valid_diff.sum().clamp(min=1.0)
                        beta_smooth = (d_log_beta.pow(2) * valid_diff).sum() / denom_diff
                    else:
                        beta_smooth = d_log_beta.pow(2).mean()

                loss = loss + self.lambda_beta * beta_reg + self.lambda_beta_smooth * beta_smooth
                reg_beta_accum += beta_reg.detach()
                reg_beta_smooth_accum += beta_smooth.detach()

            if "beta" in outputs:
                beta_mean = masked_mean(outputs["beta"], mask_chunk).mean()
                beta_mean_accum += beta_mean.detach()

            total_loss += loss
            nis_accum += nis_mean.detach()
            steps += 1

            x0 = outputs["x_filt"][:, -1].detach()
            Sigma0 = outputs["Sigma_filt"][:, -1].detach()
            hidden = outputs.get("hidden_last")
            if hidden is not None:
                hidden = hidden.detach()

        denom = max(steps, 1)
        metrics = {
            f"{stage}_nll": total_loss / denom,
            f"{stage}_nis": nis_accum / denom,
            f"{stage}_reg_delta": reg_delta_accum / denom,
            f"{stage}_reg_beta": reg_beta_accum / denom,
            f"{stage}_reg_beta_smooth": reg_beta_smooth_accum / denom,
            f"{stage}_beta_mean": beta_mean_accum / denom,
            f"{stage}_reg_L": torch.zeros((), device=self.device),
        }
        return metrics[f"{stage}_nll"], metrics

    # ------------------- Lightning API ------------------- #
    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss, metrics = self._step(batch, stage="train")
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        return loss

    def validation_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss, metrics = self._step(batch, stage="val")
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        return loss

    def test_step(self, batch, batch_idx: int) -> torch.Tensor:
        loss, metrics = self._step(batch, stage="test")
        self.log_dict(metrics, on_step=False, on_epoch=True, prog_bar=True, sync_dist=False)
        return loss

    def configure_optimizers(self):
        params = list(self.dynamics.parameters())
        if self.train_q:
            params += list(self.ekf.q_param.parameters())
        if self.train_r:
            params += list(self.ekf.r_param.parameters())
        opt_cfg = self.optimizer_cfg or {"_target_": "torch.optim.Adam", "lr": 1e-3}
        optimizer = instantiate(opt_cfg, params)
        if not self.scheduler_cfg:
            return optimizer
        scheduler = instantiate(self.scheduler_cfg, optimizer)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "monitor": "val_nll"},
            }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


__all__ = ["GruAugmentedEkfLitModule"]
