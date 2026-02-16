"""EKF with online covariance scale adaptation via NIS feedback.

This module implements a minimal online adaptation rule for the process noise scale:

    Q_k = beta_k * Q0,  beta_k > 0

The scale beta_k is updated from the Normalized Innovation Squared (NIS) statistic
using an EMA-smoothed feedback signal and a log-domain update with clipping.

Design notes:
- Predict uses beta_{k-1} (no same-step leakage).
- Update computes NIS from the innovation and innovation covariance S_k.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .EKF_fp import EKF


def _as_float(name: str, value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a float-like value.") from exc
    if not np.isfinite(result):
        raise ValueError(f"{name} must be finite.")
    return result


def _safe_cholesky(A: np.ndarray, jitter: float = 0.0, max_tries: int = 6) -> np.ndarray:
    """Cholesky with diagonal jitter escalation."""
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square 2D matrix.")
    A = np.asarray(A, dtype=float)
    diag = np.eye(A.shape[0], dtype=float)
    jitter_val = float(jitter)
    last_err: Optional[Exception] = None
    for _ in range(max_tries):
        try:
            return np.linalg.cholesky(A + jitter_val * diag)
        except np.linalg.LinAlgError as exc:
            last_err = exc
            jitter_val = 1e-9 if jitter_val <= 0.0 else jitter_val * 10.0
    raise np.linalg.LinAlgError("Cholesky failed even with jitter escalation.") from last_err


def _nis_from_innovation(innovation: np.ndarray, S: np.ndarray, jitter: float = 0.0) -> float:
    """Compute NIS = innovation^T S^{-1} innovation via Cholesky solve."""
    innovation = np.asarray(innovation, dtype=float).reshape(-1)
    S = np.asarray(S, dtype=float)
    L = _safe_cholesky(S, jitter=jitter)
    v = np.linalg.solve(L, innovation.reshape(-1, 1)).reshape(-1)
    return float(v @ v)


@dataclass(frozen=True)
class OnlineBetaConfig:
    """Hyperparameters for the beta update."""

    rho: float = 0.05
    eta: float = 0.1
    c: float = 0.5
    beta_min: float = 1e-3
    beta_max: float = 1e3
    jitter: float = 1e-9


class OnlineBetaEKF:
    """A thin wrapper around `EKF` that adapts process-noise scale online."""

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        *,
        state: Optional[np.ndarray] = None,
        covariance: Optional[np.ndarray] = None,
        transition_matrix: Optional[np.ndarray] = None,
        Q0: np.ndarray,
        measurement_noise: np.ndarray,
        beta_init: float = 1.0,
        config: Optional[OnlineBetaConfig] = None,
    ) -> None:
        self.config = config or OnlineBetaConfig()

        beta_init_f = _as_float("beta_init", beta_init)
        if beta_init_f <= 0.0:
            raise ValueError("beta_init must be > 0.")

        beta_min = _as_float("beta_min", self.config.beta_min)
        beta_max = _as_float("beta_max", self.config.beta_max)
        if beta_min <= 0.0 or beta_max <= 0.0 or beta_min >= beta_max:
            raise ValueError("Require 0 < beta_min < beta_max.")

        rho = _as_float("rho", self.config.rho)
        if not (0.0 < rho <= 1.0):
            raise ValueError("rho must be in (0, 1].")

        self.Q0 = np.asarray(Q0, dtype=float)
        if self.Q0.shape != (state_dim, state_dim):
            raise ValueError(f"Q0 must be shape {(state_dim, state_dim)}.")

        self._log_beta = float(np.clip(np.log(beta_init_f), np.log(beta_min), np.log(beta_max)))
        self._r_bar = 1.0

        self.ekf = EKF(
            state_dim,
            obs_dim,
            state=state,
            covariance=covariance,
            process_noise=self.beta * self.Q0,
            measurement_noise=measurement_noise,
            transition_matrix=transition_matrix,
        )

        self.last_nis: Optional[float] = None
        self.last_nis_norm: Optional[float] = None

    @property
    def beta(self) -> float:
        beta = float(np.exp(self._log_beta))
        return float(np.clip(beta, float(self.config.beta_min), float(self.config.beta_max)))

    @property
    def r_bar(self) -> float:
        return float(self._r_bar)

    def reset(
        self,
        *,
        state: Optional[np.ndarray] = None,
        covariance: Optional[np.ndarray] = None,
        beta: Optional[float] = None,
        r_bar: float = 1.0,
    ) -> None:
        self.ekf.reset(state=state, covariance=covariance)
        if beta is not None:
            beta_f = _as_float("beta", beta)
            if beta_f <= 0.0:
                raise ValueError("beta must be > 0.")
            self._log_beta = float(
                np.clip(
                    np.log(beta_f),
                    np.log(self.config.beta_min),
                    np.log(self.config.beta_max),
                )
            )
        self._r_bar = _as_float("r_bar", r_bar)
        self.last_nis = None
        self.last_nis_norm = None

    def predict(
        self,
        *,
        state_transition=None,
        transition_jacobian=None,
        control_input: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        # Use beta_{k-1} for the current predicted covariance (no same-step leakage).
        self.ekf.set_model(process_noise=self.beta * self.Q0)
        return self.ekf.predict(
            state_transition=state_transition,
            transition_jacobian=transition_jacobian,
            control_input=control_input,
        )

    def update(
        self,
        measurement: np.ndarray,
        *,
        measurement_fn,
        measurement_jacobian,
        measurement_noise: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        x, P, innovation, S = self.ekf.update(
            measurement,
            measurement_fn=measurement_fn,
            measurement_jacobian=measurement_jacobian,
            measurement_noise=measurement_noise,
        )

        nis = _nis_from_innovation(innovation, S, jitter=self.config.jitter)
        nis_norm = nis / float(self.ekf.obs_dim)
        self.last_nis = float(nis)
        self.last_nis_norm = float(nis_norm)

        # EMA of the normalized NIS ratio; target mean is 1 under correct stats.
        rho = float(self.config.rho)
        self._r_bar = (1.0 - rho) * float(self._r_bar) + rho * float(nis_norm)

        # Log-domain beta update with inner/outer clipping.
        eta = float(self.config.eta)
        c = float(self.config.c)
        beta_min = float(self.config.beta_min)
        beta_max = float(self.config.beta_max)

        log_r_bar = float(np.log(max(self._r_bar, 1e-12)))
        log_r_bar = float(np.clip(log_r_bar, -c, c))
        self._log_beta = float(self._log_beta + eta * log_r_bar)
        self._log_beta = float(np.clip(self._log_beta, np.log(beta_min), np.log(beta_max)))

        return x, P, innovation, S, self.beta


__all__ = ["OnlineBetaEKF", "OnlineBetaConfig"]
