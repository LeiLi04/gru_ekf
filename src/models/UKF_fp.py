"""Generic Unscented Kalman Filter (UKF) implementation."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

StateTransitionFn = Callable[[np.ndarray], np.ndarray]
MeasurementFn = Callable[[np.ndarray], np.ndarray]


def _as_vector(name: str, vector: np.ndarray, size: int) -> np.ndarray:
    array = np.asarray(vector, dtype=float).reshape(-1)
    if array.shape[0] != size:
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    return array


def _as_square_matrix(name: str, matrix: np.ndarray, size: int) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.shape != (size, size):
        raise ValueError(f"{name} must have shape ({size}, {size}), got {array.shape}")
    return array


def _apply_fn(transform_fn: Callable[[np.ndarray], np.ndarray], points: np.ndarray) -> np.ndarray:
    points_array = np.asarray(points, dtype=float)
    if points_array.ndim == 1:
        return np.asarray(transform_fn(points_array), dtype=float)
    try:
        result = transform_fn(points_array)
        result = np.asarray(result, dtype=float)
        if result.shape[0] == points_array.shape[0]:
            return result
    except Exception:
        pass
    return np.asarray([transform_fn(point) for point in points_array], dtype=float)


class UKF:
    """Generic UKF for nonlinear state and measurement models."""

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        state: Optional[np.ndarray] = None,
        covariance: Optional[np.ndarray] = None,
        process_noise: Optional[np.ndarray] = None,
        measurement_noise: Optional[np.ndarray] = None,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
    ) -> None:
        self.state_dim = int(state_dim)
        self.obs_dim = int(obs_dim)
        self.state = _as_vector("state", state if state is not None else np.zeros(self.state_dim), self.state_dim)
        self.covariance = _as_square_matrix(
            "covariance",
            covariance if covariance is not None else np.eye(self.state_dim),
            self.state_dim,
        )
        self.process_noise = (
            _as_square_matrix("process_noise", process_noise, self.state_dim) if process_noise is not None else None
        )
        self.measurement_noise = (
            _as_square_matrix("measurement_noise", measurement_noise, self.obs_dim) if measurement_noise is not None else None
        )
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.kappa = float(kappa)
        self._sigma_points_pred: Optional[np.ndarray] = None
        self._weights_mean: Optional[np.ndarray] = None
        self._weights_cov: Optional[np.ndarray] = None
        self.last_innovation: Optional[np.ndarray] = None
        self.last_innovation_cov: Optional[np.ndarray] = None

    def set_model(
        self,
        process_noise: Optional[np.ndarray] = None,
        measurement_noise: Optional[np.ndarray] = None,
    ) -> None:
        if process_noise is not None:
            self.process_noise = _as_square_matrix("process_noise", process_noise, self.state_dim)
        if measurement_noise is not None:
            self.measurement_noise = _as_square_matrix("measurement_noise", measurement_noise, self.obs_dim)

    def reset(self, state: Optional[np.ndarray] = None, covariance: Optional[np.ndarray] = None) -> None:
        if state is not None:
            self.state = _as_vector("state", state, self.state_dim)
        if covariance is not None:
            self.covariance = _as_square_matrix("covariance", covariance, self.state_dim)
        self._sigma_points_pred = None
        self._weights_mean = None
        self._weights_cov = None
        self.last_innovation = None
        self.last_innovation_cov = None

    def _sigma_points(self, state: np.ndarray, covariance: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        scaling_param = self.alpha * self.alpha * (self.state_dim + self.kappa) - self.state_dim
        scaling = self.state_dim + scaling_param
        sqrt_matrix = np.linalg.cholesky(scaling * covariance)
        points = [state]
        for index in range(self.state_dim):
            offset = sqrt_matrix[:, index]
            points.append(state + offset)
            points.append(state - offset)
        sigma_points = np.stack(points, axis=0)
        weights_mean = np.full(2 * self.state_dim + 1, 1.0 / (2.0 * scaling), dtype=float)
        weights_cov = weights_mean.copy()
        weights_mean[0] = scaling_param / scaling
        weights_cov[0] = scaling_param / scaling + (1.0 - self.alpha * self.alpha + self.beta)
        return sigma_points, weights_mean, weights_cov

    def predict(self, state_transition: StateTransitionFn) -> Tuple[np.ndarray, np.ndarray]:
        if self.process_noise is None:
            raise ValueError("process_noise is required before calling predict().")

        sigma_points, weights_mean, weights_cov = self._sigma_points(self.state, self.covariance)
        sigma_points_pred = _apply_fn(state_transition, sigma_points)
        if sigma_points_pred.ndim == 1:
            sigma_points_pred = sigma_points_pred[:, None]
        predicted_state = np.sum(weights_mean[:, None] * sigma_points_pred, axis=0)

        predicted_covariance = np.array(self.process_noise, copy=True)
        for point_index in range(sigma_points_pred.shape[0]):
            diff = sigma_points_pred[point_index] - predicted_state
            predicted_covariance += weights_cov[point_index] * np.outer(diff, diff)

        self.state = predicted_state
        self.covariance = predicted_covariance
        self._sigma_points_pred = sigma_points_pred
        self._weights_mean = weights_mean
        self._weights_cov = weights_cov
        return self.state, self.covariance

    def update(
        self,
        measurement: np.ndarray,
        measurement_fn: MeasurementFn,
        measurement_noise: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        noise_matrix = measurement_noise
        if noise_matrix is None:
            noise_matrix = self.measurement_noise
        if noise_matrix is None:
            raise ValueError("measurement_noise is required before calling update().")

        measurement_vector = _as_vector("measurement", measurement, self.obs_dim)

        if self._sigma_points_pred is None or self._weights_mean is None or self._weights_cov is None:
            sigma_points_pred, weights_mean, weights_cov = self._sigma_points(self.state, self.covariance)
        else:
            sigma_points_pred = self._sigma_points_pred
            weights_mean = self._weights_mean
            weights_cov = self._weights_cov

        predicted_measurements = _apply_fn(measurement_fn, sigma_points_pred)
        if predicted_measurements.ndim == 1:
            predicted_measurements = predicted_measurements[:, None]
        measurement_mean = np.sum(weights_mean[:, None] * predicted_measurements, axis=0)

        innovation_cov = np.array(noise_matrix, copy=True)
        cross_covariance = np.zeros((self.state_dim, self.obs_dim), dtype=float)

        predicted_state = self.state.copy()
        for point_index in range(sigma_points_pred.shape[0]):
            state_diff = sigma_points_pred[point_index] - predicted_state
            meas_diff = predicted_measurements[point_index] - measurement_mean
            innovation_cov += weights_cov[point_index] * np.outer(meas_diff, meas_diff)
            cross_covariance += weights_cov[point_index] * np.outer(state_diff, meas_diff)

        kalman_gain = cross_covariance @ np.linalg.inv(innovation_cov)
        innovation = measurement_vector - measurement_mean
        self.state = self.state + kalman_gain @ innovation
        self.covariance = self.covariance - kalman_gain @ innovation_cov @ kalman_gain.T

        self.last_innovation = innovation
        self.last_innovation_cov = innovation_cov
        return self.state, self.covariance, innovation, innovation_cov


__all__ = ["UKF"]
