"""Generic Extended Kalman Filter (EKF) implementation."""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np

StateTransitionFn = Callable[[np.ndarray], np.ndarray]
StateTransitionWithControlFn = Callable[[np.ndarray, np.ndarray], np.ndarray]
JacobianFn = Callable[[np.ndarray], np.ndarray]
MeasurementFn = Callable[[np.ndarray], np.ndarray]


def _as_vector(name: str, vector: np.ndarray, size: int) -> np.ndarray:
    array = np.asarray(vector, dtype=float).reshape(-1)
    if array.shape[0] != size:
        raise ValueError(f"{name} must have shape ({size},), got {array.shape}")
    return array


def _as_matrix(name: str, matrix: np.ndarray, rows: int, cols: int) -> np.ndarray:
    array = np.asarray(matrix, dtype=float)
    if array.shape != (rows, cols):
        raise ValueError(f"{name} must have shape ({rows}, {cols}), got {array.shape}")
    return array


def _as_square_matrix(name: str, matrix: np.ndarray, size: int) -> np.ndarray:
    return _as_matrix(name, matrix, size, size)


def _call_state_transition(
    state_transition: Callable[..., np.ndarray],
    state: np.ndarray,
    control_input: Optional[np.ndarray],
) -> np.ndarray:
    if control_input is None:
        return state_transition(state)
    try:
        return state_transition(state, control_input)
    except TypeError:
        return state_transition(state)


class EKF:
    """Generic EKF for nonlinear observation models."""

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        state: Optional[np.ndarray] = None,
        covariance: Optional[np.ndarray] = None,
        process_noise: Optional[np.ndarray] = None,
        measurement_noise: Optional[np.ndarray] = None,
        transition_matrix: Optional[np.ndarray] = None,
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
        self.transition_matrix = (
            _as_square_matrix("transition_matrix", transition_matrix, self.state_dim)
            if transition_matrix is not None
            else None
        )
        self.last_innovation: Optional[np.ndarray] = None
        self.last_innovation_cov: Optional[np.ndarray] = None

    def set_model(
        self,
        transition_matrix: Optional[np.ndarray] = None,
        process_noise: Optional[np.ndarray] = None,
        measurement_noise: Optional[np.ndarray] = None,
    ) -> None:
        if transition_matrix is not None:
            self.transition_matrix = _as_square_matrix("transition_matrix", transition_matrix, self.state_dim)
        if process_noise is not None:
            self.process_noise = _as_square_matrix("process_noise", process_noise, self.state_dim)
        if measurement_noise is not None:
            self.measurement_noise = _as_square_matrix("measurement_noise", measurement_noise, self.obs_dim)

    def reset(self, state: Optional[np.ndarray] = None, covariance: Optional[np.ndarray] = None) -> None:
        if state is not None:
            self.state = _as_vector("state", state, self.state_dim)
        if covariance is not None:
            self.covariance = _as_square_matrix("covariance", covariance, self.state_dim)
        self.last_innovation = None
        self.last_innovation_cov = None

    def predict(
        self,
        state_transition: Optional[Callable[..., np.ndarray]] = None,
        transition_jacobian: Optional[JacobianFn] = None,
        control_input: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if self.process_noise is None:
            raise ValueError("process_noise is required before calling predict().")

        if state_transition is None:
            if self.transition_matrix is None:
                raise ValueError("transition_matrix or state_transition must be provided.")
            transition_matrix = self.transition_matrix
            predicted_state = transition_matrix @ self.state
            if control_input is not None:
                predicted_state = predicted_state + control_input
        else:
            predicted_state = _call_state_transition(state_transition, self.state, control_input)
            if transition_jacobian is not None:
                transition_matrix = _as_square_matrix(
                    "transition_jacobian",
                    transition_jacobian(self.state),
                    self.state_dim,
                )
            elif self.transition_matrix is not None:
                transition_matrix = self.transition_matrix
            else:
                raise ValueError("transition_jacobian or transition_matrix must be provided.")

        self.state = _as_vector("state", predicted_state, self.state_dim)
        self.covariance = transition_matrix @ self.covariance @ transition_matrix.T + self.process_noise
        return self.state, self.covariance

    def update(
        self,
        measurement: np.ndarray,
        measurement_fn: MeasurementFn,
        measurement_jacobian: JacobianFn,
        measurement_noise: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        noise_matrix = measurement_noise
        if noise_matrix is None:
            noise_matrix = self.measurement_noise
        if noise_matrix is None:
            raise ValueError("measurement_noise is required before calling update().")

        measurement_vector = _as_vector("measurement", measurement, self.obs_dim)
        predicted_measurement = _as_vector("predicted_measurement", measurement_fn(self.state), self.obs_dim)
        measurement_jacobian_matrix = _as_matrix(
            "measurement_jacobian",
            measurement_jacobian(self.state),
            self.obs_dim,
            self.state_dim,
        )

        innovation_cov = (
            measurement_jacobian_matrix @ self.covariance @ measurement_jacobian_matrix.T + noise_matrix
        )
        kalman_gain = self.covariance @ measurement_jacobian_matrix.T @ np.linalg.inv(innovation_cov)
        innovation = measurement_vector - predicted_measurement
        self.state = self.state + kalman_gain @ innovation

        identity = np.eye(self.state_dim)
        correction = identity - kalman_gain @ measurement_jacobian_matrix
        self.covariance = correction @ self.covariance @ correction.T + kalman_gain @ noise_matrix @ kalman_gain.T

        self.last_innovation = innovation
        self.last_innovation_cov = innovation_cov
        return self.state, self.covariance, innovation, innovation_cov


__all__ = ["EKF"]
