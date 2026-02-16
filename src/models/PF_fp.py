"""Generic Particle Filter (PF) implementation."""

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


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.ones_like(weights) / weights.size
    return weights / total


def _effective_sample_size(weights: np.ndarray) -> float:
    return 1.0 / float(np.sum(np.square(weights)))


def _weighted_covariance(samples: np.ndarray, mean: np.ndarray, weights: np.ndarray) -> np.ndarray:
    centered = samples - mean
    return centered.T @ (centered * weights[:, None])


class ParticleFilter:
    """Bootstrap particle filter with Gaussian process and measurement noise."""

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        num_particles: int = 500,
        state: Optional[np.ndarray] = None,
        covariance: Optional[np.ndarray] = None,
        process_noise: Optional[np.ndarray] = None,
        measurement_noise: Optional[np.ndarray] = None,
        resample_threshold: float = 0.5,
        rng: Optional[np.random.Generator] = None,
    ) -> None:
        self.state_dim = int(state_dim)
        self.obs_dim = int(obs_dim)
        self.num_particles = int(num_particles)
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
        self.resample_threshold = float(resample_threshold)
        self.rng = rng if rng is not None else np.random.default_rng()

        self.particles: Optional[np.ndarray] = None
        self.weights: Optional[np.ndarray] = None
        self.last_innovation: Optional[np.ndarray] = None
        self.last_innovation_cov: Optional[np.ndarray] = None

        if state is not None:
            self.initialize(self.state, self.covariance)

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
        self.initialize(self.state, self.covariance)
        self.last_innovation = None
        self.last_innovation_cov = None

    def initialize(self, state: np.ndarray, covariance: np.ndarray) -> None:
        state_vector = _as_vector("state", state, self.state_dim)
        cov_matrix = _as_square_matrix("covariance", covariance, self.state_dim)
        self.particles = self.rng.multivariate_normal(state_vector, cov_matrix, size=self.num_particles)
        self.weights = np.ones(self.num_particles, dtype=float) / self.num_particles
        self.state = np.average(self.particles, weights=self.weights, axis=0)
        self.covariance = cov_matrix

    def _systematic_resample(self, weights: np.ndarray) -> np.ndarray:
        positions = (np.arange(self.num_particles) + self.rng.random()) / self.num_particles
        cumulative_sum = np.cumsum(weights)
        indices = np.searchsorted(cumulative_sum, positions)
        return indices

    def predict(self, state_transition: StateTransitionFn) -> np.ndarray:
        if self.particles is None or self.weights is None:
            raise ValueError("Particles are not initialized. Call initialize() first.")
        if self.process_noise is None:
            raise ValueError("process_noise is required before calling predict().")

        predicted_particles = _apply_fn(state_transition, self.particles)
        if predicted_particles.ndim == 1:
            predicted_particles = predicted_particles[:, None]
        noise = self.rng.multivariate_normal(
            np.zeros(self.state_dim),
            self.process_noise,
            size=self.num_particles,
        )
        self.particles = predicted_particles + noise
        self.state = np.average(self.particles, weights=self.weights, axis=0)
        return self.state

    def update(
        self,
        measurement: np.ndarray,
        measurement_fn: MeasurementFn,
        measurement_noise: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if self.particles is None or self.weights is None:
            raise ValueError("Particles are not initialized. Call initialize() first.")

        noise_matrix = measurement_noise
        if noise_matrix is None:
            noise_matrix = self.measurement_noise
        if noise_matrix is None:
            raise ValueError("measurement_noise is required before calling update().")

        measurement_vector = _as_vector("measurement", measurement, self.obs_dim)

        predicted_measurements = _apply_fn(measurement_fn, self.particles)
        if predicted_measurements.ndim == 1:
            predicted_measurements = predicted_measurements[:, None]
        innovation_matrix = measurement_vector - predicted_measurements
        inv_noise = np.linalg.inv(noise_matrix)
        exponent_terms = -0.5 * np.einsum("pi,ij,pj->p", innovation_matrix, inv_noise, innovation_matrix)
        likelihood = np.exp(exponent_terms)
        updated_weights = _normalize_weights(self.weights * likelihood)
        self.weights = updated_weights

        measurement_mean = np.average(predicted_measurements, weights=self.weights, axis=0)
        innovation_mean = measurement_vector - measurement_mean
        innovation_cov = noise_matrix + _weighted_covariance(predicted_measurements, measurement_mean, self.weights)

        self.last_innovation = innovation_mean
        self.last_innovation_cov = innovation_cov
        self.state = np.average(self.particles, weights=self.weights, axis=0)

        if _effective_sample_size(self.weights) < self.resample_threshold * self.num_particles:
            indices = self._systematic_resample(self.weights)
            self.particles = self.particles[indices]
            self.weights = np.ones(self.num_particles, dtype=float) / self.num_particles
            self.state = np.average(self.particles, weights=self.weights, axis=0)

        return self.state, self.weights, innovation_mean, innovation_cov


PF = ParticleFilter

__all__ = ["ParticleFilter", "PF"]
