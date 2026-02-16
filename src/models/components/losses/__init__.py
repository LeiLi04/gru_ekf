"""Loss functions used by GRU-augmented EKF components."""

from .innov_nll import innovation_nll

__all__ = ["innovation_nll"]

