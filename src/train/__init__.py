"""Training utilities."""

__all__ = [
    "covariance_matching_warm_start",
    "set_psd_parameter_from_matrix",
]


def __getattr__(name: str):
    # Lazy import to avoid importing torch-heavy modules at package import time.
    if name in {"covariance_matching_warm_start", "set_psd_parameter_from_matrix"}:
        from .warmstart_q0 import covariance_matching_warm_start, set_psd_parameter_from_matrix

        return {
            "covariance_matching_warm_start": covariance_matching_warm_start,
            "set_psd_parameter_from_matrix": set_psd_parameter_from_matrix,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
