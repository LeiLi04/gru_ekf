from .EKF_fp import EKF
from .UKF_fp import UKF
from .PF_fp import ParticleFilter, PF
from .ekf_online_beta import OnlineBetaEKF, OnlineBetaConfig

__all__ = [
    "GruAugmentedEkfLitModule",
    "EKF",
    "UKF",
    "ParticleFilter",
    "PF",
    "OnlineBetaEKF",
    "OnlineBetaConfig",
]


def __getattr__(name: str):
    # Lazy import to avoid circular imports between `src.models` and `src.train`.
    if name == "GruAugmentedEkfLitModule":
        from .lit_module import GruAugmentedEkfLitModule as _Lit

        return _Lit
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
