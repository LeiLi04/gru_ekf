"""Innovation negative log-likelihood (NLL).

This loss is computed in the innovation domain using the whitened innovation
and the Cholesky log-determinant of the innovation covariance.
"""

from __future__ import annotations

import torch

from src.utils.linear_algebra import chol_logdet, chol_solve, safe_cholesky
from src.utils.masking import masked_mean


def innovation_nll(
    innovation: torch.Tensor,
    S: torch.Tensor,
    logdet_S: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute mean innovation NLL for a sequence.

    Args:
        innovation: (B, T, m) innovations.
        S: (B, T, m, m) innovation covariances.
        logdet_S: optional (B, T) precomputed log det of S for efficiency.
        mask: optional (B, T) boolean mask where True means invalid/padded.
    """
    B, T, m = innovation.shape
    innov_vec = innovation.reshape(B * T, m, 1)
    S_mat = S.reshape(B * T, m, m)
    L, _ = safe_cholesky(S_mat)
    sol = chol_solve(L, innov_vec).squeeze(-1)
    quad = (sol * innovation.reshape(B * T, m)).sum(dim=-1).reshape(B, T)
    if logdet_S is None:
        logdet = chol_logdet(L).reshape(B, T)
    else:
        logdet = logdet_S
    nll = quad + logdet
    if mask is None:
        return nll.mean()
    return masked_mean(nll, mask=mask).mean()


__all__ = ["innovation_nll"]
