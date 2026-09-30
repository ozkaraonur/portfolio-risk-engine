"""Covariance and correlation estimation from return series."""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import numpy.typing as npt
import pandas as pd

TRADING_DAYS = 252
EWMA_LAMBDA = 0.94  # RiskMetrics decay for daily data


class CovMethod(StrEnum):
    SAMPLE = "sample"
    EWMA = "ewma"
    SHRINKAGE = "shrinkage"


def _check(returns: pd.DataFrame) -> None:
    if len(returns) < 2:
        raise ValueError("At least two return observations are required.")
    if returns.isna().any().any():
        raise ValueError("Returns contain NaN values.")


def covariance_matrix(returns: pd.DataFrame, *, annualize: bool = False) -> pd.DataFrame:
    """Sample covariance (ddof=1) of periodic returns; optionally annualised (x252)."""
    _check(returns)
    cov = returns.cov()
    return cov * TRADING_DAYS if annualize else cov


def ewma_weights(n: int, lam: float = EWMA_LAMBDA) -> npt.NDArray[np.float64]:
    """Normalised exponential weights for ``n`` observations, newest observation weighted most."""
    if not 0.0 < lam < 1.0:
        raise ValueError("lam must be in (0, 1).")
    raw = lam ** np.arange(n - 1, -1, -1, dtype=float)
    return np.asarray(raw / raw.sum(), dtype=np.float64)


def ewma_covariance(returns: pd.DataFrame, lam: float = EWMA_LAMBDA) -> pd.DataFrame:
    """RiskMetrics exponentially weighted covariance (zero-mean): recent days count more."""
    _check(returns)
    x = returns.to_numpy(dtype=float)
    cov = (x * ewma_weights(len(x), lam)[:, None]).T @ x
    return pd.DataFrame(cov, index=returns.columns, columns=returns.columns)


def shrunk_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """Ledoit-Wolf covariance shrunk towards a scaled identity (well-conditioned for small T)."""
    _check(returns)
    x = returns.to_numpy(dtype=float)
    x = x - x.mean(axis=0)
    n, p = x.shape
    sample = x.T @ x / n
    mu = float(np.trace(sample)) / p
    d2 = float(((sample - mu * np.eye(p)) ** 2).sum()) / p
    if d2 == 0.0:
        return returns.cov()
    b2_bar = float(((x**2).sum(axis=1) ** 2).sum() - n * (sample**2).sum()) / (n**2 * p)
    b2 = min(b2_bar, d2)
    shrunk = (b2 / d2) * mu * np.eye(p) + ((d2 - b2) / d2) * sample
    return pd.DataFrame(shrunk * n / (n - 1), index=returns.columns, columns=returns.columns)


def estimate_covariance(
    returns: pd.DataFrame, method: CovMethod = CovMethod.SAMPLE
) -> pd.DataFrame:
    """Covariance of periodic returns by the chosen estimator."""
    if method is CovMethod.EWMA:
        return ewma_covariance(returns)
    if method is CovMethod.SHRINKAGE:
        return shrunk_covariance(returns)
    return covariance_matrix(returns)


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix of periodic returns."""
    _check(returns)
    corr = returns.corr()
    # Constant series have undefined correlation; report them as uncorrelated (diag = 1).
    corr = corr.fillna(0.0)
    values = corr.to_numpy(copy=True)
    np.fill_diagonal(values, 1.0)
    return pd.DataFrame(values, index=corr.index, columns=corr.columns)
