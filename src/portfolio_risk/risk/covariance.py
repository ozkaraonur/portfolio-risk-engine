"""Covariance and correlation estimation from return series."""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


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


def correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Pearson correlation matrix of periodic returns."""
    _check(returns)
    corr = returns.corr()
    # Constant series have undefined correlation; report them as uncorrelated (diag = 1).
    corr = corr.fillna(0.0)
    values = corr.to_numpy(copy=True)
    np.fill_diagonal(values, 1.0)
    return pd.DataFrame(values, index=corr.index, columns=corr.columns)
