"""Value-at-Risk and Conditional VaR (Expected Shortfall).

Conventions: ``exposures`` are signed currency amounts per asset; results are positive
loss amounts in currency. ``confidence`` is e.g. 0.95; ``horizon`` is in trading days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def _validate(confidence: float, horizon: int) -> None:
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be in (0.5, 1).")
    if horizon < 1:
        raise ValueError("horizon must be >= 1 day.")


def parametric_var_cvar(
    exposures: pd.Series[float],
    cov: pd.DataFrame,
    confidence: float,
    horizon: int = 1,
) -> tuple[float, float]:
    """Variance-covariance VaR/CVaR assuming zero-mean normal returns.

    sigma_p = sqrt(w' C w) * sqrt(h);  VaR = z_c * sigma_p;
    CVaR = sigma_p * phi(z_c) / (1 - c).
    """
    _validate(confidence, horizon)
    symbols = list(exposures.index)
    w = exposures.to_numpy(dtype=float)
    c = cov.loc[symbols, symbols].to_numpy(dtype=float)
    sigma = float(np.sqrt(max(w @ c @ w, 0.0) * horizon))
    z = float(norm.ppf(confidence))
    return z * sigma, sigma * float(norm.pdf(z)) / (1.0 - confidence)


def horizon_returns(returns: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Overlapping ``horizon``-day compounded returns from daily simple returns."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1 day.")
    if horizon == 1:
        return returns
    log_r = pd.DataFrame(np.log1p(returns.to_numpy()), index=returns.index, columns=returns.columns)
    rolled = log_r.rolling(horizon).sum().dropna()
    return pd.DataFrame(np.expm1(rolled.to_numpy()), index=rolled.index, columns=rolled.columns)


def historical_var_cvar(
    exposures: pd.Series[float],
    returns: pd.DataFrame,
    confidence: float,
    horizon: int = 1,
) -> tuple[float, float]:
    """Historical-simulation VaR/CVaR from daily simple ``returns``.

    P&L scenarios apply each (overlapping, compounded) historical return vector to the
    current exposures. VaR is the (1 - c) quantile loss; CVaR the mean loss beyond it.
    """
    _validate(confidence, horizon)
    scenarios = horizon_returns(returns[list(exposures.index)], horizon)
    if scenarios.empty:
        raise ValueError("Not enough history for the requested horizon.")
    pnl = scenarios.to_numpy(dtype=float) @ exposures.to_numpy(dtype=float)
    cutoff = float(np.quantile(pnl, 1.0 - confidence))
    tail = pnl[pnl <= cutoff]
    return -cutoff, float(-tail.mean())
