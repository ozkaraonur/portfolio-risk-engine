"""Risk attribution: how much of the portfolio VaR / CVaR each position contributes.

Both VaR and CVaR are homogeneous of degree one in the exposures, so by Euler's theorem the
component contributions add up exactly to the portfolio figure. Diversifying positions have a
negative component.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from portfolio_risk.risk.covariance import covariance_matrix
from portfolio_risk.risk.estimators import Method
from portfolio_risk.risk.var import _validate, horizon_returns


@dataclass(frozen=True)
class RiskContribution:
    method: Method
    confidence: float
    horizon: int
    exposures: pd.Series[float]
    component_var: pd.Series[float]
    component_cvar: pd.Series[float]

    @property
    def var(self) -> float:
        return float(self.component_var.sum())

    @property
    def cvar(self) -> float:
        return float(self.component_cvar.sum())

    @property
    def var_share(self) -> pd.Series[float]:
        """Each position's fraction of the portfolio VaR (sums to 1)."""
        total = self.var
        return self.component_var / total if total else self.component_var * 0.0

    @property
    def marginal_var(self) -> pd.Series[float]:
        """VaR added per currency unit of extra exposure (``component / exposure``)."""
        safe = self.exposures.where(self.exposures != 0.0)
        return (self.component_var / safe).fillna(0.0)


def _parametric(
    exposures: pd.Series[float], window: pd.DataFrame, confidence: float, horizon: int
) -> tuple[pd.Series[float], pd.Series[float]]:
    w = exposures.to_numpy(dtype=float)
    cov = covariance_matrix(window).to_numpy(dtype=float)
    sigma = float(np.sqrt(max(w @ cov @ w, 0.0)))
    if sigma == 0.0:
        zeros = exposures * 0.0
        return zeros, zeros.copy()
    share = w * (cov @ w) / sigma  # sums to the one-day sigma
    scale = np.sqrt(horizon)
    z = float(norm.ppf(confidence))
    var = pd.Series(z * scale * share, index=exposures.index)
    cvar = pd.Series(scale * float(norm.pdf(z)) / (1.0 - confidence) * share, index=exposures.index)
    return var, cvar


def _historical(
    exposures: pd.Series[float], window: pd.DataFrame, confidence: float, horizon: int
) -> tuple[pd.Series[float], pd.Series[float]]:
    scenarios = horizon_returns(window, horizon)
    if scenarios.empty:
        raise ValueError("Not enough history for the requested horizon.")
    contributions = scenarios.to_numpy(dtype=float) * exposures.to_numpy(dtype=float)
    pnl = contributions.sum(axis=1)
    cutoff = float(np.quantile(pnl, 1.0 - confidence))
    tail = pnl <= cutoff
    cvar_parts = -contributions[tail].mean(axis=0)
    cvar = pd.Series(cvar_parts, index=exposures.index)
    total = float(cvar.sum())
    # The VaR quantile is a single scenario, too noisy to split, so VaR is allocated in
    # proportion to the expected-shortfall components.
    var = cvar * (-cutoff / total) if total else cvar * 0.0
    return var, cvar


def risk_contributions(
    exposures: pd.Series[float],
    returns: pd.DataFrame,
    *,
    method: Method = Method.PARAMETRIC,
    confidence: float = 0.99,
    horizon: int = 1,
) -> RiskContribution:
    """Component VaR / CVaR of each position (parametric, or historical via the loss tail)."""
    _validate(confidence, horizon)
    if method not in (Method.PARAMETRIC, Method.HISTORICAL):
        raise ValueError("Attribution supports the parametric and historical methods.")
    window = returns[list(exposures.index)]
    compute = _parametric if method is Method.PARAMETRIC else _historical
    var, cvar = compute(exposures, window, confidence, horizon)
    return RiskContribution(method, confidence, horizon, exposures, var, cvar)
