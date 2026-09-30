"""One entry point for every VaR/CVaR estimation method."""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import pandas as pd
from scipy.stats import norm

from portfolio_risk.risk.covariance import CovMethod, estimate_covariance, ewma_weights
from portfolio_risk.risk.linalg import FloatArray
from portfolio_risk.risk.tail_models import (
    cornish_fisher_var_cvar,
    fhs_var_cvar,
    student_t_var_cvar,
)
from portfolio_risk.risk.var import _validate, historical_var_cvar, parametric_var_cvar


class Method(StrEnum):
    PARAMETRIC = "parametric"
    HISTORICAL = "historical"
    EWMA = "ewma"
    STUDENT_T = "student-t"
    CORNISH_FISHER = "cornish-fisher"
    FHS = "fhs"


# Methods shown in the standard reports; the rest are compared by ``pre backtest``.
CORE_METHODS = (Method.PARAMETRIC, Method.HISTORICAL)


def estimate_var_cvar(
    method: Method,
    exposures: pd.Series[float],
    returns: pd.DataFrame,
    confidence: float,
    horizon: int = 1,
) -> tuple[float, float]:
    """VaR/CVaR of ``exposures`` (signed currency per asset) from daily simple ``returns``."""
    window = returns[list(exposures.index)]
    if method is Method.PARAMETRIC:
        return parametric_var_cvar(
            exposures, estimate_covariance(window, CovMethod.SAMPLE), confidence, horizon
        )
    if method is Method.EWMA:
        return parametric_var_cvar(
            exposures, estimate_covariance(window, CovMethod.EWMA), confidence, horizon
        )
    if method is Method.HISTORICAL:
        return historical_var_cvar(exposures, window, confidence, horizon)
    pnl = np.asarray(
        window.to_numpy(dtype=np.float64) @ exposures.to_numpy(dtype=np.float64), dtype=np.float64
    )
    if method is Method.STUDENT_T:
        return student_t_var_cvar(pnl, confidence, horizon)
    if method is Method.CORNISH_FISHER:
        return cornish_fisher_var_cvar(pnl, confidence, horizon)
    return fhs_var_cvar(pnl, confidence, horizon)


def one_day_var_cvar(method: Method, pnl: FloatArray, confidence: float) -> tuple[float, float]:
    """One-day VaR/CVaR straight from a P&L window (positive = gain).

    Equivalent to ``estimate_var_cvar`` with ``horizon=1`` on the returns that produced ``pnl``,
    but much cheaper, which is what the rolling backtest needs.
    """
    _validate(confidence, 1)
    if method is Method.PARAMETRIC or method is Method.EWMA:
        if method is Method.PARAMETRIC:
            sigma = float(np.std(pnl, ddof=1))
        else:
            sigma = float(np.sqrt(np.sum(ewma_weights(len(pnl)) * pnl**2)))
        z = float(norm.ppf(confidence))
        return z * sigma, sigma * float(norm.pdf(z)) / (1.0 - confidence)
    if method is Method.HISTORICAL:
        cutoff = float(np.quantile(pnl, 1.0 - confidence))
        return -cutoff, float(-pnl[pnl <= cutoff].mean())
    if method is Method.STUDENT_T:
        return student_t_var_cvar(pnl, confidence)
    if method is Method.CORNISH_FISHER:
        return cornish_fisher_var_cvar(pnl, confidence)
    return fhs_var_cvar(pnl, confidence)
