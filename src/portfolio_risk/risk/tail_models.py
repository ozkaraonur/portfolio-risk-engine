"""Fat-tail VaR/CVaR models working on a series of portfolio P&L.

``pnl`` is the daily P&L of today's exposures over the estimation window (positive = gain).
All models return positive loss amounts ``(VaR, CVaR)`` and scale to a multi-day horizon with the
square-root-of-time rule (Cornish-Fisher scales skewness and kurtosis by the iid sum rules).
"""

from __future__ import annotations

import numpy as np
from scipy.signal import lfilter
from scipy.stats import kurtosis, norm, skew
from scipy.stats import t as student_t

from portfolio_risk.risk.covariance import EWMA_LAMBDA
from portfolio_risk.risk.linalg import FloatArray

MIN_DF = 4.1  # keeps the t distribution's kurtosis finite
MAX_DF = 100.0  # beyond this the t is indistinguishable from a normal
_CF_GRID = 200


def _validate(pnl: FloatArray, confidence: float, horizon: int) -> None:
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be in (0.5, 1).")
    if horizon < 1:
        raise ValueError("horizon must be >= 1 day.")
    if pnl.size < 4:
        raise ValueError("At least four P&L observations are required.")


def estimate_t_df(pnl: FloatArray) -> float:
    """Degrees of freedom of a Student-t matching the sample excess kurtosis (moment method).

    For a t distribution the excess kurtosis is ``6 / (df - 4)``, so ``df = 4 + 6 / k``,
    clamped to ``[MIN_DF, MAX_DF]``; samples with no excess kurtosis map to ``MAX_DF``.
    """
    excess = float(kurtosis(pnl, fisher=True, bias=False))
    if not np.isfinite(excess) or excess <= 6.0 / (MAX_DF - 4.0):
        return MAX_DF
    return float(np.clip(4.0 + 6.0 / excess, MIN_DF, MAX_DF))


def student_t_var_cvar(
    pnl: FloatArray, confidence: float, horizon: int = 1, df: float | None = None
) -> tuple[float, float]:
    """Zero-mean Student-t VaR/CVaR with the sample volatility and moment-matched ``df``.

    The t is rescaled to unit variance, so the volatility equals the parametric-normal one and
    only the tail shape differs.
    """
    _validate(pnl, confidence, horizon)
    nu = estimate_t_df(pnl) if df is None else df
    if nu <= 2.0:
        raise ValueError("df must be greater than 2.")
    sigma = float(np.std(pnl, ddof=1)) * np.sqrt(horizon)
    scale = sigma * np.sqrt((nu - 2.0) / nu)
    q = float(student_t.ppf(confidence, nu))
    var = scale * q
    cvar = scale * float(student_t.pdf(q, nu)) / (1.0 - confidence) * (nu + q**2) / (nu - 1.0)
    return max(var, 0.0), max(cvar, 0.0)


def _cornish_fisher_quantile(z: FloatArray | float, s: float, k: float) -> FloatArray:
    z = np.asarray(z, dtype=np.float64)
    return np.asarray(
        z
        + (z**2 - 1.0) * s / 6.0
        + (z**3 - 3.0 * z) * k / 24.0
        - (2.0 * z**3 - 5.0 * z) * s**2 / 36.0,
        dtype=np.float64,
    )


def cornish_fisher_var_cvar(
    pnl: FloatArray, confidence: float, horizon: int = 1
) -> tuple[float, float]:
    """Cornish-Fisher VaR/CVaR: the normal quantile corrected for sample skew and kurtosis.

    The expansion is only valid for moderate skew/kurtosis: with extreme sample moments the
    polynomial stops being monotone and can *lower* the quantile. The corrected quantile is
    therefore never taken below the normal one, and CVaR never falls below VaR. CVaR averages
    the corrected quantile over the tail.
    """
    _validate(pnl, confidence, horizon)
    losses = -pnl
    sigma = float(np.std(pnl, ddof=1)) * np.sqrt(horizon)
    s = float(skew(losses, bias=False)) / np.sqrt(horizon)
    k = float(kurtosis(losses, fisher=True, bias=False)) / horizon
    if not (np.isfinite(s) and np.isfinite(k)):
        s = k = 0.0
    z = float(norm.ppf(confidence))
    var = sigma * max(float(_cornish_fisher_quantile(z, s, k)), z)
    u = confidence + (1.0 - confidence) * (np.arange(_CF_GRID) + 0.5) / _CF_GRID
    z_tail = norm.ppf(u)
    tail = np.maximum(_cornish_fisher_quantile(z_tail, s, k), z_tail)
    return max(var, 0.0), max(sigma * float(tail.mean()), var, 0.0)


def ewma_volatility(series: FloatArray, lam: float = EWMA_LAMBDA) -> FloatArray:
    """One-step-ahead EWMA volatility: entry ``i`` is the forecast for ``series[i]``.

    The recursion starts at the sample variance of the first (up to) 30 observations.
    """
    if not 0.0 < lam < 1.0:
        raise ValueError("lam must be in (0, 1).")
    start = float(np.var(series[:30])) or 1e-12
    # f_0 = start, f_i = lam * f_{i-1} + (1 - lam) * x_{i-1}^2, evaluated as a linear filter.
    drive = np.concatenate([[start], (1.0 - lam) * series[:-1] ** 2])
    return np.asarray(np.sqrt(lfilter([1.0], [1.0, -lam], drive)), dtype=np.float64)


def _next_ewma_vol(series: FloatArray, lam: float) -> float:
    vol = ewma_volatility(series, lam)
    return float(np.sqrt(lam * vol[-1] ** 2 + (1.0 - lam) * float(series[-1]) ** 2))


def fhs_var_cvar(
    pnl: FloatArray, confidence: float, horizon: int = 1, lam: float = EWMA_LAMBDA
) -> tuple[float, float]:
    """Filtered historical simulation: replay standardised residuals at today's EWMA volatility.

    Each historical P&L is divided by the volatility forecast made the day before, then rescaled
    with the forecast for tomorrow, so the tail reflects the current volatility regime while
    keeping the empirical (fat-tailed) shape of the shocks.
    """
    _validate(pnl, confidence, horizon)
    vols = ewma_volatility(pnl, lam)
    if np.any(vols <= 0):
        raise ValueError("Degenerate volatility in the estimation window.")
    residuals = pnl / vols
    scenarios = residuals * _next_ewma_vol(pnl, lam) * np.sqrt(horizon)
    cutoff = float(np.quantile(scenarios, 1.0 - confidence))
    tail = scenarios[scenarios <= cutoff]
    return -cutoff, float(-tail.mean())
