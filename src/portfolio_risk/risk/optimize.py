"""Long-only portfolio optimisation over the risky assets (cash is left out).

Works on annualised inputs: covariance ``252 * Sigma_daily`` and arithmetic mean returns
``252 * mean_daily``. Expected returns estimated from a short history are very noisy, so the
return-based objectives (max-Sharpe, frontier) are indicative only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, minimize

from portfolio_risk.risk.covariance import TRADING_DAYS, covariance_matrix
from portfolio_risk.risk.linalg import FloatArray


class Objective(StrEnum):
    MIN_VARIANCE = "min-variance"
    RISK_PARITY = "risk-parity"
    MAX_SHARPE = "max-sharpe"


@dataclass(frozen=True)
class Allocation:
    objective: Objective
    weights: pd.Series[float]  # long-only, sums to 1 over the risky assets
    expected_return: float  # annualised
    volatility: float  # annualised


@dataclass(frozen=True)
class FrontierPoint:
    expected_return: float
    volatility: float
    weights: pd.Series[float]


def expected_returns(returns: pd.DataFrame) -> pd.Series[float]:
    """Annualised arithmetic mean of daily returns."""
    return returns.mean() * TRADING_DAYS


def _inputs(returns: pd.DataFrame) -> tuple[FloatArray, FloatArray]:
    cov = np.asarray(covariance_matrix(returns), dtype=np.float64) * TRADING_DAYS
    mu = np.asarray(expected_returns(returns), dtype=np.float64)
    return np.asarray(cov, dtype=np.float64), mu


def _check_cap(n: int, max_weight: float) -> None:
    if not 0.0 < max_weight <= 1.0:
        raise ValueError("max_weight must be in (0, 1].")
    if max_weight * n < 1.0 - 1e-12:
        raise ValueError(f"max_weight {max_weight:g} is infeasible for {n} assets.")


def _solve(
    fun: Callable[[FloatArray], float],
    n: int,
    max_weight: float,
    extra: list[LinearConstraint] | None = None,
    start: FloatArray | None = None,
) -> FloatArray:
    constraints = [LinearConstraint(np.ones((1, n)), 1.0, 1.0), *(extra or [])]
    x0: FloatArray = (
        np.full(n, 1.0 / n, dtype=np.float64) if start is None else np.asarray(start, np.float64)
    )
    result = minimize(
        fun,
        x0,
        method="SLSQP",
        bounds=[(0.0, max_weight)] * n,
        constraints=constraints,
        tol=1e-12,
        options={"maxiter": 500},
    )
    if not result.success:
        raise ValueError(f"Optimisation did not converge: {result.message}")
    w = np.clip(np.asarray(result.x, dtype=np.float64), 0.0, max_weight)
    return np.asarray(w / w.sum(), dtype=np.float64)


def min_variance_weights(cov: FloatArray, max_weight: float = 1.0) -> FloatArray:
    _check_cap(cov.shape[0], max_weight)
    return _solve(lambda w: float(w @ cov @ w), cov.shape[0], max_weight)


def risk_parity_weights(cov: FloatArray) -> FloatArray:
    """Equal risk contributions (long-only), from the convex problem min 0.5 wᵀΣw - (1/n) Σ ln wᵢ.

    Its solution, rescaled to sum to one, equalises ``wᵢ (Σw)ᵢ``. Position caps do not apply.
    """
    n = cov.shape[0]

    def objective(w: FloatArray) -> float:
        return float(0.5 * w @ cov @ w - np.log(w).sum() / n)

    def gradient(w: FloatArray) -> FloatArray:
        return np.asarray(cov @ w - 1.0 / (n * w), dtype=np.float64)

    result = minimize(
        objective,
        np.full(n, 1.0 / n),
        jac=gradient,
        method="L-BFGS-B",
        bounds=[(1e-9, None)] * n,
        options={"maxiter": 1000, "ftol": 1e-15, "gtol": 1e-10},
    )
    if not result.success:
        raise ValueError(f"Optimisation did not converge: {result.message}")
    w = np.asarray(result.x, dtype=np.float64)
    return np.asarray(w / w.sum(), dtype=np.float64)


def max_sharpe_weights(
    cov: FloatArray, mu: FloatArray, risk_free: float = 0.0, max_weight: float = 1.0
) -> FloatArray:
    _check_cap(cov.shape[0], max_weight)
    if float((mu - risk_free).max()) <= 0.0:
        raise ValueError("No asset has a positive excess return; max-Sharpe is undefined.")

    def neg_sharpe(w: FloatArray) -> float:
        return float(-(w @ mu - risk_free) / np.sqrt(max(float(w @ cov @ w), 1e-16)))

    best: FloatArray | None = None
    best_value = np.inf
    n = cov.shape[0]
    starts = [np.full(n, 1.0 / n), *(np.eye(n)[i] * 0.5 + 0.5 / n for i in range(n))]
    for start in starts:
        try:
            w = _solve(neg_sharpe, n, max_weight, start=start / start.sum())
        except ValueError:
            continue
        value = neg_sharpe(w)
        if value < best_value:
            best, best_value = w, value
    if best is None:
        raise ValueError("Optimisation did not converge.")
    return best


def optimize_portfolio(
    objective: Objective,
    returns: pd.DataFrame,
    *,
    max_weight: float = 1.0,
    risk_free: float = 0.0,
) -> Allocation:
    """Optimal long-only weights over the columns of ``returns`` for the given objective."""
    if returns.shape[1] < 2:
        raise ValueError("Optimisation needs at least two assets.")
    cov, mu = _inputs(returns)
    if objective is Objective.MIN_VARIANCE:
        w = min_variance_weights(cov, max_weight)
    elif objective is Objective.RISK_PARITY:
        w = risk_parity_weights(cov)
    else:
        w = max_sharpe_weights(cov, mu, risk_free, max_weight)
    return Allocation(
        objective=objective,
        weights=pd.Series(w, index=returns.columns),
        expected_return=float(w @ mu),
        volatility=float(np.sqrt(w @ cov @ w)),
    )


def efficient_frontier(
    returns: pd.DataFrame, *, points: int = 20, max_weight: float = 1.0
) -> list[FrontierPoint]:
    """Minimum-volatility portfolios from the global minimum-variance point to the best asset."""
    if returns.shape[1] < 2:
        raise ValueError("The frontier needs at least two assets.")
    if points < 2:
        raise ValueError("points must be at least 2.")
    cov, mu = _inputs(returns)
    n = cov.shape[0]
    _check_cap(n, max_weight)
    start = min_variance_weights(cov, max_weight)
    lo = float(start @ mu)
    # Highest return reachable under the cap: fill the best assets up to the cap.
    hi = 0.0
    remaining = 1.0
    for value in np.sort(mu)[::-1]:
        take = min(max_weight, remaining)
        hi += take * float(value)
        remaining -= take
        if remaining <= 1e-12:
            break
    frontier: list[FrontierPoint] = []
    for target in np.linspace(lo, max(hi, lo), points):
        try:
            w = _solve(
                lambda x: float(x @ cov @ x),
                n,
                max_weight,
                extra=[LinearConstraint(mu[None, :], target, target)],
                start=start,
            )
        except ValueError:
            continue
        frontier.append(
            FrontierPoint(
                expected_return=float(w @ mu),
                volatility=float(np.sqrt(w @ cov @ w)),
                weights=pd.Series(w, index=returns.columns),
            )
        )
    return frontier


def portfolio_stats(weights: pd.Series[float], returns: pd.DataFrame) -> tuple[float, float]:
    """Annualised ``(expected return, volatility)`` of ``weights`` (columns of ``returns``)."""
    cov, mu = _inputs(returns[list(weights.index)])
    w = weights.to_numpy(dtype=np.float64)
    return float(w @ mu), float(np.sqrt(max(float(w @ cov @ w), 0.0)))


def rebalance_trades(values: pd.Series[float], target: pd.Series[float]) -> pd.Series[float]:
    """Currency to buy (+) or sell (-) per asset so the risky assets match ``target`` weights.

    The total invested amount is kept, so the trades sum to zero; cash is untouched.
    """
    aligned = target.reindex(values.index)
    if aligned.isna().any():
        raise ValueError("Target weights must cover every held asset.")
    return aligned * float(values.sum()) - values
