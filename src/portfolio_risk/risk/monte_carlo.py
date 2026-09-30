"""Monte Carlo simulation of portfolio value paths (correlated multivariate GBM).

Positions are held constant (buy-and-hold); cash earns nothing. Daily log-returns are
``(mu - sigma^2 / 2) + L z`` with ``L @ L.T = Sigma`` (Cholesky) and ``z ~ N(0, I)``, so the
simple-return mean equals ``mu`` and the covariance equals ``Sigma``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from portfolio_risk.models import Portfolio
from portfolio_risk.risk.covariance import covariance_matrix
from portfolio_risk.risk.linalg import FloatArray, cholesky_factor
from portfolio_risk.risk.report import MIN_OBS


def simulate_returns(
    cov: FloatArray,
    mean: FloatArray,
    n_simulations: int,
    days: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Correlated daily log-returns, shape ``(n_simulations, days, n_assets)``."""
    if n_simulations < 1 or days < 1:
        raise ValueError("n_simulations and days must be >= 1.")
    if mean.shape != (cov.shape[0],):
        raise ValueError("mean must have one entry per asset.")
    chol = cholesky_factor(cov)
    drift = mean - 0.5 * np.diag(cov)
    shocks = rng.standard_normal((n_simulations, days, cov.shape[0]))
    return np.asarray(shocks @ chol.T + drift, dtype=np.float64)


def simulate_paths(
    exposures: FloatArray,
    cash: float,
    cov: FloatArray,
    mean: FloatArray,
    n_simulations: int,
    days: int,
    rng: np.random.Generator,
) -> FloatArray:
    """Portfolio value paths, shape ``(n_simulations, days + 1)``; column 0 is today."""
    log_r = simulate_returns(cov, mean, n_simulations, days, rng)
    asset_factors = np.exp(np.cumsum(log_r, axis=1))  # (N, T, k) growth vs today
    values = asset_factors @ exposures + cash
    initial = float(exposures.sum() + cash)
    return np.concatenate([np.full((n_simulations, 1), initial), values], axis=1)


def max_drawdowns(paths: FloatArray) -> FloatArray:
    """Maximum peak-to-trough decline of each path, as a fraction in [0, 1]."""
    running_peak = np.maximum.accumulate(paths, axis=1)
    drawdown = (running_peak - paths) / running_peak
    return np.asarray(drawdown.max(axis=1), dtype=np.float64)


def _var_cvar(losses: FloatArray, confidence: float) -> tuple[float, float]:
    cutoff = float(np.quantile(losses, confidence))
    tail = losses[losses >= cutoff]
    return max(cutoff, 0.0), max(float(tail.mean()), 0.0)


@dataclass(frozen=True)
class MonteCarloReport:
    n_simulations: int
    days: int
    initial_value: float
    final_values: npt.NDArray[np.float64]
    max_drawdowns: npt.NDArray[np.float64]
    var: dict[float, float]  # loss over the full horizon, by confidence
    cvar: dict[float, float]
    loss_threshold: float  # e.g. 0.3 = a 30% loss of initial value
    prob_ruin: float  # P(value touches the threshold at any time)
    prob_loss_at_end: float  # P(final value is at or below the threshold)

    @property
    def mean_final(self) -> float:
        return float(self.final_values.mean())

    @property
    def median_final(self) -> float:
        return float(np.median(self.final_values))

    def final_percentile(self, pct: float) -> float:
        """Percentile (0-100) of terminal portfolio value."""
        return float(np.percentile(self.final_values, pct))

    def drawdown_percentile(self, pct: float) -> float:
        return float(np.percentile(self.max_drawdowns, pct))

    def prob_drawdown_exceeds(self, level: float) -> float:
        """P(max drawdown >= ``level``), with ``level`` a fraction such as 0.2."""
        return float((self.max_drawdowns >= level).mean())


def summarize_paths(
    paths: FloatArray,
    *,
    confidences: tuple[float, ...] = (0.95, 0.99),
    loss_threshold: float = 0.3,
) -> MonteCarloReport:
    if not 0.0 < loss_threshold <= 1.0:
        raise ValueError("loss_threshold must be in (0, 1].")
    if any(not 0.5 < c < 1.0 for c in confidences):
        raise ValueError("confidence levels must be in (0.5, 1).")
    initial = float(paths[0, 0])
    if initial <= 0:
        raise ValueError("Portfolio has no value to simulate.")
    final = paths[:, -1]
    losses = initial - final
    stats = {c: _var_cvar(losses, c) for c in confidences}
    barrier = initial * (1.0 - loss_threshold)
    return MonteCarloReport(
        n_simulations=paths.shape[0],
        days=paths.shape[1] - 1,
        initial_value=initial,
        final_values=final,
        max_drawdowns=max_drawdowns(paths),
        var={c: v for c, (v, _) in stats.items()},
        cvar={c: t for c, (_, t) in stats.items()},
        loss_threshold=loss_threshold,
        prob_ruin=float((paths.min(axis=1) <= barrier).mean()),
        prob_loss_at_end=float((final <= barrier).mean()),
    )


def run_monte_carlo(
    portfolio: Portfolio,
    prices: pd.DataFrame,
    *,
    seed: int,
    n_simulations: int = 1000,
    days: int = 252,
    confidences: tuple[float, ...] = (0.95, 0.99),
    loss_threshold: float = 0.3,
    use_drift: bool = False,
) -> MonteCarloReport:
    """Estimate mu/Sigma from ``prices`` and simulate the portfolio forward ``days`` days.

    Drift is zero by default (risk-focused, consistent with parametric VaR); set
    ``use_drift`` to use each asset's historical mean daily return.
    """
    returns = prices[portfolio.symbols].pct_change().dropna()
    if len(returns) < MIN_OBS:
        raise ValueError(f"Need at least {MIN_OBS} return observations, got {len(returns)}.")
    latest = {str(k): float(v) for k, v in prices.iloc[-1].items()}
    values = portfolio.market_values(latest)
    exposures = np.array([values[s] for s in returns.columns], dtype=np.float64)
    cov = covariance_matrix(returns).to_numpy(dtype=np.float64)
    mean = returns.mean().to_numpy(dtype=np.float64) if use_drift else np.zeros(len(exposures))
    paths = simulate_paths(
        exposures, portfolio.total_cash, cov, mean, n_simulations, days, np.random.default_rng(seed)
    )
    return summarize_paths(paths, confidences=confidences, loss_threshold=loss_threshold)
