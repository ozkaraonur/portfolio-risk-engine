"""Rolling one-day VaR backtest against realised portfolio P&L.

Positions are held at today's exposures (a "hypothetical" backtest): each day's P&L is that day's
asset returns applied to the current exposures, so the test isolates the risk model from trading.
For every test day the VaR is estimated only from the ``window`` returns before it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from portfolio_risk.models import Portfolio
from portfolio_risk.risk.covariance import covariance_matrix
from portfolio_risk.risk.coverage import (
    BoolArray,
    LikelihoodRatioTest,
    Zone,
    basel_zone,
    christoffersen_independence,
    conditional_coverage,
    kupiec_pof,
)
from portfolio_risk.risk.report import MIN_OBS, Method
from portfolio_risk.risk.var import historical_var_cvar, parametric_var_cvar

DEFAULT_WINDOW = 250


@dataclass(frozen=True)
class BacktestResult:
    method: Method
    confidence: float
    window: int
    dates: pd.DatetimeIndex
    pnl: npt.NDArray[np.float64]  # realised daily P&L on today's exposures
    var: npt.NDArray[np.float64]  # forecast made the day before, as a positive loss amount

    @property
    def violations(self) -> BoolArray:
        """True on days where the loss exceeded the forecast VaR."""
        return np.asarray(-self.pnl > self.var, dtype=np.bool_)

    @property
    def n_obs(self) -> int:
        return int(self.pnl.size)

    @property
    def n_violations(self) -> int:
        return int(self.violations.sum())

    @property
    def expected_violations(self) -> float:
        return self.n_obs * (1.0 - self.confidence)

    @property
    def violation_rate(self) -> float:
        return self.n_violations / self.n_obs

    @property
    def kupiec(self) -> LikelihoodRatioTest:
        return kupiec_pof(self.violations, self.confidence)

    @property
    def independence(self) -> LikelihoodRatioTest:
        return christoffersen_independence(self.violations)

    @property
    def conditional_coverage(self) -> LikelihoodRatioTest:
        return conditional_coverage(self.violations, self.confidence)

    @property
    def zone(self) -> Zone:
        return basel_zone(self.n_violations, self.n_obs, self.confidence)


def rolling_var(
    exposures: pd.Series[float],
    returns: pd.DataFrame,
    *,
    method: Method,
    confidence: float,
    window: int,
) -> npt.NDArray[np.float64]:
    """One-day VaR for each day after the first ``window``, using only the preceding window."""
    forecasts = np.empty(len(returns) - window, dtype=np.float64)
    for i in range(forecasts.size):
        history = returns.iloc[i : i + window]
        if method is Method.PARAMETRIC:
            forecasts[i] = parametric_var_cvar(exposures, covariance_matrix(history), confidence)[0]
        else:
            forecasts[i] = historical_var_cvar(exposures, history, confidence)[0]
    return forecasts


def run_backtest(
    portfolio: Portfolio,
    prices: pd.DataFrame,
    *,
    method: Method,
    confidence: float = 0.99,
    window: int = DEFAULT_WINDOW,
) -> BacktestResult:
    """Backtest ``method``'s one-day VaR on the price history (cash carries no risk)."""
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be in (0.5, 1).")
    if window < MIN_OBS:
        raise ValueError(f"window must be at least {MIN_OBS} observations.")
    returns = prices[portfolio.symbols].pct_change().dropna()
    if len(returns) <= window:
        raise ValueError(
            f"Need more than {window} return observations to backtest, got {len(returns)}."
        )
    latest = {str(k): float(v) for k, v in prices.iloc[-1].items()}
    exposures = pd.Series(portfolio.market_values(latest), dtype=float)[list(returns.columns)]

    var = rolling_var(exposures, returns, method=method, confidence=confidence, window=window)
    test_returns = returns.iloc[window:]
    pnl = np.asarray(
        test_returns.to_numpy(dtype=np.float64) @ exposures.to_numpy(dtype=np.float64),
        dtype=np.float64,
    )
    return BacktestResult(
        method=method,
        confidence=confidence,
        window=window,
        dates=pd.DatetimeIndex(test_returns.index),
        pnl=pnl,
        var=var,
    )
