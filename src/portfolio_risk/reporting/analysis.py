"""Run the full analytics suite once and bundle the results for the renderers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from portfolio_risk.models import Portfolio
from portfolio_risk.risk import (
    BUILTIN_SCENARIOS,
    CORE_METHODS,
    BacktestResult,
    Method,
    MonteCarloReport,
    RiskReport,
    StressReport,
    analyze_risk,
    correlation_matrix,
    run_backtest,
    run_monte_carlo,
    run_stress,
)

CONFIDENCES = (0.95, 0.99)
HORIZONS = (1, 10)
HEADLINE_CONFIDENCE = 0.99
HEADLINE_HORIZON = 10
BACKTEST_CONFIDENCE = 0.99
BACKTEST_WINDOW = 250
MIN_BACKTEST_DAYS = 60  # test days needed on top of the estimation window


@dataclass(frozen=True)
class AssetRow:
    symbol: str
    asset_class: str
    value: float
    weight: float
    standalone_var: float  # headline confidence / horizon, parametric


@dataclass(frozen=True)
class RiskAnalysis:
    portfolio: Portfolio
    as_of: date
    total_value: float
    cash: float
    assets: tuple[AssetRow, ...]
    correlation: pd.DataFrame
    var_reports: tuple[RiskReport, ...]
    monte_carlo: MonteCarloReport
    stress: StressReport
    backtests: tuple[BacktestResult, ...]  # empty when the history is too short
    seed: int
    observations: int  # daily return observations used for estimation

    @property
    def cash_ratio(self) -> float:
        return self.cash / self.total_value if self.total_value > 0 else 0.0

    def var_report(self, method: Method, confidence: float, horizon: int) -> RiskReport:
        for r in self.var_reports:
            if r.method is method and r.confidence == confidence and r.horizon == horizon:
                return r
        raise KeyError((method, confidence, horizon))

    @property
    def headline_var(self) -> RiskReport:
        """Parametric 10-day 99% VaR report."""
        return self.var_report(Method.PARAMETRIC, HEADLINE_CONFIDENCE, HEADLINE_HORIZON)

    @property
    def top_risk_asset(self) -> AssetRow | None:
        return max(self.assets, key=lambda a: a.standalone_var, default=None)


def build_analysis(
    portfolio: Portfolio,
    prices: pd.DataFrame,
    *,
    simulations: int = 10_000,
    mc_days: int = 252,
    seed: int = 42,
    loss_threshold: float = 0.3,
) -> RiskAnalysis:
    """Run VaR/CVaR, Monte Carlo and the built-in stress scenarios on ``prices``."""
    if not portfolio.positions:
        raise ValueError("Portfolio has no positions to analyse.")
    latest = {str(k): float(v) for k, v in prices.iloc[-1].items()}
    values = portfolio.market_values(latest)
    total = portfolio.total_value(latest)

    var_reports = tuple(
        analyze_risk(portfolio, prices, method=m, confidence=c, horizon=h)
        for m in CORE_METHODS
        for c in CONFIDENCES
        for h in HORIZONS
    )
    headline = next(
        r
        for r in var_reports
        if r.method is Method.PARAMETRIC
        and r.confidence == HEADLINE_CONFIDENCE
        and r.horizon == HEADLINE_HORIZON
    )
    classes = {a.symbol: a.asset_class.value for a in portfolio.assets}
    assets = tuple(
        AssetRow(s, classes[s], v, v / total, headline.standalone_var[s]) for s, v in values.items()
    )
    returns = prices[portfolio.symbols].pct_change().dropna()
    backtests = (
        tuple(
            run_backtest(
                portfolio,
                prices,
                method=m,
                confidence=BACKTEST_CONFIDENCE,
                window=BACKTEST_WINDOW,
            )
            for m in Method
        )
        if len(returns) >= BACKTEST_WINDOW + MIN_BACKTEST_DAYS
        else ()
    )
    return RiskAnalysis(
        portfolio=portfolio,
        as_of=prices.index[-1].date(),
        total_value=total,
        cash=portfolio.total_cash,
        assets=assets,
        correlation=correlation_matrix(returns),
        var_reports=var_reports,
        monte_carlo=run_monte_carlo(
            portfolio,
            prices,
            n_simulations=simulations,
            days=mc_days,
            seed=seed,
            loss_threshold=loss_threshold,
        ),
        stress=run_stress(portfolio, latest, list(BUILTIN_SCENARIOS.values())),
        backtests=backtests,
        seed=seed,
        observations=len(returns),
    )
