"""Apply stress scenarios to a portfolio."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from portfolio_risk.models import AssetClass, Portfolio
from portfolio_risk.risk.scenarios import Scenario


@dataclass(frozen=True)
class AssetImpact:
    symbol: str
    asset_class: AssetClass
    value: float  # current market value
    shock: float  # fractional price change, e.g. -0.3

    @property
    def pnl(self) -> float:
        return self.value * self.shock

    @property
    def stressed_value(self) -> float:
        return self.value + self.pnl


@dataclass(frozen=True)
class ScenarioResult:
    scenario: Scenario
    impacts: tuple[AssetImpact, ...]
    cash: float

    @property
    def portfolio_value(self) -> float:
        return sum(i.value for i in self.impacts) + self.cash

    @property
    def total_pnl(self) -> float:
        return sum(i.pnl for i in self.impacts)

    @property
    def pnl_pct(self) -> float:
        """Portfolio P&L as a fraction of total value (cash included)."""
        value = self.portfolio_value
        return self.total_pnl / value if value > 0 else 0.0

    @property
    def stressed_value(self) -> float:
        return self.portfolio_value + self.total_pnl


@dataclass(frozen=True)
class StressReport:
    results: tuple[ScenarioResult, ...]

    @property
    def worst_case(self) -> ScenarioResult:
        """Scenario with the largest nominal loss (most negative P&L)."""
        return min(self.results, key=lambda r: r.total_pnl)

    def breaches(self, loss_threshold: float) -> list[ScenarioResult]:
        """Scenarios whose loss is at least ``loss_threshold`` of portfolio value."""
        return [r for r in self.results if -r.pnl_pct >= loss_threshold]


def run_stress(
    portfolio: Portfolio,
    prices: Mapping[str, float],
    scenarios: Sequence[Scenario],
) -> StressReport:
    """Revalue ``portfolio`` at latest ``prices`` under each scenario's instant shocks."""
    if not scenarios:
        raise ValueError("At least one scenario is required.")
    values = portfolio.market_values(prices)
    assets = portfolio.assets
    known = {a.symbol for a in assets}
    for scenario in scenarios:
        unknown = sorted(set(scenario.symbol_shocks) - known)
        if unknown:
            raise ValueError(f"Scenario '{scenario.name}' shocks unknown symbols: {unknown}")
    results = tuple(
        ScenarioResult(
            scenario=s,
            impacts=tuple(
                AssetImpact(a.symbol, a.asset_class, values[a.symbol], s.shock_for(a))
                for a in assets
            ),
            cash=portfolio.total_cash,
        )
        for s in scenarios
    )
    return StressReport(results)
