"""Portfolio-level risk report: VaR, CVaR and diversification benefit."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import pandas as pd

from portfolio_risk.models import Portfolio
from portfolio_risk.risk.covariance import covariance_matrix
from portfolio_risk.risk.var import historical_var_cvar, parametric_var_cvar

MIN_OBS = 20


class Method(StrEnum):
    PARAMETRIC = "parametric"
    HISTORICAL = "historical"


@dataclass(frozen=True)
class RiskReport:
    method: Method
    confidence: float
    horizon: int
    portfolio_value: float  # positions + cash
    var: float
    cvar: float
    standalone_var: dict[str, float]  # each asset's VaR as if held alone

    @property
    def undiversified_var(self) -> float:
        """Sum of standalone VaRs (no diversification credit)."""
        return sum(self.standalone_var.values())

    @property
    def diversification_benefit(self) -> float:
        """Currency risk reduction: sum of standalone VaRs minus portfolio VaR."""
        return self.undiversified_var - self.var

    @property
    def diversification_ratio(self) -> float:
        """Benefit as a fraction of undiversified VaR (0 = none, -> 1 = fully diversified)."""
        total = self.undiversified_var
        return self.diversification_benefit / total if total > 0 else 0.0


def analyze_risk(
    portfolio: Portfolio,
    prices: pd.DataFrame,
    *,
    method: Method,
    confidence: float = 0.95,
    horizon: int = 1,
) -> RiskReport:
    """Compute VaR/CVaR for ``portfolio`` from a price history (cash carries no risk)."""
    returns = prices[portfolio.symbols].pct_change().dropna()
    if len(returns) < MIN_OBS:
        raise ValueError(f"Need at least {MIN_OBS} return observations, got {len(returns)}.")
    latest = {str(k): float(v) for k, v in prices.iloc[-1].items()}
    exposures = pd.Series(portfolio.market_values(latest), dtype=float)

    if method is Method.PARAMETRIC:
        cov = covariance_matrix(returns)

        def measure(e: pd.Series[float]) -> tuple[float, float]:
            return parametric_var_cvar(e, cov, confidence, horizon)
    else:

        def measure(e: pd.Series[float]) -> tuple[float, float]:
            return historical_var_cvar(e, returns, confidence, horizon)

    var, cvar = measure(exposures)
    standalone = {s: measure(exposures[[s]])[0] for s in exposures.index}
    return RiskReport(
        method=method,
        confidence=confidence,
        horizon=horizon,
        portfolio_value=portfolio.total_value(latest),
        var=var,
        cvar=cvar,
        standalone_var=standalone,
    )
