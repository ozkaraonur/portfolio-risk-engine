"""Risk limits: check an analysis against policy thresholds and report breaches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from portfolio_risk.reporting.analysis import RiskAnalysis


class RiskLimits(BaseModel):
    """Thresholds; every field is optional and only the ones set are checked."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_var_pct: float | None = Field(
        default=None,
        gt=0,
        le=1,
        description="Headline (parametric, 10-day, 99%) VaR as a fraction of portfolio value.",
    )
    max_asset_weight: float | None = Field(
        default=None, gt=0, le=1, description="Largest single-asset weight of portfolio value."
    )
    max_risk_share: float | None = Field(
        default=None, gt=0, le=1, description="Largest single-asset share of portfolio VaR."
    )
    min_cash_ratio: float | None = Field(
        default=None, ge=0, lt=1, description="Minimum cash as a fraction of portfolio value."
    )
    max_ruin_probability: float | None = Field(
        default=None,
        gt=0,
        le=1,
        description="Monte Carlo probability of hitting the ruin threshold.",
    )


@dataclass(frozen=True)
class LimitCheck:
    name: str
    limit: float
    actual: float
    is_minimum: bool  # True: actual must be >= limit; False: actual must be <= limit
    detail: str = ""  # e.g. the asset responsible

    @property
    def breached(self) -> bool:
        return self.actual < self.limit if self.is_minimum else self.actual > self.limit


def load_limits(path: Path) -> RiskLimits:
    return RiskLimits.model_validate_json(path.read_text(encoding="utf-8"))


def check_limits(analysis: RiskAnalysis, limits: RiskLimits) -> list[LimitCheck]:
    """One :class:`LimitCheck` per configured limit, in a fixed order."""
    checks: list[LimitCheck] = []
    if limits.max_var_pct is not None:
        pct = analysis.headline_var.var / analysis.total_value if analysis.total_value > 0 else 0.0
        checks.append(LimitCheck("VaR / value (99%, 10d)", limits.max_var_pct, pct, False))
    if limits.max_asset_weight is not None and analysis.assets:
        top = max(analysis.assets, key=lambda a: a.weight)
        checks.append(
            LimitCheck(
                "Largest position weight", limits.max_asset_weight, top.weight, False, top.symbol
            )
        )
    if limits.max_risk_share is not None:
        shares = analysis.contributions.var_share
        symbol = str(shares.idxmax())
        checks.append(
            LimitCheck(
                "Largest share of VaR", limits.max_risk_share, float(shares.max()), False, symbol
            )
        )
    if limits.min_cash_ratio is not None:
        checks.append(LimitCheck("Cash ratio", limits.min_cash_ratio, analysis.cash_ratio, True))
    if limits.max_ruin_probability is not None:
        checks.append(
            LimitCheck(
                "Monte Carlo ruin probability",
                limits.max_ruin_probability,
                analysis.monte_carlo.prob_ruin,
                False,
            )
        )
    return checks
