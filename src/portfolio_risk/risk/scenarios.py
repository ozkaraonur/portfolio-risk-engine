"""Stress scenario definitions: historical shocks, user shocks and beta-driven market shocks."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from portfolio_risk.models import Asset, AssetClass


@dataclass(frozen=True)
class Scenario:
    """Instantaneous percentage price shocks (-0.3 = -30%); cash is never shocked.

    Precedence for an asset: ``symbol_shocks`` > ``tag_shocks`` (most severe matching tag)
    > ``class_shocks`` > no shock.
    """

    name: str
    description: str = ""
    class_shocks: Mapping[AssetClass, float] = field(default_factory=dict)
    tag_shocks: Mapping[str, float] = field(default_factory=dict)
    symbol_shocks: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for shock in (*self.class_shocks.values(), *self.tag_shocks.values()):
            _check_shock(shock)
        for shock in self.symbol_shocks.values():
            _check_shock(shock)

    def shock_for(self, asset: Asset) -> float:
        if asset.symbol in self.symbol_shocks:
            return self.symbol_shocks[asset.symbol]
        tagged = [self.tag_shocks[t] for t in asset.tags if t in self.tag_shocks]
        if tagged:
            return min(tagged)
        return self.class_shocks.get(asset.asset_class, 0.0)


def _check_shock(shock: float) -> None:
    if not np.isfinite(shock) or shock < -1.0:
        raise ValueError(f"Shock must be a finite value >= -1 (-100%), got {shock}.")


# Historical scenarios. Crypto did not exist in 2008, so "crypto" there means high-risk assets.
# 2022: only Tech/Growth (-35%) was specified; broad equities use -20% (S&P 500 was ~-19%).
BUILTIN_SCENARIOS: dict[str, Scenario] = {
    "gfc-2008": Scenario(
        name="gfc-2008",
        description="2008 Global Financial Crisis (Lehman)",
        class_shocks={
            AssetClass.EQUITY: -0.45,
            AssetClass.CRYPTO: -0.60,
            AssetClass.COMMODITY: 0.15,
        },
    ),
    "covid-2020": Scenario(
        name="covid-2020",
        description="March 2020 COVID liquidity shock",
        class_shocks={
            AssetClass.EQUITY: -0.30,
            AssetClass.CRYPTO: -0.50,
            AssetClass.COMMODITY: -0.25,
        },
    ),
    "inflation-2022": Scenario(
        name="inflation-2022",
        description="2022 inflation / rate shock and tech sell-off",
        class_shocks={
            AssetClass.EQUITY: -0.20,
            AssetClass.CRYPTO: -0.65,
            AssetClass.COMMODITY: 0.25,
        },
        tag_shocks={"tech": -0.35, "growth": -0.35},
    ),
}


def get_scenario(name: str) -> Scenario:
    try:
        return BUILTIN_SCENARIOS[name.strip().lower()]
    except KeyError:
        valid = ", ".join(BUILTIN_SCENARIOS)
        raise ValueError(f"Unknown scenario '{name}'. Available: {valid}.") from None


def parse_custom_shocks(spec: str, name: str = "custom") -> Scenario:
    """Parse ``"equity=-0.15,crypto=-0.30,AAPL=-0.5"``.

    Keys that are asset classes shock the whole class; ``tag:<label>`` shocks a tag; any other
    key is treated as an asset symbol.
    """
    classes: dict[AssetClass, float] = {}
    tags: dict[str, float] = {}
    symbols: dict[str, float] = {}
    for part in filter(None, (p.strip() for p in spec.split(","))):
        key, sep, raw = part.partition("=")
        key = key.strip()
        if not sep or not key:
            raise ValueError(f"Invalid shock '{part}'; expected key=value.")
        try:
            shock = float(raw)
        except ValueError:
            raise ValueError(f"Invalid shock value in '{part}'.") from None
        lowered = key.lower()
        if lowered in {c.value for c in AssetClass}:
            classes[AssetClass(lowered)] = shock
        elif lowered.startswith("tag:"):
            tags[lowered.removeprefix("tag:")] = shock
        else:
            symbols[key.upper()] = shock
    if not (classes or tags or symbols):
        raise ValueError("No shocks given.")
    return Scenario(
        name=name,
        description="User-defined shock",
        class_shocks=classes,
        tag_shocks=tags,
        symbol_shocks=symbols,
    )


def compute_betas(returns: pd.DataFrame, market: pd.Series[float]) -> dict[str, float]:
    """OLS beta of each column of ``returns`` vs ``market``: cov(r, m) / var(m)."""
    joined = returns.join(market.rename("__market__"), how="inner").dropna()
    if len(joined) < 2:
        raise ValueError("Need at least two overlapping observations to estimate betas.")
    m = joined["__market__"].to_numpy(dtype=float)
    var_m = float(np.var(m, ddof=1))
    if var_m <= 0:
        raise ValueError("Market returns have zero variance; beta is undefined.")
    result: dict[str, float] = {}
    for column in returns.columns:
        r = joined[column].to_numpy(dtype=float)
        result[str(column)] = float(np.cov(r, m, ddof=1)[0, 1] / var_m)
    return result


def portfolio_beta(
    exposures: Mapping[str, float], betas: Mapping[str, float], portfolio_value: float
) -> float:
    """Value-weighted beta of the whole portfolio (cash counts as zero beta)."""
    if portfolio_value <= 0:
        raise ValueError("Portfolio has zero value.")
    return sum(v * betas[s] for s, v in exposures.items()) / portfolio_value


def beta_scenario(
    market_shock: float, betas: Mapping[str, float], benchmark: str = "SPY"
) -> Scenario:
    """Scenario where each asset moves ``beta * market_shock`` (floored at -100%)."""
    if not np.isfinite(market_shock) or market_shock <= -1.0:
        raise ValueError("market_shock must be finite and greater than -1.")
    return Scenario(
        name=f"market{market_shock:+.0%}",
        description=f"Market ({benchmark}) moves {market_shock:+.0%}; assets move by beta",
        symbol_shocks={s: max(b * market_shock, -1.0) for s, b in betas.items()},
    )
