"""Portfolio, position and cash models."""

from __future__ import annotations

from collections.abc import Mapping

from pydantic import BaseModel, ConfigDict, Field, model_validator

from portfolio_risk.models.asset import Asset


class Position(BaseModel):
    """A holding of ``quantity`` units of ``asset`` at a given broker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    asset: Asset
    quantity: float = Field(gt=0)
    broker: str = Field(default="default", min_length=1)


class CashBalance(BaseModel):
    """Cash held at a broker (assumed to be in the portfolio base currency)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    broker: str = Field(default="default", min_length=1)
    amount: float = Field(ge=0)
    currency: str = Field(default="USD", min_length=3, max_length=3)


class Portfolio(BaseModel):
    """A multi-broker portfolio of positions plus cash."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "portfolio"
    base_currency: str = Field(default="USD", min_length=3, max_length=3)
    positions: tuple[Position, ...] = ()
    cash: tuple[CashBalance, ...] = ()

    @model_validator(mode="after")
    def _check_currencies(self) -> Portfolio:
        # Single-currency valuation for now; FX conversion is out of scope for milestone 1.
        foreign = {c.currency for c in self.cash} | {p.asset.currency for p in self.positions}
        foreign.discard(self.base_currency)
        if foreign:
            raise ValueError(f"Non-base currencies not supported yet: {sorted(foreign)}")
        return self

    @property
    def symbols(self) -> list[str]:
        """Unique asset symbols, in first-seen order."""
        return list(dict.fromkeys(p.asset.symbol for p in self.positions))

    @property
    def assets(self) -> list[Asset]:
        """Unique assets, in first-seen order."""
        return list({p.asset.symbol: p.asset for p in self.positions}.values())

    @property
    def brokers(self) -> list[str]:
        return sorted({p.broker for p in self.positions} | {c.broker for c in self.cash})

    @property
    def total_cash(self) -> float:
        return sum(c.amount for c in self.cash)

    def quantities(self) -> dict[str, float]:
        """Total quantity per symbol, aggregated across brokers."""
        totals: dict[str, float] = {}
        for p in self.positions:
            totals[p.asset.symbol] = totals.get(p.asset.symbol, 0.0) + p.quantity
        return totals

    def market_values(self, prices: Mapping[str, float]) -> dict[str, float]:
        """Market value per symbol given latest ``prices``."""
        missing = [s for s in self.symbols if s not in prices]
        if missing:
            raise KeyError(f"Missing prices for: {missing}")
        return {s: q * prices[s] for s, q in self.quantities().items()}

    def total_value(self, prices: Mapping[str, float]) -> float:
        """Total portfolio value: positions plus cash."""
        return sum(self.market_values(prices).values()) + self.total_cash

    def weights(self, prices: Mapping[str, float]) -> dict[str, float]:
        """Weight per symbol, plus ``CASH``, relative to total value (sums to 1)."""
        total = self.total_value(prices)
        if total <= 0:
            raise ValueError("Portfolio has zero value; weights are undefined.")
        result = {s: v / total for s, v in self.market_values(prices).items()}
        if self.total_cash > 0:
            result["CASH"] = self.total_cash / total
        return result

    @classmethod
    def from_weights(
        cls,
        total_value: float,
        weights: Mapping[Asset, float],
        prices: Mapping[str, float],
        *,
        cash_weight: float = 0.0,
        broker: str = "default",
        name: str = "portfolio",
        base_currency: str = "USD",
    ) -> Portfolio:
        """Build a portfolio from target weights; ``weights`` + ``cash_weight`` must sum to 1."""
        if total_value <= 0:
            raise ValueError("total_value must be positive.")
        if cash_weight < 0 or any(w < 0 for w in weights.values()):
            raise ValueError("Weights must be non-negative.")
        if abs(sum(weights.values()) + cash_weight - 1.0) > 1e-9:
            raise ValueError("Weights and cash_weight must sum to 1.")
        positions = tuple(
            Position(asset=a, quantity=total_value * w / prices[a.symbol], broker=broker)
            for a, w in weights.items()
            if w > 0
        )
        cash = (
            (CashBalance(broker=broker, amount=total_value * cash_weight, currency=base_currency),)
            if cash_weight > 0
            else ()
        )
        return cls(name=name, base_currency=base_currency, positions=positions, cash=cash)
