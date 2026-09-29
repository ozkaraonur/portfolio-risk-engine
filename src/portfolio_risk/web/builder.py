"""UI-independent helpers: convert editable tables to/from a ``Portfolio``."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from portfolio_risk.models import Asset, AssetClass, CashBalance, Portfolio, Position

BROKERS = ["InteractiveBrokers", "Binance", "BIST", "Custom"]
ASSET_CLASSES = [c.value for c in AssetClass]

COL_BROKER = "Broker"
COL_SYMBOL = "Sembol"
COL_CLASS = "Varlık Sınıfı"
COL_QTY = "Miktar / Lot"
COL_TAGS = "Etiketler"
COL_AMOUNT = "Tutar (USD)"
POSITION_COLUMNS = [COL_BROKER, COL_SYMBOL, COL_CLASS, COL_QTY, COL_TAGS]
CASH_COLUMNS = [COL_BROKER, COL_AMOUNT]

# Sample-portfolio lookup: repository checkout first, then the working directory.
SAMPLE_CANDIDATES = (
    Path(__file__).resolve().parents[3] / "examples" / "portfolio.json",
    Path.cwd() / "examples" / "portfolio.json",
)


class PortfolioInputError(ValueError):
    """User-facing validation error for the portfolio tables."""


def empty_positions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            COL_BROKER: pd.Series(dtype="object"),
            COL_SYMBOL: pd.Series(dtype="object"),
            COL_CLASS: pd.Series(dtype="object"),
            COL_QTY: pd.Series(dtype="float64"),
            COL_TAGS: pd.Series(dtype="object"),
        }
    )


def empty_cash() -> pd.DataFrame:
    return pd.DataFrame(
        {COL_BROKER: pd.Series(dtype="object"), COL_AMOUNT: pd.Series(dtype="float64")}
    )


def _blank(value: object) -> bool:
    if value is None or value is pd.NA:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return str(value).strip() == ""


def _to_float(value: object) -> float:
    return float(str(value))


def parse_tags(raw: object) -> tuple[str, ...]:
    """``"tech, growth"`` -> ``("tech", "growth")``."""
    if _blank(raw):
        return ()
    return tuple(t.strip() for t in str(raw).replace(";", ",").split(",") if t.strip())


def frames_to_portfolio(
    positions: pd.DataFrame, cash: pd.DataFrame, name: str = "web-portfolio"
) -> Portfolio:
    """Validate the editor tables and build a ``Portfolio`` (fully empty rows are ignored)."""
    built_positions: list[Position] = []
    for idx, row in enumerate(positions.to_dict("records"), start=1):
        symbol, qty = row.get(COL_SYMBOL), row.get(COL_QTY)
        if _blank(symbol) and _blank(qty):
            continue
        if _blank(symbol) or _blank(qty):
            raise PortfolioInputError(f"Pozisyon satırı {idx}: sembol ve miktar birlikte gerekli.")
        asset_class = row.get(COL_CLASS)
        if _blank(asset_class):
            raise PortfolioInputError(f"Pozisyon satırı {idx}: varlık sınıfı seçilmeli.")
        try:
            asset = Asset(
                symbol=str(symbol),
                asset_class=AssetClass(str(asset_class)),
                tags=parse_tags(row.get(COL_TAGS)),
            )
            broker = row.get(COL_BROKER)
            built_positions.append(
                Position(
                    asset=asset,
                    quantity=_to_float(qty),
                    broker="Custom" if _blank(broker) else str(broker),
                )
            )
        except (ValidationError, ValueError) as exc:
            raise PortfolioInputError(f"Pozisyon satırı {idx} ({symbol}): geçersiz değer.") from exc

    built_cash: list[CashBalance] = []
    for idx, row in enumerate(cash.to_dict("records"), start=1):
        amount = row.get(COL_AMOUNT)
        if _blank(amount):
            continue
        try:
            broker = row.get(COL_BROKER)
            built_cash.append(
                CashBalance(
                    broker="Custom" if _blank(broker) else str(broker), amount=_to_float(amount)
                )
            )
        except (ValidationError, ValueError) as exc:
            raise PortfolioInputError(f"Nakit satırı {idx}: tutar negatif olamaz.") from exc

    if not built_positions:
        raise PortfolioInputError("En az bir pozisyon ekleyin.")
    return Portfolio(name=name, positions=tuple(built_positions), cash=tuple(built_cash))


def portfolio_to_frames(portfolio: Portfolio) -> tuple[pd.DataFrame, pd.DataFrame]:
    positions = pd.DataFrame(
        [
            {
                COL_BROKER: p.broker,
                COL_SYMBOL: p.asset.symbol,
                COL_CLASS: p.asset.asset_class.value,
                COL_QTY: p.quantity,
                COL_TAGS: ", ".join(p.asset.tags),
            }
            for p in portfolio.positions
        ],
        columns=POSITION_COLUMNS,
    )
    cash = pd.DataFrame(
        [{COL_BROKER: c.broker, COL_AMOUNT: c.amount} for c in portfolio.cash],
        columns=CASH_COLUMNS,
    )
    return positions, cash


def load_sample_portfolio() -> Portfolio:
    for path in SAMPLE_CANDIDATES:
        if path.is_file():
            return Portfolio.model_validate_json(path.read_text(encoding="utf-8"))
    raise FileNotFoundError("examples/portfolio.json bulunamadı.")


def broker_options(*frames: pd.DataFrame) -> list[str]:
    """Dropdown options: the standard brokers plus any others present in the tables."""
    extra = {
        str(b) for f in frames for b in f[COL_BROKER].dropna() if str(b) and str(b) not in BROKERS
    }
    return [*BROKERS, *sorted(extra)]
