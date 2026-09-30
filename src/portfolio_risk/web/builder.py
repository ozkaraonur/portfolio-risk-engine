"""UI-independent helpers: catalog-driven position/cash tables <-> ``Portfolio``."""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd
from pydantic import ValidationError

from portfolio_risk.catalog import CatalogEntry, get_entry
from portfolio_risk.models import CashBalance, Portfolio, Position

BROKERS = ["InteractiveBrokers", "Binance", "BIST", "Custom"]

COL_DELETE = "Sil"
COL_CATEGORY = "Kategori"
COL_SYMBOL = "Sembol"
COL_NAME = "Varlık"
COL_CLASS = "Sınıf"
COL_TAGS = "Etiketler"
COL_QTY = "Miktar / Adet"
COL_BROKER = "Broker"
COL_AMOUNT = "Tutar"  # internal key; the shown label carries the chosen currency
POSITION_COLUMNS = [COL_DELETE, COL_CATEGORY, COL_SYMBOL, COL_NAME, COL_CLASS, COL_TAGS, COL_QTY]
CASH_COLUMNS = [COL_BROKER, COL_AMOUNT]

# Sample-portfolio lookup: repository checkout first, then the working directory.
SAMPLE_CANDIDATES = (
    Path(__file__).resolve().parents[3] / "examples" / "portfolio.json",
    Path.cwd() / "examples" / "portfolio.json",
)


class PortfolioInputError(ValueError):
    """User-facing validation error for the portfolio tables.

    ``key`` and ``params`` name the translatable message (see ``web.i18n``); the plain text is
    the Turkish default.
    """

    def __init__(self, message: str, key: str | None = None, **params: object) -> None:
        super().__init__(message)
        self.key = key
        self.params = params


def empty_positions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            COL_DELETE: pd.Series(dtype="bool"),
            COL_CATEGORY: pd.Series(dtype="object"),
            COL_SYMBOL: pd.Series(dtype="object"),
            COL_NAME: pd.Series(dtype="object"),
            COL_CLASS: pd.Series(dtype="object"),
            COL_TAGS: pd.Series(dtype="object"),
            COL_QTY: pd.Series(dtype="float64"),
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


def _row(entry: CatalogEntry, quantity: float) -> dict[str, object]:
    return {
        COL_DELETE: False,
        COL_CATEGORY: entry.category,
        COL_SYMBOL: entry.symbol,
        COL_NAME: entry.name,
        COL_CLASS: entry.asset_class.value,
        COL_TAGS: ", ".join(entry.tags),
        COL_QTY: float(quantity),
    }


def add_position(positions: pd.DataFrame, entry: CatalogEntry, quantity: float) -> pd.DataFrame:
    """Add ``quantity`` of ``entry``; an asset already in the table has its quantity increased."""
    if not quantity > 0:
        raise PortfolioInputError("Miktar sıfırdan büyük olmalı.", "err_qty_positive")
    frame = positions.copy()
    frame[COL_QTY] = frame[COL_QTY].astype(float)
    existing = frame[COL_SYMBOL] == entry.symbol
    if existing.any():
        frame.loc[existing, COL_QTY] = frame.loc[existing, COL_QTY].astype(float) + quantity
        return frame.reset_index(drop=True)
    new = pd.DataFrame([_row(entry, quantity)], columns=POSITION_COLUMNS)
    return pd.concat([frame, new], ignore_index=True) if len(frame) else new


def remove_marked(positions: pd.DataFrame) -> pd.DataFrame:
    """Drop the rows whose delete box is ticked."""
    keep = ~positions[COL_DELETE].fillna(False).astype(bool)
    return positions[keep].reset_index(drop=True)


def frames_to_portfolio(
    positions: pd.DataFrame,
    cash: pd.DataFrame,
    name: str = "web-portfolio",
    base_currency: str = "USD",
) -> Portfolio:
    """Build a ``Portfolio`` from the tables; asset details always come from the catalog.

    Cash amounts are taken to be in ``base_currency``.
    """
    built_positions: list[Position] = []
    for idx, row in enumerate(positions.to_dict("records"), start=1):
        symbol, qty = row.get(COL_SYMBOL), row.get(COL_QTY)
        if _blank(symbol):
            continue
        try:
            entry = get_entry(str(symbol))
        except KeyError as exc:
            raise PortfolioInputError(
                f"Pozisyon satırı {idx}: {exc.args[0]}", "err_row_unknown", idx=idx, symbol=symbol
            ) from exc
        try:
            quantity = _to_float(qty)
            built_positions.append(
                Position(asset=entry.to_asset(), quantity=quantity, broker=entry.broker)
            )
        except (ValidationError, ValueError) as exc:
            raise PortfolioInputError(
                f"Pozisyon satırı {idx} ({entry.symbol}): miktar sıfırdan büyük olmalı.",
                "err_row_qty",
                idx=idx,
                symbol=entry.symbol,
            ) from exc

    built_cash: list[CashBalance] = []
    for idx, row in enumerate(cash.to_dict("records"), start=1):
        amount = row.get(COL_AMOUNT)
        if _blank(amount):
            continue
        try:
            broker = row.get(COL_BROKER)
            built_cash.append(
                CashBalance(
                    broker="Custom" if _blank(broker) else str(broker),
                    amount=_to_float(amount),
                    currency=base_currency,
                )
            )
        except (ValidationError, ValueError) as exc:
            raise PortfolioInputError(
                f"Nakit satırı {idx}: tutar negatif olamaz.", "err_cash_negative", idx=idx
            ) from exc

    if not built_positions:
        raise PortfolioInputError("En az bir pozisyon ekleyin.", "err_no_positions")
    return Portfolio(
        name=name,
        base_currency=base_currency,
        positions=tuple(built_positions),
        cash=tuple(built_cash),
    )


def portfolio_to_frames(portfolio: Portfolio) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inverse of :func:`frames_to_portfolio`; symbols must exist in the catalog."""
    frame = empty_positions()
    for symbol, quantity in portfolio.quantities().items():
        try:
            entry = get_entry(symbol)
        except KeyError as exc:
            raise PortfolioInputError(exc.args[0], "err_sample_unknown", symbol=symbol) from exc
        frame = add_position(frame, entry, quantity)
    cash = pd.DataFrame(
        [{COL_BROKER: c.broker, COL_AMOUNT: c.amount} for c in portfolio.cash],
        columns=CASH_COLUMNS,
    )
    return frame, cash


def load_sample_portfolio() -> Portfolio:
    for path in SAMPLE_CANDIDATES:
        if path.is_file():
            return Portfolio.model_validate_json(path.read_text(encoding="utf-8"))
    raise FileNotFoundError("examples/portfolio.json bulunamadı.")


def broker_options(cash: pd.DataFrame) -> list[str]:
    """Cash-broker dropdown: the standard brokers plus any others present in the table."""
    extra = {str(b) for b in cash[COL_BROKER].dropna() if str(b) and str(b) not in BROKERS}
    return [*BROKERS, *sorted(extra)]
