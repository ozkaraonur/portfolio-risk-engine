"""Import broker position exports (CSV) into a :class:`Portfolio`.

One tolerant reader covers the common exports (Interactive Brokers, Schwab, Binance, generic
spreadsheets): columns are matched by name aliases, the delimiter is sniffed, and number formats
with thousands separators or decimal commas are accepted. The broker name is supplied by the
caller, so several files can be merged into one multi-broker portfolio.
"""

from __future__ import annotations

import contextlib
import csv
import io
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from portfolio_risk.catalog import CATALOG
from portfolio_risk.models import Asset, AssetClass, CashBalance, Portfolio, Position

FIAT = frozenset({"USD", "EUR", "GBP", "CHF", "JPY", "CAD", "AUD", "TRY"})
STABLECOINS = frozenset({"USDT", "USDC", "BUSD", "DAI"})  # treated as USD cash

_ALIASES: dict[str, tuple[str, ...]] = {
    "symbol": ("symbol", "ticker", "instrument", "coin", "asset", "code"),
    "quantity": ("quantity", "qty", "shares", "position", "total", "balance", "units", "amount"),
    "currency": ("currency", "ccy"),
    "asset_class": ("assetclass", "assetcategory", "type", "class"),
}
_CLASS_NAMES: dict[str, str] = {
    "stk": "equity",
    "stock": "equity",
    "equity": "equity",
    "etf": "equity",
    "fund": "equity",
    "crypto": "crypto",
    "cryptocurrency": "crypto",
    "cmdty": "commodity",
    "commodity": "commodity",
    "fut": "commodity",
    "future": "commodity",
    "cash": "cash",
    "forex": "cash",
    "fx": "cash",
}
_GROUPED = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")


class ImportFormatError(ValueError):
    """The file is not a recognisable positions export."""


@dataclass(frozen=True)
class ImportResult:
    broker: str
    positions: tuple[Position, ...]
    cash: tuple[CashBalance, ...]
    skipped: tuple[str, ...] = field(default=())  # human-readable notes on ignored rows


def parse_number(text: str) -> float:
    """``1,234.50``, ``1.234,50``, ``1234,5`` and ``$ 12`` all parse; anything else raises."""
    cleaned = re.sub(r"[^\d,.\-]", "", text.strip())
    if not cleaned or cleaned in {"-", ".", ","}:
        raise ValueError(f"not a number: {text!r}")
    if "," in cleaned and "." in cleaned:
        if cleaned.rfind(",") > cleaned.rfind("."):  # 1.234,50
            cleaned = cleaned.replace(".", "").replace(",", ".")
        else:  # 1,234.50
            cleaned = cleaned.replace(",", "")
    elif "," in cleaned:
        cleaned = cleaned.replace(",", "") if _GROUPED.match(cleaned) else cleaned.replace(",", ".")
    try:
        return float(cleaned)
    except ValueError:
        raise ValueError(f"not a number: {text!r}") from None


def _norm(header: str) -> str:
    return re.sub(r"[^a-z0-9]", "", header.lower())


def _column_map(headers: Sequence[str]) -> dict[str, str]:
    by_norm = {_norm(h): h for h in headers if h.strip()}
    found: dict[str, str] = {}
    for field_name, aliases in _ALIASES.items():
        for alias in aliases:
            if alias in by_norm and by_norm[alias] not in found.values():
                found[field_name] = by_norm[alias]
                break
    missing = [f for f in ("symbol", "quantity") if f not in found]
    if missing:
        raise ImportFormatError(
            f"Could not find a {' / '.join(missing)} column; headers are {list(headers)}."
        )
    return found


def _rows(text: str) -> tuple[list[str], list[dict[str, str]]]:
    text = text.lstrip("﻿")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ImportFormatError("The file has no data rows.")
    dialect: type[csv.Dialect] = csv.excel
    with contextlib.suppress(csv.Error):  # single-column header: keep the comma dialect
        dialect = csv.Sniffer().sniff(lines[0], delimiters=",;\t")
    reader = csv.DictReader(io.StringIO("\n".join(lines)), dialect=dialect)
    headers = [h or "" for h in (reader.fieldnames or [])]
    return headers, [{k: (v or "").strip() for k, v in row.items() if k} for row in reader]


def _asset_class(raw: str, symbol: str) -> str:
    if raw:
        mapped = _CLASS_NAMES.get(_norm(raw))
        if mapped is None:
            raise ValueError(f"unknown asset class {raw!r}")
        return mapped
    if symbol in CATALOG:
        return CATALOG[symbol].asset_class.value
    return "equity"


def parse_broker_csv(text: str, broker: str, *, base_currency: str = "USD") -> ImportResult:
    """Read one broker export. Zero quantities are skipped; short positions are rejected."""
    headers, rows = _rows(text)
    cols = _column_map(headers)
    base = base_currency.upper()
    positions: list[Position] = []
    cash: dict[str, float] = {}
    skipped: list[str] = []

    for line, row in enumerate(rows, start=2):
        symbol = row.get(cols["symbol"], "").upper()
        if not symbol:
            skipped.append(f"line {line}: empty symbol")
            continue
        try:
            quantity = parse_number(row.get(cols["quantity"], ""))
        except ValueError as exc:
            raise ImportFormatError(f"line {line} ({symbol}): {exc}") from exc
        if quantity == 0:
            skipped.append(f"line {line}: {symbol} has zero quantity")
            continue
        if quantity < 0:
            raise ImportFormatError(f"line {line} ({symbol}): short positions are not supported.")
        entry = CATALOG.get(symbol)
        currency = row.get(cols.get("currency", ""), "").upper() or (
            entry.currency if entry else base
        )
        try:
            kind = _asset_class(row.get(cols.get("asset_class", ""), ""), symbol)
        except ValueError as exc:
            raise ImportFormatError(f"line {line} ({symbol}): {exc}") from exc

        if symbol in STABLECOINS:
            cash["USD"] = cash.get("USD", 0.0) + quantity
        elif kind == "cash" or (symbol in FIAT and not row.get(cols.get("asset_class", ""), "")):
            code = symbol if symbol in FIAT else currency
            cash[code] = cash.get(code, 0.0) + quantity
        else:
            asset = Asset(
                symbol=symbol,
                asset_class=AssetClass(kind),
                name=entry.name if entry else None,
                currency=currency,
                data_symbol=entry.data_symbol if entry else None,
                tags=entry.tags if entry else (),
            )
            positions.append(Position(asset=asset, quantity=quantity, broker=broker))

    if not positions and not cash:
        raise ImportFormatError("No usable rows found.")
    balances = tuple(CashBalance(broker=broker, amount=a, currency=c) for c, a in cash.items())
    return ImportResult(broker, tuple(positions), balances, tuple(skipped))


def build_portfolio(
    sources: Mapping[str, str], *, name: str = "imported", base_currency: str = "USD"
) -> tuple[Portfolio, list[str]]:
    """Merge several ``{broker: csv text}`` exports; returns the portfolio and skip notes."""
    positions: list[Position] = []
    cash: list[CashBalance] = []
    notes: list[str] = []
    for broker, text in sources.items():
        result = parse_broker_csv(text, broker, base_currency=base_currency)
        positions += result.positions
        cash += result.cash
        notes += [f"{broker}: {n}" for n in result.skipped]
    portfolio = Portfolio(
        name=name, base_currency=base_currency.upper(), positions=tuple(positions), cash=tuple(cash)
    )
    return portfolio, notes
