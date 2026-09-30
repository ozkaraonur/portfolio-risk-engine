from __future__ import annotations

import pytest

from portfolio_risk.importers import (
    ImportFormatError,
    build_portfolio,
    parse_broker_csv,
    parse_number,
)
from portfolio_risk.models import AssetClass


@pytest.mark.parametrize(
    ("text", "value"),
    [
        ("12", 12.0),
        ("1,234.50", 1234.5),
        ("1.234,50", 1234.5),
        ("1234,5", 1234.5),
        ("1,234", 1234.0),
        ("$ 1,000", 1000.0),
        ("-3.5", -3.5),
        ("0.000123", 0.000123),
    ],
)
def test_parse_number_handles_common_formats(text: str, value: float) -> None:
    assert parse_number(text) == pytest.approx(value)


@pytest.mark.parametrize("text", ["", "abc", "-", "n/a"])
def test_parse_number_rejects_garbage(text: str) -> None:
    with pytest.raises(ValueError, match="not a number"):
        parse_number(text)


IBKR = """Symbol,Position,Currency,Asset Category
AAPL,50,USD,STK
SAP,"1,200",EUR,STK
GC,3,USD,CMDTY
EUR,1500,EUR,CASH
NVDA,0,USD,STK
"""


def test_ibkr_style_export() -> None:
    result = parse_broker_csv(IBKR, "ibkr")
    by_symbol = {p.asset.symbol: p for p in result.positions}
    assert set(by_symbol) == {"AAPL", "SAP", "GC"}
    assert by_symbol["SAP"].quantity == 1200
    assert by_symbol["SAP"].asset.currency == "EUR"
    assert by_symbol["GC"].asset.asset_class is AssetClass.COMMODITY
    assert by_symbol["AAPL"].asset.name == "Apple"  # enriched from the catalog
    assert all(p.broker == "ibkr" for p in result.positions)
    assert [(c.currency, c.amount) for c in result.cash] == [("EUR", 1500.0)]
    assert any("NVDA" in note and "zero" in note for note in result.skipped)


BINANCE = "Coin;Total\nBTC;0,25\nETH;2\nUSDT;1.500,00\nTRY;1000\n"


def test_binance_style_semicolon_export_with_decimal_commas() -> None:
    result = parse_broker_csv(BINANCE, "binance")
    quantities = {p.asset.symbol: p.quantity for p in result.positions}
    assert quantities == {"BTC": 0.25, "ETH": 2.0}
    assert {p.asset.asset_class for p in result.positions} == {AssetClass.CRYPTO}
    cash = {c.currency: c.amount for c in result.cash}
    assert cash == {"USD": 1500.0, "TRY": 1000.0}  # stablecoin counts as USD cash


def test_unknown_symbols_default_to_equity_and_utf8_bom_is_ignored() -> None:
    result = parse_broker_csv("﻿Ticker,Qty\nzzz,4\n", "b")
    only = result.positions[0]
    assert only.asset.symbol == "ZZZ"
    assert only.asset.asset_class is AssetClass.EQUITY
    assert only.asset.currency == "USD"


def test_base_currency_is_the_default_asset_currency() -> None:
    result = parse_broker_csv("Symbol,Quantity\nTHYAO,10\n", "b", base_currency="TRY")
    assert result.positions[0].asset.currency == "TRY"


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("Name,Value\nx,1\n", "symbol"),
        ("Symbol,Quantity\n", "no data rows"),
        ("Symbol,Quantity\nAAPL,ten\n", "line 2 .AAPL.: not a number"),
        ("Symbol,Quantity\nAAPL,-5\n", "short positions"),
        ("Symbol,Quantity,Type\nAAPL,5,bond\n", "unknown asset class"),
        ("Symbol,Quantity\nNVDA,0\n", "No usable rows"),
    ],
)
def test_bad_files_give_precise_errors(text: str, message: str) -> None:
    with pytest.raises(ImportFormatError, match=message):
        parse_broker_csv(text, "b")


def test_several_brokers_merge_into_one_portfolio() -> None:
    portfolio, notes = build_portfolio(
        {"ibkr": IBKR, "binance": BINANCE}, name="all", base_currency="USD"
    )
    assert portfolio.name == "all"
    assert portfolio.brokers == ["binance", "ibkr"]
    assert portfolio.quantities()["BTC"] == 0.25
    assert len(portfolio.cash) == 3
    assert notes == ["ibkr: line 6: NVDA has zero quantity"]
