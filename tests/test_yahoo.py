from __future__ import annotations

import json
from datetime import date
from urllib.parse import unquote

import pandas as pd
import pytest

from portfolio_risk.data import (
    DataUnavailableError,
    StooqProvider,
    YahooFx,
    YahooProvider,
)
from portfolio_risk.models import Asset, AssetClass

START, END = date(2024, 1, 1), date(2024, 1, 10)
# 2024-01-02, 03, 04 at 14:30 UTC (a US market open) and a crypto-style midnight stamp.
STAMPS = [1704205800, 1704292200, 1704378600]


def payload(closes: list[float | None], adj: list[float | None] | None = None) -> str:
    indicators: dict[str, object] = {"quote": [{"close": closes}]}
    if adj is not None:
        indicators["adjclose"] = [{"adjclose": adj}]
    result = {"timestamp": STAMPS[: len(closes)], "indicators": indicators}
    return json.dumps({"chart": {"result": [result], "error": None}})


def provider(files: dict[str, str]) -> tuple[YahooProvider, list[str]]:
    urls: list[str] = []

    def fetch(url: str) -> str:
        urls.append(unquote(url))
        ticker = unquote(url.split("/chart/")[1].split("?")[0])
        if ticker not in files:
            raise DataUnavailableError(f"Yahoo returned HTTP 404 for {ticker}")
        return files[ticker]

    return YahooProvider(fetcher=fetch), urls


def test_symbol_mapping() -> None:
    def sym(symbol: str, cls: AssetClass, data_symbol: str | None = None) -> str:
        return YahooProvider.yahoo_symbol(
            Asset(symbol=symbol, asset_class=cls, data_symbol=data_symbol)
        )

    assert sym("AAPL", AssetClass.EQUITY) == "AAPL"
    assert sym("THYAO", AssetClass.EQUITY) == "THYAO.IS"  # BIST, from the catalog
    assert sym("BTC", AssetClass.CRYPTO) == "BTC-USD"
    assert sym("GOLD", AssetClass.COMMODITY, "gc.f") == "GC=F"  # catalog wins over a Stooq symbol
    assert sym("XYZ", AssetClass.EQUITY, "XYZ.L") == "XYZ.L"  # explicit ticker outside the catalog
    with pytest.raises(DataUnavailableError, match="commodity"):
        sym("URANIUM", AssetClass.COMMODITY)


def test_prices_use_adjusted_closes_and_drop_missing_values() -> None:
    files = {
        "AAPL": payload([100.0, None, 102.0], adj=[99.0, None, 101.0]),
        "BTC-USD": payload([40000.0, 41000.0, 42000.0]),  # no adjclose block: falls back to close
    }
    yahoo, _ = provider(files)
    aapl = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
    btc = Asset(symbol="BTC", asset_class=AssetClass.CRYPTO)
    prices = yahoo.get_prices([aapl, btc], START, END)
    assert list(prices.columns) == ["AAPL", "BTC"]
    assert prices["AAPL"].tolist() == [99.0, 99.0, 101.0]  # the gap is forward-filled
    assert prices["BTC"].tolist() == [40000.0, 41000.0, 42000.0]
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert prices.index.is_monotonic_increasing


def test_request_covers_the_end_date_and_asks_for_daily_bars() -> None:
    yahoo, urls = provider({"AAPL": payload([1.0, 2.0, 3.0])})
    yahoo.get_prices([Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)], START, END)
    assert "interval=1d" in urls[0]
    assert "period1=1704067200" in urls[0]  # 2024-01-01 00:00 UTC
    assert "period2=1704931200" in urls[0]  # 2024-01-11 00:00 UTC: end date is inclusive


@pytest.mark.parametrize(
    "body",
    [
        "not json",
        "{}",
        json.dumps({"chart": {"result": None, "error": None}}),
        json.dumps({"chart": {"result": [], "error": None}}),
        json.dumps(
            {"chart": {"result": [{"timestamp": [], "indicators": {"quote": [{"close": []}]}}]}}
        ),
        json.dumps({"chart": {"result": None, "error": {"description": "No data found"}}}),
    ],
)
def test_bad_payloads_raise_data_unavailable(body: str) -> None:
    yahoo, _ = provider({"AAPL": body})
    with pytest.raises(DataUnavailableError):
        yahoo.get_prices([Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)], START, END)


def test_unknown_symbol_error_propagates() -> None:
    yahoo, _ = provider({})
    with pytest.raises(DataUnavailableError, match="404"):
        yahoo.get_prices([Asset(symbol="ZZZZ", asset_class=AssetClass.EQUITY)], START, END)


def test_fx_direct_pair_and_inverse_fallback() -> None:
    yahoo, urls = provider(
        {
            "EURUSD=X": payload([1.10, 1.20, 1.25]),
            "USDGBP=X": payload([0.50, 0.25, 0.20]),  # only the reverse pair exists
        }
    )
    rates = YahooFx(yahoo).get_rates(["EUR", "GBP"], "USD", START, END)
    assert rates["EUR"].tolist() == pytest.approx([1.10, 1.20, 1.25])
    assert rates["GBP"].tolist() == pytest.approx([2.0, 4.0, 5.0])
    assert any("GBPUSD=X" in u for u in urls)  # the direct pair was tried first


def test_fx_missing_pair_raises() -> None:
    yahoo, _ = provider({})
    with pytest.raises(DataUnavailableError):
        YahooFx(yahoo).get_rates(["EUR"], "USD", START, END)


def test_stooq_javascript_challenge_gives_an_actionable_message() -> None:
    html = "<!DOCTYPE html><html><body>This site requires JavaScript</body></html>"
    stooq = StooqProvider(fetcher=lambda url: html)
    with pytest.raises(DataUnavailableError, match="--provider yahoo"):
        stooq.get_prices([Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)], START, END)
