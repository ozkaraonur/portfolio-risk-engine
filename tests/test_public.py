from datetime import date

import pytest

from portfolio_risk.data import DataUnavailableError, StooqProvider
from portfolio_risk.models import Asset, AssetClass

AAPL = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
BTC = Asset(symbol="BTC", asset_class=AssetClass.CRYPTO)
START, END = date(2024, 1, 1), date(2024, 1, 10)

HEADER = "Date,Open,High,Low,Close,Volume\n"
CSV = {
    "aapl.us": HEADER + "2024-01-02,1,1,1,100,1\n2024-01-03,1,1,1,110,1\n",
    "btc.v": HEADER + "2024-01-02,1,1,1,40000,1\n2024-01-03,1,1,1,42000,1\n",
}


def fake_fetch(url: str) -> str:
    for symbol, body in CSV.items():
        if f"s={symbol}" in url:
            return body
    return "No data"


def test_symbol_mapping() -> None:
    assert StooqProvider.stooq_symbol(AAPL) == "aapl.us"
    assert StooqProvider.stooq_symbol(BTC) == "btc.v"
    override = Asset(symbol="GC", asset_class=AssetClass.COMMODITY, data_symbol="GC.F")
    assert StooqProvider.stooq_symbol(override) == "gc.f"


def test_parses_and_aligns() -> None:
    prices = StooqProvider(fetcher=fake_fetch).get_prices([AAPL, BTC], START, END)
    assert list(prices.columns) == ["AAPL", "BTC"]
    assert prices["AAPL"].iloc[-1] == 110
    returns = StooqProvider(fetcher=fake_fetch).get_returns([AAPL], START, END)
    assert returns["AAPL"].iloc[0] == pytest.approx(0.1)


def test_unavailable_symbol_raises() -> None:
    unknown = Asset(symbol="ZZZ", asset_class=AssetClass.EQUITY)
    with pytest.raises(DataUnavailableError):
        StooqProvider(fetcher=fake_fetch).get_prices([unknown], START, END)
