from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from portfolio_risk.data import (
    DataUnavailableError,
    StooqFx,
    StooqProvider,
    SyntheticFx,
    SyntheticProvider,
    convert_to_base,
    foreign_currencies,
)
from portfolio_risk.models import Asset, AssetClass, CashBalance, Portfolio, Position

START, END = date(2024, 1, 1), date(2024, 3, 1)
SAP = Asset(symbol="SAP", asset_class=AssetClass.EQUITY, currency="EUR")
AAPL = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)


def test_synthetic_rates_are_reproducible_and_end_at_the_anchor() -> None:
    a = SyntheticFx(seed=1).get_rates(["EUR", "GBP"], "USD", START, END)
    b = SyntheticFx(seed=1).get_rates(["EUR", "GBP"], "USD", START, END)
    pd.testing.assert_frame_equal(a, b)
    assert a["EUR"].iloc[-1] == pytest.approx(1.08)  # anchored at the end date
    assert not a.equals(SyntheticFx(seed=2).get_rates(["EUR", "GBP"], "USD", START, END))


def test_synthetic_crosses_are_consistent_and_the_base_is_one() -> None:
    fx = SyntheticFx(seed=3)
    usd = fx.get_rates(["EUR", "GBP", "USD"], "USD", START, END)
    in_gbp = fx.get_rates(["EUR", "GBP"], "GBP", START, END)
    assert (usd["USD"] == 1.0).all()
    assert in_gbp["GBP"].to_numpy() == pytest.approx(1.0)
    assert in_gbp["EUR"].to_numpy() == pytest.approx((usd["EUR"] / usd["GBP"]).to_numpy())


def test_unknown_synthetic_currency_raises() -> None:
    with pytest.raises(DataUnavailableError, match="XYZ"):
        SyntheticFx().get_rates(["XYZ"], "USD", START, END)


def test_foreign_currencies_lists_positions_and_cash() -> None:
    pf = Portfolio(
        positions=(Position(asset=SAP, quantity=1), Position(asset=AAPL, quantity=1)),
        cash=(CashBalance(amount=5, currency="GBP"), CashBalance(amount=5)),
    )
    assert foreign_currencies(pf) == ["EUR", "GBP"]
    assert foreign_currencies(Portfolio(positions=(Position(asset=AAPL, quantity=1),))) == []


def _portfolio() -> Portfolio:
    return Portfolio(
        positions=(
            Position(asset=SAP, quantity=10, broker="xetra"),
            Position(asset=AAPL, quantity=2),
        ),
        cash=(
            CashBalance(broker="xetra", amount=1000, currency="EUR"),
            CashBalance(broker="xetra", amount=500, currency="USD"),
            CashBalance(broker="ibkr", amount=200, currency="USD"),
        ),
    )


def test_prices_and_cash_are_converted_to_the_base_currency() -> None:
    pf = _portfolio()
    prices = SyntheticProvider(seed=5).get_prices(pf.assets, START, END)
    fx = SyntheticFx(seed=5)
    converted, in_usd = convert_to_base(pf, prices, fx, START, END)
    rates = fx.get_rates(["EUR"], "USD", START, END)["EUR"]

    assert in_usd["SAP"].to_numpy() == pytest.approx((prices["SAP"] * rates).to_numpy())
    assert in_usd["AAPL"].equals(prices["AAPL"])  # base-currency asset is untouched
    assert (prices["SAP"] != in_usd["SAP"]).all()  # ... and the input frame is not mutated
    assert {c.currency for c in converted.cash} == {"USD"}
    by_broker = {c.broker: c.amount for c in converted.cash}
    assert by_broker["xetra"] == pytest.approx(500 + 1000 * rates.iloc[-1])
    assert by_broker["ibkr"] == 200
    latest = {str(k): float(v) for k, v in in_usd.iloc[-1].items()}
    assert converted.total_cash > 0
    assert converted.total_value(latest) == pytest.approx(
        10 * in_usd["SAP"].iloc[-1] + 2 * in_usd["AAPL"].iloc[-1] + converted.total_cash
    )


def test_fx_moves_enter_the_returns() -> None:
    pf = Portfolio(positions=(Position(asset=SAP, quantity=1),))
    prices = pd.DataFrame(
        {"SAP": [100.0, 100.0, 100.0]}, index=pd.bdate_range("2024-01-01", periods=3)
    )

    class Fixed(SyntheticFx):
        def get_rates(self, currencies: object, base: str, start: date, end: date) -> pd.DataFrame:
            return pd.DataFrame({"EUR": [1.0, 1.1, 1.21]}, index=prices.index)

    _, in_usd = convert_to_base(pf, prices, Fixed(), START, END)
    assert in_usd["SAP"].pct_change().dropna().tolist() == pytest.approx([0.1, 0.1])


def test_nothing_to_convert_returns_the_inputs() -> None:
    pf = Portfolio(positions=(Position(asset=AAPL, quantity=1),))
    prices = SyntheticProvider().get_prices(pf.assets, START, END)
    same_pf, same_prices = convert_to_base(pf, prices, SyntheticFx(), START, END)
    assert same_pf is pf
    assert same_prices is prices


def test_rates_that_miss_part_of_the_history_are_rejected() -> None:
    pf = Portfolio(positions=(Position(asset=SAP, quantity=1),))
    prices = pd.DataFrame({"SAP": [1.0, 2.0]}, index=pd.bdate_range("2024-01-01", periods=2))

    class Nan(SyntheticFx):
        def get_rates(self, currencies: object, base: str, start: date, end: date) -> pd.DataFrame:
            return pd.DataFrame({"EUR": [float("nan")] * 2}, index=prices.index)

    with pytest.raises(DataUnavailableError, match="cover"):
        convert_to_base(pf, prices, Nan(), START, END)


HEADER = "Date,Open,High,Low,Close,Volume\n"


def _stooq_fx(files: dict[str, str]) -> StooqFx:
    def fetch(url: str) -> str:
        for symbol, body in files.items():
            if f"s={symbol}&" in url:
                return body
        return "No data"

    return StooqFx(StooqProvider(fetcher=fetch))


def test_stooq_fx_uses_the_direct_pair_and_inverts_when_only_the_reverse_exists() -> None:
    direct = HEADER + "2024-01-02,1,1,1,1.10,0\n2024-01-03,1,1,1,1.20,0\n"
    inverse = HEADER + "2024-01-02,1,1,1,0.50,0\n2024-01-03,1,1,1,0.25,0\n"
    fx = _stooq_fx({"eurusd": direct, "usdgbp": inverse})
    rates = fx.get_rates(["EUR", "GBP"], "USD", START, END)
    assert rates["EUR"].tolist() == pytest.approx([1.10, 1.20])
    assert rates["GBP"].tolist() == pytest.approx([2.0, 4.0])


def test_stooq_fx_reports_a_missing_pair() -> None:
    with pytest.raises(DataUnavailableError):
        _stooq_fx({}).get_rates(["EUR"], "USD", START, END)
