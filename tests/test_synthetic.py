from datetime import date

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.data import GBMParams, SyntheticProvider, simulate_gbm
from portfolio_risk.models import Asset, AssetClass

START, END = date(2020, 1, 1), date(2024, 12, 31)
AAPL = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
BTC = Asset(symbol="BTC", asset_class=AssetClass.CRYPTO)
GOLD = Asset(symbol="GC", asset_class=AssetClass.COMMODITY)


def test_shape_and_positive_prices() -> None:
    prices = SyntheticProvider().get_prices([AAPL, BTC], START, END)
    assert list(prices.columns) == ["AAPL", "BTC"]
    assert prices.index.is_monotonic_increasing
    assert (prices > 0).all().all()
    assert not prices.isna().any().any()


def test_deterministic_and_seed_sensitive() -> None:
    a = SyntheticProvider(seed=1).get_prices([AAPL], START, END)
    b = SyntheticProvider(seed=1).get_prices([AAPL], START, END)
    c = SyntheticProvider(seed=2).get_prices([AAPL], START, END)
    pd.testing.assert_frame_equal(a, b)
    assert not a.equals(c)


def test_asset_path_independent_of_other_assets() -> None:
    alone = SyntheticProvider().get_prices([AAPL], START, END)["AAPL"]
    together = SyntheticProvider().get_prices([BTC, AAPL], START, END)["AAPL"]
    pd.testing.assert_series_equal(alone, together)


def test_volatility_and_correlation_realistic() -> None:
    provider = SyntheticProvider(seed=7, correlation=0.5)
    rets = provider.get_returns([AAPL, BTC, GOLD], date(2010, 1, 1), date(2030, 1, 1), log=True)
    ann_vol = rets.std() * np.sqrt(252)
    assert ann_vol["AAPL"] == pytest.approx(0.22, abs=0.03)
    assert ann_vol["BTC"] == pytest.approx(0.75, abs=0.08)
    assert rets.corr().loc["AAPL", "BTC"] == pytest.approx(0.5, abs=0.06)


def test_zero_vol_gbm_is_deterministic_growth() -> None:
    path = simulate_gbm(GBMParams(mu=0.1, sigma=0.0, start_price=100.0), np.zeros(252))
    assert path[0] == 100.0
    assert path[-1] == pytest.approx(100.0 * np.exp(0.1))


def test_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="before"):
        SyntheticProvider().get_prices([AAPL], END, START)
    with pytest.raises(ValueError, match="correlation"):
        SyntheticProvider(correlation=1.0)
