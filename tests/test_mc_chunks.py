from __future__ import annotations

from datetime import date

import pytest

from portfolio_risk.data import SyntheticProvider
from portfolio_risk.models import Asset, AssetClass, Portfolio, Position
from portfolio_risk.risk import monte_carlo, run_monte_carlo

AAPL = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
BTC = Asset(symbol="BTC", asset_class=AssetClass.CRYPTO)


def _run(df: float | None) -> monte_carlo.MonteCarloReport:
    pf = Portfolio(positions=(Position(asset=AAPL, quantity=10), Position(asset=BTC, quantity=1)))
    prices = SyntheticProvider(seed=2).get_prices(pf.assets, date(2022, 1, 1), date(2023, 6, 1))
    return run_monte_carlo(pf, prices, seed=4, n_simulations=900, days=20, df=df)


def test_batching_does_not_change_normal_results(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(monte_carlo, "CHUNK_PATHS", 900)
    single = _run(None)
    monkeypatch.setattr(monte_carlo, "CHUNK_PATHS", 250)  # four batches
    chunked = _run(None)
    assert chunked.n_simulations == 900
    assert chunked.final_values.shape == (900,)
    # A normal stream is consumed identically however it is split into batches.
    assert chunked.final_values == pytest.approx(single.final_values)
    assert chunked.var == pytest.approx(single.var)
    assert chunked.max_drawdowns == pytest.approx(single.max_drawdowns)
    assert chunked.prob_ruin == single.prob_ruin


def test_student_t_batches_are_deterministic(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(monte_carlo, "CHUNK_PATHS", 250)
    first, second = _run(5.0), _run(5.0)
    assert first.var == second.var
    assert first.final_values.shape == (900,)
