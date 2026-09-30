from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from portfolio_risk.data import SyntheticProvider
from portfolio_risk.models import Asset, AssetClass, Portfolio, Position
from portfolio_risk.risk import (
    Method,
    Zone,
    basel_zone,
    christoffersen_independence,
    conditional_coverage,
    estimate_var_cvar,
    kupiec_pof,
    rolling_var,
    run_backtest,
)

AAPL = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)


def _portfolio() -> Portfolio:
    return Portfolio(
        name="bt",
        base_currency="USD",
        positions=[Position(asset=AAPL, quantity=100.0, broker="b")],
    )


def _flags(pattern: list[int]) -> np.ndarray:
    return np.array(pattern, dtype=np.bool_)


def _prices_from_returns(returns: np.ndarray) -> pd.DataFrame:
    index = pd.bdate_range("2015-01-01", periods=len(returns) + 1)
    return pd.DataFrame({"AAPL": 100.0 * np.cumprod(np.r_[1.0, 1.0 + returns])}, index=index)


def _student_t_returns(n: int, df: float, daily_vol: float, seed: int) -> np.ndarray:
    raw = np.random.default_rng(seed).standard_t(df, n)
    return np.asarray(raw * daily_vol / np.sqrt(df / (df - 2.0)))


# --- statistical tests ---------------------------------------------------------------------


def test_kupiec_matches_hand_computed_likelihood_ratio() -> None:
    n, x, p = 250, 5, 0.01
    v = np.zeros(n, dtype=np.bool_)
    v[:x] = True
    q = x / n
    expected = -2 * (
        (n - x) * np.log(1 - p) + x * np.log(p) - (n - x) * np.log(1 - q) - x * np.log(q)
    )
    result = kupiec_pof(v, 0.99)
    assert result.statistic == pytest.approx(expected)
    assert result.p_value == pytest.approx(chi2.sf(expected, 1))


def test_kupiec_accepts_expected_rate_and_rejects_excess() -> None:
    at_expected = np.zeros(1000, dtype=np.bool_)
    at_expected[:10] = True
    ok = kupiec_pof(at_expected, 0.99)
    assert ok.statistic == pytest.approx(0.0, abs=1e-9)
    assert not ok.rejects()

    too_many = np.zeros(1000, dtype=np.bool_)
    too_many[:30] = True
    assert kupiec_pof(too_many, 0.99).rejects(0.01)


def test_kupiec_handles_zero_and_all_violations() -> None:
    none = kupiec_pof(np.zeros(250, dtype=np.bool_), 0.99)
    assert np.isfinite(none.statistic)
    assert none.statistic == pytest.approx(-2 * 250 * np.log(0.99))  # closed form for x = 0
    all_ = kupiec_pof(np.ones(50, dtype=np.bool_), 0.99)
    assert np.isfinite(all_.statistic)
    assert all_.rejects(0.001)


def test_independence_rejects_clustered_and_accepts_spread_violations() -> None:
    clustered = np.zeros(500, dtype=np.bool_)
    clustered[100:110] = True
    clustered[300:310] = True
    assert christoffersen_independence(clustered).rejects(0.01)

    spread = np.zeros(500, dtype=np.bool_)
    spread[::50] = True  # never two in a row
    assert not christoffersen_independence(spread).rejects()


def test_independence_is_uninformative_without_violations() -> None:
    result = christoffersen_independence(np.zeros(100, dtype=np.bool_))
    assert result.statistic == 0.0
    assert result.p_value == 1.0


def test_conditional_coverage_combines_both_statistics() -> None:
    v = np.zeros(400, dtype=np.bool_)
    v[50:56] = True
    combined = conditional_coverage(v, 0.99)
    parts = kupiec_pof(v, 0.99).statistic + christoffersen_independence(v).statistic
    assert combined.statistic == pytest.approx(parts)
    assert combined.p_value == pytest.approx(chi2.sf(parts, 2))


def test_coverage_input_validation() -> None:
    with pytest.raises(ValueError, match="empty"):
        kupiec_pof(np.zeros(0, dtype=np.bool_), 0.99)
    with pytest.raises(ValueError, match="confidence"):
        kupiec_pof(_flags([0, 1]), 1.0)
    with pytest.raises(ValueError, match="two observations"):
        christoffersen_independence(_flags([1]))


@pytest.mark.parametrize(
    ("violations", "zone"),
    [
        (0, Zone.GREEN),
        (4, Zone.GREEN),
        (5, Zone.YELLOW),
        (9, Zone.YELLOW),
        (10, Zone.RED),
        (25, Zone.RED),
    ],
)
def test_basel_zones_match_the_regulatory_table(violations: int, zone: Zone) -> None:
    assert basel_zone(violations, 250, 0.99) is zone


def test_basel_zone_validates_input() -> None:
    with pytest.raises(ValueError, match="n_obs"):
        basel_zone(11, 10, 0.99)


# --- rolling backtest ---------------------------------------------------------------------


def test_rolling_var_uses_only_past_data() -> None:
    returns = pd.DataFrame(
        {"AAPL": _student_t_returns(400, 30.0, 0.01, seed=1)},
        index=pd.bdate_range("2020-01-01", periods=400),
    )
    exposures = pd.Series({"AAPL": 1000.0})
    base = rolling_var(exposures, returns, method=Method.HISTORICAL, confidence=0.99, window=100)
    assert base.shape == (300,)

    shocked = returns.copy()
    shocked.iloc[250:] = 0.5  # rewrite the future relative to forecast #100 (window ends at 199)
    changed = rolling_var(exposures, shocked, method=Method.HISTORICAL, confidence=0.99, window=100)
    assert np.array_equal(base[:150], changed[:150])
    assert not np.array_equal(base[150:], changed[150:])


def test_backtest_shapes_and_alignment() -> None:
    prices = _prices_from_returns(_student_t_returns(600, 30.0, 0.01, seed=2))
    result = run_backtest(_portfolio(), prices, method=Method.PARAMETRIC, window=250)
    assert result.n_obs == 600 - 250
    assert result.var.shape == result.pnl.shape == (result.n_obs,)
    assert len(result.dates) == result.n_obs
    assert result.dates[-1] == prices.index[-1]
    assert result.expected_violations == pytest.approx(result.n_obs * 0.01)
    assert result.violation_rate == pytest.approx(result.n_violations / result.n_obs)


def test_backtest_violation_definition_is_loss_exceeding_var() -> None:
    prices = _prices_from_returns(_student_t_returns(400, 30.0, 0.01, seed=3))
    result = run_backtest(_portfolio(), prices, method=Method.HISTORICAL, window=100)
    manual = (-result.pnl) > result.var
    assert np.array_equal(result.violations, manual)


def test_gaussian_data_passes_parametric_backtest() -> None:
    prices = _prices_from_returns(np.random.default_rng(7).normal(0.0, 0.01, 1500))
    result = run_backtest(_portfolio(), prices, method=Method.PARAMETRIC, confidence=0.99)
    assert not result.kupiec.rejects(0.01)
    assert not result.conditional_coverage.rejects(0.01)
    assert result.zone is not Zone.RED


def test_fat_tails_break_the_gaussian_model() -> None:
    prices = _prices_from_returns(_student_t_returns(6000, 3.0, 0.01, seed=11))
    result = run_backtest(_portfolio(), prices, method=Method.PARAMETRIC, confidence=0.99)
    assert result.n_violations > result.expected_violations
    assert result.kupiec.rejects(0.05)


def test_volatility_clustering_produces_dependent_violations() -> None:
    rng = np.random.default_rng(5)
    blocks = [rng.normal(0.0, vol, 60) for vol in [0.005, 0.03] * 25]
    prices = _prices_from_returns(np.concatenate(blocks))
    result = run_backtest(_portfolio(), prices, method=Method.PARAMETRIC, confidence=0.95)
    assert result.independence.rejects(0.05)


def test_backtest_on_synthetic_provider_runs_end_to_end() -> None:
    prices = SyntheticProvider(seed=4).get_prices([AAPL], date(2020, 1, 1), date(2024, 1, 1))
    for method in Method:
        result = run_backtest(_portfolio(), prices, method=method)
        assert result.n_obs > 500
        assert 0.0 <= result.kupiec.p_value <= 1.0


def test_backtest_validates_inputs() -> None:
    prices = _prices_from_returns(_student_t_returns(100, 30.0, 0.01, seed=6))
    with pytest.raises(ValueError, match="more than 250"):
        run_backtest(_portfolio(), prices, method=Method.PARAMETRIC)
    with pytest.raises(ValueError, match="window"):
        run_backtest(_portfolio(), prices, method=Method.PARAMETRIC, window=5)
    with pytest.raises(ValueError, match="confidence"):
        run_backtest(_portfolio(), prices, method=Method.PARAMETRIC, confidence=1.0, window=50)


# --- model comparison ----------------------------------------------------------------------


def _garch_returns(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    var = 1e-4
    for i in range(n):
        out[i] = np.sqrt(var) * rng.standard_normal()
        var = 2e-6 + 0.1 * out[i] ** 2 + 0.88 * var
    return out


def test_ewma_beats_parametric_under_volatility_clustering() -> None:
    prices = _prices_from_returns(_garch_returns(4000, seed=21))
    parametric = run_backtest(_portfolio(), prices, method=Method.PARAMETRIC)
    ewma = run_backtest(_portfolio(), prices, method=Method.EWMA)
    assert parametric.conditional_coverage.rejects(0.01)
    assert not ewma.conditional_coverage.rejects(0.05)
    assert ewma.independence.p_value > parametric.independence.p_value


def test_tail_aware_models_violate_less_than_normal_on_fat_tails() -> None:
    prices = _prices_from_returns(_student_t_returns(3500, 3.0, 0.01, seed=11))
    counts = {
        m: run_backtest(_portfolio(), prices, method=m).n_violations
        for m in (Method.PARAMETRIC, Method.STUDENT_T, Method.FHS, Method.CORNISH_FISHER)
    }
    assert counts[Method.STUDENT_T] < counts[Method.PARAMETRIC]
    assert counts[Method.FHS] < counts[Method.PARAMETRIC]
    assert counts[Method.CORNISH_FISHER] < counts[Method.PARAMETRIC]


def test_every_method_backtests_on_synthetic_prices() -> None:
    prices = SyntheticProvider(seed=8).get_prices([AAPL], date(2021, 1, 1), date(2024, 1, 1))
    for method in Method:
        result = run_backtest(_portfolio(), prices, method=method, window=120)
        assert result.n_obs > 300
        assert np.isfinite(result.var).all()


@pytest.mark.parametrize("method", list(Method))
def test_fast_rolling_forecast_matches_the_full_estimator(method: Method) -> None:
    returns = pd.DataFrame(
        {
            "A": _student_t_returns(160, 5.0, 0.01, seed=31),
            "B": _student_t_returns(160, 8.0, 0.02, seed=32),
        },
        index=pd.bdate_range("2020-01-01", periods=160),
    )
    exposures = pd.Series({"A": 700.0, "B": 300.0})
    fast = rolling_var(exposures, returns, method=method, confidence=0.975, window=100)
    for i in (0, 17, 59):
        full = estimate_var_cvar(method, exposures, returns.iloc[i : i + 100], 0.975)[0]
        assert fast[i] == pytest.approx(full, rel=1e-9)
