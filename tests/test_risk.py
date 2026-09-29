from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from portfolio_risk.data import SyntheticProvider
from portfolio_risk.models import Asset, AssetClass, CashBalance, Portfolio, Position
from portfolio_risk.risk import (
    Method,
    analyze_risk,
    correlation_matrix,
    covariance_matrix,
    historical_var_cvar,
    horizon_returns,
    parametric_var_cvar,
)

Z99 = float(norm.ppf(0.99))
AAPL = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
BTC = Asset(symbol="BTC", asset_class=AssetClass.CRYPTO)


def frame(**cols: list[float]) -> pd.DataFrame:
    return pd.DataFrame(cols)


def expo(**values: float) -> pd.Series[float]:
    return pd.Series(values, dtype=float)


def cov1(variance: float) -> pd.DataFrame:
    return pd.DataFrame([[variance]], index=["A"], columns=["A"])


# Exactly orthogonal, zero-mean, equal-variance columns (sample correlation == 0).
ORTHO = frame(A=[0.01, -0.01, 0.01, -0.01], B=[0.01, 0.01, -0.01, -0.01])
# Perfectly correlated pair.
CORR1 = frame(A=[0.01, -0.02, 0.03, -0.01], B=[0.01, -0.02, 0.03, -0.01])


def test_covariance_and_correlation() -> None:
    cov = covariance_matrix(ORTHO)
    assert cov.loc["A", "B"] == pytest.approx(0.0)
    assert cov.loc["A", "A"] == pytest.approx(4 * 0.01**2 / 3)
    annual = covariance_matrix(ORTHO, annualize=True)
    np.testing.assert_allclose(annual.to_numpy(), cov.to_numpy() * 252)
    corr = correlation_matrix(CORR1)
    assert corr.loc["A", "B"] == pytest.approx(1.0)
    assert np.allclose(np.diag(corr.to_numpy()), 1.0)


def test_covariance_rejects_bad_input() -> None:
    with pytest.raises(ValueError, match="two"):
        covariance_matrix(ORTHO.iloc[:1])
    with pytest.raises(ValueError, match="NaN"):
        covariance_matrix(pd.DataFrame({"A": [0.1, np.nan, 0.2]}))


def test_parametric_single_asset_known_values() -> None:
    var, cvar = parametric_var_cvar(expo(A=1_000_000), cov1(0.01**2), 0.99)
    assert var == pytest.approx(Z99 * 0.01 * 1e6)
    assert cvar == pytest.approx(0.01 * norm.pdf(Z99) / 0.01 * 1e6)
    assert cvar > var


def test_parametric_horizon_scales_with_sqrt_time() -> None:
    v1, c1 = parametric_var_cvar(expo(A=1000), cov1(1e-4), 0.95, 1)
    v10, c10 = parametric_var_cvar(expo(A=1000), cov1(1e-4), 0.95, 10)
    assert v10 == pytest.approx(v1 * np.sqrt(10))
    assert c10 == pytest.approx(c1 * np.sqrt(10))


def test_parametric_perfect_correlation_has_no_diversification() -> None:
    cov = covariance_matrix(CORR1)
    e = expo(A=500, B=500)
    port, _ = parametric_var_cvar(e, cov, 0.95)
    alone = sum(parametric_var_cvar(e[[s]], cov, 0.95)[0] for s in e.index)
    assert port == pytest.approx(alone)


def test_parametric_zero_correlation_diversification() -> None:
    cov = covariance_matrix(ORTHO)
    e = expo(A=1000, B=1000)
    port, _ = parametric_var_cvar(e, cov, 0.95)
    single, _ = parametric_var_cvar(e[["A"]], cov, 0.95)
    assert port == pytest.approx(np.sqrt(2) * single)
    assert 1 - port / (2 * single) == pytest.approx(1 - 1 / np.sqrt(2))


def test_parametric_offsetting_positions_cancel() -> None:
    var, cvar = parametric_var_cvar(expo(A=1000, B=-1000), covariance_matrix(CORR1), 0.99)
    assert var == pytest.approx(0.0, abs=1e-9)
    assert cvar == pytest.approx(0.0, abs=1e-9)


def test_historical_known_quantile_and_tail() -> None:
    # P&L scenarios are exactly -1, -2, ..., -100 for exposure 1000.
    rets = frame(A=[-k / 1000 for k in range(1, 101)])
    var, cvar = historical_var_cvar(expo(A=1000), rets, 0.95)
    assert var == pytest.approx(95.05)  # 5% linear-interpolated quantile
    assert cvar == pytest.approx(98.0)  # mean of the five worst: 96..100


def test_historical_horizon_compounds_returns() -> None:
    rets = frame(A=[-0.01] * 30)
    var, cvar = historical_var_cvar(expo(A=1000), rets, 0.99, 10)
    expected = 1000 * (1 - 0.99**10)
    assert var == pytest.approx(expected)
    assert cvar == pytest.approx(expected)
    assert len(horizon_returns(rets, 10)) == 21


def test_historical_perfect_correlation_has_no_diversification() -> None:
    e = expo(A=1000, B=1000)
    port, _ = historical_var_cvar(e, CORR1, 0.75)
    alone = sum(historical_var_cvar(e[[s]], CORR1, 0.75)[0] for s in e.index)
    assert port == pytest.approx(alone)


def test_historical_uncorrelated_pair_worst_case() -> None:
    rets = pd.concat([ORTHO] * 25, ignore_index=True)
    var, cvar = historical_var_cvar(expo(A=1000, B=1000), rets, 0.99)
    # Worst scenario (both assets -1%) is 25% of observations: VaR = CVaR = 20.
    assert var == pytest.approx(20.0)
    assert cvar == pytest.approx(20.0)


@pytest.mark.parametrize(("confidence", "horizon"), [(0.4, 1), (1.0, 1), (0.95, 0)])
def test_invalid_parameters(confidence: float, horizon: int) -> None:
    with pytest.raises(ValueError, match="must be"):
        parametric_var_cvar(expo(A=1), cov1(1.0), confidence, horizon)
    with pytest.raises(ValueError, match="must be"):
        historical_var_cvar(expo(A=1), frame(A=[0.1, -0.1]), confidence, horizon)


def test_analyze_risk_on_synthetic_portfolio() -> None:
    pf = Portfolio(
        positions=(
            Position(asset=AAPL, quantity=100, broker="a"),
            Position(asset=BTC, quantity=5, broker="b"),
        ),
    )
    prices = SyntheticProvider(seed=3).get_prices(pf.assets, date(2021, 1, 1), date(2024, 1, 1))
    for method in Method:
        r95 = analyze_risk(pf, prices, method=method, confidence=0.95)
        r99 = analyze_risk(pf, prices, method=method, confidence=0.99)
        r10 = analyze_risk(pf, prices, method=method, confidence=0.95, horizon=10)
        assert 0 < r95.var < r95.cvar
        assert r99.var > r95.var
        assert r10.var > r95.var
        assert r95.diversification_benefit > 0
        assert 0 < r95.diversification_ratio < 1
        assert set(r95.standalone_var) == {"AAPL", "BTC"}
    # Parametric VaR scales exactly with sqrt(horizon).
    p1 = analyze_risk(pf, prices, method=Method.PARAMETRIC)
    p10 = analyze_risk(pf, prices, method=Method.PARAMETRIC, horizon=10)
    assert p10.var == pytest.approx(p1.var * np.sqrt(10))


def test_analyze_risk_cash_adds_value_not_risk() -> None:
    base = Portfolio(positions=(Position(asset=AAPL, quantity=10),))
    with_cash = Portfolio(positions=base.positions, cash=(CashBalance(amount=1_000_000),))
    prices = SyntheticProvider().get_prices(base.assets, date(2022, 1, 1), date(2024, 1, 1))
    a = analyze_risk(base, prices, method=Method.PARAMETRIC)
    b = analyze_risk(with_cash, prices, method=Method.PARAMETRIC)
    assert a.var == pytest.approx(b.var)
    assert b.portfolio_value == pytest.approx(a.portfolio_value + 1_000_000)


def test_analyze_risk_requires_enough_history() -> None:
    pf = Portfolio(positions=(Position(asset=AAPL, quantity=1),))
    prices = SyntheticProvider().get_prices(pf.assets, date(2024, 1, 1), date(2024, 1, 10))
    with pytest.raises(ValueError, match="observations"):
        analyze_risk(pf, prices, method=Method.HISTORICAL)
