from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.risk import (
    Method,
    historical_var_cvar,
    parametric_var_cvar,
    risk_contributions,
)
from portfolio_risk.risk.covariance import covariance_matrix


def _returns(seed: int = 0, n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    a = rng.normal(0, 0.01, n)
    b = 0.6 * a + rng.normal(0, 0.008, n)
    c = rng.normal(0, 0.02, n)
    return pd.DataFrame({"A": a, "B": b, "C": c})


EXPOSURES = pd.Series({"A": 5000.0, "B": 3000.0, "C": 2000.0})


@pytest.mark.parametrize("horizon", [1, 10])
@pytest.mark.parametrize("confidence", [0.95, 0.99])
def test_parametric_components_add_up_to_the_portfolio_figures(
    confidence: float, horizon: int
) -> None:
    returns = _returns()
    result = risk_contributions(EXPOSURES, returns, confidence=confidence, horizon=horizon)
    var, cvar = parametric_var_cvar(EXPOSURES, covariance_matrix(returns), confidence, horizon)
    assert result.var == pytest.approx(var)
    assert result.cvar == pytest.approx(cvar)
    assert result.var_share.sum() == pytest.approx(1.0)


def test_marginal_var_matches_a_finite_difference() -> None:
    returns = _returns(1)
    cov = covariance_matrix(returns)
    result = risk_contributions(EXPOSURES, returns, confidence=0.99)
    for symbol in EXPOSURES.index:
        bumped = EXPOSURES.copy()
        bumped[symbol] += 1.0
        change = (
            parametric_var_cvar(bumped, cov, 0.99)[0] - parametric_var_cvar(EXPOSURES, cov, 0.99)[0]
        )
        assert result.marginal_var[symbol] == pytest.approx(change, rel=1e-3)


def test_uncorrelated_equal_positions_share_risk_equally() -> None:
    rng = np.random.default_rng(2)
    x = rng.standard_normal(4000)
    y = rng.standard_normal(4000)
    x = (x - x.mean()) / x.std(ddof=1)
    y = y - y.mean()
    y = y - x * (x @ y) / (x @ x)  # exactly uncorrelated with x
    y = y / y.std(ddof=1)
    frame = pd.DataFrame({"X": x * 0.01, "Y": y * 0.01})
    result = risk_contributions(pd.Series({"X": 1000.0, "Y": 1000.0}), frame, confidence=0.95)
    assert result.var_share["X"] == pytest.approx(0.5, abs=1e-9)


def test_a_hedge_has_a_negative_component() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(0, 0.01, 1000)
    hedge = -0.8 * a + rng.normal(0, 0.002, 1000)
    frame = pd.DataFrame({"A": a, "H": hedge})
    result = risk_contributions(pd.Series({"A": 10_000.0, "H": 3_000.0}), frame)
    assert result.component_var["H"] < 0
    assert result.component_var["A"] > result.var  # the hedge reduces A's stand-alone share
    assert result.marginal_var["H"] < 0


def test_historical_components_add_up_to_the_historical_figures() -> None:
    returns = _returns(4)
    for horizon in (1, 5):
        result = risk_contributions(
            EXPOSURES, returns, method=Method.HISTORICAL, confidence=0.95, horizon=horizon
        )
        var, cvar = historical_var_cvar(EXPOSURES, returns, 0.95, horizon)
        assert result.var == pytest.approx(var)
        assert result.cvar == pytest.approx(cvar)


def test_zero_exposure_and_zero_volatility_are_handled() -> None:
    returns = _returns(5)
    zero = risk_contributions(EXPOSURES * 0.0, returns)
    assert zero.var == 0.0
    assert (zero.var_share == 0.0).all()
    partial = risk_contributions(pd.Series({"A": 1000.0, "B": 0.0}), returns)
    assert partial.component_var["B"] == pytest.approx(0.0)
    assert partial.marginal_var["B"] == 0.0  # no exposure -> defined as zero, not NaN


def test_attribution_validates_input() -> None:
    returns = _returns()
    with pytest.raises(ValueError, match="parametric and historical"):
        risk_contributions(EXPOSURES, returns, method=Method.FHS)
    with pytest.raises(ValueError, match="confidence"):
        risk_contributions(EXPOSURES, returns, confidence=1.0)
    with pytest.raises(ValueError, match="horizon"):
        risk_contributions(EXPOSURES, returns, horizon=0)
