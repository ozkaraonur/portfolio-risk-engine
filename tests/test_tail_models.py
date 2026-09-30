from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm
from scipy.stats import t as student_t

from portfolio_risk.risk import (
    CORE_METHODS,
    CovMethod,
    Method,
    cornish_fisher_var_cvar,
    covariance_matrix,
    estimate_covariance,
    estimate_t_df,
    estimate_var_cvar,
    ewma_covariance,
    ewma_volatility,
    fhs_var_cvar,
    parametric_var_cvar,
    shrunk_covariance,
    student_t_var_cvar,
)
from portfolio_risk.risk.covariance import ewma_weights

Z99 = float(norm.ppf(0.99))
TailModel = Callable[..., tuple[float, float]]


def _t_series(n: int, df: float, seed: int, vol: float = 1.0) -> np.ndarray:
    raw = np.random.default_rng(seed).standard_t(df, n)
    return np.asarray(raw * vol / np.sqrt(df / (df - 2.0)))


# --- covariance estimators -------------------------------------------------------------------


def test_ewma_weights_favour_recent_observations() -> None:
    w = ewma_weights(50, 0.94)
    assert w.sum() == pytest.approx(1.0)
    assert bool((w[1:] > w[:-1]).all())
    assert w[-1] / w[-2] == pytest.approx(1 / 0.94)
    with pytest.raises(ValueError, match="lam"):
        ewma_weights(10, 1.0)


def test_ewma_covariance_reacts_to_recent_volatility() -> None:
    calm = np.random.default_rng(0).normal(0, 0.005, (200, 2))
    wild = np.random.default_rng(1).normal(0, 0.03, (20, 2))
    frame = pd.DataFrame(np.vstack([calm, wild]), columns=["A", "B"])
    sample = float(covariance_matrix(frame).to_numpy()[0, 0])
    assert float(ewma_covariance(frame).to_numpy()[0, 0]) > 3 * sample
    assert float(ewma_covariance(frame.iloc[::-1]).to_numpy()[0, 0]) < sample


def test_ewma_covariance_is_symmetric_and_labelled() -> None:
    frame = pd.DataFrame(np.random.default_rng(2).normal(0, 0.01, (100, 3)), columns=list("XYZ"))
    cov = ewma_covariance(frame)
    assert list(cov.columns) == list("XYZ")
    assert np.allclose(cov, cov.T)
    assert np.all(np.linalg.eigvalsh(cov.to_numpy()) > 0)


def test_shrinkage_conditions_a_wide_sample() -> None:
    frame = pd.DataFrame(np.random.default_rng(3).normal(0, 0.01, (30, 25)))
    sample = covariance_matrix(frame).to_numpy()
    shrunk = shrunk_covariance(frame).to_numpy()
    assert np.linalg.cond(shrunk) < np.linalg.cond(sample)
    assert np.linalg.eigvalsh(shrunk).min() > 0
    assert np.allclose(shrunk, shrunk.T)


def test_shrinkage_converges_to_sample_with_lots_of_data() -> None:
    frame = pd.DataFrame(np.random.default_rng(4).normal(0, 0.01, (50_000, 3)))
    ratio = shrunk_covariance(frame).to_numpy() / covariance_matrix(frame).to_numpy()
    assert np.diag(ratio) == pytest.approx(1.0, abs=0.01)


def test_estimate_covariance_dispatches_on_method() -> None:
    frame = pd.DataFrame(np.random.default_rng(5).normal(0, 0.01, (80, 2)), columns=["A", "B"])
    assert estimate_covariance(frame).equals(covariance_matrix(frame))
    assert estimate_covariance(frame, CovMethod.EWMA).equals(ewma_covariance(frame))
    assert estimate_covariance(frame, CovMethod.SHRINKAGE).equals(shrunk_covariance(frame))


# --- Student-t -------------------------------------------------------------------------------


def test_student_t_matches_closed_form_for_fixed_df() -> None:
    pnl = np.random.default_rng(6).normal(0, 10.0, 500)
    sigma = float(np.std(pnl, ddof=1))
    var, cvar = student_t_var_cvar(pnl, 0.99, df=5.0)
    q = float(student_t.ppf(0.99, 5.0))
    scale = sigma * np.sqrt(3.0 / 5.0)
    assert var == pytest.approx(scale * q)
    assert cvar == pytest.approx(scale * float(student_t.pdf(q, 5.0)) / 0.01 * (5.0 + q**2) / 4.0)
    assert cvar > var


def test_student_t_with_huge_df_approaches_normal() -> None:
    pnl = np.random.default_rng(7).normal(0, 10.0, 500)
    var, _ = student_t_var_cvar(pnl, 0.99, df=1e6)
    assert var == pytest.approx(Z99 * float(np.std(pnl, ddof=1)), rel=1e-3)


def test_student_t_has_fatter_far_tail_than_normal() -> None:
    pnl = _t_series(3000, 4.5, seed=8)
    normal_var = float(norm.ppf(0.999)) * float(np.std(pnl, ddof=1))
    assert student_t_var_cvar(pnl, 0.999)[0] > normal_var


def test_estimate_t_df_recovers_tail_thickness() -> None:
    assert 4.5 < estimate_t_df(_t_series(200_000, 6.0, seed=9)) < 9.0
    assert estimate_t_df(np.random.default_rng(10).normal(size=200_000)) >= 50.0


def test_t_scales_with_square_root_of_time() -> None:
    pnl = _t_series(400, 5.0, seed=11)
    assert student_t_var_cvar(pnl, 0.99, horizon=9)[0] == pytest.approx(
        3 * student_t_var_cvar(pnl, 0.99)[0]
    )


def test_student_t_rejects_df_at_or_below_two() -> None:
    with pytest.raises(ValueError, match="df"):
        student_t_var_cvar(_t_series(100, 5.0, seed=1), 0.99, df=2.0)


# --- Cornish-Fisher --------------------------------------------------------------------------


def test_cornish_fisher_is_normal_for_symmetric_mesokurtic_data() -> None:
    # Quantiles of a normal on a fine grid: exactly symmetric, kurtosis ~ 0.
    grid = norm.ppf((np.arange(20_000) + 0.5) / 20_000)
    var, cvar = cornish_fisher_var_cvar(grid, 0.99)
    assert var == pytest.approx(Z99, rel=0.02)
    assert cvar == pytest.approx(float(norm.pdf(Z99)) / 0.01, rel=0.03)


def test_cornish_fisher_increases_var_for_negative_skew() -> None:
    rng = np.random.default_rng(12)
    base = rng.normal(0, 1.0, 5000)
    crash_days = np.where(rng.random(5000) < 0.03, base - 4.0, base)  # P&L skewed to losses
    assert (
        cornish_fisher_var_cvar(crash_days, 0.99)[0] > cornish_fisher_var_cvar(-crash_days, 0.99)[0]
    )


def test_cornish_fisher_cvar_never_below_var() -> None:
    for seed in range(5):
        var, cvar = cornish_fisher_var_cvar(_t_series(250, 3.5, seed), 0.99)
        assert var >= 0
        assert cvar >= var


# --- filtered historical simulation ----------------------------------------------------------


def test_ewma_volatility_shape_and_start() -> None:
    series = np.random.default_rng(13).normal(0, 2.0, 100)
    vol = ewma_volatility(series)
    assert vol.shape == series.shape
    assert vol[0] == pytest.approx(float(np.std(series[:30])))
    with pytest.raises(ValueError, match="lam"):
        ewma_volatility(series, 1.5)


def test_fhs_scales_up_after_a_volatility_spike_and_down_after_calm() -> None:
    rng = np.random.default_rng(14)
    base = rng.normal(0, 1.0, 250)
    spike = np.concatenate([base[:230], rng.normal(0, 3.0, 20)])
    calm = np.concatenate([base[:230], rng.normal(0, 0.3, 20)])
    hist_spike = -float(np.quantile(spike, 0.01))
    hist_calm = -float(np.quantile(calm, 0.01))
    assert fhs_var_cvar(spike, 0.99)[0] > 1.2 * hist_spike
    assert fhs_var_cvar(calm, 0.99)[0] < 0.8 * hist_calm


def test_fhs_is_close_to_historical_in_a_stable_regime() -> None:
    pnl = np.random.default_rng(15).normal(0, 1.0, 2000)
    fhs = fhs_var_cvar(pnl, 0.95)[0]
    assert fhs == pytest.approx(-float(np.quantile(pnl, 0.05)), rel=0.15)


def test_fhs_cvar_exceeds_var_and_scales_with_horizon() -> None:
    pnl = _t_series(500, 5.0, seed=16)
    var, cvar = fhs_var_cvar(pnl, 0.975)
    assert cvar >= var
    assert fhs_var_cvar(pnl, 0.975, horizon=4)[0] == pytest.approx(2 * var)


@pytest.mark.parametrize("model", [student_t_var_cvar, cornish_fisher_var_cvar, fhs_var_cvar])
def test_tail_models_validate_inputs(model: TailModel) -> None:
    pnl = np.random.default_rng(17).normal(size=50)
    with pytest.raises(ValueError, match="confidence"):
        model(pnl, 1.0)
    with pytest.raises(ValueError, match="horizon"):
        model(pnl, 0.99, 0)
    with pytest.raises(ValueError, match="four"):
        model(pnl[:3], 0.99)


# --- dispatcher ------------------------------------------------------------------------------


def _returns() -> pd.DataFrame:
    rng = np.random.default_rng(18)
    return pd.DataFrame(
        {"A": _t_series(300, 5.0, 1, 0.01), "B": rng.normal(0, 0.02, 300)}, columns=["A", "B"]
    )


def test_every_method_returns_ordered_positive_risk() -> None:
    exposures = pd.Series({"A": 1000.0, "B": 500.0})
    for method in Method:
        var, cvar = estimate_var_cvar(method, exposures, _returns(), 0.99)
        assert var > 0, method
        assert cvar >= var, method


def test_core_methods_are_the_classic_pair() -> None:
    assert CORE_METHODS == (Method.PARAMETRIC, Method.HISTORICAL)


def test_dispatcher_matches_underlying_estimators() -> None:
    exposures = pd.Series({"A": 1000.0, "B": 500.0})
    returns = _returns()
    expected = parametric_var_cvar(exposures, covariance_matrix(returns), 0.95, 5)
    assert estimate_var_cvar(Method.PARAMETRIC, exposures, returns, 0.95, 5) == expected
    ewma = parametric_var_cvar(exposures, ewma_covariance(returns), 0.95, 5)
    assert estimate_var_cvar(Method.EWMA, exposures, returns, 0.95, 5) == ewma


def test_dispatcher_uses_only_the_requested_assets() -> None:
    returns = _returns()
    alone = estimate_var_cvar(Method.STUDENT_T, pd.Series({"A": 1000.0}), returns, 0.99)
    both = estimate_var_cvar(Method.STUDENT_T, pd.Series({"A": 1000.0, "B": 0.0}), returns, 0.99)
    assert alone[0] == pytest.approx(both[0], rel=1e-9)
