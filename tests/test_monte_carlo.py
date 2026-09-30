from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from portfolio_risk.data import SyntheticProvider
from portfolio_risk.models import Asset, AssetClass, CashBalance, Portfolio, Position
from portfolio_risk.risk import (
    Method,
    analyze_risk,
    cholesky_factor,
    max_drawdowns,
    nearest_psd,
    run_monte_carlo,
    simulate_paths,
    simulate_returns,
    summarize_paths,
)
from portfolio_risk.risk.linalg import FloatArray

AAPL = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
BTC = Asset(symbol="BTC", asset_class=AssetClass.CRYPTO)


def arr(*rows: list[float]) -> FloatArray:
    return np.array(rows, dtype=np.float64)


# ---- linear algebra ---------------------------------------------------------------


def test_cholesky_reconstructs_positive_definite() -> None:
    cov = arr([4e-4, 1e-4, 0.0], [1e-4, 9e-4, 2e-4], [0.0, 2e-4, 1e-4])
    chol = cholesky_factor(cov)
    assert chol.shape == (3, 3)
    assert np.allclose(chol, np.tril(chol))
    assert np.allclose(chol @ chol.T, cov)


def test_cholesky_handles_singular_matrix() -> None:
    # Perfectly correlated assets: rank-deficient, plain np.linalg.cholesky may fail.
    cov = arr([1e-4, 2e-4], [2e-4, 4e-4])
    chol = cholesky_factor(cov)
    assert np.isfinite(chol).all()
    assert np.allclose(chol @ chol.T, cov, atol=1e-9)


def test_cholesky_handles_indefinite_matrix() -> None:
    bad = arr([1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0])
    assert np.linalg.eigvalsh(bad).min() < 0
    chol = cholesky_factor(bad)
    assert np.allclose(np.diag(chol @ chol.T), 1.0)


def test_nearest_psd_properties() -> None:
    bad = arr([1.0, 0.9, 0.9], [0.9, 1.0, -0.9], [0.9, -0.9, 1.0])
    fixed = nearest_psd(bad)
    assert np.linalg.eigvalsh(fixed).min() > 0
    assert np.allclose(fixed, fixed.T)
    assert np.allclose(np.diag(fixed), np.diag(bad))  # variances preserved


def test_nearest_psd_leaves_valid_matrix_unchanged() -> None:
    cov = arr([4e-4, 1e-4], [1e-4, 9e-4])
    assert np.allclose(nearest_psd(cov), cov)


def test_cholesky_size_and_input_validation() -> None:
    assert cholesky_factor(arr([4.0])).tolist() == [[2.0]]
    with pytest.raises(ValueError, match="square"):
        cholesky_factor(np.ones((2, 3)))
    with pytest.raises(ValueError, match="NaN"):
        cholesky_factor(arr([1.0, np.nan], [np.nan, 1.0]))
    with pytest.raises(ValueError, match="mean"):
        simulate_returns(arr([1e-4]), np.zeros(2), 10, 1, np.random.default_rng(0))
    with pytest.raises(ValueError, match=">= 1"):
        simulate_returns(arr([1e-4]), np.zeros(1), 0, 1, np.random.default_rng(0))


# ---- simulation -------------------------------------------------------------------


def test_simulation_is_deterministic_for_seed() -> None:
    cov, mean = arr([1e-4, 0.0], [0.0, 4e-4]), np.zeros(2)
    a = simulate_paths(np.array([100.0, 100.0]), 0.0, cov, mean, 50, 10, np.random.default_rng(1))
    b = simulate_paths(np.array([100.0, 100.0]), 0.0, cov, mean, 50, 10, np.random.default_rng(1))
    c = simulate_paths(np.array([100.0, 100.0]), 0.0, cov, mean, 50, 10, np.random.default_rng(2))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.shape == (50, 11)
    assert (a[:, 0] == 200.0).all()


def test_convergence_of_mean_and_volatility() -> None:
    mu, sigma, days, n = 0.0004, 0.01, 252, 20_000
    paths = simulate_paths(
        np.array([1000.0]),
        0.0,
        arr([sigma**2]),
        np.array([mu]),
        n,
        days,
        np.random.default_rng(42),
    )
    log_growth = np.log(paths[:, -1] / 1000.0)
    # log-return ~ N((mu - sigma^2/2) T, sigma^2 T); standard error of the mean ~ 0.0011.
    assert log_growth.mean() == pytest.approx((mu - 0.5 * sigma**2) * days, abs=0.005)
    assert log_growth.std() == pytest.approx(sigma * np.sqrt(days), rel=0.02)
    # Arithmetic mean growth equals exp(mu T) by construction.
    assert paths[:, -1].mean() / 1000.0 == pytest.approx(np.exp(mu * days), rel=0.01)


def test_convergence_of_correlation_structure() -> None:
    s1, s2, rho = 0.01, 0.02, 0.6
    cov = arr([s1**2, rho * s1 * s2], [rho * s1 * s2, s2**2])
    r = simulate_returns(cov, np.zeros(2), 200_000, 1, np.random.default_rng(7))[:, 0, :]
    sample = np.cov(r.T)
    assert np.allclose(sample, cov, rtol=0.03)
    assert np.corrcoef(r[:, 0], r[:, 1]).tolist()[0][1] == pytest.approx(rho, abs=0.01)


def test_perfectly_correlated_assets_move_together() -> None:
    cov = arr([1e-4, 1e-4], [1e-4, 1e-4])
    r = simulate_returns(cov, np.zeros(2), 1000, 1, np.random.default_rng(3))[:, 0, :]
    assert np.allclose(r[:, 0], r[:, 1], atol=1e-6)


# ---- metrics ----------------------------------------------------------------------


def test_max_drawdowns_known_paths() -> None:
    paths = arr([100, 120, 90, 110], [100, 101, 102, 103], [100, 80, 100, 50])
    dd = max_drawdowns(paths)
    assert dd[0] == pytest.approx(0.25)
    assert dd[1] == 0.0
    assert dd[2] == pytest.approx(0.5)  # 100 -> 50


def test_summary_known_values() -> None:
    finals = np.arange(1, 101, dtype=np.float64)  # 1..100
    paths = np.column_stack([np.full(100, 100.0), finals])
    rep = summarize_paths(paths, confidences=(0.95,), loss_threshold=0.5)
    # losses are 99..0; 95% quantile = 94.05; tail = losses >= that -> 95..99 (mean 97)
    assert rep.var[0.95] == pytest.approx(94.05)
    assert rep.cvar[0.95] == pytest.approx(97.0)
    assert rep.median_final == pytest.approx(50.5)
    assert rep.final_percentile(5) == pytest.approx(5.95)
    assert rep.prob_loss_at_end == pytest.approx(0.5)  # finals 1..50
    assert rep.prob_ruin == pytest.approx(0.5)
    assert rep.prob_drawdown_exceeds(0.5) == pytest.approx(0.5)  # finals 1..50


def test_ruin_touch_probability_exceeds_terminal_probability() -> None:
    paths = arr([100, 60, 100], [100, 90, 95])
    rep = summarize_paths(paths, loss_threshold=0.3)
    assert rep.prob_ruin == 0.5
    assert rep.prob_loss_at_end == 0.0


def test_summary_validation() -> None:
    paths = arr([100, 90])
    with pytest.raises(ValueError, match="loss_threshold"):
        summarize_paths(paths, loss_threshold=0.0)
    with pytest.raises(ValueError, match="confidence"):
        summarize_paths(paths, confidences=(0.3,))
    with pytest.raises(ValueError, match="no value"):
        summarize_paths(arr([0, 0]))


# ---- end to end -------------------------------------------------------------------


def _portfolio() -> Portfolio:
    return Portfolio(
        positions=(
            Position(asset=AAPL, quantity=100, broker="a"),
            Position(asset=BTC, quantity=5, broker="b"),
        ),
        cash=(CashBalance(amount=1000),),
    )


def test_run_monte_carlo_matches_parametric_var_at_one_day() -> None:
    pf = _portfolio()
    prices = SyntheticProvider(seed=3).get_prices(pf.assets, date(2021, 1, 1), date(2024, 1, 1))
    mc = run_monte_carlo(pf, prices, n_simulations=100_000, days=1, seed=11)
    parametric = analyze_risk(pf, prices, method=Method.PARAMETRIC, confidence=0.95)
    assert mc.var[0.95] == pytest.approx(parametric.var, rel=0.03)
    assert mc.cvar[0.95] == pytest.approx(parametric.cvar, rel=0.03)
    assert mc.cvar[0.99] > mc.var[0.99] > mc.var[0.95]


def test_run_monte_carlo_year_horizon_ordering() -> None:
    pf = _portfolio()
    prices = SyntheticProvider(seed=3).get_prices(pf.assets, date(2021, 1, 1), date(2024, 1, 1))
    rep = run_monte_carlo(pf, prices, n_simulations=2000, days=252, seed=5)
    again = run_monte_carlo(pf, prices, n_simulations=2000, days=252, seed=5)
    assert np.array_equal(rep.final_values, again.final_values)
    assert rep.final_percentile(5) < rep.median_final < rep.final_percentile(95)
    assert 0 <= rep.prob_loss_at_end <= rep.prob_ruin <= 1
    assert (rep.max_drawdowns >= 0).all()
    assert rep.prob_drawdown_exceeds(0.1) >= rep.prob_drawdown_exceeds(0.5)


def test_zero_volatility_portfolio_is_riskless() -> None:
    paths = simulate_paths(
        np.array([500.0]), 500.0, arr([0.0]), np.array([0.001]), 20, 10, np.random.default_rng(0)
    )
    rep = summarize_paths(paths)
    assert rep.var[0.99] == 0.0
    assert rep.prob_ruin == 0.0
    assert rep.max_drawdowns.max() == pytest.approx(0.0, abs=1e-6)
    assert rep.final_values[0] == pytest.approx(500 * np.exp(0.01) + 500, rel=1e-6)


def test_run_monte_carlo_requires_history() -> None:
    pf = _portfolio()
    prices = SyntheticProvider().get_prices(pf.assets, date(2024, 1, 1), date(2024, 1, 10))
    with pytest.raises(ValueError, match="observations"):
        run_monte_carlo(pf, prices, seed=0)
