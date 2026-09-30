from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.risk import (
    Objective,
    covariance_matrix,
    efficient_frontier,
    expected_returns,
    max_sharpe_weights,
    min_variance_weights,
    optimize_portfolio,
    rebalance_trades,
    risk_parity_weights,
)


def _returns(
    vols: list[float], mus: list[float], corr: float, n: int = 2000, seed: int = 0
) -> pd.DataFrame:
    k = len(vols)
    corr_m = np.full((k, k), corr) + np.eye(k) * (1.0 - corr)
    chol = np.linalg.cholesky(corr_m)
    z = np.random.default_rng(seed).standard_normal((n, k)) @ chol.T
    z = (z - z.mean(axis=0)) / z.std(axis=0, ddof=1)  # exact sample moments
    data = z * np.array(vols) + np.array(mus)
    return pd.DataFrame(data, columns=[f"A{i}" for i in range(k)])


def _annual_cov(returns: pd.DataFrame) -> np.ndarray:
    return np.asarray(covariance_matrix(returns), dtype=float) * 252


# --- min variance ------------------------------------------------------------------------------


def test_min_variance_two_uncorrelated_assets_has_closed_form() -> None:
    cov = np.diag([0.04, 0.09])
    w = min_variance_weights(cov)
    assert w == pytest.approx([0.09 / 0.13, 0.04 / 0.13], abs=1e-6)


def test_min_variance_matches_inverse_covariance_solution() -> None:
    cov = _annual_cov(_returns([0.01, 0.015, 0.02], [0, 0, 0], 0.2))
    inv = np.linalg.solve(cov, np.ones(3))
    assert min_variance_weights(cov) == pytest.approx(inv / inv.sum(), abs=1e-5)


def test_min_variance_is_long_only_when_the_unconstrained_optimum_shorts() -> None:
    sig = np.array([0.1, 0.3])
    cov = np.array([[sig[0] ** 2, 0.9 * sig[0] * sig[1]], [0.9 * sig[0] * sig[1], sig[1] ** 2]])
    assert min_variance_weights(cov) == pytest.approx([1.0, 0.0], abs=1e-6)


def test_weight_cap_is_respected_and_validated() -> None:
    cov = np.diag([0.01, 0.04, 0.09])
    w = min_variance_weights(cov, max_weight=0.5)
    assert w.max() <= 0.5 + 1e-9
    assert w.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError, match="infeasible"):
        min_variance_weights(cov, max_weight=0.3)
    with pytest.raises(ValueError, match="max_weight"):
        min_variance_weights(cov, max_weight=0.0)


# --- risk parity -------------------------------------------------------------------------------


def test_risk_parity_uncorrelated_is_inverse_volatility() -> None:
    cov = np.diag([0.04, 0.09, 0.16])
    w = risk_parity_weights(cov)
    inv_vol = 1 / np.sqrt(np.diag(cov))
    assert w == pytest.approx(inv_vol / inv_vol.sum(), abs=1e-5)


def test_risk_parity_equalises_risk_contributions() -> None:
    cov = _annual_cov(_returns([0.01, 0.02, 0.03, 0.015], [0] * 4, 0.35))
    w = risk_parity_weights(cov)
    contributions = w * (cov @ w)
    assert contributions == pytest.approx(np.full(4, contributions.mean()), rel=1e-4)
    assert w.sum() == pytest.approx(1.0)
    assert (w > 0).all()


# --- max Sharpe --------------------------------------------------------------------------------


def test_max_sharpe_matches_a_grid_search() -> None:
    cov = np.array([[0.04, 0.006], [0.006, 0.09]])
    mu = np.array([0.08, 0.14])
    w = max_sharpe_weights(cov, mu)
    grid = np.linspace(0, 1, 2001)
    best = max(
        (g * mu[0] + (1 - g) * mu[1]) / np.sqrt(np.array([g, 1 - g]) @ cov @ np.array([g, 1 - g]))
        for g in grid
    )
    sharpe = float(w @ mu / np.sqrt(w @ cov @ w))
    assert sharpe == pytest.approx(best, rel=1e-6)


def test_max_sharpe_beats_equal_weight_and_single_assets() -> None:
    returns = _returns([0.01, 0.02, 0.015], [0.0004, 0.0006, 0.0002], 0.3)
    cov, mu = _annual_cov(returns), np.asarray(expected_returns(returns))
    w = max_sharpe_weights(cov, mu)

    def sharpe(x: np.ndarray) -> float:
        return float(x @ mu / np.sqrt(x @ cov @ x))

    assert sharpe(w) >= sharpe(np.full(3, 1 / 3)) - 1e-9
    assert all(sharpe(w) >= sharpe(np.eye(3)[i]) - 1e-9 for i in range(3))


def test_max_sharpe_needs_a_positive_excess_return() -> None:
    with pytest.raises(ValueError, match="positive excess"):
        max_sharpe_weights(np.diag([0.04, 0.09]), np.array([-0.01, 0.02]), risk_free=0.05)


# --- public API --------------------------------------------------------------------------------


def test_optimize_portfolio_reports_annualised_stats() -> None:
    returns = _returns([0.01, 0.02], [0.0003, 0.0005], 0.1)
    alloc = optimize_portfolio(Objective.MIN_VARIANCE, returns)
    w = alloc.weights.to_numpy()
    assert alloc.weights.sum() == pytest.approx(1.0)
    assert list(alloc.weights.index) == ["A0", "A1"]
    assert alloc.volatility == pytest.approx(float(np.sqrt(w @ _annual_cov(returns) @ w)))
    assert alloc.expected_return == pytest.approx(float(w @ (returns.mean().to_numpy() * 252)))


@pytest.mark.parametrize("objective", list(Objective))
def test_every_objective_returns_valid_long_only_weights(objective: Objective) -> None:
    returns = _returns([0.01, 0.02, 0.03], [0.0004, 0.0005, 0.0006], 0.25)
    alloc = optimize_portfolio(objective, returns, max_weight=1.0)
    assert alloc.weights.sum() == pytest.approx(1.0)
    assert (alloc.weights >= -1e-9).all()


def test_min_variance_never_has_higher_volatility_than_other_objectives() -> None:
    returns = _returns([0.01, 0.02, 0.03], [0.0004, 0.0005, 0.0006], 0.25)
    vol = {o: optimize_portfolio(o, returns).volatility for o in Objective}
    assert vol[Objective.MIN_VARIANCE] <= min(vol.values()) + 1e-9


def test_optimize_needs_two_assets() -> None:
    with pytest.raises(ValueError, match="two assets"):
        optimize_portfolio(Objective.MIN_VARIANCE, _returns([0.01], [0.0], 0.0))


# --- frontier and rebalancing ------------------------------------------------------------------


def test_efficient_frontier_is_monotone_and_starts_at_min_variance() -> None:
    returns = _returns([0.01, 0.02, 0.03], [0.0002, 0.0005, 0.0009], 0.2)
    frontier = efficient_frontier(returns, points=12)
    assert len(frontier) >= 10
    rets = [p.expected_return for p in frontier]
    vols = [p.volatility for p in frontier]
    assert rets == sorted(rets)
    assert vols == sorted(vols)
    assert vols[0] == pytest.approx(optimize_portfolio(Objective.MIN_VARIANCE, returns).volatility)
    assert all(p.weights.sum() == pytest.approx(1.0) for p in frontier)
    assert rets[-1] == pytest.approx(0.0009 * 252, rel=1e-3)  # all-in on the best asset


def test_frontier_respects_the_cap() -> None:
    returns = _returns([0.01, 0.02, 0.03], [0.0002, 0.0005, 0.0009], 0.2)
    frontier = efficient_frontier(returns, points=6, max_weight=0.5)
    assert all(p.weights.max() <= 0.5 + 1e-9 for p in frontier)
    with pytest.raises(ValueError, match="points"):
        efficient_frontier(returns, points=1)


def test_rebalance_trades_are_self_financing() -> None:
    values = pd.Series({"A": 6000.0, "B": 3000.0, "C": 1000.0})
    target = pd.Series({"A": 0.4, "B": 0.4, "C": 0.2})
    trades = rebalance_trades(values, target)
    assert trades.sum() == pytest.approx(0.0)
    assert (values + trades).tolist() == pytest.approx([4000.0, 4000.0, 2000.0])
    with pytest.raises(ValueError, match="cover"):
        rebalance_trades(values, target.drop("C"))
