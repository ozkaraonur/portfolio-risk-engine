from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from portfolio_risk.models import Asset, AssetClass, CashBalance, Portfolio, Position
from portfolio_risk.risk import (
    BUILTIN_SCENARIOS,
    Scenario,
    beta_scenario,
    compute_betas,
    get_scenario,
    parse_custom_shocks,
    portfolio_beta,
    run_stress,
)

NVDA = Asset(symbol="NVDA", asset_class=AssetClass.EQUITY, tags=("Tech",))
KO = Asset(symbol="KO", asset_class=AssetClass.EQUITY)
BTC = Asset(symbol="BTC", asset_class=AssetClass.CRYPTO)
GLD = Asset(symbol="GLD", asset_class=AssetClass.COMMODITY)
PRICES = {"NVDA": 100.0, "KO": 50.0, "BTC": 20_000.0, "GLD": 200.0}


def make_portfolio() -> Portfolio:
    return Portfolio(
        positions=(
            Position(asset=NVDA, quantity=10, broker="a"),  # 1,000
            Position(asset=KO, quantity=20, broker="a"),  # 1,000
            Position(asset=BTC, quantity=0.05, broker="b"),  # 1,000
            Position(asset=GLD, quantity=5, broker="b"),  # 1,000
            Position(asset=KO, quantity=20, broker="b"),  # +1,000 KO across brokers
        ),
        cash=(CashBalance(amount=1000),),
    )


def test_tags_are_normalised() -> None:
    assert NVDA.tags == ("tech",)


def test_builtin_scenarios_match_spec() -> None:
    gfc = BUILTIN_SCENARIOS["gfc-2008"]
    assert gfc.class_shocks == {
        AssetClass.EQUITY: -0.45,
        AssetClass.CRYPTO: -0.60,
        AssetClass.COMMODITY: 0.15,
    }
    covid = get_scenario("COVID-2020")
    assert covid.class_shocks[AssetClass.COMMODITY] == -0.25
    infl = get_scenario("inflation-2022")
    assert infl.shock_for(NVDA) == -0.35  # tech tag beats equity class shock
    assert infl.shock_for(KO) == -0.20
    assert infl.shock_for(BTC) == -0.65
    assert infl.shock_for(GLD) == 0.25


def test_unknown_scenario_lists_options() -> None:
    with pytest.raises(ValueError, match="gfc-2008"):
        get_scenario("nope")


def test_gfc_impact_is_exact() -> None:
    report = run_stress(make_portfolio(), PRICES, [get_scenario("gfc-2008")])
    (res,) = report.results
    by_symbol = {i.symbol: i for i in res.impacts}
    assert res.portfolio_value == pytest.approx(6000)  # 5,000 invested + 1,000 cash
    assert by_symbol["NVDA"].pnl == pytest.approx(-450)
    assert by_symbol["KO"].value == pytest.approx(2000)  # aggregated across brokers
    assert by_symbol["KO"].pnl == pytest.approx(-900)
    assert by_symbol["BTC"].pnl == pytest.approx(-600)
    assert by_symbol["GLD"].pnl == pytest.approx(150)
    assert res.total_pnl == pytest.approx(-1800)
    assert res.pnl_pct == pytest.approx(-0.30)  # cash is in the denominator, unshocked
    assert res.stressed_value == pytest.approx(4200)


def test_worst_case_and_breaches() -> None:
    scenarios = [get_scenario(n) for n in BUILTIN_SCENARIOS]
    report = run_stress(make_portfolio(), PRICES, scenarios)
    worst = report.worst_case
    assert worst.total_pnl == min(r.total_pnl for r in report.results)
    assert worst.scenario.name == "gfc-2008"  # -1800 vs covid -1650 vs 2022 -1600
    names = [r.scenario.name for r in report.breaches(0.30)]
    assert names == ["gfc-2008"]
    assert report.breaches(0.99) == []


def test_symbol_shock_takes_precedence_and_cash_untouched() -> None:
    s = Scenario(
        name="x",
        class_shocks={AssetClass.EQUITY: -0.1},
        tag_shocks={"tech": -0.2},
        symbol_shocks={"NVDA": -0.5},
    )
    assert s.shock_for(NVDA) == -0.5
    res = run_stress(make_portfolio(), PRICES, [s]).results[0]
    assert res.cash == 1000
    assert res.total_pnl == pytest.approx(-500 - 200)  # NVDA -500, KO(2000) -200


def test_most_severe_tag_wins() -> None:
    a = Asset(symbol="Z", asset_class=AssetClass.EQUITY, tags=("tech", "growth"))
    s = Scenario(name="x", tag_shocks={"tech": -0.1, "growth": -0.4})
    assert s.shock_for(a) == -0.4


def test_shock_validation() -> None:
    with pytest.raises(ValueError, match=">= -1"):
        Scenario(name="bad", class_shocks={AssetClass.EQUITY: -1.5})
    with pytest.raises(ValueError, match="unknown symbols"):
        run_stress(make_portfolio(), PRICES, [Scenario(name="x", symbol_shocks={"ZZZ": -0.1})])
    with pytest.raises(ValueError, match="At least one"):
        run_stress(make_portfolio(), PRICES, [])


def test_parse_custom_shocks() -> None:
    s = parse_custom_shocks("equity=-0.15, crypto=-0.30, nvda=-0.5, tag:Growth=-0.4")
    assert s.class_shocks == {AssetClass.EQUITY: -0.15, AssetClass.CRYPTO: -0.30}
    assert s.symbol_shocks == {"NVDA": -0.5}
    assert s.tag_shocks == {"growth": -0.4}
    res = run_stress(make_portfolio(), PRICES, [parse_custom_shocks("equity=-0.2,crypto=-0.4")])
    assert res.results[0].total_pnl == pytest.approx(-0.2 * 3000 - 0.4 * 1000)


@pytest.mark.parametrize("spec", ["", "equity", "equity=abc", "=0.1", "equity=-2"])
def test_parse_custom_shocks_rejects_bad_input(spec: str) -> None:
    with pytest.raises(ValueError, match=r"(?i)shock"):
        parse_custom_shocks(spec)


def test_compute_betas_known_values() -> None:
    rng = np.random.default_rng(0)
    m = pd.Series(rng.normal(0, 0.01, 500))
    returns = pd.DataFrame({"HI": 1.5 * m, "LO": 0.5 * m + 0.0, "NEG": -1.0 * m, "MKT": m})
    betas = compute_betas(returns, m)
    assert betas["HI"] == pytest.approx(1.5)
    assert betas["LO"] == pytest.approx(0.5)
    assert betas["NEG"] == pytest.approx(-1.0)
    assert betas["MKT"] == pytest.approx(1.0)


def test_compute_betas_with_noise_and_alpha_is_close() -> None:
    rng = np.random.default_rng(1)
    m = pd.Series(rng.normal(0, 0.01, 20_000))
    r = pd.DataFrame({"A": 1.2 * m + rng.normal(0.001, 0.005, len(m))})
    assert compute_betas(r, m)["A"] == pytest.approx(1.2, abs=0.03)


def test_compute_betas_errors() -> None:
    with pytest.raises(ValueError, match="zero variance"):
        compute_betas(pd.DataFrame({"A": [0.1, 0.2, 0.3]}), pd.Series([0.0, 0.0, 0.0]))
    with pytest.raises(ValueError, match="two"):
        compute_betas(pd.DataFrame({"A": [0.1]}), pd.Series([0.1]))


def test_beta_scenario_propagates_market_shock() -> None:
    betas = {"NVDA": 2.0, "KO": 0.5, "BTC": 1.0, "GLD": -0.2}
    scenario = beta_scenario(-0.10, betas)
    assert scenario.name == "market-10%"
    res = run_stress(make_portfolio(), PRICES, [scenario]).results[0]
    by = {i.symbol: i for i in res.impacts}
    assert by["NVDA"].shock == pytest.approx(-0.20)
    assert by["GLD"].shock == pytest.approx(0.02)  # negative beta gains when market falls
    expected = -0.2 * 1000 - 0.05 * 2000 - 0.1 * 1000 + 0.02 * 1000
    assert res.total_pnl == pytest.approx(expected)
    # Portfolio-level: value-weighted beta x market shock x invested value.
    exposures = {"NVDA": 1000.0, "KO": 2000.0, "BTC": 1000.0, "GLD": 1000.0}
    pbeta = portfolio_beta(exposures, betas, 6000.0)
    assert pbeta == pytest.approx((2000 + 1000 + 1000 - 200) / 6000)
    assert res.total_pnl == pytest.approx(pbeta * -0.10 * 6000)


def test_beta_scenario_floors_at_total_loss_and_validates() -> None:
    scenario = beta_scenario(-0.5, {"X": 5.0})
    assert scenario.symbol_shocks["X"] == -1.0
    with pytest.raises(ValueError, match="greater than -1"):
        beta_scenario(-1.0, {})
    with pytest.raises(ValueError, match="zero value"):
        portfolio_beta({}, {}, 0.0)
