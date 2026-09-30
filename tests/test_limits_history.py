from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from portfolio_risk.cli import app
from portfolio_risk.data import SyntheticProvider
from portfolio_risk.models import Asset, AssetClass, CashBalance, Portfolio, Position
from portfolio_risk.reporting import RiskAnalysis, build_analysis
from portfolio_risk.reporting.history import load_history, record, snapshot_of
from portfolio_risk.reporting.limits import RiskLimits, check_limits, load_limits

runner = CliRunner()
EXAMPLE = Path(__file__).parent.parent / "examples" / "portfolio.json"
LIMITS = Path(__file__).parent.parent / "examples" / "limits.json"
AAPL = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
BTC = Asset(symbol="BTC", asset_class=AssetClass.CRYPTO)


def make_analysis(name: str = "test") -> RiskAnalysis:
    pf = Portfolio(
        name=name,
        positions=(Position(asset=AAPL, quantity=100), Position(asset=BTC, quantity=2)),
        cash=(CashBalance(amount=5000),),
    )
    prices = SyntheticProvider(seed=3).get_prices(pf.assets, date(2022, 1, 1), date(2024, 1, 1))
    return build_analysis(pf, prices, simulations=500, mc_days=60, seed=1)


def test_only_configured_limits_are_checked() -> None:
    analysis = make_analysis()
    assert check_limits(analysis, RiskLimits()) == []
    checks = check_limits(analysis, RiskLimits(min_cash_ratio=0.5, max_asset_weight=1.0))
    assert [c.name for c in checks] == ["Largest position weight", "Cash ratio"]


def test_breach_direction_and_values() -> None:
    analysis = make_analysis()
    top = max(analysis.assets, key=lambda a: a.weight)

    tight = check_limits(analysis, RiskLimits(max_asset_weight=top.weight / 2))[0]
    loose = check_limits(analysis, RiskLimits(max_asset_weight=1.0))[0]
    assert tight.breached
    assert tight.actual == pytest.approx(top.weight)
    assert tight.detail == top.symbol
    assert not loose.breached

    cash_high = check_limits(analysis, RiskLimits(min_cash_ratio=0.99))[0]
    cash_low = check_limits(analysis, RiskLimits(min_cash_ratio=0.0))[0]
    assert cash_high.breached  # a minimum is breached when the actual is below it
    assert not cash_low.breached


def test_var_and_ruin_limits_use_the_analysis_figures() -> None:
    analysis = make_analysis()
    checks = {
        c.name: c
        for c in check_limits(analysis, RiskLimits(max_var_pct=1.0, max_ruin_probability=1.0))
    }
    var_check = checks["VaR / value (99%, 10d)"]
    assert var_check.actual == pytest.approx(analysis.headline_var.var / analysis.total_value)
    assert checks["Monte Carlo ruin probability"].actual == analysis.monte_carlo.prob_ruin
    assert not any(c.breached for c in checks.values())


def test_limits_file_validation(tmp_path: Path) -> None:
    assert load_limits(LIMITS).max_var_pct == 0.12
    bad = tmp_path / "bad.json"
    bad.write_text('{"max_var_pct": 5}')
    with pytest.raises(ValidationError):
        load_limits(bad)
    bad.write_text('{"unknown_limit": 0.1}')
    with pytest.raises(ValidationError):
        load_limits(bad)


def test_history_roundtrip_and_filtering(tmp_path: Path) -> None:
    db = tmp_path / "sub" / "history.sqlite"
    assert load_history(db) == []  # a missing database is an empty history, not an error
    first = snapshot_of(make_analysis("alpha"), breaches=2)
    second = snapshot_of(make_analysis("beta"))
    assert record(db, first) == 1
    assert record(db, second) == 2

    rows = load_history(db)
    assert [r.portfolio for r in rows] == ["beta", "alpha"]  # newest first
    only = load_history(db, "alpha")
    assert len(only) == 1
    assert only[0].breaches == 2
    assert only[0].var == pytest.approx(first.var)
    assert only[0].total_value == pytest.approx(first.total_value)
    assert only[0].recorded_at == first.recorded_at
    assert load_history(db, limit=1)[0].portfolio == "beta"
    assert rows[0].breaches is None
    assert 0 < only[0].var_pct < 1


def test_snapshot_reflects_the_headline_figures() -> None:
    analysis = make_analysis()
    snap = snapshot_of(analysis)
    assert snap.var == analysis.headline_var.var
    assert snap.mc_var == analysis.monte_carlo.var[0.99]
    assert snap.top_symbol in {a.symbol for a in analysis.assets}
    assert snap.top_risk_share == pytest.approx(float(analysis.contributions.var_share.max()))


def test_history_accumulates_across_connections(tmp_path: Path) -> None:
    db = tmp_path / "h.sqlite"
    record(db, snapshot_of(make_analysis()))
    with closing(sqlite3.connect(db)) as conn:
        assert conn.execute("SELECT COUNT(*) FROM snapshots").fetchone()[0] == 1
    record(db, snapshot_of(make_analysis()))
    assert len(load_history(db)) == 2


# --- CLI ---------------------------------------------------------------------------------------


def test_check_command_exit_codes_and_history(tmp_path: Path) -> None:
    db = tmp_path / "h.sqlite"
    strict = tmp_path / "strict.json"
    strict.write_text('{"max_risk_share": 0.5}')
    loose = tmp_path / "loose.json"
    loose.write_text('{"max_risk_share": 1.0, "min_cash_ratio": 0.0}')

    breached = runner.invoke(
        app, ["check", str(EXAMPLE), str(strict), "--db", str(db), "--record", "-n", "200"]
    )
    assert breached.exit_code == 2, breached.output
    assert "BREACH" in breached.output
    assert "Recorded snapshot #1" in breached.output

    ok = runner.invoke(
        app, ["check", str(EXAMPLE), str(loose), "--db", str(db), "--record", "-n", "200"]
    )
    assert ok.exit_code == 0, ok.output
    assert "All limits respected" in ok.output

    shown = runner.invoke(app, ["history", "--db", str(db)])
    assert shown.exit_code == 0
    lines = [ln for ln in shown.output.splitlines() if ln.startswith("20")]
    assert len(lines) == 2
    assert "demo" in lines[0]
    assert lines[0].split()[-1] == "1"  # first run had one breach
    assert lines[1].split()[-1] == "0"


def test_check_command_rejects_bad_input(tmp_path: Path) -> None:
    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    assert runner.invoke(app, ["check", str(EXAMPLE), str(empty)]).exit_code == 1
    bad = tmp_path / "bad.json"
    bad.write_text('{"max_var_pct": 0}')
    assert runner.invoke(app, ["check", str(EXAMPLE), str(bad)]).exit_code == 1


def test_history_with_no_database_is_friendly(tmp_path: Path) -> None:
    result = runner.invoke(app, ["history", "--db", str(tmp_path / "none.sqlite")])
    assert result.exit_code == 0
    assert "No snapshots" in result.output


def test_report_can_record_a_snapshot(tmp_path: Path) -> None:
    db = tmp_path / "h.sqlite"
    result = runner.invoke(
        app, ["report", str(EXAMPLE), "--quiet", "-n", "200", "--record", "--db", str(db)]
    )
    assert result.exit_code == 0, result.output
    assert len(load_history(db, "demo")) == 1
