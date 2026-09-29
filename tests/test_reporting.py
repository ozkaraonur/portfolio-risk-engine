from __future__ import annotations

import io
import runpy
from datetime import date
from pathlib import Path

import pytest
from rich.console import Console
from typer.testing import CliRunner

from portfolio_risk.cli import app
from portfolio_risk.data import SyntheticProvider
from portfolio_risk.models import Asset, AssetClass, CashBalance, Portfolio, Position
from portfolio_risk.reporting import (
    RiskAnalysis,
    build_analysis,
    render_dashboard,
    render_html,
    render_markdown,
)
from portfolio_risk.risk import Method, analyze_risk

ROOT = Path(__file__).parent.parent
AAPL = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY, tags=("tech",))
BTC = Asset(symbol="BTC", asset_class=AssetClass.CRYPTO)


def make_analysis(name: str = "test") -> RiskAnalysis:
    pf = Portfolio(
        name=name,
        positions=(
            Position(asset=AAPL, quantity=100, broker="a"),
            Position(asset=BTC, quantity=2, broker="b"),
        ),
        cash=(CashBalance(amount=5000),),
    )
    prices = SyntheticProvider(seed=3).get_prices(pf.assets, date(2022, 1, 1), date(2024, 1, 1))
    return build_analysis(pf, prices, simulations=500, mc_days=60, seed=1)


@pytest.fixture(scope="module")
def analysis() -> RiskAnalysis:
    return make_analysis()


def test_executive_summary_values(analysis: RiskAnalysis) -> None:
    latest_total = sum(a.value for a in analysis.assets) + analysis.cash
    assert analysis.total_value == pytest.approx(latest_total)
    assert analysis.cash_ratio == pytest.approx(5000 / latest_total)
    assert sum(a.weight for a in analysis.assets) + analysis.cash_ratio == pytest.approx(1.0)
    top = analysis.top_risk_asset
    assert top is not None
    assert top.standalone_var == max(a.standalone_var for a in analysis.assets)
    assert len(analysis.var_reports) == 8  # 2 methods x 2 confidences x 2 horizons
    assert len(analysis.stress.results) == 3


def test_headline_var_is_parametric_10d_99(analysis: RiskAnalysis) -> None:
    pf = analysis.portfolio
    prices = SyntheticProvider(seed=3).get_prices(pf.assets, date(2022, 1, 1), date(2024, 1, 1))
    expected = analyze_risk(pf, prices, method=Method.PARAMETRIC, confidence=0.99, horizon=10)
    assert analysis.headline_var.var == pytest.approx(expected.var)
    with pytest.raises(KeyError):
        analysis.var_report(Method.PARAMETRIC, 0.9, 3)


def test_build_analysis_rejects_empty_portfolio() -> None:
    prices = SyntheticProvider().get_prices([AAPL], date(2022, 1, 1), date(2024, 1, 1))
    with pytest.raises(ValueError, match="no positions"):
        build_analysis(Portfolio(cash=(CashBalance(amount=1),)), prices)


def test_html_is_self_contained_and_complete(analysis: RiskAnalysis) -> None:
    html = render_html(analysis)
    assert html.startswith("<!doctype html>")
    for forbidden in ("http://", "https://", "<script", "<link", "<img", "@import", "src="):
        assert forbidden not in html
    for section in (
        "Executive Risk Summary",
        "Allocation &amp; Correlation",
        "Statistical Risk",
        "Monte Carlo",
        "Crisis Resilience",
    ):
        assert section in html
    for token in ("gfc-2008", "covid-2020", "inflation-2022", "AAPL", "BTC", "CASH", "<svg"):
        assert token in html
    assert f"{analysis.total_value:,.2f}" in html
    assert f"{analysis.headline_var.var:,.2f}" in html


def test_html_escapes_untrusted_text() -> None:
    html = render_html(make_analysis(name="<script>alert(1)</script>"))
    assert "<script" not in html
    assert "&lt;script&gt;" in html


def test_reports_are_deterministic(analysis: RiskAnalysis) -> None:
    again = make_analysis()
    assert render_html(analysis) == render_html(again)
    assert render_markdown(analysis) == render_markdown(again)


def test_markdown_sections_and_numbers(analysis: RiskAnalysis) -> None:
    md = render_markdown(analysis)
    for heading in (
        "# Risk Report: test",
        "## Executive Risk Summary",
        "## Allocation",
        "### Correlation matrix",
        "## Statistical Risk",
        "## Monte Carlo",
        "## Crisis Resilience",
    ):
        assert heading in md
    assert f"{analysis.total_value:,.2f}" in md
    assert "**Worst case:**" in md
    assert "| Scenario |" in md


def test_terminal_dashboard(analysis: RiskAnalysis) -> None:
    buffer = io.StringIO()
    render_dashboard(analysis, Console(file=buffer, width=110, color_system=None))
    text = buffer.getvalue()
    for token in ("Risk Dashboard", "Allocation", "Correlation", "VaR / CVaR", "Monte Carlo"):
        assert token in text
    assert "Stress tests" in text
    assert "Worst case" in text


def test_cli_report_writes_files(tmp_path: Path) -> None:
    html, md = tmp_path / "out" / "r.html", tmp_path / "out" / "r.md"
    result = CliRunner().invoke(
        app,
        [
            "report",
            str(ROOT / "examples" / "portfolio.json"),
            "--html",
            str(html),
            "--markdown",
            str(md),
            "-n",
            "200",
            "--mc-days",
            "30",
            "--quiet",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "Risk Dashboard" not in result.output
    assert "Executive Risk Summary" in html.read_text(encoding="utf-8")
    assert "# Risk Report" in md.read_text(encoding="utf-8")


def test_cli_report_dashboard_only() -> None:
    result = CliRunner().invoke(
        app, ["report", str(ROOT / "examples" / "portfolio.json"), "-n", "200", "--mc-days", "20"]
    )
    assert result.exit_code == 0, result.output
    assert "Risk Dashboard" in result.output


def test_cli_report_bad_input(tmp_path: Path) -> None:
    empty = tmp_path / "empty.json"
    empty.write_text('{"cash": [{"amount": 10}]}')
    assert CliRunner().invoke(app, ["report", str(empty)]).exit_code == 1


def test_demo_script_generates_reports(tmp_path: Path) -> None:
    module = runpy.run_path(str(ROOT / "examples" / "run_demo.py"), run_name="demo")
    module["main"](tmp_path)
    assert (tmp_path / "risk-report.html").stat().st_size > 5000
    assert (tmp_path / "risk-report.md").read_text(encoding="utf-8").startswith("# Risk Report")
