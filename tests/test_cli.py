from pathlib import Path

from typer.testing import CliRunner

from portfolio_risk import __version__
from portfolio_risk.cli import app

runner = CliRunner()
EXAMPLE = Path(__file__).parent.parent / "examples" / "portfolio.json"


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_summary_synthetic() -> None:
    result = runner.invoke(app, ["summary", str(EXAMPLE), "--seed", "1"])
    assert result.exit_code == 0, result.output
    for token in ("AAPL", "BTC", "GC", "CASH", "TOTAL"):
        assert token in result.output


def test_summary_invalid_file(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text('{"positions": [{"quantity": -1}]}')
    result = runner.invoke(app, ["summary", str(bad)])
    assert result.exit_code == 1


def test_risk_command() -> None:
    result = runner.invoke(
        app, ["risk", str(EXAMPLE), "--confidence", "0.99", "--horizon", "10", "--seed", "1"]
    )
    assert result.exit_code == 0, result.output
    for token in ("parametric", "historical", "CVaR", "Correlation", "AAPL", "BTC", "99.0%"):
        assert token in result.output


def test_risk_rejects_bad_confidence() -> None:
    result = runner.invoke(app, ["risk", str(EXAMPLE), "--confidence", "0.2"])
    assert result.exit_code != 0


def test_simulate_command() -> None:
    result = runner.invoke(app, ["simulate", str(EXAMPLE), "-n", "500", "-t", "60", "--seed", "1"])
    assert result.exit_code == 0, result.output
    for token in ("500 paths", "VaR", "CVaR", "median", "Maximum drawdown", "ruin"):
        assert token in result.output


def test_simulate_is_deterministic() -> None:
    args = ["simulate", str(EXAMPLE), "-n", "200", "-t", "30", "--seed", "9"]
    assert runner.invoke(app, args).output == runner.invoke(app, args).output


def test_simulate_rejects_too_few_simulations() -> None:
    assert runner.invoke(app, ["simulate", str(EXAMPLE), "-n", "1"]).exit_code != 0


def test_stress_all_builtin_scenarios() -> None:
    result = runner.invoke(app, ["stress", str(EXAMPLE)])
    assert result.exit_code == 0, result.output
    for token in ("gfc-2008", "covid-2020", "inflation-2022", "Worst case", "CASH"):
        assert token in result.output


def test_stress_single_custom_and_market_shock() -> None:
    args = ["stress", str(EXAMPLE), "--scenario", "gfc-2008", "--custom", "equity=-0.5"]
    result = runner.invoke(app, [*args, "--market-shock", "-0.1", "--loss-threshold", "0.05"])
    assert result.exit_code == 0, result.output
    assert "custom" in result.output
    assert "market-10%" in result.output
    assert "portfolio beta" in result.output
    assert "covid-2020" not in result.output
    assert "WARNING" in result.output


def test_stress_custom_only_skips_builtins() -> None:
    result = runner.invoke(app, ["stress", str(EXAMPLE), "--custom", "crypto=-0.3"])
    assert result.exit_code == 0, result.output
    assert "gfc-2008" not in result.output


def test_stress_errors() -> None:
    assert runner.invoke(app, ["stress", str(EXAMPLE), "--scenario", "nope"]).exit_code == 1
    assert runner.invoke(app, ["stress", str(EXAMPLE), "--custom", "ZZZ=-0.1"]).exit_code == 1
