from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from portfolio_risk import __version__
from portfolio_risk.cli import app, load_portfolio

runner = CliRunner()
EXAMPLE = Path(__file__).parent.parent / "examples" / "portfolio.json"


def test_version() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_summary_synthetic() -> None:
    result = runner.invoke(app, ["summary", str(EXAMPLE), "--seed", "1"])
    assert result.exit_code == 0, result.output
    for token in ("AAPL", "BTC", "GOLD", "CASH", "TOTAL"):
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


def test_backtest_command() -> None:
    result = runner.invoke(app, ["backtest", str(EXAMPLE), "--seed", "1", "--window", "120"])
    assert result.exit_code == 0, result.output
    for token in ("parametric", "historical", "VIOLATIONS", "KUPIEC", "ZONE", "99.0%"):
        assert token in result.output


def test_backtest_needs_enough_history() -> None:
    result = runner.invoke(app, ["backtest", str(EXAMPLE), "--days", "200"])
    assert result.exit_code == 1
    assert "Error" in result.output


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


def test_backtest_lists_every_method() -> None:
    result = runner.invoke(app, ["backtest", str(EXAMPLE), "--seed", "1", "--window", "120"])
    assert result.exit_code == 0, result.output
    for method in ("ewma", "student-t", "cornish-fisher", "fhs"):
        assert method in result.output


def test_risk_all_methods() -> None:
    result = runner.invoke(app, ["risk", str(EXAMPLE), "--seed", "1", "--all-methods"])
    assert result.exit_code == 0, result.output
    for method in ("parametric", "historical", "ewma", "student-t", "cornish-fisher", "fhs"):
        assert method in result.output
    default = runner.invoke(app, ["risk", str(EXAMPLE), "--seed", "1"])
    assert "student-t" not in default.output


def test_simulate_with_fat_tails_and_shrinkage() -> None:
    args = ["simulate", str(EXAMPLE), "-n", "400", "-t", "30", "--seed", "1"]
    fat = runner.invoke(app, [*args, "--df", "4", "--cov-method", "shrinkage"])
    assert fat.exit_code == 0, fat.output
    assert "Student-t (df=4)" in fat.output
    assert "shrinkage" in fat.output
    assert "normal" in runner.invoke(app, args).output


def test_simulate_rejects_df_at_or_below_two() -> None:
    result = runner.invoke(app, ["simulate", str(EXAMPLE), "--df", "2"])
    assert result.exit_code != 0


def test_attribute_command() -> None:
    result = runner.invoke(app, ["attribute", str(EXAMPLE), "--seed", "1"])
    assert result.exit_code == 0, result.output
    for token in ("VaR CONTRIB.", "SHARE", "MARGINAL/1000", "AAPL", "BTC", "GOLD"):
        assert token in result.output
    hist = runner.invoke(
        app, ["attribute", str(EXAMPLE), "--seed", "1", "--method", "historical", "--horizon", "1"]
    )
    assert hist.exit_code == 0, hist.output
    assert "historical" in hist.output


def test_attribute_rejects_unsupported_method() -> None:
    result = runner.invoke(app, ["attribute", str(EXAMPLE), "--method", "fhs"])
    assert result.exit_code == 1
    assert "parametric and historical" in result.output


def test_optimize_command_lists_objectives_trades_and_frontier() -> None:
    result = runner.invoke(
        app, ["optimize", str(EXAMPLE), "--seed", "1", "--frontier", "3", "--max-weight", "0.7"]
    )
    assert result.exit_code == 0, result.output
    for token in ("current", "min-variance", "risk-parity", "max-sharpe", "Trades for", "FRONTIER"):
        assert token in result.output


def test_optimize_single_objective_and_errors() -> None:
    one = runner.invoke(
        app, ["optimize", str(EXAMPLE), "--seed", "1", "--objective", "min-variance"]
    )
    assert one.exit_code == 0, one.output
    assert "risk-parity" not in one.output
    bad = runner.invoke(app, ["optimize", str(EXAMPLE), "--objective", "magic"])
    assert bad.exit_code == 1
    assert "unknown objective" in bad.output
    infeasible = runner.invoke(app, ["optimize", str(EXAMPLE), "--max-weight", "0.2"])
    assert infeasible.exit_code == 1
    assert "infeasible" in infeasible.output


# --- data sources: import, multi-currency, cache -------------------------------------------------


def test_import_merges_broker_exports_into_a_portfolio(tmp_path: Path) -> None:
    ibkr = tmp_path / "ibkr.csv"
    ibkr.write_text("Symbol,Position,Currency\nAAPL,10,USD\nSAP,5,EUR\nEUR,100,CASH\n")
    binance = tmp_path / "coins.csv"
    binance.write_text("Coin,Total\nBTC,0.5\nUSDT,250\n")
    out = tmp_path / "out" / "portfolio.json"
    result = runner.invoke(
        app, ["import", f"ibkr={ibkr}", str(binance), "-o", str(out), "--name", "mine"]
    )
    assert result.exit_code == 0, result.output
    assert "3 positions" in result.output
    portfolio = load_portfolio(out)
    assert portfolio.name == "mine"
    assert portfolio.brokers == ["coins", "ibkr"]  # file stem is the default broker name
    assert portfolio.quantities()["BTC"] == 0.5


def test_import_errors(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("Name,Value\nx,1\n")
    out = tmp_path / "p.json"
    assert runner.invoke(app, ["import", str(bad), "-o", str(out)]).exit_code == 1
    assert (
        runner.invoke(app, ["import", str(tmp_path / "missing.csv"), "-o", str(out)]).exit_code == 1
    )
    assert not out.exists()


def test_multi_currency_portfolio_is_valued_in_the_base_currency(tmp_path: Path) -> None:
    portfolio = tmp_path / "eur.json"
    portfolio.write_text(
        '{"base_currency": "USD", "positions": ['
        '{"asset": {"symbol": "SAP", "asset_class": "equity", "currency": "EUR"}, "quantity": 10},'
        '{"asset": {"symbol": "AAPL", "asset_class": "equity"}, "quantity": 5}],'
        '"cash": [{"amount": 1000, "currency": "EUR"}, {"amount": 500}]}'
    )
    for command in ("summary", "risk", "backtest"):
        result = runner.invoke(app, [command, str(portfolio), "--seed", "3"])
        assert result.exit_code == 0, (command, result.output)
    once = runner.invoke(app, ["summary", str(portfolio), "--seed", "3"])
    again = runner.invoke(app, ["summary", str(portfolio), "--seed", "3"])
    assert once.output == again.output
    assert "CASH" in once.output


def test_stooq_prices_are_cached_between_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fetched: list[str] = []
    days = pd.bdate_range(date.today() - timedelta(days=800), date.today())

    def fake_fetch(url: str) -> str:
        fetched.append(url)
        rows = ["Date,Open,High,Low,Close,Volume"]
        rows += [f"{d.date()},1,1,1,{100 + i % 7 + 0.1 * i:.2f},1" for i, d in enumerate(days)]
        return "\n".join(rows)

    monkeypatch.setattr("portfolio_risk.data.public._http_fetch", fake_fetch)
    args = ["--cache-dir", str(tmp_path / "cache"), "summary", str(EXAMPLE), "--provider", "stooq"]
    first = runner.invoke(app, args)
    assert first.exit_code == 0, first.output
    downloads = len(fetched)
    assert downloads == 3  # AAPL, BTC, GOLD
    second = runner.invoke(app, args)
    assert second.exit_code == 0
    assert len(fetched) == downloads
    assert second.output == first.output
    uncached = runner.invoke(app, ["--no-cache", *args[2:]])
    assert uncached.exit_code == 0
    assert len(fetched) == 2 * downloads


def test_invalid_files_give_one_readable_line(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text("{bad")
    result = runner.invoke(app, ["summary", str(bad)])
    assert result.exit_code == 1
    assert result.output.startswith(f"Error: {bad}: ")
    assert "errors.pydantic.dev" not in result.output
    assert len(result.output.strip().splitlines()) == 1

    wrong = tmp_path / "wrong.json"
    wrong.write_text(
        '{"positions": [{"asset": {"symbol": "A", "asset_class": "bond"}, "quantity": -1}]}'
    )
    result = runner.invoke(app, ["summary", str(wrong)])
    assert result.exit_code == 1
    assert "positions.0.asset.asset_class" in result.output
    assert "positions.0.quantity" in result.output

    limits = tmp_path / "limits.json"
    limits.write_text('{"max_var_pct": 5}')
    result = runner.invoke(app, ["check", str(EXAMPLE), str(limits)])
    assert result.exit_code == 1
    assert "max_var_pct" in result.output
    assert "errors.pydantic.dev" not in result.output
