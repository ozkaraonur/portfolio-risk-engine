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
