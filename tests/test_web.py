from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from typer.testing import CliRunner

from portfolio_risk import cli
from portfolio_risk.cli import app
from portfolio_risk.models import AssetClass
from portfolio_risk.web.builder import (
    COL_AMOUNT,
    COL_BROKER,
    COL_CLASS,
    COL_QTY,
    COL_SYMBOL,
    COL_TAGS,
    PortfolioInputError,
    broker_options,
    empty_cash,
    empty_positions,
    frames_to_portfolio,
    load_sample_portfolio,
    parse_tags,
    portfolio_to_frames,
)

APP = Path(cli.__file__).parent / "web" / "app.py"


def positions(*rows: tuple[object, object, object, object, object]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[COL_BROKER, COL_SYMBOL, COL_CLASS, COL_QTY, COL_TAGS])


def cash(*rows: tuple[object, object]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[COL_BROKER, COL_AMOUNT])


# ---- builder ------------------------------------------------------------------------


def test_parse_tags() -> None:
    assert parse_tags("tech, Growth ;  ") == ("tech", "Growth")
    assert parse_tags(None) == ()
    assert parse_tags(float("nan")) == ()


def test_frames_to_portfolio_builds_positions_and_cash() -> None:
    pf = frames_to_portfolio(
        positions(
            ("Binance", " btc ", "crypto", 0.5, None),
            ("BIST", "AAPL", "equity", 10, "Tech, growth"),
            (None, None, None, None, None),  # blank row ignored
        ),
        cash(("Binance", 1000.0), (None, None)),
    )
    assert pf.symbols == ["BTC", "AAPL"]
    assert pf.assets[1].tags == ("tech", "growth")
    assert pf.assets[0].asset_class is AssetClass.CRYPTO
    assert pf.total_cash == 1000
    assert pf.brokers == ["BIST", "Binance"]


def test_missing_broker_defaults_to_custom() -> None:
    pf = frames_to_portfolio(positions((None, "GC", "commodity", 2, None)), empty_cash())
    assert pf.positions[0].broker == "Custom"


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ((("Binance", "BTC", "crypto", None, None),), "birlikte"),
        ((("Binance", None, "crypto", 1, None),), "birlikte"),
        ((("Binance", "BTC", None, 1, None),), "sınıfı"),
        ((("Binance", "BTC", "bond", 1, None),), "geçersiz"),
        ((("Binance", "BTC", "crypto", -1, None),), "geçersiz"),
        ((), "En az bir pozisyon"),
    ],
)
def test_invalid_positions(rows: tuple[tuple[object, ...], ...], message: str) -> None:
    frame = positions(*rows) if rows else empty_positions()  # type: ignore[arg-type]
    with pytest.raises(PortfolioInputError, match=message):
        frames_to_portfolio(frame, empty_cash())


def test_negative_cash_rejected() -> None:
    with pytest.raises(PortfolioInputError, match="Nakit"):
        frames_to_portfolio(positions(("BIST", "A", "equity", 1, None)), cash(("BIST", -5.0)))


def test_sample_roundtrip() -> None:
    sample = load_sample_portfolio()
    pos_frame, cash_frame = portfolio_to_frames(sample)
    rebuilt = frames_to_portfolio(pos_frame, cash_frame, name=sample.name)
    assert rebuilt.quantities() == sample.quantities()
    assert rebuilt.total_cash == sample.total_cash
    assert {a.symbol: a.tags for a in rebuilt.assets} == {a.symbol: a.tags for a in sample.assets}


def test_broker_options_include_custom_names() -> None:
    pos_frame, cash_frame = portfolio_to_frames(load_sample_portfolio())
    options = broker_options(pos_frame, cash_frame)
    assert options[:4] == ["InteractiveBrokers", "Binance", "BIST", "Custom"]
    assert {"ibkr", "schwab"} <= set(options)


# ---- Streamlit app ------------------------------------------------------------------


def test_app_renders_empty_state() -> None:
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert any("Pozisyonları girin" in i.value for i in at.info)


def test_app_sample_to_report_flow() -> None:
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    at.sidebar.button(key="load_sample").click().run()
    assert not at.exception
    assert len(at.session_state["positions"]) == 4

    at.sidebar.button(key="run_analysis").click().run()
    assert not at.exception
    assert not at.error
    html = at.session_state["report_html"]
    assert "Executive Risk Summary" in html
    assert at.session_state["report_md"].startswith("# Risk Report")
    labels = [m.label for m in at.metric]
    assert "Toplam Portföy Değeri" in labels
    assert "Nakit Oranı" in labels


def test_app_reports_input_errors() -> None:
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    at.sidebar.button(key="run_analysis").click().run()
    assert not at.exception
    assert any("En az bir pozisyon" in e.value for e in at.error)
    assert "analysis" not in at.session_state


# ---- CLI ----------------------------------------------------------------------------


def test_web_command_launches_streamlit(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[list[str]] = []

    def fake_call(command: list[str]) -> int:
        captured.append(command)
        return 0

    monkeypatch.setattr("portfolio_risk.cli.subprocess.call", fake_call)
    result = CliRunner().invoke(app, ["web", "--port", "9000", "--host", "0.0.0.0", "--no-browser"])
    assert result.exit_code == 0, result.output
    (command,) = captured
    assert command[1:5] == ["-m", "streamlit", "run", str(APP)]
    assert command[command.index("--server.port") + 1] == "9000"
    assert command[command.index("--server.headless") + 1] == "true"
    assert command[command.index("--server.address") + 1] == "0.0.0.0"


def test_web_command_propagates_exit_code(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("portfolio_risk.cli.subprocess.call", lambda command: 3)
    assert CliRunner().invoke(app, ["web"]).exit_code == 3
