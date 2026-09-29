from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest
from typer.testing import CliRunner

from portfolio_risk import cli
from portfolio_risk.catalog import get_entry
from portfolio_risk.cli import app
from portfolio_risk.models import AssetClass
from portfolio_risk.web.builder import (
    COL_AMOUNT,
    COL_BROKER,
    COL_DELETE,
    COL_QTY,
    COL_SYMBOL,
    PortfolioInputError,
    add_position,
    broker_options,
    empty_cash,
    empty_positions,
    frames_to_portfolio,
    load_sample_portfolio,
    portfolio_to_frames,
    remove_marked,
)

APP = Path(cli.__file__).parent / "web" / "app.py"


def cash(*rows: tuple[object, object]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=[COL_BROKER, COL_AMOUNT])


def table(*items: tuple[str, float]) -> pd.DataFrame:
    frame = empty_positions()
    for symbol, qty in items:
        frame = add_position(frame, get_entry(symbol), qty)
    return frame


# ---- builder ------------------------------------------------------------------------


def test_add_position_links_catalog_details() -> None:
    frame = table(("THYAO", 100))
    row = frame.iloc[0]
    assert row[COL_SYMBOL] == "THYAO"
    assert row["Varlık"] == "Türk Hava Yolları"
    assert row["Sınıf"] == "equity"
    assert "aviation" in row["Etiketler"]
    assert row[COL_QTY] == 100
    assert not row[COL_DELETE]


def test_adding_same_asset_merges_quantity() -> None:
    frame = table(("BTC", 1), ("ETH", 2), ("BTC", 0.5))
    assert list(frame[COL_SYMBOL]) == ["BTC", "ETH"]
    assert frame[COL_QTY].tolist() == [1.5, 2]


def test_add_position_rejects_non_positive_quantity() -> None:
    with pytest.raises(PortfolioInputError, match="sıfırdan"):
        add_position(empty_positions(), get_entry("BTC"), 0)


def test_remove_marked_rows() -> None:
    frame = table(("AAPL", 1), ("BTC", 1), ("GOLD", 1))
    frame.loc[frame[COL_SYMBOL] == "BTC", COL_DELETE] = True
    assert list(remove_marked(frame)[COL_SYMBOL]) == ["AAPL", "GOLD"]
    assert remove_marked(empty_positions()).empty


def test_frames_to_portfolio_uses_catalog_and_category_brokers() -> None:
    frame = table(("THYAO", 100), ("SOL", 3), ("AAPL", 5), ("GOLD", 2))
    pf = frames_to_portfolio(frame, cash(("Custom", 250.0), (None, None)))
    assert pf.symbols == ["THYAO", "SOL", "AAPL", "GOLD"]
    by = {p.asset.symbol: p for p in pf.positions}
    assert by["THYAO"].broker == "BIST"
    assert by["SOL"].broker == "Binance"
    assert by["AAPL"].broker == "InteractiveBrokers"
    assert by["SOL"].asset.asset_class is AssetClass.CRYPTO
    assert by["AAPL"].asset.tags[:2] == ("tech", "growth")
    assert by["GOLD"].asset.data_symbol == "gc.f"
    assert pf.total_cash == 250


def test_blank_rows_ignored_and_bad_rows_rejected() -> None:
    frame = table(("AAPL", 1))
    frame.loc[len(frame)] = [False, None, None, None, None, None, None]
    assert frames_to_portfolio(frame, empty_cash()).symbols == ["AAPL"]

    bad_qty = table(("AAPL", 1))
    bad_qty[COL_QTY] = -3.0
    with pytest.raises(PortfolioInputError, match="miktar"):
        frames_to_portfolio(bad_qty, empty_cash())
    bad_qty[COL_QTY] = float("nan")
    with pytest.raises(PortfolioInputError, match="miktar"):
        frames_to_portfolio(bad_qty, empty_cash())

    unknown = table(("AAPL", 1))
    unknown[COL_SYMBOL] = "NOPE"
    with pytest.raises(PortfolioInputError, match="katalogda yok"):
        frames_to_portfolio(unknown, empty_cash())

    with pytest.raises(PortfolioInputError, match="En az bir pozisyon"):
        frames_to_portfolio(empty_positions(), empty_cash())
    with pytest.raises(PortfolioInputError, match="Nakit"):
        frames_to_portfolio(table(("AAPL", 1)), cash(("BIST", -5.0)))


def test_sample_roundtrip_merges_brokers_by_symbol() -> None:
    sample = load_sample_portfolio()
    positions, cash_frame = portfolio_to_frames(sample)
    assert list(positions[COL_SYMBOL]) == ["AAPL", "BTC", "GOLD"]
    assert positions[COL_QTY].tolist() == [70, 0.2, 3]  # AAPL 50 + 20 across brokers
    rebuilt = frames_to_portfolio(positions, cash_frame)
    assert rebuilt.quantities() == sample.quantities()
    assert rebuilt.total_cash == sample.total_cash


def test_broker_options_include_custom_names() -> None:
    _, cash_frame = portfolio_to_frames(load_sample_portfolio())
    options = broker_options(cash_frame)
    assert options[:4] == ["InteractiveBrokers", "Binance", "BIST", "Custom"]
    assert "ibkr" in options


# ---- Streamlit app ------------------------------------------------------------------


def new_app() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=120).run()


def test_app_renders_empty_state() -> None:
    at = new_app()
    assert not at.exception
    assert any("Pozisyon ekleyin" in i.value for i in at.info)
    assert at.selectbox(key="add_category").options == ["ABD Hisseleri", "BIST", "Kripto", "Emtia"]


def test_asset_dropdown_follows_category() -> None:
    at = new_app()
    us = at.selectbox(key="add_asset_ABD Hisseleri")
    assert any(o.startswith("AAPL - Apple") for o in us.options)
    at.selectbox(key="add_category").select("BIST").run()
    bist = at.selectbox(key="add_asset_BIST")
    assert any(o.startswith("THYAO - Türk Hava Yolları") for o in bist.options)
    assert not any(o.startswith("AAPL") for o in bist.options)
    assert not at.exception


def test_add_delete_and_analyse_from_dropdowns() -> None:
    at = new_app()
    at.selectbox(key="add_category").select("BIST").run()
    at.selectbox(key="add_asset_BIST").select_index(0).run()
    at.number_input(key="add_qty").set_value(250).run()
    at.button(key="add_position").click().run()
    assert not at.exception
    positions = at.session_state["positions"]
    assert list(positions[COL_SYMBOL]) == ["THYAO"]
    assert positions[COL_QTY].tolist() == [250]

    at.selectbox(key="add_category").select("Kripto").run()
    at.selectbox(key="add_asset_Kripto").select_index(2).run()  # SOL
    at.button(key="add_position").click().run()
    assert list(at.session_state["positions"][COL_SYMBOL]) == ["THYAO", "SOL"]

    at.sidebar.button(key="run_analysis").click().run()
    assert not at.exception
    assert not at.error
    assert "Executive Risk Summary" in at.session_state["report_html"]
    assert "THYAO" in at.session_state["report_html"]


def test_app_sample_to_report_flow() -> None:
    at = new_app()
    at.sidebar.button(key="load_sample").click().run()
    assert not at.exception
    assert len(at.session_state["positions"]) == 3

    at.sidebar.button(key="run_analysis").click().run()
    assert not at.exception
    assert not at.error
    assert "Executive Risk Summary" in at.session_state["report_html"]
    assert at.session_state["report_md"].startswith("# Risk Report")
    labels = [m.label for m in at.metric]
    assert "Toplam Portföy Değeri" in labels
    assert "Nakit Oranı" in labels
    # No provider choice is exposed any more.
    assert all("Fiyat Verisi" not in s.label for s in at.sidebar.selectbox)


def test_app_reports_empty_portfolio_error() -> None:
    at = new_app()
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


def test_web_command_defaults_to_localhost_and_propagates_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def fake_call(command: list[str]) -> int:
        captured.append(command)
        return 3

    monkeypatch.setattr("portfolio_risk.cli.subprocess.call", fake_call)
    assert CliRunner().invoke(app, ["web"]).exit_code == 3
    assert captured[0][captured[0].index("--server.address") + 1] == "localhost"
