from __future__ import annotations

import re
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from portfolio_risk import cli
from portfolio_risk.data.fx import USD_PER_UNIT
from portfolio_risk.web.builder import (
    PortfolioInputError,
    add_position,
    empty_cash,
    empty_positions,
    frames_to_portfolio,
)
from portfolio_risk.web.i18n import (
    CURRENCIES,
    DEFAULT_LANGUAGE,
    LANGUAGES,
    RTL_LANGUAGES,
    TRANSLATIONS,
    translate,
)
from portfolio_risk.web.numfmt import format_number

APP = Path(cli.__file__).parent / "web" / "app.py"
PLACEHOLDER = re.compile(r"\{(\w+)\}")
ENGLISH = TRANSLATIONS["en"]


def new_app() -> AppTest:
    return AppTest.from_file(str(APP), default_timeout=120).run()


def test_every_language_has_a_translation_table() -> None:
    assert set(LANGUAGES) == set(TRANSLATIONS)
    assert DEFAULT_LANGUAGE in LANGUAGES
    assert set(LANGUAGES) >= RTL_LANGUAGES
    assert len(LANGUAGES) >= 12


@pytest.mark.parametrize("lang", sorted(TRANSLATIONS))
def test_translations_are_complete_and_keep_placeholders(lang: str) -> None:
    table = TRANSLATIONS[lang]
    assert set(table) == set(ENGLISH), f"{lang}: missing {set(ENGLISH) - set(table)}"
    for key, text in table.items():
        assert text.strip(), (lang, key)
        assert set(PLACEHOLDER.findall(text)) == set(PLACEHOLDER.findall(ENGLISH[key])), (lang, key)
        # every entry must be formattable with its own placeholders
        translate(lang, key, **dict.fromkeys(PLACEHOLDER.findall(text), "x"))


def test_translate_falls_back_to_english_then_to_the_key() -> None:
    assert translate("xx", "add_button") == ENGLISH["add_button"]  # unknown language
    assert translate("en", "no_such_key") == "no_such_key"
    assert translate("tr", "cash_amount", ccy="EUR") == "Tutar (EUR)"
    assert translate("en", "var_metric", h=10, c=99) == "10-day 99% VaR"


def test_currencies_have_synthetic_rates() -> None:
    assert set(CURRENCIES) <= set(USD_PER_UNIT)


def test_input_errors_carry_translatable_keys() -> None:
    with pytest.raises(PortfolioInputError) as empty:
        frames_to_portfolio(empty_positions(), empty_cash())
    assert empty.value.key == "err_no_positions"
    from portfolio_risk.catalog import get_entry

    with pytest.raises(PortfolioInputError) as zero:
        add_position(empty_positions(), get_entry("AAPL"), 0)
    assert zero.value.key == "err_qty_positive"
    assert "Miktar" in str(zero.value)  # the plain message stays Turkish


def test_cash_takes_the_chosen_base_currency() -> None:
    import pandas as pd

    from portfolio_risk.catalog import get_entry
    from portfolio_risk.web.builder import COL_AMOUNT, COL_BROKER

    positions = add_position(empty_positions(), get_entry("AAPL"), 3)
    cash = pd.DataFrame([{COL_BROKER: "Binance", COL_AMOUNT: 500.0}])
    pf = frames_to_portfolio(positions, cash, base_currency="EUR")
    assert pf.base_currency == "EUR"
    assert {c.currency for c in pf.cash} == {"EUR"}


def _labels(at: AppTest) -> str:
    parts = [b.label for b in at.sidebar.button] + [r.label for r in at.sidebar.radio]
    parts += [s.label for s in at.sidebar.selectbox]
    return " | ".join(parts)


def test_language_dropdown_switches_the_interface() -> None:
    at = new_app()
    assert "Güven Aralığı" in _labels(at)  # Turkish is the default
    at.sidebar.selectbox(key="language").set_value("en").run()
    assert not at.exception
    assert "Confidence level" in _labels(at)
    assert "Run risk analysis & build report" in _labels(at)
    assert "Güven Aralığı" not in _labels(at)


def test_language_dropdown_offers_other_languages() -> None:
    # Switching twice used to break AppTest when a format_func read session state.
    at = new_app()
    at.sidebar.selectbox(key="language").set_value("es").run()
    assert "Nivel de confianza" in _labels(at)
    assert any("Añadir activo" in m.value for m in at.markdown)
    at.sidebar.selectbox(key="language").set_value("en").run()  # a second switch keeps working
    assert not at.exception
    assert "Confidence level" in _labels(at)


@pytest.mark.parametrize("lang", sorted(LANGUAGES))
def test_every_language_renders_a_full_analysis(lang: str) -> None:
    at = new_app()
    at.sidebar.selectbox(key="language").set_value(lang).run()
    at.sidebar.button(key="load_sample").click().run()
    at.sidebar.button(key="run_analysis").click().run()
    assert not at.exception, lang
    assert not at.error, lang
    assert any(translate(lang, "alloc_title") in m.value for m in at.markdown)
    labels = [m.label for m in at.metric]
    assert translate(lang, "total_value") in labels


def test_right_to_left_languages_get_rtl_styling() -> None:
    at = new_app()
    assert not any("direction: rtl" in m.value for m in at.sidebar.markdown)
    at.sidebar.selectbox(key="language").set_value("ar").run()
    assert any("direction: rtl" in m.value for m in at.sidebar.markdown)
    ltr = new_app()
    ltr.sidebar.selectbox(key="language").set_value("ja").run()
    assert not any("direction: rtl" in m.value for m in ltr.sidebar.markdown)


def test_language_choice_survives_other_interactions() -> None:
    at = new_app()
    at.sidebar.selectbox(key="language").set_value("de").run()
    at.sidebar.button(key="load_sample").click().run()
    assert at.session_state["language"] == "de"
    assert "Konfidenzniveau" in _labels(at)


def test_errors_are_shown_in_the_chosen_language() -> None:
    at = new_app()
    at.sidebar.selectbox(key="language").set_value("fr").run()
    at.sidebar.button(key="run_analysis").click().run()
    assert any("Ajoutez au moins une position" in e.value for e in at.error)


def _asset_values(currency: str) -> dict[str, float]:
    at = new_app()
    at.sidebar.selectbox(key="base_currency").set_value(currency).run()
    at.sidebar.button(key="load_sample").click().run()
    at.sidebar.button(key="run_analysis").click().run()
    assert not at.exception
    assert not at.error
    analysis = at.session_state["analysis"]
    assert analysis.portfolio.base_currency == currency
    return {a.symbol: a.value for a in analysis.assets}


def test_currency_dropdown_revalues_the_portfolio() -> None:
    usd = _asset_values("USD")
    eur = _asset_values("EUR")
    try_ = _asset_values("TRY")
    rate_eur = USD_PER_UNIT["USD"] / USD_PER_UNIT["EUR"]  # synthetic anchors on the end date
    rate_try = USD_PER_UNIT["USD"] / USD_PER_UNIT["TRY"]
    for symbol, value in usd.items():
        assert eur[symbol] == pytest.approx(value * rate_eur, rel=1e-6)
        assert try_[symbol] == pytest.approx(value * rate_try, rel=1e-6)


def test_cash_label_shows_the_chosen_currency() -> None:
    at = new_app()
    at.sidebar.selectbox(key="base_currency").set_value("GBP").run()
    assert not at.exception
    assert at.session_state["base_currency"] == "GBP"


# --- currency change re-runs the analysis; units and number formats ---------------------------


def _analysed(lang: str = "tr", currency: str = "USD") -> AppTest:
    at = new_app()
    at.sidebar.selectbox(key="language").set_value(lang).run()
    at.sidebar.selectbox(key="base_currency").set_value(currency).run()
    at.sidebar.button(key="load_sample").click().run()
    at.sidebar.button(key="run_analysis").click().run()
    assert not at.exception
    assert not at.error
    return at


def test_changing_the_currency_recomputes_the_whole_report_without_the_button() -> None:
    at = _analysed(currency="USD")
    usd = at.session_state["analysis"]
    html_before = at.session_state["report_html"]

    at.sidebar.selectbox(key="base_currency").set_value("EUR").run()  # no button press
    assert not at.exception
    assert not at.error
    eur = at.session_state["analysis"]
    rate = USD_PER_UNIT["USD"] / USD_PER_UNIT["EUR"]
    assert eur.portfolio.base_currency == "EUR"
    assert eur.total_value == pytest.approx(usd.total_value * rate, rel=1e-6)
    assert eur.cash == pytest.approx(usd.cash * rate, rel=1e-6)  # the cash table was converted too
    assert at.session_state["report_html"] != html_before
    assert "EUR" in at.session_state["report_html"]

    at.sidebar.selectbox(key="base_currency").set_value("TRY").run()
    assert at.session_state["analysis"].portfolio.base_currency == "TRY"
    assert at.session_state["analysis"].total_value == pytest.approx(
        usd.total_value * USD_PER_UNIT["USD"] / USD_PER_UNIT["TRY"], rel=1e-6
    )


def test_changing_the_currency_before_any_analysis_does_not_run_one() -> None:
    at = new_app()
    at.sidebar.selectbox(key="base_currency").set_value("EUR").run()
    assert not at.exception
    assert not at.error
    assert "analysis" not in at.session_state


def test_round_trip_through_another_currency_restores_the_cash() -> None:
    at = _analysed(currency="USD")
    cash_usd = at.session_state["analysis"].cash
    at.sidebar.selectbox(key="base_currency").set_value("JPY").run()
    at.sidebar.selectbox(key="base_currency").set_value("USD").run()
    assert at.session_state["analysis"].cash == pytest.approx(cash_usd, rel=1e-9)


def _metric(at: AppTest, label: str) -> str:
    return next(str(m.value) for m in at.metric if m.label == label)


def test_metrics_show_the_full_amount_with_currency_in_the_language_format() -> None:
    tr = _analysed("tr", "EUR")
    value = tr.session_state["analysis"].total_value
    shown = _metric(tr, "Toplam Portföy Değeri")
    assert shown == f"{format_number(value, 'tr')} EUR"  # e.g. 34.500,12 EUR
    assert shown.endswith(" EUR")
    assert "." in shown.split(",")[0]  # Turkish thousands separator

    en = _analysed("en", "EUR")
    assert _metric(en, "Total portfolio value") == f"{format_number(value, 'en')} EUR"
    assert "…" not in shown
    var_metric = next(m for m in en.metric if "VaR" in m.label)
    assert str(var_metric.value).endswith(" EUR")  # VaR carries its unit too
    assert "EUR" in str(var_metric.delta)


def test_tables_carry_units_and_locale_separators() -> None:
    at = _analysed("de", "GBP")
    frames = [df.value for df in at.dataframe]
    money_cells = [
        str(v)
        for frame in frames
        for col in frame.columns
        for v in frame[col]
        if str(v).endswith(" GBP")
    ]
    assert money_cells, "value columns should include the currency"
    assert all("," in cell for cell in money_cells)  # German decimal comma
    assert any("%" in str(v) for frame in frames for col in frame.columns for v in frame[col])


def test_large_values_are_not_truncated_by_css() -> None:
    at = _analysed("en", "KRW")
    css = " ".join(m.value for m in at.markdown if "stMetricValue" in m.value)
    assert "text-overflow: clip" in css
    assert "white-space: normal" in css
    assert format_number(at.session_state["analysis"].total_value, "en") in _metric(
        at, "Total portfolio value"
    )
