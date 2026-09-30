"""Streamlit dashboard. Launch with ``pre web`` (or ``streamlit run`` on this file)."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st
from pydantic import ValidationError
from streamlit.elements.lib.column_types import ColumnConfig

from portfolio_risk.catalog import (
    BIST,
    CATEGORIES,
    COMMODITY,
    CRYPTO,
    US,
    entries_for,
    synthetic_profiles,
)
from portfolio_risk.data import (
    CachedProvider,
    DataUnavailableError,
    FxProvider,
    PriceProvider,
    SyntheticFx,
    SyntheticProvider,
    YahooFx,
    YahooProvider,
    convert_to_base,
)
from portfolio_risk.reporting import RiskAnalysis, build_analysis, render_html, render_markdown
from portfolio_risk.risk import CORE_METHODS, Method
from portfolio_risk.web.builder import (
    COL_AMOUNT,
    COL_BROKER,
    COL_CATEGORY,
    COL_CLASS,
    COL_DELETE,
    COL_NAME,
    COL_QTY,
    COL_SYMBOL,
    COL_TAGS,
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
from portfolio_risk.web.i18n import (
    CURRENCIES,
    DEFAULT_LANGUAGE,
    LANGUAGES,
    RTL_LANGUAGES,
    translate,
)
from portfolio_risk.web.numfmt import format_money, format_number, format_percent

SEED = 42
HISTORY_DAYS = 730
DEFAULT_CURRENCY = "USD"
SOURCE_SYNTHETIC = "synthetic"
SOURCE_YAHOO = "yahoo"
SOURCE_KEYS = {SOURCE_SYNTHETIC: "source_synthetic", SOURCE_YAHOO: "source_yahoo"}
CATEGORY_KEYS = {US: "cat_us", BIST: "cat_bist", CRYPTO: "cat_crypto", COMMODITY: "cat_commodity"}


def _t(key: str, **params: object) -> str:
    """Text for the language currently chosen in the sidebar."""
    return translate(str(st.session_state.get("language", DEFAULT_LANGUAGE)), key, **params)


def _error_text(exc: Exception) -> str:
    if isinstance(exc, PortfolioInputError) and exc.key:
        return _t(exc.key, **exc.params)
    return str(exc)


def _sources(source: str) -> tuple[PriceProvider, FxProvider]:
    if source == SOURCE_YAHOO:
        return CachedProvider(YahooProvider(), namespace="yahoo"), YahooFx()
    return SyntheticProvider(seed=SEED, profiles=synthetic_profiles()), SyntheticFx(seed=SEED)


def _init_state() -> None:
    st.session_state.setdefault("positions", empty_positions())
    st.session_state.setdefault("cash", empty_cash())
    st.session_state.setdefault("editor_version", 0)
    st.session_state.setdefault("language", DEFAULT_LANGUAGE)
    st.session_state.setdefault("base_currency", DEFAULT_CURRENCY)
    st.session_state.setdefault("cash_currency", DEFAULT_CURRENCY)  # currency of the cash table


def _reset_editors() -> None:
    st.session_state["editor_version"] += 1  # forces the editors to re-initialise


def _load_sample() -> None:
    try:
        sample = load_sample_portfolio()
        positions, cash = portfolio_to_frames(sample)
        base = str(st.session_state.get("base_currency", sample.base_currency))
        if base != sample.base_currency:  # the sample's cash is in its own currency
            source = str(st.session_state.get("data_source", SOURCE_SYNTHETIC))
            cash[COL_AMOUNT] = cash[COL_AMOUNT].astype(float) * _fx_rate(
                sample.base_currency, base, source
            )
    except FileNotFoundError:
        st.session_state["sample_error"] = ("err_sample_missing", {})
        return
    except PortfolioInputError as exc:  # before ValueError, which it subclasses
        st.session_state["sample_error"] = (exc.key or "", exc.params)
        return
    except (DataUnavailableError, ValueError) as exc:
        st.session_state["sample_error"] = ("", {"raw": str(exc)})
        return
    st.session_state.update(positions=positions, cash=cash, cash_currency=base)
    st.session_state.pop("sample_error", None)
    _reset_editors()


def _fx_rate(old: str, new: str, source: str) -> float:
    """Units of ``new`` per unit of ``old`` today, from the selected data source."""
    _, fx = _sources(source)
    end = date.today()
    return float(fx.get_rates([old], new, end - timedelta(days=30), end)[old].iloc[-1])


def _on_currency_change() -> None:
    """Re-express the cash table in the new currency and ask ``main`` to redo the analysis."""
    new = str(st.session_state["base_currency"])
    old = str(st.session_state.get("cash_currency", new))
    if new == old:
        return
    try:
        rate = _fx_rate(old, new, str(st.session_state.get("data_source", SOURCE_SYNTHETIC)))
    except (DataUnavailableError, ValueError) as exc:
        st.session_state["base_currency"] = old  # keep the old currency: nothing was converted
        st.session_state["currency_error"] = str(exc)
        return
    frame = st.session_state.get("cash_latest", st.session_state["cash"]).copy()
    frame[COL_AMOUNT] = frame[COL_AMOUNT].astype(float) * rate
    st.session_state.update(cash=frame, cash_currency=new, auto_rerun=True)
    st.session_state.pop("currency_error", None)
    _reset_editors()


def _heat(value: object) -> str:
    number = float(value) if isinstance(value, int | float) else 0.0
    rgb = "197,48,48" if number >= 0 else "43,108,176"
    return f"background-color: rgba({rgb},{min(abs(number), 1.0) * 0.5:.2f})"


def _run_analysis(
    positions: pd.DataFrame, cash: pd.DataFrame, simulations: int, source: str, base: str
) -> None:
    """Prices come from the offline synthetic engine or, on request, from cached Yahoo data."""
    try:
        portfolio = frames_to_portfolio(positions, cash, base_currency=base)
        provider, fx = _sources(source)
        end, start = date.today(), date.today() - timedelta(days=HISTORY_DAYS)
        prices = provider.get_prices(portfolio.assets, start, end)
        portfolio, prices = convert_to_base(portfolio, prices, fx, start, end)
        analysis = build_analysis(portfolio, prices, simulations=simulations, seed=SEED)
    except (PortfolioInputError, ValidationError, ValueError, DataUnavailableError) as exc:
        st.session_state.pop("analysis", None)
        st.error(_t("analysis_failed", err=_error_text(exc)))
        return
    st.session_state["analysis"] = analysis
    st.session_state["report_html"] = render_html(analysis)
    st.session_state["report_md"] = render_markdown(analysis)


def _right(label: str) -> ColumnConfig:
    """Text column for pre-formatted numbers (locale separators, currency), right-aligned."""
    return st.column_config.TextColumn(label, alignment="right")


METRIC_CSS = """
<style>
[data-testid="stMetricValue"], [data-testid="stMetricValue"] * {
    white-space: normal !important; overflow: visible !important;
    text-overflow: clip !important; overflow-wrap: anywhere;
}
[data-testid="stMetricValue"] {font-size: 1.6rem; line-height: 1.25;}
</style>
"""


def _show_results(a: RiskAnalysis, confidence: float, horizon: int) -> None:
    lang = str(st.session_state.get("language", DEFAULT_LANGUAGE))
    ccy = a.portfolio.base_currency

    def money(value: float, decimals: int = 2) -> str:
        return format_money(value, ccy, lang, decimals)

    def num(value: float, decimals: int = 2) -> str:
        return format_number(value, lang, decimals)

    def pct(fraction: float, decimals: int = 1) -> str:
        return format_percent(fraction, lang, decimals)

    par = a.var_report(Method.PARAMETRIC, confidence, horizon)
    top = a.top_risk_asset
    conf = f"{confidence * 100:.0f}"

    st.markdown(METRIC_CSS, unsafe_allow_html=True)
    st.subheader(_t("summary_header"))
    c1, c2, c3, c4 = st.columns([3, 2, 2, 3])
    c1.metric(_t("total_value"), money(a.total_value))
    c2.metric(_t("cash_ratio"), pct(a.cash_ratio))
    c3.metric(_t("top_risk"), top.symbol if top else "-")
    c4.metric(
        _t("var_metric", h=horizon, c=conf),
        money(par.var),
        f"CVaR {money(par.cvar)}",
        delta_color="off",
    )

    st.markdown(f"**{_t('varcvar_title', h=horizon, c=conf)}**")
    col_method, col_var, col_cvar, col_div = (
        _t("col_method"),
        _t("col_var"),
        _t("col_cvar"),
        _t("col_div"),
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    col_method: m.value,
                    col_var: money(r.var),
                    col_cvar: money(r.cvar),
                    col_div: pct(r.diversification_ratio),
                }
                for m in CORE_METHODS
                for r in [a.var_report(m, confidence, horizon)]
            ]
        ),
        hide_index=True,
        column_config={
            col_var: _right(col_var),
            col_cvar: _right(col_cvar),
            col_div: _right(col_div),
        },
    )

    left, right = st.columns(2)
    with left:
        st.markdown(f"**{_t('alloc_title')}**")
        col_symbol, col_class, col_value, col_weight = (
            _t("col_symbol"),
            _t("col_class"),
            _t("col_value"),
            _t("col_weight"),
        )
        rows = [
            {
                col_symbol: x.symbol,
                col_class: x.asset_class,
                col_value: money(x.value),
                col_weight: pct(x.weight),
            }
            for x in a.assets
        ]
        if a.cash > 0:
            rows.append(
                {
                    col_symbol: "CASH",
                    col_class: "cash",
                    col_value: money(a.cash),
                    col_weight: pct(a.cash_ratio),
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            column_config={col_value: _right(col_value), col_weight: _right(col_weight)},
        )
    with right:
        st.markdown(f"**{_t('corr_title')}**")
        st.dataframe(
            a.correlation.style.format(
                lambda v: num(v, 2) if isinstance(v, int | float) else str(v)
            ).map(_heat)
        )

    rc = a.contributions
    st.markdown(f"**{_t('contrib_title', h=rc.horizon, c=f'{rc.confidence * 100:.0f}')}**")
    col_pos, col_vc, col_sh, col_cc = (
        _t("col_position"),
        _t("col_var_contrib"),
        _t("col_share"),
        _t("col_cvar_contrib"),
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    col_symbol: str(sym),
                    col_pos: money(rc.exposures[sym]),
                    col_vc: money(rc.component_var[sym]),
                    col_sh: pct(rc.var_share[sym]),
                    col_cc: money(rc.component_cvar[sym]),
                }
                for sym in rc.exposures.index
            ]
        ),
        hide_index=True,
        column_config={c: _right(c) for c in (col_pos, col_vc, col_sh, col_cc)},
    )

    if a.optimizations:
        base = a.optimizations[0].var
        st.markdown(f"**{_t('opt_title')}**")
        col_pf, col_er, col_vol, col_chg = (
            _t("col_portfolio"),
            _t("col_exp_return"),
            _t("col_vol"),
            _t("col_var_change"),
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        col_pf: o.objective,
                        **{sym: pct(w) for sym, w in o.weights.items()},
                        col_er: pct(o.expected_return),
                        col_vol: pct(o.volatility),
                        col_var: money(o.var),
                        col_chg: pct(o.var / base - 1.0 if o.objective != "current" else 0.0),
                    }
                    for o in a.optimizations
                ]
            ),
            hide_index=True,
            column_config={
                c: _right(c)
                for c in (*a.optimizations[0].weights, col_er, col_vol, col_var, col_chg)
            },
        )
        st.caption(_t("opt_caption"))

    if a.backtests:
        first = a.backtests[0]
        st.markdown(f"**{_t('bt_title', c=f'{first.confidence * 100:.0f}', n=first.n_obs)}**")
        col_viol, col_exp, col_kup, col_ind, col_bas = (
            _t("col_violations"),
            _t("col_expected"),
            _t("col_kupiec"),
            _t("col_indep"),
            _t("col_basel"),
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        col_method: r.method.value,
                        col_viol: str(r.n_violations),
                        col_exp: num(r.expected_violations, 1),
                        col_kup: num(r.kupiec.p_value, 3),
                        col_ind: num(r.independence.p_value, 3),
                        col_bas: r.zone.value,
                    }
                    for r in a.backtests
                ]
            ),
            hide_index=True,
            column_config={c: _right(c) for c in (col_viol, col_exp, col_kup, col_ind)},
        )

    st.markdown(f"**{_t('mc_title')}**")
    mc = a.monte_carlo
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(_t("p5"), money(mc.final_percentile(5), 0))
    m2.metric(_t("median"), money(mc.median_final, 0))
    m3.metric(_t("p95"), money(mc.final_percentile(95), 0))
    m4.metric(_t("ruin", p=f"{mc.loss_threshold * 100:.0f}"), pct(mc.prob_ruin, 2))

    st.markdown(f"**{_t('stress_title')}**")
    col_scn, col_pnl, col_lp, col_st = (
        _t("col_scenario"),
        _t("col_pnl"),
        _t("col_loss_pct"),
        _t("col_stressed"),
    )
    st.dataframe(
        pd.DataFrame(
            [
                {
                    col_scn: r.scenario.name,
                    col_pnl: money(r.total_pnl),
                    col_lp: pct(r.pnl_pct),
                    col_st: money(r.stressed_value),
                }
                for r in a.stress.results
            ]
        ),
        hide_index=True,
        column_config={c: _right(c) for c in (col_pnl, col_lp, col_st)},
    )
    worst = a.stress.worst_case
    st.warning(
        _t(
            "worst_case",
            name=worst.scenario.name,
            loss=num(-worst.total_pnl),
            ccy=ccy,
            pct=pct(-worst.pnl_pct),
            rest=num(worst.stressed_value),
        )
    )


def main() -> None:
    st.set_page_config(page_title="Portfolio Risk Engine", page_icon="📉", layout="wide")
    _init_state()
    version = st.session_state["editor_version"]

    with st.sidebar:
        st.selectbox(
            "🌐 Language / Dil",
            list(LANGUAGES),
            format_func=LANGUAGES.__getitem__,
            key="language",
        )
        # format_func closures below must not read session state: AppTest (and Streamlit itself)
        # may call them outside a script run.
        lang = str(st.session_state["language"])
        if lang in RTL_LANGUAGES:
            st.markdown(
                "<style>.stApp, [data-testid='stSidebar'] {direction: rtl;}</style>",
                unsafe_allow_html=True,
            )
        st.selectbox(
            _t("currency_label"),
            list(CURRENCIES),
            format_func=CURRENCIES.__getitem__,
            key="base_currency",
            on_change=_on_currency_change,
        )
        if "currency_error" in st.session_state:
            st.error(_t("currency_failed", err=st.session_state["currency_error"]))
        base_currency = str(st.session_state["base_currency"])

        st.header(_t("params_header"))
        st.button(_t("load_sample"), key="load_sample", on_click=_load_sample)
        if "sample_error" in st.session_state:
            error_key, error_params = st.session_state["sample_error"]
            st.error(_t(error_key, **error_params) if error_key else str(error_params.get("raw")))
        confidence = st.radio(
            _t("confidence"),
            [0.95, 0.99],
            format_func=lambda c: f"%{c * 100:.0f}" if lang == "tr" else f"{c * 100:.0f}%",
            horizontal=True,
            key="confidence",
        )
        horizon = st.radio(
            _t("horizon"),
            [1, 10],
            format_func=lambda h: translate(lang, "horizon_option", n=h),
            horizontal=True,
            key="horizon",
        )
        simulations = st.radio(
            _t("simulations"),
            [1000, 5000],
            format_func=lambda n: f"{n:,}",
            horizontal=True,
            key="simulations",
        )
        source = st.radio(
            _t("data_source"),
            list(SOURCE_KEYS),
            format_func=lambda s: translate(lang, SOURCE_KEYS[s]),
            key="data_source",
        )
        st.caption(_t("yahoo_caption" if source == SOURCE_YAHOO else "synthetic_caption"))
        st.markdown(f"**{_t('cash_header')}**")
        cash = st.data_editor(
            st.session_state["cash"],
            key=f"cash_editor_{version}",
            num_rows="dynamic",
            hide_index=True,
            column_config={
                COL_BROKER: st.column_config.SelectboxColumn(
                    _t("cash_broker"), options=broker_options(st.session_state["cash"])
                ),
                COL_AMOUNT: st.column_config.NumberColumn(
                    _t("cash_amount", ccy=base_currency), min_value=0.0, format="%.2f"
                ),
            },
        )
        st.session_state["cash_latest"] = cash  # what the table shows now (for a currency change)
        run = st.button(_t("run_button"), key="run_analysis", type="primary")

    st.title("Portfolio Risk Engine")
    st.caption(_t("subtitle"))

    st.markdown(f"**{_t('add_header')}**")
    c_cat, c_asset, c_qty, c_add = st.columns([2, 4, 2, 1], vertical_alignment="bottom")
    category = c_cat.selectbox(
        _t("category_label"),
        CATEGORIES,
        format_func=lambda c: translate(lang, CATEGORY_KEYS[c]),
        key="add_category",
    )
    entries = entries_for(category)
    entry = c_asset.selectbox(
        _t("asset_label"),
        entries,
        format_func=lambda e: e.label,
        key=f"add_asset_{category}",
    )
    quantity = c_qty.number_input(
        _t("qty_label"), min_value=0.0, value=1.0, step=1.0, format="%g", key="add_qty"
    )
    add_clicked = c_add.button(_t("add_button"), key="add_position", type="primary")
    if entry is not None:
        st.caption(
            _t(
                "asset_info",
                cls=entry.asset_class.value,
                tags=", ".join(entry.tags),
                broker=entry.broker,
            )
        )

    st.markdown(f"**{_t('positions_header')}**")
    positions = st.data_editor(
        st.session_state["positions"],
        key=f"positions_editor_{version}",
        num_rows="fixed",
        hide_index=True,
        width="stretch",
        disabled=[COL_CATEGORY, COL_SYMBOL, COL_NAME, COL_CLASS, COL_TAGS],
        column_config={
            COL_DELETE: st.column_config.CheckboxColumn(_t("col_delete"), default=False),
            COL_CATEGORY: st.column_config.TextColumn(_t("col_category")),
            COL_SYMBOL: st.column_config.TextColumn(_t("col_symbol")),
            COL_NAME: st.column_config.TextColumn(_t("col_name")),
            COL_CLASS: st.column_config.TextColumn(_t("col_class")),
            COL_TAGS: st.column_config.TextColumn(_t("col_tags")),
            COL_QTY: st.column_config.NumberColumn(_t("col_qty"), min_value=0.0, format="%.6g"),
        },
    )
    delete_clicked = st.button(_t("delete_selected"), key="delete_positions")

    if add_clicked and entry is not None:
        try:
            st.session_state["positions"] = add_position(positions, entry, float(quantity))
        except PortfolioInputError as exc:
            st.error(_error_text(exc))
        else:
            _reset_editors()
            st.rerun()
    if delete_clicked:
        st.session_state["positions"] = remove_marked(positions)
        _reset_editors()
        st.rerun()

    auto = bool(st.session_state.pop("auto_rerun", False)) and "analysis" in st.session_state
    if run or auto:  # a currency change re-runs the analysis without pressing the button
        _run_analysis(positions, cash, simulations, source, base_currency)

    analysis = st.session_state.get("analysis")
    if analysis is None:
        st.info(_t("info_start"))
        return
    _show_results(analysis, confidence, horizon)

    st.divider()
    d1, d2 = st.columns(2)
    d1.download_button(
        _t("download_html"),
        data=st.session_state["report_html"],
        file_name="risk-report.html",
        mime="text/html",
        key="download_html",
    )
    d2.download_button(
        _t("download_md"),
        data=st.session_state["report_md"],
        file_name="risk-report.md",
        mime="text/markdown",
        key="download_md",
    )
    with st.expander(_t("preview")):
        st.iframe(st.session_state["report_html"], height=900)  # our own escaped HTML


main()
