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


def _reset_editors() -> None:
    st.session_state["editor_version"] += 1  # forces the editors to re-initialise


def _load_sample() -> None:
    try:
        positions, cash = portfolio_to_frames(load_sample_portfolio())
    except FileNotFoundError:
        st.session_state["sample_error"] = ("err_sample_missing", {})
        return
    except PortfolioInputError as exc:
        st.session_state["sample_error"] = (exc.key or "", exc.params)
        return
    except ValidationError as exc:
        st.session_state["sample_error"] = ("", {"raw": str(exc)})
        return
    st.session_state.update(positions=positions, cash=cash)
    st.session_state.pop("sample_error", None)
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


def _pct(label: str) -> ColumnConfig:
    return st.column_config.NumberColumn(label, format="percent")


def _num(label: str, fmt: str = "%.2f") -> ColumnConfig:
    return st.column_config.NumberColumn(label, format=fmt)


def _show_results(a: RiskAnalysis, confidence: float, horizon: int) -> None:
    ccy = a.portfolio.base_currency
    par = a.var_report(Method.PARAMETRIC, confidence, horizon)
    top = a.top_risk_asset
    conf = f"{confidence * 100:.0f}"

    st.subheader(_t("summary_header"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric(_t("total_value"), f"{a.total_value:,.2f} {ccy}")
    c2.metric(_t("cash_ratio"), f"{a.cash_ratio:.1%}")
    c3.metric(_t("top_risk"), top.symbol if top else "-")
    c4.metric(
        _t("var_metric", h=horizon, c=conf),
        f"{par.var:,.2f}",
        f"CVaR {par.cvar:,.2f}",
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
                    col_var: r.var,
                    col_cvar: r.cvar,
                    col_div: r.diversification_ratio,
                }
                for m in CORE_METHODS
                for r in [a.var_report(m, confidence, horizon)]
            ]
        ),
        hide_index=True,
        column_config={col_var: _num(col_var), col_cvar: _num(col_cvar), col_div: _pct(col_div)},
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
                col_value: x.value,
                col_weight: x.weight,
            }
            for x in a.assets
        ]
        if a.cash > 0:
            rows.append(
                {
                    col_symbol: "CASH",
                    col_class: "cash",
                    col_value: a.cash,
                    col_weight: a.cash_ratio,
                }
            )
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            column_config={col_value: _num(col_value), col_weight: _pct(col_weight)},
        )
    with right:
        st.markdown(f"**{_t('corr_title')}**")
        st.dataframe(a.correlation.style.format("{:.2f}").map(_heat))

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
                    col_pos: rc.exposures[sym],
                    col_vc: rc.component_var[sym],
                    col_sh: rc.var_share[sym],
                    col_cc: rc.component_cvar[sym],
                }
                for sym in rc.exposures.index
            ]
        ),
        hide_index=True,
        column_config={
            col_pos: _num(col_pos),
            col_vc: _num(col_vc),
            col_sh: _pct(col_sh),
            col_cc: _num(col_cc),
        },
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
                        **o.weights,
                        col_er: o.expected_return,
                        col_vol: o.volatility,
                        col_var: o.var,
                        col_chg: o.var / base - 1.0 if o.objective != "current" else 0.0,
                    }
                    for o in a.optimizations
                ]
            ),
            hide_index=True,
            column_config={
                **{
                    sym: st.column_config.NumberColumn(format="percent")
                    for sym in a.optimizations[0].weights
                },
                col_er: _pct(col_er),
                col_vol: _pct(col_vol),
                col_var: _num(col_var),
                col_chg: _pct(col_chg),
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
                        col_viol: r.n_violations,
                        col_exp: r.expected_violations,
                        col_kup: r.kupiec.p_value,
                        col_ind: r.independence.p_value,
                        col_bas: r.zone.value,
                    }
                    for r in a.backtests
                ]
            ),
            hide_index=True,
            column_config={
                col_exp: _num(col_exp, "%.1f"),
                col_kup: _num(col_kup, "%.3f"),
                col_ind: _num(col_ind, "%.3f"),
            },
        )

    st.markdown(f"**{_t('mc_title')}**")
    mc = a.monte_carlo
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(_t("p5"), f"{mc.final_percentile(5):,.0f}")
    m2.metric(_t("median"), f"{mc.median_final:,.0f}")
    m3.metric(_t("p95"), f"{mc.final_percentile(95):,.0f}")
    m4.metric(_t("ruin", p=f"{mc.loss_threshold * 100:.0f}"), f"{mc.prob_ruin:.2%}")

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
                    col_pnl: r.total_pnl,
                    col_lp: r.pnl_pct,
                    col_st: r.stressed_value,
                }
                for r in a.stress.results
            ]
        ),
        hide_index=True,
        column_config={col_pnl: _num(col_pnl), col_lp: _pct(col_lp), col_st: _num(col_st)},
    )
    worst = a.stress.worst_case
    st.warning(
        _t(
            "worst_case",
            name=worst.scenario.name,
            loss=f"{-worst.total_pnl:,.2f}",
            ccy=ccy,
            pct=f"{-worst.pnl_pct:.1%}",
            rest=f"{worst.stressed_value:,.2f}",
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
        )
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

    if run:
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
