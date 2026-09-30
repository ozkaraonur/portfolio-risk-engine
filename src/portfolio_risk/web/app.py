"""Streamlit dashboard. Launch with ``pre web`` (or ``streamlit run`` on this file)."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from portfolio_risk.catalog import CATEGORIES, entries_for, synthetic_profiles
from portfolio_risk.data import SyntheticProvider
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

SEED = 42
HISTORY_DAYS = 730


def _init_state() -> None:
    st.session_state.setdefault("positions", empty_positions())
    st.session_state.setdefault("cash", empty_cash())
    st.session_state.setdefault("editor_version", 0)


def _reset_editors() -> None:
    st.session_state["editor_version"] += 1  # forces the editors to re-initialise


def _load_sample() -> None:
    try:
        positions, cash = portfolio_to_frames(load_sample_portfolio())
    except (FileNotFoundError, ValidationError, PortfolioInputError) as exc:
        st.session_state["sample_error"] = str(exc)
        return
    st.session_state.update(positions=positions, cash=cash)
    st.session_state.pop("sample_error", None)
    _reset_editors()


def _heat(value: object) -> str:
    number = float(value) if isinstance(value, int | float) else 0.0
    rgb = "197,48,48" if number >= 0 else "43,108,176"
    return f"background-color: rgba({rgb},{min(abs(number), 1.0) * 0.5:.2f})"


def _run_analysis(positions: pd.DataFrame, cash: pd.DataFrame, simulations: int) -> None:
    """Prices always come from the synthetic engine (catalog-calibrated, offline)."""
    try:
        portfolio = frames_to_portfolio(positions, cash)
        end = date.today()
        prices = SyntheticProvider(seed=SEED, profiles=synthetic_profiles()).get_prices(
            portfolio.assets, end - timedelta(days=HISTORY_DAYS), end
        )
        analysis = build_analysis(portfolio, prices, simulations=simulations, seed=SEED)
    except (PortfolioInputError, ValidationError, ValueError) as exc:
        st.session_state.pop("analysis", None)
        st.error(f"Analiz yapılamadı: {exc}")
        return
    st.session_state["analysis"] = analysis
    st.session_state["report_html"] = render_html(analysis)
    st.session_state["report_md"] = render_markdown(analysis)


def _show_results(a: RiskAnalysis, confidence: float, horizon: int) -> None:
    ccy = a.portfolio.base_currency
    par = a.var_report(Method.PARAMETRIC, confidence, horizon)
    top = a.top_risk_asset

    st.subheader("Yönetici Risk Özeti")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Toplam Portföy Değeri", f"{a.total_value:,.2f} {ccy}")
    c2.metric("Nakit Oranı", f"{a.cash_ratio:.1%}")
    c3.metric("En Yüksek Riskli Varlık", top.symbol if top else "-")
    c4.metric(
        f"{horizon} Günlük %{confidence * 100:.0f} VaR",
        f"{par.var:,.2f}",
        f"CVaR {par.cvar:,.2f}",
        delta_color="off",
    )

    st.markdown(f"**VaR / CVaR** ({horizon} gün, %{confidence * 100:.0f})")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Yöntem": m.value,
                    "VaR": r.var,
                    "CVaR": r.cvar,
                    "Çeşitlendirme faydası": r.diversification_ratio,
                }
                for m in CORE_METHODS
                for r in [a.var_report(m, confidence, horizon)]
            ]
        ),
        hide_index=True,
        column_config={
            "VaR": st.column_config.NumberColumn(format="%.2f"),
            "CVaR": st.column_config.NumberColumn(format="%.2f"),
            "Çeşitlendirme faydası": st.column_config.NumberColumn(format="percent"),
        },
    )

    left, right = st.columns(2)
    with left:
        st.markdown("**Varlık Dağılımı**")
        rows = [
            {"Sembol": x.symbol, "Sınıf": x.asset_class, "Değer": x.value, "Ağırlık": x.weight}
            for x in a.assets
        ]
        if a.cash > 0:
            rows.append(
                {"Sembol": "CASH", "Sınıf": "cash", "Değer": a.cash, "Ağırlık": a.cash_ratio}
            )
        st.dataframe(
            pd.DataFrame(rows),
            hide_index=True,
            column_config={
                "Değer": st.column_config.NumberColumn(format="%.2f"),
                "Ağırlık": st.column_config.NumberColumn(format="percent"),
            },
        )
    with right:
        st.markdown("**Korelasyon Isı Haritası**")
        st.dataframe(a.correlation.style.format("{:.2f}").map(_heat))

    if a.backtests:
        first = a.backtests[0]
        st.markdown(
            f"**Model Doğrulama** (VaR geriye dönük test, %{first.confidence * 100:.0f}, "
            f"{first.n_obs} gün)"
        )
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Yöntem": r.method.value,
                        "İhlal": r.n_violations,
                        "Beklenen": r.expected_violations,
                        "Kupiec p": r.kupiec.p_value,
                        "Bağımsızlık p": r.independence.p_value,
                        "Basel bölgesi": r.zone.value,
                    }
                    for r in a.backtests
                ]
            ),
            hide_index=True,
            column_config={
                "Beklenen": st.column_config.NumberColumn(format="%.1f"),
                "Kupiec p": st.column_config.NumberColumn(format="%.3f"),
                "Bağımsızlık p": st.column_config.NumberColumn(format="%.3f"),
            },
        )

    st.markdown("**Monte Carlo** (1 yıl)")
    mc = a.monte_carlo
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("5. Persentil", f"{mc.final_percentile(5):,.0f}")
    m2.metric("Medyan", f"{mc.median_final:,.0f}")
    m3.metric("95. Persentil", f"{mc.final_percentile(95):,.0f}")
    m4.metric(f"İflas olasılığı (-%{mc.loss_threshold * 100:.0f})", f"{mc.prob_ruin:.2%}")

    st.markdown("**Stres Testi Özeti**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Senaryo": r.scenario.name,
                    "K/Z": r.total_pnl,
                    "Kayıp %": r.pnl_pct,
                    "Stres sonrası değer": r.stressed_value,
                }
                for r in a.stress.results
            ]
        ),
        hide_index=True,
        column_config={
            "K/Z": st.column_config.NumberColumn(format="%.2f"),
            "Kayıp %": st.column_config.NumberColumn(format="percent"),
            "Stres sonrası değer": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    worst = a.stress.worst_case
    st.warning(
        f"En kötü senaryo: **{worst.scenario.name}**, kayıp {-worst.total_pnl:,.2f} {ccy} "
        f"({-worst.pnl_pct:.1%}); kalan sermaye {worst.stressed_value:,.2f} {ccy}."
    )


def main() -> None:
    st.set_page_config(page_title="Portfolio Risk Engine", page_icon="📉", layout="wide")
    _init_state()
    version = st.session_state["editor_version"]

    with st.sidebar:
        st.header("Analiz Parametreleri")
        st.button(
            "Örnek Portföyü Yükle (Load Sample Portfolio)", key="load_sample", on_click=_load_sample
        )
        if "sample_error" in st.session_state:
            st.error(st.session_state["sample_error"])
        confidence = st.radio(
            "Güven Aralığı",
            [0.95, 0.99],
            format_func=lambda c: f"%{c * 100:.0f}",
            horizontal=True,
        )
        horizon = st.radio("Zaman Ufku", [1, 10], format_func=lambda h: f"{h} gün", horizontal=True)
        simulations = st.radio(
            "Monte Carlo Simülasyon Sayısı",
            [1000, 5000],
            format_func=lambda n: f"{n:,}",
            horizontal=True,
        )
        st.markdown("**Nakit Bakiyesi**")
        cash = st.data_editor(
            st.session_state["cash"],
            key=f"cash_editor_{version}",
            num_rows="dynamic",
            hide_index=True,
            column_config={
                COL_BROKER: st.column_config.SelectboxColumn(
                    COL_BROKER, options=broker_options(st.session_state["cash"])
                ),
                COL_AMOUNT: st.column_config.NumberColumn(COL_AMOUNT, min_value=0.0, format="%.2f"),
            },
        )
        run = st.button("Risk Analizi Yap & Rapor Oluştur", key="run_analysis", type="primary")
        st.caption(
            "Fiyatlar, seçilen varlıklara göre kalibre edilmiş sentetik (GBM) motorla üretilir."
        )

    st.title("Portfolio Risk Engine")
    st.caption("Varlıkları listeden seçin, miktarı girin, tek tıkla risk analizini çalıştırın.")

    st.markdown("**Varlık Ekle**")
    c_cat, c_asset, c_qty, c_add = st.columns([2, 4, 2, 1], vertical_alignment="bottom")
    category = c_cat.selectbox("Borsa / Kategori", CATEGORIES, key="add_category")
    entries = entries_for(category)
    entry = c_asset.selectbox(
        "Varlık",
        entries,
        format_func=lambda e: e.label,
        key=f"add_asset_{category}",
    )
    quantity = c_qty.number_input(
        "Miktar / Adet", min_value=0.0, value=1.0, step=1.0, format="%g", key="add_qty"
    )
    add_clicked = c_add.button("Ekle", key="add_position", type="primary")
    if entry is not None:
        st.caption(
            f"Sınıf: **{entry.asset_class.value}** | Etiketler: {', '.join(entry.tags)} | "
            f"Broker: {entry.broker}"
        )

    st.markdown("**Pozisyonlar**")
    positions = st.data_editor(
        st.session_state["positions"],
        key=f"positions_editor_{version}",
        num_rows="fixed",
        hide_index=True,
        use_container_width=True,
        disabled=[COL_CATEGORY, COL_SYMBOL, COL_NAME, COL_CLASS, COL_TAGS],
        column_config={
            COL_DELETE: st.column_config.CheckboxColumn(COL_DELETE, default=False),
            COL_QTY: st.column_config.NumberColumn(COL_QTY, min_value=0.0, format="%.6g"),
        },
    )
    delete_clicked = st.button("Seçilenleri Sil", key="delete_positions")

    if add_clicked and entry is not None:
        try:
            st.session_state["positions"] = add_position(positions, entry, float(quantity))
        except PortfolioInputError as exc:
            st.error(str(exc))
        else:
            _reset_editors()
            st.rerun()
    if delete_clicked:
        st.session_state["positions"] = remove_marked(positions)
        _reset_editors()
        st.rerun()

    if run:
        _run_analysis(positions, cash, simulations)

    analysis = st.session_state.get("analysis")
    if analysis is None:
        st.info("Pozisyon ekleyin (veya örnek portföyü yükleyin), sonra analizi başlatın.")
        return
    _show_results(analysis, confidence, horizon)

    st.divider()
    d1, d2 = st.columns(2)
    d1.download_button(
        "HTML Raporu İndir",
        data=st.session_state["report_html"],
        file_name="risk-report.html",
        mime="text/html",
        key="download_html",
    )
    d2.download_button(
        "Markdown Özeti İndir",
        data=st.session_state["report_md"],
        file_name="risk-report.md",
        mime="text/markdown",
        key="download_md",
    )
    with st.expander("Tam raporun canlı önizlemesi"):
        st.components.v1.html(st.session_state["report_html"], height=900, scrolling=True)


main()
