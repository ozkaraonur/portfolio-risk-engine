"""Streamlit dashboard. Launch with ``pre web`` (or ``streamlit run`` on this file)."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import streamlit as st
from pydantic import ValidationError

from portfolio_risk.data import (
    DataUnavailableError,
    PriceProvider,
    StooqProvider,
    SyntheticProvider,
)
from portfolio_risk.reporting import RiskAnalysis, build_analysis, render_html, render_markdown
from portfolio_risk.risk import Method
from portfolio_risk.web.builder import (
    ASSET_CLASSES,
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
    portfolio_to_frames,
)

SEED = 42
HISTORY_DAYS = 730
PROVIDERS = {"Sentetik (çevrimdışı)": "synthetic", "Stooq (gerçek veri)": "stooq"}


def _init_state() -> None:
    st.session_state.setdefault("positions", empty_positions())
    st.session_state.setdefault("cash", empty_cash())
    st.session_state.setdefault("editor_version", 0)


def _load_sample() -> None:
    try:
        positions, cash = portfolio_to_frames(load_sample_portfolio())
    except (FileNotFoundError, ValidationError) as exc:
        st.session_state["sample_error"] = str(exc)
        return
    st.session_state.update(positions=positions, cash=cash)
    st.session_state["editor_version"] += 1  # forces the editors to re-initialise
    st.session_state.pop("sample_error", None)


def _make_provider(name: str) -> PriceProvider:
    return StooqProvider() if name == "stooq" else SyntheticProvider(seed=SEED)


def _heat(value: object) -> str:
    number = float(value) if isinstance(value, int | float) else 0.0
    rgb = "197,48,48" if number >= 0 else "43,108,176"
    return f"background-color: rgba({rgb},{min(abs(number), 1.0) * 0.5:.2f})"


def _run_analysis(
    positions: pd.DataFrame, cash: pd.DataFrame, provider: str, simulations: int
) -> None:
    try:
        portfolio = frames_to_portfolio(positions, cash)
        end = date.today()
        prices = _make_provider(provider).get_prices(
            portfolio.assets, end - timedelta(days=HISTORY_DAYS), end
        )
        analysis = build_analysis(portfolio, prices, simulations=simulations, seed=SEED)
    except (PortfolioInputError, ValidationError, DataUnavailableError, ValueError) as exc:
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
                for m in Method
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
            "Güven Aralığı", [0.95, 0.99], format_func=lambda c: f"%{c * 100:.0f}", horizontal=True
        )
        horizon = st.radio("Zaman Ufku", [1, 10], format_func=lambda h: f"{h} gün", horizontal=True)
        simulations = st.radio(
            "Monte Carlo Simülasyon Sayısı",
            [1000, 5000],
            format_func=lambda n: f"{n:,}",
            horizontal=True,
        )
        provider_label = st.selectbox("Fiyat Verisi", list(PROVIDERS))
        st.markdown("**Nakit Bakiyesi**")
        cash = st.data_editor(
            st.session_state["cash"],
            key=f"cash_editor_{version}",
            num_rows="dynamic",
            hide_index=True,
            column_config={
                COL_BROKER: st.column_config.SelectboxColumn(
                    COL_BROKER,
                    options=broker_options(st.session_state["positions"], st.session_state["cash"]),
                ),
                COL_AMOUNT: st.column_config.NumberColumn(COL_AMOUNT, min_value=0.0, format="%.2f"),
            },
        )
        run = st.button("Risk Analizi Yap & Rapor Oluştur", key="run_analysis", type="primary")

    st.title("Portfolio Risk Engine")
    st.caption(
        "Portföyünüzü oluşturun, tek tıkla risk analizini çalıştırın ve HTML raporunu indirin."
    )
    st.markdown("**Pozisyonlar** (satır eklemek için tablonun altındaki + simgesini kullanın)")
    positions = st.data_editor(
        st.session_state["positions"],
        key=f"positions_editor_{version}",
        num_rows="dynamic",
        hide_index=True,
        use_container_width=True,
        column_config={
            COL_BROKER: st.column_config.SelectboxColumn(
                COL_BROKER,
                options=broker_options(st.session_state["positions"], st.session_state["cash"]),
                default="Custom",
            ),
            COL_SYMBOL: st.column_config.TextColumn(COL_SYMBOL, help="Örn. AAPL, BTC, GC"),
            COL_CLASS: st.column_config.SelectboxColumn(
                COL_CLASS, options=ASSET_CLASSES, default="equity"
            ),
            COL_QTY: st.column_config.NumberColumn(COL_QTY, min_value=0.0, format="%.6g"),
            COL_TAGS: st.column_config.TextColumn(
                COL_TAGS, help="Virgülle ayırın, örn. tech, growth"
            ),
        },
    )

    if run:
        _run_analysis(positions, cash, PROVIDERS[provider_label], simulations)

    analysis = st.session_state.get("analysis")
    if analysis is None:
        st.info("Pozisyonları girin (veya örnek portföyü yükleyin), sonra analizi başlatın.")
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
