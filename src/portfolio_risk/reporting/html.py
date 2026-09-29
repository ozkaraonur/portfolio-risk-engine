"""Self-contained HTML risk report (inline CSS and SVG, no external requests)."""

from __future__ import annotations

from html import escape

import numpy as np

from portfolio_risk.reporting.analysis import (
    CONFIDENCES,
    HEADLINE_CONFIDENCE,
    HEADLINE_HORIZON,
    HORIZONS,
    RiskAnalysis,
)
from portfolio_risk.risk import Method, MonteCarloReport

CSS = """
:root{--bg:#f6f7f9;--card:#fff;--fg:#1a2230;--muted:#5f6b7a;--line:#e3e7ec;--accent:#2b6cb0;
--bad:#c53030;--good:#2f855a;--bar:#cfe0f3}
@media (prefers-color-scheme:dark){:root{--bg:#0f141b;--card:#171e28;--fg:#e6ebf1;--muted:#93a0b1;
--line:#28313d;--accent:#63a4ec;--bad:#f28b82;--good:#7bd3a1;--bar:#24405f}}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
main{max-width:980px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:26px;margin:0 0 4px}h2{font-size:19px;margin:36px 0 12px;
border-bottom:1px solid var(--line);padding-bottom:6px}h3{font-size:15px;margin:22px 0 8px}
.meta{color:var(--muted);font-size:13px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px;margin-top:18px}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:14px 16px}
.card .label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.04em}
.card .value{font-size:22px;font-weight:650;margin-top:4px}
.card .sub{color:var(--muted);font-size:12px}
table{width:100%;border-collapse:collapse;background:var(--card);border:1px solid var(--line);
border-radius:10px;overflow:hidden;font-size:14px}
th,td{padding:8px 12px;border-bottom:1px solid var(--line);text-align:right}
th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:600;font-size:12px;
text-transform:uppercase;letter-spacing:.03em}tr:last-child td{border-bottom:0}
.neg{color:var(--bad)}.pos{color:var(--good)}
.bar{display:inline-block;height:8px;background:var(--bar);border-radius:4px;vertical-align:middle;
margin-right:8px}
.scroll{overflow-x:auto}details{margin:8px 0}summary{cursor:pointer;color:var(--accent)}
svg{width:100%;height:auto;background:var(--card);border:1px solid var(--line);border-radius:10px}
svg text{fill:var(--muted);font-size:11px}
footer{margin-top:40px;color:var(--muted);font-size:12px}
@media print{body{background:#fff}.card,table,svg{break-inside:avoid}}
"""


def _cls(value: float) -> str:
    return "neg" if value < 0 else ("pos" if value > 0 else "")


def _table(header: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{escape(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return (
        f'<div class="scroll"><table><thead><tr>{head}</tr></thead>'
        f"<tbody>{body}</tbody></table></div>"
    )


def _card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return (
        f'<div class="card"><div class="label">{escape(label)}</div>'
        f'<div class="value">{value}</div>{sub_html}</div>'
    )


def _heat(value: float) -> str:
    rgb = "197,48,48" if value >= 0 else "43,108,176"
    return f"background:rgba({rgb},{min(abs(value), 1.0) * 0.45:.2f})"


def _histogram(mc: MonteCarloReport) -> str:
    width, height, pad = 720, 180, 24
    counts, edges = np.histogram(mc.final_values, bins=40)
    top = max(int(counts.max()), 1)
    lo, hi = float(edges[0]), float(edges[-1])
    span = (hi - lo) or 1.0

    def x(v: float) -> float:
        return pad + (v - lo) / span * (width - 2 * pad)

    bars = "".join(
        f'<rect x="{x(float(edges[i])):.1f}" y="{height - pad - c / top * (height - 2 * pad):.1f}" '
        f'width="{max(x(float(edges[i + 1])) - x(float(edges[i])) - 1, 0.5):.1f}" '
        f'height="{c / top * (height - 2 * pad):.1f}" fill="var(--accent)" opacity="0.75"/>'
        for i, c in enumerate(counts)
    )
    marks = ""
    for value, label in [
        (mc.initial_value, "today"),
        (mc.final_percentile(5), "p5"),
        (mc.median_final, "median"),
        (mc.final_percentile(95), "p95"),
    ]:
        if lo <= value <= hi:
            marks += (
                f'<line x1="{x(value):.1f}" x2="{x(value):.1f}" y1="{pad - 8}" y2="{height - pad}" '
                f'stroke="var(--fg)" stroke-dasharray="3 3" opacity="0.6"/>'
                f'<text x="{x(value):.1f}" y="{pad - 12}" text-anchor="middle">{label}</text>'
            )
    axis = f'<text x="{pad}" y="{height - 6}">{lo:,.0f}</text>'
    axis += f'<text x="{width - pad}" y="{height - 6}" text-anchor="end">{hi:,.0f}</text>'
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" '
        f'aria-label="Distribution of terminal portfolio value">{bars}{marks}{axis}</svg>'
    )


def render_html(a: RiskAnalysis) -> str:
    ccy = escape(a.portfolio.base_currency)
    mc = a.monte_carlo
    top = a.top_risk_asset
    head = a.headline_var
    name = escape(a.portfolio.name)

    cards = "".join(
        [
            _card("Total portfolio value", f"{a.total_value:,.2f}", ccy),
            _card("Cash ratio", f"{a.cash_ratio:.1%}", f"{a.cash:,.2f} {ccy}"),
            _card(
                "Highest-risk asset",
                escape(top.symbol) if top else "n/a",
                f"{top.standalone_var:,.2f} standalone VaR" if top else "",
            ),
            _card(
                f"{HEADLINE_HORIZON}-day {HEADLINE_CONFIDENCE:.0%} VaR",
                f'<span class="neg">{head.var:,.2f}</span>',
                f"{head.var / a.total_value:.2%} of value | CVaR {head.cvar:,.2f}",
            ),
        ]
    )

    alloc_rows = [
        [
            escape(x.symbol),
            escape(x.asset_class),
            f"{x.value:,.2f}",
            f'<span class="bar" style="width:{x.weight * 120:.0f}px"></span>{x.weight:.1%}',
            f"{x.standalone_var:,.2f}",
        ]
        for x in a.assets
    ]
    if a.cash > 0:
        alloc_rows.append(
            [
                "CASH",
                "cash",
                f"{a.cash:,.2f}",
                f'<span class="bar" style="width:{a.cash_ratio * 120:.0f}px"></span>'
                f"{a.cash_ratio:.1%}",
                "0.00",
            ]
        )
    allocation = _table(
        ["Symbol", "Class", "Value", "Weight", f"Standalone {HEADLINE_HORIZON}d VaR"], alloc_rows
    )

    corr = a.correlation
    corr_head = "".join(f"<th>{escape(str(c))}</th>" for c in corr.columns)
    corr_rows = "".join(
        f"<tr><td>{escape(str(s))}</td>"
        + "".join(f'<td style="{_heat(float(v))}">{float(v):.2f}</td>' for v in corr.loc[s])
        + "</tr>"
        for s in corr.index
    )
    correlation = (
        f'<div class="scroll"><table><thead><tr><th></th>{corr_head}</tr></thead>'
        f"<tbody>{corr_rows}</tbody></table></div>"
    )

    var_rows = [
        [
            m.value,
            f"{h}d",
            f"{c:.0%}",
            f"{a.var_report(m, c, h).var:,.2f}",
            f"{a.var_report(m, c, h).cvar:,.2f}",
            f"{a.var_report(m, c, h).diversification_benefit:,.2f} "
            f"({a.var_report(m, c, h).diversification_ratio:.1%})",
        ]
        for m in Method
        for h in HORIZONS
        for c in CONFIDENCES
    ]
    var_table = _table(
        ["Method", "Horizon", "Conf.", "VaR", "CVaR", "Diversification benefit"], var_rows
    )

    pct_rows = [
        [
            label,
            f"{v:,.2f}",
            f'<span class="{_cls(v - mc.initial_value)}">{v / mc.initial_value - 1:+.1%}</span>',
        ]
        for label, v in [
            ("5th percentile", mc.final_percentile(5)),
            ("Median", mc.median_final),
            ("Mean", mc.mean_final),
            ("95th percentile", mc.final_percentile(95)),
        ]
    ]
    mc_var = _table(
        ["Confidence", "MC VaR", "MC CVaR"],
        [[f"{c:.0%}", f"{mc.var[c]:,.2f}", f"{mc.cvar[c]:,.2f}"] for c in mc.var],
    )
    dd_rows = [
        [f"&ge; {lvl:.0%}", f"{mc.prob_drawdown_exceeds(lvl):.1%}"] for lvl in (0.1, 0.2, 0.3, 0.5)
    ]

    stress_rows = [
        [
            f"<strong>{escape(r.scenario.name)}</strong><br><span class='meta'>"
            f"{escape(r.scenario.description)}</span>",
            f'<span class="{_cls(r.total_pnl)}">{r.total_pnl:,.2f}</span>',
            f'<span class="{_cls(r.pnl_pct)}">{r.pnl_pct:.1%}</span>',
            f"{r.stressed_value:,.2f}",
        ]
        for r in a.stress.results
    ]
    details = ""
    for sr in a.stress.results:
        rows = [
            [
                escape(i.symbol),
                f"{i.value:,.2f}",
                f'<span class="{_cls(i.shock)}">{i.shock:+.1%}</span>',
                f'<span class="{_cls(i.pnl)}">{i.pnl:,.2f}</span>',
            ]
            for i in sr.impacts
        ]
        details += (
            f"<details><summary>{escape(sr.scenario.name)}: per-asset impact</summary>"
            f"{_table(['Symbol', 'Value', 'Shock', 'P&L'], rows)}</details>"
        )
    worst = a.stress.worst_case

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Risk Report - {name}</title>
<style>{CSS}</style>
</head>
<body>
<main>
<h1>Risk Report: {name}</h1>
<div class="meta">As of {a.as_of} &middot; base currency {ccy} &middot; {a.observations} daily
observations &middot; seed {a.seed}</div>

<h2>Executive Risk Summary</h2>
<div class="cards">{cards}</div>
<p>Diversification reduces {HEADLINE_HORIZON}-day {HEADLINE_CONFIDENCE:.0%} VaR by
<strong>{head.diversification_ratio:.1%}</strong> versus holding each asset in isolation.</p>

<h2>Allocation &amp; Correlation</h2>
{allocation}
<h3>Correlation matrix (daily returns)</h3>
{correlation}

<h2>Statistical Risk</h2>
{var_table}

<h2>Monte Carlo ({mc.n_simulations:,} paths &times; {mc.days} days)</h2>
{_histogram(mc)}
<h3>Terminal portfolio value</h3>
{_table(["", "Value", "Change"], pct_rows)}
<h3>Loss over horizon</h3>
{mc_var}
<h3>Maximum drawdown probability</h3>
{_table(["Max drawdown", "Probability"], dd_rows)}
<p>Probability of losing {mc.loss_threshold:.0%} or more:
<strong>{mc.prob_loss_at_end:.2%}</strong> at horizon end,
<strong>{mc.prob_ruin:.2%}</strong> at any point (ruin).</p>

<h2>Crisis Resilience</h2>
{_table(["Scenario", "P&L", "Loss %", "Stressed value"], stress_rows)}
<p><strong>Worst case:</strong> {escape(worst.scenario.name)} loses
<span class="neg">{-worst.total_pnl:,.2f} {ccy}</span> ({-worst.pnl_pct:.1%}), leaving
{worst.stressed_value:,.2f} {ccy}.</p>
{details}

<footer>Model outputs, not forecasts. Parametric VaR assumes zero-mean normal returns;
historical VaR replays the sample; Monte Carlo uses correlated GBM with buy-and-hold positions;
stress shocks are instantaneous and cash is unshocked. Generated by portfolio-risk-engine.</footer>
</main>
</body>
</html>
"""
