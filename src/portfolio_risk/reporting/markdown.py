"""Markdown risk summary."""

from __future__ import annotations

from portfolio_risk.reporting.analysis import (
    CONFIDENCES,
    HEADLINE_CONFIDENCE,
    HEADLINE_HORIZON,
    HORIZONS,
    RiskAnalysis,
)
from portfolio_risk.risk import CORE_METHODS


def _table(header: list[str], rows: list[list[str]], right_from: int = 1) -> list[str]:
    align = ["---" if i < right_from else "---:" for i in range(len(header))]
    lines = ["| " + " | ".join(header) + " |", "| " + " | ".join(align) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return [*lines, ""]


def render_markdown(a: RiskAnalysis) -> str:
    ccy = a.portfolio.base_currency
    mc = a.monte_carlo
    top = a.top_risk_asset
    headline = a.headline_var
    out = [
        f"# Risk Report: {a.portfolio.name}",
        "",
        f"*As of {a.as_of} | base currency {ccy} | {a.observations} daily observations | "
        f"seed {a.seed}*",
        "",
        "## Executive Risk Summary",
        "",
    ]
    out += _table(
        ["Metric", "Value"],
        [
            ["Total portfolio value", f"{a.total_value:,.2f} {ccy}"],
            ["Cash ratio", f"{a.cash_ratio:.1%}"],
            [
                "Highest-risk asset",
                f"{top.symbol} ({top.standalone_var:,.2f} standalone VaR)" if top else "n/a",
            ],
            [
                f"{HEADLINE_HORIZON}-day {HEADLINE_CONFIDENCE:.0%} VaR (parametric)",
                f"{headline.var:,.2f} {ccy} ({headline.var / a.total_value:.2%})",
            ],
            [
                f"{HEADLINE_HORIZON}-day {HEADLINE_CONFIDENCE:.0%} CVaR (parametric)",
                f"{headline.cvar:,.2f} {ccy}",
            ],
            ["Diversification benefit", f"{headline.diversification_ratio:.1%}"],
        ],
    )

    out += ["## Allocation", ""]
    rows = [
        [x.symbol, x.asset_class, f"{x.value:,.2f}", f"{x.weight:.1%}", f"{x.standalone_var:,.2f}"]
        for x in a.assets
    ]
    if a.cash > 0:
        rows.append(["CASH", "cash", f"{a.cash:,.2f}", f"{a.cash_ratio:.1%}", "0.00"])
    out += _table(["Symbol", "Class", "Value", "Weight", "Standalone 10d 99% VaR"], rows, 2)

    out += ["### Correlation matrix", ""]
    corr = a.correlation
    out += _table(
        ["", *map(str, corr.columns)],
        [[str(s), *(f"{v:.2f}" for v in corr.loc[s])] for s in corr.index],
    )

    out += ["## Statistical Risk", ""]
    rows = []
    for m in CORE_METHODS:
        for h in HORIZONS:
            for c in CONFIDENCES:
                r = a.var_report(m, c, h)
                rows.append(
                    [
                        m.value,
                        f"{h}d",
                        f"{c:.0%}",
                        f"{r.var:,.2f}",
                        f"{r.cvar:,.2f}",
                        f"{r.diversification_ratio:.1%}",
                    ]
                )
    out += _table(["Method", "Horizon", "Conf.", "VaR", "CVaR", "Div. benefit"], rows, 3)

    if a.backtests:
        first = a.backtests[0]
        out += [
            "## Model Validation (VaR Backtest)",
            "",
            f"One-day {first.confidence:.0%} VaR, {first.n_obs} test days, "
            f"{first.window}-day estimation window.",
            "",
        ]
        out += _table(
            ["Method", "Violations", "Expected", "Kupiec p", "Independence p", "Basel zone"],
            [
                [
                    r.method.value,
                    str(r.n_violations),
                    f"{r.expected_violations:.1f}",
                    f"{r.kupiec.p_value:.3f}",
                    f"{r.independence.p_value:.3f}",
                    r.zone.value,
                ]
                for r in a.backtests
            ],
        )

    out += [
        f"## Monte Carlo ({mc.n_simulations:,} paths x {mc.days} days)",
        "",
    ]
    out += _table(
        ["Terminal value", "Amount", "Change"],
        [
            [label, f"{v:,.2f}", f"{v / mc.initial_value - 1:+.1%}"]
            for label, v in [
                ("5th percentile", mc.final_percentile(5)),
                ("Median", mc.median_final),
                ("Mean", mc.mean_final),
                ("95th percentile", mc.final_percentile(95)),
            ]
        ],
    )
    out += _table(
        ["Confidence", "MC VaR", "MC CVaR"],
        [[f"{c:.0%}", f"{mc.var[c]:,.2f}", f"{mc.cvar[c]:,.2f}"] for c in mc.var],
    )
    out += _table(
        ["Max drawdown", "Probability"],
        [[f">= {lvl:.0%}", f"{mc.prob_drawdown_exceeds(lvl):.1%}"] for lvl in (0.1, 0.2, 0.3, 0.5)],
    )
    out += [
        f"Probability of losing {mc.loss_threshold:.0%} or more: "
        f"**{mc.prob_loss_at_end:.2%}** at horizon end, "
        f"**{mc.prob_ruin:.2%}** at any point (ruin).",
        "",
        "## Crisis Resilience (Stress Tests)",
        "",
    ]
    out += _table(
        ["Scenario", "P&L", "Loss %", "Stressed value"],
        [
            [
                f"{r.scenario.name} ({r.scenario.description})",
                f"{r.total_pnl:,.2f}",
                f"{r.pnl_pct:.1%}",
                f"{r.stressed_value:,.2f}",
            ]
            for r in a.stress.results
        ],
        1,
    )
    worst = a.stress.worst_case
    out += [
        f"**Worst case:** {worst.scenario.name} loses {-worst.total_pnl:,.2f} {ccy} "
        f"({-worst.pnl_pct:.1%}), leaving {worst.stressed_value:,.2f} {ccy}.",
        "",
        "---",
        "*Model outputs, not forecasts. VaR figures assume normal returns (parametric) or "
        "repeat of the sample (historical); stress shocks are instantaneous.*",
        "",
    ]
    return "\n".join(out)
