"""Rich terminal dashboard."""

from __future__ import annotations

from rich import box
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from portfolio_risk.reporting.analysis import (
    CONFIDENCES,
    HEADLINE_CONFIDENCE,
    HEADLINE_HORIZON,
    HORIZONS,
    RiskAnalysis,
)
from portfolio_risk.risk import CORE_METHODS


def _signed(value: float, text: str) -> Text:
    return Text(text, style="red" if value < 0 else "green" if value > 0 else "")


def _table(title: str, *columns: str) -> Table:
    table = Table(title=title, box=box.SIMPLE_HEAD, title_style="bold cyan", title_justify="left")
    for i, col in enumerate(columns):
        table.add_column(col, justify="left" if i == 0 else "right")
    return table


def render_dashboard(a: RiskAnalysis, console: Console) -> None:
    ccy = a.portfolio.base_currency
    head = a.headline_var
    top = a.top_risk_asset
    mc = a.monte_carlo

    summary = Table.grid(padding=(0, 3))
    summary.add_row(
        Text.assemble(("Value  ", "dim"), (f"{a.total_value:,.2f} {ccy}", "bold")),
        Text.assemble(("Cash  ", "dim"), (f"{a.cash_ratio:.1%}", "bold")),
        Text.assemble(("Top risk  ", "dim"), (top.symbol if top else "n/a", "bold yellow")),
        Text.assemble(
            (f"{HEADLINE_HORIZON}d {HEADLINE_CONFIDENCE:.0%} VaR  ", "dim"),
            (f"{head.var:,.2f}", "bold red"),
        ),
    )
    console.print(
        Panel(
            summary,
            title=f"[bold]Risk Dashboard: {a.portfolio.name}[/bold]",
            subtitle=f"as of {a.as_of} | {a.observations} obs | seed {a.seed}",
            border_style="cyan",
        )
    )

    alloc = _table("Allocation", "Symbol", "Class", "Value", "Weight", "Standalone VaR")
    for x in a.assets:
        alloc.add_row(
            x.symbol,
            x.asset_class,
            f"{x.value:,.2f}",
            f"{x.weight:.1%}",
            f"{x.standalone_var:,.2f}",
        )
    if a.cash > 0:
        alloc.add_row("CASH", "cash", f"{a.cash:,.2f}", f"{a.cash_ratio:.1%}", "-")

    corr = _table("Correlation", "", *map(str, a.correlation.columns))
    for s in a.correlation.index:
        corr.add_row(
            str(s),
            *(
                Text(f"{v:.2f}", style="bold" if v >= 0.7 and s != c else "")
                for c, v in a.correlation.loc[s].items()
            ),
        )

    var = _table("VaR / CVaR", "Method", "Horizon", "Conf.", "VaR", "CVaR", "Div. benefit")
    for m in CORE_METHODS:
        for h in HORIZONS:
            for c in CONFIDENCES:
                vr = a.var_report(m, c, h)
                var.add_row(
                    m.value,
                    f"{h}d",
                    f"{c:.0%}",
                    f"{vr.var:,.2f}",
                    f"{vr.cvar:,.2f}",
                    f"{vr.diversification_ratio:.1%}",
                )
    console.print(Group(alloc, corr, var))

    contrib = _table(
        "Risk attribution", "Symbol", "Exposure", "VaR contrib.", "Share", "CVaR contrib."
    )
    rc = a.contributions
    for sym in rc.exposures.index:
        contrib.add_row(
            str(sym),
            f"{rc.exposures[sym]:,.2f}",
            _signed(-rc.component_var[sym], f"{rc.component_var[sym]:,.2f}"),
            f"{rc.var_share[sym]:.1%}",
            f"{rc.component_cvar[sym]:,.2f}",
        )
    console.print(contrib)

    if a.optimizations:
        symbols = list(a.optimizations[0].weights)
        opt = _table("Optimisation", "Allocation", *symbols, "Return", "Vol.", "VaR", "VaR chg.")
        base = a.optimizations[0].var
        for o in a.optimizations:
            change = o.var / base - 1.0
            opt.add_row(
                o.objective,
                *(f"{o.weights[s]:.1%}" for s in symbols),
                f"{o.expected_return:.1%}",
                f"{o.volatility:.1%}",
                f"{o.var:,.2f}",
                "" if o.objective == "current" else _signed(-change, f"{change:+.1%}"),
            )
        console.print(opt)

    if a.backtests:
        first = a.backtests[0]
        bt = _table(
            f"VaR backtest ({first.confidence:.0%} one-day, {first.n_obs} days)",
            "Method",
            "Violations",
            "Expected",
            "Kupiec p",
            "Indep. p",
            "Zone",
        )
        colours = {"green": "green", "yellow": "yellow", "red": "red"}
        for r in a.backtests:
            bt.add_row(
                r.method.value,
                str(r.n_violations),
                f"{r.expected_violations:.1f}",
                f"{r.kupiec.p_value:.3f}",
                f"{r.independence.p_value:.3f}",
                Text(r.zone.value, style=colours[r.zone.value]),
            )
        console.print(bt)

    sim = _table(f"Monte Carlo ({mc.n_simulations:,} x {mc.days}d)", "Metric", "Value", "Change")
    for label, v in [
        ("5th pct", mc.final_percentile(5)),
        ("Median", mc.median_final),
        ("95th pct", mc.final_percentile(95)),
    ]:
        sim.add_row(
            label, f"{v:,.2f}", _signed(v - mc.initial_value, f"{v / mc.initial_value - 1:+.1%}")
        )
    for c in mc.var:
        sim.add_row(
            f"VaR {c:.0%}",
            f"{mc.var[c]:,.2f}",
            Text(f"-{mc.var[c] / mc.initial_value:.1%}", style="red"),
        )
        sim.add_row(
            f"CVaR {c:.0%}",
            f"{mc.cvar[c]:,.2f}",
            Text(f"-{mc.cvar[c] / mc.initial_value:.1%}", style="red"),
        )
    for lvl in (0.1, 0.2, 0.3):
        sim.add_row(f"P(MDD >= {lvl:.0%})", f"{mc.prob_drawdown_exceeds(lvl):.1%}", "")
    sim.add_row(f"P(ruin, -{mc.loss_threshold:.0%})", f"{mc.prob_ruin:.2%}", "")
    console.print(sim)

    stress = _table("Stress tests", "Scenario", "P&L", "Loss %", "Stressed value")
    for sr in a.stress.results:
        stress.add_row(
            sr.scenario.name,
            _signed(sr.total_pnl, f"{sr.total_pnl:,.2f}"),
            _signed(sr.pnl_pct, f"{sr.pnl_pct:.1%}"),
            f"{sr.stressed_value:,.2f}",
        )
    console.print(stress)
    worst = a.stress.worst_case
    console.print(
        Panel(
            f"[bold]{worst.scenario.name}[/bold] loses [red]{-worst.total_pnl:,.2f} {ccy}[/red] "
            f"({-worst.pnl_pct:.1%}); remaining capital {worst.stressed_value:,.2f} {ccy}",
            title="Worst case",
            border_style="red",
        )
    )
