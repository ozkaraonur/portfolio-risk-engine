"""End-to-end demo: offline synthetic data -> full analysis -> reports in ./output.

Run from the repository root:  python examples/run_demo.py [output_dir]
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

from rich.console import Console

from portfolio_risk.data import SyntheticProvider
from portfolio_risk.models import Portfolio
from portfolio_risk.reporting import build_analysis, render_dashboard, render_html, render_markdown

ROOT = Path(__file__).resolve().parent.parent
# Fixed window and seed so the generated report is reproducible.
START, END, SEED = date(2023, 1, 1), date(2025, 12, 31), 42


def main(output_dir: Path) -> None:
    portfolio = Portfolio.model_validate_json(
        (ROOT / "examples" / "portfolio.json").read_text(encoding="utf-8")
    )
    prices = SyntheticProvider(seed=SEED).get_prices(portfolio.assets, START, END)
    analysis = build_analysis(portfolio, prices, seed=SEED)

    render_dashboard(analysis, Console())
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "risk-report.html").write_text(render_html(analysis), encoding="utf-8")
    (output_dir / "risk-report.md").write_text(render_markdown(analysis), encoding="utf-8")
    print(f"\nReports written to {output_dir}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "output")
