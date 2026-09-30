# Portfolio Risk Engine

Modular multi-broker portfolio risk and stress-testing engine (Python 3.11+).

## Layout
- `src/portfolio_risk/models/` — pydantic v2 domain models (`Asset`, `Position`, `CashBalance`, `Portfolio`).
- `src/portfolio_risk/data/` — price providers behind the `PriceProvider` ABC:
  `SyntheticProvider` (seeded GBM, offline, used in tests) and `StooqProvider` (public CSV, network).
- `src/portfolio_risk/risk/` — covariance/correlation, parametric + historical VaR/CVaR (`var.py`), `analyze_risk` report with diversification benefit (`report.py`). VaR/CVaR are positive loss amounts; cash adds value but no risk.
- `src/portfolio_risk/risk/backtest.py` + `coverage.py` — rolling one-day VaR backtest (hypothetical P&L on today's exposures), Kupiec/Christoffersen tests, Basel zones.
- `src/portfolio_risk/risk/monte_carlo.py` + `linalg.py` — Cholesky/nearest-PSD, correlated GBM paths (buy-and-hold), MC VaR/CVaR, drawdown and ruin metrics.
- `src/portfolio_risk/risk/scenarios.py` + `stress.py` — built-in historical shocks, custom/tag/symbol shocks, beta-driven market shocks; `run_stress` gives per-asset P&L and worst case. Shock precedence: symbol > tag > class.
- `src/portfolio_risk/reporting/` — `build_analysis` runs everything once into a `RiskAnalysis`; renderers: `terminal.py` (rich), `html.py` (self-contained, escape all dynamic text, no external requests), `markdown.py`. Sample output in `docs/sample/` (regenerate: `python examples/run_demo.py docs/sample`).
- `src/portfolio_risk/catalog.py` — fixed asset universe (category, class, tags, broker, calibrated synthetic profile); the web panel selects from it and uses `SyntheticProvider(profiles=synthetic_profiles())` only.
- `src/portfolio_risk/web/` — Streamlit panel: `builder.py` (pure table<->Portfolio logic, unit-tested), `app.py` (UI; tested with `streamlit.testing.v1.AppTest`). Launch via `pre web`.
- `src/portfolio_risk/cli.py` — Typer CLI (`pre`, or `python -m portfolio_risk`).
- `tests/` — pytest; **must never touch the network** (inject fakes into `StooqProvider`).

## Commands
Use the project venv (`.venv`); install with `pip install -r requirements.txt && pip install -e .`.
- Tests: `pytest`
- Lint/format: `ruff check .` and `ruff format .`
- Types: `mypy --strict` (config in `pyproject.toml`; must be clean)

## Conventions
- Full type annotations; mypy strict must pass. No `Any` unless unavoidable and commented.
- Domain models are immutable (`frozen=True`) with `extra="forbid"`; validate at the boundary.
- Providers return a `pd.DataFrame` of prices: DatetimeIndex (ascending), one column per asset symbol.
- Simulations take an explicit `seed`; same inputs → same output. Never use global RNG state.
- Log with `loguru`; no `print` in library code (CLI output goes through `typer.echo`).
- Keep modules small and single-purpose; add a test with every new behavior.
- Commit only when ruff, mypy and pytest are all green.
