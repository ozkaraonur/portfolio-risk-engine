# Portfolio Risk Engine

Modular multi-broker portfolio risk and stress-testing engine (Python 3.11+).

## Layout
- `src/portfolio_risk/models/` — pydantic v2 domain models (`Asset`, `Position`, `CashBalance`, `Portfolio`).
- `src/portfolio_risk/data/` — price providers behind the `PriceProvider` ABC:
  `SyntheticProvider` (seeded GBM, offline, used in tests) and `StooqProvider` (public CSV, network).
  `cache.py` (`CachedProvider`, JSON per symbol, TTL) wraps any provider; `fx.py` (`FxProvider`, `SyntheticFx`, `StooqFx`, `convert_to_base`) expresses foreign-currency prices/cash in the base currency (the CLI does this in `fetch_prices`).
- `src/portfolio_risk/importers.py` — broker CSV -> `Portfolio` (`pre import`).
- `src/portfolio_risk/risk/` — covariance/correlation, parametric + historical VaR/CVaR (`var.py`), `analyze_risk` report with diversification benefit (`report.py`). VaR/CVaR are positive loss amounts; cash adds value but no risk.
- `src/portfolio_risk/risk/estimators.py` — `Method` enum (parametric, historical, ewma, student-t, cornish-fisher, fhs) and the `estimate_var_cvar` dispatcher; `tail_models.py` holds the P&L-based fat-tail models; `CORE_METHODS` (parametric, historical) are the ones in standard reports.
- `src/portfolio_risk/risk/backtest.py` + `coverage.py` — rolling one-day VaR backtest (hypothetical P&L on today's exposures) for every `Method`, Kupiec/Christoffersen tests, Basel zones; `build_analysis` includes it when history allows and the reports draw the violation charts.
- `src/portfolio_risk/risk/attribution.py` — component/marginal VaR and CVaR (Euler; parametric and historical). `optimize.py` — long-only min-variance / risk-parity / max-Sharpe over the risky assets, efficient frontier, `rebalance_trades`; `build_analysis` adds `contributions` and `optimizations` (current allocation first).
- `src/portfolio_risk/risk/monte_carlo.py` + `linalg.py` — Cholesky/nearest-PSD, correlated GBM paths (buy-and-hold), MC VaR/CVaR, drawdown and ruin metrics.
- `src/portfolio_risk/risk/scenarios.py` + `stress.py` — built-in historical shocks, custom/tag/symbol shocks, beta-driven market shocks; `run_stress` gives per-asset P&L and worst case. Shock precedence: symbol > tag > class.
- `src/portfolio_risk/reporting/limits.py` (`RiskLimits`, `check_limits`; `pre check` exits 2 on breach) and `history.py` (SQLite snapshots; `pre history`).
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
