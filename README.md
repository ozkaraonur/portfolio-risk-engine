# Portfolio Risk Engine

**Multi-broker portfolio risk analytics and stress testing: one command from a JSON portfolio to an
audit-ready risk report.**

[![CI](https://github.com/ozkaraonur/portfolio-risk-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/ozkaraonur/portfolio-risk-engine/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![Typing](https://img.shields.io/badge/mypy-strict-success)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

Investors who split assets across brokers (equities at one, crypto at another, commodities at a
third) rarely see a single, consolidated risk picture. Portfolio Risk Engine aggregates positions
and cash across brokers and answers the questions a risk committee asks:

- How much can we lose over 10 days at 99% confidence? (**VaR / CVaR**)
- What does the loss distribution look like over a year, and how deep can drawdowns get? (**Monte Carlo**)
- What happens in 2008, March 2020 or 2022 again? (**Stress tests**)
- How much does diversification really save us? (**Diversification benefit**)

## Highlights

| Capability | Detail |
| --- | --- |
| Multi-broker portfolios | Same asset at several brokers is aggregated; per-broker cash; equities, crypto, commodities |
| Risk metrics | Parametric and historical-simulation VaR / CVaR at 95% / 99%, 1-day / 10-day horizons |
| Monte Carlo | Cholesky-correlated multivariate GBM, terminal percentiles, drawdown distribution, first-passage ruin probability |
| Stress testing | Built-in historical shocks, custom class / tag / symbol shocks, beta-driven market shocks |
| Web panel | Streamlit UI: portfolio builder, one-click analysis, HTML report download |
| Reporting | Rich terminal dashboard, zero-dependency HTML report (inline CSS/SVG), Markdown summary |
| Data | Offline seeded GBM provider (tests, demos) and Stooq public-data provider (no API key) |
| Engineering | Pydantic v2 models, `mypy --strict`, ruff, 90+ deterministic tests, Docker, CI on 3.11 / 3.12 |

## Architecture

```mermaid
flowchart LR
    P[portfolio.json<br/>positions, brokers, cash] --> M[models<br/>Asset / Position / Portfolio]
    subgraph Data
      S[SyntheticProvider<br/>seeded GBM] --> PP((PriceProvider))
      Q[StooqProvider<br/>public CSV] --> PP
    end
    M --> A[reporting.build_analysis]
    PP -->|prices| A
    subgraph Risk
      C[covariance / correlation] --> V[parametric + historical<br/>VaR / CVaR]
      C --> L[Cholesky / nearest PSD] --> MC[Monte Carlo paths]
      SC[scenarios + betas] --> ST[stress engine]
    end
    A --> C
    A --> SC
    V --> R[RiskAnalysis]
    MC --> R
    ST --> R
    R --> T[Rich terminal dashboard]
    R --> H[risk-report.html]
    R --> D[risk-report.md]
```

```
src/portfolio_risk/
  models/      Asset, Position, CashBalance, Portfolio (immutable pydantic v2 models)
  data/        PriceProvider ABC, SyntheticProvider (GBM), StooqProvider
  risk/        covariance, var, linalg, monte_carlo, scenarios, stress, report
  reporting/   analysis (orchestration), terminal, html, markdown
  web/         Streamlit panel (app.py) and UI-independent table -> Portfolio builder
  cli.py       Typer CLI: summary | risk | simulate | stress | report | web
```

## Quick start

```bash
git clone https://github.com/ozkaraonur/portfolio-risk-engine && cd portfolio-risk-engine
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt && pip install -e .

pre report examples/portfolio.json --html output/risk-report.html --markdown output/risk-report.md
```

No internet is needed: the default provider generates reproducible synthetic prices. Use
`--provider stooq` for public market data.

### Portfolio format

```json
{
  "name": "demo",
  "base_currency": "USD",
  "positions": [
    {"asset": {"symbol": "AAPL", "asset_class": "equity", "tags": ["tech"]}, "quantity": 50, "broker": "ibkr"},
    {"asset": {"symbol": "AAPL", "asset_class": "equity", "tags": ["tech"]}, "quantity": 20, "broker": "schwab"},
    {"asset": {"symbol": "BTC", "asset_class": "crypto"}, "quantity": 0.2, "broker": "binance"}
  ],
  "cash": [{"broker": "ibkr", "amount": 5000}]
}
```

### Commands

| Command | Purpose |
| --- | --- |
| `pre summary <file>` | Latest valuation and weights |
| `pre risk <file> --confidence 0.99 --horizon 10` | Parametric and historical VaR / CVaR, diversification, correlations |
| `pre backtest <file> --confidence 0.99 --window 250` | Rolling one-day VaR backtest: violations, Kupiec / Christoffersen tests, Basel zone |
| `pre simulate <file> -n 10000 -t 252 --seed 42` | Monte Carlo VaR / CVaR, percentiles, drawdowns, ruin probability |
| `pre stress <file> [--scenario gfc-2008] [--custom "equity=-0.15,crypto=-0.30"] [--market-shock -0.10]` | Scenario P&L per asset, worst case |
| `pre report <file> --html report.html --markdown report.md` | Everything at once: dashboard plus reports |
| `pre web` | Launch the Streamlit web panel |

```bash
pre stress examples/portfolio.json --custom "equity=-0.15,tag:tech=-0.30,BTC=-0.5"
pre stress examples/portfolio.json --market-shock -0.10 --benchmark SPY --provider stooq
python examples/run_demo.py          # offline end-to-end demo -> ./output
```

### Web panel (Streamlit)

Build a portfolio in the browser (no JSON editing), run the analysis with one click and download the
HTML report:

```bash
pre web                       # opens http://localhost:8501
pre web --port 9000 --no-browser
docker run --rm -p 8501:8501 portfolio-risk web --host 0.0.0.0 --no-browser
```

Add positions by picking a **category** (ABD Hisseleri, BIST, Kripto, Emtia), then an **asset by name**
(e.g. `THYAO - Türk Hava Yolları`, `SOL - Solana`) and entering only the quantity. Symbol, asset class,
tags (`tech`, `aviation`, `crypto`, ...) and broker are linked automatically from the built-in
[asset catalog](src/portfolio_risk/catalog.py) (35+ US stocks and ETFs, 35+ BIST stocks, 13 crypto assets,
7 commodities). The sidebar holds the cash balances, the analysis parameters (confidence 95% / 99%,
horizon 1 / 10 days, 1,000 / 5,000 simulations) and **Load Sample Portfolio**. The results view shows
the executive summary cards, VaR / CVaR, the correlation heat map, Monte Carlo percentiles and the
stress summary, with **HTML Raporu İndir** (`risk-report.html`), a Markdown download and a live
preview of the full report. Prices in the panel always come from the offline synthetic engine,
calibrated per catalog asset (volatility, drift, start price and a market/group factor structure that
yields realistic cross-asset correlations, e.g. BTC-ETH ≈ 0.75, gold-silver ≈ 0.75, S&P stocks ≈ 0.55).
The panel binds to `localhost` by default.

### Docker

```bash
docker build -t portfolio-risk .
docker run --rm portfolio-risk report examples/portfolio.json
docker run --rm -v "$(pwd)/output:/app/output" portfolio-risk report examples/portfolio.json \
    --html output/risk-report.html --markdown output/risk-report.md
docker compose up          # same, using docker-compose.yml
```

## Report preview

Terminal dashboard (`python examples/run_demo.py`: synthetic data, fixed window, seed 42):

```text
+--------------------------- Risk Dashboard: demo ----------------------------+
| Value  31,402.92 USD   Cash  19.1%   Top risk  AAPL   10d 99% VaR  1,900.79 |
+------------------- as of 2025-12-31 | 782 obs | seed 42 --------------------+
Allocation
+----------------------------------------------------------+
| Symbol |     Class |     Value | Weight | Standalone VaR |
|--------+-----------+-----------+--------+----------------|
| AAPL   |    equity | 12,051.17 |  38.4% |       1,545.20 |
| BTC    |    crypto |  1,707.13 |   5.4% |         467.72 |
| GOLD   | commodity | 11,644.62 |  37.1% |         789.71 |
| CASH   |      cash |  6,000.00 |  19.1% |              - |
+----------------------------------------------------------+
Correlation
+---------------------------+
|      | AAPL |  BTC | GOLD |
|------+------+------+------|
| AAPL | 1.00 | 0.16 | 0.05 |
| BTC  | 0.16 | 1.00 | 0.04 |
| GOLD | 0.05 | 0.04 | 1.00 |
+---------------------------+
VaR / CVaR
+-------------------------------------------------------------------+
| Method     | Horizon | Conf. |      VaR |     CVaR | Div. benefit |
|------------+---------+-------+----------+----------+--------------|
| parametric |      1d |   95% |   425.00 |   532.97 |        32.2% |
| parametric |      1d |   99% |   601.08 |   688.64 |        32.2% |
| parametric |     10d |   95% | 1,343.96 | 1,685.38 |        32.2% |
| parametric |     10d |   99% | 1,900.79 | 2,177.67 |        32.2% |
| historical |      1d |   95% |   412.18 |   509.23 |        34.1% |
| historical |      1d |   99% |   565.72 |   654.41 |        32.3% |
| historical |     10d |   95% | 1,215.23 | 1,438.04 |        37.4% |
| historical |     10d |   99% | 1,567.66 | 1,750.89 |        39.5% |
+-------------------------------------------------------------------+
Monte Carlo (10,000 x 252d)
+------------------------------------+
| Metric        |     Value | Change |
|---------------+-----------+--------|
| 5th pct       | 25,411.20 | -19.1% |
| Median        | 31,036.87 |  -1.2% |
| 95th pct      | 38,931.85 | +24.0% |
| VaR 95%       |  5,991.72 | -19.1% |
| CVaR 95%      |  7,185.71 | -22.9% |
| VaR 99%       |  7,928.62 | -25.2% |
| CVaR 99%      |  8,768.12 | -27.9% |
| P(MDD >= 10%) |     78.7% |        |
| P(MDD >= 20%) |     15.3% |        |
| P(MDD >= 30%) |      0.6% |        |
| P(ruin, -30%) |     0.31% |        |
+------------------------------------+
Stress tests
+------------------------------------------------------+
| Scenario       |       P&L | Loss % | Stressed value |
|----------------+-----------+--------+----------------|
| gfc-2008       | -4,700.61 | -15.0% |      26,702.31 |
| covid-2020     | -7,380.07 | -23.5% |      24,022.85 |
| inflation-2022 | -2,416.39 |  -7.7% |      28,986.53 |
+------------------------------------------------------+
+-------------------------------- Worst case ---------------------------------+
| covid-2020 loses 7,380.07 USD (23.5%); remaining capital 24,022.85 USD      |
+-----------------------------------------------------------------------------+
```

The full artefacts of this run are committed in [`docs/sample/`](docs/sample):
[`risk-report.html`](docs/sample/risk-report.html) (self-contained, light/dark aware, prints cleanly)
and [`risk-report.md`](docs/sample/risk-report.md). The HTML report contains: Executive Risk Summary
(portfolio value, cash ratio, highest-risk asset, 10-day 99% VaR), allocation and a correlation heat
map, statistical risk table, Monte Carlo distribution chart with drawdown probabilities, and a crisis
resilience table with per-asset drill-downs.

## Methodology

Notation: `w` is the vector of asset exposures in currency, `Σ` the covariance matrix of daily
returns, `c` the confidence level, `h` the horizon in trading days, `z_c = Φ⁻¹(c)`.

### Parametric (variance-covariance) VaR and CVaR

Zero-mean normal returns and square-root-of-time scaling:

```
σ_p  = sqrt(wᵀ Σ w) · sqrt(h)
VaR  = z_c · σ_p
CVaR = σ_p · φ(z_c) / (1 − c)          (Expected Shortfall)
```

### Historical simulation

Each historical day (compounded over `h` days, overlapping windows) is applied to today's exposures
to build a P&L distribution. VaR is its `(1 − c)` quantile loss; CVaR is the mean loss beyond it. No
distributional assumption, but limited to what the sample has seen.

### Diversification benefit

```
benefit = Σᵢ VaRᵢ(standalone) − VaR(portfolio)         ratio = benefit / Σᵢ VaRᵢ
```

Perfectly correlated assets give 0; uncorrelated equal-risk pairs give `1 − 1/√2 ≈ 29.3%`.

### Monte Carlo: correlated GBM with Itô correction

Asset prices follow geometric Brownian motion. Over one day (`dt = 1` in daily units):

```
ln(Sₜ₊₁ / Sₜ) = (μ − ½ diag(Σ)) + L z,      z ~ N(0, I),   L Lᵀ = Σ
```

`L` is the Cholesky factor of `Σ`, which makes the shocks correlate as observed. The `−½σ²` Itô term
keeps the *arithmetic* mean return equal to `μ` (default 0, i.e. risk-only; `--drift` uses historical
means). Non-positive-definite or singular estimates are repaired by eigenvalue clipping while
preserving variances (`nearest_psd`). Positions are buy-and-hold and cash earns nothing. Outputs:
horizon VaR/CVaR, terminal percentiles, the distribution of maximum drawdown, and the **first-passage
ruin probability** (chance a path *touches* the loss threshold at any time, always ≥ the probability of
ending below it).

### VaR backtesting

`pre backtest` checks whether the VaR models are honest. For every test day the one-day VaR is
estimated from the trailing `window` returns only and compared with the realised P&L of today's
exposures (a hypothetical backtest, so trading is excluded). A *violation* is a day whose loss
exceeds the forecast.

- **Kupiec POF**: is the violation rate equal to `1 - confidence`? (likelihood ratio, chi-square 1 d.o.f.)
- **Christoffersen independence**: do violations cluster? A first-order Markov chain is compared
  with a constant rate. The joint **conditional coverage** test adds both statistics (2 d.o.f.).
- **Basel traffic light**: green / yellow / red from the binomial tail of the violation count
  (0-4 / 5-9 / 10+ violations for 250 days at 99%).

Synthetic prices are Gaussian, so both models usually pass on them; the tests are meant to expose
fat tails and volatility clustering in real data.

### Stress testing

Instantaneous shocks `sᵢ` applied to current values: `P&Lᵢ = Vᵢ · sᵢ`; cash is unshocked.
Precedence per asset: symbol > tag > asset class.

| Scenario | Equity | Crypto / high risk | Commodity |
| --- | ---: | ---: | ---: |
| `gfc-2008` | −45% | −60% | +15% |
| `covid-2020` | −30% | −50% | −25% |
| `inflation-2022` | −20% (tech/growth −35%) | −65% | +25% |

Beta shocks: `βᵢ = Cov(rᵢ, r_m) / Var(r_m)` against a benchmark (default SPY); a market move `m`
hits asset `i` by `βᵢ · m` (floored at −100%), and the portfolio beta is the value-weighted average
with cash at zero.

## Assumptions and limitations

- Reports are model outputs, not forecasts or investment advice.
- Single-currency valuation (no FX); long-only positions.
- Parametric VaR assumes zero-mean normal returns (no fat tails or skew); the backtest can flag this; historical VaR needs a representative sample.
- Synthetic data is for testing and demos. Its volatilities and correlations are configurable
  defaults, not market estimates.
- The Stooq provider relies on a public endpoint and is not exercised by the test suite.

## Development

```bash
pytest                    # offline, deterministic
ruff check . && ruff format --check .
mypy                      # strict, configured in pyproject.toml
```

Conventions are documented in [`CLAUDE.md`](CLAUDE.md). CI runs all checks on Python 3.11 and 3.12
and builds the Docker image.

## License

MIT, see [`LICENSE`](LICENSE).
