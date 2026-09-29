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
  cli.py       Typer CLI: summary | risk | simulate | stress | report
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
    {"asset": {"symbol": "BTC", "asset_class": "crypto"}, "quantity": 25, "broker": "binance"}
  ],
  "cash": [{"broker": "ibkr", "amount": 5000}]
}
```

### Commands

| Command | Purpose |
| --- | --- |
| `pre summary <file>` | Latest valuation and weights |
| `pre risk <file> --confidence 0.99 --horizon 10` | Parametric and historical VaR / CVaR, diversification, correlations |
| `pre simulate <file> -n 10000 -t 252 --seed 42` | Monte Carlo VaR / CVaR, percentiles, drawdowns, ruin probability |
| `pre stress <file> [--scenario gfc-2008] [--custom "equity=-0.15,crypto=-0.30"] [--market-shock -0.10]` | Scenario P&L per asset, worst case |
| `pre report <file> --html report.html --markdown report.md` | Everything at once: dashboard plus reports |

```bash
pre stress examples/portfolio.json --custom "equity=-0.15,tag:tech=-0.30,BTC=-0.5"
pre stress examples/portfolio.json --market-shock -0.10 --benchmark SPY --provider stooq
python examples/run_demo.py          # offline end-to-end demo -> ./output
```

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
| Value  13,504.94 USD   Cash  44.4%   Top risk  AAPL   10d 99% VaR  761.94   |
+------------------- as of 2025-12-31 | 782 obs | seed 42 --------------------+
Allocation
+---------------------------------------------------------+
| Symbol |     Class |    Value | Weight | Standalone VaR |
|--------+-----------+----------+--------+----------------|
| AAPL   |    equity | 6,043.33 |  44.7% |         612.28 |
| BTC    |    crypto |   766.58 |   5.7% |         262.90 |
| GC     | commodity |   695.03 |   5.1% |          65.02 |
| CASH   |      cash | 6,000.00 |  44.4% |              - |
+---------------------------------------------------------+
Correlation
+---------------------------+
|      | AAPL |  BTC |   GC |
|------+------+------+------|
| AAPL | 1.00 | 0.31 | 0.28 |
| BTC  | 0.31 | 1.00 | 0.27 |
| GC   | 0.28 | 0.27 | 1.00 |
+---------------------------+
VaR / CVaR
+---------------------------------------------------------------+
| Method     | Horizon | Conf. |    VaR |   CVaR | Div. benefit |
|------------+---------+-------+--------+--------+--------------|
| parametric |      1d |   95% | 170.36 | 213.64 |        19.0% |
| parametric |      1d |   99% | 240.95 | 276.04 |        19.0% |
| parametric |     10d |   95% | 538.73 | 675.59 |        19.0% |
| parametric |     10d |   99% | 761.94 | 872.93 |        19.0% |
| historical |      1d |   95% | 173.27 | 209.81 |        19.1% |
| historical |      1d |   99% | 235.98 | 264.30 |        18.9% |
| historical |     10d |   95% | 546.17 | 674.80 |        16.5% |
| historical |     10d |   99% | 782.31 | 878.50 |        16.2% |
+---------------------------------------------------------------+
Monte Carlo (10,000 x 252d)
+------------------------------------+
| Metric        |     Value | Change |
|---------------+-----------+--------|
| 5th pct       | 11,154.82 | -17.4% |
| Median        | 13,323.30 |  -1.3% |
| 95th pct      | 16,518.99 | +22.3% |
| VaR 95%       |  2,350.12 | -17.4% |
| CVaR 95%      |  2,761.50 | -20.4% |
| VaR 99%       |  3,002.64 | -22.2% |
| CVaR 99%      |  3,298.31 | -24.4% |
| P(MDD >= 10%) |     74.5% |        |
| P(MDD >= 20%) |     10.2% |        |
| P(MDD >= 30%) |      0.1% |        |
| P(ruin, -30%) |     0.01% |        |
+------------------------------------+
Stress tests
+------------------------------------------------------+
| Scenario       |       P&L | Loss % | Stressed value |
|----------------+-----------+--------+----------------|
| gfc-2008       | -3,075.20 | -22.8% |      10,429.75 |
| covid-2020     | -2,370.05 | -17.5% |      11,134.90 |
| inflation-2022 | -2,439.69 | -18.1% |      11,065.26 |
+------------------------------------------------------+
+-------------------------------- Worst case ---------------------------------+
| gfc-2008 loses 3,075.20 USD (22.8%); remaining capital 10,429.75 USD        |
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
- Parametric VaR ignores fat tails and skew; historical VaR needs a representative sample.
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
