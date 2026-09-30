# Risk Report: demo

*As of 2025-12-31 | base currency USD | 782 daily observations | seed 42*

## Executive Risk Summary

| Metric | Value |
| --- | ---: |
| Total portfolio value | 39,500.00 USD |
| Cash ratio | 15.2% |
| Highest-risk asset | BTC (3,605.48 standalone VaR) |
| 10-day 99% VaR (parametric) | 4,422.31 USD (11.20%) |
| 10-day 99% CVaR (parametric) | 5,066.48 USD |
| Diversification benefit | 25.7% |

## Allocation

| Symbol | Class | Value | Weight | Standalone 10d 99% VaR |
| --- | --- | ---: | ---: | ---: |
| AAPL | equity | 13,300.00 | 33.7% | 1,832.70 |
| BTC | crypto | 13,000.00 | 32.9% | 3,605.48 |
| GOLD | commodity | 7,200.00 | 18.2% | 509.86 |
| CASH | cash | 6,000.00 | 15.2% | 0.00 |

### Correlation matrix

|  | AAPL | BTC | GOLD |
| --- | ---: | ---: | ---: |
| AAPL | 1.00 | 0.22 | 0.03 |
| BTC | 0.22 | 1.00 | 0.01 |
| GOLD | 0.03 | 0.01 | 1.00 |

## Statistical Risk

| Method | Horizon | Conf. | VaR | CVaR | Div. benefit |
| --- | --- | --- | ---: | ---: | ---: |
| parametric | 1d | 95% | 988.78 | 1,239.97 | 25.7% |
| parametric | 1d | 99% | 1,398.46 | 1,602.16 | 25.7% |
| parametric | 10d | 95% | 3,126.81 | 3,921.14 | 25.7% |
| parametric | 10d | 99% | 4,422.31 | 5,066.48 | 25.7% |
| historical | 1d | 95% | 906.02 | 1,104.93 | 27.9% |
| historical | 1d | 99% | 1,234.25 | 1,428.58 | 26.2% |
| historical | 10d | 95% | 3,078.00 | 3,735.41 | 18.8% |
| historical | 10d | 99% | 4,140.39 | 4,383.86 | 23.7% |

## Risk Attribution

| Symbol | Exposure | VaR contribution | Share | CVaR contribution |
| --- | ---: | ---: | ---: | ---: |
| AAPL | 13,300.00 | 1,088.35 | 24.6% | 1,246.89 |
| BTC | 13,000.00 | 3,266.29 | 73.9% | 3,742.07 |
| GOLD | 7,200.00 | 67.66 | 1.5% | 77.52 |

## Portfolio Optimisation

| Allocation | AAPL | BTC | GOLD | Exp. return | Volatility | 10d VaR | VaR change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| current | 39.7% | 38.8% | 21.5% | 17.6% | 28.5% | 4,422.31 |  |
| min-variance | 18.6% | 3.0% | 78.4% | 7.3% | 13.6% | 2,113.98 | -52.2% |
| risk-parity | 27.5% | 13.8% | 58.8% | 10.8% | 15.8% | 2,451.82 | -44.6% |
| max-sharpe | 45.5% | 12.5% | 42.0% | 13.0% | 18.1% | 2,816.84 | -36.3% |

## Model Validation (VaR Backtest)

One-day 99% VaR, 532 test days, 250-day estimation window.

| Method | Violations | Expected | Kupiec p | Independence p | Basel zone |
| --- | ---: | ---: | ---: | ---: | ---: |
| parametric | 3 | 5.3 | 0.271 | 0.854 | green |
| historical | 6 | 5.3 | 0.772 | 0.711 | green |
| ewma | 3 | 5.3 | 0.271 | 0.854 | green |
| student-t | 3 | 5.3 | 0.271 | 0.854 | green |
| cornish-fisher | 3 | 5.3 | 0.271 | 0.854 | green |
| fhs | 5 | 5.3 | 0.888 | 0.758 | green |

## Monte Carlo (10,000 paths x 252 days)

| Terminal value | Amount | Change |
| --- | ---: | ---: |
| 5th percentile | 27,266.99 | -31.0% |
| Median | 37,660.70 | -4.7% |
| Mean | 39,577.73 | +0.2% |
| 95th percentile | 58,543.22 | +48.2% |

| Confidence | MC VaR | MC CVaR |
| --- | ---: | ---: |
| 95% | 12,233.01 | 14,052.29 |
| 99% | 15,260.79 | 16,530.04 |

| Max drawdown | Probability |
| --- | ---: |
| >= 10% | 99.8% |
| >= 20% | 70.8% |
| >= 30% | 25.6% |
| >= 50% | 0.1% |

Probability of losing 30% or more: **5.98%** at horizon end, **10.55%** at any point (ruin).

## Crisis Resilience (Stress Tests)

| Scenario | P&L | Loss % | Stressed value |
| --- | ---: | ---: | ---: |
| gfc-2008 (2008 Global Financial Crisis (Lehman)) | -12,705.00 | -32.2% | 26,795.00 |
| covid-2020 (March 2020 COVID liquidity shock) | -12,290.00 | -31.1% | 27,210.00 |
| inflation-2022 (2022 inflation / rate shock and tech sell-off) | -11,305.00 | -28.6% | 28,195.00 |

**Worst case:** gfc-2008 loses 12,705.00 USD (32.2%), leaving 26,795.00 USD.

---
*Model outputs, not forecasts. VaR figures assume normal returns (parametric) or repeat of the sample (historical); stress shocks are instantaneous.*
