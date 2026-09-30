# Risk Report: demo

*As of 2025-12-31 | base currency USD | 782 daily observations | seed 42*

## Executive Risk Summary

| Metric | Value |
| --- | ---: |
| Total portfolio value | 31,402.92 USD |
| Cash ratio | 19.1% |
| Highest-risk asset | AAPL (1,545.20 standalone VaR) |
| 10-day 99% VaR (parametric) | 1,900.79 USD (6.05%) |
| 10-day 99% CVaR (parametric) | 2,177.67 USD |
| Diversification benefit | 32.2% |

## Allocation

| Symbol | Class | Value | Weight | Standalone 10d 99% VaR |
| --- | --- | ---: | ---: | ---: |
| AAPL | equity | 12,051.17 | 38.4% | 1,545.20 |
| BTC | crypto | 1,707.13 | 5.4% | 467.72 |
| GOLD | commodity | 11,644.62 | 37.1% | 789.71 |
| CASH | cash | 6,000.00 | 19.1% | 0.00 |

### Correlation matrix

|  | AAPL | BTC | GOLD |
| --- | ---: | ---: | ---: |
| AAPL | 1.00 | 0.16 | 0.05 |
| BTC | 0.16 | 1.00 | 0.04 |
| GOLD | 0.05 | 0.04 | 1.00 |

## Statistical Risk

| Method | Horizon | Conf. | VaR | CVaR | Div. benefit |
| --- | --- | --- | ---: | ---: | ---: |
| parametric | 1d | 95% | 425.00 | 532.97 | 32.2% |
| parametric | 1d | 99% | 601.08 | 688.64 | 32.2% |
| parametric | 10d | 95% | 1,343.96 | 1,685.38 | 32.2% |
| parametric | 10d | 99% | 1,900.79 | 2,177.67 | 32.2% |
| historical | 1d | 95% | 412.18 | 509.23 | 34.1% |
| historical | 1d | 99% | 565.72 | 654.41 | 32.3% |
| historical | 10d | 95% | 1,215.23 | 1,438.04 | 37.4% |
| historical | 10d | 99% | 1,567.66 | 1,750.89 | 39.5% |

## Model Validation (VaR Backtest)

One-day 99% VaR, 532 test days, 250-day estimation window.

| Method | Violations | Expected | Kupiec p | Independence p | Basel zone |
| --- | ---: | ---: | ---: | ---: | ---: |
| parametric | 4 | 5.3 | 0.547 | 0.805 | green |
| historical | 10 | 5.3 | 0.069 | 0.536 | yellow |
| ewma | 4 | 5.3 | 0.547 | 0.805 | green |
| student-t | 4 | 5.3 | 0.547 | 0.805 | green |
| cornish-fisher | 4 | 5.3 | 0.547 | 0.805 | green |
| fhs | 9 | 5.3 | 0.145 | 0.577 | yellow |

## Monte Carlo (10,000 paths x 252 days)

| Terminal value | Amount | Change |
| --- | ---: | ---: |
| 5th percentile | 25,411.20 | -19.1% |
| Median | 31,036.87 | -1.2% |
| Mean | 31,406.42 | +0.0% |
| 95th percentile | 38,931.85 | +24.0% |

| Confidence | MC VaR | MC CVaR |
| --- | ---: | ---: |
| 95% | 5,991.72 | 7,185.71 |
| 99% | 7,928.62 | 8,768.12 |

| Max drawdown | Probability |
| --- | ---: |
| >= 10% | 78.7% |
| >= 20% | 15.3% |
| >= 30% | 0.6% |
| >= 50% | 0.0% |

Probability of losing 30% or more: **0.20%** at horizon end, **0.31%** at any point (ruin).

## Crisis Resilience (Stress Tests)

| Scenario | P&L | Loss % | Stressed value |
| --- | ---: | ---: | ---: |
| gfc-2008 (2008 Global Financial Crisis (Lehman)) | -4,700.61 | -15.0% | 26,702.31 |
| covid-2020 (March 2020 COVID liquidity shock) | -7,380.07 | -23.5% | 24,022.85 |
| inflation-2022 (2022 inflation / rate shock and tech sell-off) | -2,416.39 | -7.7% | 28,986.53 |

**Worst case:** covid-2020 loses 7,380.07 USD (23.5%), leaving 24,022.85 USD.

---
*Model outputs, not forecasts. VaR figures assume normal returns (parametric) or repeat of the sample (historical); stress shocks are instantaneous.*
