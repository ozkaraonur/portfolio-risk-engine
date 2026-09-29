# Risk Report: demo

*As of 2025-12-31 | base currency USD | 782 daily observations | seed 42*

## Executive Risk Summary

| Metric | Value |
| --- | ---: |
| Total portfolio value | 13,504.94 USD |
| Cash ratio | 44.4% |
| Highest-risk asset | AAPL (612.28 standalone VaR) |
| 10-day 99% VaR (parametric) | 761.94 USD (5.64%) |
| 10-day 99% CVaR (parametric) | 872.93 USD |
| Diversification benefit | 19.0% |

## Allocation

| Symbol | Class | Value | Weight | Standalone 10d 99% VaR |
| --- | --- | ---: | ---: | ---: |
| AAPL | equity | 6,043.33 | 44.7% | 612.28 |
| BTC | crypto | 766.58 | 5.7% | 262.90 |
| GC | commodity | 695.03 | 5.1% | 65.02 |
| CASH | cash | 6,000.00 | 44.4% | 0.00 |

### Correlation matrix

|  | AAPL | BTC | GC |
| --- | ---: | ---: | ---: |
| AAPL | 1.00 | 0.31 | 0.28 |
| BTC | 0.31 | 1.00 | 0.27 |
| GC | 0.28 | 0.27 | 1.00 |

## Statistical Risk

| Method | Horizon | Conf. | VaR | CVaR | Div. benefit |
| --- | --- | --- | ---: | ---: | ---: |
| parametric | 1d | 95% | 170.36 | 213.64 | 19.0% |
| parametric | 1d | 99% | 240.95 | 276.04 | 19.0% |
| parametric | 10d | 95% | 538.73 | 675.59 | 19.0% |
| parametric | 10d | 99% | 761.94 | 872.93 | 19.0% |
| historical | 1d | 95% | 173.27 | 209.81 | 19.1% |
| historical | 1d | 99% | 235.98 | 264.30 | 18.9% |
| historical | 10d | 95% | 546.17 | 674.80 | 16.5% |
| historical | 10d | 99% | 782.31 | 878.50 | 16.2% |

## Monte Carlo (10,000 paths x 252 days)

| Terminal value | Amount | Change |
| --- | ---: | ---: |
| 5th percentile | 11,154.82 | -17.4% |
| Median | 13,323.30 | -1.3% |
| Mean | 13,507.99 | +0.0% |
| 95th percentile | 16,518.99 | +22.3% |

| Confidence | MC VaR | MC CVaR |
| --- | ---: | ---: |
| 95% | 2,350.12 | 2,761.50 |
| 99% | 3,002.64 | 3,298.31 |

| Max drawdown | Probability |
| --- | ---: |
| >= 10% | 74.5% |
| >= 20% | 10.2% |
| >= 30% | 0.1% |
| >= 50% | 0.0% |

Probability of losing 30% or more: **0.00%** at horizon end, **0.01%** at any point (ruin).

## Crisis Resilience (Stress Tests)

| Scenario | P&L | Loss % | Stressed value |
| --- | ---: | ---: | ---: |
| gfc-2008 (2008 Global Financial Crisis (Lehman)) | -3,075.20 | -22.8% | 10,429.75 |
| covid-2020 (March 2020 COVID liquidity shock) | -2,370.05 | -17.5% | 11,134.90 |
| inflation-2022 (2022 inflation / rate shock and tech sell-off) | -2,439.69 | -18.1% | 11,065.26 |

**Worst case:** gfc-2008 loses 3,075.20 USD (22.8%), leaving 10,429.75 USD.

---
*Model outputs, not forecasts. VaR figures assume normal returns (parametric) or repeat of the sample (historical); stress shocks are instantaneous.*
