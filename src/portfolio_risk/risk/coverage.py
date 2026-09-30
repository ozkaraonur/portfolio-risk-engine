"""Statistical tests for VaR exceedance sequences (Kupiec, Christoffersen, Basel zones).

A violation is a day whose loss exceeds the VaR forecast. With confidence ``c`` a correct model
violates with probability ``p = 1 - c``, independently from day to day.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt
from scipy.stats import binom, chi2

BoolArray = npt.NDArray[np.bool_]

# Basel cumulative-binomial cut-offs (95% and 99.99%); for 250 observations at 99% confidence
# they give the Basel zones of 0-4 / 5-9 / 10+ violations.
YELLOW_FROM = 0.95
RED_FROM = 0.9999


class Zone(StrEnum):
    GREEN = "green"
    YELLOW = "yellow"
    RED = "red"


@dataclass(frozen=True)
class LikelihoodRatioTest:
    statistic: float
    p_value: float

    def rejects(self, significance: float = 0.05) -> bool:
        """True if the model is rejected at the given significance level."""
        return self.p_value < significance


def _xlogy(x: float, y: float) -> float:
    """``x * ln(y)`` with the convention ``0 * ln(0) = 0``."""
    return 0.0 if x == 0 else x * float(np.log(y))


def _check_confidence(confidence: float) -> float:
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be in (0.5, 1).")
    return 1.0 - confidence


def kupiec_pof(violations: BoolArray, confidence: float) -> LikelihoodRatioTest:
    """Kupiec proportion-of-failures test: is the violation rate equal to ``1 - confidence``?"""
    p = _check_confidence(confidence)
    n = int(violations.size)
    if n == 0:
        raise ValueError("violations must not be empty.")
    x = int(violations.sum())
    observed = x / n
    log_null = _xlogy(n - x, 1.0 - p) + _xlogy(x, p)
    log_alt = _xlogy(n - x, 1.0 - observed) + _xlogy(x, observed)
    stat = max(-2.0 * (log_null - log_alt), 0.0)
    return LikelihoodRatioTest(stat, float(chi2.sf(stat, df=1)))


def christoffersen_independence(violations: BoolArray) -> LikelihoodRatioTest:
    """Christoffersen test: are violations independent (no clustering) day to day?

    Compares a first-order Markov chain for the violation indicator against a constant rate.
    Without any violation (or with all days violating) there is no evidence against
    independence, so the p-value is 1.
    """
    if violations.size < 2:
        raise ValueError("At least two observations are required.")
    prev, curr = violations[:-1], violations[1:]
    n00 = int((~prev & ~curr).sum())
    n01 = int((~prev & curr).sum())
    n10 = int((prev & ~curr).sum())
    n11 = int((prev & curr).sum())
    total = n00 + n01 + n10 + n11
    if n01 + n11 == 0 or n00 + n10 == 0:
        return LikelihoodRatioTest(0.0, 1.0)

    pi = (n01 + n11) / total
    pi0 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi1 = n11 / (n10 + n11) if n10 + n11 else 0.0
    log_null = _xlogy(n00 + n10, 1.0 - pi) + _xlogy(n01 + n11, pi)
    log_alt = _xlogy(n00, 1.0 - pi0) + _xlogy(n01, pi0) + _xlogy(n10, 1.0 - pi1) + _xlogy(n11, pi1)
    stat = max(-2.0 * (log_null - log_alt), 0.0)
    return LikelihoodRatioTest(stat, float(chi2.sf(stat, df=1)))


def conditional_coverage(violations: BoolArray, confidence: float) -> LikelihoodRatioTest:
    """Christoffersen joint test: correct rate *and* independence (chi-square, 2 d.o.f.)."""
    stat = (
        kupiec_pof(violations, confidence).statistic
        + christoffersen_independence(violations).statistic
    )
    return LikelihoodRatioTest(stat, float(chi2.sf(stat, df=2)))


def basel_zone(n_violations: int, n_obs: int, confidence: float) -> Zone:
    """Basel traffic-light zone from the binomial tail of ``n_violations`` in ``n_obs`` days."""
    p = _check_confidence(confidence)
    if n_obs < 1 or not 0 <= n_violations <= n_obs:
        raise ValueError("Need n_obs >= 1 and 0 <= n_violations <= n_obs.")
    cumulative = float(binom.cdf(n_violations, n_obs, p))
    if cumulative < YELLOW_FROM:
        return Zone.GREEN
    return Zone.YELLOW if cumulative < RED_FROM else Zone.RED
