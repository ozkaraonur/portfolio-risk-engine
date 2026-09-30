"""Foreign-exchange rates and conversion of a portfolio into its base currency."""

from __future__ import annotations

import zlib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import date

import numpy as np
import pandas as pd

from portfolio_risk.data.base import DataUnavailableError, validate_range
from portfolio_risk.data.public import StooqProvider
from portfolio_risk.data.synthetic import EPOCH, GBMParams, simulate_gbm
from portfolio_risk.models import CashBalance, Portfolio

# Indicative USD value of one unit of each currency, used to anchor the synthetic rates.
USD_PER_UNIT = {
    "USD": 1.0,
    "EUR": 1.08,
    "GBP": 1.27,
    "CHF": 1.12,
    "JPY": 0.0067,
    "CAD": 0.73,
    "AUD": 0.65,
    "TRY": 0.031,
}
SYNTHETIC_FX_VOL = 0.09
SYNTHETIC_TRY_VOL = 0.18


class FxProvider(ABC):
    """Source of exchange rates."""

    @abstractmethod
    def get_rates(
        self, currencies: Sequence[str], base: str, start: date, end: date
    ) -> pd.DataFrame:
        """Base-currency value of one unit of each currency.

        Ascending DatetimeIndex, one column per requested currency (upper case). Must raise
        ``DataUnavailableError`` rather than return partial data.
        """


class SyntheticFx(FxProvider):
    """Reproducible random-walk rates; every pair is derived through USD so crosses agree."""

    def __init__(self, seed: int = 42) -> None:
        self._seed = seed

    def _usd_path(self, currency: str, index: pd.DatetimeIndex) -> np.ndarray:
        """USD value of one unit on each date of ``index``; the last date is the anchor rate."""
        if currency == "USD":
            return np.ones(len(index))
        if currency not in USD_PER_UNIT:
            raise DataUnavailableError(f"No synthetic FX profile for {currency}.")
        vol = SYNTHETIC_TRY_VOL if currency == "TRY" else SYNTHETIC_FX_VOL
        full = pd.bdate_range(EPOCH, index[-1])  # same date -> same shock, whatever the window
        rng = np.random.default_rng([self._seed, 3, zlib.crc32(currency.encode())])
        params = GBMParams(mu=0.0, sigma=vol, start_price=USD_PER_UNIT[currency])
        path = simulate_gbm(params, rng.standard_normal(len(full) - 1))
        anchored = np.asarray(path * (params.start_price / path[-1]), dtype=np.float64)
        return np.asarray(anchored[full.get_indexer(index)], dtype=np.float64)

    def get_rates(
        self, currencies: Sequence[str], base: str, start: date, end: date
    ) -> pd.DataFrame:
        validate_range(start, end)
        index = pd.bdate_range(start, end)
        if len(index) < 2:
            raise ValueError("Date range must span at least two business days.")
        if index[0] < EPOCH:
            raise ValueError(f"Synthetic data starts at {EPOCH.date()}; got start {start}.")
        base_path = self._usd_path(base.upper(), index)
        return pd.DataFrame(
            {c.upper(): self._usd_path(c.upper(), index) / base_path for c in currencies},
            index=index,
        )


class StooqFx(FxProvider):
    """Rates from Stooq currency pairs (``eurusd`` = USD per EUR); inverse pairs are tried too."""

    def __init__(self, stooq: StooqProvider | None = None) -> None:
        self._stooq = stooq or StooqProvider()

    def _rate(self, currency: str, base: str, start: date, end: date) -> pd.Series[float]:
        direct = f"{currency}{base}".lower()
        try:
            return self._stooq.close_series(direct, direct, start, end)
        except DataUnavailableError:
            inverse = f"{base}{currency}".lower()
            return 1.0 / self._stooq.close_series(inverse, inverse, start, end)

    def get_rates(
        self, currencies: Sequence[str], base: str, start: date, end: date
    ) -> pd.DataFrame:
        validate_range(start, end)
        columns = {c.upper(): self._rate(c.upper(), base.upper(), start, end) for c in currencies}
        frame = pd.DataFrame(columns).sort_index().ffill().dropna()
        if frame.empty:
            raise DataUnavailableError(f"No FX data for {sorted(columns)} against {base}.")
        return frame


def foreign_currencies(portfolio: Portfolio) -> list[str]:
    """Currencies (other than the base) that positions or cash are held in."""
    found = {a.currency for a in portfolio.assets} | {c.currency for c in portfolio.cash}
    found.discard(portfolio.base_currency)
    return sorted(found)


def convert_to_base(
    portfolio: Portfolio, prices: pd.DataFrame, fx: FxProvider, start: date, end: date
) -> tuple[Portfolio, pd.DataFrame]:
    """Express prices and cash in the portfolio's base currency.

    A foreign-currency asset's price series is multiplied by the daily rate, so the returns fed to
    the risk models include FX moves. Foreign cash is converted at the latest rate and carries no
    FX risk of its own (it is added to the base-currency cash as a fixed amount).
    """
    needed = foreign_currencies(portfolio)
    if not needed:
        return portfolio, prices
    rates = fx.get_rates(needed, portfolio.base_currency, start, end)
    missing = [c for c in needed if c not in rates.columns]
    if missing:
        raise DataUnavailableError(f"No FX rates for: {missing}")

    converted = prices.copy()
    for asset in portfolio.assets:
        if asset.currency == portfolio.base_currency:
            continue
        rate = rates[asset.currency].reindex(prices.index.union(rates.index)).ffill().bfill()
        converted[asset.symbol] = prices[asset.symbol] * rate.reindex(prices.index)
    if converted.isna().any().any():
        raise DataUnavailableError("FX rates do not cover the price history.")

    latest = {c: float(rates[c].iloc[-1]) for c in needed}
    totals: dict[str, float] = {}
    for bal in portfolio.cash:
        fx_rate = 1.0 if bal.currency == portfolio.base_currency else latest[bal.currency]
        totals[bal.broker] = totals.get(bal.broker, 0.0) + bal.amount * fx_rate
    cash = tuple(
        CashBalance(broker=b, amount=v, currency=portfolio.base_currency) for b, v in totals.items()
    )
    return portfolio.model_copy(update={"cash": cash}), converted
