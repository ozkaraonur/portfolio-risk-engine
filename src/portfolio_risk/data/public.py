"""Public-source price provider (Stooq daily CSV; no API key required)."""

from __future__ import annotations

import io
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from datetime import date

import pandas as pd
from loguru import logger

from portfolio_risk.data.base import DataUnavailableError, PriceProvider, validate_range
from portfolio_risk.models import Asset

Fetcher = Callable[[str], str]

STOOQ_URL = "https://stooq.com/q/d/l/"


def _http_fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "portfolio-risk-engine/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=15) as response:
            return str(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError) as exc:
        raise DataUnavailableError(f"Request failed: {url}: {exc}") from exc


class StooqProvider(PriceProvider):
    """Daily close prices from stooq.com.

    Symbols follow Stooq conventions (e.g. ``aapl.us``, ``btc.v``, ``gc.f``). Set
    ``Asset.data_symbol`` to override; otherwise ``<symbol>.us`` is used for equities and
    ``<symbol>.v`` for crypto. Pass ``fetcher`` to replace the HTTP layer (used in tests).
    """

    def __init__(self, fetcher: Fetcher | None = None) -> None:
        self._fetch = fetcher or _http_fetch

    @staticmethod
    def stooq_symbol(asset: Asset) -> str:
        if asset.data_symbol:
            return asset.data_symbol.lower()
        suffix = {"equity": ".us", "crypto": ".v", "commodity": ".f"}[asset.asset_class.value]
        return f"{asset.symbol.lower()}{suffix}"

    def _url(self, symbol: str, start: date, end: date) -> str:
        query = urllib.parse.urlencode(
            {"s": symbol, "d1": start.strftime("%Y%m%d"), "d2": end.strftime("%Y%m%d"), "i": "d"}
        )
        return f"{STOOQ_URL}?{query}"

    def _series(self, asset: Asset, start: date, end: date) -> pd.Series[float]:
        symbol = self.stooq_symbol(asset)
        logger.debug("Fetching {} from Stooq", symbol)
        text = self._fetch(self._url(symbol, start, end))
        try:
            frame = pd.read_csv(io.StringIO(text), parse_dates=["Date"], index_col="Date")
            close = frame["Close"].astype(float)
        except (ValueError, KeyError, pd.errors.ParserError) as exc:
            raise DataUnavailableError(f"No usable data for {asset.symbol} ({symbol}).") from exc
        if close.empty:
            raise DataUnavailableError(f"No data returned for {asset.symbol} ({symbol}).")
        close.name = asset.symbol
        return close.sort_index()

    def get_prices(self, assets: Sequence[Asset], start: date, end: date) -> pd.DataFrame:
        validate_range(start, end)
        series = [self._series(a, start, end) for a in assets]
        # Forward-fill so assets with different trading calendars (e.g. crypto) align.
        return pd.concat(series, axis=1).sort_index().ffill().dropna()
