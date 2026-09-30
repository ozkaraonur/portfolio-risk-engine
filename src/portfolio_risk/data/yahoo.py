"""Yahoo Finance provider (public JSON chart endpoint; no API key).

Covers US equities, BIST (``.IS``), crypto (``-USD``), futures (``=F``) and FX pairs (``=X``).
Prices are split- and dividend-adjusted closes, so the returns derived from them are total returns.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime, timedelta

import pandas as pd
from loguru import logger

from portfolio_risk.data.base import DataUnavailableError, PriceProvider, validate_range
from portfolio_risk.data.fx import FxProvider
from portfolio_risk.models import Asset, AssetClass

Fetcher = Callable[[str], str]

YAHOO_URL = "https://query1.finance.yahoo.com/v8/finance/chart/"
COMMODITY_FUTURES = {
    "GOLD": "GC=F",
    "SILVER": "SI=F",
    "PLATINUM": "PL=F",
    "OIL": "CL=F",
    "NATGAS": "NG=F",
    "COPPER": "HG=F",
}


def _http_fetch(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urllib.request.urlopen(request, timeout=20) as response:
            return str(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise DataUnavailableError(f"Yahoo returned HTTP {exc.code} for {url}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        raise DataUnavailableError(f"Request failed: {url}: {exc}") from exc


def _epoch(day: date) -> int:
    return int(datetime(day.year, day.month, day.day, tzinfo=UTC).timestamp())


class YahooProvider(PriceProvider):
    """Daily adjusted closes from Yahoo Finance. Pass ``fetcher`` to replace HTTP (tests)."""

    def __init__(self, fetcher: Fetcher | None = None) -> None:
        self._fetch = fetcher or _http_fetch

    @staticmethod
    def yahoo_symbol(asset: Asset) -> str:
        """Yahoo ticker for ``asset``.

        Catalog symbols map by category (BIST -> ``THYAO.IS``, commodities -> ``GC=F``). Other
        assets use ``data_symbol`` verbatim when set (it must then be a Yahoo ticker), otherwise
        ``SYMBOL`` for equities and ``SYMBOL-USD`` for crypto.
        """
        from portfolio_risk.catalog import BIST, CATALOG  # lazy: catalog imports this package

        entry = CATALOG.get(asset.symbol)
        if entry is not None:
            if entry.category == BIST:
                return f"{asset.symbol}.IS"
            if asset.asset_class is AssetClass.COMMODITY and asset.symbol in COMMODITY_FUTURES:
                return COMMODITY_FUTURES[asset.symbol]
        elif asset.data_symbol:
            return asset.data_symbol
        if asset.asset_class is AssetClass.CRYPTO:
            return f"{asset.symbol}-USD"
        if asset.asset_class is AssetClass.COMMODITY:
            raise DataUnavailableError(
                f"No Yahoo ticker known for commodity {asset.symbol}; set data_symbol (e.g. GC=F)."
            )
        return asset.symbol

    def close_series(self, ticker: str, label: str, start: date, end: date) -> pd.Series[float]:
        """Adjusted daily closes for a raw Yahoo ticker over ``[start, end]``."""
        query = urllib.parse.urlencode(
            {
                "period1": _epoch(start),
                "period2": _epoch(end + timedelta(days=1)),
                "interval": "1d",
                "events": "history",
            }
        )
        logger.debug("Fetching {} from Yahoo", ticker)
        text = self._fetch(f"{YAHOO_URL}{urllib.parse.quote(ticker)}?{query}")
        try:
            payload = json.loads(text)
            error = payload["chart"].get("error")
            if error:
                raise DataUnavailableError(f"Yahoo: {error.get('description', error)} ({ticker})")
            result = payload["chart"]["result"][0]
            stamps = result["timestamp"]
            adjusted = result["indicators"].get("adjclose")
            closes = (
                adjusted[0]["adjclose"] if adjusted else result["indicators"]["quote"][0]["close"]
            )
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise DataUnavailableError(f"No usable data for {label} ({ticker}).") from exc
        index = pd.to_datetime(stamps, unit="s").normalize()
        series = pd.Series(closes, index=index, dtype="float64", name=label).dropna()
        series = series[~series.index.duplicated(keep="last")].sort_index()
        if series.empty:
            raise DataUnavailableError(f"No data returned for {label} ({ticker}).")
        return series

    def get_prices(self, assets: Sequence[Asset], start: date, end: date) -> pd.DataFrame:
        validate_range(start, end)
        series = [self.close_series(self.yahoo_symbol(a), a.symbol, start, end) for a in assets]
        # Forward-fill so assets with different trading calendars (e.g. crypto) align.
        return pd.concat(series, axis=1, sort=True).ffill().dropna()


class YahooFx(FxProvider):
    """Rates from Yahoo pairs (``EURUSD=X`` = USD per EUR); the inverse pair is a fallback."""

    def __init__(self, yahoo: YahooProvider | None = None) -> None:
        self._yahoo = yahoo or YahooProvider()

    def _rate(self, currency: str, base: str, start: date, end: date) -> pd.Series[float]:
        direct = f"{currency}{base}=X"
        try:
            return self._yahoo.close_series(direct, direct, start, end)
        except DataUnavailableError:
            inverse = f"{base}{currency}=X"
            return 1.0 / self._yahoo.close_series(inverse, inverse, start, end)

    def get_rates(
        self, currencies: Sequence[str], base: str, start: date, end: date
    ) -> pd.DataFrame:
        validate_range(start, end)
        columns = {c.upper(): self._rate(c.upper(), base.upper(), start, end) for c in currencies}
        frame = pd.DataFrame(columns).sort_index().ffill().dropna()
        if frame.empty:
            raise DataUnavailableError(f"No FX data for {sorted(columns)} against {base}.")
        return frame
