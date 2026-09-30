"""On-disk price cache that wraps any :class:`PriceProvider`."""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

import pandas as pd
from loguru import logger

from portfolio_risk.data.base import PriceProvider, validate_range
from portfolio_risk.models import Asset

DEFAULT_TTL_HOURS = 12.0
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def default_cache_dir() -> Path:
    return Path.home() / ".cache" / "portfolio-risk-engine"


class CachedProvider(PriceProvider):
    """Caches each asset's price series as JSON under ``cache_dir``.

    A request is served from disk when a cached fetch covered the requested date range and is
    younger than ``ttl_hours``; otherwise the asset is re-fetched from ``inner`` for exactly the
    requested range and the entry is replaced. Assets are fetched one at a time so a symbol that
    is already cached is never downloaded again because another one was missing.

    ``namespace`` separates entries of different sources (e.g. ``"stooq"``). Pass ``clock`` to
    control time in tests.
    """

    def __init__(
        self,
        inner: PriceProvider,
        cache_dir: Path | None = None,
        *,
        namespace: str = "prices",
        ttl_hours: float = DEFAULT_TTL_HOURS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if ttl_hours < 0:
            raise ValueError("ttl_hours must be non-negative.")
        self._inner = inner
        self._dir = (cache_dir or default_cache_dir()) / _UNSAFE.sub("_", namespace)
        self._ttl = ttl_hours * 3600.0
        self._clock = clock

    def _path(self, asset: Asset) -> Path:
        key = _UNSAFE.sub("_", f"{asset.symbol}__{asset.data_symbol or ''}")
        return self._dir / f"{key}.json"

    def _read(self, asset: Asset, start: date, end: date) -> pd.Series[float] | None:
        path = self._path(asset)
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            covers = date.fromisoformat(entry["start"]) <= start and end <= date.fromisoformat(
                entry["end"]
            )
            fresh = self._clock() - float(entry["fetched_at"]) <= self._ttl
            if not (covers and fresh):
                return None
            series = pd.Series(
                entry["prices"]["values"],
                index=pd.DatetimeIndex(entry["prices"]["dates"]),
                dtype=float,
                name=asset.symbol,
            )
        except (OSError, ValueError, KeyError, TypeError):
            return None  # missing or corrupt entry: fall back to a fresh fetch
        return series.loc[pd.Timestamp(start) : pd.Timestamp(end)]

    def _write(self, asset: Asset, start: date, end: date, series: pd.Series[float]) -> None:
        entry = {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "fetched_at": self._clock(),
            "prices": {
                "dates": [ts.date().isoformat() for ts in series.index],
                "values": [float(v) for v in series.to_numpy()],
            },
        }
        try:
            self._dir.mkdir(parents=True, exist_ok=True)
            path = self._path(asset)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(entry), encoding="utf-8")
            tmp.replace(path)
        except OSError as exc:  # a read-only cache must not break the analysis
            logger.warning("Could not write price cache for {}: {}", asset.symbol, exc)

    def get_prices(self, assets: Sequence[Asset], start: date, end: date) -> pd.DataFrame:
        validate_range(start, end)
        series: list[pd.Series[float]] = []
        for asset in assets:
            cached = self._read(asset, start, end)
            if cached is not None and not cached.empty:
                logger.debug("Cache hit for {}", asset.symbol)
                series.append(cached)
                continue
            fetched = self._inner.get_prices([asset], start, end)[asset.symbol]
            self._write(asset, start, end, fetched)
            series.append(fetched)
        return pd.concat(series, axis=1).sort_index().ffill().dropna()
