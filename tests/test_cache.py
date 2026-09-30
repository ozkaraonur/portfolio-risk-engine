from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from portfolio_risk.data import CachedProvider, DataUnavailableError, PriceProvider
from portfolio_risk.models import Asset, AssetClass

AAPL = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY)
BTC = Asset(symbol="BTC", asset_class=AssetClass.CRYPTO)
START, END = date(2024, 1, 1), date(2024, 3, 1)


class CountingProvider(PriceProvider):
    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def get_prices(self, assets: Sequence[Asset], start: date, end: date) -> pd.DataFrame:
        self.calls.append([a.symbol for a in assets])
        index = pd.bdate_range(start, end)
        base = {"AAPL": 100.0, "BTC": 40_000.0}
        return pd.DataFrame(
            {a.symbol: [base[a.symbol] + i for i in range(len(index))] for a in assets}, index=index
        )


class Clock:
    def __init__(self) -> None:
        self.now = 1_000_000.0

    def __call__(self) -> float:
        return self.now


def make(tmp_path: Path, inner: PriceProvider, clock: Clock, ttl: float = 1.0) -> CachedProvider:
    return CachedProvider(inner, tmp_path, namespace="test", ttl_hours=ttl, clock=clock)


def test_second_request_is_served_from_disk(tmp_path: Path) -> None:
    inner, clock = CountingProvider(), Clock()
    first = make(tmp_path, inner, clock).get_prices([AAPL, BTC], START, END)
    # A new instance proves the data really is on disk, not in memory.
    second = make(tmp_path, inner, clock).get_prices([AAPL, BTC], START, END)
    assert inner.calls == [["AAPL"], ["BTC"]]
    pd.testing.assert_frame_equal(first, second, check_freq=False)


def test_a_sub_range_is_sliced_from_the_cache(tmp_path: Path) -> None:
    inner, clock = CountingProvider(), Clock()
    cache = make(tmp_path, inner, clock)
    cache.get_prices([AAPL], START, END)
    part = cache.get_prices([AAPL], date(2024, 2, 1), date(2024, 2, 15))
    assert len(inner.calls) == 1
    assert part.index.min() >= pd.Timestamp("2024-02-01")
    assert part.index.max() <= pd.Timestamp("2024-02-15")


def test_stale_entries_and_wider_ranges_are_refetched(tmp_path: Path) -> None:
    inner, clock = CountingProvider(), Clock()
    cache = make(tmp_path, inner, clock, ttl=1.0)
    cache.get_prices([AAPL], START, END)
    cache.get_prices([AAPL], START, date(2024, 4, 1))  # wider than what was cached
    assert len(inner.calls) == 2
    clock.now += 2 * 3600  # older than the ttl
    cache.get_prices([AAPL], START, date(2024, 4, 1))
    assert len(inner.calls) == 3


def test_only_missing_assets_are_downloaded(tmp_path: Path) -> None:
    inner, clock = CountingProvider(), Clock()
    cache = make(tmp_path, inner, clock)
    cache.get_prices([AAPL], START, END)
    cache.get_prices([AAPL, BTC], START, END)
    assert inner.calls == [["AAPL"], ["BTC"]]


def test_namespaces_and_data_symbols_do_not_collide(tmp_path: Path) -> None:
    inner, clock = CountingProvider(), Clock()
    make(tmp_path, inner, clock).get_prices([AAPL], START, END)
    other = Asset(symbol="AAPL", asset_class=AssetClass.EQUITY, data_symbol="aapl.us")
    make(tmp_path, inner, clock).get_prices([other], START, END)
    assert len(inner.calls) == 2


def test_corrupt_entry_falls_back_to_a_fresh_fetch(tmp_path: Path) -> None:
    inner, clock = CountingProvider(), Clock()
    cache = make(tmp_path, inner, clock)
    cache.get_prices([AAPL], START, END)
    for file in (tmp_path / "test").glob("*.json"):
        file.write_text("{not json", encoding="utf-8")
    prices = cache.get_prices([AAPL], START, END)
    assert len(inner.calls) == 2
    assert not prices.empty


def test_an_unwritable_cache_does_not_break_the_request(tmp_path: Path) -> None:
    blocker = tmp_path / "file"
    blocker.write_text("x", encoding="utf-8")  # a file where the directory should be
    prices = CachedProvider(CountingProvider(), blocker / "sub").get_prices([AAPL], START, END)
    assert not prices.empty


def test_errors_from_the_inner_provider_propagate_and_nothing_is_cached(tmp_path: Path) -> None:
    class Failing(PriceProvider):
        def get_prices(self, assets: Sequence[Asset], start: date, end: date) -> pd.DataFrame:
            raise DataUnavailableError("down")

    cache = CachedProvider(Failing(), tmp_path)
    with pytest.raises(DataUnavailableError):
        cache.get_prices([AAPL], START, END)
    assert not list(tmp_path.rglob("*.json"))


def test_negative_ttl_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ttl"):
        CachedProvider(CountingProvider(), tmp_path, ttl_hours=-1)
