from __future__ import annotations

from datetime import date

import pytest

from portfolio_risk.catalog import (
    BIST,
    CATALOG,
    CATEGORIES,
    COMMODITY,
    CRYPTO,
    US,
    entries_for,
    get_entry,
    synthetic_profiles,
)
from portfolio_risk.data import AssetProfile, SyntheticProvider
from portfolio_risk.models import AssetClass

START, END = date(2005, 1, 1), date(2035, 1, 1)


def test_category_sizes_meet_minimums() -> None:
    assert len(entries_for(US)) >= 30
    assert len(entries_for(BIST)) >= 30
    assert len(entries_for(CRYPTO)) >= 10
    assert len(entries_for(COMMODITY)) >= 3
    assert set(CATEGORIES) == {e.category for e in CATALOG.values()}


def test_symbols_are_unique_and_normalised() -> None:
    all_entries = [e for c in CATEGORIES for e in entries_for(c)]
    assert len(all_entries) == len(CATALOG)
    assert all(e.symbol == e.symbol.upper() and e.symbol for e in all_entries)


@pytest.mark.parametrize(
    ("symbol", "name", "category", "asset_class", "tag"),
    [
        ("AAPL", "Apple", US, AssetClass.EQUITY, "tech"),
        ("SPY", "SPDR S&P 500 ETF", US, AssetClass.EQUITY, "etf"),
        ("QQQ", "Invesco QQQ (Nasdaq-100)", US, AssetClass.EQUITY, "tech"),
        ("THYAO", "Türk Hava Yolları", BIST, AssetClass.EQUITY, "aviation"),
        ("ASELS", "Aselsan", BIST, AssetClass.EQUITY, "defense"),
        ("SOL", "Solana", CRYPTO, AssetClass.CRYPTO, "crypto"),
        ("GOLD", "Altın (Gold)", COMMODITY, AssetClass.COMMODITY, "commodity"),
        ("USO", "United States Oil Fund", COMMODITY, AssetClass.COMMODITY, "energy"),
    ],
)
def test_entries_link_class_and_tags(
    symbol: str, name: str, category: str, asset_class: AssetClass, tag: str
) -> None:
    entry = get_entry(symbol)
    assert entry.name == name
    assert entry.category == category
    assert entry.asset_class is asset_class
    assert tag in entry.tags
    asset = entry.to_asset()
    assert (asset.symbol, asset.asset_class, asset.tags) == (symbol, asset_class, entry.tags)
    assert entry.label == f"{symbol} - {name}"


def test_required_symbols_present() -> None:
    required = {
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "TSLA", "SPY", "QQQ",  # noqa: RUF100
        "THYAO", "ASELS", "GARAN", "EREGL", "KCHOL", "BIMAS", "AKBNK", "SISE", "TUPRS",
        "BTC", "ETH", "SOL", "BNB", "XRP", "AVAX",
        "GOLD", "SILVER", "OIL", "USO",
    }  # fmt: skip
    assert required <= set(CATALOG)


def test_broker_follows_category_and_lookup_errors() -> None:
    assert get_entry("thyao").broker == "BIST"
    assert get_entry("BTC").broker == "Binance"
    with pytest.raises(KeyError, match="katalogda yok"):
        get_entry("NOPE")
    with pytest.raises(KeyError):
        entries_for("Unknown")


def test_profiles_are_valid_for_every_entry() -> None:
    profiles = synthetic_profiles()
    assert set(profiles) == set(CATALOG)
    for profile in profiles.values():
        assert isinstance(profile, AssetProfile)
        assert profile.market_loading**2 + profile.group_loading**2 <= 1.0


def test_profile_validation() -> None:
    with pytest.raises(ValueError, match="loadings"):
        AssetProfile(0.1, 0.2, 100.0, market_loading=0.9, group="g", group_loading=0.9)
    with pytest.raises(ValueError, match="positive"):
        AssetProfile(0.1, 0.0, 100.0, market_loading=0.1, group="g", group_loading=0.1)


def test_synthetic_prices_follow_catalog_profiles() -> None:
    symbols = ["AAPL", "MSFT", "THYAO", "BTC", "ETH", "GOLD", "SILVER", "OIL", "USO"]
    assets = [get_entry(s).to_asset() for s in symbols]
    provider = SyntheticProvider(seed=11, profiles=synthetic_profiles())
    prices = provider.get_prices(assets, START, END)
    rets = provider.get_returns(assets, START, END, log=True)
    assert (prices > 0).all().all()
    # The path is anchored at the end date: the latest price is the catalog price.
    assert prices.iloc[-1].tolist() == pytest.approx([get_entry(s).price for s in symbols])

    vol = rets.std() * (252**0.5)
    for s in symbols:
        assert vol[s] == pytest.approx(get_entry(s).sigma, rel=0.08)

    corr = rets.corr()
    # Correlation = market loadings product (+ group loadings product within a group).
    assert corr.loc["AAPL", "MSFT"] == pytest.approx(0.62**2 + 0.40**2, abs=0.03)
    assert corr.loc["BTC", "ETH"] == pytest.approx(0.30 * 0.25 + 0.80 * 0.85, abs=0.03)
    assert corr.loc["OIL", "USO"] == pytest.approx(0.35**2 + 0.90**2, abs=0.03)
    assert corr.loc["GOLD", "SILVER"] == pytest.approx(0.05 * 0.30 + 0.85**2, abs=0.03)
    assert corr.loc["AAPL", "BTC"] == pytest.approx(0.62 * 0.30, abs=0.03)
    assert corr.loc["GOLD", "BTC"] == pytest.approx(0.0, abs=0.08)  # gold vs crypto


def test_profiles_are_deterministic_and_independent_of_other_assets() -> None:
    provider = SyntheticProvider(seed=3, profiles=synthetic_profiles())
    aapl, thy = get_entry("AAPL").to_asset(), get_entry("THYAO").to_asset()
    alone = provider.get_prices([aapl], date(2023, 1, 1), date(2024, 1, 1))["AAPL"]
    together = provider.get_prices([thy, aapl], date(2023, 1, 1), date(2024, 1, 1))["AAPL"]
    assert alone.equals(together)


def test_unknown_symbols_fall_back_to_class_defaults() -> None:
    from portfolio_risk.models import Asset

    custom = Asset(symbol="ZZZ", asset_class=AssetClass.EQUITY)
    prices = SyntheticProvider(profiles=synthetic_profiles()).get_prices(
        [custom], date(2024, 1, 1), date(2024, 6, 1)
    )
    assert prices["ZZZ"].iloc[-1] == pytest.approx(100.0)  # class default price, at the end date
