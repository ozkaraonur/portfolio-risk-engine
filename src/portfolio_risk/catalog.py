"""Pre-defined asset universe used by the web panel and the synthetic price engine.

Every entry carries its asset class, default tags and illustrative GBM parameters
(annualised drift / volatility, starting price in USD terms) plus a factor-loading structure
that gives the synthetic prices realistic cross-asset correlations. The numbers are plausible
but *not* market estimates; BIST prices are approximate USD equivalents.
"""

from __future__ import annotations

from dataclasses import dataclass

from portfolio_risk.data.fx import USD_PER_UNIT
from portfolio_risk.data.synthetic import AssetProfile
from portfolio_risk.models import Asset, AssetClass

US = "ABD Hisseleri"
BIST = "BIST"
CRYPTO = "Kripto"
COMMODITY = "Emtia"
CATEGORIES = [US, BIST, CRYPTO, COMMODITY]

CATEGORY_BROKER = {
    US: "InteractiveBrokers",
    BIST: "BIST",
    CRYPTO: "Binance",
    COMMODITY: "InteractiveBrokers",
}
CATEGORY_CURRENCY = {BIST: "TRY"}  # every other category is quoted in USD
CATEGORY_CLASS = {
    US: AssetClass.EQUITY,
    BIST: AssetClass.EQUITY,
    CRYPTO: AssetClass.CRYPTO,
    COMMODITY: AssetClass.COMMODITY,
}
# category -> (market loading, group loading, group name)
CATEGORY_FACTORS = {
    US: (0.62, 0.40, "us"),
    BIST: (0.35, 0.55, "bist"),
    CRYPTO: (0.25, 0.85, "crypto"),
    COMMODITY: (0.20, 0.30, "commodity"),
}


@dataclass(frozen=True)
class CatalogEntry:
    symbol: str
    name: str
    category: str
    asset_class: AssetClass
    tags: tuple[str, ...]
    sigma: float
    mu: float
    price: float
    market_loading: float
    group: str
    group_loading: float
    data_symbol: str | None = None

    @property
    def label(self) -> str:
        return f"{self.symbol} - {self.name}"

    @property
    def currency(self) -> str:
        return CATEGORY_CURRENCY.get(self.category, "USD")

    @property
    def broker(self) -> str:
        return CATEGORY_BROKER[self.category]

    def to_asset(self) -> Asset:
        return Asset(
            symbol=self.symbol,
            asset_class=self.asset_class,
            name=self.name,
            currency=self.currency,
            tags=self.tags,
            data_symbol=self.data_symbol,
        )

    def profile(self) -> AssetProfile:
        return AssetProfile(
            mu=self.mu,
            sigma=self.sigma,
            start_price=self.price,
            market_loading=self.market_loading,
            group=self.group,
            group_loading=self.group_loading,
        )


# (symbol, name, tags, sigma, mu, price) with optional (market, group loading, group) override.
_Row = tuple[str, str, str, float, float, float]
_Override = tuple[float, float, str]

_US_ROWS: list[_Row] = [
    ("AAPL", "Apple", "tech,growth", 0.28, 0.10, 190.0),
    ("MSFT", "Microsoft", "tech,growth", 0.26, 0.11, 420.0),
    ("NVDA", "Nvidia", "tech,growth,semiconductor", 0.50, 0.20, 120.0),
    ("AMZN", "Amazon", "tech,growth,consumer", 0.32, 0.12, 185.0),
    ("GOOGL", "Alphabet", "tech,growth", 0.29, 0.11, 170.0),
    ("META", "Meta Platforms", "tech,growth", 0.38, 0.13, 500.0),
    ("TSLA", "Tesla", "auto,growth,tech", 0.58, 0.12, 250.0),
    ("AVGO", "Broadcom", "tech,semiconductor", 0.36, 0.15, 160.0),
    ("AMD", "Advanced Micro Devices", "tech,semiconductor,growth", 0.48, 0.14, 150.0),
    ("NFLX", "Netflix", "tech,growth,media", 0.38, 0.12, 650.0),
    ("ORCL", "Oracle", "tech", 0.28, 0.09, 130.0),
    ("ADBE", "Adobe", "tech,growth", 0.33, 0.09, 520.0),
    ("CRM", "Salesforce", "tech,growth", 0.33, 0.09, 270.0),
    ("INTC", "Intel", "tech,semiconductor", 0.40, 0.03, 30.0),
    ("CSCO", "Cisco", "tech", 0.22, 0.06, 50.0),
    ("QCOM", "Qualcomm", "tech,semiconductor", 0.34, 0.10, 170.0),
    ("JPM", "JPMorgan Chase", "financial,bank", 0.24, 0.09, 200.0),
    ("BAC", "Bank of America", "financial,bank", 0.28, 0.08, 38.0),
    ("V", "Visa", "financial,payments", 0.20, 0.10, 270.0),
    ("MA", "Mastercard", "financial,payments", 0.22, 0.10, 460.0),
    ("BRKB", "Berkshire Hathaway B", "financial,value", 0.18, 0.09, 440.0),
    ("WMT", "Walmart", "consumer,defensive", 0.18, 0.08, 68.0),
    ("COST", "Costco", "consumer,defensive", 0.20, 0.11, 840.0),
    ("KO", "Coca-Cola", "consumer,defensive", 0.15, 0.06, 62.0),
    ("PEP", "PepsiCo", "consumer,defensive", 0.16, 0.06, 170.0),
    ("MCD", "McDonald's", "consumer,defensive", 0.17, 0.07, 270.0),
    ("NKE", "Nike", "consumer", 0.30, 0.07, 90.0),
    ("DIS", "Walt Disney", "media,consumer", 0.30, 0.07, 105.0),
    ("JNJ", "Johnson & Johnson", "healthcare,defensive", 0.15, 0.06, 155.0),
    ("UNH", "UnitedHealth", "healthcare", 0.24, 0.10, 500.0),
    ("PFE", "Pfizer", "healthcare,defensive", 0.24, 0.04, 28.0),
    ("XOM", "Exxon Mobil", "energy", 0.26, 0.07, 115.0),
    ("CVX", "Chevron", "energy", 0.24, 0.07, 155.0),
    ("BA", "Boeing", "industrial,aviation", 0.36, 0.06, 180.0),
    ("SPY", "SPDR S&P 500 ETF", "etf,index", 0.16, 0.09, 550.0),
    ("QQQ", "Invesco QQQ (Nasdaq-100)", "etf,index,tech", 0.21, 0.11, 470.0),
    ("VOO", "Vanguard S&P 500 ETF", "etf,index", 0.16, 0.09, 505.0),
    ("IWM", "iShares Russell 2000 ETF", "etf,index", 0.22, 0.07, 210.0),
]
_US_OVERRIDES: dict[str, _Override] = {
    "SPY": (0.90, 0.30, "us"),
    "VOO": (0.90, 0.30, "us"),
    "QQQ": (0.85, 0.35, "us"),
    "IWM": (0.80, 0.30, "us"),
}

_BIST_ROWS: list[_Row] = [
    ("THYAO", "Türk Hava Yolları", "aviation,industrial", 0.42, 0.14, 9.5),
    ("ASELS", "Aselsan", "defense,industrial", 0.40, 0.16, 3.2),
    ("GARAN", "Garanti BBVA", "financial,bank", 0.38, 0.12, 3.8),
    ("AKBNK", "Akbank", "financial,bank", 0.38, 0.12, 1.5),
    ("YKBNK", "Yapı Kredi", "financial,bank", 0.40, 0.12, 0.9),
    ("ISCTR", "İş Bankası (C)", "financial,bank", 0.38, 0.12, 0.35),
    ("HALKB", "Halkbank", "financial,bank", 0.45, 0.10, 0.6),
    ("VAKBN", "Vakıfbank", "financial,bank", 0.42, 0.10, 0.55),
    ("EREGL", "Ereğli Demir Çelik", "steel,industrial", 0.36, 0.09, 1.1),
    ("KRDMD", "Kardemir (D)", "steel,industrial", 0.40, 0.09, 0.7),
    ("KCHOL", "Koç Holding", "holding", 0.32, 0.12, 4.9),
    ("SAHOL", "Sabancı Holding", "holding", 0.34, 0.12, 2.3),
    ("DOHOL", "Doğan Holding", "holding", 0.36, 0.10, 0.4),
    ("BIMAS", "BİM Mağazalar", "retail,consumer,defensive", 0.30, 0.14, 15.0),
    ("MGROS", "Migros", "retail,consumer,defensive", 0.36, 0.14, 7.5),
    ("SISE", "Şişecam", "industrial,glass", 0.36, 0.09, 1.1),
    ("TUPRS", "Tüpraş", "energy,refinery", 0.34, 0.13, 4.3),
    ("PETKM", "Petkim", "energy,chemicals", 0.42, 0.07, 0.6),
    ("ODAS", "Odaş Elektrik", "energy,utilities", 0.50, 0.08, 0.15),
    ("ASTOR", "Astor Enerji", "energy,industrial", 0.48, 0.15, 3.5),
    ("TOASO", "Tofaş Oto", "auto,industrial", 0.38, 0.11, 5.0),
    ("FROTO", "Ford Otosan", "auto,industrial", 0.32, 0.14, 30.0),
    ("ARCLK", "Arçelik", "industrial,consumer", 0.38, 0.08, 3.6),
    ("TCELL", "Turkcell", "telecom", 0.32, 0.11, 2.8),
    ("TTKOM", "Türk Telekom", "telecom", 0.34, 0.10, 1.4),
    ("ENKAI", "Enka İnşaat", "construction,industrial", 0.30, 0.12, 1.6),
    ("PGSUS", "Pegasus", "aviation,industrial", 0.44, 0.15, 8.0),
    ("TAVHL", "TAV Havalimanları", "aviation,industrial", 0.38, 0.13, 6.0),
    ("SASA", "Sasa Polyester", "chemicals,industrial", 0.55, 0.06, 0.1),
    ("HEKTS", "Hektaş", "chemicals,agriculture", 0.50, 0.06, 0.2),
    ("KOZAL", "Koza Altın", "mining,gold", 0.42, 0.10, 0.7),
    ("EKGYO", "Emlak Konut GYO", "realestate", 0.40, 0.09, 0.3),
    ("ULKER", "Ülker Bisküvi", "food,consumer,defensive", 0.36, 0.10, 3.0),
    ("CCOLA", "Coca-Cola İçecek", "beverage,consumer,defensive", 0.32, 0.13, 1.6),
    ("AEFES", "Anadolu Efes", "beverage,consumer,defensive", 0.34, 0.11, 0.3),
]

_CRYPTO_ROWS: list[_Row] = [
    ("BTC", "Bitcoin", "crypto,store-of-value", 0.60, 0.20, 65_000.0),
    ("ETH", "Ethereum", "crypto,layer1", 0.72, 0.18, 3_200.0),
    ("SOL", "Solana", "crypto,layer1", 0.95, 0.20, 150.0),
    ("BNB", "BNB", "crypto,exchange", 0.65, 0.16, 580.0),
    ("XRP", "XRP", "crypto,payments", 0.85, 0.10, 0.55),
    ("ADA", "Cardano", "crypto,layer1", 0.88, 0.08, 0.45),
    ("AVAX", "Avalanche", "crypto,layer1", 0.95, 0.12, 35.0),
    ("DOGE", "Dogecoin", "crypto,meme", 1.05, 0.05, 0.15),
    ("DOT", "Polkadot", "crypto,layer1", 0.90, 0.08, 6.5),
    ("LINK", "Chainlink", "crypto,defi", 0.90, 0.12, 14.0),
    ("LTC", "Litecoin", "crypto,payments", 0.75, 0.06, 80.0),
    ("TRX", "TRON", "crypto,layer1", 0.70, 0.09, 0.12),
    ("TON", "Toncoin", "crypto,layer1", 0.90, 0.12, 6.0),
]
_CRYPTO_OVERRIDES: dict[str, _Override] = {"BTC": (0.30, 0.80, "crypto")}

_COMMODITY_ROWS: list[_Row] = [
    ("GOLD", "Altın (Gold)", "commodity,precious-metal,safe-haven", 0.15, 0.06, 2_400.0),
    ("SILVER", "Gümüş (Silver)", "commodity,precious-metal", 0.28, 0.06, 28.0),
    ("PLATINUM", "Platin (Platinum)", "commodity,precious-metal", 0.24, 0.04, 980.0),
    ("OIL", "Ham Petrol (WTI)", "commodity,energy", 0.35, 0.04, 78.0),
    ("USO", "United States Oil Fund", "commodity,energy,etf", 0.36, 0.03, 75.0),
    ("NATGAS", "Doğalgaz (Natural Gas)", "commodity,energy", 0.55, 0.02, 2.5),
    ("COPPER", "Bakır (Copper)", "commodity,industrial-metal", 0.24, 0.05, 4.4),
]
_COMMODITY_OVERRIDES: dict[str, _Override] = {
    "GOLD": (0.05, 0.85, "precious"),
    "SILVER": (0.30, 0.85, "precious"),
    "PLATINUM": (0.40, 0.50, "precious"),
    "OIL": (0.35, 0.90, "oil"),
    "USO": (0.35, 0.90, "oil"),
    "NATGAS": (0.15, 0.30, "gas"),
    "COPPER": (0.50, 0.30, "metals"),
}
_DATA_SYMBOLS = {
    "GOLD": "gc.f",
    "SILVER": "si.f",
    "PLATINUM": "pl.f",
    "OIL": "cl.f",
    "NATGAS": "ng.f",
    "COPPER": "hg.f",
}


def _build(
    category: str, rows: list[_Row], overrides: dict[str, _Override] | None = None
) -> list[CatalogEntry]:
    ml, gl, group = CATEGORY_FACTORS[category]
    entries = []
    for symbol, name, tags, sigma, mu, price in rows:
        if category == BIST:  # the table lists USD-equivalent prices; BIST quotes are in lira
            price = round(price / USD_PER_UNIT["TRY"], 2)
        m, g, grp = (overrides or {}).get(symbol, (ml, gl, group))
        entries.append(
            CatalogEntry(
                symbol=symbol,
                name=name,
                category=category,
                asset_class=CATEGORY_CLASS[category],
                tags=tuple(tags.split(",")),
                sigma=sigma,
                mu=mu,
                price=price,
                market_loading=m,
                group=grp,
                group_loading=g,
                data_symbol=_DATA_SYMBOLS.get(symbol),
            )
        )
    return entries


CATALOG: dict[str, CatalogEntry] = {
    e.symbol: e
    for e in (
        *_build(US, _US_ROWS, _US_OVERRIDES),
        *_build(BIST, _BIST_ROWS),
        *_build(CRYPTO, _CRYPTO_ROWS, _CRYPTO_OVERRIDES),
        *_build(COMMODITY, _COMMODITY_ROWS, _COMMODITY_OVERRIDES),
    )
}


def entries_for(category: str) -> list[CatalogEntry]:
    if category not in CATEGORIES:
        raise KeyError(f"Unknown category: {category}")
    return [e for e in CATALOG.values() if e.category == category]


def get_entry(symbol: str) -> CatalogEntry:
    try:
        return CATALOG[symbol.strip().upper()]
    except KeyError:
        raise KeyError(f"'{symbol}' katalogda yok.") from None


def synthetic_profiles() -> dict[str, AssetProfile]:
    """GBM profiles for every catalog symbol, for ``SyntheticProvider(profiles=...)``."""
    return {symbol: entry.profile() for symbol, entry in CATALOG.items()}
