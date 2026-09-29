"""Offline, deterministic price generation using Geometric Brownian Motion."""

from __future__ import annotations

import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from portfolio_risk.data.base import PriceProvider, validate_range
from portfolio_risk.models import Asset, AssetClass

TRADING_DAYS = 252


@dataclass(frozen=True)
class GBMParams:
    """Annualised drift and volatility, plus the default starting price."""

    mu: float
    sigma: float
    start_price: float


@dataclass(frozen=True)
class AssetProfile:
    """Per-symbol GBM parameters and a two-factor correlation structure.

    The standardised shock is ``m * M + g * G_group + sqrt(1 - m^2 - g^2) * e`` where ``M`` is
    the shared market factor, ``G_group`` a factor shared by assets in the same ``group``, and
    ``e`` idiosyncratic noise. Two assets' correlation is therefore ``m1*m2`` plus ``g1*g2`` if
    they share a group.
    """

    mu: float
    sigma: float
    start_price: float
    market_loading: float
    group: str
    group_loading: float

    def __post_init__(self) -> None:
        if self.sigma <= 0 or self.start_price <= 0:
            raise ValueError("sigma and start_price must be positive.")
        if self.market_loading**2 + self.group_loading**2 > 1.0 + 1e-12:
            raise ValueError("Squared factor loadings must not exceed 1.")

    @property
    def params(self) -> GBMParams:
        return GBMParams(mu=self.mu, sigma=self.sigma, start_price=self.start_price)


DEFAULT_PARAMS: dict[AssetClass, GBMParams] = {
    AssetClass.EQUITY: GBMParams(mu=0.08, sigma=0.22, start_price=100.0),
    AssetClass.CRYPTO: GBMParams(mu=0.15, sigma=0.75, start_price=1000.0),
    AssetClass.COMMODITY: GBMParams(mu=0.03, sigma=0.20, start_price=50.0),
}


def simulate_gbm(
    params: GBMParams,
    shocks: np.ndarray,  # standard-normal draws, shape (n_steps,)
    dt: float = 1.0 / TRADING_DAYS,
) -> np.ndarray:
    """Price path (length ``n_steps + 1``) using the exact GBM solution.

    S_{t+dt} = S_t * exp((mu - sigma^2/2) dt + sigma sqrt(dt) Z)
    """
    log_returns = (params.mu - 0.5 * params.sigma**2) * dt + params.sigma * np.sqrt(dt) * shocks
    return np.asarray(params.start_price * np.exp(np.concatenate([[0.0], np.cumsum(log_returns)])))


class SyntheticProvider(PriceProvider):
    """Generates reproducible GBM price histories on a business-day calendar.

    Assets share a common market factor so that returns are positively correlated
    (pairwise correlation ``correlation``). Each asset's idiosyncratic stream depends only
    on ``(seed, symbol)``, so an asset's path does not change when other assets are added
    (given the same date range).
    """

    def __init__(
        self,
        seed: int = 42,
        correlation: float = 0.3,
        params: dict[AssetClass, GBMParams] | None = None,
        profiles: Mapping[str, AssetProfile] | None = None,
    ) -> None:
        if not 0.0 <= correlation < 1.0:
            raise ValueError("correlation must be in [0, 1).")
        self._seed = seed
        self._rho = correlation
        self._params = {**DEFAULT_PARAMS, **(params or {})}
        self._profiles = dict(profiles or {})

    def get_prices(self, assets: Sequence[Asset], start: date, end: date) -> pd.DataFrame:
        validate_range(start, end)
        index = pd.bdate_range(start, end)
        n_steps = len(index) - 1
        if n_steps < 1:
            raise ValueError("Date range must span at least two business days.")

        market = np.random.default_rng([self._seed, 0]).standard_normal(n_steps)
        columns: dict[str, np.ndarray] = {}
        group_shocks: dict[str, np.ndarray] = {}
        for asset in assets:
            key = zlib.crc32(asset.symbol.encode())
            idio = np.random.default_rng([self._seed, 1, key]).standard_normal(n_steps)
            profile = self._profiles.get(asset.symbol)
            if profile is None:
                shocks = np.sqrt(self._rho) * market + np.sqrt(1.0 - self._rho) * idio
                columns[asset.symbol] = simulate_gbm(self._params[asset.asset_class], shocks)
                continue
            if profile.group not in group_shocks:
                group_key = zlib.crc32(profile.group.encode())
                group_shocks[profile.group] = np.random.default_rng(
                    [self._seed, 2, group_key]
                ).standard_normal(n_steps)
            idio_weight = np.sqrt(
                max(1.0 - profile.market_loading**2 - profile.group_loading**2, 0.0)
            )
            shocks = (
                profile.market_loading * market
                + profile.group_loading * group_shocks[profile.group]
                + idio_weight * idio
            )
            columns[asset.symbol] = simulate_gbm(profile.params, shocks)
        return pd.DataFrame(columns, index=index)
