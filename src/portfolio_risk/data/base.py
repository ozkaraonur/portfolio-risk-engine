"""Price provider interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from datetime import date

import numpy as np
import pandas as pd

from portfolio_risk.models import Asset


class PriceProvider(ABC):
    """Source of historical prices.

    ``get_prices`` returns a DataFrame with an ascending DatetimeIndex and one column per
    asset symbol. Implementations must raise ``DataUnavailableError`` rather than return
    partial data silently.
    """

    @abstractmethod
    def get_prices(self, assets: Sequence[Asset], start: date, end: date) -> pd.DataFrame: ...

    def get_returns(
        self,
        assets: Sequence[Asset],
        start: date,
        end: date,
        *,
        log: bool = False,
    ) -> pd.DataFrame:
        """Daily simple (default) or log returns derived from ``get_prices``."""
        prices = self.get_prices(assets, start, end)
        if log:
            logs = pd.DataFrame(
                np.log(prices.to_numpy()), index=prices.index, columns=prices.columns
            )
            return logs.diff().dropna(how="all")
        return prices.pct_change().dropna(how="all")


class DataUnavailableError(RuntimeError):
    """Raised when a provider cannot supply requested data."""


def validate_range(start: date, end: date) -> None:
    if start >= end:
        raise ValueError(f"start ({start}) must be before end ({end}).")
