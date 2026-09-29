"""Asset definitions."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class AssetClass(StrEnum):
    EQUITY = "equity"
    CRYPTO = "crypto"
    COMMODITY = "commodity"


class Asset(BaseModel):
    """A tradable instrument, identified by its unique ``symbol``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1)
    asset_class: AssetClass
    name: str | None = None
    currency: str = Field(default="USD", min_length=3, max_length=3)
    data_symbol: str | None = Field(
        default=None,
        description="Ticker used by external price sources, if different from ``symbol``.",
    )

    @field_validator("symbol", "currency")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.strip().upper()
