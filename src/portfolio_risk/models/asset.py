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
    tags: tuple[str, ...] = Field(
        default=(),
        description="Free-form labels (e.g. 'tech', 'growth') that stress scenarios can target.",
    )

    @field_validator("symbol", "currency")
    @classmethod
    def _upper(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("tags")
    @classmethod
    def _lower_tags(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        return tuple(dict.fromkeys(t.strip().lower() for t in value if t.strip()))
