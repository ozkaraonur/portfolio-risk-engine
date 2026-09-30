"""Locale-aware number formatting for the web panel.

Each language uses its own thousands and decimal separators (Turkish ``4.493.342,32``, English
``4,493,342.32``, French ``4 493 342,32``), so the same amount reads correctly for the person
looking at it. Change ``NUMBER_STYLE`` to adjust a language.
"""

from __future__ import annotations

import math

NARROW_NBSP = " "
NBSP = " "

# language -> (thousands separator, decimal separator)
NUMBER_STYLE: dict[str, tuple[str, str]] = {
    "en": (",", "."),
    "zh": (",", "."),
    "ja": (",", "."),
    "hi": (",", "."),
    "bn": (",", "."),
    "ar": (",", "."),
    "ur": (",", "."),
    "tr": (".", ","),
    "de": (".", ","),
    "es": (".", ","),
    "pt": (".", ","),
    "id": (".", ","),
    "fr": (NARROW_NBSP, ","),
    "ru": (NBSP, ","),
}
DEFAULT_STYLE = (",", ".")

# languages that put "%" in front of the number (Turkish) or after a space (German, French, ...)
PERCENT_PREFIX = frozenset({"tr"})
PERCENT_SPACED = frozenset({"de", "fr", "ru"})

PLACEHOLDER = "-"


def format_number(value: float, lang: str, decimals: int = 2) -> str:
    """``1234567.891`` -> ``1,234,567.89`` (English) or ``1.234.567,89`` (Turkish)."""
    if not math.isfinite(value):
        return PLACEHOLDER
    thousands, decimal = NUMBER_STYLE.get(lang, DEFAULT_STYLE)
    text = f"{value:,.{decimals}f}"
    return text.replace(",", "\0").replace(".", decimal).replace("\0", thousands)


def format_money(value: float, currency: str, lang: str, decimals: int = 2) -> str:
    """Amount followed by its currency code, e.g. ``4.493.342,32 EUR``."""
    if not math.isfinite(value):
        return PLACEHOLDER
    return f"{format_number(value, lang, decimals)} {currency}"


def format_percent(fraction: float, lang: str, decimals: int = 1) -> str:
    """``0.1234`` -> ``12.3%`` (``%12,3`` in Turkish, ``12,3 %`` in German / French / Russian)."""
    if not math.isfinite(fraction):
        return PLACEHOLDER
    number = format_number(fraction * 100.0, lang, decimals)
    if lang in PERCENT_PREFIX:
        return f"%{number}"
    return f"{number}{NBSP}%" if lang in PERCENT_SPACED else f"{number}%"
