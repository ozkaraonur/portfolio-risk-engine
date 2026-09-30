from __future__ import annotations

import math

import pytest

from portfolio_risk.web.i18n import LANGUAGES
from portfolio_risk.web.numfmt import (
    NARROW_NBSP,
    NBSP,
    NUMBER_STYLE,
    format_money,
    format_number,
    format_percent,
)


def test_every_language_has_a_number_style() -> None:
    assert set(NUMBER_STYLE) == set(LANGUAGES)


@pytest.mark.parametrize(
    ("lang", "expected"),
    [
        ("en", "4,493,342.32"),
        ("tr", "4.493.342,32"),
        ("de", "4.493.342,32"),
        ("es", "4.493.342,32"),
        ("fr", f"4{NARROW_NBSP}493{NARROW_NBSP}342,32"),
        ("ru", f"4{NBSP}493{NBSP}342,32"),
        ("ja", "4,493,342.32"),
        ("xx", "4,493,342.32"),  # unknown language: English style
    ],
)
def test_thousands_and_decimal_separators_follow_the_language(lang: str, expected: str) -> None:
    assert format_number(4493342.32, lang) == expected


def test_small_negative_and_rounded_values() -> None:
    assert format_number(-1234.5, "en") == "-1,234.50"
    assert format_number(-1234.5, "tr") == "-1.234,50"
    assert format_number(0.0004, "tr", 3) == "0,000"
    assert format_number(999.999, "en") == "1,000.00"
    assert format_number(1234567, "tr", 0) == "1.234.567"
    assert format_number(12, "tr", 0) == "12"


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_non_finite_values_show_a_dash(bad: float) -> None:
    assert format_number(bad, "en") == "-"
    assert format_money(bad, "USD", "en") == "-"
    assert format_percent(bad, "en") == "-"


def test_money_puts_the_currency_next_to_the_amount() -> None:
    assert format_money(4493342.32, "EUR", "tr") == "4.493.342,32 EUR"
    assert format_money(4493342.32, "USD", "en") == "4,493,342.32 USD"
    assert format_money(1234567.891, "TRY", "en", 0) == "1,234,568 TRY"


def test_percent_placement_follows_the_language() -> None:
    assert format_percent(0.1234, "en") == "12.3%"
    assert format_percent(0.1234, "tr") == "%12,3"
    assert format_percent(0.1234, "de") == f"12,3{NBSP}%"
    assert format_percent(0.1234, "fr", 2) == f"12,34{NBSP}%"
    assert format_percent(-0.05, "es") == "-5,0%"
