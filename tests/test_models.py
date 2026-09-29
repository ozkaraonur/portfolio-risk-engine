import pytest
from pydantic import ValidationError

from portfolio_risk.models import Asset, AssetClass, CashBalance, Portfolio, Position

AAPL = Asset(symbol=" aapl ", asset_class=AssetClass.EQUITY)
BTC = Asset(symbol="BTC", asset_class=AssetClass.CRYPTO)
PRICES = {"AAPL": 100.0, "BTC": 20_000.0}


def test_symbol_normalised() -> None:
    assert AAPL.symbol == "AAPL"


def test_position_rejects_non_positive_quantity() -> None:
    with pytest.raises(ValidationError):
        Position(asset=AAPL, quantity=0)


def test_models_are_immutable() -> None:
    with pytest.raises(ValidationError):
        AAPL.symbol = "MSFT"  # type: ignore[misc]


def test_aggregation_across_brokers() -> None:
    pf = Portfolio(
        positions=(
            Position(asset=AAPL, quantity=10, broker="a"),
            Position(asset=AAPL, quantity=5, broker="b"),
            Position(asset=BTC, quantity=0.1, broker="c"),
        ),
        cash=(CashBalance(broker="a", amount=500),),
    )
    assert pf.symbols == ["AAPL", "BTC"]
    assert pf.quantities()["AAPL"] == 15
    assert pf.brokers == ["a", "b", "c"]
    assert pf.total_value(PRICES) == pytest.approx(1500 + 2000 + 500)
    weights = pf.weights(PRICES)
    assert sum(weights.values()) == pytest.approx(1.0)
    assert weights["CASH"] == pytest.approx(500 / 4000)


def test_missing_price_raises() -> None:
    pf = Portfolio(positions=(Position(asset=BTC, quantity=1),))
    with pytest.raises(KeyError):
        pf.total_value({"AAPL": 1.0})


def test_non_base_currency_rejected() -> None:
    with pytest.raises(ValidationError):
        Portfolio(cash=(CashBalance(amount=1, currency="EUR"),))


def test_from_weights_roundtrip() -> None:
    pf = Portfolio.from_weights(10_000, {AAPL: 0.6, BTC: 0.3}, PRICES, cash_weight=0.1)
    assert pf.total_value(PRICES) == pytest.approx(10_000)
    w = pf.weights(PRICES)
    assert w["AAPL"] == pytest.approx(0.6)
    assert w["CASH"] == pytest.approx(0.1)


def test_from_weights_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match="sum to 1"):
        Portfolio.from_weights(1000, {AAPL: 0.5}, PRICES)


def test_json_roundtrip() -> None:
    pf = Portfolio(positions=(Position(asset=AAPL, quantity=3),))
    assert Portfolio.model_validate_json(pf.model_dump_json()) == pf
