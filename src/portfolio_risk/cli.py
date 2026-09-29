"""Command-line interface."""

from __future__ import annotations

from datetime import date, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from pydantic import ValidationError

from portfolio_risk import __version__
from portfolio_risk.data import (
    DataUnavailableError,
    PriceProvider,
    StooqProvider,
    SyntheticProvider,
)
from portfolio_risk.models import Portfolio

app = typer.Typer(
    help="Multi-broker portfolio risk and stress-testing engine.", no_args_is_help=True
)


class ProviderName(StrEnum):
    SYNTHETIC = "synthetic"
    STOOQ = "stooq"


def load_portfolio(path: Path) -> Portfolio:
    return Portfolio.model_validate_json(path.read_text(encoding="utf-8"))


def make_provider(name: ProviderName, seed: int) -> PriceProvider:
    if name is ProviderName.STOOQ:
        return StooqProvider()
    return SyntheticProvider(seed=seed)


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command()
def summary(
    portfolio_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    provider: Annotated[ProviderName, typer.Option(help="Price source.")] = ProviderName.SYNTHETIC,
    seed: Annotated[int, typer.Option(help="Seed for the synthetic provider.")] = 42,
    days: Annotated[int, typer.Option(min=5, help="History length in calendar days.")] = 365,
) -> None:
    """Value a portfolio at the latest price and show weights."""
    try:
        portfolio = load_portfolio(portfolio_file)
        end = date.today()
        prices = make_provider(provider, seed).get_prices(
            portfolio.assets, end - timedelta(days=days), end
        )
    except (ValidationError, DataUnavailableError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    latest = {str(k): float(v) for k, v in prices.iloc[-1].items()}
    values = portfolio.market_values(latest)
    weights = portfolio.weights(latest)
    typer.echo(f"{portfolio.name} ({portfolio.base_currency}) as of {prices.index[-1].date()}")
    typer.echo(f"{'SYMBOL':<10}{'VALUE':>14}{'WEIGHT':>10}")
    for symbol, value in values.items():
        typer.echo(f"{symbol:<10}{value:>14,.2f}{weights[symbol]:>10.1%}")
    if portfolio.total_cash > 0:
        typer.echo(f"{'CASH':<10}{portfolio.total_cash:>14,.2f}{weights['CASH']:>10.1%}")
    typer.echo(f"{'TOTAL':<10}{portfolio.total_value(latest):>14,.2f}{1.0:>10.1%}")


if __name__ == "__main__":
    app()
