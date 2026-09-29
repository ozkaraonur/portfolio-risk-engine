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
from portfolio_risk.risk import Method, analyze_risk, correlation_matrix, run_monte_carlo

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


@app.command()
def risk(
    portfolio_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    confidence: Annotated[
        float, typer.Option(min=0.9, max=0.999, help="e.g. 0.95 or 0.99.")
    ] = 0.95,
    horizon: Annotated[int, typer.Option(min=1, max=250, help="Horizon in trading days.")] = 1,
    provider: Annotated[ProviderName, typer.Option(help="Price source.")] = ProviderName.SYNTHETIC,
    seed: Annotated[int, typer.Option(help="Seed for the synthetic provider.")] = 42,
    days: Annotated[int, typer.Option(min=60, help="History length in calendar days.")] = 730,
) -> None:
    """VaR, CVaR (parametric and historical), diversification and correlations."""
    try:
        portfolio = load_portfolio(portfolio_file)
        end = date.today()
        prices = make_provider(provider, seed).get_prices(
            portfolio.assets, end - timedelta(days=days), end
        )
        reports = [
            analyze_risk(portfolio, prices, method=m, confidence=confidence, horizon=horizon)
            for m in Method
        ]
    except (ValidationError, DataUnavailableError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    parametric, historical = reports
    typer.echo(
        f"{portfolio.name}: {confidence:.1%} confidence, {horizon}-day horizon, "
        f"value {parametric.portfolio_value:,.2f} {portfolio.base_currency}"
    )
    typer.echo(
        f"{'METHOD':<12}{'VaR':>12}{'VaR %':>8}{'CVaR':>12}{'DIV. BENEFIT':>14}{'REDUCTION':>11}"
    )
    for r in reports:
        typer.echo(
            f"{r.method.value:<12}{r.var:>12,.2f}{r.var / r.portfolio_value:>8.2%}"
            f"{r.cvar:>12,.2f}{r.diversification_benefit:>14,.2f}{r.diversification_ratio:>11.1%}"
        )
    typer.echo("\nStandalone VaR (parametric / historical):")
    for symbol, value in parametric.standalone_var.items():
        typer.echo(f"{symbol:<12}{value:>12,.2f}{historical.standalone_var[symbol]:>12,.2f}")
    corr = correlation_matrix(prices[portfolio.symbols].pct_change().dropna())
    typer.echo("\nCorrelation:")
    typer.echo(corr.round(2).to_string())


@app.command()
def simulate(
    portfolio_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    simulations: Annotated[
        int, typer.Option("--simulations", "-n", min=10, help="Number of Monte Carlo paths.")
    ] = 1000,
    days: Annotated[
        int, typer.Option("--days", "-t", min=1, help="Simulation horizon in trading days.")
    ] = 252,
    seed: Annotated[int, typer.Option(help="Seed for simulation and synthetic prices.")] = 42,
    loss_threshold: Annotated[
        float, typer.Option(min=0.01, max=1.0, help="Loss fraction defining ruin, e.g. 0.3.")
    ] = 0.3,
    drift: Annotated[bool, typer.Option(help="Use historical mean returns as drift.")] = False,
    provider: Annotated[ProviderName, typer.Option(help="Price source.")] = ProviderName.SYNTHETIC,
    history_days: Annotated[
        int, typer.Option(min=60, help="Calendar days of history used to estimate risk.")
    ] = 730,
) -> None:
    """Monte Carlo value paths: VaR/CVaR, terminal percentiles, drawdowns, ruin probability."""
    try:
        portfolio = load_portfolio(portfolio_file)
        end = date.today()
        prices = make_provider(provider, seed).get_prices(
            portfolio.assets, end - timedelta(days=history_days), end
        )
        result = run_monte_carlo(
            portfolio,
            prices,
            n_simulations=simulations,
            days=days,
            seed=seed,
            loss_threshold=loss_threshold,
            use_drift=drift,
        )
    except (ValidationError, DataUnavailableError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    v0 = result.initial_value
    typer.echo(
        f"{portfolio.name}: {result.n_simulations:,} paths x {result.days} days, "
        f"start value {v0:,.2f} {portfolio.base_currency}"
    )
    typer.echo("\nLoss over horizon:")
    typer.echo(f"{'CONFIDENCE':<12}{'VaR':>14}{'VaR %':>8}{'CVaR':>14}{'CVaR %':>8}")
    for c, var in result.var.items():
        cvar = result.cvar[c]
        typer.echo(f"{c:<12.1%}{var:>14,.2f}{var / v0:>8.2%}{cvar:>14,.2f}{cvar / v0:>8.2%}")
    typer.echo("\nTerminal portfolio value:")
    rows = [
        ("5th pct", result.final_percentile(5)),
        ("median", result.median_final),
        ("mean", result.mean_final),
        ("95th pct", result.final_percentile(95)),
    ]
    for label, value in rows:
        typer.echo(f"{label:<12}{value:>14,.2f}{value / v0 - 1:>+9.1%}")
    typer.echo("\nMaximum drawdown:")
    for pct in (50, 95, 99):
        typer.echo(f"p{pct:<11}{result.drawdown_percentile(pct):>14.1%}")
    for level in (0.1, 0.2, 0.3, 0.5):
        typer.echo(f"P(MDD >= {level:.0%}){result.prob_drawdown_exceeds(level):>10.1%}")
    typer.echo(
        f"\nP(loss >= {result.loss_threshold:.0%}): at end {result.prob_loss_at_end:.2%}, "
        f"at any time (ruin) {result.prob_ruin:.2%}"
    )


if __name__ == "__main__":
    app()
