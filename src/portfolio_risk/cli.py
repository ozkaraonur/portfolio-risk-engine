"""Command-line interface."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence
from datetime import date, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from loguru import logger
from pydantic import ValidationError
from rich.console import Console

from portfolio_risk import __version__
from portfolio_risk.catalog import synthetic_profiles
from portfolio_risk.data import (
    CachedProvider,
    DataUnavailableError,
    FxProvider,
    PriceProvider,
    StooqFx,
    StooqProvider,
    SyntheticFx,
    SyntheticProvider,
    YahooFx,
    YahooProvider,
    convert_to_base,
)
from portfolio_risk.data.cache import DEFAULT_TTL_HOURS, default_cache_dir
from portfolio_risk.importers import ImportFormatError, build_portfolio
from portfolio_risk.models import Asset, AssetClass, Portfolio
from portfolio_risk.reporting import build_analysis, render_dashboard, render_html, render_markdown
from portfolio_risk.reporting.history import (
    default_db_path,
    load_history,
    record,
    snapshot_of,
)
from portfolio_risk.reporting.limits import check_limits, load_limits
from portfolio_risk.risk import (
    BUILTIN_SCENARIOS,
    CORE_METHODS,
    CovMethod,
    Method,
    Objective,
    Scenario,
    analyze_risk,
    beta_scenario,
    compute_betas,
    correlation_matrix,
    efficient_frontier,
    get_scenario,
    optimize_portfolio,
    parametric_var_cvar,
    parse_custom_shocks,
    portfolio_beta,
    portfolio_stats,
    rebalance_trades,
    risk_contributions,
    run_backtest,
    run_monte_carlo,
    run_stress,
)

app = typer.Typer(
    help="Multi-broker portfolio risk and stress-testing engine.", no_args_is_help=True
)


class ProviderName(StrEnum):
    SYNTHETIC = "synthetic"
    STOOQ = "stooq"  # blocked by a JavaScript check; kept for private mirrors / tests
    YAHOO = "yahoo"


def describe_validation_error(path: Path, exc: ValidationError) -> str:
    """One readable line per problem instead of pydantic's multi-line dump."""
    problems = "; ".join(
        f"{'.'.join(str(part) for part in err['loc']) or 'file'}: {err['msg']}"
        for err in exc.errors()
    )
    return f"{path}: {problems}"


def load_portfolio(path: Path) -> Portfolio:
    try:
        return Portfolio.model_validate_json(path.read_text(encoding="utf-8"))
    except ValidationError as exc:
        raise ValueError(describe_validation_error(path, exc)) from exc


class _Settings:
    use_cache = True
    cache_dir: Path | None = None
    cache_ttl_hours = DEFAULT_TTL_HOURS


SETTINGS = _Settings()


def make_provider(name: ProviderName, seed: int) -> PriceProvider:
    if name is not ProviderName.SYNTHETIC:
        live: PriceProvider = YahooProvider() if name is ProviderName.YAHOO else StooqProvider()
        if not SETTINGS.use_cache:
            return live
        return CachedProvider(
            live, SETTINGS.cache_dir, namespace=name.value, ttl_hours=SETTINGS.cache_ttl_hours
        )
    return SyntheticProvider(seed=seed, profiles=synthetic_profiles())


def make_fx(name: ProviderName, seed: int) -> FxProvider:
    if name is ProviderName.YAHOO:
        return YahooFx()
    return StooqFx() if name is ProviderName.STOOQ else SyntheticFx(seed=seed)


def fetch_prices(
    portfolio: Portfolio,
    name: ProviderName,
    seed: int,
    days: int,
    *,
    assets: Sequence[Asset] | None = None,
) -> tuple[Portfolio, pd.DataFrame]:
    """Prices for the last ``days`` calendar days, plus the portfolio expressed in base currency.

    Positions in foreign currencies are converted with the provider's FX rates, so returns
    include currency moves; the returned portfolio holds all cash in the base currency.
    """
    end = date.today()
    start = end - timedelta(days=days)
    prices = make_provider(name, seed).get_prices(assets or portfolio.assets, start, end)
    converted, in_base = convert_to_base(portfolio, prices, make_fx(name, seed), start, end)
    return converted, in_base


@app.callback()
def main(
    cache: Annotated[
        bool, typer.Option("--cache/--no-cache", help="Cache downloaded prices on disk.")
    ] = True,
    cache_dir: Annotated[
        Path | None,
        typer.Option(help=f"Cache directory (default: {default_cache_dir()})."),
    ] = None,
    cache_ttl_hours: Annotated[
        float, typer.Option(min=0.0, help="Re-download cached prices older than this.")
    ] = DEFAULT_TTL_HOURS,
    verbose: Annotated[bool, typer.Option("--verbose", help="Show debug logs.")] = False,
) -> None:
    logger.remove()
    logger.add(sys.stderr, level="DEBUG" if verbose else "WARNING")
    SETTINGS.use_cache = cache
    SETTINGS.cache_dir = cache_dir
    SETTINGS.cache_ttl_hours = cache_ttl_hours


@app.command()
def version() -> None:
    """Print the package version."""
    typer.echo(__version__)


@app.command(name="import")
def import_positions(
    sources: Annotated[
        list[str],
        typer.Argument(help="Broker exports as BROKER=FILE.csv (or just FILE.csv)."),
    ],
    output: Annotated[Path, typer.Option("--output", "-o", help="Portfolio JSON to write.")],
    name: Annotated[str, typer.Option(help="Portfolio name.")] = "imported",
    base_currency: Annotated[str, typer.Option(help="Reporting currency.")] = "USD",
) -> None:
    """Merge broker position exports (CSV) into one portfolio file."""
    texts: dict[str, str] = {}
    try:
        for source in sources:
            broker, sep, raw_path = source.partition("=")
            path = Path(raw_path if sep else broker)
            broker = broker if sep else path.stem
            if broker in texts:
                raise ValueError(f"Broker '{broker}' given twice.")
            texts[broker] = path.read_text(encoding="utf-8-sig")
        portfolio, notes = build_portfolio(texts, name=name, base_currency=base_currency)
    except (OSError, ImportFormatError, ValidationError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(portfolio.model_dump_json(indent=2) + "\n", encoding="utf-8", newline="\n")
    for note in notes:
        typer.echo(f"Skipped {note}", err=True)
    typer.echo(
        f"Wrote {output}: {len(portfolio.positions)} positions, {len(portfolio.cash)} cash "
        f"balances across {', '.join(portfolio.brokers)}."
    )


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
        portfolio, prices = fetch_prices(portfolio, provider, seed, days)
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
    all_methods: Annotated[
        bool, typer.Option("--all-methods", help="Also show EWMA, Student-t, Cornish-Fisher, FHS.")
    ] = False,
) -> None:
    """VaR, CVaR (parametric and historical), diversification and correlations."""
    try:
        portfolio = load_portfolio(portfolio_file)
        portfolio, prices = fetch_prices(portfolio, provider, seed, days)
        reports = [
            analyze_risk(portfolio, prices, method=m, confidence=confidence, horizon=horizon)
            for m in (Method if all_methods else CORE_METHODS)
        ]
    except (ValidationError, DataUnavailableError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    parametric, historical = reports[0], reports[1]
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
def backtest(
    portfolio_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    confidence: Annotated[
        float, typer.Option(min=0.9, max=0.999, help="e.g. 0.95 or 0.99.")
    ] = 0.99,
    window: Annotated[
        int, typer.Option(min=20, help="Trailing observations used for each VaR forecast.")
    ] = 250,
    provider: Annotated[ProviderName, typer.Option(help="Price source.")] = ProviderName.SYNTHETIC,
    seed: Annotated[int, typer.Option(help="Seed for the synthetic provider.")] = 42,
    days: Annotated[int, typer.Option(min=120, help="History length in calendar days.")] = 1460,
) -> None:
    """Backtest one-day VaR: violations, Kupiec / Christoffersen tests, Basel traffic light."""
    try:
        portfolio = load_portfolio(portfolio_file)
        portfolio, prices = fetch_prices(portfolio, provider, seed, days)
        results = [
            run_backtest(portfolio, prices, method=m, confidence=confidence, window=window)
            for m in Method
        ]
    except (ValidationError, DataUnavailableError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    first = results[0]
    typer.echo(
        f"{portfolio.name}: {confidence:.1%} one-day VaR, {first.n_obs} test days "
        f"({first.dates[0].date()} to {first.dates[-1].date()}), window {window}"
    )
    typer.echo(
        f"{'METHOD':<16}{'VIOLATIONS':>11}{'EXPECTED':>10}{'RATE':>8}"
        f"{'KUPIEC p':>10}{'INDEP. p':>10}{'COND. p':>9}{'ZONE':>8}"
    )
    for r in results:
        typer.echo(
            f"{r.method.value:<16}{r.n_violations:>11}{r.expected_violations:>10.1f}"
            f"{r.violation_rate:>8.2%}{r.kupiec.p_value:>10.3f}{r.independence.p_value:>10.3f}"
            f"{r.conditional_coverage.p_value:>9.3f}{r.zone.value:>8}"
        )
    for r in results:
        if r.conditional_coverage.rejects():
            typer.echo(f"WARNING: {r.method.value} VaR is rejected at the 5% level.")


@app.command()
def attribute(
    portfolio_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    confidence: Annotated[
        float, typer.Option(min=0.9, max=0.999, help="e.g. 0.95 or 0.99.")
    ] = 0.99,
    horizon: Annotated[int, typer.Option(min=1, max=250, help="Horizon in trading days.")] = 10,
    method: Annotated[
        Method, typer.Option(help="parametric (Euler) or historical (loss tail).")
    ] = Method.PARAMETRIC,
    provider: Annotated[ProviderName, typer.Option(help="Price source.")] = ProviderName.SYNTHETIC,
    seed: Annotated[int, typer.Option(help="Seed for the synthetic provider.")] = 42,
    days: Annotated[int, typer.Option(min=60, help="History length in calendar days.")] = 730,
) -> None:
    """Risk attribution: each position's component VaR / CVaR and marginal VaR."""
    try:
        portfolio = load_portfolio(portfolio_file)
        portfolio, prices = fetch_prices(portfolio, provider, seed, days)
        latest = {str(k): float(v) for k, v in prices.iloc[-1].items()}
        exposures = pd.Series(portfolio.market_values(latest), dtype=float)
        result = risk_contributions(
            exposures,
            prices[portfolio.symbols].pct_change().dropna(),
            method=method,
            confidence=confidence,
            horizon=horizon,
        )
    except (ValidationError, DataUnavailableError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    typer.echo(
        f"{portfolio.name}: {method.value} {confidence:.1%} {horizon}-day VaR "
        f"{result.var:,.2f} {portfolio.base_currency}, CVaR {result.cvar:,.2f}"
    )
    typer.echo(
        f"{'SYMBOL':<10}{'EXPOSURE':>13}{'VaR CONTRIB.':>14}{'SHARE':>8}"
        f"{'MARGINAL/1000':>15}{'CVaR CONTRIB.':>15}"
    )
    for symbol in result.exposures.index:
        typer.echo(
            f"{symbol:<10}{result.exposures[symbol]:>13,.2f}{result.component_var[symbol]:>14,.2f}"
            f"{result.var_share[symbol]:>8.1%}{result.marginal_var[symbol] * 1000:>15,.2f}"
            f"{result.component_cvar[symbol]:>15,.2f}"
        )
    hedges = [str(s) for s in result.exposures.index if result.component_var[s] < 0]
    if hedges:
        typer.echo(f"Diversifying (negative contribution): {', '.join(hedges)}")


@app.command()
def optimize(
    portfolio_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    objective: Annotated[
        str, typer.Option(help="min-variance, risk-parity, max-sharpe or all.")
    ] = "all",
    max_weight: Annotated[
        float, typer.Option(min=0.05, max=1.0, help="Cap per asset (not used by risk-parity).")
    ] = 1.0,
    risk_free: Annotated[float, typer.Option(help="Annual risk-free rate for max-sharpe.")] = 0.0,
    frontier: Annotated[
        int, typer.Option(min=0, max=100, help="Print this many efficient-frontier points.")
    ] = 0,
    confidence: Annotated[float, typer.Option(min=0.9, max=0.999)] = 0.99,
    horizon: Annotated[int, typer.Option(min=1, max=250)] = 10,
    provider: Annotated[ProviderName, typer.Option(help="Price source.")] = ProviderName.SYNTHETIC,
    seed: Annotated[int, typer.Option(help="Seed for the synthetic provider.")] = 42,
    days: Annotated[int, typer.Option(min=60, help="History length in calendar days.")] = 730,
) -> None:
    """Long-only optimal weights over the risky assets, with the trades to reach them."""
    try:
        objectives = list(Objective) if objective == "all" else [Objective(objective)]
    except ValueError as exc:
        typer.echo(f"Error: unknown objective '{objective}'.", err=True)
        raise typer.Exit(code=1) from exc
    try:
        portfolio = load_portfolio(portfolio_file)
        portfolio, prices = fetch_prices(portfolio, provider, seed, days)
        returns = prices[portfolio.symbols].pct_change().dropna()
        latest = {str(k): float(v) for k, v in prices.iloc[-1].items()}
        values = pd.Series(portfolio.market_values(latest), dtype=float)
        invested = float(values.sum())
        if invested <= 0.0:
            raise ValueError("Portfolio has no risky assets to optimise.")
        current = values / invested
        cov = returns.cov()
        candidates = {"current": current}
        for obj in objectives:
            candidates[obj.value] = optimize_portfolio(
                obj, returns, max_weight=max_weight, risk_free=risk_free
            ).weights
        points = (
            efficient_frontier(returns, points=frontier, max_weight=max_weight) if frontier else []
        )
    except (ValidationError, DataUnavailableError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    symbols = list(current.index)
    typer.echo(
        f"{portfolio.name}: risky assets {invested:,.2f} {portfolio.base_currency} "
        f"(cash untouched); VaR is {confidence:.1%} {horizon}-day parametric"
    )
    typer.echo(
        f"{'ALLOCATION':<14}"
        + "".join(f"{s:>9}" for s in symbols)
        + f"{'RETURN':>9}{'VOL':>8}{'VaR':>11}{'VaR CHG':>9}"
    )
    base_var = 0.0
    for name, weights in candidates.items():
        ret, vol = portfolio_stats(weights, returns)
        var = parametric_var_cvar(weights * invested, cov, confidence, horizon)[0]
        base_var = base_var or var
        change = "" if name == "current" else f"{var / base_var - 1.0:+.1%}"
        typer.echo(
            f"{name:<14}"
            + "".join(f"{weights[s]:>9.1%}" for s in symbols)
            + f"{ret:>9.1%}{vol:>8.1%}{var:>11,.2f}{change:>9}"
        )
    for name, weights in list(candidates.items())[1:]:
        typer.echo(f"\nTrades for {name}:")
        for symbol, amount in rebalance_trades(values, weights).items():
            typer.echo(f"  {symbol:<10}{amount:>+14,.2f}")
    if points:
        typer.echo(f"\n{'FRONTIER RETURN':<16}{'VOL':>8}" + "".join(f"{s:>9}" for s in symbols))
        for point in points:
            typer.echo(
                f"{point.expected_return:<16.1%}{point.volatility:>8.1%}"
                + "".join(f"{point.weights[s]:>9.1%}" for s in symbols)
            )
    typer.echo(
        "Expected returns are sample means: treat max-sharpe and the frontier as indicative."
    )


@app.command()
def check(
    portfolio_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    limits_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    record_run: Annotated[
        bool, typer.Option("--record", help="Append the result to the history database.")
    ] = False,
    db: Annotated[Path | None, typer.Option(help="History database (default: cache dir).")] = None,
    simulations: Annotated[int, typer.Option("--simulations", "-n", min=100)] = 5_000,
    seed: Annotated[int, typer.Option(help="Seed for simulation and synthetic prices.")] = 42,
    provider: Annotated[ProviderName, typer.Option(help="Price source.")] = ProviderName.SYNTHETIC,
    history_days: Annotated[int, typer.Option(min=60, help="Calendar days of history.")] = 730,
) -> None:
    """Check the portfolio against risk limits; exits with code 2 if any limit is breached."""
    try:
        portfolio = load_portfolio(portfolio_file)
        try:
            limits = load_limits(limits_file)
        except ValidationError as exc:
            raise ValueError(describe_validation_error(limits_file, exc)) from exc
        portfolio, prices = fetch_prices(portfolio, provider, seed, history_days)
        analysis = build_analysis(portfolio, prices, simulations=simulations, seed=seed)
    except (ValidationError, DataUnavailableError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    checks = check_limits(analysis, limits)
    if not checks:
        typer.echo("No limits configured in the limits file.", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"{portfolio.name} as of {analysis.as_of}")
    typer.echo(f"{'LIMIT':<32}{'ACTUAL':>10}{'THRESHOLD':>12}  STATUS")
    for c in checks:
        relation = ">=" if c.is_minimum else "<="
        status = "BREACH" if c.breached else "ok"
        note = f" ({c.detail})" if c.detail else ""
        typer.echo(
            f"{c.name:<32}{c.actual:>10.2%}{relation + ' ' + format(c.limit, '.2%'):>12}"
            f"  {status}{note}"
        )
    breaches = sum(c.breached for c in checks)
    if record_run:
        row = record(db or default_db_path(), snapshot_of(analysis, breaches=breaches))
        typer.echo(f"Recorded snapshot #{row}.")
    if breaches:
        typer.echo(f"{breaches} limit(s) breached.", err=True)
        raise typer.Exit(code=2)
    typer.echo("All limits respected.")


@app.command()
def history(
    portfolio_name: Annotated[
        str | None, typer.Option("--portfolio", help="Only this portfolio name.")
    ] = None,
    limit: Annotated[int, typer.Option(min=1, max=1000, help="Rows to show.")] = 20,
    db: Annotated[Path | None, typer.Option(help="History database (default: cache dir).")] = None,
) -> None:
    """Show recorded risk snapshots, oldest first, with the change in VaR between runs."""
    rows = load_history(db or default_db_path(), portfolio_name, limit)
    if not rows:
        typer.echo("No snapshots recorded yet (use `pre report --record` or `pre check --record`).")
        return
    rows.reverse()
    typer.echo(
        f"{'RECORDED (UTC)':<18}{'PORTFOLIO':<14}{'VALUE':>13}{'VaR 10d 99%':>13}{'VaR %':>8}"
        f"{'CHG':>8}  {'TOP RISK':<14}{'BREACH':>6}"
    )
    previous: dict[str, float] = {}
    for r in rows:
        prior = previous.get(r.portfolio)
        change = "" if prior is None or prior == 0 else f"{r.var / prior - 1.0:+.1%}"
        previous[r.portfolio] = r.var
        top = f"{r.top_symbol} {r.top_risk_share:.0%}" if r.top_symbol else "-"
        breach = "-" if r.breaches is None else str(r.breaches)
        typer.echo(
            f"{r.recorded_at:%Y-%m-%d %H:%M}   {r.portfolio:<14}{r.total_value:>13,.2f}"
            f"{r.var:>13,.2f}{r.var_pct:>8.1%}{change:>8}  {top:<14}{breach:>6}"
        )


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
    df: Annotated[
        float | None,
        typer.Option(min=2.1, help="Student-t degrees of freedom for fat-tailed shocks."),
    ] = None,
    cov_method: Annotated[
        CovMethod, typer.Option(help="Covariance estimator for the simulation.")
    ] = CovMethod.SAMPLE,
    provider: Annotated[ProviderName, typer.Option(help="Price source.")] = ProviderName.SYNTHETIC,
    history_days: Annotated[
        int, typer.Option(min=60, help="Calendar days of history used to estimate risk.")
    ] = 730,
) -> None:
    """Monte Carlo value paths: VaR/CVaR, terminal percentiles, drawdowns, ruin probability."""
    try:
        portfolio = load_portfolio(portfolio_file)
        portfolio, prices = fetch_prices(portfolio, provider, seed, history_days)
        result = run_monte_carlo(
            portfolio,
            prices,
            n_simulations=simulations,
            days=days,
            seed=seed,
            loss_threshold=loss_threshold,
            use_drift=drift,
            df=df,
            cov_method=cov_method,
        )
    except (ValidationError, DataUnavailableError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    v0 = result.initial_value
    typer.echo(
        f"{portfolio.name}: {result.n_simulations:,} paths x {result.days} days, "
        f"start value {v0:,.2f} {portfolio.base_currency}"
    )
    shocks = "normal" if df is None else f"Student-t (df={df:g})"
    typer.echo(f"Shocks: {shocks}, covariance: {cov_method.value}")
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


@app.command()
def stress(
    portfolio_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    scenario: Annotated[
        str | None,
        typer.Option(help=f"Scenario name or 'all' ({', '.join(BUILTIN_SCENARIOS)})."),
    ] = None,
    custom: Annotated[
        str | None,
        typer.Option(help='E.g. "equity=-0.15,crypto=-0.30,AAPL=-0.5,tag:tech=-0.4".'),
    ] = None,
    market_shock: Annotated[
        float | None,
        typer.Option(help="Market move (e.g. -0.10); assets move by their beta vs the benchmark."),
    ] = None,
    benchmark: Annotated[str, typer.Option(help="Market index symbol for betas.")] = "SPY",
    loss_threshold: Annotated[
        float, typer.Option(min=0.01, max=1.0, help="Flag scenarios losing at least this fraction.")
    ] = 0.3,
    seed: Annotated[int, typer.Option(help="Seed for the synthetic provider.")] = 42,
    provider: Annotated[ProviderName, typer.Option(help="Price source.")] = ProviderName.SYNTHETIC,
    history_days: Annotated[
        int, typer.Option(min=60, help="Calendar days of history used for betas.")
    ] = 730,
) -> None:
    """Stress test: historical, custom and beta-driven market shock scenarios."""
    try:
        portfolio = load_portfolio(portfolio_file)
        scenarios: list[Scenario] = []
        if scenario is not None and scenario.lower() != "all":
            scenarios.append(get_scenario(scenario))
        elif scenario is not None or (custom is None and market_shock is None):
            scenarios.extend(BUILTIN_SCENARIOS.values())
        if custom is not None:
            scenarios.append(parse_custom_shocks(custom))

        assets = portfolio.assets
        if market_shock is not None and benchmark.upper() not in portfolio.symbols:
            assets = [*assets, Asset(symbol=benchmark, asset_class=AssetClass.EQUITY)]
        portfolio, prices = fetch_prices(portfolio, provider, seed, history_days, assets=assets)
        latest = {str(k): float(v) for k, v in prices.iloc[-1].items()}
        beta_line = ""
        if market_shock is not None:
            rets = prices.pct_change().dropna()
            betas = compute_betas(rets[portfolio.symbols], rets[benchmark.upper()])
            scenarios.append(beta_scenario(market_shock, betas, benchmark.upper()))
            pbeta = portfolio_beta(
                portfolio.market_values(latest), betas, portfolio.total_value(latest)
            )
            beta_line = (
                "Betas vs "
                + f"{benchmark.upper()}: "
                + ", ".join(f"{s}={b:.2f}" for s, b in betas.items())
                + f" | portfolio beta {pbeta:.2f}"
            )
        report = run_stress(portfolio, latest, scenarios)
    except (ValidationError, DataUnavailableError, ValueError, KeyError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    value = report.results[0].portfolio_value
    typer.echo(f"{portfolio.name}: value {value:,.2f} {portfolio.base_currency}")
    if beta_line:
        typer.echo(beta_line)
    typer.echo(f"\n{'SCENARIO':<18}{'P&L':>14}{'LOSS %':>9}{'STRESSED VALUE':>17}")
    for r in report.results:
        typer.echo(
            f"{r.scenario.name:<18}{r.total_pnl:>14,.2f}{r.pnl_pct:>9.1%}{r.stressed_value:>17,.2f}"
        )
    for r in report.results:
        typer.echo(f"\n{r.scenario.name} - {r.scenario.description}")
        typer.echo(f"{'SYMBOL':<10}{'VALUE':>14}{'SHOCK':>9}{'P&L':>14}")
        for i in r.impacts:
            typer.echo(f"{i.symbol:<10}{i.value:>14,.2f}{i.shock:>9.1%}{i.pnl:>14,.2f}")
        if r.cash > 0:
            typer.echo(f"{'CASH':<10}{r.cash:>14,.2f}{0.0:>9.1%}{0.0:>14,.2f}")
    worst = report.worst_case
    typer.echo(
        f"\nWorst case: {worst.scenario.name} loses {-worst.total_pnl:,.2f} "
        f"({-worst.pnl_pct:.1%}); remaining capital {worst.stressed_value:,.2f}"
    )
    breaches = report.breaches(loss_threshold)
    if breaches:
        names = ", ".join(b.scenario.name for b in breaches)
        typer.echo(f"WARNING: loss >= {loss_threshold:.0%} in: {names}")


@app.command()
def report(
    portfolio_file: Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    html: Annotated[Path | None, typer.Option(help="Write a self-contained HTML report.")] = None,
    markdown: Annotated[Path | None, typer.Option(help="Write a Markdown summary.")] = None,
    simulations: Annotated[
        int, typer.Option("--simulations", "-n", min=100, help="Monte Carlo paths.")
    ] = 10_000,
    mc_days: Annotated[int, typer.Option(min=1, help="Monte Carlo horizon in trading days.")] = 252,
    loss_threshold: Annotated[
        float, typer.Option(min=0.01, max=1.0, help="Loss fraction defining ruin.")
    ] = 0.3,
    seed: Annotated[int, typer.Option(help="Seed for simulation and synthetic prices.")] = 42,
    provider: Annotated[ProviderName, typer.Option(help="Price source.")] = ProviderName.SYNTHETIC,
    history_days: Annotated[
        int, typer.Option(min=60, help="Calendar days of history used to estimate risk.")
    ] = 730,
    quiet: Annotated[bool, typer.Option(help="Skip the terminal dashboard.")] = False,
    record_run: Annotated[
        bool, typer.Option("--record", help="Append the headline figures to the history database.")
    ] = False,
    db: Annotated[Path | None, typer.Option(help="History database (default: cache dir).")] = None,
) -> None:
    """Run the full analysis: terminal dashboard plus optional HTML / Markdown reports."""
    try:
        portfolio = load_portfolio(portfolio_file)
        portfolio, prices = fetch_prices(portfolio, provider, seed, history_days)
        analysis = build_analysis(
            portfolio,
            prices,
            simulations=simulations,
            mc_days=mc_days,
            seed=seed,
            loss_threshold=loss_threshold,
        )
    except (ValidationError, DataUnavailableError, ValueError) as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if not quiet:
        render_dashboard(analysis, Console())
    if record_run:
        row = record(db or default_db_path(), snapshot_of(analysis))
        typer.echo(f"Recorded snapshot #{row} for '{portfolio.name}'.")
    for path, content in ((html, render_html), (markdown, render_markdown)):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content(analysis), encoding="utf-8")
            typer.echo(f"Wrote {path}")


@app.command()
def web(
    port: Annotated[int, typer.Option(help="Port to serve the panel on.")] = 8501,
    host: Annotated[str, typer.Option(help="Bind address (0.0.0.0 in containers).")] = "localhost",
    browser: Annotated[bool, typer.Option(help="Open the browser automatically.")] = True,
) -> None:
    """Launch the Streamlit web panel."""
    script = Path(__file__).parent / "web" / "app.py"
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(script),
        "--server.port",
        str(port),
        "--server.headless",
        "false" if browser else "true",
        "--browser.gatherUsageStats",
        "false",
        "--server.address",
        host,
    ]
    raise typer.Exit(code=subprocess.call(command))


if __name__ == "__main__":
    app()
