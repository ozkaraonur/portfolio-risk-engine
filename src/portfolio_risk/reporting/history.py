"""Analysis history: headline risk figures per run, kept in a local SQLite file."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from portfolio_risk.data.cache import default_cache_dir
from portfolio_risk.reporting.analysis import RiskAnalysis

_SCHEMA = """
CREATE TABLE IF NOT EXISTS snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    portfolio TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    as_of TEXT NOT NULL,
    currency TEXT NOT NULL,
    total_value REAL NOT NULL,
    cash_ratio REAL NOT NULL,
    var REAL NOT NULL,
    cvar REAL NOT NULL,
    mc_var REAL NOT NULL,
    top_symbol TEXT,
    top_risk_share REAL NOT NULL,
    breaches INTEGER
)
"""


@dataclass(frozen=True)
class Snapshot:
    portfolio: str
    recorded_at: datetime
    as_of: str
    currency: str
    total_value: float
    cash_ratio: float
    var: float  # headline: parametric, 10-day, 99%
    cvar: float
    mc_var: float  # Monte Carlo, 99%
    top_symbol: str | None
    top_risk_share: float
    breaches: int | None  # number of breached limits, if limits were checked

    @property
    def var_pct(self) -> float:
        return self.var / self.total_value if self.total_value > 0 else 0.0


def default_db_path() -> Path:
    return default_cache_dir() / "history.sqlite"


def _connect(db: Path) -> sqlite3.Connection:
    db.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db)
    conn.execute(_SCHEMA)
    return conn


def snapshot_of(analysis: RiskAnalysis, *, breaches: int | None = None) -> Snapshot:
    headline = analysis.headline_var
    shares = analysis.contributions.var_share
    return Snapshot(
        portfolio=analysis.portfolio.name,
        recorded_at=datetime.now(UTC),
        as_of=analysis.as_of.isoformat(),
        currency=analysis.portfolio.base_currency,
        total_value=analysis.total_value,
        cash_ratio=analysis.cash_ratio,
        var=headline.var,
        cvar=headline.cvar,
        mc_var=analysis.monte_carlo.var[0.99],
        top_symbol=str(shares.idxmax()) if len(shares) else None,
        top_risk_share=float(shares.max()) if len(shares) else 0.0,
        breaches=breaches,
    )


def record(db: Path, snapshot: Snapshot) -> int:
    """Append ``snapshot``; returns its row id."""
    with _connect(db) as conn:
        cursor = conn.execute(
            "INSERT INTO snapshots (portfolio, recorded_at, as_of, currency, total_value,"
            " cash_ratio, var, cvar, mc_var, top_symbol, top_risk_share, breaches)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                snapshot.portfolio,
                snapshot.recorded_at.isoformat(),
                snapshot.as_of,
                snapshot.currency,
                snapshot.total_value,
                snapshot.cash_ratio,
                snapshot.var,
                snapshot.cvar,
                snapshot.mc_var,
                snapshot.top_symbol,
                snapshot.top_risk_share,
                snapshot.breaches,
            ),
        )
        row_id = cursor.lastrowid
    conn.close()
    if row_id is None:
        raise RuntimeError("SQLite did not return a row id.")
    return row_id


def load_history(db: Path, portfolio: str | None = None, limit: int = 20) -> list[Snapshot]:
    """Most recent snapshots first; empty if the database does not exist yet."""
    if not db.exists():
        return []
    where, args = ("WHERE portfolio = ?", (portfolio, limit)) if portfolio else ("", (limit,))
    conn = _connect(db)
    try:
        rows = conn.execute(
            "SELECT portfolio, recorded_at, as_of, currency, total_value, cash_ratio, var, cvar,"
            f" mc_var, top_symbol, top_risk_share, breaches FROM snapshots {where}"
            " ORDER BY id DESC LIMIT ?",
            args,
        ).fetchall()
    finally:
        conn.close()
    return [Snapshot(r[0], datetime.fromisoformat(r[1]), *r[2:]) for r in rows]
