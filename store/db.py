"""SQLite store. Only violations are stored (redacted); clean steps are only counted."""
from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from contracts.schemas import Event, Verdict

DB_PATH = Path(os.getenv("MONITOR_DB", Path(__file__).with_name("monitor.db")))
_lock = threading.Lock()

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id        TEXT PRIMARY KEY,
    started_at    TEXT NOT NULL,
    ended_at      TEXT,
    status        TEXT NOT NULL DEFAULT 'running',
    checked       INTEGER NOT NULL DEFAULT 0,
    flagged       INTEGER NOT NULL DEFAULT 0,
    blocked       INTEGER NOT NULL DEFAULT 0,
    agent_tokens  INTEGER NOT NULL DEFAULT 0,
    agent_cost    REAL    NOT NULL DEFAULT 0,
    judge_tokens  INTEGER NOT NULL DEFAULT 0,
    judge_cost    REAL    NOT NULL DEFAULT 0,
    judge_calls   INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS violations (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id           TEXT NOT NULL REFERENCES runs(run_id),
    step             INTEGER NOT NULL,
    timestamp        TEXT NOT NULL,
    source           TEXT NOT NULL,
    type             TEXT NOT NULL,
    tool             TEXT,
    destination      TEXT,
    decision         TEXT NOT NULL,
    severity         TEXT NOT NULL,
    framework        TEXT,
    article          TEXT,
    reason           TEXT,
    redacted_summary TEXT,
    checked_by       TEXT
);
CREATE INDEX IF NOT EXISTS idx_violations_run ON violations(run_id, id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@contextmanager
def _conn() -> Iterator[sqlite3.Connection]:
    with _lock:
        con = sqlite3.connect(DB_PATH, timeout=10)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()


def init_db() -> None:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _conn() as con:
        con.executescript(SCHEMA)


def _ensure_run(con: sqlite3.Connection, run_id: str) -> None:
    con.execute("INSERT OR IGNORE INTO runs (run_id, started_at) VALUES (?, ?)", (run_id, _now()))


def start_run(run_id: str) -> None:
    with _conn() as con:
        _ensure_run(con, run_id)


def count_step(run_id: str) -> None:
    """Count a checked step without storing anything about it."""
    with _conn() as con:
        _ensure_run(con, run_id)
        con.execute("UPDATE runs SET checked = checked + 1 WHERE run_id = ?", (run_id,))


def save_violation(event: Event, verdict: Verdict) -> int:
    """Store a flag/block. Only the redacted summary is kept, never event.input."""
    if verdict.decision == "allow":
        raise ValueError("only violations are stored")
    col = "blocked" if verdict.decision == "block" else "flagged"
    with _conn() as con:
        _ensure_run(con, event.run_id)
        cur = con.execute(
            """INSERT INTO violations (run_id, step, timestamp, source, type, tool, destination, decision,
                   severity, framework, article, reason, redacted_summary, checked_by)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (event.run_id, event.step, event.timestamp.isoformat(timespec="seconds"), event.source, event.type,
             event.tool, event.destination, verdict.decision, verdict.severity, verdict.framework, verdict.article,
             verdict.reason, verdict.redacted_summary, verdict.checked_by),
        )
        con.execute(f"UPDATE runs SET {col} = {col} + 1 WHERE run_id = ?", (event.run_id,))
        return int(cur.lastrowid)


def add_usage(run_id: str, role: str, tokens: float, cost_usd: float, calls: int = 1) -> None:
    if role not in ("agent", "judge"):
        raise ValueError(role)
    with _conn() as con:
        _ensure_run(con, run_id)
        extra = ", judge_calls = judge_calls + ?" if role == "judge" else ""
        params: tuple = (int(tokens), float(cost_usd)) + ((calls,) if role == "judge" else ()) + (run_id,)
        con.execute(
            f"UPDATE runs SET {role}_tokens = {role}_tokens + ?, {role}_cost = {role}_cost + ?{extra} WHERE run_id = ?",
            params,
        )


def end_run(run_id: str, agent_tokens: int | None = None, agent_cost: float | None = None) -> None:
    with _conn() as con:
        _ensure_run(con, run_id)
        con.execute("UPDATE runs SET status = 'ended', ended_at = COALESCE(ended_at, ?) WHERE run_id = ?",
                    (_now(), run_id))
        if agent_tokens is not None:
            con.execute("UPDATE runs SET agent_tokens = ?, agent_cost = ? WHERE run_id = ?",
                        (int(agent_tokens), float(agent_cost or 0.0), run_id))


def get_runs(limit: int = 50) -> list[dict[str, Any]]:
    with _conn() as con:
        rows = con.execute("SELECT * FROM runs ORDER BY started_at DESC, rowid DESC LIMIT ?", (limit,)).fetchall()
    return [dict(r) for r in rows]


def get_run(run_id: str) -> dict[str, Any] | None:
    with _conn() as con:
        row = con.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
    return dict(row) if row else None


def get_violations(run_id: str, after_id: int = 0) -> list[dict[str, Any]]:
    with _conn() as con:
        rows = con.execute("SELECT * FROM violations WHERE run_id = ? AND id > ? ORDER BY id", (run_id, after_id)).fetchall()
    return [dict(r) for r in rows]
