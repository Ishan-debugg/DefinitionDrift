"""
store/query_log.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Persistence layer for query history.

Extracted from api/main.py so that storage logic lives in the
store/ layer, not in the API layer.

Public API:
    init_query_log()                      — call once at startup
    save_query(qid, question, ...)        — persist a completed query
    get_query_history(session_id, limit)  — read history rows
"""

import json
import sqlite3
from pathlib import Path
from typing import Optional

QLOG_DB = Path(__file__).parent.parent / "data" / "query_history.db"


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(QLOG_DB, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_query_log():
    """Create the query_history table and indexes if they don't exist."""
    QLOG_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = _get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS query_history (
            id               TEXT PRIMARY KEY,
            question         TEXT,
            status           TEXT,
            sql_result       TEXT,
            provider         TEXT,
            definitions_used TEXT,
            latency_ms       INTEGER,
            session_id       TEXT,
            created_at       TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_qh_session ON query_history(session_id);
        CREATE INDEX IF NOT EXISTS idx_qh_status  ON query_history(status);
        CREATE INDEX IF NOT EXISTS idx_qh_created ON query_history(created_at DESC);
    """)
    conn.commit()
    conn.close()


def save_query(
    qid: str,
    question: str,
    status: str,
    sql_result: Optional[dict],
    provider: str,
    definitions_used: list,
    latency_ms: int,
    session_id: str,
) -> None:
    """Persist a completed query to the history store."""
    conn = _get_conn()
    conn.execute(
        """INSERT OR REPLACE INTO query_history
           (id, question, status, sql_result, provider, definitions_used, latency_ms, session_id, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, datetime('now'))""",
        (
            qid,
            question,
            status,
            json.dumps(sql_result),
            provider,
            json.dumps(definitions_used),
            latency_ms,
            session_id,
        ),
    )
    conn.commit()
    conn.close()


def get_query_history(session_id: Optional[str] = None, limit: int = 50) -> list[dict]:
    """Return query history rows, optionally filtered by session_id."""
    conn = _get_conn()
    if session_id:
        rows = conn.execute(
            "SELECT * FROM query_history WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM query_history ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
