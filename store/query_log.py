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
    get_query_stats()                     — aggregated stats via SQL (Fix #7)
"""

import json
import sqlite3
import threading
from pathlib import Path
from typing import Optional

QLOG_DB = Path(__file__).parent.parent / "data" / "query_history.db"

# Fix #1: Persistent connection per thread
_qlog_local = threading.local()

def _get_conn() -> sqlite3.Connection:
    """Return a thread-local persistent connection (no open/close per call)."""
    if not hasattr(_qlog_local, 'conn') or _qlog_local.conn is None:
        _qlog_local.conn = sqlite3.connect(QLOG_DB, check_same_thread=False)
        _qlog_local.conn.row_factory = sqlite3.Row
        _qlog_local.conn.execute("PRAGMA journal_mode=WAL")
    return _qlog_local.conn


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
    return [dict(r) for r in rows]


def get_query_stats() -> dict:
    """Fix #7: Compute stats via SQL aggregation — O(1) memory, instant.
    
    Replaces the old pattern of loading 10K rows into Python and iterating.
    """
    conn = _get_conn()
    row = conn.execute("""
        SELECT
            COUNT(*) as total,
            SUM(CASE WHEN status='ok' THEN 1 ELSE 0 END) as ok_count,
            AVG(CASE WHEN status='ok' THEN latency_ms END) as avg_latency
        FROM query_history
    """).fetchone()
    return {
        "total": row[0] or 0,
        "ok_count": row[1] or 0,
        "avg_latency": round(row[2] or 0, 0),
    }
