"""
store/db.py
Single source of truth for all DefinitionDrift persistence.
Tables:
  - definitions        : canonical metric definitions
  - definition_versions: full version history (append-only)
  - hitl_queue         : pending conflict approvals
  - schema_snapshots   : column-level schema snapshots per table
  - drift_log          : schema change events
"""

import sqlite3
import json
import hashlib
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

try:
    import libsql_experimental
except ImportError:
    libsql_experimental = None

DB_PATH = Path(__file__).parent.parent / "definitiondrift.db"


_TURSO_URL = os.getenv("TURSO_DATABASE_URL")
_TURSO_TOKEN = os.getenv("TURSO_AUTH_TOKEN")
_USE_TURSO = bool(_TURSO_URL and _TURSO_TOKEN and libsql_experimental)

if _USE_TURSO:
    print(f"[DB] 🌐 Using Turso cloud database: {_TURSO_URL}")
else:
    print(f"[DB] 💾 Using local SQLite: {DB_PATH}")


def get_conn() -> sqlite3.Connection:
    if _USE_TURSO:
        conn = libsql_experimental.connect(_TURSO_URL, auth_token=_TURSO_TOKEN)
    else:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)

    conn.row_factory = sqlite3.Row
    if not _USE_TURSO:
        conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS definitions (
            id          TEXT PRIMARY KEY,
            name        TEXT NOT NULL UNIQUE,
            description TEXT NOT NULL,
            sql_expr    TEXT,
            tags        TEXT DEFAULT '[]',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            version     INTEGER DEFAULT 1,
            approved    INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS definition_versions (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            def_id      TEXT NOT NULL,
            version     INTEGER NOT NULL,
            name        TEXT NOT NULL,
            description TEXT NOT NULL,
            sql_expr    TEXT,
            changed_by  TEXT DEFAULT 'system',
            changed_at  TEXT NOT NULL,
            reason      TEXT
        );

        CREATE TABLE IF NOT EXISTS hitl_queue (
            id              TEXT PRIMARY KEY,
            type            TEXT NOT NULL,
            question_a      TEXT NOT NULL,
            question_b      TEXT,
            def_a           TEXT,
            def_b           TEXT,
            similarity      REAL,
            status          TEXT DEFAULT 'pending',
            resolution      TEXT,
            resolved_at     TEXT,
            created_at      TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS schema_snapshots (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            table_name  TEXT NOT NULL,
            columns     TEXT NOT NULL,
            snapshot_at TEXT NOT NULL,
            hash        TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS drift_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            table_name  TEXT NOT NULL,
            change_type TEXT NOT NULL,
            detail      TEXT NOT NULL,
            affects_def TEXT,
            logged_at   TEXT NOT NULL,
            notified    INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS query_log (
            id          TEXT PRIMARY KEY,
            session_id  TEXT,
            question    TEXT NOT NULL,
            status      TEXT NOT NULL,
            intent      TEXT,
            provider    TEXT,
            latency_ms  INTEGER,
            sql_result  TEXT,
            used_definitions TEXT,
            feedback    INTEGER,
            created_at  TEXT DEFAULT (datetime('now'))
        );
        CREATE INDEX IF NOT EXISTS idx_ql_session  ON query_log(session_id);
        CREATE INDEX IF NOT EXISTS idx_ql_status   ON query_log(status);
        CREATE INDEX IF NOT EXISTS idx_ql_created  ON query_log(created_at DESC);
        CREATE INDEX IF NOT EXISTS idx_ql_intent   ON query_log(intent);

        CREATE TABLE IF NOT EXISTS sql_cache (
            question_hash   TEXT PRIMARY KEY,
            question        TEXT NOT NULL,
            sql             TEXT NOT NULL,
            used_definitions TEXT,
            confidence      TEXT,
            provider        TEXT,
            created_at      TEXT DEFAULT (datetime('now')),
            hit_count       INTEGER DEFAULT 0,
            last_hit        TEXT
        );
    """)
    conn.commit()
    conn.close()
    print(f"[DB] Initialized at {DB_PATH}")


# ── DEFINITIONS ──────────────────────────────────────────────────────────────

def upsert_definition(name: str, description: str, sql_expr: Optional[str] = None,
                      tags: list = None, approved: bool = False, reason: str = None) -> dict:
    conn = get_conn()
    now = datetime.utcnow().isoformat()
    def_id = hashlib.md5(name.lower().encode()).hexdigest()[:12]
    tags_json = json.dumps(tags or [])

    existing = conn.execute("SELECT * FROM definitions WHERE id=?", (def_id,)).fetchone()

    if existing:
        version = existing["version"] + 1
        conn.execute("""
            UPDATE definitions
            SET description=?, sql_expr=?, tags=?, updated_at=?, version=?, approved=?
            WHERE id=?
        """, (description, sql_expr, tags_json, now, version, int(approved), def_id))
        conn.execute("""
            INSERT INTO definition_versions (def_id, version, name, description, sql_expr, changed_at, reason)
            VALUES (?,?,?,?,?,?,?)
        """, (def_id, version, name, description, sql_expr, now, reason or "updated"))
    else:
        conn.execute("""
            INSERT INTO definitions (id, name, description, sql_expr, tags, created_at, updated_at, approved)
            VALUES (?,?,?,?,?,?,?,?)
        """, (def_id, name, description, sql_expr, tags_json, now, now, int(approved)))
        conn.execute("""
            INSERT INTO definition_versions (def_id, version, name, description, sql_expr, changed_at, reason)
            VALUES (?,1,?,?,?,?,?)
        """, (def_id, name, description, sql_expr, now, reason or "created"))

    conn.commit()
    row = conn.execute("SELECT * FROM definitions WHERE id=?", (def_id,)).fetchone()
    conn.close()
    return dict(row)


def get_all_definitions(approved_only: bool = False) -> list[dict]:
    conn = get_conn()
    query = "SELECT * FROM definitions"
    if approved_only:
        query += " WHERE approved=1"
    query += " ORDER BY name"
    rows = conn.execute(query).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_definition_by_name(name: str) -> Optional[dict]:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM definitions WHERE lower(name)=lower(?)", (name,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_definition_history(def_id: str) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM definition_versions WHERE def_id=? ORDER BY version DESC", (def_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── HITL QUEUE ───────────────────────────────────────────────────────────────

def enqueue_conflict(question_a: str, question_b: str,
                     def_a: Optional[str], def_b: Optional[str],
                     similarity: float) -> dict:
    conn = get_conn()
    now = datetime.utcnow().isoformat()
    conflict_id = hashlib.md5(f"{question_a}{question_b}{now}".encode()).hexdigest()[:12]
    conn.execute("""
        INSERT INTO hitl_queue (id, type, question_a, question_b, def_a, def_b, similarity, created_at)
        VALUES (?,?,?,?,?,?,?,?)
    """, (conflict_id, "definition_conflict", question_a, question_b,
          def_a, def_b, similarity, now))
    conn.commit()
    row = conn.execute("SELECT * FROM hitl_queue WHERE id=?", (conflict_id,)).fetchone()
    conn.close()
    return dict(row)


def get_pending_conflicts() -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM hitl_queue WHERE status='pending' ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def resolve_conflict(conflict_id: str, resolution: str, chosen_def: str) -> dict:
    conn = get_conn()
    now = datetime.utcnow().isoformat()
    conn.execute("""
        UPDATE hitl_queue
        SET status='resolved', resolution=?, resolved_at=?
        WHERE id=?
    """, (chosen_def, now, conflict_id))
    conn.commit()
    row = conn.execute("SELECT * FROM hitl_queue WHERE id=?", (conflict_id,)).fetchone()
    conn.close()
    return dict(row)


# ── SCHEMA SNAPSHOTS ─────────────────────────────────────────────────────────

def save_schema_snapshot(table_name: str, columns: list[dict]) -> tuple[bool, list[dict] | None]:
    """
    Returns (changed: bool, previous_columns: list | None).
    previous_columns is the OLD snapshot if changed, else None.
    """
    conn = get_conn()
    now = datetime.utcnow().isoformat()
    columns_json = json.dumps(columns, sort_keys=True)
    snap_hash = hashlib.md5(columns_json.encode()).hexdigest()

    last = conn.execute(
        "SELECT hash, columns FROM schema_snapshots WHERE table_name=? ORDER BY snapshot_at DESC LIMIT 1",
        (table_name,)
    ).fetchone()

    changed = not last or last["hash"] != snap_hash
    prev_columns = json.loads(last["columns"]) if (last and changed) else None

    if changed:
        conn.execute(
            "INSERT INTO schema_snapshots (table_name, columns, snapshot_at, hash) VALUES (?,?,?,?)",
            (table_name, columns_json, now, snap_hash)
        )
        conn.commit()

    conn.close()
    return changed, prev_columns


def get_last_snapshot(table_name: str) -> Optional[list[dict]]:
    conn = get_conn()
    row = conn.execute(
        "SELECT columns FROM schema_snapshots WHERE table_name=? ORDER BY snapshot_at DESC LIMIT 1",
        (table_name,)
    ).fetchone()
    conn.close()
    return json.loads(row["columns"]) if row else None


# ── DRIFT LOG ─────────────────────────────────────────────────────────────────

def log_drift(table_name: str, change_type: str, detail: str, affects_def: Optional[str] = None):
    conn = get_conn()
    conn.execute("""
        INSERT INTO drift_log (table_name, change_type, detail, affects_def, logged_at)
        VALUES (?,?,?,?,?)
    """, (table_name, change_type, detail, affects_def, datetime.utcnow().isoformat()))
    conn.commit()
    conn.close()


def get_unnotified_drift() -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM drift_log WHERE notified=0 ORDER BY logged_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def mark_drift_notified(ids: list[int]):
    conn = get_conn()
    conn.execute(
        f"UPDATE drift_log SET notified=1 WHERE id IN ({','.join('?' * len(ids))})", ids
    )
    conn.commit()
    conn.close()


if __name__ == "__main__":
    init_db()
    # seed Contoso-compatible definitions for testing
    upsert_definition(
        name="gross_sales",
        description="Total gross sales amount across all in-store transactions",
        sql_expr="SUM(FactSales.SalesAmount)",
        tags=["finance", "sales"],
        approved=True,
        reason="initial seed — Contoso-compatible"
    )
    upsert_definition(
        name="net_revenue",
        description="Net revenue after subtracting returns and discounts from gross sales",
        sql_expr="SUM(FactSales.SalesAmount - FactSales.ReturnAmount - FactSales.DiscountAmount)",
        tags=["finance", "revenue"],
        approved=True,
        reason="initial seed — Contoso-compatible"
    )
    upsert_definition(
        name="total_margin",
        description="Total profit margin across all in-store sales",
        sql_expr="SUM(FactSales.Margin)",
        tags=["finance", "profitability"],
        approved=False,
        reason="initial seed — pending approval"
    )
    print("[DB] Seeded 3 definitions.")
    print("[DB] All definitions:", [d["name"] for d in get_all_definitions()])


# ── SQL CACHE ──────────────────────────────────────────────────────────────────

def get_cached_sql(question: str) -> Optional[dict]:
    """Return cached SQL for an exact question match, or None."""
    h = hashlib.md5(question.strip().lower().encode()).hexdigest()
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM sql_cache WHERE question_hash=?", (h,)
    ).fetchone()
    if row:
        conn.execute(
            "UPDATE sql_cache SET hit_count=hit_count+1, last_hit=datetime('now') WHERE question_hash=?",
            (h,)
        )
        conn.commit()
    conn.close()
    return dict(row) if row else None


def cache_sql(question: str, sql: str, used_definitions: list,
             confidence: str, provider: str) -> None:
    """Cache a high-confidence SQL result for deterministic repeated queries."""
    h = hashlib.md5(question.strip().lower().encode()).hexdigest()
    conn = get_conn()
    conn.execute("""
        INSERT OR IGNORE INTO sql_cache
        (question_hash, question, sql, used_definitions, confidence, provider)
        VALUES (?,?,?,?,?,?)
    """, (h, question.strip(), sql,
          json.dumps(used_definitions), confidence, provider))
    conn.commit()
    conn.close()


def invalidate_sql_cache_for_definition(def_name: str) -> int:
    """Remove cached SQL entries that used this definition.
    Called whenever a definition is updated to prevent stale answers.
    Returns count of invalidated entries.
    """
    conn = get_conn()
    rows = conn.execute(
        "SELECT question_hash, used_definitions FROM sql_cache"
    ).fetchall()
    to_delete = []
    for row in rows:
        used = json.loads(row["used_definitions"] or "[]")
        if def_name in used:
            to_delete.append(row["question_hash"])
    if to_delete:
        placeholders = ','.join('?' * len(to_delete))
        conn.execute(
            f"DELETE FROM sql_cache WHERE question_hash IN ({placeholders})",
            to_delete
        )
        conn.commit()
    conn.close()
    return len(to_delete)


# ── QUERY LOG ─────────────────────────────────────────────────────────────────

def log_query(id: str = None, session_id: str = None, question: str = "",
              status: str = "", intent: str = None, provider: str = None,
              latency_ms: int = None, sql_result=None,
              used_definitions: list = None) -> str:
    """Log a query attempt. Returns the query ID."""
    conn = get_conn()
    query_id = id or hashlib.md5(
        f"{session_id}{question}{datetime.utcnow().isoformat()}".encode()
    ).hexdigest()[:16]
    # Serialize complex objects
    sql_str = json.dumps(sql_result) if isinstance(sql_result, dict) else (sql_result or "")
    defs_str = json.dumps(used_definitions) if isinstance(used_definitions, list) else (used_definitions or "[]")
    conn.execute("""
        INSERT OR IGNORE INTO query_log
        (id, session_id, question, status, intent, provider, latency_ms, sql_result, used_definitions)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (
        query_id, session_id, question, status,
        intent, provider, latency_ms, sql_str, defs_str
    ))
    conn.commit()
    conn.close()
    return query_id


def update_feedback(query_id: str, feedback: int) -> bool:
    """Update thumbs up (1) / thumbs down (-1) feedback for a logged query."""
    conn = get_conn()
    conn.execute(
        "UPDATE query_log SET feedback=? WHERE id=?", (feedback, query_id)
    )
    conn.commit()
    conn.close()
    return True


def get_query_history(session_id: str = None, limit: int = 50) -> list[dict]:
    """Return recent query log entries, optionally filtered by session."""
    conn = get_conn()
    if session_id:
        rows = conn.execute(
            "SELECT * FROM query_log WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
            (session_id, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM query_log ORDER BY created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_query_stats() -> dict:
    """Return aggregate statistics from the query log."""
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
    by_status = conn.execute(
        "SELECT status, COUNT(*) as cnt FROM query_log GROUP BY status"
    ).fetchall()
    by_intent = conn.execute(
        "SELECT intent, COUNT(*) as cnt FROM query_log WHERE intent IS NOT NULL GROUP BY intent"
    ).fetchall()
    avg_latency = conn.execute(
        "SELECT AVG(latency_ms) FROM query_log WHERE latency_ms IS NOT NULL"
    ).fetchone()[0]
    conn.close()
    return {
        "total_queries": total,
        "by_status": {r["status"]: r["cnt"] for r in by_status},
        "by_intent": {r["intent"]: r["cnt"] for r in by_intent},
        "avg_latency_ms": round(avg_latency or 0, 1),
    }