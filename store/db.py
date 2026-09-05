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
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent.parent / "definitiondrift.db"


def get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
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
    # seed some definitions for testing
    upsert_definition(
        name="active_users",
        description="Users who logged in at least once in the last 7 days",
        sql_expr="SELECT COUNT(DISTINCT user_id) FROM sessions WHERE login_at >= date('now','-7 days')",
        tags=["users", "engagement"],
        approved=True,
        reason="initial seed"
    )
    upsert_definition(
        name="revenue",
        description="Sum of order totals excluding refunds, in USD",
        sql_expr="SELECT SUM(total_amount) FROM orders WHERE status != 'refunded'",
        tags=["finance"],
        approved=True,
        reason="initial seed"
    )
    upsert_definition(
        name="churn_rate",
        description="Percentage of users who did not return within 30 days of their last session",
        sql_expr=None,
        tags=["users", "retention"],
        approved=False,
        reason="initial seed — pending approval"
    )
    print("[DB] Seeded 3 definitions.")
    print("[DB] All definitions:", [d["name"] for d in get_all_definitions()])