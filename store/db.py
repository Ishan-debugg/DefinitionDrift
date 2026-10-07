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
from contextvars import ContextVar
from datetime import datetime
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

# ── Multi-tenancy ──────────────────────────────────────────────────────────────
# Every read/write below is scoped to the current user. The API sets this per
# request from the X-User-Id header; scripts/tests/MCP use the 'default' user.
DEFAULT_USER = "default"
_current_user: ContextVar[str] = ContextVar("dd_user_id", default=DEFAULT_USER)


def set_current_user(user_id: Optional[str]):
    return _current_user.set(user_id or DEFAULT_USER)


def get_current_user() -> str:
    return _current_user.get()


def _uid(user_id: Optional[str] = None) -> str:
    return user_id or _current_user.get()


def _def_id(user_id: str, name: str) -> str:
    # 'default' keeps the legacy id so existing rows/history stay valid
    key = name.lower() if user_id == DEFAULT_USER else f"{user_id}:{name.lower()}"
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _question_hash(question: str) -> str:
    uid = _uid()
    q = question.strip().lower()
    key = q if uid == DEFAULT_USER else f"{uid}:{q}"
    return hashlib.md5(key.encode()).hexdigest()

try:
    import libsql_experimental
except ImportError:
    libsql_experimental = None

DB_PATH = Path(__file__).parent.parent / "definitiondrift.db"


_TURSO_URL = os.getenv("TURSO_DATABASE_URL")
_TURSO_TOKEN = os.getenv("TURSO_AUTH_TOKEN")
_USE_TURSO = False # Temporarily disabled due to libsql_experimental row_factory bug

# Postgres is used whenever DATABASE_URL is set; otherwise local SQLite.
_DATABASE_URL = os.getenv("DATABASE_URL", "").strip()
if _DATABASE_URL.startswith("postgres://"):
    _DATABASE_URL = "postgresql://" + _DATABASE_URL[len("postgres://"):]
_USE_PG = _DATABASE_URL.startswith("postgresql://")

if _USE_PG:
    print("[DB] 🐘 Using PostgreSQL (DATABASE_URL)")
elif _USE_TURSO:
    print(f"[DB] 🌐 Using Turso cloud database: {_TURSO_URL}")
else:
    print(f"[DB] 💾 Using local SQLite: {DB_PATH}")


# ── Postgres adapter ─────────────────────────────────────────────────────────
# Lets the SQLite-flavoured SQL below run unchanged on Postgres.

class _Row(dict):
    """Row supporting both row["col"] and row[0], like sqlite3.Row."""
    def __getitem__(self, key):
        if isinstance(key, int):
            return list(self.values())[key]
        return dict.__getitem__(self, key)


_PG_NOW = "to_char(now() at time zone 'utc','YYYY-MM-DD HH24:MI:SS')"


def _to_pg(sql: str) -> str:
    sql = sql.replace("?", "%s")
    sql = sql.replace("datetime('now')", _PG_NOW)
    sql = sql.replace("INTEGER PRIMARY KEY AUTOINCREMENT", "SERIAL PRIMARY KEY")
    if "INSERT OR IGNORE INTO" in sql:
        sql = sql.replace("INSERT OR IGNORE INTO", "INSERT INTO").rstrip().rstrip(";")
        sql += " ON CONFLICT DO NOTHING"
    return sql


class _PgCursor:
    def __init__(self, cur):
        self._cur = cur

    def fetchone(self):
        r = self._cur.fetchone()
        return _Row(r) if r is not None else None

    def fetchall(self):
        return [_Row(r) for r in self._cur.fetchall()]

    @property
    def rowcount(self):
        return self._cur.rowcount


class _PgConn:
    def __init__(self, raw, pool):
        self._raw, self._pool = raw, pool

    def _cursor(self):
        from psycopg2.extras import RealDictCursor
        return self._raw.cursor(cursor_factory=RealDictCursor)

    def execute(self, sql, params=()):
        if sql.strip().upper().startswith("PRAGMA"):
            return _PgCursor(self._cursor())
        cur = self._cursor()
        cur.execute(_to_pg(sql), tuple(params))
        return _PgCursor(cur) if cur.description else _PgCursor(cur)

    def executescript(self, script):
        cur = self._cursor()
        for stmt in script.split(";"):
            if stmt.strip():
                cur.execute(_to_pg(stmt))

    def commit(self):
        self._raw.commit()

    def close(self):
        try:
            self._raw.rollback()  # no-op after commit; clears aborted txns
        finally:
            self._pool.putconn(self._raw)


_pg_pool = None


def _pg_conn():
    global _pg_pool
    if _pg_pool is None:
        from psycopg2.pool import ThreadedConnectionPool
        _pg_pool = ThreadedConnectionPool(1, 10, _DATABASE_URL)
    return _PgConn(_pg_pool.getconn(), _pg_pool)


def get_conn():
    if _USE_PG:
        return _pg_conn()
    if _USE_TURSO:
        conn = libsql_experimental.connect(_TURSO_URL, auth_token=_TURSO_TOKEN)
    else:
        conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_db():
    conn = get_conn()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            user_id     TEXT PRIMARY KEY,
            created_at  TEXT NOT NULL,
            last_seen   TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS definitions (
            id          TEXT PRIMARY KEY,
            user_id     TEXT NOT NULL DEFAULT 'default',
            name        TEXT NOT NULL,
            description TEXT NOT NULL,
            sql_expr    TEXT,
            tags        TEXT DEFAULT '[]',
            created_at  TEXT NOT NULL,
            updated_at  TEXT NOT NULL,
            version     INTEGER DEFAULT 1,
            approved    INTEGER DEFAULT 0,
            UNIQUE (user_id, name)
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
            reason      TEXT,
            user_id     TEXT NOT NULL DEFAULT 'default'
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
            created_at      TEXT NOT NULL,
            user_id         TEXT NOT NULL DEFAULT 'default'
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
            created_at  TEXT DEFAULT (datetime('now')),
            user_id     TEXT NOT NULL DEFAULT 'default'
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
            last_hit        TEXT,
            user_id         TEXT NOT NULL DEFAULT 'default'
        );
    """)
    conn.commit()
    if not _USE_PG:
        _migrate_sqlite(conn)
    conn.executescript("""
        CREATE INDEX IF NOT EXISTS idx_def_user  ON definitions(user_id);
        CREATE INDEX IF NOT EXISTS idx_hitl_user ON hitl_queue(user_id, status);
        CREATE INDEX IF NOT EXISTS idx_ql_user   ON query_log(user_id, created_at);
        CREATE INDEX IF NOT EXISTS idx_sc_user   ON sql_cache(user_id);
    """)
    conn.commit()
    conn.close()
    print(f"[DB] Initialized at {'PostgreSQL' if _USE_PG else DB_PATH}")


# ── DEFINITIONS ──────────────────────────────────────────────────────────────

def _migrate_sqlite(conn):
    """Add user_id to pre-multi-tenant local SQLite databases."""
    def cols(table):
        return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}

    for table in ("definition_versions", "hitl_queue", "query_log", "sql_cache"):
        if "user_id" not in cols(table):
            conn.execute(f"ALTER TABLE {table} ADD COLUMN user_id TEXT NOT NULL DEFAULT 'default'")

    if "user_id" not in cols("definitions"):
        # Recreate to replace the global UNIQUE(name) with UNIQUE(user_id, name)
        conn.executescript("""
            CREATE TABLE definitions_new (
                id          TEXT PRIMARY KEY,
                user_id     TEXT NOT NULL DEFAULT 'default',
                name        TEXT NOT NULL,
                description TEXT NOT NULL,
                sql_expr    TEXT,
                tags        TEXT DEFAULT '[]',
                created_at  TEXT NOT NULL,
                updated_at  TEXT NOT NULL,
                version     INTEGER DEFAULT 1,
                approved    INTEGER DEFAULT 0,
                UNIQUE (user_id, name)
            );
            INSERT INTO definitions_new
                (id, user_id, name, description, sql_expr, tags, created_at, updated_at, version, approved)
            SELECT id, 'default', name, description, sql_expr, tags, created_at, updated_at, version, approved
            FROM definitions;
            DROP TABLE definitions;
            ALTER TABLE definitions_new RENAME TO definitions;
        """)
    conn.commit()


# ── USERS ─────────────────────────────────────────────────────────────────────

_SEED_DEFINITIONS = [
    dict(name="gross_sales",
         description="Total gross sales amount across all in-store transactions",
         sql_expr="SUM(FactSales.SalesAmount)", tags=["finance", "sales"], approved=True),
    dict(name="net_revenue",
         description="Net revenue after subtracting returns and discounts from gross sales",
         sql_expr="SUM(FactSales.SalesAmount - FactSales.ReturnAmount - FactSales.DiscountAmount)",
         tags=["finance", "revenue"], approved=True),
    dict(name="total_margin",
         description="Total profit margin across all in-store sales",
         sql_expr="SUM(FactSales.Margin)", tags=["finance", "profitability"], approved=True),
]


def ensure_user(user_id: str) -> bool:
    """Create the profile on first sight and seed starter definitions.
    Returns True if the user was newly created."""
    if user_id == DEFAULT_USER:
        return False
    conn = get_conn()
    now = datetime.utcnow().isoformat()
    cur = conn.execute(
        "INSERT OR IGNORE INTO users (user_id, created_at, last_seen) VALUES (?,?,?)",
        (user_id, now, now))
    created = cur.rowcount == 1
    if not created:
        conn.execute("UPDATE users SET last_seen=? WHERE user_id=?", (now, user_id))
    conn.commit()
    templates = []
    if created:
        templates = [dict(r) for r in conn.execute(
            "SELECT name, description, sql_expr, tags FROM definitions "
            "WHERE user_id=? AND approved=1", (DEFAULT_USER,)).fetchall()]
    conn.close()

    if created:
        if templates:
            for t in templates:
                upsert_definition(name=t["name"], description=t["description"],
                                  sql_expr=t["sql_expr"], tags=json.loads(t["tags"] or "[]"),
                                  approved=True, reason="seeded for new profile", user_id=user_id)
        else:
            for s in _SEED_DEFINITIONS:
                upsert_definition(**s, reason="seeded for new profile", user_id=user_id)
    return created


def upsert_definition(name: str, description: str, sql_expr: Optional[str] = None,
                      tags: list = None, approved: bool = False, reason: str = None,
                      user_id: Optional[str] = None) -> dict:
    uid = _uid(user_id)
    conn = get_conn()
    now = datetime.utcnow().isoformat()
    def_id = _def_id(uid, name)
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
            INSERT INTO definition_versions (def_id, version, name, description, sql_expr, changed_at, reason, user_id)
            VALUES (?,?,?,?,?,?,?,?)
        """, (def_id, version, name, description, sql_expr, now, reason or "updated", uid))
    else:
        conn.execute("""
            INSERT INTO definitions (id, user_id, name, description, sql_expr, tags, created_at, updated_at, approved)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (def_id, uid, name, description, sql_expr, tags_json, now, now, int(approved)))
        conn.execute("""
            INSERT INTO definition_versions (def_id, version, name, description, sql_expr, changed_at, reason, user_id)
            VALUES (?,1,?,?,?,?,?,?)
        """, (def_id, name, description, sql_expr, now, reason or "created", uid))

    conn.commit()
    row = conn.execute("SELECT * FROM definitions WHERE id=?", (def_id,)).fetchone()
    conn.close()
    return dict(row)


def get_all_definitions(approved_only: bool = False) -> list[dict]:
    conn = get_conn()
    query = "SELECT * FROM definitions WHERE user_id=?"
    if approved_only:
        query += " AND approved=1"
    query += " ORDER BY name"
    rows = conn.execute(query, (_uid(),)).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_definition_by_name(name: str) -> Optional[dict]:
    conn = get_conn()
    row = conn.execute(
        "SELECT * FROM definitions WHERE lower(name)=lower(?) AND user_id=?", (name, _uid())
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def get_definition_history(def_id: str) -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM definition_versions WHERE def_id=? AND user_id=? ORDER BY version DESC",
        (def_id, _uid())
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── HITL QUEUE ───────────────────────────────────────────────────────────────

def enqueue_conflict(question_a: str, question_b: str,
                     def_a: Optional[str], def_b: Optional[str],
                     similarity: float) -> dict:
    uid = _uid()
    conn = get_conn()
    now = datetime.utcnow().isoformat()
    conflict_id = hashlib.md5(f"{uid}{question_a}{question_b}{now}".encode()).hexdigest()[:12]
    conn.execute("""
        INSERT INTO hitl_queue (id, type, question_a, question_b, def_a, def_b, similarity, created_at, user_id)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (conflict_id, "definition_conflict", question_a, question_b,
          def_a, def_b, float(similarity) if similarity is not None else None, now, uid))
    conn.commit()
    row = conn.execute("SELECT * FROM hitl_queue WHERE id=?", (conflict_id,)).fetchone()
    conn.close()
    return dict(row)


def get_pending_conflicts() -> list[dict]:
    conn = get_conn()
    rows = conn.execute(
        "SELECT * FROM hitl_queue WHERE status='pending' AND user_id=? ORDER BY created_at DESC",
        (_uid(),)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def resolve_conflict(conflict_id: str, resolution: str, chosen_def: str) -> dict:
    conn = get_conn()
    now = datetime.utcnow().isoformat()
    conn.execute("""
        UPDATE hitl_queue
        SET status='resolved', resolution=?, resolved_at=?
        WHERE id=? AND user_id=?
    """, (chosen_def, now, conflict_id, _uid()))
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
    h = _question_hash(question)
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
    h = _question_hash(question)
    conn = get_conn()
    conn.execute("""
        INSERT OR IGNORE INTO sql_cache
        (question_hash, question, sql, used_definitions, confidence, provider, user_id)
        VALUES (?,?,?,?,?,?,?)
    """, (h, question.strip(), sql,
          json.dumps(used_definitions), confidence, provider, _uid()))
    conn.commit()
    conn.close()


def invalidate_sql_cache_for_definition(def_name: str) -> int:
    """Remove cached SQL entries that used this definition.
    Called whenever a definition is updated to prevent stale answers.
    Returns count of invalidated entries.
    """
    conn = get_conn()
    rows = conn.execute(
        "SELECT question_hash, used_definitions FROM sql_cache WHERE user_id=?", (_uid(),)
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
    uid = _uid()
    conn = get_conn()
    query_id = id or hashlib.md5(
        f"{uid}{session_id}{question}{datetime.utcnow().isoformat()}".encode()
    ).hexdigest()[:16]
    # Serialize complex objects
    sql_str = json.dumps(sql_result) if isinstance(sql_result, dict) else (sql_result or "")
    defs_str = json.dumps(used_definitions) if isinstance(used_definitions, list) else (used_definitions or "[]")
    conn.execute("""
        INSERT OR IGNORE INTO query_log
        (id, session_id, question, status, intent, provider, latency_ms, sql_result, used_definitions, user_id)
        VALUES (?,?,?,?,?,?,?,?,?,?)
    """, (
        query_id, session_id, question, status,
        intent, provider, latency_ms, sql_str, defs_str, uid
    ))
    conn.commit()
    conn.close()
    return query_id


def update_feedback(query_id: str, feedback: int) -> bool:
    """Update thumbs up (1) / thumbs down (-1) feedback for a logged query."""
    conn = get_conn()
    conn.execute(
        "UPDATE query_log SET feedback=? WHERE id=? AND user_id=?", (feedback, query_id, _uid())
    )
    conn.commit()
    conn.close()
    return True


def get_query_history(session_id: str = None, limit: int = 50) -> list[dict]:
    """Return recent query log entries, optionally filtered by session."""
    conn = get_conn()
    if session_id:
        rows = conn.execute(
            "SELECT * FROM query_log WHERE user_id=? AND session_id=? ORDER BY created_at DESC LIMIT ?",
            (_uid(), session_id, limit)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM query_log WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
            (_uid(), limit)
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_definition_usage() -> list[dict]:
    """Per-definition usage stats for the current user (DB-agnostic)."""
    conn = get_conn()
    rows = conn.execute(
        "SELECT used_definitions, latency_ms, feedback FROM query_log "
        "WHERE user_id=? AND used_definitions IS NOT NULL", (_uid(),)
    ).fetchall()
    conn.close()
    agg: dict[str, dict] = {}
    for r in rows:
        try:
            names = json.loads(r["used_definitions"] or "[]")
        except (TypeError, ValueError):
            continue
        for n in names if isinstance(names, list) else []:
            a = agg.setdefault(n, {"name": n, "query_count": 0, "_lat": [],
                                   "thumbs_up": 0, "thumbs_down": 0})
            a["query_count"] += 1
            if r["latency_ms"] is not None:
                a["_lat"].append(r["latency_ms"])
            a["thumbs_up"] += 1 if r["feedback"] == 1 else 0
            a["thumbs_down"] += 1 if r["feedback"] == -1 else 0
    out = []
    for a in agg.values():
        lat = a.pop("_lat")
        a["avg_latency"] = round(sum(lat) / len(lat), 1) if lat else None
        out.append(a)
    return sorted(out, key=lambda x: -x["query_count"])


def get_query_stats() -> dict:
    """Return aggregate statistics from the query log."""
    uid = _uid()
    conn = get_conn()
    total = conn.execute("SELECT COUNT(*) FROM query_log WHERE user_id=?", (uid,)).fetchone()[0]
    by_status = conn.execute(
        "SELECT status, COUNT(*) as cnt FROM query_log WHERE user_id=? GROUP BY status", (uid,)
    ).fetchall()
    by_intent = conn.execute(
        "SELECT intent, COUNT(*) as cnt FROM query_log WHERE user_id=? AND intent IS NOT NULL GROUP BY intent",
        (uid,)
    ).fetchall()
    avg_latency = conn.execute(
        "SELECT AVG(latency_ms) FROM query_log WHERE user_id=? AND latency_ms IS NOT NULL", (uid,)
    ).fetchone()[0]
    conn.close()
    return {
        "total": total,
        "by_status": {r["status"]: r["cnt"] for r in by_status},
        "by_intent": {r["intent"]: r["cnt"] for r in by_intent},
        "avg_latency": round(float(avg_latency or 0), 1),
        "ok_count": next((r["cnt"] for r in by_status if r["status"] == "ok"), 0)
    }