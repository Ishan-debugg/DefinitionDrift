"""
db/connection.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SQLAlchemy connection abstraction for DefinitionDrift
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Supported dialects (zero-purchase):
  SQLite   — sqlite:///./data/contoso.db         (default / local dev)
  Postgres — postgresql+psycopg2://user:pw@host/db
  MySQL    — mysql+pymysql://user:pw@host/db

Usage:
    from db.connection import get_engine, execute_query, inspect_schema

    engine = get_engine()                          # uses DATA_DB_URL from env
    engine = get_engine("sqlite:///./my.db")       # explicit URL
    rows   = execute_query(engine, "SELECT 1")     # returns list[dict]
    schema = inspect_schema(engine)                # returns {table: [col_info]}

Environment variables:
    DATA_DB_URL   — primary (any SQLAlchemy connection string)
    DATA_DB_PATH  — legacy fallback; auto-converted to sqlite:/// URL
"""

import os
import functools
from pathlib import Path
from typing import Optional

from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine


# ── Helpers ───────────────────────────────────────────────────────────────────

def _path_to_sqlite_url(path: str) -> str:
    """Convert a bare file path to an absolute sqlite:/// URL."""
    p = Path(path).resolve()
    # SQLAlchemy on Windows needs four slashes for absolute paths
    return f"sqlite:///{p.as_posix()}"


def resolve_url(url: Optional[str] = None) -> str:
    """
    Return a fully-formed SQLAlchemy connection string.

    Priority:
      1. ``url`` argument (if provided)
      2. ``DATA_DB_URL`` environment variable
      3. ``DATA_DB_PATH`` environment variable (auto-converted to sqlite:///)
      4. Default: sqlite:///./data/contoso.db relative to repo root
    """
    if url:
        # If caller passed a bare file path, normalise it
        if not url.startswith(("sqlite", "postgresql", "mysql", "mssql", "oracle")):
            return _path_to_sqlite_url(url)
        return url

    env_url = os.getenv("DATA_DB_URL", "")
    if env_url:
        return env_url

    env_path = os.getenv("DATA_DB_PATH", "")
    if env_path:
        return _path_to_sqlite_url(env_path)

    # Absolute default — repo root / data / contoso.db
    default = Path(__file__).parent.parent / "data" / "contoso.db"
    return _path_to_sqlite_url(str(default))


# ── Engine factory (per-URL singleton) ────────────────────────────────────────

@functools.lru_cache(maxsize=8)
def get_engine(url: Optional[str] = None) -> Engine:
    """
    Return a cached SQLAlchemy Engine for the given URL.

    Engines are singletons per URL — safe to call repeatedly.
    Connection pooling is handled automatically by SQLAlchemy.

    Args:
        url: SQLAlchemy connection string or bare file path.
             Defaults to resolve_url() (reads DATA_DB_URL / DATA_DB_PATH).
    """
    resolved = resolve_url(url)
    kwargs: dict = {}

    if resolved.startswith("sqlite"):
        # sqlite needs check_same_thread=False in threaded servers (FastAPI)
        kwargs["connect_args"] = {"check_same_thread": False}
        kwargs["pool_pre_ping"] = True
    else:
        # For Postgres/MySQL use a small pool; adjust via DB_POOL_SIZE env var
        pool_size = int(os.getenv("DB_POOL_SIZE", "5"))
        kwargs["pool_size"] = pool_size
        kwargs["pool_pre_ping"] = True   # detect stale connections
        kwargs["pool_recycle"] = 1800    # recycle every 30 min

    engine = create_engine(resolved, **kwargs)
    dialect = engine.dialect.name
    print(f"[DataDB] Engine created — dialect={dialect}, url={_redact(resolved)}")
    return engine


def _redact(url: str) -> str:
    """Hide password in connection string for safe logging."""
    import re
    return re.sub(r"(://[^:]+:)[^@]+(@)", r"\1***\2", url)


# ── Query execution ───────────────────────────────────────────────────────────

def execute_query(
    engine: Engine,
    sql: str,
    params: Optional[dict] = None,
    max_rows: int = 100,
) -> dict:
    """
    Execute *sql* on *engine* and return a result dict.

    Returns:
        {"rows": [...], "row_count": N, "dialect": "sqlite|postgresql|..."}
    On error:
        {"error": "<message>", "sql_attempted": sql}
    """
    try:
        with engine.connect() as conn:
            result = conn.execute(text(sql), params or {})
            keys = list(result.keys())
            rows = [dict(zip(keys, row)) for row in result.fetchmany(max_rows)]
        return {
            "rows": rows,
            "row_count": len(rows),
            "dialect": engine.dialect.name,
        }
    except Exception as exc:
        return {"error": str(exc), "sql_attempted": sql}


# ── Schema inspection ─────────────────────────────────────────────────────────

def inspect_schema(engine: Engine) -> dict[str, list[dict]]:
    """
    Reflect the data-db schema using SQLAlchemy's Inspector.

    Works with SQLite, PostgreSQL, MySQL — no dialect-specific PRAGMA needed.

    Returns:
        {
          "TableName": [
            {"name": "col", "type": "VARCHAR", "nullable": True, "primary_key": False},
            ...
          ],
          ...
        }
    """
    inspector = inspect(engine)
    schema: dict[str, list[dict]] = {}

    for table_name in inspector.get_table_names():
        cols = []
        pk_cols = {c for c in inspector.get_pk_constraint(table_name).get("constrained_columns", [])}
        for col in inspector.get_columns(table_name):
            cols.append({
                "name": col["name"],
                "type": str(col["type"]),
                "nullable": col.get("nullable", True),
                "primary_key": col["name"] in pk_cols,
            })
        schema[table_name] = cols

    return schema


# ── Convenience re-exports ────────────────────────────────────────────────────

__all__ = [
    "get_engine",
    "resolve_url",
    "execute_query",
    "inspect_schema",
]
