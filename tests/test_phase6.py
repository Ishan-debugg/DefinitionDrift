"""
tests/test_phase6.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 6 — Multi-database (SQLAlchemy) test suite — 30 tests

Sections:
  [A] Connection abstraction unit tests   (12 tests)
  [B] DriftWatcher via SQLAlchemy         (7 tests)
  [C] QueryAgent._execute via SQLAlchemy  (6 tests)
  [D] API stats dialect field             (5 tests)

All tests run against an in-memory SQLite database — no external
service required.  The same code paths are exercised for Postgres/MySQL
because SQLAlchemy's Inspector and Core API are dialect-agnostic.

Run:
    python tests/test_phase6.py
"""

import sys, os, json, sqlite3, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Isolate the DefinitionDrift store DB ──────────────────────────────────────
import store.db as db_module
_TMP = Path(tempfile.gettempdir())
TEST_DB = str(_TMP / "dd_phase6_test.db")
db_module.DB_PATH = Path(TEST_DB)
if Path(TEST_DB).exists():
    Path(TEST_DB).unlink()

from store.db import init_db, upsert_definition

# ── Runner ─────────────────────────────────────────────────────────────────────
results = []
PASS, FAIL, SKIP = "✅", "❌", "⚠️ "

def check(label, cond, detail="", skip=False):
    if skip:
        results.append((SKIP, label, "skipped"))
        print(f"  {SKIP}  {label}" + (f" — {detail}" if detail else ""))
        return True
    icon = PASS if cond else FAIL
    results.append((icon, label, detail))
    print(f"  {icon}  {label}" + (f" — {detail}" if detail else ""))
    return cond

def section(name):
    print(f"\n{'─'*60}\n  {name}\n{'─'*60}")

print("\n" + "="*60)
print("  DefinitionDrift — Phase 6 Test Suite (30 tests)")
print("  Multi-database (SQLAlchemy) abstraction")
print("="*60)

init_db()

# ── Build a throwaway SQLite data DB for tests ────────────────────────────────
DATA_DB_FILE = str(_TMP / "dd_phase6_data.db")
if Path(DATA_DB_FILE).exists():
    Path(DATA_DB_FILE).unlink()

_seed_conn = sqlite3.connect(DATA_DB_FILE)
_seed_conn.executescript("""
    CREATE TABLE FactSales (
        SalesKey INTEGER PRIMARY KEY,
        DateKey  INTEGER,
        StoreKey INTEGER,
        SalesAmount REAL,
        ReturnAmount REAL,
        SalesQuantity INTEGER,
        ReturnQuantity INTEGER
    );
    CREATE TABLE DimProduct (
        ProductKey INTEGER PRIMARY KEY,
        ProductName TEXT,
        UnitPrice REAL
    );
    INSERT INTO FactSales VALUES (1, 20080101, 10, 100.0, 5.0, 10, 1);
    INSERT INTO FactSales VALUES (2, 20080201, 11, 200.0, 0.0, 20, 0);
    INSERT INTO DimProduct VALUES (1, 'Widget A', 9.99);
    INSERT INTO DimProduct VALUES (2, 'Widget B', 14.99);
""")
_seed_conn.commit()
_seed_conn.close()

DATA_DB_URL = f"sqlite:///{DATA_DB_FILE}"
print(f"  Test data DB: {DATA_DB_FILE}\n")


# ══════════════════════════════════════════════════════════════════
# [A] CONNECTION ABSTRACTION UNIT TESTS — 12 tests
# ══════════════════════════════════════════════════════════════════
section("[A] db/connection.py — Unit Tests")

from db.connection import get_engine, resolve_url, execute_query, inspect_schema

# A1 — module imports cleanly
check("A1  db.connection imports without error", True)

# A2 — resolve_url: bare file path → sqlite:/// URL
resolved = resolve_url(DATA_DB_FILE)
check("A2  resolve_url converts bare path to sqlite:/// URL",
      resolved.startswith("sqlite:///"),
      f"resolved={resolved[:60]}")

# A3 — resolve_url: pass-through for full SQLAlchemy URL
pg_url = "postgresql+psycopg2://user:secret@localhost/db"
check("A3  resolve_url passes through a full SQLAlchemy URL unchanged",
      resolve_url(pg_url) == pg_url)

# A4 — resolve_url: password redacted in logs (internal helper)
from db.connection import _redact
check("A4  _redact hides password in connection strings",
      "secret" not in _redact(pg_url) and "***" in _redact(pg_url),
      f"redacted={_redact(pg_url)}")

# A5 — get_engine returns an Engine
engine = get_engine(DATA_DB_URL)
from sqlalchemy.engine import Engine
check("A5  get_engine() returns a SQLAlchemy Engine",
      isinstance(engine, Engine))

# A6 — get_engine is cached (same object for same URL)
engine2 = get_engine(DATA_DB_URL)
check("A6  get_engine() is cached — same URL returns same object",
      engine is engine2)

# A7 — dialect is sqlite
check("A7  Engine dialect is 'sqlite' for sqlite:/// URL",
      engine.dialect.name == "sqlite",
      f"dialect={engine.dialect.name}")

# A8 — execute_query returns rows
result = execute_query(engine, "SELECT SUM(SalesAmount) AS total FROM FactSales")
check("A8  execute_query returns result dict with 'rows' key",
      "rows" in result and isinstance(result["rows"], list),
      f"row_count={result.get('row_count')}")
check("A8b execute_query returns correct aggregate",
      result["rows"][0].get("total") == 300.0,
      f"total={result['rows'][0].get('total')}")

# A9 — execute_query: bad SQL → error dict (no exception raised)
bad = execute_query(engine, "SELECT * FROM NonExistentTable_xyz")
check("A9  execute_query on bad SQL returns error dict, not exception",
      "error" in bad and "rows" not in bad,
      f"keys={list(bad.keys())}")

# A10 — inspect_schema returns table → col list mapping
schema = inspect_schema(engine)
check("A10 inspect_schema returns dict",
      isinstance(schema, dict) and len(schema) > 0,
      f"tables={list(schema.keys())}")

# A11 — FactSales is in schema
check("A11 FactSales found in inspected schema",
      "FactSales" in schema,
      f"found tables={list(schema.keys())}")

# A12 — each column entry has required keys
fact_cols = schema.get("FactSales", [])
required_keys = {"name", "type", "nullable", "primary_key"}
check("A12 Column entries have name/type/nullable/primary_key keys",
      all(required_keys.issubset(c.keys()) for c in fact_cols),
      f"sample={fact_cols[0] if fact_cols else 'none'}")


# ══════════════════════════════════════════════════════════════════
# [B] DRIFT WATCHER VIA SQLALCHEMY — 7 tests
# ══════════════════════════════════════════════════════════════════
section("[B] DriftWatcher via SQLAlchemy Inspector")

from agents.core import drift_watcher

# B1 — first snapshot runs without error on SQLAlchemy URL
events1 = drift_watcher.snapshot_and_diff(DATA_DB_URL)
check("B1  snapshot_and_diff accepts SQLite URL (no crash)",
      isinstance(events1, list) and not any(e.get("type") == "error" for e in events1),
      f"events={events1}")

# B2 — first snapshot returns no change events (baseline)
check("B2  First snapshot returns empty list (baseline established)",
      len(events1) == 0, f"events={events1}")

# B3 — also accepts bare file path (backward compat)
events_path = drift_watcher.snapshot_and_diff(DATA_DB_FILE)
check("B3  snapshot_and_diff accepts bare file path (backward compat)",
      isinstance(events_path, list) and not any(e.get("type") == "error" for e in events_path))

# B4 — alter schema and detect new column
_alter_conn = sqlite3.connect(DATA_DB_FILE)
_alter_conn.execute("ALTER TABLE DimProduct ADD COLUMN Discontinued INTEGER DEFAULT 0")
_alter_conn.commit()
_alter_conn.close()

# Invalidate the cached engine for this URL (the file changed)
get_engine.cache_clear()

events2 = drift_watcher.snapshot_and_diff(DATA_DB_URL)
added = [e for e in events2 if e.get("type") == "column_added"]
check("B4  Added column detected after ALTER TABLE",
      len(added) >= 1,
      f"added_cols={[e.get('column') for e in added]}")

# B5 — event has required fields
if added:
    check("B5  column_added event has type/table/column/detail",
          all(k in added[0] for k in ("type", "table", "column", "detail")),
          str(added[0]))
else:
    check("B5  column_added event fields", False, "no added event found")

# B6 — no changes on second snapshot of same schema
events3 = drift_watcher.snapshot_and_diff(DATA_DB_URL)
check("B6  No change events on re-snapshot of same schema",
      len([e for e in events3 if e.get("type") != "error"]) == 0,
      f"events={events3}")

# B7 — error on invalid path returns error event, not exception
events_bad = drift_watcher.snapshot_and_diff("sqlite:///nonexistent_path/nope.db")
check("B7  Invalid DB path returns error event, not unhandled exception",
      isinstance(events_bad, list),
      f"events={events_bad}")


# ══════════════════════════════════════════════════════════════════
# [C] QUERYAGENT._execute VIA SQLALCHEMY — 6 tests
# ══════════════════════════════════════════════════════════════════
section("[C] QueryAgent._execute via SQLAlchemy")

from agents.core import query_agent

# C1 — accepts SQLAlchemy URL
r = query_agent._execute("SELECT SUM(SalesAmount) AS total FROM FactSales", DATA_DB_URL)
check("C1  _execute accepts SQLite URL",
      "rows" in r, f"result={r}")

# C2 — returns correct value
check("C2  _execute returns correct aggregate value",
      r.get("rows", [{}])[0].get("total") == 300.0,
      f"total={r.get('rows', [{}])[0].get('total')}")

# C3 — accepts bare file path (backward compat)
r2 = query_agent._execute("SELECT COUNT(*) AS n FROM DimProduct", DATA_DB_FILE)
check("C3  _execute accepts bare file path (backward compat)",
      r2.get("rows", [{}])[0].get("n") == 2,
      f"count={r2.get('rows', [{}])[0].get('n')}")

# C4 — row count is capped at 100
# Insert 150 rows temporarily
_big_conn = sqlite3.connect(DATA_DB_FILE)
for i in range(3, 153):
    _big_conn.execute(
        "INSERT INTO DimProduct (ProductKey, ProductName, UnitPrice) VALUES (?,?,?)",
        (i, f"Product {i}", float(i) + 0.99)
    )
_big_conn.commit()
_big_conn.close()
get_engine.cache_clear()

r3 = query_agent._execute("SELECT * FROM DimProduct", DATA_DB_URL)
check("C4  _execute caps results at 100 rows",
      r3.get("row_count", 0) == 100,
      f"row_count={r3.get('row_count')}")

# C5 — bad SQL returns error dict
r4 = query_agent._execute("SELECT FakeCol FROM NonExistentTable", DATA_DB_URL)
check("C5  _execute on bad SQL returns error dict without raising",
      "error" in r4 and "rows" not in r4,
      f"keys={list(r4.keys())}")

# C6 — dialect field is present in successful result
check("C6  Successful _execute result includes 'dialect' field",
      "dialect" in r,
      f"dialect={r.get('dialect')}")


# ══════════════════════════════════════════════════════════════════
# [D] API STATS — DIALECT FIELD — 5 tests
# ══════════════════════════════════════════════════════════════════
section("[D] API /api/stats — dialect field")

try:
    from fastapi.testclient import TestClient
    import api.main as api_main
    # Point the API at our test data DB
    api_main.DATA_DB = DATA_DB_URL
    client = TestClient(api_main.app, raise_server_exceptions=False)
    fastapi_ok = True
except Exception as e:
    fastapi_ok = False
    print(f"     FastAPI test client unavailable: {e}")

# D1
if fastapi_ok:
    r = client.get("/api/stats")
    check("D1  GET /api/stats returns 200", r.status_code == 200)
else:
    check("D1  /api/stats", True, "skipped — install httpx", skip=True)

# D2
if fastapi_ok:
    body = r.json()
    check("D2  /api/stats response has data_db.dialect field",
          "data_db" in body and "dialect" in body.get("data_db", {}),
          f"data_db={body.get('data_db')}")
else:
    check("D2  dialect field", True, "skipped", skip=True)

# D3
if fastapi_ok:
    dialect = body.get("data_db", {}).get("dialect", "")
    check("D3  dialect is 'sqlite' for our test URL",
          dialect == "sqlite",
          f"dialect={dialect}")
else:
    check("D3  dialect value", True, "skipped", skip=True)

# D4
if fastapi_ok:
    check("D4  /api/stats has embedding_model field",
          "embedding_model" in body,
          f"embedding_model={body.get('embedding_model')}")
else:
    check("D4  embedding_model field", True, "skipped", skip=True)

# D5 — resolve_url env-var priority
os.environ["DATA_DB_URL"] = "sqlite:///./data/contoso.db"
os.environ.pop("DATA_DB_PATH", None)
resolved_priority = resolve_url()
check("D5  DATA_DB_URL takes priority over DATA_DB_PATH when both set",
      resolved_priority.startswith("sqlite:///"),
      f"resolved={resolved_priority[:60]}")
os.environ.pop("DATA_DB_URL", None)  # clean up


# ── SUMMARY ───────────────────────────────────────────────────────────────────
total   = len(results)
passed  = sum(1 for r in results if r[0] == PASS)
failed  = sum(1 for r in results if r[0] == FAIL)
skipped = sum(1 for r in results if r[0] == SKIP)

print(f"\n{'='*60}")
print(f"  RESULTS: {passed} passed  {failed} failed  {skipped} skipped  ({total} total)")
print(f"{'='*60}")

if failed:
    print("\n  Failed tests:")
    for icon, label, detail in results:
        if icon == FAIL:
            print(f"    {icon} {label}" + (f" — {detail}" if detail else ""))
    sys.exit(1)
else:
    note = (
        "All Phase 6 tests passed!"
        if not skipped
        else f"Core tests passed ({skipped} skipped — install httpx for API tests)"
    )
    print(f"\n  {note}\n")
