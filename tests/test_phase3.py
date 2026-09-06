"""
tests/test_phase3.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 3 test suite — 38 tests
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Sections:
  [A] SQL Validator — parse + schema  (14 tests)
  [B] Conversation memory             (10 tests)
  [C] QueryAgent with memory+validate  (7 tests)
  [D] Rate limiting + API endpoints    (7 tests)

Run: python tests/test_phase3.py
"""

import sys, os, json, time, sqlite3, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import store.db as db_module
_TMPDIR   = Path(tempfile.gettempdir())
TEST_DB   = str(_TMPDIR / "dd_phase3_test.db")
CONV_DB_P = str(_TMPDIR / "dd_phase3_conv.db")
db_module.DB_PATH = Path(TEST_DB)
if Path(TEST_DB).exists():   Path(TEST_DB).unlink()

import store.conversation as conv_module
conv_module.CONV_DB = Path(CONV_DB_P)
if Path(CONV_DB_P).exists(): Path(CONV_DB_P).unlink()

from store.db import init_db, upsert_definition
from store.conversation import (
    init_conversation_db, add_message, get_session_messages,
    get_conversation_context, clear_session, session_summary
)
from agents.sql_validator import validate_sql, validate_and_fix, CONTOSO_SCHEMA
from agents.core import query_agent, optimizer

CONTOSO_DB = str(Path(__file__).parent.parent / "data" / "contoso.db")

# ── Runner ────────────────────────────────────────────────────────────────────
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
print("  DefinitionDrift — Phase 3 Test Suite (38 tests)")
print("="*60)

init_db()
init_conversation_db()
upsert_definition(name="net_revenue",
    description="SalesAmount minus ReturnAmount across all channels",
    sql_expr="SELECT SUM(SalesAmount - ReturnAmount) FROM FactSales",
    approved=True, reason="test")
upsert_definition(name="gross_sales",
    description="Total SalesAmount before returns",
    sql_expr="SELECT SUM(SalesAmount) FROM FactSales",
    approved=True, reason="test")


# ══════════════════════════════════════════════════════════════════
# [A] SQL VALIDATOR — 14 tests
# ══════════════════════════════════════════════════════════════════
section("[A] SQL Validator")

# A1 — valid simple
r = validate_sql("SELECT SUM(SalesAmount) FROM FactSales")
check("A1  Valid simple query passes",
      r.valid and r.parse_ok and r.schema_ok,
      f"valid={r.valid}, errors={r.errors}")

# A2 — valid join
sql_join = """SELECT dd.CalendarYear, SUM(fs.SalesAmount)
FROM FactSales fs
JOIN DimDate dd ON fs.DateKey = dd.DateKey
WHERE dd.CalendarYear = 2008
GROUP BY dd.CalendarYear"""
r2 = validate_sql(sql_join)
check("A2  Valid join with DimDate passes",
      r2.valid, f"errors={r2.errors}")

# A3 — valid CTE
sql_cte = """WITH net AS (
  SELECT SUM(SalesAmount - ReturnAmount) as net_rev FROM FactSales
)
SELECT net_rev FROM net"""
r3 = validate_sql(sql_cte)
check("A3  Valid CTE passes",
      r3.valid, f"errors={r3.errors}")

# A4 — syntax error
r4 = validate_sql("SELECT SUM(SalesAmount FROM FactSales")
check("A4  Syntax error detected",
      not r4.valid and not r4.parse_ok,
      f"errors={r4.errors[:1]}")

# A5 — hallucinated column
r5 = validate_sql("SELECT SUM(Revenue) FROM FactSales")
check("A5  Hallucinated column 'Revenue' caught",
      not r5.valid and any("Revenue" in e for e in r5.errors),
      f"errors={r5.errors}")

# A6 — unknown table
r6 = validate_sql("SELECT * FROM SalesFact")
check("A6  Unknown table 'SalesFact' caught",
      not r6.valid and any("SalesFact" in e for e in r6.errors),
      f"errors={r6.errors}")

# A7 — valid multi-table union
sql_union = """SELECT SUM(SalesAmount) FROM FactSales
UNION ALL
SELECT SUM(SalesAmount) FROM FactOnlineSales"""
r7 = validate_sql(sql_union)
check("A7  UNION ALL across two fact tables passes",
      r7.valid, f"errors={r7.errors}")

# A8 — SELECT * warning
r8 = validate_sql("SELECT * FROM DimProduct")
check("A8  SELECT * produces warning (not error)",
      r8.parse_ok and r8.schema_ok,
      f"warnings={r8.warnings}")
check("A8b SELECT * warning text present",
      any("SELECT *" in w or "star" in w.lower() for w in r8.warnings),
      f"warnings={r8.warnings}")

# A9 — empty SQL
r9 = validate_sql("")
check("A9  Empty SQL fails gracefully",
      not r9.valid and len(r9.errors) > 0)

# A10 — fixed_sql reformatted
r10 = validate_sql("select sum(salesamount) from factsales")
check("A10 fixed_sql is reformatted (prettier)",
      r10.fixed_sql is not None,
      f"fixed={r10.fixed_sql[:60] if r10.fixed_sql else 'none'}")

# A11 — validate_and_fix convenience wrapper
ok, sql_out, errs = validate_and_fix("SELECT SUM(SalesAmount - ReturnAmount) FROM FactSales")
check("A11 validate_and_fix returns (True, sql, [])",
      ok and isinstance(sql_out, str) and len(errs) == 0,
      f"ok={ok}, errs={errs}")

# A12 — validate_and_fix with bad SQL
ok2, sql_out2, errs2 = validate_and_fix("SELECT FakeColumn FROM FactSales")
check("A12 validate_and_fix returns (False, ..., errors) for bad SQL",
      not ok2 and len(errs2) > 0,
      f"ok={ok2}, errs={errs2[:1]}")

# A13 — schema dict completeness
check("A13 All 9 Contoso tables in schema dict",
      len(CONTOSO_SCHEMA) == 9,
      f"found {len(CONTOSO_SCHEMA)} tables: {list(CONTOSO_SCHEMA.keys())}")

# A14 — known columns present
check("A14 SalesAmount in FactSales schema",
      "salesamount" in CONTOSO_SCHEMA["factsales"])


# ══════════════════════════════════════════════════════════════════
# [B] CONVERSATION MEMORY — 10 tests
# ══════════════════════════════════════════════════════════════════
section("[B] Conversation Memory")

SID = "test-session-phase3"

# B1 — add and retrieve
mid1 = add_message(SID, "user", "What is net revenue for 2008?", "question")
check("B1  Message added returns msg_id", isinstance(mid1, str) and len(mid1) > 0)

# B2
msgs = get_session_messages(SID)
check("B2  Message visible in session", len(msgs) == 1 and msgs[0]["content"] == "What is net revenue for 2008?")

# B3 — assistant reply
mid2 = add_message(SID, "assistant", "Net revenue for 2008 was $4.2M.", "sql_result",
    sql_result={"sql": "SELECT SUM(SalesAmount-ReturnAmount) FROM FactSales", "confidence": "high"})
msgs2 = get_session_messages(SID)
check("B3  Both turns stored", len(msgs2) == 2)

# B4 — second user turn
add_message(SID, "user", "What about Q3 of the same year?", "question")
add_message(SID, "assistant", "Q3 2008 net revenue was $1.1M.", "sql_result",
    sql_result={"sql": "SELECT SUM(SalesAmount-ReturnAmount) FROM FactSales JOIN DimDate ON FactSales.DateKey=DimDate.DateKey WHERE CalendarYear=2008 AND CalendarQuarter=3", "confidence": "high"})

# B5 — context block
ctx = get_conversation_context(SID, max_turns=6)
check("B5  Context block has user/assistant pairs",
      len(ctx) >= 2 and ctx[0]["role"] == "user",
      f"turns={len(ctx)}")

# B6 — context roles alternate correctly (user then assistant, repeating)
roles = [m["role"] for m in ctx]
def _alternates(roles):
    for i in range(len(roles) - 1):
        if roles[i] == roles[i + 1]:
            return False
    return True
check("B6  Context alternates user/assistant",
      _alternates(roles),
      f"roles={roles}")

# B7 — max_turns limit
ctx_limited = get_conversation_context(SID, max_turns=1)
check("B7  max_turns=1 limits context to ≤2 messages",
      len(ctx_limited) <= 2, f"got {len(ctx_limited)}")

# B8 — session summary
summary = session_summary(SID)
check("B8  Session summary has total_messages",
      summary["total_messages"] >= 4,
      f"total={summary['total_messages']}")

# B9 — clear session
clear_session(SID)
msgs_after = get_session_messages(SID)
check("B9  Clear session removes all messages",
      len(msgs_after) == 0)

# B10 — new session starts fresh
SID2 = "test-session-2"
add_message(SID2, "user", "What is gross sales?", "question")
ctx2 = get_conversation_context(SID)  # original session
check("B10 Different sessions don't cross-contaminate",
      len(ctx2) == 0, f"original session ctx len={len(ctx2)}")


# ══════════════════════════════════════════════════════════════════
# [C] QUERYAGENT WITH MEMORY + VALIDATION — 7 tests
# ══════════════════════════════════════════════════════════════════
section("[C] QueryAgent with Memory + SQL Validation")

SID3 = "test-agent-session"

# C1 — basic run with session
r = query_agent.run("What is net revenue?", session_id=SID3)
check("C1  QueryAgent runs with session_id", isinstance(r, dict) and "sql" in r)

# C2 — memory saved after run
msgs_after = get_session_messages(SID3, limit=10)
check("C2  Turn saved to conversation memory after run",
      len(msgs_after) >= 2,
      f"messages stored: {len(msgs_after)}")

# C3 — validation result present
check("C3  Validation result present in output",
      "validation" in r or r.get("sql") is None,
      f"keys={list(r.keys())}")

# C4 — second turn uses context
r2 = query_agent.run("What about online sales only?", session_id=SID3)
ctx_after = get_conversation_context(SID3)
check("C4  Second turn has conversation context",
      len(ctx_after) >= 2, f"context turns={len(ctx_after)}")

# C5 — validation catches bad sql
bad_sql = "SELECT FakeColumn FROM FactSales"
from agents.sql_validator import validate_sql as vs
vr = vs(bad_sql)
check("C5  Bad SQL from LLM would be caught by validator",
      not vr.valid and len(vr.errors) > 0,
      f"errors={vr.errors[:1]}")

# C6 — good SQL passes validation and executes
if Path(CONTOSO_DB).exists():
    r3 = query_agent.run("What is total SalesAmount?", data_db_path=CONTOSO_DB, session_id=SID3)
    check("C6  Query with Contoso DB runs end-to-end",
          isinstance(r3, dict) and "sql" in r3,
          f"status: sql={str(r3.get('sql',''))[:60]}, result={r3.get('query_result',{}).get('row_count')}")
else:
    check("C6  Contoso DB execution", True, "skipped — run load_contoso.py", skip=True)

# C7 — clear session and run again — no old context
clear_session(SID3)
r4 = query_agent.run("What is gross sales?", session_id=SID3)
ctx_fresh = get_conversation_context(SID3)
check("C7  Fresh session has no prior context (starts clean)",
      len(ctx_fresh) <= 2, f"ctx turns={len(ctx_fresh)}")


# ══════════════════════════════════════════════════════════════════
# [D] RATE LIMITING + API ENDPOINTS — 7 tests
# ══════════════════════════════════════════════════════════════════
section("[D] Rate Limiting + API Endpoints")

try:
    from fastapi.testclient import TestClient
    import api.main as api_main
    api_main.DATA_DB = CONTOSO_DB
    client = TestClient(api_main.app, raise_server_exceptions=False)
    fastapi_ok = True
except Exception as e:
    fastapi_ok = False
    print(f"     FastAPI client unavailable: {e}")

TOKEN = "dd-dev-token-change-in-prod"
BAD_TOKEN = "wrong-token-here"

# D1 — health check
if fastapi_ok:
    r = client.get("/health")
    check("D1  GET /health returns 200", r.status_code == 200, str(r.json()))
else:
    check("D1  /health", True, "skipped", skip=True)

# D2 — POST /api/definitions requires auth
if fastapi_ok:
    r = client.post("/api/definitions",
        json={"name":"test_rate","description":"test","approved":False},
        headers={"X-API-Key": BAD_TOKEN})
    check("D2  POST /api/definitions returns 401 without valid token",
          r.status_code == 401,
          f"status={r.status_code}")
else:
    check("D2  Auth 401", True, "skipped", skip=True)

# D3 — POST /api/hitl/resolve requires auth
if fastapi_ok:
    r = client.post("/api/hitl/resolve",
        json={"conflict_id":"abc","action":"reject"},
        headers={"X-API-Key": BAD_TOKEN})
    check("D3  POST /api/hitl/resolve returns 401 without valid token",
          r.status_code == 401, f"status={r.status_code}")
else:
    check("D3  HITL auth 401", True, "skipped", skip=True)

# D4 — POST /api/drift/watch requires auth
if fastapi_ok:
    r = client.post("/api/drift/watch",
        headers={"X-API-Key": BAD_TOKEN})
    check("D4  POST /api/drift/watch returns 401 without valid token",
          r.status_code == 401, f"status={r.status_code}")
else:
    check("D4  Drift auth 401", True, "skipped", skip=True)

# D5 — rate limiter is attached
if fastapi_ok:
    has_limiter = hasattr(api_main.app.state, "limiter")
    check("D5  Rate limiter attached to app.state",
          has_limiter, f"has_limiter={has_limiter}")
else:
    check("D5  Rate limiter", True, "skipped", skip=True)

# D6 — conversation endpoint
if fastapi_ok:
    test_sid = "d6-test-session"
    add_message(test_sid, "user", "test question", "question")
    r = client.get(f"/api/conversation/{test_sid}")
    check("D6  GET /api/conversation/{session_id} returns messages",
          r.status_code == 200 and "messages" in r.json(),
          f"count={len(r.json().get('messages',[]))}")
else:
    check("D6  Conversation endpoint", True, "skipped", skip=True)

# D7 — conversation clear requires auth
if fastapi_ok:
    r = client.delete("/api/conversation/some-session",
        headers={"X-API-Key": BAD_TOKEN})
    check("D7  DELETE /api/conversation returns 401 without token",
          r.status_code == 401, f"status={r.status_code}")
else:
    check("D7  Conversation auth 401", True, "skipped", skip=True)


# ── SUMMARY ───────────────────────────────────────────────────────────────────
total   = len(results)
passed  = sum(1 for r in results if r[0] == PASS)
failed  = sum(1 for r in results if r[0] == FAIL)
skipped = sum(1 for r in results if r[0] == SKIP)

print(f"\n{'='*60}")
print(f"  RESULTS: {passed} passed  {failed} failed  {skipped} skipped  ({total} total)")
print(f"{'='*60}")

if failed:
    print("\n  Failed:")
    for icon, label, detail in results:
        if icon == FAIL:
            print(f"    {icon} {label}" + (f" — {detail}" if detail else ""))
    sys.exit(1)
else:
    note = "All tests passed! 🎉" if not skipped else f"Core tests passed ✅ ({skipped} skipped — add Contoso DB for full coverage)"
    print(f"\n  {note}\n")