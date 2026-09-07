"""
tests/test_phase5.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Phase 5 test suite — 38 tests

Sections:
  [A] Embedding engine upgrade    (7 tests)
  [B] Conflict detection P/R     (15 tests) — 10 TP + 10 TN, precision/recall
  [C] API auth — 401 on every    (16 tests)
      protected endpoint

Run:
    python tests/test_phase5.py

Notes:
  - sentence-transformers must be installed for [B] precision/recall.
    Without it, [B] tests are skipped with a clear message.
  - [C] requires httpx (pip install httpx) for the FastAPI test client.
    Without it, [C] tests are skipped.
"""

import sys, os, json, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Isolate DB ─────────────────────────────────────────────────────────────────
import store.db as db_module
_TMP = Path(tempfile.gettempdir())
TEST_DB = str(_TMP / "dd_phase5_test.db")
db_module.DB_PATH = Path(TEST_DB)
if Path(TEST_DB).exists():
    Path(TEST_DB).unlink()

from store.db import init_db, upsert_definition, get_all_definitions

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
print("  DefinitionDrift — Phase 5 Test Suite (38 tests)")
print("="*60)

init_db()

# Seed definitions used by conflict and API tests
for d in [
    dict(name="net_revenue",
         description="Total SalesAmount minus ReturnAmount across all channels",
         sql_expr="SELECT SUM(SalesAmount - ReturnAmount) FROM FactSales",
         approved=True),
    dict(name="gross_sales",
         description="Sum of SalesAmount across all channels before any deductions or returns",
         sql_expr="SELECT SUM(SalesAmount) FROM FactSales",
         approved=True),
    dict(name="return_rate",
         description="ReturnQuantity divided by SalesQuantity expressed as a percentage",
         sql_expr="SELECT ROUND(SUM(ReturnQuantity)*100.0/SUM(SalesQuantity),2) FROM FactSales",
         approved=True),
    dict(name="online_revenue",
         description="Total SalesAmount from online channel only",
         sql_expr="SELECT SUM(SalesAmount) FROM FactOnlineSales",
         approved=True),
    dict(name="active_stores",
         description="Number of stores with at least one sale",
         sql_expr="SELECT COUNT(DISTINCT StoreKey) FROM FactSales WHERE SalesAmount > 0",
         approved=True),
    dict(name="avg_transaction_value",
         description="Average SalesAmount per transaction",
         sql_expr="SELECT AVG(SalesAmount) FROM FactSales",
         approved=True),
    dict(name="units_sold",
         description="Total SalesQuantity of products sold across all channels",
         sql_expr="SELECT SUM(SalesQuantity) FROM FactSales",
         approved=True),
]:
    upsert_definition(reason="phase5 seed", **d)


# ══════════════════════════════════════════════════════════════════
# [A] EMBEDDING ENGINE UPGRADE — 7 tests
# ══════════════════════════════════════════════════════════════════
section("[A] Embedding Engine Upgrade")

from embeddings.engine import (
    embed, cosine_similarity, get_active_model,
    _ST_AVAILABLE, _sentence_transformers_installed,
)

# A1 — _sentence_transformers_installed() reflects reality
import importlib.util as _ilu
real_avail = _ilu.find_spec("sentence_transformers") is not None
check("A1  _sentence_transformers_installed() matches importlib.util",
      _sentence_transformers_installed() == real_avail,
      f"_ST_AVAILABLE={_ST_AVAILABLE}, real={real_avail}")

# A2 — _ST_AVAILABLE is a bool
check("A2  _ST_AVAILABLE is a bool constant",
      isinstance(_ST_AVAILABLE, bool), f"type={type(_ST_AVAILABLE).__name__}")

# A3 — get_active_model() returns one of the three known tiers
active_model = get_active_model()
check("A3  get_active_model() returns a recognised tier",
      active_model in ("all-MiniLM-L6-v2", "claude-haiku", "charfreq"),
      f"active={active_model}")

# A4 — if ST is available, active model must be all-MiniLM-L6-v2
if _ST_AVAILABLE:
    check("A4  sentence-transformers installed → active model is all-MiniLM-L6-v2",
          active_model == "all-MiniLM-L6-v2",
          f"active={active_model}")
else:
    check("A4  sentence-transformers not installed → active model is NOT all-MiniLM-L6-v2",
          active_model != "all-MiniLM-L6-v2",
          f"active={active_model} (install sentence-transformers for semantic quality)")

# A5 — embed() returns correct dimensions
vec, model = embed("What is net revenue?")
check("A5  embed() returns non-empty vector",
      isinstance(vec, list) and len(vec) > 0,
      f"dim={len(vec)}, model={model}")

# A6 — MiniLM produces 384-dim vectors; charfreq produces 64-dim
if _ST_AVAILABLE:
    check("A6  all-MiniLM-L6-v2 produces 384-dim vectors",
          len(vec) == 384, f"dim={len(vec)}")
else:
    check("A6  charfreq fallback produces 64-dim vectors",
          len(vec) == 64, f"dim={len(vec)}")

# A7 — cosine_similarity of identical embeddings = 1.0
vec2, _ = embed("What is net revenue?")
sim = cosine_similarity(vec, vec2)
check("A7  Cosine similarity of identical text = 1.0 (cache hit)",
      abs(sim - 1.0) < 1e-4, f"sim={sim:.6f}")


# ══════════════════════════════════════════════════════════════════
# [B] CONFLICT DETECTION PRECISION / RECALL — 15 tests
# Labeled dataset: 10 True Positives + 10 True Negatives
# ══════════════════════════════════════════════════════════════════
section("[B] Conflict Detection Precision / Recall")

from agents.core import conflict_agent

# Skip the full P/R evaluation when running without semantic embeddings —
# charfreq is known to have poor discrimination and would poison the metrics.
_USE_SEMANTIC = _ST_AVAILABLE

if not _USE_SEMANTIC:
    print(
        f"  {SKIP}  sentence-transformers not installed — P/R evaluation skipped.\n"
        f"       Install it for meaningful conflict detection:\n"
        f"           pip install sentence-transformers\n"
    )

# Ground-truth labels
# True  → question IS semantically close to a stored definition → should trigger
# False → question is unrelated → should NOT trigger
CONFLICT_PAIRS = [
    # ── TRUE POSITIVES (10) — paraphrase of stored definitions ──────────────
    ("Total sales minus total returns",                        True),   # ≈ net_revenue
    ("What is revenue after deducting returns?",               True),   # ≈ net_revenue
    ("Show me gross revenue before returns",                   True),   # ≈ gross_sales
    ("Sum of all sales amounts",                               True),   # ≈ gross_sales
    ("Return quantity over sales quantity times 100",          True),   # ≈ return_rate
    ("What percentage of sold items were returned?",           True),   # ≈ return_rate
    ("How much was sold through the web store?",               True),   # ≈ online_revenue
    ("Average dollar value per sale",                          True),   # ≈ avg_transaction_value
    ("How many items were sold in total?",                     True),   # ≈ units_sold
    ("Count of stores that made at least one sale",            True),   # ≈ active_stores

    # ── TRUE NEGATIVES (10) — completely unrelated to stored definitions ────
    ("What is the weather forecast for next week?",            False),
    ("List the top 5 customers by name",                       False),
    ("When is the next public holiday in the US?",             False),
    ("Which employee had the highest attendance this year?",   False),
    ("How do I reset my password?",                            False),
    ("Show me the inventory reorder schedule",                 False),
    ("What is our marketing budget for Q4?",                   False),
    ("List all products in the electronics category",          False),
    ("How many new customer accounts were created today?",     False),
    ("What is the average shipping time by carrier?",          False),
]

tp = fp = tn = fn = 0
cr_results = []

for i, (question, expected_conflict) in enumerate(CONFLICT_PAIRS, 1):
    if not _USE_SEMANTIC:
        # Register as skip but still iterate for completeness
        results.append((SKIP, f"CR-{i:02d}", "skipped — no sentence-transformers"))
        continue

    conflict_result = conflict_agent.check(question)
    predicted = conflict_result is not None

    if expected_conflict and predicted:
        outcome, tp = "TP", tp + 1
    elif not expected_conflict and not predicted:
        outcome, tn = "TN", tn + 1
    elif not expected_conflict and predicted:
        outcome, fp = "FP", fp + 1
    else:
        outcome, fn = "FN", fn + 1

    label_str = "CONFLICT" if expected_conflict else "CLEAN  "
    verdict   = PASS if outcome in ("TP", "TN") else FAIL
    sim_str   = f"sim={conflict_result['similarity']:.3f}" if conflict_result else "sim=none"
    cr_results.append({"outcome": outcome, "q": question, "sim_str": sim_str})
    print(f"  {verdict} [{outcome}] [{label_str}] {question[:48]}  ({sim_str})")

if _USE_SEMANTIC and cr_results:
    total_pairs = len(CONFLICT_PAIRS)
    precision   = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall      = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1          = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
    accuracy    = (tp + tn) / total_pairs

    print()
    print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"  Precision: {precision:.3f}   Recall: {recall:.3f}   F1: {f1:.3f}")
    print()

    check("B1  Recall >= 60% — system catches most real conflicts",
          recall >= 0.60, f"recall={recall:.2f}  (tp={tp}, fn={fn})")
    check("B2  Precision >= 60% — flagged items are real conflicts",
          precision >= 0.60 if (tp + fp) > 0 else True,
          f"precision={precision:.2f}  (tp={tp}, fp={fp})")
    check("B3  F1 >= 0.55 — balanced precision/recall",
          f1 >= 0.55, f"f1={f1:.3f}")
    check("B4  True negatives >= 6/10 — system doesn't over-trigger",
          tn >= 6, f"tn={tn}/10")
    check("B5  No crash across all 20 conflict checks",
          len(cr_results) == total_pairs, f"completed={len(cr_results)}/{total_pairs}")
else:
    for lbl in ["B1 Recall", "B2 Precision", "B3 F1", "B4 True negatives", "B5 No crash"]:
        check(lbl, True, "skipped — install sentence-transformers", skip=True)


# ══════════════════════════════════════════════════════════════════
# [C] API AUTH — 401 on every protected endpoint — 16 tests
# ══════════════════════════════════════════════════════════════════
section("[C] API Auth — 401 on Protected Endpoints")

VALID_TOKEN = "dd-dev-token-change-in-prod"
BAD_TOKEN   = "totally-wrong-key"
CONTOSO_DB  = str(Path(__file__).parent.parent / "data" / "contoso.db")

try:
    from fastapi.testclient import TestClient
    import api.main as api_main
    api_main.DATA_DB = CONTOSO_DB
    # Use raise_server_exceptions=False so 401s come back as HTTP responses
    client = TestClient(api_main.app, raise_server_exceptions=False)
    fastapi_ok = True
except Exception as e:
    fastapi_ok = False
    print(f"     FastAPI test client unavailable: {e}")
    print("     Install httpx:  pip install httpx")

def _skip_if_no_fastapi(label):
    check(label, True, "skipped — install httpx for FastAPI test client", skip=True)


# ── 401 with bad token ─────────────────────────────────────────────────────────

# C1 — POST /api/definitions (create)
if fastapi_ok:
    r = client.post("/api/definitions",
        json={"name": "test_auth", "description": "auth test"},
        headers={"X-API-Key": BAD_TOKEN})
    check("C1  POST /api/definitions returns 401 with bad token",
          r.status_code == 401, f"status={r.status_code}")
else:
    _skip_if_no_fastapi("C1  POST /api/definitions 401")

# C2 — POST /api/definitions without any token header
if fastapi_ok:
    r = client.post("/api/definitions",
        json={"name": "test_auth_no_key", "description": "no key"})
    check("C2  POST /api/definitions returns 401 with no token",
          r.status_code == 401, f"status={r.status_code}")
else:
    _skip_if_no_fastapi("C2  POST /api/definitions 401 (no header)")

# C3 — DELETE /api/definitions/{name}
if fastapi_ok:
    r = client.delete("/api/definitions/net_revenue",
        headers={"X-API-Key": BAD_TOKEN})
    check("C3  DELETE /api/definitions/{name} returns 401 with bad token",
          r.status_code == 401, f"status={r.status_code}")
else:
    _skip_if_no_fastapi("C3  DELETE /api/definitions 401")

# C4 — DELETE /api/definitions/{name} without any token header
if fastapi_ok:
    r = client.delete("/api/definitions/net_revenue")
    check("C4  DELETE /api/definitions/{name} returns 401 with no token",
          r.status_code == 401, f"status={r.status_code}")
else:
    _skip_if_no_fastapi("C4  DELETE /api/definitions 401 (no header)")

# C5 — POST /api/hitl/resolve with bad token
if fastapi_ok:
    r = client.post("/api/hitl/resolve",
        json={"conflict_id": "fake-id", "action": "reject"},
        headers={"X-API-Key": BAD_TOKEN})
    check("C5  POST /api/hitl/resolve returns 401 with bad token",
          r.status_code == 401, f"status={r.status_code}")
else:
    _skip_if_no_fastapi("C5  POST /api/hitl/resolve 401")

# C6 — POST /api/hitl/resolve without any token
if fastapi_ok:
    r = client.post("/api/hitl/resolve",
        json={"conflict_id": "fake-id", "action": "reject"})
    check("C6  POST /api/hitl/resolve returns 401 with no token",
          r.status_code == 401, f"status={r.status_code}")
else:
    _skip_if_no_fastapi("C6  POST /api/hitl/resolve 401 (no header)")

# C7 — POST /api/drift/watch with bad token
if fastapi_ok:
    r = client.post("/api/drift/watch",
        headers={"X-API-Key": BAD_TOKEN})
    check("C7  POST /api/drift/watch returns 401 with bad token",
          r.status_code == 401, f"status={r.status_code}")
else:
    _skip_if_no_fastapi("C7  POST /api/drift/watch 401")

# C8 — POST /api/drift/watch without any token
if fastapi_ok:
    r = client.post("/api/drift/watch")
    check("C8  POST /api/drift/watch returns 401 with no token",
          r.status_code == 401, f"status={r.status_code}")
else:
    _skip_if_no_fastapi("C8  POST /api/drift/watch 401 (no header)")

# C9 — DELETE /api/conversation/{session_id} with bad token
if fastapi_ok:
    r = client.delete("/api/conversation/some-session",
        headers={"X-API-Key": BAD_TOKEN})
    check("C9  DELETE /api/conversation returns 401 with bad token",
          r.status_code == 401, f"status={r.status_code}")
else:
    _skip_if_no_fastapi("C9  DELETE /api/conversation 401")

# C10 — DELETE /api/conversation/{session_id} without any token
if fastapi_ok:
    r = client.delete("/api/conversation/some-session")
    check("C10 DELETE /api/conversation returns 401 with no token",
          r.status_code == 401, f"status={r.status_code}")
else:
    _skip_if_no_fastapi("C10 DELETE /api/conversation 401 (no header)")

# ── Valid token lets write operations through ──────────────────────────────────

# C11 — POST /api/definitions succeeds with valid token
if fastapi_ok:
    r = client.post("/api/definitions",
        json={"name": "auth_test_valid", "description": "valid token test",
              "approved": False, "reason": "phase5 auth test"},
        headers={"X-API-Key": VALID_TOKEN})
    check("C11 POST /api/definitions returns 200 with valid token",
          r.status_code == 200 and r.json().get("status") == "ok",
          f"status={r.status_code}, body_status={r.json().get('status')}")
else:
    _skip_if_no_fastapi("C11 POST /api/definitions 200 (valid token)")

# C12 — DELETE /api/definitions/{name} succeeds with valid token
if fastapi_ok:
    r = client.delete("/api/definitions/auth_test_valid",
        headers={"X-API-Key": VALID_TOKEN})
    check("C12 DELETE /api/definitions returns 200 with valid token",
          r.status_code == 200, f"status={r.status_code}")
else:
    _skip_if_no_fastapi("C12 DELETE /api/definitions 200 (valid token)")

# ── Public (unauthenticated) endpoints should still return 200 ─────────────────

# C13 — GET /health (public)
if fastapi_ok:
    r = client.get("/health")
    check("C13 GET /health returns 200 (public endpoint)",
          r.status_code == 200, f"status={r.status_code}")
else:
    _skip_if_no_fastapi("C13 GET /health 200")

# C14 — GET /api/definitions (public read)
if fastapi_ok:
    r = client.get("/api/definitions")
    check("C14 GET /api/definitions returns 200 (public read endpoint)",
          r.status_code == 200 and "definitions" in r.json(),
          f"status={r.status_code}")
else:
    _skip_if_no_fastapi("C14 GET /api/definitions 200 (public)")

# C15 — GET /api/hitl/queue (public read)
if fastapi_ok:
    r = client.get("/api/hitl/queue")
    check("C15 GET /api/hitl/queue returns 200 (public read endpoint)",
          r.status_code == 200 and "pending_count" in r.json(),
          f"status={r.status_code}")
else:
    _skip_if_no_fastapi("C15 GET /api/hitl/queue 200 (public)")

# C16 — GET /api/stats (public)
if fastapi_ok:
    r = client.get("/api/stats")
    check("C16 GET /api/stats returns 200 and includes embedding_model field",
          r.status_code == 200 and "embedding_model" in r.json(),
          f"status={r.status_code}, has_embedding_model={'embedding_model' in r.json()}")
else:
    _skip_if_no_fastapi("C16 GET /api/stats 200 (public)")


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
        "All Phase 5 tests passed!"
        if not skipped
        else f"Core tests passed ({skipped} skipped)"
    )
    print(f"\n  {note}\n")
