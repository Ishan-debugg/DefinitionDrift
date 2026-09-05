"""
tests/test_evals.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DefinitionDrift Evaluation Suite — 3 Benchmarks
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

[1] End-to-End SQL Accuracy   (20 questions, sqlglot AST comparison)
[2] Token Optimizer Accuracy  (10 questions, token reduction ≥ 50%)
[3] Conflict Detection P/R    (20 labeled pairs, precision & recall)

Run:
    python tests/test_evals.py

Requirements:
    pip install sqlglot
    GROQ_API_KEY or GEMINI_API_KEY in .env for live LLM calls
    sentence-transformers for best embedding quality
"""

import sys, os, json, time, math
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Use a fresh test DB so evals never pollute production ─────────────────────
import store.db as db_module
EVAL_DB = Path(__file__).parent.parent / "data" / "eval_test.db"
db_module.DB_PATH = EVAL_DB
if EVAL_DB.exists():
    EVAL_DB.unlink()

from store.db import init_db, upsert_definition, get_all_definitions
from agents.core import query_agent, conflict_agent, optimizer

import sqlglot

# ── Shared helpers ────────────────────────────────────────────────────────────
PASS, FAIL, SKIP, INFO = "✅", "❌", "⚠️ ", "ℹ️ "
results = []

def check(label, cond, detail="", skip=False):
    if skip:
        results.append((SKIP, label, "skipped"))
        print(f"  {SKIP} {label}" + (f" — {detail}" if detail else ""))
        return True
    icon = PASS if cond else FAIL
    results.append((icon, label, detail))
    print(f"  {icon} {label}" + (f" — {detail}" if detail else ""))
    return cond

def section(title):
    bar = "═" * 65
    print(f"\n{bar}\n  {title}\n{bar}")

def info(msg):
    print(f"  {INFO}  {msg}")

# ── Init ──────────────────────────────────────────────────────────────────────
init_db()
HAS_API = bool(os.getenv("GROQ_API_KEY") or os.getenv("GEMINI_API_KEY"))

print("\n" + "═" * 65)
print("  DefinitionDrift Evaluation Suite")
print(f"  LLM API available : {'✅ YES — live SQL generation' if HAS_API else '❌ NO  — set GROQ_API_KEY for live tests'}")
print("═" * 65)


# ══════════════════════════════════════════════════════════════════════════════
# SEED DEFINITIONS  (shared across all 3 eval sections)
# ══════════════════════════════════════════════════════════════════════════════
DEFINITIONS = [
    dict(
        name="net_revenue",
        description="Total SalesAmount minus ReturnAmount across all channels",
        sql_expr="SELECT SUM(SalesAmount - ReturnAmount) FROM FactSales",
        approved=True,
    ),
    dict(
        name="gross_sales",
        description="Sum of SalesAmount across all channels before any deductions or returns",
        sql_expr="SELECT SUM(SalesAmount) FROM FactSales",
        approved=True,
    ),
    dict(
        name="return_rate",
        description="ReturnQuantity divided by SalesQuantity expressed as a percentage",
        sql_expr="SELECT ROUND(SUM(ReturnQuantity)*100.0/SUM(SalesQuantity),2) FROM FactSales",
        approved=True,
    ),
    dict(
        name="online_revenue",
        description="Total SalesAmount from online channel only",
        sql_expr="SELECT SUM(SalesAmount) FROM FactOnlineSales",
        approved=True,
    ),
    dict(
        name="active_stores",
        description="Number of stores with at least one sale",
        sql_expr="SELECT COUNT(DISTINCT StoreKey) FROM FactSales WHERE SalesAmount > 0",
        approved=True,
    ),
    dict(
        name="avg_transaction_value",
        description="Average SalesAmount per transaction",
        sql_expr="SELECT AVG(SalesAmount) FROM FactSales",
        approved=True,
    ),
    dict(
        name="product_count",
        description="Total number of distinct products sold",
        sql_expr="SELECT COUNT(DISTINCT ProductKey) FROM FactSales",
        approved=True,
    ),
    dict(
        name="units_sold",
        description="Total SalesQuantity of products sold across all channels",
        sql_expr="SELECT SUM(SalesQuantity) FROM FactSales",
        approved=True,
    ),
]

for d in DEFINITIONS:
    upsert_definition(reason="eval seed", **d)

info(f"Seeded {len(DEFINITIONS)} approved definitions for evaluation")


# ══════════════════════════════════════════════════════════════════════════════
# [1]  END-TO-END SQL ACCURACY  (20 questions)
# ══════════════════════════════════════════════════════════════════════════════
section("[1] End-to-End SQL Accuracy — 20 Questions (sqlglot AST)")

# Each entry: (question, expected_canonical_sql or None)
# None → we only check structure (SQL is non-null and parseable), not exact AST
SQL_TEST_CASES = [
    # ── Direct definition matches ──────────────────────────────────────────
    ("What is the total net revenue?",
     "SELECT SUM(SalesAmount - ReturnAmount) FROM FactSales"),

    ("Show me gross sales across all channels",
     "SELECT SUM(SalesAmount) FROM FactSales"),

    ("What is the return rate?",
     "SELECT ROUND(SUM(ReturnQuantity)*100.0/SUM(SalesQuantity),2) FROM FactSales"),

    ("How much revenue came from the online channel?",
     "SELECT SUM(SalesAmount) FROM FactOnlineSales"),

    ("How many active stores do we have?",
     "SELECT COUNT(DISTINCT StoreKey) FROM FactSales WHERE SalesAmount > 0"),

    ("What is the average transaction value?",
     "SELECT AVG(SalesAmount) FROM FactSales"),

    ("How many distinct products have been sold?",
     "SELECT COUNT(DISTINCT ProductKey) FROM FactSales"),

    ("What are total units sold?",
     "SELECT SUM(SalesQuantity) FROM FactSales"),

    # ── Paraphrase matches (LLM should still use the definition SQL) ────────
    ("What is our revenue after subtracting returns?",
     "SELECT SUM(SalesAmount - ReturnAmount) FROM FactSales"),

    ("Total revenue before any refunds",
     "SELECT SUM(SalesAmount) FROM FactSales"),

    ("What fraction of items sold were returned, as a percent?",
     "SELECT ROUND(SUM(ReturnQuantity)*100.0/SUM(SalesQuantity),2) FROM FactSales"),

    ("Revenue generated through our e-commerce platform",
     "SELECT SUM(SalesAmount) FROM FactOnlineSales"),

    ("What is the mean sale amount per order?",
     "SELECT AVG(SalesAmount) FROM FactSales"),

    ("Count of unique products we have sold",
     "SELECT COUNT(DISTINCT ProductKey) FROM FactSales"),

    # ── Structure-only checks (no exact expected SQL) ──────────────────────
    # These test that the system produces valid, parseable SQL
    ("What were the top selling products last month?",         None),
    ("Compare online vs. store revenue",                       None),
    ("Which store had the highest return rate?",               None),
    ("What is revenue growth year over year?",                 None),
    ("Show me revenue broken down by product category",        None),
    ("How many transactions occurred per store?",              None),
]

def normalize_sql(sql: str) -> str:
    """Strip whitespace/casing for fuzzy match before AST compare."""
    return " ".join(sql.lower().split())

def ast_match(generated: str, expected: str) -> bool:
    """
    True if sqlglot ASTs are structurally equivalent.
    Falls back to normalized string match if parsing fails.
    """
    try:
        ast_gen = sqlglot.parse_one(generated)
        ast_exp = sqlglot.parse_one(expected)
        return ast_gen == ast_exp
    except Exception:
        return normalize_sql(generated) == normalize_sql(expected)

def sql_is_parseable(sql: str) -> bool:
    try:
        sqlglot.parse_one(sql)
        return True
    except Exception:
        return False

sql_results = []
sql_start   = time.time()

for i, (question, expected_sql) in enumerate(SQL_TEST_CASES, 1):
    result = query_agent.run(question)
    generated_sql = result.get("sql")
    confidence    = result.get("confidence", "unknown")

    if expected_sql is not None:
        # Exact check: AST must match canonical definition SQL
        if generated_sql:
            match = ast_match(generated_sql, expected_sql)
            status = "AST_MATCH" if match else "AST_MISMATCH"
        else:
            match = False
            status = "NO_SQL"
        sql_results.append({
            "q": i, "match": match, "status": status,
            "question": question, "expected": expected_sql,
            "generated": generated_sql, "confidence": confidence,
        })
        verdict = PASS if match else FAIL
        print(f"  {verdict} Q{i:02d} [{status}] {question[:55]}")
        if not match:
            print(f"       Expected : {expected_sql[:80]}")
            print(f"       Generated: {str(generated_sql)[:80]}")

    else:
        # Structure check only: must be non-null and parseable
        parseable = bool(generated_sql) and sql_is_parseable(generated_sql)
        sql_results.append({
            "q": i, "match": parseable, "status": "STRUCTURE_OK" if parseable else "STRUCTURE_FAIL",
            "question": question, "generated": generated_sql, "confidence": confidence,
        })
        verdict = PASS if parseable else FAIL
        print(f"  {verdict} Q{i:02d} [STRUCTURE] {question[:55]}")

sql_elapsed    = time.time() - sql_start
exact_cases    = [r for r in sql_results if "MISMATCH" in r["status"] or "MATCH" in r["status"]]
struct_cases   = [r for r in sql_results if "STRUCTURE" in r["status"]]
exact_passed   = sum(1 for r in exact_cases  if r["match"])
struct_passed  = sum(1 for r in struct_cases if r["match"])
total_passed   = sum(1 for r in sql_results  if r["match"])

no_sql_cases   = [r for r in sql_results if r.get("status") == "NO_SQL"]

print()
info(f"Exact AST matches : {exact_passed}/{len(exact_cases)}"
     + (" ← no API key, SQL not generated" if not HAS_API and exact_passed == 0 else ""))
info(f"Structure OK      : {struct_passed}/{len(struct_cases)}")
info(f"Overall accuracy  : {total_passed}/{len(sql_results)} ({total_passed/len(sql_results)*100:.1f}%)")
info(f"No-SQL responses  : {len(no_sql_cases)} (expected if no API key)")
info(f"Time              : {sql_elapsed:.1f}s")

check("SQL-1  Accuracy ≥ 60% on exact cases (requires API key)",
      exact_passed >= round(len(exact_cases) * 0.6) if HAS_API else True,
      f"{exact_passed}/{len(exact_cases)}" if HAS_API else "skipped — set GROQ_API_KEY",
      skip=not HAS_API)

check("SQL-2  Structure check: parseable SQL produced",
      struct_passed == len(struct_cases) if HAS_API else True,
      f"{struct_passed}/{len(struct_cases)}" if HAS_API else "skipped — no API key",
      skip=not HAS_API)

check("SQL-3  Pipeline completes all 20 questions without crash",
      len(sql_results) == 20, f"completed={len(sql_results)}/20")

check("SQL-4  Confidence field always present",
      all(r.get("confidence") for r in sql_results),
      "all responses have confidence")


# ══════════════════════════════════════════════════════════════════════════════
# [2]  TOKEN OPTIMIZER ACCURACY  (10 questions)
# ══════════════════════════════════════════════════════════════════════════════
section("[2] Token Optimizer Accuracy — 10 Questions")

# Build a "full context" (all definitions injected, no optimizer)
def full_context_tokens() -> int:
    """Approximate token count if we injected ALL definitions verbatim."""
    all_defs = get_all_definitions(approved_only=True)
    lines = ["## All metric definitions\n"]
    for d in all_defs:
        lines.append(f"**{d['name']}**: {d['description']}")
        if d.get("sql_expr"):
            lines.append(f"  SQL: `{d['sql_expr']}`")
        lines.append("")
    full_text = "\n".join(lines)
    # rough token estimate: 1 token ≈ 4 chars
    return len(full_text) // 4

OPTIMIZER_QUESTIONS = [
    # Expected relevant defs shown in comment
    ("What is net revenue after returns?",                  ["net_revenue"]),
    ("Show total gross sales",                              ["gross_sales"]),
    ("What is the return rate?",                            ["return_rate"]),
    ("How much did we sell online?",                        ["online_revenue"]),
    ("Count active stores this quarter",                    ["active_stores"]),
    ("Average order value?",                                ["avg_transaction_value"]),
    ("How many unique products sold?",                      ["product_count"]),
    ("Total units sold across all channels",                ["units_sold"]),
    ("Revenue minus refunds and returns",                   ["net_revenue"]),
    ("xyz_completely_unrelated_nonsense_444",               []),   # expect 0 defs
]

baseline_tokens = full_context_tokens()
opt_results = []

print()
info(f"Full context (all defs, no optimizer): ~{baseline_tokens} tokens")
print()

for i, (question, expected_defs) in enumerate(OPTIMIZER_QUESTIONS, 1):
    ctx, selected_defs = optimizer.build_context_block(question)
    selected_names = [d["name"] for d in selected_defs]

    # Token count for optimized context
    opt_tokens = len(ctx) // 4 if ctx else 0
    reduction_pct = ((baseline_tokens - opt_tokens) / baseline_tokens * 100) if baseline_tokens > 0 else 0

    # Relevance: did it pick at least one expected def?
    relevant_hit = (
        len(expected_defs) == 0 and len(selected_names) == 0   # correctly picked nothing
    ) or (
        len(expected_defs) > 0 and any(d in selected_names for d in expected_defs)
    )

    opt_results.append({
        "q": i, "question": question,
        "selected": selected_names, "expected": expected_defs,
        "opt_tokens": opt_tokens, "reduction_pct": reduction_pct,
        "relevant_hit": relevant_hit,
    })

    verdict = PASS if relevant_hit else FAIL
    hit_str = "✓ hit" if relevant_hit else "✗ miss"
    print(f"  {verdict} Q{i:02d} [{hit_str}] {question[:50]}")
    print(f"        Selected : {selected_names}")
    print(f"        Expected : {expected_defs}")
    print(f"        Tokens   : {opt_tokens} optimized vs {baseline_tokens} full  ({reduction_pct:.0f}% reduction)")

print()

# Metrics
relevance_hits     = sum(1 for r in opt_results if r["relevant_hit"])
avg_reduction      = sum(r["reduction_pct"] for r in opt_results) / len(opt_results)
min_reduction      = min(r["reduction_pct"] for r in opt_results)
over_50_pct_count  = sum(1 for r in opt_results if r["reduction_pct"] >= 50)
last_q_tokens      = opt_results[-1]["opt_tokens"]   # unrelated query

info(f"Relevance hits        : {relevance_hits}/{len(opt_results)}")
info(f"Average token reduction : {avg_reduction:.1f}%")
info(f"Min token reduction     : {min_reduction:.1f}%")
info(f"Queries with ≥50% cut   : {over_50_pct_count}/{len(opt_results)}")
info(f"Unrelated query tokens  : {last_q_tokens} (should be 0 or very low)")

check("OPT-1  Relevance: ≥70% questions get a correct definition hit",
      relevance_hits >= 7,
      f"{relevance_hits}/10")

check("OPT-2  Average token reduction ≥ 50%",
      avg_reduction >= 50.0,
      f"{avg_reduction:.1f}%")

check("OPT-3  ≥8 of 10 queries achieve ≥50% token reduction",
      over_50_pct_count >= 8,
      f"{over_50_pct_count}/10 queries hit ≥50% cut")

check("OPT-4  Unrelated query injects 0 or minimal tokens",
      last_q_tokens <= (baseline_tokens * 0.3),
      f"unrelated query injected {last_q_tokens} tokens (threshold={int(baseline_tokens*0.3)})")

check("OPT-5  Optimizer never selects more than TOP_K=4 defs per query",
      all(len(r["selected"]) <= 4 for r in opt_results),
      f"max_selected={max(len(r['selected']) for r in opt_results)}")


# ══════════════════════════════════════════════════════════════════════════════
# [3]  CONFLICT DETECTION PRECISION / RECALL  (20 labeled pairs)
# ══════════════════════════════════════════════════════════════════════════════
section("[3] Conflict Detection Precision & Recall — 20 Labeled Pairs")

# GROUND TRUTH LABELS
# Format: (question, label)
# label = True  → SHOULD trigger a conflict (semantically close to a stored def)
# label = False → should NOT trigger a conflict (genuinely different topic)

CONFLICT_PAIRS = [
    # ── TRUE POSITIVES (10) — very similar to stored definitions ───────────
    ("Total sales minus total returns",                        True),   # ≈ net_revenue
    ("What is revenue after deducting returns?",              True),   # ≈ net_revenue
    ("Show me gross revenue before returns",                  True),   # ≈ gross_sales
    ("Sum of all sales amounts",                              True),   # ≈ gross_sales
    ("Return quantity over sales quantity times 100",         True),   # ≈ return_rate
    ("What percentage of sold items were returned?",          True),   # ≈ return_rate
    ("How much was sold through the web store?",              True),   # ≈ online_revenue
    ("Average dollar value per sale",                         True),   # ≈ avg_transaction_value
    ("How many items were sold in total?",                    True),   # ≈ units_sold
    ("Count of stores that made at least one sale",           True),   # ≈ active_stores

    # ── TRUE NEGATIVES (10) — genuinely unrelated to any stored def ────────
    ("What is the weather forecast for next week?",           False),
    ("List the top 5 customers by name",                      False),
    ("When is the next public holiday?",                      False),
    ("Which employee had the highest attendance this year?",  False),
    ("How do I reset my password?",                           False),
    ("Show me the inventory reorder schedule",                False),
    ("What is our marketing budget for Q4?",                  False),
    ("List all products in the electronics category",         False),
    ("How many new customer accounts were created today?",    False),
    ("What is the average shipping time by carrier?",         False),
]

print()
info("Running 20 conflict checks (this may take 10-30s with API embeddings)...")
print()

tp = fp = tn = fn = 0
cr_results = []

for i, (question, expected_conflict) in enumerate(CONFLICT_PAIRS, 1):
    conflict_result = conflict_agent.check(question)
    predicted_conflict = conflict_result is not None

    if expected_conflict and predicted_conflict:
        outcome, tp = "TP", tp + 1
    elif not expected_conflict and not predicted_conflict:
        outcome, tn = "TN", tn + 1
    elif not expected_conflict and predicted_conflict:
        outcome, fp = "FP", fp + 1
    else:
        outcome, fn = "FN", fn + 1

    cr_results.append({
        "q": i, "question": question,
        "expected": expected_conflict, "predicted": predicted_conflict,
        "outcome": outcome,
        "similarity": round(conflict_result["similarity"], 3) if conflict_result else None,
    })

    label_str = "CONFLICT" if expected_conflict else "CLEAN  "
    verdict   = PASS if outcome in ("TP", "TN") else FAIL
    sim_str   = f"sim={conflict_result['similarity']:.3f}" if conflict_result else "sim=none"
    print(f"  {verdict} Q{i:02d} [{outcome}] [{label_str}] {question[:50]}  ({sim_str})")

print()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
accuracy  = (tp + tn) / len(CONFLICT_PAIRS)

info(f"True  Positives  : {tp}/10")
info(f"True  Negatives  : {tn}/10")
info(f"False Positives  : {fp}  (flagged clean questions as conflicts)")
info(f"False Negatives  : {fn}  (missed real conflicts)")
info(f"Precision        : {precision:.3f}  ({precision*100:.1f}%)")
info(f"Recall           : {recall:.3f}  ({recall*100:.1f}%)")
info(f"F1 Score         : {f1:.3f}")
info(f"Accuracy         : {accuracy:.3f}  ({accuracy*100:.1f}%)")
info("NOTE: char-frequency embeddings (no sentence-transformers) will score lower.")
info("      pip install sentence-transformers for semantic accuracy.")

check("CD-1  Recall ≥ 60% — system catches most real conflicts",
      recall >= 0.60,
      f"recall={recall:.2f}  (tp={tp}, fn={fn})")

check("CD-2  Precision ≥ 60% — most flagged items are real conflicts",
      precision >= 0.60 if (tp + fp) > 0 else True,
      f"precision={precision:.2f}  (tp={tp}, fp={fp})"
      if (tp + fp) > 0 else "no conflicts flagged — check embedding quality")

check("CD-3  F1 ≥ 0.55 — balanced precision/recall",
      f1 >= 0.55 if (tp + fp + fn) > 0 else True,
      f"f1={f1:.3f}")

check("CD-4  True negatives ≥ 6/10 — system doesn't over-trigger",
      tn >= 6,
      f"tn={tn}/10")

check("CD-5  No system crash across all 20 checks",
      len(cr_results) == 20, f"completed={len(cr_results)}/20")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
total   = len(results)
passed  = sum(1 for r in results if r[0] == PASS)
failed  = sum(1 for r in results if r[0] == FAIL)
skipped = sum(1 for r in results if r[0] == SKIP)

print("\n" + "═" * 65)
print(f"  EVALUATION RESULTS")
print("─" * 65)
print(f"  ✅  Passed  : {passed}")
print(f"  ❌  Failed  : {failed}")
print(f"  ⚠️   Skipped : {skipped}")
print(f"  Total      : {total}")
print("─" * 65)
print(f"  SQL Accuracy   — exact: {exact_passed}/{len(exact_cases)}, "
      f"structure: {struct_passed}/{len(struct_cases)}")
print(f"  Token Optimizer— avg reduction: {avg_reduction:.1f}%, "
      f"relevance hits: {relevance_hits}/10")
print(f"  Conflict P/R   — precision: {precision:.2f}, recall: {recall:.2f}, "
      f"F1: {f1:.2f}")
print("═" * 65)

if not HAS_API:
    print("""
  ⚠️  No LLM API key detected.
     SQL generation tests are skipped/degraded.
     Add GROQ_API_KEY to .env for full evaluation coverage.
     (Free at: console.groq.com — no credit card required)
""")

if failed > 0:
    print("\n  Failed checks:")
    for icon, label, detail in results:
        if icon == FAIL:
            print(f"    {icon} {label}" + (f" — {detail}" if detail else ""))
    sys.exit(1)
else:
    print(f"\n  {'🎉 All checks passed!' if not skipped else '✅ Core checks passed (some skipped — add API key for full run)'}\n")
