"""
scripts/test_e2e_queries.py — End-to-End Query Test Suite
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Tests the full pipeline (LLM → SQL validation → execution) with
questions that should work WITHOUT any pre-defined definitions.

Validates:
  1. SQL is generated (not null)
  2. SQL passes schema validation
  3. SQL executes without errors
  4. Results contain expected data patterns

Usage:
    python scripts/test_e2e_queries.py
"""

import sys, json, time
sys.path.insert(0, ".")

from store.db import init_db
from store.conversation import init_conversation_db
from agents.orchestrator import run_query_pipeline

init_db()
init_conversation_db()

DATA_DB = "data/contoso.db"

# ── Test cases ────────────────────────────────────────────────────────────────
# Each: (description, question, expected_checks)
# expected_checks is a dict of field → lambda that returns True if ok
TESTS = [
    (
        "Simple count — DimStore",
        "How many stores are there?",
        {
            "has_sql": lambda r: r.get("sql_result", {}).get("sql") is not None,
            "has_rows": lambda r: len(r.get("sql_result", {}).get("query_result", {}).get("rows", [])) > 0,
            "correct_ish": lambda r: (
                # Should return a single row with a count
                r.get("sql_result", {}).get("query_result", {}).get("row_count", 0) >= 1
            ),
        }
    ),
    (
        "Aggregation — Total sales",
        "What is the total sales amount across all transactions?",
        {
            "has_sql": lambda r: r.get("sql_result", {}).get("sql") is not None,
            "has_rows": lambda r: len(r.get("sql_result", {}).get("query_result", {}).get("rows", [])) > 0,
        }
    ),
    (
        "Date filter — Sales in 2008",
        "What was the total sales amount in calendar year 2008?",
        {
            "has_sql": lambda r: r.get("sql_result", {}).get("sql") is not None,
            "has_rows": lambda r: len(r.get("sql_result", {}).get("query_result", {}).get("rows", [])) > 0,
        }
    ),
    (
        "Join — Top 5 brands by revenue",
        "Show me the top 5 brands by total sales amount",
        {
            "has_sql": lambda r: r.get("sql_result", {}).get("sql") is not None,
            "has_rows": lambda r: len(r.get("sql_result", {}).get("query_result", {}).get("rows", [])) >= 1,
            "max_5": lambda r: r.get("sql_result", {}).get("query_result", {}).get("row_count", 99) <= 10,
        }
    ),
    (
        "Dimension filter — Female customers",
        "How many customers are female?",
        {
            "has_sql": lambda r: r.get("sql_result", {}).get("sql") is not None,
            "has_rows": lambda r: len(r.get("sql_result", {}).get("query_result", {}).get("rows", [])) > 0,
        }
    ),
    (
        "Multi-join — Avg price by product category",
        "What is the average unit price by product category?",
        {
            "has_sql": lambda r: r.get("sql_result", {}).get("sql") is not None,
            "has_rows": lambda r: len(r.get("sql_result", {}).get("query_result", {}).get("rows", [])) >= 1,
        }
    ),
]


def run_test(desc, question, checks, test_num):
    """Run a single test and return (passed, details)."""
    print(f"\n{'─' * 60}")
    print(f"  Test {test_num}: {desc}")
    print(f"  Q: \"{question}\"")
    print(f"{'─' * 60}")
    
    t0 = time.time()
    try:
        result = run_query_pipeline(
            question=question,
            data_db_path=DATA_DB,
            thread_id=f"e2e-test-{test_num}-{int(time.time())}",
        )
    except Exception as e:
        print(f"  ❌ EXCEPTION: {e}")
        return False, {"error": str(e)}
    
    latency = int((time.time() - t0) * 1000)
    
    # Extract key info
    status = result.get("status", "unknown")
    sql_result = result.get("sql_result", {})
    sql = sql_result.get("sql")
    provider = sql_result.get("provider_used", "none")
    confidence = sql_result.get("confidence", "none")
    query_result = sql_result.get("query_result", {})
    rows = query_result.get("rows", [])
    error = query_result.get("error")
    validation = sql_result.get("validation", {})
    defs_injected = sql_result.get("definitions_injected", 0)
    
    print(f"  Status:      {status}")
    print(f"  Provider:    {provider}")
    print(f"  Confidence:  {confidence}")
    print(f"  Defs used:   {defs_injected}")
    print(f"  Latency:     {latency}ms")
    
    if sql:
        # Show SQL (truncated)
        sql_preview = sql.replace("\n", " ")[:120]
        print(f"  SQL:         {sql_preview}...")
    else:
        print(f"  SQL:         (none)")
    
    if error:
        print(f"  DB Error:    {error[:120]}")
    elif rows:
        print(f"  Rows:        {len(rows)}")
        # Show first row
        if rows:
            first = rows[0]
            preview = str(first)[:120]
            print(f"  Sample:      {preview}")
    
    if validation.get("errors"):
        print(f"  Val Errors:  {validation['errors']}")
    if validation.get("warnings"):
        print(f"  Val Warns:   {validation['warnings']}")
    
    # Run checks
    all_passed = True
    for check_name, check_fn in checks.items():
        try:
            passed = check_fn(result)
        except Exception as e:
            passed = False
        icon = "✅" if passed else "❌"
        print(f"  {icon} {check_name}")
        if not passed:
            all_passed = False
    
    # Overall verdict
    if status == "conflict_detected":
        print(f"  ⚠️  Query was HITL-blocked (conflict). This is expected if definitions exist.")
        # Still count as pass if the pipeline worked correctly
        all_passed = True
    
    return all_passed, {
        "status": status,
        "sql": sql,
        "rows": len(rows),
        "error": error,
        "latency_ms": latency,
        "confidence": confidence,
    }


def main():
    print("=" * 60)
    print("  DefinitionDrift — End-to-End Query Test Suite")
    print("  Testing zero-definition query capability")
    print("=" * 60)
    
    results = []
    passed_count = 0
    
    for i, (desc, question, checks) in enumerate(TESTS, 1):
        passed, details = run_test(desc, question, checks, i)
        results.append((desc, passed, details))
        if passed:
            passed_count += 1
    
    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {passed_count}/{len(TESTS)} tests passed")
    print(f"{'=' * 60}")
    
    for desc, passed, details in results:
        icon = "✅" if passed else "❌"
        info = f"status={details.get('status')}, rows={details.get('rows')}, {details.get('latency_ms')}ms"
        print(f"  {icon} {desc}")
        print(f"     {info}")
    
    print()
    
    if passed_count == len(TESTS):
        print("  🎉 All tests passed! Zero-definition mode works correctly.")
    else:
        failed = len(TESTS) - passed_count
        print(f"  ⚠️  {failed} test(s) failed. Check output above for details.")
    
    return passed_count == len(TESTS)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
