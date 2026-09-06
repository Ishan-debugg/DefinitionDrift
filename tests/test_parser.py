"""
Quick smoke test for _parse_llm_response() — the new robust JSON extractor.
Tests all 6 response formats the function is designed to handle.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agents.core import _parse_llm_response

cases = [
    (
        "1. Clean JSON (happy path)",
        '{"sql":"SELECT 1","used_definitions":[],"confidence":"high","explanation":"test","warning":null}',
        "SELECT 1",
    ),
    (
        "2. JSON inside ```json ... ``` fence",
        '```json\n{"sql":"SELECT 2","used_definitions":[],"confidence":"high","explanation":"e","warning":null}\n```',
        "SELECT 2",
    ),
    (
        "3. JSON inside plain ``` fence with trailing prose",
        '```\n{"sql":"SELECT 3","used_definitions":[],"confidence":"high","explanation":"e","warning":null}\n```\nSome trailing text.',
        "SELECT 3",
    ),
    (
        "4. JSON object buried inside prose",
        'Here is your answer: {"sql":"SELECT 4","used_definitions":[],"confidence":"high","explanation":"e","warning":null} Done.',
        "SELECT 4",
    ),
    (
        "5. Pure prose with embedded SELECT statement",
        "Net revenue is SalesAmount minus ReturnAmount.\n\nSELECT SUM(SalesAmount - ReturnAmount) FROM FactSales\n\nHope that helps!",
        "SELECT SUM(SalesAmount - ReturnAmount) FROM FactSales",
    ),
    (
        "6. Total failure — no JSON, no SELECT",
        "I cannot generate SQL for this question.",
        None,
    ),
]

all_pass = True
print("\n=== _parse_llm_response() smoke test ===\n")
for label, raw, expected_sql in cases:
    r = _parse_llm_response(raw)
    got = r.get("sql")
    if expected_sql is None:
        ok = got is None
    else:
        ok = got is not None and expected_sql.strip() in got.strip()
    icon = "✅" if ok else "❌"
    print(f"  {icon}  {label}")
    if not ok:
        print(f"       expected sql = {repr(expected_sql)}")
        print(f"       got     sql = {repr(got)}")
        all_pass = False

print()
if all_pass:
    print("  All 6 parser cases passed! ✅\n")
else:
    print("  FAILURES detected ❌\n")
    sys.exit(1)
