"""
scripts/fix_bad_defs.py — One-time migration to fix broken seed definitions
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Problem:
  The original seed definitions reference tables/columns that don't exist
  in Contoso (e.g. `orders`, `sessions`). When these get injected into the
  LLM prompt, they actively mislead the model.

What this does:
  1. Marks existing broken definitions as unapproved (soft delete)
  2. Creates correct Contoso-compatible definitions with proper SQL expressions

Run once:
    python scripts/fix_bad_defs.py
"""

import sys
sys.path.insert(0, ".")

from store.db import init_db, get_all_definitions, upsert_definition

init_db()

print("=" * 60)
print("  DefinitionDrift — Fix Bad Seed Definitions")
print("=" * 60)

# ── Step 1: Show current state ────────────────────────────────────────────────
all_defs = get_all_definitions()
print(f"\n[Before] Total definitions: {len(all_defs)}")
for d in all_defs:
    status = "✅ approved" if d["approved"] else "⬜ draft"
    sql = d.get("sql_expr", "None")
    sql_preview = sql[:60] if sql else "None"
    print(f"  [{d['id'][:8]}] {d['name']:20s} {status}  sql={sql_preview}")

# ── Step 2: Unapprove broken definitions ──────────────────────────────────────
broken_names = []
for d in all_defs:
    sql = d.get("sql_expr", "") or ""
    # Flag definitions whose SQL references tables NOT in Contoso
    has_bad_table = any(t in sql.lower() for t in ["orders", "sessions", "users", "total_amount"])
    # Flag definitions with no SQL that are approved (useless context injection)
    is_empty_approved = d["approved"] and not sql.strip()
    
    if has_bad_table or is_empty_approved:
        broken_names.append(d["name"])
        upsert_definition(
            name=d["name"],
            description=d["description"],
            sql_expr=d.get("sql_expr"),
            tags=[],
            approved=False,
            reason="auto-unapproved: references non-Contoso tables or has no SQL"
        )
        print(f"\n  ❌ Unapproved '{d['name']}' — ", end="")
        if has_bad_table:
            print("SQL references non-existent tables")
        elif is_empty_approved:
            print("approved but has no SQL expression")

if not broken_names:
    print("\n  ✅ No broken definitions found!")

# ── Step 3: Create correct Contoso-compatible definitions ─────────────────────
print("\n\n[Creating] Correct Contoso definitions...\n")

correct_defs = [
    {
        "name": "gross_sales",
        "description": "Total gross sales amount across all in-store transactions",
        "sql_expr": "SUM(FactSales.SalesAmount)",
        "tags": ["finance", "sales"],
        "approved": True,
        "reason": "Contoso-compatible definition for gross sales"
    },
    {
        "name": "net_revenue",
        "description": "Net revenue after subtracting returns and discounts from gross sales",
        "sql_expr": "SUM(FactSales.SalesAmount - FactSales.ReturnAmount - FactSales.DiscountAmount)",
        "tags": ["finance", "revenue"],
        "approved": True,
        "reason": "Contoso-compatible definition for net revenue"
    },
    {
        "name": "total_margin",
        "description": "Total profit margin across all in-store sales",
        "sql_expr": "SUM(FactSales.Margin)",
        "tags": ["finance", "profitability"],
        "approved": True,
        "reason": "Contoso-compatible definition for margin"
    },
    {
        "name": "total_customers",
        "description": "Count of distinct customers in the customer dimension table",
        "sql_expr": "COUNT(DISTINCT DimCustomer.CustomerKey)",
        "tags": ["customers"],
        "approved": True,
        "reason": "Contoso-compatible definition for customer count"
    },
    {
        "name": "return_rate",
        "description": "Percentage of units returned out of total units sold (in-store)",
        "sql_expr": "ROUND(100.0 * SUM(FactSales.ReturnQuantity) / NULLIF(SUM(FactSales.SalesQuantity), 0), 2)",
        "tags": ["operations", "returns"],
        "approved": True,
        "reason": "Contoso-compatible definition for return rate"
    },
    {
        "name": "avg_order_value",
        "description": "Average sales amount per transaction across in-store sales",
        "sql_expr": "AVG(FactSales.SalesAmount)",
        "tags": ["finance", "sales"],
        "approved": True,
        "reason": "Contoso-compatible definition for average order value"
    },
]

for d in correct_defs:
    result = upsert_definition(**d)
    print(f"  ✅ Created/Updated '{result['name']}' v{result['version']} (approved={result['approved']})")
    print(f"     SQL: {d['sql_expr']}")

# ── Step 4: Final state ──────────────────────────────────────────────────────
print(f"\n{'=' * 60}")
all_defs = get_all_definitions()
approved = [d for d in all_defs if d["approved"]]
print(f"[After] Total: {len(all_defs)}, Approved: {len(approved)}")
for d in all_defs:
    status = "✅" if d["approved"] else "⬜"
    print(f"  {status} {d['name']:25s} v{d['version']}")
print(f"\n{'=' * 60}")
print("Done! Broken definitions unapproved, correct definitions created.")
