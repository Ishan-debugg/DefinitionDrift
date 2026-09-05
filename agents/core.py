"""
agents/core.py  (Phase 2 — free provider edition)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Four agents — now using free LLM providers (Groq / Gemini / Cerebras)
instead of Anthropic. Zero ongoing cost.
"""

import os, json, sqlite3, math
from typing import Optional

from store.db import (
    get_all_definitions, enqueue_conflict,
    save_schema_snapshot, log_drift,
)
from embeddings.engine import embed, cosine_similarity
from agents.llm_router import call_llm


# ── 1. TOKEN OPTIMIZER ───────────────────────────────────────────────────────
class TokenOptimizer:
    SIMILARITY_THRESHOLD = 0.35
    TOP_K = 4

    def select_relevant(self, question: str, approved_only: bool = True) -> list[dict]:
        all_defs = get_all_definitions(approved_only=approved_only)
        if not all_defs:
            return []
        q_vec, _ = embed(question)
        scored = []
        for d in all_defs:
            d_vec, _ = embed(f"{d['name']} {d['description']}")
            score = cosine_similarity(q_vec, d_vec)
            scored.append((score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for score, d in scored[:self.TOP_K] if score >= self.SIMILARITY_THRESHOLD]

    def build_context_block(self, question: str) -> tuple[str, list[dict]]:
        relevant = self.select_relevant(question)
        if not relevant:
            return "", []
        lines = ["## Approved metric definitions (use these EXACTLY)\n"]
        for d in relevant:
            lines.append(f"**{d['name']}**: {d['description']}")
            if d.get("sql_expr"):
                lines.append(f"  SQL reference: `{d['sql_expr']}`")
            lines.append("")
        return "\n".join(lines), relevant

optimizer = TokenOptimizer()


# ── 2. CONFLICT AGENT ────────────────────────────────────────────────────────
class ConflictAgent:
    CONFLICT_THRESHOLD = 0.82

    def check(self, question: str) -> Optional[dict]:
        all_defs = get_all_definitions(approved_only=True)
        if not all_defs:
            return None
        q_vec, _ = embed(question)
        best_score, best_def = 0.0, None
        for d in all_defs:
            d_vec, _ = embed(f"{d['name']} {d['description']}")
            score = cosine_similarity(q_vec, d_vec)
            if score > best_score:
                best_score, best_def = score, d
        if best_score >= self.CONFLICT_THRESHOLD and best_def:
            conflict = enqueue_conflict(
                question_a=question, question_b=best_def["description"],
                def_a=None, def_b=best_def["name"], similarity=best_score,
            )
            return {
                "conflict": True,
                "conflict_id": conflict["id"],
                "matched_definition": best_def["name"],
                "similarity": round(best_score, 3),
                "message": (
                    f"Your question is {round(best_score*100)}% similar to "
                    f"'{best_def['name']}': \"{best_def['description']}\". "
                    f"Queued for human review (ID: {conflict['id']})."
                ),
            }
        return None

conflict_agent = ConflictAgent()


# ── 3. DRIFT WATCHER ─────────────────────────────────────────────────────────
class DriftWatcher:
    def snapshot_and_diff(self, db_path: str) -> list[dict]:
        events = []
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
            for table_row in tables:
                table = table_row["name"]
                if table.startswith("sqlite_"):
                    continue
                cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
                current_cols = [{"name": c["name"], "type": c["type"], "notnull": c["notnull"]} for c in cols]
                changed, prev_cols = save_schema_snapshot(table, current_cols)
                if changed and prev_cols:
                    prev_names = {c["name"] for c in prev_cols}
                    curr_names = {c["name"] for c in current_cols}
                    for col in prev_names - curr_names:
                        affected = self._find_affected(col)
                        detail = f"Column '{col}' removed from '{table}'"
                        log_drift(table, "column_removed", detail, affected)
                        events.append({"type": "column_removed", "table": table,
                                       "column": col, "affected_definitions": affected, "detail": detail})
                    for col in curr_names - prev_names:
                        detail = f"Column '{col}' added to '{table}'"
                        log_drift(table, "column_added", detail)
                        events.append({"type": "column_added", "table": table, "column": col, "detail": detail})
            conn.close()
        except Exception as e:
            events.append({"type": "error", "detail": str(e)})
        return events

    def _find_affected(self, column_name: str) -> Optional[str]:
        affected = [d["name"] for d in get_all_definitions()
                    if d.get("sql_expr") and column_name.lower() in d["sql_expr"].lower()]
        return json.dumps(affected) if affected else None

drift_watcher = DriftWatcher()


# ── 4. QUERY AGENT ───────────────────────────────────────────────────────────
class QueryAgent:
    SYSTEM_PROMPT = """\
You are a data analyst for a Contoso Retail SQLite database.

SCHEMA (Contoso tables available):
  FactSales        — SalesKey, DateKey, StoreKey, ProductKey, CustomerKey,
                     UnitCost, UnitPrice, SalesQuantity, ReturnQuantity,
                     ReturnAmount, DiscountAmount, TotalCost, SalesAmount, Margin
  FactOnlineSales  — same columns, online channel only (StoreKey=306)
  DimProduct       — ProductKey, ProductName, BrandName, UnitCost, UnitPrice, Status
  DimStore         — StoreKey, StoreName, StoreType, Status, GeographyKey
  DimCustomer      — CustomerKey, FirstName, LastName, AnnualIncome, Occupation, Gender
  DimDate          — DateKey (YYYYMMDD int), CalendarYear, CalendarMonth,
                     CalendarQuarter, FiscalYear, FiscalMonth, FiscalQuarter
  DimProductCategory      — ProductCategoryKey, ProductCategoryName
  DimProductSubcategory   — ProductSubcategoryKey, ProductSubcategoryName, ProductCategoryKey

RULES (strict):
1. Use approved metric definitions EXACTLY — never rewrite their SQL.
2. If the question maps to an approved definition, embed its SQL as a CTE.
3. Never use column or table names not listed in SCHEMA above.
4. Use YYYYMMDD integer format for DateKey comparisons (e.g. 20080101).
5. Join DimDate on FactSales.DateKey = DimDate.DateKey for date filtering.
6. Return ONLY this exact JSON — no markdown, no extra text:
{
  "sql": "<valid SQLite SQL or null>",
  "used_definitions": ["names of definitions used"],
  "confidence": "high|medium|low",
  "explanation": "one sentence describing what this measures",
  "warning": "<caveat or null>"
}
7. temperature=0 — be deterministic. Same question must produce identical SQL."""

    def run(self, question: str, data_db_path: Optional[str] = None) -> dict:
        context_block, used_defs = optimizer.build_context_block(question)
        user_msg = (f"{context_block}\n\n## Question\n{question}"
                    if context_block else f"## Question\n{question}")

        raw, provider = call_llm(
            system=self.SYSTEM_PROMPT, user=user_msg,
            task="sql_generation", max_tokens=512,
        )

        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1] if lines and lines[-1].strip() == "```" else lines[1:])

        try:
            result = json.loads(text)
        except Exception:
            result = {"sql": None, "used_definitions": [],
                      "confidence": "low", "explanation": text[:300],
                      "warning": "Could not parse LLM response as JSON"}

        result["provider_used"] = provider
        result["definitions_injected"] = len(used_defs)

        if result.get("sql") and data_db_path:
            result["query_result"] = self._execute(result["sql"], data_db_path)

        return result

    def _execute(self, sql: str, db_path: str) -> dict:
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql).fetchmany(100)
            conn.close()
            return {"rows": [dict(r) for r in rows], "row_count": len(rows)}
        except Exception as e:
            return {"error": str(e), "sql_attempted": sql}

query_agent = QueryAgent()