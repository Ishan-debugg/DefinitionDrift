"""
agents/core.py  (Phase 3 — multi-turn memory + SQL validation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Four agents + two new capabilities:
  - Multi-turn conversation memory (session-scoped)
  - SQL validation via sqlglot AST + Contoso schema check
"""

import json, sqlite3, re
from typing import Optional

from store.db import (
    get_all_definitions, enqueue_conflict,
    save_schema_snapshot, log_drift,
)
from store.conversation import get_conversation_context, add_message
from embeddings.engine import embed, cosine_similarity
from agents.llm_router import call_llm
from agents.sql_validator import validate_sql
from config import settings


# ── LLM response parser ──────────────────────────────────────────────────────
def _parse_llm_response(raw: str) -> dict:
    """
    Robustly extract a structured result dict from whatever the LLM returned.

    Handles (in order):
      1. Clean JSON string  — happy path
      2. JSON inside ```json ... ``` fence
      3. JSON inside ``` ... ``` fence
      4. JSON object buried anywhere inside prose
      5. Prose with a SELECT statement — extract SQL via regex
      6. Total failure fallback — sql=None, raw text → explanation
    """
    text = raw.strip()

    # ── 1. Direct JSON parse ──────────────────────────────────────────────────
    try:
        return json.loads(text)
    except Exception:
        pass

    # ── 2 & 3. JSON inside markdown fences (```json or ```) ──────────────────
    fence_match = re.search(r'```(?:json)?\s*\n(.*?)\n```', text, re.DOTALL)
    if fence_match:
        inner = fence_match.group(1).strip()
        try:
            return json.loads(inner)
        except Exception:
            pass

    # ── 4. JSON object embedded somewhere in prose ────────────────────────────
    # Finds the first {...} block that spans multiple keys — avoids false matches
    json_match = re.search(r'(\{[^{}]*"sql"[^{}]*\})', text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except Exception:
            pass

    # ── 5. Prose response — extract SQL via regex ─────────────────────────────
    # Matches SELECT ... until a blank line or end of string
    sql_match = re.search(
        r'\b(SELECT[\s\S]+?)(?:\n{2,}|$)',
        text, re.IGNORECASE
    )
    extracted_sql = sql_match.group(1).strip() if sql_match else None

    return {
        "sql": extracted_sql,
        "used_definitions": [],
        "confidence": "low",
        "explanation": text[:400],
        "warning": (
            "SQL extracted from prose — LLM ignored JSON instruction. "
            "Add a valid GROQ_API_KEY for reliable structured output."
            if extracted_sql
            else "Could not parse LLM response. Check API keys in .env."
        ),
    }


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

    def run(
        self,
        question: str,
        data_db_path: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> dict:
        # ── Pass 1: build conversation history block ──────────────────────────
        history_block = ""
        if session_id:
            history = get_conversation_context(session_id, max_turns=6)
            if history:
                lines = ["## Conversation history (most recent last)"]
                for m in history:
                    role_label = "User" if m["role"] == "user" else "Assistant"
                    lines.append(f"{role_label}: {m['content']}")
                lines.append("")
                history_block = "\n".join(lines)

        # ── Pass 2: inject relevant definitions (token optimizer) ─────────────
        context_block, used_defs = optimizer.build_context_block(question)

        # ── Build final prompt ────────────────────────────────────────────────
        parts = []
        if history_block:
            parts.append(history_block)
        if context_block:
            parts.append(context_block)
        parts.append(f"## Current question\n{question}")
        user_msg = "\n\n".join(parts)

        # ── Pass 3: generate SQL via free LLM ─────────────────────────────────
        raw, provider = call_llm(
            system=self.SYSTEM_PROMPT, user=user_msg,
            task="sql_generation", max_tokens=512,
        )

        result = _parse_llm_response(raw)

        result["provider_used"] = provider
        result["definitions_injected"] = len(used_defs)
        result["escalated"] = False  # default; set True if we escalate below

        # ── Confidence-based model escalation ────────────────────────────────
        # When the fast model admits low confidence, re-run with the smart model.
        # LOW_CONFIDENCE_THRESHOLD = 0.60 (see config/settings.py), but the LLM
        # reports categorical "low"|"medium"|"high" — we escalate on "low".
        if result.get("confidence") == "low":
            print(
                f"[QueryAgent] confidence=low — escalating to {settings.QUERY_MODEL_SMART}"
            )
            raw2, provider2 = call_llm(
                system=self.SYSTEM_PROMPT,
                user=user_msg,
                task="sql_generation_smart",
                max_tokens=settings.MAX_TOKENS_QUERY,
                model_override=settings.QUERY_MODEL_SMART,
            )
            result2 = _parse_llm_response(raw2)
            result2["provider_used"] = provider2
            result2["definitions_injected"] = len(used_defs)
            result2["escalated"] = True
            result2["escalated_from_provider"] = provider
            # Keep the escalated result only if it actually produced SQL or higher confidence
            if result2.get("sql") or result2.get("confidence", "low") != "low":
                result = result2

        # ── Pass 4: SQL validation (AST parse + schema check) ─────────────────
        if result.get("sql"):
            validation = validate_sql(result["sql"])
            result["validation"] = validation.to_dict()

            if not validation.valid:
                # Surface validation errors to user — do NOT execute bad SQL
                result["warning"] = (
                    (result.get("warning") or "") +
                    " | Validation errors: " + "; ".join(validation.errors)
                ).strip(" |")
                result["confidence"] = "low"
                # Use reformatted SQL if parse succeeded (schema error only)
                if validation.parse_ok and validation.fixed_sql:
                    result["sql_original"] = result["sql"]
                    result["sql"] = validation.fixed_sql
                # Do not execute if schema errors exist
                if not validation.schema_ok:
                    result["query_result"] = {
                        "error": "SQL blocked by schema validation: " + "; ".join(validation.errors),
                        "validation_errors": validation.errors,
                    }
                    self._save_to_memory(session_id, question, result)
                    return result
            else:
                # Use prettier reformatted SQL
                if validation.fixed_sql:
                    result["sql"] = validation.fixed_sql
                if validation.warnings:
                    result["validation_warnings"] = validation.warnings

        # ── Pass 5: execute SQL ───────────────────────────────────────────────
        if result.get("sql") and data_db_path:
            result["query_result"] = self._execute(result["sql"], data_db_path)

        # ── Save to conversation memory ───────────────────────────────────────
        self._save_to_memory(session_id, question, result)
        return result

    def _save_to_memory(self, session_id: Optional[str], question: str, result: dict):
        """Persist turn to conversation memory for multi-turn context."""
        if not session_id:
            return
        try:
            add_message(session_id, "user", question, "question")
            explanation = result.get("explanation", "")
            add_message(
                session_id, "assistant", explanation, "sql_result",
                sql_result={
                    "sql": result.get("sql"),
                    "used_definitions": result.get("used_definitions", []),
                    "confidence": result.get("confidence"),
                }
            )
        except Exception:
            pass  # memory failure never blocks the query

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