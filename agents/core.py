"""
agents/core.py  (Phase 3 — multi-turn memory + SQL validation)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Four agents + two new capabilities:
  - Multi-turn conversation memory (session-scoped)
  - SQL validation via sqlglot AST + Contoso schema check
  - Dynamic schema introspection (zero-definition mode)
"""

import json, sqlite3, re, os, time
from typing import Optional
from pathlib import Path

from store.db import (
    get_all_definitions, enqueue_conflict,
    save_schema_snapshot, log_drift,
)
from store.conversation import get_conversation_context, add_message
from embeddings.engine import embed, cosine_similarity
from agents.llm_router import call_llm
from agents.sql_validator import validate_sql
from db.connection import get_engine, execute_query, inspect_schema, resolve_url
from config import settings


# ── Fix #2: Definitions cache with TTL ────────────────────────────────────────
_defs_cache: dict = {"key": None, "data": None, "ts": 0}
_DEFS_CACHE_TTL = 5  # seconds

def _get_definitions_cached(approved_only: bool = True) -> list[dict]:
    """Return definitions with a 5-second TTL cache to avoid redundant DB reads."""
    now = time.time()
    key = f"approved_{approved_only}"
    if _defs_cache["key"] == key and now - _defs_cache["ts"] < _DEFS_CACHE_TTL:
        return _defs_cache["data"]
    data = get_all_definitions(approved_only=approved_only)
    _defs_cache.update({"key": key, "data": data, "ts": now})
    return data


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

# Keyword synonyms: map natural-language terms → definition names they relate to.
# When a question contains these terms, the corresponding definition gets a score
# boost so that semantic near-misses still rank correctly.
_KEYWORD_ALIASES: dict[str, list[str]] = {
    "return":     ["return_rate", "net_revenue"],
    "returned":   ["return_rate"],
    "sent back":  ["return_rate"],
    "refund":     ["net_revenue", "return_rate"],
    "refunds":    ["net_revenue", "return_rate"],
    "margin":     ["gross_margin", "gross_margin_pct"],
    "profit":     ["gross_margin", "gross_margin_pct"],
    "gross":      ["gross_sales", "gross_margin", "gross_margin_pct"],
    "deduction":  ["net_revenue", "gross_sales"],
    "deductions": ["net_revenue", "gross_sales"],
    "net":        ["net_revenue"],
    "revenue":    ["net_revenue", "gross_sales"],
    "sales":      ["gross_sales", "net_revenue"],
    "rate":       ["return_rate"],
    "online":     ["online_units_sold"],
    "units":      ["units_sold", "online_units_sold", "return_rate"],
    "aov":        ["avg_order_value"],
    "average":    ["avg_order_value"],
    "transaction":["avg_order_value"],
}

KEYWORD_BOOST = 0.15  # score bonus when question keywords match a definition


class TokenOptimizer:
    SIMILARITY_THRESHOLD = settings.OPTIMIZER_SIMILARITY_THRESHOLD  # 0.45 from config
    TOP_K = settings.OPTIMIZER_TOP_K                                # 4 from config

    def __init__(self):
        self._def_cache: dict[str, list[float]] = {}  # def_id → vector

    def _get_def_vec(self, d: dict) -> list[float]:
        """Return cached embedding vector for a definition, computing on first access.
        
        Uses an enriched text that includes the definition name, description, and tags
        to give the embedding model more semantic surface area.
        """
        key = f"{d['id']}v{d['version']}"
        if key not in self._def_cache:
            # Enrich: include tags for broader semantic coverage
            tags_str = ""
            if d.get("tags"):
                try:
                    import json as _json
                    tags = _json.loads(d["tags"]) if isinstance(d["tags"], str) else d["tags"]
                    tags_str = " ".join(tags)
                except Exception:
                    pass
            embed_text = f"{d['name']} {d['description']} {tags_str}".strip()
            vec, _ = embed(embed_text)
            self._def_cache[key] = vec
        return self._def_cache[key]

    def _keyword_boost(self, question: str, def_name: str) -> float:
        """Return a score bonus if the question contains keywords associated with this definition."""
        q_lower = question.lower()
        for keyword, def_names in _KEYWORD_ALIASES.items():
            if keyword in q_lower and def_name in def_names:
                return KEYWORD_BOOST
        return 0.0

    def select_relevant(self, question: str, approved_only: bool = True,
                        q_vec: Optional[list[float]] = None) -> list[dict]:
        """Return top-K definitions above the similarity threshold, using cached vectors.
        
        Scoring = cosine_similarity + keyword_boost (capped at 1.0).
        The keyword boost ensures that questions containing obvious domain terms
        (e.g. 'return rate') always surface the correct definition even when the
        embedding similarity is borderline.
        
        Args:
            q_vec: Pre-computed question embedding (Fix #3 — avoids double embedding).
        """
        all_defs = _get_definitions_cached(approved_only=approved_only)
        if not all_defs:
            return []
        if q_vec is None:
            q_vec, _ = embed(question)
        scored = []
        for d in all_defs:
            base_score = cosine_similarity(q_vec, self._get_def_vec(d))
            boost = self._keyword_boost(question, d["name"])
            final_score = min(1.0, base_score + boost)
            scored.append((final_score, d))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for score, d in scored[:self.TOP_K] if score >= self.SIMILARITY_THRESHOLD]

    def build_context_block(self, question: str,
                            q_vec: Optional[list[float]] = None) -> tuple[str, list[dict]]:
        """Build context block with relevant definitions.
        
        Args:
            q_vec: Pre-computed question embedding (Fix #3).
        """
        relevant = self.select_relevant(question, q_vec=q_vec)
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


# ── SCHEMA INTROSPECTOR ──────────────────────────────────────────────────────
class SchemaIntrospector:
    """Reads the real database schema and builds a dynamic system prompt section.

    This allows the QueryAgent to answer ANY question about the database
    without requiring pre-defined metric definitions — the LLM sees the
    actual tables, columns, types, and sample data.
    """

    def __init__(self):
        self._schema_block: str = ""
        self._tables: dict[str, list[dict]] = {}

    def introspect(self, db_url: str) -> str:
        """Introspect the database and cache the schema block."""
        try:
            engine = get_engine(resolve_url(db_url))
            self._tables = inspect_schema(engine)
            self._schema_block = self._build_schema_block(engine)
            print(f"[SchemaIntrospector] Loaded {len(self._tables)} tables from {engine.dialect.name}")
        except Exception as e:
            print(f"[SchemaIntrospector] Failed to introspect: {e} — using hardcoded fallback")
            self._schema_block = ""  # will fall through to hardcoded
        return self._schema_block

    def _build_schema_block(self, engine) -> str:
        """Build a detailed schema description from introspected metadata.
        
        Fix #8: Skips per-table COUNT(*) queries — column list is sufficient
        for SQL generation and avoids N extra queries on startup.
        """
        lines = ["SCHEMA (auto-introspected from live database):"]
        for table_name, columns in sorted(self._tables.items()):
            if table_name in ("sqlite_sequence",):  # skip internal tables
                continue
            col_names = [c["name"] for c in columns]
            lines.append(f"  {table_name} — {', '.join(col_names)}")
        return "\n".join(lines)

    def get_schema_block(self) -> str:
        return self._schema_block

    def get_tables(self) -> dict[str, list[dict]]:
        return self._tables


schema_introspector = SchemaIntrospector()


# ── 2. CONFLICT AGENT ────────────────────────────────────────────────────────
class ConflictAgent:
    # Base threshold — intentionally conservative so DATA queries don't trigger.
    # IntentRouter overrides to 0.45 (DEFINE) / 0.65 (AMBIGUOUS) at runtime.
    CONFLICT_THRESHOLD = 0.50
    
    def check(self, question: str, q_vec: Optional[list[float]] = None) -> Optional[dict]:
        """Check if question conflicts with existing definitions.
        
        Args:
            q_vec: Pre-computed question embedding (Fix #3 — avoids double embedding).
        """
        all_defs = _get_definitions_cached(approved_only=True)
        if not all_defs:
            return None
        if q_vec is None:
            q_vec, _ = embed(question)
        # use optimizer's pre-computed vectors
        best_score, best_def = 0.0, None
        for d in all_defs:
            score = cosine_similarity(q_vec, optimizer._get_def_vec(d))  # ← reuse cache
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
        """
        Snapshot the schema of a data database and return change events.

        Accepts a SQLAlchemy connection URL **or** a bare SQLite file path
        (auto-converted via resolve_url for backward compatibility).

        Works with SQLite, PostgreSQL, and MySQL via SQLAlchemy Inspector.
        """
        events = []
        try:
            engine = get_engine(resolve_url(db_path))
            schema = inspect_schema(engine)  # {table: [col_info, ...]}

            for table, columns in schema.items():
                # Normalise to the same shape used by save_schema_snapshot
                current_cols = [
                    {"name": c["name"], "type": c["type"], "notnull": not c["nullable"]}
                    for c in columns
                ]
                changed, prev_cols = save_schema_snapshot(table, current_cols)
                if changed and prev_cols:
                    prev_names = {c["name"] for c in prev_cols}
                    curr_names = {c["name"] for c in current_cols}
                    for col in prev_names - curr_names:
                        affected = self._find_affected(col)
                        detail = f"Column '{col}' removed from '{table}'"
                        log_drift(table, "column_removed", detail, affected)
                        events.append({"type": "column_removed", "table": table,
                                       "column": col, "affected_definitions": affected,
                                       "detail": detail})
                    for col in curr_names - prev_names:
                        detail = f"Column '{col}' added to '{table}'"
                        log_drift(table, "column_added", detail)
                        events.append({"type": "column_added", "table": table,
                                       "column": col, "detail": detail})
        except Exception as e:
            events.append({"type": "error", "detail": str(e)})
        return events

    def _find_affected(self, column_name: str) -> Optional[str]:
        affected = [d["name"] for d in _get_definitions_cached()
                    if d.get("sql_expr") and column_name.lower() in d["sql_expr"].lower()]
        return json.dumps(affected) if affected else None

drift_watcher = DriftWatcher()


# ── 4. QUERY AGENT ───────────────────────────────────────────────────────────

# Hardcoded fallback schema — used when dynamic introspection hasn't run yet
_FALLBACK_SCHEMA = """\
SCHEMA (Contoso tables available):
  FactSales        — SalesKey, DateKey, StoreKey, ProductKey, CustomerKey,
                     ChannelKey, UnitCost, UnitPrice, SalesQuantity, ReturnQuantity,
                     ReturnAmount, DiscountAmount, TotalCost, SalesAmount, Margin
  FactOnlineSales  — OnlineSalesKey, DateKey, StoreKey, ProductKey, CustomerKey,
                     PromotionKey, UnitCost, UnitPrice, SalesQuantity, ReturnQuantity,
                     ReturnAmount, DiscountAmount, TotalCost, SalesAmount, Margin
  DimProduct       — ProductKey, ProductName, ProductLabel, ProductDescription,
                     ProductSubcategoryKey, BrandName, UnitCost, UnitPrice, Status
  DimStore         — StoreKey, StoreName, StoreType, StoreManager, StorePhone,
                     SellingAreaSize, OpenDate, Status, GeographyKey
  DimCustomer      — CustomerKey, FirstName, LastName, BirthDate, MaritalStatus,
                     Gender, EmailAddress, AnnualIncome, TotalChildren,
                     EducationLevel, Occupation, HouseOwnerFlag, CustomerType, GeographyKey
  DimDate          — DateKey (YYYYMMDD int), FullDateLabel, CalendarYear, CalendarQuarter,
                     CalendarMonth, CalendarWeek, DayNumberOfWeek, DayNameOfWeek,
                     IsWeekend, FiscalYear, FiscalQuarter, FiscalMonth
  DimProductCategory      — ProductCategoryKey, ProductCategoryName, ProductCategoryLabel, Description
  DimProductSubcategory   — ProductSubcategoryKey, ProductSubcategoryName, ProductCategoryKey
  DimGeography    — GeographyKey, GeographyType, ContinentName, CityName, StateName, RegionCountryName"""

_SYSTEM_PROMPT_TEMPLATE = """\
You are a data analyst for a Contoso Retail SQLite database.

{schema_block}

RULES (strict):
1. If approved metric definitions are provided, use their SQL EXACTLY — never rewrite.
2. If the question maps to an approved definition, embed its SQL as a CTE.
3. If NO definitions are provided, generate SQL directly from the SCHEMA above.
   You do NOT need definitions to answer — the schema is sufficient.
4. Never use column or table names not listed in SCHEMA above.
5. Use YYYYMMDD integer format for DateKey comparisons (e.g. 20080101).
6. Join DimDate on FactSales.DateKey = DimDate.DateKey for date filtering.
7. ALWAYS fully qualify column names with their table names
   (e.g., FactSales.StoreKey instead of StoreKey) to avoid ambiguous column errors.
8. Return ONLY this exact JSON — no markdown, no extra text:
{{
  "sql": "<valid SQLite SQL or null>",
  "used_definitions": ["names of definitions used, or empty list if none"],
  "confidence": "high|medium|low",
  "explanation": "one sentence describing what this measures",
  "warning": "<caveat or null>"
}}
9. temperature=0 — be deterministic. Same question must produce identical SQL.

## Examples

Q: What is net revenue for 2008?
A: {{"sql": "SELECT SUM(SalesAmount - ReturnAmount) FROM FactSales fs JOIN DimDate dd ON fs.DateKey = dd.DateKey WHERE dd.CalendarYear = 2008", "confidence": "high", "used_definitions": ["net_revenue"], "explanation": "Net revenue filters to 2008 via CalendarYear.", "warning": null}}

Q: What is the return rate across all channels?
A: {{"sql": "SELECT ROUND(SUM(ReturnQuantity)*100.0/NULLIF(SUM(SalesQuantity),0),2) FROM (SELECT ReturnQuantity, SalesQuantity FROM FactSales UNION ALL SELECT ReturnQuantity, SalesQuantity FROM FactOnlineSales)", "confidence": "high", "used_definitions": ["return_rate"], "explanation": "Combines store and online channels.", "warning": null}}
"""


class QueryAgent:

    def _build_system_prompt(self) -> str:
        """Build the system prompt with dynamic or fallback schema."""
        dynamic = schema_introspector.get_schema_block()
        schema_block = dynamic if dynamic else _FALLBACK_SCHEMA
        return _SYSTEM_PROMPT_TEMPLATE.format(schema_block=schema_block)

    def run(
        self,
        question: str,
        data_db_path: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> dict:
        # ── Ensure schema introspection has run ──────────────────────────────
        if not schema_introspector.get_schema_block() and data_db_path:
            schema_introspector.introspect(data_db_path)

        system_prompt = self._build_system_prompt()

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

        # ── Fix #3: Embed question once, reuse vector for conflict + optimizer ──
        q_vec, _ = embed(question)

        # ── Pass 2: inject relevant definitions (token optimizer) ─────────────
        context_block, used_defs = optimizer.build_context_block(question, q_vec=q_vec)

        # ── Build final prompt ────────────────────────────────────────────────
        parts = []
        if history_block:
            parts.append(history_block)
        if context_block:
            parts.append(context_block)
        else:
            # Zero-definition mode: tell the LLM it can answer from schema alone
            parts.append(
                "## Note\n"
                "No approved metric definitions matched this question. "
                "Generate SQL directly from the database schema provided above."
            )
        parts.append(f"## Current question\n{question}")
        user_msg = "\n\n".join(parts)

        # ── Pass 3: generate SQL via free LLM ─────────────────────────────────
        task = "multiturn" if history_block else "sql_generation"
        raw, provider = call_llm(
            system=system_prompt, user=user_msg,
            task=task, max_tokens=512,
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
                f"[QueryAgent] confidence=low — escalating via task routing"
            )
            raw2, provider2 = call_llm(
                system=system_prompt,
                user=user_msg,
                task="hitl_explain",
                max_tokens=settings.MAX_TOKENS_QUERY,
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

    def _execute(self, sql: str, db_url: str) -> dict:
        """
        Execute *sql* against the data database.

        Accepts a SQLAlchemy URL **or** a bare SQLite file path.
        Results are capped at 100 rows to avoid memory issues.
        """
        engine = get_engine(resolve_url(db_url))
        return execute_query(engine, sql, max_rows=100)

query_agent = QueryAgent()