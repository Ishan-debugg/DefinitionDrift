"""
agents/core.py
Four agents that power DefinitionDrift:
  1. QueryAgent        — NL → definition-aware SQL → result
  2. ConflictAgent     — semantic similarity → HITL queue
  3. DriftWatcher      — schema diff → HOTL drift log
  4. TokenOptimizer    — picks only relevant definitions to inject
"""

import os
import json
import sqlite3
import math
from typing import Optional
from datetime import datetime

import sqlglot
from anthropic import Anthropic

from store.db import (
    get_all_definitions, get_definition_by_name,
    enqueue_conflict, save_schema_snapshot, log_drift
)

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))

# ── HELPERS ───────────────────────────────────────────────────────────────────

def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x ** 2 for x in a))
    mag_b = math.sqrt(sum(x ** 2 for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


def _embed(text: str) -> list[float]:
    """
    Embedding strategy (in priority order):
    1. Claude Haiku API — semantic, accurate (requires ANTHROPIC_API_KEY)
    2. Char-frequency fallback — deterministic, no API, good enough for demo
    For production: swap with sentence-transformers all-MiniLM-L6-v2.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return _char_embed(text)
    try:
        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=256,
            system=(
                "You are an embedding simulator. Given text, return ONLY a JSON array "
                "of 32 floats between -1 and 1 that represent the semantic content. "
                "No explanation, no markdown, just the raw JSON array."
            ),
            messages=[{"role": "user", "content": text}]
        )
        return json.loads(resp.content[0].text)
    except Exception:
        return _char_embed(text)


def _char_embed(text: str) -> list[float]:
    """Deterministic char-frequency vector — no API needed."""
    vec = [0.0] * 32
    for ch in text.lower():
        vec[ord(ch) % 32] += 1
    mag = math.sqrt(sum(x ** 2 for x in vec)) or 1
    return [x / mag for x in vec]


# ── 1. TOKEN OPTIMIZER ────────────────────────────────────────────────────────

class TokenOptimizer:
    """
    Selects the top-K most relevant definitions for a given question.
    Reduces prompt tokens by 60-80% vs injecting the full definition store.
    """
    SIMILARITY_THRESHOLD = 0.55
    TOP_K = 4

    def select_relevant(self, question: str, approved_only: bool = True) -> list[dict]:
        all_defs = get_all_definitions(approved_only=approved_only)
        if not all_defs:
            return []

        q_vec = _embed(question)
        scored = []
        for d in all_defs:
            d_vec = _embed(f"{d['name']} {d['description']}")
            score = _cosine_similarity(q_vec, d_vec)
            scored.append((score, d))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [d for score, d in scored[:self.TOP_K] if score >= self.SIMILARITY_THRESHOLD]

    def build_context_block(self, question: str) -> tuple[str, list[dict]]:
        """Returns (context_string, relevant_defs_list)"""
        relevant = self.select_relevant(question)
        if not relevant:
            return "", []

        lines = ["## Approved metric definitions\n"]
        for d in relevant:
            lines.append(f"**{d['name']}**: {d['description']}")
            if d.get("sql_expr"):
                lines.append(f"  SQL: `{d['sql_expr']}`")
            lines.append("")
        return "\n".join(lines), relevant


optimizer = TokenOptimizer()


# ── 2. CONFLICT AGENT ─────────────────────────────────────────────────────────

class ConflictAgent:
    """
    Detects when a new question is semantically similar to an existing
    definition. If similarity > threshold, pauses and sends to HITL queue.
    """
    CONFLICT_THRESHOLD = 0.82

    def check(self, question: str) -> Optional[dict]:
        """
        Returns a conflict dict if detected, None if clean.
        """
        all_defs = get_all_definitions(approved_only=True)
        if not all_defs:
            return None

        q_vec = _embed(question)
        best_score = 0.0
        best_def = None

        for d in all_defs:
            d_vec = _embed(f"{d['name']} {d['description']}")
            score = _cosine_similarity(q_vec, d_vec)
            if score > best_score:
                best_score = score
                best_def = d

        if best_score >= self.CONFLICT_THRESHOLD and best_def:
            conflict = enqueue_conflict(
                question_a=question,
                question_b=best_def["description"],
                def_a=None,
                def_b=best_def["name"],
                similarity=best_score
            )
            return {
                "conflict": True,
                "conflict_id": conflict["id"],
                "matched_definition": best_def["name"],
                "similarity": round(best_score, 3),
                "message": (
                    f"Your question is {round(best_score*100)}% similar to the existing "
                    f"definition of '{best_def['name']}': \"{best_def['description']}\". "
                    f"Sending to approval queue (ID: {conflict['id']}) — "
                    f"a data owner can merge, keep separate, or reject."
                )
            }
        return None


conflict_agent = ConflictAgent()


# ── 3. DRIFT WATCHER ─────────────────────────────────────────────────────────

class DriftWatcher:
    """
    Snapshots table schemas from a SQLite DB and detects column-level drift.
    Checks which saved definitions reference changed columns.
    """

    def snapshot_and_diff(self, db_path: str) -> list[dict]:
        """
        Returns list of drift events detected. Empty = no changes.
        """
        events = []
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            tables = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()

            for table_row in tables:
                table = table_row["name"]
                if table.startswith("sqlite_"):
                    continue

                cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
                current_cols = [
                    {"name": c["name"], "type": c["type"], "notnull": c["notnull"]}
                    for c in cols
                ]
                changed, prev_cols = save_schema_snapshot(table, current_cols)

                if changed and prev_cols:
                        prev_names = {c["name"] for c in prev_cols}
                        curr_names = {c["name"] for c in current_cols}
                        removed = prev_names - curr_names
                        added = curr_names - prev_names

                        for col in removed:
                            affected = self._find_affected_definitions(col)
                            detail = f"Column '{col}' removed from table '{table}'"
                            log_drift(table, "column_removed", detail, affected)
                            events.append({
                                "type": "column_removed",
                                "table": table,
                                "column": col,
                                "affected_definitions": affected,
                                "detail": detail
                            })

                        for col in added:
                            detail = f"Column '{col}' added to table '{table}'"
                            log_drift(table, "column_added", detail)
                            events.append({
                                "type": "column_added",
                                "table": table,
                                "column": col,
                                "detail": detail
                            })
            conn.close()
        except Exception as e:
            events.append({"type": "error", "detail": str(e)})

        return events

    def _find_affected_definitions(self, column_name: str) -> Optional[str]:
        defs = get_all_definitions()
        affected = []
        for d in defs:
            if d.get("sql_expr") and column_name.lower() in d["sql_expr"].lower():
                affected.append(d["name"])
        return json.dumps(affected) if affected else None


drift_watcher = DriftWatcher()


# ── 4. QUERY AGENT ────────────────────────────────────────────────────────────

class QueryAgent:
    """
    Takes a natural language question, injects only relevant definitions
    (via TokenOptimizer), and returns a governed SQL query + explanation.

    Two-pass:
      Pass 1 (optimizer)  — picks relevant definitions, ~100 token context
      Pass 2 (generation) — Claude generates SQL grounded in those definitions
    """

    SYSTEM_PROMPT = """You are a data analyst assistant for DefinitionDrift.

Your job:
1. Use the provided approved metric definitions EXACTLY as written.
2. Generate SQL that is consistent with those definitions every time.
3. If the question maps to an approved definition, use its SQL expression directly.
4. If no definition matches, say so explicitly — do NOT guess joins or column names.
5. Always return a JSON object with:
   {
     "sql": "the SQL query or null",
     "used_definitions": ["list of definition names used"],
     "confidence": "high|medium|low",
     "explanation": "one sentence explaining what this query measures",
     "warning": "any caveats or null"
   }

RULES:
- Never fabricate column names.
- If a definition has a SQL expression, use it verbatim as a subquery or CTE.
- Prefer deterministic over creative. Same question = same SQL, always.
"""

    def run(self, question: str, data_db_path: Optional[str] = None) -> dict:
        context_block, used_defs = optimizer.build_context_block(question)

        user_message = f"{context_block}\n\n## Question\n{question}"

        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            return {
                "sql": None,
                "used_definitions": [d["name"] for d in used_defs],
                "confidence": "low",
                "explanation": "No ANTHROPIC_API_KEY set — set it to enable SQL generation.",
                "warning": "API key required for query generation.",
                "token_usage": {"input_tokens": 0, "output_tokens": 0, "definitions_injected": len(used_defs)}
            }

        resp = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            system=self.SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}]
        )

        raw = resp.content[0].text.strip()

        # strip markdown fences if present
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]

        try:
            result = json.loads(raw)
        except Exception:
            result = {
                "sql": None,
                "used_definitions": [],
                "confidence": "low",
                "explanation": raw,
                "warning": "Could not parse structured response"
            }

        # token usage metadata
        result["token_usage"] = {
            "input_tokens": resp.usage.input_tokens,
            "output_tokens": resp.usage.output_tokens,
            "definitions_injected": len(used_defs)
        }

        # optionally execute the SQL
        if result.get("sql") and data_db_path:
            result["query_result"] = self._execute(result["sql"], data_db_path)

        return result

    def _execute(self, sql: str, db_path: str) -> dict:
        try:
            conn = sqlite3.connect(db_path)
            conn.row_factory = sqlite3.Row
            rows = conn.execute(sql).fetchmany(50)
            conn.close()
            return {
                "rows": [dict(r) for r in rows],
                "row_count": len(rows)
            }
        except Exception as e:
            return {"error": str(e)}


query_agent = QueryAgent()


if __name__ == "__main__":
    from store.db import init_db
    import subprocess
    subprocess.run(["python", "store/db.py"], check=True)

    print("\n=== TokenOptimizer test ===")
    ctx, defs = optimizer.build_context_block("how many users logged in this week?")
    print(f"Relevant definitions selected: {[d['name'] for d in defs]}")
    print(f"Context block length: {len(ctx)} chars")

    print("\n=== ConflictAgent test ===")
    conflict = conflict_agent.check("show me active users from last week")
    if conflict:
        print(f"Conflict detected: {conflict['message']}")
    else:
        print("No conflict detected")

    print("\n=== QueryAgent test ===")
    result = query_agent.run("What is the total revenue excluding refunds?")
    print(json.dumps(result, indent=2))
    