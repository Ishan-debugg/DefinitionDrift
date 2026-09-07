"""
scripts/eval.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Batch NL2SQL evaluation runner for DefinitionDrift.

Reads a JSONL file of {question, expected_sql} pairs, runs each through
the full pipeline (conflict check + SQL generation + optional execution),
and outputs per-pair JSONL results plus an aggregated summary.

Metrics computed:
  exact_match        — normalised SQL string equality (whitespace/case)
  token_f1           — sqlglot-tokenized set overlap F1
  exec_match         — result-set equality (requires --data-db)
  confidence         — model-reported confidence level
  escalated          — whether confidence escalation fired
  provider_used      — which LLM backend served the call
  latency_ms         — wall-clock time for the pipeline call
  input_tok          — LLM input tokens (from usage DB)
  output_tok         — LLM output tokens (from usage DB)

Usage:
  python scripts/eval.py --input scripts/sample_eval.jsonl
  python scripts/eval.py --input eval_set.jsonl --data-db data/contoso.db \\
      --output results/eval_$(date +%Y%m%d).jsonl --quiet
"""

import sys
import os
import argparse
import json
import time
import re
import sqlite3
from pathlib import Path
from typing import Optional

# ── Bootstrap project root ───────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from store.db import init_db
from agents.orchestrator import run_query_pipeline
from agents.llm_router import USAGE_DB


# ── SQL normalisation helpers ────────────────────────────────────────────────

def _normalise_sql(sql: str) -> str:
    """Lowercase, collapse whitespace, strip trailing semicolon."""
    if not sql:
        return ""
    s = sql.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(";").strip()
    return s


def _sql_tokens(sql: str) -> set[str]:
    """
    Tokenise SQL into a set of meaningful tokens using sqlglot when available,
    falling back to a simple regex split.
    """
    if not sql:
        return set()
    try:
        import sqlglot
        toks = sqlglot.tokenize(sql)
        return {t.text.lower() for t in toks
                if t.text.strip() and t.text not in ("(", ")", ",", ";")}
    except Exception:
        # simple fallback — split on non-word characters
        return {t.lower() for t in re.split(r"\W+", sql) if t}


def _token_f1(pred_sql: str, gold_sql: str) -> dict:
    pred_toks = _sql_tokens(pred_sql)
    gold_toks = _sql_tokens(gold_sql)
    if not pred_toks and not gold_toks:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_toks or not gold_toks:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    common = pred_toks & gold_toks
    precision = len(common) / len(pred_toks)
    recall    = len(common) / len(gold_toks)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4)}


# ── Execution match ──────────────────────────────────────────────────────────

def _execute_sql(sql: str, db_path: str) -> Optional[list]:
    """Run SQL against db_path; return sorted list of row tuples or None on error."""
    try:
        conn = sqlite3.connect(db_path)
        cur  = conn.execute(sql)
        rows = [tuple(r) for r in cur.fetchmany(200)]
        conn.close()
        return sorted(rows)
    except Exception as e:
        return None


def _exec_match(pred_sql: str, gold_sql: str, db_path: str) -> Optional[bool]:
    """Return True if both SQLs produce identical ordered result sets."""
    pred_rows = _execute_sql(pred_sql, db_path)
    gold_rows = _execute_sql(gold_sql, db_path)
    if pred_rows is None or gold_rows is None:
        return None   # execution error — can't compare
    return pred_rows == gold_rows


# ── Usage token lookup ───────────────────────────────────────────────────────

def _last_call_tokens() -> tuple[int, int]:
    """Return (input_tok, output_tok) from the most recent LLM call in usage DB."""
    try:
        conn = sqlite3.connect(USAGE_DB)
        row  = conn.execute(
            "SELECT input_tok, output_tok FROM llm_calls ORDER BY id DESC LIMIT 1"
        ).fetchone()
        conn.close()
        return (row[0] or 0, row[1] or 0) if row else (0, 0)
    except Exception:
        return (0, 0)


# ── Core evaluation loop ─────────────────────────────────────────────────────

def run_eval(pairs: list[dict], data_db: Optional[str], quiet: bool) -> list[dict]:
    results = []
    n = len(pairs)

    for i, pair in enumerate(pairs, 1):
        question    = pair.get("question", "").strip()
        expected    = pair.get("expected_sql", pair.get("gold_sql", "")).strip()
        pair_id     = pair.get("id", f"pair-{i:04d}")

        if not question:
            print(f"[{i}/{n}] SKIP — no question field", flush=True)
            continue

        if not quiet:
            print(f"[{i}/{n}] {question[:80]}...", flush=True)

        t0 = time.time()
        pipeline_result = run_query_pipeline(
            question=question,
            data_db_path=data_db,
            thread_id=f"eval-{pair_id}",
        )
        latency_ms = int((time.time() - t0) * 1000)

        sql_result  = pipeline_result.get("sql_result") or {}
        pred_sql    = sql_result.get("sql") or ""
        confidence  = sql_result.get("confidence", "unknown")
        escalated   = sql_result.get("escalated", False)
        provider    = sql_result.get("provider_used", "unknown")
        status      = pipeline_result.get("status", "unknown")

        # Metrics
        exact = _normalise_sql(pred_sql) == _normalise_sql(expected)
        tf1   = _token_f1(pred_sql, expected)
        exec_m = None
        if data_db and pred_sql and expected:
            exec_m = _exec_match(pred_sql, expected, data_db)

        # Token usage from LLM tracker
        in_tok, out_tok = _last_call_tokens()

        row = {
            "id":           pair_id,
            "question":     question,
            "expected_sql": expected,
            "predicted_sql": pred_sql,
            "status":        status,
            "exact_match":   exact,
            "token_precision": tf1["precision"],
            "token_recall":    tf1["recall"],
            "token_f1":        tf1["f1"],
            "exec_match":      exec_m,
            "confidence":      confidence,
            "escalated":       escalated,
            "provider_used":   provider,
            "latency_ms":      latency_ms,
            "input_tok":       in_tok,
            "output_tok":      out_tok,
        }

        if not quiet:
            em_str  = "✓" if exact else "✗"
            xm_str  = ("✓" if exec_m else "✗") if exec_m is not None else "-"
            esc_str = " [ESC]" if escalated else ""
            print(
                f"     exact={em_str}  exec={xm_str}  f1={tf1['f1']:.2f}"
                f"  conf={confidence}{esc_str}  {latency_ms}ms",
                flush=True,
            )

        results.append(row)

    return results


# ── Summary report ───────────────────────────────────────────────────────────

def print_summary(results: list[dict]):
    n = len(results)
    if n == 0:
        print("No results to summarise.")
        return

    exact_n   = sum(1 for r in results if r["exact_match"])
    exec_n    = sum(1 for r in results if r["exec_match"] is True)
    exec_eval = sum(1 for r in results if r["exec_match"] is not None)
    avg_f1    = sum(r["token_f1"] for r in results) / n
    low_conf  = sum(1 for r in results if r["confidence"] == "low")
    escalated = sum(1 for r in results if r["escalated"])
    avg_lat   = sum(r["latency_ms"] for r in results) / n
    total_tok = sum(r["input_tok"] + r["output_tok"] for r in results)

    print()
    print("── Eval Summary " + "─" * 44)
    print(f"  Pairs evaluated  : {n}")
    print(f"  Exact match      : {exact_n} / {n}  ({100*exact_n/n:.1f}%)")
    print(f"  Token F1 (avg)   : {avg_f1:.3f}")
    if exec_eval:
        print(f"  Exec match       : {exec_n} / {exec_eval}  ({100*exec_n/exec_eval:.1f}%)")
    else:
        print(f"  Exec match       : n/a  (no --data-db provided)")
    print(f"  Low-confidence   : {low_conf} / {n}")
    print(f"  Escalated        : {escalated} / {n}")
    print(f"  Avg latency      : {avg_lat:.0f} ms")
    print(f"  Total tokens     : {total_tok:,}")
    print("─" * 60)


# ── CLI entry point ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DefinitionDrift NL2SQL batch evaluator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/eval.py --input scripts/sample_eval.jsonl
  python scripts/eval.py --input eval_set.jsonl --data-db data/contoso.db
  python scripts/eval.py --input eval_set.jsonl --output results/run1.jsonl --quiet
        """,
    )
    parser.add_argument(
        "--input", "-i", required=True,
        help="Path to JSONL input file. Each line: {\"question\":\"...\",\"expected_sql\":\"...\"}",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Path to write per-pair JSONL results (default: <input>.results.jsonl)",
    )
    parser.add_argument(
        "--data-db", default=None,
        help="Path to SQLite data DB for execution-match metric (optional)",
    )
    parser.add_argument(
        "--limit", "-n", type=int, default=None,
        help="Evaluate only the first N pairs (useful for quick sanity checks)",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress per-row output; only print the final summary",
    )
    args = parser.parse_args()

    input_path  = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    output_path = Path(args.output) if args.output else input_path.with_suffix(".results.jsonl")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load pairs ─────────────────────────────────────────────────────────────
    pairs = []
    with input_path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"WARNING: Skipping malformed JSON on line {lineno}: {e}", file=sys.stderr)

    if args.limit:
        pairs = pairs[: args.limit]

    print(f"DefinitionDrift Eval  ·  {len(pairs)} pairs  ·  input: {input_path}")
    if args.data_db:
        print(f"Execution match ON    ·  data-db: {args.data_db}")
    else:
        print("Execution match OFF   ·  pass --data-db to enable")
    print()

    # ── Bootstrap DB ───────────────────────────────────────────────────────────
    init_db()

    # ── Run evaluation ─────────────────────────────────────────────────────────
    results = run_eval(pairs, args.data_db, args.quiet)

    # ── Write output ───────────────────────────────────────────────────────────
    with output_path.open("w", encoding="utf-8") as fh:
        for row in results:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nResults written to: {output_path}")

    # ── Print summary ───────────────────────────────────────────────────────────
    print_summary(results)


if __name__ == "__main__":
    main()
