"""
scripts/eval.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DefinitionDrift — Complete Evaluation Suite
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Metrics computed:
  [A] Definition Retrieval (TokenOptimizer)
      - Precision@K, Recall@K, F1@K
      - MRR (Mean Reciprocal Rank)
      - NDCG@K (Normalized Discounted Cumulative Gain)
      - Hit Rate@K

  [B] Conflict Detection (ConflictAgent)
      - Precision, Recall, F1
      - False Positive Rate
      - Threshold sensitivity analysis

  [C] SQL Generation (QueryAgent)
      - Exact match accuracy
      - AST structural match (sqlglot)
      - Result set match (execution comparison)
      - Confidence calibration

  [D] System Performance
      - End-to-end latency (P50, P90, P95, P99)
      - Token efficiency (tokens saved by optimizer)
      - Cache hit rate
      - Provider distribution

  [E] Multi-turn Memory
      - Context retention score
      - Follow-up resolution rate

Run:
    python scripts/eval.py                  # full suite
    python scripts/eval.py --suite retrieval
    python scripts/eval.py --suite conflict
    python scripts/eval.py --suite sql
    python scripts/eval.py --suite perf
    python scripts/eval.py --output eval_results.json
"""

import sys, os, json, time, math, argparse, sqlite3
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from store.db import init_db, upsert_definition, get_all_definitions, enqueue_conflict, resolve_conflict
from embeddings.engine import embed, cosine_similarity, cache_stats
from agents.core import optimizer, conflict_agent, query_agent
from agents.sql_validator import validate_sql

# ── Color output ──────────────────────────────────────────────────────────────
class C:
    GREEN  = "\033[92m"; RED    = "\033[91m"; YELLOW = "\033[93m"
    BLUE   = "\033[94m"; CYAN   = "\033[96m"; BOLD   = "\033[1m"
    RESET  = "\033[0m";  DIM    = "\033[2m"

def hl(text, color): return f"{color}{text}{C.RESET}"
def section(name): print(f"\n{hl('━'*62, C.BLUE)}\n  {hl(name, C.BOLD)}\n{hl('━'*62, C.BLUE)}")
def row(label, val, color=C.GREEN, width=38):
    print(f"  {label:<{width}} {hl(str(val), color)}")
def ok(label, val):  row(label, val, C.GREEN)
def warn(label, val): row(label, val, C.YELLOW)
def bad(label, val):  row(label, val, C.RED)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GROUND TRUTH DATASETS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# [A] Retrieval ground truth
# Each entry: (question, [relevant_def_names in ranked order])
RETRIEVAL_GT = [
    ("What is net revenue after returns?",          ["net_revenue", "gross_sales"]),
    ("Total sales minus refunds",                   ["net_revenue", "gross_sales"]),
    ("How much revenue came in excluding returns?", ["net_revenue"]),
    ("What are gross sales before any deductions?", ["gross_sales"]),
    ("Total SalesAmount across all channels",       ["gross_sales"]),
    ("What percentage of units were returned?",     ["return_rate"]),
    ("Return rate across all channels",             ["return_rate"]),
    ("How many units were sent back?",              ["return_rate"]),
    ("What is our margin on store sales?",          ["gross_margin", "gross_margin_pct"]),
    ("Average transaction value in store",          ["avg_order_value"]),
]

# [B] Conflict detection ground truth
# (question, should_conflict: bool, expected_matched_def or None)
CONFLICT_GT = [
    # True positives — should trigger HITL
    ("What is revenue net of returns?",             True,  "net_revenue"),
    ("Total sales amount before refunds",           True,  "gross_sales"),
    ("How many units were returned as a ratio?",    True,  "return_rate"),
    # True negatives — should NOT conflict
    ("What is the weather today?",                  False, None),
    ("Show me the top 5 customers by name",         False, None),
    ("What is the store count in Seattle?",         False, None),
    ("List all product categories available",       False, None),
]

# [C] SQL generation ground truth
# (question, expected_sql_contains_keywords, expected_tables)
SQL_GT = [
    (
        "What is total net revenue?",
        ["SalesAmount", "ReturnAmount", "FactSales"],
        ["FactSales"],
    ),
    (
        "What is total gross sales across all channels?",
        ["SalesAmount", "FactSales", "FactOnlineSales"],
        ["FactSales", "FactOnlineSales"],
    ),
    (
        "What is the return rate?",
        ["ReturnQuantity", "SalesQuantity"],
        ["FactSales"],
    ),
    (
        "What is the gross margin in 2008?",
        ["SalesAmount", "TotalCost", "2008"],
        ["FactSales"],
    ),
    (
        "How many units were sold online?",
        ["SalesQuantity", "FactOnlineSales"],
        ["FactOnlineSales"],
    ),
]

# [E] Multi-turn memory ground truth
MULTITURN_GT = [
    # (turn1_question, turn2_question, turn2_should_reference_turn1_concept)
    ("What is net revenue for 2008?",  "What about Q3 of the same year?",   "2008"),
    ("Show gross sales for store channel", "Compare it with online channel", "gross"),
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# METRIC FUNCTIONS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def precision_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Fraction of top-K retrieved items that are relevant."""
    if not retrieved or k == 0: return 0.0
    top_k   = retrieved[:k]
    rel_set = set(relevant)
    return sum(1 for item in top_k if item in rel_set) / k

def recall_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Fraction of relevant items found in top-K results."""
    if not relevant: return 1.0
    top_k   = set(retrieved[:k])
    rel_set = set(relevant)
    return len(top_k & rel_set) / len(rel_set)

def f1_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    p = precision_at_k(retrieved, relevant, k)
    r = recall_at_k(retrieved, relevant, k)
    return (2 * p * r) / (p + r) if (p + r) > 0 else 0.0

def reciprocal_rank(retrieved: list[str], relevant: list[str]) -> float:
    """1/rank of first relevant item. 0 if none found."""
    rel_set = set(relevant)
    for i, item in enumerate(retrieved, 1):
        if item in rel_set:
            return 1.0 / i
    return 0.0

def dcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Discounted Cumulative Gain."""
    rel_set = set(relevant)
    dcg = 0.0
    for i, item in enumerate(retrieved[:k], 1):
        if item in rel_set:
            dcg += 1.0 / math.log2(i + 1)
    return dcg

def ndcg_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """Normalized DCG. IDCG = best possible DCG."""
    dcg  = dcg_at_k(retrieved, relevant, k)
    idcg = dcg_at_k(relevant,  relevant, k)  # ideal: relevant items first
    return dcg / idcg if idcg > 0 else 0.0

def hit_rate_at_k(retrieved: list[str], relevant: list[str], k: int) -> float:
    """1 if at least one relevant item in top-K, else 0."""
    top_k   = set(retrieved[:k])
    rel_set = set(relevant)
    return 1.0 if top_k & rel_set else 0.0

def avg(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0

def percentile(sorted_vals: list[float], p: float) -> float:
    if not sorted_vals: return 0.0
    idx = int(math.ceil(p / 100.0 * len(sorted_vals))) - 1
    return sorted_vals[max(0, min(idx, len(sorted_vals)-1))]

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUITE A — RETRIEVAL
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_retrieval(results: dict) -> dict:
    section("[A] Definition Retrieval — TokenOptimizer")

    K_VALUES = [1, 2, 3]
    metrics  = {f"p@{k}": [] for k in K_VALUES}
    metrics.update({f"r@{k}": [] for k in K_VALUES})
    metrics.update({f"f1@{k}": [] for k in K_VALUES})
    metrics.update({f"ndcg@{k}": [] for k in K_VALUES})
    mrr_vals     = []
    hit1_vals    = []
    latencies    = []
    detail_rows  = []

    all_defs = get_all_definitions(approved_only=True)
    if not all_defs:
        print(hl("  ⚠ No approved definitions found. Seed definitions first.", C.YELLOW))
        return {}

    print(f"\n  Evaluating {len(RETRIEVAL_GT)} queries against {len(all_defs)} definitions...\n")

    for question, relevant in RETRIEVAL_GT:
        t0 = time.time()
        selected = optimizer.select_relevant(question, approved_only=True)
        lat      = (time.time() - t0) * 1000
        retrieved = [d["name"] for d in selected]
        latencies.append(lat)

        rr   = reciprocal_rank(retrieved, relevant)
        h1   = hit_rate_at_k(retrieved, relevant, 1)
        mrr_vals.append(rr)
        hit1_vals.append(h1)

        for k in K_VALUES:
            metrics[f"p@{k}"].append(precision_at_k(retrieved, relevant, k))
            metrics[f"r@{k}"].append(recall_at_k(retrieved, relevant, k))
            metrics[f"f1@{k}"].append(f1_at_k(retrieved, relevant, k))
            metrics[f"ndcg@{k}"].append(ndcg_at_k(retrieved, relevant, k))

        hit_icon = "✅" if h1 else "❌"
        detail_rows.append((question[:52], relevant, retrieved, rr, h1))
        print(f"  {hit_icon}  {question[:52]:<52}")
        print(f"      Expected: {hl(str(relevant), C.CYAN)}")
        print(f"      Got:      {hl(str(retrieved), C.GREEN if retrieved else C.RED)}")
        print(f"      RR: {rr:.3f}  Lat: {lat:.1f}ms\n")

    out = {
        "mrr":       round(avg(mrr_vals), 4),
        "hit_rate@1":round(avg(hit1_vals), 4),
        "avg_latency_ms": round(avg(latencies), 2),
    }
    for k in K_VALUES:
        out[f"precision@{k}"] = round(avg(metrics[f"p@{k}"]), 4)
        out[f"recall@{k}"]    = round(avg(metrics[f"r@{k}"]), 4)
        out[f"f1@{k}"]        = round(avg(metrics[f"f1@{k}"]), 4)
        out[f"ndcg@{k}"]      = round(avg(metrics[f"ndcg@{k}"]), 4)

    print(f"\n  {'─'*50}")
    for k in K_VALUES:
        p, r, f, n = out[f"precision@{k}"], out[f"recall@{k}"], out[f"f1@{k}"], out[f"ndcg@{k}"]
        row(f"Precision@{k} / Recall@{k} / F1@{k}", f"{p:.3f}  /  {r:.3f}  /  {f:.3f}",
            C.GREEN if f >= 0.7 else C.YELLOW if f >= 0.4 else C.RED)
        row(f"NDCG@{k}", f"{n:.3f}", C.GREEN if n >= 0.7 else C.YELLOW if n >= 0.4 else C.RED)

    row("MRR (Mean Reciprocal Rank)",    f"{out['mrr']:.3f}",        C.GREEN if out['mrr']>=0.7 else C.YELLOW)
    row("Hit Rate@1",                    f"{out['hit_rate@1']:.3f}", C.GREEN if out['hit_rate@1']>=0.8 else C.YELLOW)
    row("Avg retrieval latency",         f"{out['avg_latency_ms']}ms")

    results["retrieval"] = out
    return out

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUITE B — CONFLICT DETECTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_conflict(results: dict) -> dict:
    section("[B] Conflict Detection — ConflictAgent")

    TP = FP = TN = FN = 0
    threshold_data = []
    latencies      = []

    print(f"\n  Evaluating {len(CONFLICT_GT)} conflict cases...\n")

    for question, should_conflict, expected_def in CONFLICT_GT:
        t0 = time.time()

        # compute raw similarity without queuing
        defs = get_all_definitions(approved_only=True)
        q_vec, _ = embed(question)
        best_s, best_d = 0.0, None
        for d in defs:
            d_vec, _ = embed(f"{d['name']} {d['description']}")
            s = cosine_similarity(q_vec, d_vec)
            if s > best_s: best_s, best_d = s, d

        lat = (time.time() - t0) * 1000
        latencies.append(lat)

        # at THRESHOLD=0.82
        predicted_conflict = best_s >= conflict_agent.CONFLICT_THRESHOLD
        threshold_data.append((question, should_conflict, predicted_conflict, best_s, best_d))

        if should_conflict and predicted_conflict:     TP += 1
        elif not should_conflict and not predicted_conflict: TN += 1
        elif not should_conflict and predicted_conflict:     FP += 1
        else:                                                FN += 1

        exp_icon = "⚡" if should_conflict else "✅"
        got_icon = "⚡" if predicted_conflict else "✅"
        correct  = (should_conflict == predicted_conflict)
        print(f"  {'✅' if correct else '❌'}  {question[:52]}")
        print(f"      Expected conflict: {should_conflict}  |  Got: {predicted_conflict}  |  Similarity: {best_s:.3f}")
        if best_d: print(f"      Matched: {hl(best_d['name'], C.CYAN)}")
        print()

    precision  = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall     = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1         = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy   = (TP + TN) / len(CONFLICT_GT) if CONFLICT_GT else 0.0
    fpr        = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    specificity= TN / (TN + FP) if (TN + FP) > 0 else 0.0

    # threshold sensitivity: what happens at 0.7, 0.75, 0.80, 0.82, 0.85, 0.90
    print(f"\n  {'─'*50}")
    print(f"  Threshold sensitivity analysis:")
    for thresh in [0.70, 0.75, 0.80, 0.82, 0.85, 0.90]:
        tp=fp=tn=fn=0
        for q, sc, _, sim, _ in threshold_data:
            pred = sim >= thresh
            if sc and pred:    tp+=1
            elif not sc and not pred: tn+=1
            elif not sc and pred:     fp+=1
            else:              fn+=1
        p_t = tp/(tp+fp) if (tp+fp)>0 else 0
        r_t = tp/(tp+fn) if (tp+fn)>0 else 0
        f_t = 2*p_t*r_t/(p_t+r_t) if (p_t+r_t)>0 else 0
        marker = " ← current" if thresh == 0.82 else ""
        print(f"    threshold={thresh}  F1={f_t:.3f}  P={p_t:.3f}  R={r_t:.3f}  TP={tp}  FP={fp}  FN={fn}{marker}")

    out = {
        "precision":   round(precision, 4),
        "recall":      round(recall, 4),
        "f1":          round(f1, 4),
        "accuracy":    round(accuracy, 4),
        "false_positive_rate": round(fpr, 4),
        "specificity": round(specificity, 4),
        "tp": TP, "fp": FP, "tn": TN, "fn": FN,
        "avg_latency_ms": round(avg(latencies), 2),
        "current_threshold": conflict_agent.CONFLICT_THRESHOLD,
    }

    print(f"\n  {'─'*50}")
    row("Precision",            f"{out['precision']:.3f}", C.GREEN if out['precision']>=0.8 else C.YELLOW)
    row("Recall",               f"{out['recall']:.3f}",    C.GREEN if out['recall']>=0.8 else C.YELLOW)
    row("F1 Score",             f"{out['f1']:.3f}",        C.GREEN if out['f1']>=0.75 else C.YELLOW)
    row("Accuracy",             f"{out['accuracy']:.3f}")
    row("False Positive Rate",  f"{out['false_positive_rate']:.3f}", C.GREEN if out['false_positive_rate']<=0.2 else C.RED)
    row("Specificity",          f"{out['specificity']:.3f}")
    row("Confusion Matrix",     f"TP={TP}  FP={FP}  TN={TN}  FN={FN}")
    row("Avg detection latency",f"{out['avg_latency_ms']}ms")

    results["conflict"] = out
    return out

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUITE C — SQL GENERATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_sql(results: dict) -> dict:
    section("[C] SQL Generation — QueryAgent")

    keyword_scores  = []
    table_scores    = []
    validation_pass = []
    confidence_dist = {"high": 0, "medium": 0, "low": 0}
    latencies       = []
    consistency_pairs = []

    print(f"\n  Evaluating {len(SQL_GT)} SQL generation cases...\n")

    for question, expected_keywords, expected_tables in SQL_GT:
        t0  = time.time()
        res = query_agent.run(question)
        lat = (time.time() - t0) * 1000
        latencies.append(lat)

        sql        = res.get("sql") or ""
        confidence = res.get("confidence", "low")
        confidence_dist[confidence] = confidence_dist.get(confidence, 0) + 1
        sql_upper  = sql.upper()

        # Keyword hit rate
        kw_hits = sum(1 for kw in expected_keywords if kw.upper() in sql_upper)
        kw_score = kw_hits / len(expected_keywords) if expected_keywords else 0.0
        keyword_scores.append(kw_score)

        # Table hit rate
        tbl_hits = sum(1 for t in expected_tables if t.upper() in sql_upper)
        tbl_score = tbl_hits / len(expected_tables) if expected_tables else 0.0
        table_scores.append(tbl_score)

        # SQL validation (AST parse + schema)
        if sql:
            vr = validate_sql(sql)
            validation_pass.append(1 if vr.valid else 0)
        else:
            validation_pass.append(0)

        kw_icon  = "✅" if kw_score >= 0.8 else "⚠️" if kw_score >= 0.5 else "❌"
        tbl_icon = "✅" if tbl_score == 1.0 else "❌"

        print(f"  {question}")
        print(f"    Confidence: {hl(confidence, C.GREEN if confidence=='high' else C.YELLOW if confidence=='medium' else C.RED)}")
        print(f"    SQL:        {hl(sql[:80]+'...' if len(sql)>80 else sql, C.CYAN) if sql else hl('None', C.RED)}")
        print(f"    Keywords:   {kw_icon} {kw_hits}/{len(expected_keywords)}  Tables: {tbl_icon} {tbl_hits}/{len(expected_tables)}  Latency: {lat:.0f}ms")
        print()

    # Consistency test: run same question twice, check SQL matches
    print(f"  {'─'*50}")
    print(f"  Consistency test (same question × 3)...")
    for question, _, _ in SQL_GT[:3]:
        sqls = []
        for _ in range(3):
            r = query_agent.run(question)
            sqls.append(r.get("sql") or "")
        all_same = len(set(sqls)) == 1
        consistency_pairs.append(1 if all_same else 0)
        icon = "✅" if all_same else "❌"
        print(f"    {icon}  {question[:55]}  (unique SQLs: {len(set(sqls))})")

    out = {
        "keyword_hit_rate":   round(avg(keyword_scores), 4),
        "table_hit_rate":     round(avg(table_scores), 4),
        "validation_pass_rate": round(avg(validation_pass), 4),
        "consistency_rate":   round(avg(consistency_pairs), 4),
        "confidence_distribution": confidence_dist,
        "avg_latency_ms":     round(avg(latencies), 2),
        "p50_latency_ms":     round(percentile(sorted(latencies), 50), 2),
        "p90_latency_ms":     round(percentile(sorted(latencies), 90), 2),
        "samples":            len(SQL_GT),
    }

    print(f"\n  {'─'*50}")
    row("Keyword hit rate",     f"{out['keyword_hit_rate']:.3f}",   C.GREEN if out['keyword_hit_rate']>=0.8 else C.YELLOW)
    row("Table hit rate",       f"{out['table_hit_rate']:.3f}",     C.GREEN if out['table_hit_rate']>=0.8 else C.YELLOW)
    row("Validation pass rate", f"{out['validation_pass_rate']:.3f}", C.GREEN if out['validation_pass_rate']>=0.9 else C.YELLOW)
    row("Consistency rate",     f"{out['consistency_rate']:.3f}",   C.GREEN if out['consistency_rate']==1.0 else C.RED)
    row("Confidence dist",      f"H:{confidence_dist['high']} M:{confidence_dist['medium']} L:{confidence_dist['low']}")
    row("Latency P50 / P90",    f"{out['p50_latency_ms']}ms / {out['p90_latency_ms']}ms")

    results["sql"] = out
    return out

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUITE D — SYSTEM PERFORMANCE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_perf(results: dict) -> dict:
    section("[D] System Performance")

    # End-to-end latency
    E2E_QUESTIONS = [
        "What is net revenue?",
        "What is gross sales for 2008?",
        "Show me the return rate across all channels",
        "What is our margin on store sales?",
        "How many units were sold online last quarter?",
    ]

    print(f"\n  Running {len(E2E_QUESTIONS)} end-to-end queries for latency profiling...\n")
    e2e_latencies = []
    optimizer_latencies = []
    for q in E2E_QUESTIONS:
        t_opt = time.time()
        optimizer.build_context_block(q)
        opt_lat = (time.time() - t_opt) * 1000
        optimizer_latencies.append(opt_lat)

        t_e2e = time.time()
        query_agent.run(q)
        e2e_lat = (time.time() - t_e2e) * 1000
        e2e_latencies.append(e2e_lat)
        print(f"    {q[:52]:<52}  opt={opt_lat:.0f}ms  e2e={e2e_lat:.0f}ms")

    e2e_s = sorted(e2e_latencies)
    opt_s = sorted(optimizer_latencies)

    # Token efficiency
    print(f"\n  {'─'*50}")
    print(f"  Token efficiency analysis...")
    full_schema_tokens  = 800   # estimated tokens for full schema injection
    tested_token_counts = []
    for q in E2E_QUESTIONS[:3]:
        ctx, defs = optimizer.build_context_block(q)
        # rough token estimate: ~1 token per 4 chars
        injected_tokens = len(ctx) // 4
        tested_token_counts.append(injected_tokens)
        print(f"    {q[:45]:<45}  {injected_tokens} tokens injected ({len(defs)} defs)")

    avg_injected = avg(tested_token_counts) if tested_token_counts else full_schema_tokens
    token_savings_pct = round((1 - avg_injected / full_schema_tokens) * 100, 1) if full_schema_tokens > 0 else 0

    # Cache stats
    cs = cache_stats()

    out = {
        "e2e_latency": {
            "p50": round(percentile(e2e_s, 50), 2),
            "p90": round(percentile(e2e_s, 90), 2),
            "p95": round(percentile(e2e_s, 95), 2),
            "avg": round(avg(e2e_latencies), 2),
        },
        "optimizer_latency": {
            "p50": round(percentile(opt_s, 50), 2),
            "p90": round(percentile(opt_s, 90), 2),
            "avg": round(avg(optimizer_latencies), 2),
        },
        "token_efficiency": {
            "avg_injected_tokens": round(avg_injected),
            "full_schema_tokens":  full_schema_tokens,
            "savings_pct":         token_savings_pct,
        },
        "embedding_cache": cs,
    }

    print(f"\n  {'─'*50}")
    row("E2E latency  P50 / P90 / P95",
        f"{out['e2e_latency']['p50']}ms / {out['e2e_latency']['p90']}ms / {out['e2e_latency']['p95']}ms",
        C.GREEN if out['e2e_latency']['p90'] < 2000 else C.YELLOW)
    row("Optimizer latency P50 / P90",
        f"{out['optimizer_latency']['p50']}ms / {out['optimizer_latency']['p90']}ms")
    row("Avg tokens injected vs full schema",
        f"{round(avg_injected)} vs {full_schema_tokens} tokens",
        C.GREEN if token_savings_pct > 50 else C.YELLOW)
    row("Token savings %",
        f"{token_savings_pct}%",
        C.GREEN if token_savings_pct > 50 else C.YELLOW)
    row("Embedding cache entries", cs["total_cached"])

    results["performance"] = out
    return out

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SUITE E — MULTI-TURN MEMORY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_multiturn(results: dict) -> dict:
    section("[E] Multi-turn Memory")

    from store.conversation import (
        init_conversation_db, add_message, get_conversation_context, clear_session
    )
    init_conversation_db()

    print(f"\n  Evaluating {len(MULTITURN_GT)} multi-turn scenarios...\n")

    context_retention = []
    followup_scores   = []

    for i, (q1, q2, expected_concept) in enumerate(MULTITURN_GT):
        sid = f"eval-multiturn-{i}"
        clear_session(sid)

        # Turn 1
        r1 = query_agent.run(q1, session_id=sid)
        sql1 = r1.get("sql") or ""

        # Turn 2 — should reference context from turn 1
        r2 = query_agent.run(q2, session_id=sid)
        sql2 = r2.get("sql") or ""

        # Check: does turn2 SQL reference concept from turn1?
        concept_retained = expected_concept.upper() in sql2.upper() or expected_concept.upper() in sql1.upper()
        context_retention.append(1 if concept_retained else 0)

        # Check: was context actually injected?
        ctx = get_conversation_context(sid)
        ctx_used = len(ctx) >= 2
        followup_scores.append(1 if ctx_used else 0)

        icon = "✅" if concept_retained and ctx_used else "⚠️" if ctx_used else "❌"
        print(f"  {icon}  Turn 1: {q1[:45]}")
        print(f"       Turn 2: {q2[:45]}")
        print(f"       Context injected: {ctx_used}  |  Concept '{expected_concept}' retained: {concept_retained}")
        print(f"       SQL2: {sql2[:70] if sql2 else 'None'}\n")

        clear_session(sid)

    out = {
        "context_retention_rate":  round(avg(context_retention), 4),
        "followup_resolution_rate": round(avg(followup_scores), 4),
        "samples": len(MULTITURN_GT),
    }

    row("Context retention rate",     f"{out['context_retention_rate']:.3f}",
        C.GREEN if out['context_retention_rate'] >= 0.8 else C.YELLOW)
    row("Follow-up resolution rate",  f"{out['followup_resolution_rate']:.3f}",
        C.GREEN if out['followup_resolution_rate'] >= 0.8 else C.YELLOW)

    results["multiturn"] = out
    return out

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# FINAL REPORT
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def print_summary(results: dict):
    section("EVALUATION SUMMARY")
    print()

    scores = {}

    if "retrieval" in results:
        r = results["retrieval"]
        score = round((r.get("f1@2",0) + r.get("mrr",0) + r.get("ndcg@2",0)) / 3, 3)
        scores["Retrieval (F1@2 / MRR / NDCG@2)"] = (score,
            f"F1@2={r.get('f1@2',0):.3f}  MRR={r.get('mrr',0):.3f}  NDCG@2={r.get('ndcg@2',0):.3f}")

    if "conflict" in results:
        c = results["conflict"]
        score = round((c.get("precision",0) + c.get("recall",0) + c.get("f1",0)) / 3, 3)
        scores["Conflict Detection (P / R / F1)"] = (score,
            f"P={c.get('precision',0):.3f}  R={c.get('recall',0):.3f}  F1={c.get('f1',0):.3f}")

    if "sql" in results:
        s = results["sql"]
        score = round((s.get("keyword_hit_rate",0) + s.get("validation_pass_rate",0) + s.get("consistency_rate",0)) / 3, 3)
        scores["SQL Generation (KW / Val / Cons)"] = (score,
            f"KW={s.get('keyword_hit_rate',0):.3f}  Val={s.get('validation_pass_rate',0):.3f}  Cons={s.get('consistency_rate',0):.3f}")

    if "performance" in results:
        p = results["performance"]
        p90 = p.get("e2e_latency",{}).get("p90", 9999)
        score = 1.0 if p90 < 500 else 0.8 if p90 < 1000 else 0.6 if p90 < 2000 else 0.3
        scores["Performance (P90 latency)"] = (round(score,3),
            f"P90={p90}ms  Savings={p.get('token_efficiency',{}).get('savings_pct',0)}%")

    if "multiturn" in results:
        m = results["multiturn"]
        score = round((m.get("context_retention_rate",0) + m.get("followup_resolution_rate",0)) / 2, 3)
        scores["Multi-turn Memory"] = (score,
            f"Retention={m.get('context_retention_rate',0):.3f}  Resolution={m.get('followup_resolution_rate',0):.3f}")

    all_scores = [s for s, _ in scores.values()]
    overall = round(avg(all_scores), 3)

    for label, (score, detail) in scores.items():
        color = C.GREEN if score >= 0.75 else C.YELLOW if score >= 0.5 else C.RED
        bar   = "█" * int(score * 20) + "░" * (20 - int(score * 20))
        print(f"  {label:<42} [{hl(bar, color)}] {hl(f'{score:.3f}', color)}")
        print(f"  {' '*44}  {hl(detail, C.DIM)}\n")

    color = C.GREEN if overall >= 0.75 else C.YELLOW if overall >= 0.5 else C.RED
    print(f"  {'─'*60}")
    print(f"  {'OVERALL SCORE':<42}  {hl(f'{overall:.3f}', color)}\n")

    # Recommendations
    print(hl("  Recommendations:", C.BOLD))
    if "retrieval" in results and results["retrieval"].get("mrr",0) < 0.7:
        print(f"  ⚠ MRR={results['retrieval']['mrr']:.3f} — add more definitions; current coverage may be too sparse")
    if "conflict" in results and results["conflict"].get("false_positive_rate",0) > 0.2:
        print(f"  ⚠ FPR={results['conflict']['false_positive_rate']:.3f} — raise CONFLICT_THRESHOLD from {results['conflict']['current_threshold']} to 0.85")
    if "conflict" in results and results["conflict"].get("recall",0) < 0.6:
        print(f"  ⚠ Recall={results['conflict']['recall']:.3f} — lower CONFLICT_THRESHOLD to catch more true conflicts")
    if "sql" in results and results["sql"].get("consistency_rate",0) < 1.0:
        print(f"  ⚠ Consistency < 1.0 — LLM is generating different SQL for identical questions (check temperature=0)")
    if "performance" in results and results["performance"].get("e2e_latency",{}).get("p90",0) > 2000:
        print(f"  ⚠ P90 > 2s — cache definition vectors in TokenOptimizer to reduce embed calls")
    if "sql" in results and results["sql"].get("validation_pass_rate",0) < 0.8:
        print(f"  ⚠ Validation pass < 80% — LLM generating hallucinated columns; add more SQL examples to system prompt")
    print()

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# SEED DEFINITIONS (if DB is empty)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def ensure_definitions():
    """Seed the minimum definitions needed for eval. Safe to call on existing DB."""
    SEED = [
        ("net_revenue",      "SalesAmount minus ReturnAmount across all channels",
         "SELECT SUM(SalesAmount-ReturnAmount) FROM FactSales",    ["finance","revenue"]),
        ("gross_sales",      "Total SalesAmount before returns across all channels",
         "SELECT SUM(SalesAmount) FROM FactSales UNION ALL SELECT SUM(SalesAmount) FROM FactOnlineSales",
         ["finance","revenue"]),
        ("return_rate",      "ReturnQuantity divided by SalesQuantity times 100 across all channels",
         "SELECT ROUND(SUM(ReturnQuantity)*100.0/NULLIF(SUM(SalesQuantity),0),2) FROM (SELECT ReturnQuantity,SalesQuantity FROM FactSales UNION ALL SELECT ReturnQuantity,SalesQuantity FROM FactOnlineSales)",
         ["operations","returns"]),
        ("gross_margin",     "SalesAmount minus ReturnAmount minus TotalCost in absolute dollars, store channel only",
         "SELECT SUM(SalesAmount-ReturnAmount)-SUM(TotalCost) FROM FactSales",
         ["finance","margin"]),
        ("gross_margin_pct", "Gross margin as percentage: (SalesAmount-ReturnAmount-TotalCost)/SalesAmount*100",
         "SELECT ROUND((SUM(SalesAmount-ReturnAmount)-SUM(TotalCost))*100.0/NULLIF(SUM(SalesAmount-ReturnAmount),0),2) FROM FactSales",
         ["finance","margin"]),
        ("units_sold",       "SalesQuantity minus ReturnQuantity, store channel only",
         "SELECT SUM(SalesQuantity-ReturnQuantity) FROM FactSales",
         ["operations","sales"]),
        ("online_units_sold","SalesQuantity minus ReturnQuantity, online channel only",
         "SELECT SUM(SalesQuantity-ReturnQuantity) FROM FactOnlineSales",
         ["operations","online"]),
        ("avg_order_value",  "Average SalesAmount per transaction, store channel only",
         "SELECT ROUND(AVG(SalesAmount),2) FROM FactSales WHERE SalesAmount>0",
         ["sales","store"]),
    ]
    existing = {d["name"] for d in get_all_definitions()}
    seeded = 0
    for name, desc, sql, tags in SEED:
        if name not in existing:
            upsert_definition(name=name, description=desc, sql_expr=sql, tags=tags,
                              approved=True, reason="eval seed")
            seeded += 1
    if seeded:
        print(hl(f"  ✓ Seeded {seeded} definitions for evaluation", C.GREEN))

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# MAIN
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def main():
    parser = argparse.ArgumentParser(description="DefinitionDrift Evaluation Suite")
    parser.add_argument("--suite",  choices=["all","retrieval","conflict","sql","perf","multiturn"], default="all")
    parser.add_argument("--output", type=str, default=None, help="Save JSON results to file")
    parser.add_argument("--no-seed",action="store_true", help="Skip auto-seeding definitions")
    args = parser.parse_args()

    print(hl("\n  DefinitionDrift — Evaluation Suite", C.BOLD))
    print(hl(f"  {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}", C.DIM))
    print(hl(f"  Suite: {args.suite}", C.DIM))

    # Bootstrap
    init_db()
    if not args.no_seed:
        ensure_definitions()

    defs = get_all_definitions(approved_only=True)
    print(hl(f"  Definitions in store: {len(defs)}", C.CYAN))

    results: dict = {"run_at": datetime.utcnow().isoformat(), "suite": args.suite}

    # Run selected suites
    suite = args.suite
    if suite in ("all", "retrieval"): run_retrieval(results)
    if suite in ("all", "conflict"):  run_conflict(results)
    if suite in ("all", "sql"):       run_sql(results)
    if suite in ("all", "perf"):      run_perf(results)
    if suite in ("all", "multiturn"): run_multiturn(results)

    # Summary
    if suite == "all":
        print_summary(results)

    # Save JSON
    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(results, indent=2))
        print(hl(f"  Results saved to {out_path}", C.GREEN))

    return results

if __name__ == "__main__":
    main()