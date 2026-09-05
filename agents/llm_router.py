"""
agents/llm_router.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Free LLM router — zero ongoing cost
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Priority order (all free, no credit card):
  1. Groq       — Llama 3.3 70B  — SQL generation (fastest, OpenAI-compat)
  2. Gemini     — Flash 1.5       — HITL explanations (most quota)
  3. Cerebras   — Llama 3.3 70B  — batch eval (1M tok/day)
  4. OpenRouter — auto:free       — universal fallback

All providers use OpenAI-compatible endpoints so swapping is
a single env var change, zero code change.

Get free keys (no card needed):
  Groq:       console.groq.com
  Gemini:     aistudio.google.com/app/apikey
  Cerebras:   cloud.cerebras.ai
  OpenRouter: openrouter.ai/keys
"""

import os
import json
import time
import sqlite3
from pathlib import Path
from typing import Optional
from datetime import datetime

# OpenAI-compatible client works for Groq, Cerebras, OpenRouter
from openai import OpenAI

# Gemini needs its own SDK
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

# ── Provider configs ──────────────────────────────────────────────────────────

PROVIDERS = {
    "groq": {
        "base_url":  "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model":     "llama-3.3-70b-versatile",
        "rpm":       30,
        "rpd":       1000,
        "best_for":  ["sql_generation", "structured_output"],
    },
    "cerebras": {
        "base_url":  "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
        "model":     "llama-3.3-70b",
        "rpm":       30,
        "rpd":       99999,   # 1M tokens/day, not request-capped
        "best_for":  ["batch_eval", "bulk_queries"],
    },
    "openrouter": {
        "base_url":  "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "model":     "meta-llama/llama-3.3-70b-instruct:free",
        "rpm":       20,
        "rpd":       50,
        "best_for":  ["fallback"],
    },
}

# ── Usage tracker (SQLite so it survives restarts) ────────────────────────────

USAGE_DB = Path(__file__).parent.parent / "data" / "llm_usage.db"

def _init_usage_db():
    USAGE_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(USAGE_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS llm_calls (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            provider    TEXT NOT NULL,
            model       TEXT NOT NULL,
            task        TEXT,
            input_tok   INTEGER DEFAULT 0,
            output_tok  INTEGER DEFAULT 0,
            latency_ms  INTEGER DEFAULT 0,
            success     INTEGER DEFAULT 1,
            error       TEXT,
            called_at   TEXT DEFAULT (datetime('now'))
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS daily_counts (
            date        TEXT NOT NULL,
            provider    TEXT NOT NULL,
            calls       INTEGER DEFAULT 0,
            PRIMARY KEY (date, provider)
        )
    """)
    conn.commit()
    conn.close()

_init_usage_db()

def _log_call(provider: str, model: str, task: str,
              input_tok: int, output_tok: int, latency_ms: int,
              success: bool, error: str = None):
    conn = sqlite3.connect(USAGE_DB)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    conn.execute("""
        INSERT INTO llm_calls (provider, model, task, input_tok, output_tok, latency_ms, success, error)
        VALUES (?,?,?,?,?,?,?,?)
    """, (provider, model, task, input_tok, output_tok, latency_ms, int(success), error))
    conn.execute("""
        INSERT INTO daily_counts (date, provider, calls) VALUES (?,?,1)
        ON CONFLICT(date, provider) DO UPDATE SET calls = calls + 1
    """, (today, provider))
    conn.commit()
    conn.close()

def _today_calls(provider: str) -> int:
    conn = sqlite3.connect(USAGE_DB)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    row = conn.execute(
        "SELECT calls FROM daily_counts WHERE date=? AND provider=?",
        (today, provider)
    ).fetchone()
    conn.close()
    return row[0] if row else 0

def get_usage_stats() -> dict:
    conn = sqlite3.connect(USAGE_DB)
    today = datetime.utcnow().strftime("%Y-%m-%d")
    rows = conn.execute(
        "SELECT provider, calls FROM daily_counts WHERE date=?", (today,)
    ).fetchall()
    total = conn.execute("SELECT COUNT(*), SUM(input_tok+output_tok) FROM llm_calls").fetchone()
    conn.close()
    return {
        "today": {r[0]: r[1] for r in rows},
        "total_calls": total[0] or 0,
        "total_tokens": total[1] or 0,
        "limits": {p: PROVIDERS[p]["rpd"] for p in PROVIDERS}
    }

# ── Core call function ────────────────────────────────────────────────────────

def _call_openai_compat(provider_name: str, system: str, user: str,
                         max_tokens: int = 512, task: str = "general") -> Optional[str]:
    """Call any OpenAI-compatible provider."""
    cfg = PROVIDERS[provider_name]
    api_key = os.getenv(cfg["api_key_env"], "")
    if not api_key:
        return None

    # check daily limit
    if _today_calls(provider_name) >= cfg["rpd"]:
        print(f"[LLM Router] {provider_name} daily limit reached ({cfg['rpd']} calls)")
        return None

    client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
    start = time.time()
    try:
        resp = client.chat.completions.create(
            model=cfg["model"],
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            max_tokens=max_tokens,
            temperature=0.0,   # deterministic — same question, same SQL
        )
        latency = int((time.time() - start) * 1000)
        text = resp.choices[0].message.content.strip()
        usage = resp.usage
        _log_call(provider_name, cfg["model"], task,
                  usage.prompt_tokens if usage else 0,
                  usage.completion_tokens if usage else 0,
                  latency, True)
        print(f"[LLM Router] {provider_name} OK ({latency}ms)")
        return text
    except Exception as e:
        latency = int((time.time() - start) * 1000)
        _log_call(provider_name, cfg["model"], task, 0, 0, latency, False, str(e))
        print(f"[LLM Router] {provider_name} failed: {e}")
        return None


def _call_gemini(system: str, user: str, max_tokens: int = 512,
                 task: str = "general") -> Optional[str]:
    """Call Google Gemini Flash — best for HITL explanations."""
    if not GEMINI_AVAILABLE:
        print("[LLM Router] google-generativeai not installed (pip install google-generativeai)")
        return None
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        return None
    if _today_calls("gemini") >= 1500:
        print("[LLM Router] Gemini daily limit reached (1500)")
        return None

    start = time.time()
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            "gemini-1.5-flash",
            system_instruction=system,
            generation_config={"max_output_tokens": max_tokens, "temperature": 0.0}
        )
        resp = model.generate_content(user)
        latency = int((time.time() - start) * 1000)
        text = resp.text.strip()
        tok_in  = resp.usage_metadata.prompt_token_count if resp.usage_metadata else 0
        tok_out = resp.usage_metadata.candidates_token_count if resp.usage_metadata else 0
        _log_call("gemini", "gemini-1.5-flash", task, tok_in, tok_out, latency, True)
        print(f"[LLM Router] gemini OK ({latency}ms)")
        return text
    except Exception as e:
        latency = int((time.time() - start) * 1000)
        _log_call("gemini", "gemini-1.5-flash", task, 0, 0, latency, False, str(e))
        print(f"[LLM Router] gemini failed: {e}")
        return None


# ── Public router ─────────────────────────────────────────────────────────────

def call_llm(system: str, user: str,
             task: str = "sql_generation",
             max_tokens: int = 512) -> tuple[str, str]:
    """
    Routes to best available free provider for the given task.
    Returns (response_text, provider_name_used).

    task options:
      "sql_generation"  → Groq primary (fastest, best structured output)
      "hitl_explain"    → Gemini primary (most quota, long context)
      "batch_eval"      → Cerebras primary (most tokens/day)
      "fallback"        → OpenRouter

    Falls through providers automatically if one fails or hits limits.
    """
    providers_by_task = {
        "sql_generation": ["groq", "cerebras", "openrouter"],
        "hitl_explain":   ["gemini", "groq", "openrouter"],
        "batch_eval":     ["cerebras", "groq", "openrouter"],
        "conflict_check": ["groq", "openrouter"],
        "general":        ["groq", "gemini", "cerebras", "openrouter"],
    }

    ordered = providers_by_task.get(task, ["groq", "gemini", "cerebras", "openrouter"])

    for provider in ordered:
        if provider == "gemini":
            result = _call_gemini(system, user, max_tokens, task)
        else:
            result = _call_openai_compat(provider, system, user, max_tokens, task)

        if result:
            return result, provider

    # complete fallback — char-frequency + template SQL (zero API)
    print("[LLM Router] ALL providers failed — using offline fallback")
    return json.dumps({
        "sql": None,
        "used_definitions": [],
        "confidence": "low",
        "explanation": "All LLM providers unavailable. Check your API keys in .env.",
        "warning": "Set GROQ_API_KEY or GEMINI_API_KEY for free SQL generation."
    }), "offline"


if __name__ == "__main__":
    print("=== LLM Router Test ===\n")
    system = "You are a helpful assistant. Reply in one sentence."
    user = "What is 2 + 2?"
    resp, provider = call_llm(system, user, task="general", max_tokens=64)
    print(f"Provider used: {provider}")
    print(f"Response: {resp}")
    print(f"\nUsage stats: {json.dumps(get_usage_stats(), indent=2)}")