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
from datetime import datetime
from typing import Optional, Literal
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# OpenAI-compatible client works for Groq, Cerebras, OpenRouter
from openai import OpenAI

# Gemini needs its own SDK
try:
    from google import genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    
import functools

_provider_health: dict[str, float] = {}  # provider → last_failure_timestamp

def _is_healthy(provider: str, cooldown_secs: int = 60) -> bool:
    last_fail = _provider_health.get(provider, 0)
    return (time.time() - last_fail) > cooldown_secs

def _mark_failed(provider: str):
    _provider_health[provider] = time.time()

# ── Provider configs ──────────────────────────────────────────────────────────

PROVIDERS = {
    # Groq fast — Qwen3 27B: fastest text-gen model available on this key
    "groq_fast": {
        "base_url":    "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model":       "qwen/qwen3.8-27b",
        "rpd":         1000,
    },
    # Groq smart — GPT-OSS 120B: largest model available on this key
    "groq_smart": {
        "base_url":    "https://api.groq.com/openai/v1",
        "api_key_env": "GROQ_API_KEY",
        "model":       "openai/gpt-oss-120b",
        "rpd":         1000,
    },
    # Cerebras — 70B with massive token budget for batch workloads
    # Correct model slug is llama3.1-70b on Cerebras cloud free tier
    "cerebras": {
        "base_url":  "https://api.cerebras.ai/v1",
        "api_key_env": "CEREBRAS_API_KEY",
        "model":     "llama3.1-70b",
        "rpm":       30,
        "rpd":       99999,
        "best_for":  ["batch_eval", "bulk_queries"],
    },
    # OpenRouter — last-resort fallback only (credit-limited)
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

# ── In-memory caches (Fix #5, #6, #10) ────────────────────────────────────────

# Fix #5: Cache OpenAI client instances per provider (avoid TCP/SSL setup per call)
_clients: dict[str, OpenAI] = {}

def _get_client(provider_name: str) -> Optional[OpenAI]:
    """Return a cached OpenAI-compatible client, creating on first use."""
    if provider_name in _clients:
        return _clients[provider_name]
    cfg = PROVIDERS[provider_name]
    api_key = os.getenv(cfg["api_key_env"], "")
    if not api_key:
        return None
    client = OpenAI(api_key=api_key, base_url=cfg["base_url"])
    _clients[provider_name] = client
    return client

# Fix #6: Circuit breaker — skip providers that fail 3+ times in a row
_circuit_breaker: dict[str, dict] = {}  # provider → {"failures": N, "skip_until": timestamp}
_CIRCUIT_BREAKER_THRESHOLD = 3
_CIRCUIT_BREAKER_COOLDOWN = 600  # 10 minutes

def _record_failure(provider: str):
    """Record a provider failure. After threshold, open the circuit."""
    cb = _circuit_breaker.setdefault(provider, {"failures": 0, "skip_until": 0})
    cb["failures"] += 1
    if cb["failures"] >= _CIRCUIT_BREAKER_THRESHOLD:
        cb["skip_until"] = time.time() + _CIRCUIT_BREAKER_COOLDOWN
        print(f"[LLM Router] Circuit OPEN for {provider} — skipping for {_CIRCUIT_BREAKER_COOLDOWN}s")

def _record_success(provider: str):
    """Reset circuit breaker on success."""
    if provider in _circuit_breaker:
        _circuit_breaker[provider] = {"failures": 0, "skip_until": 0}

def _is_circuit_open(provider: str) -> bool:
    """Check if a provider should be skipped (circuit is open)."""
    cb = _circuit_breaker.get(provider)
    if not cb:
        return False
    if cb["failures"] >= _CIRCUIT_BREAKER_THRESHOLD:
        if time.time() < cb["skip_until"]:
            return True  # still in cooldown — skip
        else:
            cb["failures"] = 0  # cooldown expired — reset and retry
    return False

# Fix #10: In-memory daily count cache (avoid SQLite per provider attempt)
_daily_count_cache: dict[str, dict] = {}  # provider → {"date": str, "count": int}

def _today_calls_cached(provider: str) -> int:
    """Fast in-memory check of daily call count."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    cached = _daily_count_cache.get(provider)
    if cached and cached["date"] == today:
        return cached["count"]
    # Cold start or date change — read from DB
    count = _today_calls_db(provider)
    _daily_count_cache[provider] = {"date": today, "count": count}
    return count

def _increment_daily_count(provider: str):
    """Increment the in-memory counter after a call."""
    today = datetime.utcnow().strftime("%Y-%m-%d")
    cached = _daily_count_cache.get(provider)
    if cached and cached["date"] == today:
        cached["count"] += 1
    else:
        _daily_count_cache[provider] = {"date": today, "count": 1}

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
    _increment_daily_count(provider)

def _today_calls_db(provider: str) -> int:
    """Read daily count from SQLite (cold start only)."""
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
                         max_tokens: int = 512, task: str = "general",
                         model_override: Optional[str] = None) -> Optional[str]:
    """Call any OpenAI-compatible provider.

    model_override: if set, replaces the provider's default model string.
    Used by confidence-based escalation to force QUERY_MODEL_SMART.
    """
    cfg = PROVIDERS[provider_name]
    api_key = os.getenv(cfg["api_key_env"], "")
    if not api_key:
        return None

    if not _is_healthy(provider_name):
        return None

    # Fix #6: Circuit breaker — skip providers that keep failing
    if _is_circuit_open(provider_name):
        return None

    # Fix #10: Check daily limit via in-memory cache
    if _today_calls_cached(provider_name) >= cfg["rpd"]:
        print(f"[LLM Router] {provider_name} daily limit reached ({cfg['rpd']} calls)")
        return None

    model = model_override or cfg["model"]
    # Fix #5: Reuse cached client
    client = _get_client(provider_name)
    if client is None:
        return None
    start = time.time()
    
    @retry(
        stop=stop_after_attempt(2),
        wait=wait_exponential(multiplier=0.5, min=0.5, max=3),
        retry=retry_if_exception_type((Exception,)),
        reraise=True
    )
    def _do_call():
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            max_tokens=max_tokens,
            temperature=0.0,   # deterministic — same question, same SQL
        )
    
    try:
        resp = _do_call()
        latency = int((time.time() - start) * 1000)
        text = resp.choices[0].message.content.strip()
        usage = resp.usage
        _log_call(provider_name, model, task,
                  usage.prompt_tokens if usage else 0,
                  usage.completion_tokens if usage else 0,
                  latency, True)
        _record_success(provider_name)
        print(f"[LLM Router] {provider_name}/{model} OK ({latency}ms)")
        return text
    except Exception as e:
        _mark_failed(provider_name)
        latency = int((time.time() - start) * 1000)
        err_str = str(e)
        # Permanent errors (bad API key, model gone, no credits) — open circuit now
        # so we don't waste retry budget on them
        if any(code in err_str for code in ["402", "401", "model_not_found",
                                             "model_decommissioned", "invalid_api_key"]):
            _circuit_breaker[provider_name] = {
                "failures": _CIRCUIT_BREAKER_THRESHOLD,
                "skip_until": time.time() + _CIRCUIT_BREAKER_COOLDOWN
            }
            print(f"[LLM Router] {provider_name} permanent error — circuit opened: {e}")
        _log_call(provider_name, model, task, 0, 0, latency, False, err_str)
        _record_failure(provider_name)
        print(f"[LLM Router] {provider_name}/{model} failed: {e}")
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
    if _today_calls_cached("gemini") >= 1500:
        print("[LLM Router] Gemini daily limit reached (1500)")
        return None

    start = time.time()
    try:
        client = genai.Client(api_key=api_key)
        prompt = f"{system}\n\n{user}"
        resp = client.models.generate_content(model="gemini-3.6-flash", contents=prompt)
        latency = int((time.time() - start) * 1000)
        text = resp.text.strip() if hasattr(resp, "text") and resp.text else ""
        _log_call("gemini", "gemini-3.6-flash", task, 0, 0, latency, True)
        print(f"[LLM Router] gemini OK ({latency}ms)")
        return text
    except Exception as e:
        latency = int((time.time() - start) * 1000)
        err_str = str(e)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
            print("[LLM Router] gemini 429 quota exhausted — cooling down 60s")
            _provider_health["gemini"] = time.time()  # mark unhealthy immediately
        _log_call("gemini", "gemini-3.6-flash", task, 0, 0, latency, False, err_str)
        print(f"[LLM Router] gemini failed: {e}")
        return None


# ── Public router ─────────────────────────────────────────────────────────────

def call_llm(system: str, user: str,
             task: str = "sql_generation",
             max_tokens: int = 512,
             model_override: Optional[str] = None) -> tuple[str, str]:
    """
    Routes to best available free provider for the given task.
    Returns (response_text, provider_name_used).

    task options:
      "sql_generation"       → Groq primary (fastest, best structured output)
      "sql_generation_smart" → same providers but with model_override for Sonnet/70B
      "hitl_explain"         → Gemini primary (most quota, long context)
      "batch_eval"           → Cerebras primary (most tokens/day)
      "fallback"             → OpenRouter

    model_override: if set, forces a specific model string on each provider attempt.
    Falls through providers automatically if one fails or hits limits.
    """
    TASK_ORDER = {
        # Short prompt, JSON output → fastest model with most quota on Groq
        "sql_generation":  ["groq_smart", "groq_fast", "cerebras", "gemini"],
    
        # Long conversation history → highest context (Gemini 1M ctx)
        "multiturn":       ["gemini", "groq_smart", "cerebras"],
    
        # Reasoning tasks → large model first
        "hitl_explain":    ["groq_smart", "gemini", "cerebras"],
    
        # Bulk, stateless, short → highest throughput
        "batch_eval":      ["cerebras", "groq_fast"],
    
        # Similarity/conflict (very short prompt) → smallest fast model
        "conflict_check":  ["groq_fast", "groq_smart"],
    }

    ordered = TASK_ORDER.get(task, ["groq_smart", "groq_fast", "gemini", "cerebras"])

    for provider in ordered:
        if provider == "gemini":
            # Gemini does not support model_override via its SDK in this wrapper
            result = _call_gemini(system, user, max_tokens, task)
        else:
            result = _call_openai_compat(provider, system, user, max_tokens, task,
                                         model_override=model_override)

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


def call_llm_stream(system, user, task="sql_generation", max_tokens=512):
    """Yields text chunks as they arrive from Groq."""
    cfg = PROVIDERS.get("groq", {})
    key = os.getenv(cfg.get("api_key_env",""), "")
    if not key:
        yield json.dumps({"sql":None,"confidence":"low","explanation":"No API key"})
        return
    client = OpenAI(api_key=key, base_url=cfg["base_url"])
    stream = client.chat.completions.create(
        model=cfg["model"], temperature=0.0, max_tokens=max_tokens, stream=True,
        messages=[{"role":"system","content":system},{"role":"user","content":user}]
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        if delta:
            yield delta


if __name__ == "__main__":
    print("=== LLM Router Test ===\n")
    system = "You are a helpful assistant. Reply in one sentence."
    user = "What is 2 + 2?"
    resp, provider = call_llm(system, user, task="general", max_tokens=64)
    print(f"Provider used: {provider}")
    print(f"Response: {resp}")
    print(f"\nUsage stats: {json.dumps(get_usage_stats(), indent=2)}")