"""
config/settings.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Central config for DefinitionDrift — all tunable values in one place
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
from pathlib import Path

ROOT = Path(__file__).parent.parent

# ── LLM ───────────────────────────────────────────────────────────────────────
ANTHROPIC_API_KEY   = os.getenv("ANTHROPIC_API_KEY", "")

# Model routing (zero purchase — Haiku for 95% of calls)
# Switch to Sonnet when confidence < LOW_CONFIDENCE_THRESHOLD
QUERY_MODEL_FAST    = "claude-haiku-4-5-20251001"   # $1/$5 per MTok
QUERY_MODEL_SMART   = "claude-sonnet-4-6"            # $3/$15 per MTok — for low-confidence queries
LOW_CONFIDENCE_THRESHOLD = 0.60                      # below this → escalate to Sonnet

MAX_TOKENS_QUERY    = 512
MAX_TOKENS_EXPLAIN  = 256

# ── EMBEDDING ──────────────────────────────────────────────────────────────────
# Priority: local → haiku → charfreq
EMBEDDING_MODEL_LOCAL = "all-MiniLM-L6-v2"  # free, runs on CPU
EMBEDDING_DIM_LOCAL   = 384
EMBEDDING_DIM_HAIKU   = 64
EMBEDDING_CACHE_DB    = ROOT / "embeddings" / "cache.db"

# ── CONFLICT DETECTION ────────────────────────────────────────────────────────
CONFLICT_SIMILARITY_THRESHOLD = 0.82   # above this → HITL queue
OPTIMIZER_SIMILARITY_THRESHOLD = 0.45  # above this → inject into context
OPTIMIZER_TOP_K = 4                    # max definitions to inject per query

# ── DATABASES ──────────────────────────────────────────────────────────────────
DEFINITION_DB  = ROOT / "definitiondrift.db"
DATA_DB_PATH   = Path(os.getenv("DATA_DB_PATH", str(ROOT / "data" / "contoso.db")))
EMBEDDING_CACHE = ROOT / "embeddings" / "cache.db"

# ── API ───────────────────────────────────────────────────────────────────────
API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_RELOAD = os.getenv("ENV", "dev") == "dev"

# ── COST TRACKING ─────────────────────────────────────────────────────────────
# Used to estimate running costs in /api/stats
COST_PER_1K_INPUT_HAIKU  = 0.001    # $1/MTok = $0.001/KTok
COST_PER_1K_OUTPUT_HAIKU = 0.005
COST_PER_1K_INPUT_SONNET = 0.003
COST_PER_1K_OUTPUT_SONNET = 0.015

# ── DRIFT WATCHER ─────────────────────────────────────────────────────────────
DRIFT_CRON_SCHEDULE = "0 9 * * 1"  # every Monday 9am (HOTL digest)
DRIFT_WEBHOOK_URL   = os.getenv("DRIFT_WEBHOOK_URL", "")  # Slack/Discord webhook

# ── FEATURE FLAGS ─────────────────────────────────────────────────────────────
ENABLE_PROMPT_CACHING = True   # saves 90% on repeated system prompts
ENABLE_BATCH_API      = False  # enable for eval runs (50% off, async)
ENABLE_LOCAL_EMBED    = True   # False to force Haiku embed (testing)
