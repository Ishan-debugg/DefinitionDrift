"""
IntentRouter — classifies user intent before routing to agents.

Intent classes:
  DATA   — user wants to query/retrieve data
           "What is net revenue?" / "Show top 5 stores" / "How many units sold?"
  DEFINE — user wants to create/modify/name a metric
           "Define active users as..." / "Let's call this gross margin" /
           "I want to track churn as..." / "Update the revenue definition"
  AMBIGUOUS — could be either (gets a lightweight conflict check at 0.65)

Classification is done with a tiny prompt to the fastest available model
(Groq llama-3.1-8b). Cost: ~50 input tokens, ~5 output tokens = $0.00006/call.
Latency: ~80ms. Cheaper than running ConflictAgent on every query.
"""

import os, json, re
from typing import Literal
from agents.llm_router import call_llm

IntentType = Literal["DATA", "DEFINE", "AMBIGUOUS"]

# Keywords that strongly signal DEFINE intent — used as a fast pre-filter
# before calling the LLM classifier. If any match, skip LLM entirely.
DEFINE_SIGNALS = [
    "define", "definition", "let's call", "we should call", "i want to call",
    "create a metric", "add a metric", "new metric", "track as", "measure as",
    "update the definition", "change the definition", "rename", "what should we call",
    "let's say", "going forward", "from now on", "standardize", "canonical",
    "how do we define", "how should we define",
]

# Keywords that strongly signal DATA intent — skip LLM if matched
DATA_SIGNALS = [
    "what is", "what was", "what are", "show me", "how many", "how much",
    "total", "sum", "average", "count", "list", "give me", "get me",
    "top", "bottom", "highest", "lowest", "compare", "breakdown",
    "chart", "report", "between", "for 2008", "last month", "this quarter",
    "by category", "by store", "by region", "per customer",
]

INTENT_SYSTEM = """\
You are an intent classifier for a data analytics system.
Classify the user's message into exactly one of three categories:

DATA     — The user wants to retrieve, query, or analyze existing data.
           Examples: "What is net revenue?", "Show top 5 stores", "How many returns last month?"

DEFINE   — The user wants to create, modify, rename, or establish a new metric definition.
           Examples: "Define active users as...", "Let's track churn as 30-day no-login",
                     "Update the revenue definition", "What should we call this metric?"

AMBIGUOUS — Could be either, or unclear.
           Examples: "What is active users?", "Tell me about gross margin"

Reply with ONLY one word: DATA, DEFINE, or AMBIGUOUS.
No explanation. No punctuation. Just the word."""


class IntentRouter:
    """
    Fast 3-class intent classifier.
    Uses keyword pre-filter first (0ms), LLM only if ambiguous.
    """

    def classify(self, question: str) -> tuple[IntentType, str]:
        """
        Returns (intent, method) where method is how it was classified:
          "keyword_define", "keyword_data", "llm", "default"
        """
        q_lower = question.lower().strip()

        # Fast path 1 — keyword DEFINE signal
        for signal in DEFINE_SIGNALS:
            if signal in q_lower:
                return "DEFINE", "keyword_define"

        # Fast path 2 — keyword DATA signal
        # Only use if no DEFINE signals present (already checked above)
        data_hits = sum(1 for s in DATA_SIGNALS if s in q_lower)
        if data_hits >= 2:
            return "DATA", "keyword_data"

        # Slow path — LLM classifier for ambiguous cases
        # Uses cheapest/fastest model: groq_fast (8B model)
        try:
            raw, provider = call_llm(
                system=INTENT_SYSTEM,
                user=question,
                task="conflict_check",
                max_tokens=5,
            )
            intent_str = raw.strip().upper().replace(".", "")
            if intent_str in ("DATA", "DEFINE", "AMBIGUOUS"):
                return intent_str, f"llm:{provider}"
        except Exception:
            pass

        # Fallback — default to DATA (conservative, avoids false DEFINE blocks)
        return "DATA", "default"


intent_router = IntentRouter()


# ── Threshold config per intent ───────────────────────────────────────────────
CONFLICT_THRESHOLD_BY_INTENT = {
    "DEFINE":    0.40,   # user explicitly defining — high recall, catch everything
    "AMBIGUOUS": 0.60,   # might be defining — moderate sensitivity
    "DATA":      None,   # skip conflict check entirely
}
