"""Test: what happens when querying with zero definitions."""
import sys, json
sys.path.insert(0, ".")

from store.db import init_db, get_all_definitions
from agents.core import optimizer, query_agent

init_db()

# 1. Check existing definitions
all_defs = get_all_definitions()
approved = [d for d in all_defs if d["approved"]]
print(f"=== Current Definitions ===")
print(f"  Total: {len(all_defs)}, Approved: {len(approved)}")
for d in all_defs[:10]:
    print(f"  [{d['id'][:8]}] name='{d['name']}' approved={d['approved']} desc='{d['description'][:60]}...'")

# 2. Test build_context_block with a simple question
print(f"\n=== Token Optimizer Test ===")
test_q = "What was total sales in 2008?"
ctx_block, used = optimizer.build_context_block(test_q)
print(f"  Question: {test_q}")
print(f"  Definitions injected: {len(used)}")
print(f"  Context block:\n{ctx_block[:200] if ctx_block else '(empty)'}")

# 3. Test with a question that has no definitions at all
test_q2 = "How many customers are from California?"
ctx_block2, used2 = optimizer.build_context_block(test_q2)
print(f"\n  Question: {test_q2}")
print(f"  Definitions injected: {len(used2)}")
print(f"  Context block:\n{ctx_block2[:200] if ctx_block2 else '(empty)'}")

# 4. Show what the LLM system prompt looks like
print(f"\n=== QueryAgent System Prompt ===")
print(query_agent.SYSTEM_PROMPT[:500])
