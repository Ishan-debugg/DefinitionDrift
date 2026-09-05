"""
agents/orchestrator.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LangGraph HITL orchestrator for DefinitionDrift
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

State machine:
  START
    │
    ▼
  [check_conflict]  ──conflict──▶  [hitl_interrupt]  ◀── human resolves
    │ clean                              │ resumed
    ▼                                    ▼
  [run_query]  ◀──────────────── [run_query]
    │
    ▼
  [check_drift]
    │
    ▼
  END

HITL interrupt:
  - Graph pauses at hitl_interrupt node
  - Writes conflict to DB queue
  - Resumes when /api/hitl/resolve is called with the conflict_id
  - LangGraph checkpoint stores full state across the pause
"""

import os, json
from typing import TypedDict, Optional, Annotated
from pathlib import Path

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from agents.core import conflict_agent, query_agent, drift_watcher
from store.db import get_pending_conflicts, resolve_conflict


# ── State schema ──────────────────────────────────────────────────────────────
class QueryState(TypedDict):
    question:       str
    data_db_path:   Optional[str]
    conflict:       Optional[dict]
    conflict_id:    Optional[str]
    hitl_resolved:  bool
    sql_result:     Optional[dict]
    drift_events:   list[dict]
    error:          Optional[str]
    step_log:       list[str]


# ── Nodes ─────────────────────────────────────────────────────────────────────

def check_conflict_node(state: QueryState) -> QueryState:
    """Pass 1: detect if question conflicts with existing definitions."""
    log = state.get("step_log", [])
    log.append("check_conflict: running semantic similarity check")

    result = conflict_agent.check(state["question"])

    if result:
        log.append(f"check_conflict: conflict detected (sim={result['similarity']}) → HITL")
        return {**state,
                "conflict": result,
                "conflict_id": result["conflict_id"],
                "step_log": log}
    else:
        log.append("check_conflict: clean — proceeding to SQL generation")
        return {**state, "conflict": None, "step_log": log}


def hitl_interrupt_node(state: QueryState) -> QueryState:
    """
    HITL gate. In production this node:
      - Returns the conflict to the caller immediately
      - The graph checkpoint is saved here
      - When /api/hitl/resolve is called, the graph resumes from this checkpoint
    For CLI/testing: waits for keyboard input.
    """
    log = state.get("step_log", [])
    conflict = state["conflict"]
    log.append(f"hitl_interrupt: paused at conflict {state['conflict_id'][:8]}")
    log.append("hitl_interrupt: waiting for human resolution via /api/hitl/resolve")

    # In interactive mode (CLI), prompt immediately
    if os.getenv("DD_INTERACTIVE", "0") == "1":
        print(f"\n[HITL] Conflict detected:")
        print(f"  Question:    {state['question']}")
        print(f"  Matched def: {conflict['matched_definition']}")
        print(f"  Similarity:  {conflict['similarity']}")
        print(f"  Conflict ID: {state['conflict_id']}")
        choice = input("\n  Proceed anyway? [y/N]: ").strip().lower()
        if choice == "y":
            log.append("hitl_interrupt: human approved — resuming query")
            return {**state, "hitl_resolved": True, "step_log": log}

    # Non-interactive: block the query, surface to caller
    return {**state, "hitl_resolved": False, "step_log": log}


def run_query_node(state: QueryState) -> QueryState:
    """Pass 2: generate SQL via free LLM and optionally execute."""
    log = state.get("step_log", [])
    log.append("run_query: generating SQL via LLM router")

    result = query_agent.run(
        question=state["question"],
        data_db_path=state.get("data_db_path"),
    )

    log.append(f"run_query: done (provider={result.get('provider_used')}, "
               f"confidence={result.get('confidence')})")
    return {**state, "sql_result": result, "step_log": log}


def check_drift_node(state: QueryState) -> QueryState:
    """After query: snapshot schema and detect drift."""
    log = state.get("step_log", [])
    db_path = state.get("data_db_path")

    if db_path and Path(db_path).exists():
        log.append("check_drift: running schema snapshot diff")
        events = drift_watcher.snapshot_and_diff(db_path)
        if events:
            log.append(f"check_drift: {len(events)} drift event(s) detected")
        else:
            log.append("check_drift: no drift detected")
        return {**state, "drift_events": events, "step_log": log}

    log.append("check_drift: skipped (no data_db_path)")
    return {**state, "drift_events": [], "step_log": log}


# ── Routing ───────────────────────────────────────────────────────────────────

def route_after_conflict(state: QueryState) -> str:
    """Route: conflict → hitl_interrupt, clean → run_query."""
    if state.get("conflict"):
        return "hitl_interrupt"
    return "run_query"


def route_after_hitl(state: QueryState) -> str:
    """Route: resolved → run_query, blocked → END."""
    if state.get("hitl_resolved"):
        return "run_query"
    return END


# ── Build graph ───────────────────────────────────────────────────────────────

def build_graph():
    """Build and compile the LangGraph state machine with memory checkpointing."""
    checkpointer = MemorySaver()

    g = StateGraph(QueryState)

    g.add_node("check_conflict",  check_conflict_node)
    g.add_node("hitl_interrupt",  hitl_interrupt_node)
    g.add_node("run_query",       run_query_node)
    g.add_node("check_drift",     check_drift_node)

    g.add_edge(START,            "check_conflict")
    g.add_conditional_edges("check_conflict",  route_after_conflict,
                             {"hitl_interrupt": "hitl_interrupt", "run_query": "run_query"})
    g.add_conditional_edges("hitl_interrupt",  route_after_hitl,
                             {"run_query": "run_query", END: END})
    g.add_edge("run_query",      "check_drift")
    g.add_edge("check_drift",    END)

    return g.compile(checkpointer=checkpointer)


# ── Public runner ─────────────────────────────────────────────────────────────

_graph = None

def get_graph():
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def run_query_pipeline(question: str,
                       data_db_path: Optional[str] = None,
                       thread_id: str = "default") -> dict:
    """
    Main entry point for the full HITL pipeline.

    Returns:
      {
        "status": "ok" | "conflict_detected" | "error",
        "question": ...,
        "sql_result": {...},        # if status=ok
        "conflict": {...},          # if status=conflict_detected
        "drift_events": [...],
        "step_log": [...]
      }
    """
    graph = get_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: QueryState = {
        "question":      question,
        "data_db_path":  data_db_path,
        "conflict":      None,
        "conflict_id":   None,
        "hitl_resolved": False,
        "sql_result":    None,
        "drift_events":  [],
        "error":         None,
        "step_log":      [],
    }

    try:
        final = graph.invoke(initial_state, config=config)

        if final.get("conflict") and not final.get("hitl_resolved"):
            return {
                "status": "conflict_detected",
                "question": question,
                "conflict": final["conflict"],
                "conflict_id": final["conflict_id"],
                "message": final["conflict"]["message"],
                "action_required": (
                    f"Resolve at: POST /api/hitl/resolve "
                    f"with conflict_id='{final['conflict_id']}'"
                ),
                "drift_events": [],
                "step_log": final.get("step_log", []),
            }

        return {
            "status": "ok",
            "question": question,
            "sql_result": final.get("sql_result"),
            "drift_events": final.get("drift_events", []),
            "step_log": final.get("step_log", []),
        }

    except Exception as e:
        return {"status": "error", "question": question, "error": str(e)}


if __name__ == "__main__":
    from store.db import init_db
    init_db()

    print("=== LangGraph Orchestrator Test ===\n")

    # Test 1: clean query
    result = run_query_pipeline(
        question="What is the total gross sales for 2008?",
        thread_id="test-clean"
    )
    print("Clean query:")
    print(f"  Status: {result['status']}")
    print(f"  Steps:  {result.get('step_log', [])}")
    if result.get("sql_result"):
        print(f"  SQL:    {result['sql_result'].get('sql', 'none')}")
        print(f"  Provider: {result['sql_result'].get('provider_used')}")

    print()

    # Test 2: HITL in interactive mode
    os.environ["DD_INTERACTIVE"] = "0"   # non-interactive for test
    result2 = run_query_pipeline(
        question="how much revenue did we make",
        thread_id="test-conflict"
    )
    print("Potential conflict query:")
    print(f"  Status: {result2['status']}")
    print(f"  Steps:  {result2.get('step_log', [])}")