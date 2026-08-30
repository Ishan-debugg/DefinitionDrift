"""
api/main.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DefinitionDrift FastAPI backend
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Endpoints:
  GET  /health                   — liveness probe
  GET  /api/definitions          — list all definitions
  POST /api/definitions          — create/update a definition
  GET  /api/definitions/{id}     — get one definition + version history
  GET  /api/hitl/queue           — list pending conflicts
  POST /api/hitl/resolve         — resolve a conflict (approve/merge/reject)
  GET  /api/drift                — list unnotified drift events
  POST /api/drift/watch          — trigger schema diff on data DB
  GET  /api/stats                — system stats (token savings, cache, counts)

Run:
  uvicorn api.main:app --reload --port 8000

CORS is open for local dev. Tighten ALLOW_ORIGINS for production.
"""

import sys, os, json
from pathlib import Path
from datetime import datetime
from typing import Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from store.db import (
    init_db, get_all_definitions, get_definition_by_name,
    get_definition_history, upsert_definition,
    get_pending_conflicts, resolve_conflict as db_resolve,
    get_unnotified_drift, mark_drift_notified
)
from agents.core import query_agent, conflict_agent, drift_watcher
from embeddings.engine import cache_stats

# ── Init ───────────────────────────────────────────────────────────────────────
init_db()
DATA_DB = os.getenv("DATA_DB_PATH", str(Path(__file__).parent.parent / "data" / "contoso.db"))

app = FastAPI(
    title="DefinitionDrift API",
    description="Talk-to-Data agent with HITL definition governance",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten in prod
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Pydantic models ────────────────────────────────────────────────────────────
class DefinitionCreate(BaseModel):
    name: str
    description: str
    sql_expr: Optional[str] = None
    tags: Optional[List[str]] = []
    approved: Optional[bool] = False
    reason: Optional[str] = None

class ConflictResolve(BaseModel):
    conflict_id: str
    action: str          # approve_a | approve_b | merge | reject
    merged_definition: Optional[str] = None

class QueryRequest(BaseModel):
    question: str
    run_query: Optional[bool] = False

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


# ── Definitions ───────────────────────────────────────────────────────────────

@app.get("/api/definitions")
def list_definitions(approved_only: bool = Query(False)):
    defs = get_all_definitions(approved_only=approved_only)
    return {
        "count": len(defs),
        "definitions": [
            {
                **d,
                "tags": json.loads(d.get("tags", "[]")),
                "approved": bool(d["approved"])
            }
            for d in defs
        ]
    }

@app.post("/api/definitions")
def create_definition(body: DefinitionCreate):
    d = upsert_definition(
        name=body.name,
        description=body.description,
        sql_expr=body.sql_expr,
        tags=body.tags or [],
        approved=body.approved or False,
        reason=body.reason or "created via API"
    )
    return {"status": "ok", "definition": {**d, "tags": json.loads(d.get("tags", "[]"))}}

@app.get("/api/definitions/{name}")
def get_definition(name: str):
    d = get_definition_by_name(name)
    if not d:
        raise HTTPException(status_code=404, detail=f"Definition '{name}' not found")
    history = get_definition_history(d["id"])
    return {
        "definition": {**d, "tags": json.loads(d.get("tags", "[]")), "approved": bool(d["approved"])},
        "version_count": len(history),
        "history": history
    }


# ── HITL Queue ────────────────────────────────────────────────────────────────

@app.get("/api/hitl/queue")
def get_queue():
    conflicts = get_pending_conflicts()
    return {
        "pending_count": len(conflicts),
        "conflicts": [
            {
                **c,
                "similarity_pct": round(c["similarity"] * 100, 1) if c.get("similarity") else None
            }
            for c in conflicts
        ]
    }

@app.post("/api/hitl/resolve")
def resolve_conflict(body: ConflictResolve):
    pending = get_pending_conflicts()
    conflict = next((c for c in pending if c["id"] == body.conflict_id), None)

    if not conflict:
        raise HTTPException(
            status_code=404,
            detail=f"Conflict '{body.conflict_id}' not found or already resolved"
        )

    if body.action not in ("approve_a", "approve_b", "merge", "reject"):
        raise HTTPException(status_code=400, detail="action must be approve_a|approve_b|merge|reject")

    if body.action == "merge" and not body.merged_definition:
        raise HTTPException(status_code=400, detail="merged_definition required for action=merge")

    # execute resolution
    if body.action == "approve_b":
        resolution = conflict["def_b"]
        db_resolve(body.conflict_id, body.action, resolution)

    elif body.action == "approve_a":
        name_key = conflict["question_a"][:40].lower().replace(" ", "_")
        name_key = "".join(c for c in name_key if c.isalnum() or c == "_")
        upsert_definition(
            name=name_key,
            description=conflict["question_a"],
            approved=True,
            reason=f"approved via HITL conflict {body.conflict_id}"
        )
        resolution = conflict["question_a"]
        db_resolve(body.conflict_id, body.action, resolution)

    elif body.action == "merge":
        name_key = (conflict.get("def_b") or "merged")
        upsert_definition(
            name=name_key,
            description=body.merged_definition,
            approved=True,
            reason=f"merged via HITL conflict {body.conflict_id}"
        )
        resolution = body.merged_definition
        db_resolve(body.conflict_id, body.action, resolution)

    else:  # reject
        db_resolve(body.conflict_id, "rejected", "discarded")
        resolution = "discarded"

    return {
        "status": "resolved",
        "conflict_id": body.conflict_id,
        "action": body.action,
        "resolution": resolution
    }


# ── Query endpoint ────────────────────────────────────────────────────────────

@app.post("/api/query")
def run_query(body: QueryRequest):
    # HITL gate
    conflict = conflict_agent.check(body.question)
    if conflict:
        return {
            "status": "conflict_detected",
            "conflict_id": conflict["conflict_id"],
            "matched_definition": conflict["matched_definition"],
            "similarity": conflict["similarity"],
            "message": conflict["message"],
            "action_required": "Resolve conflict at /api/hitl/resolve before this query can run."
        }

    db_path = DATA_DB if body.run_query else None
    result = query_agent.run(body.question, data_db_path=db_path)
    return {"status": "ok", "question": body.question, **result}


# ── Drift ──────────────────────────────────────────────────────────────────────

@app.get("/api/drift")
def get_drift():
    events = get_unnotified_drift()
    return {
        "unnotified_count": len(events),
        "events": events
    }

@app.post("/api/drift/watch")
def trigger_watch():
    if not Path(DATA_DB).exists():
        raise HTTPException(status_code=404, detail=f"Data DB not found: {DATA_DB}")
    events = drift_watcher.snapshot_and_diff(DATA_DB)
    return {
        "status": "ok",
        "db_path": DATA_DB,
        "events_detected": len(events),
        "events": events
    }

@app.post("/api/drift/acknowledge")
def acknowledge_drift(ids: List[int]):
    mark_drift_notified(ids)
    return {"status": "ok", "acknowledged": ids}


# ── Stats ─────────────────────────────────────────────────────────────────────

@app.get("/api/stats")
def stats():
    all_defs   = get_all_definitions()
    approved   = [d for d in all_defs if d["approved"]]
    pending_q  = get_pending_conflicts()
    drift_evts = get_unnotified_drift()
    emb_cache  = cache_stats()

    return {
        "definitions": {
            "total": len(all_defs),
            "approved": len(approved),
            "pending_approval": len(all_defs) - len(approved)
        },
        "hitl_queue": {
            "pending": len(pending_q)
        },
        "drift": {
            "unnotified_events": len(drift_evts)
        },
        "embedding_cache": emb_cache,
        "data_db": {
            "path": DATA_DB,
            "exists": Path(DATA_DB).exists(),
            "size_mb": round(Path(DATA_DB).stat().st_size / 1024 / 1024, 2) if Path(DATA_DB).exists() else 0
        }
    }