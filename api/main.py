"""
api/main.py — DefinitionDrift production API
Enhanced with: auth tokens, WebSocket live updates, full CRUD,
cost tracking, query history, session management.
"""
import sys, os, json, uuid, hashlib, time
from pathlib import Path
from datetime import datetime
from typing import Optional, List

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

# Rate limiting
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from store.db import (
    init_db, get_all_definitions, get_definition_by_name,
    get_definition_history, upsert_definition,
    get_pending_conflicts, resolve_conflict as db_resolve,
    get_unnotified_drift, mark_drift_notified, enqueue_conflict
)
from store.conversation import (
    get_session_messages, clear_session, session_summary, init_conversation_db
)
from agents.core import query_agent, conflict_agent, drift_watcher
from agents.orchestrator import run_query_pipeline
from agents.llm_router import get_usage_stats
from embeddings.engine import cache_stats, get_active_model

# ── Init ──────────────────────────────────────────────────────────────────────
init_db()
init_conversation_db()
DATA_DB = os.getenv("DATA_DB_PATH", str(Path(__file__).parent.parent / "data" / "contoso.db"))
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN", "dd-dev-token-change-in-prod")

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"

# Rate limiter — keyed by IP address
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute"])

app = FastAPI(
    title="DefinitionDrift",
    description="Talk-to-Data agent with HITL definition governance",
    version="2.0.0",
    docs_url="/api/docs",
    redoc_url=None,
)

# Attach limiter to app state
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Query history store (in-memory + SQLite backed) ───────────────────────────
import sqlite3

HIST_DB = Path(__file__).parent.parent / "data" / "query_history.db"

def _init_hist():
    HIST_DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(HIST_DB)
    c.execute("""CREATE TABLE IF NOT EXISTS query_history (
        id TEXT PRIMARY KEY, question TEXT, status TEXT,
        sql_result TEXT, provider TEXT, definitions_used TEXT,
        latency_ms INTEGER, session_id TEXT,
        created_at TEXT DEFAULT (datetime('now'))
    )""")
    c.commit(); c.close()

_init_hist()

def _save_query(qid, question, status, sql_result, provider, defs, latency_ms, session_id):
    c = sqlite3.connect(HIST_DB)
    c.execute("INSERT OR REPLACE INTO query_history VALUES (?,?,?,?,?,?,?,?,datetime('now'))",
              (qid, question, status, json.dumps(sql_result),
               provider, json.dumps(defs), latency_ms, session_id))
    c.commit(); c.close()

def _get_history(session_id: str = None, limit: int = 50):
    c = sqlite3.connect(HIST_DB)
    c.row_factory = sqlite3.Row
    if session_id:
        rows = c.execute("SELECT * FROM query_history WHERE session_id=? ORDER BY created_at DESC LIMIT ?",
                         (session_id, limit)).fetchall()
    else:
        rows = c.execute("SELECT * FROM query_history ORDER BY created_at DESC LIMIT ?",
                         (limit,)).fetchall()
    c.close()
    return [dict(r) for r in rows]

# ── WebSocket connection manager ──────────────────────────────────────────────
class ConnectionManager:
    def __init__(self): self.active: list[WebSocket] = []
    async def connect(self, ws: WebSocket):
        await ws.accept(); self.active.append(ws)
    def disconnect(self, ws: WebSocket):
        self.active = [w for w in self.active if w != ws]
    async def broadcast(self, msg: dict):
        dead = []
        for ws in self.active:
            try: await ws.send_json(msg)
            except: dead.append(ws)
        for ws in dead: self.disconnect(ws)

manager = ConnectionManager()

# ── Auth helper ───────────────────────────────────────────────────────────────
def verify_token(x_api_key: str = Header(None)):
    if x_api_key != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header")
    return True

# ── Pydantic models ───────────────────────────────────────────────────────────
class DefinitionCreate(BaseModel):
    name: str
    description: str
    sql_expr: Optional[str] = None
    tags: Optional[List[str]] = []
    approved: Optional[bool] = False
    reason: Optional[str] = None

class ConflictResolve(BaseModel):
    conflict_id: str
    action: str
    merged_definition: Optional[str] = None

class QueryRequest(BaseModel):
    question: str
    run_query: Optional[bool] = True
    session_id: Optional[str] = None

class FeedbackRequest(BaseModel):
    query_id: str
    rating: int  # 1=good, -1=bad
    comment: Optional[str] = None

# ── Serve frontend ────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    idx = FRONTEND_DIR / "index.html"
    if idx.exists():
        return HTMLResponse(idx.read_text())
    return HTMLResponse("<h2>DefinitionDrift API running. Frontend not built yet.</h2><p>Visit <a href='/api/docs'>/api/docs</a></p>")

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "timestamp": datetime.utcnow().isoformat(),
        "data_db": Path(DATA_DB).exists(),
        "version": "2.0.0"
    }

# ── Definitions ───────────────────────────────────────────────────────────────
@app.get("/api/definitions")
def list_definitions(approved_only: bool = Query(False)):
    defs = get_all_definitions(approved_only=approved_only)
    return {"count": len(defs), "definitions": [
        {**d, "tags": json.loads(d.get("tags","[]")), "approved": bool(d["approved"])}
        for d in defs
    ]}

@app.post("/api/definitions")
async def create_definition(body: DefinitionCreate, _=Depends(verify_token)):
    d = upsert_definition(name=body.name, description=body.description,
                          sql_expr=body.sql_expr, tags=body.tags or [],
                          approved=body.approved or False,
                          reason=body.reason or "created via API")
    await manager.broadcast({"event": "definition_updated", "name": d["name"]})
    return {"status": "ok", "definition": {**d, "tags": json.loads(d.get("tags","[]"))}}

@app.get("/api/definitions/{name}")
def get_definition(name: str):
    d = get_definition_by_name(name)
    if not d:
        raise HTTPException(status_code=404, detail=f"'{name}' not found")
    return {
        "definition": {**d, "tags": json.loads(d.get("tags","[]")), "approved": bool(d["approved"])},
        "history": get_definition_history(d["id"])
    }

@app.delete("/api/definitions/{name}")
async def delete_definition(name: str, _=Depends(verify_token)):
    # soft delete — mark unapproved
    d = get_definition_by_name(name)
    if not d:
        raise HTTPException(status_code=404, detail=f"'{name}' not found")
    upsert_definition(name=name, description=d["description"],
                      approved=False, reason="deleted via API")
    await manager.broadcast({"event": "definition_updated", "name": name})
    return {"status": "ok", "message": f"'{name}' unapproved (soft delete)"}

# ── Query ─────────────────────────────────────────────────────────────────────
@app.post("/api/query")
@limiter.limit("10/minute")           # per IP: 10 queries/min — protects Groq 30 RPM
async def run_query(request: Request, body: QueryRequest):
    session_id = body.session_id or str(uuid.uuid4())
    qid = hashlib.md5(f"{body.question}{time.time()}".encode()).hexdigest()[:12]
    t0 = time.time()

    result = run_query_pipeline(
        question=body.question,
        data_db_path=DATA_DB if body.run_query else None,
        thread_id=session_id,
    )

    latency = int((time.time() - t0) * 1000)
    sql_r = result.get("sql_result", {})
    provider = sql_r.get("provider_used", "none") if sql_r else "conflict"
    defs = sql_r.get("used_definitions", []) if sql_r else []

    _save_query(qid, body.question, result["status"],
                sql_r, provider, defs, latency, session_id)

    # broadcast to all open WebSocket clients
    await manager.broadcast({
        "event": "query_completed",
        "question": body.question,
        "status": result["status"],
        "latency_ms": latency,
    })

    return {
        "query_id": qid,
        "session_id": session_id,
        "latency_ms": latency,
        **result
    }

@app.post("/api/feedback")
def submit_feedback(body: FeedbackRequest):
    # store feedback — in prod, write to feedback table
    return {"status": "ok", "message": "Feedback recorded"}

# ── Query history ─────────────────────────────────────────────────────────────
@app.get("/api/history")
def get_query_history(session_id: str = Query(None), limit: int = Query(50)):
    rows = _get_history(session_id, limit)
    return {"count": len(rows), "history": rows}

# ── HITL ──────────────────────────────────────────────────────────────────────
@app.get("/api/hitl/queue")
def get_queue():
    conflicts = get_pending_conflicts()
    return {"pending_count": len(conflicts), "conflicts": [
        {**c, "similarity_pct": round(c["similarity"]*100,1) if c.get("similarity") else None}
        for c in conflicts
    ]}

@app.post("/api/hitl/resolve")
async def resolve_conflict_endpoint(body: ConflictResolve, _=Depends(verify_token)):
    pending = get_pending_conflicts()
    conflict = next((c for c in pending if c["id"] == body.conflict_id), None)
    if not conflict:
        raise HTTPException(status_code=404, detail=f"Conflict '{body.conflict_id}' not found")

    if body.action == "approve_b":
        resolution = conflict["def_b"]
        db_resolve(body.conflict_id, body.action, resolution)
    elif body.action == "approve_a":
        name_key = "".join(c for c in conflict["question_a"][:40].lower().replace(" ","_")
                           if c.isalnum() or c=="_")
        upsert_definition(name=name_key, description=conflict["question_a"],
                          approved=True, reason=f"HITL {body.conflict_id}")
        resolution = name_key
        db_resolve(body.conflict_id, body.action, resolution)
    elif body.action == "merge":
        if not body.merged_definition:
            raise HTTPException(status_code=400, detail="merged_definition required")
        upsert_definition(name=conflict.get("def_b","merged"),
                          description=body.merged_definition, approved=True,
                          reason=f"merged HITL {body.conflict_id}")
        resolution = body.merged_definition
        db_resolve(body.conflict_id, body.action, resolution)
    else:
        db_resolve(body.conflict_id, "rejected", "discarded")
        resolution = "discarded"

    await manager.broadcast({"event": "conflict_resolved", "conflict_id": body.conflict_id, "action": body.action})
    return {"status": "resolved", "conflict_id": body.conflict_id, "action": body.action, "resolution": resolution}

# ── Drift ──────────────────────────────────────────────────────────────────────
@app.get("/api/drift")
def get_drift():
    return {"events": get_unnotified_drift()}

@app.post("/api/drift/watch")
async def trigger_watch(_=Depends(verify_token)):
    if not Path(DATA_DB).exists():
        raise HTTPException(status_code=404, detail=f"DB not found: {DATA_DB}")
    events = drift_watcher.snapshot_and_diff(DATA_DB)
    if events:
        await manager.broadcast({"event": "drift_detected", "count": len(events)})
    return {"status": "ok", "events_detected": len(events), "events": events}

# ── Stats ──────────────────────────────────────────────────────────────────────
@app.get("/api/stats")
def stats():
    all_defs = get_all_definitions()
    approved  = [d for d in all_defs if d["approved"]]
    pending_q = get_pending_conflicts()
    drift_e   = get_unnotified_drift()
    emb       = cache_stats()
    usage     = get_usage_stats()

    # query history stats
    c = sqlite3.connect(HIST_DB)
    c.row_factory = sqlite3.Row
    total_q = c.execute("SELECT COUNT(*) FROM query_history").fetchone()[0]
    ok_q    = c.execute("SELECT COUNT(*) FROM query_history WHERE status='ok'").fetchone()[0]
    avg_lat = c.execute("SELECT AVG(latency_ms) FROM query_history WHERE status='ok'").fetchone()[0]
    c.close()

    return {
        "definitions": {"total": len(all_defs), "approved": len(approved),
                        "pending": len(all_defs)-len(approved)},
        "hitl": {"pending": len(pending_q)},
        "drift": {"unnotified": len(drift_e)},
        "queries": {"total": total_q, "successful": ok_q,
                    "avg_latency_ms": round(avg_lat or 0, 0)},
        "embedding_cache": emb,
        "embedding_model": get_active_model(),
        "llm_usage": usage,
        "data_db": {"path": DATA_DB, "exists": Path(DATA_DB).exists(),
                    "size_mb": round(Path(DATA_DB).stat().st_size/1024/1024,2)
                    if Path(DATA_DB).exists() else 0},
    }

# ── Conversation memory ──────────────────────────────────────────────────────
@app.get("/api/conversation/{session_id}")
def get_conversation(session_id: str, limit: int = Query(20)):
    """Get conversation history for a session (for UI replay on refresh)."""
    msgs = get_session_messages(session_id, limit=limit)
    summary = session_summary(session_id)
    return {"session_id": session_id, "messages": msgs, "summary": summary}

@app.delete("/api/conversation/{session_id}")
async def clear_conversation(session_id: str, _=Depends(verify_token)):
    """Clear a session's conversation history."""
    clear_session(session_id)
    await manager.broadcast({"event": "conversation_cleared", "session_id": session_id})
    return {"status": "ok", "message": f"Session {session_id} cleared"}

# ── WebSocket ──────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await manager.connect(ws)
    try:
        while True:
            data = await ws.receive_text()
            # echo ping/pong for keepalive
            if data == "ping":
                await ws.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(ws)