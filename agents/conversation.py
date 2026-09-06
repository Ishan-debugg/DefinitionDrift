"""
store/conversation.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Multi-turn conversation memory for DefinitionDrift
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Each session_id gets its own conversation thread.
Messages are stored in SQLite and retrieved per-session.

The memory serves two purposes:
  1. LLM context  — last N turns are injected into the prompt
                    so "show me the same for Q3" resolves correctly
  2. UI history   — the chat panel replays the session on refresh

Message roles: "user" | "assistant" | "system"
Message types: "question" | "sql_result" | "conflict" | "error"
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import Optional

CONV_DB = Path(__file__).parent.parent / "data" / "conversations.db"


def _conn() -> sqlite3.Connection:
    CONV_DB.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(CONV_DB, check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")
    return c


def init_conversation_db():
    c = _conn()
    c.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            session_id  TEXT NOT NULL,
            msg_id      TEXT NOT NULL,
            role        TEXT NOT NULL,
            msg_type    TEXT NOT NULL DEFAULT 'question',
            content     TEXT NOT NULL,
            sql_result  TEXT,
            metadata    TEXT DEFAULT '{}',
            created_at  TEXT DEFAULT (datetime('now')),
            PRIMARY KEY (session_id, msg_id)
        );
        CREATE INDEX IF NOT EXISTS idx_conv_session
            ON conversations(session_id, created_at);
    """)
    c.commit()
    c.close()


init_conversation_db()


# ── Write ─────────────────────────────────────────────────────────────────────

def add_message(
    session_id: str,
    role: str,
    content: str,
    msg_type: str = "question",
    sql_result: Optional[dict] = None,
    metadata: Optional[dict] = None,
) -> str:
    import hashlib, time
    msg_id = hashlib.md5(f"{session_id}{content}{time.time()}".encode()).hexdigest()[:12]
    c = _conn()
    c.execute("""
        INSERT INTO conversations (session_id, msg_id, role, msg_type, content, sql_result, metadata)
        VALUES (?,?,?,?,?,?,?)
    """, (
        session_id, msg_id, role, msg_type, content,
        json.dumps(sql_result) if sql_result else None,
        json.dumps(metadata or {})
    ))
    c.commit()
    c.close()
    return msg_id


# ── Read ──────────────────────────────────────────────────────────────────────

def get_session_messages(session_id: str, limit: int = 20) -> list[dict]:
    """Returns the last N messages for a session, oldest first."""
    c = _conn()
    rows = c.execute("""
        SELECT * FROM conversations
        WHERE session_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    """, (session_id, limit)).fetchall()
    c.close()
    return list(reversed([dict(r) for r in rows]))


def get_conversation_context(session_id: str, max_turns: int = 6) -> list[dict]:
    """
    Returns the last max_turns question+answer pairs formatted
    for LLM context injection.

    Format:
      [
        {"role": "user",      "content": "What is net revenue?"},
        {"role": "assistant", "content": "net_revenue = SalesAmount - ReturnAmount. SQL: SELECT ..."},
        ...
      ]

    Only includes question/sql_result pairs — not conflict or error messages.
    This keeps the context small and relevant.
    """
    msgs = get_session_messages(session_id, limit=max_turns * 2)
    context = []
    for m in msgs:
        if m["role"] == "user" and m["msg_type"] == "question":
            context.append({"role": "user", "content": m["content"]})
        elif m["role"] == "assistant" and m["msg_type"] == "sql_result":
            sr = json.loads(m["sql_result"]) if m["sql_result"] else {}
            # Compact assistant turn: explanation + SQL only (no raw rows)
            assistant_text = m["content"]
            if sr.get("sql"):
                assistant_text += f"\nSQL: {sr['sql']}"
            if sr.get("used_definitions"):
                assistant_text += f"\nDefinitions used: {', '.join(sr['used_definitions'])}"
            context.append({"role": "assistant", "content": assistant_text})
    return context


def clear_session(session_id: str):
    c = _conn()
    c.execute("DELETE FROM conversations WHERE session_id=?", (session_id,))
    c.commit()
    c.close()


def session_summary(session_id: str) -> dict:
    c = _conn()
    total = c.execute(
        "SELECT COUNT(*) FROM conversations WHERE session_id=?", (session_id,)
    ).fetchone()[0]
    last = c.execute(
        "SELECT created_at FROM conversations WHERE session_id=? ORDER BY created_at DESC LIMIT 1",
        (session_id,)
    ).fetchone()
    c.close()
    return {
        "session_id": session_id,
        "total_messages": total,
        "last_message_at": last[0] if last else None,
    }


if __name__ == "__main__":
    sid = "test-session-001"

    # Simulate a multi-turn conversation
    add_message(sid, "user", "What is net revenue for 2008?", "question")
    add_message(sid, "assistant",
                "Net revenue for 2008 was $4.2M after returns.",
                "sql_result",
                sql_result={"sql": "SELECT SUM(SalesAmount-ReturnAmount) FROM FactSales JOIN DimDate ON FactSales.DateKey=DimDate.DateKey WHERE CalendarYear=2008",
                            "used_definitions": ["net_revenue"], "confidence": "high"})
    add_message(sid, "user", "What about Q3 of the same year?", "question")

    ctx = get_conversation_context(sid)
    print("Context for LLM:")
    for m in ctx:
        print(f"  [{m['role']}] {m['content'][:80]}")

    print(f"\nSession summary: {session_summary(sid)}")