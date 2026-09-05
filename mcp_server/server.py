"""
mcp_server/server.py
DefinitionDrift MCP Server
Exposes 4 tools that any MCP client (Claude.ai, Cursor, your chat UI) can call:

  1. query_data          — NL question → governed SQL + result
  2. list_definitions    — browse all approved definitions
  3. resolve_conflict    — human approves/merges a HITL queue item
  4. watch_schema        — trigger a schema diff on a connected DB

Run:  python mcp_server/server.py
Config for Claude Desktop (~/.claude/claude_desktop_config.json):
{
  "mcpServers": {
    "definitiondrift": {
      "command": "python",
      "args": ["/absolute/path/to/definitiondrift/mcp_server/server.py"]
    }
  }
}
"""

import sys
import os
import json
from pathlib import Path

# make project root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from store.db import (
    init_db, get_all_definitions, get_pending_conflicts,
    resolve_conflict as db_resolve_conflict, upsert_definition
)
from agents.core import query_agent, conflict_agent, drift_watcher

import mcp.server.stdio
import mcp.types as types
from mcp.server import Server

# ── Bootstrap ─────────────────────────────────────────────────────────────────
init_db()

DATA_DB = os.getenv("DATA_DB_PATH", str(Path(__file__).parent.parent / "demo_data.db"))

app = Server("definitiondrift")


# ── Tool definitions ──────────────────────────────────────────────────────────

@app.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(
            name="query_data",
            description=(
                "Ask a natural language question about your data. "
                "DefinitionDrift automatically injects only the relevant approved "
                "metric definitions, generates governed SQL, and returns the result. "
                "Always produces consistent answers — same question = same SQL."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Your data question in plain English, e.g. 'What is total revenue this month?'"
                    },
                    "run_query": {
                        "type": "boolean",
                        "description": "If true, execute the SQL against the connected database and return rows.",
                        "default": False
                    }
                },
                "required": ["question"]
            }
        ),

        types.Tool(
            name="list_definitions",
            description=(
                "List all metric definitions in the DefinitionDrift store. "
                "Shows name, description, SQL expression, approval status, and version. "
                "Use this to understand what governed metrics are available."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "approved_only": {
                        "type": "boolean",
                        "description": "If true, only return human-approved definitions.",
                        "default": True
                    }
                }
            }
        ),

        types.Tool(
            name="resolve_conflict",
            description=(
                "Resolve a pending HITL (human-in-the-loop) conflict in the approval queue. "
                "When two questions are semantically similar, DefinitionDrift pauses and "
                "asks a human to canonicalize the definition. Use this tool to approve, "
                "merge, or reject a queued conflict."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "conflict_id": {
                        "type": "string",
                        "description": "The conflict ID from the HITL queue (e.g. 'a3f9b1c2d4e5')"
                    },
                    "action": {
                        "type": "string",
                        "enum": ["approve_a", "approve_b", "merge", "reject"],
                        "description": (
                            "approve_a: use question_a as the canonical definition. "
                            "approve_b: keep existing definition (question_b). "
                            "merge: create a new combined definition. "
                            "reject: discard the conflict, keep both."
                        )
                    },
                    "merged_definition": {
                        "type": "string",
                        "description": "Required only if action='merge'. The new canonical definition text."
                    }
                },
                "required": ["conflict_id", "action"]
            }
        ),

        types.Tool(
            name="watch_schema",
            description=(
                "Run a schema snapshot diff on the connected database. "
                "Detects added/removed columns, checks which metric definitions "
                "are affected by schema changes, and logs drift events. "
                "Use after any migration or schema change."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "db_path": {
                        "type": "string",
                        "description": "Path to the SQLite database to inspect. Defaults to the configured DATA_DB."
                    }
                }
            }
        ),
    ]


# ── Tool handlers ─────────────────────────────────────────────────────────────

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:

    # ── query_data ────────────────────────────────────────────────────────────
    if name == "query_data":
        question = arguments["question"]
        run_query = arguments.get("run_query", False)

        # Step 1: conflict check (HITL gate)
        conflict = conflict_agent.check(question)
        if conflict:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "conflict_detected",
                    "conflict_id": conflict["conflict_id"],
                    "matched_definition": conflict["matched_definition"],
                    "similarity": conflict["similarity"],
                    "message": conflict["message"],
                    "action_required": (
                        "A data owner must resolve this conflict at "
                        "/api/hitl_queue or via the resolve_conflict tool "
                        "before this query can proceed."
                    )
                }, indent=2)
            )]

        # Step 2: run query agent
        db_path = DATA_DB if run_query else None
        result = query_agent.run(question, data_db_path=db_path)

        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "ok",
                "question": question,
                **result
            }, indent=2)
        )]

    # ── list_definitions ──────────────────────────────────────────────────────
    elif name == "list_definitions":
        approved_only = arguments.get("approved_only", True)
        defs = get_all_definitions(approved_only=approved_only)

        if not defs:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "empty",
                    "message": "No definitions found. Add definitions via the API or store/db.py seed."
                })
            )]

        summary = []
        for d in defs:
            summary.append({
                "id": d["id"],
                "name": d["name"],
                "description": d["description"],
                "sql_expr": d.get("sql_expr"),
                "version": d["version"],
                "approved": bool(d["approved"]),
                "tags": json.loads(d.get("tags", "[]"))
            })

        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "ok",
                "count": len(summary),
                "definitions": summary
            }, indent=2)
        )]

    # ── resolve_conflict ──────────────────────────────────────────────────────
    elif name == "resolve_conflict":
        conflict_id = arguments["conflict_id"]
        action = arguments["action"]
        merged_def = arguments.get("merged_definition")

        pending = get_pending_conflicts()
        conflict = next((c for c in pending if c["id"] == conflict_id), None)

        if not conflict:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "message": f"Conflict ID '{conflict_id}' not found or already resolved."
                })
            )]

        if action == "approve_b":
            resolution = conflict["def_b"]
            db_resolve_conflict(conflict_id, action, resolution)

        elif action == "approve_a":
            upsert_definition(
                name=conflict["question_a"][:40].lower().replace(" ", "_"),
                description=conflict["question_a"],
                approved=True,
                reason=f"approved via HITL resolution of conflict {conflict_id}"
            )
            resolution = conflict["question_a"]
            db_resolve_conflict(conflict_id, action, resolution)

        elif action == "merge":
            if not merged_def:
                return [types.TextContent(
                    type="text",
                    text=json.dumps({"status": "error", "message": "merged_definition is required for action='merge'"})
                )]
            name_key = (conflict.get("def_b") or conflict["question_a"][:30]).replace(" ", "_").lower()
            upsert_definition(
                name=name_key,
                description=merged_def,
                approved=True,
                reason=f"merged via HITL conflict {conflict_id}"
            )
            resolution = merged_def
            db_resolve_conflict(conflict_id, action, resolution)

        elif action == "reject":
            db_resolve_conflict(conflict_id, "rejected", "discarded")
            resolution = "discarded"

        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "resolved",
                "conflict_id": conflict_id,
                "action": action,
                "resolution": resolution if action != "reject" else "discarded",
                "message": f"Conflict {conflict_id} resolved via '{action}'. Future similar queries will use the canonical definition."
            }, indent=2)
        )]

    # ── watch_schema ──────────────────────────────────────────────────────────
    elif name == "watch_schema":
        db_path = arguments.get("db_path", DATA_DB)

        if not Path(db_path).exists():
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "error",
                    "message": f"Database not found at: {db_path}"
                })
            )]

        events = drift_watcher.snapshot_and_diff(db_path)

        if not events:
            return [types.TextContent(
                type="text",
                text=json.dumps({
                    "status": "ok",
                    "message": "No schema changes detected. All definitions are stable.",
                    "drift_events": []
                })
            )]

        critical = [e for e in events if e.get("affected_definitions")]
        return [types.TextContent(
            type="text",
            text=json.dumps({
                "status": "drift_detected",
                "total_events": len(events),
                "critical_events": len(critical),
                "drift_events": events,
                "action_required": (
                    f"{len(critical)} definition(s) may be broken by schema changes. "
                    "Review affected definitions via list_definitions tool."
                ) if critical else "No definitions affected."
            }, indent=2)
        )]

    return [types.TextContent(type="text", text=json.dumps({"error": f"Unknown tool: {name}"}))]


# ── Entry point ───────────────────────────────────────────────────────────────

async def main():
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await app.run(
            read_stream,
            write_stream,
            app.create_initialization_options()
        )

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())