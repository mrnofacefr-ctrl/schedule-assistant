"""
tools.py
--------
The two tools the agent can call. Both are exposed to Claude via Anthropic's
tool-use (function calling) API in agent.py. The agent decides FOR ITSELF
whether a user turn needs retrieval (get_schedule), a write (update_schedule),
both, or neither (e.g. plain chit-chat) -- that decision is not hardcoded
here, it happens inside the LLM's tool-use loop.

get_schedule    -> the RAG retrieval tool. Exact date/date-range lookups hit
                    SQLite directly (fast + exact); free-text queries go
                    through the ChromaDB semantic search (vector_store.py).
update_schedule -> the write tool. Always writes to SQLite first (source of
                    truth), then re-embeds the affected row(s) into ChromaDB
                    so retrieval never sees stale data.
"""

from app import db, vector_store

# ---------------------------------------------------------------------------
# Anthropic tool schemas (JSON Schema, per Anthropic tool-use spec)
# ---------------------------------------------------------------------------

TOOL_DEFINITIONS = [
    {
        "name": "get_schedule",
        "description": (
            "Retrieve schedule entries (meetings, workshops, tasks, appointments). "
            "Use this to answer any question about what's on the calendar, whether "
            "the user is free/busy at a time, or to look up an entry before updating "
            "it. Prefer passing an explicit `date` or `date_range` when the user "
            "refers to a specific day or period (resolve relative terms like "
            "'tomorrow', 'next Friday' to an absolute YYYY-MM-DD date yourself using "
            "the current date given in the system prompt). Use `query` for fuzzy / "
            "topical lookups (e.g. 'my dentist appointment', 'anything about budget')."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Free-text search query for semantic retrieval. "
                                    "Can be empty if date/date_range fully specifies the request.",
                },
                "date": {
                    "type": "string",
                    "description": "Exact date in YYYY-MM-DD format to filter by, if the "
                                    "user asked about a single specific day.",
                },
                "date_range_start": {
                    "type": "string",
                    "description": "Start of a date range (YYYY-MM-DD), if the user asked "
                                    "about a period spanning multiple days.",
                },
                "date_range_end": {
                    "type": "string",
                    "description": "End of a date range (YYYY-MM-DD), used with date_range_start.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "Max number of results for free-text semantic search. Default 5.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "update_schedule",
        "description": (
            "Add, modify, or remove a schedule entry. Always call get_schedule first "
            "if you are not already certain of the exact entry_id you need to update "
            "or delete (e.g. resolve 'my 2 PM meeting' to a real entry before moving it). "
            "For 'add', omit entry_id and provide the new entry's fields. For 'update' "
            "or 'delete', provide entry_id. For 'update', only include the fields that "
            "are changing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["add", "update", "delete"],
                    "description": "Which write operation to perform.",
                },
                "entry_id": {
                    "type": "string",
                    "description": "Required for 'update' and 'delete'. The id of the "
                                    "existing entry, obtained from a prior get_schedule call.",
                },
                "title": {"type": "string", "description": "Event title."},
                "type": {
                    "type": "string",
                    "enum": ["meeting", "workshop", "task", "appointment"],
                    "description": "Event category.",
                },
                "date": {"type": "string", "description": "YYYY-MM-DD"},
                "start_time": {"type": "string", "description": "HH:MM in 24-hour time."},
                "end_time": {"type": "string", "description": "HH:MM in 24-hour time."},
                "location": {"type": "string"},
                "description": {"type": "string"},
                "attendees": {"type": "string", "description": "Comma-separated names."},
            },
            "required": ["action"],
        },
    },
]


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def run_get_schedule(args: dict) -> dict:
    date = args.get("date")
    start = args.get("date_range_start")
    end = args.get("date_range_end")
    query = (args.get("query") or "").strip()
    top_k = args.get("top_k") or 5

    if date:
        entries = db.get_entries_by_date(date)
        return {"mode": "exact_date", "date": date, "count": len(entries), "entries": entries}

    if start and end:
        entries = db.get_entries_by_date_range(start, end)
        return {"mode": "date_range", "start": start, "end": end,
                "count": len(entries), "entries": entries}

    if not query:
        # nothing specific given -> return everything (bounded) as a fallback
        entries = db.get_all_entries()[:20]
        return {"mode": "all", "count": len(entries), "entries": entries}

    # --- RAG step: semantic search over ChromaDB, then hydrate from SQLite ---
    hits = vector_store.semantic_search(query, n_results=top_k)
    entries = []
    for h in hits:
        entries.append({
            "id": h["id"],
            "score": round(h["score"], 3),
            "title": h["metadata"]["title"],
            "type": h["metadata"]["type"],
            "date": h["metadata"]["date"],
            "start_time": h["metadata"]["start_time"],
            "end_time": h["metadata"]["end_time"],
            "summary": h["document"],
        })
    return {"mode": "semantic_search", "query": query, "count": len(entries), "entries": entries}


def run_update_schedule(args: dict) -> dict:
    action = args.get("action")

    if action == "add":
        required = ["title", "date", "start_time", "end_time"]
        missing = [f for f in required if not args.get(f)]
        if missing:
            return {"success": False, "error": f"Missing required fields: {missing}"}
        entry = {
            "title": args["title"],
            "type": args.get("type", "meeting"),
            "date": args["date"],
            "start_time": args["start_time"],
            "end_time": args["end_time"],
            "location": args.get("location", ""),
            "description": args.get("description", ""),
            "attendees": args.get("attendees", ""),
        }
        entry_id = db.insert_entry(entry)
        entry["id"] = entry_id
        vector_store.upsert_entry(entry)
        return {"success": True, "action": "add", "entry": entry}

    if action == "update":
        entry_id = args.get("entry_id")
        if not entry_id:
            return {"success": False, "error": "entry_id is required for update"}
        existing = db.get_entry(entry_id)
        if not existing:
            return {"success": False, "error": f"No entry found with id {entry_id}"}
        fields = {k: v for k, v in args.items()
                  if k in {"title", "type", "date", "start_time", "end_time",
                           "location", "description", "attendees"} and v is not None}
        if not fields:
            return {"success": False, "error": "No fields provided to update"}
        db.update_entry(entry_id, fields)
        updated = db.get_entry(entry_id)
        vector_store.upsert_entry(updated)
        return {"success": True, "action": "update", "entry": updated}

    if action == "delete":
        entry_id = args.get("entry_id")
        if not entry_id:
            return {"success": False, "error": "entry_id is required for delete"}
        existing = db.get_entry(entry_id)
        if not existing:
            return {"success": False, "error": f"No entry found with id {entry_id}"}
        db.delete_entry(entry_id)
        vector_store.delete_entry(entry_id)
        return {"success": True, "action": "delete", "deleted_entry": existing}

    return {"success": False, "error": f"Unknown action '{action}'"}


TOOL_DISPATCH = {
    "get_schedule": run_get_schedule,
    "update_schedule": run_update_schedule,
}


def execute_tool(name: str, args: dict) -> dict:
    if name not in TOOL_DISPATCH:
        return {"success": False, "error": f"Unknown tool '{name}'"}
    return TOOL_DISPATCH[name](args)
