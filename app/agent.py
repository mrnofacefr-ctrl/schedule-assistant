"""
agent.py
--------
The agent itself. Uses Claude's native tool-use (function calling) to decide,
per user turn, whether to:
  - answer directly (no tool),
  - call get_schedule (RAG retrieval),
  - call update_schedule (write), or
  - chain both (e.g. "move my 2pm to 4pm" -> get_schedule to find the entry,
    then update_schedule to change it).

This is a standard agentic loop: send messages + tool defs -> if Claude
returns tool_use blocks, execute them locally -> feed tool_result back ->
repeat until Claude returns a plain text (end_turn) response.
"""

import os
from datetime import datetime

import anthropic

from app.tools import TOOL_DEFINITIONS, execute_tool

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-5-20250929")
MAX_TOOL_ITERATIONS = 6


def _system_prompt() -> str:
    today = datetime.now()
    return f"""You are a helpful, concise Schedule Assistant that manages the user's
calendar for the next 30 days (meetings, workshops, tasks, appointments).

Today's date is {today.strftime('%A, %Y-%m-%d')}. Resolve any relative dates the
user mentions ("tomorrow", "next Friday", "this weekend") into absolute
YYYY-MM-DD dates yourself before calling a tool.

You have two tools:
- get_schedule: retrieve schedule info (by exact date, date range, or free-text
  semantic search). Use this for any question about what's scheduled, or to
  check availability, or to look up an entry's id before modifying it.
- update_schedule: add, update, or delete a schedule entry. If the user wants
  to change or remove something ("move my meeting", "cancel the dentist
  appointment"), first call get_schedule to find the exact entry_id, then call
  update_schedule with that id. Never guess an entry_id.

Guidelines:
- Only call tools when the request actually needs schedule data. For general
  chit-chat or questions unrelated to the calendar, just answer directly.
- When checking "am I free" at a time, retrieve that day's schedule and reason
  over the start/end times yourself.
- After a write (add/update/delete), confirm back to the user in plain
  language what changed (title, date, time).
- Keep answers short and conversational; use bullet points for multiple
  events.
"""


def _client() -> anthropic.Anthropic:
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Add it to your environment / .env file."
        )
    return anthropic.Anthropic(api_key=api_key)


def run_agent(user_message: str, history: list[dict] | None = None) -> dict:
    """
    Runs one full agentic turn. `history` is a list of prior
    {"role": "user"|"assistant", "content": ...} messages (already in
    Anthropic message format) so the conversation can be multi-turn.
    Returns {"reply": str, "history": updated_history, "tool_trace": [...]}
    """
    client = _client()
    messages = list(history) if history else []
    messages.append({"role": "user", "content": user_message})

    tool_trace = []

    for _ in range(MAX_TOOL_ITERATIONS):
        response = client.messages.create(
            model=MODEL,
            max_tokens=1024,
            system=_system_prompt(),
            tools=TOOL_DEFINITIONS,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            final_text = "".join(
                block.text for block in response.content if block.type == "text"
            )
            return {"reply": final_text, "history": messages, "tool_trace": tool_trace}

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result = execute_tool(block.name, block.input)
            tool_trace.append({"tool": block.name, "input": block.input, "output": result})
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": _to_text(result),
            })
        messages.append({"role": "user", "content": tool_results})

    return {
        "reply": "I had trouble completing that request after several tool calls. "
                 "Could you rephrase or simplify it?",
        "history": messages,
        "tool_trace": tool_trace,
    }


def _to_text(obj) -> str:
    import json
    return json.dumps(obj, default=str)
