"""
main.py
-------
FastAPI web app. Serves a single-page chat UI (static/index.html) and a
/api/chat endpoint that drives the agent. Session history is kept
server-side in memory, keyed by a session_id the browser generates.

Run locally:  uvicorn app.main:app --reload --port 8000
"""

import os
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import db, data_generator, vector_store
from app.agent import run_agent

BASE_DIR = Path(__file__).resolve().parent.parent

app = FastAPI(title="Agentic RAG Schedule Assistant")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Safety net so the frontend always gets parseable JSON back, even if
    # something crashes outside the try/except blocks below.
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


# in-memory per-session conversation history (fine for a demo/assignment;
# swap for redis/db-backed sessions for real multi-user production use)
_SESSIONS: dict[str, list] = {}


@app.on_event("startup")
def startup():
    db.init_db()
    entries = db.get_all_entries()
    if not entries:
        data_generator.seed_database(reset=True)
        entries = db.get_all_entries()
    # (Re)build the vector index from the source of truth on boot.
    vector_store.sync_from_db(entries)


class ChatRequest(BaseModel):
    message: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    reply: str
    tool_trace: list


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not req.message.strip():
        raise HTTPException(400, "message cannot be empty")
    history = _SESSIONS.get(req.session_id, [])
    try:
        result = run_agent(req.message, history=history)
    except Exception as e:
        # Catch-all: without this, an unhandled exception (bad model name,
        # Anthropic API error, network blip, etc.) causes FastAPI/Starlette
        # to return a *plain-text* "Internal Server Error" page instead of
        # JSON, which breaks the frontend's response.json() call. Always
        # returning valid JSON here means the UI can show a real error.
        raise HTTPException(500, f"{type(e).__name__}: {e}")
    _SESSIONS[req.session_id] = result["history"]
    return ChatResponse(reply=result["reply"], tool_trace=result["tool_trace"])


@app.post("/api/reset")
def reset_session(session_id: str = "default"):
    _SESSIONS.pop(session_id, None)
    return {"status": "ok"}


@app.get("/api/schedule")
def list_schedule():
    return {"entries": db.get_all_entries()}


@app.get("/api/health")
def health():
    return {"status": "ok", "entries": len(db.get_all_entries())}


app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")


@app.get("/")
def root():
    return FileResponse(str(BASE_DIR / "static" / "index.html"))
