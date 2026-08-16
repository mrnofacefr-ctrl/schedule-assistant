# Agentic RAG Schedule Assistant

An agent that manages a 30-day schedule (meetings, workshops, tasks,
appointments), backed by a RAG pipeline over ChromaDB, with two tools
(`get_schedule`, `update_schedule`) that Claude decides when to call.

## Architecture

```
User message
     │
     ▼
FastAPI /api/chat  ──►  agent.py (Claude tool-use loop)
                              │
                 decides: answer directly? retrieve? write?
                              │
              ┌───────────────┴───────────────┐
              ▼                                ▼
      get_schedule tool                update_schedule tool
              │                                │
   exact date/range → SQLite            add/update/delete → SQLite
   free-text query   → ChromaDB               │ (source of truth)
   (RAG semantic search)                       ▼
              │                        re-embed changed row
              ▼                        into ChromaDB (stays in sync)
      results → Claude → final
      natural-language reply
```

**Why two stores?** SQLite (`app/db.py`) is the source of truth for exact
CRUD — dates, times, IDs. ChromaDB (`app/vector_store.py`) is a *derived*
semantic index, rebuilt/synced from SQLite, used only for retrieval (RAG).
This avoids the classic failure mode of doing exact updates/deletes against
a vector index (cosine similarity is a bad way to guarantee "delete THIS
exact meeting").

**Why is the agent "agentic"?** The routing between "just answer",
"retrieve (`get_schedule`)", "write (`update_schedule`)", or "retrieve then
write" is not hardcoded with keyword rules — it's Claude's own decision via
native tool use (function calling), driven by the tool descriptions and a
system prompt (`app/agent.py`). For a request like *"move my meeting from 2
PM to 4 PM"*, the agent first calls `get_schedule` to resolve which entry
that refers to, then calls `update_schedule` with the resolved `entry_id`.

**Embeddings**: by default, retrieval uses a small offline, dependency-free
`HashingVectorizer` (scikit-learn) as the ChromaDB embedding function — no
model download, no extra API key, works immediately on any host. If you set
`OPENAI_API_KEY`, it automatically switches to OpenAI's
`text-embedding-3-small` for stronger semantic matching. Swap in Pinecone by
replacing `app/vector_store.py`'s client with the Pinecone SDK — the rest of
the pipeline (tools, agent) doesn't need to change.

## Project layout

```
app/
  db.py             SQLite source of truth (schema + CRUD)
  data_generator.py Generates 30 days of sample events
  vector_store.py   ChromaDB RAG index (embed, sync, semantic_search)
  tools.py          get_schedule / update_schedule tool implementations + schemas
  agent.py          Claude tool-use agentic loop
  main.py           FastAPI app (chat API + serves the UI)
static/index.html   Minimal chat UI
data/               SQLite db + Chroma persistent store (created at runtime)
```

## Run locally

```bash
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env
# edit .env and paste your ANTHROPIC_API_KEY (get one at
# https://console.anthropic.com/settings/keys)

# (optional) explicitly seed sample data — main.py also does this
# automatically on first boot if the DB is empty:
python -m app.data_generator

uvicorn app.main:app --reload --port 8000
```

Open http://127.0.0.1:8000 and try:
- "What do I have scheduled tomorrow?"
- "Am I free Friday afternoon?"
- "Add a meeting on August 20 at 3 PM called Design Review"
- "Move my meeting from 2 PM to 4 PM" (say which one if it's ambiguous)
- "Cancel my dentist appointment"

## Deploy (get your live URL)

Any host that runs a Docker container or a Python web process works. Two
easy free options:

### Option A — Render (Docker, recommended, ~5 min)
1. Push this folder to a new GitHub repo.
2. On https://render.com → **New +** → **Web Service** → connect the repo.
3. Render auto-detects the `Dockerfile`. Leave build/start commands blank.
4. Under **Environment**, add `ANTHROPIC_API_KEY` = your key.
5. Deploy. Render gives you a URL like `https://your-app.onrender.com`.

### Option B — Railway (buildpack, uses `Procfile`)
1. Push to GitHub, then on https://railway.app → **New Project** → **Deploy
   from GitHub repo**.
2. Add variable `ANTHROPIC_API_KEY` in the service's **Variables** tab.
3. Railway detects Python + `Procfile` and deploys automatically, giving you
   a `https://your-app.up.railway.app` URL.

### Option C — Hugging Face Spaces (Docker SDK)
1. Create a new Space → SDK: **Docker**.
2. Upload this folder's contents (the `Dockerfile` at the root is picked up
   automatically).
3. Add `ANTHROPIC_API_KEY` as a **Secret** in Space settings.
4. Your Space URL (`https://huggingface.co/spaces/<you>/<space>`) is your
   deployed app.

Once live, put the URL in `deployment_url.txt` for submission.

## API (used by the UI, also callable directly)

- `POST /api/chat` — `{"message": "...", "session_id": "..."}` → agent reply
  + `tool_trace` (which tools were called and with what args/results —
  useful for grading/demoing the agentic behavior).
- `GET /api/schedule` — dump all current entries.
- `GET /api/health` — liveness check + entry count.
- `POST /api/reset?session_id=...` — clear a session's conversation memory.

## Notes / things worth knowing before you submit this

- Sample data is regenerated (deterministic, seeded) from **today** each
  time you run `data_generator.seed_database(reset=True)` — so "the next 30
  days" always means the next 30 days from whenever it's deployed, not a
  fixed August 2026 window.
- Conversation history is kept in-memory per `session_id` — fine for a demo;
  swap for a DB-backed session store for real multi-user persistence.
- I could not deploy this to a live URL myself from the environment I built
  it in (no hosting access there) — the app is fully built and tested
  locally (data generation, SQLite, ChromaDB sync/retrieval, tool logic, and
  the FastAPI server all verified working). Deploying it yourself with
  Option A/B/C above takes about 5 minutes.
