"""
db.py
-----
SQLite is the *source of truth* for exact schedule data (dates, times, IDs).
ChromaDB (vector_store.py) is a derived semantic index built FROM this data,
used purely for retrieval (RAG). Every write goes through here first, and the
vector index is re-synced afterwards. This split matters because vector
similarity search is great for "what feels relevant" but bad for exact
CRUD guarantees (you don't want a delete to depend on cosine similarity).
"""

import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "schedule.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS schedule (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    type TEXT NOT NULL,              -- meeting | workshop | task | appointment
    date TEXT NOT NULL,              -- YYYY-MM-DD
    start_time TEXT NOT NULL,        -- HH:MM (24h)
    end_time TEXT NOT NULL,          -- HH:MM (24h)
    location TEXT,
    description TEXT,
    attendees TEXT,                  -- comma separated
    created_at TEXT DEFAULT (datetime('now')),
    updated_at TEXT DEFAULT (datetime('now'))
);
"""


@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    with get_conn() as conn:
        conn.execute(SCHEMA)


def clear_all():
    with get_conn() as conn:
        conn.execute("DELETE FROM schedule")


def insert_entry(entry: dict) -> str:
    entry_id = entry.get("id") or str(uuid.uuid4())[:8]
    with get_conn() as conn:
        conn.execute(
            """INSERT INTO schedule (id, title, type, date, start_time, end_time,
                                      location, description, attendees)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                entry_id,
                entry["title"],
                entry.get("type", "meeting"),
                entry["date"],
                entry["start_time"],
                entry["end_time"],
                entry.get("location", ""),
                entry.get("description", ""),
                entry.get("attendees", ""),
            ),
        )
    return entry_id


def update_entry(entry_id: str, fields: dict) -> bool:
    if not fields:
        return False
    cols = ", ".join(f"{k} = ?" for k in fields.keys())
    values = list(fields.values()) + [entry_id]
    with get_conn() as conn:
        cur = conn.execute(
            f"UPDATE schedule SET {cols}, updated_at = datetime('now') WHERE id = ?",
            values,
        )
        return cur.rowcount > 0


def delete_entry(entry_id: str) -> bool:
    with get_conn() as conn:
        cur = conn.execute("DELETE FROM schedule WHERE id = ?", (entry_id,))
        return cur.rowcount > 0


def get_entry(entry_id: str) -> Optional[dict]:
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM schedule WHERE id = ?", (entry_id,)).fetchone()
        return dict(row) if row else None


def get_all_entries() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM schedule ORDER BY date, start_time").fetchall()
        return [dict(r) for r in rows]


def get_entries_by_date(date: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM schedule WHERE date = ? ORDER BY start_time", (date,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_entries_by_date_range(start_date: str, end_date: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            """SELECT * FROM schedule WHERE date BETWEEN ? AND ?
               ORDER BY date, start_time""",
            (start_date, end_date),
        ).fetchall()
        return [dict(r) for r in rows]


def find_entries_by_title_and_date(title_substr: str, date: Optional[str] = None) -> list[dict]:
    with get_conn() as conn:
        if date:
            rows = conn.execute(
                "SELECT * FROM schedule WHERE title LIKE ? AND date = ? ORDER BY start_time",
                (f"%{title_substr}%", date),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM schedule WHERE title LIKE ? ORDER BY date, start_time",
                (f"%{title_substr}%",),
            ).fetchall()
        return [dict(r) for r in rows]
