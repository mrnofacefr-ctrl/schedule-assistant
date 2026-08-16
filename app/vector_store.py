"""
vector_store.py
----------------
ChromaDB-backed semantic index over the schedule (RAG retrieval layer).

Embedding function: by default this uses a dependency-free, offline
HashingVectorizer (scikit-learn) so the app works out-of-the-box with no
extra API key and no model download at deploy time (handy on small hosts
like Render's free tier). If you set OPENAI_API_KEY in the environment,
it will automatically switch to OpenAI's text-embedding-3-small for
higher-quality semantic retrieval -- see `_build_embedding_function()`.

The collection is always rebuilt FROM the SQLite source of truth
(app/db.py) via `sync_from_db()`, so Chroma never drifts out of sync with
real schedule state.
"""

import os
from pathlib import Path
from typing import Optional

import chromadb
from chromadb import EmbeddingFunction, Embeddings, Documents

CHROMA_DIR = Path(__file__).resolve().parent.parent / "data" / "chroma"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
COLLECTION_NAME = "schedule_entries"


class HashingEmbeddingFunction(EmbeddingFunction):
    """Offline, dependency-free embedding function (no downloads, no API key).
    Deterministic bag-of-words hashing vectorizer -- good enough for
    matching schedule text (titles, types, descriptions, dates)."""

    def __init__(self, n_features: int = 384):
        from sklearn.feature_extraction.text import HashingVectorizer
        self._vectorizer = HashingVectorizer(
            n_features=n_features, alternate_sign=False, norm="l2"
        )

    def __call__(self, input: Documents) -> Embeddings:
        matrix = self._vectorizer.transform(input)
        return matrix.toarray().tolist()


def _build_embedding_function():
    if os.getenv("OPENAI_API_KEY"):
        try:
            from chromadb.utils import embedding_functions
            return embedding_functions.OpenAIEmbeddingFunction(
                api_key=os.getenv("OPENAI_API_KEY"),
                model_name="text-embedding-3-small",
            )
        except Exception:
            pass
    return HashingEmbeddingFunction()


_client = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=_build_embedding_function(),
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def _entry_to_document(entry: dict) -> str:
    """Flatten a schedule entry into natural-language text for embedding."""
    return (
        f"{entry['type'].title()}: {entry['title']}. "
        f"Date: {entry['date']}. Time: {entry['start_time']} to {entry['end_time']}. "
        f"Location: {entry.get('location') or 'N/A'}. "
        f"Attendees: {entry.get('attendees') or 'none'}. "
        f"Notes: {entry.get('description') or ''}"
    )


def upsert_entry(entry: dict):
    """Add/update a single entry's embedding (call after any DB write)."""
    col = get_collection()
    col.upsert(
        ids=[entry["id"]],
        documents=[_entry_to_document(entry)],
        metadatas=[{
            "title": entry["title"], "type": entry["type"], "date": entry["date"],
            "start_time": entry["start_time"], "end_time": entry["end_time"],
        }],
    )


def delete_entry(entry_id: str):
    col = get_collection()
    try:
        col.delete(ids=[entry_id])
    except Exception:
        pass


def sync_from_db(entries: list[dict]):
    """Rebuild the whole vector index from the SQLite source of truth."""
    global _client, _collection
    _client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        _client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    _collection = _client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=_build_embedding_function(),
        metadata={"hnsw:space": "cosine"},
    )
    if not entries:
        return
    _collection.add(
        ids=[e["id"] for e in entries],
        documents=[_entry_to_document(e) for e in entries],
        metadatas=[{
            "title": e["title"], "type": e["type"], "date": e["date"],
            "start_time": e["start_time"], "end_time": e["end_time"],
        } for e in entries],
    )


def semantic_search(query: str, n_results: int = 5, where: Optional[dict] = None) -> list[dict]:
    """Core RAG retrieval step: embed the query, return top-k similar entries."""
    col = get_collection()
    if col.count() == 0:
        return []
    n_results = min(n_results, col.count())
    results = col.query(
        query_texts=[query], n_results=n_results, where=where,
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    ids = results.get("ids", [[]])[0]
    for _id, doc, meta, dist in zip(
        ids, results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        hits.append({"id": _id, "document": doc, "metadata": meta, "score": 1 - dist})
    return hits
