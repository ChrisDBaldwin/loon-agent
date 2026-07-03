"""Chroma-backed memory provider: semantic recall over past turns.

Same three-hook contract as :class:`SqliteMemoryProvider` (see ``provider.py``), swapped
in via ``LOON_MEMORY_BACKEND=chroma``. Turns are embedded with Chroma's bundled local
embedding model (all-MiniLM-L6-v2, ONNX, downloaded once to ``~/.cache/chroma`` on first
use) — no external embeddings API or extra backend config needed. Recall is nearest-
neighbor over embeddings rather than FTS5 keyword matching, so it survives paraphrase.
"""

from __future__ import annotations

import datetime as _dt
import threading
import uuid
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from .provider import notes_block

_COLLECTION = "turns"


class ChromaMemoryProvider:
    """Chroma/embeddings-backed long-term memory."""

    def __init__(self, db_path: Path | str, notes_path: Path | str, top_k: int = 3) -> None:
        self.notes_path = Path(notes_path)
        self.top_k = top_k
        self._lock = threading.Lock()
        client = chromadb.PersistentClient(
            path=str(db_path), settings=ChromaSettings(anonymized_telemetry=False)
        )
        self._collection = client.get_or_create_collection(_COLLECTION)

    # --- MemoryProvider contract -------------------------------------------------

    def system_prompt_block(self) -> str:
        return notes_block(self.notes_path)

    def prefetch(self, query: str, session_id: str) -> str:
        rows = self._search(query)
        if not rows:
            return ""
        lines = [f"- user: {u}\n  assistant: {a}" for u, a in rows]
        return "Earlier related exchanges:\n" + "\n".join(lines)

    def sync_turn(self, user: str, assistant: str, session_id: str) -> None:
        ts = _dt.datetime.now().isoformat(timespec="seconds")
        document = f"user: {user}\nassistant: {assistant}"
        with self._lock:
            self._collection.add(
                ids=[str(uuid.uuid4())],
                documents=[document],
                metadatas=[
                    {"session_id": session_id, "ts": ts, "user": user, "assistant": assistant}
                ],
            )

    # --- internals ---------------------------------------------------------------

    def _search(self, query: str) -> list[tuple[str, str]]:
        if not query.strip():
            return []
        with self._lock:
            count = self._collection.count()
            if count == 0:
                return []
            result = self._collection.query(
                query_texts=[query], n_results=min(self.top_k, count)
            )
        metadatas = (result.get("metadatas") or [[]])[0]
        return [(m["user"], m["assistant"]) for m in metadatas]
