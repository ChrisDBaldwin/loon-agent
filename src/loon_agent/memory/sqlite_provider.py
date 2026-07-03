"""Default memory provider: SQLite full-text search + a markdown notes file.

No vector DB and no embeddings — past turns are indexed with SQLite's FTS5 and recalled by
keyword. The markdown notes file (``MEMORY.md``) is hand- or agent-curated standing
context injected into every system prompt. This is the simplest thing that captures the
hermes ``system_prompt_block`` / ``prefetch`` / ``sync_turn`` contract; swap in OpenViking
later for richer recall.
"""

from __future__ import annotations

import datetime as _dt
import re
import sqlite3
import threading
from pathlib import Path

from .provider import notes_block

# FTS5 reserves a lot of punctuation; restrict matching to bare word tokens (OR-joined).
_WORD = re.compile(r"\w+")


class SqliteMemoryProvider:
    """SQLite/FTS5-backed long-term memory."""

    def __init__(self, db_path: Path | str, notes_path: Path | str, top_k: int = 3) -> None:
        self.notes_path = Path(notes_path)
        self.top_k = top_k
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._fts = self._init_schema()

    def _init_schema(self) -> bool:
        """Create the turn store. Returns True if FTS5 is available, else uses a fallback."""
        try:
            self._conn.execute(
                "CREATE VIRTUAL TABLE IF NOT EXISTS turns USING fts5("
                "session_id UNINDEXED, ts UNINDEXED, user, assistant)"
            )
            self._conn.commit()
            return True
        except sqlite3.OperationalError:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS turns ("
                "session_id TEXT, ts TEXT, user TEXT, assistant TEXT)"
            )
            self._conn.commit()
            return False

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
        with self._lock:
            self._conn.execute(
                "INSERT INTO turns (session_id, ts, user, assistant) VALUES (?, ?, ?, ?)",
                (session_id, ts, user, assistant),
            )
            self._conn.commit()

    # --- internals ---------------------------------------------------------------

    def _search(self, query: str) -> list[tuple[str, str]]:
        tokens = _WORD.findall(query)
        if not tokens:
            return []
        with self._lock:
            if self._fts:
                match = " OR ".join(tokens)
                cur = self._conn.execute(
                    "SELECT user, assistant FROM turns WHERE turns MATCH ? ORDER BY rank LIMIT ?",
                    (match, self.top_k),
                )
            else:
                like = f"%{tokens[0]}%"
                cur = self._conn.execute(
                    "SELECT user, assistant FROM turns "
                    "WHERE user LIKE ? OR assistant LIKE ? LIMIT ?",
                    (like, like, self.top_k),
                )
            return [(row[0], row[1]) for row in cur.fetchall()]
