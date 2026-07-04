"""Follow-up tracking: loon's durable notes-to-self.

A follow-up is a small record — topic, note, open/done — that outlives any one
conversation: chat turns and processing loops (``loops.py``) share the same store, so
"mark this for me to follow up on" in a loop iteration is visible in tomorrow's DM and
vice versa. Like the site tools, these are allowed in the chat loop: every write is a
row in a dedicated sqlite file under the data dir — no code execution, no filesystem
reach beyond that database.
"""

from __future__ import annotations

import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool

from ..mirror import utc_now

_STATUSES = ("open", "done", "all")


@dataclass(frozen=True)
class Followup:
    id: int
    topic: str
    note: str
    status: str
    created_at: str


class FollowupStore:
    """Sqlite-backed follow-up notes (one shared connection, lock-serialized)."""

    def __init__(self, db_path: Path | str) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS followups ("
            "id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "topic TEXT NOT NULL,"
            "note TEXT NOT NULL,"
            "status TEXT NOT NULL DEFAULT 'open',"
            "created_at TEXT NOT NULL,"
            "resolved_at TEXT,"
            "resolution TEXT)"
        )
        self._conn.commit()

    def add(self, topic: str, note: str) -> int:
        with self._lock:
            cursor = self._conn.execute(
                "INSERT INTO followups (topic, note, created_at) VALUES (?, ?, ?)",
                (topic, note, utc_now()),
            )
            self._conn.commit()
        return int(cursor.lastrowid)

    def resolve(self, followup_id: int, resolution: str = "") -> bool:
        """Mark an open follow-up done; False if it doesn't exist or already was."""
        with self._lock:
            cursor = self._conn.execute(
                "UPDATE followups SET status = 'done', resolved_at = ?, resolution = ? "
                "WHERE id = ? AND status = 'open'",
                (utc_now(), resolution, followup_id),
            )
            self._conn.commit()
        return cursor.rowcount > 0

    def list(self, status: str = "open") -> list[Followup]:
        query = "SELECT id, topic, note, status, created_at FROM followups"
        params: tuple = ()
        if status != "all":
            query += " WHERE status = ?"
            params = (status,)
        query += " ORDER BY id"
        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [Followup(*row) for row in rows]


def format_followups(items: list[Followup], status: str) -> str:
    if not items:
        return "no follow-ups yet." if status == "all" else f"no {status} follow-ups."
    lines = [f"#{f.id} [{f.status}] {f.topic} — {f.note} ({f.created_at[:10]})" for f in items]
    return "\n".join(lines)


def followup_tools(store: FollowupStore) -> list[BaseTool]:
    """The three chat-loop follow-up tools, bound to one store."""
    shared_note = (
        "Follow-ups are your durable notes-to-self — they survive across conversations "
        "and processing loops."
    )

    def _add(topic: str, note: str) -> str:
        topic, note = topic.strip(), note.strip()
        if not topic or not note:
            return "error: add_followup needs both a topic and a note."
        followup_id = store.add(topic, note)
        return f"recorded follow-up #{followup_id}: {topic}"

    def _list(status: str = "open") -> str:
        status = (status or "open").strip().lower()
        if status not in _STATUSES:
            return f"error: status must be one of {', '.join(_STATUSES)}."
        return format_followups(store.list(status), status)

    def _resolve(followup_id: int, resolution: str = "") -> str:
        if store.resolve(followup_id, resolution.strip()):
            return f"resolved follow-up #{followup_id}."
        return f"error: no open follow-up #{followup_id} — list_followups shows what exists."

    return [
        StructuredTool.from_function(
            _add,
            name="add_followup",
            description=(
                "Record something to follow up on later: a short topic label plus a note "
                f"saying what needs attention and why. {shared_note}"
            ),
        ),
        StructuredTool.from_function(
            _list,
            name="list_followups",
            description=(
                f"List your follow-ups by status: 'open' (default), 'done', or 'all'. {shared_note}"
            ),
        ),
        StructuredTool.from_function(
            _resolve,
            name="resolve_followup",
            description=(
                "Mark an open follow-up as done by its number, optionally noting how it "
                f"was resolved. {shared_note}"
            ),
        ),
    ]
