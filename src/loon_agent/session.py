"""Platform-neutral session identity (a pattern borrowed from hermes-agent).

Every interface (CLI now, Telegram later) produces a ``MessageEvent`` carrying a
``SessionSource``. ``build_session_key`` hashes that source into a stable, opaque key
used directly as LangGraph's ``thread_id`` — so each DM / group / topic gets its own
isolated, durable conversation, and the agent core never needs to know which platform a
message came from.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class SessionSource:
    """Where a message originated, in platform-neutral terms."""

    platform: str  # "cli", "telegram", ...
    chat_id: str  # conversation/channel id (per-DM, per-group, ...)
    user_id: str  # author id
    chat_type: str = "dm"  # "dm" | "group" | "channel" | "thread"
    thread_id: str | None = None  # sub-thread / forum topic, if any


@dataclass(frozen=True)
class MessageEvent:
    """An inbound message, normalized across platforms."""

    source: SessionSource
    text: str
    metadata: dict[str, str] = field(default_factory=dict)


def build_session_key(source: SessionSource) -> str:
    """Derive a stable thread_id from a session source.

    Human-readable platform prefix + a short hash of the identifying fields, so keys are
    stable across restarts, collision-resistant, and don't leak raw ids.
    """
    raw = "|".join(
        [
            source.platform,
            source.chat_type,
            source.chat_id,
            source.user_id,
            source.thread_id or "",
        ]
    )
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{source.platform}:{digest}"


class SessionEpochs:
    """Persistent per-conversation epoch counters — the machinery behind ``/new``.

    The base session key identifies *where* a conversation happens (this DM, this
    topic); the epoch says *which* conversation is current there. Bumping the epoch
    starts a fresh checkpointed thread while the old one stays intact in the
    checkpointer (and long-term memory still spans all of them).
    """

    def __init__(self, db_path: Path | str) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS session_epochs ("
            "base_key TEXT PRIMARY KEY, epoch INTEGER NOT NULL)"
        )
        self._conn.commit()

    def thread_id(self, base_key: str) -> str:
        """The current thread id for a base session key (epoch 0 = the key itself)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT epoch FROM session_epochs WHERE base_key = ?", (base_key,)
            ).fetchone()
        epoch = row[0] if row else 0
        return base_key if epoch == 0 else f"{base_key}:e{epoch}"

    def bump(self, base_key: str) -> str:
        """Start a fresh conversation for this base key; returns the new thread id."""
        with self._lock:
            self._conn.execute(
                "INSERT INTO session_epochs (base_key, epoch) VALUES (?, 1) "
                "ON CONFLICT(base_key) DO UPDATE SET epoch = epoch + 1",
                (base_key,),
            )
            self._conn.commit()
        return self.thread_id(base_key)
