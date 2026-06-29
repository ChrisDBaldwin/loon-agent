"""Platform-neutral session identity (a pattern borrowed from hermes-agent).

Every interface (CLI now, Telegram later) produces a ``MessageEvent`` carrying a
``SessionSource``. ``build_session_key`` hashes that source into a stable, opaque key
used directly as LangGraph's ``thread_id`` — so each DM / group / topic gets its own
isolated, durable conversation, and the agent core never needs to know which platform a
message came from.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


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
