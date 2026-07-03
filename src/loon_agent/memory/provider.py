"""The memory-provider contract.

Three lifecycle hooks, lifted from hermes-agent's design:

* ``system_prompt_block()`` — static text injected into the system prompt every turn.
* ``prefetch(query, session_id)`` — recall fetched *before* a turn and injected as context.
* ``sync_turn(user, assistant, session_id)`` — write-back *after* a turn.

Keeping this an ABC means the SQLite default and a future OpenViking provider are
interchangeable, and the agent core depends only on this interface.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


def notes_block(notes_path: Path) -> str:
    """Render the standing markdown notes file as a system-prompt block (or "")."""
    if notes_path.exists():
        notes = notes_path.read_text(encoding="utf-8").strip()
        if notes:
            return f"Standing notes (from {notes_path.name}):\n{notes}"
    return ""


class MemoryProvider(ABC):
    """Interface for long-term, cross-conversation memory."""

    @abstractmethod
    def system_prompt_block(self) -> str:
        """Static instructions/status to inject into the system prompt (may be empty)."""

    @abstractmethod
    def prefetch(self, query: str, session_id: str) -> str:
        """Recall relevant context for ``query`` (may be empty). Runs before the turn."""

    @abstractmethod
    def sync_turn(self, user: str, assistant: str, session_id: str) -> None:
        """Persist a completed turn. Runs after the turn."""


class NullMemoryProvider(MemoryProvider):
    """A no-op provider; useful for tests or when memory is disabled."""

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query: str, session_id: str) -> str:
        return ""

    def sync_turn(self, user: str, assistant: str, session_id: str) -> None:
        return None
