"""ScopedMemory: a MemoryBinding enforced structurally, not by advice."""

from __future__ import annotations

from masques_core import MemoryBinding

from loon_agent.memory.provider import MemoryProvider
from loon_agent.memory.scoped import ScopedMemory


class RecordingMemory(MemoryProvider):
    def __init__(self) -> None:
        self.prefetched: list[tuple[str, str]] = []
        self.synced: list[tuple[str, str, str]] = []

    def system_prompt_block(self) -> str:
        return "STATIC NOTES"

    def prefetch(self, query: str, session_id: str) -> str:
        self.prefetched.append((query, session_id))
        return f"recall for {session_id}"

    def sync_turn(self, user: str, assistant: str, session_id: str) -> None:
        self.synced.append((user, assistant, session_id))


def test_read_write_prefixes_session_id_with_persona_namespace() -> None:
    inner = RecordingMemory()
    scoped = ScopedMemory(inner, MemoryBinding(namespace="research"))

    assert scoped.prefetch("q", "cli:chris") == "recall for persona/research/cli:chris"
    scoped.sync_turn("u", "a", "cli:chris")

    assert inner.prefetched == [("q", "persona/research/cli:chris")]
    assert inner.synced == [("u", "a", "persona/research/cli:chris")]


def test_read_only_recalls_but_has_no_write_path() -> None:
    inner = RecordingMemory()
    scoped = ScopedMemory(inner, MemoryBinding(namespace="research", mode="read-only"))

    assert scoped.prefetch("q", "s") == "recall for persona/research/s"
    scoped.sync_turn("u", "a", "s")

    assert inner.synced == []  # structurally a no-op, not a filtered write


def test_mode_none_is_full_amnesia() -> None:
    inner = RecordingMemory()
    scoped = ScopedMemory(inner, MemoryBinding(namespace="guest", mode="none"))

    assert scoped.prefetch("q", "s") == ""
    scoped.sync_turn("u", "a", "s")

    assert inner.prefetched == []  # inner provider never touched on recall
    assert inner.synced == []


def test_prompt_block_gate() -> None:
    inner = RecordingMemory()
    on = ScopedMemory(inner, MemoryBinding(namespace="n"))
    off = ScopedMemory(inner, MemoryBinding(namespace="n", prompt_block=False))

    assert on.system_prompt_block() == "STATIC NOTES"
    assert off.system_prompt_block() == ""
