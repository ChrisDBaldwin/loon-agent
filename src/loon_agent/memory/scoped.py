"""Persona-scoped memory: structural enforcement of a MemoryBinding (Phase 3 §3).

Decorates any ``MemoryProvider`` with the scope a donned persona declares in its
sidecar. Enforcement is structural, not advisory: a ``read-only`` persona has no
code path that writes memory, and mode ``none`` never touches the inner provider
at all. Two personas sharing a namespace share recall — deliberate team memory;
distinct namespaces are isolated by key prefix.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from .provider import MemoryProvider

if TYPE_CHECKING:
    from masques_core import MemoryBinding


class ScopedMemory(MemoryProvider):
    """A MemoryProvider seen through a persona's namespace/mode/prompt-block scope."""

    def __init__(self, inner: MemoryProvider, binding: MemoryBinding) -> None:
        self.inner = inner
        self.binding = binding

    def _scoped(self, session_id: str) -> str:
        return f"persona/{self.binding.namespace}/{session_id}"

    def system_prompt_block(self) -> str:
        if not self.binding.prompt_block:
            return ""
        return self.inner.system_prompt_block()

    def prefetch(self, query: str, session_id: str) -> str:
        if self.binding.mode == "none":
            return ""
        return self.inner.prefetch(query, self._scoped(session_id))

    def sync_turn(self, user: str, assistant: str, session_id: str) -> None:
        if self.binding.mode != "read-write":
            return
        self.inner.sync_turn(user, assistant, self._scoped(session_id))
