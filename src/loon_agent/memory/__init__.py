"""Swappable long-term memory.

The :class:`MemoryProvider` contract (patterned on hermes-agent) is the seam: the default
:class:`SqliteMemoryProvider` needs no vector DB, and :class:`ChromaMemoryProvider` (or a
future OpenViking-backed one) drops in later without touching the agent core.
"""

from .chroma_provider import ChromaMemoryProvider
from .provider import MemoryProvider, NullMemoryProvider
from .sqlite_provider import SqliteMemoryProvider

__all__ = ["ChromaMemoryProvider", "MemoryProvider", "NullMemoryProvider", "SqliteMemoryProvider"]
