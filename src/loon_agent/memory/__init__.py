"""Swappable long-term memory.

The :class:`MemoryProvider` contract (patterned on hermes-agent) is the seam: the default
:class:`SqliteMemoryProvider` needs no vector DB, and an OpenViking-backed provider can
drop in later without touching the agent core.
"""

from .provider import MemoryProvider, NullMemoryProvider
from .sqlite_provider import SqliteMemoryProvider

__all__ = ["MemoryProvider", "NullMemoryProvider", "SqliteMemoryProvider"]
