"""Application assembly: wire settings -> llm, tools, checkpointer, memory, telemetry.

Kept separate from any single adapter so the CLI now (and Telegram later) build the same
agent the same way.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from .config import Settings, get_settings
from .graph import LoonAgent
from .llm import make_llm
from .memory import SqliteMemoryProvider
from .telemetry import setup_telemetry
from .tools import DEFAULT_TOOLS


def build_agent(settings: Settings | None = None) -> LoonAgent:
    """Construct a fully-wired :class:`LoonAgent` from settings."""
    settings = settings or get_settings()
    setup_telemetry(settings)

    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # One sqlite connection per store; check_same_thread=False so a future async/threaded
    # adapter can share it.
    checkpoint_conn = sqlite3.connect(data_dir / "checkpoints.sqlite", check_same_thread=False)
    checkpointer = SqliteSaver(checkpoint_conn)

    memory = SqliteMemoryProvider(
        db_path=data_dir / "memory.sqlite",
        notes_path=data_dir / "MEMORY.md",
    )

    llm = make_llm(settings=settings)
    return LoonAgent(llm, DEFAULT_TOOLS, checkpointer=checkpointer, memory=memory)
