"""CLI REPL adapter: a terminal chat loop over the platform-neutral agent core.

Reads a line, wraps it in a ``MessageEvent`` with a ``cli`` ``SessionSource``, derives the
``thread_id`` via ``build_session_key`` (identical machinery the Telegram adapter will
use), streams the turn, and surfaces tool calls as they happen.
"""

from __future__ import annotations

import getpass

from langchain_core.messages import AIMessage, ToolMessage

from ..app import build_agent
from ..config import get_settings
from ..graph import _text
from ..session import MessageEvent, SessionSource, build_session_key

_BANNER = "loon-agent — type a message, or /exit to quit."
_EXIT = {"/exit", "/quit", "exit", "quit"}


def _render(message: object) -> None:
    """Print a streamed message: tool calls, tool results, or the assistant reply."""
    if isinstance(message, AIMessage) and message.tool_calls:
        for call in message.tool_calls:
            print(f"  → tool: {call['name']}({call['args']})")
    elif isinstance(message, ToolMessage):
        print(f"  ← result: {message.content}")
    elif isinstance(message, AIMessage):
        if text := _text(message):
            print(f"loon> {text}")


def run_cli() -> None:
    settings = get_settings()
    agent = build_agent(settings)
    source = SessionSource(platform="cli", chat_id="local", user_id=getpass.getuser())
    session_key = build_session_key(source)

    print(_BANNER)
    print(f"backend={settings.backend}  session={session_key}\n")

    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line.lower() in _EXIT:
            break

        event = MessageEvent(source=source, text=line)
        for message in agent.stream(event.text, session_key):
            _render(message)

    print("bye.")
