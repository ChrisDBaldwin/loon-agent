"""CLI REPL adapter: a terminal chat loop over the platform-neutral agent core.

Reads a line, wraps it in a ``MessageEvent`` with a ``cli`` ``SessionSource``, derives the
``thread_id`` via ``build_session_key`` (identical machinery the Telegram adapter uses),
streams the turn, and surfaces tool calls as they happen. ``/skill <name> <args>`` (and
the ``/research <topic>`` alias) run a markdown skill through the deterministic engine
instead of the chat loop.
"""

from __future__ import annotations

import getpass

from langchain_core.messages import AIMessage, ToolMessage

from ..app import LoonRuntime, build_runtime
from ..config import get_settings
from ..graph import _text
from ..session import MessageEvent, SessionSource, build_session_key
from ..skills.engine import SkillRunError

_BANNER = "loon-agent — type a message, /skill <name> <args>, or /exit to quit."
_EXIT = {"/exit", "/quit", "exit", "quit"}


def parse_skill_command(line: str) -> tuple[str, str] | None:
    """``/skill name args…`` or ``/research topic`` -> (name, args text), else None."""
    if line.startswith("/research"):
        rest = line[len("/research") :].strip()
        return ("research", rest)
    if line.startswith("/skill"):
        rest = line[len("/skill") :].strip()
        if not rest:
            return ("", "")
        name, _, args = rest.partition(" ")
        return (name, args.strip())
    return None


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


def _run_skill(runtime: LoonRuntime, name: str, args_text: str) -> None:
    skill = runtime.skills.get(name)
    if skill is None:
        available = ", ".join(sorted(runtime.skills)) or "(none found)"
        print(f"unknown skill {name!r} — available: {available}")
        return
    if skill.args and not args_text:
        print(f"usage: /skill {skill.name} <{skill.args[0]}>")
        return

    args: dict[str, object] = {skill.args[0]: args_text} if skill.args else {}
    args.setdefault("max_sources", runtime.settings.research_sources)
    try:
        result = runtime.runner.run(skill, args)
    except SkillRunError as exc:
        print(f"skill failed: {exc}")
        return

    briefing = result.outputs.get("briefing")
    if isinstance(briefing, str) and briefing:
        print(f"\n{briefing}\n")
    final_output = skill.steps[-1].output
    if final_output != "briefing":
        print(f"→ {final_output}: {result.outputs.get(final_output)}")
    for note in result.failures:
        print(f"  (skipped: {note})")


def run_cli() -> None:
    settings = get_settings()
    runtime = build_runtime(settings, progress=lambda message: print(f"  … {message}"))
    agent = runtime.agent
    source = SessionSource(platform="cli", chat_id="local", user_id=getpass.getuser())
    session_key = build_session_key(source)

    print(_BANNER)
    skills = ", ".join(sorted(runtime.skills)) or "none"
    print(f"backend={settings.backend}  session={session_key}  skills: {skills}\n")

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

        if command := parse_skill_command(line):
            _run_skill(runtime, *command)
            continue

        event = MessageEvent(source=source, text=line)
        for message in agent.stream(event.text, session_key):
            _render(message)

    print("bye.")
