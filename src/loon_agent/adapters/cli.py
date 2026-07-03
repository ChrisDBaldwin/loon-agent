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

from ..app import LoonRuntime, build_runtime, parse_don_command
from ..commands import (
    HELP_TEXT,
    format_model_list,
    model_inventory,
    pick_model,
    status_text,
)
from ..config import get_settings
from ..graph import _text
from ..session import MessageEvent, SessionSource, build_session_key
from ..skills.engine import SkillRunError

_BANNER = "loon-agent — type a message, /help for commands, /exit to quit."
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


def _inventory(runtime: LoonRuntime):
    return model_inventory(
        runtime.settings,
        active_backend=runtime.active_backend,
        active_model=runtime.active_model,
    )


def _handle_don(runtime: LoonRuntime, name: str, intent: str | None) -> None:
    if not name:
        print("usage: /don <name> [intent]")
        return
    persona = runtime.don(name, intent)
    if persona is None:
        print(f"masque {name!r} not found — still baseline.")
        return
    tools = ", ".join(t.name for t in runtime.agent.tools) or "(none)"
    print(f"donned {persona.name} v{persona.version.raw} — tools: {tools}\n")


def run_cli() -> None:
    settings = get_settings()
    runtime = build_runtime(settings, progress=lambda message: print(f"  … {message}"))
    source = SessionSource(platform="cli", chat_id="local", user_id=getpass.getuser())
    base_key = build_session_key(source)
    session_key = runtime.epochs.thread_id(base_key)
    last_text = ""

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
        if line == "/new":
            session_key = runtime.epochs.bump(base_key)
            print(f"fresh session started: {session_key}\n")
            continue
        if line == "/help":
            print(HELP_TEXT + "\n")
            continue
        if line == "/status":
            print(status_text(runtime, session_key) + "\n")
            continue
        if line == "/models" or line == "/model":
            choices, notes = _inventory(runtime)
            print(format_model_list(choices, notes) + "\n")
            continue
        if line.startswith("/model "):
            choices, _ = _inventory(runtime)
            picked = pick_model(choices, line.split(maxsplit=1)[1].strip())
            if isinstance(picked, str):
                print(picked + "\n")
            else:
                runtime.switch_model(picked.backend, picked.model)
                print(f"now using {picked.model} [{picked.backend}] (reverts on restart)\n")
            continue
        if line == "/doff":
            persona = runtime.doff()
            print("baseline restored.\n" if persona else "no masque was active.\n")
            continue
        if don := parse_don_command(line):
            _handle_don(runtime, *don)
            continue
        if line == "/retry":
            if not last_text:
                print("nothing to retry yet — send a message first.\n")
                continue
            line = last_text

        if command := parse_skill_command(line):
            _run_skill(runtime, *command)
            continue

        last_text = line
        event = MessageEvent(source=source, text=line)
        for message in runtime.agent.stream(event.text, session_key):
            _render(message)

    print("bye.")
