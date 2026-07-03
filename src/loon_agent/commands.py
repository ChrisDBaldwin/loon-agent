"""Shared slash-command logic for the CLI and Telegram adapters.

The adapters own transport (parsing, replies, auth); this module owns the substance:
the live model inventory (``/models``), index-based switching (``/model 2``), the
status report, and the help text — so both surfaces behave identically.

The inventory is built by asking every configured backend's OpenAI-compatible
``/models`` endpoint what it can serve, sorted deterministically (backend name, then
model id) so the indexes are stable between ``/models`` and ``/model <n>`` without
needing per-chat state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from .config import Backend, Settings

_STARTED = time.monotonic()
_PROBE_TIMEOUT = 4.0

HELP_TEXT = """loon — commands:
/new — start a fresh conversation (old thread stays on disk)
/retry — send your previous message again
/models — list models available across configured backends
/model <n> — switch to model #n from /models (until restart)
/don <name> [intent] — become a persona (prompt + tools + memory)
/doff — return to baseline
/status — backend, model, server health, session info
/help — this text

Anything else is a chat message. Research runs via the CLI: /skill research <topic>."""


@dataclass(frozen=True)
class ModelChoice:
    index: int
    backend: str
    model: str
    active: bool = False


def probe_models(backend: Backend, timeout: float = _PROBE_TIMEOUT) -> tuple[list[str], float]:
    """Model ids a backend's server reports, plus latency (seconds). Raises on failure."""
    started = time.monotonic()
    response = httpx.get(
        f"{backend.base_url.rstrip('/')}/models",
        headers={"Authorization": f"Bearer {backend.api_key}"},
        timeout=timeout,
    )
    response.raise_for_status()
    rows = response.json().get("data", [])
    ids = sorted(
        {
            str(row["id"])
            for row in rows
            if isinstance(row, dict) and row.get("id")
            # Servers list embedding models alongside chat models (the OpenAI-compatible
            # /models has no type field); switching chat to one would brick the loop.
            and "embed" not in str(row["id"]).lower()
        }
    )
    return ids, time.monotonic() - started


def model_inventory(
    settings: Settings, *, active_backend: str, active_model: str
) -> tuple[list[ModelChoice], list[str]]:
    """Numbered (backend, model) choices across all configured backends + skip notes.

    Unreachable backends contribute a note instead of entries — switching to a dead
    server would only produce a slower failure later.
    """
    choices: list[ModelChoice] = []
    notes: list[str] = []
    for name, backend in sorted(settings.backends().items()):
        if not backend.base_url:
            continue
        try:
            ids, _ = probe_models(backend)
        except Exception as exc:
            notes.append(f"{name} ({backend.base_url}): unreachable — {_short(exc)}")
            continue
        if backend.model and backend.model not in ids:
            ids.insert(0, backend.model)  # configured model always switchable
        for model_id in ids:
            choices.append(
                ModelChoice(
                    index=len(choices) + 1,
                    backend=name,
                    model=model_id,
                    active=(name == active_backend and model_id == active_model),
                )
            )
    return choices, notes


def format_model_list(choices: list[ModelChoice], notes: list[str]) -> str:
    if not choices and not notes:
        return "No backends configured — set LOON_<NAME>_BASE_URL in .env."
    lines = ["Models (switch with /model <n>):"] if choices else []
    for choice in choices:
        marker = "→" if choice.active else " "
        lines.append(f"{marker} {choice.index}. {choice.model}  [{choice.backend}]")
    lines.extend(f"! {note}" for note in notes)
    return "\n".join(lines)


def pick_model(choices: list[ModelChoice], arg: str) -> ModelChoice | str:
    """Resolve a /model argument to a choice, or return a usage/error message."""
    if not arg.isdigit():
        return "Usage: /model <n> — run /models to see the numbered list."
    index = int(arg)
    if not choices:
        return "No models available to switch to — run /models to see why."
    if not 1 <= index <= len(choices):
        return f"No model #{index} — /models lists 1–{len(choices)}."
    return choices[index - 1]


def status_text(runtime, session_key: str) -> str:
    """One-screen health report: backend, server, session, runtime plumbing."""
    settings = runtime.settings
    backend = settings.backends().get(runtime.active_backend)

    if backend is None:
        server = f"backend {runtime.active_backend!r} not in registry"
    else:
        try:
            _, latency = probe_models(backend)
            server = f"ok ({latency * 1000:.0f} ms)"
        except Exception as exc:
            server = f"UNREACHABLE — {_short(exc)}"

    lines = [
        f"backend: {runtime.active_backend} ({backend.base_url if backend else '?'})",
        f"model: {runtime.active_model}",
        f"server: {server}",
        f"session: {session_key} · {_thread_message_count(runtime.agent, session_key)} messages",
        f"memory: {settings.memory_backend} · "
        f"skills: {', '.join(sorted(runtime.skills)) or 'none'}",
        f"up: {_human_duration(time.monotonic() - _STARTED)}",
    ]
    return "\n".join(lines)


# --- internals ---------------------------------------------------------------------


def _thread_message_count(agent, session_key: str) -> int:
    try:
        state = agent.graph.get_state({"configurable": {"thread_id": session_key}})
        return len(state.values.get("messages", []))
    except Exception:
        return 0


def _human_duration(seconds: float) -> str:
    minutes, _ = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    days, hours = divmod(hours, 24)
    if days:
        return f"{days}d {hours}h {minutes}m"
    if hours:
        return f"{hours}h {minutes}m"
    return f"{minutes}m"


def _short(exc: Exception, limit: int = 80) -> str:
    text = str(exc).replace("\n", " ") or type(exc).__name__
    return text if len(text) <= limit else text[: limit - 1] + "…"
