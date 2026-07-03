"""Application assembly: wire settings -> llm, tools, checkpointer, memory, skills.

Kept separate from any single adapter so the CLI and Telegram build the same runtime
the same way. ``build_agent`` remains the light entry point for chat-only adapters;
``build_runtime`` adds the skill engine (discovered skills, tool registry, masques).
"""

from __future__ import annotations

import re
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from .config import Settings, get_settings
from .exec.backend import ExecBackend
from .exec.docker_backend import DockerExecBackend, DockerLimits
from .graph import LoonAgent
from .llm import make_llm
from .masques import MasqueLoader
from .memory import ChromaMemoryProvider, SqliteMemoryProvider
from .memory.provider import MemoryProvider
from .report import render_report, write_report
from .session import SessionEpochs
from .skills import Skill, discover_skills
from .skills.engine import SkillRunner
from .telemetry import setup_telemetry
from .tools import DEFAULT_TOOLS
from .tools.exec import delete_file, edit_file, run_command, write_file
from .tools.web import FetchedPage, fetch_page, web_search

_MEMORY_TLDR_CHARS = 400


@dataclass
class LoonRuntime:
    """Everything an adapter needs: the chat agent plus the skill machinery."""

    agent: LoonAgent
    skills: dict[str, Skill]
    runner: SkillRunner
    settings: Settings
    epochs: SessionEpochs


def build_runtime(
    settings: Settings | None = None,
    *,
    progress: Callable[[str], None] | None = None,
) -> LoonRuntime:
    """Construct the fully-wired runtime (agent + skills) from settings."""
    settings = settings or get_settings()
    setup_telemetry(settings)

    data_dir = Path(settings.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # One sqlite connection per store; check_same_thread=False so a future async/threaded
    # adapter can share it.
    checkpoint_conn = sqlite3.connect(data_dir / "checkpoints.sqlite", check_same_thread=False)
    checkpointer = SqliteSaver(checkpoint_conn)

    memory: MemoryProvider
    if settings.memory_backend == "chroma":
        memory = ChromaMemoryProvider(
            db_path=data_dir / "chroma",
            notes_path=data_dir / "MEMORY.md",
        )
    elif settings.memory_backend == "sqlite":
        memory = SqliteMemoryProvider(
            db_path=data_dir / "memory.sqlite",
            notes_path=data_dir / "MEMORY.md",
        )
    else:
        raise ValueError(
            f"Unknown LOON_MEMORY_BACKEND {settings.memory_backend!r}; "
            "expected 'sqlite' or 'chroma'."
        )

    # Local masques/ wins on collisions; LOON_MASQUES_DIR extends the catalog
    # (point it at any masques-style personas directory).
    masque_dirs: list[Path] = [Path("masques")]
    if settings.masques_dir:
        masque_dirs.append(Path(settings.masques_dir))
    masques = MasqueLoader(masque_dirs)
    persona = masques.block(settings.masque) if settings.masque else None

    llm = make_llm(settings=settings)
    agent = LoonAgent(
        llm, DEFAULT_TOOLS, checkpointer=checkpointer, memory=memory, persona=persona
    )

    backend = settings.resolve_backend()
    tools = {
        "web_search": lambda query: web_search(str(query)),
        "fetch_page": _fetch_or_raise,
        "publish_report": _make_publish(memory, settings, model_label=backend.model),
    }
    # Exec/file tools live ONLY in the skill registry (reachable via the /code skill),
    # never in DEFAULT_TOOLS — the chat loop handles untrusted fetched web content, so an
    # exec tool must never share that loop. Added only when a backend is configured.
    tools.update(_exec_tools(settings))
    runner = SkillRunner(
        llm,
        tools,
        masque_loader=masques.block,
        input_budget=settings.step_input_budget,
        max_output_tokens=settings.step_max_tokens,
        progress=progress,
    )

    return LoonRuntime(
        agent=agent,
        skills=discover_skills(settings.skills_dir),
        runner=runner,
        settings=settings,
        epochs=SessionEpochs(data_dir / "sessions.sqlite"),
    )


def build_agent(settings: Settings | None = None) -> LoonAgent:
    """Construct just the chat agent (chat-only adapters)."""
    return build_runtime(settings).agent


# --- skill tools ------------------------------------------------------------------

_URL_RE = re.compile(r"https?://\S+")


def _cited_pages(pages: list[FetchedPage], notes: object) -> list[FetchedPage]:
    """Reorder pages to match the notes' citation order (each note leads with its URL).

    Pages that produced no note (summarize skipped them) drop out — they are already
    listed under failures. If no note can be matched back to a page (the model mangled
    the URLs), fall back to the fetched order rather than an empty source list.
    """
    remaining = list(pages)
    ordered: list[FetchedPage] = []
    for note in notes if isinstance(notes, list) else []:
        match = _URL_RE.search(str(note))
        if not match:
            continue
        url = match.group(0).rstrip(".,;)")
        for page in remaining:
            if page.url == url or page.url in url or url in page.url:
                ordered.append(page)
                remaining.remove(page)
                break
    return ordered or pages


def _fetch_or_raise(url: object) -> FetchedPage:
    """fetch_page for foreach use: bad urls / failed fetches raise so the item is
    skipped and recorded, keeping only readable pages in the pipeline."""
    url = str(url).strip()
    if not url.startswith(("http://", "https://")):
        raise ValueError(f"not a url: {url!r}")
    page = fetch_page(url)
    if not page.ok:
        raise RuntimeError(page.error)
    return page


def _make_exec_backend(settings: Settings) -> ExecBackend:
    """Build the configured isolation backend, or fail loudly on misconfiguration."""
    if settings.exec_backend == "docker":
        if not settings.exec_image:
            raise ValueError(
                "LOON_EXEC_BACKEND=docker requires LOON_EXEC_IMAGE (a pinned toolbox image)."
            )
        return DockerExecBackend(
            image=settings.exec_image,
            limits=DockerLimits(
                network=settings.exec_network,
                memory=settings.exec_memory_limit,
                cpus=settings.exec_cpu_limit,
                pids=settings.exec_pids_limit,
                user=settings.exec_user,
            ),
        )
    raise ValueError(
        f"Unknown LOON_EXEC_BACKEND {settings.exec_backend!r}; expected 'off' or 'docker'."
    )


def _exec_tools(settings: Settings) -> dict[str, Callable[[object], object]]:
    """Exec/file tools for the skill registry — empty unless a backend is configured.

    Unlike ``_fetch_or_raise``, these never raise: a denied or failed command is a
    meaningful result for a coding task, so it flows into the step results (as an
    error-carrying ``ExecResult``/``FileOpResult``) for the report step to surface —
    the deny-and-report behavior. Registered only in the skill registry, never in
    ``DEFAULT_TOOLS``.
    """
    if settings.exec_backend == "off":
        return {}

    backend = _make_exec_backend(settings)
    workspace = Path(settings.exec_workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    allowed = settings.exec_allowlist()
    timeout = settings.exec_timeout

    return {
        "run_command": lambda cmd: run_command(
            str(cmd), backend=backend, workspace=workspace,
            allowed_bins=allowed, timeout=timeout,
        ),
        "write_file": lambda item: write_file(
            str(_field(item, "path")), str(_field(item, "content")), workspace=workspace
        ),
        "edit_file": lambda item: edit_file(
            str(_field(item, "path")), str(_field(item, "content")), workspace=workspace
        ),
        "delete_file": lambda item: delete_file(str(_field(item, "path")), workspace=workspace),
    }


def _field(item: object, key: str) -> object:
    """Pull a field from a foreach item that may be a mapping or a bare value."""
    if isinstance(item, Mapping):
        return item.get(key, "")
    return item if key == "path" else ""


def _make_publish(
    memory: MemoryProvider | None, settings: Settings, *, model_label: str
) -> Callable[[Mapping[str, object]], str]:
    def publish_report(context: Mapping[str, object]) -> str:
        topic = str(context.get("topic", "untitled"))
        briefing = str(context.get("briefing", ""))
        pages = [p for p in context.get("pages") or [] if isinstance(p, FetchedPage)]
        failures = [str(f) for f in context.get("failures") or []]

        # The briefing cites [n] in *note* order, and summarize may have skipped some
        # pages — list sources in note order so citations line up.
        pages = _cited_pages(pages, context.get("notes") or [])

        html_text = render_report(
            topic=topic,
            briefing_md=briefing,
            pages=pages,
            failures=failures,
            model=model_label,
            backend=settings.backend,
        )
        path = write_report(html_text, topic, Path(settings.data_dir) / "reports")

        if memory is not None:
            tldr = briefing.strip()[:_MEMORY_TLDR_CHARS]
            memory.sync_turn(
                f"research: {topic}",
                f"{tldr}\n\nfull report: {path}",
                session_id="skill:research",
            )
        return str(path)

    return publish_report
