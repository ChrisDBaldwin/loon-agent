"""Application assembly: wire settings -> llm, tools, checkpointer, memory, skills.

Kept separate from any single adapter so the CLI and Telegram build the same runtime
the same way. ``build_agent`` remains the light entry point for chat-only adapters;
``build_runtime`` adds the skill engine (discovered skills, tool registry, masques).

The runtime is also the masques reference host (Phase 3): ``don()`` swaps system
prompt, tool access, memory scope and credentials together by rebuilding the
compiled graph — the ``bind`` tier, where a denied tool is structurally absent
from the model's schema rather than advised away.
"""

from __future__ import annotations

import logging
import re
import sqlite3
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.sqlite import SqliteSaver
from masques_core import (
    HOST_NATIVE_SERVER,
    HostSnapshot,
    Persona,
    build_capability_plan,
    compose,
)

from .config import Settings, get_settings
from .credentials import CredentialResolver
from .exec.backend import ExecBackend
from .exec.docker_backend import DockerExecBackend, DockerLimits
from .graph import LoonAgent
from .llm import make_llm
from .masques import MasqueLoader
from .memory import ChromaMemoryProvider, ScopedMemory, SqliteMemoryProvider
from .memory.provider import MemoryProvider
from .mirror import active_record, utc_now, write_mirror
from .report import render_report, write_report
from .session import SessionEpochs
from .skills import Skill, discover_skills
from .skills.engine import SkillRunner
from .telemetry import set_persona_attributes, setup_telemetry
from .tools import DEFAULT_TOOLS
from .tools.exec import delete_file, edit_file, run_command, write_file
from .tools.publish import publish_page
from .tools.site import site_base_url, site_tools
from .tools.web import FetchedPage, fetch_page, web_search

logger = logging.getLogger(__name__)

_MEMORY_TLDR_CHARS = 400


def parse_don_command(line: str) -> tuple[str, str | None] | None:
    """``/don name intent…`` -> (name, intent or None); bare ``/don`` -> ("", None)."""
    line = line.strip()
    if line != "/don" and not line.startswith("/don "):
        return None
    rest = line[len("/don") :].strip()
    if not rest:
        return ("", None)
    name, _, intent = rest.partition(" ")
    return (name, intent.strip() or None)


def bound_tools(all_tools: list[BaseTool], persona: Persona) -> list[BaseTool]:
    """The bind tier: the subset of native tools the persona's ``host`` binding allows.

    A tool outside the allow/deny is never passed to ``bind_tools``/``ToolNode``,
    so it does not exist in the model's tool schema. No sidecar or no ``host``
    binding means the full registry (a lens-only masque changes no capabilities).
    """
    binding = None
    if persona.config is not None:
        binding = next(
            (b for b in persona.config.mcp if b.server == HOST_NATIVE_SERVER), None
        )
    if binding is None:
        return list(all_tools)
    if not binding.enabled:
        return []
    tools = list(all_tools)
    if binding.allow is not None:
        tools = [t for t in tools if t.name in binding.allow]
    if binding.deny:
        tools = [t for t in tools if t.name not in binding.deny]
    return tools


# What loon offers a capability plan: its native tool registry, at the strongest
# tier, with a scoped memory seam (Phase 3 §4-§5).
HOST_SNAPSHOT = HostSnapshot(
    servers=frozenset({HOST_NATIVE_SERVER}), achievable_tier="bind", memory_seam=True
)


@dataclass
class LoonRuntime:
    """Everything an adapter needs: the chat agent plus the skill machinery."""

    agent: LoonAgent
    skills: dict[str, Skill]
    runner: SkillRunner
    settings: Settings
    epochs: SessionEpochs
    # Which (backend, model) the agent is currently talking to (mutated by /model).
    active_backend: str = ""
    active_model: str = ""
    # Don/doff machinery: the parts a graph rebuild needs again.
    llm: BaseChatModel | None = None
    checkpointer: BaseCheckpointSaver | None = None
    base_memory: MemoryProvider | None = None
    masques: MasqueLoader | None = None
    baseline_persona: str | None = None  # settings.masque block, restored at doff
    # The full chat-loop registry (DEFAULT_TOOLS + deployment-bound site tools) that
    # don/doff rebuild from; falls back to DEFAULT_TOOLS when constructed bare in tests.
    chat_tools: list[BaseTool] = field(default_factory=lambda: list(DEFAULT_TOOLS))
    credentials: CredentialResolver = field(default_factory=CredentialResolver)
    active_persona: Persona | None = None
    _mirror_active: dict | None = field(default=None, repr=False)
    _mirror_previous: dict | None = field(default=None, repr=False)

    def don(self, name: str, intent: str | None = None) -> Persona | None:
        """Become ``name``: one operation swapping prompt, tools, memory, credentials.

        Lenient like every masque path — an unknown name warns and returns None,
        leaving the current agent untouched.
        """
        persona = self.masques.resolve(name) if self.masques else None
        if persona is None:
            return None
        plan, bound_refs = build_capability_plan(persona, HOST_SNAPSHOT)
        tools = bound_tools(self.chat_tools, persona)
        memory = self.base_memory
        if memory is not None and persona.config is not None and persona.config.memory:
            memory = ScopedMemory(memory, persona.config.memory)
        block = compose(persona, intent, plan=(plan, bound_refs))["identity_block"]

        if self.active_persona is not None and self._mirror_active is not None:
            # Don-over-don: the outgoing persona doffs first (Phase 2 §4 swap-first).
            self._mirror_previous = {**self._mirror_active, "doffed_at": utc_now()}
        # The shared checkpointer keeps thread continuity across the rebuild.
        self.agent = LoonAgent(
            self.llm, tools, checkpointer=self.checkpointer, memory=memory, persona=block
        )
        self.credentials.activate(persona)
        self.active_persona = persona
        set_persona_attributes(persona.otel_attributes())
        self._mirror_active = active_record(persona, intent, plan, bound_refs)
        write_mirror(Path(self.settings.data_dir), self._mirror_active, self._mirror_previous)
        logger.info(
            "donned %s v%s (tools: %s)",
            persona.name,
            persona.version.raw,
            ", ".join(t.name for t in tools) or "none",
        )
        return persona

    def doff(self) -> Persona | None:
        """Back to baseline: all tools, unscoped memory, no persona block.

        Zeroes any credential material resolved while donned; env:// secrets are
        static, so only the in-process copies are forgotten.
        """
        outgoing = self.active_persona
        self.credentials.deactivate()
        set_persona_attributes(None)
        self.agent = LoonAgent(
            self.llm,
            self.chat_tools,
            checkpointer=self.checkpointer,
            memory=self.base_memory,
            persona=self.baseline_persona,
        )
        self.active_persona = None
        if outgoing is not None and self._mirror_active is not None:
            self._mirror_previous = {**self._mirror_active, "doffed_at": utc_now()}
            logger.info("doffed %s — baseline restored", outgoing.name)
        self._mirror_active = None
        write_mirror(Path(self.settings.data_dir), None, self._mirror_previous)
        return outgoing

    def switch_model(self, backend_name: str, model_id: str) -> None:
        """Point the chat agent and skill runner at a different backend/model.

        Runtime-only (``.env`` untouched; restart reverts). The agent is rebuilt with
        its *current* tools/memory/persona, so a donned masque — including its bind-tier
        tool scoping — survives the model switch. ``self.llm`` is updated too, so a
        later don/doff rebuild also uses the new model.
        """
        self.llm = make_llm(backend_name, settings=self.settings, model=model_id)
        current = self.agent
        self.agent = LoonAgent(
            self.llm,
            current.tools,
            checkpointer=self.checkpointer,
            memory=current.memory,
            persona=current.persona,
        )
        self.runner.llm = self.llm
        self.active_backend = backend_name
        self.active_model = model_id
        logger.info("switched model to %s [%s]", model_id, backend_name)


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
    baseline_persona = masques.block(settings.masque) if settings.masque else None

    llm = make_llm(settings=settings)
    # Chat loop = safe builtins + site management. Site tools write only markdown-rendered
    # pages inside the web root (see tools/site.py), so unlike exec they may share the
    # loop with untrusted fetched web content.
    web_root = Path(settings.web_root)
    chat_tools = [
        *DEFAULT_TOOLS,
        *site_tools(web_root, base_url=site_base_url(settings.web_port)),
    ]
    agent = LoonAgent(
        llm, chat_tools, checkpointer=checkpointer, memory=memory, persona=baseline_persona
    )

    backend = settings.resolve_backend()
    tools = {
        "web_search": lambda query: web_search(str(query)),
        "fetch_page": _fetch_or_raise,
        "publish_report": _make_publish(memory, settings, model_label=backend.model),
        # Publish a markdown page to the internal website (served by adapters/web.py).
        "publish_page": lambda ctx: str(
            publish_page(
                str(ctx.get("title") or ctx.get("topic") or "untitled"),
                str(ctx.get("page") or ""),
                web_root=web_root,
            )
        ),
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
        active_backend=settings.backend,
        active_model=backend.model,
        llm=llm,
        checkpointer=checkpointer,
        base_memory=memory,
        masques=masques,
        baseline_persona=baseline_persona,
        chat_tools=chat_tools,
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
        # Publish into the web root so the report is immediately browsable on the internal
        # site (adapters/web.py), not just a file on disk.
        path = write_report(html_text, topic, Path(settings.web_root))

        if memory is not None:
            tldr = briefing.strip()[:_MEMORY_TLDR_CHARS]
            memory.sync_turn(
                f"research: {topic}",
                f"{tldr}\n\nfull report: {path}",
                session_id="skill:research",
            )
        return str(path)

    return publish_report
