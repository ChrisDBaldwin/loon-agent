"""Self-directed processing loops: host-driven iterations of a stored prompt.

Loon has no background existence of its own — it runs only while a turn is being
processed. A *loop* gives it one anyway, from the outside in: a loop file (YAML
frontmatter + an iteration prompt, ``loops/*.md``) tells the host *what* to wake loon
up with, and an adapter with a persistent event loop (Telegram today) owns the *when*
— sleeping the interval, invoking one agent turn per iteration, and delivering the
reply back to the chat that started it.

Each iteration is an ordinary chat turn in its own fresh thread
(``loop:<name>:i<n>``), so a small local model never accumulates unbounded context
across a long-running loop. Continuity lives outside the thread instead: the
follow-ups store (``tools/followups.py``), the website pages a loop maintains, and
long-term memory write-back — the same durable state chat turns use. The turn signals
completion by ending its reply with ``LOOP_DONE`` (otherwise ``LOOP_CONTINUE``);
``max_iterations`` caps the run regardless, so a loop that never says done still
terminates.
"""

from __future__ import annotations

import re
import sqlite3
import threading
from dataclasses import dataclass
from pathlib import Path

import yaml

from .mirror import utc_now

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)

DONE_MARKER = "LOOP_DONE"
CONTINUE_MARKER = "LOOP_CONTINUE"

# Floor on the iteration interval: loops share one local model with interactive chat,
# so a misconfigured (or model-suggested) tight loop must not be able to occupy it.
MIN_INTERVAL_SECONDS = 60.0

_DEFAULT_MAX_ITERATIONS = 10
_MAX_MAX_ITERATIONS = 100

_PROTOCOL = (
    "\n\n(Autonomous loop protocol: this is iteration {iteration} of at most "
    "{max_iterations} — no human is reading this conversation live, so do not ask "
    "questions or wait for input. If the loop's overall goal is fully complete and "
    f"nothing is left to do, end your reply with the line {DONE_MARKER}. Otherwise "
    f"end it with the line {CONTINUE_MARKER}.)"
)


class LoopParseError(ValueError):
    """A loop file is malformed; the message says which file and what's wrong."""

    def __init__(self, path: Path | str, problem: str) -> None:
        super().__init__(f"{path}: {problem}")


@dataclass(frozen=True)
class LoopSpec:
    """One loop definition: what to wake the agent up with, how often, how many times."""

    name: str
    description: str
    interval: float  # seconds between iterations
    max_iterations: int
    prompt: str  # iteration prompt body ({iteration}/{max_iterations} placeholders)
    path: Path | None = None


def parse_loop(text: str, path: Path | str = "<string>") -> LoopSpec:
    """Parse a loop markdown document; raise :class:`LoopParseError` with specifics."""
    match = _FRONTMATTER_RE.match(text)
    if not match:
        raise LoopParseError(path, "missing YAML frontmatter (--- ... ---)")
    try:
        meta = yaml.safe_load(match.group(1))
    except yaml.YAMLError as exc:
        raise LoopParseError(path, f"frontmatter is not valid YAML: {exc}") from exc
    if not isinstance(meta, dict):
        raise LoopParseError(path, "frontmatter must be a YAML mapping")

    name = meta.get("name")
    if not name or not isinstance(name, str):
        raise LoopParseError(path, "frontmatter needs a string 'name'")

    interval = meta.get("interval")
    if not isinstance(interval, (int, float)) or isinstance(interval, bool):
        raise LoopParseError(path, "frontmatter needs a numeric 'interval' (seconds)")
    if interval < MIN_INTERVAL_SECONDS:
        raise LoopParseError(
            path, f"'interval' must be >= {MIN_INTERVAL_SECONDS:.0f}s, got {interval}"
        )

    max_iterations = meta.get("max_iterations", _DEFAULT_MAX_ITERATIONS)
    if (
        not isinstance(max_iterations, int)
        or isinstance(max_iterations, bool)
        or not 1 <= max_iterations <= _MAX_MAX_ITERATIONS
    ):
        raise LoopParseError(path, f"'max_iterations' must be an int in 1..{_MAX_MAX_ITERATIONS}")

    prompt = text[match.end() :].strip()
    if not prompt:
        raise LoopParseError(path, "no iteration prompt after the frontmatter")

    return LoopSpec(
        name=name,
        description=str(meta.get("description", "")),
        interval=float(interval),
        max_iterations=max_iterations,
        prompt=prompt,
        path=Path(path) if path != "<string>" else None,
    )


def load_loop(path: Path | str) -> LoopSpec:
    path = Path(path)
    return parse_loop(path.read_text(encoding="utf-8"), path)


def discover_loops(directory: Path | str) -> dict[str, LoopSpec]:
    """Load every ``*.md`` in ``directory`` (missing directory -> empty registry)."""
    directory = Path(directory)
    if not directory.is_dir():
        return {}
    loops: dict[str, LoopSpec] = {}
    for file in sorted(directory.glob("*.md")):
        spec = load_loop(file)
        if spec.name in loops:
            raise LoopParseError(file, f"duplicate loop name {spec.name!r}")
        loops[spec.name] = spec
    return loops


# --- running one iteration ----------------------------------------------------------


@dataclass(frozen=True)
class IterationResult:
    reply: str
    done: bool


def iteration_session_key(name: str, iteration: int) -> str:
    return f"loop:{name}:i{iteration}"


def iteration_prompt(spec: LoopSpec, iteration: int) -> str:
    body = spec.prompt.replace("{iteration}", str(iteration)).replace(
        "{max_iterations}", str(spec.max_iterations)
    )
    return body + _PROTOCOL.format(iteration=iteration, max_iterations=spec.max_iterations)


def is_done(reply: str) -> bool:
    """The turn declared the loop finished: DONE_MARKER on one of the last few lines.

    Checked near the end only, so prose *about* the protocol earlier in the reply
    doesn't end the loop. Absent both markers, the loop continues — the iteration cap
    is the real backstop.
    """
    tail = [line for line in reply.strip().splitlines() if line.strip()][-3:]
    return any(DONE_MARKER in line for line in tail)


def run_iteration(agent, spec: LoopSpec, iteration: int) -> IterationResult:
    """One loop iteration = one full agent turn in its own fresh thread."""
    reply = agent.invoke(
        iteration_prompt(spec, iteration), iteration_session_key(spec.name, iteration)
    )
    return IterationResult(reply=reply, done=is_done(reply))


# --- persistence ---------------------------------------------------------------------


@dataclass(frozen=True)
class LoopRun:
    """The persisted state of one loop: enough to resume it after a service restart."""

    name: str
    chat_id: str
    iteration: int
    status: str  # running | done | stopped | failed


class LoopStore:
    """Which loops are (or were) running — survives service restarts.

    One row per loop name: (re)starting a loop overwrites its previous run. An adapter
    reads ``running()`` at startup to resume loops the last process didn't finish.
    """

    def __init__(self, db_path: Path | str) -> None:
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS loop_runs ("
            "name TEXT PRIMARY KEY,"
            "chat_id TEXT NOT NULL,"
            "iteration INTEGER NOT NULL DEFAULT 0,"
            "status TEXT NOT NULL,"
            "started_at TEXT NOT NULL,"
            "updated_at TEXT NOT NULL)"
        )
        self._conn.commit()

    def activate(self, name: str, chat_id: str) -> None:
        now = utc_now()
        with self._lock:
            self._conn.execute(
                "INSERT INTO loop_runs (name, chat_id, iteration, status, started_at, updated_at) "
                "VALUES (?, ?, 0, 'running', ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET chat_id = excluded.chat_id, iteration = 0, "
                "status = 'running', started_at = excluded.started_at, "
                "updated_at = excluded.updated_at",
                (name, chat_id, now, now),
            )
            self._conn.commit()

    def record_iteration(self, name: str, iteration: int) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE loop_runs SET iteration = ?, updated_at = ? WHERE name = ?",
                (iteration, utc_now(), name),
            )
            self._conn.commit()

    def finish(self, name: str, status: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE loop_runs SET status = ?, updated_at = ? WHERE name = ?",
                (status, utc_now(), name),
            )
            self._conn.commit()

    def get(self, name: str) -> LoopRun | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT name, chat_id, iteration, status FROM loop_runs WHERE name = ?",
                (name,),
            ).fetchone()
        return LoopRun(*row) if row else None

    def running(self) -> list[LoopRun]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT name, chat_id, iteration, status FROM loop_runs "
                "WHERE status = 'running' ORDER BY name"
            ).fetchall()
        return [LoopRun(*row) for row in rows]
