"""The execution-backend contract + its result type.

An :class:`ExecBackend` runs one shell command and returns an :class:`ExecResult`. Keeping
this an ABC (mirroring :class:`loon_agent.memory.provider.MemoryProvider`) means the Docker
backend and a future subprocess/remote backend are interchangeable, and the tool layer
(``tools/exec.py``) depends only on this interface — never on Docker directly.

Like the web tools, backends **degrade instead of raising**: a failed spawn (missing
binary, unreachable daemon, timeout) comes back as an :class:`ExecResult` carrying its
``error`` so the tool layer can decide whether to surface it or skip the item.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

_OUTPUT_SAMPLE_CHARS = 2000


@dataclass(frozen=True)
class ExecResult:
    """The outcome of one command run (success or a captured failure)."""

    command: str
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    duration_s: float = 0.0
    # Set when the command could not be run/completed (spawn failure, timeout, policy
    # denial). A command that ran and exited non-zero is still ``ok`` — that's a normal
    # result, not a backend error; inspect ``exit_code`` for it.
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def __str__(self) -> str:
        if not self.ok:
            return f"[command failed: {self.command}\n{self.error}]"
        head = f"$ {self.command}  (exit {self.exit_code}, {self.duration_s:.1f}s)"
        body = self.stdout.strip()
        if self.stderr.strip():
            body = f"{body}\n[stderr]\n{self.stderr.strip()}" if body else self.stderr.strip()
        return f"{head}\n{body}".strip()

    def sample(self) -> tuple[str, str]:
        """Truncated (stdout, stderr) for telemetry attributes — never the full output."""
        return self.stdout[:_OUTPUT_SAMPLE_CHARS], self.stderr[:_OUTPUT_SAMPLE_CHARS]


class ExecBackend(ABC):
    """Runs a single command in some isolation context and returns its result."""

    @abstractmethod
    def run(self, command: str, *, cwd: Path, timeout: float) -> ExecResult:
        """Execute ``command`` with ``cwd`` as its working directory, bounded by ``timeout``.

        Must not raise for ordinary failures — return an :class:`ExecResult` with ``error``
        set instead, so the caller controls raise-vs-skip semantics.
        """
