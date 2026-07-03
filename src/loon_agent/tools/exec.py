"""The exec/file tools the skill registry calls — policy check, then act, then audit.

These are plain functions (not LangChain ``@tool``s), in the same mould as ``tools/web.py``:
error-as-value (never raise here — the ``app.py`` skill-registry wrappers decide raise-vs-skip),
frozen-dataclass results with a ``__str__`` for prompt injection. Every call runs the
:mod:`loon_agent.exec.policy` check *first* and records the verdict — plus command/exit/timing —
onto the current OTel span (the ``execute_tool`` span the skill engine already opened), so there
is an audit trail of everything loon tried to run or write, allowed or denied.

Split by risk: ``run_command`` needs the container isolation boundary; the file ops are scoped
``pathlib`` writes (a scoped write can't spawn a process) gated only by the path-policy check.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from opentelemetry import trace

from ..exec.backend import ExecBackend, ExecResult
from ..exec.policy import check_command, check_path

_MAX_SAMPLE = 2000


@dataclass(frozen=True)
class FileOpResult:
    """The outcome of a create/edit/delete on one workspace file."""

    op: str
    path: str
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def __str__(self) -> str:
        if not self.ok:
            return f"[{self.op} {self.path} failed: {self.error}]"
        return f"{self.op} ok: {self.path}"


def _annotate(**attrs: object) -> None:
    """Attach loon.exec.* attributes to the current span, if one is recording."""
    span = trace.get_current_span()
    if span is None or not span.is_recording():
        return
    for key, value in attrs.items():
        if value is not None:
            span.set_attribute(f"loon.exec.{key}", value)


def run_command(
    command: str,
    *,
    backend: ExecBackend,
    workspace: Path,
    allowed_bins: frozenset[str],
    timeout: float,
) -> ExecResult:
    """Policy-check ``command`` then run it in the isolation backend; audit either way."""
    command = str(command).strip()
    decision = check_command(command, allowed_bins)
    _annotate(command=command, workspace=str(workspace), policy_decision=decision.reason)
    if not decision.allowed:
        return ExecResult(command=command, error=f"policy {decision.reason}")

    result = backend.run(command, cwd=workspace, timeout=timeout)
    stdout_sample, stderr_sample = result.sample()
    _annotate(
        backend=type(backend).__name__,
        exit_code=result.exit_code,
        duration_s=result.duration_s,
        stdout_sample=stdout_sample,
        stderr_sample=stderr_sample,
        error=result.error,
    )
    return result


def write_file(path: str, content: str, *, workspace: Path) -> FileOpResult:
    """Create/overwrite a file, but only inside the workspace."""
    return _file_op("write", path, workspace, lambda p: _do_write(p, content))


def edit_file(path: str, content: str, *, workspace: Path) -> FileOpResult:
    """Replace a file's contents (v1: full-content replace, not a diff), workspace-scoped."""
    return _file_op(
        "edit", path, workspace,
        lambda p: _do_write(p, content) if p.exists() else _missing(p),
    )


def delete_file(path: str, *, workspace: Path) -> FileOpResult:
    """Delete a file inside the workspace."""
    return _file_op("delete", path, workspace, _do_delete)


# --- internals ------------------------------------------------------------------


def _file_op(op, path, workspace, action) -> FileOpResult:  # noqa: ANN001
    path = str(path).strip()
    decision = check_path(path, workspace)
    _annotate(command=f"{op} {path}", workspace=str(workspace), policy_decision=decision.reason)
    if not decision.allowed:
        return FileOpResult(op=op, path=path, error=f"policy {decision.reason}")

    target = Path(path) if Path(path).is_absolute() else (Path(workspace) / path)
    try:
        action(target.resolve())
    except OSError as exc:
        _annotate(error=str(exc))
        return FileOpResult(op=op, path=path, error=str(exc))
    return FileOpResult(op=op, path=path)


def _do_write(target: Path, content: str) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def _do_delete(target: Path) -> None:
    target.unlink()


def _missing(target: Path) -> None:
    raise OSError(f"no such file: {target.name}")
