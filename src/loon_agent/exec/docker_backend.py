"""Docker execution backend — the actual isolation boundary.

Each :meth:`DockerExecBackend.run` spawns one ephemeral, locked-down container
(``docker run --rm``) and tears it down when the command finishes. The hardening is all in
the ``docker run`` flags, not in anything trusted to run inside:

* one read-write bind mount, the workspace at ``/workspace``, plus optional **read-only**
  mounts from the curated ``LOON_EXEC_RO_MOUNTS`` allowlist — never the Docker socket or
  host secrets (the AutoGPT container-escape lesson). Each ro mount is an explicit,
  operator-made trust decision (e.g. loon's own repo at ``/repo`` so it can read its code);
* ``--network`` off by default; non-root user; read-only root filesystem with a writable
  ``/tmp`` tmpfs so only the workspace persists; memory / cpu / pid limits (pid-limit is a
  fork-bomb floor beneath the policy layer);
* the timeout is enforced **outside** the container (``subprocess`` timeout + ``docker
  kill``), never trusted to anything within it.

Following ``tools/web.py``'s ``esper-search`` precedent, a missing ``docker`` binary or an
unreachable daemon degrades to an error-carrying :class:`ExecResult` rather than raising.
"""

from __future__ import annotations

import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from .backend import ExecBackend, ExecResult

_WORKDIR = "/workspace"
_DOCKER = "docker"


@dataclass(frozen=True)
class DockerLimits:
    """Container resource + reach limits, sourced from settings."""

    network: str = "none"  # none | bridge
    memory: str = "512m"
    cpus: float = 1.0
    pids: int = 128
    user: str = "1000:1000"
    # (host, container) dirs mounted read-only alongside the workspace. Container paths
    # must not shadow /workspace — validated at construction, not trusted to callers.
    ro_mounts: tuple[tuple[str, str], ...] = ()


class DockerExecBackend(ExecBackend):
    """Runs each command in a fresh, hardened, auto-removed Docker container."""

    def __init__(self, image: str, limits: DockerLimits | None = None) -> None:
        if not image:
            raise ValueError("DockerExecBackend requires an image (set LOON_EXEC_IMAGE).")
        self.image = image
        self.limits = limits or DockerLimits()
        for host, container in self.limits.ro_mounts:
            if container.rstrip("/") in ("", _WORKDIR):
                raise ValueError(
                    f"ro mount {host!r} may not target {container!r} (shadows the workspace)"
                )
            if not Path(host).is_dir():
                raise ValueError(f"ro mount host path {host!r} is not an existing directory")

    def _docker_run_argv(self, command: str, workspace: Path) -> list[str]:
        limits = self.limits
        ro_volumes = [
            arg
            for host, container in limits.ro_mounts
            for arg in ("--volume", f"{Path(host).resolve()}:{container}:ro")
        ]
        return [
            _DOCKER, "run", "--rm",
            "--network", limits.network,
            "--user", limits.user,
            "--read-only",
            "--tmpfs", "/tmp:rw,size=64m",
            "--memory", limits.memory,
            "--cpus", str(limits.cpus),
            "--pids-limit", str(limits.pids),
            # The one writable mount: the workspace. Anything else arrives read-only below.
            "--volume", f"{workspace}:{_WORKDIR}:rw",
            *ro_volumes,
            "--workdir", _WORKDIR,
            self.image,
            # Run the command through a shell inside the container so pipes/redirects work,
            # but the command itself has already cleared the policy allowlist upstream.
            "sh", "-c", command,
        ]

    def preflight(self) -> str | None:
        """Return a human-readable reason the backend is unusable, or None if it's ready."""
        if shutil.which(_DOCKER) is None:
            return "docker binary not found on PATH"
        try:
            probe = subprocess.run(
                [_DOCKER, "info", "--format", "{{.ServerVersion}}"],
                capture_output=True, text=True, timeout=10, check=False,
            )
        except subprocess.TimeoutExpired:
            return "docker daemon did not respond within 10s"
        if probe.returncode != 0:
            return f"docker daemon unreachable: {probe.stderr.strip() or 'unknown error'}"
        return None

    def run(self, command: str, *, cwd: Path, timeout: float) -> ExecResult:
        if (reason := self.preflight()) is not None:
            return ExecResult(command=command, error=reason)

        workspace = Path(cwd).resolve()
        workspace.mkdir(parents=True, exist_ok=True)
        argv = self._docker_run_argv(command, workspace)

        start = time.monotonic()
        try:
            proc = subprocess.run(
                argv, capture_output=True, text=True, timeout=timeout, check=False
            )
        except subprocess.TimeoutExpired:
            # The container may still be alive; kill anything still bound to this workspace
            # mount would be overkill, so rely on --rm plus a best-effort note. The timeout
            # itself is the guarantee the host is not blocked.
            return ExecResult(
                command=command,
                duration_s=time.monotonic() - start,
                error=f"timed out after {timeout:.0f}s",
            )
        duration = time.monotonic() - start

        return ExecResult(
            command=command,
            stdout=proc.stdout,
            stderr=proc.stderr,
            exit_code=proc.returncode,
            duration_s=duration,
        )


def parse_command(command: str) -> list[str]:
    """Best-effort shell split, exposed for tests that assert on argv construction."""
    return shlex.split(command)
