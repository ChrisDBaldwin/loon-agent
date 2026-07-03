"""Sandboxed execution: a layered defense for letting loon run commands + touch files.

Three layers, none of which alone is the boundary: the :mod:`policy` allow/deny engine
(pure, default-deny), an :class:`~loon_agent.exec.backend.ExecBackend` isolation boundary
(:class:`~loon_agent.exec.docker_backend.DockerExecBackend` in v1), and audit attributes on
the skill engine's existing OTel spans. Wired only into the deterministic skill registry
(via a ``/code`` skill) — never into the chat loop's ``DEFAULT_TOOLS``, so untrusted fetched
web content can never reach an exec tool.
"""

from .backend import ExecBackend, ExecResult
from .docker_backend import DockerExecBackend, DockerLimits
from .policy import PolicyDecision, check_command, check_path

__all__ = [
    "DockerExecBackend",
    "DockerLimits",
    "ExecBackend",
    "ExecResult",
    "PolicyDecision",
    "check_command",
    "check_path",
]
