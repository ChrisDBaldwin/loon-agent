"""Configuration: typed settings + an env-defined backend registry.

A single ``ChatOpenAI(base_url=...)`` talks to every backend, so switching inference
targets is just a name lookup. Backends are defined entirely from the environment /
``.env``: setting ``LOON_<NAME>_BASE_URL`` creates backend ``<name>``, with
``LOON_<NAME>_MODEL`` / ``LOON_<NAME>_API_KEY`` alongside. One generic ``local``
backend (LM Studio's default port) ships as a starting point — see ``.env.example``.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


# Local OpenAI-compatible servers usually ignore auth; the SDK just needs a non-empty key.
# Servers with token auth enabled (e.g. LM Studio's "API token authentication") are the
# exception — set a real token via LOON_<NAME>_API_KEY for those backends.
_LOCAL_API_KEY = "not-needed"

# LOON_<NAME>_BASE_URL defines backend <name>; these pick up its other fields.
_BACKEND_ENV_RE = re.compile(r"^LOON_([A-Z0-9]+)_BASE_URL$")


@dataclass(frozen=True)
class Backend:
    """An OpenAI-compatible inference endpoint."""

    base_url: str
    model: str
    api_key: str = _LOCAL_API_KEY


# The out-of-the-box backend: an OpenAI-compatible server on this machine (LM Studio's
# default port; vLLM/Ollama/llama.cpp all work — point LOON_LOCAL_BASE_URL wherever).
# The model id must match what your server is serving: set LOON_LOCAL_MODEL.
DEFAULT_BACKENDS: dict[str, Backend] = {
    "local": Backend("http://localhost:1234/v1", ""),
}


class Settings(BaseSettings):
    """Process-wide settings, populated from the environment (prefix ``LOON_``)."""

    model_config = SettingsConfigDict(env_prefix="LOON_", extra="ignore")

    backend: str = "local"
    temperature: float = 0.0
    data_dir: Path = Path(".loon")
    # off | console | otlp
    otel: str = "off"

    # Long-term memory (see memory/provider.py). sqlite = FTS5 keyword recall, no
    # extra deps; chroma = local embeddings + nearest-neighbor recall, survives paraphrase.
    memory_backend: str = "sqlite"

    # Skills + masques (see docs/spec-research-skills.md).
    skills_dir: Path = Path("skills")
    # Self-directed processing loops (see docs/loops.md).
    loops_dir: Path = Path("loops")
    masques_dir: Path | None = None  # extra masque catalog (a masques-style personas dir)
    masque: str | None = None  # optional lens donned by the chat agent itself
    step_input_budget: int = 4000  # approx tokens per assembled skill-step prompt
    step_max_tokens: int = 3000  # output cap per step call (reasoning needs headroom)
    research_sources: int = 5  # pages fetched/summarized per research run

    # Telegram adapter (see adapters/telegram.py).
    telegram_token: str | None = None
    # Comma-separated numeric Telegram user ids. Empty -> deny everyone (safe default);
    # the refusal message includes the sender's id, which is how you discover yours.
    telegram_allowed_users: str = ""

    # Sandboxed exec/file capability (see exec/ and skills/code.md). Disabled by default:
    # the /code skill's tools are only wired in when a backend is chosen. "docker" is the
    # only isolation-backed option; there is deliberately no unsandboxed host-subprocess mode.
    exec_backend: str = "off"  # off | docker
    exec_workspace: Path = Path(".loon/workspace")
    exec_image: str = ""  # required when exec_backend=docker; no default forces an explicit choice
    exec_network: str = "none"  # none | bridge
    exec_timeout: float = 120.0
    exec_memory_limit: str = "512m"
    exec_cpu_limit: float = 1.0
    exec_pids_limit: int = 128
    exec_user: str = "1000:1000"
    # Comma-separated program names loon may run in the sandbox. Empty -> deny all (safe default).
    exec_allowed_bins: str = ""
    # Also expose sandboxed run_command in the chat loop (not just the /code skill). The
    # chat loop carries untrusted web content, so this variant always runs with the
    # network locked to "none" regardless of exec_network — see docs/exec-sandbox.md.
    exec_chat: bool = False
    # Comma-separated read-only bind mounts "host:container" (e.g. the loon repo at
    # /repo). Empty -> the sandbox sees only the workspace. Curated allowlist, not a
    # general host-FS door: each entry is an explicit trust decision.
    exec_ro_mounts: str = ""

    # Internal website (see adapters/web.py): serves the HTML loon publishes over the LAN.
    # host 0.0.0.0 makes it reachable network-wide (e.g. http://pontoon.local:8800), not just
    # localhost. web_root is a curated publish dir — NOT all of data_dir (which holds DBs).
    web_host: str = "0.0.0.0"
    web_port: int = 8800
    web_root: Path = Path(".loon/site")

    def exec_allowlist(self) -> frozenset[str]:
        """Program names the sandbox may run (basename-matched; empty = deny all)."""
        return frozenset(part.strip() for part in self.exec_allowed_bins.split(",") if part.strip())

    def exec_ro_mount_pairs(self) -> tuple[tuple[str, str], ...]:
        """Parsed (host, container) read-only mounts. Malformed entries raise ValueError —
        a mount is a trust boundary, so misconfiguration must be loud, not skipped."""
        pairs = []
        for entry in self.exec_ro_mounts.split(","):
            entry = entry.strip()
            if not entry:
                continue
            host, sep, container = entry.rpartition(":")
            if not sep or not host or not container.startswith("/"):
                raise ValueError(
                    f"LOON_EXEC_RO_MOUNTS entry {entry!r} is not 'host:container' "
                    "with an absolute container path"
                )
            pairs.append((host, container))
        return tuple(pairs)

    def telegram_allowlist(self) -> frozenset[int]:
        """Numeric user ids allowed to talk to the Telegram bot (empty = deny all)."""
        ids = set()
        for part in self.telegram_allowed_users.split(","):
            part = part.strip()
            if part:
                try:
                    ids.add(int(part))
                except ValueError as exc:
                    raise ValueError(
                        f"LOON_TELEGRAM_ALLOWED_USERS must be comma-separated numeric "
                        f"Telegram user ids; got {part!r}"
                    ) from exc
        return frozenset(ids)

    def backends(self) -> dict[str, Backend]:
        """Backend registry: defaults plus every LOON_<NAME>_BASE_URL in the environment."""
        names = set(DEFAULT_BACKENDS)
        for key in os.environ:
            if match := _BACKEND_ENV_RE.match(key):
                names.add(match.group(1).lower())

        resolved: dict[str, Backend] = {}
        for name in sorted(names):
            default = DEFAULT_BACKENDS.get(name, Backend("", ""))
            prefix = f"LOON_{name.upper()}"
            resolved[name] = Backend(
                base_url=os.environ.get(f"{prefix}_BASE_URL") or default.base_url,
                model=os.environ.get(f"{prefix}_MODEL") or default.model,
                api_key=os.environ.get(f"{prefix}_API_KEY") or default.api_key,
            )
        return resolved

    def resolve_backend(self, name: str | None = None) -> Backend:
        """Look up a backend by name (defaults to the configured ``backend``)."""
        name = name or self.backend
        registry = self.backends()
        if name not in registry:
            available = ", ".join(registry)
            raise KeyError(
                f"Unknown backend {name!r}. Available: {available}. "
                f"Define one by setting LOON_{name.upper()}_BASE_URL in .env."
            )
        backend = registry[name]
        if not backend.model:
            raise ValueError(
                f"Backend {name!r} has no model configured — set LOON_{name.upper()}_MODEL "
                "to the model id your server is serving."
            )
        return backend


def get_settings() -> Settings:
    """Construct settings from the current environment."""
    return Settings()
