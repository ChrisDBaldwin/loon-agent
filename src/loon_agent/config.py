"""Configuration: typed settings + the homelab backend registry.

A single ``ChatOpenAI(base_url=...)`` talks to every backend, so switching inference
targets is just a name lookup here. Defaults are baked in; any field can be overridden
from the environment / ``.env`` (see ``.env.example``).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()


# Local OpenAI-compatible servers usually ignore auth; the SDK just needs a non-empty key.
# LM Studio with "API token authentication" enabled is the exception — set a real token via
# LOON_<NAME>_API_KEY for those backends.
_LOCAL_API_KEY = "not-needed"


@dataclass(frozen=True)
class Backend:
    """An OpenAI-compatible inference endpoint."""

    base_url: str
    model: str
    api_key: str = _LOCAL_API_KEY


# Default homelab backends. Override per-field via
# LOON_<NAME>_BASE_URL / LOON_<NAME>_MODEL / LOON_<NAME>_API_KEY.
DEFAULT_BACKENDS: dict[str, Backend] = {
    "pontoon": Backend("http://pontoon.lan:1234/v1", "mlx-community/Qwen2.5-7B-Instruct-4bit"),
    "ironwood": Backend("http://ironwood.lan:8000/v1", "Qwen/Qwen2.5-14B-Instruct-AWQ"),
    "wsl": Backend("http://localhost:8000/v1", "Qwen/Qwen2.5-7B-Instruct-AWQ"),
}


class Settings(BaseSettings):
    """Process-wide settings, populated from the environment (prefix ``LOON_``)."""

    model_config = SettingsConfigDict(env_prefix="LOON_", extra="ignore")

    backend: str = "wsl"
    temperature: float = 0.0
    data_dir: Path = Path(".loon")
    # off | console | otlp
    otel: str = "off"

    # Skills + masques (see docs/spec-research-skills.md).
    skills_dir: Path = Path("skills")
    masques_dir: Path | None = None  # extra masque catalog (e.g. ~/git/masques/personas)
    masque: str | None = None  # optional lens donned by the chat agent itself
    step_input_budget: int = 4000  # approx tokens per assembled skill-step prompt
    step_max_tokens: int = 3000  # output cap per step call (reasoning needs headroom)
    research_sources: int = 5  # pages fetched/summarized per research run

    # Telegram adapter (see adapters/telegram.py).
    telegram_token: str | None = None
    # Comma-separated numeric Telegram user ids. Empty -> deny everyone (safe default);
    # the refusal message includes the sender's id, which is how you discover yours.
    telegram_allowed_users: str = ""

    # Per-backend overrides (None -> use DEFAULT_BACKENDS value).
    pontoon_base_url: str | None = None
    pontoon_model: str | None = None
    pontoon_api_key: str | None = None
    ironwood_base_url: str | None = None
    ironwood_model: str | None = None
    ironwood_api_key: str | None = None
    wsl_base_url: str | None = None
    wsl_model: str | None = None
    wsl_api_key: str | None = None

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
        """Resolved backend registry, applying any env overrides over the defaults."""
        resolved: dict[str, Backend] = {}
        for name, default in DEFAULT_BACKENDS.items():
            base_url = getattr(self, f"{name}_base_url") or default.base_url
            model = getattr(self, f"{name}_model") or default.model
            api_key = getattr(self, f"{name}_api_key") or default.api_key
            resolved[name] = Backend(base_url, model, api_key)
        return resolved

    def resolve_backend(self, name: str | None = None) -> Backend:
        """Look up a backend by name (defaults to the configured ``backend``)."""
        name = name or self.backend
        registry = self.backends()
        if name not in registry:
            available = ", ".join(registry)
            raise KeyError(f"Unknown backend {name!r}. Available: {available}")
        return registry[name]


def get_settings() -> Settings:
    """Construct settings from the current environment."""
    return Settings()
