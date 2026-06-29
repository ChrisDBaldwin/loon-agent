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


@dataclass(frozen=True)
class Backend:
    """An OpenAI-compatible inference endpoint."""

    base_url: str
    model: str


# Default homelab backends. Override per-field via LOON_<NAME>_BASE_URL / LOON_<NAME>_MODEL.
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

    # Per-backend overrides (None -> use DEFAULT_BACKENDS value).
    pontoon_base_url: str | None = None
    pontoon_model: str | None = None
    ironwood_base_url: str | None = None
    ironwood_model: str | None = None
    wsl_base_url: str | None = None
    wsl_model: str | None = None

    def backends(self) -> dict[str, Backend]:
        """Resolved backend registry, applying any env overrides over the defaults."""
        resolved: dict[str, Backend] = {}
        for name, default in DEFAULT_BACKENDS.items():
            base_url = getattr(self, f"{name}_base_url") or default.base_url
            model = getattr(self, f"{name}_model") or default.model
            resolved[name] = Backend(base_url, model)
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
