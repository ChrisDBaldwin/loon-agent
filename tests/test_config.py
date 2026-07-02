"""Tests for the env-defined backend registry."""

from __future__ import annotations

import pytest

from loon_agent.config import Settings


def test_default_local_backend_exists(monkeypatch) -> None:
    monkeypatch.delenv("LOON_LOCAL_BASE_URL", raising=False)
    registry = Settings().backends()
    assert "local" in registry
    assert registry["local"].base_url == "http://localhost:1234/v1"


def test_env_defines_a_new_backend(monkeypatch) -> None:
    monkeypatch.setenv("LOON_GPUBOX_BASE_URL", "http://gpubox:8000/v1")
    monkeypatch.setenv("LOON_GPUBOX_MODEL", "Qwen/Qwen2.5-14B-Instruct-AWQ")
    monkeypatch.setenv("LOON_GPUBOX_API_KEY", "sekret")

    backend = Settings(backend="gpubox").resolve_backend()

    assert backend.base_url == "http://gpubox:8000/v1"
    assert backend.model == "Qwen/Qwen2.5-14B-Instruct-AWQ"
    assert backend.api_key == "sekret"


def test_env_overrides_default_backend_fields(monkeypatch) -> None:
    monkeypatch.delenv("LOON_LOCAL_BASE_URL", raising=False)
    monkeypatch.delenv("LOON_LOCAL_API_KEY", raising=False)
    monkeypatch.setenv("LOON_LOCAL_MODEL", "my-model")
    backend = Settings().resolve_backend("local")
    assert backend.base_url == "http://localhost:1234/v1"  # default kept
    assert backend.model == "my-model"
    assert backend.api_key == "not-needed"  # placeholder for auth-less servers


def test_unknown_backend_error_says_how_to_define_one() -> None:
    with pytest.raises(KeyError, match="LOON_GHOST_BASE_URL"):
        Settings().resolve_backend("ghost")


def test_backend_without_model_errors_helpfully(monkeypatch) -> None:
    # The shipped 'local' default deliberately has no model id baked in.
    monkeypatch.delenv("LOON_LOCAL_MODEL", raising=False)
    with pytest.raises(ValueError, match="LOON_LOCAL_MODEL"):
        Settings().resolve_backend("local")
