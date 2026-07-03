"""Tests for runtime assembly (build_runtime) — memory backend selection."""

from __future__ import annotations

import pytest

from loon_agent.app import build_runtime
from loon_agent.config import Settings
from loon_agent.memory import ChromaMemoryProvider, SqliteMemoryProvider


def _settings(tmp_path, *, memory_backend: str = "sqlite", **kwargs) -> Settings:
    # otel="off" and memory_backend pinned explicitly: the real project .env sets
    # LOON_OTEL=otlp and LOON_MEMORY_BACKEND=chroma, and pydantic-settings reads
    # those ambient values for any field not passed here — pin everything these
    # tests assert on so they don't depend on what happens to be in .env.
    return Settings(
        data_dir=tmp_path, backend="local", otel="off", memory_backend=memory_backend, **kwargs
    )


def test_sqlite_is_the_default_memory_backend(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOON_LOCAL_MODEL", "test-model")
    runtime = build_runtime(_settings(tmp_path))
    assert isinstance(runtime.agent.memory, SqliteMemoryProvider)


def test_chroma_backend_selects_chroma_provider(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOON_LOCAL_MODEL", "test-model")
    runtime = build_runtime(_settings(tmp_path, memory_backend="chroma"))
    assert isinstance(runtime.agent.memory, ChromaMemoryProvider)


def test_unknown_memory_backend_errors_helpfully(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOON_LOCAL_MODEL", "test-model")
    with pytest.raises(ValueError, match="LOON_MEMORY_BACKEND"):
        build_runtime(_settings(tmp_path, memory_backend="postgres"))
