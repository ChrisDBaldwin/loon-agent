"""Tests for runtime assembly (build_runtime) — memory backend selection."""

from __future__ import annotations

import pytest

from loon_agent.app import build_runtime
from loon_agent.config import Settings
from loon_agent.memory import ChromaMemoryProvider, SqliteMemoryProvider


def _settings(tmp_path, *, memory_backend: str = "sqlite", exec_backend: str = "off", **kwargs):
    # otel/memory_backend/exec_backend pinned explicitly: the real project .env sets
    # LOON_OTEL=otlp and LOON_MEMORY_BACKEND=chroma, and pydantic-settings reads those
    # ambient values for any field not passed here — pin everything these tests assert on
    # so they don't depend on what happens to be in .env.
    return Settings(
        data_dir=tmp_path, backend="local", otel="off",
        memory_backend=memory_backend, exec_backend=exec_backend, **kwargs
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


def test_exec_off_means_no_exec_tools_in_registry(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOON_LOCAL_MODEL", "test-model")
    runtime = build_runtime(_settings(tmp_path))  # exec_backend defaults to "off"
    assert "run_command" not in runtime.runner.tools
    assert "web_search" in runtime.runner.tools  # research tools still present


def test_exec_docker_wires_exec_tools_into_skill_registry(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOON_LOCAL_MODEL", "test-model")
    runtime = build_runtime(
        _settings(tmp_path, exec_backend="docker", exec_image="loon-toolbox@sha256:abc")
    )
    for name in ("run_command", "write_file", "edit_file", "delete_file"):
        assert name in runtime.runner.tools
    # Exec tools must NOT leak into the always-on chat-loop tool set (DEFAULT_TOOLS).
    from loon_agent.tools import DEFAULT_TOOLS
    assert "run_command" not in {t.name for t in DEFAULT_TOOLS}


def test_exec_docker_without_image_errors_helpfully(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOON_LOCAL_MODEL", "test-model")
    with pytest.raises(ValueError, match="LOON_EXEC_IMAGE"):
        build_runtime(_settings(tmp_path, exec_backend="docker"))


def test_unknown_exec_backend_errors_helpfully(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOON_LOCAL_MODEL", "test-model")
    with pytest.raises(ValueError, match="LOON_EXEC_BACKEND"):
        build_runtime(_settings(tmp_path, exec_backend="firejail"))


def test_publish_page_registered_and_reports_go_to_web_root(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOON_LOCAL_MODEL", "test-model")
    web_root = tmp_path / "site"
    runtime = build_runtime(_settings(tmp_path, web_root=web_root))
    assert "publish_page" in runtime.runner.tools
    # publish_page/publish_report write into the web root, not a separate reports dir.
    from loon_agent.report import render_report, write_report
    write_report(render_report(topic="t", briefing_md="x", pages=[]), "t", web_root)
    assert list(web_root.glob("t-*.html"))


def test_site_tools_bound_into_chat_loop(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOON_LOCAL_MODEL", "test-model")
    web_root = tmp_path / "site"
    runtime = build_runtime(_settings(tmp_path, web_root=web_root))
    names = {t.name for t in runtime.agent.tools}
    assert {
        "list_site_pages",
        "read_site_page",
        "publish_site_page",
        "update_site_page",
        "delete_site_page",
    } <= names
    # And they operate on the configured web root.
    publish = next(t for t in runtime.agent.tools if t.name == "publish_site_page")
    publish.invoke({"title": "Wired", "markdown": "# Wired\n\nok."})
    assert list(web_root.glob("wired-*.html"))
