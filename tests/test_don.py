"""Runtime don/doff: the bind tier, the session mirror, and command parsing."""

from __future__ import annotations

import yaml

from loon_agent.app import build_runtime, parse_don_command
from loon_agent.config import Settings
from loon_agent.memory import ScopedMemory, SqliteMemoryProvider

SIDECAR = """\
persona: Analyst
schema_version: 1
mcp:
  - server: host
    allow: [get_current_time]
memory:
  namespace: research
  mode: read-only
credentials:
  - alias: search_api
    ref: env://LOON_TEST_SECRET
    scope: {audience: [host], required: false}
"""

ANALYST = """\
name: Analyst
version: "0.1.0"
lens: |
  Be terse and factual.
"""


def _runtime(tmp_path, monkeypatch, *, sidecar: str | None = SIDECAR):
    """A real runtime pointed at a tmp masques/ dir (model never invoked)."""
    monkeypatch.setenv("LOON_LOCAL_MODEL", "test-model")
    monkeypatch.chdir(tmp_path)  # masques/ and skills/ resolve relative to cwd
    masques = tmp_path / "masques"
    masques.mkdir()
    (masques / "analyst.yaml").write_text(ANALYST, encoding="utf-8")
    if sidecar is not None:
        (masques / "analyst.persona.yaml").write_text(sidecar, encoding="utf-8")
    settings = Settings(
        data_dir=tmp_path / "data",
        backend="local",
        otel="off",
        memory_backend="sqlite",
        masque=None,
        masques_dir=None,
    )
    return build_runtime(settings)


def _tool_names(runtime) -> list[str]:
    return [tool.name for tool in runtime.agent.tools]


# --- /don parsing ---------------------------------------------------------------


def test_parse_don_command() -> None:
    assert parse_don_command("/don analyst") == ("analyst", None)
    assert parse_don_command("/don analyst dig into the logs") == (
        "analyst",
        "dig into the logs",
    )
    assert parse_don_command("/don") == ("", None)
    assert parse_don_command("/doff") is None
    assert parse_don_command("hello /don") is None
    assert parse_don_command("/donut") is None


# --- the bind tier ----------------------------------------------------------------


def test_don_binds_only_allowed_tools_and_doff_restores(tmp_path, monkeypatch) -> None:
    runtime = _runtime(tmp_path, monkeypatch)
    baseline = _tool_names(runtime)
    assert "calculator" in baseline

    persona = runtime.don("analyst")

    assert persona is not None and persona.name == "Analyst"
    # Denied tools are structurally absent — never handed to bind_tools/ToolNode.
    assert _tool_names(runtime) == ["get_current_time"]
    assert runtime.agent.persona is not None
    assert "<masque-active" in runtime.agent.persona

    runtime.doff()

    assert _tool_names(runtime) == baseline
    assert runtime.agent.persona is None
    assert runtime.active_persona is None


def test_don_scopes_memory_read_only(tmp_path, monkeypatch) -> None:
    runtime = _runtime(tmp_path, monkeypatch)

    runtime.don("analyst")

    memory = runtime.agent.memory
    assert isinstance(memory, ScopedMemory)
    assert memory.binding.namespace == "research"
    assert memory.binding.mode == "read-only"

    runtime.doff()
    assert isinstance(runtime.agent.memory, SqliteMemoryProvider)


def test_don_without_sidecar_keeps_all_tools(tmp_path, monkeypatch) -> None:
    runtime = _runtime(tmp_path, monkeypatch, sidecar=None)

    baseline = _tool_names(runtime)
    persona = runtime.don("analyst")

    assert persona is not None
    assert _tool_names(runtime) == baseline
    assert not isinstance(runtime.agent.memory, ScopedMemory)


def test_don_unknown_masque_is_lenient_no_op(tmp_path, monkeypatch) -> None:
    runtime = _runtime(tmp_path, monkeypatch)
    before = runtime.agent

    assert runtime.don("ghost") is None
    assert runtime.agent is before  # untouched, still baseline


# --- the session mirror -------------------------------------------------------------


def _read_mirror(tmp_path) -> dict:
    return yaml.safe_load((tmp_path / "data" / "masque.session.yaml").read_text())


def test_mirror_written_at_bind_tier_with_no_material(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("LOON_TEST_SECRET", "hunter2-material-9917")
    runtime = _runtime(tmp_path, monkeypatch)

    runtime.don("analyst", "audit the logs")
    # Force a resolution so material is cached in-process before we check the mirror.
    assert runtime.credentials.material("search_api") == "hunter2-material-9917"

    mirror = _read_mirror(tmp_path)
    active = mirror["active"]
    assert active["name"] == "Analyst"
    assert active["version"] == "0.1.0"
    assert active["intent"] == "audit the logs"
    assert active["config_source"] == "local"
    assert active["bound_refs"] == [
        {"alias": "search_api", "scheme": "env://", "status": "pending"}
    ]
    plan = active["capability_plan"]
    assert plan["host_apply"] == "bind"
    (binding,) = plan["bindings"]
    assert binding["enforced"] is True
    assert binding["tier"] == "bind"
    assert binding["effective_allow"] == ["get_current_time"]
    assert plan["memory"] == {"namespace": "research", "mode": "read-only", "status": "applied"}

    raw = (tmp_path / "data" / "masque.session.yaml").read_text()
    assert "hunter2-material-9917" not in raw  # never material, only alias+scheme+status

    runtime.doff()

    mirror = _read_mirror(tmp_path)
    assert mirror["active"] is None
    assert mirror["previous"]["name"] == "Analyst"
    assert "doffed_at" in mirror["previous"]
    raw = (tmp_path / "data" / "masque.session.yaml").read_text()
    assert "hunter2-material-9917" not in raw


def test_don_over_don_rolls_previous(tmp_path, monkeypatch) -> None:
    runtime = _runtime(tmp_path, monkeypatch)
    baseline = _tool_names(runtime)
    (tmp_path / "masques" / "briefer.yaml").write_text(
        'name: Briefer\nversion: "0.2.0"\nlens: |\n  Write the TL;DR.\n', encoding="utf-8"
    )

    runtime.don("analyst")
    runtime.don("briefer")

    mirror = _read_mirror(tmp_path)
    assert mirror["active"]["name"] == "Briefer"
    assert mirror["previous"]["name"] == "Analyst"
    assert "doffed_at" in mirror["previous"]
    # Swap doffs the outgoing persona first: briefer has no sidecar => all tools.
    assert _tool_names(runtime) == baseline
