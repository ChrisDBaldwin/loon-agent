"""Tests for gen_ai telemetry: engine span tree + setup gating.

Uses an InMemorySpanExporter on the global tracer provider (the engine's module-level
tracer is a proxy that resolves lazily, so setting the provider here is enough).
"""

from __future__ import annotations

import pytest
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from fakes import FakeChat
from loon_agent.config import Settings
from loon_agent.skills import parse_skill
from loon_agent.skills.engine import SkillRunner
from loon_agent.telemetry import setup_telemetry
from test_skill_engine import RESEARCH_MINI

_EXPORTER = InMemorySpanExporter()


@pytest.fixture(scope="module", autouse=True)
def _tracing():
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(_EXPORTER))
    trace.set_tracer_provider(provider)
    yield


@pytest.fixture(autouse=True)
def _reset_spans():
    _EXPORTER.clear()
    yield


def _spans_by_name() -> dict[str, object]:
    return {span.name: span for span in _EXPORTER.get_finished_spans()}


def test_skill_run_emits_gen_ai_agent_span_tree() -> None:
    runner = SkillRunner(
        FakeChat(replies=["q1\nq2", "done"], calls=[]), {"fake_search": lambda q: [q]}
    )
    runner.run(parse_skill(RESEARCH_MINI), {"topic": "loons"})

    spans = _spans_by_name()
    agent = spans["invoke_agent mini"]
    assert agent.attributes["gen_ai.operation.name"] == "invoke_agent"
    assert agent.attributes["gen_ai.agent.name"] == "mini"
    assert agent.attributes["loon.skill.failures"] == 0

    # One step span per pipeline step, parented under the agent span.
    for name in ("step plan", "step search", "step synthesize"):
        assert spans[name].parent.span_id == agent.context.span_id
    assert spans["step plan"].attributes["loon.step.masque"] == "analyst"
    assert spans["step search"].attributes["loon.step.items_total"] == 2
    assert spans["step search"].attributes["loon.step.items_failed"] == 0

    # Each tool execution is an execute_tool span under its step span.
    tool_spans = [s for s in _EXPORTER.get_finished_spans() if s.name == "execute_tool fake_search"]
    assert len(tool_spans) == 2
    assert all(s.attributes["gen_ai.tool.name"] == "fake_search" for s in tool_spans)
    assert all(s.parent.span_id == spans["step search"].context.span_id for s in tool_spans)


def test_failed_items_and_failed_runs_are_visible_on_spans() -> None:
    def flaky(query: str) -> list[str]:
        if query == "bad":
            raise RuntimeError("kaboom")
        return [query]

    runner = SkillRunner(FakeChat(replies=["good\nbad", "done"], calls=[]), {"fake_search": flaky})
    runner.run(parse_skill(RESEARCH_MINI), {"topic": "x"})

    spans = _spans_by_name()
    assert spans["invoke_agent mini"].attributes["loon.skill.failures"] == 1
    assert spans["step search"].attributes["loon.step.items_failed"] == 1
    failed_tool = [
        s
        for s in _EXPORTER.get_finished_spans()
        if s.name == "execute_tool fake_search" and not s.status.is_ok
    ]
    assert len(failed_tool) == 1


def test_aborted_run_marks_agent_span_error() -> None:
    runner = SkillRunner(FakeChat(replies=["q1"], calls=[]), {})  # unknown tool -> abort
    with pytest.raises(Exception, match="unknown tool"):
        runner.run(parse_skill(RESEARCH_MINI), {"topic": "x"})

    agent = _spans_by_name()["invoke_agent mini"]
    assert not agent.status.is_ok
    assert "unknown tool" in agent.status.description


def test_setup_telemetry_off_is_noop_and_bad_mode_raises() -> None:
    setup_telemetry(Settings(otel="off"))  # must not touch global providers
    with pytest.raises(ValueError, match="LOON_OTEL"):
        setup_telemetry(Settings(otel="banana"))
