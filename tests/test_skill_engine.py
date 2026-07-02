"""Tests for the deterministic skill engine — fake LLM, fake tools, no network."""

from __future__ import annotations

import pytest

from fakes import FakeChat
from loon_agent.skills import parse_skill
from loon_agent.skills.engine import RunResult, SkillRunError, SkillRunner, _parse_lines
from loon_agent.textbudget import CHARS_PER_TOKEN


def _fake(replies: list[str]) -> FakeChat:
    return FakeChat(replies=replies, calls=[])


RESEARCH_MINI = """\
---
name: mini
description: Minimal end-to-end pipeline.
args: [topic]
steps:
  - {name: plan, kind: llm, masque: analyst, output: queries, parse: lines}
  - {name: search, kind: tool, tool: fake_search, foreach: queries, output: results}
  - {name: synthesize, kind: llm, output: briefing}
---

## step: plan
Queries for {topic}, one per line.

## step: synthesize
Write a briefing on {topic} from:
{results}
"""


def test_pipeline_flows_context_through_steps() -> None:
    llm = _fake(["alpha\nbeta", "the briefing"])
    searched: list[str] = []

    def fake_search(query: str) -> list[str]:
        searched.append(query)
        return [f"hit({query})"]

    runner = SkillRunner(llm, {"fake_search": fake_search})
    result = runner.run(parse_skill(RESEARCH_MINI), {"topic": "loons"})

    assert searched == ["alpha", "beta"]
    assert result.outputs["queries"] == ["alpha", "beta"]
    assert result.outputs["results"] == ["hit(alpha)", "hit(beta)"]  # per-item lists flatten
    assert result.outputs["briefing"] == "the briefing"
    assert result.failures == []
    # The synthesize prompt actually contained the searched results.
    final_prompt = llm.calls[-1][-1].content
    assert "hit(alpha)" in final_prompt


def test_masque_becomes_system_message_only_where_declared() -> None:
    llm = _fake(["q1", "done"])
    runner = SkillRunner(
        llm,
        {"fake_search": lambda q: [q]},
        masque_loader=lambda name: f"LENS:{name}",
    )
    runner.run(parse_skill(RESEARCH_MINI), {"topic": "x"})

    plan_messages, synth_messages = llm.calls[0], llm.calls[1]
    assert plan_messages[0].content == "LENS:analyst"  # plan declares masque: analyst
    assert len(synth_messages) == 1  # synthesize declares none -> no system message


def test_think_blocks_are_stripped() -> None:
    llm = _fake(["<think>pondering deeply</think>q1", "<THINKING>hm</THINKING>final text"])
    runner = SkillRunner(llm, {"fake_search": lambda q: [q]})
    result = runner.run(parse_skill(RESEARCH_MINI), {"topic": "x"})

    assert result.outputs["queries"] == ["q1"]
    assert result.outputs["briefing"] == "final text"


def test_lines_parser_is_tolerant() -> None:
    text = 'Here are the queries:\n1. alpha\n- "beta"\n\n2) gamma\n• delta'
    assert _parse_lines(text, max_lines=8) == ["alpha", "beta", "gamma", "delta"]
    assert _parse_lines(text, max_lines=2) == ["alpha", "beta"]  # hard cap


def test_substituted_variables_respect_input_budget() -> None:
    llm = _fake(["q1", "done"])
    runner = SkillRunner(llm, {"fake_search": lambda q: [q]}, input_budget=100)
    runner.run(parse_skill(RESEARCH_MINI), {"topic": "loon " * 5_000})

    for calls in llm.calls:
        prompt = calls[-1].content
        # Budget is approximate (chars/4) — allow the marker's slack, nothing more.
        assert len(prompt) <= 100 * CHARS_PER_TOKEN + 20
    assert "…[truncated]" in llm.calls[0][-1].content


def test_foreach_item_failure_skips_and_records() -> None:
    def flaky(query: str) -> list[str]:
        if query == "bad":
            raise RuntimeError("kaboom")
        return [f"hit({query})"]

    llm = _fake(["good\nbad", "done"])
    runner = SkillRunner(llm, {"fake_search": flaky})
    result = runner.run(parse_skill(RESEARCH_MINI), {"topic": "x"})

    assert result.outputs["results"] == ["hit(good)"]
    assert len(result.failures) == 1
    assert "kaboom" in result.failures[0]


def test_all_items_failing_aborts() -> None:
    def always_fail(query: str) -> list[str]:
        raise RuntimeError("dead")

    llm = _fake(["a\nb"])
    runner = SkillRunner(llm, {"fake_search": always_fail})
    with pytest.raises(SkillRunError, match="every item failed"):
        runner.run(parse_skill(RESEARCH_MINI), {"topic": "x"})


def test_empty_llm_reply_retries_then_aborts() -> None:
    llm = _fake(["", "  ", ""])
    runner = SkillRunner(llm, {"fake_search": lambda q: [q]})
    with pytest.raises(SkillRunError, match="after 2 attempts"):
        runner.run(parse_skill(RESEARCH_MINI), {"topic": "x"})
    assert len(llm.calls) == 2


def test_missing_arg_unknown_tool_and_bad_foreach_fail_readably() -> None:
    skill = parse_skill(RESEARCH_MINI)

    with pytest.raises(SkillRunError, match="missing args"):
        SkillRunner(_fake(["x"]), {"fake_search": lambda q: [q]}).run(skill, {})

    with pytest.raises(SkillRunError, match="unknown tool"):
        SkillRunner(_fake(["q1"]), {}).run(skill, {"topic": "x"})

    unknown_var = RESEARCH_MINI.replace("{results}", "{nonexistent}")
    with pytest.raises(SkillRunError, match="nonexistent"):
        SkillRunner(_fake(["q1", "z"]), {"fake_search": lambda q: [q]}).run(
            parse_skill(unknown_var), {"topic": "x"}
        )


def test_non_foreach_tool_receives_full_context() -> None:
    skill_text = """\
---
name: publishing
description: Tool gets the whole context.
args: [topic]
steps:
  - {name: note, kind: llm, output: briefing}
  - {name: publish, kind: tool, tool: publish, output: path}
---

## step: note
Say something about {topic}.
"""
    seen: dict[str, object] = {}

    def publish(context: dict[str, object]) -> str:
        seen.update(context)
        return "/reports/out.html"

    runner = SkillRunner(_fake(["a note"]), {"publish": publish})
    result = runner.run(parse_skill(skill_text), {"topic": "loons"})

    assert seen["topic"] == "loons"
    assert seen["briefing"] == "a note"
    assert result.outputs["path"] == "/reports/out.html"


def test_progress_callback_narrates_steps() -> None:
    lines: list[str] = []
    runner = SkillRunner(
        _fake(["q", "done"]), {"fake_search": lambda q: [q]}, progress=lines.append
    )
    runner.run(parse_skill(RESEARCH_MINI), {"topic": "x"})
    assert lines == ["mini: plan…", "mini: search…", "mini: synthesize…"]


def test_run_result_type() -> None:
    runner = SkillRunner(_fake(["q", "done"]), {"fake_search": lambda q: [q]})
    result = runner.run(parse_skill(RESEARCH_MINI), {"topic": "x"})
    assert isinstance(result, RunResult)
