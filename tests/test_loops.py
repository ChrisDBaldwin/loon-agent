"""Tests for processing loops — parsing, iteration protocol, and run persistence."""

from __future__ import annotations

import pytest

from fakes import FakeChat
from loon_agent.graph import LoonAgent
from loon_agent.loops import (
    CONTINUE_MARKER,
    DONE_MARKER,
    LoopParseError,
    LoopSpec,
    LoopStore,
    discover_loops,
    is_done,
    iteration_prompt,
    iteration_session_key,
    parse_loop,
    run_iteration,
)

VALID = """---
name: audit
description: look at things
interval: 300
max_iterations: 5
---
Audit iteration {iteration} of {max_iterations}: look at one thing.
"""


def _spec(**overrides) -> LoopSpec:
    defaults = dict(
        name="audit", description="", interval=300.0, max_iterations=5, prompt="look at one thing"
    )
    return LoopSpec(**{**defaults, **overrides})


# --- parsing -----------------------------------------------------------------------


def test_parse_valid_loop() -> None:
    spec = parse_loop(VALID)
    assert spec.name == "audit"
    assert spec.interval == 300.0
    assert spec.max_iterations == 5
    assert "one thing" in spec.prompt


def test_parse_rejects_missing_frontmatter() -> None:
    with pytest.raises(LoopParseError, match="frontmatter"):
        parse_loop("just a prompt")


def test_parse_rejects_missing_interval() -> None:
    with pytest.raises(LoopParseError, match="interval"):
        parse_loop("---\nname: x\n---\nprompt\n")


def test_parse_rejects_tight_interval() -> None:
    with pytest.raises(LoopParseError, match="interval"):
        parse_loop("---\nname: x\ninterval: 5\n---\nprompt\n")


def test_parse_rejects_bad_max_iterations() -> None:
    with pytest.raises(LoopParseError, match="max_iterations"):
        parse_loop("---\nname: x\ninterval: 300\nmax_iterations: 0\n---\nprompt\n")


def test_parse_rejects_empty_prompt() -> None:
    with pytest.raises(LoopParseError, match="prompt"):
        parse_loop("---\nname: x\ninterval: 300\n---\n   \n")


def test_discover_loops_reads_directory(tmp_path) -> None:
    (tmp_path / "audit.md").write_text(VALID, encoding="utf-8")
    loops = discover_loops(tmp_path)
    assert set(loops) == {"audit"}
    assert discover_loops(tmp_path / "missing") == {}


def test_shipped_self_audit_loop_parses() -> None:
    loops = discover_loops("loops")
    assert "self-audit" in loops
    assert loops["self-audit"].interval >= 60


# --- iteration protocol --------------------------------------------------------------


def test_iteration_prompt_substitutes_and_appends_protocol() -> None:
    spec = _spec(prompt="step {iteration} of {max_iterations}")
    prompt = iteration_prompt(spec, 3)
    assert "step 3 of 5" in prompt
    assert DONE_MARKER in prompt and CONTINUE_MARKER in prompt


def test_is_done_checks_only_the_tail() -> None:
    assert is_done(f"all covered.\n{DONE_MARKER}")
    early_marker = f"the protocol says {DONE_MARKER} ends the loop." + "\nmore\n" * 5
    assert not is_done(early_marker + CONTINUE_MARKER)
    assert not is_done("no marker at all")
    assert not is_done("")


def test_run_iteration_fresh_thread_per_iteration() -> None:
    assert iteration_session_key("audit", 2) == "loop:audit:i2"
    llm = FakeChat(replies=[f"did a thing\n{CONTINUE_MARKER}"], calls=[])
    agent = LoonAgent(llm, tools=[])
    result = run_iteration(agent, _spec(), 1)
    assert not result.done
    assert "did a thing" in result.reply

    llm = FakeChat(replies=[f"everything covered\n{DONE_MARKER}"], calls=[])
    assert run_iteration(LoonAgent(llm, tools=[]), _spec(), 5).done


# --- run persistence -----------------------------------------------------------------


def test_loop_store_round_trip(tmp_path) -> None:
    store = LoopStore(tmp_path / "loops.sqlite")
    assert store.get("audit") is None
    assert store.running() == []

    store.activate("audit", "555")
    run = store.get("audit")
    assert run is not None and run.status == "running" and run.chat_id == "555"

    store.record_iteration("audit", 3)
    assert store.get("audit").iteration == 3
    assert [r.name for r in store.running()] == ["audit"]

    store.finish("audit", "done")
    assert store.get("audit").status == "done"
    assert store.running() == []


def test_loop_store_restart_resets_iteration(tmp_path) -> None:
    store = LoopStore(tmp_path / "loops.sqlite")
    store.activate("audit", "555")
    store.record_iteration("audit", 4)
    store.finish("audit", "stopped")

    store.activate("audit", "777")  # started again, from another chat
    run = store.get("audit")
    assert run.iteration == 0 and run.status == "running" and run.chat_id == "777"
