"""Tests for masque loading and the chat-agent persona hook."""

from __future__ import annotations

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver

from fakes import FakeChat
from loon_agent.graph import LoonAgent
from loon_agent.masques import Masque, MasqueLoader
from loon_agent.tools import DEFAULT_TOOLS


def test_loads_loon_convention_yaml(tmp_path) -> None:
    (tmp_path / "analyst.yaml").write_text(
        "name: Analyst\nlens: |\n  Be terse.\ncontext: |\n  Helping the operator.\n",
        encoding="utf-8",
    )
    loader = MasqueLoader([tmp_path])

    masque = loader.load("analyst")
    assert masque == Masque(name="Analyst", lens="Be terse.\n", context="Helping the operator.\n")
    assert loader.block("analyst") == "Be terse.\n\nHelping the operator."


def test_loads_masques_repo_convention_and_earlier_dir_wins(tmp_path) -> None:
    local, catalog = tmp_path / "local", tmp_path / "catalog"
    local.mkdir()
    catalog.mkdir()
    (catalog / "reviewer.masque.yaml").write_text(
        "name: Reviewer\nlens: catalog lens\nattributes: {domain: review}\n", encoding="utf-8"
    )
    (local / "reviewer.yaml").write_text("name: Reviewer\nlens: local lens\n", encoding="utf-8")

    assert MasqueLoader([local, catalog]).block("reviewer") == "local lens"
    assert MasqueLoader([catalog]).block("reviewer") == "catalog lens"


def test_missing_or_invalid_masque_is_none_not_crash(tmp_path) -> None:
    (tmp_path / "broken.yaml").write_text("lens: [not: valid\n", encoding="utf-8")
    (tmp_path / "lensless.yaml").write_text("name: NoLens\n", encoding="utf-8")
    loader = MasqueLoader([tmp_path])

    assert loader.load("ghost") is None
    assert loader.block("broken") is None
    assert loader.block("lensless") is None


def test_real_repo_masques_load() -> None:
    loader = MasqueLoader(["masques"])
    assert "research analyst" in loader.block("analyst").lower()
    assert "TL;DR" in loader.block("briefer")


def test_agent_persona_lands_in_system_prompt() -> None:
    llm = FakeChat(replies=["hi"], calls=[])
    agent = LoonAgent(
        llm, DEFAULT_TOOLS, checkpointer=MemorySaver(), persona="I WEAR THE ANALYST MASQUE"
    )
    agent.invoke("hello", session_key="cli:masque-test")

    system = llm.calls[0][0]
    assert system.type == "system"
    assert "I WEAR THE ANALYST MASQUE" in system.content


def test_no_persona_leaves_system_prompt_clean() -> None:
    llm = FakeChat(replies=[AIMessage(content="hi").content], calls=[])
    agent = LoonAgent(llm, DEFAULT_TOOLS, checkpointer=MemorySaver())
    agent.invoke("hello", session_key="cli:no-masque")

    assert "MASQUE" not in llm.calls[0][0].content
