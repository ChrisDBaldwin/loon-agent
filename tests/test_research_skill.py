"""Tests for the research skill: the real skill file, run end-to-end against fakes."""

from __future__ import annotations

from pathlib import Path

import pytest

from fakes import FakeChat
from loon_agent.adapters.cli import parse_skill_command
from loon_agent.app import _fetch_or_raise, _make_publish
from loon_agent.config import Settings
from loon_agent.masques import MasqueLoader
from loon_agent.skills import load_skill
from loon_agent.skills.engine import SkillRunner
from loon_agent.tools.web import FetchedPage, SearchResult

RESEARCH = Path("skills/research.md")


class FakeMemory:
    def __init__(self) -> None:
        self.turns: list[tuple[str, str, str]] = []

    def system_prompt_block(self) -> str:
        return ""

    def prefetch(self, query: str, session_id: str) -> str:
        return ""

    def sync_turn(self, user: str, assistant: str, session_id: str) -> None:
        self.turns.append((user, assistant, session_id))


def test_research_skill_file_parses_with_expected_pipeline() -> None:
    skill = load_skill(RESEARCH)

    assert skill.args == ("topic",)
    assert [s.name for s in skill.steps] == [
        "plan", "search", "select", "fetch", "summarize", "synthesize", "publish",
    ]
    # Every analyst/briefer masque the skill declares actually exists in masques/.
    loader = MasqueLoader(["masques"])
    declared = {s.masque for s in skill.steps if s.masque}
    assert declared == {"analyst", "briefer"}
    assert all(loader.block(m) for m in declared)
    # Templates only reference variables the pipeline provides.
    assert "{max_sources}" in skill.templates["select"]


def test_research_pipeline_end_to_end_with_fakes(tmp_path) -> None:
    settings = Settings(data_dir=tmp_path)
    memory = FakeMemory()

    def fake_search(query: object) -> list[SearchResult]:
        return [
            SearchResult(
                title=f"About {query}", url=f"https://example.com/{query}", snippet="…"
            )
        ]

    def fake_fetch(url: object) -> FetchedPage:
        if "q2" in str(url):
            raise RuntimeError("paywalled")  # one source dies; the run must survive
        return FetchedPage(url=str(url), title="A Page", text="loons dive deep")

    llm = FakeChat(
        replies=[
            "q1\nq2",  # plan -> two queries
            "https://example.com/q1\nhttps://example.com/q2",  # select -> two urls
            "URL: https://example.com/q1\n- loons dive to 60m\nRELIABILITY: field guide",
            "## TL;DR\nLoons dive deep [1].\n\n## Detail\nUp to 60 m [1].",  # synthesize
        ],
        calls=[],
    )
    runner = SkillRunner(
        llm,
        {
            "web_search": fake_search,
            "fetch_page": fake_fetch,
            "publish_report": _make_publish(memory, settings, model_label="fake-model"),
        },
        masque_loader=MasqueLoader(["masques"]).block,
    )

    result = runner.run(load_skill(RESEARCH), {"topic": "loon diving", "max_sources": 2})

    # The failed fetch was skipped, recorded, and the run still published.
    assert len(result.failures) == 1 and "paywalled" in result.failures[0]
    report_path = Path(str(result.outputs["report_path"]))
    assert report_path.exists()
    html = report_path.read_text(encoding="utf-8")
    assert "Loons dive deep" in html
    assert "paywalled" in html  # skipped source surfaces in the report
    assert "https://example.com/q1" in html

    # Memory write-back is FTS-recallable later: topic + report path recorded.
    (user, assistant, session_id), = memory.turns
    assert user == "research: loon diving"
    assert str(report_path) in assistant
    assert session_id == "skill:research"

    # Masques were donned: first call (plan) ran under the analyst lens,
    # the synthesize call under the briefer lens.
    assert "research analyst" in llm.calls[0][0].content.lower()
    assert "TL;DR" in llm.calls[-1][0].content


def test_cited_pages_follow_note_order_and_drop_unsummarized() -> None:
    from loon_agent.app import _cited_pages

    a = FetchedPage(url="https://a.com/x", title="A", text="…")
    b = FetchedPage(url="https://b.com/y", title="B", text="…")
    c = FetchedPage(url="https://c.com/z", title="C", text="…")
    notes = [
        "URL: https://c.com/z\n- fact",  # cited [1]
        "URL: https://a.com/x\n- fact",  # cited [2]
        # b produced no note (summarize skipped it) -> not listed as a source
    ]

    assert _cited_pages([a, b, c], notes) == [c, a]
    # Model mangled every URL -> fall back to fetched order, never an empty list.
    assert _cited_pages([a, b], ["no urls here"]) == [a, b]


def test_fetch_or_raise_rejects_non_urls_and_failed_pages() -> None:
    with pytest.raises(ValueError, match="not a url"):
        _fetch_or_raise("Sure! Here are the URLs you asked for:")


def test_parse_skill_command_forms() -> None:
    assert parse_skill_command("/research loon migration") == ("research", "loon migration")
    assert parse_skill_command("/skill research loon migration") == (
        "research",
        "loon migration",
    )
    assert parse_skill_command("/skill") == ("", "")
    assert parse_skill_command("plain chat message") is None
