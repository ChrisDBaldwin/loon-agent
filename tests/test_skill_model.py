"""Tests for the skill markdown parser."""

from __future__ import annotations

import pytest

from loon_agent.skills import SkillParseError, discover_skills, parse_skill

GOOD = """\
---
name: research
description: Deep-dive a topic.
args: [topic]
steps:
  - {name: plan, kind: llm, masque: analyst, output: queries, parse: lines}
  - {name: search, kind: tool, tool: web_search, foreach: queries, output: results}
  - {name: synthesize, kind: llm, output: briefing}
---

## step: plan
Plan research on {topic}. One query per line.

## step: synthesize
Combine {results} into a briefing about {topic}.
"""


def test_parses_steps_templates_and_metadata() -> None:
    skill = parse_skill(GOOD)

    assert skill.name == "research"
    assert skill.args == ("topic",)
    assert [s.name for s in skill.steps] == ["plan", "search", "synthesize"]

    plan, search, synthesize = skill.steps
    assert (plan.kind, plan.parse, plan.masque) == ("llm", "lines", "analyst")
    assert (search.kind, search.tool, search.foreach) == ("tool", "web_search", "queries")
    assert synthesize.parse == "text"  # default

    assert skill.templates["plan"].startswith("Plan research on {topic}")
    assert "briefing" in skill.templates["synthesize"]


@pytest.mark.parametrize(
    ("mutation", "complaint"),
    [
        (lambda t: t.replace("---\nname", "name"), "frontmatter"),
        (lambda t: t.replace("name: research\n", ""), "'name'"),
        (lambda t: t.replace("kind: llm, masque: analyst", "kind: wizard"), "kind"),
        (lambda t: t.replace(", parse: lines", ", parse: json"), "parse"),
        (lambda t: t.replace("tool: web_search, ", ""), "needs a string 'tool'"),
        (lambda t: t.replace("## step: plan", "## step: plotting"), "template"),
        (lambda t: t.replace("output: queries, ", ""), "'output'"),
    ],
)
def test_malformed_skills_fail_readably(mutation, complaint) -> None:
    with pytest.raises(SkillParseError, match=complaint):
        parse_skill(mutation(GOOD))


def test_duplicate_step_names_rejected() -> None:
    bad = GOOD.replace("name: search", "name: plan")
    with pytest.raises(SkillParseError, match="unique"):
        parse_skill(bad)


def test_orphan_template_rejected() -> None:
    bad = GOOD + "\n## step: ghost\nBoo.\n"
    with pytest.raises(SkillParseError, match="ghost"):
        parse_skill(bad)


def test_discover_skills_from_directory(tmp_path) -> None:
    (tmp_path / "research.md").write_text(GOOD, encoding="utf-8")

    skills = discover_skills(tmp_path)

    assert list(skills) == ["research"]
    assert skills["research"].path == tmp_path / "research.md"


def test_discover_skills_missing_dir_is_empty() -> None:
    assert discover_skills("/nonexistent/skills") == {}


def test_discover_skills_rejects_duplicate_names(tmp_path) -> None:
    (tmp_path / "a.md").write_text(GOOD, encoding="utf-8")
    (tmp_path / "b.md").write_text(GOOD, encoding="utf-8")

    with pytest.raises(SkillParseError, match="duplicate"):
        discover_skills(tmp_path)
