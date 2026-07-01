"""Tests for the chars/4 budgeting helpers."""

from __future__ import annotations

from loon_agent.textbudget import CHARS_PER_TOKEN, approx_tokens, truncate_to_tokens


def test_approx_tokens_rounds_up() -> None:
    assert approx_tokens("") == 0
    assert approx_tokens("a") == 1
    assert approx_tokens("a" * CHARS_PER_TOKEN) == 1
    assert approx_tokens("a" * (CHARS_PER_TOKEN + 1)) == 2


def test_short_text_is_untouched() -> None:
    assert truncate_to_tokens("hello world", 100) == "hello world"


def test_truncation_respects_budget_and_marks_cut() -> None:
    text = "word " * 1000
    out = truncate_to_tokens(text, 50)
    assert approx_tokens(out) <= 50
    assert out.endswith("…[truncated]")
    # Cut on a word boundary, not mid-word.
    assert "word …[truncated]" == out[-len("word …[truncated]") :]


def test_solid_run_gets_hard_cut() -> None:
    out = truncate_to_tokens("x" * 10_000, 25)
    assert approx_tokens(out) <= 25
    assert out.endswith("…[truncated]")


def test_zero_or_negative_budget_is_empty() -> None:
    assert truncate_to_tokens("anything", 0) == ""
    assert truncate_to_tokens("anything", -5) == ""
