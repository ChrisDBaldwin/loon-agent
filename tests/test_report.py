"""Tests for the HTML report renderer."""

from __future__ import annotations

from loon_agent.report import render_report, write_report
from loon_agent.tools.web import FetchedPage

_PAGES = [
    FetchedPage(url="https://example.com/a", title="Loons & You", text="..."),
    FetchedPage(url="https://example.com/b", title="", text="..."),
]


def test_markdown_briefing_becomes_html_body() -> None:
    out = render_report(
        topic="Common loons",
        briefing_md="## TL;DR\nLoons **dive**.\n\n- fact one [1]\n- fact two [2]",
        pages=_PAGES,
        model="gemma-4-26b",
        backend="localbox",
    )
    assert "<h2>TL;DR</h2>" in out
    assert "<strong>dive</strong>" in out
    assert "gemma-4-26b" in out and "localbox" in out


def test_sources_are_numbered_links_with_url_fallback() -> None:
    out = render_report(topic="t", briefing_md="x", pages=_PAGES)
    assert '<a href="https://example.com/a">Loons &amp; You</a>' in out
    # Untitled source falls back to its URL as link text.
    assert '<a href="https://example.com/b">https://example.com/b</a>' in out


def test_raw_html_from_model_or_web_is_escaped() -> None:
    out = render_report(
        topic="<script>alert('t')</script>",
        briefing_md="hi <script>alert('x')</script>\n\n<img src=x onerror=steal()>",
        pages=[],
        failures=["<b>sneaky</b> failure"],
    )
    assert "<script>alert" not in out
    assert "<img" not in out
    assert "<b>sneaky</b>" not in out


def test_failures_render_as_skipped_section_only_when_present() -> None:
    with_failures = render_report(
        topic="t", briefing_md="x", pages=_PAGES, failures=["fetch of z timed out"]
    )
    assert "Skipped during the run" in with_failures
    assert "fetch of z timed out" in with_failures

    without = render_report(topic="t", briefing_md="x", pages=_PAGES)
    assert "Skipped" not in without


def test_write_report_slugs_dates_and_dedupes(tmp_path) -> None:
    first = write_report("<html>1</html>", "Qwen 3 vs. Gemma: местный?!", tmp_path)
    second = write_report("<html>2</html>", "Qwen 3 vs. Gemma: местный?!", tmp_path)

    assert first.parent == tmp_path
    assert first.name.startswith("qwen-3-vs-gemma-")
    assert first.suffix == ".html"
    assert second != first  # same topic same day -> counter suffix
    assert first.read_text(encoding="utf-8") == "<html>1</html>"
