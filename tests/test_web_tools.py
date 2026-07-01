"""Tests for web search/fetch — network fully mocked."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import httpx

from loon_agent.tools.web import FetchedPage, SearchResult, fetch_page, web_search

_HTML = """
<html><head><title>  Loons &amp; Lakes  </title></head><body>
<article><p>The common loon is a large diving waterbird.</p>
<p>It winters on coasts and summers on northern lakes.</p></article>
</body></html>
"""


# --- web_search -----------------------------------------------------------------


def test_web_search_maps_ddgs_rows() -> None:
    rows = [
        {"title": "Loon", "href": "https://example.com/loon", "body": "a bird"},
        {"title": "No url row is dropped", "body": "x"},
    ]
    with patch("loon_agent.tools.web.DDGS") as ddgs:
        ddgs.return_value.text.return_value = rows
        results = web_search("loon", max_results=5)

    assert results == [SearchResult(title="Loon", url="https://example.com/loon", snippet="a bird")]


def test_web_search_retries_then_gives_up_empty() -> None:
    with (
        patch("loon_agent.tools.web.DDGS") as ddgs,
        patch("loon_agent.tools.web.time.sleep") as nap,
    ):
        ddgs.return_value.text.side_effect = RuntimeError("rate limited")
        results = web_search("loon", retries=2)

    assert results == []
    assert ddgs.return_value.text.call_count == 3
    assert nap.call_count == 2  # backoff between attempts, none after the last


def test_web_search_recovers_on_second_attempt() -> None:
    ok = [{"title": "t", "href": "https://e.com", "body": "s"}]
    with patch("loon_agent.tools.web.DDGS") as ddgs, patch("loon_agent.tools.web.time.sleep"):
        ddgs.return_value.text.side_effect = [RuntimeError("flake"), ok]
        results = web_search("loon", retries=2)

    assert [r.url for r in results] == ["https://e.com"]


# --- fetch_page -----------------------------------------------------------------


def _response(text: str = _HTML, content_type: str = "text/html", status: int = 200) -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.text = text
    response.headers = {"content-type": content_type}
    if status >= 400:
        response.raise_for_status.side_effect = httpx.HTTPStatusError(
            f"{status}", request=MagicMock(), response=response
        )
    return response


def test_fetch_page_extracts_text_and_title() -> None:
    with patch("loon_agent.tools.web.httpx.get", return_value=_response()):
        page = fetch_page("https://example.com/loon")

    assert page.ok
    assert page.title == "Loons & Lakes"
    assert "diving waterbird" in page.text
    assert "<p>" not in page.text


def test_fetch_page_truncates_to_max_chars() -> None:
    big = _HTML.replace("northern lakes", "northern lakes " + "very " * 5000)
    with patch("loon_agent.tools.web.httpx.get", return_value=_response(big)):
        page = fetch_page("https://example.com", max_chars=500)

    assert page.ok
    assert len(page.text) <= 500


def test_fetch_page_transport_error_is_a_value() -> None:
    with patch("loon_agent.tools.web.httpx.get", side_effect=httpx.ConnectError("boom")):
        page = fetch_page("https://down.example.com")

    assert not page.ok
    assert "boom" in (page.error or "")
    assert "failed to fetch" in str(page)


def test_fetch_page_http_error_is_a_value() -> None:
    with patch("loon_agent.tools.web.httpx.get", return_value=_response(status=404)):
        page = fetch_page("https://example.com/missing")

    assert not page.ok


def test_fetch_page_rejects_non_text_content() -> None:
    with patch(
        "loon_agent.tools.web.httpx.get",
        return_value=_response(text="%PDF-1.7", content_type="application/pdf"),
    ):
        page = fetch_page("https://example.com/paper.pdf")

    assert not page.ok
    assert "unsupported type" in (page.error or "")


def test_fetch_page_empty_extraction_is_error() -> None:
    with patch(
        "loon_agent.tools.web.httpx.get",
        return_value=_response(text="<html><body><script>x()</script></body></html>"),
    ):
        page = fetch_page("https://example.com/empty")

    assert not page.ok
    assert page.error == "no extractable text"


def test_fetched_page_str_formats_source_block() -> None:
    page = FetchedPage(url="https://e.com", title="T", text="body")
    assert str(page).startswith("SOURCE: T\nURL: https://e.com\n")
