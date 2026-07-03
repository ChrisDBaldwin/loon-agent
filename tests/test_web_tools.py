"""Tests for web search/fetch — network fully mocked."""

from __future__ import annotations

import json
import subprocess
from unittest.mock import MagicMock, patch

import httpx

from loon_agent.tools.web import FetchedPage, SearchResult, fetch_page, web_search


def _cli_result(returncode: int = 0, stdout: str = "", stderr: str = "") -> MagicMock:
    result = MagicMock(spec=subprocess.CompletedProcess)
    result.returncode = returncode
    result.stdout = stdout
    result.stderr = stderr
    return result

_HTML = """
<html><head><title>  Loons &amp; Lakes  </title></head><body>
<article><p>The common loon is a large diving waterbird.</p>
<p>It winters on coasts and summers on northern lakes.</p></article>
</body></html>
"""


# --- web_search -----------------------------------------------------------------


def test_web_search_maps_searxng_rows() -> None:
    payload = json.dumps(
        {
            "results": [
                {"title": "Loon", "url": "https://example.com/loon", "content": "a bird"},
                {"title": "No url row is dropped", "content": "x"},
            ]
        }
    )
    with (
        patch("loon_agent.tools.web.shutil.which", return_value="/usr/local/bin/esper-search"),
        patch("loon_agent.tools.web.subprocess.run", return_value=_cli_result(stdout=payload)),
    ):
        results = web_search("loon", max_results=5)

    assert results == [SearchResult(title="Loon", url="https://example.com/loon", snippet="a bird")]


def test_web_search_missing_binary_returns_empty() -> None:
    with patch("loon_agent.tools.web.shutil.which", return_value=None):
        results = web_search("loon")

    assert results == []


def test_web_search_rate_limited_gives_up_without_retry() -> None:
    with (
        patch("loon_agent.tools.web.shutil.which", return_value="/usr/local/bin/esper-search"),
        patch(
            "loon_agent.tools.web.subprocess.run",
            return_value=_cli_result(returncode=2, stderr="rate limited"),
        ) as run,
        patch("loon_agent.tools.web.time.sleep") as nap,
    ):
        results = web_search("loon", retries=2)

    assert results == []
    assert run.call_count == 1  # exit 2 already waited out the window; no local retry
    assert nap.call_count == 0


def test_web_search_recovers_on_second_attempt() -> None:
    ok = json.dumps({"results": [{"title": "t", "url": "https://e.com", "content": "s"}]})
    with (
        patch("loon_agent.tools.web.shutil.which", return_value="/usr/local/bin/esper-search"),
        patch(
            "loon_agent.tools.web.subprocess.run",
            side_effect=[
                _cli_result(returncode=3, stderr="upstream error"),
                _cli_result(stdout=ok),
            ],
        ),
        patch("loon_agent.tools.web.time.sleep"),
    ):
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
