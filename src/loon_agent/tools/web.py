"""Web research primitives: search (ddgs) and fetch/extract (httpx + trafilatura).

Plain functions (not LangChain ``@tool``s) — they are called deterministically by the
skill engine's tool registry. Both degrade instead of raising where the pipeline can
survive it: a failed search retries with backoff then returns ``[]``; a failed fetch
returns a :class:`FetchedPage` carrying its error so the run can skip it and report.
"""

from __future__ import annotations

import html as _html
import logging
import re
import time
from dataclasses import dataclass

import httpx
import trafilatura
from ddgs import DDGS

logger = logging.getLogger(__name__)

_UA = "Mozilla/5.0 (compatible; loon-agent homelab research bot)"
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)

DEFAULT_MAX_RESULTS = 5
DEFAULT_PAGE_CHARS = 12_000
DEFAULT_TIMEOUT = 30.0


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str

    def __str__(self) -> str:
        return f"{self.title}\n  {self.url}\n  {self.snippet}"


@dataclass(frozen=True)
class FetchedPage:
    url: str
    title: str
    text: str
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def __str__(self) -> str:
        if not self.ok:
            return f"[failed to fetch {self.url}: {self.error}]"
        return f"SOURCE: {self.title or self.url}\nURL: {self.url}\n{self.text}"


def web_search(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    *,
    retries: int = 2,
    backoff: float = 2.0,
) -> list[SearchResult]:
    """Search the web; returns [] if every attempt fails (caller decides whether to abort)."""
    for attempt in range(retries + 1):
        try:
            rows = DDGS().text(query, max_results=max_results) or []
            return [
                SearchResult(
                    title=(row.get("title") or "").strip(),
                    url=(row.get("href") or row.get("url") or "").strip(),
                    snippet=(row.get("body") or row.get("snippet") or "").strip(),
                )
                for row in rows
                if row.get("href") or row.get("url")
            ]
        except Exception as exc:  # noqa: BLE001 - ddgs raises assorted network errors
            logger.warning("web_search %r attempt %d failed: %s", query, attempt + 1, exc)
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
    return []


def fetch_page(
    url: str,
    *,
    max_chars: int = DEFAULT_PAGE_CHARS,
    timeout: float = DEFAULT_TIMEOUT,
) -> FetchedPage:
    """Fetch a URL and extract readable text; failures come back as a value, not a raise."""
    try:
        response = httpx.get(
            url,
            headers={"User-Agent": _UA},
            timeout=timeout,
            follow_redirects=True,
        )
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001 - report any transport/HTTP failure per-source
        return FetchedPage(url=url, title="", text="", error=str(exc))

    content_type = response.headers.get("content-type", "")
    if content_type and "html" not in content_type and "text" not in content_type:
        return FetchedPage(url=url, title="", text="", error=f"unsupported type: {content_type}")

    html = response.text
    text = trafilatura.extract(html, include_comments=False) or ""
    if not text.strip():
        return FetchedPage(url=url, title="", text="", error="no extractable text")

    match = _TITLE_RE.search(html)
    title = _html.unescape(re.sub(r"\s+", " ", match.group(1)).strip()) if match else ""
    return FetchedPage(url=url, title=title, text=text.strip()[:max_chars])
