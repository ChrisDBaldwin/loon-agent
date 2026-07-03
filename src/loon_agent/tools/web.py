"""Web research primitives: search (esper-search/SearXNG) and fetch/extract (httpx +
trafilatura).

Plain functions (not LangChain ``@tool``s) — they are called deterministically by the
skill engine's tool registry. Both degrade instead of raising where the pipeline can
survive it: a failed search retries with backoff then returns ``[]``; a failed fetch
returns a :class:`FetchedPage` carrying its error so the run can skip it and report.

Search shells out to the ``esper-search`` CLI (a stdlib-only client for the private
SearXNG instance on the Esper network — see ``docs/esper-search.md``) rather than
calling SearXNG directly, so every agent on the network shares its Redis-backed query
cache and rate limit instead of each reimplementing them.
"""

from __future__ import annotations

import html as _html
import json
import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass

import httpx
import trafilatura

logger = logging.getLogger(__name__)

_UA = "Mozilla/5.0 (compatible; loon-agent homelab research bot)"
_TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)

DEFAULT_MAX_RESULTS = 5
DEFAULT_PAGE_CHARS = 12_000
DEFAULT_TIMEOUT = 30.0
ESPER_SEARCH_BIN = os.environ.get("LOON_ESPER_SEARCH_BIN", "esper-search")


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
    timeout: float = 45.0,
) -> list[SearchResult]:
    """Search via the esper-search/SearXNG CLI; returns [] if every attempt fails.

    esper-search itself already waits out a full rate-limit window (exit 2 if the
    wait cap is hit) and reports upstream failures as exit 3 — retries here are for
    the CLI being transiently missing/unreachable, not for re-fighting the rate limit.
    """
    if shutil.which(ESPER_SEARCH_BIN) is None:
        logger.warning(
            "web_search: %r not found on PATH; see docs/esper-search.md to install it",
            ESPER_SEARCH_BIN,
        )
        return []

    for attempt in range(retries + 1):
        try:
            proc = subprocess.run(
                [ESPER_SEARCH_BIN, query],
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            logger.warning("web_search %r attempt %d timed out", query, attempt + 1)
            if attempt < retries:
                time.sleep(backoff * (attempt + 1))
            continue

        if proc.returncode != 0:
            logger.warning(
                "web_search %r attempt %d failed (exit %d): %s",
                query, attempt + 1, proc.returncode, proc.stderr.strip(),
            )
            # Exit 2 = rate limited (already waited out the cap) — retrying won't help.
            if proc.returncode == 2 or attempt >= retries:
                return []
            time.sleep(backoff * (attempt + 1))
            continue

        try:
            payload = json.loads(proc.stdout)
        except json.JSONDecodeError as exc:
            logger.warning("web_search %r: bad JSON from esper-search: %s", query, exc)
            return []

        rows = payload.get("results") or []
        return [
            SearchResult(
                title=(row.get("title") or "").strip(),
                url=(row.get("url") or "").strip(),
                snippet=(row.get("content") or "").strip(),
            )
            for row in rows[:max_results]
            if row.get("url")
        ]
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
