"""Publish a page to loon's internal website.

``publish_page`` turns a model-written markdown page into a self-contained, XSS-safe HTML
file in the web root (served by ``adapters/web.py``). It reuses ``report.render_page`` so the
same escaping guarantees apply — the model writes markdown, never raw HTML.

A plain function for the skill registry, in the ``tools/web.py`` mould: returns a frozen
result carrying the served path (and any error) rather than raising.

Each page's markdown source is also stored under ``.src/`` in the web root (hidden from
the served gallery, which skips dotfiles) so the site tools (``tools/site.py``) can read
it back for round-trip editing.
"""

from __future__ import annotations

import datetime as _dt
import re
from dataclasses import dataclass
from pathlib import Path

from ..report import render_page

_SLUG_RE = re.compile(r"[^a-z0-9]+")

SOURCE_DIR = ".src"


def source_path_for(page_path: Path) -> Path:
    """Where a served page's markdown source lives: ``<web_root>/.src/<stem>.md``."""
    return page_path.parent / SOURCE_DIR / f"{page_path.stem}.md"


@dataclass(frozen=True)
class PublishResult:
    title: str
    path: str = ""
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def __str__(self) -> str:
        if not self.ok:
            return f"[publish {self.title!r} failed: {self.error}]"
        return f"published: {self.path}"


def _slug(title: str) -> str:
    return _SLUG_RE.sub("-", title.lower()).strip("-")[:60] or "page"


def publish_page(title: str, markdown: str, *, web_root: Path, subtitle: str = "") -> PublishResult:
    """Render ``markdown`` to HTML and write it into the web root as ``<slug>-<date>.html``."""
    try:
        html_text = render_page(title, markdown, subtitle=subtitle)
        root = Path(web_root)
        root.mkdir(parents=True, exist_ok=True)
        stem = f"{_slug(title)}-{_dt.date.today().isoformat()}"
        path = root / f"{stem}.html"
        counter = 2
        while path.exists():
            path = root / f"{stem}-{counter}.html"
            counter += 1
        path.write_text(html_text, encoding="utf-8")
        source = source_path_for(path)
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(markdown, encoding="utf-8")
    except OSError as exc:
        return PublishResult(title=title, error=str(exc))
    return PublishResult(title=title, path=str(path))
