"""Site management tools: let the chat loop create, edit and remove pages on loon's
internal website.

The website (``adapters/web.py``) is a read-only static server over ``LOON_WEB_ROOT`` —
managing it is filesystem work, not HTTP. These tools operate directly on the web root,
so they *are* allowed in the chat loop where exec/file tools are not (see
``tools/builtins.py``): every write goes through ``report.render_page`` (model-written
markdown, escaped title, raw HTML inert) and every filename is validated to a single
``*.html`` basename inside the web root. The worst a prompt-injecting web page can do is
publish, rewrite or delete pages on an internal LAN gallery — no code execution, no
reads or writes outside the site directory.

Round-trip editing: ``publish_page`` (tools/publish.py) stores each page's markdown
source under ``.src/`` in the web root (dotfiles are hidden from the served gallery).
``read`` returns that source so the model can edit and ``update`` re-renders it; pages
published before source storage existed fall back to their raw HTML.
"""

from __future__ import annotations

import datetime as _dt
import re
import socket
from pathlib import Path

from langchain_core.tools import BaseTool, StructuredTool

from ..report import render_page
from .publish import publish_page, source_path_for

# One served page: a bare *.html basename — no separators, no leading dot, so a model-
# (or injection-)supplied name can never address anything but a page in the web root.
_PAGE_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*\.html$")

_READ_FALLBACK_CHARS = 12_000


def site_base_url(port: int) -> str:
    """The URL other LAN nodes reach the site at, mirroring adapters/web.py's label."""
    host = socket.gethostname().lower() or "localhost"
    return f"http://{host}:{port}"


def _page_path(web_root: Path, filename: str) -> Path | None:
    """Resolve ``filename`` to a page path inside the web root, or None if invalid."""
    if not _PAGE_NAME_RE.match(filename):
        return None
    path = (web_root / filename).resolve()
    if path.parent != Path(web_root).resolve():
        return None
    return path


def list_pages(web_root: Path) -> str:
    """One line per served page (newest first): filename, modified date, size."""
    root = Path(web_root)
    if not root.is_dir():
        return "no pages yet (site directory is empty)"
    pages = sorted(
        (p for p in root.iterdir() if p.is_file() and not p.name.startswith(".")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not pages:
        return "no pages yet (site directory is empty)"
    lines = []
    for p in pages:
        stat = p.stat()
        when = _dt.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        lines.append(f"{p.name}  ({when}, {stat.st_size}B)")
    return "\n".join(lines)


def read_page(web_root: Path, filename: str) -> str:
    """The page's markdown source, or its raw HTML for pages published without one."""
    path = _page_path(web_root, filename)
    if path is None:
        return f"error: invalid page name {filename!r} (expected a bare *.html filename)"
    if not path.is_file():
        return f"error: no such page {filename!r} — use list_site_pages to see what exists"
    source = source_path_for(path)
    if source.is_file():
        return source.read_text(encoding="utf-8")
    html_text = path.read_text(encoding="utf-8", errors="replace")
    return (
        "[no markdown source stored for this page; raw HTML follows]\n"
        + html_text[:_READ_FALLBACK_CHARS]
    )


def update_page(web_root: Path, filename: str, title: str, markdown: str) -> str:
    """Re-render an existing page in place (same filename, same URL)."""
    path = _page_path(web_root, filename)
    if path is None:
        return f"error: invalid page name {filename!r} (expected a bare *.html filename)"
    if not path.is_file():
        return f"error: no such page {filename!r} — use publish_site_page to create new pages"
    try:
        path.write_text(render_page(title, markdown), encoding="utf-8")
        source = source_path_for(path)
        source.parent.mkdir(parents=True, exist_ok=True)
        source.write_text(markdown, encoding="utf-8")
    except OSError as exc:
        return f"error: {exc}"
    return f"updated: {filename}"


def delete_page(web_root: Path, filename: str) -> str:
    """Remove a page (and its stored markdown source) from the site."""
    path = _page_path(web_root, filename)
    if path is None:
        return f"error: invalid page name {filename!r} (expected a bare *.html filename)"
    if not path.is_file():
        return f"error: no such page {filename!r} — use list_site_pages to see what exists"
    try:
        path.unlink()
        source_path_for(path).unlink(missing_ok=True)
    except OSError as exc:
        return f"error: {exc}"
    return f"deleted: {filename}"


def site_tools(web_root: Path, *, base_url: str) -> list[BaseTool]:
    """The five chat-loop site tools, bound to this deployment's web root and URL."""
    web_root = Path(web_root)
    site_note = (
        f"This is your own internal website at {base_url}/ — pages are files you manage "
        "directly with the site tools; no HTTP POST is involved."
    )

    def _list() -> str:
        return f"site root: {base_url}/\n{list_pages(web_root)}"

    def _read(filename: str) -> str:
        return read_page(web_root, filename)

    def _publish(title: str, markdown: str) -> str:
        result = publish_page(title, markdown, web_root=web_root)
        if not result.ok:
            return str(result)
        return f"published: {base_url}/{Path(result.path).name}"

    def _update(filename: str, title: str, markdown: str) -> str:
        result = update_page(web_root, filename, title, markdown)
        return result if result.startswith("error") else f"{result} — {base_url}/{filename}"

    def _delete(filename: str) -> str:
        return delete_page(web_root, filename)

    return [
        StructuredTool.from_function(
            _list,
            name="list_site_pages",
            description=f"List every page on your website, newest first. {site_note}",
        ),
        StructuredTool.from_function(
            _read,
            name="read_site_page",
            description=(
                "Read one website page's markdown source (by filename from "
                f"list_site_pages) so you can edit it. {site_note}"
            ),
        ),
        StructuredTool.from_function(
            _publish,
            name="publish_site_page",
            description=(
                "Publish a new page to your website: give a title and the full page "
                f"content as markdown; returns the page's URL. {site_note}"
            ),
        ),
        StructuredTool.from_function(
            _update,
            name="update_site_page",
            description=(
                "Rewrite an existing website page in place (same filename and URL): "
                f"give its filename, a title, and the full new markdown. {site_note}"
            ),
        ),
        StructuredTool.from_function(
            _delete,
            name="delete_site_page",
            description=f"Delete a page from your website by filename. {site_note}",
        ),
    ]
