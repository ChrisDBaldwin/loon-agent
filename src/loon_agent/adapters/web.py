"""Web adapter: serve loon's published HTML artifacts over the LAN.

``python -m loon_agent web`` starts a small static file server rooted at ``LOON_WEB_ROOT``
(default ``.loon/site``), bound to ``LOON_WEB_HOST:LOON_WEB_PORT`` (default ``0.0.0.0:8800``)
so any node on the network can reach it — the way Grafana is reachable at ``ironwood:3000``.

Deliberately static and read-only: it is decoupled from the agent runtime (no model, no
skill engine) and serves only files loon has already written into the site directory. The
directory index is replaced with a styled gallery so the root is browsable rather than a
raw file listing. Zero third-party deps — stdlib ``http.server`` is enough for static
serving on a trusted LAN, matching the rest of loon's minimalist tooling.

Hardening for a LAN-exposed static server: GET/HEAD only, and a symlink-escape guard on top
of the base handler's path sanitization so nothing outside the site root is ever served.
"""

from __future__ import annotations

import datetime as _dt
import html
import logging
import socket
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from ..config import Settings, get_settings
from ..report import html_shell

logger = logging.getLogger(__name__)

_ALLOWED_METHODS = ("GET", "HEAD")


class SiteRequestHandler(SimpleHTTPRequestHandler):
    """Serves the site directory: a generated gallery at ``/``, files by name, nothing else."""

    def do_POST(self) -> None:  # noqa: N802 - http.server naming
        self._reject_method()

    # Every non-GET/HEAD verb SimpleHTTPRequestHandler doesn't implement returns 501 already,
    # but be explicit about the read-only contract for the ones a client might actually try.
    do_PUT = do_DELETE = do_PATCH = do_POST

    def _reject_method(self) -> None:
        self.send_error(HTTPStatus.METHOD_NOT_ALLOWED, "read-only site")

    def translate_path(self, path: str) -> str:
        """Resolve as the base handler does, then refuse anything that escapes the root
        (e.g. a symlink in the site dir pointing outside it)."""
        resolved = Path(super().translate_path(path)).resolve()
        root = Path(self.directory).resolve()
        if resolved != root and root not in resolved.parents:
            # Point back at the root so the base handler 404s inside bounds rather than serving.
            return str(root / "__forbidden__")
        return str(resolved)

    def list_directory(self, path: str):
        """Render the styled gallery instead of the stock plain directory listing."""
        directory = Path(path)
        try:
            entries = sorted(
                (p for p in directory.iterdir() if not p.name.startswith(".")),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            self.send_error(HTTPStatus.NOT_FOUND, "cannot list directory")
            return None

        body = _gallery_html(entries)
        encoded = html_shell("loon", body).encode("utf-8", "replace")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(encoded)
        return None

    def log_message(self, fmt: str, *args) -> None:
        logger.info("web %s - %s", self.address_string(), fmt % args)


def _gallery_html(entries: list[Path]) -> str:
    header = (
        "<header>\n  <h1>loon</h1>\n"
        "  <p>published artifacts · browse what loon has created</p>\n</header>\n"
    )
    if not entries:
        return header + '<p class="empty">Nothing published yet.</p>'

    items = []
    for p in entries:
        stat = p.stat()
        when = _dt.datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M")
        name = html.escape(p.name + ("/" if p.is_dir() else ""))
        meta = when if p.is_dir() else f"{when} · {_size(stat.st_size)}"
        href = html.escape(p.name + ("/" if p.is_dir() else ""), quote=True)
        items.append(
            f'  <li><a href="{href}"><span class="name">{name}</span>'
            f'<span class="meta">{meta}</span></a></li>'
        )
    return header + '<ul class="gallery">\n' + "\n".join(items) + "\n</ul>"


def _size(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n / (1024 * 1024):.1f}MB"


def run_web(settings: Settings | None = None) -> None:
    """Start the static site server and serve forever."""
    # Service-friendly logging, matching the telegram adapter: loon at INFO, libs at WARNING.
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("loon_agent").setLevel(logging.INFO)

    settings = settings or get_settings()
    root = Path(settings.web_root)
    root.mkdir(parents=True, exist_ok=True)

    handler = partial(SiteRequestHandler, directory=str(root))
    server = ThreadingHTTPServer((settings.web_host, settings.web_port), handler)

    # Full mDNS hostname (e.g. "pontoon.local") is what other LAN nodes actually resolve.
    host_label = socket.gethostname().lower() or settings.web_host
    logger.info(
        "loon web up — serving %s at http://%s:%d (bound %s)",
        root, host_label, settings.web_port, settings.web_host,
    )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
