"""Research report rendering: model-written markdown -> a self-contained HTML file.

The model never writes HTML. It writes a markdown briefing; this module converts it
with markdown-it in ``js-default`` mode — raw HTML in the (untrusted, web-derived)
briefing is escaped, so a hostile page can skew a summary but cannot inject script
into the report. One file, embedded CSS, no external assets.
"""

from __future__ import annotations

import datetime as _dt
import html
import re
from pathlib import Path

from markdown_it import MarkdownIt

from .tools.web import FetchedPage

_MD = MarkdownIt("js-default")  # html=False: raw HTML from the model/web stays escaped
_SLUG_RE = re.compile(r"[^a-z0-9]+")

# Shared visual language for every page loon serves (reports, published pages, the gallery
# index). Plain string — single braces — concatenated into the shell so its CSS braces never
# reach a str.format() call.
_CSS = """
  :root {
    --bg: #ffffff; --fg: #1a1d21; --muted: #667085; --accent: #0b6e4f;
    --card: #f5f7f9; --border: #e3e8ee;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #14171a; --fg: #e6e9ec; --muted: #98a2b3; --accent: #57c297;
      --card: #1d2126; --border: #2c3238;
    }
  }
  body {
    margin: 0 auto; padding: 2.5rem 1.5rem 4rem; max-width: 46rem;
    background: var(--bg); color: var(--fg);
    font: 17px/1.6 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }
  header { border-bottom: 2px solid var(--accent); margin-bottom: 2rem; }
  header h1 { font-size: 1.7rem; margin: 0 0 .3rem; }
  header p { color: var(--muted); margin: 0 0 1rem; font-size: .9rem; }
  main h1, main h2 { font-size: 1.25rem; margin-top: 2rem; }
  main h3 { font-size: 1.05rem; }
  a { color: var(--accent); }
  code { background: var(--card); padding: .1em .35em; border-radius: 4px; font-size: .9em; }
  pre { background: var(--card); padding: 1rem; border-radius: 8px; overflow-x: auto; }
  blockquote { margin: 0; padding: .2rem 1rem; border-left: 3px solid var(--border);
    color: var(--muted); }
  .sources { margin-top: 3rem; padding: 1.2rem 1.5rem; background: var(--card);
    border: 1px solid var(--border); border-radius: 10px; }
  .sources h2 { margin: 0 0 .8rem; font-size: 1rem; }
  .sources ol { margin: 0; padding-left: 1.4rem; }
  .sources li { margin: .35rem 0; font-size: .92rem; overflow-wrap: anywhere; }
  .sources .skipped { color: var(--muted); }
  .gallery { list-style: none; padding: 0; margin: 2rem 0 0; }
  .gallery li { border: 1px solid var(--border); border-radius: 10px; background: var(--card);
    margin: .6rem 0; }
  .gallery a { display: block; padding: .9rem 1.1rem; text-decoration: none; color: var(--fg); }
  .gallery a:hover { border-color: var(--accent); }
  .gallery .name { font-weight: 600; overflow-wrap: anywhere; }
  .gallery .meta { color: var(--muted); font-size: .82rem; margin-top: .2rem; }
  .empty { color: var(--muted); margin-top: 2rem; }
  footer { margin-top: 3rem; color: var(--muted); font-size: .8rem;
    border-top: 1px solid var(--border); padding-top: 1rem; }
"""


def html_shell(title: str, body_html: str) -> str:
    """Wrap a body fragment in loon's standard, self-contained HTML shell.

    ``title`` must already be escaped by the caller; ``body_html`` is inserted verbatim, so
    callers are responsible for escaping any untrusted content within it.
    """
    return (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
        f"<title>{title}</title>\n<style>{_CSS}</style>\n</head>\n<body>\n"
        f"{body_html}\n</body>\n</html>\n"
    )


def render_page(title: str, markdown: str, *, subtitle: str = "") -> str:
    """Render a titled markdown page to a self-contained HTML document (no source list).

    Same escaping guarantees as :func:`render_report`: the model writes markdown, markdown-it
    runs in ``js-default`` mode, and the title is escaped — raw HTML stays inert.
    """
    safe_title = html.escape(title)
    sub = f"<p>{html.escape(subtitle)}</p>\n" if subtitle else ""
    body = (
        f"<header>\n  <h1>{safe_title}</h1>\n  {sub}</header>\n"
        f"<main>\n{_MD.render(markdown)}\n</main>\n"
        f"<footer>published by loon-agent · {_dt.date.today().isoformat()}</footer>"
    )
    return html_shell(safe_title, body)


def render_report(
    *,
    topic: str,
    briefing_md: str,
    pages: list[FetchedPage],
    failures: list[str] | None = None,
    model: str = "",
    backend: str = "",
) -> str:
    """Render the briefing + numbered source list into a single HTML document."""
    sources = "\n".join(
        f'    <li><a href="{html.escape(page.url, quote=True)}">'
        f"{html.escape(page.title or page.url)}</a></li>"
        for page in pages
    )
    skipped = ""
    if failures:
        items = "".join(f"<li>{html.escape(note)}</li>" for note in failures)
        skipped = (
            '  <p class="skipped">Skipped during the run:</p>\n'
            f'  <ul class="skipped">{items}</ul>'
        )

    safe_title = html.escape(topic)
    date = _dt.date.today().isoformat()
    body = (
        f"<header>\n  <h1>{safe_title}</h1>\n  <p>research briefing · {date}</p>\n</header>\n"
        f"<main>\n{_MD.render(briefing_md)}\n</main>\n"
        '<section class="sources">\n  <h2>Sources</h2>\n  <ol>\n'
        f"{sources}\n  </ol>\n{skipped}\n</section>\n"
        f"<footer>generated by loon-agent · {html.escape(model)} on "
        f"{html.escape(backend)} · {date}</footer>"
    )
    return html_shell(safe_title, body)


def write_report(html_text: str, topic: str, reports_dir: Path | str) -> Path:
    """Write the report to ``<reports_dir>/<slug>-<date>.html`` (deduped) and return it."""
    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    slug = _SLUG_RE.sub("-", topic.lower()).strip("-")[:60] or "report"
    stem = f"{slug}-{_dt.date.today().isoformat()}"
    path = reports_dir / f"{stem}.html"
    counter = 2
    while path.exists():
        path = reports_dir / f"{stem}-{counter}.html"
        counter += 1

    path.write_text(html_text, encoding="utf-8")
    return path
