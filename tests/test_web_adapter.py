"""Tests for the internal website: static serving, gallery index, publish, wiring."""

from __future__ import annotations

import threading
import urllib.error
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer

import pytest

from loon_agent.adapters.web import SiteRequestHandler
from loon_agent.tools.publish import publish_page


@pytest.fixture()
def site(tmp_path):
    """A running server rooted at a tmp site dir; yields (base_url, root_path)."""
    (tmp_path / "report-a.html").write_text("<h1>Report A</h1>", encoding="utf-8")
    handler = partial(SiteRequestHandler, directory=str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", tmp_path
    finally:
        server.shutdown()
        server.server_close()


def _get(url: str):
    with urllib.request.urlopen(url, timeout=5) as resp:
        return resp.status, resp.read().decode("utf-8")


def test_serves_a_published_artifact_file(site) -> None:
    base, _ = site
    status, body = _get(f"{base}/report-a.html")
    assert status == 200
    assert "<h1>Report A</h1>" in body


def test_root_renders_the_gallery_listing_artifacts(site) -> None:
    base, _ = site
    status, body = _get(f"{base}/")
    assert status == 200
    assert 'class="gallery"' in body
    assert "report-a.html" in body
    assert "published artifacts" in body


def test_empty_site_shows_empty_message(tmp_path) -> None:
    handler = partial(SiteRequestHandler, directory=str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        _, body = _get(f"http://127.0.0.1:{server.server_address[1]}/")
        assert "Nothing published yet" in body
    finally:
        server.shutdown()
        server.server_close()


def test_directory_traversal_is_refused(site) -> None:
    base, _ = site
    # The base handler collapses .. before it reaches us; the request must not escape root.
    with pytest.raises(urllib.error.HTTPError) as exc:
        _get(f"{base}/../../etc/passwd")
    assert exc.value.code in (403, 404)


def test_non_get_method_is_rejected(site) -> None:
    base, _ = site
    req = urllib.request.Request(f"{base}/report-a.html", method="DELETE")
    with pytest.raises(urllib.error.HTTPError) as exc:
        urllib.request.urlopen(req, timeout=5)
    assert exc.value.code == 405


# --- publish_page ---------------------------------------------------------------


def test_publish_page_writes_safe_html_into_web_root(tmp_path) -> None:
    result = publish_page(
        "My Update", "# My Update\n\nbody with <script>alert(1)</script>", web_root=tmp_path
    )
    assert result.ok
    files = list(tmp_path.glob("my-update-*.html"))
    assert len(files) == 1
    html_text = files[0].read_text()
    assert "<h1>My Update</h1>" in html_text
    assert "<script>alert" not in html_text  # raw HTML stays escaped


def test_publish_page_dedupes_same_title_same_day(tmp_path) -> None:
    a = publish_page("Dupe", "# Dupe\n\none", web_root=tmp_path)
    b = publish_page("Dupe", "# Dupe\n\ntwo", web_root=tmp_path)
    assert a.path != b.path
    assert len(list(tmp_path.glob("dupe-*.html"))) == 2


# --- /publish skill end-to-end (fake LLM, real skill file) ----------------------


def test_publish_skill_runs_and_writes_a_page(tmp_path) -> None:
    from pathlib import Path

    from fakes import FakeChat
    from loon_agent.masques import MasqueLoader
    from loon_agent.skills import load_skill
    from loon_agent.skills.engine import SkillRunner

    llm = FakeChat(replies=["# Loons\n\nLoons are diving birds."], calls=[])
    runner = SkillRunner(
        llm,
        {
            "publish_page": lambda ctx: str(
                publish_page(
                    str(ctx.get("topic") or "untitled"), str(ctx.get("page") or ""),
                    web_root=tmp_path,
                )
            ),
        },
        masque_loader=MasqueLoader(["masques"]).block,
    )
    result = runner.run(load_skill(Path("skills/publish.md")), {"topic": "loons"})

    assert "published:" in str(result.outputs["result"])
    files = list(tmp_path.glob("loons-*.html"))
    assert len(files) == 1
    assert "Loons are diving birds." in files[0].read_text()
