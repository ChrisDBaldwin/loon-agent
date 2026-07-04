"""Tests for the chat-loop site management tools (tools/site.py)."""

from __future__ import annotations

from pathlib import Path

from loon_agent.tools.publish import publish_page, source_path_for
from loon_agent.tools.site import (
    delete_page,
    list_pages,
    read_page,
    site_tools,
    update_page,
)

BASE = "http://testhost:8800"


def _publish(web_root: Path, title: str = "Loon Facts", md: str = "# Loon Facts\n\nbird.") -> str:
    result = publish_page(title, md, web_root=web_root)
    assert result.ok
    return Path(result.path).name


# --- publish stores an editable source -------------------------------------------


def test_publish_stores_markdown_source(tmp_path) -> None:
    name = _publish(tmp_path)
    page = tmp_path / name
    assert page.is_file()
    source = source_path_for(page)
    assert source.read_text(encoding="utf-8") == "# Loon Facts\n\nbird."


# --- list -------------------------------------------------------------------------


def test_list_pages_shows_pages_and_hides_source_dir(tmp_path) -> None:
    name = _publish(tmp_path)
    listing = list_pages(tmp_path)
    assert name in listing
    assert ".src" not in listing


def test_list_pages_empty_root(tmp_path) -> None:
    assert "no pages yet" in list_pages(tmp_path / "missing")


# --- read -------------------------------------------------------------------------


def test_read_page_returns_markdown_source(tmp_path) -> None:
    name = _publish(tmp_path)
    assert read_page(tmp_path, name) == "# Loon Facts\n\nbird."


def test_read_page_falls_back_to_html_for_legacy_pages(tmp_path) -> None:
    legacy = tmp_path / "old-page.html"
    legacy.write_text("<html><body>old</body></html>", encoding="utf-8")
    text = read_page(tmp_path, "old-page.html")
    assert text.startswith("[no markdown source stored")
    assert "old" in text


def test_read_page_missing(tmp_path) -> None:
    assert read_page(tmp_path, "nope.html").startswith("error: no such page")


# --- update -----------------------------------------------------------------------


def test_update_page_rerenders_in_place(tmp_path) -> None:
    name = _publish(tmp_path)
    result = update_page(tmp_path, name, "Loon Facts", "# Loon Facts\n\nrevised.")
    assert result == f"updated: {name}"
    assert "revised." in (tmp_path / name).read_text(encoding="utf-8")
    assert read_page(tmp_path, name) == "# Loon Facts\n\nrevised."


def test_update_page_refuses_to_create(tmp_path) -> None:
    assert update_page(tmp_path, "new.html", "t", "m").startswith("error: no such page")


# --- delete -----------------------------------------------------------------------


def test_delete_page_removes_page_and_source(tmp_path) -> None:
    name = _publish(tmp_path)
    assert delete_page(tmp_path, name) == f"deleted: {name}"
    assert not (tmp_path / name).exists()
    assert not source_path_for(tmp_path / name).exists()


# --- filename validation (the injection surface) -----------------------------------


def test_hostile_filenames_are_rejected(tmp_path) -> None:
    (tmp_path.parent / "outside.html").write_text("x", encoding="utf-8")
    for hostile in (
        "../outside.html",
        "/etc/passwd",
        ".src/page.md",
        "sub/dir.html",
        "page.txt",
        ".hidden.html",
        "",
    ):
        assert read_page(tmp_path, hostile).startswith("error: invalid page name")
        assert delete_page(tmp_path, hostile).startswith("error: invalid page name")
        assert update_page(tmp_path, hostile, "t", "m").startswith("error: invalid page name")
    assert (tmp_path.parent / "outside.html").exists()


# --- the bound tool set -------------------------------------------------------------


def test_site_tools_names_and_urls(tmp_path) -> None:
    tools = {t.name: t for t in site_tools(tmp_path, base_url=BASE)}
    assert set(tools) == {
        "list_site_pages",
        "read_site_page",
        "publish_site_page",
        "update_site_page",
        "delete_site_page",
    }

    published = tools["publish_site_page"].invoke(
        {"title": "Hello Site", "markdown": "# Hello\n\nhi."}
    )
    assert published.startswith(f"published: {BASE}/hello-site-")
    name = published.rsplit("/", 1)[1]

    assert BASE in tools["list_site_pages"].invoke({})
    assert name in tools["list_site_pages"].invoke({})
    assert tools["read_site_page"].invoke({"filename": name}) == "# Hello\n\nhi."

    updated = tools["update_site_page"].invoke(
        {"filename": name, "title": "Hello Site", "markdown": "# Hello\n\nedited."}
    )
    assert updated == f"updated: {name} — {BASE}/{name}"

    assert tools["delete_site_page"].invoke({"filename": name}) == f"deleted: {name}"
    assert not (tmp_path / name).exists()
