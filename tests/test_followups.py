"""Tests for the follow-up store and its chat-loop tools."""

from __future__ import annotations

from loon_agent.tools.followups import FollowupStore, followup_tools


def _tools(tmp_path):
    store = FollowupStore(tmp_path / "followups.sqlite")
    return store, {t.name: t for t in followup_tools(store)}


def test_add_list_resolve_round_trip(tmp_path) -> None:
    store, tools = _tools(tmp_path)

    reply = tools["add_followup"].invoke({"topic": "exec mounts", "note": "audit ro mounts"})
    assert "#1" in reply and "exec mounts" in reply

    listed = tools["list_followups"].invoke({})
    assert "#1 [open] exec mounts — audit ro mounts" in listed

    assert "resolved" in tools["resolve_followup"].invoke({"followup_id": 1, "resolution": "done"})
    assert "no open follow-ups" in tools["list_followups"].invoke({})
    assert "exec mounts" in tools["list_followups"].invoke({"status": "done"})
    assert "exec mounts" in tools["list_followups"].invoke({"status": "all"})


def test_add_rejects_empty_fields(tmp_path) -> None:
    _, tools = _tools(tmp_path)
    assert "error" in tools["add_followup"].invoke({"topic": "  ", "note": "x"})
    assert "error" in tools["add_followup"].invoke({"topic": "x", "note": ""})


def test_resolve_unknown_or_done_id_is_an_error_message(tmp_path) -> None:
    store, tools = _tools(tmp_path)
    assert "error" in tools["resolve_followup"].invoke({"followup_id": 7})
    store.add("t", "n")
    store.resolve(1)
    assert "error" in tools["resolve_followup"].invoke({"followup_id": 1})  # already done


def test_list_rejects_unknown_status(tmp_path) -> None:
    _, tools = _tools(tmp_path)
    assert "error" in tools["list_followups"].invoke({"status": "weird"})


def test_store_survives_reopen(tmp_path) -> None:
    FollowupStore(tmp_path / "f.sqlite").add("persist", "across restarts")
    reopened = FollowupStore(tmp_path / "f.sqlite")
    items = reopened.list("open")
    assert len(items) == 1 and items[0].topic == "persist"
