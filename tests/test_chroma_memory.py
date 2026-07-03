"""Tests for the Chroma-backed memory provider."""

from __future__ import annotations

from loon_agent.memory import ChromaMemoryProvider


def _provider(tmp_path, **kwargs) -> ChromaMemoryProvider:
    return ChromaMemoryProvider(
        db_path=tmp_path / "chroma", notes_path=tmp_path / "MEMORY.md", **kwargs
    )


def test_system_prompt_block_empty_when_no_notes_file(tmp_path) -> None:
    provider = _provider(tmp_path)
    assert provider.system_prompt_block() == ""


def test_system_prompt_block_reads_notes_file(tmp_path) -> None:
    (tmp_path / "MEMORY.md").write_text("Chris prefers terse replies.\n", encoding="utf-8")
    provider = _provider(tmp_path)
    block = provider.system_prompt_block()
    assert "Standing notes (from MEMORY.md):" in block
    assert "Chris prefers terse replies." in block


def test_prefetch_empty_on_empty_store(tmp_path) -> None:
    provider = _provider(tmp_path)
    assert provider.prefetch("loons", session_id="s1") == ""


def test_sync_then_prefetch_recalls_by_semantic_similarity(tmp_path) -> None:
    provider = _provider(tmp_path)
    provider.sync_turn(
        "what is a common loon", "a large diving waterbird found on northern lakes", "s1"
    )
    provider.sync_turn("what's the capital of France", "Paris", "s1")

    recall = provider.prefetch("tell me about diving birds on lakes", session_id="s1")

    assert "Earlier related exchanges:" in recall
    assert "diving waterbird" in recall


def test_prefetch_caps_at_top_k(tmp_path) -> None:
    provider = _provider(tmp_path, top_k=1)
    for i in range(3):
        provider.sync_turn(f"question {i}", f"answer {i}", "s1")

    recall = provider.prefetch("question", session_id="s1")

    assert recall.count("- user:") == 1
