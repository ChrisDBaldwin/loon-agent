"""Tests for the Telegram adapter — pure helpers plus handler flows with mocked PTB objects."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from loon_agent.adapters.telegram import LoonTelegramBot, chunk_message, normalize_chat_type
from loon_agent.config import Settings

# --- chunking -----------------------------------------------------------------


def test_chunk_short_message_is_single_chunk() -> None:
    assert chunk_message("hello") == ["hello"]


def test_chunk_empty_message_is_no_chunks() -> None:
    assert chunk_message("   ") == []


def test_chunk_prefers_newline_boundary() -> None:
    text = "a" * 90 + "\n" + "b" * 90
    chunks = chunk_message(text, limit=100)
    assert chunks == ["a" * 90, "b" * 90]


def test_chunk_falls_back_to_space_then_hard_cut() -> None:
    spaced = "word " * 40  # 200 chars, no newlines
    chunks = chunk_message(spaced, limit=100)
    assert all(len(c) <= 100 for c in chunks)
    assert " ".join(chunks).split() == spaced.split()

    solid = "x" * 250
    chunks = chunk_message(solid, limit=100)
    assert [len(c) for c in chunks] == [100, 100, 50]


def test_chunk_never_exceeds_limit() -> None:
    text = ("paragraph one\n" * 30 + "y" * 500 + "\n") * 3
    for chunk in chunk_message(text, limit=200):
        assert 0 < len(chunk) <= 200


# --- chat-type mapping / allowlist parsing --------------------------------------


def test_normalize_chat_type_maps_telegram_vocabulary() -> None:
    assert normalize_chat_type("private") == "dm"
    assert normalize_chat_type("group") == "group"
    assert normalize_chat_type("supergroup") == "group"
    assert normalize_chat_type("channel") == "channel"
    assert normalize_chat_type("weird") == "weird"  # pass through unknowns


def test_telegram_allowlist_parses_and_defaults_empty() -> None:
    assert Settings(telegram_allowed_users="").telegram_allowlist() == frozenset()
    parsed = Settings(telegram_allowed_users=" 123, 456 ,789 ").telegram_allowlist()
    assert parsed == frozenset({123, 456, 789})


def test_telegram_allowlist_rejects_garbage() -> None:
    with pytest.raises(ValueError, match="numeric"):
        Settings(telegram_allowed_users="123,@chris").telegram_allowlist()


# --- handler flows ---------------------------------------------------------------


def _runtime(agent: object = None) -> SimpleNamespace:
    """Minimal LoonRuntime stand-in: pass-through epochs, no model registry."""
    return SimpleNamespace(
        agent=agent,
        epochs=SimpleNamespace(thread_id=lambda key: key, bump=lambda key: f"{key}:e1"),
    )


def _update(text: str = "hi", user_id: int = 99, chat_type: str = "private") -> SimpleNamespace:
    message = SimpleNamespace(
        text=text,
        is_topic_message=False,
        message_thread_id=None,
        reply_text=AsyncMock(),
    )
    return SimpleNamespace(
        effective_message=message,
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=555, type=chat_type),
    )


def test_unknown_user_is_refused_with_their_id() -> None:
    bot = LoonTelegramBot(_runtime(), allowlist=frozenset({1}))
    update = _update(user_id=99)

    asyncio.run(bot.on_message(update, context=None))

    (refusal,), _ = update.effective_message.reply_text.await_args
    assert "99" in refusal
    # The agent was never touched (it's None — a call would have raised).


def test_authorized_user_gets_agent_reply() -> None:
    agent = SimpleNamespace(invoke=lambda text, session_key: f"echo: {text}")
    bot = LoonTelegramBot(_runtime(agent), allowlist=frozenset({99}))
    update = _update(text="ping", user_id=99)
    context = SimpleNamespace(bot=AsyncMock())

    asyncio.run(bot.on_message(update, context))

    replies = [call.args[0] for call in update.effective_message.reply_text.await_args_list]
    assert replies == ["echo: ping"]


def test_agent_failure_becomes_apology_not_crash() -> None:
    def boom(text: str, session_key: str) -> str:
        raise RuntimeError("backend down")

    bot = LoonTelegramBot(_runtime(SimpleNamespace(invoke=boom)), allowlist=frozenset({99}))
    update = _update(user_id=99)
    context = SimpleNamespace(bot=AsyncMock())

    asyncio.run(bot.on_message(update, context))

    (reply,), _ = update.effective_message.reply_text.await_args
    assert "wrong" in reply.lower()


def test_session_key_distinguishes_dm_from_topic() -> None:
    seen: list[str] = []
    agent = SimpleNamespace(invoke=lambda text, session_key: seen.append(session_key) or "ok")
    bot = LoonTelegramBot(_runtime(agent), allowlist=frozenset({99}))

    for is_topic, thread_id in [(False, None), (True, 7)]:
        update = _update(user_id=99)
        update.effective_message.is_topic_message = is_topic
        update.effective_message.message_thread_id = thread_id
        asyncio.run(bot.on_message(update, SimpleNamespace(bot=AsyncMock())))

    # Same chat, same user — but the forum topic is its own durable conversation.
    assert len(seen) == 2
    assert all(key.startswith("telegram:") for key in seen)
    assert seen[0] != seen[1]


def test_start_command_tells_stranger_their_id() -> None:
    bot = LoonTelegramBot(_runtime(), allowlist=frozenset())
    update = _update(user_id=4242)

    asyncio.run(bot.on_start(update, context=None))

    (reply,), _ = update.effective_message.reply_text.await_args
    assert "4242" in reply
