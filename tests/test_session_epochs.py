"""Tests for /new session management: the epoch store and the Telegram handler."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

from loon_agent.adapters.telegram import LoonTelegramBot
from loon_agent.session import SessionEpochs


def _store(tmp_path) -> SessionEpochs:
    return SessionEpochs(tmp_path / "sessions.sqlite")


def test_epoch_zero_is_the_base_key_itself(tmp_path) -> None:
    epochs = _store(tmp_path)
    assert epochs.thread_id("telegram:abc") == "telegram:abc"


def test_bump_starts_new_threads_and_is_per_conversation(tmp_path) -> None:
    epochs = _store(tmp_path)

    assert epochs.bump("telegram:abc") == "telegram:abc:e1"
    assert epochs.bump("telegram:abc") == "telegram:abc:e2"
    assert epochs.thread_id("telegram:abc") == "telegram:abc:e2"
    # Another conversation is untouched.
    assert epochs.thread_id("cli:xyz") == "cli:xyz"


def test_epochs_persist_across_reopen(tmp_path) -> None:
    _store(tmp_path).bump("telegram:abc")
    assert _store(tmp_path).thread_id("telegram:abc") == "telegram:abc:e1"


# --- Telegram /new -----------------------------------------------------------------


def _update(user_id: int = 99, text: str = "hi") -> SimpleNamespace:
    message = SimpleNamespace(
        text=text,
        is_topic_message=False,
        message_thread_id=None,
        reply_text=AsyncMock(),
    )
    return SimpleNamespace(
        effective_message=message,
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=555, type="private"),
    )


def test_new_command_rotates_the_thread_used_by_messages(tmp_path) -> None:
    seen: list[str] = []
    agent = SimpleNamespace(invoke=lambda text, session_key: seen.append(session_key) or "ok")
    bot = LoonTelegramBot(agent, allowlist=frozenset({99}), epochs=_store(tmp_path))
    context = SimpleNamespace(bot=AsyncMock())

    asyncio.run(bot.on_message(_update(), context))
    asyncio.run(bot.on_new(_update(text="/new"), context))
    asyncio.run(bot.on_message(_update(), context))

    assert len(seen) == 2
    assert seen[0] != seen[1]
    assert seen[1].endswith(":e1")


def test_new_command_refuses_strangers(tmp_path) -> None:
    epochs = _store(tmp_path)
    bot = LoonTelegramBot(agent=None, allowlist=frozenset({1}), epochs=epochs)
    update = _update(user_id=99, text="/new")

    asyncio.run(bot.on_new(update, context=None))

    (refusal,), _ = update.effective_message.reply_text.await_args
    assert "99" in refusal
    assert epochs.thread_id("whatever") == "whatever"  # nothing was bumped
