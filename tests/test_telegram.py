"""Tests for the Telegram adapter — pure helpers plus handler flows with mocked PTB objects."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from loon_agent.adapters.telegram import (
    LoonTelegramBot,
    LoopManager,
    chunk_message,
    normalize_chat_type,
)
from loon_agent.config import Settings
from loon_agent.loops import LoopSpec, LoopStore

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


# --- processing loops --------------------------------------------------------------


def _loop_spec(**overrides) -> LoopSpec:
    defaults = dict(name="a", description="", interval=0.01, max_iterations=5, prompt="p")
    return LoopSpec(**{**defaults, **overrides})


def _loop_runtime(tmp_path, agent, specs) -> SimpleNamespace:
    return SimpleNamespace(
        agent=agent, loops=specs, loop_store=LoopStore(tmp_path / "loops.sqlite")
    )


def test_loop_runs_to_done_and_delivers_each_iteration(tmp_path) -> None:
    replies = iter(["looked at one thing\nLOOP_CONTINUE", "all covered\nLOOP_DONE"])
    agent = SimpleNamespace(invoke=lambda text, key: next(replies))
    runtime = _loop_runtime(tmp_path, agent, {"a": _loop_spec()})
    bot = AsyncMock()

    async def drive() -> None:
        manager = LoopManager(runtime, asyncio.Lock())
        assert "no loop named" in manager.start("nope", 555, bot)
        assert "started" in manager.start("a", 555, bot)
        assert "already running" in manager.start("a", 555, bot)
        await manager._tasks["a"]

    asyncio.run(drive())
    assert runtime.loop_store.get("a").status == "done"
    assert runtime.loop_store.get("a").iteration == 2
    texts = [call.args[1] for call in bot.send_message.await_args_list]
    assert any("looked at one thing" in t for t in texts)
    assert any("finished" in t for t in texts)


def test_loop_stops_at_iteration_cap(tmp_path) -> None:
    agent = SimpleNamespace(invoke=lambda text, key: "still going\nLOOP_CONTINUE")
    runtime = _loop_runtime(tmp_path, agent, {"a": _loop_spec(max_iterations=2)})
    bot = AsyncMock()

    async def drive() -> None:
        manager = LoopManager(runtime, asyncio.Lock())
        manager.start("a", 555, bot)
        await manager._tasks["a"]

    asyncio.run(drive())
    assert runtime.loop_store.get("a").iteration == 2
    texts = [call.args[1] for call in bot.send_message.await_args_list]
    assert any("cap" in t for t in texts)


def test_loop_gives_up_after_consecutive_failures(tmp_path) -> None:
    def boom(text: str, key: str) -> str:
        raise RuntimeError("backend down")

    runtime = _loop_runtime(tmp_path, SimpleNamespace(invoke=boom), {"a": _loop_spec()})
    bot = AsyncMock()

    async def drive() -> None:
        manager = LoopManager(runtime, asyncio.Lock())
        manager.start("a", 555, bot)
        await manager._tasks["a"]

    asyncio.run(drive())
    assert runtime.loop_store.get("a").status == "failed"
    assert runtime.loop_store.get("a").iteration == 3  # _LOOP_MAX_FAILURES


def test_loop_stop_cancels_between_iterations(tmp_path) -> None:
    agent = SimpleNamespace(invoke=lambda text, key: "one\nLOOP_CONTINUE")
    runtime = _loop_runtime(tmp_path, agent, {"a": _loop_spec(interval=60.0)})
    bot = AsyncMock()

    async def drive() -> None:
        manager = LoopManager(runtime, asyncio.Lock())
        assert "not running" in manager.stop("a")
        manager.start("a", 555, bot)
        task = manager._tasks["a"]
        await asyncio.sleep(0.3)  # let iteration 1 finish; the loop is now in its sleep
        assert "stopped" in manager.stop("a")
        with pytest.raises(asyncio.CancelledError):
            await task

    asyncio.run(drive())
    assert runtime.loop_store.get("a").status == "stopped"
    assert runtime.loop_store.get("a").iteration == 1


def test_loop_resume_picks_up_mid_run_state(tmp_path) -> None:
    agent = SimpleNamespace(invoke=lambda text, key: "wrap up\nLOOP_DONE")
    runtime = _loop_runtime(tmp_path, agent, {"a": _loop_spec()})
    runtime.loop_store.activate("a", "555")
    runtime.loop_store.record_iteration("a", 3)
    runtime.loop_store.activate("ghost", "555")  # stored but no definition on disk
    bot = AsyncMock()

    async def drive() -> None:
        manager = LoopManager(runtime, asyncio.Lock())
        manager.resume(bot)
        assert "ghost" not in manager._tasks
        await manager._tasks["a"]

    asyncio.run(drive())
    assert runtime.loop_store.get("a").status == "done"
    assert runtime.loop_store.get("a").iteration == 4  # resumed after the stored 3
    assert runtime.loop_store.get("ghost").status == "failed"


def test_on_loop_command_lists_and_validates(tmp_path) -> None:
    runtime = _runtime()
    runtime.loops = {"a": _loop_spec()}
    runtime.loop_store = LoopStore(tmp_path / "loops.sqlite")
    bot = LoonTelegramBot(runtime, allowlist=frozenset({99}))

    update = _update(user_id=99)
    asyncio.run(bot.on_loop(update, SimpleNamespace(args=[], bot=AsyncMock())))
    (listing,), _ = update.effective_message.reply_text.await_args
    assert "a — every" in listing

    update = _update(user_id=99)
    asyncio.run(bot.on_loop(update, SimpleNamespace(args=["bogus"], bot=AsyncMock())))
    (usage,), _ = update.effective_message.reply_text.await_args
    assert "usage" in usage

    update = _update(user_id=99)
    asyncio.run(bot.on_loop(update, SimpleNamespace(args=["stop", "a"], bot=AsyncMock())))
    (reply,), _ = update.effective_message.reply_text.await_args
    assert "not running" in reply
