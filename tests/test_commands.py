"""Tests for the shared slash-command logic (/models, /model, /status, /retry, /help)."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import httpx

from loon_agent.adapters.telegram import LoonTelegramBot
from loon_agent.app import LoonRuntime
from loon_agent.commands import (
    HELP_TEXT,
    ModelChoice,
    format_model_list,
    model_inventory,
    pick_model,
    probe_models,
    status_text,
)
from loon_agent.config import Settings

# --- inventory -------------------------------------------------------------------


def _models_response(ids: list[str]) -> MagicMock:
    response = MagicMock(spec=httpx.Response)
    response.json.return_value = {"data": [{"id": i} for i in ids]}
    return response


def test_probe_models_lists_sorted_chat_ids() -> None:
    backend = SimpleNamespace(base_url="http://box:1234/v1", api_key="k", model="m")
    with patch(
        "loon_agent.commands.httpx.get",
        return_value=_models_response(["zeta", "alpha", "text-embedding-nomic-v1.5"]),
    ) as get:
        ids, latency = probe_models(backend)

    assert ids == ["alpha", "zeta"]  # embedding models are not chat-switchable
    assert latency >= 0
    assert get.call_args.args[0] == "http://box:1234/v1/models"


def test_inventory_numbers_across_backends_and_notes_unreachable(monkeypatch) -> None:
    import os

    # Hermetic: drop any backends the developer's real .env defined.
    for key in list(os.environ):
        if key.startswith("LOON_") and key.endswith("_BASE_URL"):
            monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("LOON_AAA_BASE_URL", "http://aaa:1234/v1")
    monkeypatch.setenv("LOON_AAA_MODEL", "configured-model")
    monkeypatch.setenv("LOON_BBB_BASE_URL", "http://bbb:8000/v1")
    monkeypatch.setenv("LOON_BBB_MODEL", "b-model")
    monkeypatch.delenv("LOON_LOCAL_BASE_URL", raising=False)

    def fake_probe(backend, timeout=4.0):
        if "aaa" in backend.base_url:
            return ["served-model"], 0.01
        raise httpx.ConnectError("down")

    with patch("loon_agent.commands.probe_models", side_effect=fake_probe):
        choices, notes = model_inventory(
            Settings(), active_backend="aaa", active_model="served-model"
        )

    # aaa contributes its configured model (not reported by the server) + the served one;
    # bbb and the default 'local' are unreachable -> notes, no entries.
    assert [(c.index, c.backend, c.model) for c in choices] == [
        (1, "aaa", "configured-model"),
        (2, "aaa", "served-model"),
    ]
    assert choices[1].active
    assert any("bbb" in note for note in notes)
    assert len(notes) == 2  # bbb + local


def test_format_model_list_marks_active_and_handles_empty() -> None:
    text = format_model_list(
        [ModelChoice(1, "a", "m1"), ModelChoice(2, "b", "m2", active=True)], ["c: unreachable"]
    )
    lines = text.splitlines()
    assert "1. m1" in lines[1] and not lines[1].startswith("→")
    assert lines[2].startswith("→") and "m2" in lines[2]
    assert "! c: unreachable" in lines[3]

    assert "No backends" in format_model_list([], [])


def test_pick_model_validates_input() -> None:
    choices = [ModelChoice(1, "a", "m1"), ModelChoice(2, "a", "m2")]
    assert pick_model(choices, "two") == "Usage: /model <n> — run /models to see the numbered list."
    assert "1–2" in pick_model(choices, "7")
    assert pick_model([], "1") == "No models available to switch to — run /models to see why."
    assert pick_model(choices, "2") == ModelChoice(2, "a", "m2")


# --- switch_model ------------------------------------------------------------------


def _bare_runtime(**overrides) -> LoonRuntime:
    defaults = dict(
        agent=SimpleNamespace(),
        skills={},
        runner=SimpleNamespace(llm=None),
        settings=Settings(memory_backend="sqlite"),
        epochs=SimpleNamespace(thread_id=lambda k: k, bump=lambda k: f"{k}:e1"),
        active_backend="local",
        active_model="old-model",
    )
    defaults.update(overrides)
    return LoonRuntime(**defaults)


def test_switch_model_swaps_agent_and_runner_llm(monkeypatch) -> None:
    from fakes import FakeChat

    runtime = _bare_runtime()
    fake_llm = FakeChat(replies=["hi"], calls=[])
    with patch("loon_agent.app.make_llm", return_value=fake_llm) as make:
        runtime.switch_model("gpubox", "big-model")

    make.assert_called_once_with("gpubox", settings=runtime.settings, model="big-model")
    assert runtime.active_backend == "gpubox"
    assert runtime.active_model == "big-model"
    assert runtime.runner.llm is fake_llm
    assert runtime.agent.graph is not None  # rebuilt LoonAgent, compiled


# --- status --------------------------------------------------------------------------


def test_status_text_reports_server_session_and_memory(monkeypatch) -> None:
    monkeypatch.setenv("LOON_LOCAL_BASE_URL", "http://localhost:1234/v1")
    state = SimpleNamespace(values={"messages": [1, 2, 3]})
    agent = SimpleNamespace(graph=SimpleNamespace(get_state=lambda config: state))
    runtime = _bare_runtime(agent=agent, skills={"research": object()})

    with patch("loon_agent.commands.probe_models", return_value=(["m"], 0.05)):
        text = status_text(runtime, "cli:abc:e2")

    assert "backend: local (http://localhost:1234/v1)" in text
    assert "model: old-model" in text
    assert "server: ok (50 ms)" in text
    assert "session: cli:abc:e2 · 3 messages" in text
    assert "memory: sqlite · skills: research" in text


def test_status_text_survives_dead_server_and_empty_thread() -> None:
    def no_state(config):
        raise RuntimeError("no checkpoint")

    agent = SimpleNamespace(graph=SimpleNamespace(get_state=no_state))
    runtime = _bare_runtime(agent=agent)

    with patch("loon_agent.commands.probe_models", side_effect=httpx.ConnectError("refused")):
        text = status_text(runtime, "cli:abc")

    assert "UNREACHABLE" in text
    assert "0 messages" in text


# --- telegram handlers ----------------------------------------------------------------


def _update(text: str = "hi", user_id: int = 99) -> SimpleNamespace:
    message = SimpleNamespace(
        text=text, is_topic_message=False, message_thread_id=None, reply_text=AsyncMock()
    )
    return SimpleNamespace(
        effective_message=message,
        effective_user=SimpleNamespace(id=user_id),
        effective_chat=SimpleNamespace(id=555, type="private"),
    )


def _replies(update) -> list[str]:
    return [c.args[0] for c in update.effective_message.reply_text.await_args_list]


def test_help_is_gated_and_lists_commands() -> None:
    bot = LoonTelegramBot(_bare_runtime(), allowlist=frozenset({99}))

    stranger = _update(user_id=1)
    asyncio.run(bot.on_help(stranger, context=None))
    assert "only talk to my humans" in _replies(stranger)[0]

    friend = _update()
    asyncio.run(bot.on_help(friend, context=None))
    assert _replies(friend) == [HELP_TEXT]
    assert "/model <n>" in HELP_TEXT


def test_retry_resends_last_message_and_requires_history() -> None:
    seen: list[str] = []
    agent = SimpleNamespace(invoke=lambda text, session_key: seen.append(text) or "ok")
    bot = LoonTelegramBot(_bare_runtime(agent=agent), allowlist=frozenset({99}))
    context = SimpleNamespace(bot=AsyncMock())

    empty = _update(text="/retry")
    asyncio.run(bot.on_retry(empty, context))
    assert "Nothing to retry" in _replies(empty)[0]

    asyncio.run(bot.on_message(_update(text="what is a loon?"), context))
    asyncio.run(bot.on_retry(_update(text="/retry"), context))
    assert seen == ["what is a loon?", "what is a loon?"]


def test_model_command_switches_by_index() -> None:
    runtime = _bare_runtime()
    switched: list[tuple[str, str]] = []
    runtime.switch_model = lambda b, m: switched.append((b, m))  # type: ignore[method-assign]
    bot = LoonTelegramBot(runtime, allowlist=frozenset({99}))
    choices = [ModelChoice(1, "local", "small"), ModelChoice(2, "gpubox", "big")]

    with patch.object(bot, "_inventory", return_value=(choices, [])):
        update = _update(text="/model 2")
        asyncio.run(bot.on_model(update, SimpleNamespace(bot=AsyncMock(), args=["2"])))

    assert switched == [("gpubox", "big")]
    assert "Now using big [gpubox]" in _replies(update)[0]


def test_model_command_without_args_lists() -> None:
    bot = LoonTelegramBot(_bare_runtime(), allowlist=frozenset({99}))
    choices = [ModelChoice(1, "local", "small", active=True)]

    with patch.object(bot, "_inventory", return_value=(choices, [])):
        update = _update(text="/model")
        asyncio.run(bot.on_model(update, SimpleNamespace(bot=AsyncMock(), args=[])))

    assert "1. small" in _replies(update)[0]
