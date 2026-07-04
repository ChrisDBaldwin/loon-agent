"""Telegram adapter: a long-polling bot over the platform-neutral agent core.

Pattern borrowed from hermes-agent's telegram platform adapter, minus its gateway
framework: ``python-telegram-bot`` v22 with long-polling (no public ingress needed in
the homelab), deny-by-default user allowlist, and the same ``MessageEvent`` /
``build_session_key`` machinery the CLI uses — so each DM, group, and forum topic gets
its own durable checkpointed conversation.

Local-model realities: the turn runs in a worker thread (``asyncio.to_thread``) so the
sync agent never blocks the event loop, a background task keeps the ``typing…``
indicator alive through local-model cold starts, and replies are chunked at
Telegram's 4096-char limit.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import AsyncIterator

from telegram import Bot, BotCommand, Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from ..app import LoonRuntime, build_runtime, parse_don_command
from ..commands import (
    HELP_TEXT,
    format_model_list,
    model_inventory,
    pick_model,
    status_text,
)
from ..config import get_settings
from ..loops import LoopSpec, run_iteration
from ..session import MessageEvent, SessionSource, build_session_key

logger = logging.getLogger(__name__)

# A loop stops itself after this many consecutive failed iterations — a broken backend
# should not be retried unattended forever.
_LOOP_MAX_FAILURES = 3

TELEGRAM_MESSAGE_LIMIT = 4096
_TYPING_REFRESH_SECONDS = 4.0
_ERROR_REPLY = "Something went wrong on my end — check the loon logs."

# Telegram chat.type -> SessionSource.chat_type vocabulary.
_CHAT_TYPES = {"private": "dm", "group": "group", "supergroup": "group", "channel": "channel"}


def normalize_chat_type(telegram_chat_type: str) -> str:
    return _CHAT_TYPES.get(telegram_chat_type, telegram_chat_type)


def chunk_message(text: str, limit: int = TELEGRAM_MESSAGE_LIMIT) -> list[str]:
    """Split a reply into <=limit chunks, preferring newline then space boundaries."""
    text = text.strip()
    chunks: list[str] = []
    while len(text) > limit:
        cut = text.rfind("\n", 1, limit)
        if cut == -1:
            cut = text.rfind(" ", 1, limit)
        if cut <= 0:
            cut = limit
        chunks.append(text[:cut].rstrip())
        text = text[cut:].lstrip()
    if text:
        chunks.append(text)
    return chunks


@contextlib.asynccontextmanager
async def _typing(bot: Bot, chat_id: int) -> AsyncIterator[None]:
    """Keep the 'typing…' chat action alive while the (slow, local) turn runs."""

    async def refresh() -> None:
        while True:
            with contextlib.suppress(Exception):
                await bot.send_chat_action(chat_id, ChatAction.TYPING)
            await asyncio.sleep(_TYPING_REFRESH_SECONDS)

    task = asyncio.create_task(refresh())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task


def _source_of(message: object, user: object, chat: object) -> SessionSource:
    """Derive the platform-neutral session source for a Telegram message."""
    topic_id = message.message_thread_id if message.is_topic_message else None
    return SessionSource(
        platform="telegram",
        chat_id=str(chat.id),
        user_id=str(user.id),
        chat_type="thread" if topic_id is not None else normalize_chat_type(chat.type),
        thread_id=str(topic_id) if topic_id is not None else None,
    )


class LoopManager:
    """Drives processing loops (``loops.py``) through the shared agent.

    One asyncio task per running loop, living on the bot's event loop: sleep the
    interval, take the turn lock (so a loop iteration and a user turn never hit the
    single local model concurrently), run one iteration in a worker thread, deliver
    the reply to the chat that started the loop. Run state persists in the
    ``LoopStore``, so ``resume()`` picks up mid-run loops after a service restart.

    ``stop()`` cancels between iterations; an iteration already running in its worker
    thread finishes there but its reply is dropped.
    """

    def __init__(self, runtime: LoonRuntime, turn_lock: asyncio.Lock) -> None:
        self.runtime = runtime
        self.turn_lock = turn_lock
        self._tasks: dict[str, asyncio.Task] = {}

    def status_text(self) -> str:
        loops = self.runtime.loops
        if not loops:
            return "no loops defined — add loop files under loops/*.md."
        lines = ["Loops (/loop start <name> · /loop stop <name>):"]
        for name, spec in sorted(loops.items()):
            run = self.runtime.loop_store.get(name) if self.runtime.loop_store else None
            task = self._tasks.get(name)
            if task is not None and not task.done():
                state = (
                    f"running, iteration {run.iteration}/{spec.max_iterations}"
                    if run
                    else "running"
                )
            else:
                state = run.status if run else "never run"
            lines.append(
                f"• {name} — every {spec.interval:.0f}s, "
                f"≤{spec.max_iterations} iterations [{state}]"
            )
            if spec.description:
                lines.append(f"    {spec.description}")
        return "\n".join(lines)

    def start(self, name: str, chat_id: int, bot: Bot) -> str:
        spec = self.runtime.loops.get(name)
        if spec is None:
            known = ", ".join(sorted(self.runtime.loops)) or "none defined"
            return f"no loop named {name!r} (loops: {known})."
        task = self._tasks.get(name)
        if task is not None and not task.done():
            return f"loop {name!r} is already running — /loop stop {name} first."
        self.runtime.loop_store.activate(name, str(chat_id))
        self._spawn(spec, chat_id, bot, start_iteration=0)
        return (
            f"loop {name!r} started — one iteration now, then every {spec.interval:.0f}s, "
            f"up to {spec.max_iterations} iterations. /loop stop {name} to stop."
        )

    def stop(self, name: str) -> str:
        task = self._tasks.pop(name, None)
        if task is None or task.done():
            return f"loop {name!r} is not running."
        task.cancel()
        self.runtime.loop_store.finish(name, "stopped")
        return f"loop {name!r} stopped."

    def resume(self, bot: Bot) -> None:
        """Restart loops the previous process left mid-run (called once at startup)."""
        if self.runtime.loop_store is None:
            return
        for run in self.runtime.loop_store.running():
            spec = self.runtime.loops.get(run.name)
            if spec is None:
                logger.warning("stored loop %r has no definition — marking failed", run.name)
                self.runtime.loop_store.finish(run.name, "failed")
                continue
            logger.info("resuming loop %s at iteration %d", run.name, run.iteration)
            self._spawn(spec, int(run.chat_id), bot, start_iteration=run.iteration)

    def _spawn(self, spec: LoopSpec, chat_id: int, bot: Bot, *, start_iteration: int) -> None:
        self._tasks[spec.name] = asyncio.create_task(
            self._run(spec, chat_id, bot, start_iteration), name=f"loop:{spec.name}"
        )

    async def _run(self, spec: LoopSpec, chat_id: int, bot: Bot, start_iteration: int) -> None:
        store = self.runtime.loop_store
        iteration = start_iteration
        failures = 0
        while iteration < spec.max_iterations:
            iteration += 1
            result = None
            try:
                async with self.turn_lock:
                    result = await asyncio.to_thread(
                        run_iteration, self.runtime.agent, spec, iteration
                    )
                failures = 0
            except asyncio.CancelledError:
                raise  # stop() already recorded the status
            except Exception:
                logger.exception("loop %s iteration %d failed", spec.name, iteration)
                failures += 1
            store.record_iteration(spec.name, iteration)
            if result is not None:
                await self._deliver(
                    bot,
                    chat_id,
                    f"[{spec.name} #{iteration}/{spec.max_iterations}]\n{result.reply}",
                )
                if result.done:
                    store.finish(spec.name, "done")
                    await self._deliver(
                        bot,
                        chat_id,
                        f"loop {spec.name!r} finished — it declared itself done at "
                        f"iteration {iteration}.",
                    )
                    return
            elif failures >= _LOOP_MAX_FAILURES:
                store.finish(spec.name, "failed")
                await self._deliver(
                    bot,
                    chat_id,
                    f"loop {spec.name!r} stopped after {failures} consecutive failed "
                    "iterations — check the loon logs.",
                )
                return
            await asyncio.sleep(spec.interval)
        store.finish(spec.name, "done")
        await self._deliver(
            bot, chat_id, f"loop {spec.name!r} reached its {spec.max_iterations}-iteration cap."
        )

    async def _deliver(self, bot: Bot, chat_id: int, text: str) -> None:
        for chunk in chunk_message(text) or []:
            try:
                await bot.send_message(chat_id, chunk)
            except Exception:
                # Delivery is best-effort; the loop's real output lives on the site /
                # in follow-ups, so a Telegram hiccup must not kill the run.
                logger.exception("loop delivery to chat %s failed", chat_id)


class LoonTelegramBot:
    """Handlers binding a :class:`LoonRuntime` to a Telegram bot."""

    def __init__(self, runtime: LoonRuntime, allowlist: frozenset[int]) -> None:
        self.runtime = runtime
        self.allowlist = allowlist
        self._last_text: dict[str, str] = {}  # base session key -> last user message
        # One turn at a time — user turns and loop iterations share a single local model.
        self.turn_lock = asyncio.Lock()
        self.loop_manager = LoopManager(runtime, self.turn_lock)

    async def _gate(self, update: Update):
        """Common preamble: unpack the update and enforce the allowlist.

        Returns (message, user, chat) or None if the update is unusable/refused.
        """
        message, user, chat = update.effective_message, update.effective_user, update.effective_chat
        if message is None or user is None or chat is None:
            return None
        if user.id not in self.allowlist:
            await message.reply_text(
                f"Sorry, I only talk to my humans. (your telegram id: {user.id})"
            )
            return None
        return message, user, chat

    async def on_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message, user = update.effective_message, update.effective_user
        if message is None or user is None:
            return
        if user.id in self.allowlist:
            await message.reply_text(
                "loon here — send me a message and I'll think on my own hardware. "
                "/help lists commands."
            )
        else:
            await message.reply_text(
                f"This is a private homelab bot. Your telegram id is {user.id} — "
                "add it to LOON_TELEGRAM_ALLOWED_USERS to get access."
            )

    async def on_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if gated := await self._gate(update):
            await gated[0].reply_text(HELP_TEXT)

    async def on_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not (gated := await self._gate(update)):
            return
        message, user, chat = gated
        session_key = self._session_key(message, user, chat)
        text = await asyncio.to_thread(status_text, self.runtime, session_key)
        await message.reply_text(text)

    async def on_models(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not (gated := await self._gate(update)):
            return
        message = gated[0]
        choices, notes = await asyncio.to_thread(self._inventory)
        await message.reply_text(format_model_list(choices, notes))

    async def on_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not (gated := await self._gate(update)):
            return
        message = gated[0]
        args = context.args if context is not None and context.args else []
        if not args:
            await self.on_models(update, context)
            return
        choices, _ = await asyncio.to_thread(self._inventory)
        picked = pick_model(choices, args[0])
        if isinstance(picked, str):
            await message.reply_text(picked)
            return
        try:
            await asyncio.to_thread(self.runtime.switch_model, picked.backend, picked.model)
        except Exception as exc:
            logger.exception("model switch failed")
            await message.reply_text(f"Switch failed: {exc}")
            return
        logger.info("switched model to %s [%s]", picked.model, picked.backend)
        await message.reply_text(
            f"Now using {picked.model} [{picked.backend}]. Reverts to .env on restart."
        )

    async def on_new(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/new — start a fresh conversation for this chat (old thread stays on disk)."""
        if not (gated := await self._gate(update)):
            return
        message, user, chat = gated
        base_key = build_session_key(_source_of(message, user, chat))
        thread = self.runtime.epochs.bump(base_key)
        logger.info("fresh session started (thread=%s)", thread)
        await message.reply_text("Fresh conversation started — earlier context is set aside.")

    async def on_don(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/don <name> [intent] — become a persona (prompt + tools + memory + creds)."""
        if not (gated := await self._gate(update)):
            return
        message = gated[0]
        if not message.text:
            return
        parsed = parse_don_command(message.text)
        name, intent = parsed if parsed else ("", None)
        if not name:
            await message.reply_text("usage: /don <name> [intent]")
            return
        persona = await asyncio.to_thread(self.runtime.don, name, intent)
        if persona is None:
            await message.reply_text(f"masque {name!r} not found — still baseline.")
            return
        tools = ", ".join(t.name for t in self.runtime.agent.tools) or "(none)"
        await message.reply_text(f"donned {persona.name} v{persona.version.raw} — tools: {tools}")

    async def on_doff(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/doff — back to baseline: all tools, unscoped memory, no persona."""
        if not (gated := await self._gate(update)):
            return
        persona = await asyncio.to_thread(self.runtime.doff)
        await gated[0].reply_text("baseline restored." if persona else "no masque was active.")

    async def on_loop(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/loop [list|start <name>|stop <name>] — manage self-directed processing loops."""
        if not (gated := await self._gate(update)):
            return
        message, _user, chat = gated
        args = context.args if context is not None and context.args else []
        if not args or args[0] == "list":
            await message.reply_text(self.loop_manager.status_text())
            return
        action, name = args[0], args[1] if len(args) > 1 else ""
        if action == "start" and name:
            reply = self.loop_manager.start(name, chat.id, context.bot)
        elif action == "stop" and name:
            reply = self.loop_manager.stop(name)
        else:
            reply = "usage: /loop [list] · /loop start <name> · /loop stop <name>"
        await message.reply_text(reply)

    async def on_retry(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/retry — send the previous message again (fresh roll on a flaky local model)."""
        if not (gated := await self._gate(update)):
            return
        message, user, chat = gated
        base_key = build_session_key(_source_of(message, user, chat))
        last = self._last_text.get(base_key)
        if not last:
            await message.reply_text("Nothing to retry yet — send a message first.")
            return
        await self._run_turn(message, user, chat, context, last)

    async def on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        if not (gated := await self._gate(update)):
            return
        message, user, chat = gated
        if not message.text:
            return
        base_key = build_session_key(_source_of(message, user, chat))
        self._last_text[base_key] = message.text
        await self._run_turn(message, user, chat, context, message.text)

    # --- shared plumbing ---------------------------------------------------------

    def _session_key(self, message: object, user: object, chat: object) -> str:
        base_key = build_session_key(_source_of(message, user, chat))
        return self.runtime.epochs.thread_id(base_key)

    def _inventory(self):
        return model_inventory(
            self.runtime.settings,
            active_backend=self.runtime.active_backend,
            active_model=self.runtime.active_model,
        )

    async def _run_turn(self, message, user, chat, context, text: str) -> None:
        session_key = self._session_key(message, user, chat)
        source = _source_of(message, user, chat)
        event = MessageEvent(source=source, text=text)

        async with _typing(context.bot, chat.id):
            # The shared turn lock serializes this with any running loop iteration; the
            # typing indicator stays alive while we wait for the model to free up.
            async with self.turn_lock:
                try:
                    reply = await asyncio.to_thread(
                        self.runtime.agent.invoke, event.text, session_key
                    )
                except Exception:
                    logger.exception("turn failed (session=%s)", session_key)
                    reply = _ERROR_REPLY

        for chunk in chunk_message(reply) or ["(no reply)"]:
            await message.reply_text(chunk)
        logger.info("turn done (session=%s, reply=%d chars)", session_key, len(reply))


def run_telegram() -> None:
    # Service-friendly logging: loon at INFO, libraries at WARNING (httpx would
    # otherwise log every getUpdates poll).
    logging.basicConfig(format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger().setLevel(logging.WARNING)
    logging.getLogger("loon_agent").setLevel(logging.INFO)

    settings = get_settings()
    if not settings.telegram_token:
        raise SystemExit(
            "LOON_TELEGRAM_TOKEN is not set. Create a bot with @BotFather and put its "
            "token in .env (see .env.example)."
        )
    allowlist = settings.telegram_allowlist()
    if not allowlist:
        logger.warning(
            "LOON_TELEGRAM_ALLOWED_USERS is empty — the bot will refuse everyone. "
            "DM the bot once and it will tell you your id."
        )

    runtime = build_runtime(settings)
    bot = LoonTelegramBot(runtime, allowlist)

    async def _post_init(app: Application) -> None:
        # Populate Telegram's command menu (the "/" button in the UI).
        await app.bot.set_my_commands(
            [
                BotCommand("new", "start a fresh conversation"),
                BotCommand("retry", "send your previous message again"),
                BotCommand("models", "list models available to switch to"),
                BotCommand("model", "switch model: /model <n>"),
                BotCommand("status", "backend, server health, session info"),
                BotCommand("don", "become a persona: /don <name> [intent]"),
                BotCommand("doff", "return to baseline"),
                BotCommand("loop", "processing loops: /loop start|stop <name>"),
                BotCommand("help", "list commands"),
            ]
        )
        # Pick up loops the previous process left mid-run (kickstart restarts).
        bot.loop_manager.resume(app.bot)

    application = Application.builder().token(settings.telegram_token).post_init(_post_init).build()
    application.add_handler(CommandHandler("start", bot.on_start))
    application.add_handler(CommandHandler("help", bot.on_help))
    application.add_handler(CommandHandler("status", bot.on_status))
    application.add_handler(CommandHandler("models", bot.on_models))
    application.add_handler(CommandHandler("model", bot.on_model))
    application.add_handler(CommandHandler("new", bot.on_new))
    application.add_handler(CommandHandler("retry", bot.on_retry))
    application.add_handler(CommandHandler("don", bot.on_don))
    application.add_handler(CommandHandler("doff", bot.on_doff))
    application.add_handler(CommandHandler("loop", bot.on_loop))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.on_message))

    print(f"loon telegram bot up — backend={settings.backend}, allowed users={len(allowlist)}")
    application.run_polling()
