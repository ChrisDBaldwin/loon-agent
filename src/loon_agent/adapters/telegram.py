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
from ..session import MessageEvent, SessionSource, build_session_key

logger = logging.getLogger(__name__)

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


class LoonTelegramBot:
    """Handlers binding a :class:`LoonRuntime` to a Telegram bot."""

    def __init__(self, runtime: LoonRuntime, allowlist: frozenset[int]) -> None:
        self.runtime = runtime
        self.allowlist = allowlist
        self._last_text: dict[str, str] = {}  # base session key -> last user message

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
                BotCommand("help", "list commands"),
            ]
        )

    application = (
        Application.builder().token(settings.telegram_token).post_init(_post_init).build()
    )
    application.add_handler(CommandHandler("start", bot.on_start))
    application.add_handler(CommandHandler("help", bot.on_help))
    application.add_handler(CommandHandler("status", bot.on_status))
    application.add_handler(CommandHandler("models", bot.on_models))
    application.add_handler(CommandHandler("model", bot.on_model))
    application.add_handler(CommandHandler("new", bot.on_new))
    application.add_handler(CommandHandler("retry", bot.on_retry))
    application.add_handler(CommandHandler("don", bot.on_don))
    application.add_handler(CommandHandler("doff", bot.on_doff))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.on_message))

    print(f"loon telegram bot up — backend={settings.backend}, allowed users={len(allowlist)}")
    application.run_polling()
