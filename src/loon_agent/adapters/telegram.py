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

from telegram import Bot, Update
from telegram.constants import ChatAction
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from ..app import build_runtime
from ..config import get_settings
from ..graph import LoonAgent
from ..session import MessageEvent, SessionEpochs, SessionSource, build_session_key

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
    """Handlers binding a :class:`LoonAgent` to a Telegram bot."""

    def __init__(
        self,
        agent: LoonAgent,
        allowlist: frozenset[int],
        epochs: SessionEpochs | None = None,
    ) -> None:
        self.agent = agent
        self.allowlist = allowlist
        self.epochs = epochs

    async def on_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message, user = update.effective_message, update.effective_user
        if message is None or user is None:
            return
        if user.id in self.allowlist:
            await message.reply_text(
                "loon here — send me a message and I'll think on my own hardware."
            )
        else:
            await message.reply_text(
                f"This is a private homelab bot. Your telegram id is {user.id} — "
                "add it to LOON_TELEGRAM_ALLOWED_USERS to get access."
            )

    async def on_new(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """/new — start a fresh conversation for this chat (old thread stays on disk)."""
        message, user, chat = update.effective_message, update.effective_user, update.effective_chat
        if message is None or user is None or chat is None:
            return
        if user.id not in self.allowlist:
            await message.reply_text(
                f"Sorry, I only talk to my humans. (your telegram id: {user.id})"
            )
            return
        if self.epochs is None:
            await message.reply_text("Session management isn't enabled here.")
            return
        base_key = build_session_key(_source_of(message, user, chat))
        thread = self.epochs.bump(base_key)
        logger.info("fresh session started (thread=%s)", thread)
        await message.reply_text("Fresh conversation started — earlier context is set aside.")

    async def on_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        message, user, chat = update.effective_message, update.effective_user, update.effective_chat
        if message is None or user is None or chat is None or not message.text:
            return
        if user.id not in self.allowlist:
            await message.reply_text(
                f"Sorry, I only talk to my humans. (your telegram id: {user.id})"
            )
            return

        source = _source_of(message, user, chat)
        base_key = build_session_key(source)
        session_key = self.epochs.thread_id(base_key) if self.epochs else base_key
        event = MessageEvent(source=source, text=message.text)

        async with _typing(context.bot, chat.id):
            try:
                reply = await asyncio.to_thread(self.agent.invoke, event.text, session_key)
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
    bot = LoonTelegramBot(runtime.agent, allowlist, epochs=runtime.epochs)

    application = Application.builder().token(settings.telegram_token).build()
    application.add_handler(CommandHandler("start", bot.on_start))
    application.add_handler(CommandHandler("new", bot.on_new))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, bot.on_message))

    print(f"loon telegram bot up — backend={settings.backend}, allowed users={len(allowlist)}")
    application.run_polling()
