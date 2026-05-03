# Inspired by hermes-agent/gateway/platforms/telegram.py design pattern
"""Telegram Bot 適配器。"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from collections.abc import Sequence
from typing import Any

from mochi.channels.base import BaseChannel, SendResult
from mochi.channels.events import Attachment, AttachmentEvent, CommandEvent, MessageEvent

try:  # pragma: no cover - optional dependency is covered by adapter tests via injection.
    from telegram import InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import Application, CallbackQueryHandler, MessageHandler, filters
except ImportError:  # pragma: no cover
    InlineKeyboardButton = None  # type: ignore[assignment]
    InlineKeyboardMarkup = None  # type: ignore[assignment]
    Application = None  # type: ignore[assignment]
    CallbackQueryHandler = None  # type: ignore[assignment]
    MessageHandler = None  # type: ignore[assignment]
    filters = None  # type: ignore[assignment]


class TelegramAdapter(BaseChannel):
    """Telegram Bot 頻道適配器（python-telegram-bot）。"""

    name = "telegram"

    def __init__(
        self,
        bot_token: str,
        *,
        allowed_chat_ids: Sequence[int] | None = None,
        allowed_user_ids: Sequence[int] | None = None,
        rate_limit_per_user: int = 10,
    ) -> None:
        super().__init__()
        self._bot_token = bot_token
        self._allowed_chat_ids = set(allowed_chat_ids or [])
        self._allowed_user_ids = set(allowed_user_ids or [])
        self._rate_limit_per_user = max(1, rate_limit_per_user)
        self._recent_user_events: dict[str, deque[float]] = defaultdict(deque)
        self._application: Any | None = None

    async def start(self) -> None:
        """啟動 python-telegram-bot polling。"""
        if (
            Application is None
            or CallbackQueryHandler is None
            or MessageHandler is None
            or filters is None
        ):
            raise RuntimeError(
                "python-telegram-bot is not installed. Install channels extra: "
                "`uv sync --extra channels`."
            )
        if self._application is not None:
            return

        application = Application.builder().token(self._bot_token).build()
        application.add_handler(MessageHandler(filters.TEXT, self._handle_update))
        attachment_filters = (
            filters.PHOTO
            | filters.Document.ALL
            | filters.VOICE
            | filters.AUDIO
            | filters.VIDEO
        )
        application.add_handler(MessageHandler(attachment_filters, self._handle_update))
        application.add_handler(CallbackQueryHandler(self._handle_callback_query, pattern="^help$"))
        await application.initialize()
        if application.updater is not None:
            await application.updater.start_polling()
        await application.start()
        self._application = application

    async def stop(self) -> None:
        """停止 Telegram polling。"""
        application = self._application
        if application is None:
            return
        if application.updater is not None:
            await application.updater.stop()
        await application.stop()
        await application.shutdown()
        self._application = None

    async def send_message(
        self,
        chat_id: str,
        text: str,
        reply_to: str | None = None,
    ) -> SendResult:
        """傳送 Telegram 訊息。"""
        application = self._application
        if application is None:
            return SendResult(success=False, error="Telegram application is not started.")

        try:
            message_id: str | None = None
            reply_to_message_id = int(reply_to) if reply_to else None
            for chunk in _split_message(text, max_length=4096):
                sent = await application.bot.send_message(
                    chat_id=int(chat_id),
                    text=chunk,
                    reply_to_message_id=reply_to_message_id,
                    reply_markup=_help_reply_markup(chunk),
                )
                message_id = str(getattr(sent, "message_id", "") or message_id or "")
                reply_to_message_id = None
            return SendResult(success=True, message_id=message_id)
        except Exception as exc:
            return SendResult(success=False, error=str(exc))

    async def _handle_callback_query(self, update: Any, context: Any) -> None:
        """處理 Telegram inline keyboard callback。"""
        del context
        query = getattr(update, "callback_query", None)
        if query is None or getattr(query, "data", "") != "help":
            return

        if hasattr(query, "answer"):
            await query.answer()
        message = getattr(query, "message", None)
        chat = getattr(message, "chat", None)
        chat_id = getattr(chat, "id", None)
        if chat_id is None:
            return
        await self.send_message(
            str(chat_id),
            "Mochi is ready. Send a message and I will reply in this chat.",
            reply_to=str(getattr(message, "message_id", "") or "") or None,
        )

    async def _handle_update(self, update: Any, context: Any) -> None:
        """將 Telegram update 轉成 Mochi ChannelEvent。"""
        del context
        message = getattr(update, "effective_message", None)
        chat = getattr(update, "effective_chat", None)
        user = getattr(update, "effective_user", None)
        if message is None or chat is None or user is None:
            return

        chat_id = int(getattr(chat, "id", 0) or 0)
        user_id = int(getattr(user, "id", 0) or 0)
        if not self._is_allowed(chat_id=chat_id, user_id=user_id):
            return
        if self._is_rate_limited(str(user_id)):
            return

        message_id = str(getattr(message, "message_id", "") or "")
        attachments = _extract_attachments(message)
        if attachments:
            await self.emit_event(
                AttachmentEvent(
                    channel=self.name,
                    chat_id=str(chat_id),
                    user_id=str(user_id),
                    message_id=message_id,
                    text=str(getattr(message, "caption", "") or "").strip(),
                    attachments=attachments,
                    metadata={"telegram_chat_id": chat_id},
                )
            )
            return

        text = str(getattr(message, "text", "") or "").strip()
        if not text:
            return

        if text.startswith("/"):
            command, args = _parse_command_text(text)
            await self.emit_event(
                CommandEvent(
                    channel=self.name,
                    chat_id=str(chat_id),
                    user_id=str(user_id),
                    command=command,
                    args=args,
                    message_id=message_id,
                    metadata={"telegram_chat_id": chat_id},
                )
            )
            return

        await self.emit_event(
            MessageEvent(
                channel=self.name,
                chat_id=str(chat_id),
                user_id=str(user_id),
                text=text,
                message_id=message_id,
                metadata={"telegram_chat_id": chat_id},
            )
        )

    def _is_allowed(self, *, chat_id: int, user_id: int) -> bool:
        if self._allowed_chat_ids and chat_id not in self._allowed_chat_ids:
            return False
        return not (self._allowed_user_ids and user_id not in self._allowed_user_ids)

    def _is_rate_limited(self, user_id: str) -> bool:
        now = time.monotonic()
        window = self._recent_user_events[user_id]
        while window and now - window[0] >= 60.0:
            window.popleft()
        if len(window) >= self._rate_limit_per_user:
            return True
        window.append(now)
        return False


def _parse_command_text(content: str) -> tuple[str, list[str]]:
    parts = content.removeprefix("/").split()
    if not parts:
        return "", []
    command_part, *args = parts
    command = command_part.split("@", 1)[0] if command_part else ""
    return command, args


def _extract_attachments(message: Any) -> list[Attachment]:
    attachments: list[Attachment] = []

    document = getattr(message, "document", None)
    if document is not None:
        attachments.append(_attachment_from_file(document, attachment_type="document"))

    photo = _largest_photo(getattr(message, "photo", None))
    if photo is not None:
        attachments.append(_attachment_from_file(photo, attachment_type="photo"))

    for attachment_type in ("voice", "audio", "video"):
        media = getattr(message, attachment_type, None)
        if media is not None:
            attachments.append(_attachment_from_file(media, attachment_type=attachment_type))

    return attachments


def _largest_photo(photo_sizes: Any) -> Any | None:
    if not photo_sizes:
        return None
    return max(
        photo_sizes,
        key=lambda photo: (
            int(getattr(photo, "file_size", 0) or 0),
            int(getattr(photo, "width", 0) or 0) * int(getattr(photo, "height", 0) or 0),
        ),
    )


def _attachment_from_file(file_obj: Any, *, attachment_type: str) -> Attachment:
    metadata: dict[str, Any] = {"telegram_attachment_type": attachment_type}
    for attr in ("file_unique_id", "width", "height", "duration"):
        value = getattr(file_obj, attr, None)
        if value is not None:
            metadata[attr] = value

    return Attachment(
        id=str(getattr(file_obj, "file_id", "") or ""),
        filename=str(getattr(file_obj, "file_name", "") or ""),
        content_type=getattr(file_obj, "mime_type", None),
        size=getattr(file_obj, "file_size", None),
        metadata=metadata,
    )


def _help_reply_markup(text: str) -> Any | None:
    if InlineKeyboardButton is None or InlineKeyboardMarkup is None:
        return None
    if text != "Mochi is ready. Send a message and I will reply in this chat.":
        return None
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Ask Mochi", switch_inline_query_current_chat=""),
                InlineKeyboardButton("Help", callback_data="help"),
            ]
        ]
    )


def _split_message(text: str, *, max_length: int) -> list[str]:
    if not text:
        return [""]
    return [text[index : index + max_length] for index in range(0, len(text), max_length)]
