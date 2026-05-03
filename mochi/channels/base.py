# Inspired by hermes-agent/gateway/platforms/base.py design pattern
"""頻道適配器抽象基類。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mochi.channels.events import ChannelEvent


ChannelEventHandler = Callable[["ChannelEvent"], Awaitable[None]]


@dataclass
class SendResult:
    """訊息傳送結果。"""

    success: bool
    message_id: str | None = None
    error: str | None = None


class BaseChannel(ABC):
    """頻道適配器抽象（Discord / Telegram / 未來擴充）。"""

    name: str = ""

    def __init__(self) -> None:
        self._event_handler: ChannelEventHandler | None = None

    def set_event_handler(self, handler: ChannelEventHandler) -> None:
        """設定收到平台事件後要呼叫的統一處理器。"""
        self._event_handler = handler

    async def emit_event(self, event: ChannelEvent) -> None:
        """將平台事件交給 ChannelManager。"""
        if self._event_handler is None:
            return
        await self._event_handler(event)

    @abstractmethod
    async def start(self) -> None:
        """啟動頻道監聽。"""
        ...

    @abstractmethod
    async def stop(self) -> None:
        """停止頻道監聽。"""
        ...

    @abstractmethod
    async def send_message(
        self,
        chat_id: str,
        text: str,
        reply_to: str | None = None,
    ) -> SendResult:
        """傳送訊息到指定聊天室。"""
        ...
