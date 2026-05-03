"""頻道事件型別定義。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass
class MessageEvent:
    """收到文字訊息事件。"""

    type: Literal["message"] = field(default="message", init=False)
    channel: str = ""
    chat_id: str = ""
    user_id: str = ""
    text: str = ""
    message_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class CommandEvent:
    """收到 slash command 事件。"""

    type: Literal["command"] = field(default="command", init=False)
    channel: str = ""
    chat_id: str = ""
    user_id: str = ""
    command: str = ""
    args: list[str] = field(default_factory=list)
    message_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Attachment:
    """平台附件 metadata，不包含下載後的檔案內容。"""

    id: str = ""
    filename: str = ""
    url: str = ""
    content_type: str | None = None
    size: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttachmentEvent:
    """收到帶附件訊息事件。"""

    type: Literal["attachment"] = field(default="attachment", init=False)
    channel: str = ""
    chat_id: str = ""
    user_id: str = ""
    message_id: str = ""
    text: str = ""
    attachments: list[Attachment] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


ChannelEvent = MessageEvent | CommandEvent | AttachmentEvent
