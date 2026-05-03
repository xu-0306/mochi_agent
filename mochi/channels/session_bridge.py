"""頻道 chat_id 到 AgentSession 的映射橋接器。"""

from __future__ import annotations


class SessionBridge:
    """將 Discord/Telegram 的 chat_id 映射到對應的 AgentSession。

    session_key 格式："{channel}:{chat_id}"
    例如："discord:123456789" 或 "telegram:channel-987"
    """

    def __init__(self) -> None:
        self._mapping: dict[str, str] = {}

    def get_or_create_session_id(self, channel: str, chat_id: str) -> str:
        """取得或建立對應的 session_id。

        Args:
            channel: 頻道名稱（discord/telegram）。
            chat_id: 平台的聊天室 ID。

        Returns:
            對應的 session_id 字串。
        """
        key = f"{channel}:{chat_id}"
        if key not in self._mapping:
            self._mapping[key] = key  # Phase 4.5 改為 UUID
        return self._mapping[key]
