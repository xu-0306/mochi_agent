"""短期對話記憶管理。"""

from __future__ import annotations

from mochi.backends.types import Message


class ConversationMemory:
    """管理短期對話歷史（in-memory）。"""

    def __init__(self, max_messages: int = 50) -> None:
        self._messages: list[Message] = []
        self._max = max_messages

    def add(self, message: Message) -> None:
        """新增一條訊息到短期記憶。"""
        self._messages.append(message)
        if len(self._messages) > self._max:
            # 保留 system message（若有），移除最舊的非 system message
            non_system = [m for m in self._messages if m.role != "system"]
            system = [m for m in self._messages if m.role == "system"]
            self._messages = system + non_system[-(self._max - len(system)):]

    def get_history(self, n: int | None = None) -> list[Message]:
        """取得對話歷史。"""
        if n is None:
            return list(self._messages)
        return list(self._messages[-n:])

    def clear(self) -> None:
        """清空對話歷史。"""
        self._messages.clear()
