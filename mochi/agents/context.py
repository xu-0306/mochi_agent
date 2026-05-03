"""上下文管理器：整合短期對話與長期記憶檢索。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from mochi.agents.compaction import ConversationCompactor
from mochi.backends.types import Message
from mochi.memory.conversation import ConversationMemory

if TYPE_CHECKING:
    from mochi.memory.store import MemoryStore
else:
    class MemoryStore(Protocol):
        """長期記憶檢索介面（duck typing）。"""

        async def search(self, query: str, top_k: int = 5) -> list[Any]:
            """搜尋相關記憶條目。"""


@dataclass
class PromptContext:
    """單輪輸入前的上下文資料。"""

    history: list[Message]
    """短期對話歷史。"""

    summary: str | None = None
    """滾動對話摘要（僅注入 prompt，不回寫 canonical message）。"""

    memory_context: str | None = None
    """長期記憶檢索結果（純文字）。"""


class ContextManager:
    """管理 prompt 組裝所需上下文。"""

    def __init__(
        self,
        conversation_memory: ConversationMemory | None = None,
        memory_store: MemoryStore | None = None,
        compactor: ConversationCompactor | None = None,
        *,
        history_window: int = 20,
        memory_top_k: int = 5,
    ) -> None:
        """初始化 ContextManager。

        Args:
            conversation_memory: 短期對話記憶實例，未提供時自動建立。
            memory_store: 長期記憶儲存實例（可為 None）。
            compactor: 對話壓縮器（可選）。
            history_window: 預設回傳歷史訊息數量。
            memory_top_k: 長期記憶檢索數量上限。
        """
        self._conversation = conversation_memory or ConversationMemory()
        self._memory_store = memory_store
        self._compactor = compactor
        self._history_window = history_window
        self._memory_top_k = memory_top_k
        self._summary: str | None = None

    def add_message(self, message: Message) -> None:
        """加入一則訊息到短期對話歷史。"""
        self._conversation.add(message)

    def get_recent_history(self, limit: int | None = None) -> list[Message]:
        """取得最近對話歷史。"""
        n = self._history_window if limit is None else limit
        return self._conversation.get_history(n)

    def clear_history(self) -> None:
        """清空短期對話歷史。"""
        self._conversation.clear()
        self._summary = None

    @property
    def summary(self) -> str | None:
        """目前滾動摘要。"""
        return self._summary

    async def prepare_prompt_context(
        self,
        user_message: str,
        *,
        history_limit: int | None = None,
        memory_top_k: int | None = None,
    ) -> PromptContext:
        """為新的使用者訊息準備 prompt 上下文。"""
        self._compact_history_if_needed()
        history = self.get_recent_history(history_limit)
        memory_context = await self._retrieve_memory_context(
            query=user_message,
            top_k=memory_top_k or self._memory_top_k,
        )
        return PromptContext(
            history=history,
            summary=self._summary,
            memory_context=memory_context,
        )

    def _compact_history_if_needed(self) -> None:
        """必要時將過舊歷史壓縮為滾動摘要。"""
        if self._compactor is None:
            return

        history = self._conversation.get_history()
        result = self._compactor.compact(
            history,
            previous_summary=self._summary,
        )
        if result is None:
            return

        self._summary = result.summary
        self._conversation.clear()
        for message in result.retained_history:
            self._conversation.add(message)

    async def _retrieve_memory_context(self, query: str, top_k: int) -> str | None:
        """檢索長期記憶並格式化為 prompt 可用文字。"""
        if self._memory_store is None or not query.strip():
            return None

        try:
            entries = await self._memory_store.search(query=query, top_k=top_k)
        except Exception:
            # 長期記憶失敗不應阻斷主流程，回退為僅使用短期歷史。
            return None

        lines: list[str] = []
        for idx, entry in enumerate(entries, start=1):
            text = self._extract_entry_text(entry)
            if text:
                lines.append(f"{idx}. {text}")

        if not lines:
            return None
        return "\n".join(lines)

    def _extract_entry_text(self, entry: Any) -> str:
        """從不同型別的記憶條目抽取文字內容。"""
        if isinstance(entry, str):
            return entry.strip()

        if isinstance(entry, dict):
            for key in ("content", "text", "summary", "memory"):
                value = entry.get(key)
                if isinstance(value, str) and value.strip():
                    return value.strip()
            return ""

        for attr in ("content", "text", "summary", "memory"):
            value = getattr(entry, attr, None)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""
