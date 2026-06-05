"""Prompt context assembly and short-term memory management."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from mochi.agents.compaction import (
    CompactionDiagnostics,
    ContextBudget,
    ConversationCompactor,
    ConversationStateSummary,
)
from mochi.backends.types import Message
from mochi.memory.conversation import ConversationMemory

if TYPE_CHECKING:
    from mochi.memory.store import MemoryStore
else:
    class MemoryStore(Protocol):
        """長期記憶 store duck type。"""

        async def search(self, query: str, top_k: int = 5) -> list[Any]:
            """查詢長期記憶。"""


@dataclass
class PromptContext:
    """提供給 prompt builder 的上下文。"""

    history: list[Message]
    summary: str | None = None
    memory_context: str | None = None
    summary_state: ConversationStateSummary | None = None
    compaction_diagnostics: CompactionDiagnostics | None = None


class ContextManager:
    """Manage prompt context assembly for one session."""

    def __init__(
        self,
        conversation_memory: ConversationMemory | None = None,
        memory_store: MemoryStore | None = None,
        compactor: ConversationCompactor | None = None,
        *,
        history_window: int = 20,
        memory_top_k: int = 5,
        max_short_term_tokens: int | None = None,
        reserve_output_tokens: int = 0,
    ) -> None:
        self._conversation = conversation_memory or ConversationMemory()
        self._memory_store = memory_store
        self._compactor = compactor
        self._history_window = history_window
        self._memory_top_k = memory_top_k
        self._summary: str | None = None
        self._summary_state: ConversationStateSummary | None = None
        self._compaction_diagnostics: CompactionDiagnostics | None = None
        self._max_short_term_tokens = max_short_term_tokens
        self._reserve_output_tokens = reserve_output_tokens

    def add_message(self, message: Message) -> None:
        self._conversation.add(message)

    def get_recent_history(self, limit: int | None = None) -> list[Message]:
        n = self._history_window if limit is None else limit
        return self._conversation.get_history(n)

    def clear_history(self) -> None:
        self._conversation.clear()
        self._summary = None
        self._summary_state = None
        self._compaction_diagnostics = None

    @property
    def summary(self) -> str | None:
        return self._summary

    @property
    def summary_state(self) -> ConversationStateSummary | None:
        return self._summary_state

    @property
    def compaction_diagnostics(self) -> CompactionDiagnostics | None:
        return self._compaction_diagnostics

    async def prepare_prompt_context(
        self,
        user_message: str,
        *,
        history_limit: int | None = None,
        memory_top_k: int | None = None,
        reserve_output_tokens: int | None = None,
    ) -> PromptContext:
        self._compact_history_if_needed(
            budget=ContextBudget(
                max_input_tokens=self._max_short_term_tokens,
                reserve_output_tokens=(
                    self._reserve_output_tokens if reserve_output_tokens is None else reserve_output_tokens
                ),
            )
        )
        history = self.get_recent_history(history_limit)
        memory_context = await self._retrieve_memory_context(
            query=user_message,
            top_k=memory_top_k or self._memory_top_k,
        )
        return PromptContext(
            history=history,
            summary=self._summary,
            memory_context=memory_context,
            summary_state=self._summary_state,
            compaction_diagnostics=self._compaction_diagnostics,
        )

    async def preview_prompt_context(
        self,
        user_message: str,
        *,
        history_limit: int | None = None,
        memory_top_k: int | None = None,
    ) -> PromptContext:
        history = self.get_recent_history(history_limit)
        memory_context = await self._retrieve_memory_context(
            query=user_message,
            top_k=memory_top_k or self._memory_top_k,
        )
        return PromptContext(
            history=history,
            summary=self._summary,
            memory_context=memory_context,
            summary_state=self._summary_state,
            compaction_diagnostics=self._compaction_diagnostics,
        )

    def _compact_history_if_needed(self, *, budget: ContextBudget | None = None) -> None:
        if self._compactor is None:
            return

        history = self._conversation.get_history()
        result = self._compactor.compact(
            history,
            previous_summary=self._summary,
            previous_state=self._summary_state,
            budget=budget,
        )
        if result is None:
            return

        self._summary = result.summary
        self._summary_state = result.summary_state
        self._compaction_diagnostics = result.diagnostics
        self._conversation.clear()
        for message in result.retained_history:
            self._conversation.add(message)

    async def _retrieve_memory_context(self, query: str, top_k: int) -> str | None:
        if self._memory_store is None or not query.strip():
            return None

        try:
            entries = await self._memory_store.search(query=query, top_k=top_k)
        except Exception:
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
