"""ContextManager 單元測試。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from mochi.agents.context import ContextManager
from mochi.backends.types import Message
from mochi.memory.conversation import ConversationMemory


class _FakeMemoryStore:
    """測試用長期記憶儲存。"""

    def __init__(self, entries: list[object]) -> None:
        self.entries = entries
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, top_k: int = 5) -> list:
        self.calls.append((query, top_k))
        return self.entries[:top_k]


class _ErrorMemoryStore:
    """測試用失敗記憶儲存。"""

    async def search(self, query: str, top_k: int = 5) -> list:
        raise RuntimeError("boom")


@dataclass
class _MemoryEntry:
    """測試用記憶條目。"""

    content: str


def test_add_message_and_get_recent_history() -> None:
    """add_message() 與 get_recent_history() 應可管理短期對話。"""
    conversation = ConversationMemory(max_messages=10)
    manager = ContextManager(conversation_memory=conversation, history_window=2)

    manager.add_message(Message(role="user", content="u1"))
    manager.add_message(Message(role="assistant", content="a1"))
    manager.add_message(Message(role="user", content="u2"))

    history = manager.get_recent_history()
    assert [m.content for m in history] == ["a1", "u2"]


def test_prepare_prompt_context_with_memory_search() -> None:
    """prepare_prompt_context() 應整合 history 與 long-term memory。"""
    memory_store = _FakeMemoryStore(
        entries=[
            {"content": "記憶 A"},
            _MemoryEntry(content="記憶 B"),
            "記憶 C",
        ]
    )
    manager = ContextManager(
        conversation_memory=ConversationMemory(max_messages=10),
        memory_store=memory_store,
        history_window=5,
        memory_top_k=2,
    )
    manager.add_message(Message(role="user", content="之前問題"))
    manager.add_message(Message(role="assistant", content="之前回答"))

    context = asyncio.run(manager.prepare_prompt_context("新問題"))

    assert [m.content for m in context.history] == ["之前問題", "之前回答"]
    assert context.memory_context == "1. 記憶 A\n2. 記憶 B"
    assert memory_store.calls == [("新問題", 2)]


def test_prepare_prompt_context_without_memory_store() -> None:
    """無 memory_store 時應僅回傳 history。"""
    manager = ContextManager(
        conversation_memory=ConversationMemory(max_messages=10),
        memory_store=None,
        history_window=3,
    )
    manager.add_message(Message(role="user", content="hello"))

    context = asyncio.run(manager.prepare_prompt_context("query"))
    assert [m.content for m in context.history] == ["hello"]
    assert context.memory_context is None


def test_prepare_prompt_context_tolerates_memory_error() -> None:
    """長期記憶檢索失敗時不應中斷流程。"""
    manager = ContextManager(
        conversation_memory=ConversationMemory(max_messages=10),
        memory_store=_ErrorMemoryStore(),
    )
    manager.add_message(Message(role="user", content="x"))

    context = asyncio.run(manager.prepare_prompt_context("query"))
    assert [m.content for m in context.history] == ["x"]
    assert context.memory_context is None
