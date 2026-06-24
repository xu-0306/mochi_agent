"""Conversation compaction 測試。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from mochi.agents.compaction import ConversationCompactor
from mochi.agents.context import ContextManager
from mochi.agents.engine import AgentEngine
from mochi.backends.base import BaseLLMBackend
from mochi.backends.types import GenerationResult, Message, ModelInfo, StreamChunk
from mochi.config.schema import MochiConfig
from mochi.sessions.store import SessionStore


class _CaptureBackend(BaseLLMBackend):
    """可檢查 prompt/messages 的測試後端。"""

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

    async def generate(
        self,
        messages: list[Message],
        tools: list | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.0,
        reasoning_effort: str | None = None,
        stream: bool = False,
    ) -> GenerationResult | AsyncIterator[StreamChunk]:
        del (
            tools,
            temperature,
            max_tokens,
            top_p,
            min_p,
            top_k,
            frequency_penalty,
            presence_penalty,
            repeat_penalty,
            reasoning_effort,
            stream,
        )
        self.calls.append(messages)
        return GenerationResult(content="ok")

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="capture", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


def test_context_manager_compacts_history_into_summary() -> None:
    """ContextManager 應在 prepare 前自動壓縮過長歷史。"""
    manager = ContextManager(
        compactor=ConversationCompactor.from_max_messages(10),
        history_window=10,
    )
    for i in range(12):
        manager.add_message(Message(role="user", content=f"user-{i}"))

    prompt_context = asyncio.run(manager.prepare_prompt_context("next"))

    assert prompt_context.summary is not None
    assert "Current task:" in prompt_context.summary
    assert prompt_context.summary_state is not None
    assert prompt_context.summary_state.current_task.startswith("user-")
    assert len(prompt_context.history) <= 4


@pytest.mark.asyncio
async def test_engine_compaction_does_not_pollute_canonical_restore(tmp_path: Path) -> None:
    """compaction 只影響 prompt，不應影響 canonical message restore。"""
    backend = _CaptureBackend()
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {
                "db_path": str(tmp_path / "memory.db"),
                "max_short_term_messages": 10,
            },
            "learning": {"enabled": False},
            "security": {
                "require_approval_for_exec": False,
                "require_approval_for_file_write": False,
            },
        }
    )
    engine = AgentEngine(config)

    async def fake_load(model_spec: str) -> _CaptureBackend:
        del model_spec
        engine._router._active = backend  # noqa: SLF001
        return backend

    engine._router.load = fake_load  # type: ignore[method-assign]

    for i in range(8):
        _ = [event async for event in engine.chat(f"u{i}", session_id="s1")]

    _ = [event async for event in engine.chat("u8", session_id="s1")]
    assert backend.calls
    system_prompt = backend.calls[-1][0].content
    assert "Conversation summary:" in system_prompt

    await engine.close()

    store = SessionStore(tmp_path / "sessions")
    events = await store.load_session("s1")
    canonical_messages = [event for event in events if event.get("type") == "message"]
    assert canonical_messages
    assert all(
        not str(event.get("content", "")).startswith("Conversation summary:")
        for event in canonical_messages
    )

    restored_backend = _CaptureBackend()
    restored = AgentEngine(config)

    async def restored_load(model_spec: str) -> _CaptureBackend:
        del model_spec
        restored._router._active = restored_backend  # noqa: SLF001
        return restored_backend

    restored._router.load = restored_load  # type: ignore[method-assign]
    _ = [event async for event in restored.chat("follow-up", session_id="s1")]

    restored_messages = restored_backend.calls[-1]
    assert [message.content for message in restored_messages[1:5]] == [
        "u7",
        "ok",
        "u8",
        "ok",
    ]

    await restored.close()
