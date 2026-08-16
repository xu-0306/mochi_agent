"""Durable Chat context lifecycle coverage."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path

from mochi.agents.context_snapshot import ContextLifecycleSnapshot
from mochi.agents.engine import AgentEngine
from mochi.backends.base import BaseLLMBackend
from mochi.backends.types import GenerationResult, Message, ModelInfo, StreamChunk
from mochi.config.schema import MochiConfig
from mochi.sessions.store import SessionStore


class _LifecycleBackend(BaseLLMBackend):
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
        return GenerationResult(content=f"reply-{len(self.calls)}")

    def supports_tool_calling(self) -> bool:
        return False

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="lifecycle", backend_type="test", context_length=4096)

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


def _config(tmp_path: Path) -> MochiConfig:
    return MochiConfig.model_validate(
        {
            "model": "ollama:lifecycle",
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


def _install_backend(engine: AgentEngine, backend: _LifecycleBackend) -> None:
    async def fake_load(model_spec: str) -> _LifecycleBackend:
        del model_spec
        engine._router._active = backend  # noqa: SLF001
        return backend

    engine._router.load = fake_load  # type: ignore[method-assign]


def _snapshots(events: list[dict[str, object]]) -> list[ContextLifecycleSnapshot]:
    return [
        snapshot
        for event in events
        if (snapshot := ContextLifecycleSnapshot.from_event(event)) is not None
    ]


def test_chat_context_lifecycle_persists_after_timeline_and_restores_latest_state(
    tmp_path: Path,
) -> None:
    asyncio.run(_exercise_context_lifecycle(tmp_path))


async def _exercise_context_lifecycle(tmp_path: Path) -> None:
    config = _config(tmp_path)
    session_id = "context-lifecycle"
    engine = AgentEngine(config)
    _install_backend(engine, _LifecycleBackend())

    for index in range(8):
        _ = [
            event
            async for event in engine.chat(
                f"request-{index}",
                session_id=session_id,
                tool_mode="disabled",
            )
        ]

    store = SessionStore(tmp_path / "sessions")
    events = await store.load_session(session_id)
    snapshots = _snapshots(events)
    assert snapshots
    assert len(snapshots) == 16
    assert [snapshot.revision for snapshot in snapshots] == list(
        range(1, len(snapshots) + 1)
    )
    assert any(snapshot.phase == "pre_prompt" for snapshot in snapshots)
    assert any(snapshot.phase == "post_compaction" for snapshot in snapshots)
    assert snapshots[-1].phase == "post_response"
    assert snapshots[-1].compaction_revision > 0
    latest_snapshot = snapshots[-1]

    await engine.close()

    restored_backend = _LifecycleBackend()
    restored = AgentEngine(config)
    _install_backend(restored, restored_backend)
    await restored.initialize()
    restored_context = await restored._get_context(session_id)  # noqa: SLF001
    assert restored_context.snapshot_revision == latest_snapshot.revision
    assert restored_context.compaction_revision == latest_snapshot.compaction_revision
    assert restored_context.summary == latest_snapshot.summary
    assert [message.content for message in restored_context.get_full_history()] == [
        str(message["content"]) for message in latest_snapshot.history
    ]

    _ = [
        event
        async for event in restored.chat(
            "follow-up",
            session_id=session_id,
            tool_mode="disabled",
        )
    ]
    resumed_snapshots = _snapshots(await store.load_session(session_id))
    assert [snapshot.revision for snapshot in resumed_snapshots] == list(
        range(1, len(resumed_snapshots) + 1)
    )
    assert resumed_snapshots[-1].phase == "post_response"
    assert resumed_snapshots[-1].revision > latest_snapshot.revision

    await restored.close()
