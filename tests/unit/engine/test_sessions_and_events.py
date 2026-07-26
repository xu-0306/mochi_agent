"""AgentEngine Phase 2 整合測試。"""

from __future__ import annotations

from pathlib import Path

import pytest

from mochi.agents.context import ContextManager
from mochi.agents.engine import AgentEngine
from mochi.agents.events import (
    AssistantTruncatedEvent,
    ErrorEvent,
    FinalAnswerEvent,
    GoalStateChangedEvent,
    ToolCallCompletedEvent,
    ToolCallCreatedEvent,
)
from mochi.backends.types import (
    ResponsesReplayState,
)
from mochi.config.schema import MochiConfig
from mochi.sessions.store import SessionStore
from tests.unit.engine._support import (
    FakeBackend,
)


@pytest.mark.asyncio
async def test_engine_persists_and_restores_session_history(tmp_path: Path) -> None:
    """不同 AgentEngine 實例應可透過 SessionStore 還原歷史。"""
    fake_backend = FakeBackend()
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
            "security": {
                "require_approval_for_exec": False,
                "require_approval_for_file_write": False,
                "command_rules": [{"tokens": ["echo"], "decision": "allow", "match": "prefix"}],
                "max_file_write_size_mb": 1,
            },
        }
    )

    async def fake_load(model_spec: str) -> FakeBackend:
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine = AgentEngine(config)
    engine._router.load = fake_load  # type: ignore[method-assign]

    events = [event async for event in engine.chat("first turn", session_id="s1")]
    assert any(isinstance(event, FinalAnswerEvent) for event in events)
    await engine.close()

    store = SessionStore(tmp_path / "sessions")
    await store.save_event(
        "s1",
        {
            "type": "turn_event",
            "schema_version": 1,
            "turn_id": "turn-1",
            "event_id": "event-1",
            "seq": 1,
            "phase": "thinking",
            "timestamp": "2026-04-30T10:00:00+00:00",
            "payload": {"type": "thinking", "content": "should not enter prompt", "metadata": {}},
        },
    )

    restored_backend = FakeBackend()
    restored = AgentEngine(config)

    async def restored_load(model_spec: str) -> FakeBackend:
        restored._router._active = restored_backend  # noqa: SLF001
        return restored_backend

    restored._router.load = restored_load  # type: ignore[method-assign]
    restored_events = [event async for event in restored.chat("second turn", session_id="s1")]

    assert any(isinstance(event, FinalAnswerEvent) for event in restored_events)
    assert restored_backend.probe_calls == 1
    restored_messages = next(
        call
        for call in reversed(restored_backend.calls)
        if any(message.role == "user" and message.content == "second turn" for message in call)
    )
    assert [message.content for message in restored_messages[1:3]] == [
        "first turn",
        "fake reply",
    ]
    assert all(message.content != "should not enter prompt" for message in restored_messages)

    await restored.close()


@pytest.mark.asyncio
async def test_restore_session_history_preserves_tool_messages_and_responses_replay(tmp_path: Path) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "test-model",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
        }
    )
    engine = AgentEngine(config)
    store = SessionStore(tmp_path / "sessions")

    replay_state = ResponsesReplayState(
        response_id="resp_prev",
        assistant_output_items=[
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "web_search",
                "arguments": '{"query":"Mochi"}',
            }
        ],
        continuity_mode="manual_items",
    )

    await store.save_event(
        "s-replay",
        {
            "type": "message",
            "schema_version": 1,
            "turn_id": "turn-1",
            "role": "assistant",
            "content": "",
            "thinking": "summary text",
            "tool_calls": [
                {
                    "id": "call-1",
                    "name": "web_search",
                    "arguments": {"query": "Mochi"},
                    "index": 0,
                }
            ],
            "responses_replay": replay_state.to_dict(),
            "attachments": [],
            "timestamp": "2026-06-11T10:00:00+00:00",
        },
    )
    await store.save_event(
        "s-replay",
        {
            "type": "message",
            "schema_version": 1,
            "turn_id": "turn-1",
            "role": "tool",
            "content": '{"ok": true}',
            "tool_call_id": "call-1",
            "name": "web_search",
            "attachments": [],
            "timestamp": "2026-06-11T10:00:01+00:00",
        },
    )

    context = ContextManager()
    await engine._restore_session_history("s-replay", context)  # noqa: SLF001
    history = context.get_recent_history()

    assert [message.role for message in history] == ["assistant", "tool"]
    assert history[0].responses_replay is not None
    assert history[0].responses_replay.response_id == "resp_prev"
    assert history[0].thinking == "summary text"
    assert len(history[0].tool_calls) == 1
    assert history[0].tool_calls[0].id == "call-1"
    assert history[1].content == '{"ok": true}'
    assert history[1].tool_call_id == "call-1"
    assert history[1].name == "web_search"

    await engine.close()



def test_turn_event_payload_serializes_assistant_truncated_event(tmp_path: Path) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
        }
    )
    engine = AgentEngine(config)

    phase, payload = engine._turn_event_payload(  # noqa: SLF001
        AssistantTruncatedEvent(
            content="Model output hit the response length limit; requesting continuation.",
            finish_reason="length",
            recovery_attempt=1,
            partial_output_chars=42,
            metadata={
                "runtime_category": "truncation",
                "error_type": "output_truncated",
                "recoverability": "retrying",
            },
        )
    )

    assert phase == "assistant_truncated"
    assert payload["finish_reason"] == "length"
    assert payload["recovery_attempt"] == 1
    assert payload["partial_output_chars"] == 42
    assert payload["metadata"]["error_type"] == "output_truncated"





def test_turn_event_payload_serializes_explicit_tool_call_events(tmp_path: Path) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
        }
    )
    engine = AgentEngine(config)

    created_phase, created_payload = engine._turn_event_payload(  # noqa: SLF001
        ToolCallCreatedEvent(
            call_id="call-1",
            tool_name="write_file",
            arguments={"path": "demo.py"},
            metadata={"compat_event_type": "tool_call_request"},
        )
    )
    completed_phase, completed_payload = engine._turn_event_payload(  # noqa: SLF001
        ToolCallCompletedEvent(
            call_id="call-1",
            tool_name="write_file",
            arguments={"path": "demo.py"},
            result={"status": "ok"},
            error=None,
            metadata={"compat_event_type": "tool_call_result"},
        )
    )

    assert created_phase == "tool_call_created"
    assert created_payload == {
        "call_id": "call-1",
        "tool_name": "write_file",
        "arguments": {"path": "demo.py"},
        "metadata": {"compat_event_type": "tool_call_request"},
    }
    assert completed_phase == "tool_call_completed"
    assert completed_payload == {
        "call_id": "call-1",
        "tool_name": "write_file",
        "arguments": {"path": "demo.py"},
        "result": {"status": "ok"},
        "error": None,
        "metadata": {"compat_event_type": "tool_call_result"},
    }


def test_turn_event_payload_serializes_goal_state_changed_event(tmp_path: Path) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
        }
    )
    engine = AgentEngine(config)

    phase, payload = engine._turn_event_payload(  # noqa: SLF001
        GoalStateChangedEvent(
            goal_id="goal-1",
            previous_status="running",
            status="paused",
            attempt_id="attempt-1",
            agent_run_id="run-1",
            reason="operator emergency stop",
            metadata={"source": "operator_controls"},
        )
    )

    assert phase == "goal_state_changed"
    assert payload == {
        "goal_id": "goal-1",
        "previous_status": "running",
        "status": "paused",
        "attempt_id": "attempt-1",
        "agent_run_id": "run-1",
        "reason": "operator emergency stop",
        "metadata": {"source": "operator_controls"},
    }

def test_turn_event_payload_preserves_error_metadata(tmp_path: Path) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
        }
    )
    engine = AgentEngine(config)

    phase, payload = engine._turn_event_payload(  # noqa: SLF001
        ErrorEvent(
            message="tool turn failed",
            metadata={
                "backend": {
                    "backend_type": "ollama",
                    "tool_turn_reason": "thinking_only",
                }
            },
        )
    )

    assert phase == "error"
    assert payload == {
        "message": "tool turn failed",
        "code": "AGENT_ERROR",
        "metadata": {
            "backend": {
                "backend_type": "ollama",
                "tool_turn_reason": "thinking_only",
            }
        },
    }
