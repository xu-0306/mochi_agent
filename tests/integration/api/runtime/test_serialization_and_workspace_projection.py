"""Runtime API tests grouped by ownership."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mochi.agents.events import (
    FinalAnswerEvent,
    GoalStateChangedEvent,
    ToolCallCompletedEvent,
    ToolCallCreatedEvent,
)
from mochi.api.routes.chat import _serialize_event
from mochi.api.server import create_app
from mochi.config.schema import MochiConfig
from mochi.sessions.store import SessionStore


@pytest.mark.parametrize(
    (
        "sandbox_mode",
        "expected_policy_decision",
        "expected_sandbox_status",
        "expected_degraded",
    ),
    [
        ("off", "allow_host", "not_enforced", False),
        ("preferred", "prefer_sandbox_backend", "configured_unavailable", True),
        ("required", "reject_backend_unavailable", "configured_unavailable", True),
    ],
)
def test_session_api_protected_workspace_matrix_keeps_policy_and_effective_behavior_distinct(
    tmp_path: Path,
    change_contract_path: tuple[str, str, str | None, str],
    sandbox_mode: str,
    expected_policy_decision: str,
    expected_sandbox_status: str,
    expected_degraded: bool,
) -> None:
    change_mode, expected_file_policy, expected_shadow, expected_change_status = (
        change_contract_path
    )
    sessions_dir = tmp_path / "sessions"
    config = MochiConfig.model_validate(
        {
            "sessions_dir": str(sessions_dir),
            "security": {"change_contract_mode": change_mode},
            "sandbox": {"mode": sandbox_mode},
        }
    )
    app = create_app()
    app.state.config_factory = lambda: config
    app.state.session_store = SessionStore(sessions_dir)

    with TestClient(app) as client:
        created = client.post("/v1/sessions")
        assert created.status_code == 200
        session_id = created.json()["session_id"]
        response = client.get(f"/v1/sessions/{session_id}")

    assert response.status_code == 200
    projection = response.json()["protected_workspace"]
    assert projection["session_id"] == session_id
    assert projection["change_contract"]["mode"] == change_mode
    assert projection["change_contract"]["configured_policy_decision"] == expected_file_policy
    assert projection["change_contract"]["enforcement_active"] is False
    assert projection["change_contract"]["effective_file_behavior"] == "legacy_mutation_allowed"
    assert projection["change_contract"]["effective_undo_behavior"] == "legacy_undo_available"
    assert projection["change_contract"]["shadow_decision"] == expected_shadow
    assert projection["change_contract"]["status"] == expected_change_status
    assert projection["sandbox"]["mode"] == sandbox_mode
    assert projection["sandbox"]["configured_policy_decision"] == expected_policy_decision
    assert projection["sandbox"]["enforcement_active"] is False
    assert projection["sandbox"]["effective_exec_behavior"] == "host_execution_available"
    assert projection["sandbox"]["host_execution_allowed"] is True
    assert projection["sandbox"]["status"] == expected_sandbox_status
    assert projection["sandbox"]["degraded"] is expected_degraded
    assert projection["sandbox"]["backend"] is None
    assert projection["sandbox"]["capabilities"] == {"exec_containment": False}

def test_chat_serializer_preserves_final_answer_metadata() -> None:
    serialized = _serialize_event(
        FinalAnswerEvent(
            content="done",
            finish_reason="length",
            metadata={
                "runtime_category": "truncation",
                "error_type": "output_truncated",
                "recoverability": "partial",
                "truncated": True,
                "recovery_attempts": 1,
            },
        )
    )

    assert serialized["type"] == "final_answer"
    assert serialized["finish_reason"] == "length"
    assert serialized["metadata"]["runtime_category"] == "truncation"
    assert serialized["metadata"]["error_type"] == "output_truncated"
    assert serialized["metadata"]["truncated"] is True

def test_chat_serializer_supports_explicit_tool_and_goal_events() -> None:
    created = _serialize_event(
        ToolCallCreatedEvent(
            call_id="call-1",
            tool_name="write_file",
            arguments={"path": "demo.py"},
            metadata={"compat_event_type": "tool_call_request"},
        ),
        fallback_turn_id="turn-1",
    )
    completed = _serialize_event(
        ToolCallCompletedEvent(
            call_id="call-1",
            tool_name="write_file",
            arguments={"path": "demo.py"},
            result={"status": "ok"},
            metadata={"compat_event_type": "tool_call_result"},
        ),
        fallback_turn_id="turn-1",
    )
    goal_changed = _serialize_event(
        GoalStateChangedEvent(
            goal_id="goal-1",
            previous_status="running",
            status="paused",
            attempt_id="attempt-1",
            agent_run_id="run-1",
            reason="operator emergency stop",
            metadata={"source": "operator_controls"},
        ),
        fallback_turn_id="turn-1",
    )

    assert created == {
        "type": "tool_call_created",
        "call_id": "call-1",
        "tool_name": "write_file",
        "arguments": {"path": "demo.py"},
        "metadata": {"compat_event_type": "tool_call_request"},
        "turn_id": "turn-1",
    }
    assert completed == {
        "type": "tool_call_completed",
        "call_id": "call-1",
        "tool_name": "write_file",
        "arguments": {"path": "demo.py"},
        "result": {"status": "ok"},
        "error": None,
        "metadata": {"compat_event_type": "tool_call_result"},
        "turn_id": "turn-1",
    }
    assert goal_changed == {
        "type": "goal_state_changed",
        "goal_id": "goal-1",
        "previous_status": "running",
        "status": "paused",
        "attempt_id": "attempt-1",
        "agent_run_id": "run-1",
        "reason": "operator emergency stop",
        "metadata": {"source": "operator_controls"},
        "turn_id": "turn-1",
    }
