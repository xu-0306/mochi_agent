"""Runtime API tests grouped by ownership."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi.testclient import TestClient

from mochi.api.server import create_app
from mochi.config.schema import MochiConfig
from mochi.runtime.service import RuntimeService
from mochi.runtime.store import RuntimeStore

from ._support import (
    _CONTROLLED_SMOKE_COMMAND_RULE,
    _AgentRunModelBackedEngine,
    _RuntimeFakeEngine,
    _wait_until,
)


def test_delegated_multi_agent_task_runs_with_explicit_protocol(tmp_path: Path) -> None:
    app = create_app()
    engine = _AgentRunModelBackedEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "sessions_dir": str(tmp_path / "sessions"),
                "security": {
                    "require_approval_for_exec": False,
                    "command_rules": [_CONTROLLED_SMOKE_COMMAND_RULE],
                },
            }
        )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/tasks",
            json={
                "input_message": "Compare two deployment answers and select the stronger one.",
                "task_type": "delegated_multi_agent",
                "workspace_dir": str(tmp_path / "project-workspace"),
                "metadata": {
                    "protocol": "multi_agent_debate",
                    "protocol_config": {"rounds": 2},
                    "execution_policy": {"mode": "disabled"},
                    "selected_models_roles": {
                        "by_role": {
                            "debater_a": "debater-a-model",
                            "debater_b": "debater-b-model",
                            "judge": "judge-model",
                            "verifier": "verifier-model",
                        }
                    },
                },
            },
        )
        assert create_response.status_code == 200
        created = create_response.json()
        assert created["task_type"] == "delegated_multi_agent"

        done_payload = _wait_until(client, created["task_id"], {"succeeded"})
        assert done_payload["task_type"] == "delegated_multi_agent"
        assert "Argument B" in str(done_payload["final_answer"])
        artifact_events = [
            event for event in done_payload["events"] if event.get("type") == "artifact"
        ]
        assert any(
            event.get("payload", {}).get("name") == "debate_state"
            for event in artifact_events
        )

def test_create_delegated_subagent_task_defaults_to_generic_protocol_with_execution_policy(
    tmp_path: Path,
) -> None:
    runtime_service = RuntimeService(
        engine=_RuntimeFakeEngine(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )

    created = asyncio.run(
        runtime_service.create_delegated_subagent_task(
            objective="Summarize the strongest deployment guidance.",
            protocol="teacher_student_distill",
            session_id="chat-session-42",
            suggested_roles=["teacher", "student"],
            suggested_models={"teacher": "teacher-model", "student": "student-model"},
            execution_budget={
                "max_execution_requests": 2,
                "max_commands_per_request": 1,
                "default_timeout_sec": 45,
                "background_allowed": False,
            },
        )
    )
    task = asyncio.run(runtime_service._store.get_task_run(created["task_id"]))

    assert task is not None
    assert created["task_type"] == "delegated_multi_agent"
    assert created["status"] == "queued"
    assert isinstance(created["display_name"], str)
    assert created["display_name"]
    assert created["display_name"] != "teacher"
    assert created["display_name"] != "Teacher"
    assert created["role"] == "teacher"
    assert created["instruction"] == "Summarize the strongest deployment guidance."
    assert created["objective"] == "Summarize the strongest deployment guidance."
    assert created["parent_session_id"] == "chat-session-42"
    display_name = created["display_name"]

    metadata = task["metadata"]
    assert metadata["protocol"] == "teacher_student_distill"
    assert metadata["protocol_config"] == {}
    assert metadata["execution_policy"]["mode"] == "controlled"
    assert metadata["execution_policy"]["max_execution_requests"] == 2
    assert metadata["execution_policy"]["default_timeout_sec"] == 45
    assert metadata["execution_policy"]["background_allowed"] is False
    assert metadata["delegated_subagent"] == {
        "display_name": display_name,
        "role": "teacher",
        "instruction": "Summarize the strongest deployment guidance.",
        "objective": "Summarize the strongest deployment guidance.",
        "parent_session_id": "chat-session-42",
        "status": "queued",
    }

    payload = asyncio.run(runtime_service.get_task(created["task_id"]))
    assert payload is not None
    assert payload["metadata"]["protocol"] == "teacher_student_distill"
    assert payload["metadata"]["selected_models_roles"]["by_role"] == {
        "teacher": "teacher-model",
        "student": "student-model",
    }
    assert payload["delegated_subagent"] == metadata["delegated_subagent"]
    assert payload["events"][0]["type"] == "delegated_subagent_created"
    assert payload["events"][0]["display_name"] == display_name
    assert payload["events"][0]["role"] == "teacher"
    assert payload["events"][0]["instruction"] == "Summarize the strongest deployment guidance."
    assert payload["events"][0]["parent_session_id"] == "chat-session-42"

    asyncio.run(runtime_service._store.update_task_status(created["task_id"], "running"))
    refreshed_task = asyncio.run(runtime_service._store.get_task_run(created["task_id"]))
    assert refreshed_task is not None
    assert refreshed_task["metadata"]["delegated_subagent"]["status"] == "running"

def test_create_delegated_subagent_task_defaults_missing_protocol_to_autonomous_agent(
    tmp_path: Path,
) -> None:
    runtime_service = RuntimeService(
        engine=_RuntimeFakeEngine(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )

    created = asyncio.run(
        runtime_service.create_delegated_subagent_task(
            objective="Continue this as a direct delegated worker.",
            session_id="chat-session-43",
            execution_budget={
                "max_execution_requests": 2,
                "max_commands_per_request": 1,
            },
        )
    )
    task = asyncio.run(runtime_service._store.get_task_run(created["task_id"]))

    assert task is not None
    metadata = task["metadata"]
    assert metadata["protocol"] == "autonomous_single_agent"
    assert metadata["protocol_config"] == {}
    assert metadata["execution_policy"]["mode"] == "controlled"

    payload = asyncio.run(runtime_service.get_task(created["task_id"]))
    assert payload is not None
    assert payload["metadata"]["protocol"] == "autonomous_single_agent"

def test_delegated_task_message_appends_guidance_only_transcript(tmp_path: Path) -> None:
    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/tasks",
            json={
                "input_message": "Summarize the strongest deployment guidance.",
                "session_id": "chat-session-42",
                "task_type": "delegated_multi_agent",
                "metadata": {
                    "delegated_subagent": {
                        "display_name": "Ada",
                        "role": "teacher",
                        "instruction": "Summarize the strongest deployment guidance.",
                        "objective": "Summarize the strongest deployment guidance.",
                        "parent_session_id": "chat-session-42",
                        "status": "queued",
                    }
                },
            },
        )
        assert create_response.status_code == 200
        task_id = create_response.json()["task_id"]

        message_response = client.post(
            f"/v1/tasks/{task_id}/messages",
            json={
                "content": "Please narrow the answer to verified deployment blockers only.",
                "metadata": {"channel": "subagent-pane", "author": "user"},
            },
        )
        assert message_response.status_code == 200
        payload = message_response.json()
        assert payload["task_id"] == task_id

        message_events = [
            event
            for event in payload["events"]
            if event.get("type") == "delegated_subagent_message"
        ]
        assert len(message_events) == 1
        event = message_events[0]
        assert event["content"] == "Please narrow the answer to verified deployment blockers only."
        assert event["task_id"] == task_id
        assert event["display_name"] == "Ada"
        assert event["role"] == "teacher"
        assert event["parent_session_id"] == "chat-session-42"
        assert event["metadata"] == {"channel": "subagent-pane", "author": "user"}
        assert event["created_context"] == {
            "source": "task_message_api",
            "delivery": "guidance_only",
        }
        assert payload["delegated_subagent"]["display_name"] == "Ada"
        assert payload["delegated_subagent"]["role"] == "teacher"

def test_task_message_endpoint_rejects_non_delegated_tasks(tmp_path: Path) -> None:
    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/tasks",
            json={
                "input_message": "run something",
                "session_id": "runtime-s1",
            },
        )
        assert create_response.status_code == 200
        task_id = create_response.json()["task_id"]

        message_response = client.post(
            f"/v1/tasks/{task_id}/messages",
            json={"content": "Follow this up in the subagent pane."},
        )
        assert message_response.status_code == 400
        assert message_response.json()["detail"] == (
            "Task messages are only supported for delegated_multi_agent tasks."
        )

        task_payload = client.get(f"/v1/tasks/{task_id}")
        assert task_payload.status_code == 200
        assert all(
            event.get("type") != "delegated_subagent_message"
            for event in task_payload.json()["events"]
        )
