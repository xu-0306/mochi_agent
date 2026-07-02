"""Operator endpoint tests for agent runs."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from mochi.agents.multi_agent.orchestrator import MultiAgentRunResult
from mochi.api.server import create_app
from mochi.config.schema import MochiConfig
from mochi.runtime.service import RuntimeService
from mochi.runtime.store import RuntimeStore


def _create_operator_test_client(tmp_path: Path) -> TestClient:
    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )
    return TestClient(app)


def _wait_agent_run_until(
    client: TestClient,
    run_id: str,
    statuses: set[str],
    *,
    timeout_seconds: float = 4.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_payload: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        response = client.get(f"/v1/agent-runs/{run_id}")
        assert response.status_code == 200
        last_payload = response.json()
        if last_payload["status"] in statuses:
            return last_payload
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for statuses {statuses}; last payload={last_payload}")


def _build_recovery_result(*, state: str = "awaiting_resources") -> MultiAgentRunResult:
    checkpoint = {
        "checkpoint_index": 2,
        "stage": "teacher_generation",
        "task_input": "deployment summary",
        "protocol": "teacher_student_distill",
        "candidate_ids": ["teacher"],
        "candidate_count": 1,
        "artifact_keys": ["subagent_health_snapshot", "detached_exec_jobs"],
    }
    recovery_state = {
        "status": state,
        "action": "pause",
        "reason": "budget exhausted",
        "stage": "teacher_generation",
        "checkpoint": checkpoint,
        "unfinished_steps": ["resume from the latest checkpoint"],
        "recommended_resume_conditions": ["restore quota before resuming"],
        "resource_exhaustion_report": {
            "classification": "quota_exhausted",
            "recommended_resume_conditions": ["restore quota before resuming"],
        },
    }
    return MultiAgentRunResult(
        run_id="test-run",
        protocol="teacher_student_distill",
        state=state,
        task_input="deployment summary",
        candidates=[],
        selected_candidate_id=None,
        evaluation={},
        artifacts={
            "final_answer": "",
            "run_guard_policy": {
                "max_wall_clock_sec": 1,
                "on_budget_exhausted": "pause",
            },
            "recovery_checkpoint": checkpoint,
            "partial_summary": {
                "status": state,
                "reason": "budget exhausted",
                "stage": "teacher_generation",
                "checkpoint_index": 2,
                "candidate_count": 1,
                "artifact_keys": ["subagent_health_snapshot", "detached_exec_jobs"],
                "unfinished_steps": ["resume from the latest checkpoint"],
            },
            "resource_exhaustion_report": recovery_state["resource_exhaustion_report"],
            "subagent_health_snapshot": {
                "status": "degraded",
                "degraded": True,
                "degraded_role_ids": ["teacher"],
                "failure_counts": {"teacher": 1},
                "events": [
                    {
                        "type": "degraded",
                        "role_id": "teacher",
                        "stage": "teacher_generation",
                        "reason": "budget exhausted",
                    }
                ],
            },
            "detached_exec_jobs": {
                "items": [
                    {
                        "session_id": "sess-1",
                        "request_id": "req-1",
                        "status": "running",
                        "background": True,
                        "reattach_supported": True,
                        "log_path": str(Path("logs") / "session.log"),
                        "checkpoint_dir": str(Path("logs") / "checkpoints"),
                    }
                ]
            },
        },
        events=[],
        metadata={
            "degraded": True,
            "recovery_state": recovery_state,
        },
    )


def test_finalize_partial_promotes_recoverable_run(tmp_path: Path, monkeypatch: Any) -> None:
    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self, request
        return _build_recovery_result()

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    with _create_operator_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Recoverable run",
                "topic": "deployment summary",
                "run_policy": {
                    "max_wall_clock_sec": 1,
                    "on_budget_exhausted": "pause",
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        awaiting_payload = _wait_agent_run_until(client, run_id, {"awaiting_resources"})
        assert awaiting_payload["recovery_state"]["status"] == "awaiting_resources"

        finalize_response = client.post(f"/v1/agent-runs/{run_id}/finalize-partial")
        assert finalize_response.status_code == 200
        finalized = finalize_response.json()
        assert finalized["status"] == "partial"
        assert finalized["finished_at"] is not None
        assert finalized["recovery_state"]["status"] == "partial"
        assert finalized["recovery_state"]["action"] == "finalize_partial"

        partial_summaries = [
            artifact
            for artifact in finalized["artifacts"]
            if artifact["artifact_type"] == "partial_summary"
        ]
        assert len(partial_summaries) >= 2
        operator_summary = partial_summaries[-1]["metadata"]
        assert operator_summary["operator_action"] == "finalize_partial"
        assert operator_summary["content"]["status"] == "partial"
        assert operator_summary["content"]["detached_exec_job_count"] == 1
        assert any(
            item["artifact_type"] == "detached_exec_jobs"
            for item in operator_summary["content"]["recoverable_artifacts"]
        )
        assert any(
            event["type"] == "run_finalized_partial"
            for event in finalized["events"]
        )


def test_finalize_partial_rejects_ineligible_status(tmp_path: Path) -> None:
    with _create_operator_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Created run",
                "topic": "deployment summary",
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        finalize_response = client.post(f"/v1/agent-runs/{run_id}/finalize-partial")
        assert finalize_response.status_code == 409
        assert "created" in finalize_response.json()["detail"]


def test_agent_run_health_returns_operator_summary(tmp_path: Path, monkeypatch: Any) -> None:
    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self, request
        return _build_recovery_result(state="stalled")

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    with _create_operator_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Health run",
                "topic": "deployment summary",
                "run_policy": {
                    "heartbeat_timeout_sec": 1,
                    "on_subagent_disconnect": "pause",
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200
        _wait_agent_run_until(client, run_id, {"stalled"})

        health_response = client.get(f"/v1/agent-runs/{run_id}/health")
        assert health_response.status_code == 200
        payload = health_response.json()
        assert payload["status"] == "stalled"
        assert payload["recovery_state"]["status"] == "stalled"
        assert payload["recovery_state"]["finalize_partial_ready"] is True
        assert payload["degraded"] is True
        assert payload["subagent_health_snapshot"]["available"] is True
        assert payload["subagent_health_snapshot"]["degraded_role_ids"] == ["teacher"]
        assert payload["detached_exec_jobs"]["available"] is True
        assert payload["detached_exec_jobs"]["total"] == 1
        assert payload["detached_exec_jobs"]["reattachable_count"] == 1
        assert payload["detached_exec_jobs"]["items"][0]["session_id"] == "sess-1"


def test_create_agent_run_persists_project_and_workspace_fields(tmp_path: Path) -> None:
    with _create_operator_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Workflow chat-first run",
                "topic": "deployment summary",
                "project_id": "project-123",
                "workspace_dir": str(tmp_path / "workspace-alpha"),
            },
        )
        assert create_response.status_code == 200
        payload = create_response.json()
        assert payload["project_id"] == "project-123"
        assert payload["workspace_dir"] == str(tmp_path / "workspace-alpha")
        assert payload["summary"]["project_id"] == "project-123"
        assert payload["summary"]["workspace_dir"] == str(tmp_path / "workspace-alpha")
        detail_response = client.get(f"/v1/agent-runs/{payload['run_id']}")
        assert detail_response.status_code == 200
        detail_payload = detail_response.json()
        assert detail_payload["events"][0]["type"] == "run_created"
        assert detail_payload["events"][0]["project_id"] == "project-123"
        assert detail_payload["events"][0]["workspace_dir"] == str(tmp_path / "workspace-alpha")


def test_agent_run_messages_append_operator_and_assistant_events(tmp_path: Path) -> None:
    with _create_operator_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Workflow message run",
                "topic": "deployment summary",
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        message_response = client.post(
            f"/v1/agent-runs/{run_id}/messages",
            json={
                "role": "user",
                "content": "Please focus on the deployment checklist.",
                "project_id": "project-456",
                "workspace_dir": str(tmp_path / "workspace-beta"),
                "metadata": {"channel": "workflow-chat"},
            },
        )
        assert message_response.status_code == 200
        payload = message_response.json()
        assert payload["project_id"] == "project-456"
        assert payload["workspace_dir"] == str(tmp_path / "workspace-beta")
        assert payload["summary"]["task_input"] == "Please focus on the deployment checklist."

        operator_event = payload["events"][-2]
        assert operator_event["type"] == "operator_message"
        assert operator_event["role"] == "operator"
        assert operator_event["source_role"] == "user"
        assert operator_event["content"] == "Please focus on the deployment checklist."
        assert operator_event["project_id"] == "project-456"
        assert operator_event["workspace_dir"] == str(tmp_path / "workspace-beta")
        assert operator_event["metadata"]["channel"] == "workflow-chat"

        assistant_event = payload["events"][-1]
        assert assistant_event["type"] == "assistant_message"
        assert assistant_event["role"] == "assistant"
        assert assistant_event["project_id"] == "project-456"
        assert assistant_event["workspace_dir"] == str(tmp_path / "workspace-beta")
        assert "Please focus on the deployment checklist." in assistant_event["content"]
        assert "Project: project-456." in assistant_event["content"]
        assert f"Workspace: {tmp_path / 'workspace-beta'}." in assistant_event["content"]
        assert "Current run status: created." in assistant_event["content"]
        assert "existing orchestrator boundaries" in assistant_event["content"]
        assert payload["summary"]["guidance_messages"] == [
            "Please focus on the deployment checklist."
        ]


def test_agent_run_subagent_messages_append_targeted_event(tmp_path: Path) -> None:
    with _create_operator_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Subagent message run",
                "topic": "deployment verification",
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]
        runtime_service = client.app.state.runtime_service
        transcript = asyncio.run(
            runtime_service._store.upsert_subagent_transcript(
                subagent_id="subagent-verifier-1",
                parent_type="agent_run",
                parent_id=run_id,
                agent_run_id=run_id,
                role_id="verifier",
                title="Verifier",
                status="running",
            )
        )
        assert transcript["subagent_id"] == "subagent-verifier-1"

        attachment = {
            "name": "claims.md",
            "path": str(tmp_path / "claims.md"),
            "source": "workspace_file",
            "note": "Re-check claim 2 against this note.",
        }
        message_response = client.post(
            f"/v1/agent-runs/{run_id}/subagents/verifier/messages",
            json={
                "role": "user",
                "content": "Please verify claim 2 more strictly.",
                "project_id": "project-subagent",
                "workspace_dir": str(tmp_path / "workspace-subagent"),
                "attachments": [attachment],
                "metadata": {"channel": "subagent-chat"},
            },
        )
        assert message_response.status_code == 200
        payload = message_response.json()
        assert payload["project_id"] == "project-subagent"
        assert payload["workspace_dir"] == str(tmp_path / "workspace-subagent")

        subagent_event = payload["events"][-1]
        assert subagent_event["type"] == "subagent_message"
        assert subagent_event["role"] == "operator"
        assert subagent_event["source_role"] == "user"
        assert subagent_event["target_role_id"] == "verifier"
        assert subagent_event["subagent_id"] == "subagent-verifier-1"
        assert subagent_event["content"] == "Please verify claim 2 more strictly."
        assert subagent_event["project_id"] == "project-subagent"
        assert subagent_event["workspace_dir"] == str(tmp_path / "workspace-subagent")
        assert subagent_event["metadata"]["channel"] == "subagent-chat"
        assert subagent_event["created_context"] == {
            "source": "workflow_subagent_message_api",
            "delivery": "guidance_only",
            "target_role_id": "verifier",
            "subagent_id": "subagent-verifier-1",
        }
        assert subagent_event["attachments"] == [
            {
                "name": "claims.md",
                "path": str(tmp_path / "claims.md"),
                "size": None,
                "content_type": None,
                "source": "workspace_file",
                "line_start": None,
                "line_end": None,
                "quote": None,
                "note": "Re-check claim 2 against this note.",
            }
        ]

        detail_response = client.get(f"/v1/agent-runs/{run_id}")
        assert detail_response.status_code == 200
        detail_payload = detail_response.json()
        assert detail_payload["events"][-1] == subagent_event
        assert detail_payload["summary"]["role_guidance_messages"] == {
            "verifier": ["Please verify claim 2 more strictly."]
        }


def test_agent_run_subagent_transcript_persists_prompt_and_lifecycle_from_live_runtime(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        callback = request.runtime_event_callback
        assert callback is not None

        class _Event:
            def __init__(self, seq: int, event_type: str, payload: dict[str, Any]) -> None:
                self.seq = seq
                self.type = event_type
                self.payload = payload

            def to_dict(self) -> dict[str, Any]:
                return {
                    "seq": self.seq,
                    "type": self.type,
                    "payload": dict(self.payload),
                }

        subagent_id = f"{request.run_id}:primary:autonomous-single-agent"
        await callback(
            _Event(
                1,
                "subagent_prompt",
                {
                    "subagent_id": subagent_id,
                    "role_id": "primary",
                    "title": "Primary agent",
                    "model_id": "gpt-test",
                    "system_prompt": "System instruction",
                    "user_prompt": "User task prompt",
                    "stage": "autonomous_single_agent",
                    "execution_profile": "subagent_readonly",
                },
            )
        )
        await callback(
            _Event(
                2,
                "role_started",
                {
                    "role_id": "primary",
                    "title": "Primary agent",
                    "subagent_id": subagent_id,
                    "status": "thinking",
                    "current_action": "Primary agent is preparing a response.",
                    "stage": "autonomous_single_agent",
                    "model_id": "gpt-test",
                },
            )
        )
        await callback(
            _Event(
                3,
                "role_completed",
                {
                    "role_id": "primary",
                    "title": "Primary agent",
                    "subagent_id": subagent_id,
                    "status": "done",
                    "current_action": "Primary agent produced an output.",
                    "stage": "autonomous_single_agent",
                    "model_id": "gpt-test",
                    "summary": "Completed primary answer.",
                },
            )
        )
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="autonomous_single_agent",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "done"},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    with _create_operator_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "autonomous_single_agent",
                "title": "Live transcript run",
                "topic": "deployment verification",
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200
        _wait_agent_run_until(client, run_id, {"succeeded"})

        list_response = client.get(f"/v1/agent-runs/{run_id}/subagents")
        assert list_response.status_code == 200
        summaries = list_response.json()
        assert len(summaries) == 1
        assert summaries[0]["role_id"] == "primary"
        assert summaries[0]["title"] == "Primary agent"
        assert summaries[0]["model_id"] == "gpt-test"

        detail_response = client.get(
            f"/v1/agent-runs/{run_id}/subagents/{run_id}:primary:autonomous-single-agent"
        )
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["role_id"] == "primary"
        assert detail["title"] == "Primary agent"
        assert detail["model_id"] == "gpt-test"
        assert detail["system_prompt"] == "System instruction"
        assert detail["user_prompt"] == "User task prompt"
        assert [event["type"] for event in detail["events"]] == [
            "subagent_prompt",
            "subagent_started",
            "subagent_completed",
        ]


def test_agent_run_events_endpoint_supports_after_seq_and_limit(tmp_path: Path) -> None:
    with _create_operator_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Events run",
                "topic": "deployment verification",
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]
        runtime_service = client.app.state.runtime_service
        asyncio.run(
            runtime_service._store.append_agent_run_event(
                run_id,
                {"type": "role_started", "role_id": "verifier", "current_action": "Preparing checks."},
            )
        )
        asyncio.run(
            runtime_service._store.append_agent_run_event(
                run_id,
                {"type": "role_progress", "role_id": "verifier", "content": "Inspecting evidence."},
            )
        )

        response = client.get(f"/v1/agent-runs/{run_id}/events?after_seq=1&limit=2")
        assert response.status_code == 200
        payload = response.json()
        assert [item["seq"] for item in payload] == [2, 3]
        assert payload[0]["type"] == "role_started"
        assert payload[1]["type"] == "role_progress"


def test_agent_run_events_stream_returns_sse_frames(tmp_path: Path) -> None:
    with _create_operator_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Events stream run",
                "topic": "deployment verification",
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]
        runtime_service = client.app.state.runtime_service
        asyncio.run(
            runtime_service._store.append_agent_run_event(
                run_id,
                {
                    "type": "role_progress",
                    "role_id": "verifier",
                    "content": "Streaming evidence update.",
                },
            )
        )

        with client.stream("GET", f"/v1/agent-runs/{run_id}/events/stream?after_seq=1") as response:
            assert response.status_code == 200
            data_line = next(
                line for line in response.iter_lines() if line.startswith("data: ")
            )

        payload = json.loads(data_line.removeprefix("data: "))
        assert payload["seq"] == 2
        assert payload["type"] == "role_progress"
        assert payload["role_id"] == "verifier"


def test_agent_run_subagent_transcript_endpoint_returns_detail(tmp_path: Path) -> None:
    with _create_operator_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Transcript run",
                "topic": "deployment verification",
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]
        runtime_service = client.app.state.runtime_service
        asyncio.run(
            runtime_service._store.upsert_subagent_transcript(
                subagent_id="subagent-verifier-2",
                parent_type="agent_run",
                parent_id=run_id,
                agent_run_id=run_id,
                role_id="verifier",
                title="Verifier",
                model_id="gpt-test",
                status="running",
                system_prompt="System prompt",
                user_prompt="User prompt",
                prompt_preview="User prompt",
            )
        )
        asyncio.run(
            runtime_service._store.append_subagent_transcript_event(
                "subagent-verifier-2",
                {
                    "type": "subagent_started",
                    "parent_type": "agent_run",
                    "parent_id": run_id,
                    "subagent_id": "subagent-verifier-2",
                    "role_id": "verifier",
                    "status": "running",
                },
            )
        )
        asyncio.run(
            runtime_service._store.append_subagent_transcript_event(
                "subagent-verifier-2",
                {
                    "type": "subagent_completed",
                    "parent_type": "agent_run",
                    "parent_id": run_id,
                    "subagent_id": "subagent-verifier-2",
                    "role_id": "verifier",
                    "status": "completed",
                    "summary": "Verification complete.",
                },
            )
        )

        list_response = client.get(f"/v1/agent-runs/{run_id}/subagents")
        assert list_response.status_code == 200
        summaries = list_response.json()
        assert len(summaries) == 1
        assert summaries[0]["subagent_id"] == "subagent-verifier-2"
        assert summaries[0]["role_id"] == "verifier"
        assert summaries[0]["event_count"] == 2

        detail_response = client.get(f"/v1/agent-runs/{run_id}/subagents/subagent-verifier-2")
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["subagent_id"] == "subagent-verifier-2"
        assert detail["system_prompt"] == "System prompt"
        assert detail["user_prompt"] == "User prompt"
        assert [event["type"] for event in detail["events"]] == [
            "subagent_started",
            "subagent_completed",
        ]


def test_agent_run_subagent_message_not_found_returns_404(tmp_path: Path) -> None:
    with _create_operator_test_client(tmp_path) as client:
        response = client.post(
            "/v1/agent-runs/missing-run/subagents/verifier/messages",
            json={"content": "Please verify this."},
        )

    assert response.status_code == 404
    assert response.json() == {"detail": "Agent run not found"}
