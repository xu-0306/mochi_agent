"""Goal runtime API tests."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any

from fastapi.testclient import TestClient

from mochi.agents.multi_agent.orchestrator import MultiAgentRunResult
from mochi.api.server import create_app
from mochi.config.schema import MochiConfig
from mochi.runtime.approvals import InMemoryApprovalStore
from mochi.runtime.exec_runtime import ExecRuntime
from mochi.runtime.service import RuntimeService
from mochi.runtime.store import RuntimeStore
from mochi.utils.shell_providers import BaseShellProvider, SubprocessSpec


def _create_goal_test_client(tmp_path: Path) -> TestClient:
    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    runtime_service.set_scheduler_poll_interval(0.05)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )
    return TestClient(app)


def _create_goal_test_app(tmp_path: Path) -> tuple[Any, RuntimeService]:
    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    runtime_service.set_scheduler_poll_interval(0.05)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )
    return app, runtime_service


def _create_goal_exec_test_client(
    *,
    sessions_dir: Path,
    exec_approval_store: InMemoryApprovalStore,
) -> TestClient:
    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(sessions_dir / "runtime.db"),
        exec_approval_store=exec_approval_store,
        exec_runtime=ExecRuntime(
            providers={"test": _GoalApiPythonDirectProvider()},
            default_shell="test",
        ),
    )
    runtime_service.set_scheduler_poll_interval(0.05)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(sessions_dir)}
    )
    return TestClient(app)


class _GoalApiPythonDirectProvider(BaseShellProvider):
    @property
    def canonical_name(self) -> str:
        return "test"

    @property
    def aliases(self) -> tuple[str, ...]:
        return ("test",)

    def build_subprocess_spec(self, command: str, *, tty: bool = False) -> SubprocessSpec:
        del tty
        return SubprocessSpec(executable=sys.executable, args=("-c", command))


def _wait_goal_until(
    client: TestClient,
    goal_id: str,
    statuses: set[str],
    *,
    timeout_seconds: float = 4.0,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    last_payload: dict[str, Any] | None = None
    while time.monotonic() < deadline:
        response = client.get(f"/v1/goals/{goal_id}")
        assert response.status_code == 200
        last_payload = response.json()
        if last_payload["status"] in statuses:
            return last_payload
        time.sleep(0.05)
    raise AssertionError(f"Timed out waiting for goal statuses {statuses}; last payload={last_payload}")


def _set_goal_started_at(db_path: Path, goal_id: str, started_at: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE goals SET started_at=?, updated_at=? WHERE id=?",
            (started_at, started_at, goal_id),
        )
        conn.commit()


def _create_goal_audit_finding(
    runtime_service: RuntimeService,
    *,
    goal_id: str,
    objective: str,
    finding_code: str,
    summary: str,
    details: dict[str, Any] | None = None,
    status: str = "open",
) -> dict[str, Any]:
    asyncio.run(
        runtime_service._store.create_goal(
            goal_id=goal_id,
            objective=objective,
            summary={"phase": "operator_review"},
        )
    )
    return asyncio.run(
        runtime_service._store.upsert_goal_audit_finding(
            goal_id=goal_id,
            finding_code=finding_code,
            summary=summary,
            details=details or {},
            status=status,
        )
    )


def _build_goal_linked_exec_approval_orchestrator(
    *,
    approval_id: str,
    workdir: Path,
    final_answer: str,
) -> Any:
    async def _approval_then_success_run(self: Any, request: Any) -> MultiAgentRunResult:
        approval = self._exec_approval_store.get(approval_id)
        if approval is None or approval.status == "pending":
            if approval is None:
                self._exec_approval_store.create(
                    approval_id=approval_id,
                    command="print('goal restart approval')",
                    shell="test",
                    scope="dangerous_command",
                    reason="Exec command requires approval.",
                    command_payload={
                        "command": "print('goal restart approval')",
                        "shell": "test",
                        "workdir": str(workdir),
                        "env": None,
                        "timeout_sec": 5.0,
                        "background": False,
                        "tty": False,
                        "approval_state": "approved",
                    },
                )
            role_task_snapshot = {
                "roles": {
                    "planner": {"role_id": "planner", "status": "completed"},
                    "executor": {"role_id": "executor", "status": "completed"},
                    "controller": {
                        "role_id": "controller",
                        "status": "waiting_approval",
                        "stage": "controlled_execution_exec:req-1",
                    },
                },
                "tasks": {
                    "controlled_execution_planner": {
                        "task_key": "controlled_execution_planner",
                        "role_id": "planner",
                        "status": "completed",
                        "result_summary": {"planner_output": {"content": "plan"}},
                    },
                    "controlled_execution_executor": {
                        "task_key": "controlled_execution_executor",
                        "role_id": "executor",
                        "status": "completed",
                        "result_summary": {"execution_requests": [{"request_id": "req-1"}]},
                    },
                    "controlled_execution_controller:req-1": {
                        "task_key": "controlled_execution_controller:req-1",
                        "role_id": "controller",
                        "status": "completed",
                        "result_summary": {
                            "controller_decision": {
                                "request_id": "req-1",
                                "status": "approved",
                                "command": "print('goal restart approval')",
                                "shell": "test",
                            }
                        },
                    },
                },
            }
            resume_payload = {
                "version": 1,
                "executor": "continue_from_checkpoint",
                "strategy_default": "continue_from_checkpoint",
                "stage": "controlled_execution_exec:req-1",
                "checkpoint": {
                    "checkpoint_index": 4,
                    "stage": "controlled_execution_controller:req-1",
                },
                "guidance_messages": [],
                "role_guidance_messages": {},
                "metadata_state": {},
                "precomputed_artifacts": {},
                "protocol_artifacts": {},
                "candidates": [],
                "evidence_packets": [],
                "verifications": [],
                "role_task_snapshot": role_task_snapshot,
            }
            return MultiAgentRunResult(
                run_id=request.run_id,
                protocol="controlled_subagent_execution",
                state="awaiting_approval",
                task_input=request.task_input,
                candidates=[],
                selected_candidate_id=None,
                evaluation={},
                artifacts={
                    "final_answer": None,
                    "controlled_execution_runtime": {"approval_pending_count": 1},
                },
                events=[],
                metadata={
                    "approval_state": {
                        "status": "awaiting_approval",
                        "pending_count": 1,
                        "approval_ids": [approval_id],
                        "pending_approvals": [
                            {
                                "approval_id": approval_id,
                                "tool_name": "exec_command",
                                "request_id": "req-1",
                                "task_key": "controlled_execution_exec:req-1",
                                "stage": "controlled_execution_exec:req-1",
                                "role_id": "controller",
                                "source": "controlled_execution",
                            }
                        ],
                    },
                    "recovery_state": {
                        "status": "awaiting_approval",
                        "action": "await_approval",
                        "reason": "Execution approval required",
                        "stage": "controlled_execution_exec:req-1",
                        "checkpoint": {
                            "checkpoint_index": 4,
                            "stage": "controlled_execution_controller:req-1",
                        },
                        "role_task_snapshot": role_task_snapshot,
                        "resume_payload": resume_payload,
                    },
                },
            )
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="controlled_subagent_execution",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": final_answer},
            events=[],
            metadata={},
        )

    return _approval_then_success_run


def test_goal_create_normalizes_prompt_duration_into_runtime_contract(tmp_path: Path) -> None:
    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Please run for 2 hours and collect forum conversations for corpus building.",
                "title": "Prompt Duration Goal",
            },
        )
        assert create_response.status_code == 200
        created = create_response.json()
        goal_id = created["goal_id"]
        assert created["run_policy"]["requested_duration_text"] == "2 hours"
        assert created["run_policy"]["requested_duration_sec"] == 7_200
        assert created["run_policy"]["requested_duration_source"] == "objective"
        assert created["run_policy"]["runtime_mode"] == "fixed_duration"

        get_response = client.get(f"/v1/goals/{goal_id}")
        assert get_response.status_code == 200
        fetched = get_response.json()
        assert fetched["run_policy"] == created["run_policy"]

        list_response = client.get("/v1/goals")
        assert list_response.status_code == 200
        listed = list_response.json()
        assert listed[0]["run_policy"] == created["run_policy"]

        health_response = client.get(f"/v1/goals/{goal_id}/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()
        assert health_payload["run_policy"]["requested_duration_sec"] == 7_200
        assert health_payload["run_policy"]["runtime_mode"] == "fixed_duration"


def test_goal_execution_mode_round_trips_and_single_agent_uses_protocol_fallback(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Single-agent goal completed."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Handle this as a single-agent goal.",
                "title": "Single Agent Goal",
                "execution_mode": "single_agent",
            },
        )
        assert create_response.status_code == 200
        created = create_response.json()
        goal_id = created["goal_id"]
        assert created["execution_mode"] == "single_agent"

        list_response = client.get("/v1/goals")
        assert list_response.status_code == 200
        assert list_response.json()[0]["execution_mode"] == "single_agent"

        get_response = client.get(f"/v1/goals/{goal_id}")
        assert get_response.status_code == 200
        assert get_response.json()["execution_mode"] == "single_agent"

        health_response = client.get(f"/v1/goals/{goal_id}/health")
        assert health_response.status_code == 200
        assert health_response.json()["execution_mode"] == "single_agent"

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200
        assert start_response.json()["execution_mode"] == "single_agent"

        completed_goal = _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)
        assert completed_goal["execution_mode"] == "single_agent"
        linked_run_id = completed_goal["attempts"][0]["agent_run_id"]
        assert linked_run_id is not None

        linked_run_response = client.get(f"/v1/agent-runs/{linked_run_id}")
        assert linked_run_response.status_code == 200
        assert linked_run_response.json()["protocol_id"] == "teacher_student_distill"


def test_goal_create_explicit_runtime_policy_duration_overrides_prompt_duration(
    tmp_path: Path,
) -> None:
    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Please run for 2 hours and collect forum conversations for corpus building.",
                "title": "Explicit Duration Goal",
                "run_policy": {
                    "requested_duration_sec": 1_800,
                    "requested_duration": "30 minutes",
                    "context_handoff_threshold": 85,
                    "max_attempt_retries": "4",
                    "unknown_key": "preserve-me",
                },
            },
        )
        assert create_response.status_code == 200
        created = create_response.json()
        goal_id = created["goal_id"]
        run_policy = created["run_policy"]
        assert run_policy["requested_duration_text"] == "30 minutes"
        assert run_policy["requested_duration_sec"] == 1_800
        assert run_policy["requested_duration_source"] == "run_policy.requested_duration_sec"
        assert run_policy["runtime_mode"] == "fixed_duration"
        assert run_policy["context_handoff_threshold"] == 0.85
        assert run_policy["max_attempt_retries"] == 4
        assert run_policy["unknown_key"] == "preserve-me"

        get_response = client.get(f"/v1/goals/{goal_id}")
        assert get_response.status_code == 200
        fetched = get_response.json()
        assert fetched["run_policy"] == run_policy


def test_goal_start_preserves_normalized_runtime_policy_on_linked_agent_run(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Goal-linked agent run completed."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Please run for 2 hours and collect forum conversations for corpus building.",
                "title": "Bridge Policy Goal",
                "protocol_id": "teacher_student_distill",
                "run_policy": {
                    "requested_duration_sec": 1_800,
                    "requested_duration": "30 minutes",
                    "context_handoff_threshold": 85,
                    "max_attempt_retries": 4,
                    "unknown_key": "preserve-me",
                },
            },
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200

        completed_goal = _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)
        linked_run_id = completed_goal["attempts"][0]["agent_run_id"]
        assert linked_run_id is not None

        linked_run_response = client.get(f"/v1/agent-runs/{linked_run_id}")
        assert linked_run_response.status_code == 200
        linked_run = linked_run_response.json()
        assert linked_run["run_policy"]["requested_duration_text"] == "30 minutes"
        assert linked_run["run_policy"]["requested_duration_sec"] == 1_800
        assert linked_run["run_policy"]["requested_duration_source"] == "run_policy.requested_duration_sec"
        assert linked_run["run_policy"]["runtime_mode"] == "fixed_duration"
        assert linked_run["run_policy"]["context_handoff_threshold"] == 0.85
        assert linked_run["run_policy"]["max_attempt_retries"] == 4
        assert linked_run["run_policy"]["unknown_key"] == "preserve-me"


def test_goal_propagates_linked_recovery_state_and_checkpoint_to_health(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    async def _awaiting_resources_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="teacher_student_distill",
            state="awaiting_resources",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": None},
            events=[],
            metadata={
                "recovery_state": {
                    "status": "awaiting_resources",
                    "action": "pause",
                    "reason": "provider quota exhausted",
                    "stage": "evidence_collection",
                    "checkpoint": {
                        "checkpoint_index": 7,
                        "stage": "evidence_collection",
                        "task_input": request.task_input,
                    },
                    "recommended_resume_conditions": ["resume after provider quota resets"],
                }
            },
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _awaiting_resources_run)

    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Collect web evidence until a provider quota pause occurs.",
                "title": "Recovery State Goal",
                "protocol_id": "teacher_student_distill",
            },
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200

        goal_payload = _wait_goal_until(client, goal_id, {"awaiting_resources"}, timeout_seconds=4.0)
        attempt_summary = goal_payload["attempts"][0]["summary"]
        assert attempt_summary["linked_recovery_state"]["status"] == "awaiting_resources"
        assert attempt_summary["linked_recovery_state"]["action"] == "pause"
        assert attempt_summary["linked_recovery_state"]["checkpoint_index"] == 7
        assert attempt_summary["linked_recovery_state"]["checkpoint"]["stage"] == "evidence_collection"
        assert goal_payload["summary"]["current_recovery_state"]["status"] == "awaiting_resources"

        health_response = client.get(f"/v1/goals/{goal_id}/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()
        assert health_payload["status"] == "awaiting_resources"
        assert health_payload["current_attempt"]["status"] == "awaiting_resources"
        assert health_payload["recovery_state"]["status"] == "awaiting_resources"
        assert health_payload["recovery_state"]["reason"] == "provider quota exhausted"
        assert health_payload["recovery_state"]["checkpoint_index"] == 7
        assert health_payload["checkpoint"]["checkpoint_index"] == 7
        assert health_payload["checkpoint"]["stage"] == "evidence_collection"


def test_goal_surfaces_waiting_approval_and_checkpoint_policy_in_health(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    checkpoint_captured_at = (datetime.now(UTC) - timedelta(seconds=30)).isoformat()

    async def _approval_wait_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="teacher_student_distill",
            state="stalled",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={
                "final_answer": None,
                "subagent_runtime": {
                    "approval_pending_count": 1,
                    "approval_pending": [
                        {
                            "type": "tool_call_result",
                            "call_id": "call-1",
                            "tool_name": "exec_command",
                            "metadata": {
                                "status": "approval_pending",
                                "requires_approval": True,
                                "approval_id": "exec-approval-1",
                            },
                        }
                    ],
                },
                "controlled_execution_runtime": {
                    "approval_pending_count": 1,
                },
            },
            events=[],
            metadata={
                "recovery_state": {
                    "status": "stalled",
                    "action": "resume",
                    "reason": "execution approval required",
                    "stage": "controlled_execution_exec:req-1",
                    "checkpoint": {
                        "checkpoint_index": 3,
                        "stage": "controller_decision",
                        "captured_at": checkpoint_captured_at,
                    },
                }
            },
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _approval_wait_run)

    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Collect datasets until an execution approval is required.",
                "title": "Approval Wait Goal",
                "protocol_id": "teacher_student_distill",
                "run_policy": {
                    "checkpoint_interval_sec": 60,
                    "generation_refresh_interval_sec": 1_800,
                    "context_handoff_threshold": 0.8,
                    "approval_mode": "default",
                },
            },
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200

        goal_payload = _wait_goal_until(client, goal_id, {"waiting_approval"}, timeout_seconds=4.0)
        assert goal_payload["attempts"][0]["status"] == "waiting_approval"
        attempt_summary = goal_payload["attempts"][0]["summary"]
        assert attempt_summary["linked_approval_state"]["status"] == "waiting_approval"
        assert attempt_summary["linked_approval_state"]["pending_count"] == 1
        assert attempt_summary["linked_approval_state"]["approval_ids"] == ["exec-approval-1"]
        assert attempt_summary["linked_approval_state"]["tool_names"] == ["exec_command"]
        assert goal_payload["summary"]["current_approval_state"]["status"] == "waiting_approval"

        health_response = client.get(f"/v1/goals/{goal_id}/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()
        assert health_payload["status"] == "waiting_approval"
        assert health_payload["current_attempt"]["status"] == "waiting_approval"
        assert health_payload["approval_state"]["status"] == "waiting_approval"
        assert health_payload["approval_state"]["pending_count"] == 1
        assert health_payload["approval_state"]["approval_ids"] == ["exec-approval-1"]
        assert "approval_wait_started_at" not in health_payload["approval_state"]
        assert "approval_wait_elapsed_sec" not in health_payload["approval_state"]
        assert "approval_wait_timeout_sec" not in health_payload["approval_state"]
        assert health_payload["checkpoint_policy"]["status"] == "recorded"
        assert health_payload["checkpoint_policy"]["interval_sec"] == 60
        assert health_payload["checkpoint_policy"]["generation_refresh_interval_sec"] == 1_800
        assert health_payload["checkpoint_policy"]["context_handoff_threshold"] == 0.8
        assert health_payload["checkpoint_policy"]["checkpoint_index"] == 3
        assert health_payload["checkpoint_policy"]["stage"] == "controller_decision"
        assert health_payload["linked_agent_run"]["status"] == "stalled"
        assert health_payload["linked_agent_run"]["approval_state"]["pending_count"] == 1
        assert health_payload["linked_agent_run"]["checkpoint_policy"]["status"] == "recorded"
        assert "approval_wait_timeout" not in [
            item["finding_code"] for item in health_payload["open_findings"]
        ]

def test_goal_auto_rejects_exec_approval_when_approval_mode_is_deny(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    approval_id = "exec-approval-goal-auto-deny-1"
    approval_store = InMemoryApprovalStore()
    monkeypatch.setattr(
        "mochi.runtime.service.MultiAgentOrchestrator.run",
        _build_goal_linked_exec_approval_orchestrator(
            approval_id=approval_id,
            workdir=tmp_path,
            final_answer="This should not be reached after auto-reject.",
        ),
    )

    with _create_goal_exec_test_client(
        sessions_dir=tmp_path / "sessions",
        exec_approval_store=approval_store,
    ) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Run unattended and fail closed if execution approval is required.",
                "protocol_id": "controlled_subagent_execution",
                "run_policy": {
                    "approval_mode": "deny",
                },
            },
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200

        failed_goal = _wait_goal_until(client, goal_id, {"failed"}, timeout_seconds=4.0)
        assert failed_goal["status"] == "failed"
        assert failed_goal["attempts"][0]["status"] == "failed"
        linked_run_id = failed_goal["attempts"][0]["agent_run_id"]
        assert linked_run_id is not None

        linked_run_response = client.get(f"/v1/agent-runs/{linked_run_id}")
        assert linked_run_response.status_code == 200
        linked_run = linked_run_response.json()
        assert linked_run["status"] == "failed"
        assert "approval_mode=deny" in str(linked_run["latest_error"])

        health_response = client.get(f"/v1/goals/{goal_id}/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()
        assert health_payload["status"] == "failed"
        assert "approval_mode=deny" in str(health_payload["latest_error"])
        assert health_payload["linked_agent_run"]["status"] == "failed"

        pending_approvals_response = client.get("/v1/approvals?status=pending")
        assert pending_approvals_response.status_code == 200
        pending_approval_ids = [
            item["approval_id"]
            for item in pending_approvals_response.json()
        ]

    approval = approval_store.get(approval_id)
    assert approval is not None
    assert approval.status == "rejected"
    assert approval.reason is not None and "approval_mode=deny" in approval.reason
    assert approval_id not in pending_approval_ids


def test_goal_health_passes_through_current_generation_payload(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    app, runtime_service = _create_goal_test_app(tmp_path)

    async def _fake_get_goal_health(goal_id: str) -> dict[str, Any]:
        assert goal_id == "goal-generation-1"
        return {
            "goal_id": goal_id,
            "status": "running",
            "current_generation": {
                "generation_id": "goal-generation-1-gen-3",
                "generation_index": 3,
                "status": "active",
                "attempt_id": "goal-generation-attempt-1",
                "agent_run_id": "linked-generation-run-1",
                "started_at": "2026-06-23T03:00:00+00:00",
            },
        }

    monkeypatch.setattr(runtime_service, "get_goal_health", _fake_get_goal_health)

    with TestClient(app) as client:
        response = client.get("/v1/goals/goal-generation-1/health")
        assert response.status_code == 200
        payload = response.json()

    assert payload["current_generation"] == {
        "generation_id": "goal-generation-1-gen-3",
        "generation_index": 3,
        "status": "active",
        "attempt_id": "goal-generation-attempt-1",
        "agent_run_id": "linked-generation-run-1",
        "started_at": "2026-06-23T03:00:00+00:00",
    }


def test_goal_health_surfaces_context_handoff_telemetry_and_resolves_report_only_finding(
    tmp_path: Path,
) -> None:
    app, runtime_service = _create_goal_test_app(tmp_path)
    goal_id = "goal-context-handoff-1"
    attempt_id = "goal-context-handoff-attempt-1"
    run_id = "linked-context-handoff-run-1"

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id=goal_id,
            objective="Monitor context handoff pressure for the active linked run.",
            protocol_id="multi_agent_debate",
            run_policy={"context_handoff_threshold": 0.8},
            summary={"phase": "running"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id=attempt_id,
            goal_id=goal_id,
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id=run_id,
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            goal_id,
            "running",
            current_attempt_id=attempt_id,
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id=run_id,
            protocol_id="multi_agent_debate",
            title="Context handoff monitor run",
            topic="context handoff telemetry",
            summary={
                "goal_id": goal_id,
                "goal_attempt_id": attempt_id,
                "objective": "Monitor context handoff pressure for the active linked run.",
                "task_input": "Monitor context handoff pressure for the active linked run.",
            },
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status(run_id, "running"))
    asyncio.run(
        runtime_service._store.create_goal_worker_generation(
            goal_id=goal_id,
            attempt_id=attempt_id,
            agent_run_id=run_id,
            generation_index=1,
            status="running",
            started_at=(datetime.now(UTC) - timedelta(seconds=45)).isoformat(),
        )
    )
    now = datetime.now(UTC).isoformat()
    asyncio.run(
        runtime_service._store.upsert_goal_lease(
            goal_id=goal_id,
            owner_id=runtime_service._runtime_owner_id,
            metadata={"reason": "context_handoff_test"},
            acquired_at=now,
            heartbeat_at=now,
            expires_at=(datetime.now(UTC) + timedelta(minutes=2)).isoformat(),
        )
    )
    asyncio.run(
        runtime_service._store.append_agent_run_artifact(
            run_id,
            artifact_id=f"{run_id}:attempt:{attempt_id}:debate_context_snapshot:high",
            artifact_type="debate_context_snapshot",
            title="Debate Context Snapshot",
            uri=f"agent-run://{run_id}/artifacts/{attempt_id}/debate_context_snapshot/high",
            mime_type="application/json",
            metadata={
                "attempt_id": attempt_id,
                "content": {
                    "protocol": "multi_agent_debate",
                    "snapshots": [
                        {
                            "role_id": "judge",
                            "stage": "evaluation",
                            "usage_ratio": 0.91,
                        }
                    ],
                    "latest": {
                        "role_id": "judge",
                        "stage": "evaluation",
                        "usage_ratio": 0.91,
                    },
                },
            },
        )
    )

    original_is_live = runtime_service._agent_run_job_is_live
    runtime_service._agent_run_job_is_live = lambda candidate_run_id: candidate_run_id == run_id
    try:
        asyncio.run(runtime_service._process_goal_supervision())
    finally:
        runtime_service._agent_run_job_is_live = original_is_live

    with TestClient(app) as client:
        health_before_response = client.get(f"/v1/goals/{goal_id}/health")
        assert health_before_response.status_code == 200
        health_before = health_before_response.json()

    current_generation_before = health_before["current_generation"]
    assert current_generation_before["status"] == "running"
    assert current_generation_before["usage_ratio"] == 0.91
    assert current_generation_before["context_handoff_threshold"] == 0.8
    assert current_generation_before["context_handoff_due"] is True
    assert health_before["recommended_next_action"] == {
        "action": "refresh_worker_generation",
        "summary": (
            "Active worker generation crossed the context handoff threshold and should hand off "
            "to a fresh worker."
        ),
        "blocking": False,
        "finding_code": "context_handoff_due",
        "generation_id": 1,
        "run_id": run_id,
    }

    open_finding_codes_before = [
        item["finding_code"] for item in health_before["open_findings"]
    ]
    assert "context_handoff_due" in open_finding_codes_before

    asyncio.run(
        runtime_service._store.append_agent_run_artifact(
            run_id,
            artifact_id=f"{run_id}:attempt:{attempt_id}:debate_context_snapshot:low",
            artifact_type="debate_context_snapshot",
            title="Debate Context Snapshot",
            uri=f"agent-run://{run_id}/artifacts/{attempt_id}/debate_context_snapshot/low",
            mime_type="application/json",
            metadata={
                "attempt_id": attempt_id,
                "content": {
                    "protocol": "multi_agent_debate",
                    "snapshots": [
                        {
                            "role_id": "judge",
                            "stage": "evaluation",
                            "usage_ratio": 0.42,
                        }
                    ],
                    "latest": {
                        "role_id": "judge",
                        "stage": "evaluation",
                        "usage_ratio": 0.42,
                    },
                },
            },
        )
    )

    original_is_live = runtime_service._agent_run_job_is_live
    runtime_service._agent_run_job_is_live = lambda candidate_run_id: candidate_run_id == run_id
    try:
        asyncio.run(runtime_service._process_goal_supervision())
    finally:
        runtime_service._agent_run_job_is_live = original_is_live

    with TestClient(app) as client:
        health_after_response = client.get(f"/v1/goals/{goal_id}/health")
        assert health_after_response.status_code == 200
        health_after = health_after_response.json()

    current_generation_after = health_after["current_generation"]
    assert current_generation_after["usage_ratio"] == 0.42
    assert current_generation_after["context_handoff_threshold"] == 0.8
    assert current_generation_after["context_handoff_due"] is False
    assert health_after["recommended_next_action"] == {
        "action": "monitor",
        "summary": "Goal is actively progressing and does not currently need operator intervention.",
        "blocking": False,
        "run_id": run_id,
    }

    open_finding_codes_after = [
        item["finding_code"] for item in health_after["open_findings"]
    ]
    assert "context_handoff_due" not in open_finding_codes_after


def test_goal_context_handoff_due_is_generation_scoped_after_rollover(
    tmp_path: Path,
) -> None:
    app, runtime_service = _create_goal_test_app(tmp_path)
    goal_id = "goal-context-handoff-rollover-1"
    attempt_id = "goal-context-handoff-rollover-attempt-1"
    run_id = "linked-context-handoff-rollover-run-1"

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id=goal_id,
            objective="Do not inherit an old context handoff snapshot after a new generation starts.",
            protocol_id="multi_agent_debate",
            run_policy={"context_handoff_threshold": 0.8},
            summary={"phase": "running"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id=attempt_id,
            goal_id=goal_id,
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id=run_id,
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            goal_id,
            "running",
            current_attempt_id=attempt_id,
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id=run_id,
            protocol_id="multi_agent_debate",
            title="Context handoff rollover run",
            topic="generation-scoped context handoff",
            summary={
                "goal_id": goal_id,
                "goal_attempt_id": attempt_id,
                "objective": "Do not inherit an old context handoff snapshot after a new generation starts.",
                "task_input": "Do not inherit an old context handoff snapshot after a new generation starts.",
            },
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status(run_id, "running"))
    asyncio.run(
        runtime_service._store.create_goal_worker_generation(
            goal_id=goal_id,
            attempt_id=attempt_id,
            agent_run_id=run_id,
            generation_index=1,
            status="running",
            started_at=(datetime.now(UTC) - timedelta(minutes=2)).isoformat(),
        )
    )
    now = datetime.now(UTC).isoformat()
    asyncio.run(
        runtime_service._store.upsert_goal_lease(
            goal_id=goal_id,
            owner_id=runtime_service._runtime_owner_id,
            metadata={"reason": "context_handoff_rollover_test"},
            acquired_at=now,
            heartbeat_at=now,
            expires_at=(datetime.now(UTC) + timedelta(minutes=2)).isoformat(),
        )
    )
    asyncio.run(
        runtime_service._store.append_agent_run_artifact(
            run_id,
            artifact_id=f"{run_id}:attempt:{attempt_id}:debate_context_snapshot:gen1-high",
            artifact_type="debate_context_snapshot",
            title="Debate Context Snapshot",
            uri=f"agent-run://{run_id}/artifacts/{attempt_id}/debate_context_snapshot/gen1-high",
            mime_type="application/json",
            metadata={
                "attempt_id": attempt_id,
                "content": {
                    "protocol": "multi_agent_debate",
                    "snapshots": [
                        {
                            "role_id": "judge",
                            "stage": "evaluation",
                            "usage_ratio": 0.92,
                        }
                    ],
                    "latest": {
                        "role_id": "judge",
                        "stage": "evaluation",
                        "usage_ratio": 0.92,
                    },
                },
            },
        )
    )

    original_is_live = runtime_service._agent_run_job_is_live
    runtime_service._agent_run_job_is_live = lambda candidate_run_id: candidate_run_id == run_id
    try:
        asyncio.run(runtime_service._process_goal_supervision())
    finally:
        runtime_service._agent_run_job_is_live = original_is_live

    with TestClient(app) as client:
        health_before_response = client.get(f"/v1/goals/{goal_id}/health")
        assert health_before_response.status_code == 200
        health_before = health_before_response.json()

    assert health_before["current_generation"]["generation_id"] == 1
    assert health_before["current_generation"]["usage_ratio"] == 0.92
    assert health_before["current_generation"]["context_handoff_due"] is True
    assert "context_handoff_due" in [
        item["finding_code"] for item in health_before["open_findings"]
    ]

    time.sleep(0.05)
    asyncio.run(
        runtime_service._store.create_goal_worker_generation(
            goal_id=goal_id,
            attempt_id=attempt_id,
            agent_run_id=run_id,
            generation_index=2,
            status="running",
            parent_generation_id=1,
            rollover_reason="scheduled_refresh",
            started_at=datetime.now(UTC).isoformat(),
        )
    )

    original_is_live = runtime_service._agent_run_job_is_live
    runtime_service._agent_run_job_is_live = lambda candidate_run_id: candidate_run_id == run_id
    try:
        asyncio.run(runtime_service._process_goal_supervision())
    finally:
        runtime_service._agent_run_job_is_live = original_is_live

    with TestClient(app) as client:
        health_after_response = client.get(f"/v1/goals/{goal_id}/health")
        assert health_after_response.status_code == 200
        health_after = health_after_response.json()

    current_generation_after = health_after["current_generation"]
    assert current_generation_after["generation_id"] == 2
    assert current_generation_after["context_handoff_threshold"] == 0.8
    assert current_generation_after.get("usage_ratio") is None
    assert current_generation_after.get("context_handoff_due") is None
    assert "context_handoff_due" not in [
        item["finding_code"] for item in health_after["open_findings"]
    ]


def test_goal_health_surfaces_generation_scoped_live_subagent_runtime_telemetry(
    tmp_path: Path,
) -> None:
    app, runtime_service = _create_goal_test_app(tmp_path)
    goal_id = "goal-live-subagent-runtime-1"
    attempt_id = "goal-live-subagent-runtime-attempt-1"
    run_id = "linked-live-subagent-runtime-run-1"

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id=goal_id,
            objective="Surface live subagent runtime telemetry on current generation health.",
            protocol_id="teacher_student_distill",
            run_policy={
                "generation_refresh_interval_sec": 600,
                "generation_token_refresh_threshold": 30,
            },
            summary={"phase": "running"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id=attempt_id,
            goal_id=goal_id,
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id=run_id,
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            goal_id,
            "running",
            current_attempt_id=attempt_id,
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id=run_id,
            protocol_id="teacher_student_distill",
            title="Live subagent runtime run",
            topic="generation-scoped telemetry",
            summary={
                "goal_id": goal_id,
                "goal_attempt_id": attempt_id,
                "objective": "Surface live subagent runtime telemetry on current generation health.",
                "task_input": "Surface live subagent runtime telemetry on current generation health.",
            },
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status(run_id, "running"))
    asyncio.run(
        runtime_service._store.create_goal_worker_generation(
            goal_id=goal_id,
            attempt_id=attempt_id,
            agent_run_id=run_id,
            generation_index=1,
            status="running",
            started_at=(datetime.now(UTC) - timedelta(minutes=2)).isoformat(),
        )
    )
    asyncio.run(
        runtime_service._store.append_agent_run_artifact(
            run_id,
            artifact_id=f"{run_id}:attempt:{attempt_id}:subagent_runtime:gen1",
            artifact_type="subagent_runtime",
            title="Subagent Runtime Trace",
            uri=f"agent-run://{run_id}/artifacts/{attempt_id}/subagent_runtime/gen1",
            mime_type="application/json",
            metadata={
                "attempt_id": attempt_id,
                "content": {
                    "invocation_count": 2,
                    "completed_invocation_count": 2,
                    "token_tracked_invocation_count": 2,
                    "input_tokens": 20,
                    "output_tokens": 12,
                    "total_tokens": 32,
                    "generation_time_ms": 41.5,
                    "finish_reason_counts": {"stop": 2},
                    "approval_pending": [],
                    "risky_tool_events": [],
                    "invocations": [],
                },
            },
        )
    )

    with TestClient(app) as client:
        health_before_response = client.get(f"/v1/goals/{goal_id}/health")
        assert health_before_response.status_code == 200
        health_before = health_before_response.json()

    current_generation_before = health_before["current_generation"]
    assert current_generation_before["generation_id"] == 1
    assert current_generation_before["subagent_invocation_count"] == 2
    assert current_generation_before["subagent_completed_invocation_count"] == 2
    assert current_generation_before["subagent_token_tracked_invocation_count"] == 2
    assert current_generation_before["observed_input_tokens"] == 20
    assert current_generation_before["observed_output_tokens"] == 12
    assert current_generation_before["observed_total_tokens"] == 32
    assert current_generation_before["observed_generation_time_ms"] == 41.5
    assert current_generation_before["observed_finish_reason_counts"] == {"stop": 2}
    assert current_generation_before["generation_token_refresh_threshold"] == 30
    assert current_generation_before["token_refresh_due"] is True
    assert current_generation_before["token_refresh_over_threshold"] == 2
    assert current_generation_before["last_subagent_runtime_snapshot_at"]

    time.sleep(0.05)
    asyncio.run(
        runtime_service._store.create_goal_worker_generation(
            goal_id=goal_id,
            attempt_id=attempt_id,
            agent_run_id=run_id,
            generation_index=2,
            status="running",
            parent_generation_id=1,
            rollover_reason="scheduled_refresh",
            started_at=datetime.now(UTC).isoformat(),
        )
    )

    with TestClient(app) as client:
        health_after_response = client.get(f"/v1/goals/{goal_id}/health")
        assert health_after_response.status_code == 200
        health_after = health_after_response.json()

    current_generation_after = health_after["current_generation"]
    assert current_generation_after["generation_id"] == 2
    assert current_generation_after["generation_token_refresh_threshold"] == 30
    assert current_generation_after.get("subagent_invocation_count") is None
    assert current_generation_after.get("observed_input_tokens") is None
    assert current_generation_after.get("observed_total_tokens") is None
    assert current_generation_after.get("token_refresh_due") is None
    assert current_generation_after.get("token_refresh_over_threshold") is None
    assert current_generation_after.get("last_subagent_runtime_snapshot_at") is None


def test_goal_supervisor_resumes_stalled_run_as_context_handoff_refresh(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured_requests: list[Any] = []

    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        captured_requests.append(request)
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="multi_agent_debate",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Context-refresh resumed linked run completed."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    runtime_service.set_goal_lease_ttl_seconds(30)
    runtime_service.set_scheduler_poll_interval(0.05)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    goal_id = "goal-context-refresh-stalled-1"
    attempt_id = "goal-context-refresh-stalled-attempt-1"
    run_id = "linked-context-refresh-stalled-run-1"
    checkpoint_captured_at = "2026-06-23T05:00:00+00:00"
    stale_timestamp = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id=goal_id,
            objective="Resume a stalled linked run as a context-pressure refresh when durable handoff exists.",
            protocol_id="multi_agent_debate",
            run_policy={"context_handoff_threshold": 0.8},
            summary={"phase": "running"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id=attempt_id,
            goal_id=goal_id,
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id=run_id,
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            goal_id,
            "running",
            current_attempt_id=attempt_id,
        )
    )
    checkpoint = asyncio.run(
        runtime_service._store.create_goal_checkpoint(
            goal_id=goal_id,
            attempt_id=attempt_id,
            agent_run_id=run_id,
            checkpoint_index=4,
            stage="evaluation",
            source="operator_test",
            payload={
                "checkpoint_index": 4,
                "stage": "evaluation",
                "captured_at": checkpoint_captured_at,
            },
            metadata={"signature": "context-refresh-stalled-checkpoint-1"},
            captured_at=checkpoint_captured_at,
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_memory_snapshot(
            goal_id=goal_id,
            attempt_id=attempt_id,
            checkpoint_id=checkpoint["id"],
            snapshot_kind="compact_recovery_v1",
            snapshot={
                "goal_objective": (
                    "Resume a stalled linked run as a context-pressure refresh when durable handoff exists."
                ),
                "attempt_id": attempt_id,
                "agent_run_id": run_id,
                "protocol_id": "multi_agent_debate",
                "agent_run_status": "stalled",
                "stage": "evaluation",
                "checkpoint_index": 4,
                "unfinished_steps": ["resume the debate from the durable evaluation checkpoint"],
                "captured_at": checkpoint_captured_at,
            },
            metadata={"signature": "context-refresh-stalled-memory-1"},
            captured_at=checkpoint_captured_at,
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id=run_id,
            protocol_id="multi_agent_debate",
            title="Context refresh stalled linked run",
            topic="context handoff refresh",
            summary={
                "goal_id": goal_id,
                "goal_attempt_id": attempt_id,
                "objective": (
                    "Resume a stalled linked run as a context-pressure refresh when durable handoff exists."
                ),
                "task_input": (
                    "Resume a stalled linked run as a context-pressure refresh when durable handoff exists."
                ),
                "recovery_state": {
                    "status": "stalled",
                    "action": "resume",
                    "reason": "Previous worker stopped after context pressure grew too high.",
                    "stage": "evaluation",
                    "checkpoint": {
                        "checkpoint_index": 4,
                        "stage": "evaluation",
                        "captured_at": checkpoint_captured_at,
                    },
                    "resume_payload": {
                        "version": 1,
                        "executor": "continue_from_checkpoint",
                        "strategy_default": "continue_from_checkpoint",
                        "stage": "evaluation",
                        "checkpoint": {
                            "checkpoint_index": 4,
                            "stage": "evaluation",
                            "captured_at": checkpoint_captured_at,
                        },
                        "guidance_messages": [],
                        "metadata_state": {},
                        "precomputed_artifacts": {},
                        "protocol_artifacts": {},
                        "candidates": [],
                        "evidence_packets": [],
                        "verifications": [],
                        "role_task_snapshot": {},
                    },
                },
            },
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status(run_id, "stalled"))
    asyncio.run(
        runtime_service._store.create_goal_worker_generation(
            goal_id=goal_id,
            attempt_id=attempt_id,
            agent_run_id=run_id,
            generation_index=1,
            status="running",
            started_at=(datetime.now(UTC) - timedelta(minutes=3)).isoformat(),
        )
    )
    asyncio.run(
        runtime_service._store.append_agent_run_artifact(
            run_id,
            artifact_id=f"{run_id}:attempt:{attempt_id}:debate_context_snapshot:high-stalled",
            artifact_type="debate_context_snapshot",
            title="Debate Context Snapshot",
            uri=f"agent-run://{run_id}/artifacts/{attempt_id}/debate_context_snapshot/high-stalled",
            mime_type="application/json",
            metadata={
                "attempt_id": attempt_id,
                "content": {
                    "protocol": "multi_agent_debate",
                    "snapshots": [
                        {
                            "role_id": "judge",
                            "stage": "evaluation",
                            "usage_ratio": 0.91,
                        }
                    ],
                    "latest": {
                        "role_id": "judge",
                        "stage": "evaluation",
                        "usage_ratio": 0.91,
                    },
                },
            },
        )
    )
    asyncio.run(
        runtime_service._store.upsert_goal_lease(
            goal_id=goal_id,
            owner_id="runtime-stale-owner",
            metadata={"reason": "old_runtime"},
            acquired_at=stale_timestamp,
            heartbeat_at=stale_timestamp,
            expires_at=stale_timestamp,
            force_takeover=True,
        )
    )

    with TestClient(app) as client:
        completed_goal = _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)
        linked_run = client.get(f"/v1/agent-runs/{run_id}")
        assert linked_run.status_code == 200
        linked_run_payload = linked_run.json()

    latest_generation = asyncio.run(
        runtime_service._store.get_latest_goal_worker_generation(
            goal_id,
            attempt_id=attempt_id,
        )
    )

    assert completed_goal["attempts"][0]["agent_run_id"] == run_id
    assert completed_goal["attempts"][0]["status"] == "completed"
    assert linked_run_payload["status"] == "succeeded"
    assert linked_run_payload["summary"]["final_answer"] == "Context-refresh resumed linked run completed."
    assert len(captured_requests) == 1
    assert captured_requests[0].metadata["resume_strategy"] == "continue_from_checkpoint"
    assert captured_requests[0].metadata["resume_runtime"]["source"] == "goal_context_handoff_refresh"
    assert latest_generation is not None
    assert latest_generation["generation_index"] == 2
    assert latest_generation["rollover_reason"] == "context_pressure"


def test_goal_supervisor_resumes_stalled_run_as_worker_stall_refresh(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured_requests: list[Any] = []

    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        captured_requests.append(request)
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Worker-stall resumed linked run completed."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    runtime_service.set_goal_lease_ttl_seconds(30)
    runtime_service.set_scheduler_poll_interval(0.05)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    goal_id = "goal-worker-stall-refresh-1"
    attempt_id = "goal-worker-stall-refresh-attempt-1"
    run_id = "linked-worker-stall-refresh-run-1"
    checkpoint_captured_at = "2026-06-23T06:00:00+00:00"
    stale_timestamp = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id=goal_id,
            objective="Resume a stalled linked run as a worker-stall refresh when durable handoff exists.",
            protocol_id="teacher_student_distill",
            summary={"phase": "running"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id=attempt_id,
            goal_id=goal_id,
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id=run_id,
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            goal_id,
            "running",
            current_attempt_id=attempt_id,
        )
    )
    checkpoint = asyncio.run(
        runtime_service._store.create_goal_checkpoint(
            goal_id=goal_id,
            attempt_id=attempt_id,
            agent_run_id=run_id,
            checkpoint_index=3,
            stage="teacher_generation",
            source="operator_test",
            payload={
                "checkpoint_index": 3,
                "stage": "teacher_generation",
                "captured_at": checkpoint_captured_at,
            },
            metadata={"signature": "worker-stall-refresh-checkpoint-1"},
            captured_at=checkpoint_captured_at,
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_memory_snapshot(
            goal_id=goal_id,
            attempt_id=attempt_id,
            checkpoint_id=checkpoint["id"],
            snapshot_kind="compact_recovery_v1",
            snapshot={
                "goal_objective": (
                    "Resume a stalled linked run as a worker-stall refresh when durable handoff exists."
                ),
                "attempt_id": attempt_id,
                "agent_run_id": run_id,
                "protocol_id": "teacher_student_distill",
                "agent_run_status": "stalled",
                "stage": "teacher_generation",
                "checkpoint_index": 3,
                "unfinished_steps": ["resume the teacher generation from the durable checkpoint"],
                "captured_at": checkpoint_captured_at,
            },
            metadata={"signature": "worker-stall-refresh-memory-1"},
            captured_at=checkpoint_captured_at,
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id=run_id,
            protocol_id="teacher_student_distill",
            title="Worker-stall refresh linked run",
            topic="worker stall refresh",
            summary={
                "goal_id": goal_id,
                "goal_attempt_id": attempt_id,
                "objective": (
                    "Resume a stalled linked run as a worker-stall refresh when durable handoff exists."
                ),
                "task_input": (
                    "Resume a stalled linked run as a worker-stall refresh when durable handoff exists."
                ),
                "recovery_state": {
                    "status": "stalled",
                    "action": "resume",
                    "reason": "Previous worker exited before the linked goal finished.",
                    "stage": "teacher_generation",
                    "checkpoint": {
                        "checkpoint_index": 3,
                        "stage": "teacher_generation",
                        "captured_at": checkpoint_captured_at,
                    },
                    "resume_payload": {
                        "version": 1,
                        "executor": "continue_from_checkpoint",
                        "strategy_default": "continue_from_checkpoint",
                        "stage": "teacher_generation",
                        "checkpoint": {
                            "checkpoint_index": 3,
                            "stage": "teacher_generation",
                            "captured_at": checkpoint_captured_at,
                        },
                        "guidance_messages": [],
                        "metadata_state": {},
                        "precomputed_artifacts": {},
                        "protocol_artifacts": {},
                        "candidates": [],
                        "evidence_packets": [],
                        "verifications": [],
                        "role_task_snapshot": {},
                    },
                },
            },
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status(run_id, "stalled"))
    asyncio.run(
        runtime_service._store.create_goal_worker_generation(
            goal_id=goal_id,
            attempt_id=attempt_id,
            agent_run_id=run_id,
            generation_index=1,
            status="running",
            started_at=(datetime.now(UTC) - timedelta(minutes=3)).isoformat(),
        )
    )
    asyncio.run(
        runtime_service._store.upsert_goal_lease(
            goal_id=goal_id,
            owner_id="runtime-stale-owner",
            metadata={"reason": "old_runtime"},
            acquired_at=stale_timestamp,
            heartbeat_at=stale_timestamp,
            expires_at=stale_timestamp,
            force_takeover=True,
        )
    )

    with TestClient(app) as client:
        completed_goal = _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)
        linked_run = client.get(f"/v1/agent-runs/{run_id}")
        assert linked_run.status_code == 200
        linked_run_payload = linked_run.json()

    latest_generation = asyncio.run(
        runtime_service._store.get_latest_goal_worker_generation(
            goal_id,
            attempt_id=attempt_id,
        )
    )

    assert completed_goal["attempts"][0]["agent_run_id"] == run_id
    assert completed_goal["attempts"][0]["status"] == "completed"
    assert linked_run_payload["status"] == "succeeded"
    assert linked_run_payload["summary"]["final_answer"] == "Worker-stall resumed linked run completed."
    assert len(captured_requests) == 1
    assert captured_requests[0].metadata["resume_strategy"] == "continue_from_checkpoint"
    assert captured_requests[0].metadata["resume_runtime"]["source"] == "goal_worker_stall_refresh"
    assert latest_generation is not None
    assert latest_generation["generation_index"] == 2
    assert latest_generation["rollover_reason"] == "worker_stall"


def test_goal_supervisor_resumes_stalled_run_as_context_pressure_restart_refresh(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured_requests: list[Any] = []

    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        captured_requests.append(request)
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="multi_agent_debate",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Context-pressure restart refresh completed."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    runtime_service.set_goal_lease_ttl_seconds(30)
    runtime_service.set_scheduler_poll_interval(0.05)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    goal_id = "goal-context-restart-refresh-1"
    attempt_id = "goal-context-restart-refresh-attempt-1"
    run_id = "linked-context-restart-refresh-run-1"
    checkpoint_captured_at = "2026-06-23T08:00:00+00:00"
    stale_timestamp = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
    unfinished_step = "restart the stalled debate from compact memory instead of replaying the transcript"

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id=goal_id,
            objective="Resume a stalled linked run as a context-pressure restart refresh when durable handoff exists.",
            protocol_id="multi_agent_debate",
            run_policy={"context_handoff_threshold": 0.8},
            summary={"phase": "running"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id=attempt_id,
            goal_id=goal_id,
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id=run_id,
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            goal_id,
            "running",
            current_attempt_id=attempt_id,
        )
    )
    checkpoint = asyncio.run(
        runtime_service._store.create_goal_checkpoint(
            goal_id=goal_id,
            attempt_id=attempt_id,
            agent_run_id=run_id,
            checkpoint_index=4,
            stage="evaluation",
            source="operator_test",
            payload={
                "checkpoint_index": 4,
                "stage": "evaluation",
                "captured_at": checkpoint_captured_at,
            },
            metadata={"signature": "context-restart-refresh-checkpoint-1"},
            captured_at=checkpoint_captured_at,
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_memory_snapshot(
            goal_id=goal_id,
            attempt_id=attempt_id,
            checkpoint_id=checkpoint["id"],
            snapshot_kind="compact_recovery_v1",
            snapshot={
                "goal_objective": (
                    "Resume a stalled linked run as a context-pressure restart refresh when durable handoff exists."
                ),
                "attempt_id": attempt_id,
                "agent_run_id": run_id,
                "protocol_id": "multi_agent_debate",
                "agent_run_status": "stalled",
                "stage": "evaluation",
                "checkpoint_index": 4,
                "unfinished_steps": [unfinished_step],
                "captured_at": checkpoint_captured_at,
            },
            metadata={"signature": "context-restart-refresh-memory-1"},
            captured_at=checkpoint_captured_at,
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id=run_id,
            protocol_id="multi_agent_debate",
            title="Context restart refresh stalled linked run",
            topic="context handoff restart refresh",
            summary={
                "goal_id": goal_id,
                "goal_attempt_id": attempt_id,
                "objective": (
                    "Resume a stalled linked run as a context-pressure restart refresh when durable handoff exists."
                ),
                "task_input": (
                    "Resume a stalled linked run as a context-pressure restart refresh when durable handoff exists."
                ),
                "recovery_state": {
                    "status": "stalled",
                    "action": "resume",
                    "reason": "Previous worker stopped after context pressure grew too high.",
                    "stage": "evaluation",
                    "checkpoint": {
                        "checkpoint_index": 4,
                        "stage": "evaluation",
                        "captured_at": checkpoint_captured_at,
                    },
                },
            },
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status(run_id, "stalled"))
    asyncio.run(
        runtime_service._store.create_goal_worker_generation(
            goal_id=goal_id,
            attempt_id=attempt_id,
            agent_run_id=run_id,
            generation_index=1,
            status="running",
            started_at=(datetime.now(UTC) - timedelta(minutes=3)).isoformat(),
        )
    )
    asyncio.run(
        runtime_service._store.append_agent_run_artifact(
            run_id,
            artifact_id=f"{run_id}:attempt:{attempt_id}:debate_context_snapshot:restart-high-stalled",
            artifact_type="debate_context_snapshot",
            title="Debate Context Snapshot",
            uri=f"agent-run://{run_id}/artifacts/{attempt_id}/debate_context_snapshot/restart-high-stalled",
            mime_type="application/json",
            metadata={
                "attempt_id": attempt_id,
                "content": {
                    "protocol": "multi_agent_debate",
                    "snapshots": [
                        {
                            "role_id": "judge",
                            "stage": "evaluation",
                            "usage_ratio": 0.91,
                        }
                    ],
                    "latest": {
                        "role_id": "judge",
                        "stage": "evaluation",
                        "usage_ratio": 0.91,
                    },
                },
            },
        )
    )
    asyncio.run(
        runtime_service._store.upsert_goal_lease(
            goal_id=goal_id,
            owner_id="runtime-stale-owner",
            metadata={"reason": "old_runtime"},
            acquired_at=stale_timestamp,
            heartbeat_at=stale_timestamp,
            expires_at=stale_timestamp,
            force_takeover=True,
        )
    )

    with TestClient(app) as client:
        completed_goal = _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)
        linked_run = client.get(f"/v1/agent-runs/{run_id}")
        assert linked_run.status_code == 200
        linked_run_payload = linked_run.json()

    latest_generation = asyncio.run(
        runtime_service._store.get_latest_goal_worker_generation(
            goal_id,
            attempt_id=attempt_id,
        )
    )

    assert completed_goal["attempts"][0]["agent_run_id"] == run_id
    assert completed_goal["attempts"][0]["status"] == "completed"
    assert linked_run_payload["status"] == "succeeded"
    assert linked_run_payload["summary"]["final_answer"] == "Context-pressure restart refresh completed."
    assert len(captured_requests) == 1
    assert captured_requests[0].metadata["resume_strategy"] == "restart_attempt"
    assert captured_requests[0].metadata["resume_payload"] == {}
    assert captured_requests[0].metadata["resume_runtime"]["source"] == "goal_context_handoff_refresh"
    assert unfinished_step in "\n".join(captured_requests[0].guidance_messages)
    assert latest_generation is not None
    assert latest_generation["generation_index"] == 2
    assert latest_generation["rollover_reason"] == "context_pressure"


def test_goal_supervisor_resumes_stalled_run_as_worker_stall_restart_refresh(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured_requests: list[Any] = []

    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        captured_requests.append(request)
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Worker-stall restart refresh completed."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    runtime_service.set_goal_lease_ttl_seconds(30)
    runtime_service.set_scheduler_poll_interval(0.05)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    goal_id = "goal-worker-stall-restart-refresh-1"
    attempt_id = "goal-worker-stall-restart-refresh-attempt-1"
    run_id = "linked-worker-stall-restart-refresh-run-1"
    checkpoint_captured_at = "2026-06-23T09:00:00+00:00"
    stale_timestamp = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
    unfinished_step = "restart the teacher generation from compact memory because checkpoint continuation is unavailable"

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id=goal_id,
            objective="Resume a stalled linked run as a worker-stall restart refresh when durable handoff exists.",
            protocol_id="teacher_student_distill",
            summary={"phase": "running"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id=attempt_id,
            goal_id=goal_id,
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id=run_id,
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            goal_id,
            "running",
            current_attempt_id=attempt_id,
        )
    )
    checkpoint = asyncio.run(
        runtime_service._store.create_goal_checkpoint(
            goal_id=goal_id,
            attempt_id=attempt_id,
            agent_run_id=run_id,
            checkpoint_index=3,
            stage="teacher_generation",
            source="operator_test",
            payload={
                "checkpoint_index": 3,
                "stage": "teacher_generation",
                "captured_at": checkpoint_captured_at,
            },
            metadata={"signature": "worker-stall-restart-refresh-checkpoint-1"},
            captured_at=checkpoint_captured_at,
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_memory_snapshot(
            goal_id=goal_id,
            attempt_id=attempt_id,
            checkpoint_id=checkpoint["id"],
            snapshot_kind="compact_recovery_v1",
            snapshot={
                "goal_objective": (
                    "Resume a stalled linked run as a worker-stall restart refresh when durable handoff exists."
                ),
                "attempt_id": attempt_id,
                "agent_run_id": run_id,
                "protocol_id": "teacher_student_distill",
                "agent_run_status": "stalled",
                "stage": "teacher_generation",
                "checkpoint_index": 3,
                "unfinished_steps": [unfinished_step],
                "captured_at": checkpoint_captured_at,
            },
            metadata={"signature": "worker-stall-restart-refresh-memory-1"},
            captured_at=checkpoint_captured_at,
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id=run_id,
            protocol_id="teacher_student_distill",
            title="Worker-stall restart refresh linked run",
            topic="worker stall restart refresh",
            summary={
                "goal_id": goal_id,
                "goal_attempt_id": attempt_id,
                "objective": (
                    "Resume a stalled linked run as a worker-stall restart refresh when durable handoff exists."
                ),
                "task_input": (
                    "Resume a stalled linked run as a worker-stall restart refresh when durable handoff exists."
                ),
                "recovery_state": {
                    "status": "stalled",
                    "action": "resume",
                    "reason": "Previous worker exited before the linked goal finished.",
                    "stage": "teacher_generation",
                    "checkpoint": {
                        "checkpoint_index": 3,
                        "stage": "teacher_generation",
                        "captured_at": checkpoint_captured_at,
                    },
                },
            },
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status(run_id, "stalled"))
    asyncio.run(
        runtime_service._store.create_goal_worker_generation(
            goal_id=goal_id,
            attempt_id=attempt_id,
            agent_run_id=run_id,
            generation_index=1,
            status="running",
            started_at=(datetime.now(UTC) - timedelta(minutes=3)).isoformat(),
        )
    )
    asyncio.run(
        runtime_service._store.upsert_goal_lease(
            goal_id=goal_id,
            owner_id="runtime-stale-owner",
            metadata={"reason": "old_runtime"},
            acquired_at=stale_timestamp,
            heartbeat_at=stale_timestamp,
            expires_at=stale_timestamp,
            force_takeover=True,
        )
    )

    with TestClient(app) as client:
        completed_goal = _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)
        linked_run = client.get(f"/v1/agent-runs/{run_id}")
        assert linked_run.status_code == 200
        linked_run_payload = linked_run.json()

    latest_generation = asyncio.run(
        runtime_service._store.get_latest_goal_worker_generation(
            goal_id,
            attempt_id=attempt_id,
        )
    )

    assert completed_goal["attempts"][0]["agent_run_id"] == run_id
    assert completed_goal["attempts"][0]["status"] == "completed"
    assert linked_run_payload["status"] == "succeeded"
    assert linked_run_payload["summary"]["final_answer"] == "Worker-stall restart refresh completed."
    assert len(captured_requests) == 1
    assert captured_requests[0].metadata["resume_strategy"] == "restart_attempt"
    assert captured_requests[0].metadata["resume_payload"] == {}
    assert captured_requests[0].metadata["resume_runtime"]["source"] == "goal_worker_stall_refresh"
    assert unfinished_step in "\n".join(captured_requests[0].guidance_messages)
    assert latest_generation is not None
    assert latest_generation["generation_index"] == 2
    assert latest_generation["rollover_reason"] == "worker_stall"


def test_goal_health_surfaces_persisted_checkpoint_and_memory_snapshot(
    tmp_path: Path,
) -> None:
    app, runtime_service = _create_goal_test_app(tmp_path)
    checkpoint_captured_at = (datetime.now(UTC) - timedelta(seconds=15)).isoformat()

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-progress-1",
            objective="Persist a durable checkpoint and compact memory snapshot for handoff",
            protocol_id="controlled_subagent_execution",
            summary={"phase": "running"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-progress-attempt-1",
            goal_id="goal-progress-1",
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id="linked-progress-run-1",
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-progress-1",
            "running",
            current_attempt_id="goal-progress-attempt-1",
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id="linked-progress-run-1",
            protocol_id="controlled_subagent_execution",
            title="Progress snapshot run",
            topic="persist progress",
            summary={
                "goal_id": "goal-progress-1",
                "goal_attempt_id": "goal-progress-attempt-1",
                "objective": "Persist a durable checkpoint and compact memory snapshot for handoff",
                "task_input": "Persist a durable checkpoint and compact memory snapshot for handoff",
            },
        )
    )
    asyncio.run(
        runtime_service._store.update_agent_run_metadata(
            "linked-progress-run-1",
            summary={
                "goal_id": "goal-progress-1",
                "goal_attempt_id": "goal-progress-attempt-1",
                "objective": "Persist a durable checkpoint and compact memory snapshot for handoff",
                "task_input": "Persist a durable checkpoint and compact memory snapshot for handoff",
                "selected_candidate_id": "candidate-progress-1",
                "candidate_count": 2,
                "approval_state": {
                    "status": "awaiting_approval",
                    "pending_count": 1,
                    "approval_ids": ["exec-approval-progress-1"],
                    "pending_approvals": [
                        {
                            "approval_id": "exec-approval-progress-1",
                            "tool_name": "exec_command",
                            "request_id": "req-progress-1",
                            "task_key": "controlled_execution_exec:req-progress-1",
                            "stage": "controlled_execution_exec:req-progress-1",
                            "role_id": "controller",
                            "source": "controlled_execution",
                        }
                    ],
                },
                "recovery_state": {
                    "status": "awaiting_approval",
                    "action": "await_approval",
                    "reason": "Execution approval required",
                    "stage": "controlled_execution_exec:req-progress-1",
                    "unfinished_steps": ["resume execution after approval"],
                    "recommended_resume_conditions": ["review the pending command"],
                    "role_task_snapshot": {
                        "roles": {
                            "planner": {
                                "role_id": "planner",
                                "status": "completed",
                                "stage": "planning",
                                "checkpoint_index": 5,
                            },
                            "controller": {
                                "role_id": "controller",
                                "status": "waiting_approval",
                                "stage": "controlled_execution_exec:req-progress-1",
                                "checkpoint_index": 6,
                            },
                        },
                        "resume_plan": {
                            "assignments": {
                                "planner": {
                                    "assigned_model_id": "planner-model",
                                    "assignment_source": "checkpoint_completed",
                                    "resume_action": "skip_completed",
                                },
                                "controller": {
                                    "assigned_model_id": "controller-model",
                                    "assignment_source": "checkpoint_resume",
                                    "resume_action": "resume_pending_step",
                                },
                            },
                            "reassigned_roles": ["controller"],
                            "blocked_roles": [],
                        },
                    },
                    "checkpoint": {
                        "checkpoint_index": 6,
                        "stage": "controller_decision",
                        "captured_at": checkpoint_captured_at,
                    },
                },
                "final_answer": "Partial evidence collected.",
            },
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status("linked-progress-run-1", "awaiting_approval"))
    asyncio.run(
        runtime_service._store.append_agent_run_artifact(
            "linked-progress-run-1",
            artifact_id="artifact-progress-1",
            artifact_type="evidence_bundle",
            title="Evidence Bundle",
            uri="agent-run://linked-progress-run-1/artifacts/evidence",
            metadata={"rows": 12},
        )
    )
    asyncio.run(
        runtime_service._store.append_agent_run_artifact(
            "linked-progress-run-1",
            artifact_id="artifact-progress-brief-1",
            artifact_type="research_brief",
            title="Research Brief",
            uri="agent-run://linked-progress-run-1/artifacts/research_brief",
            metadata={
                "content": {
                    "summary": "Verified deployment can proceed with a bounded approval gate.",
                    "selected_candidate_summary": "Use the staged deployment plan with operator approval.",
                }
            },
        )
    )
    asyncio.run(
        runtime_service._store.append_agent_run_artifact(
            "linked-progress-run-1",
            artifact_id="artifact-progress-claim-map-1",
            artifact_type="claim_evidence_map",
            title="Claim Evidence Map",
            uri="agent-run://linked-progress-run-1/artifacts/claim_evidence_map",
            metadata={
                "content": {
                    "claims": [
                        {
                            "claim": "Deployment can proceed without downtime.",
                            "support_status": "supported",
                            "confidence": 0.94,
                        },
                        {
                            "claim": "Legacy endpoint is already removed.",
                            "support_status": "refuted",
                            "confidence": 0.18,
                        },
                    ]
                }
            },
        )
    )
    asyncio.run(
        runtime_service._store.append_agent_run_artifact(
            "linked-progress-run-1",
            artifact_id="artifact-progress-controller-decisions-1",
            artifact_type="controller_decisions",
            title="Controller Decisions",
            uri="agent-run://linked-progress-run-1/artifacts/controller_decisions",
            metadata={
                "content": {
                    "items": [
                        {
                            "request_id": "req-progress-rejected-1",
                            "status": "rejected",
                            "command": "rm -rf /tmp/unsafe",
                        }
                    ]
                }
            },
        )
    )
    asyncio.run(
        runtime_service._store.append_agent_run_artifact(
            "linked-progress-run-1",
            artifact_id="artifact-progress-collector-shard-1",
            artifact_type="collector_shard_manifest",
            title="Collector Shard forum-thread-1",
            uri="agent-run://linked-progress-run-1/artifacts/collector_shard_1",
            metadata={
                "attempt_id": "goal-progress-attempt-1",
                "content": {
                    "shard_id": "forum-thread-1",
                    "adapter_name": "forum_thread_adapter",
                    "status": "running",
                    "updated_at": checkpoint_captured_at,
                    "source": {
                        "url": "https://forum.example/thread-1",
                        "id": "thread-1",
                    },
                    "progress": {
                        "cursor": "post-24",
                        "items_collected": 24,
                        "items_emitted": 24,
                    },
                },
            },
        )
    )
    asyncio.run(runtime_service._sync_goal_from_agent_run_by_run_id("linked-progress-run-1"))

    with TestClient(app) as client:
        health_response = client.get("/v1/goals/goal-progress-1/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()

    assert health_payload["persisted_checkpoint"]["checkpoint_index"] == 6
    assert health_payload["persisted_checkpoint"]["stage"] == "controller_decision"
    assert health_payload["persisted_checkpoint"]["payload"]["approval_state"]["approval_ids"] == [
        "exec-approval-progress-1"
    ]
    assert (
        health_payload["persisted_checkpoint"]["payload"]["recovery_state"]["selected_candidate_id"]
        == "candidate-progress-1"
    )
    assert health_payload["persisted_checkpoint"]["payload"]["recovery_state"]["candidate_count"] == 2
    role_task_summary = health_payload["persisted_checkpoint"]["payload"]["recovery_state"][
        "role_task_summary"
    ]
    assert role_task_summary["tracked_role_count"] == 2
    assert role_task_summary["reassigned_role_count"] == 1
    assert any(
        item["role_id"] == "controller" and item["resume_action"] == "resume_pending_step"
        for item in role_task_summary["roles"]
    )
    assert health_payload["memory_snapshot"]["snapshot_kind"] == "compact_recovery_v1"
    assert health_payload["memory_snapshot"]["snapshot"]["checkpoint_index"] == 6
    assert health_payload["memory_snapshot"]["snapshot"]["selected_candidate_id"] == (
        "candidate-progress-1"
    )
    assert health_payload["memory_snapshot"]["snapshot"]["candidate_count"] == 2
    assert health_payload["memory_snapshot"]["snapshot"]["pending_approval_ids"] == [
        "exec-approval-progress-1"
    ]
    assert health_payload["memory_snapshot"]["snapshot"]["unfinished_steps"] == [
        "resume execution after approval"
    ]
    assert "review the pending command" not in health_payload["memory_snapshot"]["snapshot"].get(
        "pending_actions",
        [],
    )
    assert "resume execution after approval" in health_payload["memory_snapshot"]["snapshot"][
        "pending_actions"
    ]
    assert "controller: resume_pending_step" in health_payload["memory_snapshot"]["snapshot"][
        "pending_actions"
    ]
    assert "Resolve approval exec-approval-progress-1" in health_payload["memory_snapshot"][
        "snapshot"
    ]["pending_actions"]
    assert (
        "Verified deployment can proceed with a bounded approval gate."
        in health_payload["memory_snapshot"]["snapshot"]["accepted_facts"]
    )
    assert (
        "Deployment can proceed without downtime."
        in health_payload["memory_snapshot"]["snapshot"]["accepted_facts"]
    )
    assert (
        "Legacy endpoint is already removed."
        in health_payload["memory_snapshot"]["snapshot"]["rejected_paths"]
    )
    assert (
        "Rejected command: rm -rf /tmp/unsafe"
        in health_payload["memory_snapshot"]["snapshot"]["rejected_paths"]
    )
    assert health_payload["memory_snapshot"]["snapshot"]["role_task_summary"]["tracked_role_count"] == 2
    assert health_payload["memory_snapshot"]["snapshot"]["important_artifacts"][0]["artifact_type"] == (
        "evidence_bundle"
    )
    assert health_payload["persisted_checkpoint"]["payload"]["promotion"]["mode"] == (
        "internal_checkpoint_plus_downstream_artifacts"
    )
    promoted_types = {
        item["artifact_type"]
        for item in health_payload["persisted_checkpoint"]["payload"]["promotion"][
            "promoted_artifacts"
        ]
    }
    assert "claim_evidence_map" in promoted_types
    assert "collector_shard_manifest" in promoted_types
    assert health_payload["memory_snapshot"]["snapshot"]["checkpoint_promotion_mode"] == (
        "internal_checkpoint_plus_downstream_artifacts"
    )
    promoted_snapshot_types = {
        item["artifact_type"]
        for item in health_payload["memory_snapshot"]["snapshot"]["promoted_artifacts"]
    }
    assert "claim_evidence_map" in promoted_snapshot_types
    assert "collector_shard_manifest" in promoted_snapshot_types
    assert health_payload["collector_state"]["shard_count"] == 1
    assert health_payload["collector_state"]["shards"][0]["cursor"] == "post-24"
    assert health_payload["linked_agent_run"]["collector_state"]["shards"][0]["status"] == "running"
    assert health_payload["persisted_checkpoint"]["payload"]["collector_shard_offsets"][0]["shard_id"] == (
        "forum-thread-1"
    )
    assert (
        health_payload["memory_snapshot"]["snapshot"]["collector_shard_offsets"][0]["source_id"]
        == "thread-1"
    )


def test_goal_linked_run_derives_generation_refresh_partial_budget(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured_requests: list[Any] = []

    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        captured_requests.append(request)
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Completed with a derived generation refresh budget."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    app, runtime_service = _create_goal_test_app(tmp_path)
    with TestClient(app) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Refresh this worker generation on a bounded cadence.",
                "protocol_id": "teacher_student_distill",
                "run_policy": {
                    "generation_refresh_interval_sec": 60,
                    "max_wall_clock_sec": 600,
                },
            },
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200

        _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)

    assert len(captured_requests) == 1
    run_policy = dict(captured_requests[0].run_policy)
    assert run_policy["generation_refresh_interval_sec"] == 60
    assert run_policy["max_wall_clock_sec"] == 60
    assert run_policy["on_budget_exhausted"] == "finalize_partial"
    assert run_policy["goal_generation_refresh_owned"] is True
    assert run_policy["goal_generation_refresh_effective_sec"] == 60


def test_goal_linked_run_derives_checkpoint_cadence_partial_budget(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured_requests: list[Any] = []

    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        captured_requests.append(request)
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Completed with a derived checkpoint cadence budget."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    app, runtime_service = _create_goal_test_app(tmp_path)
    with TestClient(app) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Persist durable progress on a bounded checkpoint cadence.",
                "protocol_id": "teacher_student_distill",
                "run_policy": {
                    "checkpoint_interval_sec": 60,
                    "max_wall_clock_sec": 600,
                },
            },
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200

        _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)

    assert len(captured_requests) == 1
    run_policy = dict(captured_requests[0].run_policy)
    assert run_policy["checkpoint_interval_sec"] == 60
    assert run_policy["max_wall_clock_sec"] == 60
    assert run_policy["on_budget_exhausted"] == "finalize_partial"
    assert run_policy["goal_checkpoint_cadence_owned"] is True
    assert run_policy["goal_checkpoint_cadence_effective_sec"] == 60


def test_goal_linked_run_derives_checkpoint_step_cadence_partial_budget(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured_requests: list[Any] = []

    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        captured_requests.append(request)
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Completed with a checkpoint step cadence budget."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    app, runtime_service = _create_goal_test_app(tmp_path)
    with TestClient(app) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Persist durable progress every two checkpoint stages.",
                "protocol_id": "teacher_student_distill",
                "run_policy": {
                    "checkpoint_interval_steps": 2,
                    "max_wall_clock_sec": 600,
                },
            },
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        get_response = client.get(f"/v1/goals/{goal_id}")
        assert get_response.status_code == 200
        assert get_response.json()["run_policy"]["checkpoint_interval_steps"] == 2

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200

        _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)

    del runtime_service
    assert len(captured_requests) == 1
    run_policy = dict(captured_requests[0].run_policy)
    assert run_policy["checkpoint_interval_steps"] == 2
    assert run_policy["max_wall_clock_sec"] == 600
    assert run_policy["on_budget_exhausted"] == "finalize_partial"
    assert run_policy["goal_checkpoint_cadence_owned"] is True
    assert run_policy["goal_checkpoint_cadence_effective_steps"] == 2
    assert "goal_checkpoint_cadence_effective_sec" not in run_policy


def test_goal_resume_fresh_attempt_includes_compact_memory_snapshot_handoff_guidance(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured_requests: list[Any] = []
    handoff_step = "resume shard-17 extraction without replaying the old transcript"

    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        captured_requests.append(request)
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="controlled_subagent_execution",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Fresh attempt completed from compact handoff guidance."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    app, runtime_service = _create_goal_test_app(tmp_path)
    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-handoff-1",
            objective="Recover progress from the latest durable goal memory snapshot",
            protocol_id="controlled_subagent_execution",
            summary={"phase": "stalled"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-handoff-attempt-1",
            goal_id="goal-handoff-1",
            attempt_index=1,
            status="stalled",
            trigger="manual_start",
            agent_run_id="linked-handoff-run-1",
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-handoff-1",
            "stalled",
            current_attempt_id="goal-handoff-attempt-1",
        )
    )
    checkpoint = asyncio.run(
        runtime_service._store.create_goal_checkpoint(
            goal_id="goal-handoff-1",
            attempt_id="goal-handoff-attempt-1",
            agent_run_id="linked-handoff-run-1",
            checkpoint_index=6,
            stage="controller_decision",
            source="operator_test",
            payload={"checkpoint_index": 6, "stage": "controller_decision"},
            metadata={"signature": "handoff-checkpoint-1"},
            captured_at="2026-06-23T03:05:00+00:00",
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_memory_snapshot(
            goal_id="goal-handoff-1",
            attempt_id="goal-handoff-attempt-1",
            checkpoint_id=checkpoint["id"],
            snapshot_kind="compact_recovery_v1",
            snapshot={
                "goal_objective": "Recover progress from the latest durable goal memory snapshot",
                "attempt_id": "goal-handoff-attempt-1",
                "agent_run_id": "linked-handoff-run-1",
                "protocol_id": "controlled_subagent_execution",
                "agent_run_status": "awaiting_approval",
                "stage": "controller_decision",
                "checkpoint_index": 6,
                "unfinished_steps": [handoff_step],
                "important_artifacts": [
                    {
                        "artifact_type": "evidence_bundle",
                        "title": "Compact Handoff Artifact",
                        "uri": "artifact://goal-handoff-1/evidence-bundle",
                    }
                ],
                "captured_at": "2026-06-23T03:05:01+00:00",
            },
            metadata={"signature": "handoff-memory-1"},
            captured_at="2026-06-23T03:05:01+00:00",
        )
    )

    with TestClient(app) as client:
        resume_response = client.post("/v1/goals/goal-handoff-1/resume")
        assert resume_response.status_code == 200

        completed_goal = _wait_goal_until(client, "goal-handoff-1", {"completed"}, timeout_seconds=4.0)

    assert len(completed_goal["attempts"]) == 2
    assert len(captured_requests) == 1
    assert captured_requests[0].run_id == completed_goal["attempts"][-1]["agent_run_id"]
    assert captured_requests[0].guidance_messages
    assert handoff_step in "\n".join(captured_requests[0].guidance_messages)


def test_goal_resume_reuses_stalled_linked_run_as_manual_refresh(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured_requests: list[Any] = []

    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        captured_requests.append(request)
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Manual refresh reused the stalled linked run."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    runtime_service.set_scheduler_poll_interval(0.05)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    goal_id = "goal-manual-refresh-1"
    attempt_id = "goal-manual-refresh-attempt-1"
    run_id = "linked-manual-refresh-run-1"
    checkpoint_captured_at = "2026-06-23T10:00:00+00:00"

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id=goal_id,
            objective="Resume the current stalled linked run without opening a fresh goal attempt.",
            protocol_id="teacher_student_distill",
            summary={"phase": "stalled"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id=attempt_id,
            goal_id=goal_id,
            attempt_index=1,
            status="stalled",
            trigger="manual_start",
            agent_run_id=run_id,
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            goal_id,
            "stalled",
            current_attempt_id=attempt_id,
        )
    )
    checkpoint = asyncio.run(
        runtime_service._store.create_goal_checkpoint(
            goal_id=goal_id,
            attempt_id=attempt_id,
            agent_run_id=run_id,
            checkpoint_index=3,
            stage="research_context_prepared",
            source="operator_test",
            payload={
                "checkpoint_index": 3,
                "stage": "research_context_prepared",
                "captured_at": checkpoint_captured_at,
            },
            metadata={"signature": "manual-refresh-checkpoint-1"},
            captured_at=checkpoint_captured_at,
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_memory_snapshot(
            goal_id=goal_id,
            attempt_id=attempt_id,
            checkpoint_id=checkpoint["id"],
            snapshot_kind="compact_recovery_v1",
            snapshot={
                "goal_objective": "Resume the current stalled linked run without opening a fresh goal attempt.",
                "attempt_id": attempt_id,
                "agent_run_id": run_id,
                "protocol_id": "teacher_student_distill",
                "agent_run_status": "stalled",
                "stage": "research_context_prepared",
                "checkpoint_index": 3,
                "unfinished_steps": ["resume the linked research worker from the durable handoff"],
                "captured_at": checkpoint_captured_at,
            },
            metadata={"signature": "manual-refresh-memory-1"},
            captured_at=checkpoint_captured_at,
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id=run_id,
            protocol_id="teacher_student_distill",
            title="Manual refresh linked run",
            topic="manual refresh",
            summary={
                "goal_id": goal_id,
                "goal_attempt_id": attempt_id,
                "objective": "Resume the current stalled linked run without opening a fresh goal attempt.",
                "task_input": "Resume the current stalled linked run without opening a fresh goal attempt.",
                "recovery_state": {
                    "status": "stalled",
                    "action": "resume",
                    "reason": "Previous worker exited before the linked goal finished.",
                    "stage": "research_context_prepared",
                    "checkpoint": {
                        "checkpoint_index": 3,
                        "stage": "research_context_prepared",
                        "captured_at": checkpoint_captured_at,
                    },
                    "resume_payload": {
                        "version": 1,
                        "executor": "continue_from_checkpoint",
                        "strategy_default": "continue_from_checkpoint",
                        "stage": "research_context_prepared",
                        "checkpoint": {
                            "checkpoint_index": 3,
                            "stage": "research_context_prepared",
                            "captured_at": checkpoint_captured_at,
                        },
                        "guidance_messages": [],
                        "metadata_state": {},
                        "precomputed_artifacts": {},
                        "protocol_artifacts": {},
                        "candidates": [],
                        "evidence_packets": [],
                        "verifications": [],
                        "role_task_snapshot": {},
                    },
                },
            },
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status(run_id, "stalled"))
    asyncio.run(
        runtime_service._store.create_goal_worker_generation(
            goal_id=goal_id,
            attempt_id=attempt_id,
            agent_run_id=run_id,
            generation_index=1,
            status="running",
            started_at=(datetime.now(UTC) - timedelta(minutes=3)).isoformat(),
        )
    )

    with TestClient(app) as client:
        resume_response = client.post(
            f"/v1/goals/{goal_id}/resume",
            json={"strategy": "continue_from_checkpoint"},
        )
        assert resume_response.status_code == 200

        completed_goal = _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)
        linked_run_response = client.get(f"/v1/agent-runs/{run_id}")
        assert linked_run_response.status_code == 200
        linked_run = linked_run_response.json()

    latest_generation = asyncio.run(
        runtime_service._store.get_latest_goal_worker_generation(
            goal_id,
            attempt_id=attempt_id,
        )
    )

    assert len(completed_goal["attempts"]) == 1
    assert completed_goal["current_attempt_id"] == attempt_id
    assert completed_goal["attempts"][0]["attempt_id"] == attempt_id
    assert completed_goal["attempts"][0]["agent_run_id"] == run_id
    assert linked_run["status"] == "succeeded"
    assert linked_run["summary"]["final_answer"] == "Manual refresh reused the stalled linked run."
    assert len(captured_requests) == 1
    assert captured_requests[0].run_id == run_id
    assert captured_requests[0].metadata["resume_strategy"] == "continue_from_checkpoint"
    assert captured_requests[0].metadata["resume_runtime"]["source"] == "manual_resume"
    assert captured_requests[0].metadata["resume_runtime"]["strategy"] == "continue_from_checkpoint"
    assert latest_generation is not None
    assert latest_generation["generation_index"] == 2
    assert latest_generation["rollover_reason"] == "manual_refresh"


def test_goal_retry_failed_shard_reuses_linked_run_with_retry_guidance(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured_requests: list[Any] = []

    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        captured_requests.append(request)
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Failed shard retry reused the linked run."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    runtime_service.set_scheduler_poll_interval(0.05)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    goal_id = "goal-retry-failed-shard-1"
    attempt_id = "goal-retry-failed-shard-attempt-1"
    run_id = "linked-retry-failed-shard-run-1"
    checkpoint_captured_at = "2026-06-24T03:00:00+00:00"

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id=goal_id,
            objective="Retry the failed collector shard without opening a fresh goal attempt.",
            protocol_id="teacher_student_distill",
            summary={"phase": "collector_recovery"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id=attempt_id,
            goal_id=goal_id,
            attempt_index=1,
            status="stalled",
            trigger="manual_start",
            agent_run_id=run_id,
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            goal_id,
            "stalled",
            current_attempt_id=attempt_id,
        )
    )
    checkpoint = asyncio.run(
        runtime_service._store.create_goal_checkpoint(
            goal_id=goal_id,
            attempt_id=attempt_id,
            agent_run_id=run_id,
            checkpoint_index=4,
            stage="protocol_completed",
            source="operator_test",
            payload={
                "checkpoint_index": 4,
                "stage": "protocol_completed",
                "captured_at": checkpoint_captured_at,
            },
            metadata={"signature": "retry-failed-shard-checkpoint-1"},
            captured_at=checkpoint_captured_at,
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_memory_snapshot(
            goal_id=goal_id,
            attempt_id=attempt_id,
            checkpoint_id=checkpoint["id"],
            snapshot_kind="compact_recovery_v1",
            snapshot={
                "goal_objective": "Retry the failed collector shard without opening a fresh goal attempt.",
                "attempt_id": attempt_id,
                "agent_run_id": run_id,
                "protocol_id": "teacher_student_distill",
                "agent_run_status": "stalled",
                "stage": "protocol_completed",
                "checkpoint_index": 4,
                "unfinished_steps": ["retry failed collector shard discourse-topic-274354"],
                "captured_at": checkpoint_captured_at,
            },
            metadata={"signature": "retry-failed-shard-memory-1"},
            captured_at=checkpoint_captured_at,
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id=run_id,
            protocol_id="teacher_student_distill",
            title="Retry failed shard linked run",
            topic="retry failed shard",
            summary={
                "goal_id": goal_id,
                "goal_attempt_id": attempt_id,
                "objective": "Retry the failed collector shard without opening a fresh goal attempt.",
                "task_input": "Retry the failed collector shard without opening a fresh goal attempt.",
                "recovery_state": {
                    "status": "stalled",
                    "action": "resume",
                    "reason": "Collector shard failed and needs retry.",
                    "stage": "protocol_completed",
                    "checkpoint": {
                        "checkpoint_index": 4,
                        "stage": "protocol_completed",
                        "captured_at": checkpoint_captured_at,
                    },
                    "resume_payload": {
                        "version": 1,
                        "executor": "continue_from_checkpoint",
                        "strategy_default": "continue_from_checkpoint",
                        "stage": "protocol_completed",
                        "checkpoint": {
                            "checkpoint_index": 4,
                            "stage": "protocol_completed",
                            "captured_at": checkpoint_captured_at,
                        },
                        "guidance_messages": [],
                        "metadata_state": {},
                        "precomputed_artifacts": {},
                        "protocol_artifacts": {},
                        "candidates": [],
                        "evidence_packets": [],
                        "verifications": [],
                        "role_task_snapshot": {},
                    },
                },
            },
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status(run_id, "stalled"))
    asyncio.run(
        runtime_service._store.append_agent_run_artifact(
            run_id,
            artifact_id=f"{run_id}:attempt:{attempt_id}:collector-shard:1",
            artifact_type="collector_shard_manifest",
            title="Collector Shard discourse-topic-274354",
            uri=f"agent-run://{run_id}/artifacts/{attempt_id}/collector-shard-1",
            mime_type="application/json",
            metadata={
                "attempt_id": attempt_id,
                "content": {
                    "shards": [
                        {
                            "shard_id": "discourse-topic-274354",
                            "adapter_name": "discourse_topic_adapter",
                            "status": "failed",
                            "source": {
                                "url": "https://forum.example/t/api-examples/274354",
                                "id": "topic:274354",
                            },
                            "progress": {
                                "cursor": "101",
                                "items_collected": 2,
                                "items_emitted": 2,
                            },
                        }
                    ]
                },
            },
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_worker_generation(
            goal_id=goal_id,
            attempt_id=attempt_id,
            agent_run_id=run_id,
            generation_index=1,
            status="running",
            started_at=(datetime.now(UTC) - timedelta(minutes=2)).isoformat(),
        )
    )

    with TestClient(app) as client:
        retry_response = client.post(
            f"/v1/goals/{goal_id}/retry-failed-shard",
            json={"shard_id": "discourse-topic-274354", "strategy": "continue_from_checkpoint"},
        )
        assert retry_response.status_code == 200

        completed_goal = _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)
        linked_run_response = client.get(f"/v1/agent-runs/{run_id}")
        assert linked_run_response.status_code == 200
        linked_run = linked_run_response.json()
        operator_audit_log = client.get("/v1/goals/operator-audit-log?limit=4")
        assert operator_audit_log.status_code == 200

    assert len(completed_goal["attempts"]) == 1
    assert linked_run["status"] == "succeeded"
    assert linked_run["summary"]["final_answer"] == "Failed shard retry reused the linked run."
    assert len(captured_requests) == 1
    assert captured_requests[0].run_id == run_id
    assert captured_requests[0].metadata["resume_strategy"] == "restart_attempt"
    assert "Retry only collector shard discourse-topic-274354." in "\n".join(
        captured_requests[0].guidance_messages
    )
    audit_entries = operator_audit_log.json()
    assert any(entry["event_type"] == "collector_shard_retry" for entry in audit_entries)


def test_goal_resume_reuses_awaiting_resources_linked_run_without_new_attempt(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured_requests: list[Any] = []

    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        captured_requests.append(request)
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Awaiting-resources resume reused the linked run."},
            events=[],
            metadata={},
        )
    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    runtime_service.set_scheduler_poll_interval(0.05)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    goal_id = "goal-awaiting-resources-resume-1"
    attempt_id = "goal-awaiting-resources-resume-attempt-1"
    run_id = "linked-awaiting-resources-resume-run-1"
    checkpoint_captured_at = "2026-06-23T11:00:00+00:00"

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id=goal_id,
            objective="Resume the current awaiting-resources linked run without opening a fresh goal attempt.",
            protocol_id="teacher_student_distill",
            summary={"phase": "awaiting_resources"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id=attempt_id,
            goal_id=goal_id,
            attempt_index=1,
            status="awaiting_resources",
            trigger="manual_start",
            agent_run_id=run_id,
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            goal_id,
            "awaiting_resources",
            current_attempt_id=attempt_id,
        )
    )
    checkpoint = asyncio.run(
        runtime_service._store.create_goal_checkpoint(
            goal_id=goal_id,
            attempt_id=attempt_id,
            agent_run_id=run_id,
            checkpoint_index=2,
            stage="research_context_prepared",
            source="operator_test",
            payload={
                "checkpoint_index": 2,
                "stage": "research_context_prepared",
                "captured_at": checkpoint_captured_at,
            },
            metadata={"signature": "awaiting-resources-resume-checkpoint-1"},
            captured_at=checkpoint_captured_at,
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_memory_snapshot(
            goal_id=goal_id,
            attempt_id=attempt_id,
            checkpoint_id=checkpoint["id"],
            snapshot_kind="compact_recovery_v1",
            snapshot={
                "goal_objective": (
                    "Resume the current awaiting-resources linked run without opening a fresh goal attempt."
                ),
                "attempt_id": attempt_id,
                "agent_run_id": run_id,
                "protocol_id": "teacher_student_distill",
                "agent_run_status": "awaiting_resources",
                "stage": "research_context_prepared",
                "checkpoint_index": 2,
                "unfinished_steps": ["resume once provider quota or capacity is restored"],
                "captured_at": checkpoint_captured_at,
            },
            metadata={"signature": "awaiting-resources-resume-memory-1"},
            captured_at=checkpoint_captured_at,
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id=run_id,
            protocol_id="teacher_student_distill",
            title="Awaiting resources linked run",
            topic="awaiting resources resume",
            summary={
                "goal_id": goal_id,
                "goal_attempt_id": attempt_id,
                "objective": (
                    "Resume the current awaiting-resources linked run without opening a fresh goal attempt."
                ),
                "task_input": (
                    "Resume the current awaiting-resources linked run without opening a fresh goal attempt."
                ),
                "recovery_state": {
                    "status": "awaiting_resources",
                    "action": "pause",
                    "reason": "provider quota exhausted",
                    "stage": "research_context_prepared",
                    "checkpoint": {
                        "checkpoint_index": 2,
                        "stage": "research_context_prepared",
                        "captured_at": checkpoint_captured_at,
                    },
                    "recommended_resume_conditions": ["resume after provider quota resets"],
                    "resume_payload": {
                        "version": 1,
                        "executor": "continue_from_checkpoint",
                        "strategy_default": "continue_from_checkpoint",
                        "stage": "research_context_prepared",
                        "checkpoint": {
                            "checkpoint_index": 2,
                            "stage": "research_context_prepared",
                            "captured_at": checkpoint_captured_at,
                        },
                        "guidance_messages": [],
                        "metadata_state": {},
                        "precomputed_artifacts": {},
                        "protocol_artifacts": {},
                        "candidates": [],
                        "evidence_packets": [],
                        "verifications": [],
                        "role_task_snapshot": {},
                    },
                },
            },
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status(run_id, "awaiting_resources"))
    asyncio.run(
        runtime_service._store.create_goal_worker_generation(
            goal_id=goal_id,
            attempt_id=attempt_id,
            agent_run_id=run_id,
            generation_index=1,
            status="awaiting_resources",
            started_at=(datetime.now(UTC) - timedelta(minutes=3)).isoformat(),
        )
    )

    with TestClient(app) as client:
        resume_response = client.post(f"/v1/goals/{goal_id}/resume")
        assert resume_response.status_code == 200

        completed_goal = _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)
        linked_run_response = client.get(f"/v1/agent-runs/{run_id}")
        assert linked_run_response.status_code == 200
        linked_run = linked_run_response.json()

    latest_generation = asyncio.run(
        runtime_service._store.get_latest_goal_worker_generation(
            goal_id,
            attempt_id=attempt_id,
        )
    )

    assert len(completed_goal["attempts"]) == 1
    assert completed_goal["current_attempt_id"] == attempt_id
    assert completed_goal["attempts"][0]["attempt_id"] == attempt_id
    assert completed_goal["attempts"][0]["agent_run_id"] == run_id
    assert linked_run["status"] == "succeeded"
    assert linked_run["summary"]["final_answer"] == "Awaiting-resources resume reused the linked run."
    assert len(captured_requests) == 1
    assert captured_requests[0].run_id == run_id
    assert captured_requests[0].metadata["resume_strategy"] == "restart_attempt"
    assert captured_requests[0].metadata["resume_payload"] == {}
    assert captured_requests[0].metadata["resume_runtime"]["source"] == "manual_resume"
    assert captured_requests[0].metadata["resume_runtime"]["strategy"] == "restart_attempt"
    assert latest_generation is not None
    assert latest_generation["generation_index"] == 2
    assert latest_generation["rollover_reason"] == "manual_refresh"


def test_goal_operator_audit_log_can_filter_to_selected_goal(tmp_path: Path) -> None:
    app, runtime_service = _create_goal_test_app(tmp_path)

    asyncio.run(
        runtime_service._store.append_goal_operator_audit_log(
            event_type="estop_update",
            subject_type="goal_operator_controls",
            subject_id="global",
            action="update",
            summary="Updated global operator controls.",
            details={"controls": {"stop_all_goals": True}},
        )
    )
    asyncio.run(
        runtime_service._store.append_goal_operator_audit_log(
            event_type="collector_shard_retry",
            subject_type="goal",
            subject_id="goal-a",
            action="retry_failed_shard",
            summary="Retried collector shard for goal A.",
            details={"goal_id": "goal-a", "shard_id": "topic-1"},
        )
    )
    asyncio.run(
        runtime_service._store.append_goal_operator_audit_log(
            event_type="approval_resolution",
            subject_type="approval",
            subject_id="approval-a",
            action="approve_once",
            summary="Resolved approval for goal A.",
            details={"goal_id": "goal-a", "agent_run_id": "run-a"},
        )
    )
    asyncio.run(
        runtime_service._store.append_goal_operator_audit_log(
            event_type="collector_shard_retry",
            subject_type="goal",
            subject_id="goal-b",
            action="retry_failed_shard",
            summary="Retried collector shard for goal B.",
            details={"goal_id": "goal-b", "shard_id": "topic-2"},
        )
    )

    with TestClient(app) as client:
        response = client.get("/v1/goals/operator-audit-log?goal_id=goal-a&limit=10")
        assert response.status_code == 200
        payload = response.json()

    assert len(payload) == 3
    assert {entry["event_type"] for entry in payload} == {
        "collector_shard_retry",
        "approval_resolution",
        "estop_update",
    }
    assert all(
        entry["subject_id"] != "goal-b"
        and entry["details"].get("goal_id") != "goal-b"
        for entry in payload
    )


def test_goal_resume_reuses_paused_linked_run_without_new_attempt(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured_requests: list[Any] = []

    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        captured_requests.append(request)
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Manual refresh reused the paused linked run."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    runtime_service.set_scheduler_poll_interval(0.05)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    goal_id = "goal-paused-manual-refresh-1"
    attempt_id = "goal-paused-manual-refresh-attempt-1"
    run_id = "linked-paused-manual-refresh-run-1"
    checkpoint_captured_at = "2026-06-23T10:00:00+00:00"

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id=goal_id,
            objective="Resume the current paused linked run without opening a fresh goal attempt.",
            protocol_id="teacher_student_distill",
            summary={"phase": "paused"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id=attempt_id,
            goal_id=goal_id,
            attempt_index=1,
            status="paused",
            trigger="manual_start",
            agent_run_id=run_id,
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            goal_id,
            "paused",
            current_attempt_id=attempt_id,
        )
    )
    checkpoint = asyncio.run(
        runtime_service._store.create_goal_checkpoint(
            goal_id=goal_id,
            attempt_id=attempt_id,
            agent_run_id=run_id,
            checkpoint_index=3,
            stage="research_context_prepared",
            source="operator_test",
            payload={
                "checkpoint_index": 3,
                "stage": "research_context_prepared",
                "captured_at": checkpoint_captured_at,
            },
            metadata={"signature": "paused-manual-refresh-checkpoint-1"},
            captured_at=checkpoint_captured_at,
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_memory_snapshot(
            goal_id=goal_id,
            attempt_id=attempt_id,
            checkpoint_id=checkpoint["id"],
            snapshot_kind="compact_recovery_v1",
            snapshot={
                "goal_objective": "Resume the current paused linked run without opening a fresh goal attempt.",
                "attempt_id": attempt_id,
                "agent_run_id": run_id,
                "protocol_id": "teacher_student_distill",
                "agent_run_status": "paused",
                "stage": "research_context_prepared",
                "checkpoint_index": 3,
                "unfinished_steps": ["resume the linked research worker from the durable handoff"],
                "captured_at": checkpoint_captured_at,
            },
            metadata={"signature": "paused-manual-refresh-memory-1"},
            captured_at=checkpoint_captured_at,
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id=run_id,
            protocol_id="teacher_student_distill",
            title="Paused manual refresh linked run",
            topic="manual refresh",
            summary={
                "goal_id": goal_id,
                "goal_attempt_id": attempt_id,
                "objective": "Resume the current paused linked run without opening a fresh goal attempt.",
                "task_input": "Resume the current paused linked run without opening a fresh goal attempt.",
                "recovery_state": {
                    "status": "paused",
                    "action": "resume",
                    "reason": "Operator paused the linked goal and wants to continue on the same run.",
                    "stage": "research_context_prepared",
                    "checkpoint": {
                        "checkpoint_index": 3,
                        "stage": "research_context_prepared",
                        "captured_at": checkpoint_captured_at,
                    },
                    "resume_payload": {
                        "version": 1,
                        "executor": "continue_from_checkpoint",
                        "strategy_default": "continue_from_checkpoint",
                        "stage": "research_context_prepared",
                        "checkpoint": {
                            "checkpoint_index": 3,
                            "stage": "research_context_prepared",
                            "captured_at": checkpoint_captured_at,
                        },
                        "guidance_messages": [],
                        "metadata_state": {},
                        "precomputed_artifacts": {},
                        "protocol_artifacts": {},
                        "candidates": [],
                        "evidence_packets": [],
                        "verifications": [],
                        "role_task_snapshot": {},
                    },
                },
            },
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status(run_id, "paused"))
    asyncio.run(
        runtime_service._store.create_goal_worker_generation(
            goal_id=goal_id,
            attempt_id=attempt_id,
            agent_run_id=run_id,
            generation_index=1,
            status="paused",
            started_at=(datetime.now(UTC) - timedelta(minutes=3)).isoformat(),
        )
    )

    with TestClient(app) as client:
        resume_response = client.post(f"/v1/goals/{goal_id}/resume")
        assert resume_response.status_code == 200

        completed_goal = _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)
        linked_run_response = client.get(f"/v1/agent-runs/{run_id}")
        assert linked_run_response.status_code == 200
        linked_run = linked_run_response.json()

    latest_generation = asyncio.run(
        runtime_service._store.get_latest_goal_worker_generation(
            goal_id,
            attempt_id=attempt_id,
        )
    )

    assert len(completed_goal["attempts"]) == 1
    assert completed_goal["current_attempt_id"] == attempt_id
    assert completed_goal["attempts"][0]["attempt_id"] == attempt_id
    assert completed_goal["attempts"][0]["agent_run_id"] == run_id
    assert linked_run["status"] == "succeeded"
    assert linked_run["summary"]["final_answer"] == "Manual refresh reused the paused linked run."
    assert len(captured_requests) == 1
    assert captured_requests[0].run_id == run_id
    assert captured_requests[0].metadata["resume_strategy"] == "restart_attempt"
    assert captured_requests[0].metadata["resume_payload"] == {}
    assert captured_requests[0].metadata["resume_runtime"]["source"] == "manual_resume"
    assert captured_requests[0].metadata["resume_runtime"]["strategy"] == "restart_attempt"
    assert latest_generation is not None
    assert latest_generation["generation_index"] == 2
    assert latest_generation["rollover_reason"] == "manual_refresh"


def test_goal_resume_reuses_waiting_approval_linked_run_without_new_attempt(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured_requests: list[Any] = []

    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        captured_requests.append(request)
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="controlled_subagent_execution",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Waiting-approval resume reused the linked run."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    runtime_service.set_scheduler_poll_interval(0.05)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    goal_id = "goal-waiting-approval-resume-1"
    attempt_id = "goal-waiting-approval-resume-attempt-1"
    run_id = "linked-waiting-approval-resume-run-1"
    checkpoint_captured_at = "2026-06-23T12:00:00+00:00"

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id=goal_id,
            objective="Resume the current waiting-approval linked run without opening a fresh goal attempt.",
            protocol_id="controlled_subagent_execution",
            summary={"phase": "waiting_approval"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id=attempt_id,
            goal_id=goal_id,
            attempt_index=1,
            status="waiting_approval",
            trigger="manual_start",
            agent_run_id=run_id,
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            goal_id,
            "waiting_approval",
            current_attempt_id=attempt_id,
        )
    )
    checkpoint = asyncio.run(
        runtime_service._store.create_goal_checkpoint(
            goal_id=goal_id,
            attempt_id=attempt_id,
            agent_run_id=run_id,
            checkpoint_index=4,
            stage="controlled_execution_exec:req-1",
            source="operator_test",
            payload={
                "checkpoint_index": 4,
                "stage": "controlled_execution_exec:req-1",
                "captured_at": checkpoint_captured_at,
            },
            metadata={"signature": "waiting-approval-resume-checkpoint-1"},
            captured_at=checkpoint_captured_at,
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_memory_snapshot(
            goal_id=goal_id,
            attempt_id=attempt_id,
            checkpoint_id=checkpoint["id"],
            snapshot_kind="compact_recovery_v1",
            snapshot={
                "goal_objective": "Resume the current waiting-approval linked run without opening a fresh goal attempt.",
                "attempt_id": attempt_id,
                "agent_run_id": run_id,
                "protocol_id": "controlled_subagent_execution",
                "agent_run_status": "awaiting_approval",
                "stage": "controlled_execution_exec:req-1",
                "checkpoint_index": 4,
                "pending_approval_ids": ["exec-approval-waiting-resume-1"],
                "unfinished_steps": ["resume execution after approval is resolved"],
                "captured_at": checkpoint_captured_at,
            },
            metadata={"signature": "waiting-approval-resume-memory-1"},
            captured_at=checkpoint_captured_at,
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id=run_id,
            protocol_id="controlled_subagent_execution",
            title="Waiting approval linked run",
            topic="waiting approval resume",
            summary={
                "goal_id": goal_id,
                "goal_attempt_id": attempt_id,
                "objective": "Resume the current waiting-approval linked run without opening a fresh goal attempt.",
                "task_input": "Resume the current waiting-approval linked run without opening a fresh goal attempt.",
                "approval_state": {
                    "status": "awaiting_approval",
                    "pending_count": 1,
                    "approval_ids": ["exec-approval-waiting-resume-1"],
                },
                "recovery_state": {
                    "status": "awaiting_approval",
                    "action": "await_approval",
                    "reason": "Execution approval required",
                    "stage": "controlled_execution_exec:req-1",
                    "checkpoint": {
                        "checkpoint_index": 4,
                        "stage": "controlled_execution_exec:req-1",
                        "captured_at": checkpoint_captured_at,
                    },
                    "resume_payload": {
                        "version": 1,
                        "executor": "continue_from_checkpoint",
                        "strategy_default": "continue_from_checkpoint",
                        "stage": "controlled_execution_exec:req-1",
                        "checkpoint": {
                            "checkpoint_index": 4,
                            "stage": "controlled_execution_exec:req-1",
                            "captured_at": checkpoint_captured_at,
                        },
                        "guidance_messages": [],
                        "metadata_state": {},
                        "precomputed_artifacts": {},
                        "protocol_artifacts": {},
                        "candidates": [],
                        "evidence_packets": [],
                        "verifications": [],
                        "role_task_snapshot": {},
                    },
                },
            },
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status(run_id, "awaiting_approval"))
    asyncio.run(
        runtime_service._store.create_goal_worker_generation(
            goal_id=goal_id,
            attempt_id=attempt_id,
            agent_run_id=run_id,
            generation_index=1,
            status="awaiting_approval",
            started_at=(datetime.now(UTC) - timedelta(minutes=3)).isoformat(),
        )
    )

    with TestClient(app) as client:
        resume_response = client.post(f"/v1/goals/{goal_id}/resume")
        assert resume_response.status_code == 200

        completed_goal = _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)
        linked_run_response = client.get(f"/v1/agent-runs/{run_id}")
        assert linked_run_response.status_code == 200
        linked_run = linked_run_response.json()

    latest_generation = asyncio.run(
        runtime_service._store.get_latest_goal_worker_generation(
            goal_id,
            attempt_id=attempt_id,
        )
    )

    assert len(completed_goal["attempts"]) == 1
    assert completed_goal["current_attempt_id"] == attempt_id
    assert completed_goal["attempts"][0]["attempt_id"] == attempt_id
    assert completed_goal["attempts"][0]["agent_run_id"] == run_id
    assert linked_run["status"] == "succeeded"
    assert linked_run["summary"]["final_answer"] == "Waiting-approval resume reused the linked run."
    assert len(captured_requests) == 1
    assert captured_requests[0].run_id == run_id
    assert captured_requests[0].metadata["resume_strategy"] == "continue_from_checkpoint"
    assert captured_requests[0].metadata["resume_runtime"]["source"] == "manual_resume"
    assert latest_generation is not None
    assert latest_generation["generation_index"] == 2
    assert latest_generation["rollover_reason"] == "manual_refresh"


def test_goal_refresh_reuses_running_linked_run_as_manual_refresh(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured_requests: list[Any] = []

    async def _running_then_refresh(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        resume_runtime = (
            dict(request.metadata.get("resume_runtime") or {})
            if isinstance(getattr(request, "metadata", None), dict)
            else {}
        )
        if resume_runtime:
            captured_requests.append(request)
            return MultiAgentRunResult(
                run_id=request.run_id,
                protocol="teacher_student_distill",
                state="succeeded",
                task_input=request.task_input,
                candidates=[],
                selected_candidate_id=None,
                evaluation={},
                artifacts={"final_answer": "Live manual refresh reused the running linked run."},
                events=[],
                metadata={},
            )
        await asyncio.sleep(5)
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "unexpected initial completion"},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _running_then_refresh)

    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Refresh a running linked run onto a fresh worker generation.",
                "title": "Live Manual Refresh",
                "protocol_id": "teacher_student_distill",
            },
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200

        running_goal = _wait_goal_until(client, goal_id, {"running"}, timeout_seconds=4.0)
        attempt = running_goal["attempts"][0]
        attempt_id = attempt["attempt_id"]
        run_id = attempt["agent_run_id"]
        assert run_id is not None

        runtime_service = client.app.state.runtime_service
        checkpoint_captured_at = "2026-06-23T13:00:00+00:00"
        checkpoint = asyncio.run(
            runtime_service._store.create_goal_checkpoint(
                goal_id=goal_id,
                attempt_id=attempt_id,
                agent_run_id=run_id,
                checkpoint_index=2,
                stage="research_context_prepared",
                source="operator_test",
                payload={
                    "checkpoint_index": 2,
                    "stage": "research_context_prepared",
                    "captured_at": checkpoint_captured_at,
                },
                metadata={"signature": "live-manual-refresh-checkpoint-1"},
                captured_at=checkpoint_captured_at,
            )
        )
        asyncio.run(
            runtime_service._store.create_goal_memory_snapshot(
                goal_id=goal_id,
                attempt_id=attempt_id,
                checkpoint_id=checkpoint["id"],
                snapshot_kind="compact_recovery_v1",
                snapshot={
                    "goal_objective": "Refresh a running linked run onto a fresh worker generation.",
                    "attempt_id": attempt_id,
                    "agent_run_id": run_id,
                    "protocol_id": "teacher_student_distill",
                    "agent_run_status": "running",
                    "stage": "research_context_prepared",
                    "checkpoint_index": 2,
                    "unfinished_steps": ["resume the linked research worker from the durable handoff"],
                    "captured_at": checkpoint_captured_at,
                },
                metadata={"signature": "live-manual-refresh-memory-1"},
                captured_at=checkpoint_captured_at,
            )
        )
        run = asyncio.run(runtime_service._store.get_agent_run(run_id))
        assert run is not None
        summary = dict(run.get("summary") or {})
        summary["goal_id"] = goal_id
        summary["goal_attempt_id"] = attempt_id
        summary["objective"] = "Refresh a running linked run onto a fresh worker generation."
        summary["task_input"] = "Refresh a running linked run onto a fresh worker generation."
        summary["recovery_state"] = {
            "status": "running",
            "action": "continue",
            "reason": "Operator requested a fresh worker generation.",
            "stage": "research_context_prepared",
            "checkpoint": {
                "checkpoint_index": 2,
                "stage": "research_context_prepared",
                "captured_at": checkpoint_captured_at,
            },
            "resume_payload": {
                "version": 1,
                "executor": "continue_from_checkpoint",
                "strategy_default": "continue_from_checkpoint",
                "stage": "research_context_prepared",
                "checkpoint": {
                    "checkpoint_index": 2,
                    "stage": "research_context_prepared",
                    "captured_at": checkpoint_captured_at,
                },
                "guidance_messages": [],
                "metadata_state": {},
                "precomputed_artifacts": {},
                "protocol_artifacts": {},
                "candidates": [],
                "evidence_packets": [],
                "verifications": [],
                "role_task_snapshot": {},
            },
        }
        asyncio.run(runtime_service._store.update_agent_run_metadata(run_id, summary=summary))

        refresh_response = client.post(f"/v1/goals/{goal_id}/refresh")
        assert refresh_response.status_code == 200

        completed_goal = _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)
        linked_run_response = client.get(f"/v1/agent-runs/{run_id}")
        assert linked_run_response.status_code == 200
        linked_run = linked_run_response.json()

    latest_generation = asyncio.run(
        runtime_service._store.get_latest_goal_worker_generation(
            goal_id,
            attempt_id=attempt_id,
        )
    )

    assert len(completed_goal["attempts"]) == 1
    assert completed_goal["current_attempt_id"] == attempt_id
    assert completed_goal["attempts"][0]["attempt_id"] == attempt_id
    assert completed_goal["attempts"][0]["agent_run_id"] == run_id
    assert linked_run["status"] == "succeeded"
    assert linked_run["summary"]["final_answer"] == "Live manual refresh reused the running linked run."
    assert len(captured_requests) == 1
    assert captured_requests[0].run_id == run_id
    assert captured_requests[0].metadata["resume_strategy"] == "restart_attempt"
    assert captured_requests[0].metadata["resume_payload"] == {}
    assert captured_requests[0].metadata["resume_runtime"]["source"] == "manual_resume"
    assert latest_generation is not None
    assert latest_generation["generation_index"] == 2
    assert latest_generation["rollover_reason"] == "manual_refresh"


def test_goal_refresh_rejects_live_refresh_without_durable_handoff(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    async def _slow_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self, request
        await asyncio.sleep(5)
        return MultiAgentRunResult(
            run_id="unexpected",
            protocol="teacher_student_distill",
            state="succeeded",
            task_input="unexpected",
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "unexpected initial completion"},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _slow_run)

    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Do not refresh a running linked run without durable handoff.",
                "title": "Unsafe Live Manual Refresh",
                "protocol_id": "teacher_student_distill",
            },
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200

        running_goal = _wait_goal_until(client, goal_id, {"running"}, timeout_seconds=4.0)
        assert len(running_goal["attempts"]) == 1

        refresh_response = client.post(f"/v1/goals/{goal_id}/refresh")
        assert refresh_response.status_code == 409
        assert "durable handoff" in refresh_response.json()["detail"]

        goal_response = client.get(f"/v1/goals/{goal_id}")
        assert goal_response.status_code == 200
        goal_payload = goal_response.json()
        assert goal_payload["status"] == "running"
        assert len(goal_payload["attempts"]) == 1

        cancel_response = client.post(f"/v1/goals/{goal_id}/cancel")
        assert cancel_response.status_code == 200


def test_goal_supervisor_auto_refreshes_live_run_when_context_handoff_due(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured_requests: list[Any] = []

    async def _running_then_context_refresh(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        resume_runtime = (
            dict(request.metadata.get("resume_runtime") or {})
            if isinstance(getattr(request, "metadata", None), dict)
            else {}
        )
        if str(resume_runtime.get("source") or "") == "goal_context_handoff_refresh":
            captured_requests.append(request)
            return MultiAgentRunResult(
                run_id=request.run_id,
                protocol="multi_agent_debate",
                state="succeeded",
                task_input=request.task_input,
                candidates=[],
                selected_candidate_id=None,
                evaluation={},
                artifacts={"final_answer": "Supervisor auto-refreshed the live context-pressured run."},
                events=[],
                metadata={},
            )
        await asyncio.sleep(5)
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="multi_agent_debate",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "unexpected initial completion"},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _running_then_context_refresh)

    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Auto-refresh a live worker generation when context pressure crosses the handoff threshold.",
                "title": "Auto Context Handoff Refresh",
                "protocol_id": "multi_agent_debate",
                "run_policy": {"context_handoff_threshold": 0.8},
            },
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200

        running_goal = _wait_goal_until(client, goal_id, {"running"}, timeout_seconds=4.0)
        attempt = running_goal["attempts"][0]
        attempt_id = attempt["attempt_id"]
        run_id = attempt["agent_run_id"]
        assert run_id is not None

        runtime_service = client.app.state.runtime_service
        checkpoint_captured_at = "2026-06-23T14:00:00+00:00"
        checkpoint = asyncio.run(
            runtime_service._store.create_goal_checkpoint(
                goal_id=goal_id,
                attempt_id=attempt_id,
                agent_run_id=run_id,
                checkpoint_index=3,
                stage="debate_context_prepared",
                source="operator_test",
                payload={
                    "checkpoint_index": 3,
                    "stage": "debate_context_prepared",
                    "captured_at": checkpoint_captured_at,
                },
                metadata={"signature": "auto-context-refresh-checkpoint-1"},
                captured_at=checkpoint_captured_at,
            )
        )
        asyncio.run(
            runtime_service._store.create_goal_memory_snapshot(
                goal_id=goal_id,
                attempt_id=attempt_id,
                checkpoint_id=checkpoint["id"],
                snapshot_kind="compact_recovery_v1",
                snapshot={
                    "goal_objective": (
                        "Auto-refresh a live worker generation when context pressure crosses "
                        "the handoff threshold."
                    ),
                    "attempt_id": attempt_id,
                    "agent_run_id": run_id,
                    "protocol_id": "multi_agent_debate",
                    "agent_run_status": "running",
                    "stage": "debate_context_prepared",
                    "checkpoint_index": 3,
                    "unfinished_steps": ["resume the debate on a fresh worker generation"],
                    "captured_at": checkpoint_captured_at,
                },
                metadata={"signature": "auto-context-refresh-memory-1"},
                captured_at=checkpoint_captured_at,
            )
        )
        run = asyncio.run(runtime_service._store.get_agent_run(run_id))
        assert run is not None
        summary = dict(run.get("summary") or {})
        summary["goal_id"] = goal_id
        summary["goal_attempt_id"] = attempt_id
        summary["objective"] = (
            "Auto-refresh a live worker generation when context pressure crosses the handoff threshold."
        )
        summary["task_input"] = (
            "Auto-refresh a live worker generation when context pressure crosses the handoff threshold."
        )
        summary["recovery_state"] = {
            "status": "running",
            "action": "continue",
            "reason": "Goal worker generation is still active.",
            "stage": "debate_context_prepared",
            "checkpoint": {
                "checkpoint_index": 3,
                "stage": "debate_context_prepared",
                "captured_at": checkpoint_captured_at,
            },
            "resume_payload": {
                "version": 1,
                "executor": "continue_from_checkpoint",
                "strategy_default": "continue_from_checkpoint",
                "stage": "debate_context_prepared",
                "checkpoint": {
                    "checkpoint_index": 3,
                    "stage": "debate_context_prepared",
                    "captured_at": checkpoint_captured_at,
                },
                "guidance_messages": [],
                "metadata_state": {},
                "precomputed_artifacts": {},
                "protocol_artifacts": {},
                "candidates": [],
                "evidence_packets": [],
                "verifications": [],
                "role_task_snapshot": {},
            },
        }
        asyncio.run(runtime_service._store.update_agent_run_metadata(run_id, summary=summary))
        asyncio.run(
            runtime_service._store.append_agent_run_artifact(
                run_id,
                artifact_id=f"{run_id}:attempt:{attempt_id}:debate_context_snapshot:auto-high",
                artifact_type="debate_context_snapshot",
                title="Debate Context Snapshot",
                uri=f"agent-run://{run_id}/artifacts/{attempt_id}/debate_context_snapshot/auto-high",
                mime_type="application/json",
                metadata={
                    "attempt_id": attempt_id,
                    "content": {
                        "protocol": "multi_agent_debate",
                        "snapshots": [
                            {
                                "role_id": "judge",
                                "stage": "evaluation",
                                "usage_ratio": 0.91,
                            }
                        ],
                        "latest": {
                            "role_id": "judge",
                            "stage": "evaluation",
                            "usage_ratio": 0.91,
                        },
                    },
                },
            )
        )

        completed_goal = _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)
        linked_run_response = client.get(f"/v1/agent-runs/{run_id}")
        assert linked_run_response.status_code == 200
        linked_run = linked_run_response.json()

    latest_generation = asyncio.run(
        runtime_service._store.get_latest_goal_worker_generation(
            goal_id,
            attempt_id=attempt_id,
        )
    )

    assert len(completed_goal["attempts"]) == 1
    assert completed_goal["current_attempt_id"] == attempt_id
    assert completed_goal["attempts"][0]["attempt_id"] == attempt_id
    assert completed_goal["attempts"][0]["agent_run_id"] == run_id
    assert linked_run["status"] == "succeeded"
    assert linked_run["summary"]["final_answer"] == (
        "Supervisor auto-refreshed the live context-pressured run."
    )
    assert len(captured_requests) == 1
    assert captured_requests[0].run_id == run_id
    assert captured_requests[0].metadata["resume_strategy"] == "restart_attempt"
    assert captured_requests[0].metadata["resume_payload"] == {}
    assert captured_requests[0].metadata["resume_runtime"]["source"] == "goal_context_handoff_refresh"
    assert latest_generation is not None
    assert latest_generation["generation_index"] == 2
    assert latest_generation["rollover_reason"] == "context_pressure"


def test_goal_supervisor_auto_refreshes_live_run_when_generation_refresh_is_overdue(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured_requests: list[Any] = []

    async def _running_then_generation_refresh(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        resume_runtime = (
            dict(request.metadata.get("resume_runtime") or {})
            if isinstance(getattr(request, "metadata", None), dict)
            else {}
        )
        if str(resume_runtime.get("source") or "") == "goal_generation_refresh":
            captured_requests.append(request)
            return MultiAgentRunResult(
                run_id=request.run_id,
                protocol="teacher_student_distill",
                state="succeeded",
                task_input=request.task_input,
                candidates=[],
                selected_candidate_id=None,
                evaluation={},
                artifacts={"final_answer": "Supervisor auto-refreshed the overdue worker generation."},
                events=[],
                metadata={},
            )
        await asyncio.sleep(5)
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "unexpected initial completion"},
            events=[],
            metadata={},
        )

    monkeypatch.setattr(
        "mochi.runtime.service.MultiAgentOrchestrator.run",
        _running_then_generation_refresh,
    )

    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Auto-refresh a live worker generation when the refresh interval is overdue.",
                "title": "Auto Generation Refresh",
                "protocol_id": "teacher_student_distill",
                "run_policy": {"generation_refresh_interval_sec": 60},
            },
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200

        running_goal = _wait_goal_until(client, goal_id, {"running"}, timeout_seconds=4.0)
        attempt = running_goal["attempts"][0]
        attempt_id = attempt["attempt_id"]
        run_id = attempt["agent_run_id"]
        assert run_id is not None

        runtime_service = client.app.state.runtime_service
        checkpoint_captured_at = "2026-06-23T14:00:00+00:00"
        checkpoint = asyncio.run(
            runtime_service._store.create_goal_checkpoint(
                goal_id=goal_id,
                attempt_id=attempt_id,
                agent_run_id=run_id,
                checkpoint_index=3,
                stage="draft_answer",
                source="operator_test",
                payload={
                    "checkpoint_index": 3,
                    "stage": "draft_answer",
                    "captured_at": checkpoint_captured_at,
                },
                metadata={"signature": "auto-generation-refresh-checkpoint-1"},
                captured_at=checkpoint_captured_at,
            )
        )
        asyncio.run(
            runtime_service._store.create_goal_memory_snapshot(
                goal_id=goal_id,
                attempt_id=attempt_id,
                checkpoint_id=checkpoint["id"],
                snapshot_kind="compact_recovery_v1",
                snapshot={
                    "goal_objective": "Auto-refresh a live worker generation when the refresh interval is overdue.",
                    "attempt_id": attempt_id,
                    "agent_run_id": run_id,
                    "protocol_id": "teacher_student_distill",
                    "agent_run_status": "running",
                    "stage": "draft_answer",
                    "checkpoint_index": 3,
                    "unfinished_steps": ["continue from the latest checkpoint on a fresh worker"],
                    "captured_at": checkpoint_captured_at,
                },
                metadata={"signature": "auto-generation-refresh-memory-1"},
                captured_at=checkpoint_captured_at,
            )
        )
        run = asyncio.run(runtime_service._store.get_agent_run(run_id))
        assert run is not None
        summary = dict(run.get("summary") or {})
        summary["goal_id"] = goal_id
        summary["goal_attempt_id"] = attempt_id
        summary["objective"] = (
            "Auto-refresh a live worker generation when the refresh interval is overdue."
        )
        summary["task_input"] = (
            "Auto-refresh a live worker generation when the refresh interval is overdue."
        )
        summary["recovery_state"] = {
            "status": "running",
            "action": "continue",
            "reason": "Goal worker generation is still active.",
            "stage": "draft_answer",
            "checkpoint": {
                "checkpoint_index": 3,
                "stage": "draft_answer",
                "captured_at": checkpoint_captured_at,
            },
            "resume_payload": {
                "version": 1,
                "executor": "continue_from_checkpoint",
                "strategy_default": "continue_from_checkpoint",
                "stage": "draft_answer",
                "checkpoint": {
                    "checkpoint_index": 3,
                    "stage": "draft_answer",
                    "captured_at": checkpoint_captured_at,
                },
                "guidance_messages": [],
                "metadata_state": {},
                "precomputed_artifacts": {},
                "protocol_artifacts": {},
                "candidates": [],
                "evidence_packets": [],
                "verifications": [],
                "role_task_snapshot": {},
            },
        }
        asyncio.run(runtime_service._store.update_agent_run_metadata(run_id, summary=summary))
        latest_generation_before = asyncio.run(
            runtime_service._store.get_latest_goal_worker_generation(
                goal_id,
                attempt_id=attempt_id,
            )
        )
        assert latest_generation_before is not None
        stale_started_at = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
        with sqlite3.connect(tmp_path / "sessions" / "runtime.db") as conn:
            conn.execute(
                "UPDATE goal_worker_generations SET started_at=?, updated_at=? WHERE id=?",
                (stale_started_at, stale_started_at, latest_generation_before["id"]),
            )
            conn.commit()

        completed_goal = _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)
        linked_run_response = client.get(f"/v1/agent-runs/{run_id}")
        assert linked_run_response.status_code == 200
        linked_run = linked_run_response.json()

    latest_generation = asyncio.run(
        runtime_service._store.get_latest_goal_worker_generation(
            goal_id,
            attempt_id=attempt_id,
        )
    )

    assert len(completed_goal["attempts"]) == 1
    assert completed_goal["current_attempt_id"] == attempt_id
    assert completed_goal["attempts"][0]["attempt_id"] == attempt_id
    assert completed_goal["attempts"][0]["agent_run_id"] == run_id
    assert linked_run["status"] == "succeeded"
    assert linked_run["summary"]["final_answer"] == (
        "Supervisor auto-refreshed the overdue worker generation."
    )
    assert len(captured_requests) == 1
    assert captured_requests[0].run_id == run_id
    assert captured_requests[0].metadata["resume_strategy"] == "restart_attempt"
    assert captured_requests[0].metadata["resume_payload"] == {}
    assert captured_requests[0].metadata["resume_runtime"]["source"] == "goal_generation_refresh"
    assert latest_generation is not None
    assert latest_generation["generation_index"] == 2
    assert latest_generation["rollover_reason"] == "scheduled_refresh"


def test_goal_supervisor_auto_refreshes_live_run_when_generation_token_refresh_is_due(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured_requests: list[Any] = []

    async def _running_then_token_refresh(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        resume_runtime = (
            dict(request.metadata.get("resume_runtime") or {})
            if isinstance(getattr(request, "metadata", None), dict)
            else {}
        )
        if str(resume_runtime.get("source") or "") == "goal_token_refresh":
            captured_requests.append(request)
            return MultiAgentRunResult(
                run_id=request.run_id,
                protocol="teacher_student_distill",
                state="succeeded",
                task_input=request.task_input,
                candidates=[],
                selected_candidate_id=None,
                evaluation={},
                artifacts={"final_answer": "Supervisor auto-refreshed the token-pressured worker generation."},
                events=[],
                metadata={},
            )
        await asyncio.sleep(5)
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "unexpected initial completion"},
            events=[],
            metadata={},
        )

    monkeypatch.setattr(
        "mochi.runtime.service.MultiAgentOrchestrator.run",
        _running_then_token_refresh,
    )

    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Auto-refresh a live worker generation when the token refresh threshold is crossed.",
                "title": "Auto Token Refresh",
                "protocol_id": "teacher_student_distill",
                "run_policy": {"generation_token_refresh_threshold": 30},
            },
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200

        running_goal = _wait_goal_until(client, goal_id, {"running"}, timeout_seconds=4.0)
        attempt = running_goal["attempts"][0]
        attempt_id = attempt["attempt_id"]
        run_id = attempt["agent_run_id"]
        assert run_id is not None

        runtime_service = client.app.state.runtime_service
        checkpoint_captured_at = "2026-06-23T14:00:00+00:00"
        checkpoint = asyncio.run(
            runtime_service._store.create_goal_checkpoint(
                goal_id=goal_id,
                attempt_id=attempt_id,
                agent_run_id=run_id,
                checkpoint_index=3,
                stage="draft_answer",
                source="operator_test",
                payload={
                    "checkpoint_index": 3,
                    "stage": "draft_answer",
                    "captured_at": checkpoint_captured_at,
                },
                metadata={"signature": "auto-token-refresh-checkpoint-1"},
                captured_at=checkpoint_captured_at,
            )
        )
        asyncio.run(
            runtime_service._store.create_goal_memory_snapshot(
                goal_id=goal_id,
                attempt_id=attempt_id,
                checkpoint_id=checkpoint["id"],
                snapshot_kind="compact_recovery_v1",
                snapshot={
                    "goal_objective": "Auto-refresh a live worker generation when the token refresh threshold is crossed.",
                    "attempt_id": attempt_id,
                    "agent_run_id": run_id,
                    "protocol_id": "teacher_student_distill",
                    "agent_run_status": "running",
                    "stage": "draft_answer",
                    "checkpoint_index": 3,
                    "unfinished_steps": ["continue from the latest checkpoint on a fresh worker"],
                    "captured_at": checkpoint_captured_at,
                },
                metadata={"signature": "auto-token-refresh-memory-1"},
                captured_at=checkpoint_captured_at,
            )
        )
        run = asyncio.run(runtime_service._store.get_agent_run(run_id))
        assert run is not None
        summary = dict(run.get("summary") or {})
        summary["goal_id"] = goal_id
        summary["goal_attempt_id"] = attempt_id
        summary["objective"] = (
            "Auto-refresh a live worker generation when the token refresh threshold is crossed."
        )
        summary["task_input"] = (
            "Auto-refresh a live worker generation when the token refresh threshold is crossed."
        )
        summary["recovery_state"] = {
            "status": "running",
            "action": "continue",
            "reason": "Goal worker generation is still active.",
            "stage": "draft_answer",
            "checkpoint": {
                "checkpoint_index": 3,
                "stage": "draft_answer",
                "captured_at": checkpoint_captured_at,
            },
            "resume_payload": {
                "version": 1,
                "executor": "continue_from_checkpoint",
                "strategy_default": "continue_from_checkpoint",
                "stage": "draft_answer",
                "checkpoint": {
                    "checkpoint_index": 3,
                    "stage": "draft_answer",
                    "captured_at": checkpoint_captured_at,
                },
                "guidance_messages": [],
                "metadata_state": {},
                "precomputed_artifacts": {},
                "protocol_artifacts": {},
                "candidates": [],
                "evidence_packets": [],
                "verifications": [],
                "role_task_snapshot": {},
            },
        }
        asyncio.run(runtime_service._store.update_agent_run_metadata(run_id, summary=summary))
        asyncio.run(
            runtime_service._store.append_agent_run_artifact(
                run_id,
                artifact_id=f"{run_id}:attempt:{attempt_id}:subagent_runtime:auto-high",
                artifact_type="subagent_runtime",
                title="Subagent Runtime Trace",
                uri=f"agent-run://{run_id}/artifacts/{attempt_id}/subagent_runtime/auto-high",
                mime_type="application/json",
                metadata={
                    "attempt_id": attempt_id,
                    "content": {
                        "invocation_count": 2,
                        "completed_invocation_count": 2,
                        "token_tracked_invocation_count": 2,
                        "input_tokens": 20,
                        "output_tokens": 12,
                        "total_tokens": 32,
                        "generation_time_ms": 41.5,
                        "finish_reason_counts": {"stop": 2},
                        "approval_pending": [],
                        "risky_tool_events": [],
                        "invocations": [],
                    },
                },
            )
        )

        completed_goal = _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)
        linked_run_response = client.get(f"/v1/agent-runs/{run_id}")
        assert linked_run_response.status_code == 200
        linked_run = linked_run_response.json()

    latest_generation = asyncio.run(
        runtime_service._store.get_latest_goal_worker_generation(
            goal_id,
            attempt_id=attempt_id,
        )
    )

    assert len(completed_goal["attempts"]) == 1
    assert completed_goal["current_attempt_id"] == attempt_id
    assert completed_goal["attempts"][0]["attempt_id"] == attempt_id
    assert completed_goal["attempts"][0]["agent_run_id"] == run_id
    assert linked_run["status"] == "succeeded"
    assert linked_run["summary"]["final_answer"] == (
        "Supervisor auto-refreshed the token-pressured worker generation."
    )
    assert len(captured_requests) == 1
    assert captured_requests[0].run_id == run_id
    assert captured_requests[0].metadata["resume_strategy"] == "restart_attempt"
    assert captured_requests[0].metadata["resume_payload"] == {}
    assert captured_requests[0].metadata["resume_runtime"]["source"] == "goal_token_refresh"
    assert latest_generation is not None
    assert latest_generation["generation_index"] == 2
    assert latest_generation["rollover_reason"] == "context_pressure"


def test_goal_progress_snapshot_endpoints_list_persisted_records(
    tmp_path: Path,
) -> None:
    app, runtime_service = _create_goal_test_app(tmp_path)
    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-progress-api-1",
            objective="Inspect persisted checkpoint and memory snapshot records",
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-progress-api-attempt-1",
            goal_id="goal-progress-api-1",
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id="linked-progress-api-run-1",
        )
    )
    older_checkpoint = asyncio.run(
        runtime_service._store.create_goal_checkpoint(
            goal_id="goal-progress-api-1",
            attempt_id="goal-progress-api-attempt-1",
            agent_run_id="linked-progress-api-run-1",
            checkpoint_index=2,
            stage="evidence_collection",
            source="agent_run_sync",
            payload={"checkpoint_index": 2, "stage": "evidence_collection"},
            metadata={"signature": "api-sig-1"},
            captured_at="2026-06-23T02:00:00+00:00",
        )
    )
    newer_checkpoint = asyncio.run(
        runtime_service._store.create_goal_checkpoint(
            goal_id="goal-progress-api-1",
            attempt_id="goal-progress-api-attempt-1",
            agent_run_id="linked-progress-api-run-1",
            checkpoint_index=3,
            stage="controller_decision",
            source="agent_run_sync",
            payload={"checkpoint_index": 3, "stage": "controller_decision"},
            metadata={"signature": "api-sig-2"},
            captured_at="2026-06-23T02:05:00+00:00",
        )
    )
    memory_snapshot = asyncio.run(
        runtime_service._store.create_goal_memory_snapshot(
            goal_id="goal-progress-api-1",
            attempt_id="goal-progress-api-attempt-1",
            checkpoint_id=newer_checkpoint["id"],
            snapshot_kind="compact_recovery_v1",
            snapshot={"checkpoint_index": 3, "stage": "controller_decision"},
            metadata={"signature": "api-mem-sig-1"},
            captured_at="2026-06-23T02:05:01+00:00",
        )
    )

    with TestClient(app) as client:
        checkpoints_response = client.get(
            "/v1/goals/goal-progress-api-1/checkpoints",
            params={"attempt_id": "goal-progress-api-attempt-1"},
        )
        assert checkpoints_response.status_code == 200
        checkpoints_payload = checkpoints_response.json()

        limited_checkpoints_response = client.get(
            "/v1/goals/goal-progress-api-1/checkpoints",
            params={"attempt_id": "goal-progress-api-attempt-1", "limit": 1},
        )
        assert limited_checkpoints_response.status_code == 200
        limited_checkpoints_payload = limited_checkpoints_response.json()

        memory_snapshots_response = client.get(
            "/v1/goals/goal-progress-api-1/memory-snapshots",
            params={"attempt_id": "goal-progress-api-attempt-1"},
        )
        assert memory_snapshots_response.status_code == 200
        memory_snapshots_payload = memory_snapshots_response.json()

        missing_goal_response = client.get("/v1/goals/missing-goal/checkpoints")
        assert missing_goal_response.status_code == 404
        assert missing_goal_response.json()["detail"] == "Goal not found"

    assert [item["checkpoint_id"] for item in checkpoints_payload] == [
        newer_checkpoint["id"],
        older_checkpoint["id"],
    ]
    assert limited_checkpoints_payload[0]["checkpoint_id"] == newer_checkpoint["id"]
    assert len(limited_checkpoints_payload) == 1
    assert memory_snapshots_payload[0]["snapshot_id"] == memory_snapshot["id"]
    assert memory_snapshots_payload[0]["checkpoint_id"] == newer_checkpoint["id"]


def test_goal_maps_orchestrator_awaiting_approval_into_agent_run_and_goal_health(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    approval_wait_started_at = (datetime.now(UTC) - timedelta(seconds=75)).isoformat()

    async def _awaiting_approval_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="controlled_subagent_execution",
            state="awaiting_approval",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={
                "final_answer": None,
                "controlled_execution_runtime": {"approval_pending_count": 1},
            },
            events=[],
            metadata={
                "approval_state": {
                    "status": "awaiting_approval",
                    "pending_count": 1,
                    "approval_ids": ["exec-approval-2"],
                    "pending_approvals": [
                        {
                            "approval_id": "exec-approval-2",
                            "tool_name": "exec_command",
                            "request_id": "req-1",
                            "task_key": "controlled_execution_exec:req-1",
                            "source": "controlled_execution",
                            "created_at": approval_wait_started_at,
                            "timeout_sec": 300,
                        }
                    ],
                },
                "recovery_state": {
                    "status": "awaiting_approval",
                    "action": "await_approval",
                    "reason": "Execution approval required",
                    "stage": "controlled_execution_exec:req-1",
                    "checkpoint": {
                        "checkpoint_index": 4,
                        "stage": "controlled_execution_controller:req-1",
                    },
                    "approval_state": {
                        "status": "awaiting_approval",
                        "pending_count": 1,
                        "approval_ids": ["exec-approval-2"],
                    },
                },
            },
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _awaiting_approval_run)

    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Pause for execution approval and resume later.",
                "title": "Direct Approval Goal",
                "protocol_id": "controlled_subagent_execution",
            },
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200

        goal_payload = _wait_goal_until(client, goal_id, {"waiting_approval"}, timeout_seconds=4.0)
        linked_run_id = goal_payload["attempts"][0]["agent_run_id"]
        assert linked_run_id is not None

        linked_run_response = client.get(f"/v1/agent-runs/{linked_run_id}")
        assert linked_run_response.status_code == 200
        linked_run = linked_run_response.json()
        assert linked_run["status"] == "awaiting_approval"
        assert linked_run["summary"]["approval_state"]["pending_count"] == 1

        health_response = client.get(f"/v1/goals/{goal_id}/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()
        assert health_payload["status"] == "waiting_approval"
        assert health_payload["approval_state"]["status"] == "waiting_approval"
        assert health_payload["approval_state"]["approval_ids"] == ["exec-approval-2"]
        assert health_payload["approval_state"]["approval_wait_started_at"] == approval_wait_started_at
        assert health_payload["approval_state"]["approval_wait_timeout_sec"] == 300
        assert 60 <= health_payload["approval_state"]["approval_wait_elapsed_sec"] <= 120
        assert health_payload["linked_agent_run"]["status"] == "awaiting_approval"
        assert (
            health_payload["linked_agent_run"]["approval_state"]["approval_wait_started_at"]
            == approval_wait_started_at
        )
        assert (
            health_payload["linked_agent_run"]["approval_state"]["approval_wait_timeout_sec"] == 300
        )
        assert health_payload["recovery_state"]["status"] == "awaiting_approval"


def test_goal_resume_endpoint_resolves_exec_approval_on_linked_agent_run_without_creating_new_attempt(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    approval_id = "exec-approval-goal-resume-1"

    async def _approval_then_success_run(self: Any, request: Any) -> MultiAgentRunResult:
        approval = self._exec_approval_store.get(approval_id)
        if approval is None or approval.status == "pending":
            if approval is None:
                self._exec_approval_store.create(
                    approval_id=approval_id,
                    command="print('goal approval resume')",
                    shell="test",
                    scope="dangerous_command",
                    reason="Exec command requires approval.",
                    command_payload={
                        "command": "print('goal approval resume')",
                        "shell": "test",
                        "workdir": str(tmp_path),
                        "env": None,
                        "timeout_sec": 5.0,
                        "background": False,
                        "tty": False,
                        "approval_state": "approved",
                    },
                )
            return MultiAgentRunResult(
                run_id=request.run_id,
                protocol="controlled_subagent_execution",
                state="awaiting_approval",
                task_input=request.task_input,
                candidates=[],
                selected_candidate_id=None,
                evaluation={},
                artifacts={
                    "final_answer": None,
                    "controlled_execution_runtime": {"approval_pending_count": 1},
                },
                events=[],
                metadata={
                    "approval_state": {
                        "status": "awaiting_approval",
                        "pending_count": 1,
                        "approval_ids": [approval_id],
                        "pending_approvals": [
                            {
                                "approval_id": approval_id,
                                "tool_name": "exec_command",
                                "request_id": "req-1",
                                "task_key": "controlled_execution_exec:req-1",
                                "source": "controlled_execution",
                            }
                        ],
                    },
                    "recovery_state": {
                        "status": "awaiting_approval",
                        "action": "await_approval",
                        "reason": "Execution approval required",
                        "stage": "controlled_execution_exec:req-1",
                        "checkpoint": {
                            "checkpoint_index": 4,
                            "stage": "controlled_execution_controller:req-1",
                        },
                        "approval_state": {
                            "status": "awaiting_approval",
                            "pending_count": 1,
                            "approval_ids": [approval_id],
                        },
                    },
                },
            )
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="controlled_subagent_execution",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Goal approval resume completed."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _approval_then_success_run)

    app = create_app()
    exec_approval_store = InMemoryApprovalStore()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
        exec_approval_store=exec_approval_store,
        exec_runtime=ExecRuntime(
            providers={"test": _GoalApiPythonDirectProvider()},
            default_shell="test",
        ),
    )
    runtime_service.set_scheduler_poll_interval(0.05)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Resume the linked run once execution approval is granted.",
                "title": "Goal approval resume",
                "protocol_id": "controlled_subagent_execution",
            },
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200

        waiting_goal = _wait_goal_until(client, goal_id, {"waiting_approval"}, timeout_seconds=4.0)
        assert len(waiting_goal["attempts"]) == 1
        linked_run_id = waiting_goal["attempts"][0]["agent_run_id"]
        assert linked_run_id is not None

        resume_response = client.post(
            f"/v1/goals/{goal_id}/resume",
            json={
                "approval_id": approval_id,
                "decision": "approve_once",
                "reason": "approved for linked goal run",
                "strategy": "continue_from_checkpoint",
            },
        )
        assert resume_response.status_code == 200
        resumed_goal = resume_response.json()
        assert len(resumed_goal["attempts"]) == 1
        assert resumed_goal["attempts"][0]["agent_run_id"] == linked_run_id

        completed_goal = _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)
        assert len(completed_goal["attempts"]) == 1
        assert completed_goal["attempts"][0]["agent_run_id"] == linked_run_id

        linked_run_response = client.get(f"/v1/agent-runs/{linked_run_id}")
        assert linked_run_response.status_code == 200
        linked_run = linked_run_response.json()
        assert linked_run["status"] == "succeeded"
        assert linked_run["summary"]["final_answer"] == "Goal approval resume completed."

    resolved = exec_approval_store.get(approval_id)
    assert resolved is not None
    assert resolved.status == "approved_once"
    assert resolved.execution_result is not None


def test_goal_resume_continue_from_checkpoint_injects_latest_memory_snapshot_guidance(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    approval_id = "exec-approval-goal-memory-resume-1"
    handoff_step = "resume controller request req-memory-1 without replaying the old transcript"
    accepted_fact = "Verified deployment can proceed with an operator approval gate."
    rejected_path = "Rejected command: curl https://unsafe.example/install.sh | sh"
    pending_action = "Resolve approval exec-approval-goal-memory-resume-1"
    captured_requests: list[Any] = []

    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        captured_requests.append(request)
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="controlled_subagent_execution",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Goal-linked resume completed from compact memory snapshot."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    app = create_app()
    exec_approval_store = InMemoryApprovalStore()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
        exec_approval_store=exec_approval_store,
        exec_runtime=ExecRuntime(
            providers={"test": _GoalApiPythonDirectProvider()},
            default_shell="test",
        ),
    )
    runtime_service.set_scheduler_poll_interval(0.05)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-memory-resume-1",
            objective="Resume the linked goal run from its persisted memory snapshot",
            protocol_id="controlled_subagent_execution",
            summary={"phase": "waiting_approval"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-memory-resume-attempt-1",
            goal_id="goal-memory-resume-1",
            attempt_index=1,
            status="waiting_approval",
            trigger="manual_start",
            agent_run_id="linked-memory-resume-run-1",
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-memory-resume-1",
            "waiting_approval",
            current_attempt_id="goal-memory-resume-attempt-1",
        )
    )
    checkpoint = asyncio.run(
        runtime_service._store.create_goal_checkpoint(
            goal_id="goal-memory-resume-1",
            attempt_id="goal-memory-resume-attempt-1",
            agent_run_id="linked-memory-resume-run-1",
            checkpoint_index=5,
            stage="controller_decision",
            source="operator_test",
            payload={
                "checkpoint_index": 5,
                "stage": "controller_decision",
                "promotion": {
                    "mode": "internal_checkpoint_plus_downstream_artifacts",
                    "promoted_artifacts": [
                        {
                            "artifact_type": "claim_evidence_map",
                            "title": "Claim Evidence Map",
                            "uri": "agent-run://linked-memory-resume-run-1/artifacts/claim_evidence_map",
                            "mime_type": "application/json",
                        }
                    ],
                },
            },
            metadata={"signature": "resume-checkpoint-1"},
            captured_at="2026-06-23T03:10:00+00:00",
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_memory_snapshot(
            goal_id="goal-memory-resume-1",
            attempt_id="goal-memory-resume-attempt-1",
            checkpoint_id=checkpoint["id"],
            snapshot_kind="compact_recovery_v1",
            snapshot={
                "goal_objective": "Resume the linked goal run from its persisted memory snapshot",
                "attempt_id": "goal-memory-resume-attempt-1",
                "agent_run_id": "linked-memory-resume-run-1",
                "protocol_id": "controlled_subagent_execution",
                "agent_run_status": "awaiting_approval",
                "stage": "controller_decision",
                "checkpoint_index": 5,
                "unfinished_steps": [handoff_step],
                "pending_actions": [pending_action],
                "accepted_facts": [accepted_fact],
                "rejected_paths": [rejected_path],
                "promoted_artifacts": [
                    {
                        "artifact_type": "claim_evidence_map",
                        "title": "Claim Evidence Map",
                        "uri": "agent-run://linked-memory-resume-run-1/artifacts/claim_evidence_map",
                        "mime_type": "application/json",
                    }
                ],
                "captured_at": "2026-06-23T03:10:01+00:00",
            },
            metadata={"signature": "resume-memory-1"},
            captured_at="2026-06-23T03:10:01+00:00",
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id="linked-memory-resume-run-1",
            protocol_id="controlled_subagent_execution",
            title="Goal memory resume linked run",
            topic="goal memory resume",
            summary={
                "goal_id": "goal-memory-resume-1",
                "goal_attempt_id": "goal-memory-resume-attempt-1",
                "objective": "Resume the linked goal run from its persisted memory snapshot",
                "task_input": "Resume the linked goal run from its persisted memory snapshot",
                "approval_state": {
                    "status": "awaiting_approval",
                    "pending_count": 1,
                    "approval_ids": [approval_id],
                    "pending_approvals": [
                        {
                            "approval_id": approval_id,
                            "tool_name": "exec_command",
                            "request_id": "req-memory-1",
                            "task_key": "controlled_execution_exec:req-memory-1",
                            "stage": "controlled_execution_exec:req-memory-1",
                            "role_id": "controller",
                            "source": "controlled_execution",
                        }
                    ],
                },
                "recovery_state": {
                    "status": "awaiting_approval",
                    "action": "await_approval",
                    "reason": "Execution approval required",
                    "stage": "controlled_execution_exec:req-memory-1",
                    "checkpoint": {
                        "checkpoint_index": 5,
                        "stage": "controller_decision",
                    },
                    "approval_state": {
                        "status": "awaiting_approval",
                        "pending_count": 1,
                        "approval_ids": [approval_id],
                        "pending_approvals": [
                            {
                                "approval_id": approval_id,
                                "tool_name": "exec_command",
                                "request_id": "req-memory-1",
                                "task_key": "controlled_execution_exec:req-memory-1",
                                "stage": "controlled_execution_exec:req-memory-1",
                                "role_id": "controller",
                                "source": "controlled_execution",
                            }
                        ],
                    },
                    "resume_payload": {
                        "version": 1,
                        "executor": "continue_from_checkpoint",
                        "strategy_default": "continue_from_checkpoint",
                        "stage": "controlled_execution_exec:req-memory-1",
                        "checkpoint": {
                            "checkpoint_index": 5,
                            "stage": "controller_decision",
                        },
                        "guidance_messages": [],
                        "role_guidance_messages": {},
                        "metadata_state": {},
                        "precomputed_artifacts": {},
                        "protocol_artifacts": {},
                        "candidates": [],
                        "evidence_packets": [],
                        "verifications": [],
                        "role_task_snapshot": {"roles": {}, "tasks": {}},
                    },
                },
            },
        )
    )
    asyncio.run(
        runtime_service._store.update_agent_run_status(
            "linked-memory-resume-run-1",
            "awaiting_approval",
        )
    )
    exec_approval_store.create(
        approval_id=approval_id,
        command="print('goal memory resume approval')",
        shell="test",
        scope="dangerous_command",
        reason="Exec command requires approval.",
        command_payload={
            "command": "print('goal memory resume approval')",
            "shell": "test",
            "workdir": str(tmp_path),
            "env": None,
            "timeout_sec": 5.0,
            "background": False,
            "tty": False,
            "approval_state": "approved",
        },
    )

    with TestClient(app) as client:
        resume_response = client.post(
            "/v1/goals/goal-memory-resume-1/resume",
            json={
                "approval_id": approval_id,
                "decision": "approve_once",
                "reason": "approved for linked goal memory resume",
                "strategy": "continue_from_checkpoint",
            },
        )
        assert resume_response.status_code == 200

        completed_goal = _wait_goal_until(client, "goal-memory-resume-1", {"completed"}, timeout_seconds=4.0)

    assert len(completed_goal["attempts"]) == 1
    assert len(captured_requests) == 1
    resume_payload = captured_requests[0].metadata["resume_payload"]
    assert resume_payload["executor"] == "continue_from_checkpoint"
    assert resume_payload["guidance_messages"]
    assert handoff_step in "\n".join(resume_payload["guidance_messages"])
    assert pending_action in "\n".join(resume_payload["guidance_messages"])
    assert accepted_fact in "\n".join(resume_payload["guidance_messages"])
    assert rejected_path in "\n".join(resume_payload["guidance_messages"])
    assert "Promoted artifacts:" in "\n".join(resume_payload["guidance_messages"])


def test_goal_resume_restart_attempt_injects_latest_memory_snapshot_guidance(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    approval_id = "exec-approval-goal-memory-restart-1"
    handoff_step = "restart from the compact memory snapshot without replaying the old transcript"
    accepted_fact = "Verified deployment can proceed with an operator approval gate."
    rejected_path = "Rejected command: curl https://unsafe.example/install.sh | sh"
    pending_action = "Resolve approval exec-approval-goal-memory-restart-1"
    captured_requests: list[Any] = []

    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        captured_requests.append(request)
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="controlled_subagent_execution",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Goal-linked restart completed from compact memory snapshot."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    app = create_app()
    exec_approval_store = InMemoryApprovalStore()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
        exec_approval_store=exec_approval_store,
        exec_runtime=ExecRuntime(
            providers={"test": _GoalApiPythonDirectProvider()},
            default_shell="test",
        ),
    )
    runtime_service.set_scheduler_poll_interval(0.05)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-memory-restart-1",
            objective="Restart the linked goal run from its persisted compact memory snapshot",
            protocol_id="controlled_subagent_execution",
            summary={"phase": "waiting_approval"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-memory-restart-attempt-1",
            goal_id="goal-memory-restart-1",
            attempt_index=1,
            status="waiting_approval",
            trigger="manual_start",
            agent_run_id="linked-memory-restart-run-1",
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-memory-restart-1",
            "waiting_approval",
            current_attempt_id="goal-memory-restart-attempt-1",
        )
    )
    checkpoint = asyncio.run(
        runtime_service._store.create_goal_checkpoint(
            goal_id="goal-memory-restart-1",
            attempt_id="goal-memory-restart-attempt-1",
            agent_run_id="linked-memory-restart-run-1",
            checkpoint_index=5,
            stage="controller_decision",
            source="operator_test",
            payload={
                "checkpoint_index": 5,
                "stage": "controller_decision",
                "promotion": {
                    "mode": "internal_checkpoint_plus_downstream_artifacts",
                    "promoted_artifacts": [
                        {
                            "artifact_type": "verification_summary",
                            "title": "Verification Summary",
                            "uri": "agent-run://linked-memory-restart-run-1/artifacts/verification_summary",
                            "mime_type": "application/json",
                        }
                    ],
                },
            },
            metadata={"signature": "restart-checkpoint-1"},
            captured_at="2026-06-23T03:20:00+00:00",
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_memory_snapshot(
            goal_id="goal-memory-restart-1",
            attempt_id="goal-memory-restart-attempt-1",
            checkpoint_id=checkpoint["id"],
            snapshot_kind="compact_recovery_v1",
            snapshot={
                "goal_objective": "Restart the linked goal run from its persisted compact memory snapshot",
                "attempt_id": "goal-memory-restart-attempt-1",
                "agent_run_id": "linked-memory-restart-run-1",
                "protocol_id": "controlled_subagent_execution",
                "agent_run_status": "awaiting_approval",
                "stage": "controller_decision",
                "checkpoint_index": 5,
                "unfinished_steps": [handoff_step],
                "pending_actions": [pending_action],
                "accepted_facts": [accepted_fact],
                "rejected_paths": [rejected_path],
                "promoted_artifacts": [
                    {
                        "artifact_type": "verification_summary",
                        "title": "Verification Summary",
                        "uri": "agent-run://linked-memory-restart-run-1/artifacts/verification_summary",
                        "mime_type": "application/json",
                    }
                ],
                "captured_at": "2026-06-23T03:20:01+00:00",
            },
            metadata={"signature": "restart-memory-1"},
            captured_at="2026-06-23T03:20:01+00:00",
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id="linked-memory-restart-run-1",
            protocol_id="controlled_subagent_execution",
            title="Goal memory restart linked run",
            topic="goal memory restart",
            summary={
                "goal_id": "goal-memory-restart-1",
                "goal_attempt_id": "goal-memory-restart-attempt-1",
                "objective": "Restart the linked goal run from its persisted compact memory snapshot",
                "task_input": "Restart the linked goal run from its persisted compact memory snapshot",
                "approval_state": {
                    "status": "awaiting_approval",
                    "pending_count": 1,
                    "approval_ids": [approval_id],
                    "pending_approvals": [
                        {
                            "approval_id": approval_id,
                            "tool_name": "exec_command",
                            "request_id": "req-memory-restart-1",
                            "task_key": "controlled_execution_exec:req-memory-restart-1",
                            "stage": "controlled_execution_exec:req-memory-restart-1",
                            "role_id": "controller",
                            "source": "controlled_execution",
                        }
                    ],
                },
                "recovery_state": {
                    "status": "awaiting_approval",
                    "action": "await_approval",
                    "reason": "Execution approval required",
                    "stage": "controlled_execution_exec:req-memory-restart-1",
                    "checkpoint": {
                        "checkpoint_index": 5,
                        "stage": "controller_decision",
                    },
                    "approval_state": {
                        "status": "awaiting_approval",
                        "pending_count": 1,
                        "approval_ids": [approval_id],
                        "pending_approvals": [
                            {
                                "approval_id": approval_id,
                                "tool_name": "exec_command",
                                "request_id": "req-memory-restart-1",
                                "task_key": "controlled_execution_exec:req-memory-restart-1",
                                "stage": "controlled_execution_exec:req-memory-restart-1",
                                "role_id": "controller",
                                "source": "controlled_execution",
                            }
                        ],
                    },
                    "resume_payload": {
                        "version": 1,
                        "executor": "continue_from_checkpoint",
                        "strategy_default": "continue_from_checkpoint",
                        "stage": "controlled_execution_exec:req-memory-restart-1",
                        "checkpoint": {
                            "checkpoint_index": 5,
                            "stage": "controller_decision",
                        },
                        "guidance_messages": [],
                        "role_guidance_messages": {},
                        "metadata_state": {},
                        "precomputed_artifacts": {},
                        "protocol_artifacts": {},
                        "candidates": [],
                        "evidence_packets": [],
                        "verifications": [],
                        "role_task_snapshot": {"roles": {}, "tasks": {}},
                    },
                },
            },
        )
    )
    asyncio.run(
        runtime_service._store.update_agent_run_status(
            "linked-memory-restart-run-1",
            "awaiting_approval",
        )
    )
    exec_approval_store.create(
        approval_id=approval_id,
        command="print('goal memory restart approval')",
        shell="test",
        scope="dangerous_command",
        reason="Exec command requires approval.",
        command_payload={
            "command": "print('goal memory restart approval')",
            "shell": "test",
            "workdir": str(tmp_path),
            "env": None,
            "timeout_sec": 5.0,
            "background": False,
            "tty": False,
            "approval_state": "approved",
        },
    )

    with TestClient(app) as client:
        resume_response = client.post(
            "/v1/goals/goal-memory-restart-1/resume",
            json={
                "approval_id": approval_id,
                "decision": "approve_once",
                "reason": "approved for linked goal memory restart",
                "strategy": "restart_attempt",
            },
        )
        assert resume_response.status_code == 200

        completed_goal = _wait_goal_until(client, "goal-memory-restart-1", {"completed"}, timeout_seconds=4.0)
        linked_run_response = client.get("/v1/agent-runs/linked-memory-restart-run-1")
        assert linked_run_response.status_code == 200
        linked_run = linked_run_response.json()

    assert len(completed_goal["attempts"]) == 1
    assert len(captured_requests) == 1
    assert captured_requests[0].metadata["resume_strategy"] == "restart_attempt"
    assert captured_requests[0].metadata["resume_payload"] == {}
    assert captured_requests[0].guidance_messages
    joined_guidance = "\n".join(captured_requests[0].guidance_messages)
    assert handoff_step in joined_guidance
    assert pending_action in joined_guidance
    assert accepted_fact in joined_guidance
    assert rejected_path in joined_guidance
    assert "Promoted artifacts:" in joined_guidance
    assert linked_run["summary"]["goal_handoff"]["durable_handoff_gate"]["usable"] is True
    assert linked_run["summary"]["goal_handoff"]["checkpoint_promotion_mode"] == (
        "internal_checkpoint_plus_downstream_artifacts"
    )
    assert linked_run["summary"]["goal_handoff"]["promoted_artifacts"][0]["artifact_type"] == (
        "verification_summary"
    )


def test_goal_resume_continue_from_checkpoint_falls_back_to_restart_without_durable_handoff(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    approval_id = "exec-approval-goal-memory-fallback-1"
    captured_requests: list[Any] = []

    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        captured_requests.append(request)
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="controlled_subagent_execution",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Goal-linked resume restarted without durable handoff."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    app = create_app()
    exec_approval_store = InMemoryApprovalStore()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
        exec_approval_store=exec_approval_store,
        exec_runtime=ExecRuntime(
            providers={"test": _GoalApiPythonDirectProvider()},
            default_shell="test",
        ),
    )
    runtime_service.set_scheduler_poll_interval(0.05)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-memory-fallback-1",
            objective="Resume the linked goal run only when durable handoff exists",
            protocol_id="controlled_subagent_execution",
            summary={"phase": "waiting_approval"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-memory-fallback-attempt-1",
            goal_id="goal-memory-fallback-1",
            attempt_index=1,
            status="waiting_approval",
            trigger="manual_start",
            agent_run_id="linked-memory-fallback-run-1",
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-memory-fallback-1",
            "waiting_approval",
            current_attempt_id="goal-memory-fallback-attempt-1",
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id="linked-memory-fallback-run-1",
            protocol_id="controlled_subagent_execution",
            title="Goal memory fallback linked run",
            topic="goal memory fallback",
            summary={
                "goal_id": "goal-memory-fallback-1",
                "goal_attempt_id": "goal-memory-fallback-attempt-1",
                "objective": "Resume the linked goal run only when durable handoff exists",
                "task_input": "Resume the linked goal run only when durable handoff exists",
                "approval_state": {
                    "status": "awaiting_approval",
                    "pending_count": 1,
                    "approval_ids": [approval_id],
                    "pending_approvals": [
                        {
                            "approval_id": approval_id,
                            "tool_name": "exec_command",
                            "request_id": "req-fallback-1",
                            "task_key": "controlled_execution_exec:req-fallback-1",
                            "stage": "controlled_execution_exec:req-fallback-1",
                            "role_id": "controller",
                            "source": "controlled_execution",
                        }
                    ],
                },
                "recovery_state": {
                    "status": "awaiting_approval",
                    "action": "await_approval",
                    "reason": "Execution approval required",
                    "stage": "controlled_execution_exec:req-fallback-1",
                    "checkpoint": {
                        "checkpoint_index": 5,
                        "stage": "controller_decision",
                        "captured_at": "2026-06-23T03:10:01+00:00",
                    },
                    "approval_state": {
                        "status": "awaiting_approval",
                        "pending_count": 1,
                        "approval_ids": [approval_id],
                        "pending_approvals": [
                            {
                                "approval_id": approval_id,
                                "tool_name": "exec_command",
                                "request_id": "req-fallback-1",
                                "task_key": "controlled_execution_exec:req-fallback-1",
                                "stage": "controlled_execution_exec:req-fallback-1",
                                "role_id": "controller",
                                "source": "controlled_execution",
                            }
                        ],
                    },
                    "resume_payload": {
                        "version": 1,
                        "executor": "continue_from_checkpoint",
                        "strategy_default": "continue_from_checkpoint",
                        "stage": "controlled_execution_exec:req-fallback-1",
                        "checkpoint": {
                            "checkpoint_index": 5,
                            "stage": "controller_decision",
                            "captured_at": "2026-06-23T03:10:01+00:00",
                        },
                        "guidance_messages": [],
                        "role_guidance_messages": {},
                        "metadata_state": {},
                        "precomputed_artifacts": {},
                        "protocol_artifacts": {},
                        "candidates": [],
                        "evidence_packets": [],
                        "verifications": [],
                        "role_task_snapshot": {"roles": {}, "tasks": {}},
                    },
                },
            },
        )
    )
    asyncio.run(
        runtime_service._store.update_agent_run_status(
            "linked-memory-fallback-run-1",
            "awaiting_approval",
        )
    )
    exec_approval_store.create(
        approval_id=approval_id,
        command="print('goal memory fallback approval')",
        shell="test",
        scope="dangerous_command",
        reason="Exec command requires approval.",
        command_payload={
            "command": "print('goal memory fallback approval')",
            "shell": "test",
            "workdir": str(tmp_path),
            "env": None,
            "timeout_sec": 5.0,
            "background": False,
            "tty": False,
            "approval_state": "approved",
        },
    )

    with TestClient(app) as client:
        resume_response = client.post(
            "/v1/goals/goal-memory-fallback-1/resume",
            json={
                "approval_id": approval_id,
                "decision": "approve_once",
                "reason": "approved for linked goal memory fallback resume",
                "strategy": "continue_from_checkpoint",
            },
        )
        assert resume_response.status_code == 200

        completed_goal = _wait_goal_until(
            client,
            "goal-memory-fallback-1",
            {"completed"},
            timeout_seconds=4.0,
        )
        linked_run_response = client.get("/v1/agent-runs/linked-memory-fallback-run-1")
        assert linked_run_response.status_code == 200
        linked_run = linked_run_response.json()

    assert len(completed_goal["attempts"]) == 1
    assert len(captured_requests) == 1
    assert captured_requests[0].metadata["resume_strategy"] == "restart_attempt"
    assert captured_requests[0].metadata["resume_payload"] == {}
    assert captured_requests[0].metadata["resume_runtime"]["strategy"] == "restart_attempt"
    assert linked_run["summary"]["goal_handoff"]["resume_gate"]["effective_strategy"] == "restart_attempt"
    assert linked_run["summary"]["goal_handoff"]["resume_gate"]["reason"] == "missing_memory_snapshot"
    resumed_events = [
        event
        for event in linked_run["events"]
        if isinstance(event, dict) and event.get("type") == "run_resumed"
    ]
    assert resumed_events
    assert resumed_events[-1]["resume_strategy"] == "restart_attempt"


def test_goal_resume_continue_from_checkpoint_rehydrates_from_durable_goal_checkpoint(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured_requests: list[Any] = []

    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        captured_requests.append(request)
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Goal-linked checkpoint resume rehydrated from durable state."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    runtime_service.set_scheduler_poll_interval(0.05)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    goal_id = "goal-rehydrate-resume-1"
    attempt_id = "goal-rehydrate-resume-attempt-1"
    run_id = "linked-rehydrate-resume-run-1"
    checkpoint_captured_at = "2026-06-23T04:00:00+00:00"
    snapshot_captured_at = "2026-06-23T04:00:01+00:00"
    role_task_snapshot = {
        "version": 1,
        "roles": {
            "teacher": {
                "role_id": "teacher",
                "status": "completed",
                "stage": "teacher_generation",
                "resume_action": "reuse_output",
                "candidate": {
                    "candidate_id": "teacher-candidate-1",
                    "role_id": "teacher",
                    "content": "Teacher draft",
                    "metadata": {"model_id": "teacher-model"},
                },
            }
        },
        "tasks": {
            "teacher_generation": {
                "task_key": "teacher_generation",
                "role_id": "teacher",
                "status": "completed",
                "stage": "teacher_generation",
                "resume_action": "reuse_output",
                "candidate": {
                    "candidate_id": "teacher-candidate-1",
                    "role_id": "teacher",
                    "content": "Teacher draft",
                    "metadata": {"model_id": "teacher-model"},
                },
            }
        },
    }

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id=goal_id,
            objective="Resume a stalled linked run from its durable goal checkpoint.",
            protocol_id="teacher_student_distill",
            summary={"phase": "stalled"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id=attempt_id,
            goal_id=goal_id,
            attempt_index=1,
            status="stalled",
            trigger="manual_start",
            agent_run_id=run_id,
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            goal_id,
            "stalled",
            current_attempt_id=attempt_id,
        )
    )
    checkpoint = asyncio.run(
        runtime_service._store.create_goal_checkpoint(
            goal_id=goal_id,
            attempt_id=attempt_id,
            agent_run_id=run_id,
            checkpoint_index=4,
            stage="teacher_generation",
            source="operator_test",
            payload={
                "checkpoint_index": 4,
                "stage": "teacher_generation",
                "checkpoint": {
                    "checkpoint_index": 4,
                    "stage": "teacher_generation",
                    "captured_at": checkpoint_captured_at,
                },
                "recovery_state": {
                    "status": "stalled",
                    "action": "resume",
                    "reason": "Teacher worker disconnected before downstream stages finished.",
                    "stage": "teacher_generation",
                    "checkpoint": {
                        "checkpoint_index": 4,
                        "stage": "teacher_generation",
                        "captured_at": checkpoint_captured_at,
                    },
                    "unfinished_steps": [
                        "Continue from the teacher checkpoint without replaying the full run."
                    ],
                    "role_task_snapshot": role_task_snapshot,
                },
            },
            metadata={"signature": "rehydrate-resume-checkpoint-1"},
            captured_at=checkpoint_captured_at,
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_memory_snapshot(
            goal_id=goal_id,
            attempt_id=attempt_id,
            checkpoint_id=checkpoint["id"],
            snapshot_kind="compact_recovery_v1",
            snapshot={
                "goal_objective": "Resume a stalled linked run from its durable goal checkpoint.",
                "attempt_id": attempt_id,
                "agent_run_id": run_id,
                "protocol_id": "teacher_student_distill",
                "agent_run_status": "stalled",
                "stage": "teacher_generation",
                "checkpoint_index": 4,
                "unfinished_steps": [
                    "Continue from the teacher checkpoint without replaying the full run."
                ],
                "captured_at": snapshot_captured_at,
            },
            metadata={"signature": "rehydrate-resume-memory-1"},
            captured_at=snapshot_captured_at,
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id=run_id,
            protocol_id="teacher_student_distill",
            title="Goal checkpoint rehydrate linked run",
            topic="goal checkpoint rehydrate",
            summary={
                "goal_id": goal_id,
                "goal_attempt_id": attempt_id,
                "objective": "Resume a stalled linked run from its durable goal checkpoint.",
                "task_input": "Resume a stalled linked run from its durable goal checkpoint.",
                "recovery_state": {
                    "status": "stalled",
                    "action": "resume",
                    "reason": "Teacher worker disconnected before downstream stages finished.",
                    "stage": "teacher_generation",
                    "checkpoint": {
                        "checkpoint_index": 4,
                        "stage": "teacher_generation",
                        "captured_at": checkpoint_captured_at,
                    },
                },
            },
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status(run_id, "stalled"))

    with TestClient(app) as client:
        resume_response = client.post(
            f"/v1/goals/{goal_id}/resume",
            json={"strategy": "continue_from_checkpoint"},
        )
        assert resume_response.status_code == 200

        completed_goal = _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)
        linked_run_response = client.get(f"/v1/agent-runs/{run_id}")
        assert linked_run_response.status_code == 200
        linked_run = linked_run_response.json()

    assert len(completed_goal["attempts"]) == 1
    assert len(captured_requests) == 1
    assert captured_requests[0].metadata["resume_strategy"] == "continue_from_checkpoint"
    resume_payload = captured_requests[0].metadata["resume_payload"]
    assert resume_payload["executor"] == "continue_from_checkpoint"
    assert resume_payload["role_task_snapshot"]["roles"]["teacher"]["status"] == "completed"
    assert linked_run["summary"]["goal_handoff"]["resume_gate"]["effective_strategy"] == (
        "continue_from_checkpoint"
    )
    assert linked_run["summary"]["goal_handoff"]["resume_payload_rehydrated_from"] == (
        "durable_goal_checkpoint"
    )


def test_goal_partial_generation_refresh_auto_resumes_as_restart_attempt(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    app, runtime_service = _create_goal_test_app(tmp_path)
    goal_id = "goal-auto-refresh-1"
    attempt_id = "goal-auto-refresh-attempt-1"
    run_id = "linked-auto-refresh-run-1"

    async def _noop_ensure_agent_run_job(self: Any, run_id: str, *, job_name: str) -> None:
        del self, run_id, job_name
        return None

    monkeypatch.setattr(RuntimeService, "_ensure_agent_run_job", _noop_ensure_agent_run_job)

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id=goal_id,
            objective="Auto-resume after a planned generation refresh partial stop.",
            protocol_id="teacher_student_distill",
            run_policy={
                "generation_refresh_interval_sec": 60,
                "max_wall_clock_sec": 600,
            },
            summary={"phase": "running"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id=attempt_id,
            goal_id=goal_id,
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id=run_id,
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            goal_id,
            "running",
            current_attempt_id=attempt_id,
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id=run_id,
            protocol_id="teacher_student_distill",
            title="Planned refresh partial run",
            topic="planned generation refresh",
            run_policy={
                "generation_refresh_interval_sec": 60,
                "max_wall_clock_sec": 600,
            },
            summary={
                "goal_id": goal_id,
                "goal_attempt_id": attempt_id,
                "objective": "Auto-resume after a planned generation refresh partial stop.",
                "task_input": "Auto-resume after a planned generation refresh partial stop.",
                "recovery_state": {
                    "status": "partial",
                    "action": "finalize_partial",
                    "reason": "Run exceeded max_wall_clock_sec=60.",
                    "stage": "teacher_generation",
                    "checkpoint": {
                        "checkpoint_index": 2,
                        "stage": "teacher_generation",
                        "task_input": "Auto-resume after a planned generation refresh partial stop.",
                    },
                    "unfinished_steps": ["resume from the latest checkpoint"],
                    "recommended_resume_conditions": ["continue with a fresh worker generation"],
                    "resume_payload": {
                        "stage": "teacher_generation",
                        "checkpoint": {
                            "checkpoint_index": 2,
                            "stage": "teacher_generation",
                            "task_input": "Auto-resume after a planned generation refresh partial stop.",
                        },
                        "executor": "continue_from_checkpoint",
                        "strategy_default": "continue_from_checkpoint",
                        "guidance_messages": ["Resume from the planned refresh checkpoint."],
                        "metadata_state": {},
                        "precomputed_artifacts": {},
                        "protocol_artifacts": {},
                        "candidates": [],
                        "evidence_packets": [],
                        "verifications": [],
                        "role_task_snapshot": {},
                    },
                },
            },
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status(run_id, "partial"))

    partial_run = asyncio.run(runtime_service._store.get_agent_run(run_id))
    assert partial_run is not None
    asyncio.run(
        runtime_service._sync_goal_from_agent_run(
            goal_id=goal_id,
            attempt_id=attempt_id,
            run=partial_run,
        )
    )

    goal_after = asyncio.run(runtime_service._store.get_goal(goal_id))
    run_after = asyncio.run(runtime_service._store.get_agent_run(run_id))
    assert goal_after is not None
    assert run_after is not None
    assert goal_after["status"] == "running"
    assert run_after["status"] == "running"
    recovery_state = dict(run_after["summary"]["recovery_state"])
    resume_runtime = dict(recovery_state["resume_runtime"])
    assert resume_runtime["status"] == "active"
    assert resume_runtime["strategy"] == "restart_attempt"
    assert resume_runtime["source"] == "goal_generation_refresh"

    with TestClient(app) as client:
        health_response = client.get(f"/v1/goals/{goal_id}/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()

    assert health_payload["status"] == "running"
    assert health_payload["linked_agent_run"]["status"] == "running"
    assert health_payload["persisted_checkpoint"]["checkpoint_index"] == 2
    assert health_payload["memory_snapshot"]["snapshot"]["checkpoint_index"] == 2


def test_goal_partial_checkpoint_cadence_auto_resumes_as_restart_attempt(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    app, runtime_service = _create_goal_test_app(tmp_path)
    goal_id = "goal-auto-checkpoint-1"
    attempt_id = "goal-auto-checkpoint-attempt-1"
    run_id = "linked-auto-checkpoint-run-1"

    async def _noop_ensure_agent_run_job(self: Any, run_id: str, *, job_name: str) -> None:
        del self, run_id, job_name
        return None

    monkeypatch.setattr(RuntimeService, "_ensure_agent_run_job", _noop_ensure_agent_run_job)

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id=goal_id,
            objective="Auto-resume after a checkpoint cadence partial stop.",
            protocol_id="teacher_student_distill",
            run_policy={
                "checkpoint_interval_sec": 60,
                "max_wall_clock_sec": 600,
            },
            summary={"phase": "running"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id=attempt_id,
            goal_id=goal_id,
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id=run_id,
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            goal_id,
            "running",
            current_attempt_id=attempt_id,
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id=run_id,
            protocol_id="teacher_student_distill",
            title="Checkpoint cadence partial run",
            topic="checkpoint cadence partial refresh",
            run_policy={
                "checkpoint_interval_sec": 60,
                "max_wall_clock_sec": 60,
                "on_budget_exhausted": "finalize_partial",
                "goal_checkpoint_cadence_owned": True,
                "goal_checkpoint_cadence_effective_sec": 60,
            },
            summary={
                "goal_id": goal_id,
                "goal_attempt_id": attempt_id,
                "objective": "Auto-resume after a checkpoint cadence partial stop.",
                "task_input": "Auto-resume after a checkpoint cadence partial stop.",
                "recovery_state": {
                    "status": "partial",
                    "action": "finalize_partial",
                    "reason": "Run exceeded max_wall_clock_sec=60.",
                    "stage": "teacher_generation",
                    "checkpoint": {
                        "checkpoint_index": 2,
                        "stage": "teacher_generation",
                        "task_input": "Auto-resume after a checkpoint cadence partial stop.",
                    },
                    "unfinished_steps": ["resume from the latest checkpoint"],
                    "recommended_resume_conditions": ["continue after persisting a checkpoint cadence refresh"],
                    "resume_payload": {
                        "stage": "teacher_generation",
                        "checkpoint": {
                            "checkpoint_index": 2,
                            "stage": "teacher_generation",
                            "task_input": "Auto-resume after a checkpoint cadence partial stop.",
                        },
                        "executor": "continue_from_checkpoint",
                        "strategy_default": "continue_from_checkpoint",
                        "guidance_messages": ["Resume from the checkpoint cadence partial checkpoint."],
                        "metadata_state": {},
                        "precomputed_artifacts": {},
                        "protocol_artifacts": {},
                        "candidates": [],
                        "evidence_packets": [],
                        "verifications": [],
                        "role_task_snapshot": {},
                    },
                },
            },
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status(run_id, "partial"))

    partial_run = asyncio.run(runtime_service._store.get_agent_run(run_id))
    assert partial_run is not None
    asyncio.run(
        runtime_service._sync_goal_from_agent_run(
            goal_id=goal_id,
            attempt_id=attempt_id,
            run=partial_run,
        )
    )

    goal_after = asyncio.run(runtime_service._store.get_goal(goal_id))
    run_after = asyncio.run(runtime_service._store.get_agent_run(run_id))
    assert goal_after is not None
    assert run_after is not None
    assert goal_after["status"] == "running"
    assert run_after["status"] == "running"
    recovery_state = dict(run_after["summary"]["recovery_state"])
    resume_runtime = dict(recovery_state["resume_runtime"])
    assert resume_runtime["status"] == "active"
    assert resume_runtime["strategy"] == "restart_attempt"
    assert resume_runtime["source"] == "goal_checkpoint_cadence_refresh"


def test_goal_partial_checkpoint_step_cadence_auto_resumes_as_restart_attempt(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    app, runtime_service = _create_goal_test_app(tmp_path)
    goal_id = "goal-auto-checkpoint-steps-1"
    attempt_id = "goal-auto-checkpoint-steps-attempt-1"
    run_id = "linked-auto-checkpoint-steps-run-1"

    async def _noop_ensure_agent_run_job(self: Any, run_id: str, *, job_name: str) -> None:
        del self, run_id, job_name
        return None

    monkeypatch.setattr(RuntimeService, "_ensure_agent_run_job", _noop_ensure_agent_run_job)

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id=goal_id,
            objective="Auto-resume after a checkpoint step cadence partial stop.",
            protocol_id="teacher_student_distill",
            run_policy={
                "checkpoint_interval_steps": 2,
                "max_wall_clock_sec": 600,
            },
            summary={"phase": "running"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id=attempt_id,
            goal_id=goal_id,
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id=run_id,
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            goal_id,
            "running",
            current_attempt_id=attempt_id,
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id=run_id,
            protocol_id="teacher_student_distill",
            title="Checkpoint step cadence partial run",
            topic="checkpoint step cadence partial refresh",
            run_policy={
                "checkpoint_interval_steps": 2,
                "max_wall_clock_sec": 600,
                "on_budget_exhausted": "finalize_partial",
                "goal_checkpoint_cadence_owned": True,
                "goal_checkpoint_cadence_effective_steps": 2,
            },
            summary={
                "goal_id": goal_id,
                "goal_attempt_id": attempt_id,
                "objective": "Auto-resume after a checkpoint step cadence partial stop.",
                "task_input": "Auto-resume after a checkpoint step cadence partial stop.",
                "recovery_state": {
                    "status": "partial",
                    "action": "finalize_partial",
                    "reason": "Run reached checkpoint_interval_steps=2 at checkpoint_index=2.",
                    "stage": "teacher_generation",
                    "checkpoint": {
                        "checkpoint_index": 2,
                        "stage": "teacher_generation",
                        "task_input": "Auto-resume after a checkpoint step cadence partial stop.",
                    },
                    "unfinished_steps": ["resume from the latest checkpoint"],
                    "recommended_resume_conditions": [
                        "continue after persisting a checkpoint step cadence refresh"
                    ],
                    "resume_payload": {
                        "stage": "teacher_generation",
                        "checkpoint": {
                            "checkpoint_index": 2,
                            "stage": "teacher_generation",
                            "task_input": "Auto-resume after a checkpoint step cadence partial stop.",
                        },
                        "executor": "continue_from_checkpoint",
                        "strategy_default": "continue_from_checkpoint",
                        "guidance_messages": [
                            "Resume from the checkpoint step cadence partial checkpoint."
                        ],
                        "metadata_state": {},
                        "precomputed_artifacts": {},
                        "protocol_artifacts": {},
                        "candidates": [],
                        "evidence_packets": [],
                        "verifications": [],
                        "role_task_snapshot": {},
                    },
                },
            },
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status(run_id, "partial"))

    partial_run = asyncio.run(runtime_service._store.get_agent_run(run_id))
    assert partial_run is not None
    asyncio.run(
        runtime_service._sync_goal_from_agent_run(
            goal_id=goal_id,
            attempt_id=attempt_id,
            run=partial_run,
        )
    )

    goal_after = asyncio.run(runtime_service._store.get_goal(goal_id))
    run_after = asyncio.run(runtime_service._store.get_agent_run(run_id))
    assert goal_after is not None
    assert run_after is not None
    assert goal_after["status"] == "running"
    assert run_after["status"] == "running"
    recovery_state = dict(run_after["summary"]["recovery_state"])
    resume_runtime = dict(recovery_state["resume_runtime"])
    assert resume_runtime["status"] == "active"
    assert resume_runtime["strategy"] == "restart_attempt"
    assert resume_runtime["source"] == "goal_checkpoint_cadence_refresh"

    with TestClient(app) as client:
        health_response = client.get(f"/v1/goals/{goal_id}/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()

    assert health_payload["status"] == "running"
    assert health_payload["checkpoint_policy"]["interval_steps"] == 2

    with TestClient(app) as client:
        health_response = client.get(f"/v1/goals/{goal_id}/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()

    assert health_payload["status"] == "running"
    assert health_payload["linked_agent_run"]["status"] == "running"
    assert health_payload["persisted_checkpoint"]["checkpoint_index"] == 2
    assert health_payload["memory_snapshot"]["snapshot"]["checkpoint_index"] == 2


def test_goal_auto_resumes_when_generic_exec_approval_is_resolved(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    approval_id = "exec-approval-goal-generic-resolve-1"

    async def _approval_then_success_run(self: Any, request: Any) -> MultiAgentRunResult:
        approval = self._exec_approval_store.get(approval_id)
        if approval is None or approval.status == "pending":
            if approval is None:
                self._exec_approval_store.create(
                    approval_id=approval_id,
                    command="print('goal generic approval resolve')",
                    shell="test",
                    scope="dangerous_command",
                    reason="Exec command requires approval.",
                    command_payload={
                        "command": "print('goal generic approval resolve')",
                        "shell": "test",
                        "workdir": str(tmp_path),
                        "env": None,
                        "timeout_sec": 5.0,
                        "background": False,
                        "tty": False,
                        "approval_state": "approved",
                    },
                )
            return MultiAgentRunResult(
                run_id=request.run_id,
                protocol="controlled_subagent_execution",
                state="awaiting_approval",
                task_input=request.task_input,
                candidates=[],
                selected_candidate_id=None,
                evaluation={},
                artifacts={
                    "final_answer": None,
                    "controlled_execution_runtime": {"approval_pending_count": 1},
                },
                events=[],
                metadata={
                    "approval_state": {
                        "status": "awaiting_approval",
                        "pending_count": 1,
                        "approval_ids": [approval_id],
                        "pending_approvals": [
                            {
                                "approval_id": approval_id,
                                "tool_name": "exec_command",
                                "request_id": "req-1",
                                "task_key": "controlled_execution_exec:req-1",
                                "source": "controlled_execution",
                            }
                        ],
                    },
                    "recovery_state": {
                        "status": "awaiting_approval",
                        "action": "await_approval",
                        "reason": "Execution approval required",
                        "stage": "controlled_execution_exec:req-1",
                        "checkpoint": {
                            "checkpoint_index": 4,
                            "stage": "controlled_execution_controller:req-1",
                        },
                    },
                },
            )
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="controlled_subagent_execution",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Goal generic approval resolve completed."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _approval_then_success_run)

    app = create_app()
    exec_approval_store = InMemoryApprovalStore()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
        exec_approval_store=exec_approval_store,
        exec_runtime=ExecRuntime(
            providers={"test": _GoalApiPythonDirectProvider()},
            default_shell="test",
        ),
    )
    runtime_service.set_scheduler_poll_interval(0.05)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Resolve a linked exec approval without opening a new attempt.",
                "title": "Goal generic approval resolve",
                "protocol_id": "controlled_subagent_execution",
            },
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200

        waiting_goal = _wait_goal_until(client, goal_id, {"waiting_approval"}, timeout_seconds=4.0)
        assert len(waiting_goal["attempts"]) == 1
        linked_run_id = waiting_goal["attempts"][0]["agent_run_id"]
        assert linked_run_id is not None

        resolve_response = client.post(
            f"/v1/approvals/{approval_id}/resolve",
            json={"decision": "approve_once", "reason": "allow linked goal auto resume"},
        )
        assert resolve_response.status_code == 200
        assert resolve_response.json()["status"] == "approved_once"

        completed_goal = _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)
        assert len(completed_goal["attempts"]) == 1
        assert completed_goal["attempts"][0]["agent_run_id"] == linked_run_id

    resolved = exec_approval_store.get(approval_id)
    assert resolved is not None
    assert resolved.status == "approved_once"
    assert resolved.execution_result is not None


def test_goal_linked_exec_approval_restart_resolve_reuses_existing_attempt(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    approval_id = "exec-approval-goal-restart-resolve-1"
    sessions_dir = tmp_path / "sessions"
    monkeypatch.setattr(
        "mochi.runtime.service.MultiAgentOrchestrator.run",
        _build_goal_linked_exec_approval_orchestrator(
            approval_id=approval_id,
            workdir=tmp_path,
            final_answer="Goal restart approval completed.",
        ),
    )

    with _create_goal_exec_test_client(
        sessions_dir=sessions_dir,
        exec_approval_store=InMemoryApprovalStore(),
    ) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Resume the original linked run after a runtime restart.",
                "title": "Goal restart approval resolve",
                "protocol_id": "controlled_subagent_execution",
            },
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200

        waiting_goal = _wait_goal_until(client, goal_id, {"waiting_approval"}, timeout_seconds=4.0)
        assert len(waiting_goal["attempts"]) == 1
        first_attempt = waiting_goal["attempts"][0]
        first_attempt_id = first_attempt["attempt_id"]
        linked_run_id = first_attempt["agent_run_id"]
        assert linked_run_id is not None

    restarted_exec_approval_store = InMemoryApprovalStore()
    with _create_goal_exec_test_client(
        sessions_dir=sessions_dir,
        exec_approval_store=restarted_exec_approval_store,
    ) as restarted_client:
        resolve_response = restarted_client.post(
            f"/v1/approvals/{approval_id}/resolve",
            json={"decision": "approve_once", "reason": "allow linked goal restart auto resume"},
        )
        assert resolve_response.status_code == 200
        assert resolve_response.json()["status"] == "approved_once"

        completed_goal = _wait_goal_until(
            restarted_client,
            goal_id,
            {"completed"},
            timeout_seconds=4.0,
        )
        assert len(completed_goal["attempts"]) == 1
        assert completed_goal["current_attempt_id"] == first_attempt_id
        assert completed_goal["attempts"][0]["attempt_id"] == first_attempt_id
        assert completed_goal["attempts"][0]["agent_run_id"] == linked_run_id

        linked_run_response = restarted_client.get(f"/v1/agent-runs/{linked_run_id}")
        assert linked_run_response.status_code == 200
        linked_run = linked_run_response.json()
        assert linked_run["status"] == "succeeded"
        assert linked_run["summary"]["final_answer"] == "Goal restart approval completed."

    resolved = restarted_exec_approval_store.get(approval_id)
    assert resolved is not None
    assert resolved.status == "approved_once"
    assert resolved.execution_result is not None


def test_goal_linked_exec_approval_restart_resolve_reuses_existing_attempt_from_stalled_run(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    approval_id = "exec-approval-goal-restart-resolve-stalled-1"
    sessions_dir = tmp_path / "sessions"
    monkeypatch.setattr(
        "mochi.runtime.service.MultiAgentOrchestrator.run",
        _build_goal_linked_exec_approval_orchestrator(
            approval_id=approval_id,
            workdir=tmp_path,
            final_answer="Goal restart approval completed from stalled run.",
        ),
    )

    with _create_goal_exec_test_client(
        sessions_dir=sessions_dir,
        exec_approval_store=InMemoryApprovalStore(),
    ) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Resume a linked run that drifted to stalled before restart.",
                "title": "Goal restart approval resolve from stalled run",
                "protocol_id": "controlled_subagent_execution",
            },
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200

        waiting_goal = _wait_goal_until(client, goal_id, {"waiting_approval"}, timeout_seconds=4.0)
        assert len(waiting_goal["attempts"]) == 1
        first_attempt = waiting_goal["attempts"][0]
        first_attempt_id = first_attempt["attempt_id"]
        linked_run_id = first_attempt["agent_run_id"]
        assert linked_run_id is not None

    restarted_store = RuntimeStore(sessions_dir / "runtime.db")
    asyncio.run(
        restarted_store.update_agent_run_status(
            linked_run_id,
            "stalled",
            latest_error="worker interrupted while waiting for approval",
        )
    )

    restarted_exec_approval_store = InMemoryApprovalStore()
    with _create_goal_exec_test_client(
        sessions_dir=sessions_dir,
        exec_approval_store=restarted_exec_approval_store,
    ) as restarted_client:
        resolve_response = restarted_client.post(
            f"/v1/approvals/{approval_id}/resolve",
            json={"decision": "approve_once", "reason": "allow linked goal restart auto resume"},
        )
        assert resolve_response.status_code == 200
        assert resolve_response.json()["status"] == "approved_once"

        completed_goal = _wait_goal_until(
            restarted_client,
            goal_id,
            {"completed"},
            timeout_seconds=4.0,
        )
        assert len(completed_goal["attempts"]) == 1
        assert completed_goal["current_attempt_id"] == first_attempt_id
        assert completed_goal["attempts"][0]["attempt_id"] == first_attempt_id
        assert completed_goal["attempts"][0]["agent_run_id"] == linked_run_id

        linked_run_response = restarted_client.get(f"/v1/agent-runs/{linked_run_id}")
        assert linked_run_response.status_code == 200
        linked_run = linked_run_response.json()
        assert linked_run["status"] == "succeeded"
        assert linked_run["summary"]["final_answer"] == (
            "Goal restart approval completed from stalled run."
        )

    resolved = restarted_exec_approval_store.get(approval_id)
    assert resolved is not None
    assert resolved.status == "approved_once"
    assert resolved.execution_result is not None


def test_goal_fails_when_generic_exec_approval_is_rejected(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    approval_id = "exec-approval-goal-generic-reject-1"

    async def _approval_wait_run(self: Any, request: Any) -> MultiAgentRunResult:
        approval = self._exec_approval_store.get(approval_id)
        if approval is None:
            self._exec_approval_store.create(
                approval_id=approval_id,
                command="print('goal generic approval reject')",
                shell="test",
                scope="dangerous_command",
                reason="Exec command requires approval.",
                command_payload={
                    "command": "print('goal generic approval reject')",
                    "shell": "test",
                    "workdir": str(tmp_path),
                    "env": None,
                    "timeout_sec": 5.0,
                    "background": False,
                    "tty": False,
                    "approval_state": "approved",
                },
            )
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="controlled_subagent_execution",
            state="awaiting_approval",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={
                "final_answer": None,
                "controlled_execution_runtime": {"approval_pending_count": 1},
            },
            events=[],
            metadata={
                "approval_state": {
                    "status": "awaiting_approval",
                    "pending_count": 1,
                    "approval_ids": [approval_id],
                    "pending_approvals": [
                        {
                            "approval_id": approval_id,
                            "tool_name": "exec_command",
                            "request_id": "req-1",
                            "task_key": "controlled_execution_exec:req-1",
                            "source": "controlled_execution",
                        }
                    ],
                },
                "recovery_state": {
                    "status": "awaiting_approval",
                    "action": "await_approval",
                    "reason": "Execution approval required",
                    "stage": "controlled_execution_exec:req-1",
                    "checkpoint": {
                        "checkpoint_index": 4,
                        "stage": "controlled_execution_controller:req-1",
                    },
                },
            },
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _approval_wait_run)

    app = create_app()
    exec_approval_store = InMemoryApprovalStore()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
        exec_approval_store=exec_approval_store,
        exec_runtime=ExecRuntime(
            providers={"test": _GoalApiPythonDirectProvider()},
            default_shell="test",
        ),
    )
    runtime_service.set_scheduler_poll_interval(0.05)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Reject the linked exec approval and fail the current attempt.",
                "title": "Goal generic approval reject",
                "protocol_id": "controlled_subagent_execution",
            },
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200

        waiting_goal = _wait_goal_until(client, goal_id, {"waiting_approval"}, timeout_seconds=4.0)
        assert len(waiting_goal["attempts"]) == 1
        linked_run_id = waiting_goal["attempts"][0]["agent_run_id"]
        assert linked_run_id is not None

        resolve_response = client.post(
            f"/v1/approvals/{approval_id}/resolve",
            json={"decision": "reject", "reason": "do not run this command"},
        )
        assert resolve_response.status_code == 200
        assert resolve_response.json()["status"] == "rejected"

        failed_goal = _wait_goal_until(client, goal_id, {"failed"}, timeout_seconds=4.0)
        assert len(failed_goal["attempts"]) == 1
        assert failed_goal["attempts"][0]["agent_run_id"] == linked_run_id

        linked_run_response = client.get(f"/v1/agent-runs/{linked_run_id}")
        assert linked_run_response.status_code == 200
        linked_run = linked_run_response.json()
        assert linked_run["status"] == "failed"
        assert linked_run["latest_error"] == "do not run this command"

    resolved = exec_approval_store.get(approval_id)
    assert resolved is not None
    assert resolved.status == "rejected"
    assert resolved.execution_result is None


def test_goals_api_flow_creates_and_tracks_linked_agent_runs(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured_requests: list[Any] = []

    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        captured_requests.append(request)
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Goal-linked agent run completed."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Collect multimodal datasets for long-running training jobs",
                "title": "Dataset Collection Goal",
                "goal_type": "dataset_collection",
                "protocol_id": "teacher_student_distill",
                "topic": "multimodal corpora",
                "project_id": "proj-goal",
                "workspace_dir": str(tmp_path / "workspace"),
                "run_policy": {"max_wall_clock_sec": 18_000},
                "capability_policy": {
                    "allowed_tools": [" web_search ", "web_fetch", "web_search", ""],
                    "operator_mode": "observe",
                },
                "source_manifest": {"sources": [{"id": "hf-datasets"}]},
                "summary": {"phase": "created"},
                "metadata": {"packet": "A"},
            },
        )
        assert create_response.status_code == 200
        created = create_response.json()
        goal_id = created["goal_id"]
        expected_capability_policy = {
            "operator_mode": "observe",
            "allowed_tools": ["web_search", "web_fetch"],
        }
        assert created["status"] == "created"
        assert created["capability_policy"] == expected_capability_policy
        assert created["attempts"] == []

        list_response = client.get("/v1/goals")
        assert list_response.status_code == 200
        listed = list_response.json()
        assert len(listed) == 1
        assert listed[0]["goal_id"] == goal_id
        assert listed[0]["capability_policy"] == expected_capability_policy

        get_response = client.get(f"/v1/goals/{goal_id}")
        assert get_response.status_code == 200
        fetched = get_response.json()
        assert fetched["objective"] == "Collect multimodal datasets for long-running training jobs"
        assert fetched["run_policy"]["max_wall_clock_sec"] == 18_000
        assert fetched["capability_policy"] == expected_capability_policy

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200
        started = start_response.json()
        assert started["status"] in {"queued", "running", "completed"}
        assert len(started["attempts"]) == 1
        first_attempt = started["attempts"][0]
        assert started["current_attempt_id"] == first_attempt["attempt_id"]
        assert first_attempt["trigger"] == "manual_start"
        assert first_attempt["agent_run_id"] is not None

        completed_goal = _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)
        assert completed_goal["attempts"][0]["status"] == "completed"
        linked_run_id = completed_goal["attempts"][0]["agent_run_id"]
        assert linked_run_id is not None

        health_response = client.get(f"/v1/goals/{goal_id}/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()
        assert health_payload["status"] == "completed"
        assert health_payload["capability_policy"] == expected_capability_policy
        assert health_payload["lease"] is None

        linked_run_response = client.get(f"/v1/agent-runs/{linked_run_id}")
        assert linked_run_response.status_code == 200
        linked_run = linked_run_response.json()
        assert linked_run["status"] == "succeeded"
        assert linked_run["summary"]["goal_id"] == goal_id
        assert linked_run["summary"]["goal_attempt_id"] == first_attempt["attempt_id"]
        assert linked_run["summary"]["goal_capability_policy"] == expected_capability_policy

        agent_runs = client.get("/v1/agent-runs")
        assert agent_runs.status_code == 200
        assert len(agent_runs.json()) == 1

    assert len(captured_requests) == 1
    assert captured_requests[0].metadata["goal_capability_policy"] == expected_capability_policy


def test_goal_startup_recovery_repairs_claimed_linked_run_without_stalling(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Recovered claimed linked run."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    app, runtime_service = _create_goal_test_app(tmp_path)
    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-repair-1",
            objective="Repair a claimed linked run after supervisor recovery",
            protocol_id="teacher_student_distill",
            summary={"phase": "running"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-repair-attempt-1",
            goal_id="goal-repair-1",
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id="linked-repair-run-1",
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-repair-1",
            "running",
            current_attempt_id="goal-repair-attempt-1",
        )
    )

    with TestClient(app) as client:
        completed_goal = _wait_goal_until(client, "goal-repair-1", {"completed"}, timeout_seconds=4.0)
        assert len(completed_goal["attempts"]) == 1
        assert completed_goal["attempts"][0]["attempt_id"] == "goal-repair-attempt-1"
        assert completed_goal["attempts"][0]["agent_run_id"] == "linked-repair-run-1"

        linked_run_response = client.get("/v1/agent-runs/linked-repair-run-1")
        assert linked_run_response.status_code == 200
        linked_run = linked_run_response.json()
        assert linked_run["status"] == "succeeded"
        assert linked_run["summary"]["goal_id"] == "goal-repair-1"
        assert linked_run["summary"]["goal_attempt_id"] == "goal-repair-attempt-1"

        health_response = client.get("/v1/goals/goal-repair-1/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()
        assert health_payload["status"] == "completed"
        finding_codes = [item["finding_code"] for item in health_payload["open_findings"]]
        assert "linked_run_missing" not in finding_codes


def test_goal_pause_resume_and_cancel_manage_attempts_without_auto_relaunching_previous_runs(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    async def _slow_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        await asyncio.sleep(5)
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "slow"},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _slow_run)

    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Collect forum corpora with operator controls",
                "title": "Controllable Goal",
                "execution_mode": "single_agent",
                "protocol_id": "teacher_student_distill",
            },
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200
        started = start_response.json()
        assert started["execution_mode"] == "single_agent"
        assert len(started["attempts"]) == 1

        running_goal = _wait_goal_until(client, goal_id, {"running"}, timeout_seconds=4.0)
        assert running_goal["execution_mode"] == "single_agent"
        assert running_goal["attempts"][0]["agent_run_id"] is not None

        pause_response = client.post(f"/v1/goals/{goal_id}/pause")
        assert pause_response.status_code == 200
        paused = pause_response.json()
        assert paused["execution_mode"] == "single_agent"
        assert paused["status"] == "paused"
        assert paused["attempts"][0]["status"] == "paused"

        resume_response = client.post(f"/v1/goals/{goal_id}/resume")
        assert resume_response.status_code == 200
        resumed = resume_response.json()
        assert resumed["execution_mode"] == "single_agent"
        assert resumed["status"] == "running"
        assert len(resumed["attempts"]) == 1
        resumed_attempt = resumed["attempts"][0]
        assert resumed["current_attempt_id"] == resumed_attempt["attempt_id"]
        assert resumed_attempt["status"] == "running"

        cancel_response = client.post(f"/v1/goals/{goal_id}/cancel")
        assert cancel_response.status_code == 200
        cancelled = cancel_response.json()
        assert cancelled["execution_mode"] == "single_agent"
        assert cancelled["status"] == "cancelled"
        assert cancelled["attempts"][-1]["status"] == "cancelled"


def test_goals_api_returns_404_for_missing_goal(tmp_path: Path) -> None:
    with _create_goal_test_client(tmp_path) as client:
        response = client.get("/v1/goals/missing-goal")
        assert response.status_code == 404
        assert response.json()["detail"] == "Goal not found"


def test_goal_audit_finding_resolve_endpoint_updates_health_and_open_findings(
    tmp_path: Path,
) -> None:
    app, runtime_service = _create_goal_test_app(tmp_path)
    finding = _create_goal_audit_finding(
        runtime_service,
        goal_id="goal-audit-resolve-1",
        objective="Resolve an operator finding from the goal dashboard",
        finding_code="stale_running",
        summary="Goal lease became stale and required operator review.",
        details={"previous_owner_id": "runtime-owner-1"},
    )

    with TestClient(app) as client:
        health_before_response = client.get("/v1/goals/goal-audit-resolve-1/health")
        assert health_before_response.status_code == 200
        health_before = health_before_response.json()
        assert [item["finding_id"] for item in health_before["open_findings"]] == [finding["id"]]
        assert health_before["open_findings"][0]["status"] == "open"

        resolve_response = client.post(
            f"/v1/goals/goal-audit-resolve-1/audit-findings/{finding['id']}/resolve"
        )
        assert resolve_response.status_code == 200
        resolved_payload = resolve_response.json()

        health_after_response = client.get("/v1/goals/goal-audit-resolve-1/health")
        assert health_after_response.status_code == 200
        health_after = health_after_response.json()

    assert resolved_payload["finding_id"] == finding["id"]
    assert resolved_payload["goal_id"] == "goal-audit-resolve-1"
    assert resolved_payload["finding_code"] == "stale_running"
    assert resolved_payload["status"] == "resolved"
    assert resolved_payload["resolved_at"] is not None
    assert resolved_payload["closed_at"] is None
    assert health_after["open_findings"] == []


def test_goal_audit_finding_close_endpoint_removes_only_closed_finding_from_open_list(
    tmp_path: Path,
) -> None:
    app, runtime_service = _create_goal_test_app(tmp_path)
    closed_target = _create_goal_audit_finding(
        runtime_service,
        goal_id="goal-audit-close-1",
        objective="Close a completed operator finding from the goal dashboard",
        finding_code="linked_run_interrupted",
        summary="A linked run was interrupted and later confirmed handled.",
        details={"run_id": "linked-run-close-1"},
    )
    remaining_open = asyncio.run(
        runtime_service._store.upsert_goal_audit_finding(
            goal_id="goal-audit-close-1",
            finding_code="retry_limit_reached",
            summary="Retry budget still needs operator action.",
            details={"attempts_used": 4},
        )
    )

    with TestClient(app) as client:
        close_response = client.post(
            f"/v1/goals/goal-audit-close-1/audit-findings/{closed_target['id']}/close"
        )
        assert close_response.status_code == 200
        closed_payload = close_response.json()

        health_response = client.get("/v1/goals/goal-audit-close-1/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()

    open_finding_ids = [item["finding_id"] for item in health_payload["open_findings"]]
    assert closed_payload["finding_id"] == closed_target["id"]
    assert closed_payload["status"] == "closed"
    assert closed_payload["resolved_at"] is not None
    assert closed_payload["closed_at"] is not None
    assert closed_target["id"] not in open_finding_ids
    assert open_finding_ids == [remaining_open["id"]]


def test_goal_audit_finding_operator_endpoints_return_404_for_missing_goal_or_finding(
    tmp_path: Path,
) -> None:
    app, runtime_service = _create_goal_test_app(tmp_path)
    finding = _create_goal_audit_finding(
        runtime_service,
        goal_id="goal-audit-errors-1",
        objective="Verify operator error handling for goal audit findings",
        finding_code="stale_running",
        summary="Goal lease became stale during supervision.",
    )
    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-audit-errors-2",
            objective="Second goal for cross-goal finding checks",
            summary={"phase": "operator_review"},
        )
    )

    with TestClient(app) as client:
        missing_goal_response = client.post(
            "/v1/goals/missing-goal/audit-findings/999/resolve"
        )
        assert missing_goal_response.status_code == 404
        assert missing_goal_response.json()["detail"] == "Goal not found"

        missing_finding_response = client.post(
            "/v1/goals/goal-audit-errors-1/audit-findings/999/close"
        )
        assert missing_finding_response.status_code == 404
        assert missing_finding_response.json()["detail"] == "Goal audit finding not found"

        foreign_finding_response = client.post(
            f"/v1/goals/goal-audit-errors-2/audit-findings/{finding['id']}/resolve"
        )
        assert foreign_finding_response.status_code == 404
        assert foreign_finding_response.json()["detail"] == "Goal audit finding not found"


def test_goal_startup_recovery_takes_over_stale_lease(tmp_path: Path) -> None:
    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    runtime_service.set_goal_lease_ttl_seconds(30)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    stale_timestamp = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-recovery-1",
            objective="Recover a long-running goal after restart",
            summary={"phase": "running"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-recovery-attempt-1",
            goal_id="goal-recovery-1",
            attempt_index=1,
            status="running",
            trigger="manual_start",
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-recovery-1",
            "running",
            current_attempt_id="goal-recovery-attempt-1",
        )
    )
    asyncio.run(
        runtime_service._store.upsert_goal_lease(
            goal_id="goal-recovery-1",
            owner_id="runtime-stale-owner",
            metadata={"reason": "old_runtime"},
            acquired_at=stale_timestamp,
            heartbeat_at=stale_timestamp,
            expires_at=stale_timestamp,
            force_takeover=True,
        )
    )

    with TestClient(app) as client:
        health_response = client.get("/v1/goals/goal-recovery-1/health")
        assert health_response.status_code == 200
        payload = health_response.json()

    assert payload["status"] == "running"
    assert payload["lease"]["owner_id"] != "runtime-stale-owner"
    assert payload["lease"]["owned_by_runtime"] is True
    assert payload["lease"]["stale"] is False
    finding_codes = [item["finding_code"] for item in payload["open_findings"]]
    assert "stale_running" in finding_codes


def test_goal_supervision_does_not_take_over_fresh_foreign_lease(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    db_path = tmp_path / "sessions" / "runtime.db"
    runtime_a = RuntimeService(engine=object(), store=RuntimeStore(db_path))
    runtime_b = RuntimeService(engine=object(), store=RuntimeStore(db_path))

    now = datetime.now(UTC).isoformat()
    future_expires_at = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
    asyncio.run(
        runtime_a._store.create_goal(
            goal_id="goal-lease-service-1",
            objective="Keep a fresh foreign supervisor lease",
            summary={"phase": "running"},
        )
    )
    asyncio.run(
        runtime_a._store.create_goal_attempt(
            attempt_id="goal-lease-attempt-1",
            goal_id="goal-lease-service-1",
            attempt_index=1,
            status="running",
            trigger="manual_start",
        )
    )
    asyncio.run(
        runtime_a._store.update_goal_status(
            "goal-lease-service-1",
            "running",
            current_attempt_id="goal-lease-attempt-1",
        )
    )
    asyncio.run(
        runtime_a._store.upsert_goal_lease(
            goal_id="goal-lease-service-1",
            owner_id=runtime_a._runtime_owner_id,
            metadata={"reason": "runtime-a"},
            acquired_at=now,
            heartbeat_at=now,
            expires_at=future_expires_at,
        )
    )

    drive_calls = {"count": 0}

    async def _count_drive(goal: dict[str, Any]) -> None:
        del goal
        drive_calls["count"] += 1

    monkeypatch.setattr(runtime_b, "_drive_goal_attempt_execution", _count_drive)

    asyncio.run(runtime_b._process_goal_supervision())

    lease = asyncio.run(runtime_b._store.get_goal_lease("goal-lease-service-1"))
    assert lease is not None
    assert lease["owner_id"] == runtime_a._runtime_owner_id
    assert drive_calls["count"] == 0


def test_goal_startup_recovery_resumes_linked_stalled_agent_run(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Recovered linked run completed."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    runtime_service.set_goal_lease_ttl_seconds(30)
    runtime_service.set_scheduler_poll_interval(0.05)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    stale_timestamp = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-recovery-2",
            objective="Resume a linked stalled run after runtime restart",
            protocol_id="teacher_student_distill",
            summary={"phase": "running"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-recovery-attempt-2",
            goal_id="goal-recovery-2",
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id="linked-run-2",
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-recovery-2",
            "running",
            current_attempt_id="goal-recovery-attempt-2",
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id="linked-run-2",
            protocol_id="teacher_student_distill",
            title="Recovered linked run",
            topic="restart recovery",
            summary={
                "goal_id": "goal-recovery-2",
                "goal_attempt_id": "goal-recovery-attempt-2",
                "objective": "Resume a linked stalled run after runtime restart",
                "task_input": "Resume a linked stalled run after runtime restart",
            },
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status("linked-run-2", "stalled"))
    asyncio.run(
        runtime_service._store.upsert_goal_lease(
            goal_id="goal-recovery-2",
            owner_id="runtime-stale-owner",
            metadata={"reason": "old_runtime"},
            acquired_at=stale_timestamp,
            heartbeat_at=stale_timestamp,
            expires_at=stale_timestamp,
            force_takeover=True,
        )
    )

    with TestClient(app) as client:
        completed_goal = _wait_goal_until(client, "goal-recovery-2", {"completed"}, timeout_seconds=4.0)
        linked_run = client.get("/v1/agent-runs/linked-run-2")
        assert linked_run.status_code == 200
        linked_run_payload = linked_run.json()

    assert completed_goal["attempts"][0]["agent_run_id"] == "linked-run-2"
    assert completed_goal["attempts"][0]["status"] == "completed"
    assert linked_run_payload["status"] == "succeeded"
    assert linked_run_payload["summary"]["final_answer"] == "Recovered linked run completed."


def test_goal_startup_recovery_marks_orphan_running_linked_run_stalled(tmp_path: Path) -> None:
    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    runtime_service.set_goal_lease_ttl_seconds(30)
    runtime_service.set_scheduler_poll_interval(0.05)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    stale_timestamp = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-recovery-3",
            objective="Detect an orphan running linked run after runtime restart",
            protocol_id="teacher_student_distill",
            summary={"phase": "running"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-recovery-attempt-3",
            goal_id="goal-recovery-3",
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id="linked-run-3",
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-recovery-3",
            "running",
            current_attempt_id="goal-recovery-attempt-3",
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id="linked-run-3",
            protocol_id="teacher_student_distill",
            title="Orphan running linked run",
            topic="restart recovery",
            summary={
                "goal_id": "goal-recovery-3",
                "goal_attempt_id": "goal-recovery-attempt-3",
                "objective": "Detect an orphan running linked run after runtime restart",
                "task_input": "Detect an orphan running linked run after runtime restart",
            },
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status("linked-run-3", "running"))
    asyncio.run(
        runtime_service._store.upsert_goal_lease(
            goal_id="goal-recovery-3",
            owner_id="runtime-stale-owner",
            metadata={"reason": "old_runtime"},
            acquired_at=stale_timestamp,
            heartbeat_at=stale_timestamp,
            expires_at=stale_timestamp,
            force_takeover=True,
        )
    )

    with TestClient(app) as client:
        stalled_goal = _wait_goal_until(client, "goal-recovery-3", {"stalled"}, timeout_seconds=4.0)
        health_response = client.get("/v1/goals/goal-recovery-3/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()
        linked_run = client.get("/v1/agent-runs/linked-run-3")
        assert linked_run.status_code == 200
        linked_run_payload = linked_run.json()

    assert stalled_goal["attempts"][0]["status"] == "stalled"
    assert linked_run_payload["status"] == "stalled"
    finding_codes = [item["finding_code"] for item in health_payload["open_findings"]]
    assert "linked_run_interrupted" in finding_codes


def test_goal_startup_recovery_defers_first_orphan_running_probe_before_marking_stalled(
    tmp_path: Path,
) -> None:
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    runtime_service.set_goal_lease_ttl_seconds(30)

    stale_timestamp = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-recovery-3b",
            objective="Defer the first orphan-running probe before stalling",
            protocol_id="teacher_student_distill",
            summary={"phase": "running"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-recovery-attempt-3b",
            goal_id="goal-recovery-3b",
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id="linked-run-3b",
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-recovery-3b",
            "running",
            current_attempt_id="goal-recovery-attempt-3b",
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id="linked-run-3b",
            protocol_id="teacher_student_distill",
            title="Deferred orphan running linked run",
            topic="restart recovery",
            summary={
                "goal_id": "goal-recovery-3b",
                "goal_attempt_id": "goal-recovery-attempt-3b",
                "objective": "Defer the first orphan-running probe before stalling",
                "task_input": "Defer the first orphan-running probe before stalling",
            },
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status("linked-run-3b", "running"))
    asyncio.run(
        runtime_service._store.upsert_goal_lease(
            goal_id="goal-recovery-3b",
            owner_id="runtime-stale-owner",
            metadata={"reason": "old_runtime"},
            acquired_at=stale_timestamp,
            heartbeat_at=stale_timestamp,
            expires_at=stale_timestamp,
            force_takeover=True,
        )
    )

    asyncio.run(runtime_service._process_goal_supervision())

    goal_after_first = asyncio.run(runtime_service._store.get_goal("goal-recovery-3b"))
    run_after_first = asyncio.run(runtime_service._store.get_agent_run("linked-run-3b"))
    lease_after_first = asyncio.run(runtime_service._store.get_goal_lease("goal-recovery-3b"))

    assert goal_after_first is not None
    assert run_after_first is not None
    assert lease_after_first is not None
    assert goal_after_first["status"] == "running"
    assert run_after_first["status"] == "running"
    assert lease_after_first["owner_id"] == runtime_service._runtime_owner_id
    assert (
        lease_after_first["metadata"]["running_without_worker_probe"]["run_id"]
        == "linked-run-3b"
    )

    asyncio.run(runtime_service._process_goal_supervision())

    goal_after_second = asyncio.run(runtime_service._store.get_goal("goal-recovery-3b"))
    run_after_second = asyncio.run(runtime_service._store.get_agent_run("linked-run-3b"))

    assert goal_after_second is not None
    assert run_after_second is not None
    assert goal_after_second["status"] == "stalled"
    assert run_after_second["status"] == "stalled"


def test_goal_supervision_serializes_overlapping_supervisor_passes(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    _, runtime_service = _create_goal_test_app(tmp_path)
    state = {
        "invocations": 0,
        "active": 0,
        "max_active": 0,
    }

    async def _exercise() -> None:
        first_entered = asyncio.Event()
        release_first = asyncio.Event()

        async def _noop_heartbeat() -> None:
            return None

        async def _noop_reconcile(*, reason: str) -> None:
            del reason
            return None

        async def _gated_process_owned_goal_runs() -> None:
            state["invocations"] += 1
            state["active"] += 1
            state["max_active"] = max(state["max_active"], state["active"])
            if state["invocations"] == 1:
                first_entered.set()
                await release_first.wait()
            else:
                await asyncio.sleep(0)
            state["active"] -= 1

        monkeypatch.setattr(runtime_service, "_heartbeat_owned_goal_leases", _noop_heartbeat)
        monkeypatch.setattr(runtime_service, "_reconcile_goal_supervision", _noop_reconcile)
        monkeypatch.setattr(
            runtime_service,
            "_process_owned_goal_runs",
            _gated_process_owned_goal_runs,
        )

        first = asyncio.create_task(runtime_service._process_goal_supervision())
        await first_entered.wait()

        second = asyncio.create_task(runtime_service._process_goal_supervision())
        await asyncio.sleep(0.05)

        assert not second.done()
        assert state["invocations"] == 1

        release_first.set()
        await asyncio.gather(first, second)

    asyncio.run(_exercise())

    assert state["invocations"] == 2
    assert state["max_active"] == 1


def test_goal_startup_recovery_immediately_drives_recovered_goal_without_scheduler_delay(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Recovered goal completed immediately."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    runtime_service.set_goal_lease_ttl_seconds(30)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    stale_timestamp = (datetime.now(UTC) - timedelta(minutes=10)).isoformat()
    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-recovery-4",
            objective="Immediately recover a linked stalled run after runtime restart",
            protocol_id="teacher_student_distill",
            summary={"phase": "running"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-recovery-attempt-4",
            goal_id="goal-recovery-4",
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id="linked-run-4",
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-recovery-4",
            "running",
            current_attempt_id="goal-recovery-attempt-4",
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id="linked-run-4",
            protocol_id="teacher_student_distill",
            title="Recovered linked run",
            topic="restart recovery immediate drive",
            summary={
                "goal_id": "goal-recovery-4",
                "goal_attempt_id": "goal-recovery-attempt-4",
                "objective": "Immediately recover a linked stalled run after runtime restart",
                "task_input": "Immediately recover a linked stalled run after runtime restart",
            },
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status("linked-run-4", "stalled"))
    asyncio.run(
        runtime_service._store.upsert_goal_lease(
            goal_id="goal-recovery-4",
            owner_id="runtime-stale-owner",
            metadata={"reason": "old_runtime"},
            acquired_at=stale_timestamp,
            heartbeat_at=stale_timestamp,
            expires_at=stale_timestamp,
            force_takeover=True,
        )
    )

    with TestClient(app) as client:
        completed_goal = _wait_goal_until(client, "goal-recovery-4", {"completed"}, timeout_seconds=2.0)
        linked_run = client.get("/v1/agent-runs/linked-run-4")
        assert linked_run.status_code == 200
        linked_run_payload = linked_run.json()

    assert completed_goal["status"] == "completed"
    assert linked_run_payload["status"] == "succeeded"


def test_goal_start_blocks_when_runtime_budget_hard_stop_already_passed(tmp_path: Path) -> None:
    with _create_goal_test_client(tmp_path) as client:
        hard_stop_at = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Do not start after the hard stop passes.",
                "run_policy": {"hard_stop_at": hard_stop_at},
            },
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200
        payload = start_response.json()

        assert payload["status"] == "awaiting_resources"
        assert payload["attempts"] == []
        assert payload["latest_error"] == f"Goal hard stop reached at {hard_stop_at}."

        health_response = client.get(f"/v1/goals/{goal_id}/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()
        assert health_payload["runtime_budget"]["status"] == "hard_stop_reached"
        finding_codes = [item["finding_code"] for item in health_payload["open_findings"]]
        assert "runtime_budget_exhausted" in finding_codes


def test_goal_resume_blocks_when_retry_budget_exhausted(tmp_path: Path) -> None:
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

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-retry-budget-1",
            objective="Retry limit should block a new attempt.",
            run_policy={"max_attempt_retries": 1},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-retry-budget-attempt-1",
            goal_id="goal-retry-budget-1",
            attempt_index=1,
            status="failed",
            trigger="manual_start",
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-retry-budget-attempt-2",
            goal_id="goal-retry-budget-1",
            attempt_index=2,
            status="failed",
            trigger="manual_resume",
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-retry-budget-1",
            "failed",
            current_attempt_id="goal-retry-budget-attempt-2",
        )
    )

    with TestClient(app) as client:
        resume_response = client.post("/v1/goals/goal-retry-budget-1/resume")
        assert resume_response.status_code == 200
        payload = resume_response.json()
        health_response = client.get("/v1/goals/goal-retry-budget-1/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()

    assert payload["status"] == "failed"
    assert len(payload["attempts"]) == 2
    assert payload["latest_error"] == "Goal retry budget exhausted (1/1 retries used)."
    assert health_payload["runtime_budget"]["status"] == "retry_limit_reached"
    assert health_payload["runtime_budget"]["retry_limit_reached"] is True
    finding_codes = [item["finding_code"] for item in health_payload["open_findings"]]
    assert "runtime_budget_exhausted" in finding_codes


def test_goal_supervisor_stops_active_goal_when_requested_duration_is_exhausted(
    tmp_path: Path,
) -> None:
    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    runtime_service.set_goal_lease_ttl_seconds(30)
    runtime_service.set_scheduler_poll_interval(0.05)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-runtime-budget-1",
            objective="Supervisor should stop this goal after the requested duration is exhausted.",
            run_policy={"requested_duration_sec": 60},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-runtime-budget-attempt-1",
            goal_id="goal-runtime-budget-1",
            attempt_index=1,
            status="queued",
            trigger="manual_start",
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-runtime-budget-1",
            "running",
            current_attempt_id="goal-runtime-budget-attempt-1",
        )
    )
    stale_started_at = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    _set_goal_started_at(
        tmp_path / "sessions" / "runtime.db",
        "goal-runtime-budget-1",
        stale_started_at,
    )
    now = datetime.now(UTC).isoformat()
    expires_at = (datetime.now(UTC) + timedelta(minutes=2)).isoformat()
    asyncio.run(
        runtime_service._store.upsert_goal_lease(
            goal_id="goal-runtime-budget-1",
            owner_id=runtime_service._runtime_owner_id,
            metadata={"reason": "runtime_budget_test"},
            acquired_at=now,
            heartbeat_at=now,
            expires_at=expires_at,
        )
    )

    with TestClient(app) as client:
        payload = _wait_goal_until(client, "goal-runtime-budget-1", {"awaiting_resources"}, timeout_seconds=2.0)
        health_response = client.get("/v1/goals/goal-runtime-budget-1/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()

    assert payload["attempts"][0]["status"] == "awaiting_resources"
    assert payload["attempts"][0]["agent_run_id"] is None
    assert health_payload["runtime_budget"]["status"] == "duration_exhausted"
    assert health_payload["runtime_budget"]["requested_duration_sec"] == 60
    finding_codes = [item["finding_code"] for item in health_payload["open_findings"]]
    assert "runtime_budget_exhausted" in finding_codes


def test_goal_supervisor_opens_and_resolves_generation_refresh_overdue_report_only(
    tmp_path: Path,
) -> None:
    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    runtime_service.set_goal_lease_ttl_seconds(30)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-generation-refresh-1",
            objective="Keep the linked run active while reporting an overdue generation refresh.",
            protocol_id="teacher_student_distill",
            run_policy={"generation_refresh_interval_sec": 60},
            summary={"phase": "running"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-generation-refresh-attempt-1",
            goal_id="goal-generation-refresh-1",
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id="linked-generation-refresh-run-1",
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-generation-refresh-1",
            "running",
            current_attempt_id="goal-generation-refresh-attempt-1",
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id="linked-generation-refresh-run-1",
            protocol_id="teacher_student_distill",
            title="Generation refresh overdue linked run",
            topic="generation refresh overdue",
            summary={
                "goal_id": "goal-generation-refresh-1",
                "goal_attempt_id": "goal-generation-refresh-attempt-1",
                "objective": (
                    "Keep the linked run active while reporting an overdue generation refresh."
                ),
                "task_input": (
                    "Keep the linked run active while reporting an overdue generation refresh."
                ),
            },
        )
    )
    asyncio.run(
        runtime_service._store.update_agent_run_status("linked-generation-refresh-run-1", "running")
    )
    stale_started_at = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    asyncio.run(
        runtime_service._store.create_goal_worker_generation(
            goal_id="goal-generation-refresh-1",
            attempt_id="goal-generation-refresh-attempt-1",
            agent_run_id="linked-generation-refresh-run-1",
            generation_index=1,
            status="running",
            started_at=stale_started_at,
        )
    )
    now = datetime.now(UTC).isoformat()
    expires_at = (datetime.now(UTC) + timedelta(minutes=2)).isoformat()
    asyncio.run(
        runtime_service._store.upsert_goal_lease(
            goal_id="goal-generation-refresh-1",
            owner_id=runtime_service._runtime_owner_id,
            metadata={"reason": "generation_refresh_test"},
            acquired_at=now,
            heartbeat_at=now,
            expires_at=expires_at,
        )
    )

    original_is_live = runtime_service._agent_run_job_is_live
    runtime_service._agent_run_job_is_live = lambda run_id: run_id == "linked-generation-refresh-run-1"
    try:
        asyncio.run(runtime_service._process_goal_supervision())
    finally:
        runtime_service._agent_run_job_is_live = original_is_live

    goal_after_open = asyncio.run(runtime_service._store.get_goal("goal-generation-refresh-1"))
    run_after_open = asyncio.run(
        runtime_service._store.get_agent_run("linked-generation-refresh-run-1")
    )
    findings_after_open = asyncio.run(
        runtime_service._store.list_goal_audit_findings(
            "goal-generation-refresh-1",
            status="open",
        )
    )
    with TestClient(app) as client:
        health_open_response = client.get("/v1/goals/goal-generation-refresh-1/health")
        assert health_open_response.status_code == 200
        health_after_open = health_open_response.json()

    assert goal_after_open is not None
    assert run_after_open is not None
    assert goal_after_open["status"] == "running"
    assert run_after_open["status"] == "running"
    assert health_after_open["status"] == "running"
    assert health_after_open["current_generation"]["status"] == "running"
    assert health_after_open["current_generation"]["refresh_due"] is True
    assert health_after_open["current_generation"]["refresh_overdue_sec"] >= 240
    assert [item["finding_code"] for item in findings_after_open] == [
        "generation_refresh_overdue"
    ]

    asyncio.run(
        runtime_service._store.update_agent_run_status(
            "linked-generation-refresh-run-1",
            "succeeded",
        )
    )
    completed_run = asyncio.run(
        runtime_service._store.get_agent_run("linked-generation-refresh-run-1")
    )
    assert completed_run is not None
    asyncio.run(
        runtime_service._sync_goal_from_agent_run(
            goal_id="goal-generation-refresh-1",
            attempt_id="goal-generation-refresh-attempt-1",
            run=completed_run,
        )
    )

    with TestClient(app) as client:
        health_response = client.get("/v1/goals/goal-generation-refresh-1/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()

    goal_after_resolve = asyncio.run(runtime_service._store.get_goal("goal-generation-refresh-1"))
    findings_after_resolve = asyncio.run(
        runtime_service._store.list_goal_audit_findings(
            "goal-generation-refresh-1",
            status="open",
        )
    )

    assert goal_after_resolve is not None
    assert goal_after_resolve["status"] == "completed"
    assert health_payload["status"] == "completed"
    assert health_payload["open_findings"] == []
    assert findings_after_resolve == []


def test_goal_supervisor_opens_and_resolves_generation_token_refresh_due_report_only(
    tmp_path: Path,
) -> None:
    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    runtime_service.set_goal_lease_ttl_seconds(30)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-token-refresh-1",
            objective="Keep the linked run active while reporting an overdue token refresh generation.",
            protocol_id="teacher_student_distill",
            run_policy={"generation_token_refresh_threshold": 30},
            summary={"phase": "running"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-token-refresh-attempt-1",
            goal_id="goal-token-refresh-1",
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id="linked-token-refresh-run-1",
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-token-refresh-1",
            "running",
            current_attempt_id="goal-token-refresh-attempt-1",
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id="linked-token-refresh-run-1",
            protocol_id="teacher_student_distill",
            title="Token refresh due linked run",
            topic="generation token refresh due",
            summary={
                "goal_id": "goal-token-refresh-1",
                "goal_attempt_id": "goal-token-refresh-attempt-1",
                "objective": (
                    "Keep the linked run active while reporting an overdue token refresh generation."
                ),
                "task_input": (
                    "Keep the linked run active while reporting an overdue token refresh generation."
                ),
            },
        )
    )
    asyncio.run(
        runtime_service._store.update_agent_run_status("linked-token-refresh-run-1", "running")
    )
    asyncio.run(
        runtime_service._store.create_goal_worker_generation(
            goal_id="goal-token-refresh-1",
            attempt_id="goal-token-refresh-attempt-1",
            agent_run_id="linked-token-refresh-run-1",
            generation_index=1,
            status="running",
            started_at=(datetime.now(UTC) - timedelta(minutes=5)).isoformat(),
        )
    )
    asyncio.run(
        runtime_service._store.append_agent_run_artifact(
            "linked-token-refresh-run-1",
            artifact_id=(
                "linked-token-refresh-run-1:attempt:goal-token-refresh-attempt-1:"
                "subagent_runtime:high"
            ),
            artifact_type="subagent_runtime",
            title="Subagent Runtime Trace",
            uri=(
                "agent-run://linked-token-refresh-run-1/artifacts/"
                "goal-token-refresh-attempt-1/subagent_runtime/high"
            ),
            mime_type="application/json",
            metadata={
                "attempt_id": "goal-token-refresh-attempt-1",
                "content": {
                    "invocation_count": 2,
                    "completed_invocation_count": 2,
                    "token_tracked_invocation_count": 2,
                    "input_tokens": 20,
                    "output_tokens": 12,
                    "total_tokens": 32,
                    "generation_time_ms": 41.5,
                    "finish_reason_counts": {"stop": 2},
                    "approval_pending": [],
                    "risky_tool_events": [],
                    "invocations": [],
                },
            },
        )
    )
    now = datetime.now(UTC).isoformat()
    expires_at = (datetime.now(UTC) + timedelta(minutes=2)).isoformat()
    asyncio.run(
        runtime_service._store.upsert_goal_lease(
            goal_id="goal-token-refresh-1",
            owner_id=runtime_service._runtime_owner_id,
            metadata={"reason": "token_refresh_test"},
            acquired_at=now,
            heartbeat_at=now,
            expires_at=expires_at,
        )
    )

    original_is_live = runtime_service._agent_run_job_is_live
    runtime_service._agent_run_job_is_live = lambda run_id: run_id == "linked-token-refresh-run-1"
    try:
        asyncio.run(runtime_service._process_goal_supervision())
    finally:
        runtime_service._agent_run_job_is_live = original_is_live

    goal_after_open = asyncio.run(runtime_service._store.get_goal("goal-token-refresh-1"))
    run_after_open = asyncio.run(runtime_service._store.get_agent_run("linked-token-refresh-run-1"))
    findings_after_open = asyncio.run(
        runtime_service._store.list_goal_audit_findings(
            "goal-token-refresh-1",
            status="open",
        )
    )
    with TestClient(app) as client:
        health_open_response = client.get("/v1/goals/goal-token-refresh-1/health")
        assert health_open_response.status_code == 200
        health_after_open = health_open_response.json()

    assert goal_after_open is not None
    assert run_after_open is not None
    assert goal_after_open["status"] == "running"
    assert run_after_open["status"] == "running"
    assert health_after_open["status"] == "running"
    assert health_after_open["current_generation"]["status"] == "running"
    assert health_after_open["current_generation"]["generation_token_refresh_threshold"] == 30
    assert health_after_open["current_generation"]["token_refresh_due"] is True
    assert health_after_open["current_generation"]["token_refresh_over_threshold"] == 2
    assert [item["finding_code"] for item in findings_after_open] == [
        "generation_token_refresh_due"
    ]

    asyncio.run(
        runtime_service._store.update_agent_run_status(
            "linked-token-refresh-run-1",
            "succeeded",
        )
    )
    completed_run = asyncio.run(runtime_service._store.get_agent_run("linked-token-refresh-run-1"))
    assert completed_run is not None
    asyncio.run(
        runtime_service._sync_goal_from_agent_run(
            goal_id="goal-token-refresh-1",
            attempt_id="goal-token-refresh-attempt-1",
            run=completed_run,
        )
    )

    with TestClient(app) as client:
        health_response = client.get("/v1/goals/goal-token-refresh-1/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()

    goal_after_resolve = asyncio.run(runtime_service._store.get_goal("goal-token-refresh-1"))
    findings_after_resolve = asyncio.run(
        runtime_service._store.list_goal_audit_findings(
            "goal-token-refresh-1",
            status="open",
        )
    )

    assert goal_after_resolve is not None
    assert goal_after_resolve["status"] == "completed"
    assert health_payload["status"] == "completed"
    assert health_payload["open_findings"] == []
    assert findings_after_resolve == []


def test_goal_supervisor_suppresses_context_handoff_due_while_waiting_approval(
    tmp_path: Path,
) -> None:
    app, runtime_service = _create_goal_test_app(tmp_path)
    goal_id = "goal-context-handoff-approval-1"
    attempt_id = "goal-context-handoff-approval-attempt-1"
    run_id = "linked-context-handoff-approval-run-1"

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id=goal_id,
            objective="Do not open context handoff findings while approval is pending.",
            protocol_id="multi_agent_debate",
            run_policy={"context_handoff_threshold": 0.8},
            summary={"phase": "waiting_approval"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id=attempt_id,
            goal_id=goal_id,
            attempt_index=1,
            status="waiting_approval",
            trigger="manual_start",
            agent_run_id=run_id,
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            goal_id,
            "waiting_approval",
            current_attempt_id=attempt_id,
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id=run_id,
            protocol_id="multi_agent_debate",
            title="Approval-pending context handoff run",
            topic="context handoff awaiting approval",
            summary={
                "goal_id": goal_id,
                "goal_attempt_id": attempt_id,
                "objective": "Do not open context handoff findings while approval is pending.",
                "task_input": "Do not open context handoff findings while approval is pending.",
                "approval_state": {"status": "awaiting_approval", "pending_count": 1},
            },
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status(run_id, "awaiting_approval"))
    asyncio.run(
        runtime_service._store.create_goal_worker_generation(
            goal_id=goal_id,
            attempt_id=attempt_id,
            agent_run_id=run_id,
            generation_index=1,
            status="running",
            started_at=(datetime.now(UTC) - timedelta(seconds=45)).isoformat(),
        )
    )
    now = datetime.now(UTC).isoformat()
    asyncio.run(
        runtime_service._store.upsert_goal_lease(
            goal_id=goal_id,
            owner_id=runtime_service._runtime_owner_id,
            metadata={"reason": "context_handoff_approval_test"},
            acquired_at=now,
            heartbeat_at=now,
            expires_at=(datetime.now(UTC) + timedelta(minutes=2)).isoformat(),
        )
    )
    asyncio.run(
        runtime_service._store.append_agent_run_artifact(
            run_id,
            artifact_id=f"{run_id}:attempt:{attempt_id}:debate_context_snapshot:approval",
            artifact_type="debate_context_snapshot",
            title="Debate Context Snapshot",
            uri=f"agent-run://{run_id}/artifacts/{attempt_id}/debate_context_snapshot/approval",
            mime_type="application/json",
            metadata={
                "attempt_id": attempt_id,
                "content": {
                    "protocol": "multi_agent_debate",
                    "snapshots": [
                        {
                            "role_id": "judge",
                            "stage": "evaluation",
                            "usage_ratio": 0.93,
                        }
                    ],
                    "latest": {
                        "role_id": "judge",
                        "stage": "evaluation",
                        "usage_ratio": 0.93,
                    },
                },
            },
        )
    )

    original_is_live = runtime_service._agent_run_job_is_live
    runtime_service._agent_run_job_is_live = lambda candidate_run_id: candidate_run_id == run_id
    try:
        asyncio.run(runtime_service._process_goal_supervision())
    finally:
        runtime_service._agent_run_job_is_live = original_is_live

    with TestClient(app) as client:
        health_response = client.get(f"/v1/goals/{goal_id}/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()

    current_generation = health_payload["current_generation"]
    assert health_payload["status"] == "waiting_approval"
    assert health_payload["linked_agent_run"]["status"] == "awaiting_approval"
    assert current_generation["usage_ratio"] == 0.93
    assert current_generation["context_handoff_threshold"] == 0.8
    assert current_generation["context_handoff_due"] is True
    assert health_payload["recommended_next_action"] == {
        "action": "resolve_approval",
        "summary": "Goal is waiting on operator approval before it can continue.",
        "blocking": True,
        "run_id": run_id,
    }
    assert "context_handoff_due" not in [
        item["finding_code"] for item in health_payload["open_findings"]
    ]


def test_goal_supervisor_skips_generation_refresh_overdue_while_waiting_approval(
    tmp_path: Path,
) -> None:
    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    runtime_service.set_goal_lease_ttl_seconds(30)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-generation-refresh-approval-1",
            objective="Do not report refresh overdue while approval is pending.",
            protocol_id="teacher_student_distill",
            run_policy={"generation_refresh_interval_sec": 60},
            summary={"phase": "waiting_approval"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-generation-refresh-approval-attempt-1",
            goal_id="goal-generation-refresh-approval-1",
            attempt_index=1,
            status="waiting_approval",
            trigger="manual_start",
            agent_run_id="linked-generation-refresh-approval-run-1",
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-generation-refresh-approval-1",
            "waiting_approval",
            current_attempt_id="goal-generation-refresh-approval-attempt-1",
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id="linked-generation-refresh-approval-run-1",
            protocol_id="teacher_student_distill",
            title="Approval-pending refresh run",
            topic="generation refresh awaiting approval",
            summary={
                "goal_id": "goal-generation-refresh-approval-1",
                "goal_attempt_id": "goal-generation-refresh-approval-attempt-1",
                "objective": "Do not report refresh overdue while approval is pending.",
                "task_input": "Do not report refresh overdue while approval is pending.",
                "approval_state": {"status": "awaiting_approval", "pending_count": 1},
            },
        )
    )
    asyncio.run(
        runtime_service._store.update_agent_run_status(
            "linked-generation-refresh-approval-run-1",
            "awaiting_approval",
        )
    )
    stale_started_at = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    asyncio.run(
        runtime_service._store.create_goal_worker_generation(
            goal_id="goal-generation-refresh-approval-1",
            attempt_id="goal-generation-refresh-approval-attempt-1",
            agent_run_id="linked-generation-refresh-approval-run-1",
            generation_index=1,
            status="running",
            started_at=stale_started_at,
        )
    )
    now = datetime.now(UTC).isoformat()
    expires_at = (datetime.now(UTC) + timedelta(minutes=2)).isoformat()
    asyncio.run(
        runtime_service._store.upsert_goal_lease(
            goal_id="goal-generation-refresh-approval-1",
            owner_id=runtime_service._runtime_owner_id,
            metadata={"reason": "generation_refresh_approval_test"},
            acquired_at=now,
            heartbeat_at=now,
            expires_at=expires_at,
        )
    )

    asyncio.run(runtime_service._process_goal_supervision())

    with TestClient(app) as client:
        health_response = client.get("/v1/goals/goal-generation-refresh-approval-1/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()

    goal_after = asyncio.run(runtime_service._store.get_goal("goal-generation-refresh-approval-1"))
    run_after = asyncio.run(
        runtime_service._store.get_agent_run("linked-generation-refresh-approval-run-1")
    )
    findings_after = asyncio.run(
        runtime_service._store.list_goal_audit_findings(
            "goal-generation-refresh-approval-1",
            status="open",
        )
    )

    assert goal_after is not None
    assert run_after is not None
    assert goal_after["status"] == "waiting_approval"
    assert run_after["status"] == "awaiting_approval"
    assert findings_after == []
    assert health_payload["status"] == "waiting_approval"
    assert health_payload["linked_agent_run"]["status"] == "awaiting_approval"
    assert health_payload["open_findings"] == []


def test_goal_supervisor_opens_and_resolves_checkpoint_overdue_report_only(
    tmp_path: Path,
) -> None:
    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    runtime_service.set_goal_lease_ttl_seconds(30)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-checkpoint-overdue-1",
            objective="Report an overdue checkpoint without restarting the linked run.",
            protocol_id="teacher_student_distill",
            run_policy={"checkpoint_interval_sec": 60},
            summary={"phase": "running"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-checkpoint-overdue-attempt-1",
            goal_id="goal-checkpoint-overdue-1",
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id="linked-checkpoint-overdue-run-1",
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-checkpoint-overdue-1",
            "running",
            current_attempt_id="goal-checkpoint-overdue-attempt-1",
        )
    )
    stale_checkpoint_captured_at = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id="linked-checkpoint-overdue-run-1",
            protocol_id="teacher_student_distill",
            title="Checkpoint overdue linked run",
            topic="checkpoint overdue",
            summary={
                "goal_id": "goal-checkpoint-overdue-1",
                "goal_attempt_id": "goal-checkpoint-overdue-attempt-1",
                "objective": "Report an overdue checkpoint without restarting the linked run.",
                "task_input": "Report an overdue checkpoint without restarting the linked run.",
                "recovery_state": {
                    "status": "running",
                    "action": "continue",
                    "stage": "planner_progress",
                    "checkpoint": {
                        "checkpoint_index": 1,
                        "stage": "planner_progress",
                        "captured_at": stale_checkpoint_captured_at,
                    },
                },
            },
        )
    )
    asyncio.run(
        runtime_service._store.update_agent_run_status("linked-checkpoint-overdue-run-1", "running")
    )
    running_run = asyncio.run(runtime_service._store.get_agent_run("linked-checkpoint-overdue-run-1"))
    assert running_run is not None
    asyncio.run(
        runtime_service._sync_goal_from_agent_run(
            goal_id="goal-checkpoint-overdue-1",
            attempt_id="goal-checkpoint-overdue-attempt-1",
            run=running_run,
        )
    )
    now = datetime.now(UTC).isoformat()
    expires_at = (datetime.now(UTC) + timedelta(minutes=2)).isoformat()
    asyncio.run(
        runtime_service._store.upsert_goal_lease(
            goal_id="goal-checkpoint-overdue-1",
            owner_id=runtime_service._runtime_owner_id,
            metadata={"reason": "checkpoint_overdue_test"},
            acquired_at=now,
            heartbeat_at=now,
            expires_at=expires_at,
        )
    )

    original_is_live = runtime_service._agent_run_job_is_live
    runtime_service._agent_run_job_is_live = lambda run_id: run_id == "linked-checkpoint-overdue-run-1"
    try:
        asyncio.run(runtime_service._process_goal_supervision())
    finally:
        runtime_service._agent_run_job_is_live = original_is_live

    goal_after_open = asyncio.run(runtime_service._store.get_goal("goal-checkpoint-overdue-1"))
    run_after_open = asyncio.run(
        runtime_service._store.get_agent_run("linked-checkpoint-overdue-run-1")
    )
    findings_after_open = asyncio.run(
        runtime_service._store.list_goal_audit_findings(
            "goal-checkpoint-overdue-1",
            status="open",
        )
    )
    with TestClient(app) as client:
        health_open_response = client.get("/v1/goals/goal-checkpoint-overdue-1/health")
        assert health_open_response.status_code == 200
        health_after_open = health_open_response.json()

    assert goal_after_open is not None
    assert run_after_open is not None
    assert goal_after_open["status"] == "running"
    assert run_after_open["status"] == "running"
    assert health_after_open["status"] == "running"
    assert health_after_open["checkpoint_policy"]["status"] == "overdue"
    assert health_after_open["checkpoint_policy"]["overdue_sec"] >= 240
    assert [item["finding_code"] for item in findings_after_open] == ["checkpoint_overdue"]

    fresh_checkpoint_captured_at = datetime.now(UTC).isoformat()
    asyncio.run(
        runtime_service._store.update_agent_run_metadata(
            "linked-checkpoint-overdue-run-1",
            summary={
                "goal_id": "goal-checkpoint-overdue-1",
                "goal_attempt_id": "goal-checkpoint-overdue-attempt-1",
                "objective": "Report an overdue checkpoint without restarting the linked run.",
                "task_input": "Report an overdue checkpoint without restarting the linked run.",
                "recovery_state": {
                    "status": "running",
                    "action": "continue",
                    "stage": "executor_progress",
                    "checkpoint": {
                        "checkpoint_index": 2,
                        "stage": "executor_progress",
                        "captured_at": fresh_checkpoint_captured_at,
                    },
                },
            },
        )
    )
    refreshed_run = asyncio.run(runtime_service._store.get_agent_run("linked-checkpoint-overdue-run-1"))
    assert refreshed_run is not None
    asyncio.run(
        runtime_service._sync_goal_from_agent_run(
            goal_id="goal-checkpoint-overdue-1",
            attempt_id="goal-checkpoint-overdue-attempt-1",
            run=refreshed_run,
        )
    )

    goal_after_resolve = asyncio.run(runtime_service._store.get_goal("goal-checkpoint-overdue-1"))
    run_after_resolve = asyncio.run(
        runtime_service._store.get_agent_run("linked-checkpoint-overdue-run-1")
    )
    findings_after_resolve = asyncio.run(
        runtime_service._store.list_goal_audit_findings(
            "goal-checkpoint-overdue-1",
            status="open",
        )
    )
    with TestClient(app) as client:
        health_resolved_response = client.get("/v1/goals/goal-checkpoint-overdue-1/health")
        assert health_resolved_response.status_code == 200
        resolved_health = health_resolved_response.json()

    assert goal_after_resolve is not None
    assert run_after_resolve is not None
    assert goal_after_resolve["status"] == "running"
    assert run_after_resolve["status"] == "running"
    assert resolved_health["status"] == "running"
    assert resolved_health["checkpoint_policy"]["status"] == "recorded"
    assert resolved_health["persisted_checkpoint"]["checkpoint_index"] == 2
    assert resolved_health["persisted_checkpoint"]["stage"] == "executor_progress"
    assert findings_after_resolve == []


def test_goal_supervisor_opens_and_resolves_missing_checkpoint_report_only(
    tmp_path: Path,
) -> None:
    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    runtime_service.set_goal_lease_ttl_seconds(30)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-missing-checkpoint-1",
            objective="Report a missing first checkpoint without restarting the linked run.",
            protocol_id="teacher_student_distill",
            run_policy={"checkpoint_interval_sec": 60},
            summary={"phase": "running"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-missing-checkpoint-attempt-1",
            goal_id="goal-missing-checkpoint-1",
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id="linked-missing-checkpoint-run-1",
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-missing-checkpoint-1",
            "running",
            current_attempt_id="goal-missing-checkpoint-attempt-1",
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id="linked-missing-checkpoint-run-1",
            protocol_id="teacher_student_distill",
            title="Missing checkpoint linked run",
            topic="missing checkpoint",
            summary={
                "goal_id": "goal-missing-checkpoint-1",
                "goal_attempt_id": "goal-missing-checkpoint-attempt-1",
                "objective": "Report a missing first checkpoint without restarting the linked run.",
                "task_input": "Report a missing first checkpoint without restarting the linked run.",
            },
        )
    )
    asyncio.run(
        runtime_service._store.update_agent_run_status("linked-missing-checkpoint-run-1", "running")
    )
    now = datetime.now(UTC).isoformat()
    expires_at = (datetime.now(UTC) + timedelta(minutes=2)).isoformat()
    asyncio.run(
        runtime_service._store.upsert_goal_lease(
            goal_id="goal-missing-checkpoint-1",
            owner_id=runtime_service._runtime_owner_id,
            metadata={"reason": "missing_checkpoint_test"},
            acquired_at=now,
            heartbeat_at=now,
            expires_at=expires_at,
        )
    )

    original_is_live = runtime_service._agent_run_job_is_live
    runtime_service._agent_run_job_is_live = lambda run_id: run_id == "linked-missing-checkpoint-run-1"
    try:
        asyncio.run(runtime_service._process_goal_supervision())
    finally:
        runtime_service._agent_run_job_is_live = original_is_live

    goal_after_open = asyncio.run(runtime_service._store.get_goal("goal-missing-checkpoint-1"))
    run_after_open = asyncio.run(
        runtime_service._store.get_agent_run("linked-missing-checkpoint-run-1")
    )
    findings_after_open = asyncio.run(
        runtime_service._store.list_goal_audit_findings(
            "goal-missing-checkpoint-1",
            status="open",
        )
    )
    with TestClient(app) as client:
        health_open_response = client.get("/v1/goals/goal-missing-checkpoint-1/health")
        assert health_open_response.status_code == 200
        health_after_open = health_open_response.json()

    assert goal_after_open is not None
    assert run_after_open is not None
    assert goal_after_open["status"] == "running"
    assert run_after_open["status"] == "running"
    assert health_after_open["status"] == "running"
    assert health_after_open["checkpoint_policy"]["status"] == "pending_first_checkpoint"
    assert [item["finding_code"] for item in findings_after_open] == ["missing_checkpoint"]

    checkpoint_captured_at = datetime.now(UTC).isoformat()
    asyncio.run(
        runtime_service._store.update_agent_run_metadata(
            "linked-missing-checkpoint-run-1",
            summary={
                "goal_id": "goal-missing-checkpoint-1",
                "goal_attempt_id": "goal-missing-checkpoint-attempt-1",
                "objective": "Report a missing first checkpoint without restarting the linked run.",
                "task_input": "Report a missing first checkpoint without restarting the linked run.",
                "recovery_state": {
                    "status": "running",
                    "action": "continue",
                    "stage": "planner_progress",
                    "checkpoint": {
                        "checkpoint_index": 1,
                        "stage": "planner_progress",
                        "captured_at": checkpoint_captured_at,
                    },
                },
            },
        )
    )
    running_run_with_checkpoint = asyncio.run(
        runtime_service._store.get_agent_run("linked-missing-checkpoint-run-1")
    )
    assert running_run_with_checkpoint is not None
    asyncio.run(
        runtime_service._sync_goal_from_agent_run(
            goal_id="goal-missing-checkpoint-1",
            attempt_id="goal-missing-checkpoint-attempt-1",
            run=running_run_with_checkpoint,
        )
    )

    with TestClient(app) as client:
        health_resolved_response = client.get("/v1/goals/goal-missing-checkpoint-1/health")
        assert health_resolved_response.status_code == 200
        resolved_health = health_resolved_response.json()

    findings_after_resolve = asyncio.run(
        runtime_service._store.list_goal_audit_findings(
            "goal-missing-checkpoint-1",
            status="open",
        )
    )
    assert resolved_health["status"] == "running"
    assert resolved_health["checkpoint_policy"]["status"] == "recorded"
    assert resolved_health["persisted_checkpoint"]["checkpoint_index"] == 1
    assert "missing_checkpoint" not in [
        item["finding_code"] for item in findings_after_resolve
    ]


def test_goal_supervisor_opens_and_resolves_collector_shard_stuck_report_only(
    tmp_path: Path,
) -> None:
    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    runtime_service.set_goal_lease_ttl_seconds(30)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    stale_updated_at = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    fresh_updated_at = datetime.now(UTC).isoformat()
    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-collector-stuck-1",
            objective="Report a stuck collector shard without restarting the linked run.",
            protocol_id="teacher_student_distill",
            run_policy={"collector_shard_stall_timeout_sec": 60},
            summary={"phase": "running"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-collector-stuck-attempt-1",
            goal_id="goal-collector-stuck-1",
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id="linked-collector-stuck-run-1",
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-collector-stuck-1",
            "running",
            current_attempt_id="goal-collector-stuck-attempt-1",
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id="linked-collector-stuck-run-1",
            protocol_id="teacher_student_distill",
            title="Collector shard stall run",
            topic="collector shard stall",
            summary={
                "goal_id": "goal-collector-stuck-1",
                "goal_attempt_id": "goal-collector-stuck-attempt-1",
                "objective": "Report a stuck collector shard without restarting the linked run.",
                "task_input": "Report a stuck collector shard without restarting the linked run.",
            },
        )
    )
    asyncio.run(
        runtime_service._store.update_agent_run_metadata(
            "linked-collector-stuck-run-1",
            summary={
                "goal_id": "goal-collector-stuck-1",
                "goal_attempt_id": "goal-collector-stuck-attempt-1",
                "objective": "Report a stuck collector shard without restarting the linked run.",
                "task_input": "Report a stuck collector shard without restarting the linked run.",
                "recovery_state": {
                    "status": "running",
                    "action": "continue",
                    "stage": "collector_progress",
                    "checkpoint": {
                        "checkpoint_index": 2,
                        "stage": "collector_progress",
                        "captured_at": stale_updated_at,
                    },
                },
            },
        )
    )
    asyncio.run(
        runtime_service._store.update_agent_run_status("linked-collector-stuck-run-1", "running")
    )
    asyncio.run(
        runtime_service._store.append_agent_run_artifact(
            "linked-collector-stuck-run-1",
            artifact_id="collector-stuck-shard-1",
            artifact_type="collector_shard_manifest",
            title="Collector Shard forum-thread-1",
            uri="agent-run://linked-collector-stuck-run-1/artifacts/collector_stuck_shard_1",
            metadata={
                "attempt_id": "goal-collector-stuck-attempt-1",
                "content": {
                    "shard_id": "forum-thread-1",
                    "adapter_name": "forum_thread_adapter",
                    "status": "running",
                    "updated_at": stale_updated_at,
                    "artifact_updated_at": stale_updated_at,
                    "source": {
                        "url": "https://forum.example/thread-1",
                        "id": "thread-1",
                    },
                    "progress": {
                        "cursor": "post-10",
                        "items_collected": 10,
                        "items_emitted": 10,
                    },
                },
            },
        )
    )
    asyncio.run(runtime_service._sync_goal_from_agent_run_by_run_id("linked-collector-stuck-run-1"))

    findings_after_open = asyncio.run(
        runtime_service._store.list_goal_audit_findings(
            "goal-collector-stuck-1",
            status="open",
        )
    )
    with TestClient(app) as client:
        health_open_response = client.get("/v1/goals/goal-collector-stuck-1/health")
        assert health_open_response.status_code == 200
        health_after_open = health_open_response.json()

    assert [item["finding_code"] for item in findings_after_open] == ["collector_shard_stuck"]
    assert health_after_open["collector_state"]["stalled_shard_count"] == 1
    assert health_after_open["recommended_next_action"]["action"] == "inspect_collector_shards"
    assert health_after_open["persisted_checkpoint"]["payload"]["collector_shard_offsets"][0]["cursor"] == (
        "post-10"
    )
    assert (
        health_after_open["memory_snapshot"]["snapshot"]["collector_shard_offsets"][0]["shard_id"]
        == "forum-thread-1"
    )

    asyncio.run(
        runtime_service._store.append_agent_run_artifact(
            "linked-collector-stuck-run-1",
            artifact_id="collector-stuck-shard-1-refresh",
            artifact_type="collector_shard_manifest",
            title="Collector Shard forum-thread-1 refresh",
            uri="agent-run://linked-collector-stuck-run-1/artifacts/collector_stuck_shard_1_refresh",
            metadata={
                "attempt_id": "goal-collector-stuck-attempt-1",
                "content": {
                    "shard_id": "forum-thread-1",
                    "adapter_name": "forum_thread_adapter",
                    "status": "completed",
                    "updated_at": fresh_updated_at,
                    "completed_at": fresh_updated_at,
                    "artifact_updated_at": fresh_updated_at,
                    "source": {
                        "url": "https://forum.example/thread-1",
                        "id": "thread-1",
                    },
                    "progress": {
                        "cursor": "post-24",
                        "items_collected": 24,
                        "items_emitted": 24,
                    },
                },
            },
        )
    )
    asyncio.run(runtime_service._sync_goal_from_agent_run_by_run_id("linked-collector-stuck-run-1"))

    findings_after_resolve = asyncio.run(
        runtime_service._store.list_goal_audit_findings(
            "goal-collector-stuck-1",
            status="open",
        )
    )
    with TestClient(app) as client:
        health_resolved_response = client.get("/v1/goals/goal-collector-stuck-1/health")
        assert health_resolved_response.status_code == 200
        resolved_health = health_resolved_response.json()

    assert "collector_shard_stuck" not in [
        item["finding_code"] for item in findings_after_resolve
    ]
    assert resolved_health["collector_state"]["stalled_shard_count"] == 0
    assert resolved_health["collector_state"]["shards"][0]["status"] == "completed"


def test_goal_collector_state_is_attempt_scoped_and_refreshes_persisted_progress(
    tmp_path: Path,
) -> None:
    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    runtime_service.set_goal_lease_ttl_seconds(30)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    stale_updated_at = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    fresh_updated_at = datetime.now(UTC).isoformat()
    refreshed_updated_at = (datetime.now(UTC) + timedelta(seconds=5)).isoformat()
    goal_id = "goal-collector-refresh-1"
    attempt_id = "goal-collector-refresh-attempt-1"
    run_id = "linked-collector-refresh-run-1"

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id=goal_id,
            objective="Track collector shard progress without carrying stale shard state across attempts.",
            protocol_id="teacher_student_distill",
            run_policy={"collector_shard_stall_timeout_sec": 60},
            summary={"phase": "running"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id=attempt_id,
            goal_id=goal_id,
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id=run_id,
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            goal_id,
            "running",
            current_attempt_id=attempt_id,
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id=run_id,
            protocol_id="teacher_student_distill",
            title="Collector progress refresh run",
            topic="collector progress refresh",
            summary={
                "goal_id": goal_id,
                "goal_attempt_id": attempt_id,
                "objective": "Track collector shard progress without carrying stale shard state across attempts.",
                "task_input": "Track collector shard progress without carrying stale shard state across attempts.",
            },
        )
    )
    asyncio.run(
        runtime_service._store.update_agent_run_metadata(
            run_id,
            summary={
                "goal_id": goal_id,
                "goal_attempt_id": attempt_id,
                "objective": "Track collector shard progress without carrying stale shard state across attempts.",
                "task_input": "Track collector shard progress without carrying stale shard state across attempts.",
                "recovery_state": {
                    "status": "running",
                    "action": "continue",
                    "stage": "collector_progress",
                    "checkpoint": {
                        "checkpoint_index": 2,
                        "stage": "collector_progress",
                        "captured_at": fresh_updated_at,
                    },
                },
            },
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status(run_id, "running"))
    asyncio.run(
        runtime_service._store.append_agent_run_artifact(
            run_id,
            artifact_id="collector-refresh-shard-legacy",
            artifact_type="collector_shard_manifest",
            title="Collector Shard legacy-thread",
            uri="agent-run://linked-collector-refresh-run-1/artifacts/collector_refresh_shard_legacy",
            metadata={
                "attempt_id": "legacy-agent-attempt-1",
                "content": {
                        "shard_id": "legacy-thread",
                        "adapter_name": "forum_thread_adapter",
                        "status": "running",
                        "updated_at": stale_updated_at,
                        "artifact_updated_at": stale_updated_at,
                        "source": {
                            "url": "https://forum.example/legacy-thread",
                            "id": "legacy-thread",
                        },
                    "progress": {
                        "cursor": "post-90",
                        "items_collected": 90,
                        "items_emitted": 90,
                    },
                },
            },
        )
    )
    asyncio.run(
        runtime_service._store.append_agent_run_artifact(
            run_id,
            artifact_id="collector-refresh-shard-current",
            artifact_type="collector_shard_manifest",
            title="Collector Shard current-thread",
            uri="agent-run://linked-collector-refresh-run-1/artifacts/collector_refresh_shard_current",
            metadata={
                "attempt_id": attempt_id,
                "content": {
                        "shard_id": "current-thread",
                        "adapter_name": "forum_thread_adapter",
                        "status": "running",
                        "updated_at": fresh_updated_at,
                        "artifact_updated_at": fresh_updated_at,
                        "source": {
                            "url": "https://forum.example/current-thread",
                            "id": "current-thread",
                        },
                    "progress": {
                        "cursor": "post-3",
                        "items_collected": 3,
                        "items_emitted": 3,
                    },
                },
            },
        )
    )
    running_run = asyncio.run(runtime_service._store.get_agent_run(run_id))
    assert running_run is not None
    asyncio.run(
        runtime_service._sync_goal_from_agent_run(
            goal_id=goal_id,
            attempt_id=attempt_id,
            run=running_run,
        )
    )

    checkpoint_before_refresh = asyncio.run(
        runtime_service._store.get_latest_goal_checkpoint(goal_id, attempt_id=attempt_id)
    )
    snapshot_before_refresh = asyncio.run(
        runtime_service._store.get_latest_goal_memory_snapshot(goal_id, attempt_id=attempt_id)
    )
    findings_before_refresh = asyncio.run(
        runtime_service._store.list_goal_audit_findings(goal_id, status="open")
    )

    with TestClient(app) as client:
        initial_health_response = client.get(f"/v1/goals/{goal_id}/health")
        assert initial_health_response.status_code == 200
        initial_health = initial_health_response.json()

    assert checkpoint_before_refresh is not None
    assert snapshot_before_refresh is not None
    assert "collector_shard_stuck" not in [
        item["finding_code"] for item in findings_before_refresh
    ]
    assert initial_health["collector_state"]["shard_count"] == 1
    assert initial_health["collector_state"]["stalled_shard_count"] == 0
    assert initial_health["collector_state"]["shards"][0]["shard_id"] == "current-thread"
    assert initial_health["persisted_checkpoint"]["payload"]["collector_shard_offsets"] == [
        {
            "shard_id": "current-thread",
            "attempt_id": attempt_id,
            "status": "running",
            "adapter_name": "forum_thread_adapter",
            "source_url": "https://forum.example/current-thread",
            "source_id": "current-thread",
            "cursor": "post-3",
            "items_collected": 3,
            "items_emitted": 3,
            "last_activity_at": fresh_updated_at,
        }
    ]

    asyncio.run(
        runtime_service._store.append_agent_run_artifact(
            run_id,
            artifact_id="collector-refresh-shard-current-2",
            artifact_type="collector_shard_manifest",
            title="Collector Shard current-thread refresh",
            uri="agent-run://linked-collector-refresh-run-1/artifacts/collector_refresh_shard_current_2",
            metadata={
                "attempt_id": attempt_id,
                "content": {
                        "shard_id": "current-thread",
                        "adapter_name": "forum_thread_adapter",
                        "status": "completed",
                        "updated_at": refreshed_updated_at,
                        "completed_at": refreshed_updated_at,
                        "artifact_updated_at": refreshed_updated_at,
                        "source": {
                            "url": "https://forum.example/current-thread",
                            "id": "current-thread",
                        },
                    "progress": {
                        "cursor": "post-24",
                        "items_collected": 24,
                        "items_emitted": 24,
                    },
                },
            },
        )
    )
    refreshed_run = asyncio.run(runtime_service._store.get_agent_run(run_id))
    assert refreshed_run is not None
    asyncio.run(
        runtime_service._sync_goal_from_agent_run(
            goal_id=goal_id,
            attempt_id=attempt_id,
            run=refreshed_run,
        )
    )

    checkpoint_after_refresh = asyncio.run(
        runtime_service._store.get_latest_goal_checkpoint(goal_id, attempt_id=attempt_id)
    )
    snapshot_after_refresh = asyncio.run(
        runtime_service._store.get_latest_goal_memory_snapshot(goal_id, attempt_id=attempt_id)
    )

    with TestClient(app) as client:
        refreshed_health_response = client.get(f"/v1/goals/{goal_id}/health")
        assert refreshed_health_response.status_code == 200
        refreshed_health = refreshed_health_response.json()

    assert checkpoint_after_refresh is not None
    assert snapshot_after_refresh is not None
    assert checkpoint_after_refresh["id"] != checkpoint_before_refresh["id"]
    assert snapshot_after_refresh["id"] != snapshot_before_refresh["id"]
    assert refreshed_health["collector_state"]["shard_count"] == 1
    assert refreshed_health["collector_state"]["stalled_shard_count"] == 0
    assert refreshed_health["collector_state"]["shards"][0]["shard_id"] == "current-thread"
    assert refreshed_health["collector_state"]["shards"][0]["status"] == "completed"
    assert refreshed_health["persisted_checkpoint"]["payload"]["collector_shard_offsets"][0]["cursor"] == (
        "post-24"
    )
    assert refreshed_health["persisted_checkpoint"]["payload"]["collector_shard_offsets"][0]["status"] == (
        "completed"
    )
    assert refreshed_health["memory_snapshot"]["snapshot"]["collector_shard_offsets"][0]["cursor"] == (
        "post-24"
    )


def test_goal_supervisor_skips_checkpoint_overdue_while_waiting_approval(
    tmp_path: Path,
) -> None:
    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    runtime_service.set_goal_lease_ttl_seconds(30)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    stale_checkpoint_captured_at = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-checkpoint-overdue-approval-1",
            objective="Do not report an overdue checkpoint while approval is pending.",
            protocol_id="teacher_student_distill",
            run_policy={"checkpoint_interval_sec": 60},
            summary={"phase": "waiting_approval"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-checkpoint-overdue-approval-attempt-1",
            goal_id="goal-checkpoint-overdue-approval-1",
            attempt_index=1,
            status="waiting_approval",
            trigger="manual_start",
            agent_run_id="linked-checkpoint-overdue-approval-run-1",
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-checkpoint-overdue-approval-1",
            "waiting_approval",
            current_attempt_id="goal-checkpoint-overdue-approval-attempt-1",
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id="linked-checkpoint-overdue-approval-run-1",
            protocol_id="teacher_student_distill",
            title="Checkpoint overdue approval wait",
            topic="checkpoint overdue awaiting approval",
            summary={
                "goal_id": "goal-checkpoint-overdue-approval-1",
                "goal_attempt_id": "goal-checkpoint-overdue-approval-attempt-1",
                "objective": "Do not report an overdue checkpoint while approval is pending.",
                "task_input": "Do not report an overdue checkpoint while approval is pending.",
                "approval_state": {"status": "awaiting_approval", "pending_count": 1},
                "recovery_state": {
                    "status": "awaiting_approval",
                    "action": "await_approval",
                    "stage": "controller_decision",
                    "checkpoint": {
                        "checkpoint_index": 3,
                        "stage": "controller_decision",
                        "captured_at": stale_checkpoint_captured_at,
                    },
                },
            },
        )
    )
    asyncio.run(
        runtime_service._store.update_agent_run_status(
            "linked-checkpoint-overdue-approval-run-1",
            "awaiting_approval",
        )
    )
    waiting_run = asyncio.run(
        runtime_service._store.get_agent_run("linked-checkpoint-overdue-approval-run-1")
    )
    assert waiting_run is not None
    asyncio.run(
        runtime_service._sync_goal_from_agent_run(
            goal_id="goal-checkpoint-overdue-approval-1",
            attempt_id="goal-checkpoint-overdue-approval-attempt-1",
            run=waiting_run,
        )
    )
    now = datetime.now(UTC).isoformat()
    expires_at = (datetime.now(UTC) + timedelta(minutes=2)).isoformat()
    asyncio.run(
        runtime_service._store.upsert_goal_lease(
            goal_id="goal-checkpoint-overdue-approval-1",
            owner_id=runtime_service._runtime_owner_id,
            metadata={"reason": "checkpoint_overdue_approval_test"},
            acquired_at=now,
            heartbeat_at=now,
            expires_at=expires_at,
        )
    )

    asyncio.run(runtime_service._process_goal_supervision())

    with TestClient(app) as client:
        health_response = client.get("/v1/goals/goal-checkpoint-overdue-approval-1/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()

    goal_after = asyncio.run(runtime_service._store.get_goal("goal-checkpoint-overdue-approval-1"))
    run_after = asyncio.run(
        runtime_service._store.get_agent_run("linked-checkpoint-overdue-approval-run-1")
    )
    findings_after = asyncio.run(
        runtime_service._store.list_goal_audit_findings(
            "goal-checkpoint-overdue-approval-1",
            status="open",
        )
    )

    assert goal_after is not None
    assert run_after is not None
    assert goal_after["status"] == "waiting_approval"
    assert run_after["status"] == "awaiting_approval"
    assert findings_after == []
    assert health_payload["status"] == "waiting_approval"
    assert health_payload["linked_agent_run"]["status"] == "awaiting_approval"
    assert health_payload["open_findings"] == []


def test_goal_supervisor_skips_missing_checkpoint_while_waiting_approval(
    tmp_path: Path,
) -> None:
    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    runtime_service.set_goal_lease_ttl_seconds(30)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-missing-checkpoint-approval-1",
            objective="Do not report a missing checkpoint while approval is pending.",
            protocol_id="teacher_student_distill",
            run_policy={"checkpoint_interval_sec": 60},
            summary={"phase": "waiting_approval"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-missing-checkpoint-approval-attempt-1",
            goal_id="goal-missing-checkpoint-approval-1",
            attempt_index=1,
            status="waiting_approval",
            trigger="manual_start",
            agent_run_id="linked-missing-checkpoint-approval-run-1",
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-missing-checkpoint-approval-1",
            "waiting_approval",
            current_attempt_id="goal-missing-checkpoint-approval-attempt-1",
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id="linked-missing-checkpoint-approval-run-1",
            protocol_id="teacher_student_distill",
            title="Missing checkpoint approval wait",
            topic="missing checkpoint awaiting approval",
            summary={
                "goal_id": "goal-missing-checkpoint-approval-1",
                "goal_attempt_id": "goal-missing-checkpoint-approval-attempt-1",
                "objective": "Do not report a missing checkpoint while approval is pending.",
                "task_input": "Do not report a missing checkpoint while approval is pending.",
                "approval_state": {"status": "awaiting_approval", "pending_count": 1},
            },
        )
    )
    asyncio.run(
        runtime_service._store.update_agent_run_status(
            "linked-missing-checkpoint-approval-run-1",
            "awaiting_approval",
        )
    )
    now = datetime.now(UTC).isoformat()
    expires_at = (datetime.now(UTC) + timedelta(minutes=2)).isoformat()
    asyncio.run(
        runtime_service._store.upsert_goal_lease(
            goal_id="goal-missing-checkpoint-approval-1",
            owner_id=runtime_service._runtime_owner_id,
            metadata={"reason": "missing_checkpoint_approval_test"},
            acquired_at=now,
            heartbeat_at=now,
            expires_at=expires_at,
        )
    )

    asyncio.run(runtime_service._process_goal_supervision())

    with TestClient(app) as client:
        health_response = client.get("/v1/goals/goal-missing-checkpoint-approval-1/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()

    findings_after = asyncio.run(
        runtime_service._store.list_goal_audit_findings(
            "goal-missing-checkpoint-approval-1",
            status="open",
        )
    )
    assert health_payload["status"] == "waiting_approval"
    assert health_payload["linked_agent_run"]["status"] == "awaiting_approval"
    assert health_payload["checkpoint_policy"]["status"] == "missing_checkpoint"
    assert findings_after == []


def test_goal_supervisor_opens_and_resolves_approval_wait_timeout_report_only(
    tmp_path: Path,
) -> None:
    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    runtime_service.set_goal_lease_ttl_seconds(30)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    started_at = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-approval-timeout-1",
            objective="Surface approval wait timeout as a report-only operator finding.",
            protocol_id="controlled_subagent_execution",
            summary={"phase": "waiting_approval"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-approval-timeout-attempt-1",
            goal_id="goal-approval-timeout-1",
            attempt_index=1,
            status="waiting_approval",
            trigger="manual_start",
            agent_run_id="linked-approval-timeout-run-1",
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-approval-timeout-1",
            "waiting_approval",
            current_attempt_id="goal-approval-timeout-attempt-1",
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id="linked-approval-timeout-run-1",
            protocol_id="controlled_subagent_execution",
            title="Approval timeout linked run",
            topic="approval wait timeout",
            summary={
                "goal_id": "goal-approval-timeout-1",
                "goal_attempt_id": "goal-approval-timeout-attempt-1",
                "objective": "Surface approval wait timeout as a report-only operator finding.",
                "task_input": "Surface approval wait timeout as a report-only operator finding.",
                "approval_state": {
                    "status": "awaiting_approval",
                    "pending_count": 1,
                    "approval_ids": ["exec-approval-timeout-1"],
                    "pending_approvals": [
                        {
                            "approval_id": "exec-approval-timeout-1",
                            "tool_name": "exec_command",
                            "request_id": "req-timeout-1",
                            "task_key": "controlled_execution_exec:req-timeout-1",
                            "stage": "controlled_execution_exec:req-timeout-1",
                            "source": "controlled_execution",
                            "created_at": started_at,
                            "timeout_sec": 60,
                        }
                    ],
                },
                "recovery_state": {
                    "status": "awaiting_approval",
                    "action": "await_approval",
                    "reason": "Execution approval required",
                    "stage": "controlled_execution_exec:req-timeout-1",
                    "checkpoint": {
                        "checkpoint_index": 4,
                        "stage": "controlled_execution_controller:req-timeout-1",
                    },
                },
            },
        )
    )
    asyncio.run(
        runtime_service._store.update_agent_run_status(
            "linked-approval-timeout-run-1",
            "awaiting_approval",
        )
    )
    waiting_run = asyncio.run(
        runtime_service._store.get_agent_run("linked-approval-timeout-run-1")
    )
    assert waiting_run is not None
    asyncio.run(
        runtime_service._sync_goal_from_agent_run(
            goal_id="goal-approval-timeout-1",
            attempt_id="goal-approval-timeout-attempt-1",
            run=waiting_run,
        )
    )

    with TestClient(app) as client:
        health_response = client.get("/v1/goals/goal-approval-timeout-1/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()

    findings_after_open = asyncio.run(
        runtime_service._store.list_goal_audit_findings(
            "goal-approval-timeout-1",
            status="open",
        )
    )
    assert health_payload["status"] == "waiting_approval"
    assert health_payload["approval_state"]["approval_wait_started_at"] == started_at
    assert health_payload["approval_state"]["approval_wait_timeout_sec"] == 60
    assert health_payload["approval_state"]["approval_wait_elapsed_sec"] >= 300
    assert health_payload["recommended_next_action"] == {
        "action": "resolve_approval",
        "summary": "Goal has been waiting on operator approval longer than the configured approval wait timeout.",
        "blocking": True,
        "run_id": "linked-approval-timeout-run-1",
        "approval_ids": ["exec-approval-timeout-1"],
        "finding_code": "approval_wait_timeout",
        "approval_wait_elapsed_sec": health_payload["approval_state"]["approval_wait_elapsed_sec"],
        "approval_wait_timeout_sec": 60,
    }
    assert [item["finding_code"] for item in findings_after_open] == ["approval_wait_timeout"]
    assert findings_after_open[0]["details"]["approval_wait_timeout_sec"] == 60
    assert findings_after_open[0]["details"]["approval_wait_elapsed_sec"] >= 300

    asyncio.run(
        runtime_service._store.update_agent_run_metadata(
            "linked-approval-timeout-run-1",
            summary={
                "goal_id": "goal-approval-timeout-1",
                "goal_attempt_id": "goal-approval-timeout-attempt-1",
                "objective": "Surface approval wait timeout as a report-only operator finding.",
                "task_input": "Surface approval wait timeout as a report-only operator finding.",
                "approval_state": {
                    "status": "awaiting_approval",
                    "pending_count": 1,
                    "approval_ids": ["exec-approval-timeout-1"],
                    "pending_approvals": [
                        {
                            "approval_id": "exec-approval-timeout-1",
                            "tool_name": "exec_command",
                            "request_id": "req-timeout-1",
                            "task_key": "controlled_execution_exec:req-timeout-1",
                            "stage": "controlled_execution_exec:req-timeout-1",
                            "source": "controlled_execution",
                            "created_at": started_at,
                            "timeout_sec": 600,
                        }
                    ],
                },
                "recovery_state": {
                    "status": "awaiting_approval",
                    "action": "await_approval",
                    "reason": "Execution approval required",
                    "stage": "controlled_execution_exec:req-timeout-1",
                    "checkpoint": {
                        "checkpoint_index": 4,
                        "stage": "controlled_execution_controller:req-timeout-1",
                    },
                },
            },
        )
    )
    waiting_run_after_update = asyncio.run(
        runtime_service._store.get_agent_run("linked-approval-timeout-run-1")
    )
    assert waiting_run_after_update is not None
    asyncio.run(
        runtime_service._sync_goal_from_agent_run(
            goal_id="goal-approval-timeout-1",
            attempt_id="goal-approval-timeout-attempt-1",
            run=waiting_run_after_update,
        )
    )

    with TestClient(app) as client:
        resolved_health_response = client.get("/v1/goals/goal-approval-timeout-1/health")
        assert resolved_health_response.status_code == 200
        resolved_health = resolved_health_response.json()

    findings_after_resolve = asyncio.run(
        runtime_service._store.list_goal_audit_findings(
            "goal-approval-timeout-1",
            status="open",
        )
    )
    assert resolved_health["status"] == "waiting_approval"
    assert resolved_health["approval_state"]["approval_wait_timeout_sec"] == 600
    assert resolved_health["approval_state"]["approval_wait_elapsed_sec"] < 600
    assert resolved_health["recommended_next_action"] == {
        "action": "resolve_approval",
        "summary": "Goal is waiting on operator approval before it can continue.",
        "blocking": True,
        "run_id": "linked-approval-timeout-run-1",
        "approval_ids": ["exec-approval-timeout-1"],
    }
    assert findings_after_resolve == []


def test_goal_supervisor_opens_and_resolves_checkpoint_overdue_report_only(
    tmp_path: Path,
) -> None:
    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    runtime_service.set_goal_lease_ttl_seconds(30)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    old_checkpoint_captured_at = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-checkpoint-overdue-1",
            objective="Report an overdue checkpoint without restarting the linked run.",
            protocol_id="teacher_student_distill",
            run_policy={"checkpoint_interval_sec": 60},
            summary={"phase": "running"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-checkpoint-overdue-attempt-1",
            goal_id="goal-checkpoint-overdue-1",
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id="linked-checkpoint-overdue-run-1",
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-checkpoint-overdue-1",
            "running",
            current_attempt_id="goal-checkpoint-overdue-attempt-1",
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id="linked-checkpoint-overdue-run-1",
            protocol_id="teacher_student_distill",
            title="Checkpoint overdue linked run",
            topic="checkpoint overdue",
            summary={
                "goal_id": "goal-checkpoint-overdue-1",
                "goal_attempt_id": "goal-checkpoint-overdue-attempt-1",
                "objective": "Report an overdue checkpoint without restarting the linked run.",
                "task_input": "Report an overdue checkpoint without restarting the linked run.",
                "recovery_state": {
                    "status": "running",
                    "action": "continue",
                    "stage": "planner_progress",
                    "checkpoint": {
                        "checkpoint_index": 2,
                        "stage": "planner_progress",
                        "captured_at": old_checkpoint_captured_at,
                    },
                },
            },
        )
    )
    asyncio.run(
        runtime_service._store.update_agent_run_status("linked-checkpoint-overdue-run-1", "running")
    )
    now = datetime.now(UTC).isoformat()
    expires_at = (datetime.now(UTC) + timedelta(minutes=2)).isoformat()
    asyncio.run(
        runtime_service._store.upsert_goal_lease(
            goal_id="goal-checkpoint-overdue-1",
            owner_id=runtime_service._runtime_owner_id,
            metadata={"reason": "checkpoint_overdue_test"},
            acquired_at=now,
            heartbeat_at=now,
            expires_at=expires_at,
        )
    )

    original_is_live = runtime_service._agent_run_job_is_live
    runtime_service._agent_run_job_is_live = lambda run_id: run_id == "linked-checkpoint-overdue-run-1"
    try:
        asyncio.run(runtime_service._process_goal_supervision())
    finally:
        runtime_service._agent_run_job_is_live = original_is_live

    goal_after_open = asyncio.run(runtime_service._store.get_goal("goal-checkpoint-overdue-1"))
    run_after_open = asyncio.run(
        runtime_service._store.get_agent_run("linked-checkpoint-overdue-run-1")
    )
    findings_after_open = asyncio.run(
        runtime_service._store.list_goal_audit_findings(
            "goal-checkpoint-overdue-1",
            status="open",
        )
    )
    with TestClient(app) as client:
        health_open_response = client.get("/v1/goals/goal-checkpoint-overdue-1/health")
        assert health_open_response.status_code == 200
        health_after_open = health_open_response.json()

    assert goal_after_open is not None
    assert run_after_open is not None
    assert goal_after_open["status"] == "running"
    assert run_after_open["status"] == "running"
    assert health_after_open["status"] == "running"
    assert health_after_open["checkpoint_policy"]["status"] == "overdue"
    assert health_after_open["checkpoint_policy"]["overdue_sec"] >= 240
    assert "checkpoint_overdue" in [item["finding_code"] for item in findings_after_open]

    fresh_checkpoint_captured_at = datetime.now(UTC).isoformat()
    asyncio.run(
        runtime_service._store.update_agent_run_metadata(
            "linked-checkpoint-overdue-run-1",
            summary={
                "goal_id": "goal-checkpoint-overdue-1",
                "goal_attempt_id": "goal-checkpoint-overdue-attempt-1",
                "objective": "Report an overdue checkpoint without restarting the linked run.",
                "task_input": "Report an overdue checkpoint without restarting the linked run.",
                "recovery_state": {
                    "status": "running",
                    "action": "continue",
                    "stage": "planner_progress",
                    "checkpoint": {
                        "checkpoint_index": 3,
                        "stage": "planner_progress",
                        "captured_at": fresh_checkpoint_captured_at,
                    },
                },
            },
        )
    )
    running_run_with_fresh_checkpoint = asyncio.run(
        runtime_service._store.get_agent_run("linked-checkpoint-overdue-run-1")
    )
    assert running_run_with_fresh_checkpoint is not None
    asyncio.run(
        runtime_service._sync_goal_from_agent_run(
            goal_id="goal-checkpoint-overdue-1",
            attempt_id="goal-checkpoint-overdue-attempt-1",
            run=running_run_with_fresh_checkpoint,
        )
    )

    with TestClient(app) as client:
        health_resolved_response = client.get("/v1/goals/goal-checkpoint-overdue-1/health")
        assert health_resolved_response.status_code == 200
        resolved_health = health_resolved_response.json()

    findings_after_resolve = asyncio.run(
        runtime_service._store.list_goal_audit_findings(
            "goal-checkpoint-overdue-1",
            status="open",
        )
    )
    assert resolved_health["status"] == "running"
    assert resolved_health["checkpoint_policy"]["status"] == "recorded"
    assert resolved_health["persisted_checkpoint"]["checkpoint_index"] == 3
    assert "checkpoint_overdue" not in [
        item["finding_code"] for item in findings_after_resolve
    ]


def test_goal_supervisor_skips_checkpoint_overdue_while_waiting_approval(
    tmp_path: Path,
) -> None:
    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    runtime_service.set_goal_lease_ttl_seconds(30)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    old_checkpoint_captured_at = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-checkpoint-overdue-approval-1",
            objective="Do not report an overdue checkpoint while approval is pending.",
            protocol_id="teacher_student_distill",
            run_policy={"checkpoint_interval_sec": 60},
            summary={"phase": "waiting_approval"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-checkpoint-overdue-approval-attempt-1",
            goal_id="goal-checkpoint-overdue-approval-1",
            attempt_index=1,
            status="waiting_approval",
            trigger="manual_start",
            agent_run_id="linked-checkpoint-overdue-approval-run-1",
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-checkpoint-overdue-approval-1",
            "waiting_approval",
            current_attempt_id="goal-checkpoint-overdue-approval-attempt-1",
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id="linked-checkpoint-overdue-approval-run-1",
            protocol_id="teacher_student_distill",
            title="Checkpoint overdue approval wait",
            topic="checkpoint overdue awaiting approval",
            summary={
                "goal_id": "goal-checkpoint-overdue-approval-1",
                "goal_attempt_id": "goal-checkpoint-overdue-approval-attempt-1",
                "objective": "Do not report an overdue checkpoint while approval is pending.",
                "task_input": "Do not report an overdue checkpoint while approval is pending.",
                "approval_state": {"status": "awaiting_approval", "pending_count": 1},
                "recovery_state": {
                    "status": "awaiting_approval",
                    "action": "await_approval",
                    "stage": "planner_progress",
                    "checkpoint": {
                        "checkpoint_index": 2,
                        "stage": "planner_progress",
                        "captured_at": old_checkpoint_captured_at,
                    },
                },
            },
        )
    )
    asyncio.run(
        runtime_service._store.update_agent_run_status(
            "linked-checkpoint-overdue-approval-run-1",
            "awaiting_approval",
        )
    )
    asyncio.run(
        runtime_service._sync_goal_from_agent_run_by_run_id(
            "linked-checkpoint-overdue-approval-run-1"
        )
    )

    with TestClient(app) as client:
        health_response = client.get("/v1/goals/goal-checkpoint-overdue-approval-1/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()

    findings_after = asyncio.run(
        runtime_service._store.list_goal_audit_findings(
            "goal-checkpoint-overdue-approval-1",
            status="open",
        )
    )
    assert health_payload["status"] == "waiting_approval"
    assert health_payload["linked_agent_run"]["status"] == "awaiting_approval"
    assert health_payload["checkpoint_policy"]["status"] == "recorded"
    assert findings_after == []


def test_goal_pause_does_not_overwrite_a_concurrently_completed_goal(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
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

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-pause-race-1",
            objective="Pause should not overwrite a completed linked run",
            protocol_id="teacher_student_distill",
            summary={"phase": "running"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-pause-race-attempt-1",
            goal_id="goal-pause-race-1",
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id="linked-run-pause-race-1",
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-pause-race-1",
            "running",
            current_attempt_id="goal-pause-race-attempt-1",
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id="linked-run-pause-race-1",
            protocol_id="teacher_student_distill",
            title="Pause race linked run",
            topic="pause race",
            summary={
                "goal_id": "goal-pause-race-1",
                "goal_attempt_id": "goal-pause-race-attempt-1",
                "objective": "Pause should not overwrite a completed linked run",
                "task_input": "Pause should not overwrite a completed linked run",
            },
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status("linked-run-pause-race-1", "running"))
    asyncio.run(
        runtime_service._store.upsert_goal_lease(
            goal_id="goal-pause-race-1",
            owner_id="runtime-owner-test",
            metadata={"reason": "pause_race_test"},
            acquired_at="2026-06-22T10:00:00+00:00",
            heartbeat_at="2026-06-22T10:00:10+00:00",
            expires_at="2026-06-22T10:02:10+00:00",
        )
    )

    async def _complete_on_pause(run_id: str) -> dict[str, Any] | None:
        await runtime_service._store.update_agent_run_status(run_id, "succeeded", latest_error=None)
        run = await runtime_service._store.get_agent_run(run_id)
        assert run is not None
        await runtime_service._sync_goal_from_agent_run(
            goal_id="goal-pause-race-1",
            attempt_id="goal-pause-race-attempt-1",
            run=run,
        )
        return run

    monkeypatch.setattr(runtime_service, "pause_agent_run", _complete_on_pause)

    with TestClient(app) as client:
        pause_response = client.post("/v1/goals/goal-pause-race-1/pause")
        assert pause_response.status_code == 200
        payload = pause_response.json()
        health_response = client.get("/v1/goals/goal-pause-race-1/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()

    assert payload["status"] == "completed"
    assert payload["attempts"][0]["status"] == "completed"
    assert health_payload["lease"] is None


def test_goal_estop_persists_across_restart_and_blocks_manual_start(tmp_path: Path) -> None:
    app, _runtime_service = _create_goal_test_app(tmp_path)

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Pause long-running automation while the operator performs maintenance.",
                "protocol_id": "teacher_student_distill",
            },
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        estop_response = client.post(
            "/v1/goals/estop",
            json={"stop_all_goals": True, "reason": "maintenance window"},
        )
        assert estop_response.status_code == 200
        estop_payload = estop_response.json()
        assert estop_payload["stop_all_goals"] is True
        assert estop_payload["metadata"]["reason"] == "maintenance window"

        audit_response = client.get("/v1/goals/operator-audit-log?event_type=estop_update")
        assert audit_response.status_code == 200
        audit_entries = audit_response.json()
        assert audit_entries
        assert audit_entries[0]["action"] == "update"
        assert audit_entries[0]["details"]["controls"]["stop_all_goals"] is True

    restarted_app, _restarted_service = _create_goal_test_app(tmp_path)
    with TestClient(restarted_app) as restarted_client:
        estop_status_response = restarted_client.get("/v1/goals/estop")
        assert estop_status_response.status_code == 200
        estop_status = estop_status_response.json()
        assert estop_status["stop_all_goals"] is True
        assert estop_status["metadata"]["reason"] == "maintenance window"

        start_response = restarted_client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200
        started = start_response.json()
        assert started["status"] == "created"
        assert "operator emergency stop" in str(started["latest_error"])

        health_response = restarted_client.get(f"/v1/goals/{goal_id}/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()
        assert health_payload["operator_controls"]["stop_all_goals"] is True
        assert "operator emergency stop" in str(health_payload["latest_error"])


def test_goal_estop_update_pauses_running_goal(tmp_path: Path) -> None:
    app, runtime_service = _create_goal_test_app(tmp_path)

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-estop-running-1",
            objective="Pause a running goal when operator estop is enabled.",
            protocol_id="teacher_student_distill",
            summary={"phase": "running"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-estop-running-attempt-1",
            goal_id="goal-estop-running-1",
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id="goal-estop-linked-run-1",
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-estop-running-1",
            "running",
            current_attempt_id="goal-estop-running-attempt-1",
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id="goal-estop-linked-run-1",
            protocol_id="teacher_student_distill",
            title="Goal estop linked run",
            topic="estop pause",
            summary={
                "goal_id": "goal-estop-running-1",
                "goal_attempt_id": "goal-estop-running-attempt-1",
            },
        )
    )
    asyncio.run(
        runtime_service._store.update_agent_run_status(
            "goal-estop-linked-run-1",
            "running",
        )
    )

    with TestClient(app) as client:
        estop_response = client.post(
            "/v1/goals/estop",
            json={"stop_all_goals": True, "reason": "operator pause"},
        )
        assert estop_response.status_code == 200

        goal_response = client.get("/v1/goals/goal-estop-running-1")
        assert goal_response.status_code == 200
        goal_payload = goal_response.json()

        linked_run_response = client.get("/v1/agent-runs/goal-estop-linked-run-1")
        assert linked_run_response.status_code == 200
        linked_run = linked_run_response.json()

    assert goal_payload["status"] == "paused"
    assert goal_payload["attempts"][0]["status"] == "paused"
    assert "operator emergency stop" in str(goal_payload["latest_error"])
    assert linked_run["status"] == "paused"


def test_goal_operator_controls_propagate_into_linked_run_request_metadata(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured_requests: list[Any] = []

    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        captured_requests.append(request)
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Goal-linked agent run completed with operator controls."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    with _create_goal_test_client(tmp_path) as client:
        estop_response = client.post(
            "/v1/goals/estop",
            json={
                "blocked_tools": [" file_read ", "web_search", "file_read", ""],
                "blocked_domains": [
                    " Example.com ",
                    "blocked.example.org",
                    "example.com",
                ],
                "block_network_usage": True,
                "reason": "restrict network during unattended execution",
            },
        )
        assert estop_response.status_code == 200

        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Run with operator-imposed tool restrictions.",
                "title": "Goal Operator Controls Propagation",
                "protocol_id": "teacher_student_distill",
            },
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200

        completed_goal = _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)
        assert completed_goal["status"] == "completed"

        health_response = client.get(f"/v1/goals/{goal_id}/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()

    assert len(captured_requests) == 1
    operator_controls = captured_requests[0].metadata["goal_operator_controls"]
    assert operator_controls["stop_all_goals"] is False
    assert operator_controls["blocked_tools"] == ["file_read", "web_search"]
    assert operator_controls["blocked_domains"] == ["example.com", "blocked.example.org"]
    assert operator_controls["block_network_usage"] is True
    assert operator_controls["tool_denylist"] == [
        "file_read",
        "web_search",
        "web_fetch",
        "web_crawl",
        "arxiv_search",
        "semantic_scholar_search",
        "crossref_search",
        "pubmed_search",
    ]
    assert captured_requests[0].metadata["permission_policy"] == {
        "blocked_web_domains": ["example.com", "blocked.example.org"],
    }
    assert operator_controls["metadata"]["reason"] == "restrict network during unattended execution"
    assert health_payload["operator_controls"]["tool_denylist"] == operator_controls["tool_denylist"]
    assert health_payload["operator_controls"]["blocked_domains"] == operator_controls["blocked_domains"]


def test_goal_operator_audit_log_records_linked_approval_resolution(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    approval_id = "exec-approval-goal-audit-log-1"
    sessions_dir = tmp_path / "sessions"
    monkeypatch.setattr(
        "mochi.runtime.service.MultiAgentOrchestrator.run",
        _build_goal_linked_exec_approval_orchestrator(
            approval_id=approval_id,
            workdir=tmp_path,
            final_answer="Goal approval audit log completed.",
        ),
    )

    with _create_goal_exec_test_client(
        sessions_dir=sessions_dir,
        exec_approval_store=InMemoryApprovalStore(),
    ) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Record linked approval resolutions in the operator audit log.",
                "title": "Goal Approval Audit Log",
                "protocol_id": "controlled_subagent_execution",
            },
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200

        waiting_goal = _wait_goal_until(client, goal_id, {"waiting_approval"}, timeout_seconds=4.0)
        assert waiting_goal["status"] == "waiting_approval"

    with _create_goal_exec_test_client(
        sessions_dir=sessions_dir,
        exec_approval_store=InMemoryApprovalStore(),
    ) as restarted_client:
        resolve_response = restarted_client.post(
            f"/v1/approvals/{approval_id}/resolve",
            json={"decision": "approve_once", "reason": "allow linked goal audit logging"},
        )
        assert resolve_response.status_code == 200

        completed_goal = _wait_goal_until(
            restarted_client,
            goal_id,
            {"completed"},
            timeout_seconds=4.0,
        )
        assert completed_goal["status"] == "completed"

        audit_response = restarted_client.get(
            "/v1/goals/operator-audit-log?event_type=approval_resolution"
        )
        assert audit_response.status_code == 200
        audit_entries = audit_response.json()

    matching_entries = [item for item in audit_entries if item["subject_id"] == approval_id]
    assert matching_entries
    entry = matching_entries[0]
    assert entry["action"] == "approve_once"
    assert entry["details"]["goal_id"] == goal_id
    assert entry["details"]["linked_exec_approval_id"] == approval_id
    assert entry["details"]["source"] == "linked_exec_approval_rehydrated"


def test_goal_cancel_does_not_overwrite_a_concurrently_completed_goal(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
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

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-cancel-race-1",
            objective="Cancel should not overwrite a completed linked run",
            protocol_id="teacher_student_distill",
            summary={"phase": "running"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-cancel-race-attempt-1",
            goal_id="goal-cancel-race-1",
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id="linked-run-cancel-race-1",
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-cancel-race-1",
            "running",
            current_attempt_id="goal-cancel-race-attempt-1",
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id="linked-run-cancel-race-1",
            protocol_id="teacher_student_distill",
            title="Cancel race linked run",
            topic="cancel race",
            summary={
                "goal_id": "goal-cancel-race-1",
                "goal_attempt_id": "goal-cancel-race-attempt-1",
                "objective": "Cancel should not overwrite a completed linked run",
                "task_input": "Cancel should not overwrite a completed linked run",
            },
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status("linked-run-cancel-race-1", "running"))
    asyncio.run(
        runtime_service._store.upsert_goal_lease(
            goal_id="goal-cancel-race-1",
            owner_id="runtime-owner-test",
            metadata={"reason": "cancel_race_test"},
            acquired_at="2026-06-22T10:00:00+00:00",
            heartbeat_at="2026-06-22T10:00:10+00:00",
            expires_at="2026-06-22T10:02:10+00:00",
        )
    )

    async def _complete_on_cancel(run_id: str) -> dict[str, Any] | None:
        await runtime_service._store.update_agent_run_status(run_id, "succeeded", latest_error=None)
        run = await runtime_service._store.get_agent_run(run_id)
        assert run is not None
        await runtime_service._sync_goal_from_agent_run(
            goal_id="goal-cancel-race-1",
            attempt_id="goal-cancel-race-attempt-1",
            run=run,
        )
        return run

    monkeypatch.setattr(runtime_service, "cancel_agent_run", _complete_on_cancel)

    with TestClient(app) as client:
        cancel_response = client.post("/v1/goals/goal-cancel-race-1/cancel")
        assert cancel_response.status_code == 200
        payload = cancel_response.json()
        health_response = client.get("/v1/goals/goal-cancel-race-1/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()

    assert payload["status"] == "completed"
    assert payload["attempts"][0]["status"] == "completed"
    assert health_payload["lease"] is None
