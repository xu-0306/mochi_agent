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
        assert health_payload["checkpoint_policy"]["status"] == "recorded"
        assert health_payload["checkpoint_policy"]["interval_sec"] == 60
        assert health_payload["checkpoint_policy"]["generation_refresh_interval_sec"] == 1_800
        assert health_payload["checkpoint_policy"]["context_handoff_threshold"] == 0.8
        assert health_payload["checkpoint_policy"]["checkpoint_index"] == 3
        assert health_payload["checkpoint_policy"]["stage"] == "controller_decision"
        assert health_payload["linked_agent_run"]["status"] == "stalled"
        assert health_payload["linked_agent_run"]["approval_state"]["pending_count"] == 1
        assert health_payload["linked_agent_run"]["checkpoint_policy"]["status"] == "recorded"


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
    assert health_payload["memory_snapshot"]["snapshot"]["role_task_summary"]["tracked_role_count"] == 2
    assert health_payload["memory_snapshot"]["snapshot"]["important_artifacts"][0]["artifact_type"] == (
        "evidence_bundle"
    )


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
        assert health_payload["linked_agent_run"]["status"] == "awaiting_approval"
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
            payload={"checkpoint_index": 5, "stage": "controller_decision"},
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
                "objective": "Collect multimodal datasets for long-running training jobs",
                "title": "Dataset Collection Goal",
                "goal_type": "dataset_collection",
                "protocol_id": "teacher_student_distill",
                "topic": "multimodal corpora",
                "project_id": "proj-goal",
                "workspace_dir": str(tmp_path / "workspace"),
                "run_policy": {"max_wall_clock_sec": 18_000},
                "capability_policy": {"allowed_tools": ["web_search", "web_fetch"]},
                "source_manifest": {"sources": [{"id": "hf-datasets"}]},
                "summary": {"phase": "created"},
                "metadata": {"packet": "A"},
            },
        )
        assert create_response.status_code == 200
        created = create_response.json()
        goal_id = created["goal_id"]
        assert created["status"] == "created"
        assert created["attempts"] == []

        list_response = client.get("/v1/goals")
        assert list_response.status_code == 200
        listed = list_response.json()
        assert len(listed) == 1
        assert listed[0]["goal_id"] == goal_id

        get_response = client.get(f"/v1/goals/{goal_id}")
        assert get_response.status_code == 200
        fetched = get_response.json()
        assert fetched["objective"] == "Collect multimodal datasets for long-running training jobs"
        assert fetched["run_policy"]["max_wall_clock_sec"] == 18_000

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
        assert health_payload["lease"] is None

        linked_run_response = client.get(f"/v1/agent-runs/{linked_run_id}")
        assert linked_run_response.status_code == 200
        linked_run = linked_run_response.json()
        assert linked_run["status"] == "succeeded"
        assert linked_run["summary"]["goal_id"] == goal_id
        assert linked_run["summary"]["goal_attempt_id"] == first_attempt["attempt_id"]

        agent_runs = client.get("/v1/agent-runs")
        assert agent_runs.status_code == 200
        assert len(agent_runs.json()) == 1


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
                "protocol_id": "teacher_student_distill",
            },
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200
        started = start_response.json()
        assert len(started["attempts"]) == 1

        running_goal = _wait_goal_until(client, goal_id, {"running"}, timeout_seconds=4.0)
        assert running_goal["attempts"][0]["agent_run_id"] is not None

        pause_response = client.post(f"/v1/goals/{goal_id}/pause")
        assert pause_response.status_code == 200
        paused = pause_response.json()
        assert paused["status"] == "paused"
        assert paused["attempts"][0]["status"] == "paused"

        resume_response = client.post(f"/v1/goals/{goal_id}/resume")
        assert resume_response.status_code == 200
        resumed = resume_response.json()
        assert resumed["status"] in {"queued", "running"}
        assert len(resumed["attempts"]) == 2
        latest_attempt = resumed["attempts"][-1]
        assert resumed["current_attempt_id"] == latest_attempt["attempt_id"]
        assert latest_attempt["trigger"] == "manual_resume"

        cancel_response = client.post(f"/v1/goals/{goal_id}/cancel")
        assert cancel_response.status_code == 200
        cancelled = cancel_response.json()
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
