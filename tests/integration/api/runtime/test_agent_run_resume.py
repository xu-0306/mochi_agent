"""Runtime API tests grouped by ownership."""

from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from mochi.agents.multi_agent.orchestrator import MultiAgentRunResult
from mochi.api.server import create_app
from mochi.config.schema import MochiConfig
from mochi.runtime.approvals import APPROVAL_OWNER_TASK_ID_KEY, InMemoryApprovalStore
from mochi.runtime.exec_runtime import ExecRuntime
from mochi.runtime.service import (
    RuntimeService,
    _ensure_agent_run_resume_payload,
    _resolve_agent_run_resume_strategy,
)
from mochi.runtime.store import RuntimeStore
from tests.support.exec_providers import PythonDirectProvider as _ApiRuntimePythonDirectProvider

from ._support import (
    _ResumeStrategyRecordingService,
    _RuntimeFakeEngine,
    _wait_agent_run_until,
    _wait_until,
)


def test_agent_run_resume_defaults_to_checkpoint_continue_strategy(tmp_path: Path) -> None:
    app = create_app()
    service = _ResumeStrategyRecordingService()
    app.state.runtime_service = service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        response = client.post("/v1/agent-runs/run-resume-default/resume")
        assert response.status_code == 200
        payload = response.json()

    assert service.calls == [
        {
            "run_id": "run-resume-default",
            "strategy": "continue_from_checkpoint",
            "checkpoint_used": True,
        }
    ]
    assert payload["recovery_state"]["resume_strategy"] == "continue_from_checkpoint"
    assert payload["recovery_state"]["checkpoint"]["checkpoint_index"] == 2

def test_agent_run_resume_restart_attempt_ignores_checkpoint_payload(tmp_path: Path) -> None:
    app = create_app()
    service = _ResumeStrategyRecordingService()
    app.state.runtime_service = service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/agent-runs/run-resume-restart/resume",
            json={"strategy": "restart_attempt"},
        )
        assert response.status_code == 200
        payload = response.json()

    assert service.calls == [
        {
            "run_id": "run-resume-restart",
            "strategy": "restart_attempt",
            "checkpoint_used": False,
        }
    ]
    assert payload["recovery_state"]["resume_strategy"] == "restart_attempt"
    assert payload["recovery_state"]["checkpoint"] is None

def test_resolve_agent_run_resume_strategy_reads_structured_resume_executor() -> None:
    run = {
        "summary": {
            "recovery_state": {
                "resume_payload": {
                    "executor": "restart_attempt",
                    "stage": "protocol_completed",
                    "checkpoint": {
                        "checkpoint_index": 3,
                        "stage": "protocol_completed",
                    },
                }
            }
        }
    }

    assert _resolve_agent_run_resume_strategy(None, run) == "restart_attempt"

def test_ensure_agent_run_resume_payload_upgrades_legacy_payload_shape() -> None:
    run = {
        "protocol_id": "teacher_student_distill",
        "task": "Summarize the deployment risks.",
        "events": [
            {
                "type": "role_output",
                "payload": {
                    "candidate_id": "student",
                    "role_id": "student",
                    "content": "Student answer",
                    "model_id": "student-model",
                    "round_index": 1,
                },
            },
            {
                "type": "evaluation",
                "payload": {
                    "policy": {"mode": "judge"},
                    "selected_candidate_id": "student",
                    "scores": [{"candidate_id": "student", "score": 0.9}],
                },
            },
        ],
        "artifacts": [
            {
                "artifact_type": "run_summary",
                "metadata": {
                    "content": {
                        "protocol": "teacher_student_distill",
                        "selected_candidate_id": "student",
                        "final_answer": "Student answer",
                        "candidate_count": 1,
                        "debate_transcript": {"items": []},
                    }
                },
            },
            {
                "artifact_type": "evidence_summary",
                "metadata": {
                    "content": {
                        "evidence_packets": [
                            {
                                "evidence_id": "src-1",
                                "title": "Deployment note",
                            }
                        ],
                        "collected_packet_count": 1,
                    }
                },
            },
            {
                "artifact_type": "verification_summary",
                "metadata": {
                    "content": {
                        "verified_candidate_ids": ["student"],
                        "verifications": [
                            {
                                "candidate_id": "student",
                                "status": "verified",
                                "summary": "Matches evidence.",
                                "citations": [],
                            }
                        ],
                    }
                },
            },
            {
                "artifact_type": "role_task_snapshot",
                "metadata": {
                    "content": {
                        "roles": {
                            "teacher": {
                                "role_id": "teacher",
                                "status": "completed",
                                "stage": "teacher_generation",
                                "assigned_model_id": "teacher-model",
                                "candidate": {
                                    "candidate_id": "teacher",
                                    "role_id": "teacher",
                                    "content": "Teacher answer",
                                    "metadata": {"model_id": "teacher-model"},
                                },
                            }
                        }
                    }
                },
            },
        ],
        "summary": {
            "evidence_packets": [
                {
                    "evidence_id": "src-1",
                    "title": "Deployment note",
                }
            ]
        },
    }
    summary = {
        "recovery_state": {
            "status": "awaiting_resources",
            "action": "pause",
            "reason": "budget exhausted",
            "stage": "evaluation_completed",
            "checkpoint": {
                "checkpoint_index": 7,
                "stage": "evaluation_completed",
            },
            "resume_payload": {
                "strategy_default": "continue_from_checkpoint",
                "state": {"legacy": True},
            },
        }
    }

    upgraded = _ensure_agent_run_resume_payload(
        run=run,
        summary=summary,
        strategy="continue_from_checkpoint",
    )

    payload = upgraded["recovery_state"]["resume_payload"]
    assert payload["stage"] == "evaluation_completed"
    assert payload["executor"] == "continue_from_checkpoint"
    assert payload["checkpoint"]["checkpoint_index"] == 7
    assert payload["candidates"][0]["candidate_id"] == "student"
    assert payload["evaluation"]["selected_candidate_id"] == "student"
    assert payload["verification_summary"]["verified_candidate_ids"] == ["student"]
    assert payload["role_task_snapshot"]["roles"]["teacher"]["status"] == "completed"

def test_ensure_agent_run_resume_payload_includes_live_collector_shard_artifacts() -> None:
    run = {
        "protocol_id": "teacher_student_distill",
        "events": [],
        "artifacts": [
            {
                "artifact_type": "collector_shard_manifest",
                "artifact_id": "collector-live-1",
                "metadata": {
                    "attempt_id": "attempt-1",
                    "content": {
                        "shards": [
                            {
                                "shard_id": "discourse-topic-274354",
                                "adapter_name": "discourse_topic_adapter",
                                "status": "running",
                                "updated_at": "2026-06-24T00:00:00+00:00",
                                "progress": {
                                    "cursor": "101",
                                    "items_collected": 2,
                                    "items_emitted": 2,
                                },
                            }
                        ]
                    },
                },
            },
            {
                "artifact_type": "dataset_record",
                "artifact_id": "collector-record-live-1",
                "metadata": {
                    "attempt_id": "attempt-1",
                    "record": {
                        "input": "Topic: API examples",
                        "target": {"answer": "First collected post"},
                        "metadata": {
                            "capability_family": "dataset_collection",
                            "collector_provenance": {
                                "source_url": "https://forum.example/t/api-examples/274354/1",
                                "source_id": "topic:274354:post:1",
                                "adapter_name": "discourse_topic_adapter",
                                "tool_name": "discourse_topic_collect",
                                "policy_disposition": "allow",
                                "shard_id": "discourse-topic-274354",
                            },
                        },
                    },
                },
            },
        ],
        "summary": {},
        "schedule": {"current_attempt_id": "attempt-1"},
    }
    summary = {
        "recovery_state": {
            "status": "awaiting_resources",
            "action": "pause",
            "reason": "collector interrupted",
            "stage": "protocol_completed",
            "checkpoint": {
                "checkpoint_index": 3,
                "stage": "protocol_completed",
            },
        }
    }

    upgraded = _ensure_agent_run_resume_payload(
        run=run,
        summary=summary,
        strategy="continue_from_checkpoint",
    )

    payload = upgraded["recovery_state"]["resume_payload"]["protocol_artifacts"]
    assert payload["collector_shard_manifests"]["shards"][0]["shard_id"] == "discourse-topic-274354"
    assert payload["collector_shard_manifests"]["shards"][0]["progress"]["cursor"] == "101"
    assert payload["collector_dataset_records"]["records"][0]["target"]["answer"] == (
        "First collected post"
    )
    assert payload["collector_record_provenance"]["records"][0]["source_id"] == (
        "topic:274354:post:1"
    )

def test_agent_run_resume_duplicate_requests_do_not_enqueue_parallel_recovery_jobs(
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

    run_id = "run-resume-dedupe"
    recovery_summary = {
        "task_input": "resume the latest checkpoint",
        "recovery_state": {
            "status": "awaiting_resources",
            "action": "pause",
            "reason": "quota exhausted",
            "stage": "teacher_generation",
            "checkpoint": {
                "checkpoint_index": 2,
                "stage": "teacher_generation",
                "task_input": "resume the latest checkpoint",
            },
            "unfinished_steps": ["resume from the latest checkpoint"],
            "recommended_resume_conditions": ["restore quota before resuming"],
        },
    }
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id=run_id,
            protocol_id="teacher_student_distill",
            title="Resume dedupe run",
            topic="avoid duplicate recovery jobs",
            summary=recovery_summary,
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status(run_id, "awaiting_resources"))

    resume_started = threading.Event()
    release_resume = threading.Event()
    resume_call_count = 0

    async def _blocking_run_agent_run(self: RuntimeService, *, run_id: str) -> None:
        nonlocal resume_call_count
        del self, run_id
        resume_call_count += 1
        resume_started.set()
        await asyncio.to_thread(release_resume.wait, 1.0)

    monkeypatch.setattr(RuntimeService, "_run_agent_run", _blocking_run_agent_run)

    with TestClient(app) as client:
        first_response = client.post(f"/v1/agent-runs/{run_id}/resume")
        assert first_response.status_code == 200
        assert resume_started.wait(timeout=1.0)

        second_response = client.post(f"/v1/agent-runs/{run_id}/resume")
        assert second_response.status_code == 200
        assert second_response.json()["status"] == "running"

        time.sleep(0.1)
        release_resume.set()

    assert resume_call_count == 1

def test_agent_run_resume_upgrades_legacy_payload_before_recovery_execution(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured_requests: list[Any] = []

    async def _capture_resumed_request(self: Any, request: Any) -> MultiAgentRunResult:
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
            artifacts={"final_answer": "Legacy non-goal resume was upgraded before execution."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _capture_resumed_request)

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

    run_id = "run-legacy-resume-upgrade-1"
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id=run_id,
            protocol_id="teacher_student_distill",
            title="Legacy non-goal resume",
            topic="upgrade non-goal resume payload",
            summary={
                "task_input": "Resume the latest checkpoint without a linked goal.",
                "evidence_packets": [
                    {
                        "evidence_id": "src-1",
                        "title": "Deployment note",
                    }
                ],
                "recovery_state": {
                    "status": "awaiting_resources",
                    "action": "pause",
                    "reason": "budget exhausted",
                    "stage": "research_context_prepared",
                    "checkpoint": {
                        "checkpoint_index": 7,
                        "stage": "research_context_prepared",
                    },
                    "resume_payload": {
                        "strategy_default": "continue_from_checkpoint",
                        "state": {"legacy": True},
                    },
                },
            },
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status(run_id, "awaiting_resources"))
    asyncio.run(
        runtime_service._store.append_agent_run_event(
            run_id,
            {
                "type": "role_output",
                "payload": {
                    "candidate_id": "student",
                    "role_id": "student",
                    "content": "Student answer",
                    "model_id": "student-model",
                    "round_index": 1,
                },
            },
        )
    )
    asyncio.run(
        runtime_service._store.append_agent_run_event(
            run_id,
            {
                "type": "evaluation",
                "payload": {
                    "policy": {"mode": "judge"},
                    "selected_candidate_id": "student",
                    "scores": [{"candidate_id": "student", "score": 0.9}],
                },
            },
        )
    )
    asyncio.run(
        runtime_service._store.append_agent_run_artifact(
            run_id,
            artifact_id="run-legacy-resume-upgrade-1:run-summary",
            artifact_type="run_summary",
            title="Run Summary",
            uri=f"agent-run://{run_id}/artifacts/run-summary",
            metadata={
                "content": {
                    "protocol": "teacher_student_distill",
                    "selected_candidate_id": "student",
                    "final_answer": "Student answer",
                    "candidate_count": 1,
                    "debate_transcript": {"items": []},
                }
            },
        )
    )
    asyncio.run(
        runtime_service._store.append_agent_run_artifact(
            run_id,
            artifact_id="run-legacy-resume-upgrade-1:evidence-summary",
            artifact_type="evidence_summary",
            title="Evidence Summary",
            uri=f"agent-run://{run_id}/artifacts/evidence-summary",
            metadata={
                "content": {
                    "evidence_packets": [
                        {
                            "evidence_id": "src-1",
                            "title": "Deployment note",
                        }
                    ],
                    "collected_packet_count": 1,
                }
            },
        )
    )
    asyncio.run(
        runtime_service._store.append_agent_run_artifact(
            run_id,
            artifact_id="run-legacy-resume-upgrade-1:verification-summary",
            artifact_type="verification_summary",
            title="Verification Summary",
            uri=f"agent-run://{run_id}/artifacts/verification-summary",
            metadata={
                "content": {
                    "verified_candidate_ids": ["student"],
                    "verifications": [
                        {
                            "candidate_id": "student",
                            "status": "verified",
                            "summary": "Matches evidence.",
                            "citations": [],
                        }
                    ],
                }
            },
        )
    )
    asyncio.run(
        runtime_service._store.append_agent_run_artifact(
            run_id,
            artifact_id="run-legacy-resume-upgrade-1:role-task-snapshot",
            artifact_type="role_task_snapshot",
            title="Role Task Snapshot",
            uri=f"agent-run://{run_id}/artifacts/role-task-snapshot",
            metadata={
                "content": {
                    "roles": {
                        "teacher": {
                            "role_id": "teacher",
                            "status": "completed",
                            "stage": "teacher_generation",
                            "assigned_model_id": "teacher-model",
                            "candidate": {
                                "candidate_id": "teacher",
                                "role_id": "teacher",
                                "content": "Teacher answer",
                                "metadata": {"model_id": "teacher-model"},
                            },
                        }
                    }
                }
            },
        )
    )

    with TestClient(app) as client:
        resume_response = client.post(f"/v1/agent-runs/{run_id}/resume")
        assert resume_response.status_code == 200

        completed = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)

    assert completed["summary"]["final_answer"] == (
        "Legacy non-goal resume was upgraded before execution."
    )
    assert len(captured_requests) == 1
    assert captured_requests[0].metadata["resume_strategy"] == "continue_from_checkpoint"
    resume_payload = captured_requests[0].metadata["resume_payload"]
    assert resume_payload["stage"] == "research_context_prepared"
    assert resume_payload["executor"] == "continue_from_checkpoint"
    assert resume_payload["supported_actions"] == ["restart_attempt", "continue_from_checkpoint"]
    assert resume_payload["checkpoint"]["checkpoint_index"] == 7

def test_agent_run_resume_endpoint_resolves_exec_approval_before_resuming(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    approval_id = "exec-approval-agent-run-resume-1"

    async def _approval_then_success_run(self: Any, request: Any) -> MultiAgentRunResult:
        approval = self._exec_approval_store.get(approval_id)
        if approval is None or approval.status == "pending":
            if approval is None:
                self._exec_approval_store.create(
                    approval_id=approval_id,
                    command="print('agent run approval resume')",
                    shell="test",
                    scope="dangerous_command",
                    reason="Exec command requires approval.",
                    command_payload={
                        "command": "print('agent run approval resume')",
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
            artifacts={"final_answer": "Approved agent run completed."},
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
            providers={"test": _ApiRuntimePythonDirectProvider()},
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
            "/v1/agent-runs",
            json={
                "protocol_id": "controlled_subagent_execution",
                "title": "Approval resume run",
                "topic": "resume after approval",
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        waiting = _wait_agent_run_until(client, run_id, {"awaiting_approval"}, timeout_seconds=4.0)
        assert waiting["summary"]["approval_state"]["approval_ids"] == [approval_id]

        resume_response = client.post(
            f"/v1/agent-runs/{run_id}/resume",
            json={
                "approval_id": approval_id,
                "decision": "approve_once",
                "reason": "approved for resume",
                "strategy": "continue_from_checkpoint",
            },
        )
        assert resume_response.status_code == 200
        assert resume_response.json()["run_id"] == run_id

        completed = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        assert completed["summary"]["final_answer"] == "Approved agent run completed."

    resolved = exec_approval_store.get(approval_id)
    assert resolved is not None
    assert resolved.status == "consumed"
    assert resolved.execution_result is not None

def test_approval_resolve_endpoint_auto_resumes_linked_agent_run(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    approval_id = "exec-approval-agent-run-auto-resume-1"
    resumed_payloads: list[dict[str, Any]] = []

    async def _approval_then_success_run(self: Any, request: Any) -> MultiAgentRunResult:
        approval = self._exec_approval_store.get(approval_id)
        if approval is None or approval.status == "pending":
            if approval is None:
                self._exec_approval_store.create(
                    approval_id=approval_id,
                    command="print('agent run auto resume')",
                    shell="test",
                    scope="dangerous_command",
                    reason="Exec command requires approval.",
                    command_payload={
                        "command": "print('agent run auto resume')",
                        "shell": "test",
                        "workdir": str(tmp_path),
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
                                "command": "print('agent run auto resume')",
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
        resume_payload = (
            dict(request.metadata.get("resume_payload"))
            if isinstance(request.metadata.get("resume_payload"), dict)
            else {}
        )
        resumed_payloads.append(resume_payload)
        role_task_snapshot = (
            dict(resume_payload.get("role_task_snapshot"))
            if isinstance(resume_payload.get("role_task_snapshot"), dict)
            else {}
        )
        tasks = dict(role_task_snapshot.get("tasks")) if isinstance(role_task_snapshot.get("tasks"), dict) else {}
        exec_task = (
            dict(tasks.get("controlled_execution_exec:req-1"))
            if isinstance(tasks.get("controlled_execution_exec:req-1"), dict)
            else {}
        )
        exec_result = (
            dict(exec_task.get("result_summary", {}).get("execution_result"))
            if isinstance(exec_task.get("result_summary"), dict)
            and isinstance(exec_task.get("result_summary", {}).get("execution_result"), dict)
            else {}
        )
        assert exec_result["status"] == "completed"
        assert exec_result["request_id"] == "req-1"
        assert isinstance(exec_result.get("session_id"), str)
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="controlled_subagent_execution",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Approval auto-resume completed."},
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
            providers={"test": _ApiRuntimePythonDirectProvider()},
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
            "/v1/agent-runs",
            json={
                "protocol_id": "controlled_subagent_execution",
                "title": "Approval auto-resume run",
                "topic": "generic approval resolve auto-resume",
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        waiting = _wait_agent_run_until(client, run_id, {"awaiting_approval"}, timeout_seconds=4.0)
        assert waiting["summary"]["approval_state"]["approval_ids"] == [approval_id]

        resolve_response = client.post(
            f"/v1/approvals/{approval_id}/resolve",
            json={"decision": "approve_once", "reason": "allow linked auto resume"},
        )
        assert resolve_response.status_code == 200
        assert resolve_response.json()["status"] == "consumed"

        completed = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        assert completed["summary"]["final_answer"] == "Approval auto-resume completed."

    assert resumed_payloads
    resolved = exec_approval_store.get(approval_id)
    assert resolved is not None
    assert resolved.status == "consumed"
    assert resolved.execution_result is not None

def test_task_resume_endpoint_applies_approval_override(tmp_path: Path) -> None:
    app = create_app()
    engine = _RuntimeFakeEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/tasks",
            json={
                "input_message": "run with resume endpoint",
                "workspace_dir": str(tmp_path / "workspace"),
            },
        )
        assert create_response.status_code == 200
        task_id = create_response.json()["task_id"]

        waiting_payload = _wait_until(client, task_id, {"awaiting_approval"})
        assert waiting_payload["pending_approval"] is not None

        resume_response = client.post(
            f"/v1/tasks/{task_id}/resume",
            json={"decision": "approve_once", "reason": "manual resume"},
        )
        assert resume_response.status_code == 200
        resumed = resume_response.json()
        assert resumed["status"] in {"resumed", "running"}

        done_payload = _wait_until(client, task_id, {"succeeded"})
        assert done_payload["status"] == "succeeded"
        assert done_payload["final_answer"] == "done"

    assert len(engine.permission_policy_calls) == 2
    assert engine.permission_policy_calls[1] == {
        "autonomy_mode": "trusted_workspace",
        "require_approval_for_file_write": False,
        "require_approval_for_exec": True,
        "file_read_scope": "workspace",
        "file_write_scope": "workspace",
        APPROVAL_OWNER_TASK_ID_KEY: task_id,
        "approved_tool_calls": [
            {
                "tool_name": "exec_command",
                "arguments": {"command": "dir", "shell": "cmd"},
            }
        ],
    }
