"""Runtime API tests grouped by ownership."""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from mochi.agents.multi_agent.orchestrator import (
    MultiAgentOrchestrator,
    MultiAgentRunEvent,
    MultiAgentRunResult,
)
from mochi.api.server import create_app
from mochi.config.schema import MochiConfig
from mochi.runtime.exec_runtime import ExecRuntime
from mochi.runtime.service import RuntimeService
from mochi.runtime.store import RuntimeStore
from tests.support.exec_providers import PythonDirectProvider as _ApiRuntimePythonDirectProvider

from ._support import (
    _BACKGROUND_SMOKE_COMMAND_RULE,
    _CONTROLLED_SMOKE_COMMAND_RULE,
    _AgentRunModelBackedEngine,
    _BackgroundControlledExecAgentRunEngine,
    _LocalModelUnavailableAgentRunEngine,
    _QuotaExhaustedAgentRunEngine,
    _RuntimeFakeEngine,
    _SlowAgentRunModelBackedEngine,
    _wait_agent_run_until,
)


def test_agent_runs_api_flow(tmp_path: Path) -> None:
    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "research-protocol-v1",
                "title": "Cancer literature review",
                "topic": "oncology",
                "reasoning_effort": "high",
                "selected_models_roles": {
                    "planner": {"model": "qwen3", "role": "lead"},
                    "worker": {"model": "llama", "role": "analysis"},
                },
                "evaluation_policy": {"strict": True},
                "schedule": {"cron": "0 9 * * 1", "timezone": "Asia/Taipei"},
                "summary": {"objective": "collect evidence"},
                "evidence_status": {"task": "pending", "rag": "pending"},
                "artifacts": [
                    {
                        "artifact_id": "artifact-1",
                        "artifact_type": "report",
                        "title": "Initial brief",
                        "uri": "memory://artifact-1",
                    }
                ],
            },
        )
        assert create_response.status_code == 200
        created = create_response.json()
        run_id = created["run_id"]
        assert created["status"] == "created"
        assert created["reasoning_effort"] == "high"
        assert created["schedule"] == {"cron": "0 9 * * 1", "timezone": "Asia/Taipei"}
        assert created["artifacts"][0]["artifact_id"] == "artifact-1"

        list_response = client.get("/v1/agent-runs")
        assert list_response.status_code == 200
        listed = list_response.json()
        assert len(listed) == 1
        assert listed[0]["run_id"] == run_id
        assert listed[0]["status"] == "created"
        assert listed[0]["reasoning_effort"] == "high"

        get_response = client.get(f"/v1/agent-runs/{run_id}")
        assert get_response.status_code == 200
        fetched = get_response.json()
        assert fetched["protocol_id"] == "research-protocol-v1"
        assert fetched["reasoning_effort"] == "high"
        assert fetched["events"][0]["type"] == "run_created"

        guidance_response = client.post(
            f"/v1/agent-runs/{run_id}/guidance",
            json={
                "guidance": "Prioritize RCT sources first.",
                "author": "user",
                "metadata": {"priority": "high"},
            },
        )
        assert guidance_response.status_code == 200
        guidance_payload = guidance_response.json()
        guidance_events = [event for event in guidance_payload["events"] if event["type"] == "guidance"]
        assert guidance_events[-1]["guidance"] == "Prioritize RCT sources first."
        assert guidance_events[-1]["author"] == "user"

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200
        assert start_response.json()["status"] in {"running", "succeeded"}

        final_payload = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        assert final_payload["finished_at"] is not None
        assert len(final_payload["artifacts"]) >= 1
        assert final_payload["summary"]["final_answer"]
        assert final_payload["evidence_status"]["selected_candidate_id"]
        event_types = [event["type"] for event in final_payload["events"]]
        assert "run_started" in event_types
        assert "role_output" in event_types
        assert "evaluation" in event_types
        assert "artifact" in event_types

def test_agent_run_pause_cancels_background_job(tmp_path: Path, monkeypatch: Any) -> None:
    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    async def _slow_run(self: Any, request: Any) -> MultiAgentRunResult:
        await asyncio.sleep(5)
        return MultiAgentRunResult(
            run_id=request.run_id or "slow-run",
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _slow_run)

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Pause test",
                "topic": "verify pause behavior",
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200
        assert start_response.json()["status"] == "running"

        pause_response = client.post(f"/v1/agent-runs/{run_id}/pause")
        assert pause_response.status_code == 200
        assert pause_response.json()["status"] == "paused"

        paused_payload = _wait_agent_run_until(client, run_id, {"paused"})
        assert paused_payload["status"] == "paused"
        assert paused_payload["finished_at"] is None
        assert "run_paused" in [event["type"] for event in paused_payload["events"]]

def test_agent_runs_api_flow_uses_configured_model_generation(tmp_path: Path) -> None:
    app = create_app()
    engine = _AgentRunModelBackedEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Model-backed run",
                "topic": "deployment summary",
                "reasoning_effort": "medium",
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                        "verifier": "verifier-model",
                        "judge": "judge-model",
                    }
                },
                "summary": {
                    "evidence_packets": [
                        {
                            "evidence_id": "src-1",
                            "title": "Deployment note",
                            "content": "The approved note matches the student answer and rejects the teacher claim.",
                        }
                    ]
                }
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        assert final_payload["summary"]["final_answer"] == "Student concise final answer."
        assert final_payload["summary"]["selected_candidate_id"] == "student"
        assert final_payload["reasoning_effort"] == "medium"
        assert final_payload["evidence_status"]["evidence_gate_counts"] == {
            "verified": 1,
            "skipped": 0,
            "failed": 1,
        }
        artifact_types = [artifact["artifact_type"] for artifact in final_payload["artifacts"]]
        assert "run_summary" in artifact_types
        assert "verification_summary" in artifact_types
        assert "subagent_runtime" in artifact_types

    assert [call["model_id"] for call in engine.calls] == [
        "teacher-model",
        "student-model",
        "verifier-model",
        "judge-model",
    ]
    assert all(call["reasoning_effort"] == "medium" for call in engine.calls)

def test_agent_runs_api_flow_supports_dr_zero_dataset_package(tmp_path: Path) -> None:
    app = create_app()
    engine = _AgentRunModelBackedEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "dr_zero_self_evolve",
                "title": "Dr.Zero run",
                "topic": "deployment search self-evolution",
                "selected_models_roles": {
                    "by_role": {
                        "proposer": "proposer-model",
                        "solver": "solver-model",
                        "verifier": "verifier-model",
                        "judge": "judge-model",
                    }
                },
                "summary": {
                    "protocol_config": {
                        "proposal_sample_size": 1,
                        "solver_rollouts_per_task": 1,
                    },
                    "evidence_packets": [
                        {
                            "evidence_id": "src-1",
                            "title": "Deployment note",
                            "content": "The supported deployment answer matches the solver.",
                        }
                    ],
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        assert final_payload["protocol_id"] == "dr_zero_self_evolve"
        assert final_payload["summary"]["selected_candidate_id"] == "solver_1_1"
        artifact_types = [artifact["artifact_type"] for artifact in final_payload["artifacts"]]
        assert "dataset_record" in artifact_types
        assert "dataset_package_snapshot" in artifact_types
        assert "synthetic_tasks" in artifact_types
        assert "solver_rollouts" in artifact_types
        assert "reward_summary" in artifact_types

        dataset_response = client.get(f"/v1/agent-runs/{run_id}/packages/dataset")
        assert dataset_response.status_code == 200
        dataset_package = dataset_response.json()
        assert dataset_package["protocol_id"] == "dr_zero_self_evolve"
        assert dataset_package["training_ready_count"] == 1
        record = dataset_package["training_ready_records"][0]["record"]
        assert record["metadata"]["dataset_mode"] == "self_evolve_search"
        assert record["synthetic_tasks"]["tasks"][0]["task_id"] == "task-1"

    assert [call["model_id"] for call in engine.calls] == [
        "proposer-model",
        "solver-model",
        "verifier-model",
        "judge-model",
    ]

def test_agent_runs_api_flow_supports_controlled_execution_dataset_package(tmp_path: Path) -> None:
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
            "/v1/agent-runs",
            json={
                "protocol_id": "controlled_subagent_execution",
                "title": "Controlled execution run",
                "topic": "run a controlled smoke command",
                "selected_models_roles": {
                    "by_role": {
                        "planner": "controlled-planner-model",
                        "executor": "controlled-executor-model",
                        "controller": "controlled-controller-model",
                        "evaluator": "controlled-evaluator-model",
                        "judge": "controlled-judge-model",
                        "verifier": "controlled-verifier-model",
                    }
                },
                "summary": {
                    "protocol_config": {
                        "max_execution_requests": 1,
                        "default_timeout_sec": 30,
                        "background_allowed": False,
                    },
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        assert final_payload["protocol_id"] == "controlled_subagent_execution"
        artifact_types = [artifact["artifact_type"] for artifact in final_payload["artifacts"]]
        assert "execution_requests" in artifact_types
        assert "controller_decisions" in artifact_types
        assert "execution_results" in artifact_types
        assert "controlled_execution_runtime" in artifact_types

        dataset_response = client.get(f"/v1/agent-runs/{run_id}/packages/dataset")
        assert dataset_response.status_code == 200
        dataset_package = dataset_response.json()
        assert dataset_package["protocol_id"] == "controlled_subagent_execution"
        record = dataset_package["all_records"][0]["record"]
        assert record["metadata"]["dataset_mode"] == "agentic_execution"
        assert record["supervision"]["type"] == "agentic_execution_trace"

def test_runtime_service_persists_live_collector_shard_snapshot_before_run_completion(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    run_id = "live-collector-run-1"
    live_snapshot_artifacts: list[dict[str, Any]] = []
    live_dataset_record_artifacts: list[dict[str, Any]] = []
    received_callback: list[object | None] = []

    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id=run_id,
            protocol_id="teacher_student_distill",
            title="Live collector snapshot run",
            topic="collector live snapshot",
            summary={
                "task_input": "Collect a Discourse topic into dataset records.",
                "objective": "Collect a Discourse topic into dataset records.",
            },
        )
    )

    original_append_agent_run_artifact = runtime_service._store.append_agent_run_artifact

    async def _capturing_append_agent_run_artifact(
        run_id: str,
        *,
        artifact_id: str | None,
        artifact_type: str,
        title: str | None = None,
        uri: str | None = None,
        mime_type: str | None = None,
        size_bytes: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await original_append_agent_run_artifact(
            run_id,
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            title=title,
            uri=uri,
            mime_type=mime_type,
            size_bytes=size_bytes,
            metadata=metadata,
        )
        if (
            artifact_type == "collector_shard_manifest"
            and isinstance(artifact_id, str)
            and "collector-shard-live" in artifact_id
        ):
            live_snapshot_artifacts.append(dict(metadata or {}))
        if (
            artifact_type == "dataset_record"
            and isinstance(artifact_id, str)
            and "collector-dataset-live" in artifact_id
        ):
            live_dataset_record_artifacts.append(dict(metadata or {}))

    monkeypatch.setattr(
        runtime_service._store,
        "append_agent_run_artifact",
        _capturing_append_agent_run_artifact,
    )

    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        received_callback.append(request.runtime_event_callback or request.event_callback)
        provenance_event = MultiAgentRunEvent(
            run_id=request.run_id or run_id,
            seq=5,
            type="artifact",
            payload={
                "name": "collector_record_provenance",
                "content": {
                    "records": [
                        {
                            "source_url": "https://forum.example/t/api-examples/274354/1",
                            "source_id": "topic:274354:post:1",
                            "adapter_name": "discourse_topic_adapter",
                            "tool_name": "discourse_topic_collect",
                            "policy_disposition": "allow",
                            "shard_id": "discourse-topic-274354",
                        }
                    ]
                },
            },
            timestamp="2026-06-24T00:00:00+00:00",
        )
        dataset_event = MultiAgentRunEvent(
            run_id=request.run_id or run_id,
            seq=6,
            type="artifact",
            payload={
                "name": "collector_dataset_records",
                "content": {
                    "records": [
                        {
                            "input": "Topic: API examples",
                            "target": {"answer": "First collected post"},
                            "metadata": {},
                        }
                    ]
                },
            },
            timestamp="2026-06-24T00:00:00+00:00",
        )
        shard_event = MultiAgentRunEvent(
            run_id=request.run_id or run_id,
            seq=7,
            type="artifact",
            payload={
                "name": "collector_shard_manifests",
                "content": {
                    "shards": [
                        {
                            "shard_id": "discourse-topic-274354",
                            "adapter_name": "discourse_topic_adapter",
                            "status": "running",
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
            timestamp="2026-06-24T00:00:00+00:00",
        )
        callback = request.runtime_event_callback or request.event_callback
        if callback is not None:
            await callback(provenance_event)
            await callback(dataset_event)
            await callback(shard_event)
        assert live_snapshot_artifacts
        assert live_dataset_record_artifacts
        return MultiAgentRunResult(
            run_id=run_id,
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={
                "collector_dataset_records": {
                    "records": [
                        {
                            "input": "Topic: API examples",
                            "target": {"answer": "First collected post"},
                            "metadata": {
                                "collector_provenance": {
                                    "source_url": "https://forum.example/t/api-examples/274354/1",
                                    "source_id": "topic:274354:post:1",
                                    "adapter_name": "discourse_topic_adapter",
                                    "tool_name": "discourse_topic_collect",
                                    "policy_disposition": "allow",
                                    "shard_id": "discourse-topic-274354",
                                }
                            },
                        }
                    ]
                },
                "collector_record_provenance": {
                    "records": [
                        {
                            "source_url": "https://forum.example/t/api-examples/274354/1",
                            "source_id": "topic:274354:post:1",
                            "adapter_name": "discourse_topic_adapter",
                            "tool_name": "discourse_topic_collect",
                            "policy_disposition": "allow",
                            "shard_id": "discourse-topic-274354",
                        }
                    ]
                },
                "collector_shard_manifests": {
                    "shards": [
                        {
                            "shard_id": "discourse-topic-274354",
                            "adapter_name": "discourse_topic_adapter",
                            "status": "completed",
                            "source": {
                                "url": "https://forum.example/t/api-examples/274354",
                                "id": "topic:274354",
                            },
                            "progress": {
                                "cursor": "102",
                                "items_collected": 3,
                                "items_emitted": 3,
                            },
                        }
                    ]
                }
            },
            events=[provenance_event, dataset_event, shard_event],
            metadata={},
        )

    monkeypatch.setattr(MultiAgentOrchestrator, "run", _fake_run)

    asyncio.run(runtime_service._run_agent_run(run_id=run_id))

    assert received_callback and received_callback[0] is not None
    assert live_snapshot_artifacts
    assert live_snapshot_artifacts[0]["content"]["shards"][0]["shard_id"] == (
        "discourse-topic-274354"
    )
    assert live_snapshot_artifacts[0]["content"]["shards"][0]["status"] == "running"
    assert live_dataset_record_artifacts
    assert live_dataset_record_artifacts[0]["record"]["target"]["answer"] == "First collected post"
    assert live_dataset_record_artifacts[0]["record"]["metadata"]["collector_provenance"][
        "source_id"
    ] == "topic:274354:post:1"

    persisted_run = asyncio.run(runtime_service.get_agent_run(run_id))
    dataset_artifacts = [
        artifact
        for artifact in persisted_run["artifacts"]
        if artifact["artifact_type"] == "dataset_record"
    ]
    assert len(dataset_artifacts) == 1

def test_runtime_service_persists_live_subagent_runtime_snapshot_before_run_completion(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    run_id = "live-subagent-runtime-run-1"
    live_runtime_artifacts: list[dict[str, Any]] = []
    received_callback: list[object | None] = []

    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id=run_id,
            protocol_id="teacher_student_distill",
            title="Live subagent runtime run",
            topic="subagent runtime live snapshot",
            summary={
                "task_input": "Summarize the deployment risks.",
                "objective": "Summarize the deployment risks.",
            },
        )
    )

    original_append_agent_run_artifact = runtime_service._store.append_agent_run_artifact

    async def _capturing_append_agent_run_artifact(
        run_id: str,
        *,
        artifact_id: str | None,
        artifact_type: str,
        title: str | None = None,
        uri: str | None = None,
        mime_type: str | None = None,
        size_bytes: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await original_append_agent_run_artifact(
            run_id,
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            title=title,
            uri=uri,
            mime_type=mime_type,
            size_bytes=size_bytes,
            metadata=metadata,
        )
        if (
            artifact_type == "subagent_runtime"
            and isinstance(artifact_id, str)
            and "subagent-runtime-live" in artifact_id
        ):
            live_runtime_artifacts.append(dict(metadata or {}))

    monkeypatch.setattr(
        runtime_service._store,
        "append_agent_run_artifact",
        _capturing_append_agent_run_artifact,
    )

    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        received_callback.append(request.runtime_event_callback or request.event_callback)
        runtime_event = MultiAgentRunEvent(
            run_id=request.run_id or run_id,
            seq=4,
            type="artifact",
            payload={
                "name": "subagent_runtime",
                "content": {
                    "invocation_count": 1,
                    "completed_invocation_count": 1,
                    "token_tracked_invocation_count": 1,
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "total_tokens": 18,
                    "generation_time_ms": 12.5,
                    "finish_reason_counts": {"stop": 1},
                    "approval_pending": [],
                    "risky_tool_events": [],
                    "invocations": [],
                },
            },
            timestamp="2026-06-24T00:00:00+00:00",
        )
        callback = request.runtime_event_callback or request.event_callback
        if callback is not None:
            await callback(runtime_event)
        assert live_runtime_artifacts
        return MultiAgentRunResult(
            run_id=run_id,
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={
                "subagent_runtime": {
                    "invocation_count": 3,
                    "completed_invocation_count": 3,
                    "token_tracked_invocation_count": 3,
                    "input_tokens": 24,
                    "output_tokens": 15,
                    "total_tokens": 39,
                    "generation_time_ms": 37.5,
                    "finish_reason_counts": {"stop": 3},
                    "approval_pending": [],
                    "risky_tool_events": [],
                    "invocations": [],
                }
            },
            events=[runtime_event],
            metadata={},
        )

    monkeypatch.setattr(MultiAgentOrchestrator, "run", _fake_run)

    asyncio.run(runtime_service._run_agent_run(run_id=run_id))

    assert received_callback and received_callback[0] is not None
    assert live_runtime_artifacts
    assert live_runtime_artifacts[0]["content"]["total_tokens"] == 18
    assert live_runtime_artifacts[0]["content"]["finish_reason_counts"] == {"stop": 1}

    persisted_run = asyncio.run(runtime_service.get_agent_run(run_id))
    subagent_runtime_artifacts = [
        artifact
        for artifact in persisted_run["artifacts"]
        if artifact["artifact_type"] == "subagent_runtime"
    ]
    assert len(subagent_runtime_artifacts) >= 2

def test_agent_run_background_exec_persists_detached_job_and_supports_run_scoped_session_lookup(
    tmp_path: Path,
) -> None:
    app = create_app()
    engine = _BackgroundControlledExecAgentRunEngine()
    exec_runtime = ExecRuntime(
        providers={"test": _ApiRuntimePythonDirectProvider()},
        default_shell="test",
    )
    runtime_service = RuntimeService(
        engine=engine,
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
        exec_runtime=exec_runtime,
    )
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "sessions_dir": str(tmp_path / "sessions"),
            "security": {
                "require_approval_for_exec": False,
                "command_rules": [_BACKGROUND_SMOKE_COMMAND_RULE],
            },
        }
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "controlled_subagent_execution",
                "title": "Controlled background execution run",
                "topic": "run a detached smoke command",
                "selected_models_roles": {
                    "by_role": {
                        "planner": "controlled-planner-model",
                        "executor": "controlled-executor-model",
                        "controller": "controlled-controller-model",
                        "evaluator": "controlled-evaluator-model",
                        "judge": "controlled-judge-model",
                        "verifier": "controlled-verifier-model",
                    }
                },
                "summary": {
                    "protocol_config": {
                        "max_execution_requests": 1,
                        "default_timeout_sec": 30,
                        "background_allowed": True,
                    },
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        artifact_types = [artifact["artifact_type"] for artifact in final_payload["artifacts"]]
        assert "detached_exec_jobs" in artifact_types
        runtime_artifact = next(
            artifact
            for artifact in final_payload["artifacts"]
            if artifact["artifact_type"] == "controlled_execution_runtime"
        )
        assert runtime_artifact["metadata"]["content"]["detached_exec_job_count"] == 1
        detached_artifact = next(
            artifact
            for artifact in final_payload["artifacts"]
            if artifact["artifact_type"] == "detached_exec_jobs"
        )
        detached_jobs = detached_artifact["metadata"]["content"]["items"]
        assert len(detached_jobs) == 1
        session_id = detached_jobs[0]["session_id"]
        assert detached_jobs[0]["reattach_supported"] is True
        log_path = Path(detached_jobs[0]["log_path"])
        checkpoint_dir = Path(detached_jobs[0]["checkpoint_dir"])
        assert log_path.name == "session.log"
        assert checkpoint_dir.name == "checkpoints"
        assert checkpoint_dir.is_dir()

        session_response = client.get(
            f"/v1/agent-runs/{run_id}/exec/{session_id}",
            params={"yield_time_ms": 50},
        )
        assert session_response.status_code == 200
        session_payload = session_response.json()
        assert session_payload["run_id"] == run_id
        assert session_payload["session_id"] == session_id
        assert session_payload["associated"] is True
        assert "bg-start" in session_payload["lease"]["command"]
        assert "sleep(5)" in session_payload["lease"]["command"]
        assert Path(session_payload["lease"]["log_path"]) == log_path
        assert Path(session_payload["lease"]["checkpoint_dir"]) == checkpoint_dir
        assert session_payload["live_status"] in {"available", "unavailable"}
        if session_payload["session"] is not None:
            assert session_payload["session"]["status"] in {"running", "completed"}
        log_text = ""
        for _ in range(10):
            if log_path.exists():
                log_text = log_path.read_text(encoding="utf-8")
                if "bg-start" in log_text:
                    break
            time.sleep(0.05)
        assert "bg-start" in log_text

def test_agent_run_background_exec_supports_run_scoped_stop(tmp_path: Path) -> None:
    app = create_app()
    engine = _BackgroundControlledExecAgentRunEngine()
    exec_runtime = ExecRuntime(
        providers={"test": _ApiRuntimePythonDirectProvider()},
        default_shell="test",
    )
    runtime_service = RuntimeService(
        engine=engine,
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
        exec_runtime=exec_runtime,
    )
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "sessions_dir": str(tmp_path / "sessions"),
            "security": {
                "require_approval_for_exec": False,
                "command_rules": [_BACKGROUND_SMOKE_COMMAND_RULE],
            },
        }
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "controlled_subagent_execution",
                "title": "Controlled background execution stop run",
                "topic": "stop a detached smoke command",
                "selected_models_roles": {
                    "by_role": {
                        "planner": "controlled-planner-model",
                        "executor": "controlled-executor-model",
                        "controller": "controlled-controller-model",
                        "evaluator": "controlled-evaluator-model",
                        "judge": "controlled-judge-model",
                        "verifier": "controlled-verifier-model",
                    }
                },
                "summary": {
                    "protocol_config": {
                        "max_execution_requests": 1,
                        "default_timeout_sec": 30,
                        "background_allowed": True,
                    },
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        detached_artifact = next(
            artifact
            for artifact in final_payload["artifacts"]
            if artifact["artifact_type"] == "detached_exec_jobs"
        )
        session_id = detached_artifact["metadata"]["content"]["items"][0]["session_id"]

        stop_response = client.post(f"/v1/agent-runs/{run_id}/exec/{session_id}/stop")
        assert stop_response.status_code == 200
        stop_payload = stop_response.json()
        assert stop_payload["run_id"] == run_id
        assert stop_payload["session_id"] == session_id
        assert stop_payload["associated"] is True
        assert stop_payload["stop_status"] in {"killed", "completed"}
        if stop_payload["session"] is not None:
            assert stop_payload["session"]["status"] in {"killed", "completed"}

        updated = client.get(f"/v1/agent-runs/{run_id}")
        assert updated.status_code == 200
        stop_events = [event for event in updated.json()["events"] if event["type"] == "detached_exec_stop"]
        assert stop_events
        assert stop_events[-1]["session_id"] == session_id
        assert stop_events[-1]["status"] in {"killed", "completed"}

def test_agent_run_background_exec_supports_reattach(tmp_path: Path) -> None:
    app = create_app()
    engine = _BackgroundControlledExecAgentRunEngine()
    exec_runtime = ExecRuntime(
        providers={"test": _ApiRuntimePythonDirectProvider()},
        default_shell="test",
    )
    runtime_service = RuntimeService(
        engine=engine,
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
        exec_runtime=exec_runtime,
    )
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "sessions_dir": str(tmp_path / "sessions"),
            "security": {
                "require_approval_for_exec": False,
                "command_rules": [_BACKGROUND_SMOKE_COMMAND_RULE],
            },
        }
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "controlled_subagent_execution",
                "title": "Controlled background execution reattach run",
                "topic": "reattach a detached smoke command",
                "selected_models_roles": {
                    "by_role": {
                        "planner": "controlled-planner-model",
                        "executor": "controlled-executor-model",
                        "controller": "controlled-controller-model",
                        "evaluator": "controlled-evaluator-model",
                        "judge": "controlled-judge-model",
                        "verifier": "controlled-verifier-model",
                    }
                },
                "summary": {
                    "protocol_config": {
                        "max_execution_requests": 1,
                        "default_timeout_sec": 30,
                        "background_allowed": True,
                    },
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        detached_artifact = next(
            artifact
            for artifact in final_payload["artifacts"]
            if artifact["artifact_type"] == "detached_exec_jobs"
        )
        session_id = detached_artifact["metadata"]["content"]["items"][0]["session_id"]

        reattach_response = client.post(
            f"/v1/agent-runs/{run_id}/reattach-exec/{session_id}",
            params={"yield_time_ms": 50},
        )
        assert reattach_response.status_code == 200
        reattach_payload = reattach_response.json()
        assert reattach_payload["run_id"] == run_id
        assert reattach_payload["session_id"] == session_id
        assert reattach_payload["associated"] is True
        assert reattach_payload["reattached"] is True
        assert reattach_payload["reattach_status"] in {"available", "unavailable"}

        updated = client.get(f"/v1/agent-runs/{run_id}")
        assert updated.status_code == 200
        reattach_events = [
            event for event in updated.json()["events"] if event["type"] == "detached_exec_reattached"
        ]
        assert reattach_events
        assert reattach_events[-1]["session_id"] == session_id

def test_agent_runs_api_flow_collects_evidence_queries_and_persists_artifacts(tmp_path: Path) -> None:
    app = create_app()
    engine = _AgentRunModelBackedEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Evidence-backed run",
                "topic": "deployment summary",
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                        "verifier": "verifier-model",
                        "judge": "judge-model",
                    }
                },
                "summary": {
                    "evidence_queries": ["approved deployment note"],
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        assert final_payload["summary"]["selected_candidate_id"] == "student"
        artifact_types = [artifact["artifact_type"] for artifact in final_payload["artifacts"]]
        assert "evidence_summary" in artifact_types
        assert "verification_summary" in artifact_types

    assert engine.evidence_calls
    assert engine.evidence_calls[0]["queries"] == ["approved deployment note"]

def test_agent_run_run_policy_pause_maps_to_awaiting_resources(tmp_path: Path) -> None:
    app = create_app()
    engine = _SlowAgentRunModelBackedEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Budget pause run",
                "topic": "deployment summary",
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                    }
                },
                "run_policy": {
                    "max_wall_clock_sec": 1,
                    "on_budget_exhausted": "pause",
                },
            },
        )
        assert create_response.status_code == 200
        created = create_response.json()
        run_id = created["run_id"]
        assert created["run_policy"]["max_wall_clock_sec"] == 1
        assert created["run_policy"]["on_budget_exhausted"] == "pause"

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(client, run_id, {"awaiting_resources"}, timeout_seconds=5.0)
        assert final_payload["status"] == "awaiting_resources"
        assert final_payload["finished_at"] is None
        assert final_payload["recovery_state"]["status"] == "awaiting_resources"
        assert final_payload["recovery_state"]["action"] == "pause"
        artifact_types = [artifact["artifact_type"] for artifact in final_payload["artifacts"]]
        assert "run_guard_policy" in artifact_types
        assert "recovery_checkpoint" in artifact_types
        assert "partial_summary" in artifact_types

def test_agent_run_heartbeat_timeout_maps_to_stalled(tmp_path: Path) -> None:
    app = create_app()
    engine = _SlowAgentRunModelBackedEngine(delay_seconds=1.2)
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Heartbeat stalled run",
                "topic": "deployment summary",
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                    }
                },
                "run_policy": {
                    "heartbeat_timeout_sec": 1,
                    "max_subagent_failures_per_role": 0,
                    "on_subagent_disconnect": "pause",
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(client, run_id, {"stalled"}, timeout_seconds=5.0)
        assert final_payload["status"] == "stalled"
        assert final_payload["finished_at"] is None
        assert final_payload["recovery_state"]["status"] == "stalled"
        assert final_payload["recovery_state"]["action"] == "pause"
        artifact_types = [artifact["artifact_type"] for artifact in final_payload["artifacts"]]
        assert "subagent_health_snapshot" in artifact_types

def test_agent_run_run_policy_finalize_partial_sets_partial_status(tmp_path: Path) -> None:
    app = create_app()
    engine = _SlowAgentRunModelBackedEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Partial finalize run",
                "topic": "deployment summary",
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                    }
                },
                "run_policy": {
                    "max_wall_clock_sec": 1,
                    "on_budget_exhausted": "finalize_partial",
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(client, run_id, {"partial"}, timeout_seconds=5.0)
        assert final_payload["status"] == "partial"
        assert final_payload["finished_at"] is not None
        assert final_payload["recovery_state"]["status"] == "partial"
        assert final_payload["recovery_state"]["action"] == "finalize_partial"
        artifact_types = [artifact["artifact_type"] for artifact in final_payload["artifacts"]]
        assert "partial_summary" in artifact_types

def test_agent_run_quota_exhaustion_maps_to_awaiting_resources(tmp_path: Path) -> None:
    app = create_app()
    engine = _QuotaExhaustedAgentRunEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Quota exhaustion run",
                "topic": "deployment summary",
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                    }
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(client, run_id, {"awaiting_resources"}, timeout_seconds=5.0)
        assert final_payload["status"] == "awaiting_resources"
        assert final_payload["recovery_state"]["status"] == "awaiting_resources"
        assert final_payload["recovery_state"]["resource_exhaustion_report"]["classification"] == "quota_exhausted"
        artifact_types = [artifact["artifact_type"] for artifact in final_payload["artifacts"]]
        assert "resource_exhaustion_report" in artifact_types

def test_agent_run_uses_security_default_run_policy_when_request_omits_it(tmp_path: Path) -> None:
    app = create_app()
    engine = _SlowAgentRunModelBackedEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "sessions_dir": str(tmp_path / "sessions"),
            "security": {
                "agent_run_default_max_wall_clock_sec": 1,
                "agent_run_default_heartbeat_timeout_sec": 45,
                "agent_run_default_checkpoint_interval_steps": 3,
                "agent_run_default_max_subagent_failures_per_role": 4,
                "agent_run_default_on_budget_exhausted": "pause",
                "agent_run_default_on_subagent_disconnect": "pause",
            },
        }
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Default run policy from settings",
                "topic": "verify settings-backed run guard",
            },
        )
        assert create_response.status_code == 200
        created = create_response.json()
        run_id = created["run_id"]
        assert created["run_policy"]["max_wall_clock_sec"] == 1
        assert created["run_policy"]["heartbeat_timeout_sec"] == 45
        assert created["run_policy"]["checkpoint_interval_steps"] == 3
        assert created["run_policy"]["max_subagent_failures_per_role"] == 4
        assert created["run_policy"]["on_budget_exhausted"] == "pause"
        assert created["run_policy"]["on_subagent_disconnect"] == "pause"

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(
            client,
            run_id,
            {"awaiting_resources", "partial", "succeeded"},
            timeout_seconds=5.0,
        )
        assert final_payload["run_policy"]["max_wall_clock_sec"] == 1
        assert final_payload["run_policy"]["on_budget_exhausted"] == "pause"

def test_agent_run_local_model_unavailable_maps_to_awaiting_resources(tmp_path: Path) -> None:
    app = create_app()
    engine = _LocalModelUnavailableAgentRunEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Local runtime unavailable run",
                "topic": "deployment summary",
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                    }
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(client, run_id, {"awaiting_resources"}, timeout_seconds=5.0)
        assert final_payload["status"] == "awaiting_resources"
        assert final_payload["recovery_state"]["resource_exhaustion_report"]["classification"] == "local_model_unavailable"
        assert final_payload["latest_error"] == "Configured model 'teacher-model' is not available."

def test_agent_runs_api_flow_persists_research_debate_artifacts(tmp_path: Path) -> None:
    app = create_app()
    engine = _AgentRunModelBackedEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "multi_agent_debate",
                "title": "Research debate run",
                "topic": "deployment review",
                "selected_models_roles": {
                    "by_role": {
                        "debater_a": "debater-a-model",
                        "debater_b": "debater-b-model",
                        "judge": "judge-model",
                        "verifier": "verifier-model",
                        "planner": "planner-model",
                        "local_worker": "local-worker-model",
                        "synthesizer": "synth-model",
                    }
                },
                "summary": {
                    "evidence_queries": ["approved deployment note"],
                    "protocol_config": {"rounds": 2},
                },
                "evaluation_policy": {
                    "research": {
                        "enabled": True,
                        "preset": "smart_judge_research_debate",
                        "output_targets": ["research_brief"],
                        "source_mode": "hybrid",
                        "citation_policy": "claim_level_required",
                        "local_worker_count": 2,
                        "local_worker_count_max": 6,
                        "max_research_queries": 4,
                        "max_sources_per_query": 3,
                        "debate_rounds": 2,
                    }
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        artifact_types = [artifact["artifact_type"] for artifact in final_payload["artifacts"]]
        assert "research_plan" in artifact_types
        assert "source_quality_table" in artifact_types
        assert "claim_evidence_map" in artifact_types
        assert "research_brief" in artifact_types
        assert "dataset_record" not in artifact_types

def test_agent_runs_api_flow_keeps_multi_agent_debate_protocol_with_controlled_execution_capability(
    tmp_path: Path,
) -> None:
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
            "/v1/agent-runs",
            json={
                "protocol_id": "multi_agent_debate",
                "title": "Debate with controlled execution capability",
                "topic": "deployment review",
                "selected_models_roles": {
                    "by_role": {
                        "debater_a": "debater-a-model",
                        "debater_b": "debater-b-model",
                        "judge": "judge-model",
                        "verifier": "verifier-model",
                        "planner": "controlled-planner-model",
                        "executor": "controlled-executor-model",
                        "controller": "controlled-controller-model",
                        "evaluator": "controlled-evaluator-model",
                    }
                },
                "summary": {
                    "protocol_config": {"rounds": 2},
                    "execution_policy": {
                        "mode": "controlled",
                        "max_execution_requests": 1,
                        "max_commands_per_request": 1,
                        "default_timeout_sec": 30,
                        "background_allowed": False,
                    },
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        assert final_payload["protocol_id"] == "multi_agent_debate"
        artifact_types = [artifact["artifact_type"] for artifact in final_payload["artifacts"]]
        assert "controlled_execution_runtime" in artifact_types
        assert "execution_requests" in artifact_types
        runtime_artifact = next(
            artifact
            for artifact in final_payload["artifacts"]
            if artifact["artifact_type"] == "controlled_execution_runtime"
        )
        runtime_payload = runtime_artifact["metadata"]["content"]
        assert runtime_payload["primary_workflow"] is False
        assert (
            runtime_payload["execution_boundary"]
            == "subagents_propose_controller_approves_shared_runtime_executes"
        )

def test_agent_run_schedule_starts_due_run_automatically(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        RuntimeService,
        "_DEFAULT_SCHEDULER_POLL_INTERVAL_SECONDS",
        0.05,
    )
    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Scheduled run",
                "topic": "auto execute",
                "schedule": {
                    "enabled": True,
                    "interval_seconds": 3600,
                    "start_immediately": True,
                },
            },
        )
        assert create_response.status_code == 200
        created = create_response.json()
        run_id = created["run_id"]
        assert created["status"] == "created"

        final_payload = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        assert final_payload["summary"]["final_answer"]
        assert final_payload["schedule"]["last_scheduled_at"] is not None
        assert final_payload["schedule"]["last_completion_status"] == "succeeded"
        assert final_payload["schedule"]["next_run_at"] is not None
        assert final_payload["schedule"]["attempt_count"] == 1
        assert isinstance(final_payload["schedule"].get("recent_attempts"), list)
        latest_attempt = final_payload["schedule"]["recent_attempts"][0]
        assert latest_attempt["attempt_id"]
        assert latest_attempt["status"] == "succeeded"
        assert latest_attempt["selected_candidate_id"] == final_payload["summary"].get(
            "selected_candidate_id"
        )
        assert latest_attempt["completed_at"] == final_payload["schedule"]["last_completed_at"]
        assert latest_attempt["final_answer_preview"]
        artifact_attempt_ids = {
            artifact.get("metadata", {}).get("attempt_id")
            for artifact in final_payload["artifacts"]
            if isinstance(artifact.get("metadata"), dict)
        }
        assert latest_attempt["attempt_id"] in artifact_attempt_ids
        scheduled_events = [
            event
            for event in final_payload["events"]
            if event.get("type") in {"run_scheduled", "run_started"}
        ]
        assert scheduled_events
        assert all(event.get("attempt_id") == latest_attempt["attempt_id"] for event in scheduled_events)
        event_types = [event["type"] for event in final_payload["events"]]
        assert "run_scheduled" in event_types
        assert "run_started" in event_types

def test_agent_run_schedule_disabled_does_not_start(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        RuntimeService,
        "_DEFAULT_SCHEDULER_POLL_INTERVAL_SECONDS",
        0.05,
    )
    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    due_at = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Disabled scheduled run",
                "topic": "stay idle",
                "schedule": {
                    "enabled": False,
                    "run_at": due_at,
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        time.sleep(0.3)
        get_response = client.get(f"/v1/agent-runs/{run_id}")
        assert get_response.status_code == 200
        payload = get_response.json()
        assert payload["status"] == "created"
        assert payload["schedule"].get("last_scheduled_at") is None
        assert "run_scheduled" not in [event["type"] for event in payload["events"]]

def test_agent_run_schedule_respects_max_runs(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        RuntimeService,
        "_DEFAULT_SCHEDULER_POLL_INTERVAL_SECONDS",
        0.05,
    )
    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Max runs policy",
                "topic": "stop after one run",
                "schedule": {
                    "enabled": True,
                    "interval_seconds": 1,
                    "start_immediately": True,
                    "max_runs": 1,
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        final_payload = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        assert final_payload["schedule"]["attempt_count"] == 1
        assert final_payload["schedule"]["enabled"] is False
        assert final_payload["schedule"]["next_run_at"] is None
        assert final_payload["schedule"]["schedule_status"] == "max_runs_reached"
        assert final_payload["schedule"]["health_status"] == "completed"

        time.sleep(0.3)
        follow_up = client.get(f"/v1/agent-runs/{run_id}")
        assert follow_up.status_code == 200
        follow_up_payload = follow_up.json()
        assert follow_up_payload["schedule"]["attempt_count"] == 1

def test_agent_run_schedule_auto_pauses_after_failure(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        RuntimeService,
        "_DEFAULT_SCHEDULER_POLL_INTERVAL_SECONDS",
        0.05,
    )
    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    async def _failing_run(self: Any, request: Any) -> MultiAgentRunResult:
        raise RuntimeError("synthetic scheduled failure")

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _failing_run)

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Auto pause on failure",
                "topic": "synthetic failure",
                "schedule": {
                    "enabled": True,
                    "interval_seconds": 60,
                    "start_immediately": True,
                    "auto_pause_on_failure": True,
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        final_payload = _wait_agent_run_until(client, run_id, {"failed"}, timeout_seconds=4.0)
        assert final_payload["schedule"]["attempt_count"] == 1
        assert final_payload["schedule"]["failure_streak"] == 1
        assert final_payload["schedule"]["enabled"] is False
        assert final_payload["schedule"]["next_run_at"] is None
        assert final_payload["schedule"]["schedule_status"] == "paused_after_failure"
        assert final_payload["schedule"]["health_status"] == "paused_after_failure"
        assert final_payload["latest_error"] == "synthetic scheduled failure"

def test_agent_run_operator_messages_are_collected_as_guidance_for_runtime(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured_request: dict[str, Any] = {}

    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        captured_request["guidance_messages"] = list(request.guidance_messages)
        captured_request["metadata"] = dict(request.metadata)
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Guided result."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Workflow-guided run",
                "topic": "deployment summary",
                "project_id": "project-guidance",
                "workspace_dir": str(tmp_path / "workspace-guidance"),
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        message_response = client.post(
            f"/v1/agent-runs/{run_id}/messages",
            json={
                "role": "operator",
                "content": "Prioritize a concise answer with verified claims only.",
            },
        )
        assert message_response.status_code == 200

        guidance_response = client.post(
            f"/v1/agent-runs/{run_id}/guidance",
            json={"guidance": "Mention any uncertainty explicitly."},
        )
        assert guidance_response.status_code == 200

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        assert final_payload["summary"]["final_answer"] == "Guided result."

    assert captured_request["guidance_messages"] == [
        "Prioritize a concise answer with verified claims only.",
        "Mention any uncertainty explicitly.",
    ]
    assert captured_request["metadata"]["project_id"] == "project-guidance"
    assert captured_request["metadata"]["workspace_dir"] == str(tmp_path / "workspace-guidance")

def test_agent_run_subagent_messages_do_not_change_global_guidance_collection(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured_request: dict[str, Any] = {}

    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        captured_request["guidance_messages"] = list(request.guidance_messages)
        captured_request["role_guidance_messages"] = dict(request.role_guidance_messages)
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Subagent guidance stays additive."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Subagent-guided run",
                "topic": "deployment summary",
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        subagent_message_response = client.post(
            f"/v1/agent-runs/{run_id}/subagents/verifier/messages",
            json={
                "content": "Please inspect only claim 2 evidence.",
                "metadata": {"channel": "subagent-chat"},
            },
        )
        assert subagent_message_response.status_code == 200
        assert [
            event
            for event in subagent_message_response.json()["events"]
            if event["type"] == "subagent_message"
        ][-1]["target_role_id"] == "verifier"

        guidance_response = client.post(
            f"/v1/agent-runs/{run_id}/guidance",
            json={"guidance": "Keep the final answer concise."},
        )
        assert guidance_response.status_code == 200

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        assert final_payload["summary"]["final_answer"] == "Subagent guidance stays additive."

    assert captured_request["guidance_messages"] == [
        "Keep the final answer concise.",
    ]
    assert captured_request["role_guidance_messages"] == {
        "verifier": ["Please inspect only claim 2 evidence."],
    }

def test_agent_run_operator_message_preserves_structured_attachments(
    tmp_path: Path,
) -> None:
    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Attachment replay run",
                "topic": "structured attachments",
                "project_id": "project-attachments",
                "workspace_dir": str(tmp_path / "workspace-attachments"),
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        attachment = {
            "name": "app.py",
            "path": str(tmp_path / "workspace-attachments" / "app.py"),
            "source": "workspace_selection",
            "line_start": 10,
            "line_end": 14,
            "quote": "print('hello')",
            "note": "Investigate this block.",
        }
        message_response = client.post(
            f"/v1/agent-runs/{run_id}/messages",
            json={
                "role": "operator",
                "content": "Please inspect this selection.",
                "attachments": [attachment],
            },
        )
        assert message_response.status_code == 200
        payload = message_response.json()

    operator_events = [event for event in payload["events"] if event["type"] == "operator_message"]
    assert operator_events
    expected_attachment = {
        **attachment,
        "size": None,
        "content_type": None,
    }
    assert operator_events[-1]["attachments"] == [expected_attachment]
    assert operator_events[-1]["metadata"]["attachments"] == [expected_attachment]

def test_agent_run_operator_message_rejects_invalid_attachment_ranges(
    tmp_path: Path,
) -> None:
    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Invalid attachment run",
                "topic": "structured attachments",
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        response = client.post(
            f"/v1/agent-runs/{run_id}/messages",
            json={
                "role": "operator",
                "content": "Please inspect this selection.",
                "attachments": [
                    {
                        "name": "app.py",
                        "path": str(tmp_path / "app.py"),
                        "source": "workspace_selection",
                        "line_start": 12,
                        "line_end": 4,
                    }
                ],
            },
        )

    assert response.status_code == 422

def test_agent_run_operator_message_allows_attachment_only_payloads(
    tmp_path: Path,
) -> None:
    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Attachment only run",
                "topic": "attachment only workflow message",
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        response = client.post(
            f"/v1/agent-runs/{run_id}/messages",
            json={
                "role": "operator",
                "content": "",
                "attachments": [
                    {
                        "name": "notes.md",
                        "path": str(tmp_path / "notes.md"),
                        "source": "workspace_file",
                        "note": "Use this document as context.",
                    }
                ],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    operator_event = [event for event in payload["events"] if event["type"] == "operator_message"][-1]
    assistant_event = [event for event in payload["events"] if event["type"] == "assistant_message"][-1]
    assert operator_event["content"] == ""
    assert operator_event["attachments"] == [
        {
            "name": "notes.md",
            "path": str(tmp_path / "notes.md"),
            "size": None,
            "content_type": None,
            "source": "workspace_file",
            "line_start": None,
            "line_end": None,
            "quote": None,
            "note": "Use this document as context.",
        }
    ]
    assert "message received" in assistant_event["content"]

def test_agent_run_schedule_failure_streak_persists_when_not_auto_paused(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        RuntimeService,
        "_DEFAULT_SCHEDULER_POLL_INTERVAL_SECONDS",
        0.05,
    )
    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    async def _failing_run(self: Any, request: Any) -> MultiAgentRunResult:
        raise RuntimeError("non-pausing failure")

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _failing_run)

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Failure streak",
                "topic": "keep schedule active",
                "schedule": {
                    "enabled": True,
                    "interval_seconds": 60,
                    "start_immediately": True,
                    "auto_pause_on_failure": False,
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        final_payload = _wait_agent_run_until(client, run_id, {"failed"}, timeout_seconds=4.0)
        assert final_payload["schedule"]["attempt_count"] == 1
        assert final_payload["schedule"]["failure_streak"] == 1
        assert final_payload["schedule"]["enabled"] is True
        assert final_payload["schedule"]["next_run_at"] is not None
        assert final_payload["schedule"]["schedule_status"] == "scheduled"
        assert final_payload["schedule"]["health_status"] == "failing"
