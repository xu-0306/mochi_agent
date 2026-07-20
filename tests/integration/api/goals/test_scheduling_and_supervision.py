"""Goal API integration tests: Scheduling And Supervision."""

from ._support import *  # noqa: F401,F403


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
