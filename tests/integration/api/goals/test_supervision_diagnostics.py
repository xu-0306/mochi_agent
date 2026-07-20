"""Goal API integration tests: Supervision Diagnostics."""

from ._support import *  # noqa: F401,F403


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
        "blocker_type": "approval",
        "approval_count": 0,
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
