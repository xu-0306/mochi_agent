"""Goal API integration tests: Checkpoints Memory And Collectors."""

from ._support import *  # noqa: F401,F403

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
