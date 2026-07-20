"""Goal API integration tests: Resume Followups."""

from ._support import *  # noqa: F401,F403

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

def test_goal_resume_uses_summary_guidance_and_role_guidance_messages(
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
            artifacts={"final_answer": "Summary-backed guidance resume completed."},
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

    goal_id = "goal-summary-guidance-resume-1"
    attempt_id = "goal-summary-guidance-resume-attempt-1"
    run_id = "linked-summary-guidance-run-1"

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id=goal_id,
            objective="Resume the linked run using persisted summary guidance.",
            protocol_id="teacher_student_distill",
            summary={"phase": "manual_resume"},
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
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id=run_id,
            protocol_id="teacher_student_distill",
            title="Summary guidance linked run",
            topic="summary guidance",
            summary={
                "goal_id": goal_id,
                "goal_attempt_id": attempt_id,
                "objective": "Resume the linked run using persisted summary guidance.",
                "task_input": "Resume the linked run using persisted summary guidance.",
                "guidance_messages": [
                    "Resume from the stored summary guidance.",
                    "Preserve the existing teacher output.",
                ],
                "role_guidance_messages": {
                    "teacher": ["Teacher: keep the original evidence structure."],
                    "student": ["Student: condense the answer for final delivery."],
                },
                "recovery_state": {
                    "status": "stalled",
                    "action": "resume",
                    "reason": "Operator requested resume.",
                    "stage": "teacher_generation",
                    "checkpoint": {
                        "checkpoint_index": 2,
                        "stage": "teacher_generation",
                    },
                    "resume_payload": {
                        "version": 1,
                        "executor": "continue_from_checkpoint",
                        "strategy_default": "continue_from_checkpoint",
                        "stage": "teacher_generation",
                        "checkpoint": {
                            "checkpoint_index": 2,
                            "stage": "teacher_generation",
                        },
                        "guidance_messages": [],
                        "role_guidance_messages": {},
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
            started_at=(datetime.now(UTC) - timedelta(minutes=2)).isoformat(),
        )
    )

    with TestClient(app) as client:
        resume_response = client.post(
            f"/v1/goals/{goal_id}/resume",
            json={"strategy": "continue_from_checkpoint"},
        )
        assert resume_response.status_code == 200
        completed_goal = _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)

    assert len(completed_goal["attempts"]) == 1
    assert len(captured_requests) == 1
    assert captured_requests[0].metadata["resume_strategy"] == "restart_attempt"
    resume_payload = captured_requests[0].metadata["resume_payload"]
    assert resume_payload["role_guidance_messages"] == {
        "teacher": ["Teacher: keep the original evidence structure."],
        "student": ["Student: condense the answer for final delivery."],
    }
    assert "Resume from the stored summary guidance." in resume_payload["guidance_messages"]
    assert "Preserve the existing teacher output." in resume_payload["guidance_messages"]
    carryover_messages = [
        item
        for item in resume_payload["guidance_messages"]
        if item.startswith("Goal carryover state: ")
    ]
    assert len(carryover_messages) == 1
    carryover = json.loads(carryover_messages[0].removeprefix("Goal carryover state: "))
    assert carryover["goal_id"] == goal_id
    assert carryover["attempt_id"] == attempt_id
    assert carryover["agent_run_id"] == run_id
    assert carryover["operator_guidance"] == [
        "Resume from the stored summary guidance.",
        "Preserve the existing teacher output.",
    ]
    assert captured_requests[0].guidance_messages == resume_payload["guidance_messages"]
    assert captured_requests[0].role_guidance_messages == {
        "teacher": ["Teacher: keep the original evidence structure."],
        "student": ["Student: condense the answer for final delivery."],
    }

def test_goal_resume_carries_follow_up_guidance_inside_resume_payload(
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
            artifacts={"final_answer": "Follow-up guidance reached the resumed run."},
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

    goal_id = "goal-follow-up-resume-payload-1"
    attempt_id = "goal-follow-up-resume-payload-attempt-1"
    run_id = "linked-follow-up-resume-payload-run-1"
    follow_up = "Continue in this direction and keep the comparison grounded in the earlier evidence."

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id=goal_id,
            objective="Resume the linked run with follow-up guidance embedded in the resume payload.",
            protocol_id="teacher_student_distill",
            summary={"phase": "manual_resume"},
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
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id=run_id,
            protocol_id="teacher_student_distill",
            title="Goal follow-up resume payload run",
            topic="goal follow-up resume payload",
            summary={
                "goal_id": goal_id,
                "goal_attempt_id": attempt_id,
                "objective": "Resume the linked run with follow-up guidance embedded in the resume payload.",
                "task_input": "Resume the linked run with follow-up guidance embedded in the resume payload.",
                "recovery_state": {
                    "status": "stalled",
                    "action": "resume",
                    "reason": "Operator requested follow-up.",
                    "stage": "research_context_prepared",
                    "checkpoint": {
                        "checkpoint_index": 1,
                        "stage": "research_context_prepared",
                    },
                    "resume_payload": {
                        "version": 1,
                        "executor": "continue_from_checkpoint",
                        "strategy_default": "continue_from_checkpoint",
                        "supported_actions": [
                            "restart_attempt",
                            "continue_from_checkpoint",
                        ],
                        "stage": "research_context_prepared",
                        "checkpoint": {
                            "checkpoint_index": 1,
                            "stage": "research_context_prepared",
                        },
                        "guidance_messages": [],
                        "role_guidance_messages": {},
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

    with TestClient(app) as client:
        message_response = client.post(
            f"/v1/agent-runs/{run_id}/messages",
            json={
                "role": "user",
                "content": follow_up,
                "metadata": {"channel": "goal-chat"},
            },
        )
        assert message_response.status_code == 200
        message_payload = message_response.json()
        assert message_payload["summary"]["guidance_messages"] == [follow_up]
        assert (
            message_payload["summary"]["recovery_state"]["resume_payload"]["guidance_messages"]
            == [follow_up]
        )

        resume_response = client.post(
            f"/v1/goals/{goal_id}/resume",
            json={"strategy": "continue_from_checkpoint"},
        )
        assert resume_response.status_code == 200
        completed_goal = _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)

    assert completed_goal["status"] == "completed"
    assert len(captured_requests) == 1
    assert follow_up in captured_requests[0].guidance_messages
    carryover_messages = [
        item
        for item in captured_requests[0].guidance_messages
        if item.startswith("Goal carryover state: ")
    ]
    assert len(carryover_messages) == 1
    carryover = json.loads(carryover_messages[0].removeprefix("Goal carryover state: "))
    assert carryover["goal_id"] == goal_id
    assert carryover["attempt_id"] == attempt_id
    assert carryover["agent_run_id"] == run_id
    assert carryover["previous_status"] == "stalled"
    assert follow_up in carryover["operator_guidance"]
    assert follow_up in captured_requests[0].metadata["resume_payload"]["guidance_messages"]
    assert carryover_messages[0] in captured_requests[0].metadata["resume_payload"]["guidance_messages"]

def test_goal_resume_new_attempt_seeds_follow_up_guidance_into_linked_run(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured_requests: list[Any] = []
    follow_up = "Resume with the updated operator direction after reopening the goal."

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
            artifacts={"final_answer": "Fresh goal attempt consumed queued follow-up guidance."},
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

    goal_id = "goal-follow-up-new-attempt-1"
    old_attempt_id = "goal-follow-up-old-attempt-1"
    old_run_id = "goal-follow-up-old-run-1"

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id=goal_id,
            objective="Resume the goal by reopening a fresh linked run when the previous attempt is no longer resumable.",
            protocol_id="teacher_student_distill",
            summary={"phase": "stalled"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id=old_attempt_id,
            goal_id=goal_id,
            attempt_index=1,
            status="stalled",
            trigger="manual_start",
            agent_run_id=old_run_id,
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            goal_id,
            "stalled",
            current_attempt_id=old_attempt_id,
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id=old_run_id,
            protocol_id="teacher_student_distill",
            title="Unresumable linked run",
            topic="goal restart",
            summary={
                "goal_id": goal_id,
                "goal_attempt_id": old_attempt_id,
                "objective": "Resume the goal by reopening a fresh linked run when the previous attempt is no longer resumable.",
                "task_input": "Resume the goal by reopening a fresh linked run when the previous attempt is no longer resumable.",
            },
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status(old_run_id, "completed"))

    with TestClient(app) as client:
        resume_response = client.post(
            f"/v1/goals/{goal_id}/resume",
            json={
                "strategy": "continue_from_checkpoint",
                "guidance_message": follow_up,
            },
        )
        assert resume_response.status_code == 200
        completed_goal = _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)

    assert completed_goal["status"] == "completed"
    assert len(completed_goal["attempts"]) == 2
    assert completed_goal["current_attempt_id"] != old_attempt_id
    assert len(captured_requests) == 1
    assert follow_up in captured_requests[0].guidance_messages
    carryover_messages = [
        item
        for item in captured_requests[0].guidance_messages
        if item.startswith("Goal carryover state: ")
    ]
    assert len(carryover_messages) == 1
    carryover = json.loads(carryover_messages[0].removeprefix("Goal carryover state: "))
    assert carryover["goal_id"] == goal_id
    assert carryover["previous_status"] == "stalled"
    assert carryover["current_status"] == "queued"
    assert carryover["agent_run_id"] == old_run_id
    assert follow_up in carryover["operator_guidance"]
    assert captured_requests[0].metadata["summary"]["guidance_messages"] == [
        follow_up,
        carryover_messages[0],
    ]

def test_goal_resume_carryover_marks_truncated_final_answer(
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
            artifacts={"final_answer": "Recovered from a truncated prior answer."},
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

    goal_id = "goal-truncated-carryover-1"
    attempt_id = "goal-truncated-carryover-attempt-1"
    run_id = "goal-truncated-carryover-run-1"
    partial_answer = (
        "This final answer was cut off mid implementation patch.\n\n"
        "```python\n"
        "def save_recent_code() -> str:\n"
        "    return 'persist this unfinished implementation'\n"
        "```\n"
    )

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id=goal_id,
            objective="Resume a goal whose previous final answer was truncated.",
            protocol_id="teacher_student_distill",
            summary={"phase": "manual_resume"},
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
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id=run_id,
            protocol_id="teacher_student_distill",
            title="Truncated carryover run",
            topic="truncated carryover",
            summary={
                "goal_id": goal_id,
                "goal_attempt_id": attempt_id,
                "objective": "Resume a goal whose previous final answer was truncated.",
                "task_input": "Resume a goal whose previous final answer was truncated.",
                "recovery_state": {
                    "status": "stalled",
                    "action": "resume",
                    "reason": "Previous final answer was truncated.",
                    "stage": "research_context_prepared",
                    "checkpoint": {
                        "checkpoint_index": 1,
                        "stage": "research_context_prepared",
                    },
                    "resume_payload": {
                        "version": 1,
                        "executor": "continue_from_checkpoint",
                        "strategy_default": "continue_from_checkpoint",
                        "stage": "research_context_prepared",
                        "checkpoint": {
                            "checkpoint_index": 1,
                            "stage": "research_context_prepared",
                        },
                        "guidance_messages": [],
                        "role_guidance_messages": {},
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
    asyncio.run(
        runtime_service._store.append_agent_run_event(
            run_id,
            {
                "type": "final_answer",
                "payload": {
                    "type": "final_answer",
                    "content": partial_answer,
                    "finish_reason": "length",
                    "metadata": {"truncated": True, "recovery_attempts": 1},
                },
            },
        )
    )
    asyncio.run(
        runtime_service._store.append_agent_run_artifact(
            run_id,
            artifact_id=f"{run_id}:attempt:{attempt_id}:produced_artifacts",
            artifact_type="produced_artifacts",
            title="Produced Code Artifacts",
            uri=f"agent-run://{run_id}/artifacts/{attempt_id}/produced_artifacts",
            mime_type="application/json",
            metadata={
                "attempt_id": attempt_id,
                "content": {
                    "items": [
                        {
                            "path": "mochi/runtime/service.py",
                            "status": "partial",
                        }
                    ]
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

    assert completed_goal["status"] == "completed"
    assert len(captured_requests) == 1
    carryover_messages = [
        item
        for item in captured_requests[0].guidance_messages
        if item.startswith("Goal carryover state: ")
    ]
    assert len(carryover_messages) == 1
    carryover = json.loads(carryover_messages[0].removeprefix("Goal carryover state: "))
    assert carryover["finish_reason"] == "length"
    assert carryover["truncated"] is True
    assert carryover["recovery_attempts"] == 1
    assert carryover["pending_next_action"] == "continue_from_prior_truncated_final_answer"
    assert carryover["final_answer_preview"] == partial_answer.rstrip()
    assert carryover["recent_code_blocks"] == [
        {
            "language": "python",
            "preview": (
                "def save_recent_code() -> str:\n"
                "    return 'persist this unfinished implementation'"
            ),
        }
    ]
    assert carryover["recent_artifact_refs"] == [
        {
            "artifact_type": "produced_artifacts",
            "title": "Produced Code Artifacts",
            "uri": f"agent-run://{run_id}/artifacts/{attempt_id}/produced_artifacts",
            "mime_type": "application/json",
        }
    ]
    assert carryover_messages[0] in captured_requests[0].metadata["resume_payload"]["guidance_messages"]

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
