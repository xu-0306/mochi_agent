"""Goal API integration tests: Exec Approvals."""

from ._support import *  # noqa: F401,F403

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
                    "approval_mode": "auto_review",
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
        assert health_payload["approval_diagnostic"] == {
            "cause_code": "approval_required",
            "what_is_blocked": "Active goal execution is waiting on operator approval.",
            "why_blocked": (
                "The linked run still has a pending approval-required tool request in runtime state, "
                "so execution remains blocked until that request is resolved."
            ),
            "actor_required": "operator",
            "next_action": "resolve_approval",
            "auto_resume_policy": "resume_after_approval_if_run_can_continue",
            "source_event_ids": [
                f"agent_run:{health_payload['linked_agent_run']['run_id']}",
                "approval:exec-approval-1",
            ],
            "approval_ids": ["exec-approval-1"],
            "tool_names": ["exec_command"],
            "run_id": health_payload["linked_agent_run"]["run_id"],
        }
        assert health_payload["blocker_diagnostic"] == {
            "cause_code": "approval_required",
            "what_is_blocked": "Active goal execution is waiting on operator approval.",
            "why_blocked": (
                "The linked run still has a pending approval-required tool request in runtime state, "
                "so execution remains blocked until that request is resolved."
            ),
            "actor_required": "operator",
            "next_action": "resolve_approval",
            "auto_resume_policy": "resume_after_approval_if_run_can_continue",
            "source_event_ids": [
                f"agent_run:{health_payload['linked_agent_run']['run_id']}",
                "approval:exec-approval-1",
            ],
        }
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

def test_goal_subagent_transcript_preserves_pending_approval_metadata(tmp_path: Path) -> None:
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )

    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id="goal-approval-run-1",
            protocol_id="controlled_subagent_execution",
            title="Goal approval transcript",
            topic="approval transcript",
            summary={"goal_id": "goal-approval-1"},
        )
    )
    asyncio.run(
        runtime_service._store.upsert_subagent_transcript(
            subagent_id="goal-approval-run-1:controller:approval",
            parent_type="agent_run",
            parent_id="goal-approval-run-1",
            goal_id="goal-approval-1",
            agent_run_id="goal-approval-run-1",
            role_id="controller",
            title="Controller",
            model_id="gpt-5.4",
            status="blocked",
            summary="Waiting for approval.",
        )
    )
    asyncio.run(
        runtime_service._store.append_subagent_transcript_event(
            "goal-approval-run-1:controller:approval",
            {
                "type": "runtime_blocked",
                "parent_type": "agent_run",
                "parent_id": "goal-approval-run-1",
                "subagent_id": "goal-approval-run-1:controller:approval",
                "role_id": "controller",
                "status": "blocked",
                "blocker_type": "approval",
                "summary": "Controller is waiting for approval.",
                "approval_ids": ["exec-approval-goal-1"],
                "tool_names": ["exec_command"],
                "recommended_action": "resolve_approval",
                "pending_approvals": [
                    {
                        "approval_id": "exec-approval-goal-1",
                        "tool_name": "exec_command",
                        "reason": "Exec command requires approval.",
                        "approval_kind": "exec",
                        "approval_scope": "workspace",
                        "replay_safe": False,
                        "security_decision": "require_approval",
                        "policy_source": "runtime_policy",
                        "allowed_decisions": ["approve_once", "reject"],
                    }
                ],
            },
        )
    )

    detail = asyncio.run(
        runtime_service.get_agent_run_subagent(
            "goal-approval-run-1",
            "goal-approval-run-1:controller:approval",
        )
    )

    assert detail is not None
    assert detail["subagent_id"] == "goal-approval-run-1:controller:approval"
    assert detail["events"][0]["type"] == "runtime_blocked"
    assert detail["events"][0]["approval_ids"] == ["exec-approval-goal-1"]
    assert detail["events"][0]["pending_approvals"][0] == {
        "approval_id": "exec-approval-goal-1",
        "tool_name": "exec_command",
        "reason": "Exec command requires approval.",
        "approval_kind": "exec",
        "approval_scope": "workspace",
        "replay_safe": False,
        "security_decision": "require_approval",
        "policy_source": "runtime_policy",
        "allowed_decisions": ["approve_once", "reject"],
    }

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
    assert resolved.status == "consumed"
    assert resolved.execution_result is not None

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
        assert resolve_response.json()["status"] in {"approved_once", "consumed"}

        completed_goal = _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)
        assert len(completed_goal["attempts"]) == 1
        assert completed_goal["attempts"][0]["agent_run_id"] == linked_run_id

    resolved = exec_approval_store.get(approval_id)
    assert resolved is not None
    assert resolved.status == "consumed"
    assert resolved.execution_result is not None

def test_goal_resume_with_approval_id_preserves_guidance_message(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    approval_id = "exec-approval-goal-resume-guidance-1"
    captured_requests: list[Any] = []
    follow_up = "After approving this command, continue with the operator's updated constraint."

    async def _approval_then_success_run(self: Any, request: Any) -> MultiAgentRunResult:
        captured_requests.append(request)
        approval = self._exec_approval_store.get(approval_id)
        if approval is None or approval.status == "pending":
            if approval is None:
                self._exec_approval_store.create(
                    approval_id=approval_id,
                    command="print('goal approval resume guidance')",
                    shell="test",
                    scope="dangerous_command",
                    reason="Exec command requires approval.",
                    command_payload={
                        "command": "print('goal approval resume guidance')",
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
                        "resume_payload": {
                            "version": 1,
                            "executor": "continue_from_checkpoint",
                            "strategy_default": "continue_from_checkpoint",
                            "supported_actions": [
                                "restart_attempt",
                                "continue_from_checkpoint",
                            ],
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
                            "role_task_snapshot": {},
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
            artifacts={"final_answer": "Goal approval guidance resume completed."},
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
                "objective": "Resolve a linked exec approval and carry follow-up guidance.",
                "title": "Goal approval guidance resume",
                "protocol_id": "controlled_subagent_execution",
            },
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200

        waiting_goal = _wait_goal_until(client, goal_id, {"waiting_approval"}, timeout_seconds=4.0)
        linked_run_id = waiting_goal["attempts"][0]["agent_run_id"]
        assert linked_run_id is not None

        resume_response = client.post(
            f"/v1/goals/{goal_id}/resume",
            json={
                "approval_id": approval_id,
                "decision": "approve_once",
                "reason": "allow linked goal resume with guidance",
                "guidance_message": follow_up,
            },
        )
        assert resume_response.status_code == 200

        completed_goal = _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)
        linked_run_response = client.get(f"/v1/agent-runs/{linked_run_id}")
        assert linked_run_response.status_code == 200

    assert completed_goal["attempts"][0]["agent_run_id"] == linked_run_id
    assert len(captured_requests) >= 2
    assert captured_requests[-1].guidance_messages == [follow_up]
    assert follow_up in captured_requests[-1].metadata["resume_payload"]["guidance_messages"]
    linked_run = linked_run_response.json()
    assert linked_run["summary"]["guidance_messages"] == [follow_up]
    assert follow_up in linked_run["summary"]["recovery_state"]["resume_payload"]["guidance_messages"]

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
        assert resolve_response.json()["status"] in {"approved_once", "consumed"}

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
    assert resolved.status == "consumed"
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
        assert resolve_response.json()["status"] in {"approved_once", "consumed"}

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
    assert resolved.status == "consumed"
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
        "blocker_type": "approval",
        "approval_count": 1,
        "run_id": "linked-approval-timeout-run-1",
        "approval_ids": ["exec-approval-timeout-1"],
        "tool_names": ["exec_command"],
        "finding_code": "approval_wait_timeout",
        "approval_wait_elapsed_sec": health_payload["approval_state"]["approval_wait_elapsed_sec"],
        "approval_wait_timeout_sec": 60,
    }
    assert health_payload["approval_diagnostic"] == {
        "cause_code": "approval_wait_timeout",
        "what_is_blocked": "Active goal execution is waiting on operator approval.",
        "why_blocked": "A pending exec_command approval has exceeded the configured approval wait timeout for the linked run.",
        "actor_required": "operator",
        "next_action": "resolve_approval",
        "auto_resume_policy": "resume_after_approval_if_run_can_continue",
        "source_event_ids": [
            "agent_run:linked-approval-timeout-run-1",
            f"goal_finding:{health_payload['open_findings'][0]['finding_id']}",
            "approval:exec-approval-timeout-1",
        ],
        "approval_ids": ["exec-approval-timeout-1"],
        "tool_names": ["exec_command"],
        "run_id": "linked-approval-timeout-run-1",
        "wait_elapsed_seconds": health_payload["approval_state"]["approval_wait_elapsed_sec"],
        "wait_timeout_seconds": 60,
    }
    assert health_payload["blocker_diagnostic"] == {
        "cause_code": "approval_wait_timeout",
        "what_is_blocked": "Active goal execution is waiting on operator approval.",
        "why_blocked": "A pending exec_command approval has exceeded the configured approval wait timeout for the linked run.",
        "actor_required": "operator",
        "next_action": "resolve_approval",
        "auto_resume_policy": "resume_after_approval_if_run_can_continue",
        "source_event_ids": [
            "agent_run:linked-approval-timeout-run-1",
            f"goal_finding:{health_payload['open_findings'][0]['finding_id']}",
            "approval:exec-approval-timeout-1",
        ],
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
        "blocker_type": "approval",
        "approval_count": 1,
        "run_id": "linked-approval-timeout-run-1",
        "approval_ids": ["exec-approval-timeout-1"],
        "tool_names": ["exec_command"],
    }
    assert findings_after_resolve == []
