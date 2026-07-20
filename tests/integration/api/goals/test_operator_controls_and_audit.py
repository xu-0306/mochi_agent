"""Goal API integration tests: Operator Controls And Audit."""

from ._support import *  # noqa: F401,F403


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
        assert health_payload["approval_diagnostic"] is None
        assert health_payload["blocker_diagnostic"] == {
            "cause_code": "operator_emergency_stop",
            "what_is_blocked": "Active goal execution cannot continue under current operator controls.",
            "why_blocked": "Goal execution is blocked by operator emergency stop. Reason: maintenance window",
            "actor_required": "operator",
            "next_action": "clear_operator_controls",
            "auto_resume_policy": "manual_resume_required",
            "source_event_ids": [
                f"goal_operator_controls:{health_payload['operator_controls']['updated_at']}",
            ],
        }

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

        audit_response = client.get(
            "/v1/goals/operator-audit-log?event_type=goal_state_changed&goal_id=goal-estop-running-1"
        )
        assert audit_response.status_code == 200
        state_events = audit_response.json()

    assert goal_payload["status"] == "paused"
    assert goal_payload["attempts"][0]["status"] == "paused"
    assert "operator emergency stop" in str(goal_payload["latest_error"])
    assert linked_run["status"] == "paused"
    assert state_events
    pause_event = state_events[0]
    assert pause_event["event_type"] == "goal_state_changed"
    assert pause_event["subject_type"] == "goal"
    assert pause_event["subject_id"] == "goal-estop-running-1"
    assert pause_event["action"] == "operator_pause"
    assert pause_event["details"] == {
        "type": "goal_state_changed",
        "goal_id": "goal-estop-running-1",
        "previous_status": "running",
        "status": "paused",
        "attempt_id": "goal-estop-running-attempt-1",
        "agent_run_id": "goal-estop-linked-run-1",
        "reason": "Goal execution is blocked by operator emergency stop. Reason: operator pause",
        "metadata": {"source": "operator_controls"},
    }

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
