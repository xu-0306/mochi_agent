"""Goal API integration tests: Recovery And Leases."""

from ._support import (
    UTC,
    Any,
    MochiConfig,
    MultiAgentRunResult,
    Path,
    RuntimeService,
    RuntimeStore,
    TestClient,
    _create_goal_test_app,
    _create_goal_test_client,
    _set_goal_started_at,
    _wait_goal_until,
    asyncio,
    create_app,
    datetime,
    timedelta,
)


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

    assert payload["status"] in {"running", "awaiting_resources"}
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
        assert health_payload["approval_diagnostic"] is None
        assert health_payload["blocker_diagnostic"] == {
            "cause_code": "runtime_hard_stop_reached",
            "what_is_blocked": "Active goal execution cannot safely continue.",
            "why_blocked": "The configured hard stop time has passed, so goal execution cannot continue safely.",
            "actor_required": "operator",
            "next_action": "inspect_runtime_budget",
            "auto_resume_policy": "manual_resume_required",
            "source_event_ids": [
                f"goal_finding:{health_payload['open_findings'][0]['finding_id']}",
            ],
        }

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
