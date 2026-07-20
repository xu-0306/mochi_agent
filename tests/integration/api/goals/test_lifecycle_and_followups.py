"""Goal API integration tests: Lifecycle And Followups."""

from ._support import (
    UTC,
    Any,
    GoalStrategyRegistryEntryData,
    MultiAgentRunResult,
    Path,
    TestClient,
    _create_goal_test_app,
    _create_goal_test_client,
    _wait_goal_until,
    asyncio,
    datetime,
    pytest,
    registered_goal_strategy_entries_for_test,
    sqlite3,
    time,
    timedelta,
)


def test_goal_create_normalizes_prompt_duration_into_runtime_contract(tmp_path: Path) -> None:
    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Please run for 2 hours and collect forum conversations for corpus building.",
                "title": "Prompt Duration Goal",
            },
        )
        assert create_response.status_code == 200
        created = create_response.json()
        goal_id = created["goal_id"]
        assert created["run_policy"]["requested_duration_text"] == "2 hours"
        assert created["run_policy"]["requested_duration_sec"] == 7_200
        assert created["run_policy"]["requested_duration_source"] == "objective"
        assert created["run_policy"]["runtime_mode"] == "fixed_duration"

        get_response = client.get(f"/v1/goals/{goal_id}")
        assert get_response.status_code == 200
        fetched = get_response.json()
        assert fetched["run_policy"] == created["run_policy"]

        list_response = client.get("/v1/goals")
        assert list_response.status_code == 200
        listed = list_response.json()
        assert listed[0]["run_policy"] == created["run_policy"]

        health_response = client.get(f"/v1/goals/{goal_id}/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()
        assert health_payload["run_policy"]["requested_duration_sec"] == 7_200
        assert health_payload["run_policy"]["runtime_mode"] == "fixed_duration"

def test_goal_create_preserves_explicit_protocol_for_single_agent_lane_goals(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="multi_agent_debate",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Single-agent lane goal completed."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Handle this as a single-agent goal.",
                "title": "Single Agent Goal",
                "execution_mode": "single_agent",
                "protocol_id": "multi_agent_debate",
            },
        )
        assert create_response.status_code == 200
        created = create_response.json()
        goal_id = created["goal_id"]
        assert created["execution_mode"] == "single_agent"
        assert created["interaction_mode"] == "goal"
        assert created["execution_topology"] == "multi_agent"
        assert created["strategy_id"] == "multi_agent_debate"
        assert created["selection_source"] == "explicit_override"
        assert created["selection_reason"] == "Strategy explicitly set to multi_agent_debate."
        assert created["protocol_id"] == "multi_agent_debate"
        assert created["bound_run_id"] is None
        assert created["protocol_selection"] == "multi_agent_debate"
        assert created["selection_rationale"] == "Strategy explicitly set to multi_agent_debate."

        list_response = client.get("/v1/goals")
        assert list_response.status_code == 200
        assert list_response.json()[0]["execution_mode"] == "single_agent"
        assert list_response.json()[0]["strategy_id"] == "multi_agent_debate"
        assert list_response.json()[0]["selection_source"] == "explicit_override"
        assert list_response.json()[0]["selection_reason"] == "Strategy explicitly set to multi_agent_debate."
        assert list_response.json()[0]["protocol_id"] == "multi_agent_debate"

        get_response = client.get(f"/v1/goals/{goal_id}")
        assert get_response.status_code == 200
        assert get_response.json()["execution_mode"] == "single_agent"
        assert get_response.json()["strategy_id"] == "multi_agent_debate"
        assert get_response.json()["selection_source"] == "explicit_override"
        assert get_response.json()["selection_reason"] == "Strategy explicitly set to multi_agent_debate."
        assert get_response.json()["protocol_id"] == "multi_agent_debate"

        health_response = client.get(f"/v1/goals/{goal_id}/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()
        assert health_payload["execution_mode"] == "single_agent"
        assert health_payload["interaction_mode"] == "goal"
        assert health_payload["execution_topology"] == "multi_agent"
        assert health_payload["strategy_id"] == "multi_agent_debate"
        assert health_payload["selection_source"] == "explicit_override"
        assert health_payload["selection_reason"] == "Strategy explicitly set to multi_agent_debate."
        assert health_payload["protocol_id"] == "multi_agent_debate"
        assert health_payload["bound_run_id"] is None
        assert health_payload["protocol_selection"] == "multi_agent_debate"
        assert health_payload["selection_rationale"] == "Strategy explicitly set to multi_agent_debate."

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200
        started = start_response.json()
        assert started["execution_mode"] == "single_agent"
        assert started["interaction_mode"] == "goal"
        assert started["execution_topology"] == "multi_agent"
        assert started["strategy_id"] == "multi_agent_debate"
        assert started["selection_source"] == "explicit_override"
        assert started["selection_reason"] == "Strategy explicitly set to multi_agent_debate."
        assert started["protocol_id"] == "multi_agent_debate"
        assert started["protocol_selection"] == "multi_agent_debate"
        assert started["selection_rationale"] == "Strategy explicitly set to multi_agent_debate."

        completed_goal = _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)
        assert completed_goal["execution_mode"] == "single_agent"
        assert completed_goal["interaction_mode"] == "goal"
        assert completed_goal["execution_topology"] == "multi_agent"
        assert completed_goal["strategy_id"] == "multi_agent_debate"
        assert completed_goal["selection_source"] == "explicit_override"
        assert completed_goal["selection_reason"] == "Strategy explicitly set to multi_agent_debate."
        assert completed_goal["protocol_id"] == "multi_agent_debate"
        linked_run_id = completed_goal["attempts"][0]["agent_run_id"]
        assert linked_run_id is not None
        assert completed_goal["bound_run_id"] == linked_run_id
        assert completed_goal["protocol_selection"] == "multi_agent_debate"
        assert completed_goal["selection_rationale"] == "Strategy explicitly set to multi_agent_debate."

        linked_run_response = client.get(f"/v1/agent-runs/{linked_run_id}")
        assert linked_run_response.status_code == 200
        assert linked_run_response.json()["protocol_id"] == "multi_agent_debate"

def test_goal_strategy_registry_endpoint_exposes_default_and_descriptions(
    tmp_path: Path,
) -> None:
    with _create_goal_test_client(tmp_path) as client:
        response = client.get("/v1/goals/strategies")

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "goal_strategy_registry"
    assert payload["default_strategy_id"] == "autonomous_single_agent"

    entries = {entry["id"]: entry for entry in payload["entries"]}
    assert "autonomous_single_agent" in entries
    assert "multi_agent_debate" in entries
    assert entries["autonomous_single_agent"]["is_default"] is True
    assert entries["autonomous_single_agent"]["available"] is True
    assert "availability_reason" in entries["autonomous_single_agent"]
    assert entries["autonomous_single_agent"]["description"]
    assert entries["autonomous_single_agent"]["when_to_use"]
    assert entries["autonomous_single_agent"]["when_not_to_use"]
    assert entries["autonomous_single_agent"]["description"] != "autonomous_single_agent"
    assert entries["multi_agent_debate"]["description"]
    assert entries["multi_agent_debate"]["when_to_use"]
    assert entries["multi_agent_debate"]["available"] is True

    if "teacher_student_distill" in entries:
        assert entries["teacher_student_distill"]["is_default"] is False
        assert payload["default_strategy_id"] != "teacher_student_distill"
        assert entries["teacher_student_distill"]["requires_confirmation"] is True
        assert entries["teacher_student_distill"]["available"] is True
        assert entries["teacher_student_distill"]["availability_reason"]

def test_goal_strategy_registry_route_is_not_captured_as_goal_id(
    tmp_path: Path,
) -> None:
    with _create_goal_test_client(tmp_path) as client:
        strategy_response = client.get("/v1/goals/strategies")
        missing_goal_response = client.get("/v1/goals/not-a-real-goal-id")

    assert strategy_response.status_code == 200
    assert strategy_response.json()["type"] == "goal_strategy_registry"
    assert missing_goal_response.status_code == 404
    assert missing_goal_response.json()["detail"] == "Goal not found"

def test_goal_strategy_registry_endpoint_includes_test_injected_entries(
    tmp_path: Path,
) -> None:
    injected = GoalStrategyRegistryEntryData(
        id="test_registry_strategy",
        name="Test registry strategy",
        display_name="Test registry strategy",
        description="A test-only strategy with natural-language registry guidance.",
        when_to_use="Use only in tests that prove routes read from the registry.",
        when_not_to_use="Never use for production Goal execution.",
        execution_topology="single_agent",
        protocol_id=None,
        required_capabilities=("test_registry",),
        approval_profile="test_only",
        control_scope="goal",
        success_signals=("The injected entry appears in the API response.",),
        failure_modes=("The route hardcodes production strategy ids.",),
        available=True,
        availability_reason="Injected only while this test context manager is active.",
        override_label="Test strategy",
        selection_guidance="Injected strategy used for route registry tests.",
    )

    with registered_goal_strategy_entries_for_test((injected,)):
        with _create_goal_test_client(tmp_path) as client:
            response = client.get("/v1/goals/strategies")

    assert response.status_code == 200
    entries = {entry["id"]: entry for entry in response.json()["entries"]}
    assert "test_registry_strategy" in entries
    assert entries["test_registry_strategy"]["description"] == (
        "A test-only strategy with natural-language registry guidance."
    )
    assert entries["test_registry_strategy"]["protocol_id"] is None
    assert entries["test_registry_strategy"]["available"] is True
    assert entries["test_registry_strategy"]["availability_reason"] == (
        "Injected only while this test context manager is active."
    )

def test_goal_create_defaults_missing_single_agent_protocol_to_autonomous_agent(
    tmp_path: Path,
) -> None:
    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Handle this as a single-agent goal.",
                "title": "Autonomous Single Agent Goal",
                "execution_mode": "single_agent",
            },
        )
        assert create_response.status_code == 200
        created = create_response.json()

        assert created["execution_mode"] == "single_agent"
        assert created["interaction_mode"] == "goal"
        assert created["execution_topology"] == "single_agent"
        assert created["strategy_id"] == "autonomous_single_agent"
        assert created["selection_source"] == "safe_default"
        assert created["selection_reason"] == (
            "Defaulted to autonomous_single_agent because no explicit strategy was provided."
        )
        assert created["protocol_id"] == "autonomous_single_agent"
        assert created["protocol_selection"] == "autonomous_single_agent"
        assert created["selection_rationale"] == (
            "Defaulted to autonomous_single_agent because no explicit strategy was provided."
        )

def test_goal_create_defaults_missing_execution_mode_and_protocol_to_autonomous_agent(
    tmp_path: Path,
) -> None:
    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Handle this with the default goal execution path.",
                "title": "Default Goal",
            },
        )
        assert create_response.status_code == 200
        created = create_response.json()
        goal_id = created["goal_id"]

        assert created["execution_mode"] == "single_agent"
        assert created["interaction_mode"] == "goal"
        assert created["execution_topology"] == "single_agent"
        assert created["strategy_id"] == "autonomous_single_agent"
        assert created["selection_source"] == "safe_default"
        assert created["selection_reason"] == (
            "Defaulted to autonomous_single_agent because no explicit strategy was provided."
        )
        assert created["protocol_id"] == "autonomous_single_agent"
        assert created["protocol_selection"] == "autonomous_single_agent"
        assert created["selection_rationale"] == (
            "Defaulted to autonomous_single_agent because no explicit strategy was provided."
        )

        get_response = client.get(f"/v1/goals/{goal_id}")
        assert get_response.status_code == 200
        fetched = get_response.json()
        assert fetched["execution_mode"] == "single_agent"
        assert fetched["strategy_id"] == "autonomous_single_agent"
        assert fetched["selection_source"] == "safe_default"
        assert fetched["selection_reason"] == (
            "Defaulted to autonomous_single_agent because no explicit strategy was provided."
        )
        assert fetched["protocol_id"] == "autonomous_single_agent"
        assert fetched["protocol_selection"] == "autonomous_single_agent"

        list_response = client.get("/v1/goals")
        assert list_response.status_code == 200
        listed = list_response.json()[0]
        assert listed["strategy_id"] == "autonomous_single_agent"
        assert listed["selection_source"] == "safe_default"
        assert listed["selection_reason"] == (
            "Defaulted to autonomous_single_agent because no explicit strategy was provided."
        )

        health_response = client.get(f"/v1/goals/{goal_id}/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()
        assert health_payload["strategy_id"] == "autonomous_single_agent"
        assert health_payload["selection_source"] == "safe_default"
        assert health_payload["selection_reason"] == (
            "Defaulted to autonomous_single_agent because no explicit strategy was provided."
        )

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200
        started = start_response.json()
        assert started["strategy_id"] == "autonomous_single_agent"
        assert started["selection_source"] == "safe_default"
        assert started["selection_reason"] == (
            "Defaulted to autonomous_single_agent because no explicit strategy was provided."
        )

def test_goal_start_refuses_placeholder_success_when_no_model_role_selected(
    tmp_path: Path,
) -> None:
    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Research this for 20 minutes and report findings.",
                "execution_mode": "single_agent",
            },
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200

        goal_payload = _wait_goal_until(
            client,
            goal_id,
            {"awaiting_resources"},
            timeout_seconds=4.0,
        )
        assert goal_payload["status"] == "awaiting_resources"
        linked_run_id = goal_payload["attempts"][0]["agent_run_id"]
        assert linked_run_id is not None

        linked_run_response = client.get(f"/v1/agent-runs/{linked_run_id}")
        assert linked_run_response.status_code == 200
        linked_run = linked_run_response.json()
        assert linked_run["status"] == "awaiting_resources"
        assert "placeholder" not in str(linked_run.get("summary", {}).get("final_answer", "")).lower()
        assert "requires at least one selected model role" in linked_run["latest_error"]

def test_goal_start_passes_selected_model_roles_to_linked_agent_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_requests: list[Any] = []

    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        captured_requests.append(request)
        return MultiAgentRunResult(
            run_id=request.run_id or "goal-model-run",
            protocol="autonomous_single_agent",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Model-backed goal completed."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Research this with the selected chat model.",
                "execution_mode": "single_agent",
                "selected_models_roles": {
                    "by_role": {"agent": "ollama:qwen3.5:9b"},
                    "entries": [{"role": "agent", "model_id": "ollama:qwen3.5:9b"}],
                },
            },
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200
        completed_goal = _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)
        linked_run_id = completed_goal["attempts"][0]["agent_run_id"]
        assert linked_run_id is not None

        linked_run_response = client.get(f"/v1/agent-runs/{linked_run_id}")
        assert linked_run_response.status_code == 200
        linked_run = linked_run_response.json()
        assert linked_run["selected_models_roles"]["by_role"] == {
            "agent": "ollama:qwen3.5:9b",
        }
        assert captured_requests
        assert captured_requests[0].metadata["selected_models_roles"]["by_role"] == {
            "agent": "ollama:qwen3.5:9b",
        }
        assert captured_requests[0].metadata["summary"]["selected_models_roles"]["by_role"] == {
            "agent": "ollama:qwen3.5:9b",
        }

def test_goal_api_prefers_canonical_goal_fields_over_stale_legacy_summary_metadata(
    tmp_path: Path,
) -> None:
    app, runtime_service = _create_goal_test_app(tmp_path)

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Handle this with the default goal execution path.",
                "title": "Canonical Goal",
            },
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        asyncio.run(
            runtime_service._store.update_goal_metadata(  # noqa: SLF001
                goal_id,
                summary={
                    "interaction_mode": "workflow",
                    "execution_topology": "multi_agent",
                    "protocol_selection": "teacher_student_distill",
                    "selection_rationale": "Stale legacy summary still claims distill mode.",
                    "strategy_id": "teacher_student_distill",
                },
            )
        )

        get_response = client.get(f"/v1/goals/{goal_id}")
        assert get_response.status_code == 200
        fetched = get_response.json()
        assert fetched["execution_mode"] == "single_agent"
        assert fetched["interaction_mode"] == "goal"
        assert fetched["execution_topology"] == "single_agent"
        assert fetched["strategy_id"] == "autonomous_single_agent"
        assert fetched["selection_source"] == "safe_default"
        assert fetched["selection_reason"] == (
            "Defaulted to autonomous_single_agent because no explicit strategy was provided."
        )
        assert fetched["protocol_id"] == "autonomous_single_agent"
        assert fetched["protocol_selection"] == "autonomous_single_agent"
        assert fetched["selection_rationale"] == (
            "Defaulted to autonomous_single_agent because no explicit strategy was provided."
        )

        list_response = client.get("/v1/goals")
        assert list_response.status_code == 200
        listed = list_response.json()[0]
        assert listed["interaction_mode"] == "goal"
        assert listed["execution_topology"] == "single_agent"
        assert listed["protocol_selection"] == "autonomous_single_agent"
        assert listed["selection_rationale"] == (
            "Defaulted to autonomous_single_agent because no explicit strategy was provided."
        )

        health_response = client.get(f"/v1/goals/{goal_id}/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()
        assert health_payload["interaction_mode"] == "goal"
        assert health_payload["execution_topology"] == "single_agent"
        assert health_payload["protocol_selection"] == "autonomous_single_agent"
        assert health_payload["selection_rationale"] == (
            "Defaulted to autonomous_single_agent because no explicit strategy was provided."
        )

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200
        started = start_response.json()
        assert started["interaction_mode"] == "goal"
        assert started["execution_topology"] == "single_agent"
        assert started["protocol_selection"] == "autonomous_single_agent"
        assert started["selection_rationale"] == (
            "Defaulted to autonomous_single_agent because no explicit strategy was provided."
        )

def test_goal_api_legacy_workflow_without_strategy_does_not_default_to_distill(
    tmp_path: Path,
) -> None:
    app, runtime_service = _create_goal_test_app(tmp_path)

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Legacy workflow-shaped goal without strategy metadata.",
                "title": "Legacy Workflow Goal",
                "execution_mode": "workflow",
            },
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        asyncio.run(runtime_service._store.initialize())  # noqa: SLF001
        with sqlite3.connect(runtime_service._store._db_path) as conn:  # noqa: SLF001
            conn.execute(
                """
                UPDATE goals
                SET strategy_id=NULL,
                    selection_source=NULL,
                    selection_reason=NULL,
                    protocol_id=NULL,
                    summary_json=?
                WHERE id=?
                """,
                (
                    '{"interaction_mode":"workflow","execution_topology":"multi_agent"}',
                    goal_id,
                ),
            )
            conn.commit()

        get_response = client.get(f"/v1/goals/{goal_id}")
        assert get_response.status_code == 200
        fetched = get_response.json()
        assert fetched["execution_mode"] == "workflow"
        assert fetched["strategy_id"] == "autonomous_single_agent"
        assert fetched["selection_source"] == "legacy_migration"
        assert fetched["selection_reason"] == (
            "Legacy goal metadata mapped to autonomous_single_agent during strategy migration."
        )
        assert fetched["protocol_id"] == "autonomous_single_agent"
        assert fetched["protocol_selection"] == "autonomous_single_agent"
        assert fetched["selection_rationale"] == (
            "Legacy goal metadata mapped to autonomous_single_agent during strategy migration."
        )
        assert fetched["protocol_id"] != "teacher_student_distill"

def test_goal_create_selects_injected_registry_strategy_from_semantic_registry_evidence(
    tmp_path: Path,
) -> None:
    injected = GoalStrategyRegistryEntryData(
        id="panorama_synthesis",
        name="Panorama synthesis",
        display_name="Panorama synthesis",
        description=(
            "Coordinates a panorama synthesis pass that merges overlapping field notes into one reconciled map."
        ),
        when_to_use=(
            "Use when the objective is to merge overlapping field notes into one reconciled panorama map."
        ),
        when_not_to_use="Do not use for ordinary execution or debate tasks.",
        execution_topology="multi_agent",
        kind="protocol",
        protocol_id="panorama_synthesis_protocol",
        required_capabilities=("merge_pass",),
        approval_profile="standard_goal_policy",
        control_scope="goal",
        success_signals=("A reconciled panorama map is produced.",),
        failure_modes=("Merge evidence is missing.",),
        available=True,
        selection_guidance=(
            "Select for objectives that explicitly ask to merge overlapping field notes into a reconciled panorama map."
        ),
    )

    with registered_goal_strategy_entries_for_test((injected,)):
        with _create_goal_test_client(tmp_path) as client:
            create_response = client.post(
                "/v1/goals",
                json={
                    "objective": "Merge overlapping field notes into a reconciled panorama map for this expedition.",
                },
            )

    assert create_response.status_code == 200
    created = create_response.json()
    assert created["strategy_id"] == "panorama_synthesis"
    assert created["protocol_id"] == "panorama_synthesis_protocol"
    assert created["selection_source"] == "semantic_registry_selector"
    assert "panorama_synthesis" in created["selection_reason"]
    assert "registry" in created["selection_reason"]

def test_goal_create_does_not_semantically_select_confirmation_gated_distill_strategy(
    tmp_path: Path,
) -> None:
    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Use teacher student generation and compression to distill this into a smaller output.",
            },
        )

    assert create_response.status_code == 200
    created = create_response.json()
    assert created["strategy_id"] == "autonomous_single_agent"
    assert created["protocol_id"] == "autonomous_single_agent"
    assert created["selection_source"] == "safe_default"
    assert "teacher_student_distill" in created["selection_reason"]
    assert "requires explicit confirmation" in created["selection_reason"]

def test_goal_create_explicit_teacher_student_distill_preserves_override_metadata_and_linked_run_protocol(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured_requests: list[Any] = []

    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        captured_requests.append(request)
        return MultiAgentRunResult(
            run_id=request.run_id or "goal-distill-run",
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Distill goal completed."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Run the explicit distillation protocol.",
                "protocol_id": "teacher_student_distill",
            },
        )
        assert create_response.status_code == 200
        created = create_response.json()
        goal_id = created["goal_id"]
        assert created["strategy_id"] == "teacher_student_distill"
        assert created["protocol_id"] == "teacher_student_distill"
        assert created["selection_source"] == "explicit_override"
        assert created["selection_reason"] == "Strategy explicitly set to teacher_student_distill."

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200

        completed_goal = _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)
        assert completed_goal["strategy_id"] == "teacher_student_distill"
        assert completed_goal["protocol_id"] == "teacher_student_distill"
        assert completed_goal["selection_source"] == "explicit_override"
        assert captured_requests
        assert captured_requests[0].protocol["protocol"] == "teacher_student_distill"

def test_goal_create_rejects_conflicting_strategy_and_protocol_ids(
    tmp_path: Path,
) -> None:
    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Run with conflicting strategy and protocol.",
                "strategy_id": "multi_agent_debate",
                "protocol_id": "teacher_student_distill",
            },
        )

    assert create_response.status_code == 422
    assert "Conflicting strategy_id/protocol_id" in create_response.text

def test_goal_create_accepts_first_class_strategy_metadata(
    tmp_path: Path,
) -> None:
    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Run this through the debate strategy.",
                "strategy_id": "multi_agent_debate",
                "selection_source": "explicit_override",
                "selection_reason": "User selected debate mode from the goal composer.",
            },
        )
        assert create_response.status_code == 200
        created = create_response.json()
        goal_id = created["goal_id"]

        assert created["strategy_id"] == "multi_agent_debate"
        assert created["selection_source"] == "explicit_override"
        assert created["selection_reason"] == "User selected debate mode from the goal composer."
        assert created["protocol_id"] == "multi_agent_debate"
        assert created["protocol_selection"] == "multi_agent_debate"
        assert created["selection_rationale"] == "User selected debate mode from the goal composer."

        get_response = client.get(f"/v1/goals/{goal_id}")
        assert get_response.status_code == 200
        fetched = get_response.json()
        assert fetched["strategy_id"] == "multi_agent_debate"
        assert fetched["selection_source"] == "explicit_override"
        assert fetched["selection_reason"] == "User selected debate mode from the goal composer."
        assert fetched["protocol_id"] == "multi_agent_debate"

def test_goal_create_explicit_runtime_policy_duration_overrides_prompt_duration(
    tmp_path: Path,
) -> None:
    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Please run for 2 hours and collect forum conversations for corpus building.",
                "title": "Explicit Duration Goal",
                "run_policy": {
                    "requested_duration_sec": 1_800,
                    "requested_duration": "30 minutes",
                    "context_handoff_threshold": 85,
                    "max_attempt_retries": "4",
                    "unknown_key": "preserve-me",
                },
            },
        )
        assert create_response.status_code == 200
        created = create_response.json()
        goal_id = created["goal_id"]
        run_policy = created["run_policy"]
        assert run_policy["requested_duration_text"] == "30 minutes"
        assert run_policy["requested_duration_sec"] == 1_800
        assert run_policy["requested_duration_source"] == "run_policy.requested_duration_sec"
        assert run_policy["runtime_mode"] == "fixed_duration"
        assert run_policy["context_handoff_threshold"] == 0.85
        assert run_policy["max_attempt_retries"] == 4
        assert run_policy["unknown_key"] == "preserve-me"

        get_response = client.get(f"/v1/goals/{goal_id}")
        assert get_response.status_code == 200
        fetched = get_response.json()
        assert fetched["run_policy"] == run_policy

def test_goal_start_preserves_normalized_runtime_policy_on_linked_agent_run(
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
            artifacts={"final_answer": "Goal-linked agent run completed."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Please run for 2 hours and collect forum conversations for corpus building.",
                "title": "Bridge Policy Goal",
                "protocol_id": "teacher_student_distill",
                "run_policy": {
                    "requested_duration_sec": 1_800,
                    "requested_duration": "30 minutes",
                    "context_handoff_threshold": 85,
                    "max_attempt_retries": 4,
                    "unknown_key": "preserve-me",
                },
            },
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200

        completed_goal = _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)
        linked_run_id = completed_goal["attempts"][0]["agent_run_id"]
        assert linked_run_id is not None

        linked_run_response = client.get(f"/v1/agent-runs/{linked_run_id}")
        assert linked_run_response.status_code == 200
        linked_run = linked_run_response.json()
        assert linked_run["run_policy"]["requested_duration_text"] == "30 minutes"
        assert linked_run["run_policy"]["requested_duration_sec"] == 1_800
        assert linked_run["run_policy"]["requested_duration_source"] == "run_policy.requested_duration_sec"
        assert linked_run["run_policy"]["runtime_mode"] == "fixed_duration"
        assert linked_run["run_policy"]["context_handoff_threshold"] == 0.85
        assert linked_run["run_policy"]["max_attempt_retries"] == 4
        assert linked_run["run_policy"]["unknown_key"] == "preserve-me"

def test_goal_health_passes_through_current_generation_payload(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    app, runtime_service = _create_goal_test_app(tmp_path)

    async def _fake_get_goal_health(goal_id: str) -> dict[str, Any]:
        assert goal_id == "goal-generation-1"
        return {
            "goal_id": goal_id,
            "status": "running",
            "current_generation": {
                "generation_id": "goal-generation-1-gen-3",
                "generation_index": 3,
                "status": "active",
                "attempt_id": "goal-generation-attempt-1",
                "agent_run_id": "linked-generation-run-1",
                "started_at": "2026-06-23T03:00:00+00:00",
            },
        }

    monkeypatch.setattr(runtime_service, "get_goal_health", _fake_get_goal_health)

    with TestClient(app) as client:
        response = client.get("/v1/goals/goal-generation-1/health")
        assert response.status_code == 200
        payload = response.json()

    assert payload["current_generation"] == {
        "generation_id": "goal-generation-1-gen-3",
        "generation_index": 3,
        "status": "active",
        "attempt_id": "goal-generation-attempt-1",
        "agent_run_id": "linked-generation-run-1",
        "started_at": "2026-06-23T03:00:00+00:00",
    }

def test_goal_health_surfaces_generation_scoped_live_subagent_runtime_telemetry(
    tmp_path: Path,
) -> None:
    app, runtime_service = _create_goal_test_app(tmp_path)
    goal_id = "goal-live-subagent-runtime-1"
    attempt_id = "goal-live-subagent-runtime-attempt-1"
    run_id = "linked-live-subagent-runtime-run-1"

    asyncio.run(
        runtime_service._store.create_goal(
            goal_id=goal_id,
            objective="Surface live subagent runtime telemetry on current generation health.",
            protocol_id="teacher_student_distill",
            run_policy={
                "generation_refresh_interval_sec": 600,
                "generation_token_refresh_threshold": 30,
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
            title="Live subagent runtime run",
            topic="generation-scoped telemetry",
            summary={
                "goal_id": goal_id,
                "goal_attempt_id": attempt_id,
                "objective": "Surface live subagent runtime telemetry on current generation health.",
                "task_input": "Surface live subagent runtime telemetry on current generation health.",
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
    asyncio.run(
        runtime_service._store.append_agent_run_artifact(
            run_id,
            artifact_id=f"{run_id}:attempt:{attempt_id}:subagent_runtime:gen1",
            artifact_type="subagent_runtime",
            title="Subagent Runtime Trace",
            uri=f"agent-run://{run_id}/artifacts/{attempt_id}/subagent_runtime/gen1",
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

    with TestClient(app) as client:
        health_before_response = client.get(f"/v1/goals/{goal_id}/health")
        assert health_before_response.status_code == 200
        health_before = health_before_response.json()

    current_generation_before = health_before["current_generation"]
    assert current_generation_before["generation_id"] == 1
    assert current_generation_before["subagent_invocation_count"] == 2
    assert current_generation_before["subagent_completed_invocation_count"] == 2
    assert current_generation_before["subagent_token_tracked_invocation_count"] == 2
    assert current_generation_before["observed_input_tokens"] == 20
    assert current_generation_before["observed_output_tokens"] == 12
    assert current_generation_before["observed_total_tokens"] == 32
    assert current_generation_before["observed_generation_time_ms"] == 41.5
    assert current_generation_before["observed_finish_reason_counts"] == {"stop": 2}
    assert current_generation_before["generation_token_refresh_threshold"] == 30
    assert current_generation_before["token_refresh_due"] is True
    assert current_generation_before["token_refresh_over_threshold"] == 2
    assert current_generation_before["last_subagent_runtime_snapshot_at"]

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

    with TestClient(app) as client:
        health_after_response = client.get(f"/v1/goals/{goal_id}/health")
        assert health_after_response.status_code == 200
        health_after = health_after_response.json()

    current_generation_after = health_after["current_generation"]
    assert current_generation_after["generation_id"] == 2
    assert current_generation_after["generation_token_refresh_threshold"] == 30
    assert current_generation_after.get("subagent_invocation_count") is None
    assert current_generation_after.get("observed_input_tokens") is None
    assert current_generation_after.get("observed_total_tokens") is None
    assert current_generation_after.get("token_refresh_due") is None
    assert current_generation_after.get("token_refresh_over_threshold") is None
    assert current_generation_after.get("last_subagent_runtime_snapshot_at") is None

def test_goal_refresh_reuses_running_linked_run_as_manual_refresh(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured_requests: list[Any] = []

    async def _running_then_refresh(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        resume_runtime = (
            dict(request.metadata.get("resume_runtime") or {})
            if isinstance(getattr(request, "metadata", None), dict)
            else {}
        )
        if resume_runtime:
            captured_requests.append(request)
            return MultiAgentRunResult(
                run_id=request.run_id,
                protocol="teacher_student_distill",
                state="succeeded",
                task_input=request.task_input,
                candidates=[],
                selected_candidate_id=None,
                evaluation={},
                artifacts={"final_answer": "Live manual refresh reused the running linked run."},
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

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _running_then_refresh)

    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Refresh a running linked run onto a fresh worker generation.",
                "title": "Live Manual Refresh",
                "protocol_id": "teacher_student_distill",
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
        checkpoint_captured_at = "2026-06-23T13:00:00+00:00"
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
                metadata={"signature": "live-manual-refresh-checkpoint-1"},
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
                    "goal_objective": "Refresh a running linked run onto a fresh worker generation.",
                    "attempt_id": attempt_id,
                    "agent_run_id": run_id,
                    "protocol_id": "teacher_student_distill",
                    "agent_run_status": "running",
                    "stage": "research_context_prepared",
                    "checkpoint_index": 2,
                    "unfinished_steps": ["resume the linked research worker from the durable handoff"],
                    "captured_at": checkpoint_captured_at,
                },
                metadata={"signature": "live-manual-refresh-memory-1"},
                captured_at=checkpoint_captured_at,
            )
        )
        run = asyncio.run(runtime_service._store.get_agent_run(run_id))
        assert run is not None
        summary = dict(run.get("summary") or {})
        summary["goal_id"] = goal_id
        summary["goal_attempt_id"] = attempt_id
        summary["objective"] = "Refresh a running linked run onto a fresh worker generation."
        summary["task_input"] = "Refresh a running linked run onto a fresh worker generation."
        summary["recovery_state"] = {
            "status": "running",
            "action": "continue",
            "reason": "Operator requested a fresh worker generation.",
            "stage": "research_context_prepared",
            "checkpoint": {
                "checkpoint_index": 2,
                "stage": "research_context_prepared",
                "captured_at": checkpoint_captured_at,
            },
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
        }
        asyncio.run(runtime_service._store.update_agent_run_metadata(run_id, summary=summary))

        refresh_response = client.post(f"/v1/goals/{goal_id}/refresh")
        assert refresh_response.status_code == 200

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
    assert linked_run["summary"]["final_answer"] == "Live manual refresh reused the running linked run."
    assert len(captured_requests) == 1
    assert captured_requests[0].run_id == run_id
    assert captured_requests[0].metadata["resume_strategy"] == "restart_attempt"
    assert captured_requests[0].metadata["resume_payload"] == {}
    assert captured_requests[0].metadata["resume_runtime"]["source"] == "manual_resume"
    assert latest_generation is not None
    assert latest_generation["generation_index"] == 2
    assert latest_generation["rollover_reason"] == "manual_refresh"

def test_goal_refresh_rejects_live_refresh_without_durable_handoff(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    async def _slow_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self, request
        await asyncio.sleep(5)
        return MultiAgentRunResult(
            run_id="unexpected",
            protocol="teacher_student_distill",
            state="succeeded",
            task_input="unexpected",
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "unexpected initial completion"},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _slow_run)

    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Do not refresh a running linked run without durable handoff.",
                "title": "Unsafe Live Manual Refresh",
                "protocol_id": "teacher_student_distill",
            },
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200

        running_goal = _wait_goal_until(client, goal_id, {"running"}, timeout_seconds=4.0)
        assert len(running_goal["attempts"]) == 1

        refresh_response = client.post(f"/v1/goals/{goal_id}/refresh")
        assert refresh_response.status_code == 409
        assert "durable handoff" in refresh_response.json()["detail"]

        goal_response = client.get(f"/v1/goals/{goal_id}")
        assert goal_response.status_code == 200
        goal_payload = goal_response.json()
        assert goal_payload["status"] == "running"
        assert len(goal_payload["attempts"]) == 1

        cancel_response = client.post(f"/v1/goals/{goal_id}/cancel")
        assert cancel_response.status_code == 200

def test_goal_progress_snapshot_endpoints_list_persisted_records(
    tmp_path: Path,
) -> None:
    app, runtime_service = _create_goal_test_app(tmp_path)
    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-progress-api-1",
            objective="Inspect persisted checkpoint and memory snapshot records",
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-progress-api-attempt-1",
            goal_id="goal-progress-api-1",
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id="linked-progress-api-run-1",
        )
    )
    older_checkpoint = asyncio.run(
        runtime_service._store.create_goal_checkpoint(
            goal_id="goal-progress-api-1",
            attempt_id="goal-progress-api-attempt-1",
            agent_run_id="linked-progress-api-run-1",
            checkpoint_index=2,
            stage="evidence_collection",
            source="agent_run_sync",
            payload={"checkpoint_index": 2, "stage": "evidence_collection"},
            metadata={"signature": "api-sig-1"},
            captured_at="2026-06-23T02:00:00+00:00",
        )
    )
    newer_checkpoint = asyncio.run(
        runtime_service._store.create_goal_checkpoint(
            goal_id="goal-progress-api-1",
            attempt_id="goal-progress-api-attempt-1",
            agent_run_id="linked-progress-api-run-1",
            checkpoint_index=3,
            stage="controller_decision",
            source="agent_run_sync",
            payload={"checkpoint_index": 3, "stage": "controller_decision"},
            metadata={"signature": "api-sig-2"},
            captured_at="2026-06-23T02:05:00+00:00",
        )
    )
    memory_snapshot = asyncio.run(
        runtime_service._store.create_goal_memory_snapshot(
            goal_id="goal-progress-api-1",
            attempt_id="goal-progress-api-attempt-1",
            checkpoint_id=newer_checkpoint["id"],
            snapshot_kind="compact_recovery_v1",
            snapshot={"checkpoint_index": 3, "stage": "controller_decision"},
            metadata={"signature": "api-mem-sig-1"},
            captured_at="2026-06-23T02:05:01+00:00",
        )
    )

    with TestClient(app) as client:
        checkpoints_response = client.get(
            "/v1/goals/goal-progress-api-1/checkpoints",
            params={"attempt_id": "goal-progress-api-attempt-1"},
        )
        assert checkpoints_response.status_code == 200
        checkpoints_payload = checkpoints_response.json()

        limited_checkpoints_response = client.get(
            "/v1/goals/goal-progress-api-1/checkpoints",
            params={"attempt_id": "goal-progress-api-attempt-1", "limit": 1},
        )
        assert limited_checkpoints_response.status_code == 200
        limited_checkpoints_payload = limited_checkpoints_response.json()

        memory_snapshots_response = client.get(
            "/v1/goals/goal-progress-api-1/memory-snapshots",
            params={"attempt_id": "goal-progress-api-attempt-1"},
        )
        assert memory_snapshots_response.status_code == 200
        memory_snapshots_payload = memory_snapshots_response.json()

        missing_goal_response = client.get("/v1/goals/missing-goal/checkpoints")
        assert missing_goal_response.status_code == 404
        assert missing_goal_response.json()["detail"] == "Goal not found"

    assert [item["checkpoint_id"] for item in checkpoints_payload] == [
        newer_checkpoint["id"],
        older_checkpoint["id"],
    ]
    assert limited_checkpoints_payload[0]["checkpoint_id"] == newer_checkpoint["id"]
    assert len(limited_checkpoints_payload) == 1
    assert memory_snapshots_payload[0]["snapshot_id"] == memory_snapshot["id"]
    assert memory_snapshots_payload[0]["checkpoint_id"] == newer_checkpoint["id"]

def test_goals_api_flow_creates_and_tracks_linked_agent_runs(
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
            artifacts={"final_answer": "Goal-linked agent run completed."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={
                "objective": "Collect multimodal datasets for long-running training jobs",
                "title": "Dataset Collection Goal",
                "goal_type": "dataset_collection",
                "protocol_id": "teacher_student_distill",
                "topic": "multimodal corpora",
                "project_id": "proj-goal",
                "workspace_dir": str(tmp_path / "workspace"),
                "run_policy": {"max_wall_clock_sec": 18_000},
                "capability_policy": {
                    "allowed_tools": [" web_search ", "web_fetch", "web_search", ""],
                    "operator_mode": "observe",
                },
                "source_manifest": {"sources": [{"id": "hf-datasets"}]},
                "summary": {"phase": "created"},
                "metadata": {"packet": "A"},
            },
        )
        assert create_response.status_code == 200
        created = create_response.json()
        goal_id = created["goal_id"]
        expected_capability_policy = {
            "operator_mode": "observe",
            "allowed_tools": ["web_search", "web_fetch"],
        }
        assert created["status"] == "created"
        assert created["capability_policy"] == expected_capability_policy
        assert created["attempts"] == []

        list_response = client.get("/v1/goals")
        assert list_response.status_code == 200
        listed = list_response.json()
        assert len(listed) == 1
        assert listed[0]["goal_id"] == goal_id
        assert listed[0]["capability_policy"] == expected_capability_policy

        get_response = client.get(f"/v1/goals/{goal_id}")
        assert get_response.status_code == 200
        fetched = get_response.json()
        assert fetched["objective"] == "Collect multimodal datasets for long-running training jobs"
        assert fetched["run_policy"]["max_wall_clock_sec"] == 18_000
        assert fetched["capability_policy"] == expected_capability_policy

        start_response = client.post(f"/v1/goals/{goal_id}/start")
        assert start_response.status_code == 200
        started = start_response.json()
        assert started["status"] in {"queued", "running", "completed"}
        assert len(started["attempts"]) == 1
        first_attempt = started["attempts"][0]
        assert started["current_attempt_id"] == first_attempt["attempt_id"]
        assert first_attempt["trigger"] == "manual_start"
        assert first_attempt["agent_run_id"] is not None

        completed_goal = _wait_goal_until(client, goal_id, {"completed"}, timeout_seconds=4.0)
        assert completed_goal["attempts"][0]["status"] == "completed"
        linked_run_id = completed_goal["attempts"][0]["agent_run_id"]
        assert linked_run_id is not None

        health_response = client.get(f"/v1/goals/{goal_id}/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()
        assert health_payload["status"] == "completed"
        assert health_payload["capability_policy"] == expected_capability_policy
        assert health_payload["lease"] is None

        linked_run_response = client.get(f"/v1/agent-runs/{linked_run_id}")
        assert linked_run_response.status_code == 200
        linked_run = linked_run_response.json()
        assert linked_run["status"] == "succeeded"
        assert linked_run["summary"]["goal_id"] == goal_id
        assert linked_run["summary"]["goal_attempt_id"] == first_attempt["attempt_id"]
        assert linked_run["summary"]["goal_capability_policy"] == expected_capability_policy

        agent_runs = client.get("/v1/agent-runs")
        assert agent_runs.status_code == 200
        assert len(agent_runs.json()) == 1

    assert len(captured_requests) == 1
    assert captured_requests[0].metadata["goal_capability_policy"] == expected_capability_policy
