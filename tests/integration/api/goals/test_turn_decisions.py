"""Goal API integration tests: Turn Decisions."""

from ._support import *  # noqa: F401,F403

def test_goal_turn_decision_route_classifies_blocked_explanation_question(
    tmp_path: Path,
) -> None:
    app, runtime_service = _create_goal_test_app(tmp_path)
    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-turn-decision-blocked-1",
            objective="Explain why the goal is blocked.",
            summary={"phase": "operator_review"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-turn-decision-blocked-attempt-1",
            goal_id="goal-turn-decision-blocked-1",
            attempt_index=1,
            status="waiting_approval",
            trigger="manual_start",
            summary={
                "linked_approval_state": {
                    "status": "waiting_approval",
                    "pending_count": 1,
                    "approval_ids": ["exec-approval-turn-decision-1"],
                    "tool_names": ["exec_command"],
                }
            },
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-turn-decision-blocked-1",
            "waiting_approval",
            current_attempt_id="goal-turn-decision-blocked-attempt-1",
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/goals/goal-turn-decision-blocked-1/turn-decision",
            json={"message": "why is this blocked?"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["lane"] == "active_goal_turn"
    assert payload["kind"] == "explain_goal_state"
    assert payload["selection_source"] == "bounded_fallback"
    assert payload["requires_confirmation"] is False
    assert payload["goal_status"] == "waiting_approval"
    assert payload["recommended_action"] == "resolve_approval"

def test_goal_turn_decision_route_classifies_progress_question_as_explanatory(
    tmp_path: Path,
) -> None:
    app, runtime_service = _create_goal_test_app(tmp_path)
    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-turn-decision-progress-1",
            objective="Summarize current progress.",
            summary={"phase": "running"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-turn-decision-progress-attempt-1",
            goal_id="goal-turn-decision-progress-1",
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id="goal-turn-decision-progress-run-1",
            summary={},
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id="goal-turn-decision-progress-run-1",
            protocol_id="autonomous_single_agent",
            title="Progress monitor run",
            topic="Summarize current progress.",
            summary={},
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status("goal-turn-decision-progress-run-1", "running"))
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-turn-decision-progress-1",
            "running",
            current_attempt_id="goal-turn-decision-progress-attempt-1",
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/goals/goal-turn-decision-progress-1/turn-decision",
            json={"message": "what happened so far?"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] in {"answer_question", "explain_goal_state"}
    assert payload["kind"] != "steer"
    assert payload["selection_source"] == "bounded_fallback"
    assert payload["recommended_action"] == "monitor"

def test_goal_turn_decision_route_classifies_chinese_blocked_approval_question_as_explanatory(
    tmp_path: Path,
) -> None:
    app, runtime_service = _create_goal_test_app(tmp_path)
    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-turn-decision-blocked-zh-1",
            objective="Explain approval blockage in Chinese.",
            summary={"phase": "operator_review"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-turn-decision-blocked-zh-attempt-1",
            goal_id="goal-turn-decision-blocked-zh-1",
            attempt_index=1,
            status="waiting_approval",
            trigger="manual_start",
            agent_run_id="goal-turn-decision-blocked-zh-run-1",
            summary={
                "linked_approval_state": {
                    "status": "waiting_approval",
                    "pending_count": 1,
                    "approval_ids": ["exec-approval-turn-decision-zh-1"],
                    "tool_names": ["exec_command"],
                }
            },
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id="goal-turn-decision-blocked-zh-run-1",
            protocol_id="autonomous_single_agent",
            title="Blocked approval monitor",
            topic="Explain approval blockage in Chinese.",
            summary={},
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status("goal-turn-decision-blocked-zh-run-1", "awaiting_approval"))
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-turn-decision-blocked-zh-1",
            "waiting_approval",
            current_attempt_id="goal-turn-decision-blocked-zh-attempt-1",
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/goals/goal-turn-decision-blocked-zh-1/turn-decision",
            json={"message": "为什么这个目标还在等批准？"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] == "explain_goal_state"
    assert payload["selection_source"] == "bounded_fallback"
    assert payload["requires_confirmation"] is False
    assert payload["goal_status"] == "waiting_approval"
    assert payload["linked_run_status"] == "awaiting_approval"
    assert payload["recommended_action"] == "resolve_approval"

def test_goal_turn_decision_route_classifies_chinese_progress_question_as_explanatory(
    tmp_path: Path,
) -> None:
    app, runtime_service = _create_goal_test_app(tmp_path)
    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-turn-decision-progress-zh-1",
            objective="Summarize current progress in Chinese.",
            summary={"phase": "running"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-turn-decision-progress-zh-attempt-1",
            goal_id="goal-turn-decision-progress-zh-1",
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id="goal-turn-decision-progress-zh-run-1",
            summary={},
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id="goal-turn-decision-progress-zh-run-1",
            protocol_id="autonomous_single_agent",
            title="Chinese progress monitor run",
            topic="Summarize current progress in Chinese.",
            summary={},
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status("goal-turn-decision-progress-zh-run-1", "running"))
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-turn-decision-progress-zh-1",
            "running",
            current_attempt_id="goal-turn-decision-progress-zh-attempt-1",
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/goals/goal-turn-decision-progress-zh-1/turn-decision",
            json={"message": "現在進度怎麼樣？"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] in {"answer_question", "explain_goal_state"}
    assert payload["kind"] != "steer"
    assert payload["selection_source"] == "bounded_fallback"
    assert payload["goal_status"] == "running"
    assert payload["linked_run_status"] == "running"
    assert payload["recommended_action"] == "monitor"

@pytest.mark.parametrize(
    "message",
    [
        "¿Cuál es el progreso hasta ahora?",
        "अब तक क्या प्रगति हुई है?",
    ],
)
def test_goal_turn_decision_route_classifies_non_english_progress_questions_as_non_mutating(
    tmp_path: Path,
    message: str,
) -> None:
    app, runtime_service = _create_goal_test_app(tmp_path)
    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-turn-decision-progress-intl-1",
            objective="Summarize multilingual progress questions.",
            summary={"phase": "running"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-turn-decision-progress-intl-attempt-1",
            goal_id="goal-turn-decision-progress-intl-1",
            attempt_index=1,
            status="running",
            trigger="manual_start",
            agent_run_id="goal-turn-decision-progress-intl-run-1",
            summary={},
        )
    )
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id="goal-turn-decision-progress-intl-run-1",
            protocol_id="autonomous_single_agent",
            title="International progress monitor run",
            topic="Summarize multilingual progress questions.",
            summary={},
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status("goal-turn-decision-progress-intl-run-1", "running"))
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-turn-decision-progress-intl-1",
            "running",
            current_attempt_id="goal-turn-decision-progress-intl-attempt-1",
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/goals/goal-turn-decision-progress-intl-1/turn-decision",
            json={"message": message},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] in {"answer_question", "explain_goal_state"}
    assert payload["kind"] != "steer"
    assert payload["selection_source"] == "bounded_fallback"
    assert payload["linked_run_status"] == "running"
    assert payload["recommended_action"] == "monitor"

def test_goal_turn_decision_route_does_not_treat_explanation_question_as_steering(
    tmp_path: Path,
) -> None:
    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={"objective": "Explain implementation choices clearly."},
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        response = client.post(
            f"/v1/goals/{goal_id}/turn-decision",
            json={"message": "why did you use python for this?"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] in {"answer_question", "explain_goal_state"}
    assert payload["kind"] != "steer"
    assert payload["selection_source"] == "bounded_fallback"

def test_goal_turn_decision_route_does_not_treat_continue_question_as_steering(
    tmp_path: Path,
) -> None:
    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={"objective": "Continue only when explicitly instructed."},
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        response = client.post(
            f"/v1/goals/{goal_id}/turn-decision",
            json={"message": "can you continue with benchmark comparisons?"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] in {"answer_question", "explain_goal_state"}
    assert payload["kind"] != "steer"
    assert payload["selection_source"] == "bounded_fallback"

@pytest.mark.parametrize(
    ("message", "allowed_kinds", "requires_confirmation"),
    [
        ("maybe focus on benchmarks", {"clarify"}, True),
        ("I think focus on benchmarks might help", {"clarify"}, True),
        ("can you continue the goal?", {"answer_question", "explain_goal_state", "clarify"}, None),
        ("should we take a different approach?", {"answer_question", "explain_goal_state", "clarify"}, None),
        ("maybe take a different approach", {"clarify"}, True),
    ],
)
def test_goal_turn_decision_route_does_not_overclassify_ambiguous_or_question_mutations(
    tmp_path: Path,
    message: str,
    allowed_kinds: set[str],
    requires_confirmation: bool | None,
) -> None:
    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={"objective": "Stay conservative about mutating follow-up intent."},
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        response = client.post(
            f"/v1/goals/{goal_id}/turn-decision",
            json={"message": message},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] in allowed_kinds
    assert payload["kind"] not in {"steer", "replan", "lifecycle"}
    assert payload["selection_source"] == "bounded_fallback"
    if requires_confirmation is not None:
        assert payload["requires_confirmation"] is requires_confirmation

def test_goal_turn_decision_route_classifies_steering_instruction(
    tmp_path: Path,
) -> None:
    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={"objective": "Keep researching the topic."},
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        response = client.post(
            f"/v1/goals/{goal_id}/turn-decision",
            json={"message": "focus on benchmark comparisons next"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] == "steer"
    assert payload["selection_source"] == "bounded_fallback"
    assert payload["requires_confirmation"] is False

def test_goal_turn_decision_route_classifies_ambiguous_continue_instruction_as_clarify(
    tmp_path: Path,
) -> None:
    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={"objective": "Only continue after specific guidance."},
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        response = client.post(
            f"/v1/goals/{goal_id}/turn-decision",
            json={"message": "keep going"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] == "clarify"
    assert payload["selection_source"] == "bounded_fallback"
    assert payload["requires_confirmation"] is True

def test_goal_turn_decision_route_classifies_replan_instruction(
    tmp_path: Path,
) -> None:
    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={"objective": "Work this through the current approach."},
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        response = client.post(
            f"/v1/goals/{goal_id}/turn-decision",
            json={"message": "replan this and take a different approach"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] == "replan"
    assert payload["selection_source"] == "bounded_fallback"
    assert payload["requires_confirmation"] is False

@pytest.mark.parametrize(
    ("message", "expected_kind"),
    [
        ("pause goal", "lifecycle"),
        ("resume", "lifecycle"),
        ("stop goal", "lifecycle"),
        ("continue with benchmark comparisons", "steer"),
    ],
)
def test_goal_turn_decision_route_preserves_explicit_mutating_commands(
    tmp_path: Path,
    message: str,
    expected_kind: str,
) -> None:
    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={"objective": "Keep explicit mutating command routing stable."},
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        response = client.post(
            f"/v1/goals/{goal_id}/turn-decision",
            json={"message": message},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] == expected_kind
    assert payload["selection_source"] == "bounded_fallback"
    assert payload["requires_confirmation"] is False

def test_goal_turn_decision_route_classifies_lifecycle_command(
    tmp_path: Path,
) -> None:
    with _create_goal_test_client(tmp_path) as client:
        create_response = client.post(
            "/v1/goals",
            json={"objective": "Pause and resume lifecycle coverage."},
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        response = client.post(
            f"/v1/goals/{goal_id}/turn-decision",
            json={"message": "resume this goal"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] == "lifecycle"
    assert payload["selection_source"] == "bounded_fallback"
    assert payload["requires_confirmation"] is False

def test_goal_turn_decision_route_uses_production_semantic_selector_from_app_runtime_service(
    tmp_path: Path,
) -> None:
    class _FakeEngine:
        def __init__(self) -> None:
            self.requests: list[Any] = []

        async def invoke(self, request: Any) -> Any:
            self.requests.append(request)
            return SimpleNamespace(
                content=(
                    '{"kind":"steer","confidence":0.97,'
                    '"selection_reason":"Semantic selector recognized a non-English steering instruction.",'
                    '"requires_confirmation":false}'
                )
            )

    app = create_app()
    fake_engine = _FakeEngine()
    app.state.engine_factory = lambda: fake_engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/goals",
            json={"objective": "Accept semantic steering from the production runtime-service builder."},
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        response = client.post(
            f"/v1/goals/{goal_id}/turn-decision",
            json={"message": "다음에는 벤치마크 비교에 집중해 줘"},
        )
        assert callable(getattr(app.state.runtime_service, "_active_goal_turn_selector", None))

    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] == "steer"
    assert payload["selection_source"] == "semantic_registry_selector"
    assert payload["requires_confirmation"] is False
    assert len(fake_engine.requests) == 1
    request = fake_engine.requests[0]
    assert request.tool_mode == "disabled"
    assert request.execution_profile == "judge"
    assert request.persist_session is False
    assert '"fallback_decision"' in request.message

def test_goal_turn_decision_route_semantic_selector_can_override_non_keyword_steering(
    tmp_path: Path,
) -> None:
    def semantic_selector(context: Any) -> dict[str, Any]:
        assert context.fallback_decision.kind == "clarify"
        return {
            "kind": "steer",
            "confidence": 0.97,
            "selection_source": "semantic_registry_selector",
            "selection_reason": "Semantic selector recognized a non-English directive to steer the goal.",
            "requires_confirmation": False,
            "goal_status": context.fallback_decision.goal_status,
            "linked_run_status": context.fallback_decision.linked_run_status,
            "recommended_action": context.fallback_decision.recommended_action,
        }

    app, _runtime_service = _create_goal_test_app(
        tmp_path,
        active_goal_turn_selector=semantic_selector,
    )
    with TestClient(app) as client:
        create_response = client.post(
            "/v1/goals",
            json={"objective": "Accept semantic steering when the bounded fallback is ambiguous."},
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        response = client.post(
            f"/v1/goals/{goal_id}/turn-decision",
            json={"message": "다음에는 벤치마크 비교에 집중해 줘"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] == "steer"
    assert payload["selection_source"] == "semantic_registry_selector"
    assert payload["requires_confirmation"] is False

def test_goal_turn_decision_route_falls_back_when_semantic_selector_returns_malformed_result(
    tmp_path: Path,
) -> None:
    def semantic_selector(_context: Any) -> dict[str, Any]:
        return {
            "kind": "steer",
            "selection_source": "semantic_registry_selector",
        }

    app, _runtime_service = _create_goal_test_app(
        tmp_path,
        active_goal_turn_selector=semantic_selector,
    )
    with TestClient(app) as client:
        create_response = client.post(
            "/v1/goals",
            json={"objective": "Stay conservative when semantic selector output is malformed."},
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        response = client.post(
            f"/v1/goals/{goal_id}/turn-decision",
            json={"message": "다음에는 벤치마크 비교에 집중해 줘"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] == "clarify"
    assert payload["selection_source"] == "bounded_fallback"
    assert payload["requires_confirmation"] is True

def test_goal_turn_decision_route_low_confidence_semantic_mutation_does_not_override_explanatory_fallback(
    tmp_path: Path,
) -> None:
    def semantic_selector(context: Any) -> dict[str, Any]:
        assert context.fallback_decision.kind == "explain_goal_state"
        return {
            "kind": "steer",
            "confidence": 0.89,
            "selection_source": "semantic_registry_selector",
            "selection_reason": "Selector weakly inferred a steering intent.",
            "requires_confirmation": False,
            "goal_status": context.fallback_decision.goal_status,
            "linked_run_status": context.fallback_decision.linked_run_status,
            "recommended_action": context.fallback_decision.recommended_action,
        }

    app, runtime_service = _create_goal_test_app(
        tmp_path,
        active_goal_turn_selector=semantic_selector,
    )
    asyncio.run(
        runtime_service._store.create_goal(
            goal_id="goal-turn-decision-semantic-blocked-1",
            objective="Explain why the goal is blocked.",
            summary={"phase": "operator_review"},
        )
    )
    asyncio.run(
        runtime_service._store.create_goal_attempt(
            attempt_id="goal-turn-decision-semantic-blocked-attempt-1",
            goal_id="goal-turn-decision-semantic-blocked-1",
            attempt_index=1,
            status="waiting_approval",
            trigger="manual_start",
            summary={
                "linked_approval_state": {
                    "status": "waiting_approval",
                    "pending_count": 1,
                    "approval_ids": ["exec-approval-turn-decision-semantic-1"],
                    "tool_names": ["exec_command"],
                }
            },
        )
    )
    asyncio.run(
        runtime_service._store.update_goal_status(
            "goal-turn-decision-semantic-blocked-1",
            "waiting_approval",
            current_attempt_id="goal-turn-decision-semantic-blocked-attempt-1",
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/goals/goal-turn-decision-semantic-blocked-1/turn-decision",
            json={"message": "why is this blocked?"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] == "explain_goal_state"
    assert payload["selection_source"] == "bounded_fallback"
    assert payload["requires_confirmation"] is False

def test_goal_turn_decision_route_low_confidence_semantic_mutation_does_not_override_clarify_fallback(
    tmp_path: Path,
) -> None:
    def semantic_selector(context: Any) -> dict[str, Any]:
        assert context.fallback_decision.kind == "clarify"
        return {
            "kind": "steer",
            "confidence": 0.94,
            "selection_source": "semantic_registry_selector",
            "selection_reason": "Selector inferred steering intent, but not confidently enough.",
            "requires_confirmation": False,
            "goal_status": context.fallback_decision.goal_status,
            "linked_run_status": context.fallback_decision.linked_run_status,
            "recommended_action": context.fallback_decision.recommended_action,
        }

    app, _runtime_service = _create_goal_test_app(
        tmp_path,
        active_goal_turn_selector=semantic_selector,
    )
    with TestClient(app) as client:
        create_response = client.post(
            "/v1/goals",
            json={"objective": "Stay conservative about ambiguous continue instructions."},
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        response = client.post(
            f"/v1/goals/{goal_id}/turn-decision",
            json={"message": "keep going"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] == "clarify"
    assert payload["selection_source"] == "bounded_fallback"
    assert payload["requires_confirmation"] is True

def test_goal_turn_decision_route_confirming_semantic_mutation_does_not_override_clarify_fallback(
    tmp_path: Path,
) -> None:
    def semantic_selector(context: Any) -> dict[str, Any]:
        assert context.fallback_decision.kind == "clarify"
        return {
            "kind": "steer",
            "confidence": 0.99,
            "selection_source": "semantic_registry_selector",
            "selection_reason": "Selector inferred steering intent but still needs confirmation.",
            "requires_confirmation": True,
            "goal_status": context.fallback_decision.goal_status,
            "linked_run_status": context.fallback_decision.linked_run_status,
            "recommended_action": context.fallback_decision.recommended_action,
        }

    app, _runtime_service = _create_goal_test_app(
        tmp_path,
        active_goal_turn_selector=semantic_selector,
    )
    with TestClient(app) as client:
        create_response = client.post(
            "/v1/goals",
            json={"objective": "Stay conservative about confirming semantic steering."},
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        response = client.post(
            f"/v1/goals/{goal_id}/turn-decision",
            json={"message": "keep going"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] == "clarify"
    assert payload["selection_source"] == "bounded_fallback"
    assert payload["requires_confirmation"] is True

def test_goal_turn_decision_route_selector_exception_falls_back_without_server_error(
    tmp_path: Path,
) -> None:
    def semantic_selector(_context: Any) -> dict[str, Any]:
        raise RuntimeError("selector failure")

    app, _runtime_service = _create_goal_test_app(
        tmp_path,
        active_goal_turn_selector=semantic_selector,
    )
    with TestClient(app) as client:
        create_response = client.post(
            "/v1/goals",
            json={"objective": "Fall back when the semantic selector fails."},
        )
        assert create_response.status_code == 200
        goal_id = create_response.json()["goal_id"]

        response = client.post(
            f"/v1/goals/{goal_id}/turn-decision",
            json={"message": "다음에는 벤치마크 비교에 집중해 줘"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] == "clarify"
    assert payload["selection_source"] == "bounded_fallback"
    assert payload["requires_confirmation"] is True
