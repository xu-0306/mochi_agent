"""Goal API integration tests: Proposals And Copy."""

from ._support import (
    Any,
    MochiConfig,
    Path,
    SimpleNamespace,
    TestClient,
    create_app,
    pytest,
)


def test_pending_goal_proposal_intent_route_uses_bounded_engine_invoke(tmp_path: Path) -> None:
    class _FakeEngine:
        def __init__(self) -> None:
            self.requests: list[Any] = []

        async def invoke(self, request: Any) -> Any:
            self.requests.append(request)
            return SimpleNamespace(
                content='{"intent":"confirm_start","confidence":0.91,"rationale":"The user clearly wants to start the pending goal now."}'
            )

    app = create_app()
    fake_engine = _FakeEngine()
    app.state.engine_factory = lambda: fake_engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/goals/pending-proposal-intent",
            json={
                "message": "Please launch the pending goal now.",
                "proposal_objective": "Research ESG LLM fine-tuning methods",
                "execution_mode": "single_agent",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "type": "goal_pending_proposal_intent",
        "intent": "confirm_start",
        "confidence": 0.91,
        "rationale": "The user clearly wants to start the pending goal now.",
    }
    assert len(fake_engine.requests) == 1
    request = fake_engine.requests[0]
    assert request.tool_mode == "disabled"
    assert request.execution_profile == "judge"
    assert request.persist_session is False

def test_goal_proposal_assistant_copy_route_uses_bounded_engine_invoke(tmp_path: Path) -> None:
    class _FakeEngine:
        def __init__(self) -> None:
            self.requests: list[Any] = []

        async def invoke(self, request: Any) -> Any:
            self.requests.append(request)
            return SimpleNamespace(
                content=(
                    "\u6211\u628a\u4f60\u7684\u9700\u6c42\u6574\u7406\u6210\u4e00\u4efd goal \u8349\u7a3f\uff0c\u4f5c\u70ba\u9019\u500b\u4efb\u52d9\u7684\u57f7\u884c\u5951\u7d04\u3002 "
                    "\u76ee\u524d\u9078\u5b9a\u7684\u57f7\u884c\u7b56\u7565\u662f autonomous_single_agent\uff0c\u78ba\u8a8d\u555f\u52d5\u5f8c\u624d\u6703\u958b\u59cb\u57f7\u884c\u3002"
                )
            )

    app = create_app()
    fake_engine = _FakeEngine()
    app.state.engine_factory = lambda: fake_engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/goals/proposal-assistant-copy",
            json={
                "message": "\u5e6b\u6211\u67e5\u8a62 ESG LLM \u5fae\u8abf\u76f8\u95dc\u8ad6\u6587",
                "proposal_objective": "Research ESG LLM fine-tuning papers and compare them",
                "execution_mode": "single_agent",
                "protocol_selection": "autonomous_single_agent",
                "role_summary": "Primary agent continues the task directly with the current chat tools.",
                "runtime_mode": "Single-agent long-running execution",
                "revision_index": 0,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "type": "goal_proposal_assistant_copy",
        "explanation": "\u6211\u628a\u4f60\u7684\u9700\u6c42\u6574\u7406\u6210\u4e00\u4efd goal \u8349\u7a3f\uff0c\u4f5c\u70ba\u9019\u500b\u4efb\u52d9\u7684\u57f7\u884c\u5951\u7d04\u3002 \u76ee\u524d\u9078\u5b9a\u7684\u57f7\u884c\u7b56\u7565\u662f autonomous_single_agent\uff0c\u78ba\u8a8d\u555f\u52d5\u5f8c\u624d\u6703\u958b\u59cb\u57f7\u884c\u3002",
        "source": "model",
    }
    assert len(fake_engine.requests) == 1
    request = fake_engine.requests[0]
    assert request.tool_mode == "disabled"
    assert request.execution_profile == "judge"
    assert request.persist_session is False
    assert "Latest user request" in request.message
    assert "\u5e6b\u6211\u67e5\u8a62 ESG LLM \u5fae\u8abf\u76f8\u95dc\u8ad6\u6587" in request.message
    assert "launch directly" not in payload["explanation"]

def test_pending_goal_proposal_intent_route_uses_deterministic_chinese_start_rule(
    tmp_path: Path,
) -> None:
    class _FakeEngine:
        def __init__(self) -> None:
            self.requests: list[Any] = []

        async def invoke(self, request: Any) -> Any:
            self.requests.append(request)
            raise AssertionError("deterministic confirm-start rule should bypass bounded invoke")

    app = create_app()
    fake_engine = _FakeEngine()
    app.state.engine_factory = lambda: fake_engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/goals/pending-proposal-intent",
            json={
                "message": "\u958b\u59cb",
                "proposal_objective": "Research ESG LLM fine-tuning methods",
                "execution_mode": "single_agent",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "confirm_start"
    assert payload["confidence"] == 1.0
    assert len(fake_engine.requests) == 0

@pytest.mark.parametrize(
    ("message",),
    [
        ("\u597d \u958b\u59cb",),
        ("\u53ef\u4ee5\uff0c\u555f\u52d5",),
        ("ok start",),
    ],
)
def test_pending_goal_proposal_intent_route_accepts_mixed_affirmation_start_phrases(
    tmp_path: Path,
    message: str,
) -> None:
    class _FakeEngine:
        def __init__(self) -> None:
            self.requests: list[Any] = []

        async def invoke(self, request: Any) -> Any:
            self.requests.append(request)
            raise AssertionError("affirmation-plus-start phrases should bypass bounded invoke")

    app = create_app()
    fake_engine = _FakeEngine()
    app.state.engine_factory = lambda: fake_engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/goals/pending-proposal-intent",
            json={
                "message": message,
                "proposal_objective": "Research ESG LLM fine-tuning methods",
                "execution_mode": "single_agent",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "confirm_start"
    assert payload["confidence"] == 1.0
    assert len(fake_engine.requests) == 0

def test_goal_proposal_assistant_copy_route_falls_back_to_chinese_copy(
    tmp_path: Path,
) -> None:
    app = create_app()
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/goals/proposal-assistant-copy",
            json={
                "message": "\u5e6b\u6211\u67e5\u8a62 ESG LLM \u5fae\u8abf\u76f8\u95dc\u8ad6\u6587",
                "proposal_objective": "Research ESG LLM fine-tuning papers and compare them",
                "execution_mode": "single_agent",
                "protocol_selection": "autonomous_single_agent",
                "role_summary": "Primary agent continues the task directly with the current chat tools.",
                "runtime_mode": "Single-agent long-running execution",
                "revision_index": 0,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "fallback"
    assert "\u555f\u52d5" in payload["explanation"] or "\u57f7\u884c" in payload["explanation"]

def test_goal_follow_up_assistant_copy_route_uses_bounded_engine_invoke(tmp_path: Path) -> None:
    class _FakeEngine:
        def __init__(self) -> None:
            self.requests: list[Any] = []

        async def invoke(self, request: Any) -> Any:
            self.requests.append(request)
            return SimpleNamespace(
                content=(
                    "\u76ee\u524d\u9019\u500b goal \u9084\u5361\u5728 exec_command \u7684\u6838\u51c6\u3002 "
                    "\u5148\u8655\u7406\u6838\u51c6\u5f8c\uff0c\u6211\u624d\u80fd\u7e7c\u7e8c\u5957\u7528\u4f60\u7684\u65b0\u65b9\u5411\u3002"
                )
            )

    app = create_app()
    fake_engine = _FakeEngine()
    app.state.engine_factory = lambda: fake_engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/goals/follow-up-assistant-copy",
            json={
                "message": "\u9019\u662f\u4ec0\u9ebc\u610f\u601d",
                "kind": "manual_resolution_required",
                "goal_objective": "Research ESG LLM fine-tuning methods",
                "goal_status": "waiting_approval",
                "linked_run_status": "blocked",
                "continuation_action": "manual_resolution_required",
                "continuation_summary": "Goal is waiting on operator approval before it can continue.",
                "approval_count": 1,
                "tool_names": ["exec_command"],
                "recommended_action": "resolve_approval",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "type": "goal_follow_up_assistant_copy",
        "explanation": (
            "\u76ee\u524d\u9019\u500b goal \u9084\u5361\u5728 exec_command \u7684\u6838\u51c6\u3002 "
            "\u5148\u8655\u7406\u6838\u51c6\u5f8c\uff0c\u6211\u624d\u80fd\u7e7c\u7e8c\u5957\u7528\u4f60\u7684\u65b0\u65b9\u5411\u3002"
        ),
        "source": "model",
    }
    assert len(fake_engine.requests) == 1
    request = fake_engine.requests[0]
    assert request.tool_mode == "disabled"
    assert request.execution_profile == "judge"
    assert request.persist_session is False
    assert "Goal follow-up outcome" in request.message
    assert "manual_resolution_required" in request.message

def test_goal_follow_up_assistant_copy_route_falls_back_to_chinese_copy(
    tmp_path: Path,
) -> None:
    app = create_app()
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/goals/follow-up-assistant-copy",
            json={
                "message": "\u8acb\u7e7c\u7e8c\u8655\u7406",
                "kind": "manual_resolution_required",
                "goal_objective": "Research ESG LLM fine-tuning methods",
                "goal_status": "waiting_approval",
                "continuation_summary": "The active goal needs approval handling before it can continue.",
                "approval_count": 1,
                "tool_names": ["exec_command"],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "goal_follow_up_assistant_copy"
    assert payload["source"] == "fallback"
    assert "The active goal needs approval handling" not in payload["explanation"]
    assert "\u5f85\u6838\u51c6\u5de5\u5177" in payload["explanation"]
    assert "Goal Console" in payload["explanation"]

def test_goal_follow_up_assistant_copy_route_supports_queued_after_resolution(
    tmp_path: Path,
) -> None:
    app = create_app()
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/goals/follow-up-assistant-copy",
            json={
                "message": "Please continue after approval",
                "kind": "queued_after_resolution",
                "goal_objective": "Research ESG LLM fine-tuning methods",
                "goal_status": "waiting_approval",
                "continuation_summary": (
                    "Goal is waiting on operator approval, but the current attempt can queue "
                    "your follow-up guidance and resume with it once approval is resolved."
                ),
                "approval_count": 1,
                "tool_names": ["exec_command"],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "goal_follow_up_assistant_copy"
    assert payload["source"] == "fallback"
    assert "restating it" in payload["explanation"]
