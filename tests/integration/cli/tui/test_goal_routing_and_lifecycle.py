"""Chat TUI CLI tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from mochi.agents.events import FinalAnswerEvent
from mochi.config.schema import SecurityConfig
from mochi.terminal_goal_helpers import (
    is_natural_language_goal_request,
    resolve_goal_workflow_routing,
)

runner = CliRunner()

@pytest.mark.asyncio
@pytest.mark.asyncio
async def test_chat_tui_async_goal_lifecycle_commands_use_terminal_goal_flow(
    monkeypatch,
    capsys,
) -> None:
    from mochi.main import _chat_tui_async

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001, ARG002
            self.calls: list[tuple[str, str | None]] = []
            self.closed = False
            self.invocations: list[object] = []

        async def initialize(self) -> None:
            return None

        async def chat(
            self,
            message: str,
            session_id: str | None = None,
        ) -> AsyncIterator[object]:
            self.calls.append((message, session_id))
            yield FinalAnswerEvent(content="unexpected")

        async def invoke(self, request):  # noqa: ANN001
            self.invocations.append(request)
            if str(getattr(request, "session_id", "")).startswith("goal-proposal-copy:"):
                return SimpleNamespace(
                    content=(
                        "I framed your request as a goal draft that acts as the contract for this task. "
                        "This scope fits a single-agent strategy, and execution begins only after you confirm the start."
                    )
                )
            assert '"user_follow_up": "start it"' in request.message
            return SimpleNamespace(
                content='{"intent":"confirm_start","confidence":0.93,"rationale":"The user clearly wants to launch the pending goal now."}'
            )

        async def close(self) -> None:
            self.closed = True

    class _FakeSessionStore:
        def __init__(self, sessions_dir) -> None:  # noqa: ANN001, ARG002
            self.by_session: dict[str, list[dict[str, object]]] = {}

        async def save_event(self, session_id: str, event: dict[str, object]) -> None:
            self.by_session.setdefault(session_id, []).append(dict(event))

        async def load_session(self, session_id: str) -> list[dict[str, object]]:
            return list(self.by_session.get(session_id, []))

        async def delete_session(self, session_id: str) -> bool:
            self.by_session.pop(session_id, None)
            return True

    class _FakeRuntimeService:
        def __init__(self) -> None:
            self.goal: dict[str, object] | None = None

        async def create_goal(self, payload) -> dict[str, object]:  # noqa: ANN001
            self.goal = {
                "goal_id": "goal-1",
                "objective": payload.objective,
                "execution_mode": payload.execution_mode,
                "protocol_id": payload.protocol_id,
                "status": "created",
                "attempts": [],
                "current_attempt_id": None,
                "project_id": payload.project_id,
                "workspace_dir": payload.workspace_dir,
                "latest_error": None,
            }
            return dict(self.goal)

        async def start_goal(self, goal_id: str) -> dict[str, object]:
            assert goal_id == "goal-1"
            assert self.goal is not None
            self.goal.update(
                {
                    "status": "running",
                    "current_attempt_id": "attempt-1",
                    "attempts": [
                        {
                            "attempt_id": "attempt-1",
                            "agent_run_id": "run-1",
                        }
                    ],
                }
            )
            return dict(self.goal)

        async def get_goal(self, goal_id: str) -> dict[str, object] | None:
            assert goal_id == "goal-1"
            return dict(self.goal) if self.goal is not None else None

        async def pause_goal(self, goal_id: str) -> dict[str, object]:
            assert goal_id == "goal-1"
            assert self.goal is not None
            self.goal["status"] = "paused"
            return dict(self.goal)

        async def resume_goal(self, goal_id: str) -> dict[str, object]:
            assert goal_id == "goal-1"
            assert self.goal is not None
            self.goal["status"] = "running"
            return dict(self.goal)

        async def cancel_goal(self, goal_id: str) -> dict[str, object]:
            assert goal_id == "goal-1"
            assert self.goal is not None
            self.goal["status"] = "cancelled"
            return dict(self.goal)

        async def close(self) -> None:
            return None

    fake_store = _FakeSessionStore(None)
    fake_runtime = _FakeRuntimeService()
    fake_engine = _FakeEngine(None)
    inputs = iter(
        [
            "/workflow keep working on this for 30 minutes",
            "start it",
            "/goal status",
            "/goal pause",
            "/goal resume",
            "/goal stop",
            "/exit",
        ]
    )

    monkeypatch.setattr(
        "mochi.config.manager.load_config",
        lambda config_path=None: SimpleNamespace(  # noqa: ARG005
            model="ollama:base",
            sessions_dir="/tmp/mochi-sessions",
            security=SecurityConfig(),
        ),
    )
    monkeypatch.setattr("mochi.agents.engine.AgentEngine", lambda config: fake_engine)  # noqa: ARG005
    monkeypatch.setattr("mochi.sessions.store.SessionStore", lambda sessions_dir: fake_store)  # noqa: ARG005

    async def fake_runtime_service_factory(**kwargs):  # noqa: ANN003, ARG001
        return fake_runtime

    monkeypatch.setattr("mochi.main._create_tui_runtime_service", fake_runtime_service_factory)
    monkeypatch.setattr("mochi.main.console.input", lambda prompt="": next(inputs))  # noqa: ARG005

    await _chat_tui_async(
        model=None,
        config_path=None,
        session_id="goal-session",
        max_turns=2,
    )
    captured = capsys.readouterr().out
    normalized_captured = " ".join(captured.split())

    assert "goal draft that acts as the contract for this task" in normalized_captured
    assert "Next step:" not in captured
    assert "Launch: Send a short confirmation when you want execution to begin." not in normalized_captured
    assert "Goal started." in captured
    assert "Fetched the latest goal status." in captured
    assert "Paused the active goal." in captured
    assert "Resumed the active goal." in captured
    assert "Stopped the active goal." in captured
    assert "launch directly" not in normalized_captured
    assert fake_engine.calls == []
    assert len(fake_engine.invocations) == 1

    session_events = fake_store.by_session["goal-session"]
    workflow_updates = [
        event
        for event in session_events
        if event.get("type") == "session_meta" and event.get("event") == "workflow_state_updated"
    ]
    goal_updates = [
        event
        for event in session_events
        if event.get("type") == "session_meta" and event.get("event") == "goal_state_updated"
    ]
    assert workflow_updates[-1]["workflow"]["enabled"] is False
    assert workflow_updates[-1]["workflow"]["bound_run_id"] is None
    assert goal_updates[-1]["goal"]["active_goal_id"] is None
    assert goal_updates[-1]["goal"]["active_goal_status"] == "cancelled"
    initial_pending_proposal = goal_updates[0]["goal"]["pending_proposal"]
    assert initial_pending_proposal["assistant_explanation"] == (
        "I framed your request as a goal draft that acts as the contract for this task. "
        "This scope fits a single-agent strategy, and execution begins only after you confirm the start."
    )
    assistant_goal_cards = [
        event.get("goal_card")
        for event in session_events
        if event.get("type") == "message" and event.get("role") == "assistant" and event.get("goal_card")
    ]
    assert assistant_goal_cards == []



@pytest.mark.asyncio
async def test_chat_tui_async_goal_command_runs_as_autonomous_chat_without_pending_proposal(
    monkeypatch,
    capsys,
) -> None:
    from mochi.main import _chat_tui_async

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001, ARG002
            self.calls: list[tuple[str, str | None, dict[str, object] | None]] = []
            self.invocations: list[object] = []
            self.closed = False

        async def initialize(self) -> None:
            return None

        async def chat(
            self,
            message: str,
            session_id: str | None = None,
            inference_overrides: dict[str, object] | None = None,
        ) -> AsyncIterator[object]:
            self.calls.append((message, session_id, inference_overrides))
            yield FinalAnswerEvent(content="autonomous goal turn")

        async def invoke(self, request):  # noqa: ANN001
            self.invocations.append(request)
            return SimpleNamespace(content="unexpected")

        async def close(self) -> None:
            self.closed = True

    class _FakeSessionStore:
        def __init__(self, sessions_dir) -> None:  # noqa: ANN001, ARG002
            self.by_session: dict[str, list[dict[str, object]]] = {}

        async def save_event(self, session_id: str, event: dict[str, object]) -> None:
            self.by_session.setdefault(session_id, []).append(dict(event))

        async def load_session(self, session_id: str) -> list[dict[str, object]]:
            return list(self.by_session.get(session_id, []))

        async def delete_session(self, session_id: str) -> bool:
            self.by_session.pop(session_id, None)
            return True

    fake_store = _FakeSessionStore(None)
    fake_engine = _FakeEngine(None)
    inputs = iter(["/goal ship this fix without a proposal UI", "/exit"])

    monkeypatch.setattr(
        "mochi.config.manager.load_config",
        lambda config_path=None: SimpleNamespace(  # noqa: ARG005
            model="ollama:base",
            sessions_dir="/tmp/mochi-sessions",
            security=SecurityConfig(),
            agent=SimpleNamespace(system_prompt="Base prompt."),
        ),
    )
    monkeypatch.setattr("mochi.agents.engine.AgentEngine", lambda config: fake_engine)  # noqa: ARG005
    monkeypatch.setattr("mochi.sessions.store.SessionStore", lambda sessions_dir: fake_store)  # noqa: ARG005
    monkeypatch.setattr("mochi.main.console.input", lambda prompt="": next(inputs))  # noqa: ARG005

    await _chat_tui_async(
        model=None,
        config_path=None,
        session_id="goal-session",
        max_turns=2,
    )
    captured = capsys.readouterr().out

    assert "autonomous goal turn" in captured
    assert fake_engine.invocations == []
    assert len(fake_engine.calls) == 1
    message, session_id, inference_overrides = fake_engine.calls[0]
    assert message == "ship this fix without a proposal UI"
    assert session_id == "goal-session"
    assert isinstance(inference_overrides, dict)
    system_prompt = str(inference_overrides.get("system_prompt") or "")
    assert system_prompt.startswith("Base prompt.")
    assert "Autonomous goal mode is active for this turn." in system_prompt
    assert "Goal objective: ship this fix without a proposal UI" in system_prompt
    assert "Do not create or describe a separate goal proposal UI" in system_prompt
    assert fake_store.by_session.get("goal-session", []) == []



def test_is_natural_language_goal_request_remains_disabled_placeholder() -> None:
    messages = [
        "請研究這個主題 20分鐘，整理重點給我。",
        "Research this for 30 minutes and come back with progress",
        "Keep working on this in the background for the next 30 minutes.",
        "Investiga este tema durante 20 minutos y resume los hallazgos.",
        "Is vishay par 20 minute research karke summary do.",
        "What is the goal doing right now?",
        "What does this blocked state mean?",
        "Prioritize the failing login test first and keep the patch minimal",
        "Can you share progress?",
    ]

    for message in messages:
        assert is_natural_language_goal_request(message) is False



def test_tui_goal_routing_ignores_non_slash_natural_language_goal_phrasing() -> None:
    cases = [
        ("chinese timed research", "請研究這個主題 20分鐘，整理重點給我。", False),
        ("english timed research", "Research this for 30 minutes and come back with progress", False),
        (
            "english background work",
            "Keep working on this in the background for the next 30 minutes.",
            False,
        ),
        ("spanish timed research", "Investiga este tema durante 20 minutos y resume los hallazgos.", False),
        ("hindi timed research", "Is vishay par 20 minute research karke summary do.", False),
        ("active goal progress question", "What is the goal doing right now?", True),
        ("active goal blocked-state explanation", "What does this blocked state mean?", True),
        (
            "active goal steering instruction",
            "Prioritize the failing login test first and keep the patch minimal",
            True,
        ),
        ("active goal ambiguous follow-up", "Can you share progress?", True),
    ]

    for label, message, has_active_goal in cases:
        decision = resolve_goal_workflow_routing(
            text=message,
            has_pending_proposal=False,
            has_active_goal=has_active_goal,
        )
        assert decision.mode_command is None, label
        assert decision.goal_command is None, label
        assert decision.request_text == message, label
        assert decision.workflow_mode_requested is False, label
        assert decision.workflow_proposal_requested is False, label
        assert decision.natural_language_goal_requested is False, label
        assert decision.active_goal_follow_up_requested is False, label
        assert decision.pending_proposal_follow_up_requested is False, label
        assert decision.confirmation_requested is False, label
        assert decision.proposal_revision_requested is False, label
        assert decision.should_handle_goal_workflow_routing is False, label



def test_tui_goal_routing_only_handles_explicit_goal_workflow_commands_and_pending_follow_up() -> None:
    explicit_goal_proposal = resolve_goal_workflow_routing(
        text="/goal research this for 20 minutes",
        has_pending_proposal=False,
        has_active_goal=False,
    )
    assert explicit_goal_proposal.goal_command is not None
    assert explicit_goal_proposal.goal_command.action == "proposal"
    assert explicit_goal_proposal.goal_command.content == "research this for 20 minutes"
    assert explicit_goal_proposal.request_text == "research this for 20 minutes"
    assert explicit_goal_proposal.should_handle_goal_workflow_routing is True

    explicit_goal_lifecycle = resolve_goal_workflow_routing(
        text="/goal status",
        has_pending_proposal=False,
        has_active_goal=True,
    )
    assert explicit_goal_lifecycle.goal_command is not None
    assert explicit_goal_lifecycle.goal_command.action == "status"
    assert explicit_goal_lifecycle.request_text == "/goal status"
    assert explicit_goal_lifecycle.should_handle_goal_workflow_routing is True

    explicit_workflow = resolve_goal_workflow_routing(
        text="/workflow draft a multi-agent plan",
        has_pending_proposal=False,
        has_active_goal=False,
    )
    assert explicit_workflow.mode_command is not None
    assert explicit_workflow.mode_command.mode == "workflow"
    assert explicit_workflow.request_text == "draft a multi-agent plan"
    assert explicit_workflow.workflow_mode_requested is True
    assert explicit_workflow.workflow_proposal_requested is True
    assert explicit_workflow.should_handle_goal_workflow_routing is True

    explicit_chat = resolve_goal_workflow_routing(
        text="/chat what models are available?",
        has_pending_proposal=False,
        has_active_goal=True,
    )
    assert explicit_chat.mode_command is not None
    assert explicit_chat.mode_command.mode == "chat"
    assert explicit_chat.request_text == "what models are available?"
    assert explicit_chat.workflow_mode_requested is False
    assert explicit_chat.workflow_proposal_requested is False
    assert explicit_chat.should_handle_goal_workflow_routing is False

    pending_follow_up = resolve_goal_workflow_routing(
        text="go ahead",
        has_pending_proposal=True,
        has_active_goal=False,
    )
    assert pending_follow_up.mode_command is None
    assert pending_follow_up.goal_command is None
    assert pending_follow_up.request_text == "go ahead"
    assert pending_follow_up.pending_proposal_follow_up_requested is True
    assert pending_follow_up.should_handle_goal_workflow_routing is True

    for greeting in ("hi", "hello", "\u4f60\u597d", "\u55e8"):
        greeting_decision = resolve_goal_workflow_routing(
            text=greeting,
            has_pending_proposal=True,
            has_active_goal=False,
        )
        assert greeting_decision.goal_command is None, greeting
        assert greeting_decision.mode_command is None, greeting
        assert greeting_decision.pending_proposal_follow_up_requested is False, greeting
        assert greeting_decision.should_handle_goal_workflow_routing is False, greeting



@pytest.mark.asyncio
async def test_terminal_pending_goal_ambiguous_follow_up_clears_pending_and_returns_to_chat() -> None:
    from mochi.main import _handle_terminal_goal_input

    class _FakeSessionStore:
        def __init__(self) -> None:
            self.events: list[dict[str, object]] = [
                {
                    "type": "session_meta",
                    "event": "goal_state_updated",
                    "session_id": "goal-session",
                    "goal": {
                        "active_goal_id": None,
                        "active_goal_status": None,
                        "execution_mode": "single_agent",
                        "interaction_mode": "goal",
                        "execution_topology": "single_agent",
                        "bound_run_id": None,
                        "protocol_selection": None,
                        "selection_rationale": None,
                        "default_route": "goal",
                        "last_goal_summary": None,
                        "pending_proposal": {
                            "proposal_id": "proposal-1",
                            "objective": "Ship the task",
                            "execution_mode": "single_agent",
                            "interaction_mode": "goal",
                            "execution_topology": "single_agent",
                            "bound_run_id": None,
                            "protocol_selection": None,
                            "selection_rationale": None,
                            "protocol_id": None,
                            "models": ["ollama:base"],
                            "role_summary": "Primary agent continues the task directly.",
                            "runtime_mode": "Single-agent long-running execution",
                            "risk_note": None,
                            "revision_index": 0,
                            "assistant_explanation": "Old proposal copy.",
                        },
                    },
                    "timestamp": "2026-07-04T00:00:00+00:00",
                }
            ]

        async def save_event(self, session_id: str, event: dict[str, object]) -> None:
            assert session_id == "goal-session"
            self.events.append(dict(event))

        async def load_session(self, session_id: str) -> list[dict[str, object]]:
            assert session_id == "goal-session"
            return list(self.events)

    class _AmbiguousIntentInvoker:
        async def invoke(self, request) -> SimpleNamespace:  # noqa: ANN001
            assert '"user_follow_up": "random side question"' in request.message
            return SimpleNamespace(
                content='{"intent":"ambiguous","confidence":0.0,"rationale":"Side question is normal chat."}'
            )

    fake_store = _FakeSessionStore()
    result = await _handle_terminal_goal_input(
        text="random side question",
        session_id="goal-session",
        current_model="ollama:base",
        autonomy_mode=None,
        session_store=fake_store,
        ensure_runtime_service=lambda: None,
        intent_invoker=_AmbiguousIntentInvoker(),
    )

    assert result == {"handled": False, "chat_text": "random side question"}
    goal_updates = [
        event
        for event in fake_store.events
        if event.get("type") == "session_meta" and event.get("event") == "goal_state_updated"
    ]
    assert goal_updates[-1]["goal"]["default_route"] == "chat"
    assert goal_updates[-1]["goal"]["pending_proposal"] is None
    assistant_goal_cards = [
        event.get("goal_card")
        for event in fake_store.events
        if event.get("type") == "message" and event.get("role") == "assistant"
    ]
    assert assistant_goal_cards == []



@pytest.mark.asyncio
async def test_chat_tui_async_shows_active_goal_summary_on_start_and_session_query(
    monkeypatch,
    capsys,
) -> None:
    from mochi.main import _chat_tui_async

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001, ARG002
            self.closed = False

        async def initialize(self) -> None:
            return None

        async def close(self) -> None:
            self.closed = True

    class _FakeSessionStore:
        def __init__(self, sessions_dir) -> None:  # noqa: ANN001, ARG002
            self.by_session: dict[str, list[dict[str, object]]] = {
                "goal-session": [
                    {
                        "type": "session_meta",
                        "event": "goal_state_updated",
                        "session_id": "goal-session",
                        "goal": {
                            "active_goal_id": "goal-9",
                            "active_goal_status": "running",
                            "execution_mode": "single_agent",
                            "default_route": "goal",
                            "last_goal_summary": {
                                "goal_id": "goal-9",
                                "objective": "Finalize the release checklist",
                                "execution_mode": "single_agent",
                                "protocol_id": None,
                                "models": ["ollama:base"],
                                "role_summary": "Primary agent continues the task directly with the current chat tools.",
                                "runtime_mode": "Single-agent long-running execution",
                                "risk_note": None,
                                "status": "running",
                            },
                            "pending_proposal": None,
                        },
                        "timestamp": "2026-06-25T00:00:00+00:00",
                    }
                ]
            }

        async def save_event(self, session_id: str, event: dict[str, object]) -> None:
            self.by_session.setdefault(session_id, []).append(dict(event))

        async def load_session(self, session_id: str) -> list[dict[str, object]]:
            return list(self.by_session.get(session_id, []))

        async def delete_session(self, session_id: str) -> bool:
            self.by_session.pop(session_id, None)
            return True

    fake_store = _FakeSessionStore(None)
    inputs = iter(["/session", "/exit"])

    monkeypatch.setattr(
        "mochi.config.manager.load_config",
        lambda config_path=None: SimpleNamespace(  # noqa: ARG005
            model="ollama:base",
            sessions_dir="/tmp/mochi-sessions",
            security=SecurityConfig(),
        ),
    )
    monkeypatch.setattr("mochi.agents.engine.AgentEngine", lambda config: _FakeEngine(config))  # noqa: ARG005
    monkeypatch.setattr("mochi.sessions.store.SessionStore", lambda sessions_dir: fake_store)  # noqa: ARG005
    monkeypatch.setattr("mochi.main.console.input", lambda prompt="": next(inputs))  # noqa: ARG005

    await _chat_tui_async(
        model=None,
        config_path=None,
        session_id="goal-session",
        max_turns=2,
    )
    captured = capsys.readouterr().out

    assert captured.count("Active goal summary for this session.") >= 2
    assert "Active goal" in captured
    assert "Finalize the release checklist" in captured
