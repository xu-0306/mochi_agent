"""Chat TUI CLI tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from mochi.agents.events import FinalAnswerEvent
from tests.integration.cli.tui._support import (
    _ActiveGoalSessionStore,
    _patch_active_goal_tui_env,
)

runner = CliRunner()


@pytest.mark.asyncio
async def test_chat_tui_async_active_goal_question_uses_backend_decision_then_chat(
    monkeypatch,
    capsys,
) -> None:
    from mochi.main import _chat_tui_async

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001, ARG002
            self.calls: list[tuple[str, str | None]] = []
            self.closed = False

        async def initialize(self) -> None:
            return None

        async def chat(
            self,
            message: str,
            session_id: str | None = None,
        ) -> AsyncIterator[object]:
            self.calls.append((message, session_id))
            yield FinalAnswerEvent(content="plain chat reply")

        async def close(self) -> None:
            self.closed = True

    class _FakeRuntimeService:
        def __init__(self) -> None:
            self.decision_calls: list[tuple[str, str]] = []
            self.health_calls: list[str] = []
            self.guidance_calls: list[tuple[str, str]] = []
            self.resume_calls: list[tuple[str, str | None, str | None]] = []
            self.refresh_calls: list[tuple[str, str | None]] = []

        async def decide_active_goal_turn(self, goal_id: str, payload) -> SimpleNamespace:  # noqa: ANN001
            self.decision_calls.append((goal_id, payload.message))
            return SimpleNamespace(kind="answer_question", requires_confirmation=False)

        async def get_goal_health(self, goal_id: str) -> dict[str, object] | None:
            self.health_calls.append(goal_id)
            return None

        async def append_agent_run_guidance(self, run_id: str, payload) -> dict[str, object] | None:  # noqa: ANN001
            self.guidance_calls.append((run_id, payload.guidance))
            return {"run_id": run_id}

        async def resume_goal(
            self,
            goal_id: str,
            *,
            strategy: str | None = None,
            guidance_message: str | None = None,
        ) -> dict[str, object] | None:
            self.resume_calls.append((goal_id, strategy, guidance_message))
            return None

        async def refresh_goal(
            self,
            goal_id: str,
            *,
            strategy: str | None = None,
        ) -> dict[str, object] | None:
            self.refresh_calls.append((goal_id, strategy))
            return None

        async def close(self) -> None:
            return None

    fake_store = _ActiveGoalSessionStore(None)
    fake_engine = _FakeEngine(None)
    fake_runtime = _FakeRuntimeService()
    inputs = iter(["Can you share progress?", "/exit"])

    _patch_active_goal_tui_env(
        monkeypatch,
        fake_engine=fake_engine,
        fake_store=fake_store,
        fake_runtime=fake_runtime,
        inputs=inputs,
    )

    await _chat_tui_async(
        model=None,
        config_path=None,
        session_id="goal-session",
        max_turns=2,
    )
    captured = capsys.readouterr().out

    assert fake_engine.calls == [
        ("Can you share progress?", "goal-session"),
    ]
    assert fake_runtime.decision_calls == [("goal-1", "Can you share progress?")]
    assert fake_runtime.health_calls == []
    assert fake_runtime.guidance_calls == []
    assert fake_runtime.resume_calls == []
    assert fake_runtime.refresh_calls == []
    assert captured.count("plain chat reply") == 1



@pytest.mark.asyncio
async def test_chat_tui_async_active_goal_steering_forwards_guidance_without_chat(
    monkeypatch,
    capsys,
) -> None:
    from mochi.main import _chat_tui_async

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001, ARG002
            self.calls: list[tuple[str, str | None]] = []
            self.closed = False

        async def initialize(self) -> None:
            return None

        async def chat(
            self,
            message: str,
            session_id: str | None = None,
        ) -> AsyncIterator[object]:
            self.calls.append((message, session_id))
            yield FinalAnswerEvent(content="unexpected chat")

        async def close(self) -> None:
            self.closed = True

    class _FakeRuntimeService:
        def __init__(self) -> None:
            self.decision_calls: list[tuple[str, str]] = []
            self.health_calls: list[str] = []
            self.guidance_calls: list[tuple[str, str]] = []
            self.get_goal_calls: list[str] = []

        async def decide_active_goal_turn(self, goal_id: str, payload) -> SimpleNamespace:  # noqa: ANN001
            self.decision_calls.append((goal_id, payload.message))
            return SimpleNamespace(kind="steer", requires_confirmation=False)

        async def get_goal_health(self, goal_id: str) -> dict[str, object] | None:
            self.health_calls.append(goal_id)
            return {
                "status": "running",
                "recommended_next_action": {
                    "action": "monitor",
                    "summary": "The goal can keep running with additional guidance.",
                },
                "linked_agent_run": {
                    "run_id": "run-1",
                    "status": "running",
                },
            }

        async def append_agent_run_guidance(self, run_id: str, payload) -> dict[str, object] | None:  # noqa: ANN001
            self.guidance_calls.append((run_id, payload.guidance))
            return {"run_id": run_id}

        async def get_goal(self, goal_id: str) -> dict[str, object] | None:
            self.get_goal_calls.append(goal_id)
            return {
                "goal_id": goal_id,
                "objective": "Ship the task",
                "execution_mode": "single_agent",
                "interaction_mode": "goal",
                "execution_topology": "single_agent",
                "status": "running",
                "current_attempt_id": "attempt-1",
                "attempts": [
                    {
                        "attempt_id": "attempt-1",
                        "agent_run_id": "run-1",
                    }
                ],
            }

        async def close(self) -> None:
            return None

    fake_store = _ActiveGoalSessionStore(None)
    fake_engine = _FakeEngine(None)
    fake_runtime = _FakeRuntimeService()
    inputs = iter(["Please focus on tests first", "/exit"])

    _patch_active_goal_tui_env(
        monkeypatch,
        fake_engine=fake_engine,
        fake_store=fake_store,
        fake_runtime=fake_runtime,
        inputs=inputs,
    )

    await _chat_tui_async(
        model=None,
        config_path=None,
        session_id="goal-session",
        max_turns=2,
    )
    captured = capsys.readouterr().out

    assert fake_engine.calls == []
    assert fake_runtime.decision_calls == [("goal-1", "Please focus on tests first")]
    assert fake_runtime.health_calls == ["goal-1"]
    assert fake_runtime.guidance_calls == [("run-1", "Please focus on tests first")]
    assert fake_runtime.get_goal_calls == ["goal-1"]
    assert "Forwarded your guidance to the active goal." in captured



@pytest.mark.asyncio
async def test_chat_tui_async_active_goal_clarify_confirmation_stays_conversational(
    monkeypatch,
    capsys,
) -> None:
    from mochi.main import _chat_tui_async

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001, ARG002
            self.calls: list[tuple[str, str | None]] = []
            self.closed = False

        async def initialize(self) -> None:
            return None

        async def chat(
            self,
            message: str,
            session_id: str | None = None,
        ) -> AsyncIterator[object]:
            self.calls.append((message, session_id))
            yield FinalAnswerEvent(content="clarify reply")

        async def close(self) -> None:
            self.closed = True

    class _FakeRuntimeService:
        def __init__(self) -> None:
            self.decision_calls: list[tuple[str, str]] = []
            self.guidance_calls: list[tuple[str, str]] = []

        async def decide_active_goal_turn(self, goal_id: str, payload) -> SimpleNamespace:  # noqa: ANN001
            self.decision_calls.append((goal_id, payload.message))
            return SimpleNamespace(kind="clarify", requires_confirmation=True)

        async def append_agent_run_guidance(self, run_id: str, payload) -> dict[str, object] | None:  # noqa: ANN001
            self.guidance_calls.append((run_id, payload.guidance))
            return {"run_id": run_id}

        async def close(self) -> None:
            return None

    fake_store = _ActiveGoalSessionStore(None)
    fake_engine = _FakeEngine(None)
    fake_runtime = _FakeRuntimeService()
    inputs = iter(["maybe focus on benchmarks", "/exit"])

    _patch_active_goal_tui_env(
        monkeypatch,
        fake_engine=fake_engine,
        fake_store=fake_store,
        fake_runtime=fake_runtime,
        inputs=inputs,
    )

    await _chat_tui_async(
        model=None,
        config_path=None,
        session_id="goal-session",
        max_turns=2,
    )
    captured = capsys.readouterr().out

    assert fake_runtime.decision_calls == [("goal-1", "maybe focus on benchmarks")]
    assert fake_runtime.guidance_calls == []
    assert fake_engine.calls == [("maybe focus on benchmarks", "goal-session")]
    assert "clarify reply" in captured



@pytest.mark.asyncio
async def test_chat_tui_async_active_goal_manual_resolution_required_stays_non_mutating(
    monkeypatch,
    capsys,
) -> None:
    from mochi.main import _chat_tui_async

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001, ARG002
            self.calls: list[tuple[str, str | None]] = []
            self.closed = False

        async def initialize(self) -> None:
            return None

        async def chat(
            self,
            message: str,
            session_id: str | None = None,
        ) -> AsyncIterator[object]:
            self.calls.append((message, session_id))
            yield FinalAnswerEvent(content="unexpected chat")

        async def close(self) -> None:
            self.closed = True

    class _FakeRuntimeService:
        def __init__(self) -> None:
            self.decision_calls: list[tuple[str, str]] = []
            self.health_calls: list[str] = []
            self.guidance_calls: list[tuple[str, str]] = []
            self.resume_calls: list[tuple[str, str | None, str | None]] = []
            self.refresh_calls: list[tuple[str, str | None]] = []
            self.get_goal_calls: list[str] = []

        async def decide_active_goal_turn(self, goal_id: str, payload) -> SimpleNamespace:  # noqa: ANN001
            self.decision_calls.append((goal_id, payload.message))
            return SimpleNamespace(kind="steer", requires_confirmation=False)

        async def get_goal_health(self, goal_id: str) -> dict[str, object] | None:
            self.health_calls.append(goal_id)
            return {
                "status": "waiting_approval",
                "recommended_next_action": {
                    "action": "resolve_approval",
                    "summary": "The active goal is waiting for operator approval.",
                    "approval_ids": ["appr-1"],
                },
            }

        async def append_agent_run_guidance(self, run_id: str, payload) -> dict[str, object] | None:  # noqa: ANN001
            self.guidance_calls.append((run_id, payload.guidance))
            return {"run_id": run_id}

        async def resume_goal(
            self,
            goal_id: str,
            *,
            strategy: str | None = None,
            guidance_message: str | None = None,
        ) -> dict[str, object] | None:
            self.resume_calls.append((goal_id, strategy, guidance_message))
            return None

        async def refresh_goal(
            self,
            goal_id: str,
            *,
            strategy: str | None = None,
        ) -> dict[str, object] | None:
            self.refresh_calls.append((goal_id, strategy))
            return None

        async def get_goal(self, goal_id: str) -> dict[str, object] | None:
            self.get_goal_calls.append(goal_id)
            return None

        async def close(self) -> None:
            return None

    fake_store = _ActiveGoalSessionStore(None)
    fake_engine = _FakeEngine(None)
    fake_runtime = _FakeRuntimeService()
    inputs = iter(["please keep going after approval", "/exit"])

    _patch_active_goal_tui_env(
        monkeypatch,
        fake_engine=fake_engine,
        fake_store=fake_store,
        fake_runtime=fake_runtime,
        inputs=inputs,
    )

    await _chat_tui_async(
        model=None,
        config_path=None,
        session_id="goal-session",
        max_turns=2,
    )
    captured = capsys.readouterr().out
    normalized_captured = " ".join(captured.split())

    assert fake_engine.calls == []
    assert fake_runtime.decision_calls == [("goal-1", "please keep going after approval")]
    assert fake_runtime.health_calls == ["goal-1"]
    assert fake_runtime.guidance_calls == []
    assert fake_runtime.resume_calls == []
    assert fake_runtime.refresh_calls == []
    assert fake_runtime.get_goal_calls == []
    assert "Review the pending approval" in normalized_captured



@pytest.mark.asyncio
async def test_chat_tui_async_active_goal_blocked_stays_non_mutating(
    monkeypatch,
    capsys,
) -> None:
    from mochi.main import _chat_tui_async

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001, ARG002
            self.calls: list[tuple[str, str | None]] = []
            self.closed = False

        async def initialize(self) -> None:
            return None

        async def chat(
            self,
            message: str,
            session_id: str | None = None,
        ) -> AsyncIterator[object]:
            self.calls.append((message, session_id))
            yield FinalAnswerEvent(content="unexpected chat")

        async def close(self) -> None:
            self.closed = True

    class _FakeRuntimeService:
        def __init__(self) -> None:
            self.decision_calls: list[tuple[str, str]] = []
            self.health_calls: list[str] = []
            self.guidance_calls: list[tuple[str, str]] = []
            self.resume_calls: list[tuple[str, str | None, str | None]] = []
            self.refresh_calls: list[tuple[str, str | None]] = []
            self.get_goal_calls: list[str] = []

        async def decide_active_goal_turn(self, goal_id: str, payload) -> SimpleNamespace:  # noqa: ANN001
            self.decision_calls.append((goal_id, payload.message))
            return SimpleNamespace(kind="steer", requires_confirmation=False)

        async def get_goal_health(self, goal_id: str) -> dict[str, object] | None:
            self.health_calls.append(goal_id)
            return {
                "status": "paused",
                "recommended_next_action": {
                    "action": "inspect_runtime_budget",
                    "summary": "The goal hit a runtime budget limit.",
                },
            }

        async def append_agent_run_guidance(self, run_id: str, payload) -> dict[str, object] | None:  # noqa: ANN001
            self.guidance_calls.append((run_id, payload.guidance))
            return {"run_id": run_id}

        async def resume_goal(
            self,
            goal_id: str,
            *,
            strategy: str | None = None,
            guidance_message: str | None = None,
        ) -> dict[str, object] | None:
            self.resume_calls.append((goal_id, strategy, guidance_message))
            return None

        async def refresh_goal(
            self,
            goal_id: str,
            *,
            strategy: str | None = None,
        ) -> dict[str, object] | None:
            self.refresh_calls.append((goal_id, strategy))
            return None

        async def get_goal(self, goal_id: str) -> dict[str, object] | None:
            self.get_goal_calls.append(goal_id)
            return None

        async def close(self) -> None:
            return None

    fake_store = _ActiveGoalSessionStore(None)
    fake_engine = _FakeEngine(None)
    fake_runtime = _FakeRuntimeService()
    inputs = iter(["please continue anyway", "/exit"])

    _patch_active_goal_tui_env(
        monkeypatch,
        fake_engine=fake_engine,
        fake_store=fake_store,
        fake_runtime=fake_runtime,
        inputs=inputs,
    )

    await _chat_tui_async(
        model=None,
        config_path=None,
        session_id="goal-session",
        max_turns=2,
    )
    captured = capsys.readouterr().out
    normalized_captured = " ".join(captured.split())

    assert fake_engine.calls == []
    assert fake_runtime.decision_calls == [("goal-1", "please continue anyway")]
    assert fake_runtime.health_calls == ["goal-1"]
    assert fake_runtime.guidance_calls == []
    assert fake_runtime.resume_calls == []
    assert fake_runtime.refresh_calls == []
    assert fake_runtime.get_goal_calls == []
    assert "The goal hit a runtime budget limit." in normalized_captured
    assert "Adjust the goal from the Goal Console" in normalized_captured



@pytest.mark.asyncio
async def test_chat_tui_async_active_goal_replan_restarts_without_chat(
    monkeypatch,
    capsys,
) -> None:
    from mochi.main import _chat_tui_async

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001, ARG002
            self.calls: list[tuple[str, str | None]] = []
            self.closed = False

        async def initialize(self) -> None:
            return None

        async def chat(
            self,
            message: str,
            session_id: str | None = None,
        ) -> AsyncIterator[object]:
            self.calls.append((message, session_id))
            yield FinalAnswerEvent(content="unexpected chat")

        async def close(self) -> None:
            self.closed = True

    class _FakeRuntimeService:
        def __init__(self) -> None:
            self.decision_calls: list[tuple[str, str]] = []
            self.health_calls: list[str] = []
            self.resume_calls: list[tuple[str, str | None, str | None]] = []
            self.get_goal_calls: list[str] = []

        async def decide_active_goal_turn(self, goal_id: str, payload) -> SimpleNamespace:  # noqa: ANN001
            self.decision_calls.append((goal_id, payload.message))
            return SimpleNamespace(kind="replan", requires_confirmation=False)

        async def get_goal_health(self, goal_id: str) -> dict[str, object] | None:
            self.health_calls.append(goal_id)
            return {
                "status": "paused",
                "recommended_next_action": {
                    "action": "resume_goal",
                    "summary": "The goal should reopen with a new plan.",
                },
                "current_attempt": {
                    "agent_run_id": "run-1",
                },
            }

        async def resume_goal(
            self,
            goal_id: str,
            *,
            strategy: str | None = None,
            guidance_message: str | None = None,
        ) -> dict[str, object] | None:
            self.resume_calls.append((goal_id, strategy, guidance_message))
            return {
                "goal_id": goal_id,
                "objective": "Ship the task",
                "execution_mode": "single_agent",
                "interaction_mode": "goal",
                "execution_topology": "single_agent",
                "status": "running",
                "current_attempt_id": "attempt-2",
                "attempts": [
                    {
                        "attempt_id": "attempt-2",
                        "agent_run_id": "run-2",
                    }
                ],
            }

        async def get_goal(self, goal_id: str) -> dict[str, object] | None:
            self.get_goal_calls.append(goal_id)
            return {
                "goal_id": goal_id,
                "objective": "Ship the task",
                "execution_mode": "single_agent",
                "interaction_mode": "goal",
                "execution_topology": "single_agent",
                "status": "running",
                "current_attempt_id": "attempt-2",
                "attempts": [
                    {
                        "attempt_id": "attempt-2",
                        "agent_run_id": "run-2",
                    }
                ],
            }

        async def close(self) -> None:
            return None

    fake_store = _ActiveGoalSessionStore(None)
    fake_engine = _FakeEngine(None)
    fake_runtime = _FakeRuntimeService()
    inputs = iter(["replan this and take a different approach", "/exit"])

    _patch_active_goal_tui_env(
        monkeypatch,
        fake_engine=fake_engine,
        fake_store=fake_store,
        fake_runtime=fake_runtime,
        inputs=inputs,
    )

    await _chat_tui_async(
        model=None,
        config_path=None,
        session_id="goal-session",
        max_turns=2,
    )
    captured = capsys.readouterr().out

    assert fake_engine.calls == []
    assert fake_runtime.decision_calls == [("goal-1", "replan this and take a different approach")]
    assert fake_runtime.health_calls == ["goal-1"]
    assert fake_runtime.resume_calls == [
        ("goal-1", "restart_attempt", "replan this and take a different approach")
    ]
    assert fake_runtime.get_goal_calls == ["goal-1"]
    assert "Restarted the goal from the latest available recovery context" in captured



@pytest.mark.asyncio
async def test_chat_tui_async_active_goal_decision_failure_falls_back_to_chat(
    monkeypatch,
    capsys,
) -> None:
    from mochi.main import _chat_tui_async

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001, ARG002
            self.calls: list[tuple[str, str | None]] = []
            self.closed = False

        async def initialize(self) -> None:
            return None

        async def chat(
            self,
            message: str,
            session_id: str | None = None,
        ) -> AsyncIterator[object]:
            self.calls.append((message, session_id))
            yield FinalAnswerEvent(content="fallback chat")

        async def close(self) -> None:
            self.closed = True

    class _FakeRuntimeService:
        def __init__(self) -> None:
            self.decision_calls: list[tuple[str, str]] = []

        async def decide_active_goal_turn(self, goal_id: str, payload) -> SimpleNamespace:  # noqa: ANN001
            self.decision_calls.append((goal_id, payload.message))
            raise RuntimeError("decision unavailable")

        async def close(self) -> None:
            return None

    fake_store = _ActiveGoalSessionStore(None)
    fake_engine = _FakeEngine(None)
    fake_runtime = _FakeRuntimeService()
    inputs = iter(["please focus on tests", "/exit"])

    _patch_active_goal_tui_env(
        monkeypatch,
        fake_engine=fake_engine,
        fake_store=fake_store,
        fake_runtime=fake_runtime,
        inputs=inputs,
    )

    await _chat_tui_async(
        model=None,
        config_path=None,
        session_id="goal-session",
        max_turns=2,
    )
    captured = capsys.readouterr().out

    assert fake_runtime.decision_calls == [("goal-1", "please focus on tests")]
    assert fake_engine.calls == [("please focus on tests", "goal-session")]
    assert "fallback chat" in captured



@pytest.mark.asyncio
async def test_chat_tui_async_active_goal_health_failure_falls_back_to_chat(
    monkeypatch,
    capsys,
) -> None:
    from mochi.main import _chat_tui_async

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001, ARG002
            self.calls: list[tuple[str, str | None]] = []
            self.closed = False

        async def initialize(self) -> None:
            return None

        async def chat(
            self,
            message: str,
            session_id: str | None = None,
        ) -> AsyncIterator[object]:
            self.calls.append((message, session_id))
            yield FinalAnswerEvent(content="fallback chat")

        async def close(self) -> None:
            self.closed = True

    class _FakeRuntimeService:
        def __init__(self) -> None:
            self.decision_calls: list[tuple[str, str]] = []
            self.health_calls: list[str] = []
            self.guidance_calls: list[tuple[str, str]] = []
            self.resume_calls: list[tuple[str, str | None, str | None]] = []
            self.refresh_calls: list[tuple[str, str | None]] = []
            self.get_goal_calls: list[str] = []

        async def decide_active_goal_turn(self, goal_id: str, payload) -> SimpleNamespace:  # noqa: ANN001
            self.decision_calls.append((goal_id, payload.message))
            return SimpleNamespace(kind="steer", requires_confirmation=False)

        async def get_goal_health(self, goal_id: str) -> dict[str, object] | None:
            self.health_calls.append(goal_id)
            raise RuntimeError("health unavailable")

        async def append_agent_run_guidance(self, run_id: str, payload) -> dict[str, object] | None:  # noqa: ANN001
            self.guidance_calls.append((run_id, payload.guidance))
            return {"run_id": run_id}

        async def resume_goal(
            self,
            goal_id: str,
            *,
            strategy: str | None = None,
            guidance_message: str | None = None,
        ) -> dict[str, object] | None:
            self.resume_calls.append((goal_id, strategy, guidance_message))
            return None

        async def refresh_goal(
            self,
            goal_id: str,
            *,
            strategy: str | None = None,
        ) -> dict[str, object] | None:
            self.refresh_calls.append((goal_id, strategy))
            return None

        async def get_goal(self, goal_id: str) -> dict[str, object] | None:
            self.get_goal_calls.append(goal_id)
            return None

        async def close(self) -> None:
            return None

    fake_store = _ActiveGoalSessionStore(None)
    fake_engine = _FakeEngine(None)
    fake_runtime = _FakeRuntimeService()
    inputs = iter(["please focus on tests", "/exit"])

    _patch_active_goal_tui_env(
        monkeypatch,
        fake_engine=fake_engine,
        fake_store=fake_store,
        fake_runtime=fake_runtime,
        inputs=inputs,
    )

    await _chat_tui_async(
        model=None,
        config_path=None,
        session_id="goal-session",
        max_turns=2,
    )
    captured = capsys.readouterr().out

    assert fake_runtime.decision_calls == [("goal-1", "please focus on tests")]
    assert fake_runtime.health_calls == ["goal-1"]
    assert fake_runtime.guidance_calls == []
    assert fake_runtime.resume_calls == []
    assert fake_runtime.refresh_calls == []
    assert fake_runtime.get_goal_calls == []
    assert fake_engine.calls == [("please focus on tests", "goal-session")]
    assert "fallback chat" in captured



@pytest.mark.asyncio
async def test_chat_tui_async_active_goal_runtime_service_creation_failure_falls_back_to_chat(
    monkeypatch,
    capsys,
) -> None:
    from mochi.main import _chat_tui_async

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001, ARG002
            self.calls: list[tuple[str, str | None]] = []
            self.closed = False

        async def initialize(self) -> None:
            return None

        async def chat(
            self,
            message: str,
            session_id: str | None = None,
        ) -> AsyncIterator[object]:
            self.calls.append((message, session_id))
            yield FinalAnswerEvent(content="fallback chat")

        async def close(self) -> None:
            self.closed = True

    fake_store = _ActiveGoalSessionStore(None)
    fake_engine = _FakeEngine(None)
    inputs = iter(["please focus on tests", "/exit"])

    _patch_active_goal_tui_env(
        monkeypatch,
        fake_engine=fake_engine,
        fake_store=fake_store,
        inputs=inputs,
    )

    async def fake_runtime_service_factory(**kwargs):  # noqa: ANN003, ARG001
        raise RuntimeError("runtime unavailable")

    monkeypatch.setattr("mochi.main._create_tui_runtime_service", fake_runtime_service_factory)

    await _chat_tui_async(
        model=None,
        config_path=None,
        session_id="goal-session",
        max_turns=2,
    )
    captured = capsys.readouterr().out

    assert fake_engine.calls == [("please focus on tests", "goal-session")]
    assert "fallback chat" in captured
