"""Chat TUI CLI tests."""

from __future__ import annotations

from types import SimpleNamespace

from typer.testing import CliRunner

from mochi.config.schema import SecurityConfig

runner = CliRunner()

def _active_goal_session_event() -> dict[str, object]:
    return {
        "type": "session_meta",
        "event": "goal_state_updated",
        "session_id": "goal-session",
        "goal": {
            "active_goal_id": "goal-1",
            "active_goal_status": "running",
            "execution_mode": "single_agent",
            "default_route": "goal",
            "last_goal_summary": {
                "goal_id": "goal-1",
                "objective": "Ship the task",
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


class _ActiveGoalSessionStore:
    def __init__(self, sessions_dir) -> None:  # noqa: ANN001, ARG002
        self.by_session: dict[str, list[dict[str, object]]] = {
            "goal-session": [_active_goal_session_event()]
        }

    async def save_event(self, session_id: str, event: dict[str, object]) -> None:
        self.by_session.setdefault(session_id, []).append(dict(event))

    async def load_session(self, session_id: str) -> list[dict[str, object]]:
        return list(self.by_session.get(session_id, []))

    async def delete_session(self, session_id: str) -> bool:
        self.by_session.pop(session_id, None)
        return True


def _patch_active_goal_tui_env(
    monkeypatch,
    *,
    fake_engine: object,
    fake_store: object,
    inputs: object,
    fake_runtime: object | None = None,
) -> None:
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
    if fake_runtime is not None:
        async def fake_runtime_service_factory(**kwargs):  # noqa: ANN003, ARG001
            return fake_runtime

        monkeypatch.setattr("mochi.main._create_tui_runtime_service", fake_runtime_service_factory)
    monkeypatch.setattr("mochi.main.console.input", lambda prompt="": next(inputs))  # noqa: ARG005
