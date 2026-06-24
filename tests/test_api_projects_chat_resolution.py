"""Project-aware chat workspace resolution tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from mochi.agents.events import FinalAnswerEvent
from mochi.api.server import create_app
from mochi.config.schema import MochiConfig
from mochi.projects.store import ProjectStore
from mochi.sessions.store import SessionStore


class _WorkspaceAwareFakeEngine:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        message: str,
        session_id: str | None = None,
        inference_overrides: dict[str, Any] | None = None,
        project_id: str | None = None,
        workspace_dir: str | None = None,
        selected_skill_ids: list[str] | None = None,
        attachments: list[Any] | None = None,
    ) -> AsyncIterator[object]:
        self.calls.append(
            {
                "message": message,
                "session_id": session_id,
                "inference_overrides": inference_overrides,
                "project_id": project_id,
                "workspace_dir": workspace_dir,
                "selected_skill_ids": selected_skill_ids,
                "attachments": attachments,
            }
        )
        yield FinalAnswerEvent(content="ok")


def _build_app(
    *,
    tmp_path: Path,
) -> tuple[object, _WorkspaceAwareFakeEngine]:
    app = create_app()
    engine = _WorkspaceAwareFakeEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path / "default-workspace"),
            "sessions_dir": str(tmp_path / "sessions"),
        }
    )
    app.state.session_store = SessionStore(tmp_path / "sessions")
    app.state.project_store = ProjectStore(tmp_path / "projects.json")
    return app, engine


def test_chat_route_resolves_workspace_from_project_id(tmp_path: Path) -> None:
    """Chat requests use the referenced project workspace."""
    app, engine = _build_app(tmp_path=tmp_path)

    with TestClient(app) as client:
        project = client.post(
            "/v1/projects",
            json={
                "name": "Alpha",
                "workspace_dir": str(tmp_path / "workspace-alpha"),
            },
        ).json()

        response = client.post(
            "/v1/chat",
            json={
                "message": "hello",
                "session_id": "session-alpha",
                "project_id": project["id"],
            },
        )

    assert response.status_code == 200
    assert engine.calls == [
        {
            "message": "hello",
            "session_id": "session-alpha",
            "inference_overrides": {},
            "project_id": project["id"],
            "workspace_dir": str((tmp_path / "workspace-alpha").resolve()),
            "selected_skill_ids": None,
            "attachments": [],
        }
    ]


def test_chat_route_uses_session_project_when_request_omits_project_id(tmp_path: Path) -> None:
    """Existing session assignment defines workspace when project_id is omitted."""
    app, engine = _build_app(tmp_path=tmp_path)

    with TestClient(app) as client:
        project = client.post(
            "/v1/projects",
            json={
                "name": "Assigned",
                "workspace_dir": str(tmp_path / "workspace-assigned"),
            },
        ).json()
        session_response = client.post(
            "/v1/sessions",
            json={"session_id": "session-one", "project_id": project["id"]},
        )
        assert session_response.status_code == 200

        response = client.post(
            "/v1/chat",
            json={
                "message": "hello again",
                "session_id": "session-one",
            },
        )

    assert response.status_code == 200
    assert engine.calls == [
        {
            "message": "hello again",
            "session_id": "session-one",
            "inference_overrides": {},
            "project_id": project["id"],
            "workspace_dir": str((tmp_path / "workspace-assigned").resolve()),
            "selected_skill_ids": None,
            "attachments": [],
        }
    ]


def test_chat_route_falls_back_to_default_workspace_for_unassigned_session(tmp_path: Path) -> None:
    """Unassigned sessions still use the global fallback workspace."""
    app, engine = _build_app(tmp_path=tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat",
            json={
                "message": "plain",
                "session_id": "session-plain",
            },
        )

    assert response.status_code == 200
    assert engine.calls == [
        {
            "message": "plain",
            "session_id": "session-plain",
            "inference_overrides": {},
            "project_id": None,
            "workspace_dir": str((tmp_path / "default-workspace").resolve()),
            "selected_skill_ids": None,
            "attachments": [],
        }
    ]


def test_chat_route_passes_selected_skill_ids_to_engine(tmp_path: Path) -> None:
    """Chat requests forward explicit selected skill IDs to the engine."""
    app, engine = _build_app(tmp_path=tmp_path)

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat",
            json={
                "message": "plain",
                "session_id": "session-plain",
                "selected_skill_ids": ["skill-alpha", "skill-beta"],
            },
        )

    assert response.status_code == 200
    assert engine.calls == [
        {
            "message": "plain",
            "session_id": "session-plain",
            "inference_overrides": {},
            "project_id": None,
            "workspace_dir": str((tmp_path / "default-workspace").resolve()),
            "selected_skill_ids": ["skill-alpha", "skill-beta"],
            "attachments": [],
        }
    ]
