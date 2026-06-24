"""Project API route tests."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from mochi.api.server import create_app
from mochi.config.schema import MochiConfig
from mochi.projects.store import ProjectStore
from mochi.sessions.store import SessionStore


def _create_test_app(
    *,
    config: MochiConfig,
    project_store: ProjectStore | None = None,
    session_store: SessionStore | None = None,
):
    app = create_app()
    app.state.config_factory = lambda: config
    if project_store is not None:
        app.state.project_store = project_store
    if session_store is not None:
        app.state.session_store = session_store
    return app


def test_projects_crud_round_trip(tmp_path: Path) -> None:
    """Projects can be created, listed, fetched, updated, and deleted."""
    projects_path = tmp_path / "projects.json"
    config = MochiConfig.model_validate({"workspace_dir": str(tmp_path / "workspace")})
    app = _create_test_app(
        config=config,
        project_store=ProjectStore(projects_path),
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/projects",
            json={
                "name": "Alpha",
                "workspace_dir": str(tmp_path / "workspace-alpha"),
            },
        )
        assert create_response.status_code == 200
        created = create_response.json()
        assert created["type"] == "project"
        assert created["name"] == "Alpha"
        assert created["workspace_dir"] == str((tmp_path / "workspace-alpha").resolve())
        assert created["id"]

        list_response = client.get("/v1/projects")
        assert list_response.status_code == 200
        listed = list_response.json()["items"]
        assert len(listed) == 1
        assert listed[0] == {key: value for key, value in created.items() if key != "type"}
        assert list_response.json() == {
            "type": "projects",
            "items": listed,
        }

        get_response = client.get(f"/v1/projects/{created['id']}")
        assert get_response.status_code == 200
        assert get_response.json() == created

        update_response = client.patch(
            f"/v1/projects/{created['id']}",
            json={
                "name": "Alpha Renamed",
                "workspace_dir": str(tmp_path / "workspace-beta"),
            },
        )
        assert update_response.status_code == 200
        updated = update_response.json()
        assert updated["type"] == "project"
        assert updated["id"] == created["id"]
        assert updated["name"] == "Alpha Renamed"
        assert updated["workspace_dir"] == str((tmp_path / "workspace-beta").resolve())
        assert updated["created_at"] == created["created_at"]
        assert updated["updated_at"] != created["updated_at"]

        delete_response = client.delete(f"/v1/projects/{created['id']}")
        assert delete_response.status_code == 200
        assert delete_response.json() == {
            "type": "project",
            "project_id": created["id"],
            "deleted": True,
        }

        final_list = client.get("/v1/projects")
        assert final_list.status_code == 200
        assert final_list.json()["items"] == []


def test_deleting_project_unassigns_related_sessions(tmp_path: Path) -> None:
    """Deleting a project clears project assignment from existing sessions."""
    sessions_dir = tmp_path / "sessions"
    projects_path = tmp_path / "projects.json"
    config = MochiConfig.model_validate(
        {
            "workspace_dir": str(tmp_path / "workspace"),
            "sessions_dir": str(sessions_dir),
        }
    )
    app = _create_test_app(
        config=config,
        project_store=ProjectStore(projects_path),
        session_store=SessionStore(sessions_dir),
    )

    with TestClient(app) as client:
        created = client.post(
            "/v1/projects",
            json={
                "name": "Workspace One",
                "workspace_dir": str(tmp_path / "workspace-one"),
            },
        ).json()

        session_response = client.post(
            "/v1/sessions",
            json={"session_id": "alpha", "project_id": created["id"]},
        )
        assert session_response.status_code == 200

        delete_response = client.delete(f"/v1/projects/{created['id']}")
        assert delete_response.status_code == 200

        detail = client.get("/v1/sessions/alpha")
        assert detail.status_code == 200
        assert detail.json()["project_id"] is None
