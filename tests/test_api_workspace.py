from __future__ import annotations

import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

from mochi.api.server import create_app
from mochi.config.schema import MochiConfig
from mochi.projects.store import ProjectStore
from mochi.sessions.store import SessionStore


def _create_test_app(*, workspace_dir: Path, sessions_dir: Path, projects_path: Path):
    app = create_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(workspace_dir),
            "sessions_dir": str(sessions_dir),
        }
    )
    app.state.session_store = SessionStore(sessions_dir)
    app.state.project_store = ProjectStore(projects_path)
    return app


def _run_git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def test_workspace_routes_resolve_session_workspace_and_enforce_scope(tmp_path: Path) -> None:
    default_workspace = tmp_path / "default-workspace"
    project_workspace = tmp_path / "project-workspace"
    sessions_dir = tmp_path / "sessions"
    default_workspace.mkdir(parents=True)
    project_workspace.mkdir(parents=True)
    (project_workspace / "notes.py").write_text("print('alpha')\n", encoding="utf-8")
    outside_file = tmp_path / "outside.txt"
    outside_file.write_text("forbidden\n", encoding="utf-8")

    app = _create_test_app(
        workspace_dir=default_workspace,
        sessions_dir=sessions_dir,
        projects_path=tmp_path / "projects.json",
    )

    with TestClient(app) as client:
        project = client.post(
            "/v1/projects",
            json={
                "name": "Alpha",
                "workspace_dir": str(project_workspace),
            },
        ).json()
        session_response = client.post(
            "/v1/sessions",
            json={"session_id": "session-alpha", "project_id": project["id"]},
        )
        assert session_response.status_code == 200

        tree = client.get("/v1/workspace/tree", params={"session_id": "session-alpha"})
        assert tree.status_code == 200
        tree_payload = tree.json()
        assert tree_payload["workspace_dir"] == str(project_workspace.resolve())
        assert tree_payload["project_id"] == project["id"]
        assert tree_payload["relative_path"] == "."
        assert tree_payload["items"][0]["relative_path"] == "notes.py"

        preview = client.get(
            "/v1/workspace/preview",
            params={"session_id": "session-alpha", "path": "notes.py"},
        )
        assert preview.status_code == 200
        preview_payload = preview.json()
        assert preview_payload["workspace_dir"] == str(project_workspace.resolve())
        assert preview_payload["relative_path"] == "notes.py"
        assert "print('alpha')" in preview_payload["text"]

        denied = client.get(
            "/v1/workspace/preview",
            params={"session_id": "session-alpha", "path": str(outside_file)},
        )
        assert denied.status_code == 403


def test_workspace_changes_and_diff_report_git_backed_workspace_state(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "repo"
    sessions_dir = tmp_path / "sessions"
    workspace_dir.mkdir(parents=True)
    target = workspace_dir / "src" / "app.py"
    target.parent.mkdir(parents=True)
    target.write_text("print('before')\n", encoding="utf-8")

    _run_git(workspace_dir, "init")
    _run_git(workspace_dir, "add", "src/app.py")
    _run_git(
        workspace_dir,
        "-c",
        "user.name=Mochi",
        "-c",
        "user.email=mochi@example.com",
        "commit",
        "-m",
        "init",
    )

    target.write_text("print('after')\n", encoding="utf-8")

    app = _create_test_app(
        workspace_dir=workspace_dir,
        sessions_dir=sessions_dir,
        projects_path=tmp_path / "projects.json",
    )

    with TestClient(app) as client:
        changes = client.get("/v1/workspace/changes", params={"session_id": "session-repo"})
        assert changes.status_code == 200
        changes_payload = changes.json()
        assert changes_payload["repo_root"] == str(workspace_dir.resolve())
        assert changes_payload["items"] == [
            {
                "path": str(target.resolve()),
                "relative_path": "src/app.py",
                "status": "modified",
                "staged": False,
                "added_lines": 1,
                "deleted_lines": 1,
                "diff_available": True,
            }
        ]

        diff = client.get(
            "/v1/workspace/diff",
            params={"session_id": "session-repo", "path": "src/app.py"},
        )
        assert diff.status_code == 200
        diff_payload = diff.json()
        assert diff_payload["relative_path"] == "src/app.py"
        assert diff_payload["status"] == "modified"
        assert "-print('before')" in diff_payload["diff"]
        assert "+print('after')" in diff_payload["diff"]


def test_workspace_patch_preview_uses_resolved_session_workspace(tmp_path: Path) -> None:
    default_workspace = tmp_path / "default-workspace"
    project_workspace = tmp_path / "project-workspace"
    sessions_dir = tmp_path / "sessions"
    default_workspace.mkdir(parents=True)
    project_workspace.mkdir(parents=True)
    target = project_workspace / "notes.py"
    target.write_text("print('alpha')\n", encoding="utf-8")

    app = _create_test_app(
        workspace_dir=default_workspace,
        sessions_dir=sessions_dir,
        projects_path=tmp_path / "projects.json",
    )

    with TestClient(app) as client:
        project = client.post(
            "/v1/projects",
            json={"name": "Alpha", "workspace_dir": str(project_workspace)},
        ).json()
        session_response = client.post(
            "/v1/sessions",
            json={"session_id": "session-alpha", "project_id": project["id"]},
        )
        assert session_response.status_code == 200

        preview = client.post(
            "/v1/workspace/patch/preview",
            json={
                "session_id": "session-alpha",
                "patch": "\n".join(
                    [
                        "*** Begin Patch",
                        "*** Update File: notes.py",
                        "@@",
                        "-print('alpha')",
                        "+print('beta')",
                        "*** End Patch",
                    ]
                ),
            },
        )
        assert preview.status_code == 200
        payload = preview.json()
        assert payload["workspace_dir"] == str(project_workspace.resolve())
        assert payload["change_count"] == 1
        assert payload["file_changes"][0]["relative_path"] == "notes.py"
        assert "-print('alpha')" in payload["diff"]
        assert "+print('beta')" in payload["diff"]


def test_workspace_patch_preview_returns_validation_errors(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    sessions_dir = tmp_path / "sessions"
    workspace_dir.mkdir(parents=True)
    (workspace_dir / "notes.py").write_text("print('alpha')\n", encoding="utf-8")

    app = _create_test_app(
        workspace_dir=workspace_dir,
        sessions_dir=sessions_dir,
        projects_path=tmp_path / "projects.json",
    )

    with TestClient(app) as client:
        invalid = client.post(
            "/v1/workspace/patch/preview",
            json={"patch": "*** Begin Patch\n*** End Patch"},
        )
        assert invalid.status_code == 200
        invalid_payload = invalid.json()
        assert invalid_payload["valid"] is False
        assert invalid_payload["file_changes"] == []
        assert invalid_payload["validation_errors"]

        denied = client.post(
            "/v1/workspace/patch/preview",
            json={
                "patch": "\n".join(
                    [
                        "*** Begin Patch",
                        f"*** Update File: {tmp_path.parent / 'outside.py'}",
                        "@@",
                        "-x",
                        "+y",
                        "*** End Patch",
                    ]
                ),
            },
        )
        assert denied.status_code == 200
        denied_payload = denied.json()
        assert denied_payload["valid"] is False
        assert denied_payload["validation_errors"]
