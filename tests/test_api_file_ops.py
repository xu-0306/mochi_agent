"""File operation API tests."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from mochi.api.server import create_app
from mochi.config.schema import MochiConfig


def _create_test_app(config: MochiConfig):
    app = create_app()
    app.state.config_factory = lambda: config
    return app


def test_file_undo_restores_previous_content(tmp_path: Path) -> None:
    target = tmp_path / "demo.txt"
    target.write_text("after", encoding="utf-8")
    app = _create_test_app(
        MochiConfig.model_validate(
            {
                "workspace_dir": str(tmp_path),
                "sessions_dir": str(tmp_path / "sessions"),
            }
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/tools/file/undo",
            json={
                "file_path": str(target),
                "original_content": "before",
                "session_id": "missing-session",
                "action": "restore",
            },
        )

    assert response.status_code == 200
    assert target.read_text(encoding="utf-8") == "before"
    assert response.json()["action"] == "restore"


def test_file_undo_delete_removes_created_file(tmp_path: Path) -> None:
    target = tmp_path / "created.txt"
    target.write_text("new file", encoding="utf-8")
    app = _create_test_app(
        MochiConfig.model_validate(
            {
                "workspace_dir": str(tmp_path),
                "sessions_dir": str(tmp_path / "sessions"),
            }
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/tools/file/undo",
            json={
                "file_path": str(target),
                "original_content": None,
                "session_id": "missing-session",
                "action": "delete",
            },
        )

    assert response.status_code == 200
    assert not target.exists()
    assert response.json()["action"] == "delete"
