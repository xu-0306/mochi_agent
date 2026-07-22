from __future__ import annotations

import subprocess
from pathlib import Path

from fastapi.testclient import TestClient

from mochi.api.server import create_app
from mochi.api.routes.workspace import _parse_git_status_porcelain_v2
from mochi.config.schema import MochiConfig
from mochi.projects.store import ProjectStore
from mochi.sessions.store import SessionStore


def _create_test_app(
    *,
    workspace_dir: Path,
    sessions_dir: Path,
    projects_path: Path,
    change_contract_mode: str = "observe",
    file_read_scope: str = "workspace",
    file_write_scope: str = "workspace",
):
    app = create_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(workspace_dir),
            "sessions_dir": str(sessions_dir),
            "security": {
                "change_contract_mode": change_contract_mode,
                "file_read_scope": file_read_scope,
                "file_write_scope": file_write_scope,
            },
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


def test_workspace_routes_resolve_session_workspace_and_enforce_scope(
    tmp_path: Path,
) -> None:
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


def test_workspace_preview_honors_explicit_any_read_scope(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "external.txt"
    external.write_text("allowed read", encoding="utf-8")
    app = _create_test_app(
        workspace_dir=workspace,
        sessions_dir=tmp_path / "sessions",
        projects_path=tmp_path / "projects.json",
        file_read_scope="any",
        file_write_scope="workspace",
    )

    with TestClient(app) as client:
        response = client.get(
            "/v1/workspace/preview",
            params={"path": str(external)},
        )

    assert response.status_code == 200
    assert response.json()["text"] == "allowed read"


def test_workspace_changes_and_diff_report_git_backed_workspace_state(
    tmp_path: Path,
) -> None:
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
        changes = client.get(
            "/v1/workspace/changes", params={"session_id": "session-repo"}
        )
        assert changes.status_code == 200
        changes_payload = changes.json()
        assert changes_payload["repo_root"] == str(workspace_dir.resolve())
        assert len(changes_payload["items"]) == 1
        change = changes_payload["items"][0]
        assert change == {
            **change,
            "path": str(target.resolve()),
            "relative_path": "src/app.py",
            "status": "modified",
            "staged": False,
            "added_lines": 1,
            "deleted_lines": 1,
            "diff_available": True,
            "binary": False,
            "encoding": "utf-8",
            "eof_newline": True,
            "content_unavailable_reason": None,
        }

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


def test_workspace_git_fidelity_handles_rename_binary_non_utf8_and_eof(
    tmp_path: Path,
) -> None:
    workspace_dir = tmp_path / "repo"
    workspace_dir.mkdir(parents=True)
    literal_arrow = workspace_dir / "literal → name.txt"
    binary_file = workspace_dir / "asset.bin"
    non_utf8_file = workspace_dir / "legacy.txt"
    no_eof_file = workspace_dir / "no-eof.txt"
    untracked_file = workspace_dir / "new file.txt"
    literal_arrow.write_bytes(b"before\r\n")
    binary_file.write_bytes(b"\x00before\xff")
    non_utf8_file.write_bytes(b"\xffbefore\n")
    no_eof_file.write_bytes(b"before")

    _run_git(workspace_dir, "init")
    _run_git(workspace_dir, "add", ".")
    _run_git(
        workspace_dir,
        "-c",
        "user.name=Mochi",
        "-c",
        "user.email=mochi@example.com",
        "commit",
        "-m",
        "fixture",
    )

    renamed = workspace_dir / "renamed file.txt"
    _run_git(workspace_dir, "mv", literal_arrow.name, renamed.name)
    binary_file.write_bytes(b"\x00after\xfe")
    non_utf8_file.write_bytes(b"\xfeafter\n")
    no_eof_file.write_bytes(b"after")
    untracked_file.write_bytes(b"new\n")
    _run_git(workspace_dir, "update-index", "--chmod=+x", "no-eof.txt")

    app = _create_test_app(
        workspace_dir=workspace_dir,
        sessions_dir=tmp_path / "sessions",
        projects_path=tmp_path / "projects.json",
    )
    with TestClient(app) as client:
        changes = client.get("/v1/workspace/changes").json()["items"]
        by_path = {item["relative_path"]: item for item in changes}
        rename_diff = client.get(
            "/v1/workspace/diff", params={"path": renamed.name}
        ).json()
        binary_diff = client.get(
            "/v1/workspace/diff", params={"path": binary_file.name}
        ).json()
        non_utf8_diff = client.get(
            "/v1/workspace/diff", params={"path": non_utf8_file.name}
        ).json()
        eof_diff = client.get(
            "/v1/workspace/diff", params={"path": no_eof_file.name}
        ).json()
        untracked_diff = client.get(
            "/v1/workspace/diff", params={"path": untracked_file.name}
        ).json()

    assert by_path[renamed.name]["status"] == "renamed"
    assert by_path[renamed.name]["rename_source"] == literal_arrow.name
    assert rename_diff["rename_source"] == literal_arrow.name
    assert rename_diff["newline_style"] == "crlf"
    assert "rename from literal → name.txt" in rename_diff["diff"]
    assert "rename to renamed file.txt" in rename_diff["diff"]

    assert binary_diff["binary"] is True
    assert binary_diff["encoding"] == "binary"
    assert binary_diff["diff"] is None
    assert binary_diff["original_content"] is None
    assert binary_diff["new_content"] is None
    assert binary_diff["content_unavailable_reason"] == "binary"

    assert non_utf8_diff["binary"] is False
    assert non_utf8_diff["encoding"] == "non-utf8"
    assert non_utf8_diff["diff"] is None
    assert non_utf8_diff["original_content"] is None
    assert non_utf8_diff["new_content"] is None
    assert non_utf8_diff["content_unavailable_reason"] == "non_utf8"

    assert eof_diff["eof_newline"] is False
    assert eof_diff["mode_before"] == 0o100644
    assert eof_diff["mode_after"] == 0o100755
    assert "\\ No newline at end of file" in eof_diff["diff"]
    assert untracked_diff["status"] == "untracked"
    assert untracked_diff["added_lines"] == 1
    assert "+new" in untracked_diff["diff"]


def test_porcelain_v2_parser_never_guesses_rename_from_arrow_text() -> None:
    ordinary = (
        b"1 .M N... 100644 100644 100644 "
        + (b"0" * 40)
        + b" "
        + (b"0" * 40)
        + b" literal -> filename.txt\0"
    )
    parsed = _parse_git_status_porcelain_v2(ordinary)

    assert parsed[0]["repo_path"] == "literal -> filename.txt"
    assert parsed[0]["baseline_repo_path"] is None
    assert parsed[0]["status"] == "modified"

    copied = (
        b"2 C. N... 100644 100644 100644 "
        + (b"1" * 40)
        + b" "
        + (b"2" * 40)
        + b" C100 copied file.txt\0source file.txt\0"
    )
    copy_record = _parse_git_status_porcelain_v2(copied)[0]
    assert copy_record["status"] == "copied"
    assert copy_record["repo_path"] == "copied file.txt"
    assert copy_record["baseline_repo_path"] == "source file.txt"


def test_workspace_patch_preview_uses_resolved_session_workspace(
    tmp_path: Path,
) -> None:
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


def test_workspace_patch_preview_rejects_binary_text_patch_explicitly(
    tmp_path: Path,
) -> None:
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()
    (workspace_dir / "asset.bin").write_bytes(b"\x00binary\xff")
    app = _create_test_app(
        workspace_dir=workspace_dir,
        sessions_dir=tmp_path / "sessions",
        projects_path=tmp_path / "projects.json",
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/workspace/patch/preview",
            json={
                "patch": "\n".join(
                    [
                        "*** Begin Patch",
                        "*** Update File: asset.bin",
                        "@@",
                        "-binary",
                        "+changed",
                        "*** End Patch",
                    ]
                )
            },
        )

    payload = response.json()
    assert payload["valid"] is False
    assert payload["content_unavailable_reason"] == "binary"
    assert payload["suggested_tool"] == "binary_asset"
    assert "Binary files cannot be edited with a text patch" in payload["errors"][0]


def test_workspace_patch_preview_persists_idempotent_digest_bound_manifest(
    tmp_path: Path,
) -> None:
    workspace_dir = tmp_path / "workspace"
    sessions_dir = tmp_path / "sessions"
    workspace_dir.mkdir(parents=True)
    target = workspace_dir / "notes.py"
    target.write_text("print('alpha')\n", encoding="utf-8")
    patch = "\n".join(
        [
            "*** Begin Patch",
            "*** Update File: notes.py",
            "@@",
            "-print('alpha')",
            "+print('beta')",
            "*** End Patch",
        ]
    )
    app = _create_test_app(
        workspace_dir=workspace_dir,
        sessions_dir=sessions_dir,
        projects_path=tmp_path / "projects.json",
        change_contract_mode="enforce",
    )

    with TestClient(app) as client:
        first = client.post(
            "/v1/workspace/patch/preview",
            json={"session_id": "session-a", "patch": patch},
        )
        second = client.post(
            "/v1/workspace/patch/preview",
            json={"session_id": "session-a", "patch": patch},
        )
        other_context = client.post(
            "/v1/workspace/patch/preview",
            json={"session_id": "session-b", "patch": patch},
        )

    assert first.status_code == 200
    first_payload = first.json()
    second_payload = second.json()
    other_payload = other_context.json()
    assert first_payload["change_contract_mode"] == "enforce"
    assert first_payload["policy_version"]
    assert first_payload["expires_at"]
    assert len(first_payload["request_digest"]) == 64
    assert first_payload["change_set_id"] == second_payload["change_set_id"]
    assert first_payload["request_digest"] == second_payload["request_digest"]
    assert first_payload["change_set_id"] != other_payload["change_set_id"]
    assert first_payload["request_digest"] != other_payload["request_digest"]
