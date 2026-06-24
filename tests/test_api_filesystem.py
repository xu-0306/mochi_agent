"""filesystem API routes 測試。"""

from __future__ import annotations

from pathlib import Path
import sys
import zipfile

from fastapi.testclient import TestClient

from mochi.api.server import create_app
from mochi.config.schema import MochiConfig


def _create_test_app(config: MochiConfig):
    app = create_app()
    app.state.config_factory = lambda: config
    return app


def test_filesystem_roots_include_common_and_configured_paths(tmp_path: Path) -> None:
    """`/v1/filesystem/roots` 應回傳可瀏覽 roots 與設定路徑。"""
    workspace = tmp_path / "workspace"
    sessions = tmp_path / "sessions"
    skills = tmp_path / "skills"
    plugins = tmp_path / "plugins"
    memory_db = tmp_path / "memory" / "memory.db"
    stt_cache = tmp_path / "cache" / "models"

    config = MochiConfig.model_validate(
        {
            "workspace_dir": str(workspace),
            "sessions_dir": str(sessions),
            "skills_dir": str(skills),
            "plugins_dir": str(plugins),
            "memory": {"db_path": str(memory_db)},
            "voice": {"stt_model_cache_dir": str(stt_cache)},
        }
    )
    app = _create_test_app(config)

    with TestClient(app) as client:
        response = client.get("/v1/filesystem/roots")

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "filesystem_roots"

    by_name = {item["name"]: item["path"] for item in payload["items"]}
    assert by_name["home"] == str(Path.home())
    assert by_name["cwd"] == str(Path.cwd())
    assert by_name["workspace"] == str(workspace)
    assert by_name["sessions"] == str(sessions)
    assert by_name["skills"] == str(skills)
    assert by_name["plugins"] == str(plugins)
    assert by_name["memory-parent"] == str(memory_db.parent)
    assert by_name["stt-cache-parent"] == str(stt_cache.parent)


def test_filesystem_list_returns_single_level_metadata(tmp_path: Path) -> None:
    """`/v1/filesystem/list` 應回傳單層檔案/資料夾 metadata。"""
    root = tmp_path / "picker-root"
    child_dir = root / "alpha"
    child_file = root / "beta.txt"
    root.mkdir()
    child_dir.mkdir()
    child_file.write_text("hello", encoding="utf-8")

    app = _create_test_app(MochiConfig.model_validate({}))

    with TestClient(app) as client:
        response = client.get("/v1/filesystem/list", params={"path": str(root)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "filesystem_list"
    assert payload["current_path"] == str(root)
    assert payload["parent_path"] == str(root.parent)

    items = {item["name"]: item for item in payload["items"]}
    assert items["alpha"] == {
        "name": "alpha",
        "path": str(child_dir),
        "is_dir": True,
        "is_file": False,
    }
    assert items["beta.txt"] == {
        "name": "beta.txt",
        "path": str(child_file),
        "is_dir": False,
        "is_file": True,
    }


def test_filesystem_list_not_found_and_not_directory(tmp_path: Path) -> None:
    """無法推導 parent 的路徑才應回傳 not found。"""
    app = _create_test_app(MochiConfig.model_validate({}))

    with TestClient(app) as client:
        missing_response = client.get(
            "/v1/filesystem/list",
            params={"path": str(tmp_path / "does-not-exist" / "nested.txt")},
        )

    assert missing_response.status_code == 404
    assert missing_response.json() == {"detail": "Path not found"}


def test_filesystem_list_file_path_lists_parent_directory(tmp_path: Path) -> None:
    """檔案路徑應視為 picker selected path，並列出 parent directory。"""
    app = _create_test_app(MochiConfig.model_validate({}))
    existing_file = tmp_path / "single.txt"
    existing_file.write_text("x", encoding="utf-8")

    with TestClient(app) as client:
        response = client.get("/v1/filesystem/list", params={"path": str(existing_file)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["current_path"] == str(tmp_path)
    assert payload["selected_path"] == str(existing_file)
    assert any(item["name"] == "single.txt" and item["is_file"] for item in payload["items"])


def test_filesystem_list_missing_file_name_lists_existing_parent(tmp_path: Path) -> None:
    """尚未存在的檔案名如 1.txt 應列 parent，不應直接報錯。"""
    app = _create_test_app(MochiConfig.model_validate({}))
    target = tmp_path / "1.txt"

    with TestClient(app) as client:
        response = client.get("/v1/filesystem/list", params={"path": str(target)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["current_path"] == str(tmp_path)
    assert payload["selected_path"] == str(target)


def test_filesystem_select_directory_returns_native_selection(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Native folder picker endpoint should return the selected backend directory."""
    selected_dir = tmp_path / "STT&TTS"
    selected_dir.mkdir()

    def fake_select_native_directory(initial_dir: Path | None, title: str | None) -> str:
        assert initial_dir == tmp_path
        assert title == "Select Project Root"
        return str(selected_dir)

    filesystem_module = sys.modules["mochi.api.routes.filesystem"]
    monkeypatch.setattr(filesystem_module, "_select_native_directory", fake_select_native_directory)
    app = _create_test_app(MochiConfig.model_validate({}))

    with TestClient(app) as client:
        response = client.post(
            "/v1/filesystem/select-directory",
            json={
                "initial_path": str(tmp_path / "missing-child"),
                "title": "Select Project Root",
            },
        )

    assert response.status_code == 200
    assert response.json() == {
        "type": "filesystem_directory_selection",
        "selected": True,
        "path": str(selected_dir),
        "name": "STT&TTS",
    }


def test_filesystem_select_directory_reports_cancel(tmp_path: Path, monkeypatch) -> None:
    """Cancelling the native folder picker should be a non-error response."""

    def fake_select_native_directory(initial_dir: Path | None, title: str | None) -> None:
        assert initial_dir == tmp_path
        return None

    filesystem_module = sys.modules["mochi.api.routes.filesystem"]
    monkeypatch.setattr(filesystem_module, "_select_native_directory", fake_select_native_directory)
    app = _create_test_app(MochiConfig.model_validate({}))

    with TestClient(app) as client:
        response = client.post(
            "/v1/filesystem/select-directory",
            json={"initial_path": str(tmp_path)},
        )

    assert response.status_code == 200
    assert response.json() == {
        "type": "filesystem_directory_selection",
        "selected": False,
        "path": None,
        "name": None,
    }


def test_filesystem_import_uploads_file_to_target_dir(tmp_path: Path) -> None:
    """本機 picker 上傳的檔案應保存到後端受控目錄並回傳 server path。"""
    app = _create_test_app(MochiConfig.model_validate({}))

    with TestClient(app) as client:
        response = client.post(
            "/v1/filesystem/import",
            data={
                "target_dir": str(tmp_path),
                "package_name": "demo model",
                "relative_paths": "model.bin",
            },
            files={"files": ("model.bin", b"abc", "application/octet-stream")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "filesystem_import"
    assert payload["file_count"] == 1
    assert len(payload["files"]) == 1
    imported = Path(payload["imported_path"])
    import_root = Path(payload["import_root"])
    assert imported.exists()
    assert imported.read_bytes() == b"abc"
    assert imported.is_relative_to(tmp_path)
    assert imported == Path(payload["files"][0]["path"])
    assert imported.parent == import_root
    assert import_root.name.endswith("demo-model")
    assert "browser-imports" in import_root.parts


def test_filesystem_import_defaults_to_workspace_browser_imports(tmp_path: Path) -> None:
    """Without a target dir, imports should land under the configured workspace."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    app = _create_test_app(MochiConfig.model_validate({"workspace_dir": str(workspace)}))

    with TestClient(app) as client:
        response = client.post(
            "/v1/filesystem/import",
            data={"package_name": "demo upload", "relative_paths": "notes.txt"},
            files={"files": ("notes.txt", b"hello", "text/plain")},
        )

    assert response.status_code == 200
    payload = response.json()
    imported = Path(payload["imported_path"])
    assert imported.exists()
    assert imported.read_text(encoding="utf-8") == "hello"
    assert imported.is_relative_to(workspace / "browser-imports")
    assert payload["files"][0]["path"] == payload["imported_path"]


def test_filesystem_import_multiple_files_keep_package_root_authoritative(tmp_path: Path) -> None:
    """Multi-file imports should keep the package directory as imported_path."""
    app = _create_test_app(MochiConfig.model_validate({}))

    with TestClient(app) as client:
        response = client.post(
            "/v1/filesystem/import",
            data={
                "target_dir": str(tmp_path),
                "package_name": "batched docs",
                "relative_paths": ["docs/one.txt", "docs/two.txt"],
            },
            files=[
                ("files", ("one.txt", b"one", "text/plain")),
                ("files", ("two.txt", b"two", "text/plain")),
            ],
        )

    assert response.status_code == 200
    payload = response.json()
    import_root = Path(payload["import_root"])
    imported = Path(payload["imported_path"])
    assert payload["file_count"] == 2
    assert imported == import_root
    assert imported.is_dir()
    assert len(payload["files"]) == 2
    saved_paths = [Path(item["path"]) for item in payload["files"]]
    assert all(path.is_relative_to(import_root) for path in saved_paths)
    assert {path.read_text(encoding="utf-8") for path in saved_paths} == {"one", "two"}


def test_filesystem_file_serves_workspace_preview_asset(tmp_path: Path) -> None:
    """`/v1/filesystem/file` should serve normal local files, not just workspace files."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    image = tmp_path / "preview.txt"
    image.write_text("hello preview", encoding="utf-8")

    app = _create_test_app(MochiConfig.model_validate({"workspace_dir": str(workspace)}))

    with TestClient(app) as client:
        response = client.get("/v1/filesystem/file", params={"path": str(image)})

    assert response.status_code == 200
    assert response.text == "hello preview"


def test_filesystem_file_blocks_suspicious_path(tmp_path: Path) -> None:
    """Preview routes should still reject suspicious raw path spellings."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    app = _create_test_app(MochiConfig.model_validate({"workspace_dir": str(workspace)}))

    with TestClient(app) as client:
        response = client.get("/v1/filesystem/file", params={"path": "\\\\?\\C:\\temp\\oops.txt"})

    assert response.status_code == 403
    assert "Suspicious path denied by security policy." in response.json()["detail"]


def test_filesystem_preview_text_extracts_external_docx_text(tmp_path: Path) -> None:
    """`/v1/filesystem/preview-text` should allow readable files outside workspace roots."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    docx_path = tmp_path / "notes.docx"

    with zipfile.ZipFile(docx_path, "w") as archive:
        archive.writestr(
            "word/document.xml",
            (
                '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
                '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                "<w:body>"
                "<w:p><w:r><w:t>First paragraph.</w:t></w:r></w:p>"
                "<w:p><w:r><w:t>Second paragraph.</w:t></w:r></w:p>"
                "</w:body>"
                "</w:document>"
            ),
        )

    app = _create_test_app(MochiConfig.model_validate({"workspace_dir": str(workspace)}))

    with TestClient(app) as client:
        response = client.get("/v1/filesystem/preview-text", params={"path": str(docx_path)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "filesystem_preview_text"
    assert "First paragraph." in payload["text"]
    assert "Second paragraph." in payload["text"]
