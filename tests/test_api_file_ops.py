"""File operation API tests."""

from __future__ import annotations

import asyncio
import hashlib
import os
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mochi.api.server import create_app
from mochi.config.schema import MochiConfig
from mochi.runtime.change_sets import ChangeSetStore
from mochi.runtime.store import RuntimeStore
from mochi.security.file_contract import (
    AuthorizationContext,
    AuthorizationEnvelope,
    ChangeEntry,
    ChangeManifest,
    FileChangeRequest,
    FileIdentity,
    authorization_request_digest,
)
from mochi.tools.file_transaction import observe_undo_target


def _create_test_app(config: MochiConfig):
    app = create_app()
    app.state.config_factory = lambda: config
    return app


def _identity(path: Path) -> FileIdentity:
    info = path.stat()
    attributes = int(getattr(info, "st_file_attributes", 0))
    return FileIdentity(
        platform="windows" if os.name == "nt" else "posix",
        volume_id=str(int(info.st_dev)),
        file_id=str(int(info.st_ino)),
        link_count=max(1, int(info.st_nlink)),
        is_reparse_point=bool(attributes & 0x400),
    )


def _sha(content: bytes | None) -> str | None:
    return None if content is None else hashlib.sha256(content).hexdigest()


def _seed_authoritative_change(
    *,
    workspace: Path,
    sessions_dir: Path,
    changes: list[tuple[str, bytes | None, bytes | None, str, str | None]],
    retained_until: datetime | None = None,
) -> tuple[str, str, list[str]]:
    store = RuntimeStore(sessions_dir / "runtime.db")
    change_store = ChangeSetStore(store)
    entry_values: list[ChangeEntry] = []
    before_blobs: dict[str, str] = {}
    for index, (relative_path, before, after, operation, group) in enumerate(changes):
        entry_id = f"entry-{index + 1}"
        before_blob = None if before is None else asyncio.run(change_store.put_blob(before))
        after_blob = None if after is None else asyncio.run(change_store.put_blob(after))
        target = workspace / relative_path
        entry_values.append(
            ChangeEntry(
                entry_id=entry_id,
                relative_path=relative_path,
                operation=operation,  # type: ignore[arg-type]
                base_sha256=_sha(before),
                after_sha256=_sha(after),
                base_identity=_identity(target) if target.exists() else None,
                before_blob_id=before_blob,
                after_blob_id=after_blob,
                mode_before=None,
                mode_after=None,
                base_metadata_sha256=None,
                after_metadata_sha256=None,
                rename_source=None,
                dependency_group=group,
            )
        )
        if before_blob is not None:
            before_blobs[entry_id] = before_blob

    request = FileChangeRequest(entries=tuple(entry_values), patch_sha256=None)
    context = AuthorizationContext(
        requester_id="test-user",
        session_id="test-session",
        task_id=None,
        workspace_root=str(workspace.resolve()),
        workspace_identity=_identity(workspace),
    )
    envelope = AuthorizationEnvelope(
        schema_version=1,
        kind="file_change",
        context=context,
        policy_version="test-policy",
        file_request=request,
        exec_request=None,
    )
    digest = authorization_request_digest(envelope)
    now = datetime.now(UTC)
    change_set_id = "change-authoritative"
    manifest = ChangeManifest(
        version=1,
        change_set_id=change_set_id,
        workspace_root=str(workspace.resolve()),
        workspace_identity=_identity(workspace),
        tool_name="apply_patch",
        intent="mutate",
        entries=request.entries,
        patch_sha256=None,
        policy_version=envelope.policy_version,
        created_at=now.isoformat(),
        expires_at=(now + timedelta(hours=1)).isoformat(),
        request_digest=digest,
    )
    asyncio.run(change_store.persist_manifest(manifest, envelope))
    asyncio.run(change_store.mark_change_set_status(change_set_id, status="applied"))
    expiry = retained_until or now + timedelta(hours=1)
    for entry in entry_values:
        target = workspace / entry.relative_path
        asyncio.run(
            change_store.record_applied_entry(
                change_set_id=change_set_id,
                entry_id=entry.entry_id,
                applied_sha256=_sha(target.read_bytes()) if target.exists() else None,
                applied_identity=_identity(target) if target.exists() else None,
                applied_metadata_sha256=(
                    observe_undo_target(target).metadata_sha256
                    if target.exists()
                    else None
                ),
            )
        )
        asyncio.run(
            change_store.set_undo_retention(
                change_set_id=change_set_id,
                entry_id=entry.entry_id,
                status="retained",
                retained_until=expiry.isoformat(),
            )
        )
        before_blob = before_blobs.get(entry.entry_id)
        if before_blob is not None:
            asyncio.run(
                change_store.add_blob_reference(
                    blob_id=before_blob,
                    owner_type="change_entry",
                    owner_id=f"{change_set_id}:{entry.entry_id}",
                    purpose="undo",
                    retained_until=expiry.isoformat(),
                )
            )
    return change_set_id, digest, [entry.entry_id for entry in entry_values]


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
    assert response.json()["would_reject_legacy_undo"] is True


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
    assert response.json()["would_reject_legacy_undo"] is True

def test_enforce_rejects_legacy_raw_content_undo(tmp_path: Path) -> None:
    target = tmp_path / "demo.txt"
    target.write_text("after", encoding="utf-8")
    app = _create_test_app(
        MochiConfig.model_validate(
            {
                "workspace_dir": str(tmp_path),
                "sessions_dir": str(tmp_path / "sessions"),
                "security": {"change_contract_mode": "enforce"},
            }
        )
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/tools/file/undo",
            json={
                "file_path": str(target),
                "original_content": "forged",
                "session_id": "missing-session",
                "action": "restore",
            },
        )

    assert response.status_code == 409
    assert response.json()["detail"] == "legacy_undo_requires_change_id"
    assert target.read_text(encoding="utf-8") == "after"


def test_authoritative_undo_restores_retained_server_blob(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    target = tmp_path / "demo.txt"
    target.write_bytes(b"after")
    change_id, digest, entry_ids = _seed_authoritative_change(
        workspace=tmp_path,
        sessions_dir=sessions_dir,
        changes=[("demo.txt", b"before", b"after", "update", None)],
    )
    app = _create_test_app(
        MochiConfig.model_validate(
            {
                "workspace_dir": str(tmp_path),
                "sessions_dir": str(sessions_dir),
                "security": {"change_contract_mode": "enforce"},
            }
        )
    )

    with TestClient(app) as client:
        response = client.post(
            f"/v1/changes/{change_id}/undo",
            json={"entry_ids": entry_ids, "request_digest": digest},
        )

    assert response.status_code == 200, response.json()
    assert response.json()["status"] == "applied"
    assert response.json()["inverse_change_set_id"] != change_id
    assert target.read_bytes() == b"before"


def test_authoritative_undo_rejects_same_content_new_inode(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    target = tmp_path / "demo.txt"
    target.write_bytes(b"after")
    change_id, digest, entry_ids = _seed_authoritative_change(
        workspace=tmp_path,
        sessions_dir=sessions_dir,
        changes=[("demo.txt", b"before", b"after", "update", None)],
    )
    original_identity = _identity(target)
    replacement = tmp_path / "replacement.txt"
    replacement.write_bytes(b"after")
    replacement.replace(target)
    if _identity(target) == original_identity:
        pytest.skip("filesystem reused the same file identity")
    app = _create_test_app(
        MochiConfig.model_validate(
            {
                "workspace_dir": str(tmp_path),
                "sessions_dir": str(sessions_dir),
                "security": {"change_contract_mode": "enforce"},
            }
        )
    )

    with TestClient(app) as client:
        response = client.post(
            f"/v1/changes/{change_id}/undo",
            json={"entry_ids": entry_ids, "request_digest": digest},
        )

    assert response.status_code == 409
    assert response.json()["detail"] == "undo_target_changed"
    assert target.read_bytes() == b"after"


def test_authoritative_undo_expired_retention_returns_410(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    target = tmp_path / "demo.txt"
    target.write_bytes(b"after")
    change_id, digest, entry_ids = _seed_authoritative_change(
        workspace=tmp_path,
        sessions_dir=sessions_dir,
        changes=[("demo.txt", b"before", b"after", "update", None)],
        retained_until=datetime.now(UTC) - timedelta(minutes=1),
    )
    app = _create_test_app(
        MochiConfig.model_validate(
            {"workspace_dir": str(tmp_path), "sessions_dir": str(sessions_dir)}
        )
    )

    with TestClient(app) as client:
        response = client.post(
            f"/v1/changes/{change_id}/undo",
            json={"entry_ids": entry_ids, "request_digest": digest},
        )

    assert response.status_code == 410
    assert response.json()["detail"] == "undo_retention_expired"
    assert target.read_bytes() == b"after"
    with sqlite3.connect(sessions_dir / "runtime.db") as conn:
        retention = conn.execute(
            """
            SELECT status, expired_at FROM undo_retention
            WHERE change_set_id=? AND entry_id=?
            """,
            (change_id, entry_ids[0]),
        ).fetchone()
        reference_state = conn.execute(
            """
            SELECT state FROM blob_references
            WHERE owner_type='change_entry' AND owner_id=? AND purpose='undo'
            """,
            (f"{change_id}:{entry_ids[0]}",),
        ).fetchone()
    assert retention is not None
    assert retention[0] == "expired"
    assert retention[1] is not None
    assert reference_state is not None
    assert reference_state[0] == "expired"


def test_authoritative_undo_not_retained_returns_410(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    target = tmp_path / "demo.txt"
    target.write_bytes(b"after")
    change_id, digest, entry_ids = _seed_authoritative_change(
        workspace=tmp_path,
        sessions_dir=sessions_dir,
        changes=[("demo.txt", b"before", b"after", "update", None)],
    )
    with sqlite3.connect(sessions_dir / "runtime.db") as conn:
        conn.execute(
            "DELETE FROM undo_retention WHERE change_set_id=? AND entry_id=?",
            (change_id, entry_ids[0]),
        )

    app = _create_test_app(
        MochiConfig.model_validate(
            {"workspace_dir": str(tmp_path), "sessions_dir": str(sessions_dir)}
        )
    )
    with TestClient(app) as client:
        response = client.post(
            f"/v1/changes/{change_id}/undo",
            json={"entry_ids": entry_ids, "request_digest": digest},
        )

    assert response.status_code == 410
    assert response.json()["detail"] == "undo_not_retained"
    assert target.read_bytes() == b"after"

def test_authoritative_undo_rejects_partial_dependency_group(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_bytes(b"after-one")
    second.write_bytes(b"after-two")
    change_id, digest, entry_ids = _seed_authoritative_change(
        workspace=tmp_path,
        sessions_dir=sessions_dir,
        changes=[
            ("first.txt", b"before-one", b"after-one", "update", "rename-1"),
            ("second.txt", b"before-two", b"after-two", "update", "rename-1"),
        ],
    )
    app = _create_test_app(
        MochiConfig.model_validate(
            {"workspace_dir": str(tmp_path), "sessions_dir": str(sessions_dir)}
        )
    )

    with TestClient(app) as client:
        response = client.post(
            f"/v1/changes/{change_id}/undo",
            json={"entry_ids": entry_ids[:1], "request_digest": digest},
        )

    assert response.status_code == 409
    assert response.json()["detail"] == "partial_dependency_group"
    assert first.read_bytes() == b"after-one"
    assert second.read_bytes() == b"after-two"
