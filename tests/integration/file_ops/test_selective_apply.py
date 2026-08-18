from __future__ import annotations

import asyncio
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from mochi.config.schema import SecurityConfig
from mochi.runtime.change_sets import ChangeSetConflict, ChangeSetStore
from mochi.runtime.store import RuntimeStore
from mochi.security.file_contract import (
    AuthorizationContext,
    AuthorizationEnvelope,
    ChangeEntry,
    ChangeManifest,
    FileChangeRequest,
    FileIdentity,
    authorization_request_digest,
    derive_file_change_subset,
)
from mochi.tools.file_ops import (
    prepare_file_change_subset_contract,
    prepare_patch_change_contract,
)


def _identity() -> FileIdentity:
    return FileIdentity(
        platform="windows",
        volume_id="1",
        file_id="2",
        link_count=1,
        is_reparse_point=False,
    )


def _entry(entry_id: str, group: str | None) -> ChangeEntry:
    digest = hashlib.sha256(entry_id.encode("utf-8")).hexdigest()
    return ChangeEntry(
        entry_id=entry_id,
        relative_path=f"{entry_id}.txt",
        operation="update",
        base_sha256=digest,
        after_sha256=digest,
        base_identity=_identity(),
        before_blob_id=None,
        after_blob_id=None,
        mode_before=None,
        mode_after=None,
        base_metadata_sha256=None,
        after_metadata_sha256=None,
        rename_source=None,
        dependency_group=group,
    )


def _envelope(request: FileChangeRequest) -> AuthorizationEnvelope:
    return AuthorizationEnvelope(
        schema_version=2,
        kind="file_change",
        context=AuthorizationContext(
            requester_id="requester",
            session_id="session",
            task_id="task",
            workspace_root="D:/workspace",
            workspace_identity=_identity(),
        ),
        policy_version="file-policy-v1",
        file_request=request,
        exec_request=None,
    )


def _manifest(
    envelope: AuthorizationEnvelope,
    *,
    change_set_id: str,
) -> ChangeManifest:
    request = envelope.file_request
    assert request is not None
    return ChangeManifest(
        version=2,
        change_set_id=change_set_id,
        workspace_root=envelope.context.workspace_root,
        workspace_identity=envelope.context.workspace_identity,
        tool_name="apply_patch",
        intent="mutate",
        entries=request.entries,
        patch_sha256=request.patch_sha256,
        policy_version=envelope.policy_version,
        created_at="2026-08-06T00:00:00+00:00",
        expires_at="2026-08-07T00:00:00+00:00",
        request_digest=authorization_request_digest(envelope),
        parent_request_digest=request.parent_request_digest,
        selected_entry_ids=request.selected_entry_ids,
    )


def test_selective_apply_requires_complete_dependency_group_and_new_approval() -> None:
    parent_request = FileChangeRequest(
        entries=(
            _entry("rename-source", "rename-1"),
            _entry("rename-target", "rename-1"),
            _entry("independent", None),
        ),
        patch_sha256=hashlib.sha256(b"parent patch").hexdigest(),
    )
    parent_digest = authorization_request_digest(_envelope(parent_request))

    with pytest.raises(ValueError, match="partial_dependency_group"):
        derive_file_change_subset(
            parent_request=parent_request,
            parent_request_digest=parent_digest,
            selected_entry_ids=("rename-source",),
        )

    subset = derive_file_change_subset(
        parent_request=parent_request,
        parent_request_digest=parent_digest,
        selected_entry_ids=("rename-source", "rename-target"),
    )
    subset_digest = authorization_request_digest(_envelope(subset))

    assert subset.parent_request_digest == parent_digest
    assert subset.selected_entry_ids == ("rename-source", "rename-target")
    assert subset_digest != parent_digest
    assert subset_digest != subset.parent_request_digest


def test_subset_manifest_requires_persisted_parent_and_preserves_lineage(tmp_path: Path) -> None:
    parent_request = FileChangeRequest(
        entries=(_entry("first", "group-1"), _entry("second", "group-1")),
        patch_sha256=hashlib.sha256(b"parent patch").hexdigest(),
    )
    parent_envelope = _envelope(parent_request)
    parent_digest = authorization_request_digest(parent_envelope)
    subset_request = derive_file_change_subset(
        parent_request=parent_request,
        parent_request_digest=parent_digest,
        selected_entry_ids=("first", "second"),
    )
    subset_envelope = _envelope(subset_request)
    change_store = ChangeSetStore(RuntimeStore(tmp_path / "runtime.db"))

    with pytest.raises(ChangeSetConflict, match="parent_request_missing"):
        asyncio.run(
            change_store.persist_manifest(
                _manifest(subset_envelope, change_set_id="subset"),
                subset_envelope,
            )
        )

    asyncio.run(change_store.persist_manifest(
        _manifest(parent_envelope, change_set_id="parent"),
        parent_envelope,
    ))
    stored_subset = asyncio.run(change_store.persist_manifest(
        _manifest(subset_envelope, change_set_id="subset"),
        subset_envelope,
    ))

    assert stored_subset["manifest"].parent_request_digest == parent_digest


def test_subset_manifest_rejects_an_entry_changed_after_selection(tmp_path: Path) -> None:
    parent_request = FileChangeRequest(
        entries=(_entry("first", "group-1"), _entry("second", "group-1")),
        patch_sha256=hashlib.sha256(b"parent patch").hexdigest(),
    )
    parent_envelope = _envelope(parent_request)
    parent_digest = authorization_request_digest(parent_envelope)
    subset_request = derive_file_change_subset(
        parent_request=parent_request,
        parent_request_digest=parent_digest,
        selected_entry_ids=("first", "second"),
    )
    tampered_request = FileChangeRequest(
        entries=(
            replace(subset_request.entries[0], relative_path="different.txt"),
            subset_request.entries[1],
        ),
        patch_sha256=subset_request.patch_sha256,
        parent_request_digest=subset_request.parent_request_digest,
        selected_entry_ids=subset_request.selected_entry_ids,
    )
    change_store = ChangeSetStore(RuntimeStore(tmp_path / "runtime.db"))
    asyncio.run(change_store.persist_manifest(
        _manifest(parent_envelope, change_set_id="parent"),
        parent_envelope,
    ))
    tampered_envelope = _envelope(tampered_request)

    with pytest.raises(ChangeSetConflict, match="subset_entry_changed"):
        asyncio.run(
            change_store.persist_manifest(
                _manifest(tampered_envelope, change_set_id="tampered"),
                tampered_envelope,
            )
        )


def test_subset_preview_derives_and_persists_a_child_from_server_state(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "alpha.txt").write_text("alpha\n", encoding="utf-8")
    (workspace / "beta.txt").write_text("beta\n", encoding="utf-8")
    runtime_store = RuntimeStore(tmp_path / "runtime.db")
    patch = "\n".join(
        [
            "*** Begin Patch",
            "*** Update File: alpha.txt",
            "@@",
            "-alpha",
            "+alpha updated",
            "*** Update File: beta.txt",
            "@@",
            "-beta",
            "+beta updated",
            "*** End Patch",
        ]
    )
    task = {
        "id": "task-1",
        "session_id": "session-1",
        "task_workspace_dir": str(workspace),
    }
    _, change_payload, parent_contract = asyncio.run(
        prepare_patch_change_contract(
            runtime_store=runtime_store,
            patch=patch,
            workspace_dir=workspace,
            security=SecurityConfig(),
            requester_id="requester-1",
            session_id="session-1",
            task_id="task-1",
        )
    )
    parent_entries = change_payload["file_changes"]
    alpha_entry = next(
        entry for entry in parent_entries if entry["relative_path"] == "alpha.txt"
    )
    approval = {
        "request_digest": parent_contract["request_digest"],
        "metadata": {**change_payload, **parent_contract},
        "arguments": {"patch": patch},
    }

    child_payload, child_contract = asyncio.run(
        prepare_file_change_subset_contract(
            runtime_store=runtime_store,
            approval=approval,
            task=task,
            security=SecurityConfig(),
            selected_entry_ids=(alpha_entry["entry_id"],),
        )
    )

    assert child_contract["request_digest"] != parent_contract["request_digest"]
    assert child_contract["parent_request_digest"] == parent_contract["request_digest"]
    assert child_contract["selected_entry_ids"] == [alpha_entry["entry_id"]]
    assert len(child_payload["file_changes"]) == 1
    assert child_payload["file_changes"][0]["relative_path"] == "alpha.txt"
    assert child_payload["file_changes"][0]["entry_id"] != alpha_entry["entry_id"]
