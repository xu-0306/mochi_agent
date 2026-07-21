"""File operation API routes, including server-authoritative undo."""

from __future__ import annotations

import asyncio
import hashlib
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from mochi.api.routes.projects import _get_project_store
from mochi.api.server import _get_config
from mochi.projects.execution_scope import ExecutionScopeResolver
from mochi.runtime.change_sets import (
    ChangeSetStore,
    UndoConflict,
    UndoUnavailable,
)
from mochi.runtime.store import RuntimeStore
from mochi.security.file_contract import (
    AuthorizationContext,
    AuthorizationEnvelope,
    ChangeEntry,
    ChangeManifest,
    FileChangeRequest,
    FileIdentity,
    authorization_request_digest,
    canonical_json,
)
from mochi.sessions.store import SessionStore
from mochi.tools.file_ops import file_change_policy_version
from mochi.tools.file_transaction import (
    UndoCASConflict,
    UndoCASObservation,
    UndoMutation,
    execute_authoritative_undo,
)
from mochi.utils.security import (
    content_size_bytes,
    resolve_path_with_scope,
    size_limit_bytes,
)

router = APIRouter(prefix="/v1", tags=["file_ops"])


class FileUndoRequest(BaseModel):
    """Legacy client-authored undo request, accepted only in observe mode."""

    file_path: str = Field(min_length=1)
    original_content: str | None = None
    session_id: str = Field(min_length=1)
    action: Literal["restore", "delete"] = "restore"
    encoding: str = "utf-8"


class ChangeUndoRequest(BaseModel):
    """Identity-only request for authoritative retained undo material."""

    entry_ids: list[str] = Field(min_length=1)
    request_digest: str = Field(min_length=64, max_length=64)


@router.post("/tools/file/undo")
async def undo_file_write(
    request: Request,
    payload: FileUndoRequest,
) -> dict[str, str | int | bool | None]:
    """Run the legacy raw-content undo only while the rollout is observing."""

    config = await _get_config(request.app)
    if config.security.change_contract_mode == "enforce":
        raise HTTPException(status_code=409, detail="legacy_undo_requires_change_id")
    scope = config.security.file_write_scope
    workspace_dir = await _resolve_workspace_dir_for_session(
        request,
        payload.session_id,
    )

    try:
        target = resolve_path_with_scope(payload.file_path, workspace_dir, scope)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if payload.action == "delete":
        if target.exists() and target.is_dir():
            raise HTTPException(status_code=400, detail="Path is not a file")
        if target.exists():
            await asyncio.to_thread(target.unlink)
        return {
            "type": "file_undo",
            "action": "delete",
            "path": str(target),
            "bytes_written": 0,
            "change_contract_mode": "observe",
            "would_reject_legacy_undo": True,
        }

    original_content = payload.original_content or ""
    limit_bytes = size_limit_bytes(config.security.file_undo_max_size_mb)
    if (
        limit_bytes > 0
        and content_size_bytes(original_content, encoding=payload.encoding)
        > limit_bytes
    ):
        raise HTTPException(status_code=400, detail="Undo content exceeds size limit")

    if target.exists() and target.is_dir():
        raise HTTPException(status_code=400, detail="Path is not a file")

    def _write() -> int:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(original_content, encoding=payload.encoding)
        return len(original_content.encode(payload.encoding))

    bytes_written = await asyncio.to_thread(_write)
    return {
        "type": "file_undo",
        "action": "restore",
        "path": str(target),
        "bytes_written": bytes_written,
        "change_contract_mode": "observe",
        "would_reject_legacy_undo": True,
    }


@router.post("/changes/{change_set_id}/undo")
async def undo_change_set(
    request: Request,
    change_set_id: str,
    payload: ChangeUndoRequest,
) -> dict[str, Any]:
    """Create and execute a reverse manifest from retained server state."""

    config = await _get_config(request.app)
    runtime_store = await _get_runtime_store(request)
    change_store = ChangeSetStore(runtime_store)
    now = datetime.now(UTC)
    entry_ids = tuple(payload.entry_ids)
    try:
        material = await change_store.get_undo_material(
            change_set_id=change_set_id,
            entry_ids=entry_ids,
            request_digest=payload.request_digest,
            now=now.isoformat(),
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="change_set_not_found") from exc
    except UndoUnavailable as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc
    except UndoConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    original_manifest = material["manifest"]
    original_envelope = material["envelope"]
    workspace = Path(original_manifest.workspace_root)
    if _path_identity(workspace) != original_manifest.workspace_identity:
        raise HTTPException(status_code=409, detail="undo_workspace_changed")

    inverse_entries: list[ChangeEntry] = []
    mutations: list[UndoMutation] = []
    for ordinal, item in enumerate(material["entries"]):
        entry = item["entry"]
        applied = item["applied"]
        before_content = item["before_content"]
        operation = "delete" if entry.operation == "add" else "restore"
        inverse_operation = (
            "delete"
            if entry.operation == "add"
            else "add"
            if entry.operation == "delete"
            else "update"
        )
        seed = canonical_json(
            {
                "source_change_set_id": change_set_id,
                "source_entry_id": entry.entry_id,
                "ordinal": ordinal,
                "request_digest": payload.request_digest,
            }
        )
        inverse_entry_id = hashlib.sha256(seed.encode("utf-8")).hexdigest()
        inverse_entries.append(
            ChangeEntry(
                entry_id=inverse_entry_id,
                relative_path=entry.relative_path,
                operation=inverse_operation,
                base_sha256=applied.applied_sha256,
                after_sha256=entry.base_sha256,
                base_identity=applied.applied_identity,
                before_blob_id=entry.after_blob_id,
                after_blob_id=entry.before_blob_id,
                mode_before=entry.mode_after,
                mode_after=entry.mode_before,
                base_metadata_sha256=applied.applied_metadata_sha256,
                after_metadata_sha256=entry.base_metadata_sha256,
                rename_source=entry.rename_source,
                dependency_group=entry.dependency_group,
            )
        )
        mutations.append(
            UndoMutation(
                entry_id=inverse_entry_id,
                relative_name=entry.relative_path,
                operation=operation,
                expected=UndoCASObservation(
                    identity=applied.applied_identity,
                    content_sha256=applied.applied_sha256,
                    metadata_sha256=applied.applied_metadata_sha256,
                ),
                restore_content=before_content,
                restore_mode=entry.mode_before,
            )
        )

    inverse_request = FileChangeRequest(
        entries=tuple(inverse_entries),
        patch_sha256=None,
    )
    context = AuthorizationContext(
        requester_id=original_envelope.context.requester_id,
        session_id=original_envelope.context.session_id,
        task_id=original_envelope.context.task_id,
        workspace_root=original_envelope.context.workspace_root,
        workspace_identity=original_envelope.context.workspace_identity,
    )
    envelope = AuthorizationEnvelope(
        schema_version=1,
        kind="file_change",
        context=context,
        policy_version=file_change_policy_version(config.security),
        file_request=inverse_request,
        exec_request=None,
    )
    inverse_digest = authorization_request_digest(envelope)
    inverse_manifest = ChangeManifest(
        version=1,
        change_set_id=str(uuid4()),
        workspace_root=str(workspace),
        workspace_identity=context.workspace_identity,
        tool_name="undo_change",
        intent="undo",
        entries=inverse_request.entries,
        patch_sha256=None,
        policy_version=envelope.policy_version,
        created_at=now.isoformat(),
        expires_at=(now + timedelta(minutes=15)).isoformat(),
        request_digest=inverse_digest,
        ui_metadata={
            "source_change_set_id": change_set_id,
            "source_entry_ids": list(entry_ids),
        },
    )
    persisted = await change_store.persist_manifest(inverse_manifest, envelope)
    inverse_manifest = persisted["manifest"]
    try:
        observations = await asyncio.to_thread(
            execute_authoritative_undo,
            workspace,
            tuple(mutations),
        )
    except UndoCASConflict as exc:
        await change_store.mark_change_set_status(
            inverse_manifest.change_set_id,
            status="conflicted",
        )
        raise HTTPException(status_code=409, detail="undo_target_changed") from exc
    except (OSError, RuntimeError) as exc:
        await change_store.mark_change_set_status(
            inverse_manifest.change_set_id,
            status="conflicted",
        )
        raise HTTPException(status_code=409, detail="undo_transaction_failed") from exc

    applied_at = datetime.now(UTC).isoformat()
    for entry in inverse_manifest.entries:
        observation = observations[entry.entry_id]
        await change_store.record_applied_entry(
            change_set_id=inverse_manifest.change_set_id,
            entry_id=entry.entry_id,
            applied_sha256=observation.content_sha256,
            applied_identity=observation.identity,
            applied_metadata_sha256=observation.metadata_sha256,
            applied_at=applied_at,
        )
    await change_store.mark_change_set_status(
        inverse_manifest.change_set_id,
        status="applied",
        applied_at=applied_at,
    )
    await change_store.consume_undo_retention(
        change_set_id=change_set_id,
        entry_ids=entry_ids,
    )
    return {
        "type": "change_undo",
        "status": "applied",
        "change_set_id": change_set_id,
        "entry_ids": list(entry_ids),
        "request_digest": payload.request_digest,
        "inverse_change_set_id": inverse_manifest.change_set_id,
        "inverse_request_digest": inverse_manifest.request_digest,
        "change_contract_mode": config.security.change_contract_mode,
    }


def _path_identity(path: Path) -> FileIdentity:
    info = path.stat()
    attributes = int(getattr(info, "st_file_attributes", 0))
    return FileIdentity(
        platform="windows" if os.name == "nt" else "posix",
        volume_id=str(int(info.st_dev)),
        file_id=str(int(info.st_ino)),
        link_count=max(1, int(info.st_nlink)),
        is_reparse_point=bool(attributes & 0x400),
    )


async def _get_runtime_store(request: Request) -> RuntimeStore:
    existing = getattr(request.app.state, "runtime_store", None)
    if isinstance(existing, RuntimeStore):
        await existing.initialize()
        return existing
    config = await _get_config(request.app)
    store = RuntimeStore(Path(config.sessions_dir) / "runtime.db")
    await store.initialize()
    request.app.state.runtime_store = store
    return store


async def _resolve_workspace_dir_for_session(request: Request, session_id: str) -> str:
    """Resolve workspace boundary from session project assignment."""

    config = await _get_config(request.app)
    resolver = ExecutionScopeResolver(
        default_workspace_dir=str(config.workspace_dir),
        session_store=await _get_session_store(request),
        project_store=_get_project_store(request.app, config=config),
    )
    try:
        scope = await resolver.resolve(session_id=session_id)
    except LookupError:
        return str(config.workspace_dir)
    return scope.workspace_dir


async def _get_session_store(request: Request) -> SessionStore:
    existing = getattr(request.app.state, "session_store", None)
    if isinstance(existing, SessionStore):
        return existing

    config = await _get_config(request.app)
    store = SessionStore(config.sessions_dir)
    request.app.state.session_store = store
    return store
