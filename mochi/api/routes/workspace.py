"""Session-aware workspace browsing and diff API routes."""

from __future__ import annotations

import asyncio
import mimetypes
import os
import subprocess
from pathlib import Path
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from mochi.api.routes.filesystem import _preview_docx, _preview_pdf, _preview_text_file
from mochi.api.routes.projects import _get_project_store
from mochi.api.server import _get_config
from mochi.api.session_store_binding import resolve_route_session_store
from mochi.projects.execution_scope import ExecutionScopeResolver
from mochi.runtime.approvals import ApprovalConflict
from mochi.runtime.store import RuntimeStore
from mochi.sessions.store import SessionStore
from mochi.tools.file_mutations import PatchValidationError
from mochi.tools.file_ops import (
    file_change_policy_version,
    prepare_patch_change_contract,
)
from mochi.utils.security import (
    is_path_within_workspace,
    normalize_workspace_dir,
    resolve_path_in_workspace,
    resolve_path_with_scope,
)

router = APIRouter(prefix="/v1/workspace", tags=["workspace"])


class WorkspacePatchPreviewRequest(BaseModel):
    """Patch preview request payload."""

    model_config = ConfigDict(populate_by_name=True)

    patch: str = Field(min_length=1, alias="patch_text")
    session_id: str | None = None
    project_id: str | None = None
    approval_id: str | None = None
    encoding: str = "utf-8"


@router.get("/tree")
async def get_workspace_tree(
    request: Request,
    session_id: str | None = Query(default=None),
    project_id: str | None = Query(default=None),
    path: str | None = Query(default=None),
) -> dict[str, Any]:
    resolved_project_id, workspace_root = await resolve_workspace_scope(
        request,
        session_id=session_id,
        project_id=project_id,
    )
    config = await _get_config(request.app)
    read_scope = config.security.file_read_scope
    current, selected_path = _coerce_workspace_browse_directory(
        workspace_root,
        path,
        read_scope,
    )

    try:
        entries = await asyncio.to_thread(lambda: list(current.iterdir()))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="Permission denied") from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    items: list[dict[str, Any]] = []
    for entry in sorted(entries, key=lambda item: item.name.lower()):
        resolved_entry = entry.resolve(strict=False)
        if read_scope == "workspace" and not is_path_within_workspace(
            resolved_entry, workspace_root
        ):
            continue
        try:
            is_dir = entry.is_dir()
            is_file = entry.is_file()
        except (PermissionError, OSError):
            continue
        item: dict[str, Any] = {
            "name": entry.name,
            "path": str(resolved_entry),
            "relative_path": _relative_path(workspace_root, resolved_entry),
            "is_dir": is_dir,
            "is_file": is_file,
        }
        if is_file:
            try:
                item["size"] = entry.stat().st_size
            except OSError:
                item["size"] = None
        items.append(item)

    parent = current.parent if current != workspace_root else None
    return {
        "type": "workspace_tree",
        "session_id": session_id or "draft-session",
        "project_id": resolved_project_id,
        "workspace_dir": str(workspace_root),
        "current_path": str(current),
        "relative_path": _relative_path(workspace_root, current),
        "parent_path": str(parent) if parent is not None else None,
        "selected_path": str(selected_path) if selected_path is not None else None,
        "items": items,
    }


@router.get("/file")
async def get_workspace_file(
    request: Request,
    path: str = Query(..., min_length=1),
    session_id: str | None = Query(default=None),
    project_id: str | None = Query(default=None),
) -> FileResponse:
    _, workspace_root = await resolve_workspace_scope(
        request,
        session_id=session_id,
        project_id=project_id,
    )
    config = await _get_config(request.app)
    target = _resolve_workspace_file_target(
        workspace_root,
        path,
        config.security.file_read_scope,
    )
    media_type, _ = mimetypes.guess_type(target.name)
    return FileResponse(target, media_type=media_type or "application/octet-stream")


@router.get("/preview")
async def preview_workspace_file(
    request: Request,
    path: str = Query(..., min_length=1),
    session_id: str | None = Query(default=None),
    project_id: str | None = Query(default=None),
    max_chars: int = Query(default=12000, ge=1, le=100000),
) -> dict[str, Any]:
    resolved_project_id, workspace_root = await resolve_workspace_scope(
        request,
        session_id=session_id,
        project_id=project_id,
    )
    config = await _get_config(request.app)
    target = _resolve_workspace_file_target(
        workspace_root,
        path,
        config.security.file_read_scope,
    )
    suffix = target.suffix.lower()
    if suffix == ".docx":
        payload = await _preview_docx(target, max_chars)
    elif suffix == ".pdf":
        payload = await _preview_pdf(target, max_chars)
    else:
        payload = await _preview_text_file(target, max_chars)

    return {
        "type": "workspace_preview",
        "session_id": session_id or "draft-session",
        "project_id": resolved_project_id,
        "workspace_dir": str(workspace_root),
        "path": str(target),
        "relative_path": _relative_path(workspace_root, target),
        "name": target.name,
        "text": payload["text"],
        "truncated": payload["truncated"],
        "media_type": payload["media_type"],
    }


@router.get("/changes")
async def list_workspace_changes(
    request: Request,
    session_id: str | None = Query(default=None),
    project_id: str | None = Query(default=None),
    path: str | None = Query(default=None),
) -> dict[str, Any]:
    resolved_project_id, workspace_root = await resolve_workspace_scope(
        request,
        session_id=session_id,
        project_id=project_id,
    )
    filter_path = _resolve_workspace_path_filter(workspace_root, path)
    repo_root = _find_git_repo_root(filter_path or workspace_root)
    if repo_root is None:
        return {
            "type": "workspace_changes",
            "session_id": session_id or "draft-session",
            "project_id": resolved_project_id,
            "workspace_dir": str(workspace_root),
            "repo_root": None,
            "items": [],
        }

    items = await asyncio.to_thread(
        _collect_workspace_changes,
        repo_root,
        workspace_root,
        filter_path,
        False,
        3,
    )
    return {
        "type": "workspace_changes",
        "session_id": session_id or "draft-session",
        "project_id": resolved_project_id,
        "workspace_dir": str(workspace_root),
        "repo_root": str(repo_root),
        "items": items,
    }


@router.get("/diff")
async def get_workspace_diff(
    request: Request,
    path: str = Query(..., min_length=1),
    session_id: str | None = Query(default=None),
    project_id: str | None = Query(default=None),
    context_lines: int = Query(default=3, ge=0, le=20),
) -> dict[str, Any]:
    resolved_project_id, workspace_root = await resolve_workspace_scope(
        request,
        session_id=session_id,
        project_id=project_id,
    )
    try:
        target = resolve_path_in_workspace(path, workspace_root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    repo_root = _find_git_repo_root(target.parent if target.suffix else target)
    if repo_root is None:
        raise HTTPException(
            status_code=404, detail="Workspace is not inside a git repository."
        )

    items = await asyncio.to_thread(
        _collect_workspace_changes,
        repo_root,
        workspace_root,
        target,
        True,
        context_lines,
    )
    if not items:
        raise HTTPException(
            status_code=404, detail="No diff available for the requested path."
        )

    item = items[0]
    return {
        "type": "workspace_diff",
        "session_id": session_id or "draft-session",
        "project_id": resolved_project_id,
        "workspace_dir": str(workspace_root),
        "repo_root": str(repo_root),
        **item,
    }


@router.post("/patch/preview")
async def preview_workspace_patch(
    request: Request,
    payload: WorkspacePatchPreviewRequest,
) -> dict[str, Any]:
    resolved_project_id, workspace_root = await resolve_workspace_scope(
        request,
        session_id=payload.session_id,
        project_id=payload.project_id,
        approval_id=payload.approval_id,
    )
    config = await _get_config(request.app)
    runtime_store = await _get_runtime_store(request)
    approval: dict[str, Any] | None = None
    task: dict[str, Any] | None = None
    if payload.approval_id:
        approval = await runtime_store.get_approval_request(payload.approval_id)
        if approval is None:
            raise HTTPException(status_code=404, detail="Approval not found")
        task_id = approval.get("task_id")
        task = (
            await runtime_store.get_task_run(task_id)
            if isinstance(task_id, str) and task_id
            else None
        )

    requester_id = (
        str(approval.get("requester_id") or f"runtime-task:{approval.get('task_id')}")
        if approval is not None
        else "workspace-preview"
    )
    session_id = (
        str(task.get("session_id") or "draft-session")
        if task is not None
        else payload.session_id or "draft-session"
    )
    task_id = (
        str(approval.get("task_id"))
        if approval is not None and approval.get("task_id")
        else None
    )
    try:
        _, change_payload, contract = await prepare_patch_change_contract(
            runtime_store=runtime_store,
            patch=payload.patch,
            workspace_dir=workspace_root,
            security=config.security,
            requester_id=requester_id,
            session_id=session_id,
            task_id=task_id,
            encoding=payload.encoding,
        )
    except PatchValidationError as exc:
        return {
            "type": "workspace_patch_preview",
            "session_id": payload.session_id or "draft-session",
            "project_id": resolved_project_id,
            "workspace_dir": str(workspace_root),
            "valid": False,
            "summary": None,
            "patch_text": payload.patch,
            "editable_patch_text": payload.patch,
            "file_changes": [],
            "change_count": 0,
            "paths": [],
            "diff_available": False,
            "errors": [str(exc)],
            "validation_errors": [str(exc)],
            "warnings": [],
            "change_contract_mode": config.security.change_contract_mode,
            "change_set_id": None,
            "request_digest": None,
            "expires_at": None,
            "policy_version": file_change_policy_version(config.security),
            "replacement_approval_id": None,
            "approval_state": "invalid",
            "would_reject_edited_patch": False,
            "content_unavailable_reason": exc.metadata.get("content_kind"),
            "suggested_tool": exc.metadata.get("suggested_tool"),
        }

    replacement_approval_id: str | None = None
    approval_state = "preview_only"
    would_reject_edited_patch = False
    if approval is not None:
        approval_metadata = approval.get("metadata")
        approval_metadata = (
            approval_metadata if isinstance(approval_metadata, dict) else {}
        )
        current_approval_state = str(approval.get("status") or "pending")
        stored_arguments = approval.get("arguments")
        stored_arguments = (
            stored_arguments if isinstance(stored_arguments, dict) else {}
        )
        stored_patch = stored_arguments.get("patch")
        patch_changed = (
            not isinstance(stored_patch, str) or stored_patch != payload.patch
        )
        if config.security.change_contract_mode == "observe":
            would_reject_edited_patch = patch_changed
            approval_state = "shadow_preview"
        elif not patch_changed:
            approval_state = current_approval_state
        elif current_approval_state == "superseded":
            existing_replacement_id = approval_metadata.get("superseded_by_approval_id")
            existing_replacement = (
                await runtime_store.get_approval_request(existing_replacement_id)
                if isinstance(existing_replacement_id, str)
                else None
            )
            if (
                existing_replacement is None
                or existing_replacement.get("request_digest")
                != contract["request_digest"]
            ):
                raise HTTPException(
                    status_code=409,
                    detail="Approval was superseded by another patch preview.",
                )
            replacement_approval_id = existing_replacement_id
            approval_state = "replacement_pending"
        else:
            replacement_approval_id = str(uuid4())
            replacement_metadata = {
                **(
                    dict(approval.get("metadata"))
                    if isinstance(approval.get("metadata"), dict)
                    else {}
                ),
                **change_payload,
                **contract,
                "approval_state": "replacement_pending",
            }
            try:
                await runtime_store.supersede_and_create_approval_request(
                    str(approval["id"]),
                    replacement_approval_id=replacement_approval_id,
                    tool_name="apply_patch",
                    arguments={"patch": payload.patch, "encoding": payload.encoding},
                    metadata=replacement_metadata,
                    requester_id=requester_id,
                    request_digest=str(contract["request_digest"]),
                    context_digest=str(contract["context_digest"]),
                    expires_at=str(contract["expires_at"]),
                )
            except ApprovalConflict as exc:
                raise HTTPException(status_code=409, detail=str(exc)) from exc
            approval_state = "replacement_pending"

    warnings: list[str] = []
    if would_reject_edited_patch:
        warnings.append(
            "Observe mode executed the legacy edited patch; enforce mode would "
            "require the replacement approval."
        )
    return {
        "type": "workspace_patch_preview",
        "session_id": payload.session_id or session_id,
        "project_id": resolved_project_id,
        "workspace_dir": str(workspace_root),
        "valid": True,
        "summary": (
            "1 file change prepared."
            if int(change_payload.get("change_count") or 0) == 1
            else f"{int(change_payload.get('change_count') or 0)} file changes prepared."
        ),
        "patch_text": payload.patch,
        **change_payload,
        **contract,
        "replacement_approval_id": replacement_approval_id,
        "approval_state": approval_state,
        "would_reject_edited_patch": would_reject_edited_patch,
        "errors": [],
        "validation_errors": [],
        "warnings": warnings,
    }


async def resolve_workspace_scope(
    request: Request,
    *,
    session_id: str | None,
    project_id: str | None,
    approval_id: str | None = None,
) -> tuple[str | None, Path]:
    """Resolve the effective session/project workspace for one request."""
    if approval_id:
        approval_scope = await _resolve_workspace_scope_from_approval(
            request, approval_id
        )
        if approval_scope is not None:
            return approval_scope
        raise HTTPException(status_code=404, detail="Approval not found")

    config = await _get_config(request.app)
    resolver = ExecutionScopeResolver(
        default_workspace_dir=str(config.workspace_dir),
        session_store=await _get_session_store(request),
        project_store=_get_project_store(request.app, config=config),
    )
    try:
        scope = await resolver.resolve(
            session_id=session_id or "draft-session",
            project_id=project_id,
        )
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return scope.project_id, normalize_workspace_dir(scope.workspace_dir)


async def _get_session_store(request: Request) -> SessionStore:
    config = await _get_config(request.app)
    return resolve_route_session_store(request.app, config)


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


async def _resolve_workspace_scope_from_approval(
    request: Request,
    approval_id: str,
) -> tuple[str | None, Path] | None:
    store = await _get_runtime_store(request)
    approval = await store.get_approval_request(approval_id)
    if approval is None:
        return None
    task_id = approval.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        return None
    task = await store.get_task_run(task_id)
    if task is None:
        return None
    workspace_dir = (
        task.get("task_workspace_dir")
        or task.get("project_workspace_dir")
        or task.get("workspace_dir")
    )
    if not isinstance(workspace_dir, str) or not workspace_dir.strip():
        return None
    return task.get("project_id"), normalize_workspace_dir(workspace_dir)


def _coerce_workspace_browse_directory(
    workspace_root: Path,
    raw_path: str | None,
    read_scope: str,
) -> tuple[Path, Path | None]:
    if raw_path is None or not raw_path.strip():
        return workspace_root, None

    try:
        requested = resolve_path_with_scope(raw_path, workspace_root, read_scope)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if requested.exists():
        if requested.is_dir():
            return requested, None
        if requested.is_file():
            return requested.parent, requested
        raise HTTPException(status_code=400, detail="Path is not a directory")

    parent = requested.parent
    if (
        (read_scope == "any" or is_path_within_workspace(parent, workspace_root))
        and parent.exists()
        and parent.is_dir()
    ):
        return parent, requested
    raise HTTPException(status_code=404, detail="Path not found")


def _resolve_workspace_file_target(
    workspace_root: Path,
    raw_path: str,
    read_scope: str,
) -> Path:
    try:
        target = resolve_path_with_scope(raw_path, workspace_root, read_scope)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return target


def _resolve_workspace_path_filter(
    workspace_root: Path, raw_path: str | None
) -> Path | None:
    if raw_path is None or not raw_path.strip():
        return None
    try:
        return resolve_path_in_workspace(raw_path, workspace_root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


def _relative_path(workspace_root: Path, path: Path) -> str:
    try:
        relative = path.resolve(strict=False).relative_to(workspace_root)
    except ValueError:
        return path.name
    return "." if not relative.parts else relative.as_posix()


def _find_git_repo_root(start_path: Path) -> Path | None:
    current = start_path.resolve(strict=False)
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    return None


def _collect_workspace_changes(
    repo_root: Path,
    workspace_root: Path,
    filter_path: Path | None,
    include_diff: bool,
    context_lines: int,
) -> list[dict[str, Any]]:
    # Rename/copy detection needs both sides of a status record.  Restricting
    # Git to only the requested destination path degrades a type-2 record into
    # an ordinary add/delete pair, so collect the workspace and filter after
    # parsing the NUL-safe records.
    pathspec = _git_pathspec(repo_root, workspace_root)
    if pathspec is None:
        return []

    result = _run_git_bytes(
        repo_root,
        "status",
        "--porcelain=v2",
        "-z",
        "--untracked-files=all",
        "--",
        pathspec,
    )
    if result.returncode != 0:
        return []

    has_head = _run_git(repo_root, "rev-parse", "--verify", "HEAD").returncode == 0
    items: list[dict[str, Any]] = []
    for parsed in _parse_git_status_porcelain_v2(result.stdout):
        current_abs = (repo_root / parsed["repo_path"]).resolve(strict=False)
        if not is_path_within_workspace(current_abs, workspace_root):
            continue
        if filter_path is not None and not _path_matches_filter(
            current_abs, filter_path
        ):
            continue

        diff_payload = _build_workspace_diff_payload(
            repo_root=repo_root,
            workspace_root=workspace_root,
            current_abs=current_abs,
            current_repo_path=parsed["repo_path"],
            baseline_repo_path=parsed.get("baseline_repo_path"),
            status=parsed["status"],
            include_diff=include_diff,
            context_lines=context_lines,
            has_head=has_head,
        )
        item = {
            "path": str(current_abs),
            "relative_path": _relative_path(workspace_root, current_abs),
            "status": parsed["status"],
            "staged": parsed["staged"],
            "added_lines": diff_payload["added_lines"],
            "deleted_lines": diff_payload["deleted_lines"],
            "diff_available": diff_payload["diff_available"],
            "binary": diff_payload["binary"],
            "encoding": diff_payload["encoding"],
            "newline_style": diff_payload["newline_style"],
            "eof_newline": diff_payload["eof_newline"],
            "mode_before": parsed.get("mode_before"),
            "mode_after": parsed.get("mode_after"),
            "rename_source": parsed.get("baseline_repo_path"),
            "copy_source": (
                parsed.get("baseline_repo_path")
                if parsed.get("status") == "copied"
                else None
            ),
            "content_unavailable_reason": diff_payload["content_unavailable_reason"],
        }
        if include_diff:
            item["diff"] = diff_payload["diff"]
            item["original_content"] = diff_payload["original_content"]
            item["new_content"] = diff_payload["new_content"]
        items.append(item)

    items.sort(key=lambda item: str(item["relative_path"]).lower())
    return items


def _git_pathspec(repo_root: Path, target_path: Path) -> str | None:
    try:
        return (
            target_path.resolve(strict=False).relative_to(repo_root).as_posix() or "."
        )
    except ValueError:
        return None


def _decode_git_path(value: bytes) -> str:
    """Decode Git's NUL-delimited raw path without lossy replacement."""

    return value.decode("utf-8", errors="surrogateescape")


def _parse_git_mode(value: bytes) -> int | None:
    if value in {b"", b"000000"}:
        return None
    try:
        return int(value, 8)
    except ValueError:
        return None


def _parse_git_status_porcelain_v2(payload: bytes) -> list[dict[str, Any]]:
    """Parse ``git status --porcelain=v2 -z`` records.

    Type-2 rename/copy records carry their original path in the following NUL
    field.  Parsing bytes and record types avoids path quoting and the unsafe
    historical ``" -> "`` filename heuristic.
    """

    fields = payload.split(b"\0")
    items: list[dict[str, Any]] = []
    index = 0
    while index < len(fields):
        record = fields[index]
        index += 1
        if not record:
            continue
        kind = record[:1]
        if kind == b"#" or kind == b"!":
            continue
        if kind == b"?":
            items.append(
                {
                    "repo_path": _decode_git_path(record[2:]),
                    "baseline_repo_path": None,
                    "status": "untracked",
                    "staged": False,
                    "mode_before": None,
                    "mode_after": None,
                }
            )
            continue
        if kind == b"1":
            parts = record.split(b" ", 8)
            if len(parts) != 9:
                continue
            xy = parts[1].decode("ascii", errors="strict")
            items.append(
                {
                    "repo_path": _decode_git_path(parts[8]),
                    "baseline_repo_path": None,
                    "status": _normalize_git_status(xy[0], xy[1]),
                    "staged": xy[0] != ".",
                    "mode_before": _parse_git_mode(parts[3]),
                    "mode_after": _parse_git_mode(
                        parts[4] if xy[0] != "." else parts[5]
                    ),
                }
            )
            continue
        if kind == b"2":
            parts = record.split(b" ", 9)
            if len(parts) != 10 or index >= len(fields):
                continue
            original_path = fields[index]
            index += 1
            xy = parts[1].decode("ascii", errors="strict")
            score = parts[8][:1]
            items.append(
                {
                    "repo_path": _decode_git_path(parts[9]),
                    "baseline_repo_path": _decode_git_path(original_path),
                    "status": "copied" if score == b"C" else "renamed",
                    "staged": xy[0] != ".",
                    "mode_before": _parse_git_mode(parts[3]),
                    "mode_after": _parse_git_mode(
                        parts[4] if xy[0] != "." else parts[5]
                    ),
                }
            )
            continue
        if kind == b"u":
            parts = record.split(b" ", 10)
            if len(parts) == 11:
                items.append(
                    {
                        "repo_path": _decode_git_path(parts[10]),
                        "baseline_repo_path": None,
                        "status": "conflicted",
                        "staged": True,
                        "mode_before": _parse_git_mode(parts[3]),
                        "mode_after": _parse_git_mode(parts[6]),
                    }
                )
    return items


def _normalize_git_status(index_status: str, worktree_status: str) -> str:
    index_status = " " if index_status == "." else index_status
    worktree_status = " " if worktree_status == "." else worktree_status
    combined = f"{index_status}{worktree_status}"
    if combined == "??":
        return "untracked"
    if "U" in combined:
        return "conflicted"
    if "R" in combined:
        return "renamed"
    if "C" in combined:
        return "copied"
    if "D" in combined:
        return "deleted"
    if "A" in combined:
        return "added"
    if "M" in combined:
        return "modified"
    return "changed"


def _path_matches_filter(path: Path, filter_path: Path) -> bool:
    filter_resolved = filter_path.resolve(strict=False)
    path_resolved = path.resolve(strict=False)
    if filter_resolved == path_resolved:
        return True
    if filter_resolved.is_dir():
        try:
            path_resolved.relative_to(filter_resolved)
            return True
        except ValueError:
            return False
    return False


def _build_workspace_diff_payload(
    *,
    repo_root: Path,
    workspace_root: Path,
    current_abs: Path,
    current_repo_path: str,
    baseline_repo_path: str | None,
    status: str,
    include_diff: bool,
    context_lines: int,
    has_head: bool,
) -> dict[str, Any]:
    before_bytes: bytes | None = None
    after_bytes: bytes | None = None
    if has_head and status != "untracked":
        before_bytes = _read_git_blob_bytes(
            repo_root,
            baseline_repo_path or current_repo_path,
        )
    if status != "deleted" and current_abs.exists() and current_abs.is_file():
        after_bytes = current_abs.read_bytes()

    before_fidelity = _inspect_content_fidelity(before_bytes)
    after_fidelity = _inspect_content_fidelity(after_bytes)
    selected_fidelity = after_fidelity if after_bytes is not None else before_fidelity
    binary = bool(before_fidelity["binary"] or after_fidelity["binary"])
    non_utf8 = any(
        value is not None and fidelity["encoding"] == "non-utf8"
        for value, fidelity in (
            (before_bytes, before_fidelity),
            (after_bytes, after_fidelity),
        )
    )

    pathspecs = [current_repo_path]
    if baseline_repo_path and baseline_repo_path != current_repo_path:
        pathspecs.insert(0, baseline_repo_path)
    if status == "untracked":
        diff_result = _run_git_bytes(
            repo_root,
            "-c",
            "core.quotePath=false",
            "diff",
            "--no-index",
            "--no-color",
            "--binary",
            f"--unified={context_lines}",
            "--",
            os.devnull,
            str(current_abs),
        )
    else:
        diff_result = _run_git_bytes(
            repo_root,
            "-c",
            "core.quotePath=false",
            "diff",
            "--no-ext-diff",
            "--no-textconv",
            "--no-color",
            "--binary",
            "--full-index",
            "--find-renames",
            "--find-copies",
            f"--unified={context_lines}",
            "HEAD",
            "--",
            *pathspecs,
        )
    diff_bytes = diff_result.stdout
    unavailable_reason = "binary" if binary else "non_utf8" if non_utf8 else None
    diff_text: str | None = None
    if diff_bytes and unavailable_reason is None:
        try:
            diff_text = diff_bytes.decode("utf-8", errors="strict").rstrip("\n")
        except UnicodeDecodeError:
            unavailable_reason = "non_utf8"

    added_lines, deleted_lines = _count_git_diff_lines(diff_bytes, binary=binary)
    return {
        "path": str(current_abs),
        "relative_path": _relative_path(workspace_root, current_abs),
        "status": status,
        "added_lines": added_lines,
        "deleted_lines": deleted_lines,
        "diff_available": diff_text is not None,
        "diff": diff_text if include_diff else None,
        "original_content": (
            _decode_utf8_content(before_bytes, before_fidelity)
            if include_diff
            else None
        ),
        "new_content": (
            _decode_utf8_content(after_bytes, after_fidelity) if include_diff else None
        ),
        "binary": binary,
        "encoding": selected_fidelity["encoding"],
        "newline_style": selected_fidelity["newline_style"],
        "eof_newline": selected_fidelity["eof_newline"],
        "content_unavailable_reason": unavailable_reason,
    }


def _inspect_content_fidelity(content: bytes | None) -> dict[str, Any]:
    if content is None:
        return {
            "binary": False,
            "encoding": None,
            "newline_style": None,
            "eof_newline": None,
        }
    binary = b"\0" in content
    if binary:
        encoding = "binary"
    else:
        try:
            content.decode(
                "utf-8-sig" if content.startswith(b"\xef\xbb\xbf") else "utf-8"
            )
            encoding = "utf-8-bom" if content.startswith(b"\xef\xbb\xbf") else "utf-8"
        except UnicodeDecodeError:
            encoding = "non-utf8"
    crlf = content.count(b"\r\n")
    without_crlf = content.replace(b"\r\n", b"")
    lf = without_crlf.count(b"\n")
    cr = without_crlf.count(b"\r")
    kinds = sum(value > 0 for value in (crlf, lf, cr))
    newline_style = (
        "mixed"
        if kinds > 1
        else "crlf" if crlf else "lf" if lf else "cr" if cr else "none"
    )
    return {
        "binary": binary,
        "encoding": encoding,
        "newline_style": newline_style,
        "eof_newline": content.endswith((b"\n", b"\r")),
    }


def _decode_utf8_content(content: bytes | None, fidelity: dict[str, Any]) -> str | None:
    if content is None or fidelity["binary"] or fidelity["encoding"] == "non-utf8":
        return None
    encoding = "utf-8-sig" if fidelity["encoding"] == "utf-8-bom" else "utf-8"
    return content.decode(encoding, errors="strict")


def _count_git_diff_lines(diff: bytes, *, binary: bool) -> tuple[int, int]:
    if binary:
        return 0, 0
    additions = 0
    deletions = 0
    in_hunk = False
    for line in diff.splitlines():
        if line.startswith(b"@@ "):
            in_hunk = True
            continue
        if not in_hunk:
            continue
        if line.startswith(b"+"):
            additions += 1
        elif line.startswith(b"-"):
            deletions += 1
    return additions, deletions


def _read_git_blob_bytes(repo_root: Path, repo_path: str) -> bytes:
    result = _run_git_bytes(repo_root, "show", f"HEAD:{repo_path}")
    if result.returncode != 0:
        return b""
    return result.stdout


def _run_git(repo_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _run_git_bytes(repo_root: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-C", str(repo_root), *args],
        check=False,
        capture_output=True,
        text=False,
    )
