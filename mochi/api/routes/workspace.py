"""Session-aware workspace browsing and diff API routes."""

from __future__ import annotations

import asyncio
import mimetypes
import subprocess
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import FileResponse

from mochi.api.routes.filesystem import _preview_docx, _preview_pdf, _preview_text_file
from mochi.api.routes.projects import _get_project_store
from mochi.api.server import _get_config
from mochi.projects.execution_scope import ExecutionScopeResolver
from mochi.sessions.store import SessionStore
from mochi.utils.security import is_path_within_workspace, normalize_workspace_dir, resolve_path_in_workspace

router = APIRouter(prefix="/v1/workspace", tags=["workspace"])


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
    current, selected_path = _coerce_workspace_browse_directory(workspace_root, path)

    try:
        entries = await asyncio.to_thread(lambda: list(current.iterdir()))
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="Permission denied") from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    items: list[dict[str, Any]] = []
    for entry in sorted(entries, key=lambda item: item.name.lower()):
        resolved_entry = entry.resolve(strict=False)
        if not is_path_within_workspace(resolved_entry, workspace_root):
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
    target = _resolve_workspace_file_target(workspace_root, path)
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
    target = _resolve_workspace_file_target(workspace_root, path)
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
        raise HTTPException(status_code=404, detail="Workspace is not inside a git repository.")

    items = await asyncio.to_thread(
        _collect_workspace_changes,
        repo_root,
        workspace_root,
        target,
        True,
        context_lines,
    )
    if not items:
        raise HTTPException(status_code=404, detail="No diff available for the requested path.")

    item = items[0]
    return {
        "type": "workspace_diff",
        "session_id": session_id or "draft-session",
        "project_id": resolved_project_id,
        "workspace_dir": str(workspace_root),
        "repo_root": str(repo_root),
        **item,
    }


async def resolve_workspace_scope(
    request: Request,
    *,
    session_id: str | None,
    project_id: str | None,
) -> tuple[str | None, Path]:
    """Resolve the effective session/project workspace for one request."""
    config = await _get_config(request.app)
    resolver = ExecutionScopeResolver(
        default_workspace_dir=str(getattr(config, "workspace_dir")),
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
    existing = getattr(request.app.state, "session_store", None)
    if isinstance(existing, SessionStore):
        return existing

    config = await _get_config(request.app)
    store = SessionStore(config.sessions_dir)
    request.app.state.session_store = store
    return store


def _coerce_workspace_browse_directory(
    workspace_root: Path,
    raw_path: str | None,
) -> tuple[Path, Path | None]:
    if raw_path is None or not raw_path.strip():
        return workspace_root, None

    try:
        requested = resolve_path_in_workspace(raw_path, workspace_root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if requested.exists():
        if requested.is_dir():
            return requested, None
        if requested.is_file():
            return requested.parent, requested
        raise HTTPException(status_code=400, detail="Path is not a directory")

    parent = requested.parent
    if is_path_within_workspace(parent, workspace_root) and parent.exists() and parent.is_dir():
        return parent, requested
    raise HTTPException(status_code=404, detail="Path not found")


def _resolve_workspace_file_target(workspace_root: Path, raw_path: str) -> Path:
    try:
        target = resolve_path_in_workspace(raw_path, workspace_root)
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")
    return target


def _resolve_workspace_path_filter(workspace_root: Path, raw_path: str | None) -> Path | None:
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
    pathspec = _git_pathspec(repo_root, filter_path or workspace_root)
    if pathspec is None:
        return []

    result = _run_git(
        repo_root,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--",
        pathspec,
    )
    if result.returncode != 0:
        return []

    has_head = _run_git(repo_root, "rev-parse", "--verify", "HEAD").returncode == 0
    items: list[dict[str, Any]] = []
    for raw_line in result.stdout.splitlines():
        parsed = _parse_git_status_line(raw_line)
        if parsed is None:
            continue
        current_abs = (repo_root / parsed["repo_path"]).resolve(strict=False)
        if not is_path_within_workspace(current_abs, workspace_root):
            continue
        if filter_path is not None and not _path_matches_filter(current_abs, filter_path):
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
        }
        if include_diff:
            item["diff"] = diff_payload["diff"]
        items.append(item)

    items.sort(key=lambda item: str(item["relative_path"]).lower())
    return items


def _git_pathspec(repo_root: Path, target_path: Path) -> str | None:
    try:
        return target_path.resolve(strict=False).relative_to(repo_root).as_posix() or "."
    except ValueError:
        return None


def _parse_git_status_line(raw_line: str) -> dict[str, Any] | None:
    if len(raw_line) < 4:
        return None

    index_status = raw_line[0]
    worktree_status = raw_line[1]
    path_text = raw_line[3:].strip()
    if not path_text:
        return None

    baseline_repo_path: str | None = None
    repo_path = path_text
    if " -> " in path_text:
        before, after = path_text.split(" -> ", 1)
        baseline_repo_path = before.strip()
        repo_path = after.strip()

    status = _normalize_git_status(index_status, worktree_status)
    return {
        "repo_path": repo_path,
        "baseline_repo_path": baseline_repo_path,
        "status": status,
        "staged": index_status not in {" ", "?"},
    }


def _normalize_git_status(index_status: str, worktree_status: str) -> str:
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
    import difflib

    before_text = ""
    after_text = ""
    if has_head and status != "untracked":
        before_text = _read_git_blob(
            repo_root,
            baseline_repo_path or current_repo_path,
        )
    if status != "deleted" and current_abs.exists() and current_abs.is_file():
        after_text = current_abs.read_text(encoding="utf-8", errors="replace")

    diff_lines = list(
        difflib.unified_diff(
            before_text.splitlines(),
            after_text.splitlines(),
            fromfile=f"a/{(baseline_repo_path or current_repo_path)}",
            tofile=f"b/{current_repo_path}",
            lineterm="",
            n=context_lines,
        )
    )
    diff_text = "\n".join(diff_lines) if diff_lines else None
    added_lines = sum(
        1
        for line in diff_lines
        if line.startswith("+") and not line.startswith("+++")
    )
    deleted_lines = sum(
        1
        for line in diff_lines
        if line.startswith("-") and not line.startswith("---")
    )
    return {
        "path": str(current_abs),
        "relative_path": _relative_path(workspace_root, current_abs),
        "status": status,
        "added_lines": added_lines,
        "deleted_lines": deleted_lines,
        "diff_available": diff_text is not None,
        "diff": diff_text if include_diff else None,
    }


def _read_git_blob(repo_root: Path, repo_path: str) -> str:
    result = _run_git(repo_root, "show", f"HEAD:{repo_path}")
    if result.returncode != 0:
        return ""
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
