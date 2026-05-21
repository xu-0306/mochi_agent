"""File operation API routes (undo support)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field

from mochi.api.routes.projects import _get_project_store
from mochi.api.server import _get_config
from mochi.projects.execution_scope import ExecutionScopeResolver
from mochi.sessions.store import SessionStore
from mochi.utils.security import (
    content_size_bytes,
    resolve_path_with_scope,
    size_limit_bytes,
)

router = APIRouter(prefix="/v1/tools/file", tags=["file_ops"])


class FileUndoRequest(BaseModel):
    """Undo file write request."""

    file_path: str = Field(min_length=1)
    original_content: str | None = None
    session_id: str = Field(min_length=1)
    action: Literal["restore", "delete"] = "restore"
    encoding: str = "utf-8"


@router.post("/undo")
async def undo_file_write(request: Request, payload: FileUndoRequest) -> dict[str, str | int | None]:
    """Restore a file to its previous content or delete it when requested."""
    config = await _get_config(request.app)
    scope = config.security.file_ops_scope
    workspace_dir = await _resolve_workspace_dir_for_session(request, payload.session_id)

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
        }

    original_content = payload.original_content or ""
    limit_bytes = size_limit_bytes(config.security.file_undo_max_size_mb)
    if limit_bytes > 0 and content_size_bytes(original_content, encoding=payload.encoding) > limit_bytes:
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
    }


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
