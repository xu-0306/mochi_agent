"""Approval API routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from fastapi import APIRouter, HTTPException, Request
from fastapi import FastAPI

from mochi.api.server import _get_config, _get_or_create_engine
from mochi.runtime.models import ApprovalResolution
from mochi.runtime.service import RuntimeService
from mochi.runtime.store import RuntimeStore

router = APIRouter(prefix="/v1")


@router.get("/approvals")
async def list_approvals(request: Request, status: str | None = None) -> list[dict[str, Any]]:
    service = await _get_runtime_service(request.app)
    return await service.list_approvals(status=status)


@router.post("/approvals/{approval_id}/resolve")
async def resolve_approval(
    request: Request,
    approval_id: str,
    payload: ApprovalResolution,
) -> dict[str, Any]:
    service = await _get_runtime_service(request.app)
    approval = await service.resolve_approval(
        approval_id,
        approved=payload.approved,
        reason=payload.reason,
    )
    if approval is None:
        raise HTTPException(status_code=404, detail="Approval not found")
    return approval


async def _get_runtime_service(app: FastAPI) -> RuntimeService:
    existing = cast(RuntimeService | None, getattr(app.state, "runtime_service", None))
    config = await _get_config(app)
    if existing is not None:
        existing.update_security_config(config.security)
        await existing.start()
        return existing

    engine = await _get_or_create_engine(app)
    store = RuntimeStore(Path(config.sessions_dir) / "runtime.db")
    await store.initialize()
    service = RuntimeService(engine=engine, store=store)
    service.update_security_config(config.security)
    service.set_runtime_tasks_root(Path(config.sessions_dir) / "runtime-tasks")
    await service.start()
    app.state.runtime_service = service
    return service
