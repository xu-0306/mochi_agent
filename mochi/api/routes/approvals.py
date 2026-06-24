"""Approval API routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from fastapi import APIRouter, HTTPException, Request
from fastapi import FastAPI

from mochi.api.server import _get_config, _get_or_create_engine
from mochi.runtime.approvals import PersistentApprovalStore
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
    try:
        approval = await service.resolve_approval(
            approval_id,
            decision=payload.decision,
            reason=payload.reason,
            rule=payload.rule,
            replay_override=payload.replay_override.model_dump() if payload.replay_override is not None else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if approval is None:
        raise HTTPException(status_code=404, detail="Approval not found")
    return approval


@router.get("/approvals/{approval_id}/exec-session")
async def get_approval_exec_session(
    request: Request,
    approval_id: str,
    yield_time_ms: int | None = None,
) -> dict[str, Any]:
    service = await _get_runtime_service(request.app)
    payload = await service.get_approval_exec_session(
        approval_id,
        yield_time_ms=yield_time_ms,
    )
    if isinstance(payload, tuple):
        if payload[0] == "session_unavailable":
            raise HTTPException(status_code=409, detail="No live exec session available for this approval")
        raise HTTPException(status_code=404, detail="Exec session not available for this approval")
    return payload


@router.post("/approvals/{approval_id}/exec-session/stop")
async def stop_approval_exec_session(
    request: Request,
    approval_id: str,
) -> dict[str, Any]:
    service = await _get_runtime_service(request.app)
    payload = await service.stop_approval_exec_session(approval_id)
    if isinstance(payload, tuple):
        if payload[0] == "session_unavailable":
            raise HTTPException(status_code=409, detail="No live exec session available for this approval")
        raise HTTPException(status_code=404, detail="Exec session not available for this approval")
    return payload


async def _get_runtime_service(app: FastAPI) -> RuntimeService:
    existing = cast(RuntimeService | None, getattr(app.state, "runtime_service", None))
    config = await _get_config(app)
    if existing is not None:
        existing.update_security_config(config.security)
        existing.bind_app_config(config=config, config_path=getattr(app.state, "config_path", None))
        await existing.start()
        return existing

    engine = await _get_or_create_engine(app)
    store = RuntimeStore(Path(config.sessions_dir) / "runtime.db")
    await store.initialize()
    service = RuntimeService(
        engine=engine,
        store=store,
        exec_approval_store=PersistentApprovalStore(Path(config.sessions_dir) / "exec-approvals.db"),
    )
    service.update_security_config(config.security)
    service.bind_app_config(config=config, config_path=getattr(app.state, "config_path", None))
    service.set_runtime_tasks_root(Path(config.sessions_dir) / "runtime-tasks")
    await service.start()
    app.state.runtime_service = service
    return service
