"""Approval API routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from fastapi import APIRouter, FastAPI, HTTPException, Request

from mochi.api.server import _get_config, _get_or_create_engine
from mochi.api.session_store_binding import resolve_route_session_store
from mochi.runtime.active_goal_turn_selector import build_active_goal_turn_selector
from mochi.runtime.approvals import (
    ApprovalConflict,
    ApprovalExpired,
    ApprovalRequesterMismatch,
    PersistentApprovalStore,
)
from mochi.runtime.models import ApprovalResolution
from mochi.runtime.service import RuntimeService
from mochi.runtime.store import RuntimeStore
from mochi.runtime.ordinary_chat_session_gate import (
    OrdinaryChatSessionGate,
    OrdinaryChatSessionGateError,
)

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
    is_ordinary_chat, session_id = _ordinary_chat_approval_owner(service, approval_id)
    current_permission_policy = None
    if is_ordinary_chat:
        try:
            current_permission_policy = await _resolve_verified_ordinary_chat_policy(
                request.app,
                session_id,
            )
        except OrdinaryChatSessionGateError as exc:
            raise _ordinary_chat_session_validation_http_error(
                exc,
                operation="resolve",
            ) from exc
    try:
        approval = await service.resolve_approval(
            approval_id,
            decision=payload.decision,
            reason=payload.reason,
            rule=payload.rule,
            replay_override=payload.replay_override.model_dump() if payload.replay_override is not None else None,
            current_permission_policy=current_permission_policy,
        )
    except ApprovalExpired as exc:
        raise HTTPException(status_code=410, detail=str(exc)) from exc
    except ApprovalRequesterMismatch as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except ApprovalConflict as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if approval is None:
        raise HTTPException(status_code=404, detail="Approval not found")
    return approval


@router.post("/approvals/{approval_id}/reconcile")
async def reconcile_ordinary_chat_approval(
    request: Request,
    approval_id: str,
) -> dict[str, Any]:
    """Resume a recovered ordinary-Chat continuation with server policy only."""
    service = await _get_runtime_service(request.app)
    is_ordinary_chat, _session_id = _ordinary_chat_approval_owner(service, approval_id)
    if not is_ordinary_chat:
        raise HTTPException(
            status_code=404,
            detail={"code": "ordinary_chat_reconciliation_not_found"},
        )
    try:
        outcome = await service.reconcile_recovered_ordinary_chat_approval(
            approval_id=approval_id,
        )
    except OrdinaryChatSessionGateError as exc:
        raise _ordinary_chat_session_validation_http_error(
            exc,
            operation="reconciliation",
        ) from exc
    except ApprovalConflict as exc:
        raise HTTPException(
            status_code=409,
            detail={"code": "ordinary_chat_reconciliation_conflict", "message": str(exc)},
        ) from exc
    status = outcome.get("status")
    if status == "continued":
        return outcome
    reason = str(outcome.get("reason") or "reconciliation_unavailable")
    status_code = 404 if reason == "ordinary_chat_approval_missing" else 409
    code = reason if reason.startswith("ordinary_chat_") else f"ordinary_chat_{reason}"
    raise HTTPException(
        status_code=status_code,
        detail={"code": code},
    )


def _ordinary_chat_approval_owner(
    service: RuntimeService,
    approval_id: str,
) -> tuple[bool, str | None]:
    owner_lookup = getattr(service, "ordinary_chat_approval_owner", None)
    if callable(owner_lookup):
        is_ordinary_chat, session_id = owner_lookup(approval_id)
        return bool(is_ordinary_chat), session_id if isinstance(session_id, str) else None

    session_lookup = getattr(service, "ordinary_chat_approval_session_id", None)
    session_id = session_lookup(approval_id) if callable(session_lookup) else None
    return isinstance(session_id, str) and bool(session_id), session_id


def _ordinary_chat_session_validation_http_error(
    error: OrdinaryChatSessionGateError,
    *,
    operation: str,
) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail={"code": f"ordinary_chat_{operation}_session_{error.reason}"},
    )


async def _resolve_verified_ordinary_chat_policy(
    app: FastAPI,
    session_id: str | None,
) -> dict[str, Any]:
    """Use the shared strict ordinary-Chat session and policy gate."""
    config = await _get_config(app)
    session_store = resolve_route_session_store(app, config)
    gate = OrdinaryChatSessionGate(
        session_store=session_store,
        security=config.security,
    )
    return await gate.effective_policy(session_id)


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
        existing.update_sandbox_config(config.sandbox)
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
        active_goal_turn_selector=build_active_goal_turn_selector(engine),
    )
    service.update_security_config(config.security)
    service.update_sandbox_config(config.sandbox)
    service.bind_app_config(config=config, config_path=getattr(app.state, "config_path", None))
    service.set_runtime_tasks_root(Path(config.sessions_dir) / "runtime-tasks")
    await service.start()
    app.state.runtime_service = service
    return service
