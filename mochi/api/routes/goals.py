"""Goal runtime API routes."""

from __future__ import annotations

import inspect
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from mochi.api.routes.approvals import _get_runtime_service
from mochi.runtime.models import GoalCreateRequest, GoalResponse

router = APIRouter(prefix="/v1/goals")


class GoalResumeRequest(BaseModel):
    """Request payload for resuming a Goal, optionally resolving an approval first."""

    strategy: Literal["continue_from_checkpoint", "restart_attempt"] = (
        "continue_from_checkpoint"
    )
    approval_id: str | None = None
    decision: Literal["approve_once", "approve_and_save_rule", "reject"] = "approve_once"
    reason: str | None = None
    rule: dict[str, Any] | None = None


class GoalRefreshRequest(BaseModel):
    """Request payload for refreshing a running Goal onto a fresh worker generation."""

    strategy: Literal["continue_from_checkpoint", "restart_attempt"] | None = None


class GoalRetryFailedShardRequest(BaseModel):
    """Request payload for retrying one failed collector shard on the current goal run."""

    shard_id: str | None = None
    strategy: Literal["continue_from_checkpoint", "restart_attempt"] | None = "continue_from_checkpoint"


class GoalEstopUpdateRequest(BaseModel):
    """Request payload for updating persistent goal emergency-stop controls."""

    stop_all_goals: bool | None = None
    blocked_tools: list[str] | None = None
    blocked_domains: list[str] | None = None
    block_network_usage: bool | None = None
    reason: str | None = None


def _current_goal_attempt(goal: dict[str, Any]) -> dict[str, Any] | None:
    attempts = goal.get("attempts")
    if not isinstance(attempts, list) or not attempts:
        return None
    current_attempt_id = goal.get("current_attempt_id")
    if isinstance(current_attempt_id, str) and current_attempt_id.strip():
        for attempt in attempts:
            if not isinstance(attempt, dict):
                continue
            if str(attempt.get("attempt_id") or "") == current_attempt_id:
                return attempt
    for attempt in reversed(attempts):
        if isinstance(attempt, dict):
            return attempt
    return None


async def _resume_linked_agent_run(
    service: Any,
    run_id: str,
    *,
    strategy: str,
) -> dict[str, Any] | None:
    resume_agent_run = service.resume_agent_run
    try:
        supports_strategy = "strategy" in inspect.signature(resume_agent_run).parameters
    except (TypeError, ValueError):
        supports_strategy = False
    if supports_strategy:
        return await resume_agent_run(run_id, strategy=strategy)
    return await resume_agent_run(run_id)


@router.post("", response_model=GoalResponse)
async def create_goal(
    request: Request,
    payload: GoalCreateRequest,
) -> GoalResponse:
    service = await _get_runtime_service(request.app)
    return GoalResponse.model_validate(await service.create_goal(payload))


@router.get("", response_model=list[GoalResponse])
async def list_goals(request: Request) -> list[GoalResponse]:
    service = await _get_runtime_service(request.app)
    goals = await service.list_goals()
    return [GoalResponse.model_validate(goal) for goal in goals]


@router.get("/estop")
async def get_goal_estop(request: Request) -> dict[str, object]:
    service = await _get_runtime_service(request.app)
    return await service.get_goal_operator_controls()


@router.post("/estop")
async def update_goal_estop(
    request: Request,
    payload: GoalEstopUpdateRequest,
) -> dict[str, object]:
    service = await _get_runtime_service(request.app)
    return await service.update_goal_operator_controls(
        stop_all_goals=payload.stop_all_goals,
        blocked_tools=payload.blocked_tools,
        blocked_domains=payload.blocked_domains,
        block_network_usage=payload.block_network_usage,
        reason=payload.reason,
    )


@router.get("/operator-audit-log")
async def list_goal_operator_audit_log(
    request: Request,
    event_type: str | None = None,
    goal_id: str | None = None,
    limit: int | None = None,
) -> list[dict[str, object]]:
    service = await _get_runtime_service(request.app)
    return await service.list_goal_operator_audit_log(
        event_type=event_type,
        goal_id=goal_id,
        limit=limit,
    )


@router.get("/{goal_id}", response_model=GoalResponse)
async def get_goal(request: Request, goal_id: str) -> GoalResponse:
    service = await _get_runtime_service(request.app)
    goal = await service.get_goal(goal_id)
    if goal is None:
        raise HTTPException(status_code=404, detail="Goal not found")
    return GoalResponse.model_validate(goal)


@router.get("/{goal_id}/health")
async def get_goal_health(request: Request, goal_id: str) -> dict[str, object]:
    service = await _get_runtime_service(request.app)
    payload = await service.get_goal_health(goal_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Goal not found")
    return payload


@router.get("/{goal_id}/checkpoints")
async def list_goal_checkpoints(
    request: Request,
    goal_id: str,
    attempt_id: str | None = None,
    limit: int | None = None,
) -> list[dict[str, object]]:
    service = await _get_runtime_service(request.app)
    checkpoints = await service.list_goal_checkpoints(
        goal_id,
        attempt_id=attempt_id,
        limit=limit,
    )
    if checkpoints is None:
        raise HTTPException(status_code=404, detail="Goal not found")
    return checkpoints


@router.get("/{goal_id}/memory-snapshots")
async def list_goal_memory_snapshots(
    request: Request,
    goal_id: str,
    attempt_id: str | None = None,
    limit: int | None = None,
) -> list[dict[str, object]]:
    service = await _get_runtime_service(request.app)
    snapshots = await service.list_goal_memory_snapshots(
        goal_id,
        attempt_id=attempt_id,
        limit=limit,
    )
    if snapshots is None:
        raise HTTPException(status_code=404, detail="Goal not found")
    return snapshots


@router.get("/{goal_id}/audit-findings")
async def list_goal_audit_findings(
    request: Request,
    goal_id: str,
    status: Literal["open", "resolved", "closed"] | None = None,
) -> list[dict[str, object]]:
    service = await _get_runtime_service(request.app)
    findings = await service.list_goal_audit_findings(goal_id, status=status)
    if findings is None:
        raise HTTPException(status_code=404, detail="Goal not found")
    return findings


@router.post("/{goal_id}/audit-findings/{finding_id}/resolve")
async def resolve_goal_audit_finding(
    request: Request,
    goal_id: str,
    finding_id: int,
) -> dict[str, object]:
    service = await _get_runtime_service(request.app)
    goal = await service.get_goal(goal_id)
    if goal is None:
        raise HTTPException(status_code=404, detail="Goal not found")
    finding = await service.resolve_goal_audit_finding(goal_id, finding_id)
    if finding is None:
        raise HTTPException(status_code=404, detail="Goal audit finding not found")
    return finding


@router.post("/{goal_id}/audit-findings/{finding_id}/close")
async def close_goal_audit_finding(
    request: Request,
    goal_id: str,
    finding_id: int,
) -> dict[str, object]:
    service = await _get_runtime_service(request.app)
    goal = await service.get_goal(goal_id)
    if goal is None:
        raise HTTPException(status_code=404, detail="Goal not found")
    finding = await service.close_goal_audit_finding(goal_id, finding_id)
    if finding is None:
        raise HTTPException(status_code=404, detail="Goal audit finding not found")
    return finding


@router.post("/{goal_id}/start", response_model=GoalResponse)
async def start_goal(request: Request, goal_id: str) -> GoalResponse:
    service = await _get_runtime_service(request.app)
    goal = await service.start_goal(goal_id)
    if goal is None:
        raise HTTPException(status_code=404, detail="Goal not found")
    return GoalResponse.model_validate(goal)


@router.post("/{goal_id}/pause", response_model=GoalResponse)
async def pause_goal(request: Request, goal_id: str) -> GoalResponse:
    service = await _get_runtime_service(request.app)
    goal = await service.pause_goal(goal_id)
    if goal is None:
        raise HTTPException(status_code=404, detail="Goal not found")
    return GoalResponse.model_validate(goal)


@router.post("/{goal_id}/resume", response_model=GoalResponse)
async def resume_goal(
    request: Request,
    goal_id: str,
    payload: GoalResumeRequest | None = None,
) -> GoalResponse:
    service = await _get_runtime_service(request.app)
    approval_id = (
        payload.approval_id.strip()
        if payload is not None and isinstance(payload.approval_id, str) and payload.approval_id.strip()
        else None
    )
    if approval_id is not None:
        current_goal = await service.get_goal(goal_id)
        if current_goal is None:
            raise HTTPException(status_code=404, detail="Goal not found")

        current_attempt = _current_goal_attempt(current_goal)
        linked_run_id = (
            str(current_attempt.get("agent_run_id") or "").strip()
            if isinstance(current_attempt, dict)
            else ""
        )

        try:
            approval = await service.resolve_approval(
                approval_id,
                decision=payload.decision,
                reason=payload.reason,
                rule=payload.rule,
                auto_resume_linked_run=False,
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if approval is None:
            raise HTTPException(status_code=404, detail="Approval not found")
        if payload.decision == "reject":
            refreshed_goal = await service.get_goal(goal_id)
            return GoalResponse.model_validate(refreshed_goal or current_goal)

        if linked_run_id:
            resumed_run = await _resume_linked_agent_run(
                service,
                linked_run_id,
                strategy=payload.strategy,
            )
            if resumed_run is None:
                raise HTTPException(status_code=404, detail="Linked agent run not found")
            refreshed_goal = await service.get_goal(goal_id)
            return GoalResponse.model_validate(refreshed_goal or current_goal)

    goal = await service.resume_goal(
        goal_id,
        strategy=payload.strategy if payload is not None else None,
    )
    if goal is None:
        raise HTTPException(status_code=404, detail="Goal not found")
    return GoalResponse.model_validate(goal)


@router.post("/{goal_id}/refresh", response_model=GoalResponse)
async def refresh_goal(
    request: Request,
    goal_id: str,
    payload: GoalRefreshRequest | None = None,
) -> GoalResponse:
    service = await _get_runtime_service(request.app)
    try:
        goal = await service.refresh_goal(
            goal_id,
            strategy=payload.strategy if payload is not None else None,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if goal is None:
        raise HTTPException(status_code=404, detail="Goal not found")
    return GoalResponse.model_validate(goal)


@router.post("/{goal_id}/retry-failed-shard", response_model=GoalResponse)
async def retry_goal_failed_shard(
    request: Request,
    goal_id: str,
    payload: GoalRetryFailedShardRequest | None = None,
) -> GoalResponse:
    service = await _get_runtime_service(request.app)
    try:
        goal = await service.retry_goal_failed_shard(
            goal_id,
            shard_id=(
                payload.shard_id.strip()
                if payload is not None and isinstance(payload.shard_id, str) and payload.shard_id.strip()
                else None
            ),
            strategy=payload.strategy if payload is not None else "continue_from_checkpoint",
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if goal is None:
        raise HTTPException(status_code=404, detail="Goal not found")
    return GoalResponse.model_validate(goal)


@router.post("/{goal_id}/cancel", response_model=GoalResponse)
async def cancel_goal(request: Request, goal_id: str) -> GoalResponse:
    service = await _get_runtime_service(request.app)
    goal = await service.cancel_goal(goal_id)
    if goal is None:
        raise HTTPException(status_code=404, detail="Goal not found")
    return GoalResponse.model_validate(goal)
