"""Goal runtime API routes."""

from __future__ import annotations

import inspect
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from mochi.api.routes.approvals import _get_runtime_service
from mochi.goal_intent import classify_goal_proposal_follow_up_intent
from mochi.goal_proposal_copy import (
    build_goal_follow_up_assistant_copy_fallback,
    build_goal_proposal_assistant_copy_fallback,
    generate_goal_follow_up_assistant_copy,
    generate_goal_proposal_assistant_copy,
)
from mochi.runtime.goal_strategy_registry import (
    default_goal_strategy_entry,
    list_goal_strategy_entries,
)
from mochi.runtime.models import (
    ActiveGoalTurnDecision,
    ActiveGoalTurnDecisionRequest,
    GoalCreateRequest,
    GoalResponse,
    GoalStrategyRegistryEntry,
    GoalStrategyRegistryResponse,
)

router = APIRouter(prefix="/v1/goals")


class GoalResumeRequest(BaseModel):
    """Request payload for resuming a Goal, optionally resolving an approval first."""

    strategy: Literal["continue_from_checkpoint", "restart_attempt"] = (
        "continue_from_checkpoint"
    )
    guidance_message: str | None = None
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


class PendingGoalProposalIntentRequest(BaseModel):
    """Bounded intent classification request for a pending goal proposal follow-up."""

    message: str
    proposal_objective: str
    execution_mode: Literal["single_agent", "workflow"]


class PendingGoalProposalIntentResponse(BaseModel):
    """Bounded intent classification result for a pending goal proposal follow-up."""

    type: Literal["goal_pending_proposal_intent"] = "goal_pending_proposal_intent"
    intent: Literal["confirm_start", "revise_proposal", "exit_goal_lane", "ambiguous"]
    confidence: float | None = None
    rationale: str


class GoalProposalAssistantCopyRequest(BaseModel):
    """Bounded assistant-copy request for a goal proposal explanation."""

    message: str
    proposal_objective: str
    execution_mode: Literal["single_agent", "workflow"]
    protocol_selection: str | None = None
    role_summary: str | None = None
    runtime_mode: str | None = None
    revision_index: int = 0


class GoalProposalAssistantCopyResponse(BaseModel):
    """Assistant explanation text for a goal proposal."""

    type: Literal["goal_proposal_assistant_copy"] = "goal_proposal_assistant_copy"
    explanation: str
    source: Literal["model", "fallback"]


class GoalFollowUpAssistantCopyRequest(BaseModel):
    """Bounded assistant-copy request for an active goal follow-up reply."""

    message: str
    kind: Literal[
        "queued_after_resolution",
        "manual_resolution_required",
        "blocked",
        "no_live_attempt",
        "restarted_forwarded",
        "refreshed_forwarded",
        "resumed_forwarded",
        "forwarded",
    ]
    goal_objective: str
    goal_status: str
    linked_run_status: str | None = None
    continuation_action: str | None = None
    continuation_summary: str | None = None
    approval_count: int = 0
    tool_names: list[str] | None = None
    operator_control_hint: str | None = None
    recommended_action: str | None = None
    latest_error: str | None = None


class GoalFollowUpAssistantCopyResponse(BaseModel):
    """Assistant explanation text for an active goal follow-up reply."""

    type: Literal["goal_follow_up_assistant_copy"] = "goal_follow_up_assistant_copy"
    explanation: str
    source: Literal["model", "fallback"]


async def _get_or_create_goal_intent_engine(request: Request) -> Any:
    # Import lazily to avoid module-import cycles while still reusing the canonical app helper.
    from mochi.api.server import _get_or_create_engine

    return await _get_or_create_engine(request.app)


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


@router.get("/strategies", response_model=GoalStrategyRegistryResponse)
async def list_goal_strategies() -> GoalStrategyRegistryResponse:
    entries = [
        GoalStrategyRegistryEntry.model_validate(entry.to_dict())
        for entry in list_goal_strategy_entries()
    ]
    return GoalStrategyRegistryResponse(
        default_strategy_id=default_goal_strategy_entry().id,
        entries=entries,
    )


@router.post("/pending-proposal-intent", response_model=PendingGoalProposalIntentResponse)
async def classify_pending_goal_proposal_intent(
    request: Request,
    payload: PendingGoalProposalIntentRequest,
) -> PendingGoalProposalIntentResponse:
    engine = await _get_or_create_goal_intent_engine(request)
    invoke = getattr(engine, "invoke", None)
    if not callable(invoke):
        raise HTTPException(
            status_code=501,
            detail="Engine does not support bounded goal intent classification.",
        )
    result = await classify_goal_proposal_follow_up_intent(
        engine,
        user_message=payload.message,
        proposal_objective=payload.proposal_objective,
        execution_mode=payload.execution_mode,
    )
    return PendingGoalProposalIntentResponse(
        intent=result.intent,
        confidence=result.confidence,
        rationale=result.rationale,
    )


@router.post("/proposal-assistant-copy", response_model=GoalProposalAssistantCopyResponse)
async def build_goal_proposal_assistant_copy(
    request: Request,
    payload: GoalProposalAssistantCopyRequest,
) -> GoalProposalAssistantCopyResponse:
    engine = await _get_or_create_goal_intent_engine(request)
    invoke = getattr(engine, "invoke", None)
    if not callable(invoke):
        return GoalProposalAssistantCopyResponse(
            explanation=build_goal_proposal_assistant_copy_fallback(
                user_message=payload.message,
                proposal_objective=payload.proposal_objective,
                execution_mode=payload.execution_mode,
                protocol_selection=payload.protocol_selection,
                revision_index=payload.revision_index,
            ),
            source="fallback",
        )
    result = await generate_goal_proposal_assistant_copy(
        engine,
        user_message=payload.message,
        proposal_objective=payload.proposal_objective,
        execution_mode=payload.execution_mode,
        protocol_selection=payload.protocol_selection,
        role_summary=payload.role_summary,
        runtime_mode=payload.runtime_mode,
        revision_index=payload.revision_index,
    )
    return GoalProposalAssistantCopyResponse(
        explanation=result.explanation,
        source=result.source,
    )


@router.post("/follow-up-assistant-copy", response_model=GoalFollowUpAssistantCopyResponse)
async def build_goal_follow_up_assistant_copy(
    request: Request,
    payload: GoalFollowUpAssistantCopyRequest,
) -> GoalFollowUpAssistantCopyResponse:
    engine = await _get_or_create_goal_intent_engine(request)
    invoke = getattr(engine, "invoke", None)
    if not callable(invoke):
        return GoalFollowUpAssistantCopyResponse(
            explanation=build_goal_follow_up_assistant_copy_fallback(
                user_message=payload.message,
                kind=payload.kind,
                summary=payload.continuation_summary,
                approval_count=payload.approval_count,
                tool_names=payload.tool_names,
                operator_control_hint=payload.operator_control_hint,
            ),
            source="fallback",
        )
    result = await generate_goal_follow_up_assistant_copy(
        engine,
        user_message=payload.message,
        kind=payload.kind,
        goal_objective=payload.goal_objective,
        goal_status=payload.goal_status,
        linked_run_status=payload.linked_run_status,
        continuation_action=payload.continuation_action,
        continuation_summary=payload.continuation_summary,
        approval_count=payload.approval_count,
        tool_names=payload.tool_names,
        operator_control_hint=payload.operator_control_hint,
        recommended_action=payload.recommended_action,
        latest_error=payload.latest_error,
    )
    return GoalFollowUpAssistantCopyResponse(
        explanation=result.explanation,
        source=result.source,
    )


@router.post("/{goal_id}/turn-decision", response_model=ActiveGoalTurnDecision)
async def decide_active_goal_turn(
    request: Request,
    goal_id: str,
    payload: ActiveGoalTurnDecisionRequest,
) -> ActiveGoalTurnDecision:
    service = await _get_runtime_service(request.app)
    decision = await service.decide_active_goal_turn(goal_id, payload)
    if decision is None:
        raise HTTPException(status_code=404, detail="Goal not found")
    return decision


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
            queued_guidance_message = str(payload.guidance_message or "").strip()
            if queued_guidance_message:
                queue_guidance = getattr(service, "_queue_agent_run_guidance_message", None)
                if callable(queue_guidance):
                    await queue_guidance(linked_run_id, queued_guidance_message)
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
        guidance_message=payload.guidance_message if payload is not None else None,
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
