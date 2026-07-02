"""Agent Run runtime API routes."""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import AsyncIterator
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from mochi.api.routes.approvals import _get_runtime_service
from mochi.runtime.models import (
    AgentRunAttemptPackageResponse,
    AgentRunCreateRequest,
    AgentRunDatasetPackageResponse,
    AgentRunGuidanceRequest,
    AgentRunMessageRequest,
    AgentRunSubagentMessageRequest,
    AgentRunResponse,
    ExecutionTranscriptEvent,
    SubagentTranscriptDetail,
    SubagentTranscriptSummary,
)

router = APIRouter(prefix="/v1/agent-runs")

_AGENT_RUN_EVENTS_STREAM_POLL_INTERVAL_SEC = 0.25
_AGENT_RUN_EVENTS_STREAM_HEARTBEAT_SEC = 1.0
_AGENT_RUN_EVENTS_STREAM_TIMEOUT_SEC = 5.0


class AgentRunResumeRequest(BaseModel):
    """Request payload for resuming an Agent Run, optionally resolving an approval first."""

    strategy: Literal["continue_from_checkpoint", "restart_attempt"] = (
        "continue_from_checkpoint"
    )
    approval_id: str | None = None
    decision: Literal["approve_once", "approve_and_save_rule", "reject"] = "approve_once"
    reason: str | None = None
    rule: dict[str, Any] | None = None


@router.post("", response_model=AgentRunResponse)
async def create_agent_run(
    request: Request,
    payload: AgentRunCreateRequest,
) -> AgentRunResponse:
    service = await _get_runtime_service(request.app)
    return AgentRunResponse.model_validate(await service.create_agent_run(payload))


@router.get("", response_model=list[AgentRunResponse])
async def list_agent_runs(request: Request) -> list[AgentRunResponse]:
    service = await _get_runtime_service(request.app)
    runs = await service.list_agent_runs()
    return [AgentRunResponse.model_validate(run) for run in runs]


@router.get("/{run_id}", response_model=AgentRunResponse)
async def get_agent_run(request: Request, run_id: str) -> AgentRunResponse:
    service = await _get_runtime_service(request.app)
    run = await service.get_agent_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return AgentRunResponse.model_validate(run)


@router.get("/{run_id}/events", response_model=list[dict[str, Any]])
async def list_agent_run_events(
    request: Request,
    run_id: str,
    after_seq: int | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    service = await _get_runtime_service(request.app)
    events = await service.list_agent_run_events(run_id, after_seq=after_seq, limit=limit)
    if events is None:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return [dict(event) for event in events]


@router.get("/{run_id}/events/stream")
async def stream_agent_run_events(
    request: Request,
    run_id: str,
    after_seq: int | None = None,
    limit: int | None = None,
) -> StreamingResponse:
    service = await _get_runtime_service(request.app)
    probe = await service.list_agent_run_events(run_id, after_seq=after_seq, limit=0)
    if probe is None:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return StreamingResponse(
        _stream_agent_run_events(
            request,
            service,
            run_id,
            after_seq=after_seq,
            limit=limit,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/{run_id}/subagents", response_model=list[SubagentTranscriptSummary])
async def list_agent_run_subagents(
    request: Request,
    run_id: str,
) -> list[SubagentTranscriptSummary]:
    service = await _get_runtime_service(request.app)
    items = await service.list_agent_run_subagents(run_id)
    if items is None:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return [SubagentTranscriptSummary.model_validate(item) for item in items]


@router.get("/{run_id}/subagents/{subagent_id}", response_model=SubagentTranscriptDetail)
async def get_agent_run_subagent(
    request: Request,
    run_id: str,
    subagent_id: str,
) -> SubagentTranscriptDetail:
    service = await _get_runtime_service(request.app)
    item = await service.get_agent_run_subagent(run_id, subagent_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Agent run subagent not found")
    return SubagentTranscriptDetail.model_validate(item)


@router.get("/{run_id}/exec/{session_id}")
async def get_agent_run_exec_session(
    request: Request,
    run_id: str,
    session_id: str,
    yield_time_ms: int | None = None,
) -> dict[str, object]:
    service = await _get_runtime_service(request.app)
    payload = await service.get_agent_run_exec_session(
        run_id,
        session_id,
        yield_time_ms=yield_time_ms,
    )
    if isinstance(payload, tuple):
        if payload[0] == "run_not_found":
            raise HTTPException(status_code=404, detail="Agent run not found")
        raise HTTPException(status_code=404, detail="Exec session not associated with this agent run")
    return payload


@router.post("/{run_id}/exec/{session_id}/stop")
async def stop_agent_run_exec_session(
    request: Request,
    run_id: str,
    session_id: str,
) -> dict[str, object]:
    service = await _get_runtime_service(request.app)
    payload = await service.stop_agent_run_exec_session(run_id, session_id)
    if isinstance(payload, tuple):
        if payload[0] == "run_not_found":
            raise HTTPException(status_code=404, detail="Agent run not found")
        raise HTTPException(status_code=404, detail="Exec session not associated with this agent run")
    return payload


@router.post("/{run_id}/reattach-exec/{session_id}")
async def reattach_agent_run_exec_session(
    request: Request,
    run_id: str,
    session_id: str,
    yield_time_ms: int | None = None,
) -> dict[str, object]:
    service = await _get_runtime_service(request.app)
    payload = await service.reattach_agent_run_exec_session(
        run_id,
        session_id,
        yield_time_ms=yield_time_ms,
    )
    if isinstance(payload, tuple):
        if payload[0] == "run_not_found":
            raise HTTPException(status_code=404, detail="Agent run not found")
        raise HTTPException(status_code=404, detail="Exec session not associated with this agent run")
    return payload


@router.post("/{run_id}/finalize-partial", response_model=AgentRunResponse)
async def finalize_agent_run_partial(request: Request, run_id: str) -> AgentRunResponse:
    service = await _get_runtime_service(request.app)
    payload = await service.finalize_agent_run_partial(run_id)
    if isinstance(payload, tuple):
        if payload[0] == "run_not_found":
            raise HTTPException(status_code=404, detail="Agent run not found")
        raise HTTPException(
            status_code=409,
            detail=f"Agent run status '{payload[1] or 'unknown'}' cannot be finalized as partial",
        )
    return AgentRunResponse.model_validate(payload)


@router.get("/{run_id}/health")
async def get_agent_run_health(request: Request, run_id: str) -> dict[str, object]:
    service = await _get_runtime_service(request.app)
    payload = await service.get_agent_run_health(run_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return payload


@router.get(
    "/{run_id}/packages/attempts/{attempt_id}",
    response_model=AgentRunAttemptPackageResponse,
)
async def get_agent_run_attempt_package(
    request: Request,
    run_id: str,
    attempt_id: str,
) -> AgentRunAttemptPackageResponse:
    service = await _get_runtime_service(request.app)
    payload = await service.get_agent_run_attempt_package(run_id, attempt_id)
    if isinstance(payload, tuple):
        if payload[0] == "run_not_found":
            raise HTTPException(status_code=404, detail="Agent run not found")
        raise HTTPException(status_code=404, detail="Attempt package not found")
    return AgentRunAttemptPackageResponse.model_validate(payload)


@router.get(
    "/{run_id}/packages/dataset",
    response_model=AgentRunDatasetPackageResponse,
)
async def get_agent_run_dataset_package(
    request: Request,
    run_id: str,
) -> AgentRunDatasetPackageResponse:
    service = await _get_runtime_service(request.app)
    payload = await service.get_agent_run_dataset_package(run_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return AgentRunDatasetPackageResponse.model_validate(payload)


@router.post("/{run_id}/start", response_model=AgentRunResponse)
async def start_agent_run(request: Request, run_id: str) -> AgentRunResponse:
    service = await _get_runtime_service(request.app)
    run = await service.start_agent_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return AgentRunResponse.model_validate(run)


@router.post("/{run_id}/pause", response_model=AgentRunResponse)
async def pause_agent_run(request: Request, run_id: str) -> AgentRunResponse:
    service = await _get_runtime_service(request.app)
    run = await service.pause_agent_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return AgentRunResponse.model_validate(run)


@router.post("/{run_id}/resume", response_model=AgentRunResponse)
async def resume_agent_run(
    request: Request,
    run_id: str,
    payload: AgentRunResumeRequest | None = None,
) -> AgentRunResponse:
    service = await _get_runtime_service(request.app)
    approval_id = (
        payload.approval_id.strip()
        if payload is not None and isinstance(payload.approval_id, str) and payload.approval_id.strip()
        else None
    )
    if approval_id is not None:
        current = await service.get_agent_run(run_id)
        if current is None:
            raise HTTPException(status_code=404, detail="Agent run not found")
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
            updated = await service.get_agent_run(run_id)
            return AgentRunResponse.model_validate(updated or current)

    strategy = payload.strategy if payload is not None else "continue_from_checkpoint"
    resume_agent_run = service.resume_agent_run
    try:
        supports_strategy = "strategy" in inspect.signature(resume_agent_run).parameters
    except (TypeError, ValueError):
        supports_strategy = False
    if supports_strategy:
        run = await resume_agent_run(run_id, strategy=strategy)
    else:
        run = await resume_agent_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return AgentRunResponse.model_validate(run)


@router.post("/{run_id}/cancel", response_model=AgentRunResponse)
async def cancel_agent_run(request: Request, run_id: str) -> AgentRunResponse:
    service = await _get_runtime_service(request.app)
    run = await service.cancel_agent_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return AgentRunResponse.model_validate(run)


@router.post("/{run_id}/guidance", response_model=AgentRunResponse)
async def append_agent_run_guidance(
    request: Request,
    run_id: str,
    payload: AgentRunGuidanceRequest,
) -> AgentRunResponse:
    service = await _get_runtime_service(request.app)
    run = await service.append_agent_run_guidance(run_id, payload)
    if run is None:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return AgentRunResponse.model_validate(run)


@router.post("/{run_id}/messages", response_model=AgentRunResponse)
async def append_agent_run_message(
    request: Request,
    run_id: str,
    payload: AgentRunMessageRequest,
) -> AgentRunResponse:
    service = await _get_runtime_service(request.app)
    run = await service.append_agent_run_message(run_id, payload)
    if run is None:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return AgentRunResponse.model_validate(run)


@router.post("/{run_id}/subagents/{role_id}/messages", response_model=AgentRunResponse)
async def append_agent_run_subagent_message(
    request: Request,
    run_id: str,
    role_id: str,
    payload: AgentRunSubagentMessageRequest,
) -> AgentRunResponse:
    service = await _get_runtime_service(request.app)
    run = await service.append_agent_run_subagent_message(run_id, role_id, payload)
    if run is None:
        raise HTTPException(status_code=404, detail="Agent run not found")
    return AgentRunResponse.model_validate(run)


async def _stream_agent_run_events(
    request: Request,
    service: Any,
    run_id: str,
    *,
    after_seq: int | None = None,
    limit: int | None = None,
) -> AsyncIterator[str]:
    deadline = asyncio.get_running_loop().time() + _AGENT_RUN_EVENTS_STREAM_TIMEOUT_SEC
    next_after_seq = after_seq
    last_heartbeat_at = asyncio.get_running_loop().time()

    while True:
        if await request.is_disconnected():
            return

        events = await service.list_agent_run_events(run_id, after_seq=next_after_seq, limit=limit)
        if events is None:
            return
        if events:
            for event in events:
                seq = event.get("seq")
                if isinstance(seq, int):
                    next_after_seq = seq
                yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            continue

        now = asyncio.get_running_loop().time()
        if now >= deadline:
            return
        if now - last_heartbeat_at >= _AGENT_RUN_EVENTS_STREAM_HEARTBEAT_SEC:
            last_heartbeat_at = now
            yield ": keep-alive\n\n"
        await asyncio.sleep(_AGENT_RUN_EVENTS_STREAM_POLL_INTERVAL_SEC)
