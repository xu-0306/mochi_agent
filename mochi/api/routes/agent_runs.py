"""Agent Run runtime API routes."""

from __future__ import annotations

import inspect

from fastapi import APIRouter, HTTPException, Request

from mochi.api.routes.approvals import _get_runtime_service
from mochi.runtime.models import (
    AgentRunAttemptPackageResponse,
    AgentRunCreateRequest,
    AgentRunDatasetPackageResponse,
    AgentRunGuidanceRequest,
    AgentRunResumeRequest,
    AgentRunResponse,
)

router = APIRouter(prefix="/v1/agent-runs")


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
