"""Task runtime API routes."""

from __future__ import annotations

from fastapi import APIRouter, Body, HTTPException, Request
from pydantic import BaseModel

from mochi.api.routes.approvals import _get_runtime_service
from mochi.runtime.models import ApprovalResolution, TaskCreateRequest, TaskMessageRequest

router = APIRouter(prefix="/v1")


class TaskListResponse(BaseModel):
    type: str = "tasks"
    items: list[dict]


@router.post("/tasks")
async def create_task(request: Request, payload: TaskCreateRequest) -> dict:
    service = await _get_runtime_service(request.app)
    return await service.create_task(payload)


@router.get("/tasks", response_model=list[dict])
async def list_tasks(request: Request) -> list[dict]:
    service = await _get_runtime_service(request.app)
    return await service.list_tasks()


@router.get("/tasks/{task_id}")
async def get_task(request: Request, task_id: str) -> dict:
    service = await _get_runtime_service(request.app)
    payload = await service.get_task(task_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return payload


@router.post("/tasks/{task_id}/messages")
async def append_task_message(
    request: Request,
    task_id: str,
    payload: TaskMessageRequest,
) -> dict:
    service = await _get_runtime_service(request.app)
    try:
        result = await service.append_task_message(task_id, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if result is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return result


@router.post("/tasks/{task_id}/resume")
async def resume_task(
    request: Request,
    task_id: str,
    payload: ApprovalResolution | None = Body(default=None),
) -> dict:
    service = await _get_runtime_service(request.app)
    result = await service.resume_task(
        task_id,
        decision="approve_once" if payload is None else payload.decision,
        reason=None if payload is None else payload.reason,
        rule=None if payload is None else payload.rule,
    )
    if result is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return result


@router.post("/tasks/{task_id}/cancel")
async def cancel_task(request: Request, task_id: str) -> dict:
    service = await _get_runtime_service(request.app)
    result = await service.cancel_task(task_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return result
