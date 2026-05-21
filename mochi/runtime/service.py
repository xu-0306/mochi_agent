"""Background runtime service for tasks and approvals."""

from __future__ import annotations

import asyncio
from typing import Any
from uuid import uuid4

from mochi.api.routes.chat import _serialize_event
from mochi.config.schema import SecurityConfig
from mochi.runtime.models import TaskCreateRequest
from mochi.runtime.store import RuntimeStore
from mochi.security.policy import build_runtime_permission_policy_dict

TASK_STATUS_RUNNING = {"queued", "running", "resumed"}


class RuntimeService:
    """Coordinate background task execution and approval workflow."""

    def __init__(self, *, engine: Any, store: RuntimeStore) -> None:
        self._engine = engine
        self._store = store
        self._active_jobs: dict[str, asyncio.Task[None]] = {}
        self._security_config = SecurityConfig()

    def update_security_config(self, security: SecurityConfig) -> None:
        self._security_config = security

    async def create_task(self, payload: TaskCreateRequest) -> dict[str, Any]:
        task_id = str(uuid4())
        summary = await self._store.create_task_run(
            task_id=task_id,
            input_text=payload.input,
            session_id=payload.session_id,
            project_id=payload.project_id,
            workspace_dir=payload.workspace_dir,
            inference_overrides=payload.inference_overrides,
        )
        self._active_jobs[task_id] = asyncio.create_task(
            self._run_task(task_id=task_id),
            name=f"runtime-task-{task_id}",
        )
        summary["pending_approval"] = None
        return _task_summary(summary)

    async def list_tasks(self) -> list[dict[str, Any]]:
        tasks = await self._store.list_task_runs()
        items: list[dict[str, Any]] = []
        for task in tasks:
            task["pending_approval"] = await self._store.get_pending_approval_for_task(task["id"])
            items.append(_task_summary(task))
        return items

    async def get_task(self, task_id: str) -> dict[str, Any] | None:
        task = await self._store.get_task_run(task_id)
        if task is None:
            return None
        events = await self._store.get_task_events(task_id)
        pending_approval = await self._store.get_pending_approval_for_task(task_id)
        task["pending_approval"] = pending_approval
        payload = _task_summary(task)
        payload["events"] = events
        payload["pending_approval"] = pending_approval
        return payload

    async def cancel_task(self, task_id: str) -> dict[str, Any] | None:
        task = await self._store.get_task_run(task_id)
        if task is None:
            return None
        if task["status"] not in TASK_STATUS_RUNNING and task["status"] != "awaiting_approval":
            return _task_summary(task)

        if task["status"] in TASK_STATUS_RUNNING:
            active = self._active_jobs.get(task_id)
            if active is not None and not active.done():
                active.cancel()
        await self._store.update_task_status(task_id, "cancelled", error="Cancelled by user")
        return _task_summary((await self._store.get_task_run(task_id)) or task)

    async def resume_task(
        self,
        task_id: str,
        *,
        approved: bool,
        reason: str | None = None,
    ) -> dict[str, Any] | None:
        task = await self._store.get_task_run(task_id)
        if task is None:
            return None

        approval = await self._store.get_pending_approval_for_task(task_id)
        if approval is None:
            return _task_summary(task)

        await self._store.resolve_approval_request(
            approval["id"],
            approved=approved,
            reason=reason,
        )
        if not approved:
            await self._store.update_task_status(task_id, "cancelled", error="Approval rejected")
            return _task_summary((await self._store.get_task_run(task_id)) or task)

        tool_name = str(approval.get("tool_name", ""))
        override = _approval_override(tool_name)
        await self._store.update_task_status(
            task_id,
            "resumed",
            error=None,
            permission_override=override,
        )
        self._active_jobs[task_id] = asyncio.create_task(
            self._run_task(task_id=task_id, permission_override=override),
            name=f"runtime-task-resume-{task_id}",
        )
        return _task_summary((await self._store.get_task_run(task_id)) or task)

    async def list_approvals(self, *, status: str | None = None) -> list[dict[str, Any]]:
        approvals = await self._store.list_approval_requests(status=status)
        return [_approval_summary(item) for item in approvals]

    async def resolve_approval(
        self,
        approval_id: str,
        *,
        approved: bool,
        reason: str | None = None,
    ) -> dict[str, Any] | None:
        current = await self._store.get_approval_request(approval_id)
        if current is None:
            return None
        approval = await self._store.resolve_approval_request(
            approval_id,
            approved=approved,
            reason=reason,
        )
        if approval is None:
            return None
        if not approved:
            await self._store.update_task_status(
                approval["task_id"],
                "cancelled",
                error="Approval rejected",
            )
            return _approval_summary(approval)
        override = _approval_override(current)
        await self._store.update_task_status(
            approval["task_id"],
            "resumed",
            error=None,
            permission_override=override,
        )
        self._active_jobs[approval["task_id"]] = asyncio.create_task(
            self._run_task(task_id=approval["task_id"], permission_override=override),
            name=f"runtime-task-resume-{approval['task_id']}",
        )
        return _approval_summary(approval)

    async def _run_task(
        self,
        *,
        task_id: str,
        permission_override: dict[str, Any] | None = None,
    ) -> None:
        task = await self._store.get_task_run(task_id)
        if task is None:
            return

        await self._store.update_task_status(task_id, "running", error=None)
        pending_requests: dict[str, dict[str, Any]] = {}
        final_answer: str | None = None

        try:
            effective_permission_policy = build_runtime_permission_policy_dict(
                self._security_config,
                overrides=permission_override,
            )
            stream = self._engine.chat(
                task["input"],
                session_id=task.get("session_id"),
                inference_overrides=task.get("inference_overrides") or {},
                project_id=task.get("project_id"),
                workspace_dir=task.get("workspace_dir"),
                permission_policy=effective_permission_policy,
            )
            async for event in stream:
                serialized = _serialize_event(event)
                await self._store.append_task_event(task_id, serialized)
                if serialized.get("type") == "final_answer":
                    content = serialized.get("content")
                    if isinstance(content, str):
                        final_answer = content

                if serialized.get("type") == "tool_call_request":
                    call_id = serialized.get("call_id")
                    if isinstance(call_id, str) and call_id:
                        pending_requests[call_id] = serialized
                    continue

                if serialized.get("type") != "tool_call_result":
                    continue
                metadata = serialized.get("metadata")
                requires_approval = (
                    isinstance(metadata, dict) and metadata.get("requires_approval") is True
                )
                if not requires_approval:
                    continue

                call_id = serialized.get("call_id")
                if not isinstance(call_id, str) or call_id not in pending_requests:
                    continue
                request_event = pending_requests[call_id]
                await self._store.create_approval_request(
                    approval_id=str(uuid4()),
                    task_id=task_id,
                    call_id=call_id,
                    tool_name=str(request_event.get("tool_name", "")),
                    arguments=dict(request_event.get("arguments") or {}),
                )
                await self._store.update_task_status(task_id, "awaiting_approval", final_answer=final_answer)
                return

            await self._store.update_task_status(task_id, "succeeded", final_answer=final_answer)
        except asyncio.CancelledError:
            await self._store.update_task_status(task_id, "cancelled", error="Cancelled by user", final_answer=final_answer)
            raise
        except Exception as exc:
            await self._store.update_task_status(task_id, "failed", error=str(exc), final_answer=final_answer)
        finally:
            active = self._active_jobs.get(task_id)
            current = asyncio.current_task()
            if active is not None and (active is current or active.done()):
                self._active_jobs.pop(task_id, None)


def _task_summary(task: dict[str, Any]) -> dict[str, Any]:
    pending_approval = task.get("pending_approval") if isinstance(task.get("pending_approval"), dict) else None
    return {
        "task_id": task["id"],
        "status": task["status"],
        "input_message": task["input"],
        "session_id": task.get("session_id"),
        "project_id": task.get("project_id"),
        "final_answer": task.get("final_answer"),
        "error": task.get("error"),
        "started_at": task.get("started_at"),
        "finished_at": task.get("finished_at"),
        "created_at": task["created_at"],
        "updated_at": task["updated_at"],
        "pending_approval_id": pending_approval.get("id") if pending_approval else None,
    }


def _approval_override(approval: dict[str, Any]) -> dict[str, Any]:
    return {
        "approved_tool_calls": [
            {
                "tool_name": str(approval.get("tool_name", "")),
                "arguments": dict(approval.get("arguments") or {}),
            }
        ]
    }


def _approval_summary(approval: dict[str, Any]) -> dict[str, Any]:
    status = str(approval.get("status", "pending"))
    decision: str | None = None
    if status == "approved":
        decision = "approve"
    elif status == "rejected":
        decision = "reject"
    return {
        "approval_id": approval["id"],
        "task_id": approval["task_id"],
        "status": status,
        "tool_name": approval["tool_name"],
        "arguments": approval.get("arguments") or {},
        "created_at": approval["created_at"],
        "resolved_at": approval.get("resolved_at"),
        "decision": decision,
    }
