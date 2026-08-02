"""Bounded chat API routes."""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections.abc import AsyncIterator
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from mochi.agents.events import (
    AssistantTruncatedEvent,
    ErrorEvent,
    FinalAnswerEvent,
    GoalStateChangedEvent,
    StatusEvent,
    SubagentCompletedEvent,
    SubagentProgressEvent,
    SubagentPromptEvent,
    SubagentStartedEvent,
    ThinkingEvent,
    ToolCallCompletedEvent,
    ToolCallCreatedEvent,
    ToolCallRequestEvent,
    ToolCallResultEvent,
)
from mochi.agents.invocation import ToolMode
from mochi.api.attachment_schema import AttachmentPayload
from mochi.api.session_store_binding import resolve_route_session_store
from mochi.api.tool_workflow_observability import (
    build_tool_workflow_observability,
)
from mochi.api.tool_workflow_outbox import (
    ToolWorkflowOutboxError,
    ToolWorkflowOutboxRepository,
)
from mochi.backends.base import BackendRequestError
from mochi.backends.inference_capabilities import ReasoningEffort
from mochi.backends.types import AttachmentRef
from mochi.runtime.recovery import classify_agent_run_recovery_issue
from mochi.security.policy import EffectivePolicyResolver
from mochi.sessions.store import SessionStore
from mochi.utils.streaming import sse_stream

router = APIRouter(prefix="/v1")

_CHAT_SUBAGENT_LIVE_DRAIN_IDLE_SECONDS = 0.15
_CHAT_SUBAGENT_LIVE_DRAIN_MAX_SECONDS = 1.0

_CHAT_MODEL_ERROR_MESSAGE = (
    "The configured model request failed. No tools were run. "
    "Retry later or select another model."
)
_CHAT_PROVIDER_AUTH_ERROR_MESSAGES = {
    401: (
        "The model provider rejected authentication (HTTP 401 Unauthorized). "
        "No tools were run. Check the API key and retry."
    ),
    403: (
        "The model provider denied the request (HTTP 403 Forbidden). "
        "No tools were run. Check the API key's model/provider permissions "
        "or select another model."
    ),
}
_CHAT_RESOURCE_ERROR_MESSAGES = {
    "quota_exhausted": (
        "The configured model has no available quota. No tools were run. "
        "Restore quota or select another model, then retry."
    ),
    "rate_limited": (
        "The model provider is rate-limiting requests. No tools were run. "
        "Wait briefly or select another model, then retry."
    ),
    "provider_timeout": (
        "The configured model provider timed out. No tools were run. "
        "Retry after the provider recovers or select another model."
    ),
    "provider_unavailable": (
        "The configured model is currently unavailable from its provider. "
        "No tools were run. Retry later or select an available model."
    ),
    "local_model_unavailable": (
        "The configured local model is unavailable. No tools were run. "
        "Restart the local runtime or select another model, then retry."
    ),
}


class ChatRequest(BaseModel):
    """`POST /v1/chat` request payload。"""

    message: str = Field(min_length=0)
    session_id: str | None = None
    project_id: str | None = None
    model: str | None = Field(default=None, min_length=1)
    tool_mode: ToolMode = "auto"
    system_prompt: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=131072)
    reserve_output_tokens: int | None = Field(default=None, ge=0, le=131072)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    repeat_penalty: float | None = Field(default=None, ge=0.0, le=2.0)
    reasoning_effort: ReasoningEffort | None = None
    selected_skill_ids: list[str] | None = None
    attachments: list[AttachmentPayload] | None = None
    client_timezone: str | None = Field(default=None, max_length=255)
    expected_policy_version: str | None = Field(default=None, min_length=1)


class ChatResponse(BaseModel):
    """`POST /v1/chat` response payload。"""

    type: str = "chat_response"
    session_id: str
    turn_id: str | None = None
    final_answer: str
    trajectory_id: str | None = None
    events: list[dict[str, Any]]
    tool_workflow: dict[str, Any]
    tool_workflow_aggregate: dict[str, Any] | None = None


class ChatContextResponse(BaseModel):
    """`POST /v1/chat/context` response payload."""

    type: str = "chat_context"
    session_id: str
    model: str
    backend_type: str = ""
    context_length: int
    estimated_prompt_tokens: int
    reserved_output_tokens: int
    remaining_tokens: int
    usage_ratio: float
    summary_tokens: int
    history_tokens: int
    memory_tokens: int
    skills_tokens: int
    tool_tokens: int
    draft_tokens: int
    compaction_triggered: bool
    compaction_reason: str | None = None
    compaction_mode: Literal["legacy", "semantic"] = "legacy"
    summary_mode: Literal["deterministic", "hybrid"] | None = None
    state_tokens: int = 0
    recent_raw_tokens: int = 0
    approximate: bool = True
    reasoning_effort: ReasoningEffort | None = None
    tool_workflow: dict[str, Any]


class ChatCancelRequest(BaseModel):
    turn_id: str | None = None


class ChatCancelResponse(BaseModel):
    status: Literal["cancel_requested", "already_completed", "not_found"]
    session_id: str
    turn_id: str | None = None
    run_state: Literal["running", "cancelling", "cancelled", "completed"] | None = None
    cancel_outcome: Literal["cancelled", "completed", "pending"] | None = None
    cancel_reason: str | None = None


@router.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
async def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    """執行 bounded 單輪文字對話並回傳完整事件列表。"""
    if payload.model:
        from mochi.api.routes.models import switch_model_runtime

        await switch_model_runtime(request, payload.model)
    engine = await _get_or_create_chat_engine(request)
    await _ensure_runtime_delegate(request)
    session_id = payload.session_id or str(uuid4())
    permission_policy = await _resolve_chat_effective_permission_policy(
        request,
        session_id,
    )
    resolved_project_id, resolved_workspace_dir = await _resolve_chat_project_context(
        request,
        payload,
        session_id,
    )
    turn_id = str(uuid4())

    stream = await _start_engine_chat(
        engine,
        message=payload.message,
        session_id=session_id,
        inference_overrides=_build_inference_overrides(payload),
        project_id=resolved_project_id,
        workspace_dir=resolved_workspace_dir,
        selected_skill_ids=payload.selected_skill_ids,
        attachments=_resolve_chat_attachments(payload),
        client_timezone=payload.client_timezone,
        turn_id=turn_id,
        tool_mode=payload.tool_mode,
        permission_policy=permission_policy,
    )
    events, final_answer, trajectory_id = await _collect_chat_result(stream)
    subagent_events = _synthesize_subagent_events(events)
    if subagent_events:
        events = _merge_chat_and_subagent_events(events, subagent_events)
    response_turn_id = _response_turn_id(events) or turn_id
    if _response_turn_id(events) is None:
        await _persist_turn_events(request, session_id, events, turn_id=response_turn_id)
    if subagent_events:
        await _persist_chat_subagent_events(
            request,
            session_id=session_id,
            turn_id=response_turn_id,
            events=subagent_events,
        )
    aggregate_snapshot = await _get_chat_tool_workflow_snapshot(
        request,
        session_id=session_id,
        turn_id=response_turn_id,
    )
    if aggregate_snapshot.get("aggregate") is None:
        aggregate_snapshot = None

    return ChatResponse(
        session_id=session_id,
        turn_id=response_turn_id,
        final_answer=final_answer,
        trajectory_id=trajectory_id,
        events=events,
        tool_workflow=build_tool_workflow_observability(
            events=events,
            effective_policy=permission_policy,
            expected_policy_version=payload.expected_policy_version,
        ),
        tool_workflow_aggregate=aggregate_snapshot,
    )


@router.post("/chat/context", response_model=ChatContextResponse)
async def chat_context(request: Request, payload: ChatRequest) -> ChatContextResponse:
    """Preview the next-request context budget without sending a chat turn."""
    if payload.model:
        from mochi.api.routes.models import switch_model_runtime

        await switch_model_runtime(request, payload.model)
    engine = await _get_or_create_chat_engine(request)
    await _ensure_runtime_delegate(request)
    session_id = payload.session_id or "draft-session"
    permission_policy = await _resolve_chat_effective_permission_policy(
        request,
        session_id,
    )
    resolved_project_id, resolved_workspace_dir = await _resolve_chat_project_context(
        request,
        payload,
        session_id,
    )

    preview = await _start_engine_chat_context_preview(
        engine,
        message=payload.message,
        session_id=session_id,
        inference_overrides=_build_inference_overrides(payload),
        project_id=resolved_project_id,
        workspace_dir=resolved_workspace_dir,
        selected_skill_ids=payload.selected_skill_ids,
        attachments=_resolve_chat_attachments(payload),
        client_timezone=payload.client_timezone,
        permission_policy=permission_policy,
    )
    if isinstance(preview, dict):
        return ChatContextResponse.model_validate(
            {
                **preview,
                "tool_workflow": build_tool_workflow_observability(
                    events=(),
                    effective_policy=permission_policy,
                    expected_policy_version=payload.expected_policy_version,
                ),
            }
        )
    raise HTTPException(status_code=500, detail="Engine did not return a chat context snapshot.")


@router.post("/chat/stream")
async def chat_stream(request: Request, payload: ChatRequest) -> StreamingResponse:
    """以 SSE 串流回傳 chat event stream。"""
    if payload.model:
        from mochi.api.routes.models import switch_model_runtime

        await switch_model_runtime(request, payload.model)
    engine = await _get_or_create_chat_engine(request)
    await _ensure_runtime_delegate(request)
    session_id = payload.session_id or str(uuid4())
    permission_policy = await _resolve_chat_effective_permission_policy(
        request,
        session_id,
    )
    resolved_project_id, resolved_workspace_dir = await _resolve_chat_project_context(
        request,
        payload,
        session_id,
    )
    turn_id = str(uuid4())

    stream = await _start_engine_chat(
        engine,
        message=payload.message,
        session_id=session_id,
        inference_overrides=_build_inference_overrides(payload),
        project_id=resolved_project_id,
        workspace_dir=resolved_workspace_dir,
        selected_skill_ids=payload.selected_skill_ids,
        attachments=_resolve_chat_attachments(payload),
        client_timezone=payload.client_timezone,
        turn_id=turn_id,
        tool_mode=payload.tool_mode,
        permission_policy=permission_policy,
    )
    headers = {
        "Cache-Control": "no-cache",
        "X-Session-ID": session_id,
        "X-Turn-ID": turn_id,
        "X-Chat-Run-ID": turn_id,
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        sse_stream(_stream_chat_events(request, session_id, stream, fallback_turn_id=turn_id)),
        media_type="text/event-stream",
        headers=headers,
    )


@router.post("/chat/{session_id}/cancel", response_model=ChatCancelResponse)
async def cancel_chat_stream_run(
    request: Request,
    session_id: str,
    payload: ChatCancelRequest,
) -> ChatCancelResponse:
    engine = await _get_or_create_chat_engine(request)
    cancel_chat_run = getattr(engine, "cancel_chat_run", None)
    if not callable(cancel_chat_run):
        raise HTTPException(status_code=501, detail="Active chat cancellation is not supported by this engine.")
    result = await _maybe_await_result(
        cancel_chat_run(
            session_id,
            turn_id=payload.turn_id,
        )
    )
    if not isinstance(result, dict):
        raise HTTPException(status_code=500, detail="Chat cancellation did not return a valid response.")
    return ChatCancelResponse.model_validate(result)


async def _collect_chat_result(
    stream: AsyncIterator[Any],
) -> tuple[list[dict[str, Any]], str, str | None]:
    """收斂 chat event stream。"""
    events: list[dict[str, Any]] = []
    final_answer = ""
    trajectory_id: str | None = None

    async for event in stream:
        serialized = _serialize_event(event)
        events.append(serialized)
        if serialized["type"] == "final_answer":
            final_answer = str(serialized.get("content", ""))
            raw_trajectory_id = serialized.get("trajectory_id")
            trajectory_id = str(raw_trajectory_id) if raw_trajectory_id is not None else None

    return events, final_answer, trajectory_id


async def _stream_chat_events(
    request: Request,
    session_id: str,
    stream: AsyncIterator[Any],
    *,
    fallback_turn_id: str,
) -> AsyncIterator[dict[str, Any]]:
    """串流 serialized chat events，必要時補做 session replay 持久化。"""
    events: list[dict[str, Any]] = []
    fallback_replay_events: list[dict[str, Any]] = []
    persisted_replay_seqs: set[int] = set()
    pending_subagent_requests: dict[str, dict[str, Any]] = {}
    delegated_task_ids: set[str] = set()
    stream_task: asyncio.Task[Any] | None = None
    live_task: asyncio.Task[Any] | None = None
    live_subscription: Any | None = None
    aggregate_seq = 0

    from mochi.api.routes.approvals import _get_runtime_service

    runtime_service = await _get_runtime_service(request.app)
    subscribe_live = getattr(runtime_service, "subscribe_delegated_subagent_runtime_events", None)
    if callable(subscribe_live):
        live_subscription = subscribe_live(session_id=session_id)
        live_task = asyncio.create_task(
            live_subscription.get(),
            name=f"chat-subagent-live-{session_id}",
        )

    try:
        stream_task = asyncio.create_task(anext(stream), name=f"chat-stream-{fallback_turn_id}")
        while stream_task is not None:
            wait_tasks: set[asyncio.Task[Any]] = {stream_task}
            if live_task is not None and delegated_task_ids:
                wait_tasks.add(live_task)
            done, _ = await asyncio.wait(wait_tasks, return_when=asyncio.FIRST_COMPLETED)

            if live_task is not None and live_task in done:
                envelope = live_task.result()
                live_task = asyncio.create_task(
                    live_subscription.get(),
                    name=f"chat-subagent-live-{session_id}",
                )
                live_event = _serialize_live_subagent_event(
                    envelope,
                    fallback_turn_id=fallback_turn_id,
                    delegated_task_ids=delegated_task_ids,
                )
                if live_event is not None:
                    events.append(live_event)
                    yield live_event
                continue

            if stream_task not in done:
                continue
            try:
                event = stream_task.result()
            except StopAsyncIteration:
                stream_task = None
                break
            serialized = _serialize_event(event, fallback_turn_id=fallback_turn_id)
            task_id = _delegated_task_id_from_tool_result(serialized)
            if task_id:
                delegated_task_ids.add(task_id)
            events.append(serialized)
            phase = _event_phase(serialized)
            persisted_seq = _persisted_replay_seq(event, turn_id=fallback_turn_id)
            if phase is not None:
                if persisted_seq is None:
                    fallback_replay_events.append(serialized)
                else:
                    persisted_replay_seqs.add(persisted_seq)
            yield serialized
            for snapshot in await _get_chat_tool_workflow_updates(
                request,
                session_id=session_id,
                turn_id=fallback_turn_id,
                after_seq=aggregate_seq,
            ):
                aggregate = snapshot.get("aggregate")
                if not isinstance(aggregate, dict):
                    continue
                aggregate_seq = max(aggregate_seq, int(aggregate["seq"]))
                yield {
                    "_sse_event": "tool_workflow_aggregate",
                    "_sse_id": aggregate["event_id"],
                    "_sse_data": snapshot,
                }
            synthesized = _synthesize_incremental_subagent_events(
                serialized,
                pending_requests=pending_subagent_requests,
            )
            for subagent_event in synthesized:
                synthesized_task_id = _delegated_task_id_from_subagent_event(subagent_event)
                if synthesized_task_id:
                    delegated_task_ids.add(synthesized_task_id)
                events.append(subagent_event)
                yield subagent_event
            stream_task = asyncio.create_task(anext(stream), name=f"chat-stream-{fallback_turn_id}")

        if live_subscription is not None and delegated_task_ids:
            async for live_event in _drain_live_subagent_events(
                live_subscription,
                delegated_task_ids=delegated_task_ids,
                fallback_turn_id=fallback_turn_id,
            ):
                events.append(live_event)
                yield live_event
    except Exception as exc:
        error_payload = _chat_stream_error_payload(exc)
        error_event = _attach_turn_id(
            None,
            error_payload,
            fallback_turn_id=fallback_turn_id,
        )
        events.append(error_event)
        fallback_replay_events.append(error_event)
        yield error_event
    finally:
        for task in (stream_task, live_task):
            if task is None or task.done():
                continue
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        if live_subscription is not None:
            close_live = getattr(live_subscription, "close", None)
            if callable(close_live):
                close_live()
        close_stream = getattr(stream, "aclose", None)
        if callable(close_stream):
            with contextlib.suppress(asyncio.CancelledError, RuntimeError, StopAsyncIteration):
                payload = close_stream()
                if inspect.isawaitable(payload):
                    await payload
        if fallback_replay_events:
            await _persist_turn_events(
                request,
                session_id,
                fallback_replay_events,
                turn_id=fallback_turn_id,
                start_seq=max(persisted_replay_seqs, default=0),
            )
        await _persist_chat_subagent_events(
            request,
            session_id=session_id,
            turn_id=fallback_turn_id,
            events=events,
        )
        if events:
            for snapshot in await _get_chat_tool_workflow_updates(
                request,
                session_id=session_id,
                turn_id=fallback_turn_id,
                after_seq=aggregate_seq,
            ):
                aggregate = snapshot.get("aggregate")
                if not isinstance(aggregate, dict):
                    continue
                aggregate_seq = max(aggregate_seq, int(aggregate["seq"]))
                yield {
                    "_sse_event": "tool_workflow_aggregate",
                    "_sse_id": aggregate["event_id"],
                    "_sse_data": snapshot,
                }


def _chat_stream_error_payload(exc: Exception) -> dict[str, Any]:
    if not isinstance(exc, BackendRequestError):
        return {
            "type": "error",
            "error": str(exc),
            "code": "CHAT_STREAM_ERROR",
        }
    backend_metadata = dict(exc.metadata or {})
    status_code = _coerce_http_status(backend_metadata.get("status_code"))
    backend_name = _clean_backend_diagnostic(backend_metadata.get("backend_name"))
    model = _clean_backend_diagnostic(backend_metadata.get("model"))
    if status_code in _CHAT_PROVIDER_AUTH_ERROR_MESSAGES:
        classification = (
            "provider_authentication_failed"
            if status_code == 401
            else "provider_access_denied"
        )
        code = (
            "MODEL_PROVIDER_AUTHENTICATION_FAILED"
            if status_code == 401
            else "MODEL_PROVIDER_ACCESS_DENIED"
        )
        return {
            "type": "error",
            "error": _CHAT_PROVIDER_AUTH_ERROR_MESSAGES[status_code],
            "code": code,
            "metadata": {
                "classification": classification,
                "retryable": False,
                "backend_name": backend_name,
                "status_code": status_code,
                "model": model,
            },
        }
    issue = classify_agent_run_recovery_issue(exc)
    if issue is None:
        error_message = _CHAT_MODEL_ERROR_MESSAGE
        error_metadata: dict[str, Any] = {
            "classification": "unclassified_backend_failure",
            "retryable": False,
        }
        if status_code is not None:
            error_message = (
                f"The configured model request failed (provider HTTP {status_code}). "
                "No tools were run. Check the model/provider configuration and retry."
            )
            error_metadata["status_code"] = status_code
        if backend_name is not None:
            error_metadata["backend_name"] = backend_name
        if model is not None:
            error_metadata["model"] = model
        return {
            "type": "error",
            "error": error_message,
            "code": "MODEL_REQUEST_FAILED",
            "metadata": error_metadata,
        }
    return {
        "type": "error",
        "error": _CHAT_RESOURCE_ERROR_MESSAGES[issue.kind],
        "code": f"MODEL_{issue.kind.upper()}",
        "metadata": {
            "classification": issue.kind,
            "retryable": issue.retryable,
            "backend_name": issue.backend_name,
            "status_code": issue.status_code,
        },
    }


def _coerce_http_status(value: Any) -> int | None:
    try:
        status_code = int(value)
    except (TypeError, ValueError):
        return None
    return status_code if 100 <= status_code <= 599 else None


def _clean_backend_diagnostic(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    text = value.strip()
    return text[:128] if text else None


async def _drain_live_subagent_events(
    live_subscription: Any,
    *,
    delegated_task_ids: set[str],
    fallback_turn_id: str,
) -> AsyncIterator[dict[str, Any]]:
    loop = asyncio.get_running_loop()
    idle_deadline = loop.time() + _CHAT_SUBAGENT_LIVE_DRAIN_IDLE_SECONDS
    max_deadline = loop.time() + _CHAT_SUBAGENT_LIVE_DRAIN_MAX_SECONDS
    while True:
        timeout = min(idle_deadline, max_deadline) - loop.time()
        if timeout <= 0:
            return
        try:
            envelope = await asyncio.wait_for(live_subscription.get(), timeout=timeout)
        except TimeoutError:
            return
        live_event = _serialize_live_subagent_event(
            envelope,
            fallback_turn_id=fallback_turn_id,
            delegated_task_ids=delegated_task_ids,
        )
        if live_event is None:
            continue
        idle_deadline = loop.time() + _CHAT_SUBAGENT_LIVE_DRAIN_IDLE_SECONDS
        yield live_event


def _serialize_live_subagent_event(
    envelope: Any,
    *,
    fallback_turn_id: str,
    delegated_task_ids: set[str],
) -> dict[str, Any] | None:
    if not isinstance(envelope, dict):
        return None
    task_id = str(envelope.get("task_id") or "").strip()
    if task_id not in delegated_task_ids:
        return None
    event = envelope.get("event")
    if not isinstance(event, dict):
        return None
    event_type = str(event.get("type") or "").strip()
    subagent_id = str(event.get("subagent_id") or "").strip()
    if not subagent_id or event_type not in {
        "subagent_started",
        "subagent_prompt",
        "subagent_progress",
        "subagent_completed",
        "subagent_thinking",
        "subagent_message_accepted",
        "subagent_message_applied",
        "subagent_message_cancelled",
        "subagent_message_deferred",
        "subagent_message_queued",
        "subagent_interrupted",
        "subagent_tool_call",
        "subagent_tool_cancel_requested",
        "subagent_tool_cancelled",
        "subagent_tool_cancel_deferred",
        "subagent_tool_result",
        "runtime_blocked",
    }:
        return None
    return _attach_turn_id(
        None,
        jsonable_encoder(dict(event)),
        fallback_turn_id=fallback_turn_id,
    )


def _delegated_task_id_from_tool_result(event: dict[str, Any]) -> str | None:
    if event.get("type") != "tool_call_result" or event.get("tool_name") != "delegate_subagent_task":
        return None
    result_payload = event.get("result")
    metadata = event.get("metadata")
    task_id = (
        result_payload.get("task_id")
        if isinstance(result_payload, dict)
        else None
    ) or (
        metadata.get("task_id")
        if isinstance(metadata, dict)
        else None
    )
    task_text = str(task_id or "").strip()
    return task_text or None


def _delegated_task_id_from_subagent_event(event: dict[str, Any]) -> str | None:
    metadata = event.get("metadata")
    task_id = metadata.get("task_id") if isinstance(metadata, dict) else event.get("subagent_id")
    task_text = str(task_id or "").strip()
    return task_text or None


def _serialize_event(
    event: Any,
    *,
    fallback_turn_id: str | None = None,
) -> dict[str, Any]:
    """將 AgentEvent 轉成 JSON-safe dict。"""
    if isinstance(event, ThinkingEvent):
        return _attach_turn_id(
            event,
            {
                "type": event.type,
                "content": event.content,
                "metadata": jsonable_encoder(event.metadata),
            },
            fallback_turn_id=fallback_turn_id,
        )
    if isinstance(event, StatusEvent):
        return _attach_turn_id(
            event,
            {
                "type": event.type,
                "content": event.content,
                "metadata": jsonable_encoder(event.metadata),
            },
            fallback_turn_id=fallback_turn_id,
        )
    if isinstance(event, SubagentStartedEvent):
        return _attach_turn_id(
            event,
            {
                "type": event.type,
                "subagent_id": event.subagent_id,
                "parent_type": event.parent_type,
                "parent_id": event.parent_id,
                "role_id": event.role_id,
                "title": event.title,
                "model_id": event.model_id,
                "prompt_preview": event.prompt_preview,
                "status": event.status,
                "summary": event.summary,
                "metadata": jsonable_encoder(event.metadata),
            },
            fallback_turn_id=fallback_turn_id,
        )
    if isinstance(event, SubagentPromptEvent):
        return _attach_turn_id(
            event,
            {
                "type": event.type,
                "subagent_id": event.subagent_id,
                "parent_type": event.parent_type,
                "parent_id": event.parent_id,
                "role_id": event.role_id,
                "title": event.title,
                "model_id": event.model_id,
                "system_prompt": event.system_prompt,
                "user_prompt": event.user_prompt,
                "prompt_preview": event.prompt_preview,
                "status": event.status,
                "metadata": jsonable_encoder(event.metadata),
            },
            fallback_turn_id=fallback_turn_id,
        )
    if isinstance(event, SubagentProgressEvent):
        return _attach_turn_id(
            event,
            {
                "type": event.type,
                "subagent_id": event.subagent_id,
                "parent_type": event.parent_type,
                "parent_id": event.parent_id,
                "role_id": event.role_id,
                "title": event.title,
                "content": event.content,
                "status": event.status,
                "metadata": jsonable_encoder(event.metadata),
            },
            fallback_turn_id=fallback_turn_id,
        )
    if isinstance(event, SubagentCompletedEvent):
        return _attach_turn_id(
            event,
            {
                "type": event.type,
                "subagent_id": event.subagent_id,
                "parent_type": event.parent_type,
                "parent_id": event.parent_id,
                "role_id": event.role_id,
                "title": event.title,
                "model_id": event.model_id,
                "status": event.status,
                "summary": event.summary,
                "content": event.content,
                "metadata": jsonable_encoder(event.metadata),
            },
            fallback_turn_id=fallback_turn_id,
        )
    if isinstance(event, AssistantTruncatedEvent):
        return _attach_turn_id(
            event,
            {
                "type": event.type,
                "content": event.content,
                "finish_reason": event.finish_reason,
                "recovery_attempt": event.recovery_attempt,
                "partial_output_chars": event.partial_output_chars,
                "metadata": jsonable_encoder(event.metadata),
            },
            fallback_turn_id=fallback_turn_id,
        )
    if isinstance(event, ToolCallCreatedEvent):
        return _attach_turn_id(
            event,
            {
                "type": event.type,
                "call_id": event.call_id,
                "tool_name": event.tool_name,
                "arguments": jsonable_encoder(event.arguments),
                "metadata": jsonable_encoder(event.metadata),
            },
            fallback_turn_id=fallback_turn_id,
        )
    if isinstance(event, ToolCallCompletedEvent):
        return _attach_turn_id(
            event,
            {
                "type": event.type,
                "call_id": event.call_id,
                "tool_name": event.tool_name,
                "arguments": jsonable_encoder(event.arguments),
                "result": _json_safe(event.result),
                "error": event.error,
                "metadata": jsonable_encoder(event.metadata),
            },
            fallback_turn_id=fallback_turn_id,
        )
    if isinstance(event, GoalStateChangedEvent):
        return _attach_turn_id(
            event,
            {
                "type": event.type,
                "goal_id": event.goal_id,
                "previous_status": event.previous_status,
                "status": event.status,
                "attempt_id": event.attempt_id,
                "agent_run_id": event.agent_run_id,
                "reason": event.reason,
                "metadata": jsonable_encoder(event.metadata),
            },
            fallback_turn_id=fallback_turn_id,
        )
    if isinstance(event, ToolCallRequestEvent):
        return _attach_turn_id(
            event,
            {
                "type": event.type,
                "call_id": event.call_id,
                "tool_name": event.tool_name,
                "arguments": jsonable_encoder(event.arguments),
            },
            fallback_turn_id=fallback_turn_id,
        )
    if isinstance(event, ToolCallResultEvent):
        return _attach_turn_id(
            event,
            {
                "type": event.type,
                "call_id": event.call_id,
                "tool_name": event.tool_name,
                "result": _json_safe(event.result),
                "error": event.error,
                "metadata": jsonable_encoder(event.metadata),
            },
            fallback_turn_id=fallback_turn_id,
        )
    if isinstance(event, FinalAnswerEvent):
        payload = {
            "type": event.type,
            "content": event.content,
            "trajectory_id": event.trajectory_id,
            "input_tokens": event.input_tokens,
            "output_tokens": event.output_tokens,
            "generation_time_ms": event.generation_time_ms,
            "finish_reason": event.finish_reason,
        }
        if event.metadata:
            payload["metadata"] = jsonable_encoder(event.metadata)
        return _attach_turn_id(
            event,
            payload,
            fallback_turn_id=fallback_turn_id,
        )
    if isinstance(event, ErrorEvent):
        return _attach_turn_id(
            event,
            {
                "type": event.type,
                "error": event.message,
                "code": event.code,
                "metadata": jsonable_encoder(event.metadata),
            },
            fallback_turn_id=fallback_turn_id,
        )
    if is_dataclass(event):
        return _attach_turn_id(
            event,
            jsonable_encoder(asdict(event)),
            fallback_turn_id=fallback_turn_id,
        )
    if isinstance(event, dict):
        return _attach_turn_id(
            None,
            jsonable_encoder(event),
            fallback_turn_id=fallback_turn_id,
        )
    return _attach_turn_id(
        None,
        {"type": "unknown", "content": _json_safe(event)},
        fallback_turn_id=fallback_turn_id,
    )


def _build_inference_overrides(payload: ChatRequest) -> dict[str, Any]:
    """從 chat payload 擷取推理參數覆蓋。"""
    field_map = {
        "system_prompt": "system_prompt",
        "temperature": "temperature",
        "max_tokens": "max_tokens",
        "reserve_output_tokens": "reserve_output_tokens",
        "top_p": "top_p",
        "min_p": "min_p",
        "top_k": "top_k",
        "frequency_penalty": "frequency_penalty",
        "presence_penalty": "presence_penalty",
        "repeat_penalty": "repeat_penalty",
        "reasoning_effort": "reasoning_effort",
    }
    overrides: dict[str, Any] = {}
    for field_name, override_key in field_map.items():
        if field_name not in payload.model_fields_set:
            continue
        overrides[override_key] = getattr(payload, field_name)
    return overrides


def _resolve_chat_attachments(payload: ChatRequest) -> list[AttachmentRef]:
    return [attachment.to_attachment_ref() for attachment in payload.attachments or []]


async def _persist_turn_events(
    request: Request,
    session_id: str,
    events: list[dict[str, Any]],
    *,
    turn_id: str | None = None,
    start_seq: int = 0,
) -> str | None:
    """將本輪 replay event 以 `turn_event` schema 追加到 session JSONL。"""
    if not events:
        return None

    store = await _get_session_store(request)
    resolved_turn_id = turn_id or str(uuid4())

    for index, event in enumerate(events, start=start_seq + 1):
        phase = _event_phase(event)
        if phase is None:
            continue

        await store.save_event(
            session_id,
            {
                "type": "turn_event",
                "schema_version": 1,
                "turn_id": resolved_turn_id,
                "event_id": f"{resolved_turn_id}:{index}",
                "seq": index,
                "phase": phase,
                "timestamp": datetime.now(UTC).isoformat(),
                "payload": event,
            },
        )
    return resolved_turn_id


def _attach_turn_id(
    event: Any,
    payload: dict[str, Any],
    *,
    fallback_turn_id: str | None = None,
) -> dict[str, Any]:
    """若 AgentEvent 帶有 turn_id，附加到 API event。"""
    turn_id = getattr(event, "turn_id", None)
    if isinstance(turn_id, str) and turn_id:
        payload["turn_id"] = turn_id
    elif (
        fallback_turn_id is not None
        and not isinstance(payload.get("turn_id"), str)
    ):
        payload["turn_id"] = fallback_turn_id
    return payload


def _response_turn_id(events: list[dict[str, Any]]) -> str | None:
    """從 serialized events 取得本輪 turn_id。"""
    for event in events:
        turn_id = event.get("turn_id")
        if isinstance(turn_id, str) and turn_id:
            return turn_id
    return None


def _merge_chat_and_subagent_events(
    events: list[dict[str, Any]],
    subagent_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not subagent_events:
        return list(events)
    merged: list[dict[str, Any]] = []
    inserted = False
    for event in events:
        merged.append(event)
        if (
            not inserted
            and event.get("type") == "tool_call_result"
            and event.get("tool_name") == "delegate_subagent_task"
        ):
            merged.extend(subagent_events)
            inserted = True
    if not inserted:
        merged.extend(subagent_events)
    return merged


def _persisted_replay_seq(event: Any, *, turn_id: str) -> int | None:
    """Read the private engine-to-route replay authority marker."""
    if getattr(event, "_session_replay_persisted", False) is not True:
        return None
    seq = getattr(event, "_session_replay_seq", None)
    event_id = getattr(event, "_session_replay_event_id", None)
    if (
        not isinstance(seq, int)
        or seq < 1
        or event_id != f"{turn_id}:{seq}"
        or getattr(event, "turn_id", None) != turn_id
    ):
        return None
    return seq


async def _get_session_store(request: Request) -> SessionStore:
    """取得 chat route 可共用的 SessionStore。"""
    config = await _get_chat_config(request)
    return resolve_route_session_store(request.app, config)


async def _get_chat_tool_workflow_snapshot(
    request: Request,
    *,
    session_id: str,
    turn_id: str,
) -> dict[str, Any]:
    """Read the latest durable aggregate without making it chat authority."""

    config = await _get_chat_config(request)
    store = await _get_session_store(request)
    engine = getattr(request.app.state, "engine", None)
    outbox = getattr(engine, "_tool_workflow_outbox", None)
    if not (
        isinstance(outbox, ToolWorkflowOutboxRepository)
        and getattr(outbox, "_session_store", None) is store
    ):
        outbox = ToolWorkflowOutboxRepository(
            store,
            enabled=bool(config.agent.tool_observability_v1),
            publication_gate=getattr(engine, "tool_workflow_publication_gate", None),
        )
    try:
        records = list(await outbox.list(session_id, turn_id=turn_id))
    except ToolWorkflowOutboxError as exc:
        return {
            "storage_id": store.storage_id,
            "session_id": session_id,
            "turn_id": turn_id,
            "aggregate": None,
            "publication_enabled": outbox.enabled,
            "authoritative": False,
            "unsupported": True,
            "error": type(exc).__name__,
        }
    records.sort(key=lambda record: int(record["seq"]))
    return {
        "storage_id": store.storage_id,
        "session_id": session_id,
        "turn_id": turn_id,
        "aggregate": records[-1] if records else None,
        "publication_enabled": outbox.enabled,
        "authoritative": outbox.enabled,
    }


async def _get_chat_tool_workflow_updates(
    request: Request,
    *,
    session_id: str,
    turn_id: str,
    after_seq: int,
) -> list[dict[str, Any]]:
    snapshot = await _get_chat_tool_workflow_snapshot(
        request,
        session_id=session_id,
        turn_id=turn_id,
    )
    if not snapshot.get("authoritative"):
        return []
    aggregate = snapshot.get("aggregate")
    if not isinstance(aggregate, dict) or int(aggregate.get("seq", 0)) <= after_seq:
        return []
    return [snapshot]


async def _ensure_runtime_delegate(request: Request) -> None:
    from mochi.api.routes.approvals import _get_runtime_service

    await _get_runtime_service(request.app)


async def _start_engine_chat(
    engine: Any,
    *,
    message: str,
    session_id: str,
    inference_overrides: dict[str, Any] | None,
    project_id: str | None,
    workspace_dir: str,
    selected_skill_ids: list[str] | None,
    attachments: list[AttachmentRef],
    client_timezone: str | None,
    turn_id: str,
    tool_mode: ToolMode,
    permission_policy: dict[str, Any],
) -> AsyncIterator[Any]:
    chat_callable = engine.chat
    kwargs: dict[str, Any] = {
        "message": message,
        "session_id": session_id,
        "inference_overrides": inference_overrides,
        "project_id": project_id,
        "workspace_dir": workspace_dir,
        "selected_skill_ids": selected_skill_ids,
        "attachments": attachments,
    }
    try:
        signature = inspect.signature(chat_callable)
    except (TypeError, ValueError):
        signature = None
    if signature is None or "tool_mode" in signature.parameters:
        kwargs["tool_mode"] = tool_mode
    if signature is None or "turn_id" in signature.parameters:
        kwargs["turn_id"] = turn_id
    if signature is None or "client_timezone" in signature.parameters:
        kwargs["client_timezone"] = client_timezone
    if signature is None or "permission_policy" in signature.parameters:
        kwargs["permission_policy"] = permission_policy
    return await _maybe_await_result(chat_callable(**kwargs))


async def _start_engine_chat_context_preview(
    engine: Any,
    *,
    message: str,
    session_id: str,
    inference_overrides: dict[str, Any] | None,
    project_id: str | None,
    workspace_dir: str,
    selected_skill_ids: list[str] | None,
    attachments: list[AttachmentRef],
    client_timezone: str | None,
    permission_policy: dict[str, Any],
) -> Any:
    preview_callable = engine.preview_chat_context
    kwargs: dict[str, Any] = {
        "message": message,
        "session_id": session_id,
        "inference_overrides": inference_overrides,
        "project_id": project_id,
        "workspace_dir": workspace_dir,
        "selected_skill_ids": selected_skill_ids,
        "attachments": attachments,
    }
    try:
        signature = inspect.signature(preview_callable)
    except (TypeError, ValueError):
        signature = None
    if signature is None or "permission_policy" in signature.parameters:
        kwargs["permission_policy"] = permission_policy
    if signature is None or "client_timezone" in signature.parameters:
        kwargs["client_timezone"] = client_timezone
    return await _maybe_await_result(preview_callable(**kwargs))


async def _resolve_chat_effective_permission_policy(
    request: Request,
    session_id: str,
) -> dict[str, Any]:
    """Resolve the server-owned policy snapshot for one ordinary chat request."""
    from mochi.api.routes.sessions import _session_security_override

    config = await _get_chat_config(request)
    store = await _get_session_store(request)
    events = await store.load_session(session_id)
    session_override = _session_security_override(events)
    return EffectivePolicyResolver().resolve(
        config.security,
        session_overrides=session_override,
    ).to_dict()


async def _resolve_chat_project_context(
    request: Request,
    payload: ChatRequest,
    session_id: str,
) -> tuple[str | None, str]:
    """Resolve effective project assignment and workspace for one request."""
    resolved_project_id, workspace_root = await _resolve_workspace_scope(
        request,
        session_id=session_id,
        project_id=payload.project_id,
    )
    return resolved_project_id, str(workspace_root)


async def _get_or_create_chat_engine(request: Request) -> Any:
    from mochi.api.server import _get_or_create_engine

    return await _get_or_create_engine(request.app)


async def _get_chat_config(request: Request) -> Any:
    from mochi.api.server import _get_config

    return await _get_config(request.app)


async def _maybe_await_result(value: Any) -> Any:
    from mochi.api.server import _maybe_await

    return await _maybe_await(value)


async def _resolve_workspace_scope(
    request: Request,
    *,
    session_id: str | None,
    project_id: str | None,
) -> tuple[str | None, Any]:
    from mochi.api.routes.workspace import resolve_workspace_scope

    return await resolve_workspace_scope(
        request,
        session_id=session_id,
        project_id=project_id,
    )


def _event_phase(event: dict[str, Any]) -> str | None:
    """將 API event type 映射為 replay phase。"""
    event_type = event.get("type")
    if event_type == "thinking":
        return "thinking"
    if event_type == "status":
        return "status"
    if event_type in {"subagent_started", "subagent_prompt", "subagent_progress", "subagent_completed"}:
        return str(event_type)
    if event_type == "assistant_truncated":
        return "assistant_truncated"
    if event_type == "tool_call_created":
        return "tool_call_created"
    if event_type == "tool_call_completed":
        return "tool_call_completed"
    if event_type == "goal_state_changed":
        return "goal_state_changed"
    if event_type == "tool_call_request":
        return "tool_call_request"
    if event_type == "tool_call_result":
        return "tool_call_result"
    if event_type == "error":
        return "error"
    if event_type == "final_answer":
        return "final_answer"
    return None


def _json_safe(value: Any) -> Any:
    """將任意值收斂為 JSON 相容內容。"""
    if is_dataclass(value):
        return jsonable_encoder(asdict(value))
    return jsonable_encoder(value)


def _synthesize_subagent_events(
    events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    synthesized: list[dict[str, Any]] = []
    pending_calls: dict[str, dict[str, Any]] = {}
    for event in events:
        synthesized.extend(
            _synthesize_incremental_subagent_events(event, pending_requests=pending_calls)
        )
    return synthesized


def _synthesize_incremental_subagent_events(
    event: dict[str, Any],
    *,
    pending_requests: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    event_type = str(event.get("type") or "")
    if event_type == "tool_call_request" and event.get("tool_name") == "delegate_subagent_task":
        call_id = str(event.get("call_id") or "").strip()
        if call_id:
            pending_requests[call_id] = event
        return []
    if event_type != "tool_call_result" or event.get("tool_name") != "delegate_subagent_task":
        return []

    call_id = str(event.get("call_id") or "").strip()
    request_event = pending_requests.pop(call_id, {})
    request_args = request_event.get("arguments")
    result_payload = event.get("result")
    tool_metadata = event.get("metadata")
    if not isinstance(request_args, dict) or not isinstance(result_payload, dict):
        return []
    return _build_subagent_lifecycle_events(
        request_event=request_event,
        request_args=request_args,
        result_payload=result_payload,
        tool_metadata=tool_metadata if isinstance(tool_metadata, dict) else {},
        turn_id=str(event.get("turn_id") or request_event.get("turn_id") or "").strip() or None,
    )


def _build_subagent_lifecycle_events(
    *,
    request_event: dict[str, Any],
    request_args: dict[str, Any],
    result_payload: dict[str, Any],
    tool_metadata: dict[str, Any],
    turn_id: str | None,
) -> list[dict[str, Any]]:
    task_id = str(
        result_payload.get("task_id")
        or tool_metadata.get("task_id")
        or tool_metadata.get("subagent_id")
        or ""
    ).strip()
    if not task_id:
        return []

    objective = str(request_args.get("objective") or "").strip()
    protocol = str(request_args.get("protocol") or "").strip() or None
    title = str(result_payload.get("display_name") or "Delegated subagent task").strip()
    role_id = "delegate_subagent_task"
    model_id = _first_model_id_hint(request_args)
    created_at = datetime.now(UTC).isoformat()
    parent_type = "chat_turn"
    parent_id = turn_id or str(request_event.get("turn_id") or "").strip() or task_id
    prompt_preview = _truncate_preview(objective, 160)
    raw_status = str(result_payload.get("status") or tool_metadata.get("status") or "queued").strip()
    summary = raw_status or "queued"
    normalized_status, terminal = _normalize_delegate_subagent_status(summary)
    active_status = "queued" if normalized_status == "queued" else "running"
    approval_state = (
        result_payload.get("approval_state")
        if isinstance(result_payload.get("approval_state"), dict)
        else tool_metadata.get("approval_state")
        if isinstance(tool_metadata.get("approval_state"), dict)
        else {}
    )
    pending_approvals = (
        approval_state.get("pending_approvals")
        if isinstance(approval_state.get("pending_approvals"), list)
        else []
    )
    approval_ids = [
        str(item).strip()
        for item in approval_state.get("approval_ids", [])
        if str(item).strip()
    ] if isinstance(approval_state, dict) else []
    tool_names = [
        str(item).strip()
        for item in approval_state.get("tool_names", [])
        if str(item).strip()
    ] if isinstance(approval_state, dict) else []
    common_metadata = {
        "source": "delegate_subagent_task",
        "task_id": task_id,
        "task_type": result_payload.get("task_type") or tool_metadata.get("task_type"),
        "protocol": protocol,
        "parent_session_id": result_payload.get("parent_session_id") or tool_metadata.get("parent_session_id"),
        "delegate_status": summary,
        **({"approval_ids": approval_ids} if approval_ids else {}),
        **({"tool_names": tool_names} if tool_names else {}),
        **({"pending_approvals": pending_approvals} if pending_approvals else {}),
        **({"recommended_action": "resolve_approval"} if approval_ids or pending_approvals else {}),
    }
    if isinstance(approval_state, dict):
        for key in (
            "approval_wait_started_at",
            "approval_wait_timeout_sec",
            "approval_wait_elapsed_sec",
            "approval_wait_expires_at",
            "status",
        ):
            if approval_state.get(key) is not None:
                common_metadata[key] = approval_state.get(key)
    lifecycle_event_type = "subagent_completed" if terminal else "subagent_progress"
    lifecycle_summary = (
        f"Delegated background task finished with status: {summary}."
        if terminal
        else f"Delegated background task is {normalized_status}."
    )
    lifecycle_events = [
        _attach_turn_id(
            None,
            {
                "type": "subagent_started",
                "subagent_id": task_id,
                "parent_type": parent_type,
                "parent_id": parent_id,
                "role_id": role_id,
                "title": title,
                "model_id": model_id,
                "prompt_preview": prompt_preview,
                "status": active_status,
                "summary": summary,
                "metadata": common_metadata,
                "created_at": created_at,
            },
            fallback_turn_id=turn_id,
        ),
        _attach_turn_id(
            None,
            {
                "type": "subagent_prompt",
                "subagent_id": task_id,
                "parent_type": parent_type,
                "parent_id": parent_id,
                "role_id": role_id,
                "title": title,
                "model_id": model_id,
                "system_prompt": "Delegate a bounded read-only background subagent task.",
                "user_prompt": objective,
                "prompt_preview": prompt_preview,
                "status": active_status,
                "metadata": {
                    **common_metadata,
                    "expected_artifacts": list(request_args.get("expected_artifacts") or []),
                    "suggested_roles": list(request_args.get("suggested_roles") or []),
                    "execution_boundary": "delegated_subagents_remain_read_research_evidence_by_default",
                },
                "created_at": created_at,
            },
            fallback_turn_id=turn_id,
        ),
    ]
    if normalized_status == "waiting_approval":
        first_pending = pending_approvals[0] if pending_approvals and isinstance(pending_approvals[0], dict) else {}
        lifecycle_events.append(
            _attach_turn_id(
                None,
                {
                    "type": "subagent_tool_result",
                    "subagent_id": task_id,
                    "parent_type": parent_type,
                    "parent_id": parent_id,
                    "role_id": role_id,
                    "title": title,
                    "model_id": model_id,
                    "tool_call_id": str(first_pending.get("request_id") or "approval-request").strip() or None,
                    "tool_name": str(first_pending.get("tool_name") or tool_names[0] if tool_names else "").strip() or None,
                    "status": "approval_required",
                    "summary": "Delegated subagent is waiting for approval before the requested tool can run.",
                    "metadata": {
                        **common_metadata,
                        "approval_id": first_pending.get("approval_id") or (approval_ids[0] if approval_ids else None),
                        "reason": first_pending.get("reason"),
                        "approval_kind": first_pending.get("approval_kind"),
                        "approval_scope": first_pending.get("approval_scope"),
                        "replay_safe": first_pending.get("replay_safe"),
                        "security_decision": first_pending.get("security_decision"),
                        "policy_source": first_pending.get("policy_source"),
                        "allowed_decisions": first_pending.get("allowed_decisions"),
                    },
                    "created_at": created_at,
                },
                fallback_turn_id=turn_id,
            )
        )
        lifecycle_events.append(
            _attach_turn_id(
                None,
                {
                    "type": "runtime_blocked",
                    "parent_type": parent_type,
                    "parent_id": parent_id,
                    "subagent_id": task_id,
                    "role_id": role_id,
                    "title": title,
                    "status": "blocked",
                    "blocker_type": "approval",
                    "summary": "Delegated subagent is waiting on approval.",
                    "approval_ids": approval_ids,
                    "tool_names": tool_names,
                    "recommended_action": "resolve_approval",
                    "pending_approvals": pending_approvals,
                    "metadata": common_metadata,
                    "created_at": created_at,
                },
                fallback_turn_id=turn_id,
            )
        )
        return lifecycle_events
    lifecycle_events.extend([
        _attach_turn_id(
            None,
            {
                "type": lifecycle_event_type,
                "subagent_id": task_id,
                "parent_type": parent_type,
                "parent_id": parent_id,
                "role_id": role_id,
                "title": title,
                "model_id": model_id,
                "status": normalized_status,
                "summary": lifecycle_summary,
                "content": (
                    _truncate_preview(
                        str(result_payload.get("final_answer") or result_payload.get("display_name") or title),
                        240,
                    )
                    if terminal
                    else None
                ),
                "metadata": common_metadata,
                "created_at": created_at,
            },
            fallback_turn_id=turn_id,
        ),
    ])
    return lifecycle_events


def _normalize_delegate_subagent_status(status: str) -> tuple[str, bool]:
    normalized = status.strip().lower()
    if normalized in {"awaiting_approval", "waiting_approval", "approval_required", "approval_pending"}:
        return "waiting_approval", False
    if normalized in {"completed", "succeeded", "done"}:
        return "completed", True
    if normalized in {"failed", "error"}:
        return "failed", True
    if normalized in {"cancelled", "canceled"}:
        return "cancelled", True
    if normalized in {"queued", "pending", "created"}:
        return "queued", False
    if normalized in {"running", "resumed", "in_progress", "in-progress"}:
        return "running", False
    return normalized or "queued", False


def _first_model_id_hint(request_args: dict[str, Any]) -> str | None:
    suggested_models = request_args.get("suggested_models")
    if isinstance(suggested_models, dict):
        for value in suggested_models.values():
            text = str(value).strip()
            if text:
                return text
    return "gpt-5.4"


def _truncate_preview(value: str, max_chars: int) -> str:
    text = value.strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 1].rstrip() + "..."


async def _persist_chat_subagent_events(
    request: Request,
    *,
    session_id: str,
    turn_id: str,
    events: list[dict[str, Any]],
) -> None:
    if not events:
        return
    from mochi.api.routes.approvals import _get_runtime_service

    service = await _get_runtime_service(request.app)
    for event in events:
        event_type = str(event.get("type") or "")
        if event_type not in {
            "subagent_started",
            "subagent_prompt",
            "subagent_progress",
            "subagent_thinking",
            "subagent_message_accepted",
            "subagent_message_applied",
            "subagent_message_cancelled",
            "subagent_message_deferred",
            "subagent_message_queued",
            "subagent_interrupted",
            "subagent_tool_call",
            "subagent_tool_cancel_requested",
            "subagent_tool_cancelled",
            "subagent_tool_cancel_deferred",
            "subagent_completed",
            "subagent_tool_result",
            "runtime_blocked",
        }:
            continue
        metadata = event.get("metadata")
        if (
            isinstance(metadata, dict)
            and metadata.get("source") == "delegate_subagent_task_runtime"
        ):
            continue
        subagent_id = str(event.get("subagent_id") or "").strip()
        if not subagent_id:
            continue
        await service._store.upsert_subagent_transcript(  # noqa: SLF001
            subagent_id=subagent_id,
            parent_type="chat_turn",
            parent_id=turn_id,
            session_id=session_id,
            parent_turn_id=turn_id,
            role_id=str(event.get("role_id") or "").strip() or None,
            title=str(event.get("title") or "").strip() or None,
            model_id=str(event.get("model_id") or "").strip() or None,
            status=str(event.get("status") or "running").strip() or "running",
            system_prompt=event.get("system_prompt"),
            user_prompt=event.get("user_prompt"),
            prompt_preview=event.get("prompt_preview") or event.get("content"),
            summary=event.get("summary") or event.get("content"),
            metadata=dict(event.get("metadata") or {}),
        )
        await service._store.append_subagent_transcript_event(subagent_id, dict(event))  # noqa: SLF001
