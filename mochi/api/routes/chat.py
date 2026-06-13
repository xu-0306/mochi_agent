"""Bounded chat API routes."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from typing import Any, Literal
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from mochi.backends.inference_capabilities import ReasoningEffort
from mochi.backends.types import AttachmentRef
from mochi.api.attachment_schema import AttachmentPayload
from mochi.agents.events import (
    ErrorEvent,
    FinalAnswerEvent,
    StatusEvent,
    ThinkingEvent,
    ToolCallRequestEvent,
    ToolCallResultEvent,
)
from mochi.api.routes.workspace import resolve_workspace_scope
from mochi.api.server import _get_config, _get_or_create_engine, _maybe_await
from mochi.sessions.store import SessionStore
from mochi.utils.streaming import sse_stream

router = APIRouter(prefix="/v1")


class ChatRequest(BaseModel):
    """`POST /v1/chat` request payload。"""

    message: str = Field(min_length=0)
    session_id: str | None = None
    project_id: str | None = None
    model: str | None = Field(default=None, min_length=1)
    system_prompt: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=131072)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    repeat_penalty: float | None = Field(default=None, ge=0.0, le=2.0)
    reasoning_effort: ReasoningEffort | None = None
    selected_skill_ids: list[str] | None = None
    attachments: list[AttachmentPayload] | None = None


class ChatResponse(BaseModel):
    """`POST /v1/chat` response payload。"""

    type: str = "chat_response"
    session_id: str
    turn_id: str | None = None
    final_answer: str
    trajectory_id: str | None = None
    events: list[dict[str, Any]]


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


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    """執行 bounded 單輪文字對話並回傳完整事件列表。"""
    if payload.model:
        from mochi.api.routes.models import switch_model_runtime

        await switch_model_runtime(request, payload.model)
    engine = await _get_or_create_engine(request.app)
    await _ensure_runtime_delegate(request)
    session_id = payload.session_id or str(uuid4())
    resolved_project_id, resolved_workspace_dir = await _resolve_chat_project_context(
        request,
        payload,
        session_id,
    )

    stream = await _maybe_await(
        engine.chat(
            payload.message,
            session_id=session_id,
            inference_overrides=_build_inference_overrides(payload),
            project_id=resolved_project_id,
            workspace_dir=resolved_workspace_dir,
            selected_skill_ids=payload.selected_skill_ids,
            attachments=_resolve_chat_attachments(payload),
        )
    )
    events, final_answer, trajectory_id = await _collect_chat_result(stream)
    turn_id = _response_turn_id(events)
    if turn_id is None:
        turn_id = await _persist_turn_events(request, session_id, events)

    return ChatResponse(
        session_id=session_id,
        turn_id=turn_id,
        final_answer=final_answer,
        trajectory_id=trajectory_id,
        events=events,
    )


@router.post("/chat/context", response_model=ChatContextResponse)
async def chat_context(request: Request, payload: ChatRequest) -> ChatContextResponse:
    """Preview the next-request context budget without sending a chat turn."""
    if payload.model:
        from mochi.api.routes.models import switch_model_runtime

        await switch_model_runtime(request, payload.model)
    engine = await _get_or_create_engine(request.app)
    await _ensure_runtime_delegate(request)
    session_id = payload.session_id or "draft-session"
    resolved_project_id, resolved_workspace_dir = await _resolve_chat_project_context(
        request,
        payload,
        session_id,
    )

    preview = await _maybe_await(
        engine.preview_chat_context(
            payload.message,
            session_id=session_id,
            inference_overrides=_build_inference_overrides(payload),
            project_id=resolved_project_id,
            workspace_dir=resolved_workspace_dir,
            selected_skill_ids=payload.selected_skill_ids,
            attachments=_resolve_chat_attachments(payload),
        )
    )
    if isinstance(preview, dict):
        return ChatContextResponse.model_validate(preview)
    raise HTTPException(status_code=500, detail="Engine did not return a chat context snapshot.")


@router.post("/chat/stream")
async def chat_stream(request: Request, payload: ChatRequest) -> StreamingResponse:
    """以 SSE 串流回傳 chat event stream。"""
    if payload.model:
        from mochi.api.routes.models import switch_model_runtime

        await switch_model_runtime(request, payload.model)
    engine = await _get_or_create_engine(request.app)
    session_id = payload.session_id or str(uuid4())
    resolved_project_id, resolved_workspace_dir = await _resolve_chat_project_context(
        request,
        payload,
        session_id,
    )

    stream = await _maybe_await(
        engine.chat(
            payload.message,
            session_id=session_id,
            inference_overrides=_build_inference_overrides(payload),
            project_id=resolved_project_id,
            workspace_dir=resolved_workspace_dir,
            selected_skill_ids=payload.selected_skill_ids,
            attachments=_resolve_chat_attachments(payload),
        )
    )
    headers = {
        "Cache-Control": "no-cache",
        "X-Session-ID": session_id,
        "X-Accel-Buffering": "no",
    }
    return StreamingResponse(
        sse_stream(_stream_chat_events(request, session_id, stream)),
        media_type="text/event-stream",
        headers=headers,
    )


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
) -> AsyncIterator[dict[str, Any]]:
    """串流 serialized chat events，必要時補做 session replay 持久化。"""
    fallback_turn_id = str(uuid4())
    events: list[dict[str, Any]] = []

    try:
        async for event in stream:
            serialized = _serialize_event(event, fallback_turn_id=fallback_turn_id)
            events.append(serialized)
            yield serialized
    except Exception as exc:
        error_event = _attach_turn_id(
            None,
            {
                "type": "error",
                "error": str(exc),
                "code": "CHAT_STREAM_ERROR",
            },
            fallback_turn_id=fallback_turn_id,
        )
        events.append(error_event)
        yield error_event
    finally:
        if _should_persist_fallback_events(events, fallback_turn_id):
            await _persist_turn_events(request, session_id, events, turn_id=fallback_turn_id)


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
        return _attach_turn_id(
            event,
            {
                "type": event.type,
                "content": event.content,
                "trajectory_id": event.trajectory_id,
                "input_tokens": event.input_tokens,
                "output_tokens": event.output_tokens,
                "generation_time_ms": event.generation_time_ms,
                "finish_reason": event.finish_reason,
            },
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
    overrides = {
        "system_prompt": payload.system_prompt,
        "temperature": payload.temperature,
        "max_tokens": payload.max_tokens,
        "top_p": payload.top_p,
        "min_p": payload.min_p,
        "top_k": payload.top_k,
        "frequency_penalty": payload.frequency_penalty,
        "presence_penalty": payload.presence_penalty,
        "repeat_penalty": payload.repeat_penalty,
        "reasoning_effort": payload.reasoning_effort,
    }
    return {key: value for key, value in overrides.items() if value is not None}


def _resolve_chat_attachments(payload: ChatRequest) -> list[AttachmentRef]:
    return [attachment.to_attachment_ref() for attachment in payload.attachments or []]


async def _persist_turn_events(
    request: Request,
    session_id: str,
    events: list[dict[str, Any]],
    *,
    turn_id: str | None = None,
) -> str | None:
    """將本輪 replay event 以 `turn_event` schema 追加到 session JSONL。"""
    if not events:
        return None

    store = await _get_session_store(request)
    resolved_turn_id = turn_id or str(uuid4())

    for index, event in enumerate(events, start=1):
        phase = _event_phase(event)
        if phase is None:
            continue

        await store.save_event(
            session_id,
            {
                "type": "turn_event",
                "schema_version": 1,
                "turn_id": resolved_turn_id,
                "event_id": str(uuid4()),
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


def _should_persist_fallback_events(
    events: list[dict[str, Any]],
    fallback_turn_id: str,
) -> bool:
    """判斷是否需要以 route fallback 寫入 turn replay events。"""
    if not events:
        return False
    return _response_turn_id(events) == fallback_turn_id


async def _get_session_store(request: Request) -> SessionStore:
    """取得 chat route 可共用的 SessionStore。"""
    existing = getattr(request.app.state, "session_store", None)
    if isinstance(existing, SessionStore):
        return existing

    config = await _get_config(request.app)
    store = SessionStore(config.sessions_dir)
    request.app.state.session_store = store
    return store


async def _ensure_runtime_delegate(request: Request) -> None:
    from mochi.api.routes.approvals import _get_runtime_service

    await _get_runtime_service(request.app)


async def _resolve_chat_project_context(
    request: Request,
    payload: ChatRequest,
    session_id: str,
) -> tuple[str | None, str]:
    """Resolve effective project assignment and workspace for one request."""
    resolved_project_id, workspace_root = await resolve_workspace_scope(
        request,
        session_id=session_id,
        project_id=payload.project_id,
    )
    return resolved_project_id, str(workspace_root)


def _event_phase(event: dict[str, Any]) -> str | None:
    """將 API event type 映射為 replay phase。"""
    event_type = event.get("type")
    if event_type == "thinking":
        return "thinking"
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
