"""Bounded chat API routes."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Request
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from mochi.agents.events import (
    ErrorEvent,
    FinalAnswerEvent,
    ThinkingEvent,
    ToolCallRequestEvent,
    ToolCallResultEvent,
)
from mochi.api.server import _get_config, _get_or_create_engine, _maybe_await
from mochi.sessions.store import SessionStore

router = APIRouter(prefix="/v1")


class ChatRequest(BaseModel):
    """`POST /v1/chat` request payload。"""

    message: str = Field(min_length=1)
    session_id: str | None = None


class ChatResponse(BaseModel):
    """`POST /v1/chat` response payload。"""

    type: str = "chat_response"
    session_id: str
    turn_id: str | None = None
    final_answer: str
    trajectory_id: str | None = None
    events: list[dict[str, Any]]


@router.post("/chat", response_model=ChatResponse)
async def chat(request: Request, payload: ChatRequest) -> ChatResponse:
    """執行 bounded 單輪文字對話並回傳完整事件列表。"""
    engine = await _get_or_create_engine(request.app)
    session_id = payload.session_id or str(uuid4())

    stream = await _maybe_await(engine.chat(payload.message, session_id=session_id))
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


def _serialize_event(event: Any) -> dict[str, Any]:
    """將 AgentEvent 轉成 JSON-safe dict。"""
    if isinstance(event, ThinkingEvent):
        return _attach_turn_id(event, {"type": event.type, "content": event.content})
    if isinstance(event, ToolCallRequestEvent):
        return _attach_turn_id(event, {
            "type": event.type,
            "call_id": event.call_id,
            "tool_name": event.tool_name,
            "arguments": jsonable_encoder(event.arguments),
        })
    if isinstance(event, ToolCallResultEvent):
        return _attach_turn_id(event, {
            "type": event.type,
            "call_id": event.call_id,
            "tool_name": event.tool_name,
            "result": _json_safe(event.result),
            "error": event.error,
        })
    if isinstance(event, FinalAnswerEvent):
        return _attach_turn_id(event, {
            "type": event.type,
            "content": event.content,
            "trajectory_id": event.trajectory_id,
        })
    if isinstance(event, ErrorEvent):
        return _attach_turn_id(event, {"type": event.type, "error": event.message, "code": event.code})
    if is_dataclass(event):
        return jsonable_encoder(asdict(event))
    if isinstance(event, dict):
        return jsonable_encoder(event)
    return {"type": "unknown", "content": _json_safe(event)}


async def _persist_turn_events(
    request: Request,
    session_id: str,
    events: list[dict[str, Any]],
) -> str | None:
    """將本輪 replay event 以 `turn_event` schema 追加到 session JSONL。"""
    if not events:
        return None

    store = await _get_session_store(request)
    turn_id = str(uuid4())

    for index, event in enumerate(events, start=1):
        phase = _event_phase(event)
        if phase is None:
            continue

        await store.save_event(
            session_id,
            {
                "type": "turn_event",
                "schema_version": 1,
                "turn_id": turn_id,
                "event_id": str(uuid4()),
                "seq": index,
                "phase": phase,
                "timestamp": datetime.now(UTC).isoformat(),
                "payload": event,
            },
        )
    return turn_id


def _attach_turn_id(event: Any, payload: dict[str, Any]) -> dict[str, Any]:
    """若 AgentEvent 帶有 turn_id，附加到 API event。"""
    turn_id = getattr(event, "turn_id", None)
    if isinstance(turn_id, str) and turn_id:
        payload["turn_id"] = turn_id
    return payload


def _response_turn_id(events: list[dict[str, Any]]) -> str | None:
    """從 serialized events 取得本輪 turn_id。"""
    for event in events:
        turn_id = event.get("turn_id")
        if isinstance(turn_id, str) and turn_id:
            return turn_id
    return None


async def _get_session_store(request: Request) -> SessionStore:
    """取得 chat route 可共用的 SessionStore。"""
    existing = getattr(request.app.state, "session_store", None)
    if isinstance(existing, SessionStore):
        return existing

    config = await _get_config(request.app)
    store = SessionStore(config.sessions_dir)
    request.app.state.session_store = store
    return store


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
