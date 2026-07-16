"""Session bounded API routes。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping, cast
from uuid import uuid4

from fastapi import APIRouter, Body, HTTPException, Request
from pydantic import BaseModel, Field

from mochi.api.routes.approvals import _get_runtime_service
from mochi.api.routes.projects import _get_project_store
from mochi.api.server import _get_config
from mochi.runtime.models import (
    AgentRunMessageRequest,
    SessionSubagentMessageRequest,
    SubagentTranscriptDetail,
    SubagentTranscriptSummary,
)
from mochi.sessions.store import SessionStore
from mochi.terminal_goal_helpers import normalize_goal_session_state

router = APIRouter(prefix="/v1", tags=["sessions"])


class CreateSessionRequest(BaseModel):
    """建立 session request。"""

    session_id: str | None = None
    project_id: str | None = None
    fork_from_session_id: str | None = None
    fork_until_turn_id: str | None = None


class UpdateSessionRequest(BaseModel):
    """更新 session metadata request。"""

    title: str | None = None
    workflow: dict[str, object] | None = None
    goal: dict[str, object] | None = None
    security_override: dict[str, object] | None = None


class UpdateSessionProjectRequest(BaseModel):
    """Update session project assignment request."""

    project_id: str | None = None


class RewriteSessionFromTurnRequest(BaseModel):
    """Rewrite a session by removing conversation events from one turn onward."""

    from_turn_id: str


class AppendSessionEventsRequest(BaseModel):
    """Append one or more replayable session events."""

    events: list[dict[str, object]]


class SessionSubagentActionResponse(BaseModel):
    """Result of an action against a session-scoped subagent transcript."""

    type: str = "session_subagent_action"
    action: str
    session_id: str
    subagent_id: str
    task_id: str
    task: dict[str, Any] | None = None
    transcript: SubagentTranscriptDetail


class SessionSubagentResumeRequest(BaseModel):
    """Optional guidance payload for resuming a session-scoped subagent task."""

    role: str | None = "operator"
    content: str = ""
    guidance: str | None = None
    project_id: str | None = None
    workspace_dir: str | None = None
    attachments: list[dict[str, Any]] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_message_request(self) -> AgentRunMessageRequest | None:
        content = (self.content or "").strip() or (self.guidance or "").strip()
        if not content and not self.attachments:
            return None
        role = "operator" if self.role not in {"user", "operator"} else self.role
        return AgentRunMessageRequest.model_validate(
            {
                "role": role,
                "content": content,
                "project_id": self.project_id,
                "workspace_dir": self.workspace_dir,
                "attachments": self.attachments,
                "metadata": self.metadata,
            }
        )


def _initial_subagent_message_delivery(
    payload: SessionSubagentMessageRequest,
) -> tuple[str, str | None]:
    if payload.delivery_mode == "resume_only":
        return "accepted", None
    if payload.cancel_current_tool:
        return "queued", "tool_cancel_pending"
    if payload.interrupt:
        return "queued", "interrupt_pending"
    return "queued", "runtime_safe_point_pending"


def _message_request_with_delivery_metadata(
    payload: SessionSubagentMessageRequest,
) -> tuple[SessionSubagentMessageRequest, dict[str, Any]]:
    delivery_status, delivery_reason = _initial_subagent_message_delivery(payload)
    delivery: dict[str, Any] = {
        "message_id": uuid4().hex,
        "delivery_mode": payload.delivery_mode,
        "delivery_status": delivery_status,
        "interrupt": payload.interrupt,
        "cancel_current_tool": payload.cancel_current_tool,
    }
    if delivery_reason:
        delivery["delivery_reason"] = delivery_reason

    metadata = dict(payload.metadata or {})
    metadata.update(delivery)
    return payload.model_copy(update={"metadata": metadata}), delivery


def _subagent_message_response_with_delivery(
    transcript: SubagentTranscriptDetail,
    delivery: Mapping[str, Any],
) -> SubagentTranscriptDetail:
    message_id = str(delivery.get("message_id") or "")
    events = []
    for event in transcript.events:
        event_message_id = str(event.metadata.get("message_id") or event.message_id or "")
        if message_id and event_message_id == message_id:
            event = event.model_copy(update=dict(delivery))
        events.append(event)
    return transcript.model_copy(update={**dict(delivery), "events": events})


def _get_session_store(app: object, *, config: object | None = None) -> SessionStore:
    """從 app state 或 config 取得 SessionStore。"""
    existing = getattr(app.state, "session_store", None)
    if isinstance(existing, SessionStore):
        return existing

    if config is None:
        raise RuntimeError("config is required when app.state.session_store is not set.")

    store = SessionStore(cast(object, config).sessions_dir)
    app.state.session_store = store
    return store


async def _protected_workspace_projection(app: object, session_id: str) -> dict[str, Any]:
    service = await _get_runtime_service(app)
    return service.protected_workspace_session_projection(session_id=session_id)


async def _list_session_summaries(store: SessionStore) -> list[dict[str, object]]:
    """掃描 sessions 目錄，回傳摘要列表。"""
    sessions_dir = Path(store._sessions_dir).expanduser()  # noqa: SLF001
    if not sessions_dir.exists():
        return []

    summaries: list[dict[str, object]] = []
    for path in await asyncio.to_thread(lambda: sorted(sessions_dir.glob("*.jsonl"))):
        events = await store.load_session(path.stem)
        updated_at = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC).isoformat()
        title = _session_title(path.stem, events)
        summaries.append(
            {
                "session_id": path.stem,
                "title": title,
                "event_count": len(events),
                "updated_at": updated_at,
                "project_id": _session_project_id(events),
                "workflow": _session_workflow_state(events),
                "goal": _session_goal_state(events),
                "security_override": _session_security_override(events),
            }
        )

    summaries.sort(key=lambda item: str(item["updated_at"]), reverse=True)
    return summaries


def _session_title(session_id: str, events: list[dict]) -> str:
    """從 metadata 或首則 user message 推導 session 顯示名稱。"""
    for event in reversed(events):
        if (
            event.get("type") == "session_meta"
            and event.get("event") == "renamed"
            and isinstance(event.get("title"), str)
            and event["title"].strip()
        ):
            return event["title"].strip()

    for event in events:
        if (
            event.get("type") == "message"
            and event.get("role") == "user"
            and isinstance(event.get("content"), str)
            and event["content"].strip()
        ):
            return event["content"].strip()[:80]

        if event.get("type") == "message" and event.get("role") == "user":
            attachments = event.get("attachments")
            if isinstance(attachments, list):
                names = [
                    item.get("name", "").strip()
                    for item in attachments
                    if isinstance(item, dict) and isinstance(item.get("name"), str)
                ]
                if names:
                    return ", ".join(names)[:80]

    return session_id


def _session_project_id(events: list[dict]) -> str | None:
    """Resolve latest project assignment from metadata events."""
    for event in reversed(events):
        if event.get("type") != "session_meta":
            continue
        if event.get("event") != "project_assigned":
            continue
        project_id = event.get("project_id")
        if project_id is None:
            return None
        if isinstance(project_id, str) and project_id.strip():
            return project_id.strip()
    return None


def _session_workflow_state(events: list[dict]) -> dict[str, object] | None:
    """Resolve latest workflow state from metadata events."""
    for event in reversed(events):
        if event.get("type") != "session_meta":
            continue
        if event.get("event") != "workflow_state_updated":
            continue
        workflow = event.get("workflow")
        if isinstance(workflow, dict):
            return dict(workflow)
    return None


def _session_goal_state(events: list[dict]) -> dict[str, object] | None:
    """Resolve latest goal state from metadata events."""
    for event in reversed(events):
        if event.get("type") != "session_meta":
            continue
        if event.get("event") != "goal_state_updated":
            continue
        goal = event.get("goal")
        if isinstance(goal, dict):
            return normalize_goal_session_state(dict(goal))
    return None


def _normalize_session_security_override(
    value: dict[str, object] | None,
) -> dict[str, object] | None:
    if not isinstance(value, dict):
        return None
    autonomy_mode = value.get("autonomy_mode")
    if autonomy_mode not in {"strict", "trusted_workspace", "auto_review", "high_autonomy"}:
        return None
    return {"autonomy_mode": str(autonomy_mode)}


def _session_security_override(events: list[dict]) -> dict[str, object] | None:
    """Resolve latest security override from metadata events."""
    for event in reversed(events):
        if event.get("type") != "session_meta":
            continue
        if event.get("event") != "security_override_updated":
            continue
        override = event.get("security_override")
        if isinstance(override, dict):
            return _normalize_session_security_override(dict(override))
    return None


async def _append_project_assignment_event(
    store: SessionStore,
    session_id: str,
    project_id: str | None,
) -> None:
    await store.save_event(
        session_id,
        {
            "type": "session_meta",
            "event": "project_assigned",
            "session_id": session_id,
            "project_id": project_id,
            "timestamp": datetime.now(tz=UTC).isoformat(),
        },
    )


def _cloneable_session_events(
    events: list[dict],
    *,
    until_turn_id: str,
) -> list[dict]:
    """Return replayable events through the selected assistant turn."""
    cloned: list[dict] = []

    for event in events:
        if event.get("type") == "session_meta":
            continue

        cloned.append(dict(event))
        if (
            event.get("type") == "message"
            and event.get("role") == "assistant"
            and event.get("turn_id") == until_turn_id
        ):
            return cloned

    raise HTTPException(status_code=404, detail="Fork turn not found")


def _rewriteable_session_events_before_turn(
    events: list[dict],
    *,
    from_turn_id: str,
) -> list[dict]:
    """Keep session metadata and conversation events strictly before the target turn."""
    rewritten: list[dict] = []
    found_target = False

    for event in events:
        if event.get("type") == "session_meta":
            rewritten.append(dict(event))
            continue

        if event.get("turn_id") == from_turn_id:
            found_target = True
            continue

        if found_target:
            continue

        rewritten.append(dict(event))

    if not found_target:
        raise HTTPException(status_code=404, detail="Target turn not found")

    return rewritten


async def _clear_project_from_sessions(
    app: object,
    project_id: str,
    *,
    config: object | None = None,
) -> None:
    store = _get_session_store(app, config=config)
    sessions_dir = Path(store._sessions_dir).expanduser()  # noqa: SLF001
    if not sessions_dir.exists():
        return

    for path in await asyncio.to_thread(lambda: sorted(sessions_dir.glob("*.jsonl"))):
        events = await store.load_session(path.stem)
        if _session_project_id(events) == project_id:
            await _append_project_assignment_event(store, path.stem, None)


@router.post("/sessions")
async def create_session(
    request: CreateSessionRequest | None = None,
    *,
    http_request: Request,
) -> dict[str, object]:
    """建立新 session，並寫入 metadata event。"""
    app = http_request.app
    config = await _get_config(app)
    store = _get_session_store(app, config=config)
    session_id = (request.session_id if request is not None else None) or str(uuid4())
    now = datetime.now(tz=UTC).isoformat()

    if request is not None and request.fork_from_session_id is not None:
        source_session_id = request.fork_from_session_id.strip()
        fork_until_turn_id = (request.fork_until_turn_id or "").strip()

        if not source_session_id:
            raise HTTPException(status_code=422, detail="fork_from_session_id must not be empty")
        if not fork_until_turn_id:
            raise HTTPException(
                status_code=422,
                detail="fork_until_turn_id is required when fork_from_session_id is provided",
            )

        source_events = await store.load_session(source_session_id)
        if not source_events:
            raise HTTPException(status_code=404, detail="Source session not found")

        effective_project_id = request.project_id
        if effective_project_id is None:
            effective_project_id = _session_project_id(source_events)

        if effective_project_id is not None:
            project_store = _get_project_store(app, config=config)
            project = await project_store.get_project(effective_project_id)
            if project is None:
                raise HTTPException(status_code=404, detail="Project not found")

        await store.save_event(
            session_id,
            {
                "type": "session_meta",
                "event": "created",
                "session_id": session_id,
                "timestamp": now,
            },
        )
        if effective_project_id is not None:
            await _append_project_assignment_event(store, session_id, effective_project_id)

        for event in _cloneable_session_events(source_events, until_turn_id=fork_until_turn_id):
            await store.save_event(session_id, event)

        return {
            "type": "session",
            "session_id": session_id,
            "protected_workspace": await _protected_workspace_projection(app, session_id),
        }

    if request is not None and request.project_id is not None:
        project_store = _get_project_store(app, config=config)
        project = await project_store.get_project(request.project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")

    await store.save_event(
        session_id,
        {
            "type": "session_meta",
            "event": "created",
            "session_id": session_id,
            "timestamp": now,
        },
    )
    if request is not None and request.project_id is not None:
        await _append_project_assignment_event(store, session_id, request.project_id)
    return {
        "type": "session",
        "session_id": session_id,
        "protected_workspace": await _protected_workspace_projection(app, session_id),
    }


@router.get("/sessions")
async def list_sessions(http_request: Request) -> dict[str, object]:
    """列出所有 session 摘要。"""
    app = http_request.app
    config = await _get_config(app)
    store = _get_session_store(app, config=config)
    items = await _list_session_summaries(store)
    for item in items:
        session_id = str(item["session_id"])
        item["protected_workspace"] = await _protected_workspace_projection(
            app, session_id
        )
    return {"type": "sessions", "items": items}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str, http_request: Request) -> dict[str, object]:
    """讀取單一 session 的事件列表。"""
    app = http_request.app
    config = await _get_config(app)
    store = _get_session_store(app, config=config)
    events = await store.load_session(session_id)
    return {
        "type": "session",
        "session_id": session_id,
        "title": _session_title(session_id, events),
        "project_id": _session_project_id(events),
        "workflow": _session_workflow_state(events),
        "goal": _session_goal_state(events),
        "security_override": _session_security_override(events),
        "protected_workspace": await _protected_workspace_projection(app, session_id),
        "events": events,
    }


@router.get("/sessions/{session_id}/subagents", response_model=list[SubagentTranscriptSummary])
async def list_session_subagents(
    session_id: str,
    http_request: Request,
) -> list[SubagentTranscriptSummary]:
    service = await _get_runtime_service(http_request.app)
    config = await _get_config(http_request.app)
    store = _get_session_store(http_request.app, config=config)
    if not await store.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    items = await service.list_session_subagents(session_id)
    return [SubagentTranscriptSummary.model_validate(item) for item in items]


@router.get(
    "/sessions/{session_id}/subagents/{subagent_id}",
    response_model=SubagentTranscriptDetail,
)
async def get_session_subagent(
    session_id: str,
    subagent_id: str,
    http_request: Request,
) -> SubagentTranscriptDetail:
    service = await _get_runtime_service(http_request.app)
    config = await _get_config(http_request.app)
    store = _get_session_store(http_request.app, config=config)
    if not await store.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    item = await service.get_session_subagent(session_id, subagent_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Session subagent not found")
    return SubagentTranscriptDetail.model_validate(item)


@router.post(
    "/sessions/{session_id}/subagents/{subagent_id}/messages",
    response_model=SubagentTranscriptDetail,
)
async def append_session_subagent_message(
    session_id: str,
    subagent_id: str,
    payload: SessionSubagentMessageRequest,
    http_request: Request,
) -> SubagentTranscriptDetail:
    service = await _get_runtime_service(http_request.app)
    config = await _get_config(http_request.app)
    store = _get_session_store(http_request.app, config=config)
    if not await store.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    payload_with_delivery, delivery = _message_request_with_delivery_metadata(payload)
    item = await service.append_session_subagent_message(
        session_id,
        subagent_id,
        payload_with_delivery,
    )
    if item is None:
        raise HTTPException(status_code=404, detail="Session subagent not found")
    transcript = SubagentTranscriptDetail.model_validate(item)
    return _subagent_message_response_with_delivery(transcript, delivery)


async def _get_session_subagent_or_404(
    http_request: Request,
    session_id: str,
    subagent_id: str,
) -> tuple[Any, SubagentTranscriptDetail]:
    service = await _get_runtime_service(http_request.app)
    config = await _get_config(http_request.app)
    store = _get_session_store(http_request.app, config=config)
    if not await store.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")
    item = await service.get_session_subagent(session_id, subagent_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Session subagent not found")
    return service, SubagentTranscriptDetail.model_validate(item)


def _metadata_task_id(metadata: Mapping[str, Any] | None) -> str | None:
    if not isinstance(metadata, Mapping):
        return None
    task_id = metadata.get("task_id")
    if isinstance(task_id, str) and task_id.strip():
        return task_id.strip()
    return None


def _resolve_session_subagent_task_id(transcript: SubagentTranscriptDetail) -> str | None:
    task_id = _metadata_task_id(transcript.metadata)
    if task_id is not None:
        return task_id

    for event in transcript.events:
        task_id = _metadata_task_id(event.metadata)
        if task_id is not None:
            return task_id

    if transcript.parent_type == "delegated_task" and transcript.parent_id.strip():
        return transcript.parent_id.strip()

    for event in transcript.events:
        if event.parent_type == "delegated_task" and event.parent_id and event.parent_id.strip():
            return event.parent_id.strip()

    return None


async def _refresh_session_subagent_transcript(
    service: Any,
    session_id: str,
    subagent_id: str,
    fallback: SubagentTranscriptDetail,
) -> SubagentTranscriptDetail:
    item = await service.get_session_subagent(session_id, subagent_id)
    if item is None:
        return fallback
    return SubagentTranscriptDetail.model_validate(item)


@router.post(
    "/sessions/{session_id}/subagents/{subagent_id}/cancel",
    response_model=SessionSubagentActionResponse,
)
async def cancel_session_subagent(
    session_id: str,
    subagent_id: str,
    http_request: Request,
) -> SessionSubagentActionResponse:
    service, transcript = await _get_session_subagent_or_404(
        http_request,
        session_id,
        subagent_id,
    )
    task_id = _resolve_session_subagent_task_id(transcript)
    if task_id is None:
        raise HTTPException(
            status_code=409,
            detail="Session subagent is not linked to a delegated task",
        )

    task = await service.cancel_task(task_id)
    refreshed = await _refresh_session_subagent_transcript(
        service,
        session_id,
        subagent_id,
        transcript,
    )
    return SessionSubagentActionResponse(
        action="cancel",
        session_id=session_id,
        subagent_id=subagent_id,
        task_id=task_id,
        task=task,
        transcript=refreshed,
    )


@router.post(
    "/sessions/{session_id}/subagents/{subagent_id}/resume",
    response_model=SessionSubagentActionResponse,
)
async def resume_session_subagent(
    session_id: str,
    subagent_id: str,
    http_request: Request,
    payload: SessionSubagentResumeRequest | None = Body(default=None),
) -> SessionSubagentActionResponse:
    service, transcript = await _get_session_subagent_or_404(
        http_request,
        session_id,
        subagent_id,
    )
    task_id = _resolve_session_subagent_task_id(transcript)
    if task_id is None:
        raise HTTPException(
            status_code=409,
            detail="Session subagent is not linked to a delegated task",
        )

    message_payload = payload.to_message_request() if payload is not None else None
    if message_payload is not None:
        updated = await service.append_session_subagent_message(session_id, subagent_id, message_payload)
        if updated is None:
            raise HTTPException(status_code=404, detail="Session subagent not found")
        transcript = SubagentTranscriptDetail.model_validate(updated)

    task = await service.resume_task(
        task_id,
        decision="approve_once",
        reason="Session subagent resume requested",
        rule=None,
    )
    refreshed = await _refresh_session_subagent_transcript(
        service,
        session_id,
        subagent_id,
        transcript,
    )
    return SessionSubagentActionResponse(
        action="resume",
        session_id=session_id,
        subagent_id=subagent_id,
        task_id=task_id,
        task=task,
        transcript=refreshed,
    )


@router.post("/sessions/{session_id}/rewrite-from-turn")
async def rewrite_session_from_turn(
    session_id: str,
    payload: RewriteSessionFromTurnRequest,
    http_request: Request,
) -> dict[str, object]:
    """Rewrite one existing session by removing conversation turns from the target turn onward."""
    target_turn_id = payload.from_turn_id.strip()
    if not target_turn_id:
        raise HTTPException(status_code=422, detail="from_turn_id must not be empty")

    app = http_request.app
    config = await _get_config(app)
    store = _get_session_store(app, config=config)
    events = await store.load_session(session_id)
    if not events:
        raise HTTPException(status_code=404, detail="Session not found")

    rewritten = _rewriteable_session_events_before_turn(events, from_turn_id=target_turn_id)
    await store.replace_session(session_id, rewritten)

    return {
        "type": "session",
        "session_id": session_id,
        "title": _session_title(session_id, rewritten),
        "project_id": _session_project_id(rewritten),
        "workflow": _session_workflow_state(rewritten),
        "goal": _session_goal_state(rewritten),
        "security_override": _session_security_override(rewritten),
        "events": rewritten,
    }


@router.post("/sessions/{session_id}/events")
async def append_session_events(
    session_id: str,
    payload: AppendSessionEventsRequest,
    http_request: Request,
) -> dict[str, object]:
    """Append replayable events to one session."""
    app = http_request.app
    config = await _get_config(app)
    store = _get_session_store(app, config=config)
    if not await store.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    for event in payload.events:
        await store.save_event(session_id, dict(event))

    events = await store.load_session(session_id)
    return {
        "type": "session",
        "session_id": session_id,
        "title": _session_title(session_id, events),
        "project_id": _session_project_id(events),
        "workflow": _session_workflow_state(events),
        "goal": _session_goal_state(events),
        "security_override": _session_security_override(events),
        "events": events,
    }


@router.patch("/sessions/{session_id}")
async def update_session(
    session_id: str,
    payload: UpdateSessionRequest,
    http_request: Request,
) -> dict[str, object]:
    """更新 session 顯示 metadata。"""
    title = payload.title.strip() if isinstance(payload.title, str) else None
    workflow = dict(payload.workflow) if isinstance(payload.workflow, dict) else None
    goal = normalize_goal_session_state(dict(payload.goal)) if isinstance(payload.goal, dict) else None
    security_override = _normalize_session_security_override(
        dict(payload.security_override) if isinstance(payload.security_override, dict) else None
    )
    if title is None and workflow is None and goal is None and security_override is None:
        raise HTTPException(
            status_code=422,
            detail="title, workflow, goal, or security_override is required",
        )
    if title is not None and not title:
        raise HTTPException(status_code=422, detail="title must not be empty")

    app = http_request.app
    config = await _get_config(app)
    store = _get_session_store(app, config=config)
    if not await store.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    now = datetime.now(tz=UTC).isoformat()
    if title is not None:
        await store.save_event(
            session_id,
            {
                "type": "session_meta",
                "event": "renamed",
                "session_id": session_id,
                "title": title,
                "timestamp": now,
            },
        )
    if workflow is not None:
        await store.save_event(
            session_id,
            {
                "type": "session_meta",
                "event": "workflow_state_updated",
                "session_id": session_id,
                "workflow": workflow,
                "timestamp": now,
            },
        )
    if goal is not None:
        await store.save_event(
            session_id,
            {
                "type": "session_meta",
                "event": "goal_state_updated",
                "session_id": session_id,
                "goal": goal,
                "timestamp": now,
            },
        )
    if security_override is not None:
        await store.save_event(
            session_id,
            {
                "type": "session_meta",
                "event": "security_override_updated",
                "session_id": session_id,
                "security_override": security_override,
                "timestamp": now,
            },
        )
    events = await store.load_session(session_id)
    return {
        "type": "session",
        "session_id": session_id,
        "title": _session_title(session_id, events),
        "project_id": _session_project_id(events),
        "workflow": _session_workflow_state(events),
        "goal": _session_goal_state(events),
        "security_override": _session_security_override(events),
        "protected_workspace": await _protected_workspace_projection(app, session_id),
        "events": events,
    }


@router.patch("/sessions/{session_id}/project")
async def update_session_project(
    session_id: str,
    payload: UpdateSessionProjectRequest,
    http_request: Request,
) -> dict[str, object]:
    """Update session project assignment."""
    app = http_request.app
    config = await _get_config(app)
    store = _get_session_store(app, config=config)
    if not await store.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    if payload.project_id is not None:
        project_store = _get_project_store(app, config=config)
        project = await project_store.get_project(payload.project_id)
        if project is None:
            raise HTTPException(status_code=404, detail="Project not found")

    await _append_project_assignment_event(store, session_id, payload.project_id)
    return {
        "type": "session",
        "session_id": session_id,
        "project_id": payload.project_id,
    }


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, http_request: Request) -> dict[str, object]:
    """刪除單一 session。"""
    app = http_request.app
    config = await _get_config(app)
    store = _get_session_store(app, config=config)
    deleted = await store.delete_session(session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"type": "session", "session_id": session_id, "deleted": True}
