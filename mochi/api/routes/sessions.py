"""Session bounded API routes。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from mochi.api.routes.projects import _get_project_store
from mochi.api.server import _get_config
from mochi.sessions.store import SessionStore

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


class UpdateSessionProjectRequest(BaseModel):
    """Update session project assignment request."""

    project_id: str | None = None


class RewriteSessionFromTurnRequest(BaseModel):
    """Rewrite a session by removing conversation events from one turn onward."""

    from_turn_id: str


class AppendSessionEventsRequest(BaseModel):
    """Append one or more replayable session events."""

    events: list[dict[str, object]]


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
) -> dict[str, str]:
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

        return {"type": "session", "session_id": session_id}

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
    return {"type": "session", "session_id": session_id}


@router.get("/sessions")
async def list_sessions(http_request: Request) -> dict[str, object]:
    """列出所有 session 摘要。"""
    app = http_request.app
    config = await _get_config(app)
    store = _get_session_store(app, config=config)
    return {"type": "sessions", "items": await _list_session_summaries(store)}


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
        "events": events,
    }


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
    if title is None and workflow is None:
        raise HTTPException(status_code=422, detail="title or workflow is required")
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
    events = await store.load_session(session_id)
    return {
        "type": "session",
        "session_id": session_id,
        "title": _session_title(session_id, events),
        "project_id": _session_project_id(events),
        "workflow": _session_workflow_state(events),
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
