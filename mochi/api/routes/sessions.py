"""Session bounded API routes。"""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path
from typing import cast
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from mochi.api.server import _get_config
from mochi.sessions.store import SessionStore

router = APIRouter(prefix="/v1", tags=["sessions"])


class CreateSessionRequest(BaseModel):
    """建立 session request。"""

    session_id: str | None = None


class UpdateSessionRequest(BaseModel):
    """更新 session metadata request。"""

    title: str


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

    return session_id


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

    await store.save_event(
        session_id,
        {
            "type": "session_meta",
            "event": "created",
            "session_id": session_id,
            "timestamp": now,
        },
    )
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
        "events": events,
    }


@router.patch("/sessions/{session_id}")
async def update_session(
    session_id: str,
    payload: UpdateSessionRequest,
    http_request: Request,
) -> dict[str, object]:
    """更新 session 顯示 metadata。"""
    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=422, detail="title must not be empty")

    app = http_request.app
    config = await _get_config(app)
    store = _get_session_store(app, config=config)
    if not await store.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    now = datetime.now(tz=UTC).isoformat()
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
    return {"type": "session", "session_id": session_id, "title": title}


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
