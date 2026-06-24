from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, AsyncIterator

from fastapi.testclient import TestClient

from mochi.agents.events import FinalAnswerEvent, ThinkingEvent
from mochi.api.server import create_app
from mochi.backends.types import AttachmentRef, ModelInfo
from mochi.config.schema import MochiConfig
from mochi.sessions.store import SessionStore


def _create_test_app(*, config: MochiConfig, session_store: SessionStore | None = None):
    app = create_app()
    app.state.config_factory = lambda: config
    if session_store is not None:
        app.state.session_store = session_store
    return app


class _ContextPreviewEngine:
    def __init__(self) -> None:
        self.preview_calls: list[dict[str, Any]] = []
        self.model_info = ModelInfo(
            name="gpt-test",
            backend_type="openai_compat",
            context_length=4096,
            supports_tool_calling=True,
            metadata={"api_mode": "responses"},
        )

    async def preview_chat_context(
        self,
        message: str,
        session_id: str | None = None,
        inference_overrides: dict[str, Any] | None = None,
        project_id: str | None = None,
        workspace_dir: str | None = None,
        selected_skill_ids: list[str] | None = None,
        attachments: list[AttachmentRef] | None = None,
    ) -> dict[str, Any]:
        self.preview_calls.append(
            {
                "message": message,
                "session_id": session_id,
                "inference_overrides": inference_overrides,
                "project_id": project_id,
                "workspace_dir": workspace_dir,
                "selected_skill_ids": selected_skill_ids,
                "attachments": attachments,
            }
        )
        return {
            "type": "chat_context",
            "session_id": session_id or "draft-session",
            "model": self.model_info.name,
            "context_length": self.model_info.context_length,
            "estimated_prompt_tokens": 1200,
            "reserved_output_tokens": 512,
            "remaining_tokens": 2384,
            "usage_ratio": 0.418,
            "summary_tokens": 240,
            "history_tokens": 760,
            "memory_tokens": 120,
            "skills_tokens": 30,
            "tool_tokens": 50,
            "draft_tokens": 36,
            "compaction_triggered": True,
            "compaction_reason": "history_window",
            "approximate": True,
            "reasoning_effort": inference_overrides.get("reasoning_effort") if inference_overrides else None,
        }

    def get_model_info(self) -> ModelInfo:
        return self.model_info


def test_chat_context_preview_returns_budget_and_compaction_snapshot(tmp_path) -> None:
    """`POST /v1/chat/context` 應回傳下一輪推論所需的 context 預覽。"""
    config = MochiConfig.model_validate(
        {
            "model": "gpt-test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
        }
    )
    app = _create_test_app(config=config, session_store=SessionStore(tmp_path / "sessions"))
    engine = _ContextPreviewEngine()
    app.state.engine = engine

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/context",
            json={
                "message": "Please summarize the latest changes.",
                "session_id": "session-ctx",
                "model": "gpt-test",
                "system_prompt": "You are Mochi.",
                "max_tokens": 512,
                "reasoning_effort": "high",
                "selected_skill_ids": ["skill-a"],
                "attachments": [
                    {
                        "name": "notes.docx",
                        "path": str(tmp_path / "notes.docx"),
                        "source": "workspace_selection",
                        "line_start": 4,
                        "line_end": 6,
                        "quote": "selected summary block",
                        "note": "Summarize only this section.",
                        "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    }
                ],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "chat_context"
    assert payload["session_id"] == "session-ctx"
    assert payload["context_length"] == 4096
    assert payload["remaining_tokens"] == 2384
    assert payload["compaction_triggered"] is True
    assert payload["reasoning_effort"] == "high"
    assert engine.preview_calls[0]["selected_skill_ids"] == ["skill-a"]
    assert engine.preview_calls[0]["attachments"] == [
        AttachmentRef(
            name="notes.docx",
            path=str(tmp_path / "notes.docx"),
            size=None,
            content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            source="workspace_selection",
            line_start=4,
            line_end=6,
            quote="selected summary block",
            note="Summarize only this section.",
        )
    ]


def test_chat_context_preview_rejects_invalid_reasoning_effort(tmp_path) -> None:
    """`reasoning_effort` should be normalized at the REST boundary."""
    config = MochiConfig.model_validate(
        {
            "model": "gpt-test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
        }
    )
    app = _create_test_app(config=config, session_store=SessionStore(tmp_path / "sessions"))
    app.state.engine = _ContextPreviewEngine()

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/context",
            json={
                "message": "Please summarize the latest changes.",
                "reasoning_effort": "extreme",
            },
        )

    assert response.status_code == 422


def test_chat_context_preview_accepts_xhigh_reasoning_effort(tmp_path) -> None:
    """`reasoning_effort` should accept newer normalized values at the REST boundary."""
    config = MochiConfig.model_validate(
        {
            "model": "gpt-test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
        }
    )
    app = _create_test_app(config=config, session_store=SessionStore(tmp_path / "sessions"))
    app.state.engine = _ContextPreviewEngine()

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat/context",
            json={
                "message": "Please summarize the latest changes.",
                "reasoning_effort": "xhigh",
            },
        )

    assert response.status_code == 200
    assert response.json()["reasoning_effort"] == "xhigh"
