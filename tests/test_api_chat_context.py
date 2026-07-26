from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

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
        permission_policy: dict[str, Any] | None = None,
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
                "permission_policy": permission_policy,
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
                "reserve_output_tokens": 768,
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
    workflow = payload["tool_workflow"]
    assert workflow["tool_inventory"]["catalog_scope"] == "policy_eligible"
    assert workflow["effective_policy"]["policy_snapshot_id"].startswith("policy-")
    assert workflow["effective_policy"]["review_semantics"] == "concrete_call_only"
    assert workflow["activation"]["status"] == "not_observed"
    assert engine.preview_calls[0]["inference_overrides"]["reserve_output_tokens"] == 768
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


def test_chat_context_preview_preserves_explicit_auto_token_overrides(tmp_path) -> None:
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
                "message": "Use auto token budgeting.",
                "max_tokens": None,
                "reserve_output_tokens": None,
            },
        )

    assert response.status_code == 200
    assert engine.preview_calls[0]["inference_overrides"]["max_tokens"] is None
    assert engine.preview_calls[0]["inference_overrides"]["reserve_output_tokens"] is None


def test_chat_context_preview_receives_persisted_session_policy(tmp_path) -> None:
    sessions_dir = tmp_path / "sessions"
    config = MochiConfig.model_validate(
        {
            "model": "gpt-test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(sessions_dir),
            "security": {"autonomy_mode": "strict"},
        }
    )
    app = _create_test_app(config=config, session_store=SessionStore(sessions_dir))
    engine = _ContextPreviewEngine()
    app.state.engine = engine

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/sessions",
            json={
                "session_id": "session-context-policy",
                "security_override": {"autonomy_mode": "auto_review"},
            },
        )
        assert create_response.status_code == 200
        response = client.post(
            "/v1/chat/context",
            json={
                "message": "preview with the persisted policy",
                "session_id": "session-context-policy",
            },
        )

    assert response.status_code == 200
    policy = engine.preview_calls[-1]["permission_policy"]
    assert policy["autonomy_mode"] == "auto_review"
    assert policy["require_approval_for_file_write"] is False
    assert policy["require_approval_for_exec"] is False
    assert policy["source_chain"] == ["security_config", "session_override"]
    assert str(policy["policy_snapshot_id"]).startswith("policy-")
    workflow = response.json()["tool_workflow"]
    assert workflow["effective_policy"]["autonomy_mode"] == "auto_review"
    assert workflow["effective_policy"]["source_chain"] == [
        "security_config",
        "session_override",
    ]
    assert workflow["effective_policy"]["expectation_status"] == "not_provided"


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
