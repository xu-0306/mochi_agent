from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

from fastapi.testclient import TestClient

from mochi.agents.events import (
    FinalAnswerEvent,
    StatusEvent,
    ToolCallRequestEvent,
    ToolCallResultEvent,
)
from mochi.api.server import create_app
from mochi.backends.types import AttachmentRef
from mochi.config.schema import MochiConfig
from mochi.sessions.store import SessionStore


class _AttachmentCaptureEngine:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def chat(
        self,
        message: str,
        session_id: str | None = None,
        inference_overrides: dict[str, Any] | None = None,
        project_id: str | None = None,
        workspace_dir: str | None = None,
        selected_skill_ids: list[str] | None = None,
        attachments: list[AttachmentRef] | None = None,
    ) -> AsyncIterator[object]:
        self.calls.append(
            {
                "message": message,
                "session_id": session_id,
                "attachments": attachments,
                "project_id": project_id,
                "workspace_dir": workspace_dir,
                "selected_skill_ids": selected_skill_ids,
                "inference_overrides": inference_overrides,
            }
        )
        yield FinalAnswerEvent(content="ok")


class _DiagnosticsCaptureEngine:
    async def chat(
        self,
        message: str,
        session_id: str | None = None,
        inference_overrides: dict[str, Any] | None = None,
        project_id: str | None = None,
        workspace_dir: str | None = None,
        selected_skill_ids: list[str] | None = None,
        attachments: list[AttachmentRef] | None = None,
    ) -> AsyncIterator[object]:
        del message, session_id, inference_overrides, project_id, workspace_dir, selected_skill_ids, attachments
        yield StatusEvent(
            content="Prepared workspace tool exposure.",
            metadata={
                "tool_exposure": {
                    "exposed_tools": ["file_read", "grep_search"],
                    "workspace_bound": True,
                    "attachment_count": 2,
                }
            },
        )
        yield ToolCallRequestEvent(
            call_id="call-diag-1",
            tool_name="file_read",
            arguments={"path": "notes.txt"},
        )
        yield ToolCallResultEvent(
            call_id="call-diag-1",
            tool_name="file_read",
            result={"preview": "done"},
            metadata={
                "transport": {
                    "summary_applied": True,
                    "overflow_persisted": True,
                    "reference_id": "file_read-abc123",
                    "artifact_path": "H:/tmp/tool-results/file_read-abc123.txt",
                    "source_path": "H:/workspace/notes.txt",
                }
            },
        )
        yield FinalAnswerEvent(content="ok")


def test_chat_route_forwards_structured_attachments(tmp_path) -> None:
    app = create_app()
    engine = _AttachmentCaptureEngine()
    app.state.engine = engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
        }
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat",
            json={
                "message": "Summarize this file",
                "session_id": "session-attachments",
                "attachments": [
                    {
                        "name": "brief.docx",
                        "path": str(tmp_path / "brief.docx"),
                        "size": 2048,
                        "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert engine.calls == [
        {
            "message": "Summarize this file",
            "session_id": "session-attachments",
            "attachments": [
                AttachmentRef(
                    name="brief.docx",
                    path=str(tmp_path / "brief.docx"),
                    size=2048,
                    content_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            ],
            "project_id": None,
            "workspace_dir": str(tmp_path),
            "selected_skill_ids": None,
            "inference_overrides": {},
        }
    ]


def test_chat_route_accepts_richer_attachment_metadata_and_camel_case_fields(tmp_path) -> None:
    app = create_app()
    engine = _AttachmentCaptureEngine()
    app.state.engine = engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
        }
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat",
            json={
                "message": "Review this selection",
                "session_id": "session-selection",
                "attachments": [
                    {
                        "name": "server.py",
                        "path": str(tmp_path / "server.py"),
                        "source": "workspace_selection",
                        "lineStart": 12,
                        "lineEnd": 18,
                        "quote": "def handle_request(...):",
                        "note": "Focus on the validation branch.",
                        "contentType": "text/x-python",
                    }
                ],
            },
        )

    assert response.status_code == 200
    assert engine.calls[0]["attachments"] == [
        AttachmentRef(
            name="server.py",
            path=str(tmp_path / "server.py"),
            size=None,
            content_type="text/x-python",
            source="workspace_selection",
            line_start=12,
            line_end=18,
            quote="def handle_request(...):",
            note="Focus on the validation branch.",
        )
    ]


def test_chat_route_serializes_workspace_tool_exposure_and_transport_diagnostics(tmp_path) -> None:
    app = create_app()
    app.state.engine = _DiagnosticsCaptureEngine()
    sessions_dir = tmp_path / "sessions"
    app.state.session_store = SessionStore(sessions_dir)
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(sessions_dir),
        }
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat",
            json={
                "message": "Inspect the workspace tool diagnostics",
                "session_id": "session-diagnostics",
            },
        )
        session_response = client.get("/v1/sessions/session-diagnostics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["events"][0] == {
        "type": "status",
        "content": "Prepared workspace tool exposure.",
        "metadata": {
            "tool_exposure": {
                "exposed_tools": ["file_read", "grep_search"],
                "workspace_bound": True,
                "attachment_count": 2,
            }
        },
    }
    assert payload["events"][2] == {
        "type": "tool_call_result",
        "call_id": "call-diag-1",
        "tool_name": "file_read",
        "result": {"preview": "done"},
        "error": None,
        "metadata": {
            "transport": {
                "summary_applied": True,
                "overflow_persisted": True,
                "reference_id": "file_read-abc123",
                "artifact_path": "H:/tmp/tool-results/file_read-abc123.txt",
                "source_path": "H:/workspace/notes.txt",
            }
        },
    }

    assert session_response.status_code == 200
    session_events = session_response.json()["events"]
    assert [event["phase"] for event in session_events] == [
        "status",
        "tool_call_request",
        "tool_call_result",
        "final_answer",
    ]
    assert session_events[0]["payload"] == payload["events"][0]
    assert session_events[2]["payload"] == payload["events"][2]
