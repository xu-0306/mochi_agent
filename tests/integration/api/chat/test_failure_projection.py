"""FailureEnvelope v1 chat SSE and persisted-session projection coverage."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from mochi.agents.events import ErrorEvent
from mochi.agents.failures import FailureEnvelope
from mochi.config.schema import MochiConfig
from mochi.sessions.store import SessionStore

from ._support import _build_app, _FakeEngine


def test_canonical_failure_survives_chat_sse_and_session_replay(tmp_path: Path) -> None:
    """The same v1 envelope must reach the client and durable turn payload."""

    envelope = FailureEnvelope(
        schema_version="1.0",
        kind="backend_error",
        origin="backend",
        recoverability="manual_retry",
        retry_policy="manual",
        terminal=True,
        inject_into_model_context=False,
        telemetry_key="failure.backend_error",
        ui_hint="retry",
        diagnostics_ref="diag-opaque-42",
    ).to_dict()

    class _FailureEngine(_FakeEngine):
        async def chat(
            self,
            message: str,
            session_id: str | None = None,
            **kwargs: Any,
        ) -> AsyncIterator[object]:
            del kwargs
            self.chat_calls.append((message, session_id))
            yield ErrorEvent(
                message="The configured model request failed.",
                code="MODEL_REQUEST_FAILED",
                metadata={"backend": {"name": "openai_compat"}},
                failure=envelope,
            )

    sessions_dir = tmp_path / "sessions"
    store = SessionStore(sessions_dir)
    app, engine = _build_app(engine=_FailureEngine())
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"model": "ollama:configured", "sessions_dir": str(sessions_dir)}
    )
    app.state.session_store = store

    with TestClient(app) as client, client.stream(
        "POST",
        "/v1/chat/stream",
        json={"message": "trigger failure", "session_id": "failure-projection"},
    ) as response:
        frames = [
            line.removeprefix("data: ")
            for line in response.iter_lines()
            if line.startswith("data: ")
        ]

    assert response.status_code == 200
    assert engine.chat_calls == [("trigger failure", "failure-projection")]
    assert len(frames) == 1

    import json

    streamed_event = json.loads(frames[0])
    assert streamed_event["type"] == "error"
    assert streamed_event["failure"] == envelope

    persisted_events = asyncio.run(store.load_session("failure-projection"))
    assert len(persisted_events) == 1
    assert persisted_events[0]["type"] == "turn_event"
    assert persisted_events[0]["payload"]["failure"] == envelope
