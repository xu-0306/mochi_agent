"""Request-scoped client timezone propagation tests."""

from __future__ import annotations

from types import MethodType
from typing import Any

import pytest

from mochi.agents.engine import AgentEngine
from mochi.agents.invocation import AgentInvocationRequest
from mochi.config.schema import MochiConfig


@pytest.mark.asyncio
async def test_engine_chat_carries_client_timezone_into_invocation_request() -> None:
    engine = AgentEngine.__new__(AgentEngine)
    captured: list[AgentInvocationRequest] = []

    async def fake_run_chat(
        self: AgentEngine,
        request: AgentInvocationRequest,
    ) -> Any:
        del self
        captured.append(request)
        if False:
            yield None

    engine._run_chat = MethodType(fake_run_chat, engine)  # type: ignore[method-assign]

    events = [
        event
        async for event in engine.chat(
            "what time is it?",
            session_id="timezone-request",
            client_timezone="Africa/Nairobi",
        )
    ]

    assert events == []
    assert captured[0].client_timezone == "Africa/Nairobi"


def test_engine_tool_context_keeps_client_timezones_request_scoped(tmp_path: Any) -> None:
    engine = AgentEngine.__new__(AgentEngine)
    engine._config = MochiConfig.model_validate(  # type: ignore[attr-defined]
        {"workspace_dir": str(tmp_path)}
    )
    engine._tool_execution_contexts = {}  # type: ignore[attr-defined]

    taipei = engine._get_tool_execution_context(
        session_id="shared-session",
        workspace_dir=str(tmp_path),
        client_timezone="Asia/Taipei",
    )
    nairobi = engine._get_tool_execution_context(
        session_id="shared-session",
        workspace_dir=str(tmp_path),
        client_timezone="Africa/Nairobi",
    )
    cached_base = engine._get_tool_execution_context(
        session_id="shared-session",
        workspace_dir=str(tmp_path),
    )

    assert taipei is not nairobi
    assert taipei.client_timezone == "Asia/Taipei"
    assert nairobi.client_timezone == "Africa/Nairobi"
    assert cached_base.client_timezone is None
