"""Phase 6A chat/models API routes tests."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from mochi.agents.events import (
    FinalAnswerEvent,
    ThinkingEvent,
    ToolCallRequestEvent,
    ToolCallResultEvent,
)
from mochi.agents.engine import (
    AgentEngine,
    _build_response_language_prompt_addendum,
    _merge_prompt_addenda,
)
from mochi.agents.invocation import AgentInvocationDiagnostics, AgentInvocationRequest, AgentInvocationResult
from mochi.api.server import create_app
from mochi.auth.models import OpenAICodexAuthProfile
from mochi.auth.openai_codex import (
    OPENAI_CODEX_REFRESH_LOCK_STALE_SECONDS,
    OpenAICodexAuthService,
    _profile_refresh_lock_path,
)
from mochi.backends.router import BackendRouter
from mochi.backends.local_models import LocalModelConvertExecutionResult
from mochi.backends.types import AttachmentRef, ModelInfo
from mochi.config.manager import load_config
from mochi.config.schema import MochiConfig
from mochi.sessions.store import SessionStore


class _FakeEngine:
    def __init__(self) -> None:
        self.chat_calls: list[tuple[str, str | None]] = []
        self.chat_attachment_calls: list[list[AttachmentRef] | None] = []
        self.switch_calls: list[str] = []
        self.ollama_switch_calls: list[tuple[str, str | None]] = []
        self.openai_switch_calls: list[tuple[str, str, str, str]] = []
        self.openai_codex_switch_calls: list[tuple[str, str, str | None]] = []
        self.test_connection_calls: list[dict[str, Any]] = []
        self.model_info = ModelInfo(
            name="ollama:test",
            backend_type="ollama",
            context_length=8192,
            supports_tool_calling=True,
            metadata={"provider": "fake"},
        )
        self.tool_probe_result: dict[str, Any] | None = None
        self.unload_active_local_model_calls = 0
        self.apply_config_calls: list[tuple[str, bool]] = []

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
        _ = (inference_overrides, project_id, workspace_dir, selected_skill_ids)
        self.chat_calls.append((message, session_id))
        self.chat_attachment_calls.append(attachments)
        yield ThinkingEvent(content="分析中")
        yield ToolCallRequestEvent(
            call_id="call-1",
            tool_name="clock",
            arguments={"timezone": "Asia/Taipei"},
        )
        yield ToolCallResultEvent(
            call_id="call-1",
            tool_name="clock",
            result={"now": datetime(2026, 4, 27, 9, 30, tzinfo=UTC)},
        )
        yield FinalAnswerEvent(
            content=f"已收到：{message}",
            trajectory_id="traj-123",
            input_tokens=128,
            output_tokens=32,
            generation_time_ms=250.0,
            finish_reason="stop",
        )

    def get_model_info(self) -> ModelInfo:
        return self.model_info

    async def probe_active_tool_calling(self) -> dict[str, Any] | None:
        return self.tool_probe_result

    async def test_model_connection(
        self,
        *,
        provider: str,
        model: str,
        base_url: str | None = None,
        api_key: str = "",
        auth_profile_id: str | None = None,
    ) -> ModelInfo:
        self.test_connection_calls.append(
            {
                "provider": provider,
                "model": model,
                "base_url": base_url,
                "api_key": api_key,
                "auth_profile_id": auth_profile_id,
            }
        )
        backend_type = "gguf" if provider == "local" and model.lower().endswith(".gguf") else (
            "safetensors" if provider == "local" else (
                "ollama" if provider == "ollama" else (
                    "openai_codex" if provider == "openai_codex" else "openai_compat"
                )
            )
        )
        return ModelInfo(
            name=model,
            provider=None if provider == "local" else provider,
            backend_type=backend_type,
            context_length=4096 if provider == "local" else None,
            supports_tool_calling=provider != "local",
            metadata={
                "base_url": base_url,
                "api_key_configured": bool(api_key),
                "auth_profile_id": auth_profile_id,
                "tested": True,
            },
        )

    async def switch_model(self, model: str) -> ModelInfo:
        self.switch_calls.append(model)
        self.model_info = ModelInfo(
            name=model,
            backend_type="gguf",
            context_length=4096,
            supports_tool_calling=False,
            metadata={"switched": True},
        )
        return self.model_info

    async def switch_ollama_backend(
        self,
        *,
        model: str,
        base_url: str | None = None,
    ) -> ModelInfo:
        self.ollama_switch_calls.append((model, base_url))
        self.model_info = ModelInfo(
            name=model,
            backend_type="ollama",
            context_length=8192,
            supports_tool_calling=True,
            metadata={"base_url": base_url or "http://localhost:11434"},
        )
        return self.model_info

    async def switch_openai_compat_backend(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        provider: str = "openai_compat",
    ) -> ModelInfo:
        self.openai_switch_calls.append((base_url, model, api_key, provider))
        self.model_info = ModelInfo(
            name=model,
            backend_type="openai_compat",
            context_length=None,
            supports_tool_calling=True,
            metadata={"base_url": base_url, "api_key_configured": bool(api_key)},
        )
        return self.model_info

    async def switch_openai_codex_backend(
        self,
        *,
        base_url: str,
        model: str,
        auth_profile_id: str | None = None,
    ) -> ModelInfo:
        self.openai_codex_switch_calls.append((base_url, model, auth_profile_id))
        self.model_info = ModelInfo(
            name=model,
            provider="openai_codex",
            backend_type="openai_codex",
            context_length=None,
            supports_tool_calling=True,
            metadata={"base_url": base_url, "auth_profile_id": auth_profile_id},
        )
        return self.model_info

    async def unload_active_local_model(self) -> ModelInfo | None:
        self.unload_active_local_model_calls += 1
        if self.model_info.backend_type not in {"gguf", "safetensors"}:
            return None
        self.model_info = ModelInfo(
            name=self.model_info.name,
            backend_type=self.model_info.backend_type,
            context_length=self.model_info.context_length,
            supports_tool_calling=self.model_info.supports_tool_calling,
            metadata={
                **self.model_info.metadata,
                "loaded": False,
                "idle_unloaded": False,
            },
        )
        return self.model_info

    async def apply_config(self, config: MochiConfig, *, reload_voice: bool = False) -> None:
        self.apply_config_calls.append((config.model, reload_voice))


class _FakeManagedVLLMRuntimeManager:
    def __init__(self, *, base_url: str = "http://localhost:8000/v1") -> None:
        self.base_url = base_url
        self.running = False
        self.active_model_id: str | None = None
        self.active_model_spec: str | None = None
        self.start_calls: list[tuple[str | None, str, str, str]] = []
        self.stop_calls = 0

    async def status(self, **_: Any) -> dict[str, Any]:
        return {
            "state": "running" if self.running else "stopped",
            "running": self.running,
            "launch_mode": "managed",
            "active_model_id": self.active_model_id,
            "active_model_spec": self.active_model_spec,
            "base_url": self.base_url,
        }

    async def start(
        self,
        *,
        model_id: str | None = None,
        model_spec: str,
        base_url: str | None = None,
        launch_mode: str = "managed",
        **_: Any,
    ) -> dict[str, Any]:
        self.running = True
        self.active_model_id = model_id
        self.active_model_spec = model_spec
        self.base_url = base_url or self.base_url
        self.start_calls.append((model_id, model_spec, self.base_url, launch_mode))
        return await self.status()

    async def stop(self, **_: Any) -> dict[str, Any]:
        self.stop_calls += 1
        self.running = False
        self.active_model_id = None
        self.active_model_spec = None
        return await self.status()


def _build_app(
    *,
    engine: _FakeEngine | None = None,
    config_path: Path | None = None,
    vllm_runtime_manager: _FakeManagedVLLMRuntimeManager | None = None,
    workspace_dir: Path | None = None,
) -> tuple[object, _FakeEngine]:
    app = create_app()
    fake_engine = engine or _FakeEngine()
    app.state.engine_factory = lambda: fake_engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "workspace_dir": str(workspace_dir) if workspace_dir is not None else ".mochi",
            "local_models": {
                "roots": [],
                "scan_max_depth": 3,
                "scan_max_entries": 500,
                "runtime": "inprocess",
            },
            "channels": {
                "discord": {"bot_token": SecretStr("discord-secret-token")},
                "telegram": {"bot_token": SecretStr("telegram-secret-token")},
            },
            "voice": {
                "stt_openai_api_key": SecretStr("stt-secret-token"),
                "tts_openai_api_key": SecretStr("tts-secret-token"),
            },
        }
    )
    app.state.config_path = config_path
    if vllm_runtime_manager is not None:
        app.state.vllm_runtime_manager = vllm_runtime_manager
    return app, fake_engine


def test_chat_route_returns_bounded_response_with_serialized_events() -> None:
    """`POST /v1/chat` 應收斂事件流並回傳 final answer/trajectory。"""
    app, engine = _build_app()

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat",
            json={"message": "現在幾點？", "session_id": "session-42"},
        )

    assert response.status_code == 200
    assert engine.chat_calls == [("現在幾點？", "session-42")]
    payload = response.json()
    assert payload["turn_id"]
    payload["turn_id"] = "turn-id"
    assert payload == {
        "type": "chat_response",
        "session_id": "session-42",
        "turn_id": "turn-id",
        "final_answer": "已收到：現在幾點？",
        "trajectory_id": "traj-123",
        "events": [
            {"type": "thinking", "content": "分析中", "metadata": {}},
            {
                "type": "tool_call_request",
                "call_id": "call-1",
                "tool_name": "clock",
                "arguments": {"timezone": "Asia/Taipei"},
            },
            {
                "type": "tool_call_result",
                "call_id": "call-1",
                "tool_name": "clock",
                "result": {"now": "2026-04-27T09:30:00+00:00"},
                "error": None,
                "metadata": {},
            },
            {
                "type": "final_answer",
                "content": "已收到：現在幾點？",
                "trajectory_id": "traj-123",
                "input_tokens": 128,
                "output_tokens": 32,
                "generation_time_ms": 250.0,
                "finish_reason": "stop",
            },
        ],
    }


def test_chat_route_applies_selected_available_model_before_chat() -> None:
    """`POST /v1/chat` 帶模型 id 時應先切換到該模型再執行對話。"""
    config = MochiConfig.model_validate(
        {
            "model": "ollama:qwen2.5",
            "model_setup": {
                "configured_models": [
                    {
                        "id": "openai_compat:https://api.example.com/v1:gpt-test",
                        "provider": "openai_compat",
                        "model": "gpt-test",
                        "model_spec": "https://api.example.com/v1",
                        "base_url": "https://api.example.com/v1",
                    }
                ]
            },
            "openai_compat": {
                "provider": "openai_compat",
                "base_url": "https://api.example.com/v1",
                "model": "gpt-test",
                "api_key": "sk-secret-value",
            },
        }
    )
    app, engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat",
            json={
                "message": "hello",
                "session_id": "session-model",
                "model": "openai_compat:https://api.example.com/v1:gpt-test",
            },
        )

    assert response.status_code == 200
    assert engine.openai_switch_calls == [
        ("https://api.example.com/v1", "gpt-test", "sk-secret-value", "openai_compat")
    ]
    assert engine.chat_calls == [("hello", "session-model")]
    assert "sk-secret-value" not in response.text


def test_chat_route_does_not_reswitch_current_model() -> None:
    """聊天頁每輪都帶目前模型 id 時，後端不應重複切換已 active 的模型。"""
    config = MochiConfig.model_validate(
        {
            "model": "ollama:qwen2.5",
            "ollama": {"base_url": "http://localhost:11434"},
            "model_setup": {
                "configured_models": [
                    {
                        "id": "ollama:qwen2.5",
                        "provider": "ollama",
                        "model": "qwen2.5",
                        "model_spec": "ollama:qwen2.5",
                        "base_url": "http://localhost:11434",
                    }
                ]
            },
        }
    )
    app, engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.post(
            "/v1/chat",
            json={
                "message": "hello",
                "session_id": "session-model",
                "model": "ollama:qwen2.5",
            },
        )

    assert response.status_code == 200
    assert engine.ollama_switch_calls == []
    assert engine.chat_calls == [("hello", "session-model")]


def test_chat_route_persists_turn_events_and_sessions_route_returns_them(tmp_path: Path) -> None:
    """`POST /v1/chat` 後應將 replay `turn_event` 寫入 session JSONL。"""
    sessions_dir = tmp_path / "sessions"
    app, engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "sessions_dir": str(sessions_dir),
        }
    )
    app.state.session_store = SessionStore(sessions_dir)

    with TestClient(app) as client:
        post_response = client.post(
            "/v1/chat",
            json={"message": "現在幾點？", "session_id": "session-42"},
        )
        get_response = client.get("/v1/sessions/session-42")

    assert post_response.status_code == 200
    assert engine.chat_calls == [("現在幾點？", "session-42")]

    store_events = asyncio.run(SessionStore(sessions_dir).load_session("session-42"))
    assert [event["type"] for event in store_events] == ["turn_event"] * 4
    assert [event["phase"] for event in store_events] == [
        "thinking",
        "tool_call_request",
        "tool_call_result",
        "final_answer",
    ]
    assert [event["seq"] for event in store_events] == [1, 2, 3, 4]
    assert all(event["schema_version"] == 1 for event in store_events)
    assert len({event["event_id"] for event in store_events}) == 4
    assert len({event["turn_id"] for event in store_events}) == 1
    assert store_events[0]["payload"] == {"type": "thinking", "content": "分析中", "metadata": {}}
    assert store_events[-1]["payload"] == {
        "type": "final_answer",
        "content": "已收到：現在幾點？",
        "trajectory_id": "traj-123",
        "input_tokens": 128,
        "output_tokens": 32,
        "generation_time_ms": 250.0,
        "finish_reason": "stop",
    }

    assert get_response.status_code == 200
    assert get_response.json()["events"] == store_events


def test_chat_stream_route_returns_sse_events_incrementally() -> None:
    """`POST /v1/chat/stream` 應以 SSE 逐筆送出 serialized chat events。"""
    app, engine = _build_app()

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/stream",
            json={"message": "現在幾點？", "session_id": "session-42"},
        ) as response:
            chunks = [
                line.removeprefix("data: ")
                for line in response.iter_lines()
                if line.startswith("data: ")
            ]
            session_id = response.headers["x-session-id"]
            cache_control = response.headers["cache-control"]
            content_type = response.headers["content-type"]

    assert response.status_code == 200
    assert engine.chat_calls == [("現在幾點？", "session-42")]
    assert session_id == "session-42"
    assert cache_control == "no-cache"
    assert content_type.startswith("text/event-stream")

    events = [json.loads(chunk) for chunk in chunks]
    turn_ids = {event["turn_id"] for event in events}
    assert len(turn_ids) == 1
    turn_id = next(iter(turn_ids))
    assert turn_id
    assert events == [
        {"type": "thinking", "content": "分析中", "metadata": {}, "turn_id": turn_id},
        {
            "type": "tool_call_request",
            "call_id": "call-1",
            "tool_name": "clock",
            "arguments": {"timezone": "Asia/Taipei"},
            "turn_id": turn_id,
        },
            {
                "type": "tool_call_result",
                "call_id": "call-1",
                "tool_name": "clock",
                "result": {"now": "2026-04-27T09:30:00+00:00"},
                "error": None,
                "metadata": {},
                "turn_id": turn_id,
            },
        {
            "type": "final_answer",
            "content": "已收到：現在幾點？",
            "trajectory_id": "traj-123",
            "input_tokens": 128,
            "output_tokens": 32,
            "generation_time_ms": 250.0,
            "finish_reason": "stop",
            "turn_id": turn_id,
        },
    ]


def test_agent_engine_chat_yields_events_before_invocation_finishes() -> None:
    """`AgentEngine.chat()` should surface intermediate events before the invocation fully completes."""
    engine = AgentEngine.__new__(AgentEngine)

    async def _fake_invoke(
        self: AgentEngine,
        request: AgentInvocationRequest,
        *,
        event_callback=None,
    ) -> AgentInvocationResult:
        del self, request
        thinking = ThinkingEvent(content="streaming-thought")
        thinking.turn_id = "turn-stream"  # type: ignore[attr-defined]
        final = FinalAnswerEvent(content="streaming-answer")
        final.turn_id = "turn-stream"  # type: ignore[attr-defined]

        if event_callback is not None:
            await event_callback(thinking)
        await asyncio.sleep(0.05)
        if event_callback is not None:
            await event_callback(final)
        await asyncio.sleep(0.2)

        return AgentInvocationResult(
            content="streaming-answer",
            events=[thinking, final],
            diagnostics=AgentInvocationDiagnostics(
                execution_profile="chat",
                tool_mode="auto",
            ),
        )

    engine._invoke_shared_runtime = _fake_invoke.__get__(engine, AgentEngine)  # type: ignore[attr-defined]

    async def _collect() -> tuple[list[object], float, float]:
        start = time.perf_counter()
        first_event_elapsed: float | None = None
        events: list[object] = []

        async for event in engine.chat("hello", session_id="session-stream"):
            events.append(event)
            if first_event_elapsed is None:
                first_event_elapsed = time.perf_counter() - start

        total_elapsed = time.perf_counter() - start
        assert first_event_elapsed is not None
        return events, first_event_elapsed, total_elapsed

    events, first_event_elapsed, total_elapsed = asyncio.run(_collect())

    assert [event.type for event in events] == ["thinking", "final_answer"]
    assert first_event_elapsed < 0.15
    assert total_elapsed >= 0.25


def test_response_language_addendum_tracks_traditional_chinese_messages() -> None:
    addendum = _build_response_language_prompt_addendum(
        "same_as_user",
        "幫我查詢 ESG 相關 LLM 微調資訊，方法等",
    )

    assert addendum is not None
    assert "Reply in the same language as the user's latest message" in addendum
    assert "Traditional Chinese" in addendum
    assert "current user message is in Traditional Chinese" in addendum


def test_response_language_addendum_tracks_japanese_messages_without_chinese_bias() -> None:
    addendum = _build_response_language_prompt_addendum(
        "same_as_user",
        "ハイ",
    )

    assert addendum is not None
    assert "Reply in the same language as the user's latest message" in addendum
    assert "The current user message is in Japanese. Reply in Japanese." in addendum
    assert "Traditional Chinese" not in addendum


def test_response_language_addendum_tracks_latin_script_messages_without_language_switch() -> None:
    addendum = _build_response_language_prompt_addendum(
        "same_as_user",
        "hi there",
    )

    assert addendum is not None
    assert "Reply in the same language as the user's latest message" in addendum
    assert "The current user message is written in a Latin-script language." in addendum
    assert "Traditional Chinese" not in addendum


def test_response_language_addendum_respects_explicit_language_preference() -> None:
    addendum = _build_response_language_prompt_addendum(
        "en-US",
        "請用中文回覆這段測試",
    )

    assert addendum is not None
    assert "Default response language: en-US." in addendum
    assert "Keep using that language unless the user explicitly requests another language." in addendum


def test_merge_prompt_addenda_preserves_existing_invocation_context() -> None:
    merged = _merge_prompt_addenda(
        "Language Policy:\n- Reply in Traditional Chinese.",
        "Goal context:\n- Active goal is blocked.",
    )

    assert merged == (
        "Language Policy:\n- Reply in Traditional Chinese.\n\n"
        "Goal context:\n- Active goal is blocked."
    )


def test_chat_stream_route_persists_fallback_turn_events(tmp_path: Path) -> None:
    """stream route 對未自行持久化 turn replay 的 engine 應補寫 session turn_event。"""
    sessions_dir = tmp_path / "sessions"
    app, engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "sessions_dir": str(sessions_dir),
        }
    )
    app.state.session_store = SessionStore(sessions_dir)

    with TestClient(app) as client:
        with client.stream(
            "POST",
            "/v1/chat/stream",
            json={"message": "現在幾點？", "session_id": "session-stream"},
        ) as response:
            _ = list(response.iter_lines())
        get_response = client.get("/v1/sessions/session-stream")

    assert response.status_code == 200
    assert engine.chat_calls == [("現在幾點？", "session-stream")]

    store_events = asyncio.run(SessionStore(sessions_dir).load_session("session-stream"))
    assert [event["type"] for event in store_events] == ["turn_event"] * 4
    assert [event["phase"] for event in store_events] == [
        "thinking",
        "tool_call_request",
        "tool_call_result",
        "final_answer",
    ]
    assert [event["seq"] for event in store_events] == [1, 2, 3, 4]
    assert len({event["turn_id"] for event in store_events}) == 1
    assert store_events[0]["payload"]["turn_id"] == store_events[0]["turn_id"]
    assert store_events[-1]["payload"] == {
        "type": "final_answer",
        "content": "已收到：現在幾點？",
        "trajectory_id": "traj-123",
        "input_tokens": 128,
        "output_tokens": 32,
        "generation_time_ms": 250.0,
        "finish_reason": "stop",
        "turn_id": store_events[-1]["turn_id"],
    }

    assert get_response.status_code == 200
    assert get_response.json()["events"] == store_events


def test_models_route_returns_active_model_without_leaking_secrets() -> None:
    """`GET /v1/models` 應只回傳非敏感模型資訊。"""
    app, _engine = _build_app()

    with TestClient(app) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "models_status"
    assert payload["configured_model"] == "ollama:configured"
    assert payload["active_model"] == {
        "name": "ollama:test",
        "backend_type": "ollama",
        "context_length": 8192,
        "supports_tool_calling": True,
        "metadata": {"provider": "fake"},
    }
    assert payload["available_models"] == [
        {
            "id": "ollama:configured",
            "provider": "ollama",
            "model": "configured",
            "model_spec": "ollama:configured",
            "base_url": "http://localhost:11434",
            "label": "configured",
            "backend_type": "ollama",
            "api_key_configured": False,
        }
    ]
    assert [item["pattern"] for item in payload["supported_model_spec_formats"]] == [
        "ollama:<model>",
        "/path/to/model.gguf",
        "/path/to/model_dir/",
        "https://host/v1",
    ]
    assert "secret-token" not in response.text


def test_models_switch_route_calls_engine_switch_model() -> None:
    """`POST /v1/models/switch` 應委派給 engine.switch_model。"""
    app, engine = _build_app()

    with TestClient(app) as client:
        response = client.post("/v1/models/switch", json={"model": "/models/demo.gguf"})

    assert response.status_code == 200
    assert engine.switch_calls == ["/models/demo.gguf"]
    assert response.json() == {
        "type": "model_switch",
        "active_model": {
            "name": "/models/demo.gguf",
            "backend_type": "gguf",
            "context_length": 4096,
            "supports_tool_calling": False,
            "metadata": {"switched": True},
        },
    }


def test_models_configure_route_supports_ollama_without_leaking_key() -> None:
    """`POST /v1/models/configure` 應支援 Ollama base_url/model 設定。"""
    app, engine = _build_app()

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/configure",
            json={
                "provider": "ollama",
                "base_url": "http://localhost:11434",
                "model": "qwen2.5",
            },
        )

    assert response.status_code == 200
    assert engine.ollama_switch_calls == [("qwen2.5", "http://localhost:11434")]
    assert response.json()["provider"] == "ollama"
    assert response.json()["api_key_configured"] is False
    assert response.json()["active_model"]["name"] == "qwen2.5"
    assert response.json()["available_models"][0]["id"] == "ollama:qwen2.5"
    assert response.json()["available_models"][0]["model"] == "qwen2.5"


def test_models_configure_route_supports_openai_compat_without_returning_api_key() -> None:
    """`POST /v1/models/configure` 應接收 API key 但不得回傳原文。"""
    app, engine = _build_app()

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/configure",
            json={
                "provider": "openai_compat",
                "base_url": "https://api.example.com/v1",
                "model": "gpt-test",
                "api_key": "sk-secret-value",
            },
        )

    assert response.status_code == 200
    assert engine.openai_switch_calls == [
        ("https://api.example.com/v1", "gpt-test", "sk-secret-value", "openai_compat")
    ]
    assert response.json()["provider"] == "openai_compat"
    assert response.json()["api_key_configured"] is True
    assert response.json()["active_model"]["name"] == "gpt-test"
    assert response.json()["available_models"][0]["id"] == (
        "openai_compat:https://api.example.com/v1:gpt-test"
    )
    assert response.json()["available_models"][0]["model_spec"] == "https://api.example.com/v1"
    assert "sk-secret-value" not in response.text


@pytest.mark.parametrize(
    ("provider", "default_base_url"),
    [
        ("sglang", "http://localhost:30000/v1"),
        ("tensorrt_llm", "http://localhost:8000/v1"),
    ],
)
def test_models_configure_route_supports_external_openai_compat_presets_without_managed_vllm_path(
    provider: str,
    default_base_url: str,
) -> None:
    manager = _FakeManagedVLLMRuntimeManager()
    app, engine = _build_app(vllm_runtime_manager=manager)
    requested_model = "Qwen/Qwen2.5-7B-Instruct"

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/configure",
            json={
                "provider": provider,
                "model": requested_model,
                "api_key": "sk-provider-secret",
            },
        )

    assert response.status_code == 200
    assert manager.start_calls == []
    assert engine.openai_switch_calls == [
        (default_base_url, requested_model, "sk-provider-secret", provider)
    ]
    payload = response.json()
    assert payload["provider"] == provider
    assert payload["api_key_configured"] is True
    assert payload["active_model"]["name"] == requested_model
    assert payload["active_model"]["backend_type"] == "openai_compat"
    assert payload["available_models"][0]["provider"] == provider
    assert payload["available_models"][0]["base_url"] == default_base_url
    assert payload["available_models"][0]["model_spec"] == default_base_url
    assert payload["available_models"][0]["launch_mode"] == "external"
    assert payload["available_models"][0]["backend_type"] == "openai_compat"
    assert "sk-provider-secret" not in response.text


def test_openai_codex_import_route_stores_cli_login_under_mochi_state_root(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """OpenAI Codex CLI import should populate the separate auth store."""
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": "header.eyJleHAiOjE5MDAwMDAwMDAsImVtYWlsIjoiY29kZXhAZXhhbXBsZS5jb20ifQ.sig",
                    "refresh_token": "refresh-token",
                    "account_id": "acct_123",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    workspace_dir = tmp_path / "workspace"
    app, _engine = _build_app(workspace_dir=workspace_dir)

    with TestClient(app) as client:
        response = client.post("/v1/model-auth/openai-codex/import-codex-cli")

    assert response.status_code == 200
    payload = response.json()
    assert payload["profile"]["profile_id"] == "openai_codex:default"
    assert "store_path" not in payload
    assert "source_path" not in payload["profile"]
    store_path = workspace_dir / ".mochi" / "auth.json"
    assert store_path.is_file()
    raw = store_path.read_text(encoding="utf-8")
    assert "refresh-token" in raw
    assert "codex@example.com" in raw


def _fake_jwt(exp: int, *, email: str = "codex@example.com", name: str | None = None) -> str:
    payload: dict[str, Any] = {"exp": exp, "email": email}
    if name is not None:
        payload["name"] = name
    encoded = base64.urlsafe_b64encode(
        json.dumps(payload, separators=(",", ":")).encode("utf-8")
    ).decode("utf-8").rstrip("=")
    return f"header.{encoded}.sig"


def test_openai_codex_status_route_redacts_tokens_and_paths(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Auth status should expose safe profile metadata only."""
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": "header.eyJleHAiOjE5MDAwMDAwMDAsImVtYWlsIjoiY29kZXhAZXhhbXBsZS5jb20ifQ.sig",
                    "refresh_token": "refresh-token",
                    "account_id": "acct_123",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    workspace_dir = tmp_path / "workspace"
    app, _engine = _build_app(workspace_dir=workspace_dir)

    with TestClient(app) as client:
        import_response = client.post("/v1/model-auth/openai-codex/import-codex-cli")
        status_response = client.get("/v1/model-auth/openai-codex/status")

    assert import_response.status_code == 200
    assert status_response.status_code == 200
    payload = status_response.json()
    assert payload["configured"] is True
    assert payload["active_profile_id"] == "openai_codex:default"
    assert payload["profiles"][0]["email"] == "codex@example.com"
    assert "access_token" not in status_response.text
    assert "refresh-token" not in status_response.text
    assert "store_path" not in status_response.text
    assert "source_path" not in status_response.text


def test_openai_codex_service_refreshes_expired_access_token_on_resolve(tmp_path: Path, monkeypatch: Any) -> None:
    """Expired access tokens should refresh automatically on access-token resolution."""
    workspace_dir = tmp_path / "workspace"
    service = OpenAICodexAuthService(str(workspace_dir))
    expired_token = _fake_jwt(1_700_000_000)
    refreshed_token = _fake_jwt(4_100_000_000, name="Codex User")
    service._store.upsert_openai_codex_profile(  # noqa: SLF001
        OpenAICodexAuthProfile(
            profile_id="openai_codex:default",
            access_token=expired_token,
            refresh_token="refresh-token",
            email="codex@example.com",
            display_name="Codex User",
            expires_at=1_700_000_000,
            source_path=None,
        )
    )

    monkeypatch.setattr(
        service,
        "_request_token_refresh",
        lambda refresh_token: {
            "access_token": refreshed_token,
            "refresh_token": f"{refresh_token}-next",
        },
    )

    resolved = service.resolve_access_token("openai_codex:default")
    saved = service.get_profile("openai_codex:default")

    assert resolved == refreshed_token
    assert saved is not None
    assert saved.refresh_token.get_secret_value() == "refresh-token-next"
    assert saved.last_refresh_error is None
    assert service.get_profile_summary("openai_codex:default").status == "ready"  # type: ignore[union-attr]


def test_openai_codex_service_records_refresh_failure_status(tmp_path: Path, monkeypatch: Any) -> None:
    """Refresh failures should be persisted as auth diagnostics instead of surfacing only as backend 401s."""
    workspace_dir = tmp_path / "workspace"
    service = OpenAICodexAuthService(str(workspace_dir))
    expired_token = _fake_jwt(1_700_000_000)
    service._store.upsert_openai_codex_profile(  # noqa: SLF001
        OpenAICodexAuthProfile(
            profile_id="openai_codex:default",
            access_token=expired_token,
            refresh_token="refresh-token",
            email="codex@example.com",
            display_name="Codex User",
            expires_at=1_700_000_000,
            source_path=None,
        )
    )

    def _raise_refresh_error(_refresh_token: str) -> dict[str, Any]:
        raise RuntimeError("invalid_grant")

    monkeypatch.setattr(service, "_request_token_refresh", _raise_refresh_error)

    with pytest.raises(RuntimeError, match="OpenAI Codex auth refresh failed"):
        service.resolve_access_token("openai_codex:default")

    profile = service.get_profile("openai_codex:default")
    summary = service.get_profile_summary("openai_codex:default")
    assert profile is not None
    assert profile.last_refresh_error == "invalid_grant"
    assert summary is not None
    assert summary.status == "refresh_failed"
    assert summary.last_refresh_error == "invalid_grant"


def test_openai_codex_refresh_access_token_recovers_stale_file_lock(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A stale cross-process refresh lock should not block token refresh forever."""
    workspace_dir = tmp_path / "workspace"
    service = OpenAICodexAuthService(str(workspace_dir))
    expired_token = _fake_jwt(1_700_000_000)
    refreshed_token = _fake_jwt(4_100_000_000)
    service._store.upsert_openai_codex_profile(  # noqa: SLF001
        OpenAICodexAuthProfile(
            profile_id="openai_codex:default",
            access_token=expired_token,
            refresh_token="refresh-token",
            email="codex@example.com",
            display_name="Codex User",
            expires_at=1_700_000_000,
            source_path=None,
        )
    )
    lock_path = _profile_refresh_lock_path(service.store_path, "openai_codex:default")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("stale", encoding="utf-8")
    stale_age = time.time() - (OPENAI_CODEX_REFRESH_LOCK_STALE_SECONDS + 5.0)
    os.utime(lock_path, (stale_age, stale_age))

    monkeypatch.setattr(
        service,
        "_request_token_refresh",
        lambda refresh_token: {
            "access_token": refreshed_token,
            "refresh_token": refresh_token,
        },
    )

    resolved = service.resolve_access_token("openai_codex:default")

    assert resolved == refreshed_token
    assert not lock_path.exists()


def test_openai_codex_refresh_access_token_times_out_on_live_file_lock(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """A live cross-process refresh lock should fail fast with a bounded timeout."""
    workspace_dir = tmp_path / "workspace"
    service = OpenAICodexAuthService(str(workspace_dir))
    expired_token = _fake_jwt(1_700_000_000)
    service._store.upsert_openai_codex_profile(  # noqa: SLF001
        OpenAICodexAuthProfile(
            profile_id="openai_codex:default",
            access_token=expired_token,
            refresh_token="refresh-token",
            email="codex@example.com",
            display_name="Codex User",
            expires_at=1_700_000_000,
            source_path=None,
        )
    )
    lock_path = _profile_refresh_lock_path(service.store_path, "openai_codex:default")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_text("live", encoding="utf-8")
    monkeypatch.setattr(
        "mochi.auth.openai_codex.OPENAI_CODEX_REFRESH_LOCK_TIMEOUT_SECONDS",
        0.05,
    )
    monkeypatch.setattr(
        "mochi.auth.openai_codex.OPENAI_CODEX_REFRESH_LOCK_POLL_SECONDS",
        0.01,
    )

    with pytest.raises(RuntimeError, match="Timed out waiting for OpenAI Codex refresh lock"):
        service.refresh_access_token("openai_codex:default", force=True)


def test_openai_codex_import_route_returns_400_when_cli_login_missing(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Import should fail clearly when the local Codex CLI auth file does not exist."""
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    workspace_dir = tmp_path / "workspace"
    app, _engine = _build_app(workspace_dir=workspace_dir)

    with TestClient(app) as client:
        response = client.post("/v1/model-auth/openai-codex/import-codex-cli")

    assert response.status_code == 400
    assert "was not found" in response.json()["detail"]


def test_openai_codex_import_route_rejects_apikey_cli_state_with_actionable_message(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Import should explain that API-key Codex CLI state is not importable as ChatGPT OAuth."""
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "apikey",
                "OPENAI_API_KEY": "sk-test",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    workspace_dir = tmp_path / "workspace"
    app, _engine = _build_app(workspace_dir=workspace_dir)

    with TestClient(app) as client:
        response = client.post("/v1/model-auth/openai-codex/import-codex-cli")

    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "API key mode" in detail
    assert "Connect ChatGPT" in detail


def test_openai_codex_status_route_reports_cli_auth_diagnostics_without_leaking_paths(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Status should expose safe CLI diagnostics even when no Mochi auth profile is saved yet."""
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "apikey",
                "OPENAI_API_KEY": "sk-test",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    workspace_dir = tmp_path / "workspace"
    app, _engine = _build_app(workspace_dir=workspace_dir)

    with TestClient(app) as client:
        response = client.get("/v1/model-auth/openai-codex/status")

    assert response.status_code == 200
    payload = response.json()
    assert payload["configured"] is False
    assert payload["cli_auth_state"] == "apikey"
    assert payload["cli_auth_mode"] == "apikey"
    assert payload["cli_auth_can_import"] is False
    assert "API key mode" in payload["cli_auth_message"]
    assert str(codex_home) not in response.text
    assert "sk-test" not in response.text


def test_openai_codex_refresh_route_updates_status_without_reimport(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Refresh route should renew an expired imported profile instead of requiring a new CLI import."""
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": _fake_jwt(1_700_000_000),
                    "refresh_token": "refresh-token",
                    "account_id": "acct_123",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    monkeypatch.setattr(
        OpenAICodexAuthService,
        "_request_token_refresh",
        lambda self, refresh_token: {
            "access_token": _fake_jwt(4_100_000_000, name="Codex User"),
            "refresh_token": f"{refresh_token}-next",
        },
    )
    workspace_dir = tmp_path / "workspace"
    app, _engine = _build_app(workspace_dir=workspace_dir)

    with TestClient(app) as client:
        import_response = client.post("/v1/model-auth/openai-codex/import-codex-cli")
        refresh_response = client.post("/v1/model-auth/openai-codex/refresh")
        status_response = client.get("/v1/model-auth/openai-codex/status")

    assert import_response.status_code == 200
    assert refresh_response.status_code == 200
    assert status_response.status_code == 200
    refresh_payload = refresh_response.json()
    status_payload = status_response.json()
    assert refresh_payload["profile"]["status"] == "ready"
    assert status_payload["status"] == "ready"
    assert status_payload["last_refresh_error"] is None


def test_openai_codex_status_route_surfaces_refresh_failure_diagnostics(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Failed refresh attempts should be visible in auth status diagnostics."""
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": _fake_jwt(1_700_000_000),
                    "refresh_token": "refresh-token",
                    "account_id": "acct_123",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))

    def _raise_refresh_error(self, _refresh_token: str) -> dict[str, Any]:
        raise RuntimeError("invalid_grant")

    monkeypatch.setattr(OpenAICodexAuthService, "_request_token_refresh", _raise_refresh_error)
    workspace_dir = tmp_path / "workspace"
    app, _engine = _build_app(workspace_dir=workspace_dir)

    with TestClient(app) as client:
        import_response = client.post("/v1/model-auth/openai-codex/import-codex-cli")
        refresh_response = client.post("/v1/model-auth/openai-codex/refresh")
        status_response = client.get("/v1/model-auth/openai-codex/status")

    assert import_response.status_code == 200
    assert refresh_response.status_code == 503
    assert "refresh failed" in refresh_response.json()["detail"].lower()
    assert status_response.status_code == 200
    status_payload = status_response.json()
    assert status_payload["status"] == "refresh_failed"
    assert status_payload["last_refresh_error"] == "invalid_grant"


def test_models_configure_route_supports_openai_codex_without_leaking_oauth_tokens(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """OpenAI Codex configure should use auth_profile_id and keep tokens out of config.yaml."""
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": "header.eyJleHAiOjE5MDAwMDAwMDAsImVtYWlsIjoiY29kZXhAZXhhbXBsZS5jb20ifQ.sig",
                    "refresh_token": "refresh-token",
                    "account_id": "acct_123",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    workspace_dir = tmp_path / "workspace"
    config_path = tmp_path / "config.yaml"
    app, engine = _build_app(workspace_dir=workspace_dir, config_path=config_path)

    with TestClient(app) as client:
        import_response = client.post("/v1/model-auth/openai-codex/import-codex-cli")
        response = client.post(
            "/v1/models/configure",
            json={
                "provider": "openai_codex",
                "base_url": "https://chatgpt.com/backend-api",
                "model": "gpt-5.4",
            },
        )

    assert import_response.status_code == 200
    assert response.status_code == 200
    assert engine.openai_codex_switch_calls == [
        ("https://chatgpt.com/backend-api", "gpt-5.4", "openai_codex:default")
    ]
    payload = response.json()
    assert payload["provider"] == "openai_codex"
    assert payload["api_key_configured"] is False
    assert payload["available_models"][0]["auth_profile_id"] == "openai_codex:default"
    assert payload["available_models"][0]["auth_mode"] == "oauth"
    saved = config_path.read_text(encoding="utf-8")
    assert "auth_profile_id: openai_codex:default" in saved
    assert "refresh-token" not in saved
    assert "access_token" not in saved


def test_models_configure_route_rejects_non_official_openai_codex_base_url(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """OpenAI Codex OAuth tokens must never be routed to arbitrary hosts."""
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": "header.eyJleHAiOjE5MDAwMDAwMDAsImVtYWlsIjoiY29kZXhAZXhhbXBsZS5jb20ifQ.sig",
                    "refresh_token": "refresh-token",
                    "account_id": "acct_123",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    workspace_dir = tmp_path / "workspace"
    app, engine = _build_app(workspace_dir=workspace_dir)

    with TestClient(app) as client:
        import_response = client.post("/v1/model-auth/openai-codex/import-codex-cli")
        response = client.post(
            "/v1/models/configure",
            json={
                "provider": "openai_codex",
                "base_url": "https://example.invalid/backend-api",
                "model": "gpt-5.4",
            },
        )

    assert import_response.status_code == 200
    assert response.status_code == 400
    assert "official ChatGPT backend endpoint" in response.json()["detail"]
    assert engine.openai_codex_switch_calls == []


def test_stale_openai_codex_profile_id_is_not_reported_as_configured(tmp_path: Path) -> None:
    """Missing auth profiles should not keep Codex selected in status or settings payloads."""
    workspace_dir = tmp_path / "workspace"
    app = create_app()
    fake_engine = _FakeEngine()
    app.state.engine_factory = lambda: fake_engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "https://chatgpt.com/backend-api",
            "workspace_dir": str(workspace_dir),
            "openai_codex": {
                "base_url": "https://chatgpt.com/backend-api",
                "model": "gpt-5.4",
                "auth_profile_id": "missing-profile",
            },
        }
    )

    with TestClient(app) as client:
        status_response = client.get("/v1/model-auth/openai-codex/status")
        settings_response = client.get("/v1/settings")
        models_response = client.get("/v1/models")

    assert status_response.status_code == 200
    assert settings_response.status_code == 200
    assert models_response.status_code == 200

    status_payload = status_response.json()
    settings_payload = settings_response.json()
    models_payload = models_response.json()

    assert status_payload["configured"] is False
    assert status_payload["active_profile_id"] is None
    assert settings_payload["model_config"]["provider"] == "openai_compat"
    assert settings_payload["model_config"]["openai_codex_auth_profile_id"] is None
    assert settings_payload["model_config"]["openai_codex_auth_configured"] is False
    assert models_payload["configured_remote_provider"] == "openai_compat"


def test_openai_codex_logout_route_removes_active_profile(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    """Logout should remove the imported profile and clear active auth state."""
    codex_home = tmp_path / "codex-home"
    codex_home.mkdir()
    (codex_home / "auth.json").write_text(
        json.dumps(
            {
                "auth_mode": "chatgpt",
                "tokens": {
                    "access_token": "header.eyJleHAiOjE5MDAwMDAwMDAsImVtYWlsIjoiY29kZXhAZXhhbXBsZS5jb20ifQ.sig",
                    "refresh_token": "refresh-token",
                    "account_id": "acct_123",
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("CODEX_HOME", str(codex_home))
    workspace_dir = tmp_path / "workspace"
    app, _engine = _build_app(workspace_dir=workspace_dir)

    with TestClient(app) as client:
        import_response = client.post("/v1/model-auth/openai-codex/import-codex-cli")
        logout_response = client.post("/v1/model-auth/openai-codex/logout")
        status_response = client.get("/v1/model-auth/openai-codex/status")

    assert import_response.status_code == 200
    assert logout_response.status_code == 200
    assert status_response.status_code == 200
    logout_payload = logout_response.json()
    status_payload = status_response.json()
    assert logout_payload["deleted"] is True
    assert logout_payload["active_profile_id"] is None
    assert "store_path" not in logout_response.text
    assert status_payload["configured"] is False
    assert status_payload["active_profile_id"] is None
    assert status_payload["profiles"] == []


def test_models_configure_route_appends_to_available_models() -> None:
    """多次成功設定後 `/v1/models` 應回傳可供聊天頁選擇的模型列表。"""
    app, engine = _build_app()

    with TestClient(app) as client:
        first_response = client.post(
            "/v1/models/configure",
            json={
                "provider": "ollama",
                "base_url": "http://localhost:11434",
                "model": "qwen2.5",
            },
        )
        second_response = client.post(
            "/v1/models/configure",
            json={
                "provider": "openai_compat",
                "base_url": "https://api.example.com/v1",
                "model": "gpt-test",
                "api_key": "sk-secret-value",
            },
        )
        models_response = client.get("/v1/models")

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert models_response.status_code == 200
    assert engine.ollama_switch_calls == [("qwen2.5", "http://localhost:11434")]
    assert engine.openai_switch_calls == [
        ("https://api.example.com/v1", "gpt-test", "sk-secret-value", "openai_compat")
    ]
    assert [model["id"] for model in models_response.json()["available_models"]] == [
        "openai_compat:https://api.example.com/v1:gpt-test",
        "ollama:qwen2.5",
    ]
    assert "sk-secret-value" not in models_response.text


def test_models_switch_route_accepts_available_model_id() -> None:
    """聊天頁模型下拉以 available model id 切換時應還原 provider/model/base_url。"""
    config = MochiConfig.model_validate(
        {
            "model": "ollama:qwen2.5",
            "model_setup": {
                "configured_models": [
                    {
                        "id": "openai_compat:https://api.example.com/v1:gpt-test",
                        "provider": "openai_compat",
                        "model": "gpt-test",
                        "model_spec": "https://api.example.com/v1",
                        "base_url": "https://api.example.com/v1",
                        "label": "gpt-test (openai_compat)",
                        "backend_type": "openai_compat",
                    },
                    {
                        "id": "ollama:qwen2.5",
                        "provider": "ollama",
                        "model": "qwen2.5",
                        "model_spec": "ollama:qwen2.5",
                        "base_url": "http://localhost:11434",
                        "label": "qwen2.5",
                        "backend_type": "ollama",
                    },
                ]
            },
            "openai_compat": {
                "provider": "openai_compat",
                "base_url": "https://api.example.com/v1",
                "model": "gpt-test",
                "api_key": "sk-secret-value",
            },
        }
    )
    app, engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/switch",
            json={"model": "openai_compat:https://api.example.com/v1:gpt-test"},
        )

    assert response.status_code == 200
    assert engine.openai_switch_calls == [
        ("https://api.example.com/v1", "gpt-test", "sk-secret-value", "openai_compat")
    ]
    assert response.json()["active_model"]["name"] == "gpt-test"
    assert "sk-secret-value" not in response.text


def test_models_switch_route_accepts_bare_ollama_model_name() -> None:
    """Ollama active model 回傳裸模型名時，switch API 應還原到已設定清單項目。"""
    config = MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "ollama": {"base_url": "http://localhost:11434"},
            "model_setup": {
                "configured_models": [
                    {
                        "id": "ollama:qwen2.5",
                        "provider": "ollama",
                        "model": "qwen2.5",
                        "model_spec": "ollama:qwen2.5",
                        "base_url": "http://localhost:11434",
                        "label": "qwen2.5",
                        "backend_type": "ollama",
                    },
                ]
            },
        }
    )
    app, engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.post("/v1/models/switch", json={"model": "qwen2.5"})

    assert response.status_code == 200
    assert engine.ollama_switch_calls == [("qwen2.5", "http://localhost:11434")]
    assert engine.switch_calls == []
    assert response.json()["active_model"]["name"] == "qwen2.5"


def test_models_switch_route_accepts_bare_active_ollama_model_name() -> None:
    """已 active 的 Ollama 模型以裸名切換時不應落到通用 switch_model。"""
    config = MochiConfig.model_validate(
        {
            "model": "ollama:qwen2.5",
            "ollama": {"base_url": "http://localhost:11434"},
            "model_setup": {
                "configured_models": [
                    {
                        "id": "ollama:qwen2.5",
                        "provider": "ollama",
                        "model": "qwen2.5",
                        "model_spec": "ollama:qwen2.5",
                        "base_url": "http://localhost:11434",
                        "label": "qwen2.5",
                        "backend_type": "ollama",
                    },
                ]
            },
        }
    )
    app, engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.post("/v1/models/switch", json={"model": "qwen2.5"})

    assert response.status_code == 200
    assert engine.ollama_switch_calls == []
    assert engine.switch_calls == []


def test_models_configure_route_persists_ollama_selection(tmp_path: Path) -> None:
    """成功切換模型後應保存到指定 YAML，避免重啟後回到預設模型。"""
    config_path = tmp_path / "config.yaml"
    app, engine = _build_app(config_path=config_path)

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/configure",
            json={
                "provider": "ollama",
                "base_url": "http://localhost:11434",
                "model": "qwen2.5",
            },
        )

    assert response.status_code == 200
    assert engine.ollama_switch_calls == [("qwen2.5", "http://localhost:11434")]
    assert response.json()["persisted"] is True
    assert "qwen2.5" in config_path.read_text(encoding="utf-8")
    assert "ollama:qwen2.5" in config_path.read_text(encoding="utf-8")


def test_models_configure_route_persists_multiple_available_models(tmp_path: Path) -> None:
    """連續新增模型後，重載 YAML 應保留完整可用模型清單。"""
    config_path = tmp_path / "config.yaml"
    app, engine = _build_app(config_path=config_path)

    with TestClient(app) as client:
        first_response = client.post(
            "/v1/models/configure",
            json={
                "provider": "ollama",
                "base_url": "http://localhost:11434",
                "model": "qwen2.5",
            },
        )
        second_response = client.post(
            "/v1/models/configure",
            json={
                "provider": "openai_compat",
                "base_url": "https://api.example.com/v1",
                "model": "gpt-test",
                "api_key": "sk-secret-value",
            },
        )
        models_response = client.get("/v1/models")

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert models_response.status_code == 200
    assert engine.ollama_switch_calls == [("qwen2.5", "http://localhost:11434")]
    assert engine.openai_switch_calls == [
        ("https://api.example.com/v1", "gpt-test", "sk-secret-value", "openai_compat")
    ]

    expected_ids = [
        "openai_compat:https://api.example.com/v1:gpt-test",
        "ollama:qwen2.5",
    ]
    assert [model["id"] for model in second_response.json()["available_models"]] == expected_ids
    assert [model["id"] for model in models_response.json()["available_models"]] == expected_ids

    saved_config = load_config(config_path)
    assert [model.id for model in saved_config.model_setup.configured_models] == expected_ids
    assert saved_config.model == "https://api.example.com/v1"
    assert saved_config.openai_compat.model == "gpt-test"
    assert "sk-secret-value" not in models_response.text


def test_models_configure_route_supports_gemini_preset_without_leaking_key() -> None:
    """Gemini provider preset 應走 OpenAI-compatible backend 並保存 provider。"""
    app, engine = _build_app()

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/configure",
            json={
                "provider": "gemini",
                "model": "gemini-2.5-flash",
                "api_key": "gemini-secret",
            },
        )

    assert response.status_code == 200
    assert engine.openai_switch_calls == [
        (
            "https://generativelanguage.googleapis.com/v1beta/openai",
            "gemini-2.5-flash",
            "gemini-secret",
            "gemini",
        )
    ]
    assert response.json()["provider"] == "gemini"
    assert response.json()["active_model"]["name"] == "gemini-2.5-flash"
    assert "gemini-secret" not in response.text


def test_models_configure_route_supports_vllm_preset_without_leaking_key() -> None:
    """vLLM provider preset 應走 OpenAI-compatible backend 並保存 provider。"""
    app, engine = _build_app()

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/configure",
            json={
                "provider": "vllm",
                "model": "qwen2.5-7b-instruct",
                "api_key": "vllm-secret",
            },
        )

    assert response.status_code == 200
    assert engine.openai_switch_calls == [
        (
            "http://localhost:8000/v1",
            "qwen2.5-7b-instruct",
            "vllm-secret",
            "vllm",
        )
    ]
    payload = response.json()
    assert payload["provider"] == "vllm"
    assert payload["active_model"]["name"] == "qwen2.5-7b-instruct"
    assert payload["available_models"][0]["provider"] == "vllm"
    assert payload["available_models"][0]["backend_type"] == "openai_compat"
    assert payload["available_models"][0]["id"] == (
        "vllm:http://localhost:8000/v1:qwen2.5-7b-instruct"
    )
    assert "vllm-secret" not in response.text


def test_models_vllm_runtime_status_start_stop_with_managed_entry() -> None:
    """vLLM managed runtime endpoints 應回報狀態並可啟停單一 instance。"""
    manager = _FakeManagedVLLMRuntimeManager()
    config = MochiConfig.model_validate(
        {
            "model": "ollama:qwen2.5",
            "model_setup": {
                "configured_models": [
                    {
                        "id": "vllm-managed-qwen",
                        "provider": "vllm",
                        "model": "Qwen/Qwen2.5-7B-Instruct",
                        "model_spec": "Qwen/Qwen2.5-7B-Instruct",
                        "base_url": "http://localhost:8000/v1",
                        "label": "Qwen/Qwen2.5-7B-Instruct (vllm managed)",
                        "backend_type": "openai_compat",
                    }
                ]
            },
        }
    )
    app, _engine = _build_app(vllm_runtime_manager=manager)
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        status_before = client.get("/v1/models/vllm/runtime")
        started = client.post(
            "/v1/models/vllm/runtime/start",
            json={"model_id": "vllm-managed-qwen"},
        )
        stopped = client.post("/v1/models/vllm/runtime/stop")

    assert status_before.status_code == 200
    assert status_before.json()["running"] is False

    assert started.status_code == 200
    start_payload = started.json()
    assert start_payload["action"] == "start"
    assert start_payload["runtime_status"]["running"] is True
    assert start_payload["runtime_status"]["active_model_spec"] == "Qwen/Qwen2.5-7B-Instruct"
    assert manager.start_calls == [
        ("vllm-managed-qwen", "Qwen/Qwen2.5-7B-Instruct", "http://localhost:8000/v1", "managed")
    ]

    assert stopped.status_code == 200
    stop_payload = stopped.json()
    assert stop_payload["action"] == "stop"
    assert stop_payload["runtime_status"]["running"] is False
    assert manager.stop_calls == 2


def test_models_vllm_runtime_start_rejects_managed_gguf_model_spec() -> None:
    """managed vLLM start 應拒絕 .gguf target。"""
    manager = _FakeManagedVLLMRuntimeManager()
    config = MochiConfig.model_validate(
        {
            "model": "ollama:qwen2.5",
            "model_setup": {
                "configured_models": [
                    {
                        "id": "vllm-managed-gguf",
                        "provider": "vllm",
                        "model": "demo.gguf",
                        "model_spec": "demo.gguf",
                        "base_url": "http://localhost:8000/v1",
                        "label": "demo.gguf (vllm managed)",
                        "backend_type": "openai_compat",
                    }
                ]
            },
        }
    )
    app, _engine = _build_app(vllm_runtime_manager=manager)
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/vllm/runtime/start",
            json={"model_id": "vllm-managed-gguf"},
        )

    assert response.status_code == 400
    assert ".gguf" in response.json()["detail"]
    assert manager.start_calls == []


def test_models_configure_route_supports_managed_vllm_target_and_starts_runtime() -> None:
    """vLLM managed configure 路徑應啟動 runtime 並走 openai_compat backend。"""
    manager = _FakeManagedVLLMRuntimeManager()
    app, engine = _build_app(vllm_runtime_manager=manager)

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/configure",
            json={
                "provider": "vllm",
                "model": "Qwen/Qwen2.5-7B-Instruct",
            },
        )

    assert response.status_code == 200
    assert manager.start_calls == [
        (None, "Qwen/Qwen2.5-7B-Instruct", "http://localhost:8000/v1", "managed")
    ]
    assert engine.openai_switch_calls == [
        (
            "http://localhost:8000/v1",
            "Qwen/Qwen2.5-7B-Instruct",
            "",
            "vllm",
        )
    ]

    payload = response.json()
    assert payload["provider"] == "vllm"
    assert payload["active_model"]["name"] == "Qwen/Qwen2.5-7B-Instruct"
    assert payload["available_models"][0]["provider"] == "vllm"
    assert payload["available_models"][0]["model_spec"] == "Qwen/Qwen2.5-7B-Instruct"
    assert payload["available_models"][0]["backend_type"] == "openai_compat"


def test_models_switch_route_accepts_available_vllm_model_id() -> None:
    """`POST /v1/models/switch` 應可透過 vLLM configured model id 還原 provider。"""
    config = MochiConfig.model_validate(
        {
            "model": "ollama:qwen2.5",
            "model_setup": {
                "configured_models": [
                    {
                        "id": "vllm:http://localhost:8000/v1:qwen2.5-7b-instruct",
                        "provider": "vllm",
                        "model": "qwen2.5-7b-instruct",
                        "model_spec": "http://localhost:8000/v1",
                        "base_url": "http://localhost:8000/v1",
                        "label": "qwen2.5-7b-instruct (vllm)",
                        "backend_type": "openai_compat",
                    },
                    {
                        "id": "ollama:qwen2.5",
                        "provider": "ollama",
                        "model": "qwen2.5",
                        "model_spec": "ollama:qwen2.5",
                        "base_url": "http://localhost:11434",
                        "label": "qwen2.5",
                        "backend_type": "ollama",
                    },
                ]
            },
            "openai_compat": {
                "provider": "vllm",
                "base_url": "http://localhost:8000/v1",
                "model": "qwen2.5-7b-instruct",
                "api_key": "vllm-secret",
            },
        }
    )
    app, engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/switch",
            json={"model": "vllm:http://localhost:8000/v1:qwen2.5-7b-instruct"},
        )

    assert response.status_code == 200
    assert engine.openai_switch_calls == [
        (
            "http://localhost:8000/v1",
            "qwen2.5-7b-instruct",
            "vllm-secret",
            "vllm",
        )
    ]
    assert response.json()["active_model"]["name"] == "qwen2.5-7b-instruct"
    assert response.json()["active_model"]["id"] == "vllm:http://localhost:8000/v1:qwen2.5-7b-instruct"
    assert "vllm-secret" not in response.text


@pytest.mark.parametrize(
    "provider",
    ["sglang", "tensorrt_llm"],
)
def test_models_switch_route_keeps_external_provider_out_of_managed_vllm_path(
    provider: str,
) -> None:
    base_url = "http://remote.example.test/v1"
    model_name = "qwen2.5-7b-instruct"
    api_key = "sk-provider-secret"
    config = MochiConfig.model_validate(
        {
            "model": "ollama:qwen2.5",
            "model_setup": {
                "configured_models": [
                    {
                        "id": f"{provider}:{base_url}:{model_name}",
                        "provider": provider,
                        "model": model_name,
                        "model_spec": base_url,
                        "base_url": base_url,
                        "label": f"{model_name} ({provider})",
                        "backend_type": "openai_compat",
                        "launch_mode": "managed",
                    }
                ]
            },
            "openai_compat": {
                "provider": provider,
                "base_url": base_url,
                "model": model_name,
                "api_key": api_key,
            },
        }
    )
    manager = _FakeManagedVLLMRuntimeManager()
    app, engine = _build_app(vllm_runtime_manager=manager)
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/switch",
            json={"model": f"{provider}:{base_url}:{model_name}"},
        )

    assert response.status_code == 200
    assert manager.start_calls == []
    assert engine.openai_switch_calls == [
        (base_url, model_name, api_key, provider)
    ]
    payload = response.json()
    assert payload["active_model"]["id"] == f"{provider}:{base_url}:{model_name}"
    assert payload["active_model"]["provider"] == provider
    assert payload["active_model"]["backend_type"] == "openai_compat"
    assert "sk-provider-secret" not in response.text


def test_models_switch_route_starts_managed_vllm_runtime_for_managed_entry() -> None:
    """切換到 managed vLLM configured model 時應先啟動 runtime。"""
    manager = _FakeManagedVLLMRuntimeManager()
    config = MochiConfig.model_validate(
        {
            "model": "ollama:qwen2.5",
            "model_setup": {
                "configured_models": [
                    {
                        "id": "vllm-managed-qwen",
                        "provider": "vllm",
                        "model": "Qwen/Qwen2.5-7B-Instruct",
                        "model_spec": "Qwen/Qwen2.5-7B-Instruct",
                        "base_url": "http://localhost:8000/v1",
                        "label": "Qwen/Qwen2.5-7B-Instruct (vllm managed)",
                        "backend_type": "openai_compat",
                    },
                    {
                        "id": "ollama:qwen2.5",
                        "provider": "ollama",
                        "model": "qwen2.5",
                        "model_spec": "ollama:qwen2.5",
                        "base_url": "http://localhost:11434",
                        "label": "qwen2.5",
                        "backend_type": "ollama",
                    },
                ]
            },
            "openai_compat": {
                "provider": "vllm",
                "base_url": "http://localhost:8000/v1",
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "api_key": "vllm-secret",
            },
        }
    )
    app, engine = _build_app(vllm_runtime_manager=manager)
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/switch",
            json={"model": "vllm-managed-qwen"},
        )

    assert response.status_code == 200
    assert manager.start_calls == [
        ("vllm-managed-qwen", "Qwen/Qwen2.5-7B-Instruct", "http://localhost:8000/v1", "managed")
    ]
    assert engine.openai_switch_calls == [
        (
            "http://localhost:8000/v1",
            "Qwen/Qwen2.5-7B-Instruct",
            "vllm-secret",
            "vllm",
        )
    ]
    assert response.json()["active_model"]["name"] == "Qwen/Qwen2.5-7B-Instruct"
    assert response.json()["active_model"]["id"] == "vllm-managed-qwen"


def test_models_status_preserves_configured_vllm_active_model_id() -> None:
    """`GET /v1/models` should keep the configured vLLM model id on the active model payload."""
    config = MochiConfig.model_validate(
        {
            "model": "http://127.0.0.1:18000/v1",
            "openai_compat": {
                "provider": "vllm",
                "base_url": "http://127.0.0.1:18000/v1",
                "model": "google/gemma-4-26B-A4B-it",
                "api_key": "",
            },
            "model_setup": {
                "configured_models": [
                    {
                        "id": "vllm:http://127.0.0.1:18000/v1:google/gemma-4-26B-A4B-it",
                        "provider": "vllm",
                        "model": "google/gemma-4-26B-A4B-it",
                        "model_spec": "http://127.0.0.1:18000/v1",
                        "base_url": "http://127.0.0.1:18000/v1",
                        "label": "google/gemma-4-26B-A4B-it (vllm)",
                        "backend_type": "openai_compat",
                    }
                ]
            },
        }
    )
    app, engine = _build_app()
    engine.model_info = ModelInfo(
        name="google/gemma-4-26B-A4B-it",
        provider="vllm",
        backend_type="openai_compat",
        supports_tool_calling=True,
        metadata={"base_url": "http://127.0.0.1:18000/v1"},
    )
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["active_model"]["id"] == "vllm:http://127.0.0.1:18000/v1:google/gemma-4-26B-A4B-it"
    assert payload["active_model"]["model_spec"] == "http://127.0.0.1:18000/v1"
    assert payload["active_model"]["base_url"] == "http://127.0.0.1:18000/v1"


def test_models_configured_patch_updates_vllm_remote_entry_without_leaking_api_key() -> None:
    """`PATCH /v1/models/configured/{id}` 應支援更新 vLLM remote entry。"""
    config = MochiConfig.model_validate(
        {
            "model": "http://localhost:8000/v1",
            "openai_compat": {
                "provider": "vllm",
                "base_url": "http://localhost:8000/v1",
                "model": "qwen2.5-7b-instruct",
                "api_key": "vllm-old-secret",
            },
            "model_setup": {
                "configured_models": [
                    {
                        "id": "vllm:http://localhost:8000/v1:qwen2.5-7b-instruct",
                        "provider": "vllm",
                        "model": "qwen2.5-7b-instruct",
                        "model_spec": "http://localhost:8000/v1",
                        "base_url": "http://localhost:8000/v1",
                        "label": "qwen2.5-7b-instruct (vllm)",
                        "backend_type": "openai_compat",
                    }
                ]
            },
        }
    )
    app, _engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.patch(
            "/v1/models/configured/vllm%3Ahttp%3A%2F%2Flocalhost%3A8000%2Fv1%3Aqwen2.5-7b-instruct",
            json={
                "provider": "vllm",
                "model": "qwen3-8b",
                "model_spec": "http://localhost:9000/v1",
                "base_url": "http://localhost:9000/v1",
                "api_key": "vllm-new-secret",
                "persist": False,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "model_entry_update"
    assert payload["updated_model"]["provider"] == "vllm"
    assert payload["updated_model"]["model"] == "qwen3-8b"
    assert payload["updated_model"]["model_spec"] == "http://localhost:9000/v1"
    assert payload["updated_model"]["base_url"] == "http://localhost:9000/v1"
    assert payload["updated_model"]["id"] == "vllm:http://localhost:9000/v1:qwen3-8b"
    assert payload["updated_model"]["backend_type"] == "openai_compat"
    assert payload["updated_model"]["launch_mode"] == "external"
    assert payload["api_key_configured"] is True
    assert payload["configured_model"] == "http://localhost:9000/v1"
    assert "vllm-new-secret" not in response.text
    assert "vllm-old-secret" not in response.text


def test_models_probe_tool_calling_returns_probe_payload() -> None:
    app, fake_engine = _build_app()
    fake_engine.model_info = ModelInfo(
        name="google/gemma-4-26B-A4B-it",
        provider="vllm",
        backend_type="openai_compat",
        supports_tool_calling=False,
        metadata={
            "tool_call_mode": "simulated_fallback",
            "native_tool_calling_status": "rejected_missing_parser",
        },
    )
    fake_engine.tool_probe_result = {
        "status": "rejected_missing_parser",
        "message": "vLLM rejected native auto tool choice.",
    }

    with TestClient(app) as client:
        response = client.post("/v1/models/probe-tool-calling")

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "tool_calling_probe"
    assert payload["probe"]["status"] == "rejected_missing_parser"
    assert payload["active_model"]["metadata"]["tool_call_mode"] == "simulated_fallback"


def test_models_probe_tool_calling_returns_post_probe_active_model_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:qwen2.5",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
        }
    )
    app, _fake_engine = _build_app()
    real_engine = AgentEngine(config)

    class _ProbeBackend:
        def __init__(self) -> None:
            self.metadata = {
                "tool_call_mode": "simulated_fallback",
                "native_tool_calling_status": "native_tool_calls_missing",
            }
            self.probe_calls = 0
            self.closed = False
            self.close_calls = 0

        async def probe_tool_calling(self) -> dict[str, Any] | None:
            self.probe_calls += 1
            self.metadata.update(
                {
                    "tool_call_mode": "native",
                    "native_tool_calling_status": "supported",
                }
            )
            return {
                "status": "supported",
                "message": "native structured tool calling succeeded",
                "metadata": dict(self.metadata),
            }

        def get_model_info(self) -> ModelInfo:
            supports_tool_calling = not (
                self.metadata.get("tool_call_mode") == "unavailable"
                or self.metadata.get("tool_calling_blocked") is True
            )
            return ModelInfo(
                name="qwen2.5",
                provider="ollama",
                backend_type="ollama",
                supports_tool_calling=supports_tool_calling,
                metadata=dict(self.metadata),
            )

        async def close(self) -> None:
            self.close_calls += 1
            self.closed = True

    class _StaleInfoBackend:
        def __init__(self) -> None:
            self.get_model_info_calls = 0

        def get_model_info(self) -> ModelInfo:
            self.get_model_info_calls += 1
            return ModelInfo(
                name="qwen2.5",
                provider="ollama",
                backend_type="ollama",
                supports_tool_calling=True,
                metadata={
                    "tool_call_mode": "simulated_fallback",
                    "native_tool_calling_status": "native_tool_calls_missing",
                },
            )

    probe_backend = _ProbeBackend()
    stale_info_backend = _StaleInfoBackend()
    resolve_calls = 0

    async def fake_acquire_temporary_backend(*, model_spec: str, **kwargs: Any) -> _ProbeBackend:
        del model_spec, kwargs
        return probe_backend

    def fake_resolve(model_spec: str, **kwargs: Any) -> _StaleInfoBackend:
        nonlocal resolve_calls
        del model_spec, kwargs
        resolve_calls += 1
        return stale_info_backend

    monkeypatch.setattr(real_engine._router, "acquire_temporary_backend", fake_acquire_temporary_backend)  # noqa: SLF001
    monkeypatch.setattr(real_engine._router, "_resolve", fake_resolve)  # noqa: SLF001
    app.state.engine = real_engine

    with TestClient(app) as client:
        response = client.post("/v1/models/probe-tool-calling")

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "tool_calling_probe"
    assert payload["probe"]["status"] == "supported"
    assert payload["active_model"]["metadata"]["tool_call_mode"] == "native"
    assert payload["active_model"]["metadata"]["native_tool_calling_status"] == "supported"
    assert payload["active_model"]["supports_tool_calling"] is True
    assert probe_backend.probe_calls == 1
    assert probe_backend.close_calls == 1
    assert probe_backend.closed is True
    assert resolve_calls == 0
    assert stale_info_backend.get_model_info_calls == 0


def test_models_test_connection_route_validates_explicit_remote_payload_without_switching() -> None:
    app, fake_engine = _build_app()

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/test-connection",
            json={
                "provider": "openai_compat",
                "model": "gpt-4.1-mini",
                "base_url": "https://api.example.com/v1",
                "api_key": "test-secret",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "model_connection_test"
    assert payload["provider"] == "openai_compat"
    assert payload["tested_model"]["model_spec"] == "https://api.example.com/v1"
    assert payload["tested_model"]["base_url"] == "https://api.example.com/v1"
    assert payload["tested_model"]["metadata"]["tested"] is True
    assert fake_engine.test_connection_calls == [
        {
            "provider": "openai_compat",
            "model": "gpt-4.1-mini",
            "base_url": "https://api.example.com/v1",
            "api_key": "test-secret",
            "auth_profile_id": None,
        }
    ]
    assert fake_engine.switch_calls == []
    assert fake_engine.openai_switch_calls == []


def test_models_test_connection_route_supports_saved_model_id() -> None:
    config = MochiConfig.model_validate(
        {
            "model": "https://api.example.com/v1",
            "openai_compat": {
                "provider": "openai_compat",
                "base_url": "https://api.example.com/v1",
                "model": "gpt-4.1-mini",
                "api_key": "saved-secret",
            },
            "model_setup": {
                "configured_models": [
                    {
                        "id": "openai_compat:https://api.example.com/v1:gpt-4.1-mini",
                        "provider": "openai_compat",
                        "model": "gpt-4.1-mini",
                        "model_spec": "https://api.example.com/v1",
                        "base_url": "https://api.example.com/v1",
                        "label": "gpt-4.1-mini (openai_compat)",
                        "backend_type": "openai_compat",
                    }
                ]
            },
        }
    )
    app, fake_engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/test-connection",
            json={
                "model_id": "openai_compat:https://api.example.com/v1:gpt-4.1-mini",
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["provider"] == "openai_compat"
    assert payload["tested_model"]["id"] == "openai_compat:https://api.example.com/v1:gpt-4.1-mini"
    assert payload["tested_model"]["base_url"] == "https://api.example.com/v1"
    assert fake_engine.test_connection_calls == [
        {
            "provider": "openai_compat",
            "model": "gpt-4.1-mini",
            "base_url": "https://api.example.com/v1",
            "api_key": "saved-secret",
            "auth_profile_id": None,
        }
    ]
    assert fake_engine.switch_calls == []
    assert fake_engine.openai_switch_calls == []


def test_models_test_connection_route_prefers_saved_model_api_key_over_global_runtime_key() -> None:
    config = MochiConfig.model_validate(
        {
            "model": "https://active.example.com/v1",
            "openai_compat": {
                "provider": "openai_compat",
                "base_url": "https://active.example.com/v1",
                "model": "active-model",
                "api_key": "runtime-active-secret",
            },
            "model_setup": {
                "configured_models": [
                    {
                        "id": "openai_compat:https://api.example.com/v1:gpt-4.1-mini",
                        "provider": "openai_compat",
                        "model": "gpt-4.1-mini",
                        "model_spec": "https://api.example.com/v1",
                        "base_url": "https://api.example.com/v1",
                        "label": "gpt-4.1-mini (openai_compat)",
                        "backend_type": "openai_compat",
                        "api_key": "saved-model-secret",
                    }
                ]
            },
        }
    )
    app, fake_engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/test-connection",
            json={
                "model_id": "openai_compat:https://api.example.com/v1:gpt-4.1-mini",
            },
        )

    assert response.status_code == 200
    assert fake_engine.test_connection_calls == [
        {
            "provider": "openai_compat",
            "model": "gpt-4.1-mini",
            "base_url": "https://api.example.com/v1",
            "api_key": "saved-model-secret",
            "auth_profile_id": None,
        }
    ]
    assert "saved-model-secret" not in response.text
    assert "runtime-active-secret" not in response.text


def test_models_configured_patch_preserves_managed_vllm_entry_model_spec() -> None:
    """PATCH managed vLLM configured entry 時應保留 managed model_spec 與 launch_mode。"""
    config = MochiConfig.model_validate(
        {
            "model": "http://localhost:8000/v1",
            "openai_compat": {
                "provider": "vllm",
                "base_url": "http://localhost:8000/v1",
                "model": "Qwen/Qwen2.5-7B-Instruct",
                "api_key": "vllm-old-secret",
            },
            "model_setup": {
                "configured_models": [
                    {
                        "id": "vllm-managed-qwen",
                        "provider": "vllm",
                        "model": "Qwen/Qwen2.5-7B-Instruct",
                        "model_spec": "Qwen/Qwen2.5-7B-Instruct",
                        "base_url": "http://localhost:8000/v1",
                        "label": "Qwen/Qwen2.5-7B-Instruct (vllm managed)",
                        "backend_type": "openai_compat",
                        "launch_mode": "managed",
                    }
                ]
            },
        }
    )
    app, _engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.patch(
            "/v1/models/configured/vllm-managed-qwen",
            json={
                "api_key": "vllm-new-secret",
                "persist": False,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "model_entry_update"
    assert payload["updated_model"]["provider"] == "vllm"
    assert payload["updated_model"]["id"] == "vllm-managed-qwen"
    assert payload["updated_model"]["model_spec"] == "Qwen/Qwen2.5-7B-Instruct"
    assert payload["updated_model"]["launch_mode"] == "managed"
    assert payload["available_models"][0]["id"] == "vllm-managed-qwen"
    assert payload["available_models"][0]["model_spec"] == "Qwen/Qwen2.5-7B-Instruct"
    assert payload["available_models"][0]["launch_mode"] == "managed"
    assert payload["api_key_configured"] is True
    assert payload["configured_model"] == "http://localhost:8000/v1"
    assert "vllm-new-secret" not in response.text
    assert "vllm-old-secret" not in response.text


def test_models_configured_patch_updates_managed_vllm_entry_from_model_field() -> None:
    """managed vLLM PATCH 應允許以 model 欄位更新 managed target 並保留 managed 模式。"""
    config = MochiConfig.model_validate(
        {
            "model": "http://localhost:8000/v1",
            "openai_compat": {
                "provider": "vllm",
                "base_url": "http://localhost:8000/v1",
                "model": "Qwen/Qwen2.5-7B-Instruct",
            },
            "model_setup": {
                "configured_models": [
                    {
                        "id": "vllm-managed-qwen",
                        "provider": "vllm",
                        "model": "Qwen/Qwen2.5-7B-Instruct",
                        "model_spec": "Qwen/Qwen2.5-7B-Instruct",
                        "base_url": "http://localhost:8000/v1",
                        "label": "Qwen/Qwen2.5-7B-Instruct (vllm managed)",
                        "backend_type": "openai_compat",
                        "launch_mode": "managed",
                    }
                ]
            },
        }
    )
    app, _engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.patch(
            "/v1/models/configured/vllm-managed-qwen",
            json={
                "model": "Qwen/Qwen3-8B",
                "persist": False,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["updated_model"]["model"] == "Qwen/Qwen3-8B"
    assert payload["updated_model"]["model_spec"] == "Qwen/Qwen3-8B"
    assert payload["updated_model"]["launch_mode"] == "managed"
    assert payload["available_models"][0]["model"] == "Qwen/Qwen3-8B"
    assert payload["available_models"][0]["model_spec"] == "Qwen/Qwen3-8B"
    assert payload["available_models"][0]["launch_mode"] == "managed"


def test_models_configured_patch_updates_remote_entry_without_leaking_api_key() -> None:
    """`PATCH /v1/models/configured/{id}` 應可更新 remote entry 並保留 secret 規則。"""
    config = MochiConfig.model_validate(
        {
            "model": "https://api.example.com/v1",
            "openai_compat": {
                "provider": "openai_compat",
                "base_url": "https://api.example.com/v1",
                "model": "gpt-test",
                "api_key": "sk-old-secret",
            },
            "model_setup": {
                "configured_models": [
                    {
                        "id": "openai_compat:https://api.example.com/v1:gpt-test",
                        "provider": "openai_compat",
                        "model": "gpt-test",
                        "model_spec": "https://api.example.com/v1",
                        "base_url": "https://api.example.com/v1",
                        "label": "gpt-test (openai_compat)",
                        "backend_type": "openai_compat",
                    }
                ]
            },
        }
    )
    app, _engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.patch(
            "/v1/models/configured/openai_compat%3Ahttps%3A%2F%2Fapi.example.com%2Fv1%3Agpt-test",
            json={
                "provider": "openai_compat",
                "model": "gpt-new",
                "model_spec": "https://api.new-example.com/v1",
                "base_url": "https://api.new-example.com/v1",
                "api_key": "sk-new-secret",
                "persist": False,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "model_entry_update"
    assert payload["updated_model"]["model"] == "gpt-new"
    assert payload["updated_model"]["model_spec"] == "https://api.new-example.com/v1"
    assert payload["updated_model"]["base_url"] == "https://api.new-example.com/v1"
    assert payload["api_key_configured"] is True
    assert payload["configured_model"] == "https://api.new-example.com/v1"
    assert "sk-new-secret" not in response.text
    assert "sk-old-secret" not in response.text


def test_models_configured_patch_updates_local_entry_path(tmp_path: Path) -> None:
    """`PATCH /v1/models/configured/{id}` 應可更新 local entry 路徑。"""
    first = tmp_path / "first.gguf"
    second = tmp_path / "second.gguf"
    first.write_text("gguf", encoding="utf-8")
    second.write_text("gguf", encoding="utf-8")

    config = MochiConfig.model_validate(
        {
            "model": str(first.resolve()),
            "local_models": {"roots": [str(tmp_path)]},
            "model_setup": {
                "configured_models": [
                    {
                        "id": str(first.resolve()),
                        "provider": "local",
                        "model": first.name,
                        "model_spec": str(first.resolve()),
                        "label": first.name,
                        "backend_type": "gguf",
                    }
                ]
            },
        }
    )
    app, _engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.patch(
            f"/v1/models/configured/{first.resolve()}",
            json={
                "provider": "local",
                "model": second.name,
                "model_spec": str(second.resolve()),
                "persist": False,
            },
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["updated_model"]["provider"] == "local"
    assert payload["updated_model"]["model_spec"] == str(second.resolve())
    assert payload["configured_model"] == str(second.resolve())
    assert payload["api_key_configured"] is False


def test_models_configured_delete_removes_entry_and_falls_back_to_remaining_model() -> None:
    """`DELETE /v1/models/configured/{id}` 應刪除指定 entry 並維持可用 configured model。"""
    config = MochiConfig.model_validate(
        {
            "model": "ollama:qwen2.5",
            "ollama": {"base_url": "http://localhost:11434"},
            "model_setup": {
                "configured_models": [
                    {
                        "id": "ollama:qwen2.5",
                        "provider": "ollama",
                        "model": "qwen2.5",
                        "model_spec": "ollama:qwen2.5",
                        "base_url": "http://localhost:11434",
                        "label": "qwen2.5",
                        "backend_type": "ollama",
                    },
                    {
                        "id": "openai_compat:https://api.example.com/v1:gpt-test",
                        "provider": "openai_compat",
                        "model": "gpt-test",
                        "model_spec": "https://api.example.com/v1",
                        "base_url": "https://api.example.com/v1",
                        "label": "gpt-test (openai_compat)",
                        "backend_type": "openai_compat",
                    },
                ]
            },
        }
    )
    app, _engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.request(
            "DELETE",
            "/v1/models/configured/ollama%3Aqwen2.5",
            json={"persist": False},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "model_entry_delete"
    assert payload["deleted_model_id"] == "ollama:qwen2.5"
    assert payload["configured_model"] == "https://api.example.com/v1"
    assert [item["id"] for item in payload["available_models"]] == [
        "openai_compat:https://api.example.com/v1:gpt-test"
    ]


def test_models_configured_delete_accepts_wsl_alias_for_windows_local_model() -> None:
    """本地模型刪除應接受對應的 WSL 路徑識別。"""
    config = MochiConfig.model_validate(
        {
            "model": r"J:\_models\Qwen3.5-9B",
            "model_setup": {
                "configured_models": [
                    {
                        "id": r"J:\_models\Qwen3.5-9B",
                        "provider": "local",
                        "model": "Qwen3.5-9B",
                        "model_spec": r"J:\_models\Qwen3.5-9B",
                        "label": "Qwen3.5-9B",
                        "backend_type": "safetensors",
                    }
                ]
            },
        }
    )
    app, _engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.request(
            "DELETE",
            "/v1/models/configured/%2Fmnt%2Fj%2F_models%2FQwen3.5-9B",
            json={"persist": False},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["deleted_model_id"] == r"J:\_models\Qwen3.5-9B"
    assert payload["available_models"] == []


def test_models_configured_delete_active_last_remote_resets_to_default_model() -> None:
    """Deleting the active last saved remote model should stop it from being rehydrated as a saved entry."""
    config = MochiConfig.model_validate(
        {
            "model": "https://co.yes.vg/v1",
            "openai_compat": {
                "provider": "openai_compat",
                "base_url": "https://co.yes.vg/v1",
                "model": "gpt-5.4",
            },
            "model_setup": {
                "default_provider": "ollama",
                "default_model": "llama3.2",
                "default_model_spec": "ollama:llama3.2",
                "configured_models": [
                    {
                        "id": "openai_compat:https://co.yes.vg/v1:gpt-5.4",
                        "provider": "openai_compat",
                        "model": "gpt-5.4",
                        "model_spec": "https://co.yes.vg/v1",
                        "base_url": "https://co.yes.vg/v1",
                        "label": "gpt-5.4 (openai_compat)",
                        "backend_type": "openai_compat",
                    }
                ],
            },
        }
    )
    app, engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.request(
            "DELETE",
            "/v1/models/configured/openai_compat%3Ahttps%3A%2F%2Fco.yes.vg%2Fv1%3Agpt-5.4",
            json={"persist": False},
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["deleted_model_id"] == "openai_compat:https://co.yes.vg/v1:gpt-5.4"
        assert payload["available_models"] == []
        assert payload["configured_model"] == "ollama:llama3.2"
        assert any(
            item["name"] == "active_configured_model_deleted"
            and item["reason"] == "deleted_active_model_switched_to_default_model"
            and item["from"] == "openai_compat:https://co.yes.vg/v1:gpt-5.4"
            and item["to"] == "ollama:llama3.2"
            for item in payload["diagnostics"]
        )

        models_response = client.get("/v1/models")

    assert models_response.status_code == 200
    models_payload = models_response.json()
    assert models_payload["configured_model"] == "ollama:llama3.2"
    assert all(
        item["id"] != "openai_compat:https://co.yes.vg/v1:gpt-5.4"
        for item in models_payload["available_models"]
    )
    assert engine.apply_config_calls[-1] == ("ollama:llama3.2", False)


def test_models_route_does_not_duplicate_local_model_when_wsl_path_alias_matches_windows_config() -> None:
    """`GET /v1/models` 不應因 Windows/WSL 路徑別名再補一筆 local fallback。"""
    config = MochiConfig.model_validate(
        {
            "model": "/mnt/j/_models/Qwen3.5-9B",
            "model_setup": {
                "configured_models": [
                    {
                        "id": r"J:\_models\Qwen3.5-9B",
                        "provider": "local",
                        "model": "Qwen3.5-9B",
                        "model_spec": r"J:\_models\Qwen3.5-9B",
                        "label": "Qwen3.5-9B",
                        "backend_type": "safetensors",
                    }
                ]
            },
        }
    )
    app, _engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload["available_models"]) == 1
    assert payload["available_models"][0]["id"] == r"J:\_models\Qwen3.5-9B"


def test_dump_saved_configured_models_returns_only_explicit_entries() -> None:
    """edit/delete response 應只回傳實際保存的 configured models。"""
    from mochi.api.routes.models import _dump_saved_configured_models

    config = MochiConfig.model_validate(
        {
            "model": "/mnt/j/_models/Qwen3.5-9B",
            "model_setup": {
                "configured_models": [
                    {
                        "id": r"J:\_models\Qwen3.5-9B",
                        "provider": "local",
                        "model": "Qwen3.5-9B",
                        "model_spec": r"J:\_models\Qwen3.5-9B",
                        "label": "Qwen3.5-9B",
                        "backend_type": "safetensors",
                    }
                ]
            },
        }
    )

    payload = _dump_saved_configured_models(config)
    assert len(payload) == 1
    assert payload[0]["id"] == r"J:\_models\Qwen3.5-9B"


def test_models_ollama_discovery_returns_model_names(monkeypatch) -> None:
    """`GET /v1/models/ollama` 應解析 Ollama `/api/tags` model names。"""
    app, _engine = _build_app()

    class _FakeResponse:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, Any]:
            return {
                "models": [
                    {"name": "qwen2.5:latest"},
                    {"name": "llama3.2"},
                    {"name": ""},
                    {"id": "ignored"},
                ]
            }

    class _FakeAsyncClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            self.calls: list[str] = []

        async def __aenter__(self) -> _FakeAsyncClient:
            return self

        async def __aexit__(self, *args: Any) -> None:
            return None

        async def get(self, url: str) -> _FakeResponse:
            self.calls.append(url)
            return _FakeResponse()

    monkeypatch.setattr("mochi.api.routes.models.httpx.AsyncClient", _FakeAsyncClient)

    with TestClient(app) as client:
        response = client.get("/v1/models/ollama", params={"base_url": "http://localhost:11434"})

    assert response.status_code == 200
    assert response.json() == {
        "type": "ollama_models",
        "base_url": "http://localhost:11434",
        "models": ["llama3.2", "qwen2.5:latest"],
    }


def test_models_local_discovery_returns_gguf_and_hf_candidates(tmp_path: Path) -> None:
    """`GET /v1/models/local` 應回傳可辨識的本地模型候選。"""
    gguf = tmp_path / "demo.gguf"
    gguf.write_text("gguf", encoding="utf-8")
    hf_dir = tmp_path / "Qwen2.5-7B-Instruct"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    (hf_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (hf_dir / "model.safetensors").write_text("x", encoding="utf-8")

    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "local_models": {
                "roots": [str(tmp_path)],
                "scan_max_depth": 3,
                "scan_max_entries": 100,
            },
        }
    )

    with TestClient(app) as client:
        response = client.get("/v1/models/local", params={"root": str(tmp_path)})

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "local_models"
    assert payload["root"] == str(tmp_path.resolve())
    specs = {item["model_spec"] for item in payload["models"]}
    assert str(gguf.resolve()) in specs
    assert str(hf_dir.resolve()) in specs


def test_models_configure_route_supports_local_provider(tmp_path: Path) -> None:
    """`POST /v1/models/configure` 應支援 local provider。"""
    gguf = tmp_path / "demo.gguf"
    gguf.write_text("gguf", encoding="utf-8")
    app, engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "local_models": {
                "roots": [str(tmp_path)],
            },
        }
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/configure",
            json={
                "provider": "local",
                "model": str(gguf),
            },
        )

    assert response.status_code == 200
    assert engine.switch_calls == [str(gguf.resolve())]
    payload = response.json()
    assert payload["provider"] == "local"
    assert payload["api_key_configured"] is False
    assert payload["available_models"][0]["provider"] == "local"
    assert payload["available_models"][0]["model_spec"] == str(gguf.resolve())


def test_models_configure_route_returns_503_for_local_runtime_failure(tmp_path: Path) -> None:
    """local provider runtime 無法啟動時，應回傳可讀 API 錯誤而非 500。"""

    class _FailingLocalEngine(_FakeEngine):
        async def switch_model(self, model: str) -> ModelInfo:
            raise RuntimeError(
                f"Backend switch rejected unhealthy backend for '{model}': "
                "Missing dependencies: transformers, accelerate. Install with `uv sync --extra hf`."
            )

    hf_dir = tmp_path / "Qwen3.5-9B"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    (hf_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (hf_dir / "model.safetensors").write_text("x", encoding="utf-8")
    app, _engine = _build_app(engine=_FailingLocalEngine())
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "local_models": {
                "roots": [str(tmp_path)],
            },
        }
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/configure",
            json={
                "provider": "local",
                "model": str(hf_dir),
            },
        )

    assert response.status_code == 503
    assert "Missing dependencies: transformers, accelerate" in response.json()["detail"]


def test_models_switch_route_accepts_saved_local_model_entry(tmp_path: Path) -> None:
    """`/v1/models/switch` 應可切換已保存 local entry。"""
    gguf = tmp_path / "demo.gguf"
    gguf.write_text("gguf", encoding="utf-8")
    config = MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "model_setup": {
                "configured_models": [
                    {
                        "id": str(gguf.resolve()),
                        "provider": "local",
                        "model": gguf.name,
                        "model_spec": str(gguf.resolve()),
                        "label": gguf.name,
                        "backend_type": "gguf",
                    }
                ]
            },
        }
    )
    app, engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.post("/v1/models/switch", json={"model": str(gguf.resolve())})

    assert response.status_code == 200
    assert engine.switch_calls == [str(gguf.resolve())]
    assert response.json()["active_model"]["name"] == str(gguf.resolve())


def test_models_route_serializes_saved_local_entries(tmp_path: Path) -> None:
    """`GET /v1/models` 應正確序列化 provider=local 的已保存模型。"""
    gguf = tmp_path / "demo.gguf"
    gguf.write_text("gguf", encoding="utf-8")
    config = MochiConfig.model_validate(
        {
            "model": str(gguf.resolve()),
            "model_setup": {
                "configured_models": [
                    {
                        "id": str(gguf.resolve()),
                        "provider": "local",
                        "model": gguf.name,
                        "model_spec": str(gguf.resolve()),
                        "label": gguf.name,
                        "backend_type": "gguf",
                    }
                ]
            },
        }
    )
    app, _engine = _build_app()
    app.state.config_factory = lambda: config

    with TestClient(app) as client:
        response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["available_models"][0]["provider"] == "local"
    assert payload["available_models"][0]["model_spec"] == str(gguf.resolve())


def test_models_local_capabilities_returns_gguf_and_hardware_summary(tmp_path: Path) -> None:
    """`GET /v1/models/local/capabilities` 應回傳 GGUF 量化能力摘要。"""
    hf_dir = tmp_path / "Qwen2.5-7B-Instruct"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
    (hf_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (hf_dir / "model.safetensors").write_text("x", encoding="utf-8")

    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "local_models": {
                "roots": [str(tmp_path)],
            },
        }
    )

    with TestClient(app) as client:
        response = client.get(
            "/v1/models/local/capabilities",
            params={"model_spec": str(hf_dir)},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "local_model_quantization_capabilities"
    assert payload["model_spec"] == str(hf_dir.resolve())
    assert payload["model_dir"] == str(hf_dir.resolve())
    assert payload["model_family"] == "qwen2"
    by_format = {item["format_id"]: item for item in payload["formats"]}
    assert set(by_format.keys()) == {"gguf"}
    assert by_format["gguf"]["supported"] is True
    assert by_format["gguf"]["priority"] == "primary"
    assert by_format["gguf"]["suggested_default_quantization"] in {
        "Q3_K_M",
        "Q4_K_M",
        "Q5_K_M",
        "Q6_K",
        "Q8_0",
    }
    option_ids = {item["id"] for item in by_format["gguf"]["quantization_options"]}
    assert {"Q2_K", "Q3_K_M", "Q4_K_M", "Q5_K_M", "Q6_K", "Q8_0", "F16", "BF16"} <= option_ids
    assert payload["hardware"] is not None


def test_models_local_capabilities_rejects_non_hf_or_missing_paths(tmp_path: Path) -> None:
    """非 HF 目錄或不存在路徑應回傳可讀 4xx 錯誤。"""
    gguf = tmp_path / "demo.gguf"
    gguf.write_text("gguf", encoding="utf-8")
    broken_dir = tmp_path / "broken-hf"
    broken_dir.mkdir()
    (broken_dir / "config.json").write_text("{}", encoding="utf-8")

    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "local_models": {
                "roots": [str(tmp_path)],
            },
        }
    )

    with TestClient(app) as client:
        as_file = client.get("/v1/models/local/capabilities", params={"model_spec": str(gguf)})
        missing = client.get(
            "/v1/models/local/capabilities",
            params={"model_spec": str(tmp_path / "missing-model")},
        )
        broken = client.get(
            "/v1/models/local/capabilities",
            params={"model_spec": str(broken_dir)},
        )

    assert as_file.status_code == 400
    assert "HuggingFace model directories only" in as_file.json()["detail"]
    assert missing.status_code == 404
    assert "does not exist" in missing.json()["detail"]
    assert broken.status_code == 400
    assert "not a valid HuggingFace safetensors directory" in broken.json()["detail"]


def test_models_local_convert_persists_converted_gguf_to_available_models(tmp_path: Path) -> None:
    """`POST /v1/models/local/convert` 成功且 persist=true 時應寫入 configured_models。"""
    hf_dir = tmp_path / "Qwen2.5-7B-Instruct"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text('{"model_type":"qwen2"}', encoding="utf-8")
    (hf_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (hf_dir / "model.safetensors").write_text("x", encoding="utf-8")
    output_path = tmp_path / "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    config_path = tmp_path / "mochi.yaml"
    app, _engine = _build_app(config_path=config_path)
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "local_models": {
                "roots": [str(tmp_path)],
            },
        }
    )

    class _FakeConverter:
        async def convert(self, request: Any) -> LocalModelConvertExecutionResult:
            assert request.target_format == "gguf"
            assert request.quantization == "Q4_K_M"
            return LocalModelConvertExecutionResult(
                target_format="gguf",
                quantization="Q4_K_M",
                source_model_dir=str(hf_dir.resolve()),
                output_model_path=str(output_path.resolve()),
                converted=True,
                message="fake converter done",
            )

    app.state.local_model_converter = _FakeConverter()

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/local/convert",
            json={
                "source_model_dir": str(hf_dir),
                "target_format": "gguf",
                "quantization": "Q4_K_M",
                "persist": True,
            },
        )
        models_response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "local_model_convert"
    assert payload["target_format"] == "gguf"
    assert payload["quantization"] == "Q4_K_M"
    assert payload["provider"] == "local"
    assert payload["source_model_dir"] == str(hf_dir.resolve())
    assert payload["output_model_path"] == str(output_path.resolve())
    assert payload["converted"] is True
    assert payload["persisted"] is True
    assert payload["active_model"]["model_spec"] == str(output_path.resolve())
    assert payload["config_path"] == str(config_path)
    assert payload["warnings"] == []
    assert payload["saved_as_model"]["provider"] == "local"
    assert payload["saved_as_model"]["backend_type"] == "gguf"
    assert payload["saved_as_model"]["model_spec"] == str(output_path.resolve())
    assert isinstance(payload["available_models"], list)
    assert payload["available_models"][0]["model_spec"] == str(output_path.resolve())
    assert payload["available_models"][0]["backend_type"] == "gguf"
    assert models_response.status_code == 200
    assert models_response.json()["available_models"][0]["model_spec"] == str(output_path.resolve())

    saved_config = load_config(config_path)
    assert saved_config.model_setup.configured_models[0].model_spec == str(output_path.resolve())
    assert saved_config.model_setup.configured_models[0].backend_type == "gguf"


def test_models_local_convert_rejects_invalid_quantization(tmp_path: Path) -> None:
    """不支援的 GGUF 量化值應回傳 400。"""
    hf_dir = tmp_path / "Qwen2.5-7B-Instruct"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    (hf_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (hf_dir / "model.safetensors").write_text("x", encoding="utf-8")
    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "local_models": {"roots": [str(tmp_path)]},
        }
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/local/convert",
            json={
                "source_model_dir": str(hf_dir),
                "target_format": "gguf",
                "quantization": "Q9_FAKE",
                "persist": False,
            },
        )

    assert response.status_code == 400
    assert "Unsupported GGUF quantization" in response.json()["detail"]


def test_models_local_convert_rejects_non_hf_source_dir(tmp_path: Path) -> None:
    """非 HF safetensors 目錄應回傳 400。"""
    broken_dir = tmp_path / "broken-hf"
    broken_dir.mkdir()
    (broken_dir / "config.json").write_text("{}", encoding="utf-8")
    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "local_models": {"roots": [str(tmp_path)]},
        }
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/local/convert",
            json={
                "source_model_dir": str(broken_dir),
                "target_format": "gguf",
                "quantization": "Q4_K_M",
                "persist": False,
            },
        )

    assert response.status_code == 400
    assert "not a valid HuggingFace safetensors directory" in response.json()["detail"]


def test_models_local_convert_returns_503_when_converter_runtime_unavailable(tmp_path: Path) -> None:
    """預設 placeholder converter 在缺 runtime 時應回傳 503。"""
    hf_dir = tmp_path / "Qwen2.5-7B-Instruct"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    (hf_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (hf_dir / "model.safetensors").write_text("x", encoding="utf-8")
    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "local_models": {"roots": [str(tmp_path)]},
        }
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/local/convert",
            json={
                "source_model_dir": str(hf_dir),
                "target_format": "gguf",
                "quantization": "Q4_K_M",
                "persist": False,
            },
        )

    assert response.status_code == 503
    assert "runtime is unavailable" in response.json()["detail"]


def test_models_local_convert_runtime_unavailable_error_preserves_actionable_tooling_hint(
    tmp_path: Path,
) -> None:
    """runtime unavailable 錯誤應保留可操作訊息（例如缺少 llama.cpp 工具）。"""
    hf_dir = tmp_path / "Qwen2.5-7B-Instruct"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    (hf_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (hf_dir / "model.safetensors").write_text("x", encoding="utf-8")
    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "local_models": {"roots": [str(tmp_path)]},
        }
    )

    class _MissingToolConverter:
        async def convert(self, request: Any) -> LocalModelConvertExecutionResult:
            raise RuntimeError("unexpected")

    class _ActionableUnavailableConverter:
        async def convert(self, request: Any) -> LocalModelConvertExecutionResult:
            from mochi.backends.local_models import LocalModelConversionRuntimeUnavailableError

            raise LocalModelConversionRuntimeUnavailableError(
                "GGUF llama.cpp tools/runtime is unavailable: missing `llama-quantize` in PATH."
            )

    app.state.local_model_converter = _ActionableUnavailableConverter()

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/local/convert",
            json={
                "source_model_dir": str(hf_dir),
                "target_format": "gguf",
                "quantization": "Q4_K_M",
                "persist": False,
            },
        )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert "runtime is unavailable" in detail.lower()
    assert "llama-quantize" in detail


def test_models_local_convert_success_without_persist_does_not_mutate_available_models(
    tmp_path: Path,
) -> None:
    """persist=false 成功轉換時，不應寫入 configured_models/available_models。"""
    hf_dir = tmp_path / "Qwen2.5-7B-Instruct"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    (hf_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (hf_dir / "model.safetensors").write_text("x", encoding="utf-8")
    output_path = tmp_path / "Qwen2.5-7B-Instruct-F16.gguf"

    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "local_models": {"roots": [str(tmp_path)]},
            "model_setup": {
                "configured_models": [
                    {
                        "id": "ollama:qwen2.5",
                        "provider": "ollama",
                        "model": "qwen2.5",
                        "model_spec": "ollama:qwen2.5",
                        "base_url": "http://localhost:11434",
                        "label": "qwen2.5",
                        "backend_type": "ollama",
                    }
                ]
            },
        }
    )

    class _FakeConverter:
        async def convert(self, request: Any) -> LocalModelConvertExecutionResult:
            assert request.target_format == "gguf"
            assert request.quantization == "F16"
            return LocalModelConvertExecutionResult(
                target_format="gguf",
                quantization="F16",
                source_model_dir=str(hf_dir.resolve()),
                output_model_path=str(output_path.resolve()),
                converted=True,
                message="fake converter done",
            )

    app.state.local_model_converter = _FakeConverter()

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/local/convert",
            json={
                "source_model_dir": str(hf_dir),
                "target_format": "gguf",
                "quantization": "F16",
                "persist": False,
            },
        )
        models_response = client.get("/v1/models")

    assert response.status_code == 200
    payload = response.json()
    assert payload["converted"] is True
    assert payload["persisted"] is False
    assert payload["config_path"] is None
    assert payload["saved_as_model"] is None
    assert payload["available_models"] is None
    assert payload["active_model"] is None
    assert payload["output_model_path"] == str(output_path.resolve())

    assert models_response.status_code == 200
    listed = models_response.json()["available_models"]
    assert any(item["model_spec"] == "ollama:qwen2.5" for item in listed)
    assert all(item["model_spec"] != str(output_path.resolve()) for item in listed)


def test_models_local_convert_rejects_duplicate_in_progress_conversion(tmp_path: Path) -> None:
    """同一 source model 重複轉換時應回傳 409，避免共享中間檔競爭。"""
    hf_dir = tmp_path / "Qwen2.5-7B-Instruct"
    hf_dir.mkdir()
    (hf_dir / "config.json").write_text("{}", encoding="utf-8")
    (hf_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    (hf_dir / "model.safetensors").write_text("x", encoding="utf-8")

    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "local_models": {"roots": [str(tmp_path)]},
        }
    )
    app.state.local_model_conversion_in_progress = {str(hf_dir.resolve())}

    class _UnexpectedConverter:
        async def convert(self, request: Any) -> LocalModelConvertExecutionResult:
            raise AssertionError("converter should not run when the source model is already converting")

    app.state.local_model_converter = _UnexpectedConverter()

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/local/convert",
            json={
                "source_model_dir": str(hf_dir),
                "target_format": "gguf",
                "quantization": "Q4_K_M",
                "persist": False,
            },
        )

    assert response.status_code == 409
    assert "already in progress" in response.json()["detail"]


def test_models_local_runtime_status_reports_missing_runtime_actions(tmp_path: Path) -> None:
    """`GET /v1/models/local/runtime` 在未發現 runtime 時應回傳可操作狀態。"""
    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "workspace_dir": str(tmp_path / "workspace"),
            "local_models": {"roots": [str(tmp_path)]},
        }
    )

    with TestClient(app) as client:
        response = client.get("/v1/models/local/runtime")

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "local_model_runtime_status"
    assert payload["runtime"] == "llama.cpp"
    assert payload["readiness"] in {"missing", "degraded"}
    assert "register_existing_path" in payload["actions"]
    assert "prepare_managed_runtime" in payload["actions"]
    assert isinstance(payload["missing_components"], list)
    assert payload["install_dir"] == str((tmp_path / "workspace" / "runtimes" / "llama.cpp" / "b9058").resolve())


def test_models_local_runtime_status_includes_hardware_recommendation(tmp_path: Path, monkeypatch) -> None:
    """`GET /v1/models/local/runtime` should include hardware-based backend recommendation."""
    from mochi.backends.local_models import HardwareSummary

    monkeypatch.setattr(
        "mochi.api.routes.models._detect_hardware_summary",
        lambda: HardwareSummary(
            provider="torch",
            cuda_available=False,
            gpu_count=1,
            gpu_vendor="amd",
            primary_gpu_name="AMD Radeon RX 7900 XTX",
            total_vram_gb=24.0,
            recommended_runtime_backend="hip",
            recommended_runtime_label="HIP",
            warnings=[],
        ),
    )

    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "workspace_dir": str(tmp_path / "workspace"),
            "local_models": {"roots": [str(tmp_path)]},
        }
    )

    with TestClient(app) as client:
        response = client.get("/v1/models/local/runtime")

    assert response.status_code == 200
    payload = response.json()
    assert payload["hardware"] is not None
    assert payload["hardware"]["gpu_vendor"] == "amd"
    assert payload["hardware"]["recommended_runtime_backend"] == "hip"
    assert payload["hardware"]["recommended_runtime_label"] == "HIP"


def test_models_route_aligns_active_gguf_runtime_root_with_runtime_status(tmp_path: Path) -> None:
    workspace_dir = tmp_path / "workspace"
    runtime_root = workspace_dir / "runtimes" / "llama.cpp" / "b9058"
    build_bin = runtime_root / "build" / "bin"
    build_bin.mkdir(parents=True, exist_ok=True)
    (runtime_root / "convert_hf_to_gguf.py").write_text("#!/usr/bin/env python3", encoding="utf-8")
    (build_bin / "llama-quantize").write_text("bin", encoding="utf-8")
    (build_bin / "llama-server").write_text("bin", encoding="utf-8")
    model_path = tmp_path / "demo.gguf"
    model_path.write_text("gguf", encoding="utf-8")

    config = MochiConfig.model_validate(
        {
            "model": str(model_path.resolve()),
            "workspace_dir": str(workspace_dir),
            "sessions_dir": str(tmp_path / "sessions"),
            "skills_dir": str(tmp_path / "skills"),
            "plugins_dir": str(tmp_path / "plugins"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
            "local_models": {
                "roots": [str(tmp_path)],
                "llama_cpp": {
                    "source": "managed",
                    "python_executable": "/usr/bin/python3",
                    "version": "b9058",
                },
            },
        }
    )

    class _RouterBackedEngine:
        def __init__(self, runtime_config: MochiConfig) -> None:
            self._config = runtime_config
            self._router = BackendRouter(
                ollama_base_url=runtime_config.ollama.base_url,
                openai_default_model=runtime_config.openai_compat.model,
                openai_api_key="",
                gguf_config=runtime_config.gguf,
                huggingface_config=runtime_config.huggingface,
                llama_cpp_runtime=runtime_config.local_models.llama_cpp,
                workspace_dir=runtime_config.workspace_dir,
            )
            self._loaded = False

        async def get_model_info(self) -> ModelInfo:
            if not self._loaded:
                await self._router.load(self._config.model)
                self._loaded = True
            return self._router.active.get_model_info()

    app, _engine = _build_app()
    app.state.config_factory = lambda: config
    app.state.engine_factory = lambda: _RouterBackedEngine(config)

    with TestClient(app) as client:
        runtime_response = client.get("/v1/models/local/runtime")
        models_response = client.get("/v1/models")

    assert runtime_response.status_code == 200
    assert models_response.status_code == 200

    runtime_payload = runtime_response.json()
    models_payload = models_response.json()

    assert runtime_payload["readiness"] == "ready"
    assert runtime_payload["root_dir"] == str(runtime_root.resolve())
    assert models_payload["active_model"]["backend_type"] == "gguf"
    assert models_payload["active_model"]["metadata"]["runtime_root"] == str(runtime_root.resolve())


def test_models_active_local_runtime_status_reports_loaded_active_model(tmp_path: Path) -> None:
    """`GET /v1/models/local/active-runtime` 應回報目前 active local model 載入狀態。"""
    app, engine = _build_app()
    model_path = tmp_path / "demo.gguf"
    model_path.write_text("gguf", encoding="utf-8")
    engine.model_info = ModelInfo(
        name=str(model_path.resolve()),
        backend_type="gguf",
        context_length=4096,
        supports_tool_calling=False,
        metadata={
            "loaded": True,
            "idle_unloaded": False,
        },
    )

    with TestClient(app) as client:
        response = client.get("/v1/models/local/active-runtime")

    assert response.status_code == 200
    assert response.json() == {
        "type": "local_active_model_runtime_status",
        "has_active_local_model": True,
        "model_spec": str(model_path.resolve()),
        "backend_type": "gguf",
        "loaded": True,
        "idle_unloaded": False,
        "can_unload": True,
    }


def test_models_active_local_runtime_unload_unloads_current_local_model(tmp_path: Path) -> None:
    """`POST /v1/models/local/active-runtime/unload` 應釋放目前 active local model。"""
    app, engine = _build_app()
    model_path = tmp_path / "demo.gguf"
    model_path.write_text("gguf", encoding="utf-8")
    engine.model_info = ModelInfo(
        name=str(model_path.resolve()),
        backend_type="gguf",
        context_length=4096,
        supports_tool_calling=False,
        metadata={
            "loaded": True,
            "idle_unloaded": False,
        },
    )

    with TestClient(app) as client:
        response = client.post("/v1/models/local/active-runtime/unload")

    assert response.status_code == 200
    assert engine.unload_active_local_model_calls == 1
    assert response.json() == {
        "type": "local_active_model_runtime_unload",
        "unloaded": True,
        "active_runtime": {
            "type": "local_active_model_runtime_status",
            "has_active_local_model": True,
            "model_spec": str(model_path.resolve()),
            "backend_type": "gguf",
            "loaded": False,
            "idle_unloaded": False,
            "can_unload": True,
        },
    }


def test_models_local_runtime_install_prepare_managed_persists_runtime_metadata(tmp_path: Path) -> None:
    """`POST /v1/models/local/runtime/install` prepare_managed 應執行安裝並保存 metadata。"""
    config_path = tmp_path / "mochi.yaml"
    app, _engine = _build_app(config_path=config_path)
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "workspace_dir": str(tmp_path / "workspace"),
            "local_models": {"roots": [str(tmp_path)]},
        }
    )
    runtime_dir = tmp_path / "workspace" / "runtimes" / "llama.cpp" / "b9058"

    from mochi.api.routes import models as models_route

    async def _fake_install_managed_llama_cpp_runtime(**_: object) -> object:
        runtime_dir.mkdir(parents=True, exist_ok=True)
        (runtime_dir / "convert_hf_to_gguf.py").write_text("#!/usr/bin/env python3", encoding="utf-8")
        build_bin = runtime_dir / "build" / "bin"
        build_bin.mkdir(parents=True, exist_ok=True)
        (build_bin / "llama-quantize").write_text("bin", encoding="utf-8")

        class _Result:
            state = "installed"
            source = "managed"
            action = "install"
            version = "b9058"
            root_dir = str(runtime_dir.resolve())
            python_executable = "/usr/bin/python3"
            warnings: list[str] = []
            message = "Installed managed llama.cpp runtime b9058."

        return _Result()

    original_install = models_route.install_managed_llama_cpp_runtime
    models_route.install_managed_llama_cpp_runtime = _fake_install_managed_llama_cpp_runtime

    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/models/local/runtime/install",
                json={"action": "prepare_managed", "persist": True},
            )
    finally:
        models_route.install_managed_llama_cpp_runtime = original_install

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "local_model_runtime_install"
    assert payload["runtime"] == "llama.cpp"
    assert payload["action"] == "prepare_managed"
    assert payload["source"] == "managed"
    assert payload["persisted"] is True
    assert payload["config_path"] == str(config_path)
    assert payload["message"] == "Installed managed llama.cpp runtime b9058."
    assert payload["version"] == "b9058"
    assert payload["runtime_status"]["source"] == "managed"
    assert payload["runtime_status"]["readiness"] == "ready"
    assert payload["runtime_status"]["root_dir"] == str(runtime_dir.resolve())

    saved_config = load_config(config_path)
    assert saved_config.local_models.llama_cpp.source == "managed"
    assert saved_config.local_models.llama_cpp.root_dir == runtime_dir.resolve()
    assert saved_config.local_models.llama_cpp.version == "b9058"
    assert saved_config.local_models.llama_cpp.python_executable == "/usr/bin/python3"


def test_models_local_runtime_install_prepare_managed_surfaces_installer_failure(tmp_path: Path) -> None:
    """managed installer 失敗時，API 應映射成穩定 HTTP error。"""
    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "workspace_dir": str(tmp_path / "workspace"),
            "local_models": {"roots": [str(tmp_path)]},
        }
    )

    from mochi.api.routes import models as models_route
    from mochi.backends.local_models import ManagedLlamaCppInstallNetworkError

    async def _fake_install_failure(**_: object) -> object:
        raise ManagedLlamaCppInstallNetworkError("download failed")

    original_install = models_route.install_managed_llama_cpp_runtime
    models_route.install_managed_llama_cpp_runtime = _fake_install_failure

    try:
        with TestClient(app) as client:
            response = client.post(
                "/v1/models/local/runtime/install",
                json={"action": "prepare_managed", "persist": False},
            )
    finally:
        models_route.install_managed_llama_cpp_runtime = original_install

    assert response.status_code == 503
    assert response.json()["detail"] == "download failed"


def test_models_local_runtime_install_register_existing_path_persists_existing_runtime(
    tmp_path: Path,
) -> None:
    """`POST /v1/models/local/runtime/install` register_existing_path 應保存既有路徑。"""
    runtime_dir = tmp_path / "llama.cpp"
    runtime_dir.mkdir()
    (runtime_dir / "convert_hf_to_gguf.py").write_text("#!/usr/bin/env python3", encoding="utf-8")
    build_bin = runtime_dir / "build" / "bin"
    build_bin.mkdir(parents=True)
    (build_bin / "llama-quantize").write_text("bin", encoding="utf-8")

    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "workspace_dir": str(tmp_path / "workspace"),
            "local_models": {"roots": [str(tmp_path)]},
        }
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/local/runtime/install",
            json={
                "action": "register_existing_path",
                "existing_path": str(runtime_dir),
                "persist": False,
            },
        )
        status = client.get("/v1/models/local/runtime")

    assert response.status_code == 200
    payload = response.json()
    assert payload["source"] == "existing_path"
    assert payload["root_dir"] == str(runtime_dir.resolve())
    assert payload["runtime_status"]["readiness"] == "ready"
    assert payload["runtime_status"]["convert_script"] == str((runtime_dir / "convert_hf_to_gguf.py").resolve())
    assert payload["runtime_status"]["quantize_binary"] == str((build_bin / "llama-quantize").resolve())
    assert status.status_code == 200
    assert status.json()["source"] == "existing_path"
    assert status.json()["readiness"] == "ready"


def test_models_local_runtime_install_applies_updated_config_to_existing_engine(tmp_path: Path) -> None:
    """Runtime install/register should refresh the existing engine config, not only app.state.config."""
    runtime_dir = tmp_path / "llama.cpp"
    runtime_dir.mkdir()
    (runtime_dir / "convert_hf_to_gguf.py").write_text("#!/usr/bin/env python3", encoding="utf-8")
    build_bin = runtime_dir / "build" / "bin"
    build_bin.mkdir(parents=True)
    (build_bin / "llama-quantize").write_text("bin", encoding="utf-8")

    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "workspace_dir": str(tmp_path / "workspace"),
            "local_models": {"roots": [str(tmp_path)]},
        }
    )

    class _FakeEngine:
        def __init__(self) -> None:
            self.received_config: MochiConfig | None = None

        async def apply_config(self, config: MochiConfig, *, reload_voice: bool = False) -> None:
            self.received_config = config

    fake_engine = _FakeEngine()
    app.state.engine = fake_engine

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/local/runtime/install",
            json={
                "action": "register_existing_path",
                "existing_path": str(runtime_dir),
                "persist": False,
            },
        )

    assert response.status_code == 200
    assert fake_engine.received_config is not None
    assert fake_engine.received_config.local_models.llama_cpp.root_dir == runtime_dir.resolve()
    assert fake_engine.received_config.local_models.llama_cpp.source == "existing_path"


def test_models_local_runtime_install_prepare_managed_maps_ready_status_when_tools_ready(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """`prepare_managed` 成功路徑：若 runtime discovery ready，API 應映射 readiness=ready。"""
    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "workspace_dir": str(tmp_path / "workspace"),
            "local_models": {"roots": [str(tmp_path)]},
        }
    )

    runtime_dir = tmp_path / "workspace" / "runtimes" / "llama.cpp" / "b9058"

    async def _fake_install_managed_llama_cpp_runtime(**_: object) -> object:
        runtime_dir.mkdir(parents=True, exist_ok=True)
        (runtime_dir / "convert_hf_to_gguf.py").write_text("#!/usr/bin/env python3", encoding="utf-8")
        build_bin = runtime_dir / "build" / "bin"
        build_bin.mkdir(parents=True, exist_ok=True)
        (build_bin / "llama-quantize").write_text("bin", encoding="utf-8")

        class _Result:
            state = "installed"
            source = "managed"
            action = "install"
            version = "b9058"
            root_dir = str(runtime_dir.resolve())
            python_executable = "/usr/bin/python3"
            warnings: list[str] = []
            message = "Installed managed llama.cpp runtime b9058."

        return _Result()

    monkeypatch.setattr(
        "mochi.api.routes.models.install_managed_llama_cpp_runtime",
        _fake_install_managed_llama_cpp_runtime,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/local/runtime/install",
            json={"action": "prepare_managed", "persist": False},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["action"] == "prepare_managed"
    assert payload["source"] == "managed"
    assert payload["state"] == "ready"
    assert payload["runtime_status"]["readiness"] == "ready"
    assert payload["runtime_status"]["installed"] is True
    assert payload["runtime_status"]["actions"] == ["ready_for_conversion"]
    assert payload["runtime_status"]["version"] == "b9058"


def test_models_local_runtime_install_prepare_managed_failure_mapping_preserves_http_error(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """`prepare_managed` 失敗路徑：installer failure 應映射為穩定 HTTP error。"""
    app, _engine = _build_app()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "model": "ollama:configured",
            "workspace_dir": str(tmp_path / "workspace"),
            "local_models": {"roots": [str(tmp_path)]},
        }
    )

    from mochi.backends.local_models import ManagedLlamaCppInstallNetworkError

    async def _fake_install_failure(**_: object) -> object:
        raise ManagedLlamaCppInstallNetworkError("Managed installer backend unavailable.")

    monkeypatch.setattr(
        "mochi.api.routes.models.install_managed_llama_cpp_runtime",
        _fake_install_failure,
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/models/local/runtime/install",
            json={"action": "prepare_managed", "persist": False},
        )

    assert response.status_code == 503
    assert "Managed installer backend unavailable." in response.json()["detail"]
