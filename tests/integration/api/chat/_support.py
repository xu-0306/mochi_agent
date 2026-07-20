# ruff: noqa: F401
"""Phase 6A chat/models API routes tests."""

from __future__ import annotations

import asyncio
import base64
import contextlib
import json
import os
import threading
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr
from starlette.requests import Request as StarletteRequest

from mochi.agents.engine import (
    AgentEngine,
    _build_response_language_prompt_addendum,
    _merge_prompt_addenda,
)
from mochi.agents.events import (
    FinalAnswerEvent,
    ThinkingEvent,
    ToolCallRequestEvent,
    ToolCallResultEvent,
)
from mochi.agents.invocation import (
    AgentInvocationDiagnostics,
    AgentInvocationRequest,
    AgentInvocationResult,
)
from mochi.agents.multi_agent.orchestrator import (
    MultiAgentOrchestrator,
    MultiAgentRunEvent,
    MultiAgentRunResult,
)
from mochi.api.routes.chat import _stream_chat_events
from mochi.api.server import create_app
from mochi.auth.models import OpenAICodexAuthProfile
from mochi.auth.openai_codex import (
    OPENAI_CODEX_REFRESH_LOCK_STALE_SECONDS,
    OpenAICodexAuthService,
    _profile_refresh_lock_path,
)
from mochi.backends.local_models import LocalModelConvertExecutionResult
from mochi.backends.router import BackendRouter
from mochi.backends.types import AttachmentRef, GenerationResult, ModelInfo
from mochi.config.manager import load_config
from mochi.config.schema import MochiConfig
from mochi.runtime.service import RuntimeService
from mochi.runtime.store import RuntimeStore
from mochi.sessions.store import SessionStore
from mochi.tools.base import (
    ActiveToolController,
    RunCancellationContext,
    ToolCancellationResult,
    cancel_asyncio_task,
)


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

def _build_app(
    *,
    engine: _FakeEngine | None = None,
    config_path: Path | None = None,
    vllm_runtime_manager: Any | None = None,
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

__all__ = [name for name in globals() if not name.startswith('__')]
