"""AgentEngine Phase 2 整合測試。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from mochi.agents import engine as engine_module
from mochi.agents.context import ContextManager
from mochi.agents.engine import AgentEngine
from mochi.agents.events import AgentEvent, FinalAnswerEvent
from mochi.agents.invocation import AgentInvocationDiagnostics, AgentInvocationRequest
from mochi.agents.tool_exposure import ToolExposurePlan
from mochi.agents.tool_intent_router import ToolIntentRoute
from mochi.backends.base import BaseLLMBackend
from mochi.backends.openai_compat import OpenAICompatBackend
from mochi.backends.types import AttachmentRef, GenerationResult, Message, ModelInfo, ResponsesReplayState, StreamChunk
from mochi.config.schema import MochiConfig
from mochi.sessions.store import SessionStore


class FakeBackend(BaseLLMBackend):
    """測試用後端。"""

    def __init__(
        self,
        backend_type: str = "test",
        metadata: dict | None = None,
        probe_result: dict | None = None,
    ) -> None:
        self.calls: list[list[Message]] = []
        self.tool_calls_seen: list[list[str]] = []
        self.closed = False
        self.backend_type = backend_type
        self.metadata = metadata or {}
        self.probe_result = probe_result
        self.probe_calls = 0

    async def generate(
        self,
        messages: list[Message],
        tools: list | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.0,
        stream: bool = False,
    ) -> GenerationResult | AsyncIterator[StreamChunk]:
        self.calls.append(messages)
        self.tool_calls_seen.append([tool.name for tool in tools or []])
        return GenerationResult(content="fake reply")

    def supports_tool_calling(self) -> bool:
        return not (
            self.metadata.get("tool_call_mode") == "unavailable"
            or self.metadata.get("tool_calling_blocked") is True
        )

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="fake", backend_type=self.backend_type, metadata=dict(self.metadata))

    async def probe_tool_calling(self) -> dict | None:
        self.probe_calls += 1
        if self.probe_result:
            self.metadata.update(self.probe_result.get("metadata", {}))
        return self.probe_result

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_engine_preflight_probe_removes_tools_when_openai_provider_blocks_tool_protocols(
    tmp_path: Path,
) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "https://api.example.com/v1",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
        }
    )
    engine = AgentEngine(config)
    backend = FakeBackend(
        backend_type="openai_compat",
        metadata={"native_tool_calling_status": "unknown"},
        probe_result={
            "status": "all_tool_protocols_rejected_by_provider",
            "metadata": {
                "tool_call_mode": "unavailable",
                "tool_calling_blocked": True,
            },
        },
    )
    plan = ToolExposurePlan(tool_names=["web_search", "web_fetch"], matched_groups=["web"], limit=10)

    filtered = await engine._probe_tool_calling_before_exposure(backend, plan)  # noqa: SLF001

    assert backend.probe_calls == 1
    assert filtered.tool_names == []
    assert filtered.limit == 0


@pytest.mark.asyncio
async def test_engine_preflight_probe_calls_backend_probe_when_status_unknown(
    tmp_path: Path,
) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
        }
    )
    engine = AgentEngine(config)
    backend = FakeBackend(
        backend_type="ollama",
        metadata={"native_tool_calling_status": "unknown"},
        probe_result={
            "status": "supported",
            "metadata": {
                "tool_call_mode": "native",
                "native_tool_calling_status": "supported",
            },
        },
    )
    plan = ToolExposurePlan(tool_names=["web_search"], matched_groups=["web"], limit=10)

    filtered = await engine._probe_tool_calling_before_exposure(backend, plan)  # noqa: SLF001

    assert backend.probe_calls == 1
    assert filtered.tool_names == ["web_search"]


@pytest.mark.asyncio
async def test_engine_preflight_probe_retries_recoverable_fallback_state(
    tmp_path: Path,
) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
        }
    )
    engine = AgentEngine(config)
    backend = FakeBackend(
        backend_type="ollama",
        metadata={
            "tool_call_mode": "simulated_fallback",
            "native_tool_calling_status": "native_tool_calls_missing",
        },
        probe_result={
            "status": "supported",
            "metadata": {
                "tool_call_mode": "native",
                "native_tool_calling_status": "supported",
            },
        },
    )
    plan = ToolExposurePlan(tool_names=["web_search"], matched_groups=["web"], limit=10)

    filtered = await engine._probe_tool_calling_before_exposure(backend, plan)  # noqa: SLF001

    assert backend.probe_calls == 1
    assert filtered.tool_names == ["web_search"]


@pytest.mark.asyncio
async def test_engine_preflight_probe_calls_capable_backend_for_unresolved_state(
    tmp_path: Path,
) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
        }
    )
    engine = AgentEngine(config)
    backend = FakeBackend(
        backend_type="custom_backend",
        metadata={"native_tool_calling_status": "unknown"},
        probe_result={
            "status": "supported",
            "metadata": {
                "tool_call_mode": "native",
                "native_tool_calling_status": "supported",
            },
        },
    )
    plan = ToolExposurePlan(tool_names=["web_search"], matched_groups=["web"], limit=10)

    filtered = await engine._probe_tool_calling_before_exposure(backend, plan)  # noqa: SLF001

    assert backend.probe_calls == 1
    assert filtered.tool_names == ["web_search"]


@pytest.mark.asyncio
async def test_engine_preflight_probe_skips_resolved_supported_state_without_reprobe(
    tmp_path: Path,
) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
        }
    )
    engine = AgentEngine(config)
    backend = FakeBackend(
        backend_type="ollama",
        metadata={
            "tool_call_mode": "native",
            "native_tool_calling_status": "supported",
        },
        probe_result={
            "status": "supported",
            "metadata": {
                "tool_call_mode": "native",
                "native_tool_calling_status": "supported",
            },
        },
    )
    plan = ToolExposurePlan(tool_names=["web_search"], matched_groups=["web"], limit=10)

    filtered = await engine._probe_tool_calling_before_exposure(backend, plan)  # noqa: SLF001

    assert backend.probe_calls == 0
    assert filtered.tool_names == ["web_search"]


@pytest.mark.asyncio
async def test_engine_preflight_probe_skips_terminal_state_without_reprobe(
    tmp_path: Path,
) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
        }
    )
    engine = AgentEngine(config)
    backend = FakeBackend(
        backend_type="ollama",
        metadata={
            "tool_call_mode": "unavailable",
            "native_tool_calling_status": "simulated_protocol_rejected",
            "tool_calling_blocked": True,
        },
        probe_result={
            "status": "supported",
            "metadata": {
                "tool_call_mode": "native",
                "native_tool_calling_status": "supported",
            },
        },
    )
    plan = ToolExposurePlan(tool_names=["web_search"], matched_groups=["web"], limit=10)

    filtered = await engine._probe_tool_calling_before_exposure(backend, plan)  # noqa: SLF001

    assert backend.probe_calls == 0
    assert filtered.tool_names == []
    assert filtered.limit == 0


@pytest.mark.asyncio
async def test_engine_preview_and_chat_invoke_share_classifier_first_tool_intent_contract(
    tmp_path: Path,
) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
        }
    )
    engine = AgentEngine(config)
    backend = FakeBackend(backend_type="openai_compat")
    scoped_workspace = tmp_path / "scoped-workspace"
    scoped_workspace.mkdir()
    route_calls: list[bool] = []

    async def fake_load(model_spec: str) -> FakeBackend:
        engine._router._active = backend  # noqa: SLF001
        return backend

    async def fake_route(
        *,
        user_message: str,
        session_bound_workspace: bool,
        attachment_count: int = 0,
        workspace_attachment_count: int = 0,
        classifier=None,
    ) -> ToolIntentRoute:
        del user_message, session_bound_workspace, attachment_count, workspace_attachment_count
        route_calls.append(classifier is not None)
        return ToolIntentRoute(
            intent="ambiguous",
            confidence=0.0,
            source="fallback_keyword",
            rationale="test route",
        )

    engine._router.load = fake_load  # type: ignore[method-assign]
    engine._tool_intent_router.route = fake_route  # type: ignore[method-assign]

    await engine.preview_chat_context(
        "Summarize foo.py",
        session_id="preview-parity",
        workspace_dir=str(scoped_workspace),
    )
    await engine.invoke(
        AgentInvocationRequest(
            message="Summarize foo.py",
            session_id="preview-parity",
            workspace_dir=str(scoped_workspace),
            tool_mode="auto",
            execution_profile="chat",
            persist_session=False,
        )
    )

    assert route_calls == [True, True]
    await engine.close()


@pytest.mark.asyncio
async def test_engine_persists_and_restores_session_history(tmp_path: Path) -> None:
    """不同 AgentEngine 實例應可透過 SessionStore 還原歷史。"""
    fake_backend = FakeBackend()
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
            "security": {
                "require_approval_for_exec": False,
                "require_approval_for_file_write": False,
                "command_rules": [{"tokens": ["echo"], "decision": "allow", "match": "prefix"}],
                "max_file_write_size_mb": 1,
            },
        }
    )

    async def fake_load(model_spec: str) -> FakeBackend:
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine = AgentEngine(config)
    engine._router.load = fake_load  # type: ignore[method-assign]

    events = [event async for event in engine.chat("first turn", session_id="s1")]
    assert any(isinstance(event, FinalAnswerEvent) for event in events)
    await engine.close()

    store = SessionStore(tmp_path / "sessions")
    await store.save_event(
        "s1",
        {
            "type": "turn_event",
            "schema_version": 1,
            "turn_id": "turn-1",
            "event_id": "event-1",
            "seq": 1,
            "phase": "thinking",
            "timestamp": "2026-04-30T10:00:00+00:00",
            "payload": {"type": "thinking", "content": "should not enter prompt", "metadata": {}},
        },
    )

    restored_backend = FakeBackend()
    restored = AgentEngine(config)

    async def restored_load(model_spec: str) -> FakeBackend:
        restored._router._active = restored_backend  # noqa: SLF001
        return restored_backend

    restored._router.load = restored_load  # type: ignore[method-assign]
    restored_events = [event async for event in restored.chat("second turn", session_id="s1")]

    assert any(isinstance(event, FinalAnswerEvent) for event in restored_events)
    assert restored_backend.probe_calls == 1
    assert len(restored_backend.calls) == 2
    restored_messages = restored_backend.calls[-1]
    assert [message.content for message in restored_messages[1:3]] == [
        "first turn",
        "fake reply",
    ]
    assert all(message.content != "should not enter prompt" for message in restored_messages)

    await restored.close()


@pytest.mark.asyncio
async def test_restore_session_history_preserves_tool_messages_and_responses_replay(tmp_path: Path) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "test-model",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
        }
    )
    engine = AgentEngine(config)
    store = SessionStore(tmp_path / "sessions")

    replay_state = ResponsesReplayState(
        response_id="resp_prev",
        assistant_output_items=[
            {
                "type": "function_call",
                "call_id": "call-1",
                "name": "web_search",
                "arguments": '{"query":"Mochi"}',
            }
        ],
        continuity_mode="manual_items",
    )

    await store.save_event(
        "s-replay",
        {
            "type": "message",
            "schema_version": 1,
            "turn_id": "turn-1",
            "role": "assistant",
            "content": "",
            "thinking": "summary text",
            "tool_calls": [
                {
                    "id": "call-1",
                    "name": "web_search",
                    "arguments": {"query": "Mochi"},
                    "index": 0,
                }
            ],
            "responses_replay": replay_state.to_dict(),
            "attachments": [],
            "timestamp": "2026-06-11T10:00:00+00:00",
        },
    )
    await store.save_event(
        "s-replay",
        {
            "type": "message",
            "schema_version": 1,
            "turn_id": "turn-1",
            "role": "tool",
            "content": '{"ok": true}',
            "tool_call_id": "call-1",
            "name": "web_search",
            "attachments": [],
            "timestamp": "2026-06-11T10:00:01+00:00",
        },
    )

    context = ContextManager()
    await engine._restore_session_history("s-replay", context)  # noqa: SLF001
    history = context.get_recent_history()

    assert [message.role for message in history] == ["assistant", "tool"]
    assert history[0].responses_replay is not None
    assert history[0].responses_replay.response_id == "resp_prev"
    assert history[0].thinking == "summary text"
    assert len(history[0].tool_calls) == 1
    assert history[0].tool_calls[0].id == "call-1"
    assert history[1].content == '{"ok": true}'
    assert history[1].tool_call_id == "call-1"
    assert history[1].name == "web_search"

    await engine.close()


@pytest.mark.asyncio
async def test_engine_initializes_responses_backend_with_configured_api_key(
    tmp_path: Path,
) -> None:
    """直接以 `/v1/responses` 作為 config.model 啟動時應帶入已保存 API key。"""
    config = MochiConfig.model_validate(
        {
            "model": "https://co.yes.vg/v1/responses",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
            "openai_compat": {
                "provider": "openai_compat",
                "base_url": "https://co.yes.vg/v1/responses",
                "model": "gpt-test",
                "api_key": "sk-configured",
            },
        }
    )
    engine = AgentEngine(config)

    await engine.initialize()

    backend = engine._router.active  # noqa: SLF001
    assert isinstance(backend, OpenAICompatBackend)
    assert backend.base_url == "https://co.yes.vg/v1/responses"
    assert backend.api_key == "sk-configured"

    await engine.close()


@pytest.mark.asyncio
async def test_engine_switch_openai_compat_backend_accepts_vllm_provider(
    tmp_path: Path,
) -> None:
    """`switch_openai_compat_backend` 應接受 provider=vllm 並更新 config。"""
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
        }
    )
    engine = AgentEngine(config)

    class _SwitchedBackend:
        def get_model_info(self) -> ModelInfo:
            return ModelInfo(
                name="qwen2.5-7b-instruct",
                backend_type="openai_compat",
                supports_tool_calling=True,
            )

    async def fake_switch_openai_compat(
        *,
        base_url: str,
        model: str,
        api_key: str,
        provider: str,
    ) -> _SwitchedBackend:
        assert base_url == "http://localhost:8000/v1"
        assert model == "qwen2.5-7b-instruct"
        assert api_key == "vllm-key"
        assert provider == "vllm"
        return _SwitchedBackend()

    engine._router.switch_openai_compat = fake_switch_openai_compat  # type: ignore[method-assign]

    model_info = await engine.switch_openai_compat_backend(
        base_url="http://localhost:8000/v1",
        model="qwen2.5-7b-instruct",
        api_key="vllm-key",
        provider="vllm",
    )

    assert model_info.name == "qwen2.5-7b-instruct"
    assert model_info.backend_type == "openai_compat"
    assert engine._config.model == "http://localhost:8000/v1"  # noqa: SLF001
    assert engine._config.openai_compat.base_url == "http://localhost:8000/v1"  # noqa: SLF001
    assert engine._config.openai_compat.model == "qwen2.5-7b-instruct"  # noqa: SLF001
    assert engine._config.openai_compat.provider == "vllm"  # noqa: SLF001
    assert engine._config.openai_compat.api_key is not None  # noqa: SLF001
    assert engine._config.openai_compat.api_key.get_secret_value() == "vllm-key"  # noqa: SLF001


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider", "base_url", "model_name", "api_key"),
    [
        ("sglang", "http://localhost:30000/v1", "Qwen/Qwen2.5-7B-Instruct", "sglang-key"),
        ("tensorrt_llm", "http://localhost:8000/v1", "meta/llama-3.1-8b-instruct", "trtllm-key"),
    ],
)
async def test_engine_switch_openai_compat_backend_accepts_external_provider_presets(
    provider: str,
    base_url: str,
    model_name: str,
    api_key: str,
    tmp_path: Path,
) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
        }
    )
    engine = AgentEngine(config)

    class _SwitchedBackend:
        def get_model_info(self) -> ModelInfo:
            return ModelInfo(
                name=model_name,
                backend_type="openai_compat",
                supports_tool_calling=True,
            )

    async def fake_switch_openai_compat(
        *,
        base_url: str,
        model: str,
        api_key: str,
        provider: str,
    ) -> _SwitchedBackend:
        assert base_url == base_url_expected
        assert model == model_name
        assert api_key == api_key_expected
        assert provider == provider_expected
        return _SwitchedBackend()

    provider_expected = provider
    base_url_expected = base_url
    api_key_expected = api_key
    engine._router.switch_openai_compat = fake_switch_openai_compat  # type: ignore[method-assign]

    model_info = await engine.switch_openai_compat_backend(
        base_url=base_url,
        model=model_name,
        api_key=api_key,
        provider=provider,  # type: ignore[arg-type]
    )

    assert model_info.name == model_name
    assert model_info.backend_type == "openai_compat"
    assert engine._config.model == base_url  # noqa: SLF001
    assert engine._config.openai_compat.base_url == base_url  # noqa: SLF001
    assert engine._config.openai_compat.model == model_name  # noqa: SLF001
    assert engine._config.openai_compat.provider == provider  # noqa: SLF001
    assert engine._config.openai_compat.api_key is not None  # noqa: SLF001
    assert engine._config.openai_compat.api_key.get_secret_value() == api_key  # noqa: SLF001


@pytest.mark.asyncio
async def test_engine_weather_prompt_exposes_only_web_subset_for_local_backend(
    tmp_path: Path,
) -> None:
    """Weather queries should expose only the web tool subset, capped for local backends."""
    fake_backend = FakeBackend()

    async def fake_health_check() -> bool:
        return True

    fake_backend.health_check = fake_health_check  # type: ignore[method-assign]
    fake_backend.get_model_info = lambda: ModelInfo(  # type: ignore[method-assign]
        name="local.gguf",
        backend_type="gguf",
        supports_tool_calling=True,
    )

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
        }
    )
    engine = AgentEngine(config)

    async def fake_load(model_spec: str) -> FakeBackend:
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine._router.load = fake_load  # type: ignore[method-assign]

    _ = [event async for event in engine.chat("請幫我查詢今天台中天氣", session_id="s1")]

    exposed = fake_backend.tool_calls_seen[-1]
    assert {"web_search", "web_fetch", "get_current_time"} <= set(exposed)
    assert "file_read" not in exposed
    assert len(exposed) <= 6

    await engine.close()


@pytest.mark.asyncio
async def test_engine_coding_prompt_exposes_workspace_subset(
    tmp_path: Path,
) -> None:
    """Coding prompts should expose workspace tools instead of the full registry."""
    fake_backend = FakeBackend()
    fake_backend.get_model_info = lambda: ModelInfo(  # type: ignore[method-assign]
        name="remote-model",
        backend_type="openai_compat",
        supports_tool_calling=True,
    )

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
            "security": {
                "require_approval_for_exec": False,
                "require_approval_for_file_write": False,
                "command_rules": [{"tokens": ["echo"], "decision": "allow", "match": "prefix"}],
                "max_file_write_size_mb": 1,
            },
        }
    )
    engine = AgentEngine(config)

    async def fake_load(model_spec: str) -> FakeBackend:
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine._router.load = fake_load  # type: ignore[method-assign]

    _ = [event async for event in engine.chat("請幫我 debug 這個 repo 的 test failure", session_id="s1")]

    exposed = fake_backend.tool_calls_seen[-1]
    assert {"file_read", "glob_search", "grep_search"} <= set(exposed)
    assert "web_search" not in exposed

    await engine.close()


@pytest.mark.asyncio
async def test_engine_chinese_workspace_prompt_exposes_workspace_read_baseline(
    tmp_path: Path,
) -> None:
    fake_backend = FakeBackend()
    fake_backend.get_model_info = lambda: ModelInfo(  # type: ignore[method-assign]
        name="remote-model",
        backend_type="openai_compat",
        supports_tool_calling=True,
    )

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
        }
    )
    engine = AgentEngine(config)

    async def fake_load(model_spec: str) -> FakeBackend:
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine._router.load = fake_load  # type: ignore[method-assign]

    _ = [
        event
        async for event in engine.chat(
            "請檢查這個工作區，找出包含 TODO 的地方，並查看相關內容",
            session_id="workspace-zh",
            workspace_dir=str(tmp_path / "scoped-workspace"),
        )
    ]

    exposed = fake_backend.tool_calls_seen[-1]
    assert {"file_read", "glob_search", "grep_search"} <= set(exposed)

    await engine.close()


@pytest.mark.asyncio
async def test_engine_attachment_prompt_context_distinguishes_attachment_sources(
    tmp_path: Path,
) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
        }
    )
    engine = AgentEngine(config)

    attachments = [
        AttachmentRef(
            name="brief.md",
            path=str(tmp_path / "brief.md"),
            source="upload",
        ),
        AttachmentRef(
            name="app.py",
            path=str(tmp_path / "app.py"),
            source="workspace_file",
        ),
        AttachmentRef(
            name="engine.py",
            path=str(tmp_path / "engine.py"),
            source="workspace_selection",
            line_start=10,
            line_end=14,
            quote="def execute(...):",
            note="Investigate this branch.",
        ),
        AttachmentRef(
            name="error.png",
            path=str(tmp_path / "error.png"),
            source="image",
        ),
    ]

    planner_message = engine._build_tool_planner_message("debug this flow", attachments)  # noqa: SLF001
    prompt_context = engine._build_attachment_prompt_context(  # noqa: SLF001
        attachments=attachments,
        available_tool_names=["file_read", "image_view"],
    )

    assert "[upload]" in planner_message
    assert "[workspace file]" in planner_message
    assert "[workspace selection]" in planner_message
    assert "[image]" in planner_message
    assert "lines 10-14" in planner_message

    assert "uploads, workspace references, selections, or images" in prompt_context
    assert "quote: \"def execute(...):\"" in prompt_context
    assert "note: Investigate this branch." in prompt_context
    assert "[workspace selection]" in prompt_context
    assert "[image]" in prompt_context

    await engine.close()


@pytest.mark.asyncio
async def test_engine_invoke_exposes_tool_exposure_metadata_from_final_plan(
    tmp_path: Path,
) -> None:
    fake_backend = FakeBackend()
    fake_backend.get_model_info = lambda: ModelInfo(  # type: ignore[method-assign]
        name="remote-model",
        backend_type="openai_compat",
        supports_tool_calling=True,
    )

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
        }
    )
    engine = AgentEngine(config)

    async def fake_load(model_spec: str) -> FakeBackend:
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine._router.load = fake_load  # type: ignore[method-assign]

    result = await engine.invoke(
        AgentInvocationRequest(
            message="請檢查這個工作區，找出包含 TODO 的地方，並查看相關內容",
            session_id="diagnostics-worker",
            workspace_dir=str(tmp_path / "scoped-workspace"),
            attachments=[
                AttachmentRef(
                    name="brief.md",
                    path=str(tmp_path / "brief.md"),
                    source="workspace_file",
                ),
                AttachmentRef(
                    name="notes.pdf",
                    path=str(tmp_path / "notes.pdf"),
                    source="workspace_selection",
                ),
            ],
            tool_mode="auto",
            execution_profile="chat",
            persist_session=False,
        )
    )

    tool_exposure = result.diagnostics.to_dict()["tool_exposure"]
    assert tool_exposure["exposed_tools"] == result.diagnostics.exposed_tools
    assert tool_exposure["workspace_bound"] is True
    assert tool_exposure["attachment_count"] == 2
    assert tool_exposure["intent_route"]["intent"] == "workspace_read"
    assert tool_exposure["intent_route"]["source"] == "fallback_keyword"

    await engine.close()


def test_agent_invocation_diagnostics_to_dict_serializes_tool_exposure() -> None:
    diagnostics = AgentInvocationDiagnostics(
        execution_profile="chat",
        tool_mode="auto",
        exposed_tools=["file_read"],
        matched_tool_groups=["workspace"],
        tool_exposure={
            "exposed_tools": ["file_read"],
            "workspace_bound": True,
            "attachment_count": 2,
        },
    )

    assert diagnostics.to_dict()["tool_exposure"] == diagnostics.tool_exposure


@pytest.mark.asyncio
async def test_engine_invoke_exposes_diagnostics_and_honors_disabled_tool_mode(
    tmp_path: Path,
) -> None:
    fake_backend = FakeBackend()
    fake_backend.get_model_info = lambda: ModelInfo(  # type: ignore[method-assign]
        name="remote-model",
        backend_type="openai_compat",
        supports_tool_calling=True,
    )

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
        }
    )
    engine = AgentEngine(config)

    async def fake_load(model_spec: str) -> FakeBackend:
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine._router.load = fake_load  # type: ignore[method-assign]
    started_trajectories: list[str] = []
    engine._start_trajectory = lambda message: started_trajectories.append(message) or "traj"  # type: ignore[method-assign]  # noqa: SLF001

    result = await engine.invoke(
        AgentInvocationRequest(
            message="review the attached design note",
            session_id="worker-a",
            tool_mode="disabled",
            execution_profile="subagent_readonly",
            system_prompt_addendum="Role identity: Reviewer",
            persist_session=False,
        )
    )

    assert result.content == "fake reply"
    assert result.diagnostics.execution_profile == "subagent_readonly"
    assert result.diagnostics.tool_mode == "disabled"
    assert result.diagnostics.exposed_tools == []
    assert fake_backend.tool_calls_seen[-1] == []
    assert started_trajectories == []

    await engine.close()


@pytest.mark.asyncio
async def test_engine_subagent_research_profile_keeps_tools_read_only(
    tmp_path: Path,
) -> None:
    fake_backend = FakeBackend()
    fake_backend.get_model_info = lambda: ModelInfo(  # type: ignore[method-assign]
        name="remote-model",
        backend_type="openai_compat",
        supports_tool_calling=True,
    )

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
            "security": {"autonomy_mode": "auto_review"},
        }
    )
    engine = AgentEngine(config)

    async def fake_load(model_spec: str) -> FakeBackend:
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine._router.load = fake_load  # type: ignore[method-assign]

    result = await engine.invoke(
        AgentInvocationRequest(
            message="search repo files, inspect csv data, then run training with exec_command and execute code",
            session_id="research-worker",
            tool_mode="auto",
            execution_profile="subagent_research",
            persist_session=False,
        )
    )

    risky = {
        "exec_command",
        "execute_code",
        "execute_code_v2",
        "file_write",
        "file_edit",
        "write_stdin",
        "kill_session",
        "process_stop",
        "mcp_call",
    }
    assert not (set(result.diagnostics.exposed_tools) & risky)
    assert not (set(fake_backend.tool_calls_seen[-1]) & risky)
    assert {"grep_search", "csv_read"} & set(result.diagnostics.exposed_tools)

    await engine.close()


@pytest.mark.asyncio
async def test_engine_controlled_execution_profiles_gate_risky_tools(
    tmp_path: Path,
) -> None:
    fake_backend = FakeBackend()
    fake_backend.get_model_info = lambda: ModelInfo(  # type: ignore[method-assign]
        name="remote-model",
        backend_type="openai_compat",
        supports_tool_calling=True,
    )

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
            "security": {"autonomy_mode": "auto_review"},
        }
    )
    engine = AgentEngine(config)

    async def fake_load(model_spec: str) -> FakeBackend:
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine._router.load = fake_load  # type: ignore[method-assign]

    executor_result = await engine.invoke(
        AgentInvocationRequest(
            message="delegate subagent execution, inspect files, run command, and write results",
            session_id="controlled-executor",
            tool_mode="auto",
            execution_profile="subagent_execution_request",
            persist_session=False,
        )
    )
    executor_tools = set(executor_result.diagnostics.exposed_tools)
    assert "exec_command" not in executor_tools
    assert "file_write" not in executor_tools
    assert "delegate_subagent_task" not in executor_tools

    controller_result = await engine.invoke(
        AgentInvocationRequest(
            message="review command and run command in background",
            session_id="controlled-controller",
            tool_mode="auto",
            execution_profile="controller_exec",
            persist_session=False,
        )
    )
    controller_tools = set(controller_result.diagnostics.exposed_tools)
    assert "exec_command" in controller_tools
    assert "process_poll" in controller_tools
    assert "file_write" not in controller_tools

    await engine.close()


@pytest.mark.asyncio
async def test_engine_restricted_profiles_use_hard_readonly_allowlists(
    tmp_path: Path,
) -> None:
    fake_backend = FakeBackend()
    fake_backend.get_model_info = lambda: ModelInfo(  # type: ignore[method-assign]
        name="remote-model",
        backend_type="openai_compat",
        supports_tool_calling=True,
    )

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
            "security": {"autonomy_mode": "auto_review"},
        }
    )
    engine = AgentEngine(config)

    async def fake_load(model_spec: str) -> FakeBackend:
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine._router.load = fake_load  # type: ignore[method-assign]

    blocked_memory_tools = {"memory_save", "memory_update", "memory_delete"}
    for profile in ("subagent_readonly", "judge", "verifier"):
        result = await engine.invoke(
            AgentInvocationRequest(
                message="search memory, remember this, update memory, delete memory",
                session_id=f"profile-{profile}",
                tool_mode="auto",
                execution_profile=profile,  # type: ignore[arg-type]
                persist_session=False,
            )
        )
        exposed = set(result.diagnostics.exposed_tools)
        assert "memory_search" in exposed
        assert not (exposed & blocked_memory_tools)

    await engine.close()


@pytest.mark.asyncio
async def test_engine_invocation_tool_overrides_are_limited_by_profile(
    tmp_path: Path,
) -> None:
    fake_backend = FakeBackend()
    fake_backend.get_model_info = lambda: ModelInfo(  # type: ignore[method-assign]
        name="remote-model",
        backend_type="openai_compat",
        supports_tool_calling=True,
    )

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
            "security": {"autonomy_mode": "auto_review"},
        }
    )
    engine = AgentEngine(config)

    async def fake_load(model_spec: str) -> FakeBackend:
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine._router.load = fake_load  # type: ignore[method-assign]

    result = await engine.invoke(
        AgentInvocationRequest(
            message="search memory and files",
            session_id="override-worker",
            tool_mode="auto",
            execution_profile="subagent_readonly",
            tool_names_override=[
                "file_read",
                "memory_save",
                "web_search",
                "exec_command",
                "file_read",
            ],
            tool_allowlist=["file_read", "memory_save", "web_search", "exec_command"],
            tool_denylist=["web_search"],
            persist_session=False,
        )
    )

    assert result.diagnostics.exposed_tools == ["file_read"]
    assert fake_backend.tool_calls_seen[-1] == ["file_read"]

    await engine.close()


@pytest.mark.asyncio
async def test_engine_invocation_max_iterations_override_reaches_react_loop(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_backend = FakeBackend()
    seen_iterations: list[int] = []

    class SpyReActLoop:
        def __init__(self, *args: object, max_iterations: int = 10, **kwargs: object) -> None:
            del args, kwargs
            seen_iterations.append(max_iterations)

        async def run(self, *args: object, **kwargs: object) -> AsyncIterator[AgentEvent]:
            del args, kwargs
            yield FinalAnswerEvent(content="spy reply")

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
        }
    )
    engine = AgentEngine(config)

    async def fake_load(model_spec: str) -> FakeBackend:
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine._router.load = fake_load  # type: ignore[method-assign]
    monkeypatch.setattr(engine_module, "AsyncReActLoop", SpyReActLoop)

    result = await engine.invoke(
        AgentInvocationRequest(
            message="quick review",
            session_id="iteration-worker",
            tool_mode="disabled",
            execution_profile="subagent_readonly",
            max_iterations_override=2,
            persist_session=False,
        )
    )

    assert result.content == "spy reply"
    assert seen_iterations == [2]

    await engine.close()


@pytest.mark.asyncio
async def test_engine_uses_higher_default_max_iterations_for_local_backends(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_backend = FakeBackend(backend_type="ollama")
    seen_iterations: list[int] = []

    class SpyReActLoop:
        def __init__(self, *args: object, max_iterations: int = 10, **kwargs: object) -> None:
            del args, kwargs
            seen_iterations.append(max_iterations)

        async def run(self, *args: object, **kwargs: object) -> AsyncIterator[AgentEvent]:
            del args, kwargs
            yield FinalAnswerEvent(content="spy reply")

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
            "agent": {"max_react_iterations": 10},
        }
    )
    engine = AgentEngine(config)

    async def fake_load(model_spec: str) -> FakeBackend:
        engine._router._active = fake_backend  # noqa: SLF001
        return fake_backend

    engine._router.load = fake_load  # type: ignore[method-assign]
    monkeypatch.setattr(engine_module, "AsyncReActLoop", SpyReActLoop)

    result = await engine.invoke(
        AgentInvocationRequest(
            message="quick review",
            session_id="iteration-worker",
            tool_mode="disabled",
            execution_profile="subagent_readonly",
            persist_session=False,
        )
    )

    assert result.content == "spy reply"
    assert seen_iterations == [15]

    await engine.close()
