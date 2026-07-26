"""AgentEngine Phase 2 整合測試。"""

from __future__ import annotations

from pathlib import Path

import pytest

from mochi.agents.conversation_resolver import (
    BoundedConversationContext,
    ConversationResolver,
    IntentInterpretation,
)
from mochi.agents.engine import AgentEngine
from mochi.agents.invocation import AgentInvocationRequest
from mochi.agents.tool_exposure import ToolExposurePlan
from mochi.backends.openai_compat import OpenAICompatBackend
from mochi.backends.types import (
    Message,
    ModelInfo,
)
from mochi.config.schema import MochiConfig
from tests.unit.engine._support import (
    FakeBackend,
)


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
async def test_engine_preflight_skips_native_probe_for_ollama_prompt_guided_default(
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
            "tool_calling_protocol": "prompt_guided",
            "native_tool_calling_status": "prompt_guided_default",
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
    stages = filtered.exposure_metadata()["diagnostics"]["stages"]
    assert stages[-1]["stage"] == "preflight"
    assert stages[-1]["action"] == "skip"
    assert stages[-1]["reason"] == "prompt_guided_ollama"
    assert stages[-1]["backend"]["metadata"]["tool_calling_protocol"] == "prompt_guided"


@pytest.mark.asyncio
async def test_engine_preflight_keeps_tools_for_ollama_prompt_guided_rejected_turn(
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
            "tool_calling_protocol": "prompt_guided",
            "native_tool_calling_status": "simulated_protocol_rejected",
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
    assert filtered.limit == 10
    stages = filtered.exposure_metadata()["diagnostics"]["stages"]
    assert stages[-1]["stage"] == "preflight"
    assert stages[-1]["action"] == "skip"
    assert stages[-1]["reason"] == "prompt_guided_ollama"
    assert stages[-1]["backend"]["metadata"]["tool_call_mode"] == "unavailable"


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
async def test_engine_preflight_probe_retries_terminal_ollama_state(
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
async def test_engine_preflight_probe_skips_terminal_non_ollama_state_without_reprobe(
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
        metadata={
            "tool_call_mode": "unavailable",
            "native_tool_calling_status": "all_tool_protocols_rejected_by_provider",
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
async def test_engine_preview_and_chat_invoke_share_turn_contract_resolver(
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
    class _Interpreter:
        def __init__(self) -> None:
            self.calls = 0

        async def interpret(
            self,
            context: BoundedConversationContext,
        ) -> IntentInterpretation:
            self.calls += 1
            return IntentInterpretation(
                current_speech_act="request_information",
                task_relation="standalone",
                objective=context.current_turn.content,
                operations=frozenset({"workspace_read"}),
                confidence=0.99,
            )

    interpreter = _Interpreter()
    engine = AgentEngine(
        config,
        conversation_resolver_factory=lambda backend: ConversationResolver(
            interpreter=interpreter
        ),
    )
    backend = FakeBackend(
        backend_type="openai_compat",
        metadata={"effective_context_length": 32768},
    )
    scoped_workspace = tmp_path / "scoped-workspace"
    scoped_workspace.mkdir()

    async def fake_load(model_spec: str) -> FakeBackend:
        engine._router._active = backend  # noqa: SLF001
        return backend

    engine._router.load = fake_load  # type: ignore[method-assign]

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

    assert interpreter.calls == 2
    await engine.close()


def test_engine_resolve_inference_params_accepts_max_output_tokens_alias(tmp_path: Path) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
        }
    )
    engine = AgentEngine(config)

    resolved = engine._resolve_inference_params(  # noqa: SLF001
        {
            "max_output_tokens": 2048,
            "reserve_output_tokens": 512,
        }
    )

    assert resolved["max_output_tokens"] == 2048
    assert resolved["max_tokens"] == 2048
    assert resolved["reserve_output_tokens"] == 512


def test_engine_resolve_inference_params_derives_auto_tokens_from_context_hint(tmp_path: Path) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
            "ollama": {"num_ctx": 262144},
        }
    )
    engine = AgentEngine(config)

    resolved = engine._resolve_inference_params(  # noqa: SLF001
        {
            "max_output_tokens": None,
            "reserve_output_tokens": None,
        }
    )

    assert resolved["max_output_tokens"] == 8192
    assert resolved["max_tokens"] == 8192
    assert resolved["reserve_output_tokens"] == 2816




def test_engine_context_hint_prefers_effective_context_metadata(tmp_path: Path) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
        }
    )
    engine = AgentEngine(config)
    engine._preinitialized_model_info_cache = ModelInfo(  # noqa: SLF001
        name="ollama:test",
        backend_type="ollama",
        provider="ollama",
        context_length=131072,
        metadata={
            "context_length_source": "api_show.model_info.llama.context_length",
            "effective_context_length": 8192,
            "effective_context_length_source": "config.num_ctx",
        },
    )

    resolved = engine._resolve_inference_params(  # noqa: SLF001
        {
            "max_output_tokens": None,
            "reserve_output_tokens": None,
        }
    )

    assert resolved["max_output_tokens"] == 2048
    assert resolved["max_tokens"] == 2048
    assert resolved["reserve_output_tokens"] == 768
    assert engine._snapshot_context_length(engine._preinitialized_model_info_cache) == 8192  # noqa: SLF001


def test_engine_resolve_inference_params_uses_conservative_auto_fallback_without_context_hint(
    tmp_path: Path,
) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "https://api.example.com/v1",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
            "openai_compat": {
                "provider": "openai_compat",
                "base_url": "https://api.example.com/v1",
                "model": "gpt-test",
            },
        }
    )
    engine = AgentEngine(config)

    resolved = engine._resolve_inference_params(  # noqa: SLF001
        {
            "max_output_tokens": None,
            "reserve_output_tokens": None,
        }
    )

    assert resolved["max_output_tokens"] == 4096
    assert resolved["max_tokens"] == 4096
    assert resolved["reserve_output_tokens"] == 1024


@pytest.mark.asyncio
async def test_engine_preview_runtime_and_backend_payload_keep_output_cap_and_reserve_separate(
    tmp_path: Path,
) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "https://api.example.com/v1",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {
                "db_path": str(tmp_path / "memory.db"),
                "max_short_term_messages": 20,
                "max_short_term_tokens": 256,
                "semantic_keep_recent_messages": 4,
            },
            "security": {
                "require_approval_for_exec": False,
                "require_approval_for_file_write": False,
            },
        }
    )
    class _Interpreter:
        async def interpret(
            self,
            context: BoundedConversationContext,
        ) -> IntentInterpretation:
            return IntentInterpretation(
                current_speech_act="request_information",
                task_relation="standalone",
                objective=context.current_turn.content,
                confidence=0.99,
            )

    engine = AgentEngine(
        config,
        conversation_resolver_factory=lambda backend: ConversationResolver(
            interpreter=_Interpreter()
        ),
    )
    backend = FakeBackend(backend_type="openai_compat", metadata={"api_mode": "responses"})
    backend.get_model_info = lambda: ModelInfo(  # type: ignore[method-assign]
        name="gpt-5.2",
        provider="openai_compat",
        backend_type="openai_compat",
        context_length=32768,
        supports_tool_calling=True,
        metadata={"api_mode": "responses"},
    )

    async def fake_load(model_spec: str) -> FakeBackend:
        del model_spec
        engine._router._active = backend  # noqa: SLF001
        return backend

    engine._router.load = fake_load  # type: ignore[method-assign]
    await engine.initialize()
    context = await engine._get_context("reserve-preview")  # noqa: SLF001
    for index in range(8):
        context.add_message(Message(role="user", content=f"user turn {index}"))

    preview = await engine.preview_chat_context(
        "please summarize the state",
        session_id="reserve-preview",
        inference_overrides={"max_tokens": 8192},
    )

    assert preview["reserved_output_tokens"] == 2816
    assert preview["compaction_triggered"] is True
    assert preview["compaction_reason"] == "token_budget"
    assert context.summary is None

    await engine.invoke(
        AgentInvocationRequest(
            message="please summarize the state",
            session_id="reserve-preview",
            inference_overrides={"max_tokens": 8192},
            persist_session=False,
        )
    )

    assert backend.generation_kwargs[-1]["max_tokens"] == 8192
    assert "Conversation summary:" in backend.calls[-1][0].content
    assert context.summary is not None

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
