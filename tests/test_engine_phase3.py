"""AgentEngine Phase 3 測試。"""

from __future__ import annotations

from pathlib import Path

import pytest

import mochi.agents.engine as engine_module
from mochi.agents.engine import AgentEngine
from mochi.backends.types import GenerationResult, Message, ModelInfo
from mochi.config.schema import MochiConfig
from mochi.tools.base import ToolExecutionContext, ToolResult


class _SwitchableBackend:
    """測試用可切換後端。"""

    def __init__(self, name: str) -> None:
        self.name = name

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name=self.name, backend_type="test")


@pytest.mark.asyncio
async def test_engine_switch_model_updates_config(tmp_path: Path) -> None:
    """switch_model() 應透過 router 切換並更新 config.model。"""
    config = MochiConfig.model_validate(
        {
            "model": "ollama:old",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
        }
    )
    engine = AgentEngine(config)

    switched_to: list[str] = []

    async def fake_switch(model_spec: str) -> _SwitchableBackend:
        switched_to.append(model_spec)
        return _SwitchableBackend(name=model_spec)

    engine._router.switch = fake_switch  # type: ignore[method-assign]

    info = await engine.switch_model("/models/new.gguf")

    assert switched_to == ["/models/new.gguf"]
    assert engine._config.model == "/models/new.gguf"  # noqa: SLF001
    assert info.name == "/models/new.gguf"
    assert info.backend_type == "test"


@pytest.mark.asyncio
async def test_engine_switch_model_does_not_update_config_on_failure(tmp_path: Path) -> None:
    """switch_model() 失敗時不應污染既有 config.model。"""
    config = MochiConfig.model_validate(
        {
            "model": "ollama:old",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
        }
    )
    engine = AgentEngine(config)

    async def fake_switch(model_spec: str) -> _SwitchableBackend:  # noqa: ARG001
        raise RuntimeError("Backend switch rejected unhealthy backend for '/models/bad.gguf'.")

    engine._router.switch = fake_switch  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="rejected unhealthy backend"):
        await engine.switch_model("/models/bad.gguf")

    assert engine._config.model == "ollama:old"  # noqa: SLF001


@pytest.mark.asyncio
async def test_engine_apply_config_refreshes_active_gguf_runtime_root(tmp_path: Path) -> None:
    model_path = tmp_path / "demo.gguf"
    model_path.write_text("gguf", encoding="utf-8")
    workspace_dir = tmp_path / "workspace"
    runtime_root = workspace_dir / "runtimes" / "llama.cpp" / "b9058"

    config = MochiConfig.model_validate(
        {
            "model": str(model_path),
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
    engine = AgentEngine(config)
    await engine.initialize()

    initial_info = engine.get_model_info()
    assert initial_info.metadata["runtime_root"] is None

    updated = config.model_copy(deep=True)
    updated.local_models.llama_cpp.root_dir = runtime_root.resolve()

    await engine.apply_config(updated)

    refreshed_info = engine.get_model_info()
    assert refreshed_info.metadata["runtime_root"] == str(runtime_root.resolve())


@pytest.mark.asyncio
async def test_engine_generate_with_configured_model_uses_temporary_backend_and_closes_it(
    tmp_path: Path,
) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:old",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
            "model_setup": {
                "configured_models": [
                    {
                        "id": "judge-model",
                        "provider": "openai_compat",
                        "model": "gpt-4o-mini",
                        "model_spec": "https://example.invalid/v1",
                        "base_url": "https://example.invalid/v1",
                        "label": "Judge",
                        "backend_type": "openai_compat",
                    }
                ]
            },
        }
    )
    engine = AgentEngine(config)

    class _TempBackend:
        def __init__(self) -> None:
            self.closed = False
            self.calls: list[dict[str, object]] = []

        async def generate(self, messages, **kwargs):  # type: ignore[no-untyped-def]
            self.calls.append({"messages": messages, "kwargs": kwargs})
            return GenerationResult(content='{"selected_candidate_id":"student"}', model="judge-model")

        async def close(self) -> None:
            self.closed = True

    backend = _TempBackend()
    captured: dict[str, object] = {}

    async def fake_acquire_temporary_backend(**kwargs):  # type: ignore[no-untyped-def]
        captured.update(kwargs)
        return backend

    engine._router.acquire_temporary_backend = fake_acquire_temporary_backend  # type: ignore[method-assign]

    result = await engine.generate_with_configured_model(
        model_id="judge-model",
        messages=[Message(role="user", content="Pick the best answer.")],
        temperature=0.1,
        max_tokens=256,
    )

    assert result.content == '{"selected_candidate_id":"student"}'
    assert captured == {
        "model_spec": "https://example.invalid/v1",
        "model_name": "gpt-4o-mini",
        "provider": "openai_compat",
        "base_url": "https://example.invalid/v1",
        "api_key": "",
        "auth_profile_id": None,
    }
    assert len(backend.calls) == 1
    assert backend.calls[0]["kwargs"]["temperature"] == 0.1
    assert backend.calls[0]["kwargs"]["max_tokens"] == 256
    assert backend.closed is True


@pytest.mark.asyncio
async def test_engine_collect_agent_run_evidence_uses_metadata_policy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:old",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
        }
    )
    engine = AgentEngine(config)

    class _ToolRegistryStub:
        def __init__(self) -> None:
            self.execute_calls: list[tuple[str, dict[str, object], ToolExecutionContext | None]] = []

        def get(self, name: str):  # noqa: ANN001
            return object() if name in {"web_search", "web_fetch"} else None

        async def execute(
            self,
            name: str,
            args: dict[str, object],
            *,
            context: ToolExecutionContext | None = None,
        ) -> ToolResult:
            self.execute_calls.append((name, dict(args), context))
            return ToolResult(output={"provider": "stub-search", "results": []})

    captured: dict[str, object] = {}

    async def fake_collect_evidence_packets(**kwargs):  # type: ignore[no-untyped-def]
        captured.update({key: value for key, value in kwargs.items() if key != "execute_tool"})
        execute_tool = kwargs["execute_tool"]
        captured["execute_result"] = await execute_tool("web_search", {"query": "deployment note", "top_k": 5})
        return [], {"query_count": 1, "collected_packet_count": 0, "provider_counts": {}, "queries": []}

    registry = _ToolRegistryStub()
    engine._tool_registry = registry  # type: ignore[assignment]
    monkeypatch.setattr(engine_module, "collect_evidence_packets", fake_collect_evidence_packets)

    await engine.collect_agent_run_evidence(
        queries=["deployment note"],
        metadata={
            "evaluation_policy": {
                "evidence_collection": {
                    "mode": "rag",
                    "rag_provider": "mcp_resource",
                    "rag_mcp_servers": ["docs"],
                    "enabled": True,
                    "max_results_per_query": 5,
                    "max_fetch_per_query": 1,
                    "max_content_chars": 640,
                }
            }
        },
    )

    assert captured["queries"] == ["deployment note"]
    assert captured["mode"] == "rag"
    assert captured["rag_provider"] == "mcp_resource"
    assert captured["rag_mcp_servers"] == ["docs"]
    assert captured["max_results_per_query"] == 5
    assert captured["max_fetch_per_query"] == 1
    assert captured["max_content_chars"] == 640
    execute_result = captured["execute_result"]
    assert isinstance(execute_result, ToolResult)
    assert registry.execute_calls[0][0] == "web_search"
    assert registry.execute_calls[0][1] == {"query": "deployment note", "top_k": 5}
    assert registry.execute_calls[0][2] is not None
    assert registry.execute_calls[0][2].workspace_dir == str(tmp_path)


@pytest.mark.asyncio
async def test_engine_collect_agent_run_evidence_defaults_mode_to_hybrid(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:old",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
        }
    )
    engine = AgentEngine(config)

    class _ToolRegistryStub:
        def get(self, name: str):  # noqa: ANN001
            return object() if name in {"web_search", "web_fetch"} else None

    captured: dict[str, object] = {}

    async def fake_collect_evidence_packets(**kwargs):  # type: ignore[no-untyped-def]
        captured.update({key: value for key, value in kwargs.items() if key != "execute_tool"})
        return [], {"query_count": 1, "collected_packet_count": 0, "provider_counts": {}, "queries": []}

    engine._tool_registry = _ToolRegistryStub()  # type: ignore[assignment]
    monkeypatch.setattr(engine_module, "collect_evidence_packets", fake_collect_evidence_packets)

    await engine.collect_agent_run_evidence(
        queries=["deployment note"],
        metadata={"evaluation_policy": {"evidence_collection": {"enabled": True}}},
    )

    assert captured["mode"] == "hybrid"


@pytest.mark.asyncio
async def test_engine_collect_agent_run_evidence_rag_mode_does_not_require_web_search(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:old",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
        }
    )
    engine = AgentEngine(config)

    class _ToolRegistryStub:
        def get(self, name: str):  # noqa: ANN001
            return object() if name == "memory_search" else None

        async def execute(
            self,
            name: str,
            args: dict[str, object],
            *,
            context: ToolExecutionContext | None = None,
        ) -> ToolResult:
            assert name == "memory_search"
            assert args == {"query": "deployment note", "top_k": 3}
            assert context is not None
            return ToolResult(output=[])

    async def fake_collect_evidence_packets(**kwargs):  # type: ignore[no-untyped-def]
        result = await kwargs["execute_tool"]("memory_search", {"query": "deployment note", "top_k": 3})
        assert isinstance(result, ToolResult)
        assert kwargs["search_tool"] is None
        assert kwargs["mode"] == "rag"
        return [], {"query_count": 1, "collected_packet_count": 0, "provider_counts": {}, "queries": []}

    engine._tool_registry = _ToolRegistryStub()  # type: ignore[assignment]
    monkeypatch.setattr(engine_module, "collect_evidence_packets", fake_collect_evidence_packets)

    await engine.collect_agent_run_evidence(
        queries=["deployment note"],
        metadata={"evaluation_policy": {"evidence_collection": {"enabled": True, "mode": "rag"}}},
    )


@pytest.mark.asyncio
async def test_engine_collect_agent_run_evidence_uses_metadata_workspace_scope(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    default_workspace = tmp_path / "default-workspace"
    project_workspace = tmp_path / "project-workspace"
    task_workspace = tmp_path / "task-workspace"
    project_workspace.mkdir(parents=True, exist_ok=True)
    task_workspace.mkdir(parents=True, exist_ok=True)

    config = MochiConfig.model_validate(
        {
            "model": "ollama:old",
            "workspace_dir": str(default_workspace),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
        }
    )
    engine = AgentEngine(config)

    registry_calls: list[str] = []

    class _ToolRegistryStub:
        def get(self, name: str):  # noqa: ANN001
            return object() if name in {"web_search", "web_fetch"} else None

        def list_tools(self) -> list[object]:
            return []

        async def execute(
            self,
            name: str,
            args: dict[str, object],
            *,
            context: ToolExecutionContext | None = None,
        ) -> ToolResult:
            assert name == "web_search"
            assert args == {"query": "deployment note", "top_k": 3}
            assert context is not None
            assert context.session_id == "session-evidence"
            assert context.workspace_dir == str(project_workspace)
            assert context.project_workspace == str(project_workspace)
            assert context.task_sandbox_dir == str(task_workspace)
            return ToolResult(output={"provider": "stub-search", "results": []})

    async def fake_collect_evidence_packets(**kwargs):  # type: ignore[no-untyped-def]
        await kwargs["execute_tool"]("web_search", {"query": "deployment note", "top_k": 3})
        return [], {"query_count": 1, "collected_packet_count": 0, "provider_counts": {}, "queries": []}

    def fake_create_registry(workspace_dir: str) -> _ToolRegistryStub:
        registry_calls.append(workspace_dir)
        return _ToolRegistryStub()

    monkeypatch.setattr(engine._tool_registry_factory, "create_registry", fake_create_registry)
    monkeypatch.setattr(engine_module, "collect_evidence_packets", fake_collect_evidence_packets)

    await engine.collect_agent_run_evidence(
        queries=["deployment note"],
        metadata={
            "summary": {
                "session_id": "session-evidence",
                "project_workspace_dir": str(project_workspace),
                "task_workspace_dir": str(task_workspace),
            },
            "evaluation_policy": {"evidence_collection": {"enabled": True}},
        },
    )

    assert registry_calls == [str(project_workspace)]
