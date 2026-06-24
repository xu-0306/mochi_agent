"""Phase 3：Router 與本地後端骨架測試。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mochi.backends import router as router_module
from mochi.backends.base import BaseLLMBackend
from mochi.backends.gguf import GGUFBackend
from mochi.backends.llama_cpp_server import LlamaCppServerBackend
from mochi.backends.ollama import OllamaBackend
from mochi.backends.openai_codex import OpenAICodexBackend
from mochi.backends.openai_compat import OpenAICompatBackend
from mochi.backends.router import BackendRouter
from mochi.backends.safetensors import SafetensorsBackend
from mochi.backends.types import GenerationResult, Message, ModelInfo
from mochi.config.schema import GGUFConfig, HuggingFaceConfig, LlamaCppRuntimeConfig


class _FakeBackend(BaseLLMBackend):
    """測試用 backend。"""

    def __init__(self, name: str, *, healthy: bool = True, health_error: Exception | None = None) -> None:
        self.name = name
        self.healthy = healthy
        self.health_error = health_error
        self.closed = False

    async def generate(
        self,
        messages: list[Message],
        tools=None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
    ) -> GenerationResult:
        raise NotImplementedError

    def supports_tool_calling(self) -> bool:
        return False

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name=self.name, backend_type="fake")

    async def health_check(self) -> bool:
        if self.health_error is not None:
            raise self.health_error
        return self.healthy

    async def close(self) -> None:
        self.closed = True


class _FakeLocalBackend(SafetensorsBackend):
    def __init__(self, name: str) -> None:
        super().__init__(model_dir=name)
        self.closed = False

    async def close(self) -> None:
        self.closed = True
        await super().close()


@pytest.mark.asyncio
async def test_router_resolves_ollama_backend() -> None:
    """router 應能解析 ollama: 前綴。"""
    router = BackendRouter()
    backend = await router.load("ollama:qwen2.5")
    assert isinstance(backend, OllamaBackend)
    assert backend.model == "qwen2.5"


@pytest.mark.asyncio
async def test_router_resolves_gguf_backend() -> None:
    """router 應能將 .gguf 路徑解析為 GGUFBackend。"""
    router = BackendRouter()
    backend = await router.load("/models/demo.gguf")
    assert isinstance(backend, LlamaCppServerBackend)
    assert backend.model_path == "/models/demo.gguf"


@pytest.mark.asyncio
async def test_router_resolves_safetensors_backend_for_directory(tmp_path: Path) -> None:
    """router 應能將目錄解析為 SafetensorsBackend。"""
    model_dir = tmp_path / "hf-model"
    model_dir.mkdir()

    router = BackendRouter()
    backend = await router.load(str(model_dir))
    assert isinstance(backend, SafetensorsBackend)
    assert backend.model_dir == str(model_dir)


@pytest.mark.asyncio
@pytest.mark.parametrize("base_url", ["http://api.example.com/v1", "https://api.example.com/v1/"])
async def test_router_resolves_openai_compat_backend(base_url: str) -> None:
    """router 應能解析 http(s) 為 OpenAI-compatible backend。"""
    router = BackendRouter(openai_default_model="router-default-model", openai_api_key="sk-router")
    backend = await router.load(base_url)
    assert isinstance(backend, OpenAICompatBackend)
    assert backend.base_url == base_url.rstrip("/")
    assert backend.model == "router-default-model"
    assert backend.api_key == "sk-router"


@pytest.mark.asyncio
async def test_router_resolves_responses_endpoint_with_api_key() -> None:
    """直接設定 `/v1/responses` endpoint 初始化時也應保留 API key。"""
    router = BackendRouter(openai_default_model="gpt-test", openai_api_key="sk-router")
    backend = await router.load("https://co.yes.vg/v1/responses")

    assert isinstance(backend, OpenAICompatBackend)
    assert backend.base_url == "https://co.yes.vg/v1/responses"
    assert backend.api_key == "sk-router"
    assert backend.get_model_info().metadata["api_url"] == "https://co.yes.vg/v1/responses"


@pytest.mark.asyncio
async def test_router_keeps_codex_token_separate_from_openai_compat_state() -> None:
    """OpenAI Codex token should not replace the generic OpenAI-compatible API key state."""
    router = BackendRouter(openai_default_model="compat-default", openai_api_key="sk-compat")

    await router.switch_openai_codex(
        base_url="https://chatgpt.com/backend-api",
        model="gpt-5.4",
        access_token="codex-access-token",
        auth_profile_id="openai_codex:default",
    )

    compat_backend = router._resolve(  # noqa: SLF001
        "https://api.example.com/v1",
    )
    codex_backend = router._resolve(  # noqa: SLF001
        "https://chatgpt.com/backend-api",
        provider="openai_codex",
    )

    assert isinstance(compat_backend, OpenAICompatBackend)
    assert compat_backend.api_key == "sk-compat"
    assert compat_backend.model == "compat-default"
    assert isinstance(codex_backend, OpenAICodexBackend)
    assert codex_backend.api_key == "codex-access-token"
    assert codex_backend.model == "gpt-5.4"

    await compat_backend.close()
    await codex_backend.close()


@pytest.mark.asyncio
async def test_router_rejects_unknown_spec() -> None:
    """無法識別的 model_spec 應回報 ValueError。"""
    router = BackendRouter()
    with pytest.raises(ValueError, match="Cannot resolve model_spec"):
        await router.load("invalid-model-spec")


@pytest.mark.asyncio
async def test_gguf_backend_generate_error_semantics(tmp_path: Path) -> None:
    """GGUF generate 在模型載入失敗時應回報一致錯誤碼。"""
    model_path = tmp_path / "toy.gguf"
    model_path.write_text("placeholder", encoding="utf-8")

    backend = GGUFBackend(model_path=str(model_path))
    backend._dependency_error = None
    backend._model_loader = lambda: (_ for _ in ()).throw(RuntimeError("load failed"))  # noqa: SLF001

    with pytest.raises(RuntimeError, match=r"gguf generate unavailable \[model_load_failed\]"):
        await backend.generate([Message(role="user", content="hi")])

    info = backend.get_model_info()
    assert info.backend_type == "gguf"
    assert info.metadata["model_path"] == str(model_path)
    assert info.metadata["loaded"] is False
    assert await backend.health_check() is True

    backend._model = object()
    await backend.close()
    assert backend._model is None


@pytest.mark.asyncio
async def test_safetensors_backend_generate_error_semantics(tmp_path: Path) -> None:
    """Safetensors generate 在 pipeline 載入失敗時應回報一致錯誤碼。"""
    model_dir = tmp_path / "hf-model"
    model_dir.mkdir()

    backend = SafetensorsBackend(model_dir=str(model_dir))
    backend._dependency_error = None
    backend._pipeline_factory = lambda: (_ for _ in ()).throw(RuntimeError("load failed"))  # noqa: SLF001

    with pytest.raises(
        RuntimeError,
        match=r"safetensors generate unavailable \[model_load_failed\]",
    ):
        await backend.generate([Message(role="user", content="hi")])

    info = backend.get_model_info()
    assert info.backend_type == "safetensors"
    assert info.metadata["model_dir"] == str(model_dir)
    assert info.metadata["loaded"] is False
    assert await backend.health_check() is True

    backend._pipeline = object()
    await backend.close()
    assert backend._pipeline is None


@pytest.mark.asyncio
async def test_router_switch_replaces_active_backend_and_closes_previous_backend() -> None:
    """switch 應在 health_check 成功後才替換 active backend。"""
    router = BackendRouter()
    previous = _FakeBackend("previous")
    candidate = _FakeBackend("candidate", healthy=True)
    router._active = previous  # noqa: SLF001
    router._resolve = lambda model_spec: candidate  # type: ignore[method-assign]  # noqa: ARG005, SLF001

    backend = await router.switch("ollama:new")

    assert backend is candidate
    assert router.active is candidate
    assert previous.closed is True
    assert candidate.closed is False


@pytest.mark.asyncio
async def test_router_switch_allows_unhealthy_candidate_backend() -> None:
    """switch 失敗時應保留舊 backend，並關閉候選 backend。"""
    router = BackendRouter()
    previous = _FakeBackend("previous")
    candidate = _FakeBackend("candidate", healthy=False)
    router._active = previous  # noqa: SLF001
    router._resolve = lambda model_spec: candidate  # type: ignore[method-assign]  # noqa: ARG005, SLF001

    backend = await router.switch("/models/bad.gguf")

    assert backend is candidate
    assert router.active is candidate
    assert previous.closed is True
    assert candidate.closed is False


@pytest.mark.asyncio
async def test_router_acquire_temporary_backend_reports_dependency_detail_for_unhealthy_backend() -> None:
    """unhealthy backend 若帶依賴診斷，應補進 switch 錯誤訊息。"""
    router = BackendRouter()
    candidate = SafetensorsBackend(model_dir="/models/hf-demo")
    candidate._dependency_error = "Missing dependencies: transformers. Install with `uv sync --extra hf`."  # noqa: SLF001
    candidate.closed = False  # type: ignore[attr-defined]  # noqa: SLF001

    async def _close() -> None:
        candidate.closed = True  # type: ignore[attr-defined]  # noqa: SLF001

    candidate.close = _close  # type: ignore[method-assign]
    router._resolve = lambda model_spec, **kwargs: candidate  # type: ignore[method-assign]  # noqa: ARG005, ANN003, SLF001

    with pytest.raises(RuntimeError, match=r"Missing dependencies: transformers"):
        await router.acquire_temporary_backend(model_spec="/models/hf-demo")

    assert candidate.closed is True  # type: ignore[attr-defined]  # noqa: SLF001


@pytest.mark.asyncio
async def test_router_switch_allows_candidate_when_health_check_would_raise() -> None:
    """switch health_check 例外時應保留舊 backend。"""
    router = BackendRouter()
    previous = _FakeBackend("previous")
    candidate = _FakeBackend("candidate", health_error=RuntimeError("boom"))
    router._active = previous  # noqa: SLF001
    router._resolve = lambda model_spec: candidate  # type: ignore[method-assign]  # noqa: ARG005, SLF001

    backend = await router.switch("http://bad-host/v1")

    assert backend is candidate
    assert router.active is candidate
    assert previous.closed is True
    assert candidate.closed is False


@pytest.mark.asyncio
async def test_router_switch_ollama_updates_base_url_even_when_backend_is_unhealthy(monkeypatch) -> None:
    """switch_ollama 失敗時不應污染既有 Ollama base_url。"""

    candidates: list[_FakeBackend] = []

    class _UnhealthyOllamaBackend(_FakeBackend):
        def __init__(self, model: str, base_url: str) -> None:
            super().__init__(name=model, healthy=False)
            self.model = model
            self.base_url = base_url
            candidates.append(self)

    router = BackendRouter(ollama_base_url="http://old-host:11434")
    previous = _FakeBackend("previous")
    router._active = previous  # noqa: SLF001
    monkeypatch.setattr(router_module, "OllamaBackend", _UnhealthyOllamaBackend)

    backend = await router.switch_ollama(model="qwen2.5", base_url="http://new-host:11434")

    assert backend is candidates[0]
    assert router._ollama_base_url == "http://new-host:11434"  # noqa: SLF001
    assert router.active is candidates[0]
    assert previous.closed is True
    assert candidates[0].closed is False


@pytest.mark.asyncio
async def test_router_acquire_temporary_backend_rejects_unhealthy_candidate() -> None:
    """temporary backend 仍應拒絕 health_check=False 的候選 backend。"""
    router = BackendRouter()
    candidate = _FakeBackend("candidate", healthy=False)
    router._resolve = lambda model_spec, **kwargs: candidate  # type: ignore[method-assign]  # noqa: ARG005, ANN003, SLF001

    with pytest.raises(RuntimeError, match="rejected unhealthy backend"):
        await router.acquire_temporary_backend(model_spec="ollama:temp")

    assert candidate.closed is True


@pytest.mark.asyncio
async def test_router_uses_configured_gguf_and_hf_backend_parameters(tmp_path: Path) -> None:
    """router 建立本地 backend 時應套用 config 參數。"""
    hf_dir = tmp_path / "hf-model"
    hf_dir.mkdir()
    gguf_path = tmp_path / "demo.gguf"

    router = BackendRouter(
        gguf_config=GGUFConfig(
            n_ctx=8192,
            n_gpu_layers=16,
            n_threads=6,
            n_batch=1024,
            n_ubatch=256,
            n_threads_batch=3,
            flash_attn=True,
            offload_kqv=False,
            use_mmap=False,
            use_mlock=True,
        ),
        huggingface_config=HuggingFaceConfig(device="cpu", torch_dtype="float16"),
    )

    gguf_backend = await router.load(str(gguf_path))
    assert isinstance(gguf_backend, LlamaCppServerBackend)
    assert gguf_backend.n_ctx == 8192
    assert gguf_backend.n_gpu_layers == 16
    assert gguf_backend.n_threads == 6
    assert gguf_backend.n_batch == 1024
    assert gguf_backend.n_ubatch == 256
    assert gguf_backend.n_threads_batch == 3
    assert gguf_backend.flash_attn is True
    assert gguf_backend.offload_kqv is False
    assert gguf_backend.use_mmap is False
    assert gguf_backend.use_mlock is True
    assert gguf_backend.runtime_root is None

    hf_backend = await router.load(str(hf_dir))
    assert isinstance(hf_backend, SafetensorsBackend)
    assert hf_backend.device == "cpu"
    assert hf_backend.torch_dtype == "float16"


@pytest.mark.asyncio
async def test_router_passes_configured_runtime_root_to_llama_cpp_server_backend(tmp_path: Path) -> None:
    gguf_path = tmp_path / "demo.gguf"
    runtime_root = tmp_path / "llama-runtime"
    runtime_root.mkdir()

    from mochi.config.schema import LlamaCppRuntimeConfig

    router = BackendRouter(
        gguf_config=GGUFConfig(),
        llama_cpp_runtime=LlamaCppRuntimeConfig(
            source="managed",
            root_dir=runtime_root,
            python_executable="/usr/bin/python3",
            version="b9058",
        ),
    )

    gguf_backend = await router.load(str(gguf_path))

    assert isinstance(gguf_backend, LlamaCppServerBackend)
    assert gguf_backend.runtime_root == str(runtime_root)


@pytest.mark.asyncio
async def test_router_discovers_managed_runtime_root_when_root_dir_is_missing(tmp_path: Path) -> None:
    gguf_path = tmp_path / "demo.gguf"
    gguf_path.write_text("gguf", encoding="utf-8")
    workspace_dir = tmp_path / "workspace"
    runtime_root = workspace_dir / "runtimes" / "llama.cpp" / "b9058"
    runtime_root.mkdir(parents=True)
    (runtime_root / "convert_hf_to_gguf.py").write_text("#!/usr/bin/env python3", encoding="utf-8")
    build_bin = runtime_root / "build" / "bin"
    build_bin.mkdir(parents=True)
    (build_bin / "llama-quantize").write_text("bin", encoding="utf-8")
    (build_bin / "llama-server").write_text("bin", encoding="utf-8")

    router = BackendRouter(
        gguf_config=GGUFConfig(),
        llama_cpp_runtime=LlamaCppRuntimeConfig(
            source="managed",
            root_dir=None,
            python_executable="/usr/bin/python3",
            version="b9058",
        ),
        workspace_dir=str(workspace_dir),
    )

    gguf_backend = await router.load(str(gguf_path))

    assert isinstance(gguf_backend, LlamaCppServerBackend)
    assert gguf_backend.runtime_root == str(runtime_root.resolve())


@pytest.mark.asyncio
async def test_router_keeps_runtime_root_unset_when_discovery_fails(tmp_path: Path) -> None:
    gguf_path = tmp_path / "demo.gguf"
    gguf_path.write_text("gguf", encoding="utf-8")

    router = BackendRouter(
        gguf_config=GGUFConfig(),
        llama_cpp_runtime=LlamaCppRuntimeConfig(
            source="managed",
            root_dir=None,
            python_executable="/usr/bin/python3",
            version="b9058",
        ),
        workspace_dir=str(tmp_path / "workspace"),
    )

    gguf_backend = await router.load(str(gguf_path))

    assert isinstance(gguf_backend, LlamaCppServerBackend)
    assert gguf_backend.runtime_root is None
    assert gguf_backend.get_model_info().metadata["dependency_error"].endswith(
        "runtime root is not set."
    )


@pytest.mark.asyncio
async def test_router_schedules_idle_unload_for_local_backends() -> None:
    """本地 backend 閒置一段時間後應自動 close。"""
    router = BackendRouter(
        local_model_idle_unload_enabled=True,
        local_model_idle_unload_seconds=1,
    )
    backend = _FakeLocalBackend("/models/local")

    router._active = backend  # noqa: SLF001
    router._schedule_idle_unload_if_needed(backend)  # noqa: SLF001

    await asyncio.sleep(1.3)

    assert backend.closed is True


@pytest.mark.asyncio
async def test_router_mark_backend_busy_cancels_idle_unload() -> None:
    """request 開始時應取消既有閒置卸載排程。"""
    router = BackendRouter(
        local_model_idle_unload_enabled=True,
        local_model_idle_unload_seconds=1,
    )
    backend = _FakeLocalBackend("/models/local")

    router._active = backend  # noqa: SLF001
    router._schedule_idle_unload_if_needed(backend)  # noqa: SLF001

    await router.mark_backend_busy(backend)
    await asyncio.sleep(1.3)

    assert backend.closed is False


@pytest.mark.asyncio
async def test_router_mark_backend_idle_reschedules_idle_unload() -> None:
    """request 結束後應重新安排閒置卸載。"""
    router = BackendRouter(
        local_model_idle_unload_enabled=True,
        local_model_idle_unload_seconds=1,
    )
    backend = _FakeLocalBackend("/models/local")

    router._active = backend  # noqa: SLF001

    await router.mark_backend_busy(backend)
    await router.mark_backend_idle(backend)
    await asyncio.sleep(1.3)

    assert backend.closed is True


@pytest.mark.asyncio
async def test_router_does_not_schedule_idle_unload_when_disabled() -> None:
    """停用閒置卸載時，本地 backend 在 idle 後不應自動 close。"""
    router = BackendRouter(
        local_model_idle_unload_enabled=False,
        local_model_idle_unload_seconds=1,
    )
    backend = _FakeLocalBackend("/models/local")

    router._active = backend  # noqa: SLF001
    router._schedule_idle_unload_if_needed(backend)  # noqa: SLF001

    await asyncio.sleep(1.3)

    assert backend.closed is False
