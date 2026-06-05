"""Backend router for configured model specs."""

from __future__ import annotations

import asyncio
from pathlib import Path

try:
    from loguru import logger
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal test envs
    import logging

    logger = logging.getLogger(__name__)

from mochi.backends.base import BaseLLMBackend
from mochi.backends.gguf import GGUFBackend
from mochi.backends.llama_cpp_server import LlamaCppServerBackend
from mochi.backends.local_models import discover_llama_cpp_toolchain
from mochi.backends.ollama import OllamaBackend
from mochi.backends.openai_compat import OpenAICompatBackend
from mochi.backends.safetensors import SafetensorsBackend
from mochi.config.schema import GGUFConfig, HuggingFaceConfig, LlamaCppRuntimeConfig


class BackendRouter:
    """Resolve and manage the currently active LLM backend."""

    _SWITCH_HEALTH_CHECK_TIMEOUT_SECONDS = 5.0

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        openai_default_model: str = "auto",
        openai_api_key: str = "",
        gguf_config: GGUFConfig | None = None,
        huggingface_config: HuggingFaceConfig | None = None,
        llama_cpp_runtime: LlamaCppRuntimeConfig | None = None,
        workspace_dir: str | None = None,
        local_model_idle_unload_enabled: bool = False,
        local_model_idle_unload_seconds: int | None = None,
    ) -> None:
        self._ollama_base_url = ollama_base_url
        self._openai_default_model = openai_default_model
        self._openai_api_key = openai_api_key
        self._gguf_config = gguf_config or GGUFConfig()
        self._huggingface_config = huggingface_config or HuggingFaceConfig()
        self._llama_cpp_runtime = llama_cpp_runtime or LlamaCppRuntimeConfig()
        self._workspace_dir = workspace_dir
        self._local_model_idle_unload_enabled = local_model_idle_unload_enabled
        self._local_model_idle_unload_seconds = local_model_idle_unload_seconds
        self._active: BaseLLMBackend | None = None
        self._idle_unload_task: asyncio.Task[None] | None = None

    def apply_settings(
        self,
        *,
        ollama_base_url: str,
        openai_default_model: str,
        openai_api_key: str,
        gguf_config: GGUFConfig,
        huggingface_config: HuggingFaceConfig,
        llama_cpp_runtime: LlamaCppRuntimeConfig,
        workspace_dir: str | None,
        local_model_idle_unload_enabled: bool,
        local_model_idle_unload_seconds: int | None,
    ) -> None:
        self._ollama_base_url = ollama_base_url
        self._openai_default_model = openai_default_model
        self._openai_api_key = openai_api_key
        self._gguf_config = gguf_config
        self._huggingface_config = huggingface_config
        self._llama_cpp_runtime = llama_cpp_runtime
        self._workspace_dir = workspace_dir
        self._local_model_idle_unload_enabled = local_model_idle_unload_enabled
        self._local_model_idle_unload_seconds = local_model_idle_unload_seconds

    async def load(self, model_spec: str) -> BaseLLMBackend:
        backend = self._resolve(model_spec)
        self._cancel_idle_unload_task()
        if self._active is not None:
            await self._active.close()
        self._active = backend
        self._schedule_idle_unload_if_needed(backend)
        logger.info("Loaded backend: {} for '{}'", type(backend).__name__, model_spec)
        return backend

    async def switch(self, model_spec: str) -> BaseLLMBackend:
        backend = self._resolve(model_spec)
        return await self._switch_to_backend(backend, model_spec)

    async def switch_ollama(
        self,
        *,
        model: str,
        base_url: str | None = None,
    ) -> BaseLLMBackend:
        normalized_model = model.strip()
        if not normalized_model:
            raise ValueError("Ollama model must not be empty.")
        normalized_base_url = (base_url or self._ollama_base_url).strip().rstrip("/")
        if not normalized_base_url:
            raise ValueError("Ollama base_url must not be empty.")
        backend = OllamaBackend(model=normalized_model, base_url=normalized_base_url)
        active_backend = await self._switch_to_backend(backend, f"ollama:{normalized_model}")
        self._ollama_base_url = normalized_base_url
        return active_backend

    async def switch_openai_compat(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "",
        provider: str = "openai_compat",
    ) -> BaseLLMBackend:
        normalized_base_url = base_url.strip().rstrip("/")
        normalized_model = model.strip()
        if not normalized_base_url:
            raise ValueError("OpenAI-compatible base_url must not be empty.")
        if not normalized_model:
            raise ValueError("OpenAI-compatible model must not be empty.")
        backend = OpenAICompatBackend(
            base_url=normalized_base_url,
            model=normalized_model,
            api_key=api_key,
            provider=provider,
        )
        self._openai_default_model = normalized_model
        self._openai_api_key = api_key
        return await self._switch_to_backend(backend, normalized_base_url)

    async def acquire_temporary_backend(
        self,
        *,
        model_spec: str,
        model_name: str | None = None,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> BaseLLMBackend:
        backend = self._resolve(
            model_spec,
            model_name=model_name,
            provider=provider,
            base_url=base_url,
            api_key=api_key,
        )
        try:
            await self._ensure_backend_ready(backend, model_spec)
        except Exception:
            await backend.close()
            raise
        return backend

    async def _switch_to_backend(
        self,
        backend: BaseLLMBackend,
        model_spec: str,
    ) -> BaseLLMBackend:
        try:
            await self._ensure_backend_ready(backend, model_spec)
        except Exception:
            await backend.close()
            raise

        previous = self._active
        self._cancel_idle_unload_task()
        self._active = backend
        if previous is not None:
            try:
                await previous.close()
            except Exception as exc:
                logger.warning("Failed to close previous backend after switch: {}", exc)
        self._schedule_idle_unload_if_needed(backend)
        logger.info("Switched backend: {} for '{}'", type(backend).__name__, model_spec)
        return backend

    def _build_unhealthy_backend_error(
        self,
        backend: BaseLLMBackend,
        model_spec: str,
    ) -> RuntimeError:
        detail = ""
        try:
            metadata = backend.get_model_info().metadata
        except Exception:
            metadata = {}

        dependency_error = metadata.get("dependency_error")
        if isinstance(dependency_error, str) and dependency_error.strip():
            detail = dependency_error.strip()
        else:
            model_path = metadata.get("model_path")
            if isinstance(model_path, str) and model_path and not Path(model_path).is_file():
                detail = f"GGUF model file not found: {model_path}"
            model_dir = metadata.get("model_dir")
            if not detail and isinstance(model_dir, str) and model_dir and not Path(model_dir).is_dir():
                detail = f"Model directory not found: {model_dir}"

        if detail:
            return RuntimeError(
                f"Backend switch rejected unhealthy backend for '{model_spec}': {detail}"
            )
        return RuntimeError(
            f"Backend switch rejected unhealthy backend for '{model_spec}'."
        )

    @property
    def active(self) -> BaseLLMBackend:
        if self._active is None:
            raise RuntimeError("No backend loaded. Call load() first.")
        return self._active

    async def mark_backend_busy(self, backend: BaseLLMBackend) -> None:
        if self._active is backend:
            self._cancel_idle_unload_task()

    async def mark_backend_idle(self, backend: BaseLLMBackend) -> None:
        if self._active is backend:
            self._schedule_idle_unload_if_needed(backend)

    async def close(self) -> None:
        self._cancel_idle_unload_task()
        backend = self._active
        self._active = None
        if backend is not None:
            await backend.close()

    async def unload_active_local_model(self) -> BaseLLMBackend | None:
        backend = self._active
        if backend is None or not self._supports_idle_unload(backend):
            return None

        self._cancel_idle_unload_task()
        await backend.close()
        return backend

    def _resolve(
        self,
        model_spec: str,
        *,
        model_name: str | None = None,
        provider: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> BaseLLMBackend:
        if model_spec.startswith("ollama:"):
            resolved_model_name = (model_name or model_spec[len("ollama:"):]).strip()
            if not resolved_model_name:
                raise ValueError("Ollama model spec must be 'ollama:<model_name>'.")
            resolved_base_url = (base_url or self._ollama_base_url).strip().rstrip("/")
            return OllamaBackend(model=resolved_model_name, base_url=resolved_base_url)

        if model_spec.startswith(("http://", "https://")):
            resolved_base_url = (base_url or model_spec).strip().rstrip("/")
            resolved_model_name = (model_name or self._openai_default_model).strip()
            resolved_api_key = self._openai_api_key if api_key is None else api_key
            _ = provider
            return OpenAICompatBackend(
                base_url=resolved_base_url,
                model=resolved_model_name,
                api_key=resolved_api_key,
                provider=provider or "openai_compat",
            )

        if model_spec.lower().endswith(".gguf"):
            runtime_root = (
                str(self._llama_cpp_runtime.root_dir)
                if self._llama_cpp_runtime.root_dir is not None
                else None
            )
            if runtime_root is None:
                toolchain = discover_llama_cpp_toolchain(
                    cwd=Path.cwd(),
                    managed_root=self._workspace_dir,
                    preferred_python=self._llama_cpp_runtime.python_executable,
                    preferred_version=self._llama_cpp_runtime.version,
                    preferred_source=self._llama_cpp_runtime.source,
                )
                if toolchain.root_dir is not None:
                    runtime_root = str(toolchain.root_dir)
            logger.info(
                "Resolved GGUF backend: model_path={} runtime_root={} workspace_dir={}",
                model_spec,
                runtime_root,
                self._workspace_dir,
            )
            return LlamaCppServerBackend(
                model_path=model_spec,
                runtime_root=runtime_root,
                n_ctx=self._gguf_config.n_ctx,
                n_gpu_layers=self._gguf_config.n_gpu_layers,
                n_threads=self._gguf_config.n_threads,
                n_batch=self._gguf_config.n_batch,
                n_ubatch=self._gguf_config.n_ubatch,
                n_threads_batch=self._gguf_config.n_threads_batch,
                flash_attn=self._gguf_config.flash_attn,
                offload_kqv=self._gguf_config.offload_kqv,
                use_mmap=self._gguf_config.use_mmap,
                use_mlock=self._gguf_config.use_mlock,
            )

        if Path(model_spec).is_dir() or model_spec.endswith(("/", "\\")):
            return SafetensorsBackend(
                model_dir=model_spec,
                device=self._huggingface_config.device,
                torch_dtype=self._huggingface_config.torch_dtype,
            )

        raise ValueError(
            f"Cannot resolve model_spec '{model_spec}'. "
            "Expected formats: 'ollama:<model>', '/path/to/model.gguf', "
            "'/path/to/dir/', 'http://host/v1'"
        )

    async def _ensure_backend_ready(
        self,
        backend: BaseLLMBackend,
        model_spec: str,
    ) -> None:
        try:
            is_ready = await asyncio.wait_for(
                backend.health_check(),
                timeout=self._SWITCH_HEALTH_CHECK_TIMEOUT_SECONDS,
            )
        except TimeoutError as exc:
            raise RuntimeError(
                f"Backend switch health check timed out for '{model_spec}' "
                f"after {self._SWITCH_HEALTH_CHECK_TIMEOUT_SECONDS:.2f}s."
            ) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Backend switch health check failed for '{model_spec}': {exc}"
            ) from exc

        if not is_ready:
            raise self._build_unhealthy_backend_error(backend, model_spec)

    def _schedule_idle_unload_if_needed(self, backend: BaseLLMBackend) -> None:
        self._cancel_idle_unload_task()
        idle_seconds = self._resolve_idle_unload_seconds()
        if idle_seconds is None or not self._supports_idle_unload(backend):
            return

        self._idle_unload_task = asyncio.create_task(
            self._idle_unload_after_delay(backend, idle_seconds)
        )

    def _cancel_idle_unload_task(self) -> None:
        if self._idle_unload_task is None:
            return
        self._idle_unload_task.cancel()
        self._idle_unload_task = None

    def _resolve_idle_unload_seconds(self) -> int | None:
        if not self._local_model_idle_unload_enabled:
            return None
        if (
            isinstance(self._local_model_idle_unload_seconds, int)
            and self._local_model_idle_unload_seconds > 0
        ):
            return self._local_model_idle_unload_seconds
        return None

    def _supports_idle_unload(self, backend: BaseLLMBackend) -> bool:
        return isinstance(backend, (GGUFBackend, LlamaCppServerBackend, SafetensorsBackend))

    async def _idle_unload_after_delay(
        self,
        backend: BaseLLMBackend,
        idle_seconds: int,
    ) -> None:
        current_task = asyncio.current_task()
        try:
            await asyncio.sleep(idle_seconds)
            if self._active is not backend:
                return
            logger.info(
                "Auto-unloading idle local backend {} after {}s",
                type(backend).__name__,
                idle_seconds,
            )
            await backend.close()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Idle local backend unload failed: {}", exc)
        finally:
            if self._idle_unload_task is current_task:
                self._idle_unload_task = None
