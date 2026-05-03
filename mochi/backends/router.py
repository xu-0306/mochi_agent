"""後端路由器 — 根據 model_spec 字串自動選擇並載入正確後端。"""

from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger

from mochi.backends.base import BaseLLMBackend
from mochi.backends.gguf import GGUFBackend
from mochi.backends.ollama import OllamaBackend
from mochi.backends.openai_compat import OpenAICompatBackend
from mochi.backends.safetensors import SafetensorsBackend


class BackendRouter:
    """智慧路由器，解析 model_spec 字串並初始化對應後端。

    支援格式：
        "ollama:<model>"         → OllamaBackend
        "/path/to/model.gguf"   → GGUFBackend
        "/path/to/model_dir/"   → SafetensorsBackend
        "http://host/v1"        → OpenAICompatBackend
    """
    _SWITCH_HEALTH_CHECK_TIMEOUT_SECONDS = 5.0

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        openai_default_model: str = "auto",
    ) -> None:
        """初始化路由器。

        Args:
            ollama_base_url: Ollama 服務地址。
            openai_default_model: OpenAI-compatible API 預設模型名稱。
        """
        self._ollama_base_url = ollama_base_url
        self._openai_default_model = openai_default_model
        self._active: BaseLLMBackend | None = None

    async def load(self, model_spec: str) -> BaseLLMBackend:
        """根據 model_spec 載入並回傳後端實例。

        Args:
            model_spec: 模型規格字串。

        Returns:
            已初始化的後端實例。

        Raises:
            ValueError: 當 model_spec 格式無法識別時。
        """
        backend = self._resolve(model_spec)
        if self._active is not None:
            await self._active.close()
        self._active = backend
        logger.info(f"Loaded backend: {type(backend).__name__} for '{model_spec}'")
        return backend

    async def switch(self, model_spec: str) -> BaseLLMBackend:
        """切換到新模型，僅在健康檢查成功後才替換 active backend。

        Args:
            model_spec: 新的模型規格字串。
        """
        backend = self._resolve(model_spec)
        return await self._switch_to_backend(backend, model_spec)

    async def switch_ollama(
        self,
        *,
        model: str,
        base_url: str | None = None,
    ) -> BaseLLMBackend:
        """以指定 Ollama base URL 與模型切換後端。"""
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
    ) -> BaseLLMBackend:
        """以 OpenAI-compatible endpoint、model 與 API key 切換後端。"""
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
        )
        self._openai_default_model = normalized_model
        return await self._switch_to_backend(backend, normalized_base_url)

    async def _switch_to_backend(
        self,
        backend: BaseLLMBackend,
        model_spec: str,
    ) -> BaseLLMBackend:
        """健康檢查成功後替換 active backend。"""
        try:
            is_ready = await asyncio.wait_for(
                backend.health_check(),
                timeout=self._SWITCH_HEALTH_CHECK_TIMEOUT_SECONDS,
            )
        except TimeoutError as exc:
            await backend.close()
            raise RuntimeError(
                f"Backend switch health check timed out for '{model_spec}' "
                f"after {self._SWITCH_HEALTH_CHECK_TIMEOUT_SECONDS:.2f}s."
            ) from exc
        except Exception as exc:
            await backend.close()
            raise RuntimeError(
                f"Backend switch health check failed for '{model_spec}': {exc}"
            ) from exc

        if not is_ready:
            await backend.close()
            raise RuntimeError(
                f"Backend switch rejected unhealthy backend for '{model_spec}'."
            )

        previous = self._active
        self._active = backend
        if previous is not None:
            try:
                await previous.close()
            except Exception as exc:
                logger.warning(f"Failed to close previous backend after switch: {exc}")
        logger.info(f"Switched backend: {type(backend).__name__} for '{model_spec}'")
        return backend

    @property
    def active(self) -> BaseLLMBackend:
        """取得當前活躍的後端實例。

        Raises:
            RuntimeError: 若尚未載入任何後端。
        """
        if self._active is None:
            raise RuntimeError("No backend loaded. Call load() first.")
        return self._active

    def _resolve(self, model_spec: str) -> BaseLLMBackend:
        """解析 model_spec 並建立對應後端。"""
        if model_spec.startswith("ollama:"):
            model_name = model_spec[len("ollama:"):]
            if not model_name:
                raise ValueError("Ollama model spec must be 'ollama:<model_name>'.")
            return OllamaBackend(model=model_name, base_url=self._ollama_base_url)

        if model_spec.startswith(("http://", "https://")):
            return OpenAICompatBackend(
                base_url=model_spec.rstrip("/"),
                model=self._openai_default_model,
            )

        if model_spec.lower().endswith(".gguf"):
            return GGUFBackend(model_path=model_spec)

        if Path(model_spec).is_dir() or model_spec.endswith(("/", "\\")):
            return SafetensorsBackend(model_dir=model_spec)

        raise ValueError(
            f"Cannot resolve model_spec '{model_spec}'. "
            "Expected formats: 'ollama:<model>', '/path/to/model.gguf', "
            "'/path/to/dir/', 'http://host/v1'"
        )
