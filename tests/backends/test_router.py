"""Backend router tests."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from mochi.backends.ollama import OllamaBackend
from mochi.backends.openai_compat import OpenAICompatBackend
from mochi.backends.router import BackendRouter


@pytest.mark.asyncio
async def test_backend_router_ollama() -> None:
    """BackendRouter 應能解析 ollama: 前綴並回傳 OllamaBackend。"""
    router = BackendRouter()
    with (
        patch.object(OllamaBackend, "health_check", new_callable=AsyncMock, return_value=True),
        patch.object(OllamaBackend, "prime_model_info", new_callable=AsyncMock),
    ):
        backend_inst = await router.load("ollama:qwen2.5")

    assert isinstance(backend_inst, OllamaBackend)
    assert backend_inst.model == "qwen2.5"

@pytest.mark.asyncio
async def test_backend_router_openai_compat() -> None:
    """BackendRouter 應能解析 http(s) 並回傳 OpenAICompatBackend。"""
    router = BackendRouter()
    backend = await router.load("http://localhost:8080/v1")

    assert isinstance(backend, OpenAICompatBackend)
    assert backend.base_url == "http://localhost:8080/v1"
