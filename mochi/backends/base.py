# Inspired by zeroclaw/src/providers/traits.rs design pattern
"""LLM 後端抽象基類。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from mochi.backends.types import (
    GenerationResult,
    Message,
    ModelInfo,
    StreamChunk,
    ToolSchema,
)


class BaseLLMBackend(ABC):
    """統一 LLM 後端介面。

    所有後端實作皆繼承此類，上層元件只依賴此抽象介面，
    不關心底層是 Ollama、GGUF 還是 OpenAI-compatible API。
    """

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
    ) -> GenerationResult | AsyncIterator[StreamChunk]:
        """執行生成推理。

        Args:
            messages: 對話訊息列表。
            tools: 可用工具定義列表，None 表示不使用工具。
            temperature: 採樣溫度（0.0–2.0）。
            max_tokens: 最大輸出 token 數。
            stream: 是否啟用串流輸出。

        Returns:
            非串流時回傳 GenerationResult；串流時回傳 AsyncIterator[StreamChunk]。
        """
        ...

    @abstractmethod
    def supports_tool_calling(self) -> bool:
        """回報此後端是否原生支援 tool calling。"""
        ...

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """取得當前模型的基本資訊。"""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """執行健康檢查，回傳後端是否可用。"""
        ...

    async def close(self) -> None:
        """釋放後端資源（如 HTTP client）。子類可選擇覆寫。"""
        return None
