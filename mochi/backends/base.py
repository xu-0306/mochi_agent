# Inspired by zeroclaw/src/providers/traits.rs design pattern
"""Shared base contract for LLM backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from mochi.backends.types import (
    GenerationResult,
    Message,
    ModelInfo,
    StreamChunk,
    ToolSchema,
)


@dataclass
class BackendRequestError(RuntimeError):
    """Normalized backend request failure with structured diagnostics."""

    message: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        super().__init__(self.message)


class BaseLLMBackend(ABC):
    """Abstract backend surface used by the agent runtime."""

    @abstractmethod
    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.0,
        reasoning_effort: str | None = None,
        stream: bool = False,
    ) -> GenerationResult | AsyncIterator[StreamChunk]:
        """Run one generation request."""

    @abstractmethod
    def supports_tool_calling(self) -> bool:
        """Return whether the current backend instance should expose tools."""

    @abstractmethod
    def get_model_info(self) -> ModelInfo:
        """Return model metadata for diagnostics and capability checks."""

    @abstractmethod
    async def health_check(self) -> bool:
        """Return whether the backend endpoint is reachable."""

    async def probe_tool_calling(self) -> dict[str, Any] | None:
        """Optionally verify native tool calling for this backend."""
        return None

    async def close(self) -> None:
        """Release backend resources such as HTTP clients."""
        return None
