"""AgentEngine Phase 2 整合測試。"""

from __future__ import annotations

from collections.abc import AsyncIterator

from mochi.backends.base import BaseLLMBackend
from mochi.backends.types import (
    GenerationResult,
    Message,
    ModelInfo,
    StreamChunk,
)
from mochi.tools.base import BaseTool, ToolResult


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
        self.generation_kwargs: list[dict[str, object]] = []
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
        reasoning_effort: str | None = None,
        stream: bool = False,
    ) -> GenerationResult | AsyncIterator[StreamChunk]:
        self.calls.append(messages)
        self.tool_calls_seen.append([tool.name for tool in tools or []])
        self.generation_kwargs.append(
            {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "min_p": min_p,
                "top_k": top_k,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "repeat_penalty": repeat_penalty,
                "reasoning_effort": reasoning_effort,
                "stream": stream,
            }
        )
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


class EchoTool(BaseTool):
    @property
    def name(self) -> str:
        return "echo_tool"

    @property
    def description(self) -> str:
        return "Echo a short value."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
        }

    async def execute(self, **kwargs):  # type: ignore[no-untyped-def]
        return ToolResult(output={"echo": kwargs.get("value")})


class FileWriteProbeTool(BaseTool):
    def __init__(self, *, error: str | None = None) -> None:
        self._error = error

    @property
    def name(self) -> str:
        return "file_write"

    @property
    def description(self) -> str:
        return "Write a file in the workspace."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        }

    async def execute(self, **kwargs):  # type: ignore[no-untyped-def]
        if self._error:
            return ToolResult(
                error=self._error,
                metadata={
                    "file_changes": [{"path": kwargs.get("path", "report.md")}],
                    "change_count": 1,
                },
            )
        return ToolResult(
            output=str(kwargs.get("path", "report.md")),
            metadata={
                "file_changes": [{"path": kwargs.get("path", "report.md")}],
                "change_count": 1,
                "bytes_written": len(str(kwargs.get("content", "")).encode("utf-8")),
            },
        )
