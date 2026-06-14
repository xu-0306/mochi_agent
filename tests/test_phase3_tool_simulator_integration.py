"""Phase 3：非原生 tool calling backend 整合測試。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from mochi.agents.events import ErrorEvent, FinalAnswerEvent, StatusEvent, TextChunkEvent, ThinkingEvent, ToolCallRequestEvent, ToolCallResultEvent
from mochi.agents.react_loop import AsyncReActLoop
from mochi.backends.base import BaseLLMBackend
from mochi.backends.gguf import GGUFBackend
from mochi.backends.safetensors import SafetensorsBackend
from mochi.backends.types import GenerationResult, Message, ModelInfo, StreamChunk, ToolCall, ToolSchema
from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult
from mochi.tools.file_ops import FileReadTool
from mochi.tools.registry import ToolRegistry


class _FakeGGUFModel:
    """測試用 GGUF 模型假物件。"""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict] = []

    def create_chat_completion(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)
        return {
            "choices": [
                {
                    "message": {
                        "content": self.text,
                    }
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        }


class _FakePipeline:
    """測試用 transformers pipeline 假物件。"""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls: list[dict] = []

    def __call__(self, prompt: str, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append({"prompt": prompt, **kwargs})
        return [{"generated_text": prompt + self.text}]


class _ReactiveGGUFModel:
    """會依工具結果切換回覆的 GGUF 假模型。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create_chat_completion(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)
        messages = kwargs.get("messages", [])
        has_tool_result = any(
            isinstance(message, dict)
            and (
                (
                    message.get("role") == "tool"
                    and message.get("name") == "web_search"
                )
                or (
                    message.get("role") == "user"
                    and "Tool web_search result:" in str(message.get("content", ""))
                )
            )
            for message in messages
        )
        if has_tool_result:
            return {
                "choices": [
                    {
                        "message": {
                            "content": "我已根據 web_search 的結果完成整理。",
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 18, "completion_tokens": 9},
            }

        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '<tool_call>{"name":"web_search",'
                            '"arguments":{"query":"Mochi AI"}}</tool_call>'
                        ),
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 6},
        }


class _NativeReactiveGGUFModel:
    """Allowlisted native tool-calling roundtrip model."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.chat_format = "functionary-v2"

    def create_chat_completion(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)
        messages = kwargs.get("messages", [])
        has_native_tool_result = any(
            isinstance(message, dict)
            and message.get("role") == "tool"
            and message.get("name") == "web_search"
            for message in messages
        )
        if has_native_tool_result:
            return {
                "choices": [
                    {
                        "message": {"content": "native path done"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 18, "completion_tokens": 9},
            }

        return {
            "choices": [
                {
                    "message": {
                        "content": "Need to call tool.",
                        "tool_calls": [
                            {
                                "id": "native-call-1",
                                "function": {
                                    "name": "web_search",
                                    "arguments": '{"query":"Mochi AI"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 6},
        }


class _StrictReactiveGGUFModel:
    """模擬 llama.cpp chat template 不接受 OpenAI-style tool messages。"""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create_chat_completion(self, **kwargs):  # type: ignore[no-untyped-def]
        self.calls.append(kwargs)
        messages = kwargs.get("messages", [])

        for message in messages:
            if not isinstance(message, dict):
                continue
            if message.get("role") == "tool":
                raise TypeError("Can only get item pairs from a mapping.")
            if message.get("tool_calls"):
                raise TypeError("Can only get item pairs from a mapping.")

        if any(
            isinstance(message, dict)
            and message.get("role") == "user"
            and "Tool web_search result:" in str(message.get("content", ""))
            for message in messages
        ):
            return {
                "choices": [
                    {
                        "message": {"content": "整理完工具結果，這是最終回答。"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 20, "completion_tokens": 8},
            }

        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '<tool_call>{"name":"web_search",'
                            '"arguments":{"query":"Mochi AI"}}</tool_call>'
                        )
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 6},
        }


class _ReactiveBackend(BaseLLMBackend):
    """會檢查 tool message content 的測試 backend。"""

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

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
        stream: bool = False,
    ) -> GenerationResult:
        del (
            tools,
            temperature,
            max_tokens,
            top_p,
            min_p,
            top_k,
            frequency_penalty,
            presence_penalty,
            repeat_penalty,
            stream,
        )
        self.calls.append(messages)
        has_tool_result = any(
            message.role == "tool"
            or (
                message.role == "user"
                and "Tool structured_tool result:" in message.content
            )
            for message in messages
        )
        if has_tool_result:
            return GenerationResult(content="done")
        return GenerationResult(
            content="",
            tool_calls=[
                ToolCall(
                    id="call-json",
                    name="structured_tool",
                    arguments={"query": "Mochi"},
                )
            ],
            finish_reason="tool_calls",
        )

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="reactive", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _BackendRejectingJsonToolMessages(BaseLLMBackend):
    """Simulate a backend that mis-parses JSON-ish tool messages as file inputs."""

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

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
        stream: bool = False,
    ) -> GenerationResult:
        del (
            tools,
            temperature,
            max_tokens,
            top_p,
            min_p,
            top_k,
            frequency_penalty,
            presence_penalty,
            repeat_penalty,
            stream,
        )
        self.calls.append(messages)
        tool_message = next((message for message in messages if message.role == "tool"), None)
        if tool_message is not None:
            if tool_message.content.lstrip().startswith("{"):
                raise RuntimeError("backend rejected 1.txt style tool payload")
            return GenerationResult(content="guarded")
        return GenerationResult(
            content="",
            tool_calls=[
                ToolCall(
                    id="call-json",
                    name="structured_tool",
                    arguments={"query": "Mochi"},
                )
            ],
            finish_reason="tool_calls",
        )

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="reject-json", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _EmptyFinalBackend(BaseLLMBackend):
    """Simulate a backend that succeeds but returns an empty final answer."""

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
        stream: bool = False,
    ) -> GenerationResult:
        del (
            messages,
            tools,
            temperature,
            max_tokens,
            top_p,
            min_p,
            top_k,
            frequency_penalty,
            presence_penalty,
            repeat_penalty,
            stream,
        )
        return GenerationResult(content="")

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="empty-final", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _NeverFinalToolLoopBackend(BaseLLMBackend):
    """Simulate a backend that keeps requesting tools and never produces a final answer."""

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

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
        stream: bool = False,
    ) -> GenerationResult:
        del (
            tools,
            temperature,
            max_tokens,
            top_p,
            min_p,
            top_k,
            frequency_penalty,
            presence_penalty,
            repeat_penalty,
            stream,
        )
        self.calls.append(messages)
        return GenerationResult(
            content="",
            tool_calls=[
                ToolCall(
                    id=f"loop-call-{len(self.calls)}",
                    name="web_search",
                    arguments={"query": "paper summary"},
                )
            ],
            finish_reason="tool_calls",
        )

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="never-final", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _NeverFinalRemoteToolLoopBackend(_NeverFinalToolLoopBackend):
    """Simulate a remote OpenAI-compatible backend that loops without a final answer."""

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            name="gpt-remote",
            backend_type="openai_compat",
            metadata={
                "api_mode": "responses",
                "tool_call_mode": "simulated_fallback",
                "native_tool_calling_status": "rejected_missing_parser",
                "request_shape": "chat_completions",
            },
        )


class _StreamingRemoteToolLoopBackend(BaseLLMBackend):
    """Simulate a native streaming OpenAI-compatible backend with tool calls."""

    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

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
        stream: bool = False,
    ) -> GenerationResult | AsyncIterator[StreamChunk]:
        del (
            tools,
            temperature,
            max_tokens,
            top_p,
            min_p,
            top_k,
            frequency_penalty,
            presence_penalty,
            repeat_penalty,
        )
        self.calls.append({"messages": messages, "stream": stream})
        if not stream:
            raise AssertionError("streaming backend should be consumed with stream=True")

        has_tool_result = any(message.role == "tool" for message in messages)

        async def _iterator() -> AsyncIterator[StreamChunk]:
            if not has_tool_result:
                yield StreamChunk(thinking_delta="Searching evidence")
                yield StreamChunk(
                    tool_call_delta=ToolCall(
                        id="stream-call-1",
                        name="web_search",
                        arguments={"query": "Taipei weather"},
                        index=0,
                    )
                )
                yield StreamChunk(is_final=True, finish_reason="tool_calls")
                return

            yield StreamChunk(thinking_delta="Summarizing evidence")
            yield StreamChunk(delta="Taipei is cloudy today.")
            yield StreamChunk(is_final=True, finish_reason="stop")

        return _iterator()

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            name="gpt-stream",
            backend_type="openai_compat",
            metadata={
                "api_mode": "chat_completions",
                "tool_call_mode": "native",
                "native_tool_calling_status": "supported",
                "request_shape": "chat_completions",
            },
        )

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _LiteralThinkStreamingBackend(BaseLLMBackend):
    """Simulate models that stream literal <think> blocks through content."""

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
        del messages, tools, temperature, max_tokens, top_p, min_p, top_k
        del frequency_penalty, presence_penalty, repeat_penalty, reasoning_effort
        assert stream is True

        async def _iterator() -> AsyncIterator[StreamChunk]:
            for char in "<think>plan first</think>Visible answer":
                yield StreamChunk(delta=char)
            yield StreamChunk(is_final=True, finish_reason="stop")

        return _iterator()

    def supports_tool_calling(self) -> bool:
        return False

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="literal-think-stream", backend_type="openai_compat")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _StreamingToolCallWithoutReasoningBackend(BaseLLMBackend):
    """Simulate OpenAI native tool-call streaming with no exposed reasoning block."""

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
        del tools, temperature, max_tokens, top_p, min_p, top_k
        del frequency_penalty, presence_penalty, repeat_penalty, reasoning_effort
        assert stream is True
        has_tool_result = any(message.role == "tool" for message in messages)

        async def _iterator() -> AsyncIterator[StreamChunk]:
            if not has_tool_result:
                yield StreamChunk(
                    tool_call_delta=ToolCall(
                        id="stream-call-1",
                        name="web_search",
                        arguments={"query": "Taipei weather"},
                        index=0,
                    )
                )
                yield StreamChunk(is_final=True, finish_reason="tool_calls")
                return
            yield StreamChunk(delta="Final answer")
            yield StreamChunk(is_final=True, finish_reason="stop")

        return _iterator()

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            name="gpt-stream-no-reasoning",
            backend_type="openai_compat",
            metadata={"api_mode": "chat_completions", "tool_call_mode": "native"},
        )

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _StreamingToolCallWithVisiblePreludeBackend(BaseLLMBackend):
    """Simulate providers that stream visible text before deciding to call a tool."""

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
        del tools, temperature, max_tokens, top_p, min_p, top_k
        del frequency_penalty, presence_penalty, repeat_penalty, reasoning_effort
        assert stream is True
        has_tool_result = any(message.role == "tool" for message in messages)

        async def _iterator() -> AsyncIterator[StreamChunk]:
            if not has_tool_result:
                yield StreamChunk(delta="preventing")
                yield StreamChunk(
                    tool_call_delta=ToolCall(
                        id="stream-call-1",
                        name="web_search",
                        arguments={"query": "Taipei weather"},
                        index=0,
                    )
                )
                yield StreamChunk(is_final=True, finish_reason="tool_calls")
                return
            yield StreamChunk(delta="成功")
            yield StreamChunk(is_final=True, finish_reason="stop")

        return _iterator()

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            name="gpt-stream-visible-prelude",
            backend_type="openai_compat",
            metadata={"api_mode": "chat_completions", "tool_call_mode": "native"},
        )

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _LowInformationFetchBackend(BaseLLMBackend):
    """Use the low-information fetch note to decide whether to continue retrieval."""

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

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
        del tools, temperature, max_tokens, top_p, min_p, top_k
        del frequency_penalty, presence_penalty, repeat_penalty, reasoning_effort, stream
        self.calls.append(list(messages))
        tool_messages = [message for message in messages if message.role == "tool"]
        if not tool_messages:
            return GenerationResult(
                content="",
                tool_calls=[
                    ToolCall(
                        id="fetch-1",
                        name="web_fetch",
                        arguments={"url": "https://example.com/shell"},
                    )
                ],
                finish_reason="tool_calls",
            )
        if "Evidence quality note:" in tool_messages[-1].content:
            return GenerationResult(
                content="",
                tool_calls=[
                    ToolCall(
                        id="search-1",
                        name="web_search",
                        arguments={"query": "specific source with current values"},
                    )
                ],
                finish_reason="tool_calls",
            )
        return GenerationResult(content="final from corroborated evidence")

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="low-info-fetch", backend_type="openai_compat")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _PrematureLowInformationFetchBackend(BaseLLMBackend):
    """Try to answer immediately after a low-information fetch."""

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

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
        del tools, temperature, max_tokens, top_p, min_p, top_k
        del frequency_penalty, presence_penalty, repeat_penalty, reasoning_effort, stream
        self.calls.append(list(messages))

        if not any(message.role == "tool" for message in messages):
            return GenerationResult(
                content="",
                tool_calls=[
                    ToolCall(
                        id="fetch-1",
                        name="web_fetch",
                        arguments={"url": "https://example.com/shell"},
                    )
                ],
                finish_reason="tool_calls",
            )

        if any(
            message.role == "user"
            and "Do not answer from that incomplete page alone" in message.content
            for message in messages
        ):
            if not any(message.role == "tool" and message.name == "web_search" for message in messages):
                return GenerationResult(
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="search-1",
                            name="web_search",
                            arguments={"query": "specific source with current values"},
                        )
                    ],
                    finish_reason="tool_calls",
                )
            return GenerationResult(content="final from corroborated evidence")

        return GenerationResult(content="premature answer from insufficient evidence")

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="premature-low-info-fetch", backend_type="openai_compat")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _RepeatedLowInformationSearchBackend(BaseLLMBackend):
    """Try to answer after a follow-up search repeats the low-information URL."""

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

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
        del tools, temperature, max_tokens, top_p, min_p, top_k
        del frequency_penalty, presence_penalty, repeat_penalty, reasoning_effort, stream
        self.calls.append(list(messages))

        if not any(message.role == "tool" for message in messages):
            return GenerationResult(
                content="",
                tool_calls=[
                    ToolCall(
                        id="fetch-1",
                        name="web_fetch",
                        arguments={"url": "https://example.com/shell"},
                    )
                ],
                finish_reason="tool_calls",
            )

        web_search_count = sum(
            1 for message in messages
            if message.role == "tool" and message.name == "web_search"
        )
        if web_search_count == 0:
            return GenerationResult(
                content="",
                tool_calls=[
                    ToolCall(
                        id="search-1",
                        name="web_search",
                        arguments={"query": "official source"},
                    )
                ],
                finish_reason="tool_calls",
            )
        if web_search_count == 1 and any(
            message.role == "user"
            and "Do not answer from that incomplete page alone" in message.content
            for message in messages
        ):
            return GenerationResult(
                content="",
                tool_calls=[
                    ToolCall(
                        id="search-2",
                        name="web_search",
                        arguments={"query": "alternate source"},
                    )
                ],
                finish_reason="tool_calls",
            )
        return GenerationResult(content="final from alternate evidence")

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="repeated-low-info-search", backend_type="openai_compat")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _NearDuplicateToolLoopBackend(BaseLLMBackend):
    """Simulate near-duplicate tool calls that only differ by casing/spacing."""

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []
        self._queries = [
            "Graph RAG papers",
            " graph   rag papers ",
            "GRAPH-RAG papers",
        ]

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
        stream: bool = False,
    ) -> GenerationResult:
        del (
            tools,
            temperature,
            max_tokens,
            top_p,
            min_p,
            top_k,
            frequency_penalty,
            presence_penalty,
            repeat_penalty,
            stream,
        )
        self.calls.append(messages)
        query = self._queries[min(len(self.calls) - 1, len(self._queries) - 1)]
        return GenerationResult(
            content="",
            tool_calls=[
                ToolCall(
                    id=f"near-dup-{len(self.calls)}",
                    name="web_search",
                    arguments={"query": query},
                )
            ],
            finish_reason="tool_calls",
        )

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="near-duplicate", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _RepeatedFailingWebFetchBackend(BaseLLMBackend):
    """Simulate a model that keeps retrying the same failing URL with minor argument changes."""

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []
        self.url = "https://example.com/paper"

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
        stream: bool = False,
    ) -> GenerationResult:
        del (
            tools,
            temperature,
            max_tokens,
            top_p,
            min_p,
            top_k,
            frequency_penalty,
            presence_penalty,
            repeat_penalty,
            stream,
        )
        self.calls.append(messages)
        last_tool_message = next((message for message in reversed(messages) if message.role == "tool"), None)
        if last_tool_message is not None and "already failed multiple times" in last_tool_message.content:
            return GenerationResult(content="I will stop retrying that URL and summarize from other evidence.")
        return GenerationResult(
            content="",
            tool_calls=[
                ToolCall(
                    id=f"fetch-{len(self.calls)}",
                    name="web_fetch",
                    arguments={"url": self.url, "max_bytes": 1000 + len(self.calls)},
                )
            ],
            finish_reason="tool_calls",
        )

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="repeat-fetch", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _RepeatedFailingWebSearchBackend(BaseLLMBackend):
    """Simulate a model that keeps retrying the same failing search query."""

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []
        self.queries = [
            "taipei weather tomorrow",
            " taipei   weather tomorrow ",
            "TAIPEI-WEATHER-TOMORROW",
        ]

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
        stream: bool = False,
    ) -> GenerationResult:
        del (
            tools,
            temperature,
            max_tokens,
            top_p,
            min_p,
            top_k,
            frequency_penalty,
            presence_penalty,
            repeat_penalty,
            stream,
        )
        self.calls.append(messages)
        last_tool_message = next((message for message in reversed(messages) if message.role == "tool"), None)
        if last_tool_message is not None and "already failed multiple times" in last_tool_message.content:
            return GenerationResult(content="I will stop retrying that search and explain the limitation.")
        index = min(len(self.calls) - 1, len(self.queries) - 1)
        query = self.queries[index]
        return GenerationResult(
            content="",
            tool_calls=[
                ToolCall(
                    id=f"search-{len(self.calls)}",
                    name="web_search",
                    arguments={"query": query, "top_k": 5 + index},
                )
            ],
            finish_reason="tool_calls",
        )

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="repeat-search", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _LiteratureSummaryGuardBackend(BaseLLMBackend):
    """Simulate a model that tries to keep researching even after enough papers were found."""

    def __init__(self) -> None:
        self.calls: list[list[Message]] = []

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
        stream: bool = False,
    ) -> GenerationResult:
        del (
            tools,
            temperature,
            max_tokens,
            top_p,
            min_p,
            top_k,
            frequency_penalty,
            presence_penalty,
            repeat_penalty,
            stream,
        )
        self.calls.append(messages)
        last_tool_message = next((message for message in reversed(messages) if message.role == "tool"), None)
        if last_tool_message is not None and "Sufficient literature evidence is already collected" in last_tool_message.content:
            return GenerationResult(content="Here is the synthesized paper overview.")
        if len(self.calls) == 1:
            return GenerationResult(
                content="",
                tool_calls=[
                    ToolCall(
                        id="lit-search-1",
                        name="semantic_scholar_search",
                        arguments={"query": "graph rag literature"},
                    )
                ],
                finish_reason="tool_calls",
            )
        return GenerationResult(
            content="",
            tool_calls=[
                ToolCall(
                    id="lit-fetch-2",
                    name="web_fetch",
                    arguments={"url": "https://arxiv.org/abs/2401.12345"},
                )
            ],
            finish_reason="tool_calls",
        )

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="literature-guard", backend_type="test")

    async def health_check(self) -> bool:
        return True

    async def close(self) -> None:
        return None


class _WebSearchTool(BaseTool):
    """測試用 web_search 工具。"""

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "搜尋網頁"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer"},
            },
            "required": ["query"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs) -> ToolResult:  # type: ignore[no-untyped-def]
        return ToolResult(output=f"found: {kwargs['query']}")


class _StructuredFollowupWebSearchTool(BaseTool):
    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs) -> ToolResult:  # type: ignore[no-untyped-def]
        query = str(kwargs["query"])
        if "alternate" not in query:
            return ToolResult(
                output={
                    "query": query,
                    "provider": "test",
                    "results": [
                        {
                            "title": "Official shell page",
                            "url": "https://example.com/shell",
                            "snippet": "",
                        }
                    ],
                },
                metadata={"provider": "test", "count": 1},
            )
        return ToolResult(
            output={
                "query": query,
                "provider": "test",
                "results": [
                    {
                        "title": "Alternate source",
                        "url": "https://example.com/alternate",
                        "snippet": (
                            "A detailed alternate source with enough extracted factual context "
                            "to support a response after the shell page failed."
                        ),
                    },
                    {
                        "title": "Another corroborating source",
                        "url": "https://example.org/details",
                        "snippet": "A second result corroborates the detailed alternate source.",
                    },
                ],
            },
            metadata={"provider": "test", "count": 2},
        )


class _StructuredTool(BaseTool):
    """回傳 dict/list 的測試工具。"""

    @property
    def name(self) -> str:
        return "structured_tool"

    @property
    def description(self) -> str:
        return "回傳結構化資料"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs) -> ToolResult:  # type: ignore[no-untyped-def]
        return ToolResult(output={"items": [{"title": kwargs["query"], "url": "https://example.com"}]})


class _LargeTextTool(BaseTool):
    """回傳超長文字，模擬 web_fetch 類工具輸出。"""

    @property
    def name(self) -> str:
        return "large_text_tool"

    @property
    def description(self) -> str:
        return "回傳大量純文字"

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs) -> ToolResult:  # type: ignore[no-untyped-def]
        return ToolResult(output="A" * 12000)


class _FailingWebFetchTool(BaseTool):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "Fetch one URL."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "max_bytes": {"type": "integer"},
            },
            "required": ["url"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs) -> ToolResult:  # type: ignore[no-untyped-def]
        self.calls += 1
        url = str(kwargs["url"])
        return ToolResult(
            error=f"HTTP 403 from {url}",
            metadata={"url": url, "status_code": 403},
        )


class _LowInformationWebFetchTool(BaseTool):
    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "Fetch one URL."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs) -> ToolResult:  # type: ignore[no-untyped-def]
        return ToolResult(
            output="County forecast",
            metadata={"url": str(kwargs["url"]), "status_code": 200, "extractor": "htmlparser"},
        )


class _StructuralLowInformationWebFetchTool(BaseTool):
    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "Fetch one URL."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs) -> ToolResult:  # type: ignore[no-untyped-def]
        return ToolResult(
            output=(
                "臺中市縣市預報\n"
                "臺中市中區鄉鎮預報\n"
                "今天天氣描述\n"
                "最高溫\n"
                "最低溫\n"
                "降雨機率\n"
                "舒適度\n"
                "風向風速\n"
            ),
            metadata={"url": str(kwargs["url"]), "status_code": 200, "extractor": "htmlparser"},
        )


class _FailingWebSearchTool(BaseTool):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        return "Search the web."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs) -> ToolResult:  # type: ignore[no-untyped-def]
        self.calls += 1
        query = str(kwargs["query"])
        return ToolResult(
            error=f"DuckDuckGo returned an anti-bot challenge for {query}",
            metadata={"provider": "duckduckgo_html", "blocked": True},
            retryable=True,
        )


class _LiteratureSearchTool(BaseTool):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "semantic_scholar_search"

    @property
    def description(self) -> str:
        return "Search papers."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs) -> ToolResult:  # type: ignore[no-untyped-def]
        self.calls += 1
        return ToolResult(
            output=[
                {"title": "Paper A", "abstract": "Abstract A", "url": "https://example.com/a"},
                {"title": "Paper B", "abstract": "Abstract B", "url": "https://example.com/b"},
                {"title": "Paper C", "abstract": "Abstract C", "url": "https://example.com/c"},
            ],
            metadata={"source": "semantic_scholar", "count": 3},
        )


class _UnusedWebFetchTool(BaseTool):
    def __init__(self) -> None:
        self.calls = 0

    @property
    def name(self) -> str:
        return "web_fetch"

    @property
    def description(self) -> str:
        return "Fetch one URL."

    @property
    def parameters_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {"url": {"type": "string"}},
            "required": ["url"],
            "additionalProperties": False,
        }

    async def execute(self, **kwargs) -> ToolResult:  # type: ignore[no-untyped-def]
        self.calls += 1
        return ToolResult(output=f"fetched: {kwargs['url']}")


TOOLS = [
    ToolSchema(
        name="web_search",
        description="搜尋網頁",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
    )
]


@pytest.mark.asyncio
async def test_gguf_backend_can_emit_tool_calls_via_simulator(tmp_path: Path) -> None:
    """GGUFBackend 應可透過 simulator 解析文字中的 tool_call。"""
    model_path = tmp_path / "toy.gguf"
    model_path.write_text("placeholder", encoding="utf-8")

    backend = GGUFBackend(model_path=str(model_path))
    backend._dependency_error = None  # noqa: SLF001
    backend._model = _FakeGGUFModel(  # noqa: SLF001
        '<tool_call>{"name":"web_search","arguments":{"query":"Mochi"}}</tool_call>'
    )

    result = await backend.generate(
        messages=[Message(role="user", content="幫我搜尋 Mochi")],
        tools=TOOLS,
        stream=False,
    )

    assert result.tool_calls
    assert result.tool_calls[0].name == "web_search"
    assert result.tool_calls[0].arguments == {"query": "Mochi"}


@pytest.mark.asyncio
async def test_safetensors_backend_can_emit_tool_calls_via_simulator(tmp_path: Path) -> None:
    """SafetensorsBackend 應可透過 simulator 解析文字中的 tool_call。"""
    model_dir = tmp_path / "hf-model"
    model_dir.mkdir()

    backend = SafetensorsBackend(model_dir=str(model_dir))
    backend._dependency_error = None  # noqa: SLF001
    backend._pipeline = _FakePipeline(  # noqa: SLF001
        '\n<tool_call>{"name":"web_search","arguments":{"query":"Python"}}</tool_call>'
    )

    result = await backend.generate(
        messages=[Message(role="user", content="幫我搜尋 Python")],
        tools=TOOLS,
        stream=False,
    )

    assert result.tool_calls
    assert result.tool_calls[0].name == "web_search"
    assert result.tool_calls[0].arguments == {"query": "Python"}


@pytest.mark.asyncio
async def test_react_loop_executes_simulated_tool_calls_end_to_end(tmp_path: Path) -> None:
    """ReAct 應可完成非原生 tool_call 到工具執行再到最終回答。"""
    model_path = tmp_path / "toy.gguf"
    model_path.write_text("placeholder", encoding="utf-8")
    fake_model = _ReactiveGGUFModel()

    backend = GGUFBackend(
        model_path=str(model_path),
        model_loader=lambda: fake_model,
    )
    backend._dependency_error = None  # noqa: SLF001

    registry = ToolRegistry(discover_builtin=False)
    registry.register(_WebSearchTool())

    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        max_iterations=3,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="你是測試助理",
            history=[],
            user_message="幫我查 Mochi AI",
        )
    ]

    request_event = next(event for event in events if isinstance(event, ToolCallRequestEvent))
    result_event = next(event for event in events if isinstance(event, ToolCallResultEvent))
    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))

    assert request_event.tool_name == "web_search"
    assert request_event.arguments == {"query": "Mochi AI"}
    assert result_event.tool_name == "web_search"
    assert result_event.result == "found: Mochi AI"
    assert result_event.error is None
    assert final_event.content == "我已根據 web_search 的結果完成整理。"
    assert len(fake_model.calls) == 2
    assert any(
        message.get("role") == "user"
        and message.get("content") == "Tool web_search result:\nfound: Mochi AI"
        for message in fake_model.calls[1]["messages"]
        if isinstance(message, dict)
    )


@pytest.mark.asyncio
async def test_react_loop_sanitizes_simulated_tool_messages_for_gguf(tmp_path: Path) -> None:
    """GGUF 模擬 tool calling 的後續輪次不應送出 OpenAI-style tool 訊息。"""
    model_path = tmp_path / "toy.gguf"
    model_path.write_text("placeholder", encoding="utf-8")
    fake_model = _StrictReactiveGGUFModel()

    backend = GGUFBackend(
        model_path=str(model_path),
        model_loader=lambda: fake_model,
    )
    backend._dependency_error = None  # noqa: SLF001

    registry = ToolRegistry(discover_builtin=False)
    registry.register(_WebSearchTool())

    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        max_iterations=3,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="你是測試助理。",
            history=[],
            user_message="請搜尋 Mochi AI",
        )
    ]

    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))

    assert final_event.content == "整理完工具結果，這是最終回答。"
    assert len(fake_model.calls) == 2
    second_messages = fake_model.calls[1]["messages"]
    assert all(
        not (isinstance(message, dict) and message.get("role") == "tool")
        for message in second_messages
    )
    assert all(
        not (isinstance(message, dict) and message.get("tool_calls"))
        for message in second_messages
    )
    assert any(
        isinstance(message, dict)
        and message.get("role") == "user"
        and "Tool web_search result:" in str(message.get("content", ""))
        for message in second_messages
    )


@pytest.mark.asyncio
async def test_react_loop_executes_native_tool_calls_end_to_end_for_allowlisted_chat_format(
    tmp_path: Path,
) -> None:
    """Allowlisted chat format should run native tool-call send/receive in ReAct loop."""
    model_path = tmp_path / "toy.gguf"
    model_path.write_text("placeholder", encoding="utf-8")
    fake_model = _NativeReactiveGGUFModel()

    backend = GGUFBackend(
        model_path=str(model_path),
        model_loader=lambda: fake_model,
    )
    backend._dependency_error = None  # noqa: SLF001

    registry = ToolRegistry(discover_builtin=False)
    registry.register(_WebSearchTool())

    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        max_iterations=3,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="你是測試助理",
            history=[],
            user_message="幫我查 Mochi AI",
        )
    ]

    request_event = next(event for event in events if isinstance(event, ToolCallRequestEvent))
    result_event = next(event for event in events if isinstance(event, ToolCallResultEvent))
    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))

    assert request_event.tool_name == "web_search"
    assert request_event.arguments == {"query": "Mochi AI"}
    assert result_event.tool_name == "web_search"
    assert result_event.result == "found: Mochi AI"
    assert final_event.content == "native path done"

    assert len(fake_model.calls) == 2
    first_messages = fake_model.calls[0]["messages"]
    second_messages = fake_model.calls[1]["messages"]
    assert first_messages[0]["content"] == "你是測試助理"
    assert "## Tool Use Instructions" not in first_messages[0]["content"]
    assert any(
        isinstance(message, dict)
        and message.get("role") == "assistant"
        and message.get("tool_calls")
        for message in second_messages
    )
    assert any(
        isinstance(message, dict)
        and message.get("role") == "tool"
        and message.get("name") == "web_search"
        for message in second_messages
    )

    info = backend.get_model_info()
    assert info.metadata["tool_call_strategy"] == "structured_native"
    assert info.metadata["detected_chat_format"] == "functionary-v2"


@pytest.mark.asyncio
async def test_react_loop_returns_structured_tool_result_as_json_message() -> None:
    """工具回灌給模型的 tool message 應保留結構化 JSON，而不是 Python repr。"""
    backend = _ReactiveBackend()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(_StructuredTool())
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        max_iterations=3,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="你是測試助理",
            history=[],
            user_message="查 Mochi",
        )
    ]

    result_event = next(event for event in events if isinstance(event, ToolCallResultEvent))
    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))
    tool_message = next(message for message in backend.calls[1] if message.role == "tool")

    assert result_event.result == {"items": [{"title": "Mochi", "url": "https://example.com"}]}
    assert tool_message.content.startswith("Tool structured_tool result:")
    assert "https://example.com" in tool_message.content
    assert not tool_message.content.lstrip().startswith("{")
    assert final_event.content == "done"


@pytest.mark.asyncio
async def test_react_loop_truncates_large_tool_result_before_reinjection(
    tmp_path: Path,
) -> None:
    """大型工具輸出回灌模型前應縮短，避免單次工具結果撐爆 context。"""
    backend = _ReactiveBackend()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(_LargeTextTool())
    context = ToolExecutionContext(
        session_id="session-large",
        tool_result_store_dir=str(tmp_path / "tool-results"),
    )
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        tool_execution_context=context,
        max_iterations=3,
    )

    async def _patched_generate(
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
        stream: bool = False,
    ) -> GenerationResult:
        del (
            tools,
            temperature,
            max_tokens,
            top_p,
            min_p,
            top_k,
            frequency_penalty,
            presence_penalty,
            repeat_penalty,
            stream,
        )
        backend.calls.append(messages)
        has_tool_result = any(message.role == "tool" for message in messages)
        if has_tool_result:
            return GenerationResult(content="done")
        return GenerationResult(
            content="",
            tool_calls=[
                ToolCall(
                    id="call-large",
                    name="large_text_tool",
                    arguments={"query": "Mochi"},
                )
            ],
            finish_reason="tool_calls",
        )

    backend.generate = _patched_generate  # type: ignore[method-assign]

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="請讀取這個頁面",
        )
    ]

    tool_result_event = next(event for event in events if isinstance(event, ToolCallResultEvent))
    tool_message = next(message for message in backend.calls[1] if message.role == "tool")

    assert isinstance(tool_result_event.result, str)
    assert len(tool_result_event.result) == 12000
    assert tool_message.content.startswith("Tool large_text_tool result")
    assert "Reference:" in tool_message.content
    assert tool_result_event.metadata["transport"]["overflow_persisted"] is True
    assert tool_result_event.metadata["transport"]["reference_id"]


@pytest.mark.asyncio
async def test_react_loop_preserves_small_file_read_text_for_followup_model_round(
    tmp_path: Path,
) -> None:
    class _FileReadBackend(BaseLLMBackend):
        def __init__(self) -> None:
            self.calls: list[list[Message]] = []

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
            stream: bool = False,
        ) -> GenerationResult:
            del (
                tools,
                temperature,
                max_tokens,
                top_p,
                min_p,
                top_k,
                frequency_penalty,
                presence_penalty,
                repeat_penalty,
                stream,
            )
            self.calls.append(messages)
            tool_message = next((message for message in messages if message.role == "tool"), None)
            if tool_message is not None:
                assert tool_message.content == "1: alpha\n2: beta"
                assert not tool_message.content.startswith("Tool file_read result:")
                return GenerationResult(content="done")
            return GenerationResult(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call-file-read",
                        name="file_read",
                        arguments={"path": "sample.txt", "offset": 1, "limit": 2, "line_numbers": True},
                    )
                ],
                finish_reason="tool_calls",
            )

        def supports_tool_calling(self) -> bool:
            return True

        def get_model_info(self) -> ModelInfo:
            return ModelInfo(name="file-read-backend", backend_type="test")

        async def health_check(self) -> bool:
            return True

        async def close(self) -> None:
            return None

    (tmp_path / "sample.txt").write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    backend = _FileReadBackend()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(FileReadTool(workspace_dir=tmp_path))
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        tool_execution_context=ToolExecutionContext(workspace_dir=str(tmp_path)),
        max_iterations=3,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="read the file",
        )
    ]

    result_event = next(event for event in events if isinstance(event, ToolCallResultEvent))
    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))
    tool_message = next(message for message in backend.calls[1] if message.role == "tool")

    assert result_event.result == "1: alpha\n2: beta"
    assert tool_message.content == "1: alpha\n2: beta"
    assert final_event.content == "done"


@pytest.mark.asyncio
async def test_react_loop_exposes_large_file_read_as_resumable_followup_contract(
    tmp_path: Path,
) -> None:
    class _LargeFileReadBackend(BaseLLMBackend):
        def __init__(self) -> None:
            self.calls: list[list[Message]] = []

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
            stream: bool = False,
        ) -> GenerationResult:
            del (
                tools,
                temperature,
                max_tokens,
                top_p,
                min_p,
                top_k,
                frequency_penalty,
                presence_penalty,
                repeat_penalty,
                stream,
            )
            self.calls.append(messages)
            tool_message = next((message for message in messages if message.role == "tool"), None)
            if tool_message is not None:
                assert "Tool file_read result preview" in tool_message.content
                assert 'file_read(path="tool-result://' in tool_message.content
                assert "offset=1, limit=200, line_numbers=True" in tool_message.content
                return GenerationResult(content="done")
            return GenerationResult(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call-file-read-large",
                        name="file_read",
                        arguments={"path": "large.txt", "offset": 1, "limit": 260, "line_numbers": True},
                    )
                ],
                finish_reason="tool_calls",
            )

        def supports_tool_calling(self) -> bool:
            return True

        def get_model_info(self) -> ModelInfo:
            return ModelInfo(name="large-file-read-backend", backend_type="test")

        async def health_check(self) -> bool:
            return True

        async def close(self) -> None:
            return None

    (tmp_path / "large.txt").write_text(
        "".join(f"line {idx}\n" for idx in range(1, 401)),
        encoding="utf-8",
    )
    backend = _LargeFileReadBackend()
    context = ToolExecutionContext(
        workspace_dir=str(tmp_path),
        session_id="session-file-read-large",
        tool_result_store_dir=str(tmp_path / "tool-results"),
    )
    registry = ToolRegistry(discover_builtin=False)
    registry.register(FileReadTool(workspace_dir=tmp_path))
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        tool_execution_context=context,
        max_iterations=3,
        max_tool_message_chars=220,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="read the large file",
        )
    ]

    result_event = next(event for event in events if isinstance(event, ToolCallResultEvent))
    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))
    transport = result_event.metadata["transport"]
    reference_id = transport["reference_id"]
    tool_message = next(message for message in backend.calls[1] if message.role == "tool")

    assert transport["overflow_persisted"] is True
    assert reference_id
    assert f'tool-result://{reference_id}' in tool_message.content
    assert context.tool_result_references[reference_id]["reference_id"] == reference_id
    assert final_event.content == "done"


@pytest.mark.asyncio
async def test_react_loop_guards_json_tool_payloads_before_backend_send() -> None:
    backend = _BackendRejectingJsonToolMessages()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(_StructuredTool())
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        max_iterations=3,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="guard structured output",
        )
    ]

    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))
    assert final_event.content == "guarded"
    tool_message = next(message for message in backend.calls[1] if message.role == "tool")
    assert not tool_message.content.lstrip().startswith("{")


@pytest.mark.asyncio
async def test_react_loop_emits_backend_error_diagnostics_for_guarded_tool_context(
    tmp_path: Path,
) -> None:
    class _ExplodingBackend(_BackendRejectingJsonToolMessages):
        async def generate(self, messages: list[Message], *args, **kwargs) -> GenerationResult:  # type: ignore[override]
            self.calls.append(messages)
            if any(message.role == "tool" for message in messages):
                raise RuntimeError("backend exploded on 1.txt")
            return GenerationResult(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call-large",
                        name="large_text_tool",
                        arguments={"query": "Mochi"},
                    )
                ],
                finish_reason="tool_calls",
            )

    backend = _ExplodingBackend()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(_LargeTextTool())
    context = ToolExecutionContext(
        session_id="session-error",
        tool_result_store_dir=str(tmp_path / "tool-results"),
    )
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        tool_execution_context=context,
        max_iterations=3,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="trigger failure",
        )
    ]

    error_event = next(event for event in events if isinstance(event, ErrorEvent))
    assert error_event.message == "backend exploded on 1.txt"
    assert error_event.metadata["backend"]["backend_type"] == "test"
    assert error_event.metadata["transport"]["last_tool_name"] == "large_text_tool"
    assert error_event.metadata["transport"]["overflow_persisted"] is True


@pytest.mark.asyncio
async def test_react_loop_emits_error_instead_of_blank_final_answer() -> None:
    backend = _EmptyFinalBackend()
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=ToolRegistry(discover_builtin=False),
        max_iterations=1,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="introduce this paper",
        )
    ]

    assert not any(isinstance(event, FinalAnswerEvent) for event in events)
    error_event = next(event for event in events if isinstance(event, ErrorEvent))
    assert error_event.message == "Model returned an empty response."
    assert error_event.metadata["backend"]["backend_type"] == "test"
    assert error_event.metadata["backend"]["finish_reason"] == "stop"


@pytest.mark.asyncio
async def test_react_loop_emits_error_when_tool_loop_hits_max_iterations() -> None:
    backend = _NeverFinalToolLoopBackend()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(_WebSearchTool())
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        max_iterations=2,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="introduce the retrieved paper",
        )
    ]

    assert not any(isinstance(event, FinalAnswerEvent) for event in events)
    assert len([event for event in events if isinstance(event, ToolCallRequestEvent)]) == 2
    error_event = next(event for event in events if isinstance(event, ErrorEvent))
    assert error_event.code == "MAX_ITERATIONS_REACHED"
    assert error_event.message == (
        "Model reached the maximum tool-call iterations without producing a final answer."
    )
    assert error_event.metadata["backend"]["backend_type"] == "test"
    assert error_event.metadata["backend"]["finish_reason"] == "tool_calls"
    assert error_event.metadata["loop"]["max_iterations"] == 2
    assert error_event.metadata["transport"]["last_tool_name"] == "web_search"


@pytest.mark.asyncio
async def test_react_loop_emits_remote_iteration_progress_diagnostics() -> None:
    backend = _NeverFinalRemoteToolLoopBackend()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(_WebSearchTool())
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        max_iterations=2,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="introduce the retrieved paper",
        )
    ]

    status_events = [event for event in events if isinstance(event, StatusEvent)]
    thinking_events = [event for event in events if isinstance(event, ThinkingEvent)]
    assert all("Mochi progress:" not in event.content for event in thinking_events)
    assert all("Model requested tool call(s):" not in event.content for event in thinking_events)
    assert [event.tool_name for event in events if isinstance(event, ToolCallRequestEvent)] == [
        "web_search",
        "web_search",
    ]
    assert status_events
    assert status_events[0].content == (
        "Mochi progress: ReAct iteration 1/2 "
        "(backend=openai_compat, model=gpt-remote, api=responses, "
        "tools=simulated_fallback, native_status=rejected_missing_parser, "
        "request_shape=chat_completions)"
    )

    error_event = next(event for event in events if isinstance(event, ErrorEvent))
    assert error_event.code == "MAX_ITERATIONS_REACHED"
    assert error_event.message == (
        "Model reached the maximum tool-call iterations without producing a final answer. "
        "(backend=openai_compat, model=gpt-remote, api=responses, "
        "tools=simulated_fallback, native_status=rejected_missing_parser, "
        "request_shape=chat_completions)"
    )
    assert error_event.metadata["backend"]["tool_call_mode"] == "simulated_fallback"
    assert error_event.metadata["backend"]["native_tool_calling_status"] == "rejected_missing_parser"
    assert error_event.metadata["backend"]["request_shape"] == "chat_completions"


@pytest.mark.asyncio
async def test_react_loop_separates_streamed_reasoning_from_text_chunks() -> None:
    backend = _StreamingRemoteToolLoopBackend()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(_WebSearchTool())
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        max_iterations=4,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="what is the weather",
        )
    ]

    text_chunk_events = [event for event in events if isinstance(event, TextChunkEvent)]
    assert [event.content for event in text_chunk_events] == ["Taipei is cloudy today."]
    thinking_events = [event for event in events if isinstance(event, ThinkingEvent)]
    assert [event.content for event in thinking_events] == [
        "Searching evidence",
        "Summarizing evidence",
    ]

    tool_request = next(event for event in events if isinstance(event, ToolCallRequestEvent))
    tool_result = next(event for event in events if isinstance(event, ToolCallResultEvent))
    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))

    assert tool_request.tool_name == "web_search"
    assert tool_request.arguments == {"query": "Taipei weather"}
    assert tool_result.tool_name == "web_search"
    assert final_event.content == "Taipei is cloudy today."
    assert len(backend.calls) == 2
    assert backend.calls[0]["stream"] is True
    assert backend.calls[1]["stream"] is True


@pytest.mark.asyncio
async def test_react_loop_routes_literal_streamed_think_blocks_to_reasoning() -> None:
    react_loop = AsyncReActLoop(
        backend=_LiteralThinkStreamingBackend(),
        tool_registry=ToolRegistry(discover_builtin=False),
        max_iterations=2,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="answer",
        )
    ]

    text_chunk_events = [event for event in events if isinstance(event, TextChunkEvent)]
    thinking_events = [event for event in events if isinstance(event, ThinkingEvent)]
    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))

    assert "".join(event.content for event in text_chunk_events) == "Visible answer"
    assert [event.content for event in thinking_events] == ["plan first"]
    assert final_event.content == "Visible answer"


@pytest.mark.asyncio
async def test_react_loop_keeps_tool_requests_out_of_reasoning_when_stream_has_no_reasoning() -> None:
    registry = ToolRegistry(discover_builtin=False)
    registry.register(_WebSearchTool())
    react_loop = AsyncReActLoop(
        backend=_StreamingToolCallWithoutReasoningBackend(),
        tool_registry=registry,
        max_iterations=3,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="what is the weather",
        )
    ]

    thinking_events = [event for event in events if isinstance(event, ThinkingEvent)]
    tool_request = next(event for event in events if isinstance(event, ToolCallRequestEvent))
    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))

    assert thinking_events == []
    assert tool_request.tool_name == "web_search"
    assert tool_request.arguments == {"query": "Taipei weather"}
    assert final_event.content == "Final answer"


@pytest.mark.asyncio
async def test_react_loop_hides_streamed_visible_prelude_until_final_answer() -> None:
    registry = ToolRegistry(discover_builtin=False)
    registry.register(_WebSearchTool())
    react_loop = AsyncReActLoop(
        backend=_StreamingToolCallWithVisiblePreludeBackend(),
        tool_registry=registry,
        max_iterations=3,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="what is the weather",
        )
    ]

    text_chunk_events = [event for event in events if isinstance(event, TextChunkEvent)]
    thinking_events = [event for event in events if isinstance(event, ThinkingEvent)]
    tool_request = next(event for event in events if isinstance(event, ToolCallRequestEvent))
    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))

    assert [event.content for event in text_chunk_events] == ["成功"]
    assert [event.content for event in thinking_events] == ["preventing"]
    assert tool_request.tool_name == "web_search"
    assert final_event.content == "成功"


@pytest.mark.asyncio
async def test_react_loop_marks_low_information_web_fetch_before_next_model_round() -> None:
    backend = _LowInformationFetchBackend()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(_LowInformationWebFetchTool())
    registry.register(_WebSearchTool())
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        max_iterations=4,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="find current values",
        )
    ]

    tool_results = [event for event in events if isinstance(event, ToolCallResultEvent)]
    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))

    assert tool_results[0].tool_name == "web_fetch"
    assert tool_results[0].metadata["evidence_quality"]["status"] == "insufficient_evidence"
    assert any(
        message.role == "tool" and "Evidence quality note:" in message.content
        for message in backend.calls[1]
    )
    thinking_events = [event for event in events if isinstance(event, ThinkingEvent)]
    assert all("web_fetch returned too little extracted page content" not in event.content for event in thinking_events)
    assert any(event.tool_name == "web_search" for event in tool_results)
    assert final_event.content == "final from corroborated evidence"


@pytest.mark.asyncio
async def test_react_loop_forces_followup_retrieval_after_low_information_web_fetch() -> None:
    backend = _PrematureLowInformationFetchBackend()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(_LowInformationWebFetchTool())
    registry.register(_WebSearchTool())
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        max_iterations=5,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="find current values",
        )
    ]

    tool_requests = [event for event in events if isinstance(event, ToolCallRequestEvent)]
    text_chunks = [event for event in events if isinstance(event, TextChunkEvent)]
    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))

    assert [event.tool_name for event in tool_requests] == ["web_fetch", "web_search"]
    assert all("premature answer" not in event.content for event in text_chunks)
    assert any(
        message.role == "user"
        and "Do not answer from that incomplete page alone" in message.content
        for message in backend.calls[2]
    )
    assert final_event.content == "final from corroborated evidence"


@pytest.mark.asyncio
async def test_react_loop_treats_structural_htmlparser_fetch_as_insufficient_evidence() -> None:
    backend = _PrematureLowInformationFetchBackend()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(_StructuralLowInformationWebFetchTool())
    registry.register(_WebSearchTool())
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        max_iterations=5,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="find current values",
        )
    ]

    tool_results = [event for event in events if isinstance(event, ToolCallResultEvent)]
    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))

    assert tool_results[0].tool_name == "web_fetch"
    assert tool_results[0].metadata["evidence_quality"]["status"] == "insufficient_evidence"
    assert tool_results[0].metadata["evidence_quality"]["reason"] == "low_information_web_fetch"
    assert final_event.content == "final from corroborated evidence"


@pytest.mark.asyncio
async def test_react_loop_keeps_searching_when_followup_search_repeats_low_information_url() -> None:
    backend = _RepeatedLowInformationSearchBackend()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(_LowInformationWebFetchTool())
    registry.register(_StructuredFollowupWebSearchTool())
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        max_iterations=6,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="find current values",
        )
    ]

    tool_requests = [event for event in events if isinstance(event, ToolCallRequestEvent)]
    tool_results = [event for event in events if isinstance(event, ToolCallResultEvent)]
    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))

    assert [event.tool_name for event in tool_requests] == [
        "web_fetch",
        "web_search",
        "web_search",
    ]
    assert tool_results[1].metadata["evidence_quality"]["reason"] == (
        "web_search_repeated_low_information_source"
    )
    assert final_event.content == "final from alternate evidence"


@pytest.mark.asyncio
async def test_react_loop_detects_repeated_tool_call_patterns_before_max_iterations() -> None:
    backend = _NeverFinalToolLoopBackend()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(_WebSearchTool())
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        max_iterations=10,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="introduce the retrieved paper",
        )
    ]

    assert not any(isinstance(event, FinalAnswerEvent) for event in events)
    assert len([event for event in events if isinstance(event, ToolCallRequestEvent)]) == 2
    assert len(backend.calls) == 3
    error_event = next(event for event in events if isinstance(event, ErrorEvent))
    assert error_event.code == "REPEATED_TOOL_LOOP"
    assert error_event.message == "Model repeated the same tool-call pattern without making progress."
    assert error_event.metadata["backend"]["finish_reason"] == "tool_calls"
    assert error_event.metadata["loop"]["repeated_tool_rounds"] == 3
    assert error_event.metadata["loop"]["tool_calls"] == [
        {"name": "web_search", "arguments": {"query": "paper summary"}}
    ]
    assert error_event.metadata["transport"]["last_tool_name"] == "web_search"


@pytest.mark.asyncio
async def test_react_loop_detects_near_duplicate_tool_call_patterns() -> None:
    backend = _NearDuplicateToolLoopBackend()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(_WebSearchTool())
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        max_iterations=10,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="find graph rag papers",
        )
    ]

    assert not any(isinstance(event, FinalAnswerEvent) for event in events)
    assert len([event for event in events if isinstance(event, ToolCallRequestEvent)]) == 2
    error_event = next(event for event in events if isinstance(event, ErrorEvent))
    assert error_event.code == "REPEATED_TOOL_LOOP"
    assert error_event.metadata["loop"]["repeated_tool_rounds"] == 3


@pytest.mark.asyncio
async def test_react_loop_blocks_repeated_failing_web_fetch_urls() -> None:
    backend = _RepeatedFailingWebFetchBackend()
    failing_tool = _FailingWebFetchTool()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(failing_tool)
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        max_iterations=6,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="fetch the paper page",
        )
    ]

    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))
    tool_result_events = [event for event in events if isinstance(event, ToolCallResultEvent)]

    assert final_event.content == "I will stop retrying that URL and summarize from other evidence."
    assert failing_tool.calls == 2
    assert len(tool_result_events) == 3
    assert tool_result_events[-1].error == (
        "Skipping repeated fetch because this URL already failed multiple times: "
        "https://example.com/paper"
    )
    assert tool_result_events[-1].metadata["guard"] == "repeated_web_fetch_failure"
    assert tool_result_events[-1].metadata["failure_count"] == 2


@pytest.mark.asyncio
async def test_react_loop_blocks_repeated_failing_web_search_queries() -> None:
    backend = _RepeatedFailingWebSearchBackend()
    failing_tool = _FailingWebSearchTool()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(failing_tool)
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        max_iterations=6,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="find tomorrow's weather",
        )
    ]

    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))
    tool_result_events = [event for event in events if isinstance(event, ToolCallResultEvent)]

    assert final_event.content == "I will stop retrying that search and explain the limitation."
    assert failing_tool.calls == 2
    assert len(tool_result_events) == 3
    assert tool_result_events[-1].error == (
        "Skipping repeated search because this query already failed multiple times: "
        "TAIPEI-WEATHER-TOMORROW"
    )
    assert tool_result_events[-1].metadata["guard"] == "repeated_web_search_failure"
    assert tool_result_events[-1].metadata["failure_count"] == 2
    assert tool_result_events[-1].metadata["provider"] == "duckduckgo_html"


@pytest.mark.asyncio
async def test_react_loop_forces_literature_summary_after_enough_evidence() -> None:
    backend = _LiteratureSummaryGuardBackend()
    literature_tool = _LiteratureSearchTool()
    web_fetch_tool = _UnusedWebFetchTool()
    registry = ToolRegistry(discover_builtin=False)
    registry.register(literature_tool)
    registry.register(web_fetch_tool)
    react_loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        max_iterations=6,
    )

    events = [
        event
        async for event in react_loop.run(
            system_prompt="system",
            history=[],
            user_message="search literature then introduce the paper",
        )
    ]

    final_event = next(event for event in events if isinstance(event, FinalAnswerEvent))
    tool_result_events = [event for event in events if isinstance(event, ToolCallResultEvent)]

    assert final_event.content == "Here is the synthesized paper overview."
    assert literature_tool.calls == 1
    assert web_fetch_tool.calls == 0
    assert any(
        message.role == "user" and "You already have enough literature evidence." in message.content
        for message in backend.calls[1]
    )
    assert tool_result_events[-1].error == (
        "Sufficient literature evidence is already collected. "
        "Do not call more search or fetch tools; synthesize the answer now."
    )
    assert tool_result_events[-1].metadata["guard"] == "literature_summary_ready"
