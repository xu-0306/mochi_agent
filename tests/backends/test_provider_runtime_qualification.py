"""Runtime qualification for normalized generation terminal handling."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from mochi.agents.events import (
    AssistantTruncatedEvent,
    ErrorEvent,
    FinalAnswerEvent,
    StatusEvent,
    ToolCallRequestEvent,
)
from mochi.agents.generation_policy import GenerationTerminal, normalize_generation_terminal
from mochi.agents.react_loop import AsyncReActLoop
from mochi.backends.base import BackendRequestError, BaseLLMBackend
from mochi.backends.ollama import OllamaBackend
from mochi.backends.openai_compat import OpenAICompatBackend
from mochi.backends.types import GenerationResult, Message, ModelInfo, ToolCall, ToolSchema
from mochi.tools.base import ToolExecutionContext
from mochi.tools.file_ops import FileReadTool
from mochi.tools.registry import ToolRegistry

from ._support import _mock_response

_DECLARED_PROVIDER_IDS = frozenset({"openai_compat", "ollama"})
_CAPABILITY_NAMES = frozenset(
    {"context", "tools", "malformed_tool", "truncation", "streaming", "cancellation"}
)
_CAPABILITY_STATUSES = frozenset({"supported", "deferred", "unsupported"})


@dataclass(frozen=True)
class _ProviderQualification:
    provider_id: str
    capabilities: dict[str, str]


class _MockStreamContext:
    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.raise_for_status = MagicMock()

    async def __aenter__(self) -> _MockStreamContext:
        return self

    async def __aexit__(self, exc_type: object, exc: object, tb: object) -> None:
        return None

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line


def _assert_complete_provider_report(report: list[_ProviderQualification]) -> None:
    provider_ids = {qualification.provider_id for qualification in report}
    assert provider_ids == _DECLARED_PROVIDER_IDS, "provider rows must be exact and complete"
    assert len(report) == len(provider_ids), "provider rows must not be duplicated"
    for qualification in report:
        assert set(qualification.capabilities) == _CAPABILITY_NAMES, (
            f"{qualification.provider_id} must declare every runtime capability"
        )
        unknown_statuses = set(qualification.capabilities.values()) - _CAPABILITY_STATUSES
        assert not unknown_statuses, (
            f"{qualification.provider_id} has unsupported capability statuses: {unknown_statuses}"
        )


def _assert_truncation_terminal(reason: str | None) -> None:
    decision = normalize_generation_terminal(
        terminal_signal=reason,
        has_structured_tool_calls=False,
        output="partial output",
    )
    assert decision.terminal is GenerationTerminal.OUTPUT_TRUNCATED


async def _qualify_openai_compat() -> _ProviderQualification:
    serving_backend = OpenAICompatBackend(
        "https://example.test/v1",
        "proxy-model",
        configured_context_length=8192,
    )
    advertised_backend = OpenAICompatBackend("https://example.test/v1", "proxy-model")
    model_payload = {"data": [{"id": "proxy-model", "context_length": 32768}]}
    try:
        with patch.object(
            serving_backend._client,  # noqa: SLF001
            "get",
            new_callable=AsyncMock,
            return_value=_mock_response(model_payload),
        ):
            await serving_backend._ensure_capability_discovery()  # noqa: SLF001
        blocking_result = serving_backend._parse_chat_completions_result(  # noqa: SLF001
            {
                "context_length": 4096,
                "choices": [{"message": {"content": "partial"}, "finish_reason": "length"}],
            }
        )
        stream_lines = [
            'data: {"choices":[{"delta":{"content":"partial"},"finish_reason":null}]}',
            'data: {"choices":[{"delta":{},"finish_reason":"length"}]}',
        ]
        with patch.object(
            serving_backend._client,  # noqa: SLF001
            "stream",
            return_value=_MockStreamContext(stream_lines),
        ):
            stream_chunks = [
                chunk
                async for chunk in serving_backend._stream_generate(  # noqa: SLF001
                    {"stream": True},
                    request_url=serving_backend._chat_completions_url,  # noqa: SLF001
                )
            ]
        with patch.object(
            advertised_backend._client,  # noqa: SLF001
            "get",
            new_callable=AsyncMock,
            return_value=_mock_response(model_payload),
        ):
            await advertised_backend._ensure_capability_discovery()  # noqa: SLF001

        serving_info = serving_backend.get_model_info()
        advertised_info = advertised_backend.get_model_info()
    finally:
        await serving_backend.close()
        await advertised_backend.close()

    assert serving_info.context_length == 4096
    assert serving_info.metadata["serving_context_length"] == 4096
    assert serving_info.metadata["advertised_context_length"] == 32768
    assert serving_info.metadata["context_source"] == "serving"
    assert serving_info.metadata["context_confidence"] == "high"
    assert serving_info.metadata["context_is_hard_limit"] is True
    assert advertised_info.context_length == 32768
    assert advertised_info.metadata["context_source"] == "advertised"
    assert advertised_info.metadata["context_confidence"] == "medium"
    assert advertised_info.metadata["context_is_hard_limit"] is False
    assert sum(chunk.is_final for chunk in stream_chunks) == 1
    stream_reason = next(chunk.finish_reason for chunk in stream_chunks if chunk.is_final)
    _assert_truncation_terminal(blocking_result.finish_reason)
    _assert_truncation_terminal(stream_reason)
    assert serving_backend.supports_tool_calling() is True
    assert not hasattr(serving_backend, "cancel_generation")
    return _ProviderQualification(
        provider_id="openai_compat",
        capabilities={
            "context": "supported",
            "tools": "supported",
            "malformed_tool": "supported",
            "truncation": "supported",
            "streaming": "supported",
            "cancellation": "unsupported",
        },
    )


async def _qualify_ollama() -> _ProviderQualification:
    serving_backend = OllamaBackend(model="llama3.2", base_url="http://localhost:11434")
    advertised_backend = OllamaBackend(model="llama3.2", base_url="http://localhost:11434")
    try:
        with patch.object(
            serving_backend._client,  # noqa: SLF001
            "post",
            new_callable=AsyncMock,
            return_value=_mock_response(
                {
                    "parameters": "num_ctx 4096",
                    "model_info": {"llama.context_length": 32768},
                }
            ),
        ):
            await serving_backend.prime_model_info()
        with patch.object(
            advertised_backend._client,  # noqa: SLF001
            "post",
            new_callable=AsyncMock,
            return_value=_mock_response({"model_info": {"llama.context_length": 32768}}),
        ):
            await advertised_backend.prime_model_info()

        blocking_result = serving_backend._parse_generation_result(  # noqa: SLF001
            {
                "model": "llama3.2",
                "message": {"content": "partial"},
                "done": True,
                "done_reason": "length",
            },
            tools=None,
            use_native_tools=True,
        )
        stream_lines = [
            json.dumps({"message": {"content": "partial"}, "done": False}),
            json.dumps({"message": {"content": ""}, "done": True, "done_reason": "length"}),
        ]
        with patch.object(
            serving_backend._client,  # noqa: SLF001
            "stream",
            return_value=_MockStreamContext(stream_lines),
        ):
            stream_chunks = [
                chunk
                async for chunk in serving_backend._stream_generate(  # noqa: SLF001
                    {"model": "llama3.2", "messages": [], "stream": True}
                )
            ]
        serving_info = serving_backend.get_model_info()
        advertised_info = advertised_backend.get_model_info()
    finally:
        await serving_backend.close()
        await advertised_backend.close()

    assert serving_info.context_length == 4096
    assert serving_info.metadata["serving_context_length"] == 4096
    assert serving_info.metadata["advertised_context_length"] == 32768
    assert serving_info.metadata["effective_context_source"] == "serving"
    assert serving_info.metadata["effective_context_confidence"] == "high"
    assert serving_info.metadata["effective_context_is_hard_limit"] is True
    assert advertised_info.context_length == 32768
    assert advertised_info.metadata["effective_context_source"] == "advertised"
    assert advertised_info.metadata["effective_context_confidence"] == "medium"
    assert advertised_info.metadata["effective_context_is_hard_limit"] is False
    assert sum(chunk.is_final for chunk in stream_chunks) == 1
    stream_reason = next(chunk.finish_reason for chunk in stream_chunks if chunk.is_final)
    _assert_truncation_terminal(blocking_result.finish_reason)
    _assert_truncation_terminal(stream_reason)
    assert serving_backend.supports_tool_calling() is True
    assert not hasattr(serving_backend, "cancel_generation")
    return _ProviderQualification(
        provider_id="ollama",
        capabilities={
            "context": "supported",
            "tools": "supported",
            "malformed_tool": "supported",
            "truncation": "supported",
            "streaming": "supported",
            "cancellation": "unsupported",
        },
    )


@pytest.mark.asyncio
async def test_provider_qualification_matrix_uses_real_adapter_mappings() -> None:
    report = [await _qualify_openai_compat(), await _qualify_ollama()]

    _assert_complete_provider_report(report)


def test_provider_qualification_report_rejects_missing_or_unknown_capabilities() -> None:
    report = [
        _ProviderQualification(
            provider_id="openai_compat",
            capabilities={
                "context": "supported",
                "tools": "supported",
                "malformed_tool": "supported",
                "truncation": "supported",
                "streaming": "supported",
                "cancellation": "unsupported",
            },
        ),
        _ProviderQualification(
            provider_id="ollama",
            capabilities={
                "context": "supported",
                "tools": "supported",
                "malformed_tool": "supported",
                "truncation": "supported",
                "streaming": "unsupported",
                "cancellation": "unsupported",
            },
        ),
    ]

    _assert_complete_provider_report(report)
    with pytest.raises(AssertionError, match="provider rows"):
        _assert_complete_provider_report(report[:-1])
    unknown_status = _ProviderQualification(
        provider_id="ollama",
        capabilities={**report[1].capabilities, "streaming": "implicit"},
    )
    with pytest.raises(AssertionError, match="unsupported capability statuses"):
        _assert_complete_provider_report([report[0], unknown_status])


class _ScriptedBackend(BaseLLMBackend):
    def __init__(self, outcomes: list[GenerationResult | Exception]) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[tuple[list[Message], list[ToolSchema]]] = []

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
    ) -> GenerationResult:
        del (
            temperature,
            max_tokens,
            top_p,
            min_p,
            top_k,
            frequency_penalty,
            presence_penalty,
            repeat_penalty,
            reasoning_effort,
            stream,
        )
        self.calls.append((list(messages), list(tools or [])))
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="scripted-terminal-backend", backend_type="test")

    async def health_check(self) -> bool:
        return True


async def _run_loop(
    backend: _ScriptedBackend,
    *,
    registry: ToolRegistry | None = None,
    workspace_dir: str = ".",
) -> list[object]:
    loop = AsyncReActLoop(
        backend=backend,
        tool_registry=registry,
        tool_execution_context=ToolExecutionContext(workspace_dir=workspace_dir),
        max_iterations=4,
    )
    return [
        event
        async for event in loop.run(
            system_prompt="system",
            history=[],
            user_message="answer the request",
        )
    ]


@pytest.mark.asyncio
async def test_incomplete_response_continues_with_nonduplicating_merge() -> None:
    backend = _ScriptedBackend(
        [
            GenerationResult(content="Partial answer", finish_reason="response.incomplete"),
            GenerationResult(content=" answer with the remainder", finish_reason="stop"),
        ]
    )

    events = await _run_loop(backend)

    final = next(event for event in events if isinstance(event, FinalAnswerEvent))

    assert final.content == "Partial answer with the remainder"
    assert final.metadata["truncated"] is True
    assert final.metadata["recovery_attempts"] == 1
    assert final.metadata["continuation_overlap_chars"] == len(" answer")
    assert final.metadata["recoverability"] == "recovered"


@pytest.mark.asyncio
async def test_malformed_textual_tool_markup_repairs_once_without_leaking() -> None:
    malformed = '<tool_call>{"name": "file_read", "arguments": {"path": "x"}'
    backend = _ScriptedBackend(
        [
            GenerationResult(content=malformed, finish_reason="stop"),
            GenerationResult(content="Recovered visible answer.", finish_reason="stop"),
        ]
    )

    events = await _run_loop(backend)

    final = next(event for event in events if isinstance(event, FinalAnswerEvent))
    repair = next(
        event
        for event in events
        if isinstance(event, StatusEvent)
        and event.metadata.get("error_type") == "invalid_tool_call"
    )

    assert final.content == "Recovered visible answer."
    assert repair.metadata["error_type"] == "invalid_tool_call"
    assert repair.metadata["recovery_attempt"] == 1
    assert all(malformed not in str(getattr(event, "content", "")) for event in events)
    assert len(backend.calls) == 2


@pytest.mark.asyncio
async def test_repeated_malformed_tool_markup_becomes_terminal_without_leaking() -> None:
    malformed = '<tool_call>{"name": "file_read", "arguments": {"path": "x"}'
    backend = _ScriptedBackend(
        [
            GenerationResult(content=malformed, finish_reason="stop"),
            GenerationResult(content=malformed, finish_reason="stop"),
        ]
    )

    events = await _run_loop(backend)

    error = next(event for event in events if isinstance(event, ErrorEvent))

    assert error.code == "INVALID_TOOL_CALL"
    assert error.metadata["error_type"] == "invalid_tool_call"
    assert error.metadata["recovery_budget"]["attempts_used"] == 1
    assert len(backend.calls) == 2
    assert not any(isinstance(event, FinalAnswerEvent) for event in events)
    assert all(malformed not in str(getattr(event, "content", "")) for event in events)


@pytest.mark.asyncio
async def test_native_invalid_turn_and_truncation_share_one_recovery_budget(
    tmp_path: Path,
) -> None:
    (tmp_path / "sample.txt").write_text("tool payload", encoding="utf-8")
    backend = _ScriptedBackend(
        [
            BackendRequestError(
                "Ollama returned an invalid native tool turn.",
                metadata={
                    "backend_name": "ollama",
                    "tool_turn_reason": "thinking_only",
                    "rejected_finish_reason": "length",
                },
            ),
            GenerationResult(content="Partial after native repair", finish_reason="length"),
        ]
    )
    registry = ToolRegistry(discover_builtin=False)
    registry.register(FileReadTool(workspace_dir=tmp_path))

    events = await _run_loop(backend, registry=registry, workspace_dir=str(tmp_path))

    final = next(event for event in events if isinstance(event, FinalAnswerEvent))

    assert len(backend.calls) == 2
    assert all(tools for _, tools in backend.calls)
    assert final.content == "Partial after native repair"
    assert final.metadata["error_type"] == "output_truncated"
    assert final.metadata["recoverability"] == "partial"
    assert final.metadata["recovery_budget"]["attempts_used"] == 1


@pytest.mark.asyncio
async def test_structured_tool_calls_override_truncation_metadata(tmp_path: Path) -> None:
    (tmp_path / "sample.txt").write_text("tool payload", encoding="utf-8")
    backend = _ScriptedBackend(
        [
            GenerationResult(
                content="",
                tool_calls=[
                    ToolCall(
                        id="call-file-read",
                        name="file_read",
                        arguments={"path": "sample.txt", "line_numbers": False},
                    )
                ],
                finish_reason="response.incomplete",
            ),
            GenerationResult(content="Tool result summarized.", finish_reason="stop"),
        ]
    )
    registry = ToolRegistry(discover_builtin=False)
    registry.register(FileReadTool(workspace_dir=tmp_path))

    events = await _run_loop(backend, registry=registry, workspace_dir=str(tmp_path))

    final = next(event for event in events if isinstance(event, FinalAnswerEvent))

    assert final.content == "Tool result summarized."
    assert any(isinstance(event, ToolCallRequestEvent) for event in events)
    assert not any(isinstance(event, AssistantTruncatedEvent) for event in events)
    assert "truncated" not in final.metadata


@pytest.mark.asyncio
async def test_unknown_terminal_metadata_cannot_complete_a_final_answer() -> None:
    backend = _ScriptedBackend(
        [GenerationResult(content="Unsafe final answer", finish_reason="still_processing")]
    )

    events = await _run_loop(backend)

    error = next(event for event in events if isinstance(event, ErrorEvent))

    assert error.code == "UNKNOWN_GENERATION_TERMINAL"
    assert error.metadata["error_type"] == "unknown_generation_terminal"
    assert not any(isinstance(event, FinalAnswerEvent) for event in events)
