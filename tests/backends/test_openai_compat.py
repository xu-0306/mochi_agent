"""OpenAI-compatible backend tests."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from mochi.agents.generation_policy import GenerationTerminal, normalize_generation_terminal
from mochi.backends.openai_compat import OpenAICompatBackend
from mochi.backends.types import Message, ToolCall, ToolSchema

from ._support import _httpx_json_response, _mock_response


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


@pytest.mark.asyncio
async def test_openai_compat_gpt54_omits_unsupported_minimal_reasoning_effort() -> None:
    backend = OpenAICompatBackend(
        base_url="https://example.test/v1",
        model="gpt-5.4",
        provider="openai_compat",
    )
    response = _mock_response(
        {
            "model": "gpt-5.4",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "{}"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        }
    )

    try:
        with patch.object(
            backend._client,
            "post",
            new_callable=AsyncMock,
            return_value=response,
        ) as post:
            await backend.generate(
                messages=[Message(role="user", content="Return JSON.")],
                reasoning_effort="minimal",
                stream=False,
            )
    finally:
        await backend.close()

    assert post.await_args.args[0] == "https://example.test/v1/chat/completions"
    payload = post.await_args.kwargs["json"]
    assert "reasoning_effort" not in payload


@pytest.mark.asyncio
async def test_openai_compat_discovers_efforts_once_and_serializes_chat_effort() -> None:
    backend = OpenAICompatBackend("https://example.test/v1", "proxy-model")
    response = _mock_response(
        {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    )
    models = _mock_response(
        {
            "data": [
                {
                    "id": "proxy-model",
                    "capabilities": {"effort": {"supported": ["low", "max"]}},
                }
            ]
        }
    )
    try:
        with patch.object(backend._client, "get", new_callable=AsyncMock, return_value=models) as get, patch.object(
            backend._client, "post", new_callable=AsyncMock, return_value=response
        ) as post:
            await backend.generate([Message(role="user", content="one")], reasoning_effort="max")
            await backend.generate([Message(role="user", content="two")], reasoning_effort="max")
    finally:
        await backend.close()

    assert get.await_count == 1
    assert post.await_args.kwargs["json"]["reasoning_effort"] == "max"
    metadata = backend.get_model_info().metadata
    assert metadata["capability_source"] == "endpoint_metadata"
    assert metadata["capability_status"] == "resolved"
    assert metadata["supported_reasoning_efforts"] == ["low", "max"]


@pytest.mark.asyncio
async def test_openai_compat_standard_models_payload_is_unavailable_then_uses_registry() -> None:
    backend = OpenAICompatBackend("https://example.test/v1", "gpt-5.4")
    models = _mock_response(
        {"data": [{"id": "gpt-5.4", "object": "model", "owned_by": "openai"}]}
    )
    response = _mock_response(
        {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    )
    try:
        with patch.object(backend._client, "get", new_callable=AsyncMock, return_value=models), patch.object(
            backend._client, "post", new_callable=AsyncMock, return_value=response
        ):
            await backend.generate([Message(role="user", content="one")], reasoning_effort="xhigh")
    finally:
        await backend.close()

    metadata = backend.get_model_info().metadata
    assert metadata["capability_source"] == "registry"
    assert metadata["capability_status"] == "unavailable"
    assert metadata["supported_reasoning_efforts"][-1] == "xhigh"


@pytest.mark.asyncio
async def test_openai_compat_concurrent_generate_discovers_once() -> None:
    backend = OpenAICompatBackend("https://example.test/v1", "proxy-model")
    models = _mock_response(
        {"data": [{"id": "proxy-model", "capabilities": {"effort": {"low": True}}}]}
    )
    response = _mock_response(
        {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    )

    async def delayed_models(*args: object, **kwargs: object) -> httpx.Response:
        del args, kwargs
        await asyncio.sleep(0)
        return models

    try:
        with patch.object(backend._client, "get", new_callable=AsyncMock, side_effect=delayed_models) as get, patch.object(
            backend._client, "post", new_callable=AsyncMock, return_value=response
        ):
            await asyncio.gather(
                backend.generate([Message(role="user", content="one")]),
                backend.generate([Message(role="user", content="two")]),
            )
    finally:
        await backend.close()

    assert get.await_count == 1


@pytest.mark.asyncio
async def test_openai_compat_discovery_failure_fails_open_without_effort() -> None:
    backend = OpenAICompatBackend("https://example.test/v1", "unknown-model")
    response = _mock_response(
        {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    )
    try:
        with patch.object(backend._client, "get", new_callable=AsyncMock, side_effect=httpx.ReadTimeout("slow")), patch.object(
            backend._client, "post", new_callable=AsyncMock, return_value=response
        ) as post:
            await backend.generate([Message(role="user", content="one")], reasoning_effort="minimal")
    finally:
        await backend.close()

    assert "reasoning_effort" not in post.await_args.kwargs["json"]
    assert backend.get_model_info().metadata["capability_status"] == "failed"


@pytest.mark.asyncio
async def test_openai_compat_responses_serializes_only_discovered_effort() -> None:
    backend = OpenAICompatBackend("https://example.test/v1/responses", "proxy-model")
    models = _mock_response(
        {"data": [{"id": "proxy-model", "capabilities": {"effort": {"supported": ["max"]}}}]}
    )
    response = _mock_response({"output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}]})
    try:
        with patch.object(backend._client, "get", new_callable=AsyncMock, return_value=models), patch.object(
            backend._client, "post", new_callable=AsyncMock, return_value=response
        ) as post:
            await backend.generate([Message(role="user", content="one")], reasoning_effort="max")
    finally:
        await backend.close()

    assert post.await_args.kwargs["json"]["reasoning"]["effort"] == "max"


@pytest.mark.asyncio
async def test_openai_compat_configured_capability_metadata_overrides_discovery() -> None:
    backend = OpenAICompatBackend(
        "https://example.test/v1",
        "proxy-model",
        capability_metadata={"capabilities": {"effort": {"supported": ["max"]}}},
    )
    response = _mock_response(
        {"choices": [{"message": {"role": "assistant", "content": "ok"}}]}
    )
    try:
        with patch.object(backend._client, "get", new_callable=AsyncMock) as get, patch.object(
            backend._client, "post", new_callable=AsyncMock, return_value=response
        ) as post:
            await backend.generate([Message(role="user", content="one")], reasoning_effort="max")
    finally:
        await backend.close()

    assert get.await_count == 0
    assert post.await_args.kwargs["json"]["reasoning_effort"] == "max"
    assert backend.get_model_info().metadata["capability_source"] == "configured_override"


@pytest.mark.asyncio
async def test_openai_compat_vllm_falls_back_when_auto_tool_choice_is_disabled() -> None:
    backend = OpenAICompatBackend(
        base_url="http://localhost:8000/v1",
        model="google/gemma-4-26B-A4B-it",
        provider="vllm",
    )
    request = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
    error_response = httpx.Response(
        400,
        request=request,
        json={
            "error": {
                "message": '"auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set',
            }
        },
    )
    status_error = httpx.HTTPStatusError(
        "400 Bad Request",
        request=request,
        response=error_response,
    )
    success_response = _mock_response(
        {
            "model": "google/gemma-4-26B-A4B-it",
            "choices": [{"message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        }
    )

    try:
        with patch.object(
            backend._client,
            "post",
            new_callable=AsyncMock,
            side_effect=[status_error, success_response],
        ) as post:
            result = await backend.generate(
                messages=[Message(role="user", content="hi")],
                tools=[
                    ToolSchema(
                        name="web_search",
                        description="Search the web",
                        parameters={"type": "object", "properties": {}},
                    )
                ],
                stream=False,
            )
    finally:
        await backend.close()

    assert result.content == "ok"
    assert backend.supports_tool_calling() is True
    assert "tools" in post.await_args_list[0].kwargs["json"]
    assert "tools" not in post.await_args_list[1].kwargs["json"]
    diagnostics = backend.get_model_info().metadata["fallback_diagnostics"]
    assert any(
        item["name"] == "native_tool_calling_disabled"
        and item["reason"] == "rejected_missing_parser"
        and item["from"] == "native"
        and item["to"] == "simulated_fallback"
        for item in diagnostics
    )

@pytest.mark.asyncio
async def test_openai_compat_falls_back_when_provider_rejects_native_tools() -> None:
    backend = OpenAICompatBackend(
        base_url="https://example.test/v1",
        model="gpt-5.4",
        provider="openai_compat",
    )
    request = httpx.Request("POST", "https://example.test/v1/chat/completions")
    error_response = httpx.Response(
        403,
        request=request,
        json={
            "error": {
                "message": "status 403",
                "type": "permission_error",
                "code": "insufficient_quota",
            }
        },
    )
    status_error = httpx.HTTPStatusError(
        "403 Forbidden",
        request=request,
        response=error_response,
    )
    success_response = _mock_response(
        {
            "model": "gpt-5.4",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '<tool_call>{"name":"web_search","arguments":{"query":"台中 天氣"}}</tool_call>',
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        }
    )

    try:
        with patch.object(
            backend._client,
            "post",
            new_callable=AsyncMock,
            side_effect=[status_error, success_response],
        ) as post:
            result = await backend.generate(
                messages=[Message(role="system", content="You are helpful."), Message(role="user", content="查天氣")],
                tools=[
                    ToolSchema(
                        name="web_search",
                        description="Search the web",
                        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                    )
                ],
                stream=False,
            )
    finally:
        await backend.close()

    first_payload = post.await_args_list[0].kwargs["json"]
    retry_payload = post.await_args_list[1].kwargs["json"]
    assert "tools" in first_payload
    assert "tools" not in retry_payload
    if "messages" in retry_payload:
        assert "## Tool Use Instructions" in retry_payload["messages"][0]["content"]
    else:
        assert "## Tool Use Instructions" in retry_payload["instructions"]
    assert backend.supports_tool_calling() is True
    assert result.content == ""
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "web_search"
    assert result.tool_calls[0].arguments == {"query": "台中 天氣"}
    assert backend.get_model_info().metadata["native_tool_calling_status"] == "native_tools_rejected_by_provider"

@pytest.mark.asyncio
async def test_openai_compat_uses_simulated_tool_mode_after_vllm_fallback() -> None:
    backend = OpenAICompatBackend(
        base_url="http://localhost:8000/v1",
        model="google/gemma-4-26B-A4B-it",
        provider="vllm",
    )
    backend._tool_state.active_mode = "simulated_fallback"  # noqa: SLF001
    success_response = _mock_response(
        {
            "model": "google/gemma-4-26B-A4B-it",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": '<tool_call>{"name":"web_search","arguments":{"query":"Mochi AI"}}</tool_call>',
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        }
    )

    try:
        with patch.object(
            backend._client,
            "post",
            new_callable=AsyncMock,
            return_value=success_response,
        ) as post:
            result = await backend.generate(
                messages=[Message(role="system", content="You are helpful."), Message(role="user", content="hi")],
                tools=[
                    ToolSchema(
                        name="web_search",
                        description="Search the web",
                        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                    )
                ],
                stream=False,
            )
    finally:
        await backend.close()

    payload = post.await_args.kwargs["json"]
    assert "tools" not in payload
    assert "## Tool Use Instructions" in payload["messages"][0]["content"]
    assert result.content == ""
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "web_search"
    assert result.tool_calls[0].arguments == {"query": "Mochi AI"}
    assert result.finish_reason == "tool_calls"

@pytest.mark.asyncio
async def test_openai_compat_flattens_tool_messages_in_simulated_mode() -> None:
    backend = OpenAICompatBackend(
        base_url="http://localhost:8000/v1",
        model="google/gemma-4-26B-A4B-it",
        provider="vllm",
    )
    backend._tool_state.active_mode = "simulated_fallback"  # noqa: SLF001
    success_response = _mock_response(
        {
            "model": "google/gemma-4-26B-A4B-it",
            "choices": [{"message": {"role": "assistant", "content": "done"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        }
    )

    try:
        with patch.object(
            backend._client,
            "post",
            new_callable=AsyncMock,
            return_value=success_response,
        ) as post:
            await backend.generate(
                messages=[
                    Message(
                        role="assistant",
                        content="",
                        tool_calls=[ToolCall(id="call-1", name="web_search", arguments={"query": "Mochi AI"})],
                    ),
                    Message(
                        role="tool",
                        content="found: Mochi AI",
                        tool_call_id="call-1",
                        name="web_search",
                    ),
                    Message(role="user", content="continue"),
                ],
                tools=[
                    ToolSchema(
                        name="web_search",
                        description="Search the web",
                        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                    )
                ],
                stream=False,
            )
    finally:
        await backend.close()

    payload_messages = post.await_args.kwargs["json"]["messages"]
    assert all("tool_calls" not in message for message in payload_messages)
    assert any(
        message["role"] == "assistant" and "Tool request: web_search" in message["content"]
        for message in payload_messages
    )
    assert any(
        message["role"] == "user" and message["content"].startswith("Tool web_search result:\nfound: Mochi AI")
        for message in payload_messages
    )

@pytest.mark.asyncio
async def test_openai_compat_simulated_thinking_only_turn_marks_backend_unavailable() -> None:
    backend = OpenAICompatBackend(
        base_url="http://localhost:8000/v1",
        model="google/gemma-4-26B-A4B-it",
        provider="vllm",
    )
    backend._tool_state.active_mode = "simulated_fallback"  # noqa: SLF001
    response = _mock_response(
        {
            "model": "google/gemma-4-26B-A4B-it",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": "",
                        "reasoning": "still deciding",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 4},
        }
    )

    try:
        with patch.object(  # noqa: SIM117
            backend._client,
            "post",
            new_callable=AsyncMock,
            return_value=response,
        ):
            with pytest.raises(RuntimeError, match="invalid tool-eligible turn"):
                await backend.generate(
                    messages=[Message(role="system", content="You are helpful."), Message(role="user", content="hi")],
                    tools=[
                        ToolSchema(
                            name="web_search",
                            description="Search the web",
                            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
                        )
                    ],
                    stream=False,
                )
    finally:
        await backend.close()

    metadata = backend.get_model_info().metadata
    assert metadata["tool_call_mode"] == "unavailable"
    assert metadata["native_tool_calling_status"] == "simulated_protocol_rejected"

@pytest.mark.asyncio
async def test_openai_compat_probe_tool_calling_reports_supported() -> None:
    backend = OpenAICompatBackend(
        base_url="http://localhost:8000/v1",
        model="google/gemma-4-26B-A4B-it",
        provider="vllm",
    )
    success_response = _mock_response(
        {
            "model": "google/gemma-4-26B-A4B-it",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "probe-call-1",
                                "function": {
                                    "name": "mochi_tool_probe",
                                    "arguments": '{"value":"ok"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
    )

    try:
        with patch.object(
            backend._client,
            "post",
            new_callable=AsyncMock,
            return_value=success_response,
        ) as post:
            result = await backend.probe_tool_calling()
    finally:
        await backend.close()

    assert result is not None
    assert result["status"] == "supported"
    assert backend.supports_tool_calling() is True
    payload = post.await_args.kwargs["json"]
    assert payload["tool_choice"] == "auto"
    assert len(payload["tools"]) == 1

@pytest.mark.asyncio
async def test_openai_compat_probe_tool_calling_reenables_native_mode_after_fallback() -> None:
    backend = OpenAICompatBackend(
        base_url="http://localhost:8000/v1",
        model="google/gemma-4-26B-A4B-it",
        provider="vllm",
    )
    backend._tool_state.active_mode = "simulated_fallback"  # noqa: SLF001
    success_response = _mock_response(
        {
            "model": "google/gemma-4-26B-A4B-it",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "probe-call-1",
                                "function": {
                                    "name": "mochi_tool_probe",
                                    "arguments": '{"value":"ok"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
        }
    )

    try:
        with patch.object(
            backend._client,
            "post",
            new_callable=AsyncMock,
            return_value=success_response,
        ):
            result = await backend.probe_tool_calling()
    finally:
        await backend.close()

    assert result is not None
    assert result["status"] == "supported"
    assert backend.supports_tool_calling() is True
    assert backend.get_model_info().metadata["tool_call_mode"] == "native"
    diagnostics = backend.get_model_info().metadata["fallback_diagnostics"]
    assert any(
        item["name"] == "native_tool_calling_recovered"
        and item["reason"] == "supported"
        and item["from"] == "simulated_fallback"
        and item["to"] == "native"
        for item in diagnostics
    )

@pytest.mark.asyncio
async def test_openai_compat_probe_switches_to_responses_when_chat_tools_fail() -> None:
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="gpt-5.4",
        provider="openai_compat",
    )
    chat_error = _httpx_json_response(
        "https://api.example.com/v1/chat/completions",
        403,
        {"error": {"type": "permission_error", "code": "insufficient_quota"}},
    )
    responses_ok = _httpx_json_response(
        "https://api.example.com/v1/responses",
        200,
        {
            "output": [
                {
                    "type": "function_call",
                    "call_id": "probe-call-1",
                    "name": "mochi_tool_probe",
                    "arguments": '{"value":"ok"}',
                }
            ]
        },
    )

    try:
        with patch.object(
            backend._client,
            "post",
            new_callable=AsyncMock,
            side_effect=[chat_error, responses_ok],
        ) as post:
            result = await backend.probe_tool_calling()
    finally:
        await backend.close()

    assert result is not None
    assert result["status"] == "supported"
    assert result["tool_protocol"] == "responses"
    assert backend.supports_tool_calling() is True
    metadata = backend.get_model_info().metadata
    assert metadata["api_mode"] == "responses"
    assert metadata["request_shape"] == "responses"
    assert metadata["tool_calling_protocol"] == "responses"
    assert metadata["tool_protocol_probe"]["selected_protocol"] == "responses"
    assert post.await_args_list[0].args[0] == "https://api.example.com/v1/chat/completions"
    assert post.await_args_list[1].args[0] == "https://api.example.com/v1/responses"

@pytest.mark.asyncio
async def test_openai_compat_probe_marks_tools_unavailable_when_all_openai_protocols_are_rejected() -> None:
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="gpt-5.4",
        provider="openai_compat",
    )
    chat_error = _httpx_json_response(
        "https://api.example.com/v1/chat/completions",
        403,
        {"error": {"type": "permission_error", "code": "insufficient_quota"}},
    )
    responses_error = _httpx_json_response(
        "https://api.example.com/v1/responses",
        429,
        {"error": {"type": "usage_limit_reached", "message": "The usage limit has been reached"}},
    )

    try:
        with patch.object(
            backend._client,
            "post",
            new_callable=AsyncMock,
            side_effect=[chat_error, responses_error],
        ):
            result = await backend.probe_tool_calling()
    finally:
        await backend.close()

    assert result is not None
    assert result["status"] == "all_tool_protocols_rejected_by_provider"
    assert backend.supports_tool_calling() is False
    metadata = backend.get_model_info().metadata
    assert metadata["tool_call_mode"] == "unavailable"
    assert metadata["tool_calling_blocked"] is True
    assert metadata["tool_protocol_probe"]["selected_protocol"] is None


@pytest.mark.asyncio
async def test_openai_compat_effective_context_prefers_observed_serving_limit() -> None:
    backend = OpenAICompatBackend(
        "https://example.test/v1",
        "proxy-model",
        configured_context_length=8192,
    )
    models = _mock_response({"data": [{"id": "proxy-model", "context_length": 32768}]})
    try:
        with patch.object(backend._client, "get", new_callable=AsyncMock, return_value=models):
            await backend._ensure_capability_discovery()  # noqa: SLF001
        backend._parse_chat_completions_result(  # noqa: SLF001
            {
                "context_length": 4096,
                "choices": [{"message": {"content": "partial"}, "finish_reason": "length"}],
            }
        )
    finally:
        await backend.close()

    info = backend.get_model_info()

    assert info.context_length == 4096
    assert info.metadata["configured_context_length"] == 8192
    assert info.metadata["serving_context_length"] == 4096
    assert info.metadata["advertised_context_length"] == 32768
    assert info.metadata["context_source"] == "serving"
    assert info.metadata["context_confidence"] == "high"
    assert info.metadata["context_is_hard_limit"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("case", "chat_finish_reason", "responses_payload", "responses_event", "expected_terminal"),
    [
        (
            "completion",
            "stop",
            {"output_text": "done", "status": "completed"},
            {"type": "response.completed", "response": {"status": "completed"}},
            GenerationTerminal.COMPLETE,
        ),
        (
            "truncation",
            "length",
            {
                "output_text": "partial",
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
            },
            {
                "type": "response.incomplete",
                "response": {
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                },
            },
            GenerationTerminal.OUTPUT_TRUNCATED,
        ),
        (
            "unreliable_metadata",
            None,
            {"output_text": "unverified"},
            None,
            GenerationTerminal.UNKNOWN,
        ),
    ],
)
async def test_openai_compat_terminal_semantics_are_symmetric_for_streaming_and_blocking(
    case: str,
    chat_finish_reason: str | None,
    responses_payload: dict[str, object],
    responses_event: dict[str, object] | None,
    expected_terminal: GenerationTerminal,
) -> None:
    del case
    chat_data: dict[str, object] = {
        "choices": [{"message": {"content": "chat output"}}],
    }
    if chat_finish_reason is not None:
        chat_data["choices"] = [
            {"message": {"content": "chat output"}, "finish_reason": chat_finish_reason}
        ]

    chat_backend = OpenAICompatBackend("https://example.test/v1", "proxy-model")
    responses_backend = OpenAICompatBackend("https://example.test/v1/responses", "proxy-model")
    try:
        chat_blocking = chat_backend._parse_chat_completions_result(chat_data)  # noqa: SLF001
        responses_blocking = responses_backend._parse_responses_result(responses_payload)  # noqa: SLF001

        chat_lines = [
            'data: {"choices":[{"delta":{"content":"prefix"},"finish_reason":null}]}',
            f"data: {json.dumps(chat_data)}",
        ]
        if chat_finish_reason is None:
            chat_lines.append("data: [DONE]")
        with patch.object(chat_backend._client, "stream", return_value=_MockStreamContext(chat_lines)):  # noqa: SLF001
            chat_stream = [
                chunk
                async for chunk in chat_backend._stream_generate(  # noqa: SLF001
                    {"stream": True}, request_url=chat_backend._chat_completions_url  # noqa: SLF001
                )
            ]

        responses_lines = (
            [f"data: {json.dumps(responses_event)}"] if responses_event is not None else ["data: [DONE]"]
        )
        with patch.object(
            responses_backend._client,
            "stream",
            return_value=_MockStreamContext(responses_lines),
        ):  # noqa: SLF001
            responses_stream = [
                chunk
                async for chunk in responses_backend._stream_generate(  # noqa: SLF001
                    {"stream": True}, request_url=responses_backend._responses_url  # noqa: SLF001
                )
            ]
    finally:
        await chat_backend.close()
        await responses_backend.close()

    terminal_signals = [
        chat_blocking.finish_reason,
        responses_blocking.finish_reason,
        next(chunk.finish_reason for chunk in reversed(chat_stream) if chunk.is_final),
        next(chunk.finish_reason for chunk in reversed(responses_stream) if chunk.is_final),
    ]
    assert sum(chunk.is_final for chunk in chat_stream) == 1
    assert sum(chunk.is_final for chunk in responses_stream) == 1
    for terminal_signal in terminal_signals:
        decision = normalize_generation_terminal(
            terminal_signal=terminal_signal,
            has_structured_tool_calls=False,
            output="partial" if expected_terminal is GenerationTerminal.OUTPUT_TRUNCATED else "",
        )
        assert decision.terminal is expected_terminal
