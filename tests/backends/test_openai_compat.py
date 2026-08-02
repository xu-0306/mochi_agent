"""OpenAI-compatible backend tests."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from mochi.backends.openai_compat import OpenAICompatBackend
from mochi.backends.types import Message, ToolCall, ToolSchema

from ._support import _httpx_json_response, _mock_response


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
