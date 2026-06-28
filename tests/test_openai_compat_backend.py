"""OpenAICompatBackend 單元測試（使用 httpx mock）。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from mochi.backends.base import BackendRequestError
from mochi.backends.openai_compat import OpenAICompatBackend
from mochi.backends.types import Message, ResponsesReplayState, ToolCall, ToolSchema


@pytest.fixture
def backend() -> OpenAICompatBackend:
    """建立 OpenAICompatBackend 測試實例。"""
    return OpenAICompatBackend(
        base_url="http://localhost:8000/v1",
        model="gpt-test",
        api_key="test-key",
    )


def _mock_response(data: dict, *, status_code: int = 200) -> MagicMock:
    """建立 httpx Response mock。"""
    resp = MagicMock()
    resp.status_code = status_code
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


class _MockStreamContext:
    """模擬 httpx AsyncClient.stream 回傳的 async context manager。"""

    def __init__(self, lines: list[str]) -> None:
        self._lines = lines
        self.raise_for_status = MagicMock()

    async def __aenter__(self) -> _MockStreamContext:
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        return None

    async def aiter_lines(self) -> AsyncIterator[str]:
        for line in self._lines:
            yield line


@pytest.mark.asyncio
async def test_health_check_success(backend: OpenAICompatBackend) -> None:
    """health_check() 在 /models 回傳 200 時應回傳 True。"""
    mock_resp = _mock_response({"data": []})
    with patch.object(backend._client, "get", new_callable=AsyncMock, return_value=mock_resp):
        result = await backend.health_check()

    assert result is True


@pytest.mark.asyncio
async def test_health_check_fail(backend: OpenAICompatBackend) -> None:
    """health_check() 在連線失敗時應回傳 False。"""
    failing_backend = OpenAICompatBackend(base_url="http://localhost:8000", model="gpt-test")
    with patch.object(
        failing_backend._client,
        "get",
        new_callable=AsyncMock,
        side_effect=httpx.ConnectError("connection refused"),
    ):
        result = await failing_backend.health_check()

    assert result is False
    await failing_backend.close()


@pytest.mark.asyncio
async def test_health_check_treats_client_error_as_reachable() -> None:
    """4xx on /models should still count as a reachable endpoint for backend switching."""
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="gpt-test",
        api_key="test-key",
    )
    mock_resp = _mock_response({"error": "forbidden"}, status_code=403)

    try:
        with patch.object(backend._client, "get", new_callable=AsyncMock, return_value=mock_resp):
            result = await backend.health_check()
    finally:
        await backend.close()

    assert result is True


@pytest.mark.asyncio
async def test_health_check_returns_false_on_server_error_response() -> None:
    """5xx on /models should still reject the endpoint as unhealthy."""
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="gpt-test",
        api_key="test-key",
    )
    mock_resp = _mock_response({"error": "bad gateway"}, status_code=502)

    try:
        with patch.object(backend._client, "get", new_callable=AsyncMock, return_value=mock_resp):
            result = await backend.health_check()
    finally:
        await backend.close()

    assert result is False


@pytest.mark.asyncio
async def test_base_v1_url_completes_chat_completions_endpoint(backend: OpenAICompatBackend) -> None:
    """只輸入到 `/v1` 時才補 `/chat/completions`。"""
    response_data = {
        "model": "gpt-test",
        "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
    }
    mock_resp = _mock_response(response_data)

    with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_resp) as post:
        result = await backend.generate(messages=[Message(role="user", content="hi")], stream=False)

    assert result.content == "ok"
    assert post.await_args.args[0] == "http://localhost:8000/v1/chat/completions"


@pytest.mark.asyncio
async def test_base_v1_gpt5_prefers_responses_transport() -> None:
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="gpt-5",
        api_key="test-key",
        provider="openai_compat",
    )
    response_data = {
        "id": "resp_1",
        "model": "gpt-5",
        "status": "completed",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "content": [{"type": "output_text", "text": "responses ok"}],
            }
        ],
        "usage": {"input_tokens": 7, "output_tokens": 3},
    }
    mock_resp = _mock_response(response_data)

    try:
        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_resp) as post:
            result = await backend.generate(messages=[Message(role="user", content="hi")], stream=False)
    finally:
        await backend.close()

    assert result.content == "responses ok"
    assert post.await_args.args[0] == "https://api.example.com/v1/responses"
    assert "input" in post.await_args.kwargs["json"]
    assert backend.get_model_info().metadata["request_shape"] == "responses"
    assert backend.get_model_info().metadata["reasoning_transport_preference"] == "responses_preferred"


@pytest.mark.asyncio
async def test_full_responses_url_is_used_without_chat_completions_suffix() -> None:
    """完整 `/v1/responses` endpoint 應原樣使用，不再補 `/chat/completions`。"""
    backend = OpenAICompatBackend(
        base_url="https://co.yes.vg/v1/responses",
        model="gpt-test",
        api_key="test-key",
    )
    response_data = {
        "model": "gpt-test",
        "output_text": "responses ok",
        "usage": {"input_tokens": 7, "output_tokens": 3},
    }
    mock_resp = _mock_response(response_data)

    try:
        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_resp) as post:
            result = await backend.generate(messages=[Message(role="user", content="hi")], stream=False)
    finally:
        await backend.close()

    assert result.content == "responses ok"
    assert result.input_tokens == 7
    assert result.output_tokens == 3
    assert post.await_args.args[0] == "https://co.yes.vg/v1/responses"
    assert post.await_args.kwargs["headers"]["Authorization"] == "Bearer test-key"


@pytest.mark.asyncio
async def test_full_responses_url_health_check_uses_v1_models() -> None:
    """`/v1/responses` 健康檢查應查 `/v1/models`，不是 `/v1/responses/models`。"""
    backend = OpenAICompatBackend(
        base_url="https://co.yes.vg/v1/responses",
        model="gpt-test",
        api_key="test-key",
    )
    mock_resp = _mock_response({"data": []})

    try:
        with patch.object(backend._client, "get", new_callable=AsyncMock, return_value=mock_resp) as get:
            result = await backend.health_check()
    finally:
        await backend.close()

    assert result is True
    assert get.await_args.args[0] == "https://co.yes.vg/v1/models"


@pytest.mark.asyncio
async def test_generate_nonstream_content(backend: OpenAICompatBackend) -> None:
    """非串流生成應正確解析 content、usage 與 finish_reason。"""
    response_data = {
        "id": "chatcmpl-1",
        "model": "gpt-test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "你好，Mochi！"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 12, "completion_tokens": 6, "total_tokens": 18},
    }
    mock_resp = _mock_response(response_data)

    with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_resp):
        result = await backend.generate(messages=[Message(role="user", content="你好")], stream=False)

    assert result.content == "你好，Mochi！"
    assert result.input_tokens == 12
    assert result.output_tokens == 6
    assert result.finish_reason == "stop"
    assert result.model == "gpt-test"


@pytest.mark.asyncio
async def test_generate_nonstream_tool_calls(backend: OpenAICompatBackend) -> None:
    """非串流生成應正確解析 tool_calls。"""
    response_data = {
        "id": "chatcmpl-2",
        "model": "gpt-test",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "web_search",
                                "arguments": '{"query":"Mochi"}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
    }
    mock_resp = _mock_response(response_data)

    with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_resp):
        result = await backend.generate(messages=[Message(role="user", content="查資料")], stream=False)

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].id == "call_123"
    assert result.tool_calls[0].name == "web_search"
    assert result.tool_calls[0].arguments == {"query": "Mochi"}
    assert result.finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_generate_nonstream_reasoning_content_is_separated_from_answer(
    backend: OpenAICompatBackend,
) -> None:
    response_data = {
        "id": "chatcmpl-reasoning",
        "model": "gpt-test",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Final answer",
                    "reasoning_content": "step 1 -> step 2",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    mock_resp = _mock_response(response_data)

    with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_resp):
        result = await backend.generate(messages=[Message(role="user", content="hi")], stream=False)

    assert result.content == "Final answer"
    assert result.thinking == "step 1 -> step 2"


@pytest.mark.asyncio
async def test_generate_nonstream_reasoning_alias_is_separated_from_answer(
    backend: OpenAICompatBackend,
) -> None:
    response_data = {
        "id": "chatcmpl-reasoning",
        "model": "gpt-test",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Final answer",
                    "reasoning": "visible provider reasoning summary",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    mock_resp = _mock_response(response_data)

    with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_resp):
        result = await backend.generate(messages=[Message(role="user", content="hi")], stream=False)

    assert result.content == "Final answer"
    assert result.thinking == "visible provider reasoning summary"


@pytest.mark.asyncio
async def test_generate_nonstream_thinking_blocks_are_separated_from_answer(
    backend: OpenAICompatBackend,
) -> None:
    response_data = {
        "id": "chatcmpl-thinking-blocks",
        "model": "gpt-test",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Final answer",
                    "thinking_blocks": [
                        {"type": "thinking", "thinking": "first block"},
                        {"type": "thinking", "thinking": "second block"},
                    ],
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    mock_resp = _mock_response(response_data)

    with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_resp):
        result = await backend.generate(messages=[Message(role="user", content="hi")], stream=False)

    assert result.content == "Final answer"
    assert result.thinking == "first block\n\nsecond block"


@pytest.mark.asyncio
async def test_simulated_fallback_content_marks_protocol_validated() -> None:
    backend = OpenAICompatBackend(
        base_url="http://localhost:8000/v1",
        model="gpt-test",
        api_key="test-key",
    )
    backend._tool_state.active_mode = "simulated_fallback"  # noqa: SLF001
    response_data = {
        "id": "chatcmpl-simulated-content",
        "model": "gpt-test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Final answer"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    mock_resp = _mock_response(response_data)

    try:
        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_resp):
            result = await backend.generate(
                messages=[Message(role="user", content="hi")],
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

    assert result.content == "Final answer"
    assert backend.get_model_info().metadata["tool_call_mode"] == "simulated_fallback"
    assert backend.get_model_info().metadata["fallback_validation_status"] == "validated"


@pytest.mark.asyncio
async def test_simulated_fallback_thinking_only_marks_backend_unavailable() -> None:
    backend = OpenAICompatBackend(
        base_url="http://localhost:8000/v1",
        model="gpt-test",
        api_key="test-key",
    )
    backend._tool_state.active_mode = "simulated_fallback"  # noqa: SLF001
    response_data = {
        "id": "chatcmpl-simulated-thinking",
        "model": "gpt-test",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "reasoning_content": "still deciding",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }
    mock_resp = _mock_response(response_data)

    try:
        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_resp):
            with pytest.raises(BackendRequestError, match="invalid tool-eligible turn"):
                await backend.generate(
                    messages=[Message(role="user", content="hi")],
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


def test_blocked_simulated_retry_overwrites_stale_supported_probe_status() -> None:
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="gpt-5.4",
        api_key="test-key",
        provider="openai_compat",
    )
    backend._native_tool_probe = {"status": "supported", "message": "ok"}  # noqa: SLF001
    request = httpx.Request("POST", "https://api.example.com/v1/chat/completions")
    response = httpx.Response(403, request=request, text="tool permission_error")
    exc = httpx.HTTPStatusError("403", request=request, response=response)

    try:
        backend._record_tool_calling_blocked_from_retry(exc)  # noqa: SLF001
        metadata = backend.get_model_info().metadata
    finally:
        asyncio.run(backend.close())

    assert metadata["tool_call_mode"] == "unavailable"
    assert metadata["tool_calling_blocked"] is True
    assert metadata["native_tool_calling_status"] == "all_tool_protocols_rejected_by_provider"


def test_chat_completions_payload_serializes_tool_call_arguments_as_json(
    backend: OpenAICompatBackend,
) -> None:
    """Chat Completions assistant tool_calls.arguments 必須是合法 JSON 字串。"""
    payload = backend._build_chat_completions_payload(  # noqa: SLF001
        messages=[
            Message(
                role="assistant",
                content="",
                tool_calls=[
                    ToolCall(
                        id="call-1",
                        name="web_search",
                        arguments={"query": "台中 南區 天氣", "top_k": 5},
                    )
                ],
            )
        ],
        tools=None,
        temperature=0.7,
        max_tokens=256,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        reasoning_effort=None,
        stream=False,
    )

    raw_arguments = payload["messages"][0]["tool_calls"][0]["function"]["arguments"]

    assert raw_arguments == '{"query": "台中 南區 天氣", "top_k": 5}'


@pytest.mark.asyncio
async def test_responses_payload_includes_function_call_output_items() -> None:
    """Responses API tool result 應用 function_call_output 回送，不應降級成一般 user 文字。"""
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1/responses",
        model="gpt-test",
        api_key="test-key",
    )

    try:
        payload = backend._build_responses_payload(  # noqa: SLF001
            messages=[
                Message(role="system", content="你是測試助理"),
                Message(role="user", content="查天氣"),
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id="call-1",
                            name="web_search",
                            arguments={"query": "台中 南區 天氣"},
                        )
                    ],
                ),
                Message(
                    role="tool",
                    content='{"ok": true, "output": []}',
                    tool_call_id="call-1",
                    name="web_search",
                ),
            ],
            tools=None,
            temperature=0.7,
            max_tokens=256,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            reasoning_effort=None,
            stream=False,
        )
    finally:
        await backend.close()

    input_items = payload["input"]

    assert payload["instructions"] == "你是測試助理"
    assert input_items == [
        {"role": "user", "content": "查天氣"},
        {
            "type": "function_call",
            "call_id": "call-1",
            "name": "web_search",
            "arguments": '{"query": "台中 南區 天氣"}',
        },
        {
            "type": "function_call_output",
            "call_id": "call-1",
            "output": '{"ok": true, "output": []}',
        },
    ]


@pytest.mark.asyncio
async def test_responses_payload_maps_reasoning_effort_to_reasoning_object() -> None:
    """Responses API should receive normalized reasoning effort."""
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1/responses",
        model="gpt-test",
        api_key="test-key",
    )

    try:
        payload = backend._build_responses_payload(  # noqa: SLF001
            messages=[Message(role="user", content="Think briefly.")],
            tools=None,
            temperature=0.7,
            max_tokens=256,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            reasoning_effort="high",
            stream=False,
        )
    finally:
        await backend.close()

    assert payload["reasoning"] == {"effort": "high", "summary": "concise"}
    assert payload["include"] == ["reasoning.encrypted_content"]


@pytest.mark.asyncio
async def test_responses_payload_maps_xhigh_reasoning_effort_to_reasoning_object() -> None:
    """Responses API should pass through newer reasoning-effort levels when supported."""
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1/responses",
        model="gpt-5.2",
        api_key="test-key",
        provider="openai_compat",
    )

    try:
        payload = backend._build_responses_payload(  # noqa: SLF001
            messages=[Message(role="user", content="Think hard.")],
            tools=None,
            temperature=0.7,
            max_tokens=256,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0,
            reasoning_effort="xhigh",
            stream=False,
        )
    finally:
        await backend.close()

    assert payload["reasoning"] == {"effort": "xhigh", "summary": "concise"}


@pytest.mark.asyncio
async def test_responses_result_extracts_reasoning_summary() -> None:
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1/responses",
        model="gpt-test",
        api_key="test-key",
    )
    response_data = {
        "id": "resp-reasoning",
        "model": "gpt-test",
        "status": "completed",
        "output": [
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "searched official source"}],
            },
            {
                "type": "message",
                "id": "msg_1",
                "content": [{"type": "output_text", "text": "Final answer"}],
            },
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }
    mock_resp = _mock_response(response_data)

    try:
        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_resp):
            result = await backend.generate(messages=[Message(role="user", content="hi")], stream=False)
    finally:
        await backend.close()

    assert result.content == "Final answer"
    assert result.thinking == "searched official source"


@pytest.mark.asyncio
async def test_responses_result_captures_replay_state() -> None:
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1/responses",
        model="gpt-5",
        api_key="test-key",
    )
    response_data = {
        "id": "resp-replay",
        "model": "gpt-5",
        "status": "completed",
        "output": [
            {
                "type": "reasoning",
                "id": "rs_1",
                "summary": [{"type": "summary_text", "text": "checked docs"}],
                "encrypted_content": "enc_123",
            },
            {
                "type": "function_call",
                "id": "fc_1",
                "call_id": "call-1",
                "name": "web_search",
                "arguments": '{"query":"Mochi"}',
            },
            {
                "type": "message",
                "id": "msg_1",
                "content": [{"type": "output_text", "text": "Final answer"}],
            },
        ],
        "usage": {"input_tokens": 10, "output_tokens": 5},
    }

    try:
        result = backend._parse_responses_result(response_data)  # noqa: SLF001
    finally:
        await backend.close()

    assert result.responses_replay is not None
    assert result.responses_replay.response_id == "resp-replay"
    assert [item["type"] for item in result.responses_replay.assistant_output_items] == [
        "reasoning",
        "function_call",
        "message",
    ]
    assert result.responses_replay.encrypted_reasoning_content == "enc_123"
    assert result.responses_replay.summary_text == "checked docs"
    assert result.responses_replay.continuity_mode == "manual_encrypted"


@pytest.mark.asyncio
async def test_responses_retry_strips_rejected_summary_and_include_fields() -> None:
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1/responses",
        model="gpt-5",
        api_key="test-key",
    )
    request = httpx.Request("POST", "https://api.example.com/v1/responses")
    rejection = httpx.Response(
        400,
        request=request,
        text='Unknown field "reasoning.summary"; unsupported include reasoning.encrypted_content',
    )
    status_error = httpx.HTTPStatusError("bad request", request=request, response=rejection)
    success_response = _mock_response(
        {
            "id": "resp_2",
            "model": "gpt-5",
            "status": "completed",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "content": [{"type": "output_text", "text": "Recovered"}],
                }
            ],
            "usage": {"input_tokens": 4, "output_tokens": 2},
        }
    )

    try:
        with patch.object(
            backend._client,
            "post",
            new_callable=AsyncMock,
            side_effect=[status_error, success_response],
        ) as post:
            result = await backend.generate(messages=[Message(role="user", content="hi")], stream=False)
    finally:
        await backend.close()

    first_payload = post.await_args_list[0].kwargs["json"]
    second_payload = post.await_args_list[1].kwargs["json"]

    assert result.content == "Recovered"
    assert first_payload["reasoning"]["summary"] == "concise"
    assert first_payload["include"] == ["reasoning.encrypted_content"]
    assert "reasoning" not in second_payload
    assert "include" not in second_payload
    assert backend.get_model_info().metadata["reasoning_summary_supported"] is False
    diagnostics = backend.get_model_info().metadata["fallback_diagnostics"]
    assert any(item["name"] == "responses_reasoning_retry" for item in diagnostics)


@pytest.mark.asyncio
async def test_responses_input_uses_previous_response_id_for_incremental_continuity() -> None:
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1/responses",
        model="gpt-5",
        api_key="test-key",
    )

    try:
        input_state = backend._build_responses_input(  # noqa: SLF001
            [
                Message(role="user", content="Find Mochi"),
                Message(
                    role="assistant",
                    content="",
                    responses_replay=ResponsesReplayState(
                        response_id="resp_prev",
                        assistant_output_items=[
                            {
                                "type": "function_call",
                                "call_id": "call-1",
                                "name": "web_search",
                                "arguments": '{"query":"Mochi"}',
                            }
                        ],
                    ),
                ),
                Message(role="tool", content='{"ok": true}', tool_call_id="call-1", name="web_search"),
                Message(role="user", content="Summarize it"),
            ]
        )
    finally:
        await backend.close()

    assert input_state.previous_response_id == "resp_prev"
    assert input_state.continuity_mode == "previous_response_id"
    assert input_state.input_items == [
        {"type": "function_call_output", "call_id": "call-1", "output": '{"ok": true}'},
        {"role": "user", "content": "Summarize it"},
    ]


@pytest.mark.asyncio
async def test_responses_input_manually_replays_assistant_items_before_tool_outputs() -> None:
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1/responses",
        model="gpt-5",
        api_key="test-key",
    )
    backend._responses_chat_completions_alias = True  # noqa: SLF001

    try:
        input_state = backend._build_responses_input(  # noqa: SLF001
            [
                Message(role="user", content="Find Mochi"),
                Message(
                    role="assistant",
                    content="",
                    responses_replay=ResponsesReplayState(
                        assistant_output_items=[
                            {
                                "type": "reasoning",
                                "summary": [{"type": "summary_text", "text": "checked docs"}],
                                "encrypted_content": "enc_123",
                            },
                            {
                                "type": "function_call",
                                "call_id": "call-1",
                                "name": "web_search",
                                "arguments": '{"query":"Mochi"}',
                            },
                        ],
                        encrypted_reasoning_content="enc_123",
                        summary_text="checked docs",
                        continuity_mode="manual_encrypted",
                    ),
                ),
                Message(role="tool", content='{"ok": true}', tool_call_id="call-1", name="web_search"),
                Message(role="user", content="Summarize it"),
            ]
        )
    finally:
        await backend.close()

    assert input_state.previous_response_id is None
    assert input_state.continuity_mode == "manual_encrypted"
    assert input_state.replayed_items == 2
    assert [item.get("type") or item.get("role") for item in input_state.input_items] == [
        "user",
        "reasoning",
        "function_call",
        "function_call_output",
        "user",
    ]


def test_chat_completions_payload_omits_reasoning_effort(
    backend: OpenAICompatBackend,
) -> None:
    """Chat Completions should not receive Responses-only reasoning controls."""
    payload = backend._build_chat_completions_payload(  # noqa: SLF001
        messages=[Message(role="user", content="hi")],
        tools=None,
        temperature=0.7,
        max_tokens=256,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        reasoning_effort="high",
        stream=False,
    )

    assert "reasoning" not in payload


@pytest.mark.asyncio
async def test_generate_stream_chunk(backend: OpenAICompatBackend) -> None:
    """串流生成應輸出可用的 StreamChunk 並包含 final chunk。"""
    lines = [
        'data: {"choices":[{"delta":{"content":"Hel"},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{"content":"lo"},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
        "data: [DONE]",
    ]
    mock_stream = _MockStreamContext(lines)

    with patch.object(backend._client, "stream", new=MagicMock(return_value=mock_stream)):
        stream_iter = await backend.generate(
            messages=[Message(role="user", content="Say hello")],
            stream=True,
        )

        chunks = []
        async for chunk in stream_iter:
            chunks.append(chunk)

    assert len(chunks) >= 3
    assert chunks[0].delta == "Hel"
    assert chunks[1].delta == "lo"
    assert chunks[-1].is_final is True
    assert chunks[-1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_generate_stream_tool_call_delta_keeps_index_and_accumulates_arguments(
    backend: OpenAICompatBackend,
) -> None:
    lines = [
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1","function":{"name":"web_search","arguments":"{\\"query\\":\\"Mo"}}]},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"function":{"arguments":"chi\\"}"}}]},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"tool_calls"}]}',
        "data: [DONE]",
    ]
    mock_stream = _MockStreamContext(lines)

    with patch.object(backend._client, "stream", new=MagicMock(return_value=mock_stream)):
        stream_iter = await backend.generate(
            messages=[Message(role="user", content="Search Mochi")],
            stream=True,
        )

        chunks = []
        async for chunk in stream_iter:
            chunks.append(chunk)

    tool_chunks = [chunk for chunk in chunks if chunk.tool_call_delta is not None]
    assert len(tool_chunks) == 2
    assert tool_chunks[0].tool_call_delta is not None
    assert tool_chunks[0].tool_call_delta.index == 0
    assert tool_chunks[0].tool_call_delta.name == "web_search"
    assert tool_chunks[1].tool_call_delta is not None
    assert tool_chunks[1].tool_call_delta.index == 0
    assert tool_chunks[1].tool_call_delta.arguments == {"query": "Mochi"}
    assert chunks[-1].is_final is True
    assert chunks[-1].finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_http_status_errors_are_wrapped_with_backend_metadata() -> None:
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1/responses",
        model="gpt-test",
        api_key="test-key",
    )
    request = httpx.Request("POST", "https://api.example.com/v1/responses")
    response = httpx.Response(400, request=request, text="backend rejected 1.txt")

    try:
        with patch.object(
            backend._client,
            "post",
            new_callable=AsyncMock,
            side_effect=httpx.HTTPStatusError("bad request", request=request, response=response),
        ):
            with pytest.raises(BackendRequestError) as exc_info:
                await backend.generate(messages=[Message(role="user", content="hi")], stream=False)
    finally:
        await backend.close()

    exc = exc_info.value
    assert exc.metadata["backend_name"] == "openai_compat"
    assert exc.metadata["api_mode"] == "responses"
    assert exc.metadata["status_code"] == 400


@pytest.mark.asyncio
async def test_generate_stream_reasoning_content_is_separated_from_text(
    backend: OpenAICompatBackend,
) -> None:
    lines = [
        'data: {"choices":[{"delta":{"reasoning_content":"plan "},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{"reasoning_content":"more"},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{"content":"answer"},"finish_reason":null}]}',
        'data: {"choices":[{"delta":{},"finish_reason":"stop"}]}',
        "data: [DONE]",
    ]
    mock_stream = _MockStreamContext(lines)

    with patch.object(backend._client, "stream", new=MagicMock(return_value=mock_stream)):
        stream_iter = await backend.generate(
            messages=[Message(role="user", content="Say hello")],
            stream=True,
        )

        chunks = []
        async for chunk in stream_iter:
            chunks.append(chunk)

    assert [(chunk.delta, chunk.thinking_delta) for chunk in chunks[:-1]] == [
        ("", "plan "),
        ("", "more"),
        ("answer", ""),
    ]
    assert chunks[-1].is_final is True
    assert chunks[-1].finish_reason == "stop"


@pytest.mark.asyncio
async def test_generate_stream_http_status_error_reads_stream_body(
    backend: OpenAICompatBackend,
) -> None:
    """Streaming HTTP errors should keep backend diagnostics instead of raising ResponseNotRead."""
    request = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
    response = httpx.Response(
        429,
        request=request,
        stream=httpx.ByteStream(b"rate limited"),
    )
    status_error = httpx.HTTPStatusError(
        "too many requests",
        request=request,
        response=response,
    )
    mock_stream = _MockStreamContext([])
    mock_stream.raise_for_status = MagicMock(side_effect=status_error)

    with patch.object(backend._client, "stream", new=MagicMock(return_value=mock_stream)):
        stream_iter = await backend.generate(
            messages=[Message(role="user", content="Say hello")],
            stream=True,
        )

        with pytest.raises(BackendRequestError) as exc_info:
            async for _chunk in stream_iter:
                pass

    exc = exc_info.value
    assert exc.metadata["status_code"] == 429
    assert exc.metadata["response_text"] == "rate limited"


@pytest.mark.asyncio
async def test_responses_alias_switches_future_requests_to_chat_payload() -> None:
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1/responses",
        model="gpt-test",
        api_key="test-key",
    )
    alias_response_1 = _mock_response(
        {
            "model": "gpt-test",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "alias ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 4},
        }
    )
    alias_response_2 = _mock_response(
        {
            "model": "gpt-test",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "alias still ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 9, "completion_tokens": 3},
        }
    )

    try:
        with patch.object(
            backend._client,
            "post",
            new_callable=AsyncMock,
            side_effect=[alias_response_1, alias_response_2],
        ) as post:
            result_1 = await backend.generate(
                messages=[Message(role="user", content="hi")],
                stream=False,
            )
            result_2 = await backend.generate(
                messages=[Message(role="user", content="hello again")],
                stream=False,
            )
    finally:
        await backend.close()

    assert result_1.content == "alias ok"
    assert result_2.content == "alias still ok"
    assert "input" in post.await_args_list[0].kwargs["json"]
    assert "messages" not in post.await_args_list[0].kwargs["json"]
    assert "messages" in post.await_args_list[1].kwargs["json"]
    assert backend.get_model_info().metadata["request_shape"] == "chat_completions"
    assert backend.get_model_info().metadata["responses_alias_detected"] is True
    diagnostics = backend.get_model_info().metadata["fallback_diagnostics"]
    assert any(
        item["name"] == "responses_alias_transport"
        and item["category"] == "transport"
        and item["from"] == "responses"
        and item["to"] == "chat_completions"
        for item in diagnostics
    )


@pytest.mark.asyncio
async def test_responses_alias_empty_result_retries_with_chat_payload_and_tools() -> None:
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1/responses",
        model="gpt-test",
        api_key="test-key",
    )
    empty_response = _mock_response(
        {
            "model": "gpt-test",
            "status": "completed",
            "output": [],
            "usage": {"input_tokens": 5, "output_tokens": 0},
        }
    )
    alias_tool_response = _mock_response(
        {
            "model": "gpt-test",
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": "call_123",
                                "type": "function",
                                "function": {
                                    "name": "web_search",
                                    "arguments": '{"query":"Mochi"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 3},
        }
    )
    tool = ToolSchema(
        name="web_search",
        description="Search the web",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
    )

    try:
        with patch.object(
            backend._client,
            "post",
            new_callable=AsyncMock,
            side_effect=[empty_response, alias_tool_response],
        ) as post:
            result = await backend.generate(
                messages=[Message(role="user", content="search Mochi")],
                tools=[tool],
                stream=False,
            )
    finally:
        await backend.close()

    first_payload = post.await_args_list[0].kwargs["json"]
    second_payload = post.await_args_list[1].kwargs["json"]

    assert "input" in first_payload
    assert first_payload["tools"] == [
        {
            "type": "function",
            "name": "web_search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {"query": {"type": "string"}}},
        }
    ]
    assert "messages" in second_payload
    assert second_payload["tools"][0]["function"]["name"] == "web_search"
    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "web_search"
    assert result.tool_calls[0].arguments == {"query": "Mochi"}
    assert result.finish_reason == "tool_calls"
    assert backend.get_model_info().metadata["responses_alias_detected"] is True


@pytest.mark.asyncio
async def test_responses_http_400_retries_with_chat_payload_and_caches_alias() -> None:
    backend = OpenAICompatBackend(
        base_url="https://www.right.codes/codex/v1/responses",
        model="gpt-test",
        api_key="test-key",
    )
    request = httpx.Request("POST", "https://www.right.codes/codex/v1/responses")
    rejection = httpx.Response(
        400,
        request=request,
        text="Bad Request: this provider expects /v1/chat/completions messages payload.",
    )
    status_error = httpx.HTTPStatusError("bad request", request=request, response=rejection)
    alias_response = _mock_response(
        {
            "model": "gpt-test",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "chat fallback ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        }
    )

    try:
        with patch.object(
            backend._client,
            "post",
            new_callable=AsyncMock,
            side_effect=[status_error, alias_response, alias_response],
        ) as post:
            result_1 = await backend.generate(
                messages=[Message(role="user", content="hi")],
                stream=False,
            )
            result_2 = await backend.generate(
                messages=[Message(role="user", content="hello again")],
                stream=False,
            )
    finally:
        await backend.close()

    assert result_1.content == "chat fallback ok"
    assert result_2.content == "chat fallback ok"
    assert "input" in post.await_args_list[0].kwargs["json"]
    assert "messages" in post.await_args_list[1].kwargs["json"]
    assert "messages" in post.await_args_list[2].kwargs["json"]
    assert backend.get_model_info().metadata["request_shape"] == "chat_completions"
    assert backend.get_model_info().metadata["responses_alias_detected"] is True
    diagnostics = backend.get_model_info().metadata["fallback_diagnostics"]
    assert any(
        item["name"] == "responses_alias_transport"
        and item["reason"] == "responses_endpoint_rejected_payload_shape"
        for item in diagnostics
    )


@pytest.mark.asyncio
async def test_probe_tool_calling_supports_responses_alias_endpoint() -> None:
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1/responses",
        model="gpt-test",
        api_key="test-key",
    )
    success_response = _mock_response(
        {
            "model": "gpt-test",
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
    assert backend.get_model_info().metadata["responses_alias_detected"] is True
    payload = post.await_args.kwargs["json"]
    assert "input" in payload
    assert "tools" in payload
    assert payload["tool_choice"] == "auto"
    diagnostics = backend.get_model_info().metadata["fallback_diagnostics"]
    assert any(
        item["name"] == "responses_alias_transport"
        and item["reason"] == "responses_probe_returned_native_chat_tool_calls"
        for item in diagnostics
    )


@pytest.mark.asyncio
async def test_probe_selected_chat_completions_pins_next_generate_to_chat_transport() -> None:
    backend = OpenAICompatBackend(
        base_url="https://api.example.com/v1",
        model="gpt-5.4",
        api_key="test-key",
        provider="openai_compat",
    )
    probe_response = _mock_response(
        {
            "model": "gpt-5.4",
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
    generate_response = _mock_response(
        {
            "model": "gpt-5.4",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "chat tool route pinned"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 4},
        }
    )
    tool = ToolSchema(
        name="web_search",
        description="Search the web",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
    )

    try:
        with patch.object(
            backend._client,
            "post",
            new_callable=AsyncMock,
            side_effect=[probe_response, generate_response],
        ) as post:
            probe_result = await backend.probe_tool_calling()
            result = await backend.generate(
                messages=[Message(role="user", content="search Mochi")],
                tools=[tool],
                stream=False,
            )
    finally:
        await backend.close()

    assert probe_result is not None
    assert probe_result["status"] == "supported"
    assert result.content == "chat tool route pinned"
    assert post.await_args_list[0].args[0] == "https://api.example.com/v1/chat/completions"
    assert post.await_args_list[1].args[0] == "https://api.example.com/v1/chat/completions"
    second_payload = post.await_args_list[1].kwargs["json"]
    assert "messages" in second_payload
    assert "input" not in second_payload
    metadata = backend.get_model_info().metadata
    assert metadata["request_shape"] == "chat_completions"
    assert metadata["tool_protocol_probe"]["selected_protocol"] == "chat_completions"


@pytest.mark.asyncio
async def test_stream_blocked_simulated_retry_overwrites_stale_supported_probe_status() -> None:
    backend = OpenAICompatBackend(
        base_url="http://localhost:8000/v1",
        model="google/gemma-4-26B-A4B-it",
        api_key="test-key",
        provider="vllm",
    )
    backend._native_tool_probe = {"status": "supported", "message": "ok"}  # noqa: SLF001
    request = httpx.Request("POST", "http://localhost:8000/v1/chat/completions")
    response = httpx.Response(
        400,
        request=request,
        stream=httpx.ByteStream(
            b'{"error":{"message":"\\"auto\\" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set"}}'
        ),
    )
    initial_error = httpx.HTTPStatusError("bad request", request=request, response=response)
    mock_stream = _MockStreamContext([])
    mock_stream.raise_for_status = MagicMock(side_effect=initial_error)
    retry_error = httpx.HTTPStatusError(
        "forbidden",
        request=request,
        response=httpx.Response(403, request=request, text="tool permission_error"),
    )
    tool = ToolSchema(
        name="web_search",
        description="Search the web",
        parameters={"type": "object", "properties": {"query": {"type": "string"}}},
    )

    try:
        with (
            patch.object(backend._client, "stream", new=MagicMock(return_value=mock_stream)),
            patch.object(backend._client, "post", new_callable=AsyncMock, side_effect=retry_error),
        ):
            stream_iter = await backend.generate(
                messages=[Message(role="user", content="search Mochi")],
                tools=[tool],
                stream=True,
            )

            with pytest.raises(BackendRequestError):
                async for _chunk in stream_iter:
                    pass
    finally:
        await backend.close()

    metadata = backend.get_model_info().metadata
    assert metadata["tool_call_mode"] == "unavailable"
    assert metadata["tool_calling_blocked"] is True
    assert metadata["native_tool_calling_status"] == "all_tool_protocols_rejected_by_provider"


@pytest.mark.asyncio
async def test_responses_stream_http_400_retries_with_chat_payload() -> None:
    backend = OpenAICompatBackend(
        base_url="https://www.right.codes/codex/v1/responses",
        model="gpt-test",
        api_key="test-key",
    )
    request = httpx.Request("POST", "https://www.right.codes/codex/v1/responses")
    response = httpx.Response(
        400,
        request=request,
        stream=httpx.ByteStream(b"Bad Request: provider only supports chat/completions."),
    )
    status_error = httpx.HTTPStatusError(
        "bad request",
        request=request,
        response=response,
    )
    mock_stream = _MockStreamContext([])
    mock_stream.raise_for_status = MagicMock(side_effect=status_error)
    fallback_response = _mock_response(
        {
            "model": "gpt-test",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "fallback stream ok"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 4},
        }
    )

    try:
        with (
            patch.object(backend._client, "stream", new=MagicMock(return_value=mock_stream)),
            patch.object(backend._client, "post", new_callable=AsyncMock, return_value=fallback_response),
        ):
            stream_iter = await backend.generate(
                messages=[Message(role="user", content="Say hello")],
                stream=True,
            )

            chunks = []
            async for chunk in stream_iter:
                chunks.append(chunk)
    finally:
        await backend.close()

    assert [chunk.delta for chunk in chunks if chunk.delta] == ["fallback stream ok"]
    assert chunks[-1].is_final is True
    assert chunks[-1].finish_reason == "stop"
    assert backend.get_model_info().metadata["request_shape"] == "chat_completions"
    assert backend.get_model_info().metadata["responses_alias_detected"] is True
