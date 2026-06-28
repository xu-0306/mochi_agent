"""OllamaBackend 單元測試（使用 httpx mock）。"""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from mochi.backends.ollama import OllamaBackend
from mochi.backends.openai_compat import OpenAICompatBackend
from mochi.backends.router import BackendRouter
from mochi.backends.simulated_tool_protocol import SimulatedToolProtocol
from mochi.backends.tool_call_contract import validate_tool_turn_result
from mochi.backends.types import GenerationResult, Message, ToolCall, ToolSchema


@pytest.fixture
def backend() -> OllamaBackend:
    """建立 OllamaBackend 測試實例。"""
    return OllamaBackend(model="llama3.2", base_url="http://localhost:11434")


def _mock_response(data: dict) -> MagicMock:
    """建立 httpx Response mock。"""
    resp = MagicMock()
    resp.status_code = 200
    resp.json.return_value = data
    resp.raise_for_status = MagicMock()
    return resp


def _httpx_json_response(url: str, status_code: int, data: dict) -> httpx.Response:
    request = httpx.Request("POST", url)
    return httpx.Response(status_code, request=request, json=data)


def test_validate_tool_turn_accepts_structured_tool_calls() -> None:
    result = GenerationResult(
        content="",
        thinking="plan",
        tool_calls=[ToolCall(id="1", name="web_search", arguments={})],
    )

    verdict = validate_tool_turn_result(result=result, tools_requested=True)

    assert verdict.is_valid is True
    assert verdict.reason == "tool_calls"


def test_validate_tool_turn_rejects_thinking_only_output() -> None:
    result = GenerationResult(content="", thinking="planning only", tool_calls=[])

    verdict = validate_tool_turn_result(result=result, tools_requested=True)

    assert verdict.is_valid is False
    assert verdict.reason == "thinking_only"


def test_simulated_tool_protocol_flattens_prior_tool_messages() -> None:
    protocol = SimulatedToolProtocol()
    tools = [
        ToolSchema(
            name="web_search",
            description="Search the web",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
    ]
    messages = [
        Message(
            role="assistant",
            content="",
            tool_calls=[ToolCall(id="1", name="web_search", arguments={"query": "Mochi"})],
        ),
        Message(
            role="tool",
            content="Tool web_search result:\nfound",
            tool_call_id="1",
            name="web_search",
        ),
    ]

    prepared = protocol.prepare_messages(messages=messages, tools=tools)

    assert any(
        message.role == "assistant" and "Tool request: web_search" in message.content
        for message in prepared
    )
    assert any(
        message.role == "user" and message.content.startswith("Tool web_search result:")
        for message in prepared
    )


@pytest.mark.asyncio
async def test_health_check_ok(backend: OllamaBackend) -> None:
    """health_check() 應在 /api/tags 回傳 200 時回傳 True。"""
    mock_resp = _mock_response({"models": []})
    with patch.object(backend._client, "get", new_callable=AsyncMock, return_value=mock_resp):
        result = await backend.health_check()
    assert result is True


@pytest.mark.asyncio
async def test_health_check_fail(backend: OllamaBackend) -> None:
    """health_check() 應在連線失敗時回傳 False。"""
    import httpx

    with patch.object(
        backend._client,
        "get",
        new_callable=AsyncMock,
        side_effect=httpx.ConnectError("refused"),
    ):
        result = await backend.health_check()
    assert result is False


@pytest.mark.asyncio
async def test_generate_nonstream_basic(backend: OllamaBackend) -> None:
    """非串流生成應正確解析回傳的 content。"""
    ollama_response = {
        "model": "llama3.2",
        "message": {"role": "assistant", "content": "你好！"},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 10,
        "eval_count": 5,
    }
    mock_resp = _mock_response(ollama_response)
    messages = [Message(role="user", content="你好")]

    with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_resp):
        result = await backend._blocking_generate(
            {"model": "llama3.2", "messages": [m.to_dict() for m in messages], "stream": False}
        )

    assert result.content == "你好！"
    assert result.input_tokens == 10
    assert result.output_tokens == 5
    assert result.finish_reason == "stop"


@pytest.mark.asyncio
async def test_generate_nonstream_thinking_only_is_kept_separate(
    backend: OllamaBackend,
) -> None:
    ollama_response = {
        "model": "llama3.2",
        "message": {
            "role": "assistant",
            "content": "",
            "thinking": "BERT is a bidirectional Transformer encoder model.",
        },
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 12,
        "eval_count": 6,
    }
    mock_resp = _mock_response(ollama_response)

    with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_resp):
        result = await backend._blocking_generate(
            {"model": "llama3.2", "messages": [], "stream": False}
        )

    assert result.content == ""
    assert result.thinking == "BERT is a bidirectional Transformer encoder model."
    assert result.finish_reason == "stop"


@pytest.mark.asyncio
async def test_generate_nonstream_empty_non_tool_response_raises_backend_error(
    backend: OllamaBackend,
) -> None:
    ollama_response = {
        "model": "llama3.2",
        "message": {"role": "assistant", "content": ""},
        "done": True,
        "done_reason": "stop",
        "prompt_eval_count": 12,
        "eval_count": 0,
    }
    mock_resp = _mock_response(ollama_response)

    with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_resp):
        with pytest.raises(RuntimeError, match="empty response"):
            await backend._blocking_generate(
                {
                    "model": "llama3.2",
                    "messages": [{"role": "user", "content": "summarize this paper"}],
                    "stream": False,
                }
            )


@pytest.mark.asyncio
async def test_generate_with_tool_calls(backend: OllamaBackend) -> None:
    """含工具呼叫的回覆應正確解析 ToolCall 列表。"""
    call_id = str(uuid.uuid4())
    ollama_response = {
        "model": "llama3.2",
        "message": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {
                    "id": call_id,
                    "function": {
                        "name": "web_search",
                        "arguments": {"query": "Mochi AI"},
                    },
                }
            ],
        },
        "done": True,
        "done_reason": "tool_calls",
        "prompt_eval_count": 20,
        "eval_count": 8,
    }
    mock_resp = _mock_response(ollama_response)

    with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_resp):
        result = await backend._blocking_generate(
            {"model": "llama3.2", "messages": [], "stream": False}
        )

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "web_search"
    assert result.tool_calls[0].arguments == {"query": "Mochi AI"}
    assert result.finish_reason == "tool_calls"


@pytest.mark.asyncio
async def test_ollama_falls_back_to_simulated_tool_mode_when_native_response_has_only_thinking(
    backend: OllamaBackend,
) -> None:
    tools = [
        ToolSchema(
            name="arxiv_search",
            description="Search arXiv",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
    ]
    native_response = _mock_response(
        {
            "model": "llama3.2",
            "message": {
                "role": "assistant",
                "content": "",
                "thinking": "I should search arXiv first.",
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 18,
            "eval_count": 7,
        }
    )
    simulated_response = _mock_response(
        {
            "model": "llama3.2",
            "message": {
                "role": "assistant",
                "content": (
                    "<tool_call>\n"
                    '{"name": "arxiv_search", "arguments": {"query": "ESG LLM fine-tuning"}}\n'
                    "</tool_call>"
                ),
            },
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 24,
            "eval_count": 11,
        }
    )

    with patch.object(
        backend._client,
        "post",
        new_callable=AsyncMock,
        side_effect=[native_response, simulated_response],
    ) as post:
        result = await backend.generate(
            messages=[Message(role="user", content="Find ESG LLM fine-tuning papers.")],
            tools=tools,
            stream=False,
        )

    assert len(result.tool_calls) == 1
    assert result.tool_calls[0].name == "arxiv_search"
    assert result.tool_calls[0].arguments == {"query": "ESG LLM fine-tuning"}
    metadata = backend.get_model_info().metadata
    assert metadata["tool_call_mode"] == "simulated_fallback"
    assert metadata["native_tool_calling_status"] == "native_tool_calls_missing"
    assert metadata["fallback_diagnostics"]

    first_payload = post.await_args_list[0].kwargs["json"]
    second_payload = post.await_args_list[1].kwargs["json"]
    assert "tools" in first_payload
    assert "tools" not in second_payload
    assert second_payload["messages"][0]["role"] == "system"
    assert "## Tool Use Instructions" in second_payload["messages"][0]["content"]


@pytest.mark.asyncio
async def test_ollama_simulated_tool_mode_flattens_prior_tool_messages() -> None:
    backend = OllamaBackend(model="llama3.2", base_url="http://localhost:11434")
    backend._tool_state.active_mode = "simulated_fallback"  # noqa: SLF001
    tools = [
        ToolSchema(
            name="web_search",
            description="Search the web",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
    ]
    mock_resp = _mock_response(
        {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "done"},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 20,
            "eval_count": 6,
        }
    )
    messages = [
        Message(
            role="assistant",
            content="",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    name="web_search",
                    arguments={"query": "Mochi AI"},
                )
            ],
        ),
        Message(
            role="tool",
            content="Tool web_search result:\nfound: Mochi AI",
            tool_call_id="call-1",
            name="web_search",
        ),
    ]

    try:
        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_resp) as post:
            await backend.generate(messages=messages, tools=tools, stream=False)
    finally:
        await backend.close()

    payload = post.await_args.kwargs["json"]
    assert "tools" not in payload
    assert payload["messages"][0]["role"] == "system"
    assert "## Tool Use Instructions" in payload["messages"][0]["content"]
    assert payload["messages"][1] == {
        "role": "assistant",
        "content": "Tool request: web_search\nArguments: {'query': 'Mochi AI'}",
    }
    assert payload["messages"][2] == {
        "role": "user",
        "content": "Tool web_search result:\nfound: Mochi AI",
    }


@pytest.mark.asyncio
async def test_ollama_retry_that_returns_only_thinking_raises_backend_error() -> None:
    backend = OllamaBackend(model="llama3.2", base_url="http://localhost:11434")
    tools = [
        ToolSchema(
            name="web_search",
            description="Search the web",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
    ]
    native_response = _mock_response(
        {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "", "thinking": "Need web context."},
            "done": True,
            "done_reason": "stop",
        }
    )
    retry_response = _mock_response(
        {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "", "thinking": "Still deciding."},
            "done": True,
            "done_reason": "stop",
        }
    )

    try:
        with patch.object(
            backend._client,
            "post",
            new_callable=AsyncMock,
            side_effect=[native_response, retry_response],
        ):
            with pytest.raises(RuntimeError, match="invalid tool-eligible turn"):
                await backend.generate(
                    messages=[Message(role="user", content="Search Mochi AI")],
                    tools=tools,
                    stream=False,
                )
    finally:
        await backend.close()

    metadata = backend.get_model_info().metadata
    assert metadata["tool_call_mode"] == "unavailable"
    assert metadata["native_tool_calling_status"] == "simulated_protocol_rejected"


@pytest.mark.asyncio
async def test_ollama_probe_reenables_native_mode_after_fallback() -> None:
    backend = OllamaBackend(model="llama3.2", base_url="http://localhost:11434")
    backend._tool_state.active_mode = "simulated_fallback"  # noqa: SLF001
    backend._tool_state.native_status = "native_tool_calls_missing"  # noqa: SLF001
    backend._tool_state.fallback_validation_status = "validated"  # noqa: SLF001
    probe_response = _mock_response(
        {
            "model": "llama3.2",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "probe-call-1",
                        "function": {"name": "mochi_tool_probe", "arguments": {"value": "ok"}},
                    }
                ],
            },
            "done": True,
            "done_reason": "tool_calls",
        }
    )

    try:
        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=probe_response):
            result = await backend.probe_tool_calling()
    finally:
        await backend.close()

    assert result is not None
    assert result["status"] == "supported"
    metadata = backend.get_model_info().metadata
    assert metadata["tool_call_mode"] == "native"
    assert metadata["native_tool_calling_status"] == "supported"


@pytest.mark.asyncio
async def test_ollama_simulated_retry_http_error_marks_backend_unavailable() -> None:
    backend = OllamaBackend(model="llama3.2", base_url="http://localhost:11434")
    tools = [
        ToolSchema(
            name="web_search",
            description="Search the web",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
    ]
    native_response = _mock_response(
        {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "", "thinking": "Need web context."},
            "done": True,
            "done_reason": "stop",
        }
    )
    request = httpx.Request("POST", "http://localhost:11434/api/chat")
    simulated_error = httpx.HTTPStatusError(
        "EOF",
        request=request,
        response=httpx.Response(500, request=request, text='{"error":"EOF"}'),
    )

    try:
        with patch.object(
            backend._client,
            "post",
            new_callable=AsyncMock,
            side_effect=[native_response, simulated_error],
        ):
            with pytest.raises(RuntimeError, match="EOF|500"):
                await backend.generate(
                    messages=[Message(role="user", content="Search Mochi AI")],
                    tools=tools,
                    stream=False,
                )
    finally:
        await backend.close()

    assert backend.get_model_info().metadata["tool_call_mode"] == "unavailable"


@pytest.mark.asyncio
async def test_ollama_probe_failure_from_native_mode_marks_backend_unavailable() -> None:
    backend = OllamaBackend(model="llama3.2", base_url="http://localhost:11434")
    failure_response = _mock_response(
        {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "", "thinking": "Need a tool."},
            "done": True,
            "done_reason": "stop",
        }
    )

    try:
        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=failure_response):
            result = await backend.probe_tool_calling()
    finally:
        await backend.close()

    assert result is not None
    assert result["status"] == "thinking_only"
    assert backend.get_model_info().metadata["tool_call_mode"] == "unavailable"


@pytest.mark.asyncio
async def test_ollama_failed_reprobe_after_validated_fallback_stays_in_simulated_mode() -> None:
    backend = OllamaBackend(model="llama3.2", base_url="http://localhost:11434")
    backend._tool_state.active_mode = "simulated_fallback"  # noqa: SLF001
    backend._tool_state.native_status = "native_tool_calls_missing"  # noqa: SLF001
    backend._tool_state.fallback_validation_status = "validated"  # noqa: SLF001
    failure_response = _mock_response(
        {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "", "thinking": "Need a tool."},
            "done": True,
            "done_reason": "stop",
        }
    )

    try:
        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=failure_response):
            result = await backend.probe_tool_calling()
    finally:
        await backend.close()

    assert result is not None
    assert result["status"] == "thinking_only"
    assert backend.get_model_info().metadata["tool_call_mode"] == "simulated_fallback"


@pytest.mark.asyncio
async def test_ollama_manual_probe_can_recover_from_unavailable_state() -> None:
    backend = OllamaBackend(model="llama3.2", base_url="http://localhost:11434")
    backend._tool_state.active_mode = "unavailable"  # noqa: SLF001
    backend._tool_state.native_status = "simulated_protocol_rejected"  # noqa: SLF001
    backend._tool_state.fallback_validation_status = "rejected"  # noqa: SLF001
    probe_response = _mock_response(
        {
            "model": "llama3.2",
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "probe-call-1",
                        "function": {"name": "mochi_tool_probe", "arguments": {"value": "ok"}},
                    }
                ],
            },
            "done": True,
            "done_reason": "tool_calls",
        }
    )

    try:
        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=probe_response):
            result = await backend.probe_tool_calling()
    finally:
        await backend.close()

    assert result is not None
    assert result["status"] == "supported"
    assert backend.get_model_info().metadata["tool_call_mode"] == "native"


def test_ollama_supports_tool_calling_false_when_mode_is_unavailable() -> None:
    backend = OllamaBackend(model="llama3.2", base_url="http://localhost:11434")
    backend._tool_state.active_mode = "unavailable"  # noqa: SLF001

    assert backend.supports_tool_calling() is False


def test_ollama_serializes_messages_in_native_shape(backend: OllamaBackend) -> None:
    messages = [
        Message(
            role="assistant",
            content="",
            thinking="Search first",
            tool_calls=[
                ToolCall(
                    id="call-1",
                    name="web_search",
                    arguments={"query": "Mochi AI"},
                    index=0,
                )
            ],
        ),
        Message(
            role="tool",
            content="Tool web_search result:\nfound: Mochi AI",
            tool_call_id="call-1",
            name="web_search",
        ),
    ]

    payload = backend._serialize_messages(messages)  # noqa: SLF001

    assert payload == [
        {
            "role": "assistant",
            "content": "",
            "thinking": "Search first",
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": {"query": "Mochi AI"},
                        "index": 0,
                    }
                }
            ],
        },
        {
            "role": "tool",
            "content": "found: Mochi AI",
            "tool_name": "web_search",
        },
    ]


@pytest.mark.asyncio
async def test_generate_with_tool_calls_keeps_thinking_separate(backend: OllamaBackend) -> None:
    ollama_response = {
        "model": "llama3.2",
        "message": {
            "role": "assistant",
            "content": "",
            "thinking": "Need web context.",
            "tool_calls": [
                {
                    "id": "call-1",
                    "function": {
                        "name": "web_search",
                        "arguments": {"query": "Mochi AI"},
                        "index": 0,
                    },
                }
            ],
        },
        "done": True,
        "done_reason": "tool_calls",
        "prompt_eval_count": 20,
        "eval_count": 8,
    }
    mock_resp = _mock_response(ollama_response)

    with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_resp):
        result = await backend._blocking_generate(
            {"model": "llama3.2", "messages": [], "stream": False}
        )

    assert result.content == ""
    assert result.thinking == "Need web context."
    assert result.tool_calls[0].index == 0


def test_ollama_serializes_tools_in_native_shape(backend: OllamaBackend) -> None:
    tools = [
        ToolSchema(
            name="web_search",
            description="Search the web",
            parameters={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
    ]

    payload = backend._serialize_tools(tools)  # noqa: SLF001

    assert payload == [
        {
            "type": "function",
            "function": {
                "name": "web_search",
                "description": "Search the web",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
    ]


def test_model_info(backend: OllamaBackend) -> None:
    """get_model_info() 應回傳正確的 ModelInfo。"""
    info = backend.get_model_info()
    assert info.name == "llama3.2"
    assert info.backend_type == "ollama"
    assert info.supports_tool_calling is True
    assert info.metadata["supports_reasoning_effort"] is False


def test_supports_tool_calling(backend: OllamaBackend) -> None:
    """Ollama 後端應回報支援 tool calling。"""
    assert backend.supports_tool_calling() is True


def test_ollama_gpt_oss_model_info_supports_reasoning_effort() -> None:
    """Ollama GPT-OSS models support low/medium/high think levels."""
    backend = OllamaBackend(model="gpt-oss:20b", base_url="http://localhost:11434")

    info = backend.get_model_info()

    assert info.metadata["supports_reasoning_effort"] is True
    assert info.metadata["reasoning_effort_param"] == "think"


@pytest.mark.asyncio
async def test_ollama_generate_maps_reasoning_effort_to_think() -> None:
    """Ollama reasoning effort should serialize as the native top-level think field."""
    backend = OllamaBackend(model="gpt-oss:20b", base_url="http://localhost:11434")
    mock_resp = _mock_response(
        {
            "model": "gpt-oss:20b",
            "message": {"role": "assistant", "content": "ok"},
            "done": True,
        }
    )

    try:
        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_resp) as post:
            await backend.generate(
                messages=[Message(role="user", content="hi")],
                reasoning_effort="high",
                stream=False,
            )
    finally:
        await backend.close()

    payload = post.await_args.kwargs["json"]
    assert payload["think"] == "high"


@pytest.mark.asyncio
async def test_ollama_generate_omits_reasoning_effort_for_unknown_models() -> None:
    """Unknown Ollama models should not receive unsupported think levels."""
    backend = OllamaBackend(model="llama3.2", base_url="http://localhost:11434")
    mock_resp = _mock_response(
        {
            "model": "llama3.2",
            "message": {"role": "assistant", "content": "ok"},
            "done": True,
        }
    )

    try:
        with patch.object(backend._client, "post", new_callable=AsyncMock, return_value=mock_resp) as post:
            await backend.generate(
                messages=[Message(role="user", content="hi")],
                reasoning_effort="high",
                stream=False,
            )
    finally:
        await backend.close()

    payload = post.await_args.kwargs["json"]
    assert "think" not in payload


@pytest.mark.asyncio
async def test_backend_router_ollama() -> None:
    """BackendRouter 應能解析 ollama: 前綴並回傳 OllamaBackend。"""
    router = BackendRouter()
    with patch.object(OllamaBackend, "health_check", new_callable=AsyncMock, return_value=True):
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
        with patch.object(
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
