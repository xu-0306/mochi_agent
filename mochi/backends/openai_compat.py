"""OpenAI-compatible API 後端實作。"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, Literal, cast
from urllib.parse import urlsplit, urlunsplit

import httpx

from mochi.backends.base import BaseLLMBackend
from mochi.backends.types import (
    GenerationResult,
    Message,
    ModelInfo,
    StreamChunk,
    ToolCall,
    ToolSchema,
)

logger = logging.getLogger(__name__)


ApiMode = Literal["chat_completions", "responses"]


@dataclass(frozen=True)
class _OpenAICompatEndpoint:
    """OpenAI-compatible URL 正規化結果。"""

    input_url: str
    request_url: str
    api_mode: ApiMode
    health_urls: tuple[str, ...]


class OpenAICompatBackend(BaseLLMBackend):
    """通用 OpenAI-compatible API 後端。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 120.0,
    ) -> None:
        """初始化 OpenAI-compatible 後端。

        Args:
            base_url: API 基底網址（可含 /v1）。
            model: 預設模型名稱。
            api_key: API 金鑰，若提供則以 Bearer token 送出。
            timeout: HTTP 請求逾時秒數。
        """
        endpoint = _normalize_openai_compat_endpoint(base_url)
        self.base_url = endpoint.input_url
        self._request_url = endpoint.request_url
        self._api_mode = endpoint.api_mode
        self._health_urls = endpoint.health_urls
        self.model = model
        self.api_key = api_key
        self._client = httpx.AsyncClient(timeout=timeout)

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
    ) -> GenerationResult | AsyncIterator[StreamChunk]:
        """呼叫 Chat Completions API 進行生成。

        Args:
            messages: 對話訊息列表。
            tools: 可用工具定義列表。
            temperature: 採樣溫度。
            max_tokens: 最大輸出 token 數。
            stream: 是否啟用串流。

        Returns:
            非串流時回傳 GenerationResult，串流時回傳 AsyncIterator[StreamChunk]。
        """
        payload = (
            self._build_responses_payload(messages, tools, temperature, max_tokens, stream)
            if self._api_mode == "responses"
            else self._build_chat_completions_payload(messages, tools, temperature, max_tokens, stream)
        )

        if stream:
            return self._stream_generate(payload)
        return await self._blocking_generate(payload)

    def supports_tool_calling(self) -> bool:
        """OpenAI-compatible 端點通常支援 tool calling。"""
        return True

    def get_model_info(self) -> ModelInfo:
        """回傳目前模型資訊。"""
        return ModelInfo(
            name=self.model,
            backend_type="openai_compat",
            supports_tool_calling=True,
            metadata={
                "base_url": self.base_url,
                "api_url": self._request_url,
                "api_mode": self._api_mode,
            },
        )

    async def health_check(self) -> bool:
        """檢查 API 是否可用，優先嘗試 /models。"""
        for url in self._health_urls:
            try:
                resp = await self._client.get(
                    url,
                    headers=self._build_headers(),
                    timeout=5.0,
                )
                if resp.status_code == 200:
                    return True
            except Exception as exc:
                logger.debug(f"OpenAI-compatible health check failed on {url}: {exc}")
        return False

    async def _blocking_generate(self, payload: dict[str, Any]) -> GenerationResult:
        """執行非串流生成。"""
        try:
            resp = await self._client.post(
                self._request_url,
                json=payload,
                headers=self._build_headers(),
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(
                f"OpenAI-compatible API error {exc.response.status_code}: {exc.response.text}"
            )
            raise
        except httpx.RequestError as exc:
            logger.error(f"OpenAI-compatible connection error: {exc}")
            raise

        data = resp.json()
        if self._api_mode == "responses":
            return self._parse_responses_result(data)

        return self._parse_chat_completions_result(data)

    def _parse_chat_completions_result(self, data: dict[str, Any]) -> GenerationResult:
        """解析 Chat Completions 回應。"""
        choices: list[dict[str, Any]] = data.get("choices", [])
        choice0 = choices[0] if choices else {}
        message = choice0.get("message", {})

        content_raw = message.get("content", "")
        if isinstance(content_raw, str):
            content = content_raw
        elif content_raw is None:
            content = ""
        else:
            content = json.dumps(content_raw, ensure_ascii=False)

        tool_calls = self._parse_tool_calls(message.get("tool_calls", []))
        usage = data.get("usage", {})
        finish_reason = choice0.get("finish_reason")
        if not finish_reason:
            finish_reason = "tool_calls" if tool_calls else "stop"

        return GenerationResult(
            content=content,
            tool_calls=tool_calls,
            input_tokens=self._as_int(usage.get("prompt_tokens")),
            output_tokens=self._as_int(usage.get("completion_tokens")),
            model=data.get("model", self.model),
            finish_reason=finish_reason,
        )

    async def _stream_generate(self, payload: dict[str, Any]) -> AsyncIterator[StreamChunk]:
        """執行串流生成（SSE / JSON lines）。"""
        if self._api_mode == "responses":
            async for chunk in self._stream_responses_generate(payload):
                yield chunk
            return

        emitted_final = False
        tool_call_buffers: dict[int, dict[str, str]] = {}

        try:
            async with self._client.stream(
                "POST",
                self._request_url,
                json=payload,
                headers=self._build_headers(),
            ) as resp:
                resp.raise_for_status()
                async for raw_line in resp.aiter_lines():
                    line = raw_line.strip()
                    if not line:
                        continue

                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if not line:
                        continue

                    if line == "[DONE]":
                        if not emitted_final:
                            yield StreamChunk(is_final=True, finish_reason="stop")
                        break

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    choices: list[dict[str, Any]] = data.get("choices", [])
                    if not choices:
                        continue

                    choice0 = choices[0]
                    delta_obj = choice0.get("delta", {})

                    text_delta = delta_obj.get("content", "")
                    if not isinstance(text_delta, str):
                        text_delta = ""

                    tool_call_delta = self._parse_tool_call_delta(
                        delta_obj.get("tool_calls"),
                        tool_call_buffers,
                    )

                    finish_reason = choice0.get("finish_reason")
                    is_final = finish_reason is not None
                    if is_final:
                        emitted_final = True

                    if text_delta or tool_call_delta is not None or is_final:
                        yield StreamChunk(
                            delta=text_delta,
                            tool_call_delta=tool_call_delta,
                            is_final=is_final,
                            finish_reason=finish_reason,
                        )
        except httpx.HTTPStatusError as exc:
            logger.error(
                f"OpenAI-compatible stream API error {exc.response.status_code}: {exc.response.text}"
            )
            raise
        except httpx.RequestError as exc:
            logger.error(f"OpenAI-compatible stream connection error: {exc}")
            raise

    async def _stream_responses_generate(self, payload: dict[str, Any]) -> AsyncIterator[StreamChunk]:
        """執行 Responses API 串流生成。"""
        emitted_final = False
        try:
            async with self._client.stream(
                "POST",
                self._request_url,
                json=payload,
                headers=self._build_headers(),
            ) as resp:
                resp.raise_for_status()
                async for raw_line in resp.aiter_lines():
                    line = raw_line.strip()
                    if not line:
                        continue

                    if line.startswith("data:"):
                        line = line[5:].strip()
                    if not line or line == "[DONE]":
                        continue

                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    event_type = str(data.get("type", ""))
                    if event_type == "response.output_text.delta":
                        delta = data.get("delta")
                        if isinstance(delta, str) and delta:
                            yield StreamChunk(delta=delta)
                        continue

                    if event_type in {"response.completed", "response.incomplete", "response.failed"}:
                        emitted_final = True
                        finish_reason = "stop" if event_type == "response.completed" else event_type
                        yield StreamChunk(is_final=True, finish_reason=finish_reason)
                        break
        except httpx.HTTPStatusError as exc:
            logger.error(
                f"OpenAI-compatible responses stream API error {exc.response.status_code}: {exc.response.text}"
            )
            raise
        except httpx.RequestError as exc:
            logger.error(f"OpenAI-compatible responses stream connection error: {exc}")
            raise

        if not emitted_final:
            yield StreamChunk(is_final=True, finish_reason="stop")

    def _build_chat_completions_payload(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> dict[str, Any]:
        """建立 Chat Completions payload。"""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if tools:
            payload["tools"] = [t.to_dict() for t in tools]
        return payload

    def _build_responses_payload(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        temperature: float,
        max_tokens: int,
        stream: bool,
    ) -> dict[str, Any]:
        """建立 Responses API payload。"""
        instructions = "\n\n".join(m.content for m in messages if m.role == "system" and m.content)
        payload: dict[str, Any] = {
            "model": self.model,
            "input": self._build_responses_input(messages),
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "stream": stream,
        }
        if instructions:
            payload["instructions"] = instructions
        if tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
                for tool in tools
            ]
        return payload

    def _build_responses_input(self, messages: list[Message]) -> list[dict[str, Any]]:
        """將內部 message history 轉成 Responses API input items。"""
        input_items: list[dict[str, Any]] = []
        for message in messages:
            if message.role == "system":
                continue

            if message.role == "tool":
                if message.tool_call_id:
                    input_items.append(
                        {
                            "type": "function_call_output",
                            "call_id": message.tool_call_id,
                            "output": message.content,
                        }
                    )
                else:
                    input_items.append(
                        {
                            "role": "user",
                            "content": f"Tool result from {message.name or 'tool'}: {message.content}",
                        }
                    )
                continue

            if message.role == "assistant" and message.tool_calls:
                if message.content:
                    input_items.append({"role": "assistant", "content": message.content})
                input_items.extend(
                    {
                        "type": "function_call",
                        "call_id": tool_call.id,
                        "name": tool_call.name,
                        "arguments": json.dumps(tool_call.arguments, ensure_ascii=False),
                    }
                    for tool_call in message.tool_calls
                )
                continue

            input_items.append(
                {
                    "role": "assistant" if message.role == "assistant" else "user",
                    "content": message.content,
                }
            )

        return input_items

    def _parse_responses_result(self, data: dict[str, Any]) -> GenerationResult:
        """解析 Responses API 回應。"""
        content = _coerce_text(data.get("output_text"))
        output = data.get("output")
        tool_calls = self._parse_responses_tool_calls(output)
        if not content and isinstance(output, list):
            output_items = cast(list[Any], output)
            for raw_item in output_items:
                if not isinstance(raw_item, dict):
                    continue
                item = cast(dict[str, Any], raw_item)
                content = _coerce_text(item.get("content"))
                if content:
                    break

        usage = data.get("usage", {})
        finish_reason = data.get("finish_reason")
        if not isinstance(finish_reason, str) or not finish_reason:
            finish_reason = "tool_calls" if tool_calls else "stop"

        return GenerationResult(
            content=content,
            tool_calls=tool_calls,
            input_tokens=self._as_int(usage.get("input_tokens")),
            output_tokens=self._as_int(usage.get("output_tokens")),
            model=data.get("model", self.model),
            finish_reason=finish_reason,
        )

    def _build_headers(self) -> dict[str, str]:
        """建立 HTTP 標頭。"""
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _parse_tool_calls(self, raw_tool_calls: Any) -> list[ToolCall]:
        """解析 non-stream 回覆中的 tool calls。"""
        if not isinstance(raw_tool_calls, list):
            return []

        parsed: list[ToolCall] = []
        raw_items = cast(list[Any], raw_tool_calls)
        for raw_item in raw_items:
            if not isinstance(raw_item, dict):
                continue
            item = cast(dict[str, Any], raw_item)
            fn = item.get("function", {})
            if not isinstance(fn, dict):
                fn = {}
            fn = cast(dict[str, Any], fn)

            raw_args = fn.get("arguments", {})
            arguments: dict[str, Any]
            if isinstance(raw_args, str):
                try:
                    decoded = json.loads(raw_args)
                    arguments = cast(dict[str, Any], decoded) if isinstance(decoded, dict) else {}
                except json.JSONDecodeError:
                    arguments = {}
            elif isinstance(raw_args, dict):
                arguments = cast(dict[str, Any], raw_args)
            else:
                arguments = {}

            call_id = item.get("id")
            if not isinstance(call_id, str) or not call_id:
                call_id = str(uuid.uuid4())

            name = fn.get("name", "")
            if not isinstance(name, str):
                name = ""

            parsed.append(ToolCall(id=call_id, name=name, arguments=arguments))

        return parsed

    def _parse_tool_call_delta(
        self,
        raw_tool_calls: Any,
        buffers: dict[int, dict[str, str]],
    ) -> ToolCall | None:
        """解析串流增量中的第一個 tool call。"""
        if not isinstance(raw_tool_calls, list) or not raw_tool_calls:
            return None

        raw_items = cast(list[Any], raw_tool_calls)
        raw_item = raw_items[0]
        if not isinstance(raw_item, dict):
            return None
        item = cast(dict[str, Any], raw_item)

        index = item.get("index", 0)
        if not isinstance(index, int):
            index = 0

        buf = buffers.setdefault(index, {"id": "", "name": "", "arguments": ""})

        call_id = item.get("id")
        if isinstance(call_id, str) and call_id:
            buf["id"] = call_id

        fn = item.get("function", {})
        if not isinstance(fn, dict):
            fn = {}
        fn = cast(dict[str, Any], fn)

        name = fn.get("name")
        if isinstance(name, str) and name:
            buf["name"] = name

        arg_piece = fn.get("arguments")
        if isinstance(arg_piece, str):
            buf["arguments"] += arg_piece

        parsed_args: dict[str, Any] = {}
        if buf["arguments"]:
            try:
                parsed = json.loads(buf["arguments"])
                if isinstance(parsed, dict):
                    parsed_args = cast(dict[str, Any], parsed)
            except json.JSONDecodeError:
                parsed_args = {}

        return ToolCall(
            id=buf["id"] or str(uuid.uuid4()),
            name=buf["name"],
            arguments=parsed_args,
        )

    def _parse_responses_tool_calls(self, output: Any) -> list[ToolCall]:
        """解析 Responses API output 中的 function_call。"""
        if not isinstance(output, list):
            return []

        tool_calls: list[ToolCall] = []
        output_items = cast(list[Any], output)
        for raw_item in output_items:
            if not isinstance(raw_item, dict):
                continue
            item = cast(dict[str, Any], raw_item)
            if item.get("type") != "function_call":
                continue

            name = item.get("name")
            if not isinstance(name, str) or not name:
                continue

            raw_arguments = item.get("arguments", {})
            arguments: dict[str, Any]
            if isinstance(raw_arguments, str):
                decoded: Any
                try:
                    decoded = json.loads(raw_arguments) if raw_arguments else {}
                except json.JSONDecodeError:
                    decoded = {}
                arguments = cast(dict[str, Any], decoded) if isinstance(decoded, dict) else {}
            elif isinstance(raw_arguments, dict):
                arguments = cast(dict[str, Any], raw_arguments)
            else:
                arguments = {}

            call_id = item.get("call_id") or item.get("id")
            tool_calls.append(
                ToolCall(
                    id=str(call_id) if call_id else str(uuid.uuid4()),
                    name=name,
                    arguments=arguments,
                )
            )

        return tool_calls

    def _as_int(self, value: Any) -> int:
        """將任意值轉成 int，失敗則回傳 0。"""
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    async def close(self) -> None:
        """關閉 HTTP client 連線。"""
        await self._client.aclose()


def _normalize_openai_compat_endpoint(raw_url: str) -> _OpenAICompatEndpoint:
    """解析 API URL，只有 base `/v1` 會補 Chat Completions endpoint。"""
    input_url = raw_url.strip().rstrip("/")
    parts = urlsplit(input_url)
    normalized_path = parts.path.rstrip("/").lower()

    if normalized_path.endswith("/chat/completions"):
        models_base = _url_without_path_suffix(parts, "/chat/completions")
        return _OpenAICompatEndpoint(
            input_url=input_url,
            request_url=input_url,
            api_mode="chat_completions",
            health_urls=(f"{models_base}/models",),
        )

    if normalized_path.endswith("/responses"):
        models_base = _url_without_path_suffix(parts, "/responses")
        return _OpenAICompatEndpoint(
            input_url=input_url,
            request_url=input_url,
            api_mode="responses",
            health_urls=(f"{models_base}/models",),
        )

    if normalized_path.endswith("/v1"):
        return _OpenAICompatEndpoint(
            input_url=input_url,
            request_url=f"{input_url}/chat/completions",
            api_mode="chat_completions",
            health_urls=(f"{input_url}/models",),
        )

    return _OpenAICompatEndpoint(
        input_url=input_url,
        request_url=f"{input_url}/chat/completions",
        api_mode="chat_completions",
        health_urls=(f"{input_url}/models", f"{input_url}/v1/models"),
    )


def _url_without_path_suffix(parts: Any, suffix: str) -> str:
    """移除完整 endpoint 尾段，取得 `/v1` 層級 model list base。"""
    path = parts.path.rstrip("/")
    if path.lower().endswith(suffix):
        path = path[: -len(suffix)] or "/"
    path = path.rstrip("/")
    return urlunsplit((parts.scheme, parts.netloc, path, "", "")).rstrip("/")


def _coerce_text(value: Any) -> str:
    """從 Responses API 巢狀 text/content 結構收斂文字。"""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(part for part in (_coerce_text(item) for item in cast(list[Any], value)) if part)
    if isinstance(value, dict):
        value = cast(dict[str, Any], value)
        for key in ("output_text", "text", "content", "value"):
            if key in value:
                text = _coerce_text(value.get(key))
                if text:
                    return text
    return ""
