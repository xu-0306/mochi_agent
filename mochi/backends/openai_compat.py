"""OpenAI-compatible API 後端實作。"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import AsyncIterator
from typing import Any

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
        self.base_url = base_url.rstrip("/")
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
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if tools:
            payload["tools"] = [t.to_dict() for t in tools]

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
        )

    async def health_check(self) -> bool:
        """檢查 API 是否可用，優先嘗試 /models。"""
        endpoints = ["/models"]
        if not self.base_url.endswith("/v1"):
            endpoints.append("/v1/models")

        for endpoint in endpoints:
            try:
                resp = await self._client.get(
                    f"{self.base_url}{endpoint}",
                    headers=self._build_headers(),
                    timeout=5.0,
                )
                if resp.status_code == 200:
                    return True
            except Exception as exc:
                logger.debug(f"OpenAI-compatible health check failed on {endpoint}: {exc}")
        return False

    async def _blocking_generate(self, payload: dict[str, Any]) -> GenerationResult:
        """執行非串流生成。"""
        try:
            resp = await self._client.post(
                f"{self.base_url}/chat/completions",
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
        emitted_final = False
        tool_call_buffers: dict[int, dict[str, str]] = {}

        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
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

    def _build_headers(self) -> dict[str, str]:
        """建立 HTTP 標頭。"""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _parse_tool_calls(self, raw_tool_calls: Any) -> list[ToolCall]:
        """解析 non-stream 回覆中的 tool calls。"""
        if not isinstance(raw_tool_calls, list):
            return []

        parsed: list[ToolCall] = []
        for item in raw_tool_calls:
            if not isinstance(item, dict):
                continue
            fn = item.get("function", {})
            if not isinstance(fn, dict):
                fn = {}

            raw_args = fn.get("arguments", {})
            arguments: dict[str, Any]
            if isinstance(raw_args, str):
                try:
                    decoded = json.loads(raw_args)
                    arguments = decoded if isinstance(decoded, dict) else {}
                except json.JSONDecodeError:
                    arguments = {}
            elif isinstance(raw_args, dict):
                arguments = raw_args
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

        item = raw_tool_calls[0]
        if not isinstance(item, dict):
            return None

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
                    parsed_args = parsed
            except json.JSONDecodeError:
                parsed_args = {}

        return ToolCall(
            id=buf["id"] or str(uuid.uuid4()),
            name=buf["name"],
            arguments=parsed_args,
        )

    def _as_int(self, value: Any) -> int:
        """將任意值轉成 int，失敗則回傳 0。"""
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    async def close(self) -> None:
        """關閉 HTTP client 連線。"""
        await self._client.aclose()
