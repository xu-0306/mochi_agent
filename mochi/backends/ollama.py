"""Ollama LLM 後端實作。"""

from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx
try:
    from loguru import logger
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal test envs
    import logging

    logger = logging.getLogger(__name__)

from mochi.backends.base import BackendRequestError, BaseLLMBackend
from mochi.backends.types import (
    GenerationResult,
    Message,
    ModelInfo,
    StreamChunk,
    ToolCall,
    ToolSchema,
)


class OllamaBackend(BaseLLMBackend):
    """Ollama HTTP API 後端。

    使用 httpx async client 呼叫 Ollama /api/chat 端點，
    支援 stream / non-stream 與原生 tool calling。
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
    ) -> None:
        """初始化 Ollama 後端。

        Args:
            model: 模型名稱（如 "llama3.2"、"qwen2.5"）。
            base_url: Ollama 服務地址。
            timeout: HTTP 請求逾時秒數。
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)

    def supports_tool_calling(self) -> bool:
        """Ollama 支援原生 tool calling。"""
        return True

    def get_model_info(self) -> ModelInfo:
        """回傳 Ollama 後端的模型資訊。"""
        supports_reasoning_effort = self._supports_reasoning_effort_model(self.model)
        return ModelInfo(
            name=self.model,
            backend_type="ollama",
            provider="ollama",
            context_length=4096,
            supports_tool_calling=True,
            metadata={
                "supports_reasoning_effort": supports_reasoning_effort,
                "reasoning_effort_param": "think" if supports_reasoning_effort else None,
            },
        )

    async def health_check(self) -> bool:
        """嘗試連線 Ollama /api/tags 端點，確認服務可用。"""
        try:
            resp = await self._client.get("/api/tags", timeout=5.0)
            return resp.status_code == 200
        except Exception as exc:
            logger.debug(f"Ollama health check failed: {exc}")
            return False

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
        """呼叫 Ollama /api/chat 進行推理。

        Args:
            messages: 對話訊息列表。
            tools: 可用工具定義列表。
            temperature: 採樣溫度。
            max_tokens: 最大輸出 token 數。
            stream: 是否啟用串流。

        Returns:
            非串流時回傳 GenerationResult，串流時回傳 AsyncIterator[StreamChunk]。
        """
        options: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._serialize_messages(messages),
            "stream": stream,
            "options": options,
        }
        think_value = self._reasoning_effort_to_think_value(reasoning_effort)
        if think_value is not None:
            payload["think"] = think_value
        if tools:
            payload["tools"] = self._serialize_tools(tools)

        if stream:
            return self._stream_generate(payload)
        return await self._blocking_generate(payload)

    async def _blocking_generate(self, payload: dict[str, Any]) -> GenerationResult:
        """執行非串流推理並回傳完整結果。"""
        try:
            resp = await self._client.post("/api/chat", json=payload)
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            logger.error(f"Ollama API error {exc.response.status_code}: {exc.response.text}")
            raise self._wrap_request_error(
                exc,
                stage="generate",
                payload_preview=self._build_payload_preview(payload),
            ) from exc
        except httpx.RequestError as exc:
            logger.error(f"Ollama connection error: {exc}")
            raise self._wrap_request_error(
                exc,
                stage="generate",
                payload_preview=self._build_payload_preview(payload),
            ) from exc

        data = resp.json()
        msg = data.get("message", {})
        content: str = msg.get("content", "")
        raw_thinking = msg.get("thinking", "")
        thinking = raw_thinking if isinstance(raw_thinking, str) else ""

        tool_calls: list[ToolCall] = []
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            raw_args = fn.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    raw_args = {}
            tool_calls.append(
                ToolCall(
                    id=tc.get("id", str(uuid.uuid4())),
                    name=fn.get("name", ""),
                    arguments=raw_args,
                    index=fn.get("index") if isinstance(fn.get("index"), int) else None,
                )
            )

        if not tool_calls:
            content = self._combine_reasoning_and_content(reasoning=thinking, content=content)
            if not content.strip():
                logger.warning("Ollama returned an empty non-tool response.")
                raise BackendRequestError(
                    "Ollama returned an empty response with no content or tool calls.",
                    metadata={
                        "backend_name": "ollama",
                        "request_url": f"{self.base_url}/api/chat",
                        "stage": "generate",
                        "model": self.model,
                        "request_payload_preview": self._build_payload_preview(payload),
                        "response_preview": self._build_response_preview(data),
                    },
                )

        usage = data.get("prompt_eval_count", 0), data.get("eval_count", 0)
        return GenerationResult(
            content=content,
            thinking=thinking,
            tool_calls=tool_calls,
            input_tokens=usage[0],
            output_tokens=usage[1],
            model=data.get("model", self.model),
            finish_reason="tool_calls" if tool_calls else data.get("done_reason", "stop"),
        )

    async def _stream_generate(self, payload: dict[str, Any]) -> AsyncIterator[StreamChunk]:
        """執行串流推理，逐 chunk 回傳 StreamChunk。"""
        try:
            async with self._client.stream("POST", "/api/chat", json=payload) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    done: bool = data.get("done", False)
                    msg = data.get("message", {})
                    delta: str = msg.get("content", "")
                    thinking_delta = msg.get("thinking", "")
                    if isinstance(thinking_delta, str) and thinking_delta:
                        delta = f"<think>{thinking_delta}</think>" if not delta else f"<think>{thinking_delta}</think>{delta}"

                    yield StreamChunk(
                        delta=delta,
                        is_final=done,
                        finish_reason=data.get("done_reason") if done else None,
                    )
                    if done:
                        break
        except httpx.RequestError as exc:
            logger.error(f"Ollama stream error: {exc}")
            raise self._wrap_request_error(exc, stage="stream_generate") from exc

    async def close(self) -> None:
        """關閉 HTTP client 連線。"""
        await self._client.aclose()

    def _serialize_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """Serialize chat history into the Ollama-native message shape."""
        serialized: list[dict[str, Any]] = []
        for message in messages:
            payload: dict[str, Any] = {
                "role": message.role,
                "content": self._serialize_tool_message_content(message)
                if message.role == "tool"
                else message.content,
            }
            if message.thinking:
                payload["thinking"] = message.thinking
            if message.role == "assistant" and message.tool_calls:
                payload["tool_calls"] = [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.name,
                            "arguments": tool_call.arguments,
                            **(
                                {"index": tool_call.index}
                                if tool_call.index is not None
                                else {}
                            ),
                        }
                    }
                    for tool_call in message.tool_calls
                ]
            if message.role == "tool" and message.name:
                payload["tool_name"] = message.name
            serialized.append(payload)
        return serialized

    def _serialize_tools(self, tools: list[ToolSchema]) -> list[dict[str, Any]]:
        """Serialize tool schemas into the Ollama-native tools contract."""
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in tools
        ]

    def _serialize_tool_message_content(self, message: Message) -> str:
        """Strip generic reinjection wrappers from tool messages for Ollama-native turns."""
        if message.role != "tool":
            return message.content

        tool_name = message.name or "tool"
        result_prefix = f"Tool {tool_name} result:\n"
        error_prefix = f"Tool {tool_name} error:\n"
        if message.content.startswith(result_prefix):
            return message.content[len(result_prefix) :]
        if message.content.startswith(error_prefix):
            return f"Error: {message.content[len(error_prefix):]}"
        return message.content

    def _wrap_request_error(
        self,
        exc: httpx.HTTPError,
        *,
        stage: str,
        payload_preview: dict[str, Any] | None = None,
    ) -> BackendRequestError:
        metadata: dict[str, Any] = {
            "backend_name": "ollama",
            "request_url": f"{self.base_url}/api/chat",
            "stage": stage,
            "model": self.model,
        }
        if payload_preview is not None:
            metadata["request_payload_preview"] = payload_preview
        if isinstance(exc, httpx.HTTPStatusError):
            metadata["status_code"] = exc.response.status_code
            metadata["response_text"] = exc.response.text
        return BackendRequestError(str(exc), metadata=metadata)

    def _build_payload_preview(self, payload: dict[str, Any]) -> dict[str, Any]:
        messages = payload.get("messages")
        summarized_messages: list[dict[str, Any]] = []
        if isinstance(messages, list):
            raw_items = messages[-3:]
            for raw in raw_items:
                if not isinstance(raw, dict):
                    continue
                entry: dict[str, Any] = {
                    "role": raw.get("role"),
                    "content_preview": self._truncate_preview(str(raw.get("content", ""))),
                }
                thinking = raw.get("thinking")
                if isinstance(thinking, str) and thinking:
                    entry["thinking_preview"] = self._truncate_preview(thinking)
                tool_name = raw.get("tool_name")
                if isinstance(tool_name, str) and tool_name:
                    entry["tool_name"] = tool_name
                tool_calls = raw.get("tool_calls")
                if isinstance(tool_calls, list):
                    entry["tool_calls"] = [
                        {
                            "id": item.get("id"),
                            "type": item.get("type"),
                            "function": (
                                {
                                    "name": item.get("function", {}).get("name"),
                                    "arguments_type": type(item.get("function", {}).get("arguments")).__name__,
                                }
                                if isinstance(item, dict) and isinstance(item.get("function"), dict)
                                else None
                            ),
                        }
                        for item in tool_calls[:4]
                        if isinstance(item, dict)
                    ]
                summarized_messages.append(entry)

        return {
            "model": payload.get("model"),
            "stream": payload.get("stream"),
            "message_count": len(messages) if isinstance(messages, list) else None,
            "tool_count": len(payload.get("tools", [])) if isinstance(payload.get("tools"), list) else None,
            "tail_messages": summarized_messages,
        }

    def _build_response_preview(self, data: dict[str, Any]) -> dict[str, Any]:
        msg = data.get("message", {})
        preview: dict[str, Any] = {
            "model": data.get("model", self.model),
            "done": data.get("done"),
            "done_reason": data.get("done_reason"),
            "prompt_eval_count": data.get("prompt_eval_count"),
            "eval_count": data.get("eval_count"),
        }
        if isinstance(msg, dict):
            preview["message"] = {
                "role": msg.get("role"),
                "content_preview": self._truncate_preview(str(msg.get("content", ""))),
                "thinking_preview": self._truncate_preview(str(msg.get("thinking", ""))),
                "tool_call_count": (
                    len(msg.get("tool_calls", []))
                    if isinstance(msg.get("tool_calls"), list)
                    else None
                ),
            }
        return preview

    @staticmethod
    def _truncate_preview(value: str, max_chars: int = 240) -> str:
        if len(value) <= max_chars:
            return value
        return value[: max_chars - 14] + "...[truncated]"

    @staticmethod
    def _combine_reasoning_and_content(*, reasoning: str, content: str) -> str:
        if reasoning and content:
            return f"<think>{reasoning}</think>\n\n{content}"
        if reasoning:
            return f"<think>{reasoning}</think>"
        return content

    @staticmethod
    def _supports_reasoning_effort_model(model: str) -> bool:
        """Return whether an Ollama model is known to accept low/medium/high think levels."""
        return "gpt-oss" in model.lower()

    def _reasoning_effort_to_think_value(self, effort: str | None) -> str | None:
        if effort not in {"low", "medium", "high"}:
            return None
        if not self._supports_reasoning_effort_model(self.model):
            return None
        return effort
