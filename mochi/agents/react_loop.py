# Inspired by openclaw/src/agents/pi-embedded-runner design pattern
"""Async ReAct 迴圈 — 核心推理引擎。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
import time
from typing import TYPE_CHECKING, Any

try:
    from loguru import logger
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal test envs
    import logging

    logger = logging.getLogger(__name__)

from mochi.agents.events import (
    AgentEvent,
    ErrorEvent,
    FinalAnswerEvent,
    ThinkingEvent,
    ToolCallRequestEvent,
    ToolCallResultEvent,
)
from mochi.backends.base import BackendRequestError
from mochi.backends.types import Message
from mochi.tools.base import ToolResult
from mochi.tools.transport_guard import ToolResultTransportGuard

if TYPE_CHECKING:
    from mochi.backends.base import BaseLLMBackend
    from mochi.backends.types import ToolSchema
    from mochi.tools.base import ToolExecutionContext
    from mochi.tools.registry import ToolRegistry


class AsyncReActLoop:
    """非同步 ReAct（Reasoning + Acting）迴圈。

    流程（非串流模式，Phase 1）：
        1. 組裝 messages（system + history + user）
        2. 呼叫 LLM（non-stream）
        3. 若 LLM 請求工具呼叫 → 執行工具 → 觀察結果 → 繼續
        4. 無工具呼叫 → 輸出最終回答，結束迴圈
    """

    def __init__(
        self,
        backend: BaseLLMBackend,
        tool_registry: ToolRegistry | None = None,
        tool_execution_context: ToolExecutionContext | None = None,
        max_iterations: int = 10,
        max_tool_message_chars: int = 2000,
    ) -> None:
        """初始化 ReAct 迴圈。

        Args:
            backend: LLM 後端實例。
            tool_registry: 工具注冊表，None 表示不使用工具。
            max_iterations: 最大迭代次數（防止無限循環）。
        """
        self._backend = backend
        self._tool_registry = tool_registry
        self._tool_execution_context = tool_execution_context
        self._max_iterations = max_iterations
        self._max_tool_message_chars = max(256, max_tool_message_chars)
        self._transport_guard = ToolResultTransportGuard()
        self._last_transport_diagnostics: dict[str, Any] | None = None

    async def run(
        self,
        system_prompt: str,
        history: list[Message],
        user_message: str,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.0,
    ) -> AsyncIterator[AgentEvent]:
        """執行完整 ReAct 迴圈，以非同步生成器形式產出事件流。

        Args:
            system_prompt: 組裝好的系統提示詞。
            history: 對話歷史（不含本輪 user message）。
            user_message: 本輪使用者輸入。
            temperature: 採樣溫度。
            max_tokens: 最大輸出 token 數。
            top_p: Top-p 取樣。
            min_p: Min-p 取樣。
            top_k: Top-k 取樣。
            frequency_penalty: Frequency penalty。
            presence_penalty: Presence penalty。
            repeat_penalty: Repeat penalty。

        Yields:
            各類 AgentEvent（工具呼叫、最終回答、錯誤等）。
        """
        messages: list[Message] = [
            Message(role="system", content=system_prompt),
            *history,
            Message(role="user", content=user_message),
        ]
        tools = self._collect_tool_schemas()
        async for event in self._run_nonstream(
            messages,
            tools,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
        ):
            yield event

    def _collect_tool_schemas(self) -> list[ToolSchema]:
        """從 tool_registry 收集工具 schema。"""
        from mochi.backends.types import ToolSchema as _TS

        if self._tool_registry is None:
            return []
        return [
            _TS(
                name=s["function"]["name"],
                description=s["function"]["description"],
                parameters=s["function"].get("parameters", {}),
            )
            for s in self._tool_registry.get_schemas()
        ]

    async def _run_nonstream(
        self,
        messages: list[Message],
        tools: list[ToolSchema],
        *,
        temperature: float,
        max_tokens: int,
        top_p: float,
        min_p: float,
        top_k: int,
        frequency_penalty: float,
        presence_penalty: float,
        repeat_penalty: float,
    ) -> AsyncIterator[AgentEvent]:
        """非串流模式 ReAct 迴圈（Phase 1 主要執行路徑）。

        Args:
            messages: 已組裝的訊息列表。
            tools: 可用工具列表。

        Yields:
            AgentEvent 事件流。
        """
        from mochi.backends.types import GenerationResult

        final_text = ""
        total_input_tokens = 0
        total_output_tokens = 0
        total_generation_time_ms = 0.0
        finish_reason = "stop"

        try:
            for iteration in range(self._max_iterations):
                logger.debug(f"ReAct iteration {iteration + 1}/{self._max_iterations}")

                started_at = time.perf_counter()
                try:
                    result = await self._backend.generate(
                        messages=messages,
                        tools=tools if tools else None,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                        min_p=min_p,
                        top_k=top_k,
                        frequency_penalty=frequency_penalty,
                        presence_penalty=presence_penalty,
                        repeat_penalty=repeat_penalty,
                        stream=False,
                    )
                except Exception as exc:
                    logger.exception(f"ReAct loop backend error: {exc}")
                    yield ErrorEvent(
                        message=str(exc),
                        metadata=self._build_backend_error_metadata(exc),
                    )
                    return
                total_generation_time_ms += (time.perf_counter() - started_at) * 1000.0

                if not isinstance(result, GenerationResult):
                    yield ErrorEvent(message="Expected GenerationResult in non-stream mode.")
                    return

                total_input_tokens += result.input_tokens
                total_output_tokens += result.output_tokens
                finish_reason = result.finish_reason or finish_reason

                # 若有工具呼叫
                if result.tool_calls:
                    if result.content:
                        yield ThinkingEvent(content=result.content)

                    messages.append(
                        Message(
                            role="assistant",
                            content=result.content,
                            tool_calls=result.tool_calls,
                        )
                    )

                    for tc in result.tool_calls:
                        yield ToolCallRequestEvent(
                            call_id=tc.id,
                            tool_name=tc.name,
                            arguments=tc.arguments,
                        )

                        tool_output: Any = None
                        tool_error: str | None = None
                        tool_metadata: dict[str, Any] = {}
                        tool_result_payload = ToolResult(output=None, error=None)
                        if self._tool_registry is not None:
                            try:
                                tr = await self._tool_registry.execute(
                                    tc.name,
                                    tc.arguments,
                                    context=self._tool_execution_context,
                                )
                                tool_output = tr.output
                                tool_error = tr.error
                                tool_metadata = tr.metadata
                                tool_result_payload = tr
                                formatted_content = self._tool_registry.format_result_for_model(
                                    tc.name,
                                    tr,
                                    max_chars=self._max_tool_message_chars,
                                )
                            except Exception as exc:
                                tool_error = str(exc)
                                tool_result_payload = ToolResult(output=tool_output, error=tool_error)
                                formatted_content = self._format_tool_message_content(
                                    output=tool_output,
                                    error=tool_error,
                                )
                        else:
                            tool_error = "No tool registry configured."
                            tool_result_payload = ToolResult(output=tool_output, error=tool_error)
                            formatted_content = self._format_tool_message_content(
                                output=tool_output,
                                error=tool_error,
                            )

                        tool_content, transport_diagnostics = self._guard_tool_message(
                            tool_name=tc.name,
                            result=tool_result_payload,
                            formatted_content=formatted_content,
                        )
                        self._last_transport_diagnostics = transport_diagnostics
                        tool_metadata = {
                            **tool_metadata,
                            "transport": transport_diagnostics,
                        }

                        yield ToolCallResultEvent(
                            call_id=tc.id,
                            tool_name=tc.name,
                            result=tool_output,
                            error=tool_error,
                            metadata=tool_metadata,
                        )

                        messages.append(
                            Message(
                                role="tool",
                                content=tool_content,
                                tool_call_id=tc.id,
                                name=tc.name,
                            )
                        )
                    continue  # 繼續下一輪

                # 無工具呼叫 → 最終回答
                final_text = result.content
                break

            else:
                logger.warning(f"ReAct loop reached max iterations ({self._max_iterations})")

        except Exception as exc:
            logger.exception(f"ReAct loop error: {exc}")
            yield ErrorEvent(
                message=str(exc),
                metadata=self._build_backend_error_metadata(exc),
            )
            return

        yield FinalAnswerEvent(
            content=final_text,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            generation_time_ms=total_generation_time_ms,
            finish_reason=finish_reason,
        )

    def _format_tool_message_content(self, *, output: Any, error: str | None) -> str:
        """將工具結果轉成可回灌給模型的穩定 JSON 文字。"""
        if error:
            payload: dict[str, Any] = {"ok": False, "error": error}
        else:
            payload = {"ok": True, "output": output}

        serialized = json.dumps(payload, ensure_ascii=False, default=str)
        if len(serialized) <= self._max_tool_message_chars:
            return serialized

        if error:
            compact_payload = {
                "ok": False,
                "error": self._truncate_text(error, self._max_tool_message_chars // 2),
                "truncated": True,
                "original_length": len(error),
            }
        elif isinstance(output, str):
            compact_payload = {
                "ok": True,
                "output": self._truncate_text(output, self._max_tool_message_chars // 2),
                "truncated": True,
                "original_length": len(output),
            }
        else:
            compact_payload = {
                "ok": True,
                "output_preview": self._truncate_text(
                    json.dumps(output, ensure_ascii=False, default=str),
                    self._max_tool_message_chars // 2,
                ),
                "truncated": True,
                "original_length": len(serialized),
            }
        return json.dumps(compact_payload, ensure_ascii=False, default=str)

    def _guard_tool_message(
        self,
        *,
        tool_name: str,
        result: ToolResult,
        formatted_content: str,
    ) -> tuple[str, dict[str, Any]]:
        backend_info = self._backend.get_model_info()
        api_mode = None
        if isinstance(backend_info.metadata, dict):
            raw_mode = backend_info.metadata.get("api_mode")
            if isinstance(raw_mode, str):
                api_mode = raw_mode
        outcome = self._transport_guard.guard(
            tool_name=tool_name,
            result=result,
            formatted_content=formatted_content,
            context=self._tool_execution_context,
            max_chars=self._max_tool_message_chars,
            backend_name=backend_info.backend_type,
            api_mode=api_mode,
        )
        diagnostics = dict(outcome.diagnostics)
        diagnostics["last_tool_name"] = tool_name
        return outcome.content, diagnostics

    def _build_backend_error_metadata(self, exc: Exception) -> dict[str, Any]:
        backend_info = self._backend.get_model_info()
        backend_metadata: dict[str, Any] = {
            "backend_type": backend_info.backend_type,
            "model": backend_info.name,
        }
        if isinstance(backend_info.metadata, dict):
            api_mode = backend_info.metadata.get("api_mode")
            if isinstance(api_mode, str):
                backend_metadata["api_mode"] = api_mode
        if isinstance(exc, BackendRequestError):
            backend_metadata.update(exc.metadata)

        metadata: dict[str, Any] = {"backend": backend_metadata}
        if self._last_transport_diagnostics is not None:
            metadata["transport"] = dict(self._last_transport_diagnostics)
        return metadata

    @staticmethod
    def _truncate_text(value: str, max_chars: int) -> str:
        if max_chars <= 0:
            return ""
        if len(value) <= max_chars:
            return value
        suffix = "...[truncated]"
        if max_chars <= len(suffix):
            return suffix[:max_chars]
        return value[: max_chars - len(suffix)] + suffix
