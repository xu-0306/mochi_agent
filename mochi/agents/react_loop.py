# Inspired by openclaw/src/agents/pi-embedded-runner design pattern
"""Async ReAct 迴圈 — 核心推理引擎。"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from loguru import logger

from mochi.agents.events import (
    AgentEvent,
    ErrorEvent,
    FinalAnswerEvent,
    ThinkingEvent,
    ToolCallRequestEvent,
    ToolCallResultEvent,
)
from mochi.backends.types import Message

if TYPE_CHECKING:
    from mochi.backends.base import BaseLLMBackend
    from mochi.backends.types import ToolSchema
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
        max_iterations: int = 10,
    ) -> None:
        """初始化 ReAct 迴圈。

        Args:
            backend: LLM 後端實例。
            tool_registry: 工具注冊表，None 表示不使用工具。
            max_iterations: 最大迭代次數（防止無限循環）。
        """
        self._backend = backend
        self._tool_registry = tool_registry
        self._max_iterations = max_iterations

    async def run(
        self,
        system_prompt: str,
        history: list[Message],
        user_message: str,
    ) -> AsyncIterator[AgentEvent]:
        """執行完整 ReAct 迴圈，以非同步生成器形式產出事件流。

        Args:
            system_prompt: 組裝好的系統提示詞。
            history: 對話歷史（不含本輪 user message）。
            user_message: 本輪使用者輸入。

        Yields:
            各類 AgentEvent（工具呼叫、最終回答、錯誤等）。
        """
        messages: list[Message] = [
            Message(role="system", content=system_prompt),
            *history,
            Message(role="user", content=user_message),
        ]
        tools = self._collect_tool_schemas()
        async for event in self._run_nonstream(messages, tools):
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

        try:
            for iteration in range(self._max_iterations):
                logger.debug(f"ReAct iteration {iteration + 1}/{self._max_iterations}")

                result = await self._backend.generate(
                    messages=messages,
                    tools=tools if tools else None,
                    stream=False,
                )

                if not isinstance(result, GenerationResult):
                    yield ErrorEvent(message="Expected GenerationResult in non-stream mode.")
                    return

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
                        if self._tool_registry is not None:
                            try:
                                tr = await self._tool_registry.execute(tc.name, tc.arguments)
                                tool_output = tr.output
                                tool_error = tr.error
                            except Exception as exc:
                                tool_error = str(exc)
                        else:
                            tool_error = "No tool registry configured."

                        tool_content = self._format_tool_message_content(
                            output=tool_output,
                            error=tool_error,
                        )

                        yield ToolCallResultEvent(
                            call_id=tc.id,
                            tool_name=tc.name,
                            result=tool_output,
                            error=tool_error,
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
            yield ErrorEvent(message=str(exc))
            return

        yield FinalAnswerEvent(content=final_text)

    def _format_tool_message_content(self, *, output: Any, error: str | None) -> str:
        """將工具結果轉成可回灌給模型的穩定 JSON 文字。"""
        if error:
            payload: dict[str, Any] = {"ok": False, "error": error}
        else:
            payload = {"ok": True, "output": output}

        return json.dumps(payload, ensure_ascii=False, default=str)
