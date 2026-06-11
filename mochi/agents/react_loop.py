# Inspired by openclaw/src/agents/pi-embedded-runner design pattern
"""Async ReAct loop for tool-capable backends."""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, cast
from urllib.parse import urlsplit, urlunsplit

try:
    from loguru import logger
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal test envs
    import logging

    logger = logging.getLogger(__name__)

from mochi.agents.events import (
    AgentEvent,
    ErrorEvent,
    FinalAnswerEvent,
    StatusEvent,
    ThinkingEvent,
    TextChunkEvent,
    ToolCallRequestEvent,
    ToolCallResultEvent,
)
from mochi.backends.base import BackendRequestError
from mochi.backends.types import Message, StreamChunk, ToolCall
from mochi.tools.base import ToolResult
from mochi.tools.transport_guard import ToolResultTransportGuard

if TYPE_CHECKING:
    from mochi.backends.base import BaseLLMBackend
    from mochi.backends.types import ToolSchema
    from mochi.tools.base import ToolExecutionContext
    from mochi.tools.registry import ToolRegistry


class AsyncReActLoop:
    """Run a non-streaming ReAct loop until a final answer is produced."""

    def __init__(
        self,
        backend: BaseLLMBackend,
        tool_registry: ToolRegistry | None = None,
        tool_execution_context: ToolExecutionContext | None = None,
        max_iterations: int = 10,
        max_tool_message_chars: int = 2000,
    ) -> None:
        self._backend = backend
        self._tool_registry = tool_registry
        self._tool_execution_context = tool_execution_context
        self._max_iterations = max_iterations
        self._max_tool_message_chars = max(256, max_tool_message_chars)
        self._max_repeated_tool_rounds = 3
        self._transport_guard = ToolResultTransportGuard()
        self._last_transport_diagnostics: dict[str, Any] | None = None
        self._turn_messages: list[Message] = []

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
        reasoning_effort: str | None = None,
    ) -> AsyncIterator[AgentEvent]:
        messages: list[Message] = [
            Message(role="system", content=system_prompt),
            *history,
            Message(role="user", content=user_message),
        ]
        self._turn_messages = []
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
            reasoning_effort=reasoning_effort,
        ):
            yield event

    @property
    def turn_messages(self) -> list[Message]:
        return [Message(**message.__dict__) for message in self._turn_messages]

    def _collect_tool_schemas(self) -> list[ToolSchema]:
        from mochi.backends.types import ToolSchema as _ToolSchema

        if self._tool_registry is None:
            return []
        return [
            _ToolSchema(
                name=schema["function"]["name"],
                description=schema["function"]["description"],
                parameters=schema["function"].get("parameters", {}),
            )
            for schema in self._tool_registry.get_schemas()
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
        reasoning_effort: str | None,
    ) -> AsyncIterator[AgentEvent]:
        from mochi.backends.types import GenerationResult

        final_text = ""
        total_input_tokens = 0
        total_output_tokens = 0
        total_generation_time_ms = 0.0
        finish_reason = "stop"
        last_tool_signature: tuple[str, ...] | None = None
        repeated_tool_rounds = 0
        web_fetch_guard_state: dict[str, Any] = {
            "last_failed_url": None,
            "failure_streak": 0,
            "blocked_urls": {},
        }
        web_search_guard_state: dict[str, Any] = {
            "last_failed_query": None,
            "failure_streak": 0,
            "blocked_queries": {},
        }
        literature_state: dict[str, Any] = {
            "research_mode": False,
            "paper_hits": 0,
            "abstract_hits": 0,
            "fetched_docs": 0,
            "fetched_chars": 0,
            "summary_ready": False,
            "prompt_injected": False,
        }
        evidence_guard_state: dict[str, Any] = {
            "requires_more_retrieval": False,
            "nudge_count": 0,
            "followup_attempts": 0,
            "last_low_info_url": None,
            "last_low_info_chars": None,
            "last_low_info_lines": None,
        }

        try:
            for iteration in range(self._max_iterations):
                logger.debug(f"ReAct iteration {iteration + 1}/{self._max_iterations}")
                progress_event = self._build_iteration_progress_event(iteration=iteration + 1)
                if progress_event is not None:
                    yield progress_event

                started_at = time.perf_counter()
                try:
                    streamed_generation = False
                    held_stream_text = ""
                    if self._should_use_streaming_generate():
                        streamed_generation = True
                        hold_visible_stream = self._should_hold_visible_stream(
                            evidence_guard_state,
                            tools,
                        )
                        stream_result = await self._backend.generate(
                            **self._build_generate_kwargs(
                                messages=messages,
                                tools=tools,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                top_p=top_p,
                                min_p=min_p,
                                top_k=top_k,
                                frequency_penalty=frequency_penalty,
                                presence_penalty=presence_penalty,
                                repeat_penalty=repeat_penalty,
                                reasoning_effort=reasoning_effort,
                                stream=True,
                            )
                        )
                        if isinstance(stream_result, GenerationResult):
                            streamed_generation = False
                            result = stream_result
                        else:
                            content_parts: list[str] = []
                            thinking_parts: list[str] = []
                            think_filter_state: dict[str, Any] = {
                                "in_think": False,
                                "buffer": "",
                            }
                            streamed_tool_calls: dict[str, ToolCall] = {}
                            streamed_finish_reason = "stop"

                            async for chunk in cast(AsyncIterator[StreamChunk], stream_result):
                                if chunk.thinking_delta:
                                    thinking_parts.append(chunk.thinking_delta)
                                if chunk.delta:
                                    visible_delta, thinking_delta = self._split_stream_thinking_delta(
                                        chunk.delta,
                                        think_filter_state,
                                    )
                                    if thinking_delta:
                                        thinking_parts.append(thinking_delta)
                                    if visible_delta:
                                        content_parts.append(visible_delta)
                                        if hold_visible_stream:
                                            held_stream_text += visible_delta
                                        else:
                                            yield TextChunkEvent(content=visible_delta)
                                if chunk.tool_call_delta is not None:
                                    self._merge_stream_tool_call(
                                        streamed_tool_calls,
                                        chunk.tool_call_delta,
                                    )
                                if chunk.finish_reason:
                                    streamed_finish_reason = chunk.finish_reason

                            visible_tail, thinking_tail = self._finalize_stream_thinking_delta(
                                think_filter_state,
                            )
                            if thinking_tail:
                                thinking_parts.append(thinking_tail)
                            if visible_tail:
                                content_parts.append(visible_tail)
                                if hold_visible_stream:
                                    held_stream_text += visible_tail
                                else:
                                    yield TextChunkEvent(content=visible_tail)
                            streamed_thinking = "".join(thinking_parts).strip()
                            streamed_tool_call_list = self._ordered_stream_tool_calls(streamed_tool_calls)
                            if streamed_tool_call_list:
                                streamed_finish_reason = "tool_calls"

                            backend_info = self._backend.get_model_info()
                            result = GenerationResult(
                                content="".join(content_parts),
                                thinking=streamed_thinking,
                                tool_calls=streamed_tool_call_list,
                                model=backend_info.name,
                                finish_reason=streamed_finish_reason
                                or ("tool_calls" if streamed_tool_call_list else "stop"),
                            )
                    else:
                        result = await self._backend.generate(
                            **self._build_generate_kwargs(
                                messages=messages,
                                tools=tools,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                top_p=top_p,
                                min_p=min_p,
                                top_k=top_k,
                                frequency_penalty=frequency_penalty,
                                presence_penalty=presence_penalty,
                                repeat_penalty=repeat_penalty,
                                reasoning_effort=reasoning_effort,
                                stream=False,
                            )
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

                if result.tool_calls:
                    current_tool_signature = self._build_tool_call_signature(result.tool_calls)
                    if current_tool_signature == last_tool_signature:
                        repeated_tool_rounds += 1
                    else:
                        last_tool_signature = current_tool_signature
                        repeated_tool_rounds = 1
                    if repeated_tool_rounds >= self._max_repeated_tool_rounds:
                        logger.warning(
                            "ReAct loop detected repeated tool-call pattern "
                            f"({repeated_tool_rounds} consecutive rounds)"
                        )
                        repeated_loop_message = "Model repeated the same tool-call pattern without making progress."
                        backend_detail = self._build_backend_runtime_detail()
                        if backend_detail:
                            repeated_loop_message = f"{repeated_loop_message} ({backend_detail})"
                        yield ErrorEvent(
                            message=repeated_loop_message,
                            code="REPEATED_TOOL_LOOP",
                            metadata=self._build_repeated_tool_loop_metadata(
                                finish_reason=finish_reason,
                                repeated_tool_rounds=repeated_tool_rounds,
                                tool_calls=result.tool_calls,
                            ),
                        )
                        return

                    visible_content, thinking_content = self._split_thinking_blocks(
                        result.content,
                        result.thinking,
                    )
                    if streamed_generation and visible_content:
                        thinking_content = self._combine_reasoning_segments(
                            thinking_content,
                            visible_content,
                        )
                        visible_content = ""
                    result.content = visible_content
                    result.thinking = thinking_content
                    thought_content = thinking_content or visible_content
                    if thought_content:
                        yield ThinkingEvent(
                            content=thought_content,
                            metadata={"source": "model_summary"},
                        )

                    assistant_message = Message(
                        role="assistant",
                        content=visible_content,
                        thinking=thinking_content,
                        tool_calls=result.tool_calls,
                        responses_replay=result.responses_replay,
                    )
                    messages.append(assistant_message)
                    self._remember_turn_message(assistant_message)

                    for tool_call in result.tool_calls:
                        yield ToolCallRequestEvent(
                            call_id=tool_call.id,
                            tool_name=tool_call.name,
                            arguments=tool_call.arguments,
                        )

                        tool_output: Any = None
                        tool_error: str | None = None
                        tool_metadata: dict[str, Any] = {}
                        tool_result_payload = ToolResult(output=None, error=None)
                        self._mark_literature_research_mode(tool_call, literature_state)
                        followup_retrieval_attempt = self._is_followup_retrieval_attempt(
                            tool_call=tool_call,
                            evidence_guard_state=evidence_guard_state,
                        )
                        guarded_tool_result = self._build_guarded_tool_result(
                            tool_call=tool_call,
                            literature_state=literature_state,
                            web_fetch_guard_state=web_fetch_guard_state,
                            web_search_guard_state=web_search_guard_state,
                        )
                        if guarded_tool_result is not None:
                            tool_output = guarded_tool_result.output
                            tool_error = guarded_tool_result.error
                            tool_metadata = (
                                dict(guarded_tool_result.metadata)
                                if isinstance(guarded_tool_result.metadata, dict)
                                else {}
                            )
                            tool_result_payload = guarded_tool_result
                            formatted_content = self._format_tool_message_content(
                                output=tool_output,
                                error=tool_error,
                            )
                        if self._tool_registry is not None:
                            if guarded_tool_result is None:
                                try:
                                    tool_result = await self._tool_registry.execute(
                                        tool_call.name,
                                        tool_call.arguments,
                                        context=self._tool_execution_context,
                                    )
                                    tool_output = tool_result.output
                                    tool_error = tool_result.error
                                    tool_metadata = (
                                        dict(tool_result.metadata)
                                        if isinstance(tool_result.metadata, dict)
                                        else {}
                                    )
                                    tool_result_payload = tool_result
                                    formatted_content = self._tool_registry.format_result_for_model(
                                        tool_call.name,
                                        tool_result,
                                        max_chars=self._max_tool_message_chars,
                                    )
                                except Exception as exc:
                                    tool_error = str(exc)
                                    tool_result_payload = ToolResult(output=tool_output, error=tool_error)
                                    formatted_content = self._format_tool_message_content(
                                        output=tool_output,
                                        error=tool_error,
                                    )
                        elif guarded_tool_result is None:
                            tool_error = "No tool registry configured."
                            tool_result_payload = ToolResult(output=tool_output, error=tool_error)
                            formatted_content = self._format_tool_message_content(
                                output=tool_output,
                                error=tool_error,
                            )

                        self._update_web_fetch_guard_state(
                            tool_call=tool_call,
                            tool_result=tool_result_payload,
                            web_fetch_guard_state=web_fetch_guard_state,
                        )
                        self._update_web_search_guard_state(
                            tool_call=tool_call,
                            tool_result=tool_result_payload,
                            web_search_guard_state=web_search_guard_state,
                        )
                        self._update_literature_state(
                            tool_call=tool_call,
                            tool_result=tool_result_payload,
                            literature_state=literature_state,
                        )

                        tool_content, transport_diagnostics = self._guard_tool_message(
                            tool_name=tool_call.name,
                            result=tool_result_payload,
                            formatted_content=formatted_content,
                        )
                        tool_content, evidence_diagnostics = self._augment_low_information_web_fetch(
                            tool_call=tool_call,
                            tool_result=tool_result_payload,
                            tool_content=tool_content,
                        )
                        if followup_retrieval_attempt:
                            evidence_guard_state["followup_attempts"] = (
                                int(evidence_guard_state.get("followup_attempts") or 0) + 1
                            )
                            if evidence_diagnostics is None:
                                tool_content, evidence_diagnostics = self._augment_insufficient_followup_retrieval(
                                    tool_call=tool_call,
                                    tool_result=tool_result_payload,
                                    tool_content=tool_content,
                                    evidence_guard_state=evidence_guard_state,
                                )
                        self._last_transport_diagnostics = transport_diagnostics
                        tool_metadata = {
                            **tool_metadata,
                            "transport": transport_diagnostics,
                        }
                        if evidence_diagnostics:
                            tool_metadata["evidence_quality"] = evidence_diagnostics
                            if evidence_diagnostics.get("reason") == "low_information_web_fetch":
                                self._mark_low_information_fetch(
                                    evidence_guard_state=evidence_guard_state,
                                    diagnostics=evidence_diagnostics,
                                )
                        elif followup_retrieval_attempt and not tool_error:
                            self._clear_low_information_fetch_guard(evidence_guard_state)

                        yield ToolCallResultEvent(
                            call_id=tool_call.id,
                            tool_name=tool_call.name,
                            result=tool_output,
                            error=tool_error,
                            metadata=tool_metadata,
                        )

                        tool_message = Message(
                            role="tool",
                            content=tool_content,
                            tool_call_id=tool_call.id,
                            name=tool_call.name,
                        )
                        messages.append(tool_message)
                        self._remember_turn_message(tool_message)
                    if literature_state["summary_ready"] and not literature_state["prompt_injected"]:
                        messages.append(
                            Message(
                                role="user",
                                content=self._build_literature_summary_prompt(literature_state),
                            )
                        )
                        literature_state["prompt_injected"] = True
                    continue

                final_text, final_thinking = self._split_thinking_blocks(result.content, result.thinking)
                if self._should_force_followup_retrieval(evidence_guard_state):
                    messages.append(
                        Message(
                            role="user",
                            content=self._build_followup_retrieval_prompt(evidence_guard_state),
                        )
                    )
                    evidence_guard_state["nudge_count"] = int(evidence_guard_state.get("nudge_count") or 0) + 1
                    continue
                if final_thinking:
                    yield ThinkingEvent(
                        content=final_thinking,
                        metadata={"source": "model_summary"},
                    )
                if not final_text.strip():
                    backend_info = self._backend.get_model_info()
                    logger.warning("ReAct loop received an empty final response from the backend.")
                    yield ErrorEvent(
                        message="Model returned an empty response.",
                        metadata={
                            "backend": {
                                "backend_type": backend_info.backend_type,
                                "model": backend_info.name,
                                "finish_reason": finish_reason,
                            }
                        },
                    )
                    return
                if streamed_generation and held_stream_text:
                    yield TextChunkEvent(content=held_stream_text)
                final_assistant_message = Message(
                    role="assistant",
                    content=final_text,
                    thinking=final_thinking,
                    responses_replay=result.responses_replay,
                )
                messages.append(final_assistant_message)
                self._remember_turn_message(final_assistant_message)
                last_tool_signature = None
                repeated_tool_rounds = 0
                break
            else:
                logger.warning(f"ReAct loop reached max iterations ({self._max_iterations})")
                max_iterations_message = (
                    "Model reached the maximum tool-call iterations without producing a final answer."
                )
                backend_detail = self._build_backend_runtime_detail()
                if backend_detail:
                    max_iterations_message = f"{max_iterations_message} ({backend_detail})"
                yield ErrorEvent(
                    message=max_iterations_message,
                    code="MAX_ITERATIONS_REACHED",
                    metadata=self._build_max_iterations_metadata(finish_reason=finish_reason),
                )
                return
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

    def _split_thinking_blocks(self, content: str, thinking: str = "") -> tuple[str, str]:
        state: dict[str, Any] = {"in_think": False, "buffer": ""}
        visible, extracted_thinking = self._split_stream_thinking_delta(content, state)
        visible_tail, thinking_tail = self._finalize_stream_thinking_delta(state)
        combined_thinking = "".join(
            part
            for part in (thinking, extracted_thinking, thinking_tail)
            if part
        ).strip()
        return (visible + visible_tail).strip(), combined_thinking

    @staticmethod
    def _split_stream_thinking_delta(
        delta: str,
        state: dict[str, Any],
    ) -> tuple[str, str]:
        visible_parts: list[str] = []
        thinking_parts: list[str] = []
        buffer = str(state.get("buffer") or "")
        in_think = bool(state.get("in_think"))

        for char in delta:
            if buffer:
                buffer += char
                target = "</think>" if in_think else "<think>"
                lowered_buffer = buffer.lower()
                if target.startswith(lowered_buffer):
                    if lowered_buffer == target:
                        buffer = ""
                        in_think = not in_think
                    continue
                if in_think:
                    thinking_parts.append(buffer)
                else:
                    visible_parts.append(buffer)
                buffer = ""
                continue

            if char == "<":
                buffer = char
                continue

            if in_think:
                thinking_parts.append(char)
            else:
                visible_parts.append(char)

        state["buffer"] = buffer
        state["in_think"] = in_think
        return "".join(visible_parts), "".join(thinking_parts)

    @staticmethod
    def _finalize_stream_thinking_delta(state: dict[str, Any]) -> tuple[str, str]:
        buffer = str(state.get("buffer") or "")
        in_think = bool(state.get("in_think"))
        state["buffer"] = ""
        if not buffer:
            return "", ""
        if in_think:
            return "", buffer
        return buffer, ""

    def _remember_turn_message(self, message: Message) -> None:
        self._turn_messages.append(Message(**message.__dict__))

    def _format_tool_message_content(self, *, output: Any, error: str | None) -> str:
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

    @staticmethod
    def _is_followup_retrieval_attempt(
        *,
        tool_call: ToolCall,
        evidence_guard_state: dict[str, Any],
    ) -> bool:
        if not evidence_guard_state.get("requires_more_retrieval"):
            return False
        return tool_call.name in {"web_search", "web_fetch"}

    @staticmethod
    def _should_hold_visible_stream(
        evidence_guard_state: dict[str, Any],
        tools: list[ToolSchema],
    ) -> bool:
        if evidence_guard_state.get("requires_more_retrieval"):
            return True
        return bool(tools)

    @staticmethod
    def _combine_reasoning_segments(*segments: str) -> str:
        cleaned = [segment.strip() for segment in segments if segment and segment.strip()]
        return "\n\n".join(cleaned)

    @staticmethod
    def _mark_low_information_fetch(
        *,
        evidence_guard_state: dict[str, Any],
        diagnostics: dict[str, Any],
    ) -> None:
        evidence_guard_state["requires_more_retrieval"] = True
        evidence_guard_state["last_low_info_url"] = diagnostics.get("url")
        evidence_guard_state["last_low_info_chars"] = diagnostics.get("chars")
        evidence_guard_state["last_low_info_lines"] = diagnostics.get("lines")

    @staticmethod
    def _clear_low_information_fetch_guard(evidence_guard_state: dict[str, Any]) -> None:
        evidence_guard_state["requires_more_retrieval"] = False
        evidence_guard_state["nudge_count"] = 0
        evidence_guard_state["followup_attempts"] = 0
        evidence_guard_state["last_low_info_url"] = None
        evidence_guard_state["last_low_info_chars"] = None
        evidence_guard_state["last_low_info_lines"] = None

    @staticmethod
    def _should_force_followup_retrieval(evidence_guard_state: dict[str, Any]) -> bool:
        if not evidence_guard_state.get("requires_more_retrieval"):
            return False
        if int(evidence_guard_state.get("followup_attempts") or 0) < 2:
            return True
        return int(evidence_guard_state.get("nudge_count") or 0) < 1

    @staticmethod
    def _build_followup_retrieval_prompt(evidence_guard_state: dict[str, Any]) -> str:
        url = evidence_guard_state.get("last_low_info_url") or "the previous URL"
        chars = evidence_guard_state.get("last_low_info_chars")
        lines = evidence_guard_state.get("last_low_info_lines")
        detail = ""
        if chars is not None and lines is not None:
            detail = f" ({chars} chars across {lines} non-empty lines)"
        return (
            "The previous web_fetch result from "
            f"{url}{detail} is insufficient evidence for a factual answer. "
            "Do not answer from that incomplete page alone. Call another retrieval tool now, "
            "such as web_search with a different query, web_search targeting another source, "
            "or web_fetch for a more specific result URL. "
            "Only provide a final answer after that follow-up retrieval attempt."
        )

    def _augment_insufficient_followup_retrieval(
        self,
        *,
        tool_call: ToolCall,
        tool_result: ToolResult,
        tool_content: str,
        evidence_guard_state: dict[str, Any],
    ) -> tuple[str, dict[str, Any] | None]:
        if not evidence_guard_state.get("requires_more_retrieval"):
            return tool_content, None
        diagnostics = self._diagnose_followup_retrieval_evidence(
            tool_call=tool_call,
            tool_result=tool_result,
            evidence_guard_state=evidence_guard_state,
        )
        if diagnostics is None:
            return tool_content, None

        guidance = (
            "\n\nEvidence quality note: this follow-up retrieval still does not provide "
            "enough evidence to answer factually. Use a different search query, another "
            "source, or a more specific result URL before answering. Only explain the "
            "limitation after multiple follow-up retrieval attempts fail."
        )
        return tool_content + guidance, diagnostics

    def _diagnose_followup_retrieval_evidence(
        self,
        *,
        tool_call: ToolCall,
        tool_result: ToolResult,
        evidence_guard_state: dict[str, Any],
    ) -> dict[str, Any] | None:
        if tool_result.error:
            return {
                "status": "insufficient_evidence",
                "reason": "followup_retrieval_error",
                "tool": tool_call.name,
                "attempts": int(evidence_guard_state.get("followup_attempts") or 0),
            }
        if tool_call.name == "web_search":
            return self._diagnose_web_search_evidence(
                tool_call=tool_call,
                tool_result=tool_result,
                evidence_guard_state=evidence_guard_state,
            )
        if tool_call.name == "web_fetch":
            return None
        return None

    def _diagnose_web_search_evidence(
        self,
        *,
        tool_call: ToolCall,
        tool_result: ToolResult,
        evidence_guard_state: dict[str, Any],
    ) -> dict[str, Any] | None:
        if isinstance(tool_result.output, str) and len(tool_result.output.strip()) >= 24:
            return None
        results = self._extract_web_search_results(tool_result.output)
        if not results:
            return {
                "status": "insufficient_evidence",
                "reason": "web_search_no_results",
                "tool": "web_search",
                "query": tool_call.arguments.get("query"),
                "attempts": int(evidence_guard_state.get("followup_attempts") or 0),
            }

        last_low_info_url = self._normalize_url(str(evidence_guard_state.get("last_low_info_url") or ""))
        normalized_urls = [
            self._normalize_url(str(item.get("url") or ""))
            for item in results
            if isinstance(item, dict)
        ]
        distinct_urls = {url for url in normalized_urls if url}
        different_urls = {
            url for url in distinct_urls
            if not last_low_info_url or url != last_low_info_url
        }
        text_chars = 0
        content_chars = 0
        for item in results:
            if not isinstance(item, dict):
                continue
            title = str(item.get("title") or "").strip()
            snippet = str(item.get("snippet") or "").strip()
            content = str(item.get("content") or "").strip()
            text_chars += len(title) + len(snippet) + len(content)
            content_chars += len(content)

        if different_urls and (text_chars >= 180 or len(different_urls) >= 2 or content_chars >= 120):
            return None

        reason = "web_search_repeated_low_information_source"
        if different_urls:
            reason = "web_search_low_information_results"
        return {
            "status": "insufficient_evidence",
            "reason": reason,
            "tool": "web_search",
            "query": tool_call.arguments.get("query"),
            "result_count": len(results),
            "distinct_url_count": len(distinct_urls),
            "different_url_count": len(different_urls),
            "text_chars": text_chars,
            "attempts": int(evidence_guard_state.get("followup_attempts") or 0),
        }

    @staticmethod
    def _extract_web_search_results(output: Any) -> list[dict[str, Any]]:
        if isinstance(output, dict):
            results = output.get("results")
            if isinstance(results, list):
                return [item for item in results if isinstance(item, dict)]
            return []
        if isinstance(output, list):
            return [item for item in output if isinstance(item, dict)]
        if isinstance(output, str) and output.strip():
            return [{"title": "", "url": "", "snippet": output.strip()}]
        return []

    def _augment_low_information_web_fetch(
        self,
        *,
        tool_call: ToolCall,
        tool_result: ToolResult,
        tool_content: str,
    ) -> tuple[str, dict[str, Any] | None]:
        if tool_call.name != "web_fetch" or tool_result.error:
            return tool_content, None
        output = tool_result.output
        if not isinstance(output, str):
            return tool_content, None
        normalized = re.sub(r"\s+", " ", output).strip()
        if not normalized:
            return tool_content, None
        non_empty_lines = [line.strip() for line in output.splitlines() if line.strip()]
        metadata = tool_result.metadata if isinstance(tool_result.metadata, dict) else {}
        if (
            len(normalized) >= 600
            and len(non_empty_lines) >= 6
            and not self._looks_like_structural_web_fetch_content(
                normalized=normalized,
                non_empty_lines=non_empty_lines,
                metadata=metadata,
            )
        ):
            return tool_content, None

        diagnostics = {
            "status": "insufficient_evidence",
            "reason": "low_information_web_fetch",
            "chars": len(normalized),
            "lines": len(non_empty_lines),
            "url": tool_call.arguments.get("url"),
        }
        guidance = (
            "\n\nEvidence quality note: this web_fetch result contains very little extracted "
            f"page content ({len(normalized)} chars across {len(non_empty_lines)} non-empty lines). "
            "Treat it as insufficient evidence for a factual answer. Use another web_search query, "
            "fetch a more specific result URL, or corroborate with another source before answering. "
            "Only explain the limitation after additional retrieval attempts fail."
        )
        return tool_content + guidance, diagnostics

    @staticmethod
    def _looks_like_structural_web_fetch_content(
        *,
        normalized: str,
        non_empty_lines: list[str],
        metadata: dict[str, Any],
    ) -> bool:
        extractor = str(metadata.get("extractor") or "").strip().lower()
        if extractor != "htmlparser":
            return False
        if len(non_empty_lines) < 6:
            return False

        lower_lines = [line.lower() for line in non_empty_lines]
        unique_lines = {line for line in lower_lines if line}
        if unique_lines and len(unique_lines) <= max(3, len(non_empty_lines) // 3):
            return True

        label_like_lines = sum(
            1
            for line in non_empty_lines
            if len(line) <= 24 and re.fullmatch(r"[\w\u4e00-\u9fff\s:/\-()%.,]+", line)
        )
        if label_like_lines >= max(4, len(non_empty_lines) // 2):
            return True

        punctuation_light = sum(1 for char in normalized if char in ":：|/»›>")
        if punctuation_light >= 6 and len(unique_lines) <= max(4, len(non_empty_lines) // 2):
            return True

        return False

    def _build_backend_error_metadata(self, exc: Exception) -> dict[str, Any]:
        backend_info = self._backend.get_model_info()
        backend_metadata: dict[str, Any] = {
            "backend_type": backend_info.backend_type,
            "model": backend_info.name,
        }
        self._append_backend_runtime_metadata(backend_metadata, backend_info.metadata)
        if isinstance(exc, BackendRequestError):
            backend_metadata.update(exc.metadata)

        metadata: dict[str, Any] = {"backend": backend_metadata}
        if self._last_transport_diagnostics is not None:
            metadata["transport"] = dict(self._last_transport_diagnostics)
        return metadata

    def _build_max_iterations_metadata(self, *, finish_reason: str) -> dict[str, Any]:
        backend_info = self._backend.get_model_info()
        backend_metadata: dict[str, Any] = {
            "backend_type": backend_info.backend_type,
            "model": backend_info.name,
            "finish_reason": finish_reason,
        }
        self._append_backend_runtime_metadata(backend_metadata, backend_info.metadata)

        metadata: dict[str, Any] = {
            "backend": backend_metadata,
            "loop": {
                "max_iterations": self._max_iterations,
            },
        }
        if self._last_transport_diagnostics is not None:
            metadata["transport"] = dict(self._last_transport_diagnostics)
        return metadata

    def _build_repeated_tool_loop_metadata(
        self,
        *,
        finish_reason: str,
        repeated_tool_rounds: int,
        tool_calls: list[ToolCall],
    ) -> dict[str, Any]:
        backend_info = self._backend.get_model_info()
        backend_metadata: dict[str, Any] = {
            "backend_type": backend_info.backend_type,
            "model": backend_info.name,
            "finish_reason": finish_reason,
        }
        self._append_backend_runtime_metadata(backend_metadata, backend_info.metadata)

        metadata: dict[str, Any] = {
            "backend": backend_metadata,
            "loop": {
                "repeated_tool_rounds": repeated_tool_rounds,
                "tool_calls": [
                    {
                        "name": tool_call.name,
                        "arguments": tool_call.arguments,
                    }
                    for tool_call in tool_calls
                ],
            },
        }
        if self._last_transport_diagnostics is not None:
            metadata["transport"] = dict(self._last_transport_diagnostics)
        return metadata

    def _build_iteration_progress_event(self, *, iteration: int) -> StatusEvent | None:
        backend_info = self._backend.get_model_info()
        backend_type = backend_info.backend_type.strip().lower()
        metadata = backend_info.metadata if isinstance(backend_info.metadata, dict) else {}
        tool_mode = metadata.get("tool_call_mode")
        if backend_type in {"gguf", "safetensors", "llama_cpp_server"} and tool_mode != "simulated_fallback":
            return None

        backend_detail = self._build_backend_runtime_detail()
        content = f"Mochi progress: ReAct iteration {iteration}/{self._max_iterations}"
        if backend_detail:
            content = f"{content} ({backend_detail})"
        return StatusEvent(
            content=content,
            metadata={
                "kind": "react_iteration_progress",
                "iteration": iteration,
                "source": "runtime_progress",
            },
        )

    def _build_backend_runtime_detail(self) -> str:
        backend_info = self._backend.get_model_info()
        details: list[str] = []
        metadata = backend_info.metadata if isinstance(backend_info.metadata, dict) else {}
        has_interesting_metadata = False

        api_mode = metadata.get("api_mode")
        if isinstance(api_mode, str) and api_mode:
            has_interesting_metadata = True
            details.append(f"api={api_mode}")

        tool_mode = metadata.get("tool_call_mode")
        if isinstance(tool_mode, str) and tool_mode:
            has_interesting_metadata = True
            details.append(f"tools={tool_mode}")

        native_status = metadata.get("native_tool_calling_status")
        if isinstance(native_status, str) and native_status and native_status != "unknown":
            has_interesting_metadata = True
            details.append(f"native_status={native_status}")

        request_shape = metadata.get("request_shape")
        if isinstance(request_shape, str) and request_shape:
            has_interesting_metadata = True
            details.append(f"request_shape={request_shape}")

        if backend_info.backend_type != "test" or has_interesting_metadata:
            details.insert(0, f"model={backend_info.name}")
            details.insert(0, f"backend={backend_info.backend_type}")

        return ", ".join(details)

    def _should_use_streaming_generate(self) -> bool:
        backend_info = self._backend.get_model_info()
        backend_type = backend_info.backend_type.strip().lower()
        if backend_type not in {"openai_compat", "openai_codex"}:
            return False

        metadata = backend_info.metadata if isinstance(backend_info.metadata, dict) else {}
        if metadata.get("request_shape") == "responses":
            return False
        tool_mode = metadata.get("tool_call_mode")
        if tool_mode == "simulated_fallback":
            return False
        return True

    @staticmethod
    def _build_generate_kwargs(
        *,
        messages: list[Message],
        tools: list[ToolSchema],
        temperature: float,
        max_tokens: int,
        top_p: float,
        min_p: float,
        top_k: int,
        frequency_penalty: float,
        presence_penalty: float,
        repeat_penalty: float,
        reasoning_effort: str | None,
        stream: bool,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "messages": messages,
            "tools": tools if tools else None,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "min_p": min_p,
            "top_k": top_k,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "repeat_penalty": repeat_penalty,
            "stream": stream,
        }
        if reasoning_effort is not None:
            kwargs["reasoning_effort"] = reasoning_effort
        return kwargs

    @staticmethod
    def _merge_stream_tool_call(
        streamed_tool_calls: dict[str, ToolCall],
        next_tool_call: ToolCall,
    ) -> None:
        key = (
            str(next_tool_call.index)
            if next_tool_call.index is not None
            else next_tool_call.id or next_tool_call.name
        )
        current = streamed_tool_calls.get(key)
        if current is None:
            streamed_tool_calls[key] = ToolCall(
                id=next_tool_call.id,
                name=next_tool_call.name,
                arguments=dict(next_tool_call.arguments),
                index=next_tool_call.index,
            )
            return

        current.id = next_tool_call.id or current.id
        current.name = next_tool_call.name or current.name
        if next_tool_call.arguments:
            current.arguments = dict(next_tool_call.arguments)
        if next_tool_call.index is not None:
            current.index = next_tool_call.index

    @staticmethod
    def _ordered_stream_tool_calls(streamed_tool_calls: dict[str, ToolCall]) -> list[ToolCall]:
        return sorted(
            streamed_tool_calls.values(),
            key=lambda tool_call: (
                tool_call.index is None,
                tool_call.index if tool_call.index is not None else 0,
                tool_call.id,
            ),
        )

    @staticmethod
    def _append_backend_runtime_metadata(
        backend_metadata: dict[str, Any],
        metadata: Any,
    ) -> None:
        if not isinstance(metadata, dict):
            return
        for key in ("api_mode", "tool_call_mode", "native_tool_calling_status", "request_shape"):
            value = metadata.get(key)
            if isinstance(value, str) and value:
                backend_metadata[key] = value

    @staticmethod
    def _build_tool_call_signature(tool_calls: list[ToolCall]) -> tuple[str, ...]:
        return tuple(
            f"{tool_call.name}:{json.dumps(AsyncReActLoop._normalize_tool_arguments(tool_call.arguments), ensure_ascii=False, sort_keys=True, default=str)}"
            for tool_call in tool_calls
        )

    @staticmethod
    def _normalize_tool_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            str(key): AsyncReActLoop._normalize_tool_argument_value(str(key), value)
            for key, value in sorted(arguments.items())
        }

    @staticmethod
    def _normalize_tool_argument_value(key: str, value: Any) -> Any:
        if isinstance(value, str):
            if key == "url":
                return AsyncReActLoop._normalize_url(value)
            if key in {"query", "title", "doi", "pmid", "paper_id", "arxiv_id"}:
                return AsyncReActLoop._normalize_text(value)
            return value.strip()
        if isinstance(value, dict):
            return {
                str(child_key): AsyncReActLoop._normalize_tool_argument_value(str(child_key), child_value)
                for child_key, child_value in sorted(value.items())
            }
        if isinstance(value, list):
            return [AsyncReActLoop._normalize_tool_argument_value(key, item) for item in value]
        return value

    @staticmethod
    def _normalize_text(value: str) -> str:
        lowered = value.strip().lower()
        lowered = re.sub(r"[\s\-_]+", " ", lowered)
        return re.sub(r"\s+", " ", lowered)

    @staticmethod
    def _normalize_url(url: str) -> str:
        cleaned = url.strip()
        if not cleaned:
            return ""
        parts = urlsplit(cleaned)
        path = parts.path.rstrip("/")
        query = "&".join(sorted(filter(None, parts.query.split("&"))))
        return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, query, ""))

    def _build_guarded_tool_result(
        self,
        *,
        tool_call: ToolCall,
        literature_state: dict[str, Any],
        web_fetch_guard_state: dict[str, Any],
        web_search_guard_state: dict[str, Any],
    ) -> ToolResult | None:
        if tool_call.name == "web_fetch":
            normalized_url = self._normalize_url(str(tool_call.arguments.get("url", "")))
            blocked_urls = web_fetch_guard_state.get("blocked_urls", {})
            if isinstance(blocked_urls, dict) and normalized_url in blocked_urls:
                blocked_info = blocked_urls[normalized_url]
                failure_count = blocked_info.get("failure_count", 2) if isinstance(blocked_info, dict) else 2
                status_code = blocked_info.get("status_code") if isinstance(blocked_info, dict) else None
                return ToolResult(
                    error=(
                        "Skipping repeated fetch because this URL already failed multiple times: "
                        f"{tool_call.arguments.get('url', '')}"
                    ),
                    metadata={
                        "url": tool_call.arguments.get("url", ""),
                        "status_code": status_code,
                        "guard": "repeated_web_fetch_failure",
                        "failure_count": failure_count,
                    },
                    retryable=False,
                    suggestion="Use another source or summarize from the evidence already collected.",
                )
        if tool_call.name == "web_search":
            normalized_query = self._normalize_text(str(tool_call.arguments.get("query", "")))
            blocked_queries = web_search_guard_state.get("blocked_queries", {})
            if isinstance(blocked_queries, dict) and normalized_query in blocked_queries:
                blocked_info = blocked_queries[normalized_query]
                failure_count = blocked_info.get("failure_count", 2) if isinstance(blocked_info, dict) else 2
                provider = blocked_info.get("provider") if isinstance(blocked_info, dict) else None
                return ToolResult(
                    error=(
                        "Skipping repeated search because this query already failed multiple times: "
                        f"{tool_call.arguments.get('query', '')}"
                    ),
                    metadata={
                        "query": tool_call.arguments.get("query", ""),
                        "provider": provider,
                        "guard": "repeated_web_search_failure",
                        "failure_count": failure_count,
                    },
                    retryable=False,
                    suggestion=(
                        "Try a different query, switch to another search provider, "
                        "or answer from the evidence already collected."
                    ),
                )

        if literature_state["summary_ready"] and self._is_research_retrieval_tool_call(tool_call):
            return ToolResult(
                error=(
                    "Sufficient literature evidence is already collected. "
                    "Do not call more search or fetch tools; synthesize the answer now."
                ),
                metadata={
                    "guard": "literature_summary_ready",
                    "paper_hits": literature_state["paper_hits"],
                    "abstract_hits": literature_state["abstract_hits"],
                    "fetched_docs": literature_state["fetched_docs"],
                    "fetched_chars": literature_state["fetched_chars"],
                },
                retryable=False,
                suggestion="Summarize the retrieved papers now and cite the most relevant evidence.",
            )
        return None

    def _update_web_fetch_guard_state(
        self,
        *,
        tool_call: ToolCall,
        tool_result: ToolResult,
        web_fetch_guard_state: dict[str, Any],
    ) -> None:
        if tool_call.name != "web_fetch":
            return
        normalized_url = self._normalize_url(str(tool_call.arguments.get("url", "")))
        if not normalized_url:
            return

        blocked_urls = web_fetch_guard_state.setdefault("blocked_urls", {})
        if not isinstance(blocked_urls, dict):
            blocked_urls = {}
            web_fetch_guard_state["blocked_urls"] = blocked_urls

        if tool_result.error:
            last_failed_url = web_fetch_guard_state.get("last_failed_url")
            failure_streak = int(web_fetch_guard_state.get("failure_streak", 0))
            failure_streak = failure_streak + 1 if last_failed_url == normalized_url else 1
            web_fetch_guard_state["last_failed_url"] = normalized_url
            web_fetch_guard_state["failure_streak"] = failure_streak
            if failure_streak >= 2:
                blocked_urls[normalized_url] = {
                    "failure_count": failure_streak,
                    "status_code": tool_result.metadata.get("status_code"),
                }
            return

        if web_fetch_guard_state.get("last_failed_url") == normalized_url:
            web_fetch_guard_state["last_failed_url"] = None
            web_fetch_guard_state["failure_streak"] = 0
        blocked_urls.pop(normalized_url, None)

    def _update_web_search_guard_state(
        self,
        *,
        tool_call: ToolCall,
        tool_result: ToolResult,
        web_search_guard_state: dict[str, Any],
    ) -> None:
        if tool_call.name != "web_search":
            return

        normalized_query = self._normalize_text(str(tool_call.arguments.get("query", "")))
        if not normalized_query:
            return

        blocked_queries = web_search_guard_state.setdefault("blocked_queries", {})
        if not isinstance(blocked_queries, dict):
            blocked_queries = {}
            web_search_guard_state["blocked_queries"] = blocked_queries

        if tool_result.error:
            last_failed_query = web_search_guard_state.get("last_failed_query")
            failure_streak = int(web_search_guard_state.get("failure_streak", 0))
            failure_streak = failure_streak + 1 if last_failed_query == normalized_query else 1
            web_search_guard_state["last_failed_query"] = normalized_query
            web_search_guard_state["failure_streak"] = failure_streak
            if failure_streak >= 2:
                provider = tool_result.metadata.get("engine") or tool_result.metadata.get("provider")
                blocked_queries[normalized_query] = {
                    "failure_count": failure_streak,
                    "provider": provider,
                }
            return

        if web_search_guard_state.get("last_failed_query") == normalized_query:
            web_search_guard_state["last_failed_query"] = None
            web_search_guard_state["failure_streak"] = 0
        blocked_queries.pop(normalized_query, None)

    def _mark_literature_research_mode(self, tool_call: ToolCall, literature_state: dict[str, Any]) -> None:
        if self._is_literature_tool_name(tool_call.name) or self._tool_call_mentions_literature(tool_call):
            literature_state["research_mode"] = True

    def _update_literature_state(
        self,
        *,
        tool_call: ToolCall,
        tool_result: ToolResult,
        literature_state: dict[str, Any],
    ) -> None:
        self._mark_literature_research_mode(tool_call, literature_state)
        if tool_result.error:
            return

        if tool_call.name in {"arxiv_search", "semantic_scholar_search", "crossref_search", "pubmed_search"}:
            output = tool_result.output if isinstance(tool_result.output, list) else []
            literature_state["paper_hits"] += len(output)
            literature_state["abstract_hits"] += sum(
                1
                for item in output
                if isinstance(item, dict)
                and any(
                    isinstance(item.get(field), str) and item.get(field, "").strip()
                    for field in ("abstract", "summary")
                )
            )
        elif tool_call.name == "web_fetch":
            url = str(tool_call.arguments.get("url", ""))
            if literature_state["research_mode"] or self._looks_like_academic_url(url):
                output = tool_result.output
                if isinstance(output, str) and output.strip():
                    literature_state["fetched_docs"] += 1
                    literature_state["fetched_chars"] += len(output)

        literature_state["summary_ready"] = self._has_sufficient_literature_evidence(literature_state)

    @staticmethod
    def _is_literature_tool_name(tool_name: str) -> bool:
        return tool_name in {"arxiv_search", "semantic_scholar_search", "crossref_search", "pubmed_search"}

    def _tool_call_mentions_literature(self, tool_call: ToolCall) -> bool:
        query = tool_call.arguments.get("query")
        if isinstance(query, str):
            normalized = self._normalize_text(query)
            if any(
                token in normalized
                for token in (
                    "paper",
                    "papers",
                    "literature",
                    "research",
                    "study",
                    "studies",
                    "doi",
                    "arxiv",
                    "pubmed",
                    "論文",
                    "文獻",
                    "研究",
                )
            ):
                return True
        url = tool_call.arguments.get("url")
        return isinstance(url, str) and self._looks_like_academic_url(url)

    def _is_research_retrieval_tool_call(self, tool_call: ToolCall) -> bool:
        return tool_call.name in {
            "web_search",
            "web_fetch",
            "arxiv_search",
            "semantic_scholar_search",
            "crossref_search",
            "pubmed_search",
        }

    def _has_sufficient_literature_evidence(self, literature_state: dict[str, Any]) -> bool:
        if not literature_state["research_mode"]:
            return False
        paper_hits = int(literature_state["paper_hits"])
        abstract_hits = int(literature_state["abstract_hits"])
        fetched_docs = int(literature_state["fetched_docs"])
        fetched_chars = int(literature_state["fetched_chars"])
        return (
            (paper_hits >= 3 and abstract_hits >= 2)
            or (paper_hits >= 2 and fetched_docs >= 1 and (abstract_hits >= 1 or fetched_chars >= 2500))
            or (paper_hits >= 2 and fetched_docs >= 2)
        )

    def _build_literature_summary_prompt(self, literature_state: dict[str, Any]) -> str:
        return (
            "You already have enough literature evidence. "
            f"Collected about {literature_state['paper_hits']} paper records, "
            f"{literature_state['abstract_hits']} abstracts, and "
            f"{literature_state['fetched_docs']} fetched documents. "
            "Stop calling more search or fetch tools unless a core fact is still missing. "
            "Now synthesize the retrieved papers, compare the most relevant findings, and mention uncertainty when needed."
        )

    @staticmethod
    def _looks_like_academic_url(url: str) -> bool:
        normalized_url = AsyncReActLoop._normalize_url(url)
        return any(
            host_fragment in normalized_url
            for host_fragment in (
                "arxiv.org",
                "doi.org",
                "semanticscholar.org",
                "pubmed.ncbi.nlm.nih.gov",
                "ncbi.nlm.nih.gov",
                "openreview.net",
                "aclanthology.org",
                "biorxiv.org",
                "medrxiv.org",
                "springer.com",
                "nature.com",
                "sciencedirect.com",
            )
        )

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
