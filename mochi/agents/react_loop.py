# Inspired by openclaw/src/agents/pi-embedded-runner design pattern
"""Async ReAct loop for tool-capable backends."""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any, Mapping, cast
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

try:
    from loguru import logger
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal test envs
    import logging

    logger = logging.getLogger(__name__)

from mochi.agents.events import (
    AgentEvent,
    AssistantTruncatedEvent,
    ErrorEvent,
    FinalAnswerEvent,
    StatusEvent,
    ThinkingEvent,
    TextChunkEvent,
    ToolCallCompletedEvent,
    ToolCallCreatedEvent,
    ToolCallRequestEvent,
    ToolCallResultEvent,
)
from mochi.backends.base import BackendRequestError
from mochi.backends.tool_call_parsers import parse_tool_calls, strip_tool_call_blocks
from mochi.backends.types import Message, StreamChunk, ToolCall
from mochi.tools.base import ToolResult
from mochi.tools.transport_guard import ToolResultTransportGuard

if TYPE_CHECKING:
    from mochi.backends.base import BaseLLMBackend
    from mochi.backends.types import ToolSchema
    from mochi.tools.base import ToolExecutionContext
    from mochi.tools.registry import ToolRegistry


_CHANNEL_MARKER_RE = re.compile(
    r"(?:<\|channel\|?>|<channel\|>|<\|message\|?>|<message\|>)",
    re.IGNORECASE,
)
_HEADER_MARKER_RE = re.compile(
    r"(?:<\|start_header_id\|>|<\|end_header_id\|>|<\|im_start\|>|<\|im_end\|>|<\|eot_id\|>)",
    re.IGNORECASE,
)
_ROLE_SENTINEL_RE = re.compile(
    r"<[|｜]\s*(?:assistant|user|system|tool)\s*[|｜]>",
    re.IGNORECASE,
)
_CHANNEL_REASONING_PREFIX_RE = re.compile(
    r"^\s*(?:(?:<\|channel\|?>|<channel\|>|<\|message\|?>|<message\|>)\s*)+"
    r"(?:(?:thought|analysis|reasoning)\s*)?"
    r"(?:(?:<\|channel\|?>|<channel\|>|<\|message\|?>|<message\|>)\s*)*",
    re.IGNORECASE,
)
_ROLE_PREFIX_RE = re.compile(
    r"^\s*(?:(?:<\|start_header_id\|>|<\|im_start\|>)\s*)+"
    r"(?:assistant|user|system|tool)\s*"
    r"(?:(?:<\|end_header_id\|>|<\|im_end\|>|<\|eot_id\|>)\s*)*",
    re.IGNORECASE,
)
_ROLE_SENTINEL_PREFIX_RE = re.compile(
    r"^\s*<[|｜]\s*(?:assistant|user|system|tool)\s*[|｜]>\s*",
    re.IGNORECASE,
)
_REASONING_TAGS: tuple[str, ...] = ("think", "analysis", "reasoning")
_REASONING_OPEN_TAGS: tuple[str, ...] = tuple(f"<{tag}>" for tag in _REASONING_TAGS)


def _find_opening_reasoning_tag(source: str) -> tuple[int, str, str] | None:
    lowered = source.lower()
    best_match: tuple[int, str, str] | None = None
    for tag in _REASONING_TAGS:
        token = f"<{tag}>"
        index = lowered.find(token)
        if index == -1:
            continue
        if best_match is None or index < best_match[0]:
            best_match = (index, tag, token)
    return best_match


def _find_closing_reasoning_tag(source: str) -> tuple[int, str, str] | None:
    lowered = source.lower()
    best_match: tuple[int, str, str] | None = None
    for tag in _REASONING_TAGS:
        token = f"</{tag}>"
        index = lowered.find(token)
        if index == -1:
            continue
        if best_match is None or index < best_match[0]:
            best_match = (index, tag, token)
    return best_match


def _find_partial_tag_suffix(source: str, candidates: tuple[str, ...]) -> str:
    if not source or not candidates:
        return ""
    lowered = source.lower()
    max_size = min(len(source), max(len(candidate) - 1 for candidate in candidates))
    for size in range(max_size, 0, -1):
        suffix = lowered[-size:]
        if any(candidate.startswith(suffix) and candidate != suffix for candidate in candidates):
            return source[-size:]
    return ""


class AsyncReActLoop:
    """Run a non-streaming ReAct loop until a final answer is produced."""

    def __init__(
        self,
        backend: BaseLLMBackend,
        tool_registry: ToolRegistry | None = None,
        tool_execution_context: ToolExecutionContext | None = None,
        max_iterations: int = 10,
        max_tool_message_chars: int = 2000,
        requires_file_mutation: bool = False,
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
        self._requires_file_mutation = bool(requires_file_mutation)

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
            start_iteration=0,
            initial_file_mutation_satisfied=None,
        ):
            yield event

    async def resume_from_ordinary_chat_approval(
        self,
        *,
        checkpoint: Mapping[str, Any],
        tool_result: ToolResult,
    ) -> AsyncIterator[AgentEvent]:
        """Continue the interrupted ReAct turn after its exact tool result is available."""
        continuation = checkpoint.get("react_continuation")
        if not isinstance(continuation, Mapping):
            raise ValueError("Ordinary-Chat approval is missing its ReAct continuation checkpoint.")
        messages = self._messages_from_ordinary_chat_checkpoint(continuation)
        cursor = checkpoint.get("resume_cursor")
        if not isinstance(cursor, Mapping):
            raise ValueError("Ordinary-Chat approval is missing its ReAct resume cursor.")
        tool_call_id = str(cursor.get("tool_call_id") or "").strip()
        tool_name = str(cursor.get("tool_name") or checkpoint.get("tool_name") or "").strip()
        if not tool_call_id or not tool_name:
            raise ValueError("Ordinary-Chat approval resume cursor is invalid.")

        tool_call = next(
            (
                candidate
                for message in reversed(messages)
                if message.role == "assistant"
                for candidate in message.tool_calls
                if candidate.id == tool_call_id and candidate.name == tool_name
            ),
            None,
        )
        if tool_call is None:
            raise ValueError("Ordinary-Chat approval cursor does not match its original tool call.")

        expected_tool_names = continuation.get("callable_tool_names")
        if not isinstance(expected_tool_names, list) or tool_name not in expected_tool_names:
            raise ValueError("Ordinary-Chat approval checkpoint does not authorize its original tool.")
        tools = self._collect_tool_schemas()
        available_tool_names = {tool.name for tool in tools}
        missing_tools = {
            str(name)
            for name in expected_tool_names
            if isinstance(name, str) and name and name not in available_tool_names
        }
        if missing_tools:
            raise ValueError("Ordinary-Chat approval continuation tools are no longer available.")

        if self._tool_registry is None:
            raise ValueError("Ordinary-Chat approval continuation has no tool registry.")
        tool_definition = self._tool_registry.get(tool_name)
        formatted_content = (
            tool_definition.format_result_for_model(
                tool_result,
                max_chars=self._max_tool_message_chars,
            )
            if tool_definition is not None
            else self._tool_registry.format_result_for_model(
                tool_name,
                tool_result,
                max_chars=self._max_tool_message_chars,
            )
        )
        tool_content, transport_diagnostics = self._guard_tool_message(
            tool_name=tool_name,
            result=tool_result,
            formatted_content=formatted_content,
        )
        metadata = (
            dict(tool_result.metadata)
            if isinstance(tool_result.metadata, dict)
            else {}
        )
        metadata.update(
            {
                "approval_continuation": True,
                "approval_continuation_source": "ordinary_chat",
                "transport": transport_diagnostics,
            }
        )
        self._turn_messages = []
        yield ToolCallResultEvent(
            call_id=tool_call.id,
            tool_name=tool_call.name,
            result=tool_result.output,
            error=tool_result.error,
            metadata=metadata,
        )
        yield ToolCallCompletedEvent(
            call_id=tool_call.id,
            tool_name=tool_call.name,
            arguments=tool_call.arguments,
            result=tool_result.output,
            error=tool_result.error,
            metadata={**metadata, "compat_event_type": "tool_call_result"},
        )
        tool_message = Message(
            role="tool",
            content=tool_content,
            tool_call_id=tool_call.id,
            name=tool_call.name,
        )
        messages.append(tool_message)
        self._remember_turn_message(tool_message)

        generation = continuation.get("generation")
        if not isinstance(generation, Mapping):
            raise ValueError("Ordinary-Chat approval checkpoint is missing generation settings.")
        async for event in self._run_nonstream(
            messages,
            tools,
            temperature=float(generation.get("temperature", 0.7)),
            max_tokens=int(generation.get("max_tokens", 4096)),
            top_p=float(generation.get("top_p", 1.0)),
            min_p=float(generation.get("min_p", 0.0)),
            top_k=int(generation.get("top_k", 0)),
            frequency_penalty=float(generation.get("frequency_penalty", 0.0)),
            presence_penalty=float(generation.get("presence_penalty", 0.0)),
            repeat_penalty=float(generation.get("repeat_penalty", 1.0)),
            reasoning_effort=(
                str(generation["reasoning_effort"])
                if isinstance(generation.get("reasoning_effort"), str)
                else None
            ),
            start_iteration=(
                int(continuation["next_iteration"])
                if isinstance(continuation.get("next_iteration"), int)
                and int(continuation["next_iteration"]) > 0
                else 0
            ),
            initial_file_mutation_satisfied=(
                bool(continuation.get("file_mutation_satisfied"))
                or (
                    self._is_file_mutation_tool(tool_name)
                    and tool_result.error is None
                )
            ),
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
        start_iteration: int = 0,
        initial_file_mutation_satisfied: bool | None = None,
    ) -> AsyncIterator[AgentEvent]:
        from mochi.backends.types import GenerationResult

        final_text = ""
        total_input_tokens = 0
        total_output_tokens = 0
        total_generation_time_ms = 0.0
        finish_reason = "stop"
        last_tool_signature: tuple[str, ...] | None = None
        repeated_tool_rounds = 0
        terminal_unavailable_mutation_signatures: set[str] = set()
        empty_final_recovery_attempts = 0
        invalid_tool_turn_recovery_attempts = 0
        truncated_final_recovery_attempts = 0
        final_was_truncated = False
        truncated_final_prefix: str | None = None
        force_plain_answer_without_tools = False
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
            "search_tools": set(),
            "search_queries": set(),
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
        file_artifact_guard_state: dict[str, Any] = {
            "satisfied": (
                initial_file_mutation_satisfied
                if initial_file_mutation_satisfied is not None
                else not self._requires_file_mutation
            ),
            "nudge_count": 0,
            "last_error": None,
            "last_tool_name": None,
            "mutation_paths": [],
        }
        final_file_artifact_blocker_metadata: dict[str, Any] | None = None
        final_plan_blocker_metadata: dict[str, Any] | None = None
        allowed_tool_names = {tool.name for tool in tools}

        try:
            for iteration in range(max(0, start_iteration), self._max_iterations):
                logger.debug(f"ReAct iteration {iteration + 1}/{self._max_iterations}")
                progress_event = self._build_iteration_progress_event(iteration=iteration + 1)
                if progress_event is not None:
                    yield progress_event

                started_at = time.perf_counter()
                iteration_tools = [] if force_plain_answer_without_tools else tools
                try:
                    streamed_generation = False
                    held_stream_text = ""
                    if self._should_use_streaming_generate():
                        streamed_generation = True
                        hold_visible_stream = self._should_hold_visible_stream(
                            evidence_guard_state,
                            iteration_tools,
                        )
                        stream_result = await self._backend.generate(
                            **self._build_generate_kwargs(
                                messages=messages,
                                tools=iteration_tools,
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
                                tools=iteration_tools,
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
                    recovery_mode = self._invalid_tool_turn_recovery_mode(
                        exc=exc,
                        messages=messages,
                        retry_count=invalid_tool_turn_recovery_attempts,
                    )
                    if recovery_mode is not None:
                        rejected_thinking = self._extract_invalid_tool_turn_thinking(exc)
                        if rejected_thinking:
                            yield ThinkingEvent(
                                content=rejected_thinking,
                                metadata={
                                    "source": "model_summary",
                                    "recovery": "invalid_tool_turn",
                                },
                            )
                        invalid_tool_turn_recovery_attempts += 1
                        force_plain_answer_without_tools = (
                            recovery_mode == "plain_answer_without_tools"
                        )
                        recovery_prompt = (
                            self._build_empty_final_response_prompt()
                            if force_plain_answer_without_tools
                            else self._build_invalid_tool_turn_repair_prompt()
                        )
                        messages.append(
                            Message(
                                role="user",
                                content=recovery_prompt,
                            )
                        )
                        continue
                    logger.exception(f"ReAct loop backend error: {exc}")
                    yield ErrorEvent(
                        message=str(exc),
                        metadata=self._with_runtime_error_taxonomy(
                            self._build_backend_error_metadata(exc),
                            error_type="backend_request_error",
                            recoverability="retryable_after_repair",
                        ),
                    )
                    return
                total_generation_time_ms += (time.perf_counter() - started_at) * 1000.0

                if not isinstance(result, GenerationResult):
                    yield ErrorEvent(
                        message="Expected GenerationResult in non-stream mode.",
                        metadata=self._runtime_error_taxonomy(
                            error_type="invalid_backend_result",
                            recoverability="not_retryable",
                        ),
                    )
                    return

                total_input_tokens += result.input_tokens
                total_output_tokens += result.output_tokens
                finish_reason = result.finish_reason or finish_reason
                force_plain_answer_without_tools = False

                if not result.tool_calls and not force_plain_answer_without_tools:
                    rescued_source_content = result.content
                    rescue_reason = "final_text_tool_call_rescue"
                    if truncated_final_prefix:
                        combined_source = f"{truncated_final_prefix}{result.content}"
                        if self._parse_final_text_tool_calls(combined_source):
                            rescued_source_content = combined_source
                            rescue_reason = "truncated_final_text_tool_call_rescue"
                    if not self._parse_final_text_tool_calls(rescued_source_content) and result.thinking:
                        combined_source = "\n".join(
                            part for part in (result.content, result.thinking) if part
                        )
                        if self._parse_final_text_tool_calls(combined_source):
                            rescued_source_content = combined_source
                            rescue_reason = "thinking_tool_call_rescue"
                    rescued_tool_calls = self._parse_final_text_tool_calls(rescued_source_content)
                    if rescued_tool_calls:
                        rescued_visible_content = strip_tool_call_blocks(result.content).strip()
                        rescued_thinking_content = strip_tool_call_blocks(result.thinking).strip()
                        result = GenerationResult(
                            content=rescued_visible_content,
                            thinking=rescued_thinking_content,
                            tool_calls=rescued_tool_calls,
                            model=result.model,
                            input_tokens=result.input_tokens,
                            output_tokens=result.output_tokens,
                            finish_reason="tool_calls",
                            responses_replay=result.responses_replay,
                        )
                        finish_reason = "tool_calls"
                        yield StatusEvent(
                            content="Recovered tool call markup from assistant text.",
                            metadata={
                                **self._runtime_error_taxonomy(
                                    error_type="final_text_tool_call_rescue",
                                    recoverability="recovered",
                                    runtime_category="tool_protocol",
                                ),
                                "reason": rescue_reason,
                                "tool_names": [tool_call.name for tool_call in rescued_tool_calls],
                                "rescued_tool_call_count": len(rescued_tool_calls),
                            },
                        )

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

                    batch_plan_guard_state: dict[str, Any] = {
                        "stale_effect_boundary": False,
                        "successful_update_plan": False,
                    }
                    for tool_call in result.tool_calls:
                        yield ToolCallCreatedEvent(
                            call_id=tool_call.id,
                            tool_name=tool_call.name,
                            arguments=tool_call.arguments,
                            metadata={"compat_event_type": "tool_call_request"},
                        )
                        yield ToolCallRequestEvent(
                            call_id=tool_call.id,
                            tool_name=tool_call.name,
                            arguments=tool_call.arguments,
                        )

                        tool_output: Any = None
                        tool_error: str | None = None
                        tool_metadata: dict[str, Any] = {}
                        tool_result_payload = ToolResult(output=None, error=None)
                        observed_operation_id: str | None = None
                        self._mark_literature_research_mode(tool_call, literature_state)
                        followup_retrieval_attempt = self._is_followup_retrieval_attempt(
                            tool_call=tool_call,
                            evidence_guard_state=evidence_guard_state,
                        )
                        tool_definition = (
                            self._tool_registry.get(tool_call.name)
                            if self._tool_registry is not None
                            else None
                        )
                        plan_guarded_tool_result = self._build_plan_guarded_tool_result(
                            tool_call=tool_call,
                            batch_plan_guard_state=batch_plan_guard_state,
                        )
                        activation_tool_result = self._request_tool_activation_if_needed(
                            tool_call=tool_call,
                            allowed_tool_names=allowed_tool_names,
                            terminal_unavailable_mutation_signatures=(
                                terminal_unavailable_mutation_signatures
                            ),
                        )
                        if (
                            activation_tool_result is not None
                            and activation_tool_result.metadata.get("status") == "tool_activated"
                        ):
                            tools = self._collect_tool_schemas()
                            allowed_tool_names = {tool.name for tool in tools}
                        unavailable_tool_result = (
                            plan_guarded_tool_result
                            or activation_tool_result
                            or self._build_unavailable_tool_result(
                                tool_call=tool_call,
                                allowed_tool_names=allowed_tool_names,
                            )
                        )
                        mutation_tool_signature = (
                            self._build_tool_call_signature([tool_call])[0]
                            if self._is_file_mutation_tool(tool_call.name)
                            else None
                        )
                        repeated_unavailable_mutation = (
                            unavailable_tool_result is not None
                            and mutation_tool_signature is not None
                            and mutation_tool_signature in terminal_unavailable_mutation_signatures
                        )
                        if (
                            unavailable_tool_result is not None
                            and mutation_tool_signature is not None
                            and unavailable_tool_result.retryable is False
                        ):
                            terminal_unavailable_mutation_signatures.add(mutation_tool_signature)
                        if repeated_unavailable_mutation and unavailable_tool_result is not None:
                            repeated_metadata = (
                                dict(unavailable_tool_result.metadata)
                                if isinstance(unavailable_tool_result.metadata, dict)
                                else {}
                            )
                            repeated_metadata.update(
                                self._runtime_error_taxonomy(
                                    error_type="repeated_unavailable_mutation_tool",
                                    recoverability="requires_replanning_or_activation",
                                    runtime_category="tool_activation",
                                )
                            )
                            repeated_metadata.update(
                                {
                                    "retryable": False,
                                    "repeated_tool_call_signature": mutation_tool_signature,
                                }
                            )
                            unavailable_tool_result = ToolResult(
                                output=unavailable_tool_result.output,
                                error=(
                                    (unavailable_tool_result.error or "Mutation tool is not available.")
                                    + " Repeating the same unavailable mutation call is blocked; replan or request activation."
                                ),
                                metadata=repeated_metadata,
                                retryable=False,
                                suggestion=(
                                    "Replan with a different callable path or request activation; "
                                    "do not repeat this exact mutation call."
                                ),
                            )
                            file_artifact_guard_state["nudge_count"] = max(
                                2,
                                int(file_artifact_guard_state.get("nudge_count") or 0),
                            )
                            # A terminal unavailable mutation is a blocker, not a
                            # normal repeated-tool loop; allow the final blocker path.
                            last_tool_signature = None
                            repeated_tool_rounds = 0
                        if unavailable_tool_result is not None:
                            tool_output = unavailable_tool_result.output
                            tool_error = unavailable_tool_result.error
                            tool_metadata = (
                                dict(unavailable_tool_result.metadata)
                                if isinstance(unavailable_tool_result.metadata, dict)
                                else {}
                            )
                            tool_result_payload = unavailable_tool_result
                            formatted_content = self._format_tool_message_content(
                                output=tool_output,
                                error=tool_error,
                            )
                        guarded_tool_result = self._build_guarded_tool_result(
                            tool_call=tool_call,
                            literature_state=literature_state,
                            web_fetch_guard_state=web_fetch_guard_state,
                            web_search_guard_state=web_search_guard_state,
                        )
                        if unavailable_tool_result is None and guarded_tool_result is not None:
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
                            if unavailable_tool_result is None and guarded_tool_result is None:
                                self._record_preplan_read_attempt(tool_call)
                                active_tool_controller = (
                                    self._tool_execution_context.active_tool_controller
                                    if self._tool_execution_context is not None
                                    else None
                                )
                                if active_tool_controller is not None:
                                    await active_tool_controller.activate_tool(
                                        tool_call_id=tool_call.id,
                                        tool_name=tool_call.name,
                                        cancellable=bool(
                                            getattr(tool_definition, "is_cancellable", False)
                                        ),
                                    )
                                self._bind_ordinary_chat_approval_cursor(tool_call)
                                self._capture_ordinary_chat_react_continuation(
                                    messages=messages,
                                    tools=tools,
                                    iteration=iteration,
                                    file_mutation_satisfied=bool(
                                        file_artifact_guard_state.get("satisfied")
                                    ),
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    top_p=top_p,
                                    min_p=min_p,
                                    top_k=top_k,
                                    frequency_penalty=frequency_penalty,
                                    presence_penalty=presence_penalty,
                                    repeat_penalty=repeat_penalty,
                                    reasoning_effort=reasoning_effort,
                                )
                                try:
                                    if self._tool_execution_context is not None:
                                        self._tool_execution_context.state[
                                            "timeline_tool_call_id"
                                        ] = tool_call.id
                                        if not self._tool_execution_context.state.get(
                                            "timeline_tool_lifecycle"
                                        ):
                                            observed_operation_id = (
                                                f"tool-execution-{uuid4().hex}"
                                            )
                                    else:
                                        observed_operation_id = f"tool-execution-{uuid4().hex}"
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
                                    if tool_definition is not None:
                                        formatted_content = tool_definition.format_result_for_model(
                                            tool_result,
                                            max_chars=self._max_tool_message_chars,
                                        )
                                    else:
                                        formatted_content = self._tool_registry.format_result_for_model(
                                            tool_call.name,
                                            tool_result,
                                            max_chars=self._max_tool_message_chars,
                                        )
                                    if (
                                        tool_call.name == "tool_activate"
                                        and tool_result.error is None
                                        and tool_result.metadata.get("status")
                                        in {"tool_activated", "tool_already_callable"}
                                    ):
                                        tools = self._collect_tool_schemas()
                                        allowed_tool_names = {tool.name for tool in tools}
                                except Exception as exc:
                                    tool_error = str(exc)
                                    tool_result_payload = ToolResult(output=tool_output, error=tool_error)
                                    formatted_content = self._format_tool_message_content(
                                        output=tool_output,
                                        error=tool_error,
                                    )
                                finally:
                                    if active_tool_controller is not None:
                                        await active_tool_controller.finish_tool()
                        elif unavailable_tool_result is None and guarded_tool_result is None:
                            tool_error = "No tool registry configured."
                            tool_result_payload = ToolResult(output=tool_output, error=tool_error)
                            formatted_content = self._format_tool_message_content(
                                output=tool_output,
                                error=tool_error,
                            )

                        tool_result_payload = self._postprocess_plan_runtime_tool_result(
                            tool_call=tool_call,
                            tool_result=tool_result_payload,
                            batch_plan_guard_state=batch_plan_guard_state,
                        )
                        tool_output = tool_result_payload.output
                        tool_error = tool_result_payload.error
                        tool_metadata = (
                            dict(tool_result_payload.metadata)
                            if isinstance(tool_result_payload.metadata, dict)
                            else {}
                        )
                        if tool_definition is not None and unavailable_tool_result is None and guarded_tool_result is None:
                            formatted_content = tool_definition.format_result_for_model(
                                tool_result_payload,
                                max_chars=self._max_tool_message_chars,
                            )
                        else:
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
                        self._update_file_artifact_guard_state(
                            tool_call=tool_call,
                            tool_result=tool_result_payload,
                            file_artifact_guard_state=file_artifact_guard_state,
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
                        if observed_operation_id is not None:
                            tool_metadata.setdefault("operation_id", observed_operation_id)
                            tool_metadata["execution_observed"] = True
                        if evidence_diagnostics:
                            tool_metadata["evidence_quality"] = evidence_diagnostics
                            if evidence_diagnostics.get("reason") == "low_information_web_fetch":
                                self._mark_low_information_fetch(
                                    evidence_guard_state=evidence_guard_state,
                                    diagnostics=evidence_diagnostics,
                                )
                        elif followup_retrieval_attempt and not tool_error:
                            self._clear_low_information_fetch_guard(evidence_guard_state)

                        if tool_metadata.get("status") == "runtime_steering":
                            yield StatusEvent(
                                content="Runtime steering: synthesize from collected literature evidence.",
                                metadata={
                                    **self._runtime_error_taxonomy(
                                        error_type="runtime_steering",
                                        recoverability="recovered",
                                        runtime_category="runtime_steering",
                                    ),
                                    "reason": "runtime_steering",
                                    "tool_name": tool_call.name,
                                    **tool_metadata,
                                },
                            )

                        yield ToolCallResultEvent(
                            call_id=tool_call.id,
                            tool_name=tool_call.name,
                            result=tool_output,
                            error=tool_error,
                            metadata=tool_metadata,
                        )
                        yield ToolCallCompletedEvent(
                            call_id=tool_call.id,
                            tool_name=tool_call.name,
                            arguments=tool_call.arguments,
                            result=tool_output,
                            error=tool_error,
                            metadata={**tool_metadata, "compat_event_type": "tool_call_result"},
                        )
                        if tool_metadata.get("timeline_fail_closed") is True:
                            return
                        if self._is_durable_approval_interrupt(tool_metadata):
                            approval_id = str(tool_metadata.get("approval_id") or "").strip()
                            approval_metadata = {
                                "status": "approval_pending",
                                "approval_id": approval_id,
                                "tool_name": tool_call.name,
                                "operation_id": tool_metadata.get("operation_id"),
                                "arguments_digest": tool_metadata.get("arguments_digest"),
                                "resume_cursor": tool_metadata.get("resume_cursor"),
                                "requires_approval": True,
                            }
                            yield StatusEvent(
                                content="Tool execution is paused until the requested approval is resolved.",
                                metadata=approval_metadata,
                            )
                            yield FinalAnswerEvent(
                                content=(
                                    "The requested tool call is waiting for your approval. "
                                    "After it is resolved, Mochi will execute this exact call once."
                                ),
                                finish_reason="approval_required",
                                metadata=approval_metadata,
                            )
                            return

                        tool_message = Message(
                            role="tool",
                            content=tool_content,
                            tool_call_id=tool_call.id,
                            name=tool_call.name,
                        )
                        messages.append(tool_message)
                        self._remember_turn_message(tool_message)
                        injected_guidance = await self._consume_live_subagent_guidance()
                        if injected_guidance is not None:
                            messages.append(injected_guidance)
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
                if streamed_generation and held_stream_text and not final_text.strip():
                    final_text = held_stream_text.strip()
                    held_stream_text = ""
                if self._should_force_followup_retrieval(evidence_guard_state):
                    messages.append(
                        Message(
                            role="user",
                            content=self._build_followup_retrieval_prompt(evidence_guard_state),
                        )
                    )
                    evidence_guard_state["nudge_count"] = int(evidence_guard_state.get("nudge_count") or 0) + 1
                    continue
                if self._should_force_plan_finalization_followup():
                    messages.append(
                        Message(
                            role="user",
                            content=self._build_plan_finalization_followup_prompt(),
                        )
                    )
                    self._increment_plan_runtime_counter("finalization_nudges_used")
                    finalization_nudges_used = self._plan_runtime_counter("finalization_nudges_used")
                    yield StatusEvent(
                        content=(
                            "A durable task plan is required for this turn; requesting a final plan update "
                            "before completing the answer."
                        ),
                        metadata={
                            **self._runtime_error_taxonomy(
                                error_type="plan_finalization_required",
                                recoverability="retrying",
                                runtime_category="task_planning",
                            ),
                            "reason": "plan_finalization_required",
                            "finalization_nudges_used": finalization_nudges_used,
                            "max_finalization_nudges": self._plan_runtime_counter(
                                "max_finalization_nudges"
                            ),
                        },
                    )
                    continue
                if self._should_force_file_artifact_followup(file_artifact_guard_state):
                    messages.append(
                        Message(
                            role="user",
                            content=self._build_file_artifact_followup_prompt(
                                file_artifact_guard_state=file_artifact_guard_state,
                                allowed_tool_names=allowed_tool_names,
                            ),
                        )
                    )
                    file_artifact_guard_state["nudge_count"] = (
                        int(file_artifact_guard_state.get("nudge_count") or 0) + 1
                    )
                    yield StatusEvent(
                        content=(
                            "The user requested a workspace file artifact, but no successful file mutation "
                            "has occurred yet; requesting a file mutation tool call."
                        ),
                        metadata={
                            **self._runtime_error_taxonomy(
                                error_type="file_artifact_missing",
                                recoverability="retrying",
                                runtime_category="deliverable_guard",
                            ),
                            "reason": "file_artifact_missing",
                            "nudge_count": file_artifact_guard_state["nudge_count"],
                            "available_file_mutation_tools": self._available_file_mutation_tools(allowed_tool_names),
                            "last_file_mutation_error": file_artifact_guard_state.get("last_error"),
                            "last_file_mutation_tool": file_artifact_guard_state.get("last_tool_name"),
                        },
                    )
                    continue
                if self._should_block_unsatisfied_plan_final(final_text):
                    final_text = self._build_plan_blocker_final()
                    final_plan_blocker_metadata = self._build_plan_blocker_metadata()
                if self._should_block_unsatisfied_file_artifact_final(file_artifact_guard_state, final_text):
                    final_text = self._build_file_artifact_blocker_final(
                        file_artifact_guard_state=file_artifact_guard_state,
                        allowed_tool_names=allowed_tool_names,
                    )
                    final_file_artifact_blocker_metadata = self._build_file_artifact_blocker_metadata(
                        file_artifact_guard_state=file_artifact_guard_state,
                        allowed_tool_names=allowed_tool_names,
                    )
                if final_thinking:
                    yield ThinkingEvent(
                        content=final_thinking,
                        metadata={"source": "model_summary"},
                    )
                if not final_text.strip():
                    if self._should_retry_empty_final_response(
                        final_thinking=final_thinking,
                        messages=messages,
                        retry_count=empty_final_recovery_attempts,
                    ):
                        empty_final_recovery_attempts += 1
                        force_plain_answer_without_tools = True
                        messages.append(
                            Message(
                                role="user",
                                content=self._build_empty_final_response_prompt(),
                            )
                        )
                        continue
                    backend_info = self._backend.get_model_info()
                    logger.warning("ReAct loop received an empty final response from the backend.")
                    yield ErrorEvent(
                        message="Model returned an empty response.",
                        metadata=self._with_runtime_error_taxonomy(
                            {
                                "backend": {
                                    "backend_type": backend_info.backend_type,
                                    "model": backend_info.name,
                                    "finish_reason": finish_reason,
                                }
                            },
                            error_type="empty_model_response",
                            recoverability="not_retryable",
                        ),
                    )
                    return
                if self._is_length_finish_reason(finish_reason):
                    if truncated_final_recovery_attempts < 1:
                        truncated_final_recovery_attempts += 1
                        final_was_truncated = True
                        truncated_final_prefix = final_text
                        messages.append(
                            Message(
                                role="assistant",
                                content=final_text,
                                thinking=final_thinking,
                                responses_replay=result.responses_replay,
                            )
                        )
                        messages.append(
                            Message(
                                role="user",
                                content=self._build_truncated_final_continuation_prompt(),
                            )
                        )
                        truncation_metadata = {
                            **self._runtime_error_taxonomy(
                                error_type="output_truncated",
                                recoverability="retrying",
                                runtime_category="truncation",
                            ),
                            "reason": "finish_reason_length",
                            "finish_reason": finish_reason,
                            "recovery_attempt": truncated_final_recovery_attempts,
                            "partial_output_chars": len(final_text),
                        }
                        yield AssistantTruncatedEvent(
                            content="Model output hit the response length limit; requesting continuation.",
                            finish_reason=finish_reason or "length",
                            recovery_attempt=truncated_final_recovery_attempts,
                            partial_output_chars=len(final_text),
                            metadata=truncation_metadata,
                        )
                        yield StatusEvent(
                            content="Model output hit the response length limit; requesting continuation.",
                            metadata=truncation_metadata,
                        )
                        continue
                    final_was_truncated = True
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
                    metadata=self._with_runtime_error_taxonomy(
                        self._build_max_iterations_metadata(finish_reason=finish_reason),
                        error_type="max_iterations_reached",
                        recoverability="not_retryable",
                    ),
                )
                return
        except Exception as exc:
            logger.exception(f"ReAct loop error: {exc}")
            yield ErrorEvent(
                message=str(exc),
                metadata=self._with_runtime_error_taxonomy(
                    self._build_backend_error_metadata(exc),
                    error_type="runtime_exception",
                    recoverability="not_retryable",
                ),
            )
            return

        final_metadata: dict[str, Any] = {}
        if final_was_truncated:
            final_metadata.update(
                self._runtime_error_taxonomy(
                    error_type="output_truncated",
                    recoverability="recovered" if not self._is_length_finish_reason(finish_reason) else "partial",
                    runtime_category="truncation",
                )
            )
            final_metadata["truncated"] = True
            final_metadata["recovery_attempts"] = truncated_final_recovery_attempts
        if final_plan_blocker_metadata:
            final_metadata.update(final_plan_blocker_metadata)
        if final_file_artifact_blocker_metadata:
            final_metadata.update(final_file_artifact_blocker_metadata)
        yield FinalAnswerEvent(
            content=final_text,
            input_tokens=total_input_tokens,
            output_tokens=total_output_tokens,
            generation_time_ms=total_generation_time_ms,
            finish_reason=finish_reason,
            metadata=final_metadata,
        )

    @staticmethod
    def _is_length_finish_reason(finish_reason: str | None) -> bool:
        normalized = (finish_reason or "").strip().lower()
        return normalized in {"length", "max_tokens", "token_limit", "context_length"}

    @staticmethod
    def _parse_final_text_tool_calls(content: str) -> list[ToolCall]:
        if "<tool_call" not in content:
            return []
        return parse_tool_calls(content)

    @staticmethod
    def _build_truncated_final_continuation_prompt() -> str:
        return (
            "Your previous answer was cut off because the response length limit was reached. "
            "Continue exactly where it stopped. Do not restart, do not repeat completed text, "
            "and include only the missing continuation."
        )

    def _bind_ordinary_chat_approval_cursor(self, tool_call: ToolCall) -> None:
        """Attach the exact ReAct cursor before a tool can create an approval."""
        context = self._tool_execution_context
        if context is None or not isinstance(context.state, dict):
            return
        approval_context = context.state.get("ordinary_chat_approval_context")
        if not isinstance(approval_context, dict) or approval_context.get("source") != "ordinary_chat":
            return
        cursor = approval_context.get("resume_cursor")
        next_cursor = dict(cursor) if isinstance(cursor, dict) else {}
        next_cursor.update(
            {
                "tool_call_id": tool_call.id,
                "tool_name": tool_call.name,
            }
        )
        approval_context["resume_cursor"] = next_cursor

    def _capture_ordinary_chat_react_continuation(
        self,
        *,
        messages: list[Message],
        tools: list[ToolSchema],
        iteration: int,
        file_mutation_satisfied: bool,
        temperature: float,
        max_tokens: int,
        top_p: float,
        min_p: float,
        top_k: int,
        frequency_penalty: float,
        presence_penalty: float,
        repeat_penalty: float,
        reasoning_effort: str | None,
    ) -> None:
        """Capture the pre-call transcript before a tool can persist an approval."""
        context = self._tool_execution_context
        if context is None or not isinstance(context.state, dict):
            return
        approval_context = context.state.get("ordinary_chat_approval_context")
        if not isinstance(approval_context, dict) or approval_context.get("source") != "ordinary_chat":
            return
        approval_context["react_continuation"] = {
            "schema_version": 1,
            "messages": [self._serialize_ordinary_chat_message(message) for message in messages],
            "callable_tool_names": [tool.name for tool in tools],
            "max_iterations": self._max_iterations,
            "requires_file_mutation": self._requires_file_mutation,
            "next_iteration": iteration + 1,
            "file_mutation_satisfied": file_mutation_satisfied,
            "tool_activation_policy": (
                dict(context.state["tool_activation_policy"])
                if isinstance(context.state.get("tool_activation_policy"), Mapping)
                else None
            ),
            "generation": {
                "temperature": temperature,
                "max_tokens": max_tokens,
                "top_p": top_p,
                "min_p": min_p,
                "top_k": top_k,
                "frequency_penalty": frequency_penalty,
                "presence_penalty": presence_penalty,
                "repeat_penalty": repeat_penalty,
                "reasoning_effort": reasoning_effort,
            },
        }

    @staticmethod
    def _serialize_ordinary_chat_message(message: Message) -> dict[str, Any]:
        return {
            "role": message.role,
            "content": message.content,
            "thinking": message.thinking,
            "tool_calls": [
                {
                    "id": tool_call.id,
                    "name": tool_call.name,
                    "arguments": dict(tool_call.arguments),
                    "index": tool_call.index,
                }
                for tool_call in message.tool_calls
            ],
            "tool_call_id": message.tool_call_id,
            "name": message.name,
            "attachments": [attachment.to_dict() for attachment in message.attachments],
            "responses_replay": (
                message.responses_replay.to_dict()
                if message.responses_replay is not None
                else None
            ),
        }

    @staticmethod
    def _messages_from_ordinary_chat_checkpoint(
        checkpoint: Mapping[str, Any],
    ) -> list[Message]:
        raw_messages = checkpoint.get("messages")
        if not isinstance(raw_messages, list) or not raw_messages:
            raise ValueError("Ordinary-Chat approval continuation transcript is missing.")
        messages: list[Message] = []
        for raw in raw_messages:
            if not isinstance(raw, Mapping):
                raise ValueError("Ordinary-Chat approval continuation transcript is invalid.")
            role = raw.get("role")
            content = raw.get("content")
            if role not in {"system", "user", "assistant", "tool"} or not isinstance(content, str):
                raise ValueError("Ordinary-Chat approval continuation message is invalid.")
            raw_tool_calls = raw.get("tool_calls")
            tool_calls: list[ToolCall] = []
            if isinstance(raw_tool_calls, list):
                for raw_call in raw_tool_calls:
                    if not isinstance(raw_call, Mapping):
                        continue
                    call_id = raw_call.get("id")
                    name = raw_call.get("name")
                    arguments = raw_call.get("arguments")
                    if isinstance(call_id, str) and call_id and isinstance(name, str) and name:
                        tool_calls.append(
                            ToolCall(
                                id=call_id,
                                name=name,
                                arguments=dict(arguments) if isinstance(arguments, Mapping) else {},
                                index=(raw_call.get("index") if isinstance(raw_call.get("index"), int) else None),
                            )
                        )
            from mochi.backends.types import AttachmentRef, ResponsesReplayState

            raw_attachments = raw.get("attachments")
            attachments = (
                [
                    attachment
                    for item in raw_attachments
                    if (attachment := AttachmentRef.from_dict(item)) is not None
                ]
                if isinstance(raw_attachments, list)
                else []
            )
            messages.append(
                Message(
                    role=role,
                    content=content,
                    thinking=raw.get("thinking") if isinstance(raw.get("thinking"), str) else "",
                    tool_calls=tool_calls,
                    tool_call_id=(raw.get("tool_call_id") if isinstance(raw.get("tool_call_id"), str) else None),
                    name=raw.get("name") if isinstance(raw.get("name"), str) else None,
                    attachments=attachments,
                    responses_replay=ResponsesReplayState.from_dict(raw.get("responses_replay")),
                )
            )
        return messages

    @staticmethod
    def _is_durable_approval_interrupt(metadata: dict[str, Any]) -> bool:
        """Only persisted approval requests may suspend an ordinary ReAct turn."""
        return (
            metadata.get("status") == "approval_pending"
            and isinstance(metadata.get("approval_id"), str)
            and bool(str(metadata.get("approval_id")).strip())
        )

    @staticmethod
    def _runtime_control_tool_names() -> frozenset[str]:
        return frozenset({"update_plan", "tool_activate"})

    def _plan_runtime_state(self) -> dict[str, Any] | None:
        context = self._tool_execution_context
        if context is None or not isinstance(context.state, dict):
            return None
        plan_runtime = context.state.get("plan_runtime")
        if not isinstance(plan_runtime, dict):
            return None
        return plan_runtime

    def _plan_ledger_snapshot(self) -> Mapping[str, Any] | None:
        context = self._tool_execution_context
        if context is None or not isinstance(context.state, dict):
            return None
        snapshot = context.state.get("plan_ledger_snapshot")
        if not isinstance(snapshot, Mapping):
            return None
        return snapshot

    def _is_runtime_control_tool(self, tool_name: str) -> bool:
        return tool_name in self._runtime_control_tool_names()

    def _tool_is_read_only(self, tool_name: str) -> bool:
        if self._tool_registry is None:
            return False
        tool = self._tool_registry.get(tool_name)
        return bool(tool is not None and tool.is_read_only)

    def _plan_runtime_counter(self, key: str) -> int:
        plan_runtime = self._plan_runtime_state()
        if plan_runtime is None:
            return 0
        return max(0, int(plan_runtime.get(key) or 0))

    def _increment_plan_runtime_counter(self, key: str) -> int:
        plan_runtime = self._plan_runtime_state()
        if plan_runtime is None:
            return 0
        next_value = max(0, int(plan_runtime.get(key) or 0)) + 1
        plan_runtime[key] = next_value
        return next_value

    def _build_plan_guarded_tool_result(
        self,
        *,
        tool_call: ToolCall,
        batch_plan_guard_state: dict[str, Any],
    ) -> ToolResult | None:
        plan_runtime = self._plan_runtime_state()
        if plan_runtime is None:
            return None
        if not bool(plan_runtime.get("enabled")) or not bool(plan_runtime.get("required")):
            return None
        if plan_runtime.get("state") == "unavailable":
            return None
        if self._is_runtime_control_tool(tool_call.name):
            return None

        if batch_plan_guard_state.get("stale_effect_boundary") and not self._tool_is_read_only(tool_call.name):
            return ToolResult(
                error=(
                    "A previous update_plan call did not commit because its plan revision was stale. "
                    "Refresh the durable plan state before executing effectful tools."
                ),
                metadata={
                    **self._runtime_error_taxonomy(
                        error_type="plan_stale_before_effect",
                        recoverability="retrying",
                        runtime_category="task_planning",
                    ),
                    "guard": "plan_effect_boundary",
                    "reason": "plan_stale_before_effect",
                    "plan_corrections_used": self._plan_runtime_counter("plan_corrections_used"),
                    "max_plan_corrections": self._plan_runtime_counter("max_plan_prompt_corrections"),
                },
                retryable=True,
                suggestion="Call update_plan view or retry update_plan with the latest revision before continuing.",
            )

        if self._tool_is_read_only(tool_call.name):
            if self._plan_requires_creation_before_effect():
                used = self._plan_runtime_counter("preplan_read_calls_used")
                max_reads = self._plan_runtime_counter("max_preplan_read_calls")
                if used >= max_reads:
                    return ToolResult(
                        error=(
                            "The pre-plan read budget is exhausted. Create or repair the durable task plan "
                            "with update_plan before making more inspection calls."
                        ),
                        metadata={
                            **self._runtime_error_taxonomy(
                                error_type="preplan_read_budget_exhausted",
                                recoverability="retrying",
                                runtime_category="task_planning",
                            ),
                            "guard": "preplan_read_budget",
                            "preplan_read_calls_used": used,
                            "max_preplan_read_calls": max_reads,
                        },
                        retryable=True,
                        suggestion="Use update_plan now, then continue with any remaining reads if still needed.",
                    )
            return None

        plan_issue = self._plan_effect_boundary_issue()
        if plan_issue is None:
            return None
        corrections_used = self._plan_runtime_counter("plan_corrections_used")
        max_corrections = self._plan_runtime_counter("max_plan_prompt_corrections")
        retryable = corrections_used < max_corrections
        if retryable:
            corrections_used = self._increment_plan_runtime_counter("plan_corrections_used")
        return ToolResult(
            error=str(plan_issue.get("error") or "A durable task plan is required before effectful tool execution."),
            metadata={
                **self._runtime_error_taxonomy(
                    error_type=str(plan_issue.get("error_type") or "plan_required_before_effect"),
                    recoverability="retrying" if retryable else "requires_replanning_or_activation",
                    runtime_category="task_planning",
                ),
                "guard": "plan_effect_boundary",
                "reason": str(plan_issue.get("reason") or "plan_required_before_effect"),
                "plan_corrections_used": corrections_used,
                "max_plan_corrections": max_corrections,
                **{
                    key: value
                    for key, value in plan_issue.items()
                    if key not in {"error", "error_type", "reason"}
                },
            },
            retryable=retryable,
            suggestion=(
                "Use update_plan to create or correct the durable plan, ensure exactly one in_progress item "
                "with satisfied dependencies, and only then call effectful tools."
            ),
        )

    def _plan_requires_creation_before_effect(self) -> bool:
        plan_runtime = self._plan_runtime_state()
        if plan_runtime is None:
            return False
        if not bool(plan_runtime.get("required")):
            return False
        if plan_runtime.get("state") == "required":
            return True
        return self._plan_ledger_snapshot() is None

    def _plan_effect_boundary_issue(self) -> dict[str, Any] | None:
        plan_runtime = self._plan_runtime_state()
        snapshot = self._plan_ledger_snapshot()
        if plan_runtime is None:
            return None
        if self._plan_requires_creation_before_effect():
            return {
                "error": "A durable task plan is required before effectful tool execution.",
                "error_type": "plan_required_before_effect",
                "reason": "plan_required_before_effect",
            }
        if not isinstance(snapshot, Mapping):
            return {
                "error": "The durable task plan snapshot is unavailable. Refresh the plan with update_plan before proceeding.",
                "error_type": "plan_stale_before_effect",
                "reason": "plan_stale_before_effect",
            }
        items = snapshot.get("items")
        if not isinstance(items, list):
            return {
                "error": "The durable task plan snapshot is malformed. Rebuild it with update_plan before proceeding.",
                "error_type": "plan_stale_before_effect",
                "reason": "plan_snapshot_invalid",
            }
        in_progress_items = [
            item
            for item in items
            if isinstance(item, Mapping) and str(item.get("status") or "") == "in_progress"
        ]
        if len(in_progress_items) != 1:
            return {
                "error": (
                    "Effectful tools require exactly one in_progress plan item. "
                    f"Current in_progress count: {len(in_progress_items)}."
                ),
                "error_type": "plan_stale_before_effect",
                "reason": "plan_in_progress_item_invalid",
                "in_progress_count": len(in_progress_items),
            }
        current_item = in_progress_items[0]
        current_item_id = str(current_item.get("item_id") or "").strip()
        item_by_id = {
            str(item.get("item_id") or ""): item
            for item in items
            if isinstance(item, Mapping) and str(item.get("item_id") or "").strip()
        }
        dependencies = current_item.get("dependencies")
        unmet_dependencies = [
            dependency
            for dependency in dependencies
            if isinstance(dependency, str)
            and str(item_by_id.get(dependency, {}).get("status") or "") != "completed"
        ] if isinstance(dependencies, list) else []
        if unmet_dependencies:
            return {
                "error": (
                    "Effectful tools require the current in_progress item to have all dependencies completed. "
                    f"Unmet dependencies for {current_item_id or 'current item'}: {', '.join(unmet_dependencies)}."
                ),
                "error_type": "plan_stale_before_effect",
                "reason": "plan_dependencies_incomplete",
                "current_item_id": current_item_id or None,
                "unmet_dependencies": unmet_dependencies,
            }
        return None

    def _record_preplan_read_attempt(self, tool_call: ToolCall) -> None:
        if not self._tool_is_read_only(tool_call.name):
            return
        if not self._plan_requires_creation_before_effect():
            return
        self._increment_plan_runtime_counter("preplan_read_calls_used")

    def _postprocess_plan_runtime_tool_result(
        self,
        *,
        tool_call: ToolCall,
        tool_result: ToolResult,
        batch_plan_guard_state: dict[str, Any],
    ) -> ToolResult:
        if tool_call.name == "update_plan":
            return self._postprocess_update_plan_result(
                tool_result=tool_result,
                batch_plan_guard_state=batch_plan_guard_state,
            )
        if tool_result.error is None:
            self._record_successful_plan_evidence_ref(tool_call, tool_result)
        return tool_result

    def _postprocess_update_plan_result(
        self,
        *,
        tool_result: ToolResult,
        batch_plan_guard_state: dict[str, Any],
    ) -> ToolResult:
        metadata = dict(tool_result.metadata) if isinstance(tool_result.metadata, dict) else {}
        output = tool_result.output if isinstance(tool_result.output, Mapping) else None
        if tool_result.error is None:
            save_status = str(metadata.get("save_status") or output.get("status") or "").strip()
            if save_status == "saved":
                batch_plan_guard_state["successful_update_plan"] = True
                batch_plan_guard_state["stale_effect_boundary"] = False
            return tool_result

        error_type = str(metadata.get("error_type") or "").strip()
        if error_type == "stale_plan_revision":
            batch_plan_guard_state["stale_effect_boundary"] = True
            plan_runtime = self._plan_runtime_state()
            if plan_runtime is not None and isinstance(metadata.get("current_revision"), int):
                plan_runtime["current_revision"] = int(metadata["current_revision"])
            return ToolResult(
                output=tool_result.output,
                error=tool_result.error,
                metadata={
                    **metadata,
                    **self._runtime_error_taxonomy(
                        error_type="stale_plan_revision",
                        recoverability="retrying",
                        runtime_category="task_planning",
                    ),
                },
                retryable=True,
                suggestion="Refresh the latest ledger revision with update_plan view, then retry the plan update.",
            )

        if error_type not in {"plan_tool_invalid_request", "plan_transition_invalid", "plan_mutation_invalid"}:
            return tool_result
        used = self._plan_runtime_counter("plan_corrections_used")
        max_corrections = self._plan_runtime_counter("max_plan_prompt_corrections")
        retryable = used < max_corrections
        if retryable:
            used = self._increment_plan_runtime_counter("plan_corrections_used")
            error_message = (
                f"{tool_result.error or 'The plan update was malformed.'} "
                "Fix the update_plan arguments or transition and retry once."
            )
        else:
            error_message = (
                f"{tool_result.error or 'The plan update was malformed.'} "
                "The plan correction budget is exhausted; do not execute effectful tools until the plan is repaired."
            )
        return ToolResult(
            output=tool_result.output,
            error=error_message,
            metadata={
                **metadata,
                **self._runtime_error_taxonomy(
                    error_type="plan_update_malformed",
                    recoverability="retrying" if retryable else "requires_replanning_or_activation",
                    runtime_category="task_planning",
                ),
                "plan_corrections_used": used,
                "max_plan_corrections": max_corrections,
            },
            retryable=retryable,
            suggestion=(
                "Call update_plan again with a valid revision and valid item transition."
                if retryable
                else "Repair the durable plan state before attempting more effectful work."
            ),
        )

    def _record_successful_plan_evidence_ref(self, tool_call: ToolCall, tool_result: ToolResult) -> None:
        context = self._tool_execution_context
        if context is None or not isinstance(context.state, dict):
            return
        raw_refs = context.state.get("recognized_plan_evidence_refs")
        recognized = (
            {str(item) for item in raw_refs if isinstance(item, str)}
            if isinstance(raw_refs, (set, frozenset, list, tuple))
            else set()
        )
        recognized.add(tool_call.id)
        metadata = tool_result.metadata if isinstance(tool_result.metadata, dict) else {}
        explicit_refs = metadata.get("evidence_refs")
        if isinstance(explicit_refs, (list, tuple, set, frozenset)):
            recognized.update(str(item) for item in explicit_refs if isinstance(item, str) and item)
        context.state["recognized_plan_evidence_refs"] = sorted(recognized)

    def _plan_finalization_issue(self) -> dict[str, Any] | None:
        plan_runtime = self._plan_runtime_state()
        if plan_runtime is None:
            return None
        if not bool(plan_runtime.get("enabled")) or not bool(plan_runtime.get("required")):
            return None
        if plan_runtime.get("state") == "unavailable":
            return None
        if self._plan_requires_creation_before_effect():
            return {
                "reason": "plan_missing_at_finalization",
                "error": "A durable task plan is still required before the turn can finalize.",
            }
        ledger_status = str(plan_runtime.get("ledger_status") or "").strip()
        if ledger_status in {"completed", "cancelled"} or plan_runtime.get("state") == "terminal":
            return None
        return {
            "reason": "plan_incomplete_at_finalization",
            "error": "The durable task plan is still active or incomplete and needs a final update before the answer can finalize.",
            "current_item_id": plan_runtime.get("current_item_id"),
        }

    def _should_force_plan_finalization_followup(self) -> bool:
        if self._plan_finalization_issue() is None:
            return False
        return self._plan_runtime_counter("finalization_nudges_used") < self._plan_runtime_counter(
            "max_finalization_nudges"
        )

    def _should_block_unsatisfied_plan_final(self, final_text: str) -> bool:
        if self._plan_finalization_issue() is None:
            return False
        return not self._looks_like_plan_blocker_final(final_text)

    @staticmethod
    def _looks_like_plan_blocker_final(final_text: str) -> bool:
        normalized = final_text.strip().lower()
        if not normalized:
            return False
        return any(
            marker in normalized
            for marker in (
                "blocked",
                "plan",
                "in_progress",
                "could not complete",
                "needs update_plan",
                "task plan",
            )
        )

    def _build_plan_finalization_followup_prompt(self) -> str:
        issue = self._plan_finalization_issue() or {}
        current_item_id = str(issue.get("current_item_id") or "").strip()
        if issue.get("reason") == "plan_missing_at_finalization":
            return (
                "A durable task plan is required for this turn before you can finalize. "
                "Use update_plan to create the plan, set exactly one item to in_progress, and then continue."
            )
        item_hint = f" Current in-progress item: {current_item_id}." if current_item_id else ""
        return (
            "The durable task plan is still incomplete and must be updated before you finalize."
            f"{item_hint} Use update_plan to reflect the latest task state. "
            "Only mark items completed with host-recognized evidence_refs from successful current-turn tool calls."
        )

    def _build_plan_blocker_final(self) -> str:
        issue = self._plan_finalization_issue() or {}
        if issue.get("reason") == "plan_missing_at_finalization":
            return "I could not finalize this turn because a durable task plan was required but was never created."
        current_item_id = str(issue.get("current_item_id") or "").strip()
        if current_item_id:
            return (
                "I could not finalize this turn because the durable task plan is still incomplete: "
                f"item {current_item_id} remains in progress."
            )
        return "I could not finalize this turn because the durable task plan is still incomplete."

    def _build_plan_blocker_metadata(self) -> dict[str, Any]:
        issue = self._plan_finalization_issue() or {}
        return {
            **self._runtime_error_taxonomy(
                error_type="plan_finalization_required",
                recoverability="partial",
                runtime_category="task_planning",
            ),
            "reason": issue.get("reason") or "plan_finalization_required",
            "plan_corrections_used": self._plan_runtime_counter("plan_corrections_used"),
            "max_plan_corrections": self._plan_runtime_counter("max_plan_prompt_corrections"),
            "finalization_nudges_used": self._plan_runtime_counter("finalization_nudges_used"),
            "max_finalization_nudges": self._plan_runtime_counter("max_finalization_nudges"),
            "current_item_id": issue.get("current_item_id"),
        }

    @staticmethod
    def _available_file_mutation_tools(allowed_tool_names: set[str]) -> list[str]:
        preferred_order = ("file_write", "file_edit", "apply_patch")
        return [tool_name for tool_name in preferred_order if tool_name in allowed_tool_names]

    @staticmethod
    def _is_file_mutation_tool(tool_name: str) -> bool:
        return tool_name in {"file_write", "file_edit", "apply_patch"}

    def _update_file_artifact_guard_state(
        self,
        *,
        tool_call: ToolCall,
        tool_result: ToolResult,
        file_artifact_guard_state: dict[str, Any],
    ) -> None:
        if not self._is_file_mutation_tool(tool_call.name):
            return
        file_artifact_guard_state["last_tool_name"] = tool_call.name
        metadata = tool_result.metadata if isinstance(tool_result.metadata, dict) else {}
        file_changes = metadata.get("file_changes")
        has_file_changes = isinstance(file_changes, list) and bool(file_changes)
        if tool_result.error:
            file_artifact_guard_state["last_error"] = tool_result.error
            return
        if has_file_changes and ("bytes_written" in metadata or tool_result.output is not None):
            file_artifact_guard_state["satisfied"] = True
            paths = metadata.get("paths")
            if isinstance(paths, list):
                file_artifact_guard_state["mutation_paths"] = [str(path) for path in paths if path]
            elif len(file_changes) == 1 and isinstance(file_changes[0], dict):
                path = file_changes[0].get("path")
                if path:
                    file_artifact_guard_state["mutation_paths"] = [str(path)]
            file_artifact_guard_state["last_error"] = None

    @staticmethod
    def _should_force_file_artifact_followup(file_artifact_guard_state: dict[str, Any]) -> bool:
        if bool(file_artifact_guard_state.get("satisfied")):
            return False
        return int(file_artifact_guard_state.get("nudge_count") or 0) < 2

    @classmethod
    def _should_block_unsatisfied_file_artifact_final(
        cls,
        file_artifact_guard_state: dict[str, Any],
        final_text: str,
    ) -> bool:
        if bool(file_artifact_guard_state.get("satisfied")):
            return False
        return not cls._looks_like_file_artifact_blocker_final(final_text)

    @staticmethod
    def _looks_like_file_artifact_blocker_final(final_text: str) -> bool:
        normalized = final_text.strip().lower()
        if not normalized:
            return False
        blocker_markers = (
            "blocked",
            "could not save",
            "couldn't save",
            "unable to save",
            "not saved",
            "did not save",
            "failed to save",
            "requires approval",
            "not callable",
            "no successful file mutation",
            "write is blocked",
            "file mutation did not succeed",
        )
        return any(marker in normalized for marker in blocker_markers)

    def _build_file_artifact_followup_prompt(
        self,
        *,
        file_artifact_guard_state: dict[str, Any],
        allowed_tool_names: set[str],
    ) -> str:
        available_tools = self._available_file_mutation_tools(allowed_tool_names)
        tool_hint = ", ".join(available_tools) if available_tools else "no file mutation tools are exposed"
        last_error = str(file_artifact_guard_state.get("last_error") or "").strip()
        if last_error:
            return (
                "The user requested a workspace file artifact, but the previous file mutation attempt failed. "
                f"Last error: {last_error}. "
                f"Available file mutation tools: {tool_hint}. "
                "Retry with a valid file mutation tool call if possible. If the write is blocked by approval, "
                "path policy, missing tool access, or another runtime error, give a final answer that clearly "
                "states the blocker and do not claim the file was saved."
            )
        return (
            "The user requested a workspace file artifact, but no successful file mutation has occurred. "
            f"Available file mutation tools: {tool_hint}. "
            "Use file_write, file_edit, or apply_patch to create or update the requested local file. "
            "If no file mutation tool is available or the write is blocked, give a final answer that clearly "
            "states the blocker and do not claim the file was saved."
        )

    def _build_file_artifact_blocker_final(
        self,
        *,
        file_artifact_guard_state: dict[str, Any],
        allowed_tool_names: set[str],
    ) -> str:
        available_tools = self._available_file_mutation_tools(allowed_tool_names)
        last_error = str(file_artifact_guard_state.get("last_error") or "").strip()
        last_tool_name = str(file_artifact_guard_state.get("last_tool_name") or "").strip()
        hidden_mutation_tool = (
            bool(last_tool_name)
            and self._is_file_mutation_tool(last_tool_name)
            and last_tool_name not in allowed_tool_names
        )
        if hidden_mutation_tool or not available_tools:
            return "I could not save the file because the required write tool was not callable in this turn."
        if last_error:
            return f"I could not save the file because the file mutation did not succeed: {last_error}"
        return "I could not save the file because no successful file mutation occurred in this turn."

    def _build_file_artifact_blocker_metadata(
        self,
        *,
        file_artifact_guard_state: dict[str, Any],
        allowed_tool_names: set[str],
    ) -> dict[str, Any]:
        last_tool_name = file_artifact_guard_state.get("last_tool_name")
        last_error = file_artifact_guard_state.get("last_error")
        return {
            **self._runtime_error_taxonomy(
                error_type="file_artifact_not_mutated",
                recoverability="requires_replanning_or_activation",
                runtime_category="deliverable_guard",
            ),
            "reason": "file_artifact_not_mutated",
            "available_file_mutation_tools": self._available_file_mutation_tools(allowed_tool_names),
            "last_file_mutation_error": last_error,
            "last_file_mutation_tool": last_tool_name,
            "mutation_paths": list(file_artifact_guard_state.get("mutation_paths") or []),
        }

    async def _consume_live_subagent_guidance(self) -> Message | None:
        if self._tool_execution_context is None:
            return None
        controller = self._tool_execution_context.active_tool_controller
        if controller is None:
            return None
        guidance_messages = await controller.consume_post_tool_messages()
        contents = [
            str(item.get("content") or "").strip()
            for item in guidance_messages
            if str(item.get("content") or "").strip()
        ]
        if not contents:
            return None
        lines = ["Live subagent guidance:"]
        lines.extend(f"- {content}" for content in contents)
        return Message(role="user", content="\n".join(lines))

    def _split_thinking_blocks(self, content: str, thinking: str = "") -> tuple[str, str]:
        closing_only_match = _find_closing_reasoning_tag(content)
        if _find_opening_reasoning_tag(content) is None and closing_only_match is not None:
            visible = content[closing_only_match[0] + len(closing_only_match[2]) :]
            combined_thinking = "".join(
                part
                for part in (thinking, content[: closing_only_match[0]])
                if part
            ).strip()
            return (
                self._sanitize_reasoning_artifacts(visible),
                self._sanitize_reasoning_artifacts(combined_thinking),
            )
        state: dict[str, Any] = {"in_think": False, "buffer": ""}
        visible, extracted_thinking = self._split_stream_thinking_delta(content, state)
        visible_tail, thinking_tail = self._finalize_stream_thinking_delta(state)
        combined_thinking = "".join(
            part
            for part in (thinking, extracted_thinking, thinking_tail)
            if part
        ).strip()
        sanitized_visible = self._sanitize_reasoning_artifacts((visible + visible_tail).strip())
        sanitized_thinking = self._sanitize_reasoning_artifacts(combined_thinking)
        return sanitized_visible, sanitized_thinking

    @staticmethod
    def _split_stream_thinking_delta(
        delta: str,
        state: dict[str, Any],
    ) -> tuple[str, str]:
        visible_parts: list[str] = []
        thinking_parts: list[str] = []
        buffer = str(state.get("buffer") or "")
        current_tag = str(state.get("current_tag") or "").lower() or None
        in_think = bool(state.get("in_think") or current_tag)

        for char in delta:
            if buffer:
                buffer += char
                target = f"</{current_tag}>" if current_tag else None
                lowered_buffer = buffer.lower()
                if target is not None and target.startswith(lowered_buffer):
                    if lowered_buffer == target:
                        buffer = ""
                        current_tag = None
                        in_think = False
                    continue
                if target is None and any(
                    candidate.startswith(lowered_buffer) for candidate in _REASONING_OPEN_TAGS
                ):
                    matched_open_tag = next(
                        (candidate for candidate in _REASONING_OPEN_TAGS if candidate == lowered_buffer),
                        None,
                    )
                    if matched_open_tag is not None:
                        buffer = ""
                        current_tag = matched_open_tag[1:-1]
                        in_think = True
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
        state["current_tag"] = current_tag or ""
        return "".join(visible_parts), "".join(thinking_parts)

    @staticmethod
    def _finalize_stream_thinking_delta(state: dict[str, Any]) -> tuple[str, str]:
        buffer = str(state.get("buffer") or "")
        current_tag = str(state.get("current_tag") or "").lower() or None
        in_think = bool(state.get("in_think") or current_tag)
        state["buffer"] = ""
        state["current_tag"] = ""
        if not buffer:
            return "", ""
        if in_think:
            return "", buffer
        return buffer, ""

    @staticmethod
    def _sanitize_reasoning_artifacts(value: str) -> str:
        if not value:
            return ""
        normalized = value.replace("\r\n", "\n")
        stripped = _CHANNEL_REASONING_PREFIX_RE.sub("", normalized, count=1)
        stripped = _ROLE_PREFIX_RE.sub("", stripped, count=1)
        stripped = _ROLE_SENTINEL_PREFIX_RE.sub("", stripped, count=1)
        stripped = _CHANNEL_MARKER_RE.sub("", stripped)
        stripped = _HEADER_MARKER_RE.sub("", stripped)
        stripped = _ROLE_SENTINEL_RE.sub("", stripped)
        stripped = re.sub(r"\n{3,}", "\n\n", stripped)
        return stripped.strip()

    def _remember_turn_message(self, message: Message) -> None:
        self._turn_messages.append(Message(**message.__dict__))

    def _request_tool_activation_if_needed(
        self,
        *,
        tool_call: ToolCall,
        allowed_tool_names: set[str],
        terminal_unavailable_mutation_signatures: set[str],
    ) -> ToolResult | None:
        if (
            not self._is_file_mutation_tool(tool_call.name)
            or tool_call.name in allowed_tool_names
            or self._tool_registry is None
            or self._tool_execution_context is None
        ):
            return None
        mutation_signature = self._build_tool_call_signature([tool_call])[0]
        if mutation_signature in terminal_unavailable_mutation_signatures:
            return ToolResult(
                error=(
                    f"Tool '{tool_call.name}' was already denied as unavailable; "
                    "repeating the exact mutation request is blocked."
                ),
                metadata={
                    "guard": "tool_not_exposed",
                    "requested_tool": tool_call.name,
                    "callable_this_turn": False,
                    "activation_required": True,
                    "activation_reason": "The exact unavailable mutation request was already denied.",
                    **self._runtime_error_taxonomy(
                        error_type="repeated_unavailable_mutation_tool",
                        recoverability="requires_replanning_or_activation",
                        runtime_category="tool_activation",
                    ),
                    "repeated_tool_call_signature": mutation_signature,
                },
                retryable=False,
                suggestion="Replan or obtain activation; do not replay the exact denied request.",
            )
        policy = self._tool_execution_context.state.get("tool_activation_policy")
        if not isinstance(policy, dict):
            return None
        return self._tool_registry.request_tool_activation(
            tool_call.name,
            context=self._tool_execution_context,
        )

    def _build_unavailable_tool_result(
        self,
        *,
        tool_call: ToolCall,
        allowed_tool_names: set[str],
    ) -> ToolResult | None:
        if tool_call.name and tool_call.name in allowed_tool_names and self._tool_registry is not None:
            if self._tool_registry.get(tool_call.name) is not None:
                return None

        available_tools = sorted(name for name in allowed_tool_names if name)
        available_preview = ", ".join(available_tools[:12])
        if len(available_tools) > 12:
            available_preview = f"{available_preview}, +{len(available_tools) - 12} more"
        available_hint = available_preview or "(none)"
        requested_tool = tool_call.name or "(empty tool name)"

        metadata: dict[str, Any] = {
            "guard": "tool_not_exposed",
            "requested_tool": requested_tool,
            "available_tools": available_tools,
        }
        if self._requires_file_mutation and self._is_file_mutation_tool(requested_tool):
            metadata.update(
                self._runtime_error_taxonomy(
                    error_type="mutation_tool_not_callable",
                    recoverability="requires_replanning_or_activation",
                    runtime_category="tool_activation",
                )
            )
            metadata.update(
                {
                    "callable_this_turn": False,
                    "activation_required": True,
                    "activation_reason": (
                        "Mutation tool is discoverable or requested but not exposed as callable in this turn."
                    ),
                }
            )

        return ToolResult(
            error=(
                f"Tool '{requested_tool}' is not available in this turn. "
                f"Use only the exposed tools: {available_hint}."
            ),
            metadata=metadata,
            retryable=False,
            suggestion=(
                "Choose one of the exposed tools for this turn, "
                "or produce the final answer without calling unavailable tools."
            ),
        )

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

    @staticmethod
    def _should_retry_empty_final_response(
        *,
        final_thinking: str,
        messages: list[Message],
        retry_count: int,
    ) -> bool:
        if retry_count >= 1:
            return False
        if final_thinking.strip():
            return True
        return any(message.role == "tool" for message in messages)

    @staticmethod
    def _build_empty_final_response_prompt() -> str:
        return (
            "Your last response contained no user-visible final answer. "
            "Return the final answer now in plain assistant text for the user, "
            "using the user's language. Do not repeat hidden reasoning. "
            "Do not call tools again unless another tool is strictly necessary."
        )

    @staticmethod
    def _build_invalid_tool_turn_repair_prompt() -> str:
        return (
            "Your last response was invalid for a tool-capable turn. "
            "Reply correctly on the next turn. If you need external information, "
            "call one of the available tools now. If no tool is needed, answer "
            "directly in plain assistant text for the user, using the user's "
            "language. If you cannot call a tool correctly, say that plainly "
            "instead of guessing. Do not output hidden reasoning only. Do not "
            "output placeholder text such as 'I'll check' or 'searching'."
        )

    @staticmethod
    def _invalid_tool_turn_recovery_mode(
        *,
        exc: Exception,
        messages: list[Message],
        retry_count: int,
    ) -> str | None:
        if retry_count >= 1 or not isinstance(exc, BackendRequestError):
            return None
        tool_turn_reason = str(exc.metadata.get("tool_turn_reason") or "").strip().lower()
        if tool_turn_reason not in {"thinking_only", "empty"}:
            return None
        if any(message.role == "tool" for message in messages):
            return "plain_answer_without_tools"
        backend_name = str(exc.metadata.get("backend_name") or "").strip().lower()
        if backend_name == "ollama":
            return "repair_tool_turn"
        return None

    @staticmethod
    def _extract_invalid_tool_turn_thinking(exc: Exception) -> str:
        if not isinstance(exc, BackendRequestError):
            return ""
        rejected_thinking = exc.metadata.get("rejected_thinking")
        return rejected_thinking if isinstance(rejected_thinking, str) else ""

    @staticmethod
    def _runtime_error_taxonomy(
        *,
        error_type: str,
        recoverability: str,
        runtime_category: str = "runtime_error",
    ) -> dict[str, Any]:
        return {
            "runtime_category": runtime_category,
            "error_type": error_type,
            "recoverability": recoverability,
        }

    @classmethod
    def _with_runtime_error_taxonomy(
        cls,
        metadata: dict[str, Any],
        *,
        error_type: str,
        recoverability: str,
        runtime_category: str = "runtime_error",
    ) -> dict[str, Any]:
        merged = dict(metadata)
        merged.update(
            cls._runtime_error_taxonomy(
                error_type=error_type,
                recoverability=recoverability,
                runtime_category=runtime_category,
            )
        )
        return merged

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
                output=(
                    "Sufficient literature evidence is already collected. "
                    "Do not call more search or fetch tools; synthesize the answer now."
                ),
                metadata={
                    "status": "runtime_steering",
                    "steering_reason": "evidence_sufficient",
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
            if output:
                search_tools = literature_state.setdefault("search_tools", set())
                if isinstance(search_tools, set):
                    search_tools.add(tool_call.name)
                query = tool_call.arguments.get("query")
                if isinstance(query, str) and query.strip():
                    search_queries = literature_state.setdefault("search_queries", set())
                    if isinstance(search_queries, set):
                        search_queries.add(self._normalize_text(query))
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
        search_tools = literature_state.get("search_tools")
        search_queries = literature_state.get("search_queries")
        distinct_search_tools = len(search_tools) if isinstance(search_tools, set) else 0
        distinct_search_queries = len(search_queries) if isinstance(search_queries, set) else 0
        has_corroborated_search = distinct_search_tools >= 2 or distinct_search_queries >= 2
        return (
            (has_corroborated_search and paper_hits >= 3 and abstract_hits >= 2)
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
