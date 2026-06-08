# Inspired by openclaw/src/agents/pi-embedded-runner design pattern
"""Async ReAct loop for tool-capable backends."""

from __future__ import annotations

import json
import re
import time
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any
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
    ThinkingEvent,
    ToolCallRequestEvent,
    ToolCallResultEvent,
)
from mochi.backends.base import BackendRequestError
from mochi.backends.types import Message, ToolCall
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
        literature_state: dict[str, Any] = {
            "research_mode": False,
            "paper_hits": 0,
            "abstract_hits": 0,
            "fetched_docs": 0,
            "fetched_chars": 0,
            "summary_ready": False,
            "prompt_injected": False,
        }

        try:
            for iteration in range(self._max_iterations):
                logger.debug(f"ReAct iteration {iteration + 1}/{self._max_iterations}")

                started_at = time.perf_counter()
                try:
                    generate_kwargs: dict[str, Any] = {
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
                        "stream": False,
                    }
                    if reasoning_effort is not None:
                        generate_kwargs["reasoning_effort"] = reasoning_effort
                    result = await self._backend.generate(**generate_kwargs)
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
                        yield ErrorEvent(
                            message="Model repeated the same tool-call pattern without making progress.",
                            code="REPEATED_TOOL_LOOP",
                            metadata=self._build_repeated_tool_loop_metadata(
                                finish_reason=finish_reason,
                                repeated_tool_rounds=repeated_tool_rounds,
                                tool_calls=result.tool_calls,
                            ),
                        )
                        return

                    thought_content = result.thinking or result.content
                    if thought_content:
                        yield ThinkingEvent(content=thought_content)

                    messages.append(
                        Message(
                            role="assistant",
                            content=result.content,
                            thinking=result.thinking,
                            tool_calls=result.tool_calls,
                        )
                    )

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
                        guarded_tool_result = self._build_guarded_tool_result(
                            tool_call=tool_call,
                            literature_state=literature_state,
                            web_fetch_guard_state=web_fetch_guard_state,
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
                        self._last_transport_diagnostics = transport_diagnostics
                        tool_metadata = {
                            **tool_metadata,
                            "transport": transport_diagnostics,
                        }

                        yield ToolCallResultEvent(
                            call_id=tool_call.id,
                            tool_name=tool_call.name,
                            result=tool_output,
                            error=tool_error,
                            metadata=tool_metadata,
                        )

                        messages.append(
                            Message(
                                role="tool",
                                content=tool_content,
                                tool_call_id=tool_call.id,
                                name=tool_call.name,
                            )
                        )
                    if literature_state["summary_ready"] and not literature_state["prompt_injected"]:
                        messages.append(
                            Message(
                                role="user",
                                content=self._build_literature_summary_prompt(literature_state),
                            )
                        )
                        literature_state["prompt_injected"] = True
                    continue

                final_text = result.content
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
                last_tool_signature = None
                repeated_tool_rounds = 0
                break
            else:
                logger.warning(f"ReAct loop reached max iterations ({self._max_iterations})")
                yield ErrorEvent(
                    message="Model reached the maximum tool-call iterations without producing a final answer.",
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

    def _build_max_iterations_metadata(self, *, finish_reason: str) -> dict[str, Any]:
        backend_info = self._backend.get_model_info()
        backend_metadata: dict[str, Any] = {
            "backend_type": backend_info.backend_type,
            "model": backend_info.name,
            "finish_reason": finish_reason,
        }
        if isinstance(backend_info.metadata, dict):
            api_mode = backend_info.metadata.get("api_mode")
            if isinstance(api_mode, str):
                backend_metadata["api_mode"] = api_mode

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
        if isinstance(backend_info.metadata, dict):
            api_mode = backend_info.metadata.get("api_mode")
            if isinstance(api_mode, str):
                backend_metadata["api_mode"] = api_mode

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
