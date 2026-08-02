"""Ollama backend implementation."""

from __future__ import annotations

import json
import re
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
from mochi.backends.simulated_tool_protocol import SimulatedToolProtocol
from mochi.backends.tool_call_contract import (
    build_invalid_tool_turn_metadata,
    validate_tool_turn_result,
)
from mochi.backends.tool_call_simulator import ToolCallSimulator
from mochi.backends.tool_call_state import ToolCallingState
from mochi.backends.types import (
    GenerationResult,
    Message,
    ModelInfo,
    StreamChunk,
    ToolCall,
    ToolSchema,
)
from mochi.diagnostics.fallbacks import append_fallback_diagnostic

_IN_PROGRESS_MARKERS = (
    "searching",
    "looking up",
    "checking",
    "querying",
    "fetching",
    "please wait",
    "one moment",
    "\u67e5\u8a62\u4e2d",
    "\u67e5\u8be2\u4e2d",
    "\u641c\u5c0b\u4e2d",
    "\u641c\u7d22\u4e2d",
    "\u6aa2\u7d22\u4e2d",
    "\u68c0\u7d22\u4e2d",
    "\u6b63\u5728\u67e5\u8a62",
    "\u6b63\u5728\u67e5\u8be2",
    "\u6b63\u5728\u641c\u5c0b",
    "\u6b63\u5728\u641c\u7d22",
    "\u7a0d\u7b49",
    "\u8acb\u7a0d\u5019",
    "\u8bf7\u7a0d\u5019",
)
_TOOL_USAGE_PHRASES = (
    "let me check",
    "i'll check",
    "i will check",
    "i need to use",
    "i should use",
    "call the tool",
    "using the tool",
    "\u6211\u4f86\u5e6b\u60a8\u67e5\u8a62",
    "\u6211\u6765\u5e2e\u60a8\u67e5\u8be2",
    "\u6211\u4f86\u67e5\u8a62",
    "\u6211\u6765\u67e5\u8be2",
    "\u6211\u6703\u4f7f\u7528",
    "\u6211\u4f1a\u4f7f\u7528",
    "\u6211\u9700\u8981\u4f7f\u7528",
)
_PLACEHOLDER_SUFFIXES = (":", "\uff1a", "...", "\u2026")


class OllamaBackend(BaseLLMBackend):
    """Backend for Ollama's `/api/chat` endpoint."""

    _CONTEXT_LENGTH_FALLBACK = 4096

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: float = 120.0,
        num_ctx: int | None = None,
        auto_num_ctx: bool = True,
        auto_num_ctx_cap: int = 32768,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._configured_num_ctx = self._normalize_context_length(num_ctx)
        self._auto_num_ctx = bool(auto_num_ctx)
        self._auto_num_ctx_cap = self._normalize_context_length(auto_num_ctx_cap) or 32768
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=timeout)
        self._tool_prompt_profile = self._resolve_tool_prompt_profile(self.model)
        self._tool_call_simulator = ToolCallSimulator(tool_prompt_profile=self._tool_prompt_profile)
        self._simulated_tool_protocol = SimulatedToolProtocol(
            self._tool_call_simulator,
            tool_prompt_profile=self._tool_prompt_profile,
        )
        self._tool_state = ToolCallingState()
        self._tool_state.recover_native("native_default")
        self._native_tool_probe: dict[str, Any] | None = None
        self._fallback_diagnostics: list[dict[str, Any]] = []
        self._context_length: int | None = None
        self._context_length_source = "unknown"
        self._runtime_context_length: int | None = None
        self._runtime_context_length_source = "unknown"
        self._model_max_context_length: int | None = None
        self._model_max_context_length_source = "unknown"
        self._apply_configured_num_ctx()

    def supports_tool_calling(self) -> bool:
        return self._tool_state.supports_tool_calling()

    def _tool_call_mode(self) -> str:
        return self._tool_state.active_mode

    def _tool_calling_protocol(self) -> str:
        if self._tool_state.active_mode == "native":
            return "native"
        if self._tool_state.active_mode == "unavailable":
            return "unavailable"
        return "prompt_guided"

    @staticmethod
    def _resolve_tool_prompt_profile(model: str) -> str:
        normalized = str(model or "").strip().lower()
        if "qwen" in normalized:
            return "qwen_xml_tool_call"
        return "json_tool_call"

    def get_model_info(self) -> ModelInfo:
        supports_reasoning_effort = self._supports_reasoning_effort_model(self.model)
        probe = self._native_tool_probe if isinstance(self._native_tool_probe, dict) else {}
        context_length = self._context_length
        request_num_ctx = self._request_num_ctx()
        if request_num_ctx is not None:
            effective_context_length = request_num_ctx
            if self._configured_num_ctx is not None:
                effective_context_length_source = "config.num_ctx"
            elif self._runtime_context_length is not None:
                effective_context_length_source = self._runtime_context_length_source
            elif self._model_max_context_length is not None:
                effective_context_length_source = "auto_num_ctx.model_max_cap"
            else:
                effective_context_length_source = "auto_num_ctx.fallback_default"
        else:
            effective_context_length = context_length or self._CONTEXT_LENGTH_FALLBACK
            effective_context_length_source = (
                self._context_length_source if context_length is not None else "fallback_default"
            )
        return ModelInfo(
            name=self.model,
            backend_type="ollama",
            provider="ollama",
            context_length=context_length,
            supports_tool_calling=self.supports_tool_calling(),
            metadata={
                "supports_reasoning_effort": supports_reasoning_effort,
                "reasoning_effort_param": "think" if supports_reasoning_effort else None,
                "context_length_source": self._context_length_source,
                "context_length_fallback": (
                    None if context_length is not None else self._CONTEXT_LENGTH_FALLBACK
                ),
                "effective_context_length": effective_context_length,
                "effective_context_length_source": effective_context_length_source,
                "configured_num_ctx": self._configured_num_ctx,
                "auto_num_ctx": self._auto_num_ctx,
                "auto_num_ctx_cap": self._auto_num_ctx_cap,
                "auto_num_ctx_value": self._auto_num_ctx_value(),
                "runtime_context_length": self._runtime_context_length,
                "runtime_context_length_source": self._runtime_context_length_source,
                "model_max_context_length": self._model_max_context_length,
                "model_max_context_length_source": self._model_max_context_length_source,
                "tool_call_mode": self._tool_call_mode(),
                "tool_calling_protocol": self._tool_calling_protocol(),
                "tool_calling_style": self._tool_calling_protocol(),
                "tool_prompt_profile": self._tool_prompt_profile,
                "native_tool_calling_status": probe.get("status", self._tool_state.native_status),
                "native_tool_calling_message": probe.get("message"),
                "native_tool_calling_checked_at": probe.get("checked_at"),
                "fallback_validation_status": self._tool_state.fallback_validation_status,
                "fallback_diagnostics": list(self._fallback_diagnostics),
            },
        )

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get("/api/tags", timeout=5.0)
            return resp.status_code == 200
        except Exception as exc:
            logger.debug(f"Ollama health check failed: {exc}")
            return False

    async def prime_model_info(self) -> None:
        try:
            resp = await self._client.post("/api/show", json={"model": self.model})
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            logger.debug("Ollama model-info prime failed for '{}': {}", self.model, exc)
            return
        except Exception as exc:
            logger.debug("Ollama model-info prime failed for '{}': {}", self.model, exc)
            return

        data = resp.json()
        if not isinstance(data, dict):
            return

        metadata = self._extract_context_metadata_from_show_payload(data)
        self._runtime_context_length = metadata["runtime_context_length"]
        self._runtime_context_length_source = metadata["runtime_context_length_source"]
        self._model_max_context_length = metadata["model_max_context_length"]
        self._model_max_context_length_source = metadata["model_max_context_length_source"]
        self._context_length = metadata["context_length"]
        self._context_length_source = metadata["context_length_source"]
        self._apply_configured_num_ctx()

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
        options: dict[str, Any] = {
            "temperature": temperature,
            "num_predict": max_tokens,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
        }
        request_num_ctx = self._request_num_ctx()
        if request_num_ctx is not None:
            options["num_ctx"] = request_num_ctx
        think_value = self._reasoning_effort_to_think_value(reasoning_effort)
        tools_requested = bool(tools)
        post_tool_continuation = self._is_post_tool_continuation(messages)
        use_native_tools = self._tool_state.active_mode == "native"
        prepared_messages = self._prepare_messages(
            messages,
            tools,
            use_native_tools=use_native_tools,
        )
        payload = self._build_request_payload(
            messages=prepared_messages,
            tools=tools if use_native_tools else None,
            options=options,
            stream=stream,
            think_value=think_value,
        )

        if stream:
            return self._stream_generate(payload)

        try:
            result = await self._blocking_generate(
                payload,
                tools=tools,
                use_native_tools=use_native_tools,
            )
        except BackendRequestError:
            if tools_requested and self._tool_state.active_mode == "simulated_fallback" and not use_native_tools:
                self._record_simulated_protocol_rejection(
                    status="simulated_protocol_error",
                    reason="Prompt-simulated tool retry failed before returning a valid turn.",
                )
            raise

        if not tools_requested:
            return result

        if use_native_tools:
            retry_trigger = self._native_tool_retry_trigger(
                result=result,
                tools=tools,
            )
            if retry_trigger is None:
                return result
            if post_tool_continuation:
                return result

            self._record_native_tool_turn_rejection(
                status=str(retry_trigger["metadata"].get("trigger") or "native_tool_turn_invalid"),
                reason=str(retry_trigger["reason"]),
                result=result,
                metadata=retry_trigger["metadata"],
            )

        if not use_native_tools:
            return self._finalize_simulated_tool_result(
                result,
                allow_incomplete_final=post_tool_continuation,
            )

        return result

    def _ensure_prompt_guided_tool_mode(self) -> None:
        if self._tool_state.active_mode == "unavailable":
            return
        self._tool_state.enter_simulated("prompt_guided_default")

    async def _blocking_generate(
        self,
        payload: dict[str, Any],
        *,
        tools: list[ToolSchema] | None = None,
        use_native_tools: bool = True,
    ) -> GenerationResult:
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
        result = self._parse_generation_result(
            data,
            tools=tools,
            use_native_tools=use_native_tools,
        )
        if result.tool_calls and use_native_tools:
            self._record_native_tool_probe(
                status="supported",
                message="Ollama returned structured native tool calls.",
                activate_native=None,
            )
        return result

    def _parse_generation_result(
        self,
        data: dict[str, Any],
        *,
        tools: list[ToolSchema] | None,
        use_native_tools: bool,
    ) -> GenerationResult:
        msg = data.get("message", {})
        content = msg.get("content", "")
        if not isinstance(content, str):
            content = ""
        raw_thinking = msg.get("thinking", "")
        thinking = raw_thinking if isinstance(raw_thinking, str) else ""

        tool_calls: list[ToolCall] = []
        for tc in msg.get("tool_calls", []):
            if not isinstance(tc, dict):
                continue
            fn = tc.get("function", {})
            if not isinstance(fn, dict):
                fn = {}
            raw_args = fn.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except json.JSONDecodeError:
                    raw_args = {}
            elif not isinstance(raw_args, dict):
                raw_args = {}
            tool_calls.append(
                ToolCall(
                    id=tc.get("id", str(uuid.uuid4())),
                    name=fn.get("name", ""),
                    arguments=raw_args,
                    index=fn.get("index") if isinstance(fn.get("index"), int) else None,
                )
            )

        if (
            not tool_calls
            and tools
            and content
            and (not use_native_tools or "<tool_call" in content.lower())
        ):
            content, tool_calls = self._simulated_tool_protocol.parse_assistant_content(content)

        if not tool_calls and not content.strip() and not thinking.strip() and not tools:
            logger.warning("Ollama returned an empty non-tool response.")
            raise BackendRequestError(
                "Ollama returned an empty response with no content or tool calls.",
                metadata={
                    "backend_name": "ollama",
                    "request_url": f"{self.base_url}/api/chat",
                    "stage": "generate",
                    "model": self.model,
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

    def _prepare_messages(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        *,
        use_native_tools: bool,
    ) -> list[Message]:
        if use_native_tools or not tools:
            return [Message(**message.__dict__) for message in messages]
        return self._simulated_tool_protocol.prepare_messages(messages=messages, tools=tools)

    def _build_request_payload(
        self,
        *,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        options: dict[str, Any],
        stream: bool,
        think_value: str | bool | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._serialize_messages(messages),
            "stream": stream,
            "options": options,
        }
        if think_value is not None:
            payload["think"] = think_value
        if tools:
            payload["tools"] = self._serialize_tools(tools)
        return payload

    def _finalize_simulated_tool_result(
        self,
        result: GenerationResult,
        *,
        allow_incomplete_final: bool = False,
    ) -> GenerationResult:
        verdict = validate_tool_turn_result(result=result, tools_requested=True)
        if verdict.is_valid:
            self._tool_state.validate_simulated()
            return result
        if allow_incomplete_final and verdict.reason == "thinking_only":
            return result

        error_metadata = build_invalid_tool_turn_metadata(result=result, reason=verdict.reason)
        self._record_simulated_protocol_rejection(
            status="simulated_protocol_rejected",
            reason="Ollama returned an invalid tool-eligible turn from prompt-simulated tool calling.",
            metadata=error_metadata,
        )
        raise BackendRequestError(
            "Ollama returned an invalid tool-eligible turn.",
            metadata={
                "backend_name": "ollama",
                "model": self.model,
                **error_metadata,
            },
        )

    @staticmethod
    def _is_post_tool_continuation(messages: list[Message]) -> bool:
        seen_tool = False
        for message in reversed(messages):
            if message.role == "tool":
                seen_tool = True
                continue
            if not seen_tool:
                if message.role == "assistant" and not message.tool_calls:
                    return False
                continue
            if message.role == "assistant":
                return bool(message.tool_calls)
            if message.role == "user":
                return True
        return seen_tool

    def _native_tool_retry_trigger(
        self,
        *,
        result: GenerationResult,
        tools: list[ToolSchema] | None,
    ) -> dict[str, Any] | None:
        verdict = validate_tool_turn_result(result=result, tools_requested=True)
        if verdict.reason in {"thinking_only", "empty"}:
            trigger = (
                "empty_native_tool_turn"
                if verdict.reason == "empty"
                else "thinking_without_native_tool_calls"
            )
            return {
                "reason": (
                    "Ollama returned no structured tool calls; retrying with "
                    "prompt-simulated tool mode."
                ),
                "metadata": {
                    "trigger": trigger,
                    "tool_count": len(tools or []),
                },
            }
        if verdict.reason == "content" and self._looks_like_incomplete_tool_intent(
            content=result.content,
            tools=tools,
        ):
            return {
                "reason": (
                    "Ollama returned placeholder tool-intent text without structured tool "
                    "calls; retrying with prompt-simulated tool mode."
                ),
                "metadata": {
                    "trigger": "contentful_tool_intent_without_native_tool_calls",
                    "tool_count": len(tools or []),
                    "content_preview": self._truncate_preview(result.content),
                },
            }
        return None

    def _looks_like_incomplete_tool_intent(
        self,
        *,
        content: str,
        tools: list[ToolSchema] | None,
    ) -> bool:
        stripped = content.strip()
        if not stripped:
            return False

        lowered = stripped.lower()
        if any(marker in lowered for marker in _IN_PROGRESS_MARKERS[:6]):
            return True
        if any(marker in stripped for marker in _IN_PROGRESS_MARKERS[6:]):
            return True

        tool_name_mentions = any(
            isinstance(tool.name, str) and tool.name and tool.name.lower() in lowered
            for tool in tools or []
        )
        uses_tool_word = " tool" in lowered or "tools" in lowered or "\u5de5\u5177" in stripped
        usage_phrase = any(phrase in lowered for phrase in _TOOL_USAGE_PHRASES[:7]) or any(
            phrase in stripped for phrase in _TOOL_USAGE_PHRASES[7:]
        )
        ends_like_placeholder = stripped.endswith(_PLACEHOLDER_SUFFIXES)
        fenced_placeholder = "```" in stripped and ends_like_placeholder

        if tool_name_mentions:
            return True
        if uses_tool_word and (usage_phrase or ends_like_placeholder):
            return True
        if usage_phrase and (ends_like_placeholder or fenced_placeholder):
            return True
        return False

    def _extract_context_metadata_from_show_payload(
        self,
        payload: dict[str, Any],
    ) -> dict[str, int | str | None]:
        runtime_context_length, runtime_context_length_source = self._extract_runtime_context_length(
            payload
        )
        model_max_context_length, model_max_context_length_source = self._extract_model_max_context_length(
            payload
        )

        effective_context_length = runtime_context_length
        effective_context_length_source = runtime_context_length_source
        if effective_context_length is None:
            effective_context_length = model_max_context_length
            effective_context_length_source = model_max_context_length_source

        return {
            "context_length": effective_context_length,
            "context_length_source": effective_context_length_source,
            "effective_context_length": effective_context_length,
            "effective_context_length_source": effective_context_length_source,
            "runtime_context_length": runtime_context_length,
            "runtime_context_length_source": runtime_context_length_source,
            "model_max_context_length": model_max_context_length,
            "model_max_context_length_source": model_max_context_length_source,
        }

    def _apply_configured_num_ctx(self) -> None:
        if self._configured_num_ctx is None:
            return
        self._runtime_context_length = self._configured_num_ctx
        self._runtime_context_length_source = "config.num_ctx"
        self._context_length = self._configured_num_ctx
        self._context_length_source = "config.num_ctx"

    def _auto_num_ctx_value(self) -> int | None:
        if not self._auto_num_ctx or self._configured_num_ctx is not None:
            return None
        if isinstance(self._runtime_context_length, int) and self._runtime_context_length > 0:
            return self._runtime_context_length
        if isinstance(self._model_max_context_length, int) and self._model_max_context_length > 0:
            return min(self._model_max_context_length, self._auto_num_ctx_cap)
        return min(self._CONTEXT_LENGTH_FALLBACK, self._auto_num_ctx_cap)

    def _request_num_ctx(self) -> int | None:
        if self._configured_num_ctx is not None:
            return self._configured_num_ctx
        return self._auto_num_ctx_value()

    def _extract_runtime_context_length(
        self,
        payload: dict[str, Any],
    ) -> tuple[int | None, str]:
        for key in ("num_ctx",):
            candidate = self._normalize_context_length(payload.get(key))
            if candidate is not None:
                return candidate, f"api_show.{key}"

        details = payload.get("details")
        if isinstance(details, dict):
            candidate = self._normalize_context_length(details.get("num_ctx"))
            if candidate is not None:
                return candidate, "api_show.details.num_ctx"

        for field_name in ("parameters", "modelfile"):
            text = payload.get(field_name)
            if not isinstance(text, str):
                continue
            parsed = self._extract_context_length_from_text(text)
            if parsed is not None:
                return parsed, f"api_show.{field_name}:num_ctx"

        model_info = payload.get("model_info")
        if isinstance(model_info, dict):
            for key, value in model_info.items():
                candidate = self._normalize_context_length(value)
                if candidate is None:
                    continue
                lowered = key.strip().lower()
                if lowered.endswith(".num_ctx") or lowered == "num_ctx":
                    return candidate, f"api_show.model_info.{key}"

        return None, "unknown"

    def _extract_model_max_context_length(
        self,
        payload: dict[str, Any],
    ) -> tuple[int | None, str]:
        for key in ("context_length", "context_window"):
            candidate = self._normalize_context_length(payload.get(key))
            if candidate is not None:
                return candidate, f"api_show.{key}"

        details = payload.get("details")
        if isinstance(details, dict):
            for key in ("context_length", "context_window"):
                candidate = self._normalize_context_length(details.get(key))
                if candidate is not None:
                    return candidate, f"api_show.details.{key}"

        model_info = payload.get("model_info")
        if isinstance(model_info, dict):
            for key, value in model_info.items():
                candidate = self._normalize_context_length(value)
                if candidate is None:
                    continue
                if key in {"context_length", "context_window"}:
                    return candidate, f"api_show.model_info.{key}"
                lowered = key.strip().lower()
                if lowered.endswith((".context_length", ".context_window")):
                    return candidate, f"api_show.model_info.{key}"

        return None, "unknown"

    @staticmethod
    def _extract_context_length_from_text(text: str) -> int | None:
        match = re.search(r"(?im)^\s*(?:parameter\s+)?num_ctx\s+(\d+)\s*$", text)
        if match is None:
            return None
        return OllamaBackend._normalize_context_length(match.group(1))

    @staticmethod
    def _normalize_context_length(value: Any) -> int | None:
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            return None
        return parsed if parsed > 0 else None

    def _record_native_tool_probe(
        self,
        *,
        status: str,
        message: str,
        activate_native: bool | None = None,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        previous_mode = self._tool_state.active_mode
        payload: dict[str, Any] = {
            "status": status,
            "message": message,
            "checked_at": self._build_probe_timestamp(),
        }
        if extra:
            payload.update(extra)
        if activate_native is True:
            changed = self._tool_state.recover_native(status)
            if changed and previous_mode != "native":
                append_fallback_diagnostic(
                    self._fallback_diagnostics,
                    category="tool_calling",
                    name="native_tool_calling_recovered",
                    reason=status,
                    kind="recovery",
                    severity="info",
                    from_state=previous_mode,
                    to_state="native",
                    metadata={"model": self.model, **(extra or {})},
                )
        elif activate_native is False:
            self._tool_state.enter_simulated(status)
        self._native_tool_probe = payload
        return payload

    def _switch_to_simulated_tool_mode(
        self,
        *,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        previous_mode = self._tool_state.active_mode
        changed = self._tool_state.enter_simulated("native_tool_calls_missing")
        self._native_tool_probe = {
            "status": "native_tool_calls_missing",
            "message": reason,
            "checked_at": self._build_probe_timestamp(),
        }
        if changed and previous_mode != "simulated_fallback":
            append_fallback_diagnostic(
                self._fallback_diagnostics,
                category="tool_calling",
                name="ollama_native_to_simulated",
                reason=reason,
                from_state=previous_mode,
                to_state="simulated_fallback",
                metadata=metadata,
            )

    def _mark_simulated_protocol_unavailable(
        self,
        *,
        status: str,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        previous_mode = self._tool_state.active_mode
        changed = self._tool_state.mark_unavailable(status)
        self._native_tool_probe = {
            "status": status,
            "message": reason,
            "checked_at": self._build_probe_timestamp(),
            **(metadata or {}),
        }
        if changed:
            append_fallback_diagnostic(
                self._fallback_diagnostics,
                category="tool_calling",
                name="tool_calling_unavailable",
                reason=status,
                kind="fallback",
                severity="warning",
                from_state=previous_mode,
                to_state="unavailable",
                metadata={"model": self.model, **(metadata or {})},
            )

    def _record_simulated_protocol_rejection(
        self,
        *,
        status: str,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        previous_mode = self._tool_state.active_mode
        changed = self._tool_state.enter_simulated(status)
        self._native_tool_probe = {
            "status": status,
            "message": reason,
            "checked_at": self._build_probe_timestamp(),
            **(metadata or {}),
        }
        append_fallback_diagnostic(
            self._fallback_diagnostics,
            category="tool_calling",
            name="prompt_guided_tool_turn_rejected",
            reason=status,
            kind="fallback",
            severity="warning",
            from_state=previous_mode,
            to_state=self._tool_state.active_mode,
            metadata={"model": self.model, **(metadata or {})},
        )

    def _record_native_tool_turn_rejection(
        self,
        *,
        status: str,
        reason: str,
        result: GenerationResult,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        error_metadata = build_invalid_tool_turn_metadata(
            result=result,
            reason=validate_tool_turn_result(result=result, tools_requested=True).reason,
        )
        self._native_tool_probe = {
            "status": status,
            "message": reason,
            "checked_at": self._build_probe_timestamp(),
            **(metadata or {}),
            **error_metadata,
        }
        append_fallback_diagnostic(
            self._fallback_diagnostics,
            category="tool_calling",
            name="native_tool_turn_rejected",
            reason=status,
            kind="native",
            severity="warning",
            from_state=self._tool_state.active_mode,
            to_state=self._tool_state.active_mode,
            metadata={"model": self.model, **(metadata or {}), **error_metadata},
        )
        raise BackendRequestError(
            "Ollama returned an invalid native tool-eligible turn.",
            metadata={
                "backend_name": "ollama",
                "model": self.model,
                **(metadata or {}),
                **error_metadata,
            },
        )

    async def probe_tool_calling(self) -> dict[str, Any] | None:
        probe_tool = ToolSchema(
            name="mochi_tool_probe",
            description="Diagnostic probe tool. Echo the requested value.",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
            },
        )
        payload = self._build_request_payload(
            messages=[Message(role="user", content="Call mochi_tool_probe with value='ok'.")],
            tools=[probe_tool],
            options={
                "temperature": 0.0,
                "num_predict": 128,
                "top_p": 1.0,
                "top_k": 0,
                "repeat_penalty": 1.0,
            },
            stream=False,
            think_value=None,
        )
        try:
            result = await self._blocking_generate(
                payload,
                tools=[probe_tool],
                use_native_tools=True,
            )
        except BackendRequestError as exc:
            status = f"http_{exc.metadata['status_code']}" if "status_code" in exc.metadata else "request_error"
            return self._record_probe_failure(
                status=status,
                message="Native tool-calling probe request failed.",
                metadata=exc.metadata,
            )

        verdict = validate_tool_turn_result(result=result, tools_requested=True)
        if verdict.reason == "tool_calls":
            return self._record_native_tool_probe(
                status="supported",
                message="Native structured tool calling succeeded; using native Ollama tools.",
                activate_native=True,
                extra={"tool_calls": len(result.tool_calls)},
            )
        return self._record_probe_failure(
            status=verdict.reason,
            message="Probe did not return structured tool calls.",
            metadata={"tool_turn_reason": verdict.reason},
        )

    def _record_probe_failure(
        self,
        *,
        status: str,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "status": status,
            "message": message,
            "checked_at": self._build_probe_timestamp(),
            **(metadata or {}),
        }
        if (
            self._tool_state.active_mode == "simulated_fallback"
            and self._tool_state.fallback_validation_status == "validated"
        ):
            self._native_tool_probe = payload
            append_fallback_diagnostic(
                self._fallback_diagnostics,
                category="tool_calling",
                name="native_tool_calling_recovery_failed",
                reason=status,
                kind="recovery",
                severity="warning",
                from_state="simulated_fallback",
                to_state="simulated_fallback",
                metadata={"model": self.model, **(metadata or {})},
            )
            return payload

        self._mark_simulated_protocol_unavailable(
            status=status,
            reason=message,
            metadata=metadata,
        )
        return self._native_tool_probe

    async def _stream_generate(self, payload: dict[str, Any]) -> AsyncIterator[StreamChunk]:
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

                    done = data.get("done", False)
                    msg = data.get("message", {})
                    delta = msg.get("content", "")
                    thinking_delta = msg.get("thinking", "")

                    yield StreamChunk(
                        delta=delta if isinstance(delta, str) else "",
                        thinking_delta=thinking_delta if isinstance(thinking_delta, str) else "",
                        is_final=done,
                        finish_reason=data.get("done_reason") if done else None,
                    )
                    if done:
                        break
        except httpx.RequestError as exc:
            logger.error(f"Ollama stream error: {exc}")
            raise self._wrap_request_error(exc, stage="stream_generate") from exc

    async def close(self) -> None:
        await self._client.aclose()

    def _serialize_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
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
                        },
                    }
                    for tool_call in message.tool_calls
                ]
            if message.role == "tool" and message.name:
                payload["tool_name"] = message.name
            serialized.append(payload)
        return serialized

    def _serialize_tools(self, tools: list[ToolSchema]) -> list[dict[str, Any]]:
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
    def _supports_reasoning_effort_model(model: str) -> bool:
        return "gpt-oss" in model.lower()

    @staticmethod
    def _build_probe_timestamp() -> str:
        from datetime import UTC, datetime

        return datetime.now(UTC).isoformat()

    def _reasoning_effort_to_think_value(
        self, effort: str | None
    ) -> str | bool | None:
        if effort == "none":
            return False
        if effort not in {"low", "medium", "high"}:
            return None
        if not self._supports_reasoning_effort_model(self.model):
            return None
        return effort
