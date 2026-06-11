"""OpenAI-compatible API 後端實作。"""

from __future__ import annotations

import copy
import json
import logging
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, cast
from urllib.parse import urlsplit, urlunsplit

import httpx

from mochi.backends.inference_capabilities import (
    ReasoningEffort,
    resolve_model_inference_capabilities,
)
from mochi.backends.base import BackendRequestError, BaseLLMBackend
from mochi.backends.tool_call_simulator import ToolCallSimulator
from mochi.backends.types import (
    GenerationResult,
    Message,
    ModelInfo,
    ResponsesContinuityMode,
    ResponsesReplayState,
    StreamChunk,
    ToolCall,
    ToolSchema,
)
from mochi.diagnostics.fallbacks import append_fallback_diagnostic

logger = logging.getLogger(__name__)


ApiMode = Literal["chat_completions", "responses"]


@dataclass(frozen=True)
class _OpenAICompatEndpoint:
    """OpenAI-compatible URL 正規化結果。"""

    input_url: str
    request_url: str
    chat_completions_url: str
    responses_url: str
    api_mode: ApiMode
    health_urls: tuple[str, ...]
    transport_locked: bool


@dataclass(frozen=True)
class _ResponsesInputState:
    input_items: list[dict[str, Any]]
    previous_response_id: str | None
    continuity_mode: ResponsesContinuityMode
    replayed_items: int


class OpenAICompatBackend(BaseLLMBackend):
    """通用 OpenAI-compatible API 後端。"""

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "",
        timeout: float = 120.0,
        provider: str = "openai_compat",
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
        self._chat_completions_url = endpoint.chat_completions_url
        self._responses_url = endpoint.responses_url
        self._health_urls = endpoint.health_urls
        self._transport_locked = endpoint.transport_locked
        self.model = model
        self.api_key = api_key
        self.provider = provider
        self._tool_calling_enabled = True
        self._tool_call_simulator = ToolCallSimulator()
        self._native_tool_probe: dict[str, Any] | None = None
        self._tool_protocol_probe: dict[str, Any] | None = None
        self._tool_calling_blocked = False
        self._responses_chat_completions_alias = False
        self._fallback_diagnostics: list[dict[str, Any]] = []
        self._responses_reasoning_summary_supported: bool | None = None
        self._responses_encrypted_include_supported: bool | None = None
        self._responses_reasoning_summary_received = False
        self._responses_last_replayed_items = 0
        self._responses_last_continuity_mode: ResponsesContinuityMode = "none"
        self._client = httpx.AsyncClient(timeout=timeout)

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
        prepared_messages = self._prepare_messages(messages, tools)
        use_chat_transport = self._uses_chat_completions_transport()
        chat_payload = self._build_chat_completions_payload(
            prepared_messages,
            tools,
            temperature,
            max_tokens,
            top_p,
            frequency_penalty,
            presence_penalty,
            reasoning_effort,
            stream,
        )

        if use_chat_transport:
            self._responses_last_continuity_mode = "none"
            self._responses_last_replayed_items = 0
            self._responses_reasoning_summary_received = False
            if stream:
                return self._stream_generate(chat_payload, request_url=self._chat_completions_url)
            return await self._blocking_generate(chat_payload, request_url=self._chat_completions_url)

        responses_payload = self._build_responses_payload(
            prepared_messages,
            tools,
            temperature,
            max_tokens,
            top_p,
            frequency_penalty,
            presence_penalty,
            reasoning_effort,
            stream,
        )

        if stream:
            return self._stream_generate(responses_payload, request_url=self._responses_url)
        return await self._blocking_generate_with_responses_alias_fallback(
            responses_payload,
            chat_payload,
            responses_url=self._responses_url,
            chat_url=self._chat_completions_url,
        )

    def supports_tool_calling(self) -> bool:
        """OpenAI-compatible 端點通常支援 tool calling。"""
        return not self._tool_calling_blocked and self._tool_calling_enabled

    def _tool_call_mode(self) -> str:
        if self._tool_calling_blocked:
            return "unavailable"
        return "native" if self._tool_calling_enabled else "simulated_fallback"

    def _uses_chat_completions_transport(self) -> bool:
        return not self._should_use_responses_transport()

    def _should_use_responses_transport(self) -> bool:
        if self._responses_chat_completions_alias:
            return False
        if self._api_mode == "responses":
            return True
        if self._transport_locked:
            return False
        if not self._is_openai_native_reasoning_model():
            return False
        if self._responses_transport_rejected():
            return False
        return True

    def _responses_transport_rejected(self) -> bool:
        if not isinstance(self._tool_protocol_probe, dict):
            return False
        protocol_results = self._tool_protocol_probe.get("protocol_results")
        if not isinstance(protocol_results, list):
            return False
        for item in protocol_results:
            if not isinstance(item, dict):
                continue
            if item.get("protocol") != "responses":
                continue
            status = item.get("status")
            if isinstance(status, str) and status.startswith("http_"):
                return True
        return False

    def _is_openai_native_reasoning_model(self) -> bool:
        normalized_provider = (self.provider or "").strip().lower()
        normalized_model = self.model.strip().lower()
        return normalized_provider in {"openai_compat", "openai_codex"} and normalized_model.startswith(
            "gpt-5"
        )

    def _reasoning_transport_preference(self) -> str:
        if self._responses_chat_completions_alias:
            return "chat_completions_alias_fallback"
        if self._api_mode == "responses":
            return "responses_explicit"
        if self._transport_locked and self._api_mode == "chat_completions":
            return "chat_completions_explicit"
        if self._should_use_responses_transport():
            return "responses_preferred"
        return "chat_completions_default"

    def get_model_info(self) -> ModelInfo:
        """回傳目前模型資訊。"""
        capabilities = resolve_model_inference_capabilities(
            ModelInfo(
                name=self.model,
                provider=self.provider,
                backend_type="openai_compat",
                supports_tool_calling=not self._tool_calling_blocked and self._tool_calling_enabled,
                metadata={"api_mode": self._api_mode},
            )
        )
        return ModelInfo(
            name=self.model,
            provider=self.provider,
            backend_type="openai_compat",
            supports_tool_calling=not self._tool_calling_blocked and self._tool_calling_enabled,
            metadata={
                "base_url": self.base_url,
                "api_url": self._chat_completions_url
                if self._uses_chat_completions_transport()
                else self._responses_url,
                "api_mode": self._api_mode,
                "provider": self.provider,
                "request_shape": "chat_completions"
                if self._uses_chat_completions_transport()
                else "responses",
                "responses_alias_detected": self._responses_chat_completions_alias,
                "reasoning_transport_preference": self._reasoning_transport_preference(),
                "reasoning_summary_requested": self._should_request_reasoning_summary(),
                "reasoning_summary_supported": self._responses_reasoning_summary_supported,
                "reasoning_summary_received": self._responses_reasoning_summary_received,
                "reasoning_items_replayed": self._responses_last_replayed_items,
                "responses_continuity_mode": self._responses_last_continuity_mode,
                "tool_call_mode": self._tool_call_mode(),
                "native_tool_calling_status": self._native_tool_probe.get("status")
                if isinstance(self._native_tool_probe, dict)
                else "unknown",
                "native_tool_calling_message": self._native_tool_probe.get("message")
                if isinstance(self._native_tool_probe, dict)
                else None,
                "native_tool_calling_checked_at": self._native_tool_probe.get("checked_at")
                if isinstance(self._native_tool_probe, dict)
                else None,
                "fallback_diagnostics": list(self._fallback_diagnostics),
                "tool_calling_blocked": self._tool_calling_blocked,
                "tool_calling_protocol": self._api_mode,
                "tool_protocol_probe": dict(self._tool_protocol_probe)
                if isinstance(self._tool_protocol_probe, dict)
                else None,
                **capabilities.to_metadata(),
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
                if 200 <= resp.status_code < 500:
                    return True
            except Exception as exc:
                logger.debug(f"OpenAI-compatible health check failed on {url}: {exc}")
        return False

    async def _blocking_generate(
        self,
        payload: dict[str, Any],
        *,
        request_url: str,
    ) -> GenerationResult:
        """執行非串流生成。"""
        try:
            resp = await self._post_json(payload, request_url=request_url)
        except httpx.HTTPStatusError as exc:
            logger.error(
                f"OpenAI-compatible API error {exc.response.status_code}: {exc.response.text}"
            )
            raise self._wrap_request_error(exc, stage="generate", request_url=request_url) from exc
        except httpx.RequestError as exc:
            logger.error(f"OpenAI-compatible connection error: {exc}")
            raise self._wrap_request_error(exc, stage="generate", request_url=request_url) from exc

        data = resp.json()
        if request_url == self._responses_url and not self._responses_chat_completions_alias:
            return self._parse_responses_result(data)

        return self._parse_chat_completions_result(data)

    async def _blocking_generate_with_responses_alias_fallback(
        self,
        responses_payload: dict[str, Any],
        chat_payload: dict[str, Any],
        *,
        responses_url: str,
        chat_url: str,
    ) -> GenerationResult:
        try:
            resp = await self._post_json(responses_payload, request_url=responses_url)
        except httpx.HTTPStatusError as exc:
            alias_result = await self._retry_chat_completions_after_responses_error(
                exc,
                chat_payload,
                responses_url=responses_url,
                chat_url=chat_url,
                stage="blocking_generate",
            )
            if alias_result is not None:
                return alias_result
            logger.error(
                f"OpenAI-compatible API error {exc.response.status_code}: {exc.response.text}"
            )
            raise self._wrap_request_error(exc, stage="generate", request_url=responses_url) from exc
        except httpx.RequestError as exc:
            logger.error(f"OpenAI-compatible connection error: {exc}")
            raise self._wrap_request_error(exc, stage="generate", request_url=responses_url) from exc

        data = resp.json()
        if _looks_like_chat_completions_response(data):
            result = self._parse_chat_completions_result(data)
            if self._has_generation_output(result):
                logger.info(
                    "Detected chat-completions-compatible payloads on responses endpoint %s; switching transport.",
                    self._request_url,
                )
                self._enable_responses_alias(
                    reason="responses_endpoint_returned_chat_completions_response",
                    metadata={"stage": "blocking_generate"},
                )
                return result

        result = self._parse_responses_result(data)
        if self._has_generation_output(result):
            return result

        logger.warning(
            "Responses endpoint %s returned an empty result; retrying with chat-completions payload.",
            self._request_url,
        )
        try:
            retry_resp = await self._post_json(chat_payload, request_url=chat_url)
            retry_result = self._parse_chat_completions_result(retry_resp.json())
        except httpx.HTTPError as exc:
            logger.warning(
                "Chat-completions alias retry failed for responses endpoint %s: %s",
                request_url,
                exc,
            )
            return result

        if self._has_generation_output(retry_result):
            logger.info(
                "Responses endpoint %s accepted chat-completions payloads; caching alias transport.",
                responses_url,
            )
            self._enable_responses_alias(
                reason="responses_endpoint_required_chat_completions_retry",
                metadata={"stage": "blocking_generate"},
            )
            return retry_result

        return result

    async def _retry_chat_completions_after_responses_error(
        self,
        exc: httpx.HTTPStatusError,
        chat_payload: dict[str, Any],
        *,
        responses_url: str,
        chat_url: str,
        stage: str,
    ) -> GenerationResult | None:
        if not self._should_fallback_responses_http_error_to_chat(exc):
            return None

        response_text = self._safe_response_text(exc.response)
        logger.warning(
            "Responses endpoint %s rejected the request with %s; retrying with chat-completions payload.",
            responses_url,
            exc.response.status_code,
        )
        try:
            retry_resp = await self._post_json(chat_payload, request_url=chat_url)
            retry_result = self._parse_chat_completions_result(retry_resp.json())
        except httpx.HTTPError as retry_exc:
            logger.warning(
                "Chat-completions fallback after responses HTTP error failed for %s: %s",
                responses_url,
                retry_exc,
            )
            return None

        if not self._has_generation_output(retry_result):
            return None

        self._enable_responses_alias(
            reason="responses_endpoint_rejected_payload_shape",
            metadata={
                "stage": stage,
                "http_status": exc.response.status_code,
                "response_text": response_text[:1000],
            },
        )
        return retry_result

    async def _post_json(
        self,
        payload: dict[str, Any],
        *,
        request_url: str,
        allow_reasoning_retry: bool = True,
    ) -> httpx.Response:
        try:
            resp = await self._client.post(
                request_url,
                json=payload,
                headers=self._build_headers(),
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            retry_payload = self._build_responses_reasoning_retry_payload(exc, payload)
            if retry_payload is not None and allow_reasoning_retry:
                retry_resp = await self._client.post(
                    request_url,
                    json=retry_payload,
                    headers=self._build_headers(),
                )
                retry_resp.raise_for_status()
                return retry_resp
            fallback = self._native_tool_fallback_for_error(exc, payload)
            if fallback is not None:
                status, message = fallback
                self._record_native_tool_probe(
                    status=status,
                    message=message,
                    extra={
                        "http_status": exc.response.status_code,
                        "response_text": exc.response.text,
                    },
                    enable_native=False,
                )
                logger.warning(
                    "Disabling tool calling for provider=%s model=%s after automatic tool choice was rejected.",
                    self.provider,
                    self.model,
                )
                retry_payload = self._build_simulated_tool_retry_payload(payload)
                retry_resp = await self._client.post(
                    request_url,
                    json=retry_payload,
                    headers=self._build_headers(),
                )
                try:
                    retry_resp.raise_for_status()
                except httpx.HTTPStatusError as retry_exc:
                    self._record_tool_calling_blocked_from_retry(retry_exc)
                    raise
                return retry_resp
            raise
        return resp

    def _record_tool_calling_blocked_from_retry(self, exc: httpx.HTTPStatusError) -> None:
        if exc.response.status_code not in {403, 429}:
            return
        response_text = self._safe_response_text(exc.response)
        lowered = response_text.lower()
        if not any(
            marker in lowered
            for marker in (
                "insufficient_quota",
                "permission_error",
                "usage_limit",
                "tool",
                "function",
            )
        ):
            return
        self._tool_calling_blocked = True
        self._tool_protocol_probe = {
            "status": "all_tool_protocols_rejected_by_provider",
            "selected_protocol": None,
            "protocol_results": [
                {
                    "protocol": self._api_mode,
                    "url": self._request_url,
                    "status": f"http_{exc.response.status_code}",
                    "http_status": exc.response.status_code,
                    "response_text": response_text[:1000],
                }
            ],
            "checked_at": datetime.now(UTC).isoformat(),
        }
        append_fallback_diagnostic(
            self._fallback_diagnostics,
            category="tool_calling",
            name="tool_calling_unavailable",
            reason="simulated_tool_retry_rejected_by_provider",
            kind="fallback",
            severity="warning",
            from_state="simulated_fallback",
            to_state="unavailable",
            metadata={
                "provider": self.provider,
                "model": self.model,
                "http_status": exc.response.status_code,
            },
        )

    def _record_native_tool_probe(
        self,
        *,
        status: str,
        message: str,
        extra: dict[str, Any] | None = None,
        enable_native: bool | None = None,
    ) -> dict[str, Any]:
        previous_tool_calling_enabled = self._tool_calling_enabled
        payload: dict[str, Any] = {
            "status": status,
            "message": message,
            "checked_at": datetime.now(UTC).isoformat(),
        }
        if extra:
            payload.update(extra)
        if enable_native is not None:
            self._tool_calling_enabled = enable_native
        if enable_native is not None and enable_native != previous_tool_calling_enabled:
            append_fallback_diagnostic(
                self._fallback_diagnostics,
                category="tool_calling",
                name=(
                    "native_tool_calling_recovered"
                    if enable_native
                    else "native_tool_calling_disabled"
                ),
                reason=status,
                kind="recovery" if enable_native else "fallback",
                severity="info" if enable_native else "warning",
                from_state="simulated_fallback" if enable_native else "native",
                to_state="native" if enable_native else "simulated_fallback",
                metadata={
                    "provider": self.provider,
                    "model": self.model,
                    **(extra or {}),
                },
            )
        self._native_tool_probe = payload
        return payload

    def _enable_responses_alias(
        self,
        *,
        reason: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if self._responses_chat_completions_alias:
            return
        self._responses_chat_completions_alias = True
        append_fallback_diagnostic(
            self._fallback_diagnostics,
            category="transport",
            name="responses_alias_transport",
            reason=reason,
            kind="fallback",
            severity="warning",
            from_state="responses",
            to_state="chat_completions",
            metadata={
                "request_url": self._request_url,
                **(metadata or {}),
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
                "additionalProperties": False,
            },
        )
        protocol_order: list[ApiMode] = [self._api_mode]
        alternate: ApiMode = "responses" if self._api_mode == "chat_completions" else "chat_completions"
        if alternate not in protocol_order:
            protocol_order.append(alternate)

        protocol_results: list[dict[str, Any]] = []
        for protocol in protocol_order:
            result = await self._probe_tool_protocol(protocol, probe_tool)
            protocol_results.append(result)
            if result.get("status") == "supported":
                if result.get("responses_alias_detected") is True:
                    self._api_mode = "responses"
                    self._request_url = self._responses_url
                    self._responses_chat_completions_alias = False
                    self._enable_responses_alias(
                        reason="responses_probe_returned_native_chat_tool_calls",
                        metadata={"stage": "probe_tool_calling"},
                    )
                else:
                    self._select_tool_protocol(protocol)
                self._tool_calling_blocked = False
                self._tool_protocol_probe = {
                    "status": "supported",
                    "selected_protocol": protocol,
                    "protocol_results": protocol_results,
                    "checked_at": datetime.now(UTC).isoformat(),
                }
                return self._record_native_tool_probe(
                    status="supported",
                    message=f"{protocol} structured tool calling succeeded.",
                    extra={
                        "tool_calls": result.get("tool_calls", 1),
                        "tool_protocol": protocol,
                        "protocol_results": protocol_results,
                    },
                    enable_native=True,
                )

        blocked = any(
            item.get("status") in {"native_tools_rejected_by_provider", "http_403", "http_429"}
            for item in protocol_results
        )
        status = "all_tool_protocols_rejected_by_provider" if blocked else "no_tool_protocol_supported"
        message = (
            "Provider rejected both OpenAI Chat Completions tools and Responses function tools."
            if blocked
            else "No OpenAI tool-calling protocol produced structured tool calls."
        )
        self._tool_calling_blocked = blocked
        self._tool_protocol_probe = {
            "status": status,
            "selected_protocol": None,
            "protocol_results": protocol_results,
            "checked_at": datetime.now(UTC).isoformat(),
        }
        return self._record_native_tool_probe(
            status=status,
            message=message,
            extra={"protocol_results": protocol_results},
            enable_native=False,
        )

    async def _probe_tool_protocol(
        self,
        protocol: ApiMode,
        probe_tool: ToolSchema,
    ) -> dict[str, Any]:
        url = self._chat_completions_url if protocol == "chat_completions" else self._responses_url
        payload = (
            self._build_tool_probe_chat_payload(probe_tool)
            if protocol == "chat_completions"
            else self._build_tool_probe_responses_payload(probe_tool)
        )
        try:
            response = await self._client.post(
                url,
                json=payload,
                headers=self._build_headers(),
            )
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            fallback = self._native_tool_fallback_for_error(exc, payload)
            response_text = self._safe_response_text(exc.response)
            status = fallback[0] if fallback is not None else f"http_{exc.response.status_code}"
            message = fallback[1] if fallback is not None else f"HTTP {exc.response.status_code}"
            return {
                "protocol": protocol,
                "url": url,
                "status": status,
                "message": message,
                "http_status": exc.response.status_code,
                "response_text": response_text[:1000],
            }
        except httpx.RequestError as exc:
            return {
                "protocol": protocol,
                "url": url,
                "status": "request_error",
                "message": str(exc),
            }

        try:
            data = response.json()
        except ValueError as exc:
            return {
                "protocol": protocol,
                "url": url,
                "status": "invalid_json",
                "message": str(exc),
                "response_text": self._safe_response_text(response)[:1000],
            }

        responses_alias_detected = False
        if protocol == "responses" and _looks_like_chat_completions_response(data):
            tool_calls = self._extract_chat_probe_tool_calls(data)
            responses_alias_detected = bool(tool_calls)
        else:
            tool_calls = (
                self._extract_chat_probe_tool_calls(data)
                if protocol == "chat_completions"
                else self._parse_responses_tool_calls(data.get("output"))
            )
        if tool_calls:
            return {
                "protocol": protocol,
                "url": url,
                "status": "supported",
                "tool_calls": len(tool_calls),
                "responses_alias_detected": responses_alias_detected,
            }

        preview = self._response_preview_for_protocol(protocol, data)
        status = "text_tool_call_only" if "<tool_call>" in preview else "no_tool_calls"
        return {
            "protocol": protocol,
            "url": url,
            "status": status,
            "message": "Probe completed without structured tool calls.",
            "response_preview": preview[:240],
        }

    def _build_tool_probe_chat_payload(self, probe_tool: ToolSchema) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": self._tool_probe_chat_messages(),
            "tools": [probe_tool.to_dict()],
            "tool_choice": "auto",
            "stream": False,
        }
        capabilities = resolve_model_inference_capabilities(self.get_model_info())
        supported = set(capabilities.supported_inference_parameters)
        if "temperature" in supported:
            payload["temperature"] = 0.0
        if "max_tokens" in supported:
            payload["max_tokens"] = 96
        return payload

    def _build_tool_probe_responses_payload(self, probe_tool: ToolSchema) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "instructions": self._tool_probe_system_prompt(),
            "input": [{"role": "user", "content": "Run the diagnostic tool probe now."}],
            "tools": [
                {
                    "type": "function",
                    "name": probe_tool.name,
                    "description": probe_tool.description,
                    "parameters": probe_tool.parameters,
                }
            ],
            "tool_choice": "auto",
            "stream": False,
        }
        capabilities = resolve_model_inference_capabilities(self.get_model_info())
        supported = set(capabilities.supported_inference_parameters)
        if "temperature" in supported:
            payload["temperature"] = 0.0
        if "max_tokens" in supported:
            payload["max_output_tokens"] = 96
        return payload

    @staticmethod
    def _tool_probe_system_prompt() -> str:
        return (
            "You are running a native tool-calling diagnostics probe. "
            "Call the mochi_tool_probe tool exactly once with {\"value\":\"ok\"}. "
            "Do not answer in plain text."
        )

    def _tool_probe_chat_messages(self) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self._tool_probe_system_prompt()},
            {"role": "user", "content": "Run the diagnostic tool probe now."},
        ]

    def _extract_chat_probe_tool_calls(self, data: dict[str, Any]) -> list[ToolCall]:
        choices = data.get("choices", [])
        choice0 = choices[0] if isinstance(choices, list) and choices else {}
        message = choice0.get("message", {}) if isinstance(choice0, dict) else {}
        if not isinstance(message, dict):
            message = {}
        return self._parse_tool_calls(message.get("tool_calls", []))

    def _response_preview_for_protocol(self, protocol: ApiMode, data: dict[str, Any]) -> str:
        if protocol == "responses":
            result = self._parse_responses_result(data)
            return result.content
        choices = data.get("choices", [])
        choice0 = choices[0] if isinstance(choices, list) and choices else {}
        message = choice0.get("message", {}) if isinstance(choice0, dict) else {}
        if not isinstance(message, dict):
            return ""
        content, _thinking = self._normalize_chat_message_parts(message)
        return content

    def _select_tool_protocol(self, protocol: ApiMode) -> None:
        self._api_mode = protocol
        self._request_url = (
            self._chat_completions_url
            if protocol == "chat_completions"
            else self._responses_url
        )
        self._responses_chat_completions_alias = False

    def _prepare_messages(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
    ) -> list[Message]:
        prepared = [Message(**message.__dict__) for message in messages]
        if self._tool_calling_enabled or not tools or not self._uses_chat_completions_transport():
            return prepared

        prepared = self._flatten_simulated_tool_messages(prepared)
        injected = False
        for message in prepared:
            if message.role == "system":
                message.content = self._tool_call_simulator.inject_tools_into_prompt(
                    message.content,
                    tools,
                )
                injected = True
                break

        if not injected:
            prepared.insert(
                0,
                Message(
                    role="system",
                    content=self._tool_call_simulator.inject_tools_into_prompt("", tools).strip(),
                ),
            )
        return prepared

    def _flatten_simulated_tool_messages(self, messages: list[Message]) -> list[Message]:
        flattened: list[Message] = []
        for message in messages:
            if message.role == "assistant" and message.tool_calls:
                if message.content:
                    flattened.append(Message(role="assistant", content=message.content))
                for tool_call in message.tool_calls:
                    flattened.append(
                        Message(
                            role="assistant",
                            content=(
                                f"Tool request: {tool_call.name}\n"
                                f"Arguments: {tool_call.arguments}"
                            ),
                        )
                    )
                continue

            if message.role == "tool":
                tool_name = message.name or "tool"
                tool_text = message.content
                if not tool_text.startswith(f"Tool {tool_name} result:") and not tool_text.startswith(
                    f"Tool {tool_name} error:"
                ):
                    tool_text = f"Tool {tool_name} result:\n{tool_text}"
                flattened.append(Message(role="user", content=tool_text))
                continue

            flattened.append(Message(**message.__dict__))
        return flattened

    def _parse_chat_completions_result(self, data: dict[str, Any]) -> GenerationResult:
        """解析 Chat Completions 回應。"""
        choices: list[dict[str, Any]] = data.get("choices", [])
        choice0 = choices[0] if choices else {}
        message = choice0.get("message", {})

        content, thinking = self._normalize_chat_message_parts(message)

        tool_calls = self._parse_tool_calls(message.get("tool_calls", []))
        if not tool_calls and not self._tool_calling_enabled and content:
            tool_calls = self._tool_call_simulator.parse_tool_calls(content)
            if tool_calls:
                content = self._tool_call_simulator.extract_text_response(content)
        usage = data.get("usage", {})
        finish_reason = choice0.get("finish_reason")
        if tool_calls:
            finish_reason = "tool_calls"
        if not finish_reason:
            finish_reason = "tool_calls" if tool_calls else "stop"

        return GenerationResult(
            content=content,
            thinking=thinking,
            tool_calls=tool_calls,
            input_tokens=self._as_int(usage.get("prompt_tokens")),
            output_tokens=self._as_int(usage.get("completion_tokens")),
            model=data.get("model", self.model),
            finish_reason=finish_reason,
        )

    async def _stream_generate(
        self,
        payload: dict[str, Any],
        *,
        request_url: str,
    ) -> AsyncIterator[StreamChunk]:
        """執行串流生成（SSE / JSON lines）。"""
        if request_url == self._responses_url and not self._responses_chat_completions_alias:
            async for chunk in self._stream_responses_generate(payload, request_url=request_url):
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
                await self._read_stream_error_body_before_raise(resp)
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

                    text_delta, thinking_delta = self._normalize_chat_delta(delta_obj)

                    tool_call_delta = self._parse_tool_call_delta(
                        delta_obj.get("tool_calls"),
                        tool_call_buffers,
                    )

                    finish_reason = choice0.get("finish_reason")
                    is_final = finish_reason is not None
                    if is_final:
                        emitted_final = True

                    if text_delta or thinking_delta or tool_call_delta is not None or is_final:
                        yield StreamChunk(
                            delta=text_delta,
                            thinking_delta=thinking_delta,
                            tool_call_delta=tool_call_delta,
                            is_final=is_final,
                            finish_reason=finish_reason,
                        )
        except httpx.HTTPStatusError as exc:
            response_text = await self._read_response_text(exc.response)
            logger.error(
                f"OpenAI-compatible stream API error {exc.response.status_code}: {response_text}"
            )
            fallback = self._native_tool_fallback_for_error(exc, payload)
            if fallback is not None:
                status, message = fallback
                self._record_native_tool_probe(
                    status=status,
                    message=message,
                    extra={
                        "http_status": exc.response.status_code,
                        "response_text": response_text,
                    },
                    enable_native=False,
                )
                retry_payload = self._build_simulated_tool_retry_payload(payload)
                try:
                    retry_result = await self._blocking_generate(retry_payload, request_url=request_url)
                except BackendRequestError as retry_exc:
                    if retry_exc.metadata.get("status_code") in {403, 429}:
                        self._tool_calling_blocked = True
                        self._tool_protocol_probe = {
                            "status": "all_tool_protocols_rejected_by_provider",
                            "selected_protocol": None,
                            "protocol_results": [
                                {
                                    "protocol": self._api_mode,
                                    "url": self._request_url,
                                    "status": f"http_{retry_exc.metadata.get('status_code')}",
                                    "http_status": retry_exc.metadata.get("status_code"),
                                    "response_text": str(retry_exc.metadata.get("response_text") or "")[:1000],
                                }
                            ],
                            "checked_at": datetime.now(UTC).isoformat(),
                        }
                    raise
                if retry_result.content:
                    yield StreamChunk(delta=retry_result.content)
                for index, tool_call in enumerate(retry_result.tool_calls):
                    yield StreamChunk(
                        tool_call_delta=ToolCall(
                            id=tool_call.id,
                            name=tool_call.name,
                            arguments=tool_call.arguments,
                            index=tool_call.index if tool_call.index is not None else index,
                        )
                    )
                yield StreamChunk(
                    is_final=True,
                    finish_reason="tool_calls" if retry_result.tool_calls else retry_result.finish_reason,
                )
                return
            raise self._wrap_request_error(exc, stage="stream_generate", request_url=request_url) from exc
        except httpx.RequestError as exc:
            logger.error(f"OpenAI-compatible stream connection error: {exc}")
            raise self._wrap_request_error(exc, stage="stream_generate", request_url=request_url) from exc

    def _normalize_chat_message_parts(self, message: dict[str, Any]) -> tuple[str, str]:
        content = self._stringify_chat_text_part(message.get("content"))
        reasoning = self._extract_chat_reasoning(message)
        return content, reasoning

    def _normalize_chat_delta(
        self,
        delta_obj: dict[str, Any],
    ) -> tuple[str, str]:
        reasoning_delta = self._extract_chat_reasoning(delta_obj)
        content_delta = self._stringify_chat_text_part(delta_obj.get("content"))
        return content_delta, reasoning_delta

    def _extract_chat_reasoning(self, payload: dict[str, Any]) -> str:
        for key in ("reasoning_content", "reasoning"):
            value = self._stringify_chat_text_part(payload.get(key))
            if value:
                return value
        thinking_blocks = payload.get("thinking_blocks")
        if isinstance(thinking_blocks, list):
            parts: list[str] = []
            for raw_block in thinking_blocks:
                if not isinstance(raw_block, dict):
                    continue
                block = cast(dict[str, Any], raw_block)
                thinking = self._stringify_chat_text_part(
                    block.get("thinking") or block.get("text") or block.get("content")
                )
                if thinking:
                    parts.append(thinking)
            return "\n\n".join(parts)
        return ""

    def _stringify_chat_text_part(self, value: Any) -> str:
        if isinstance(value, str):
            return value
        if value is None:
            return ""
        return json.dumps(value, ensure_ascii=False)

    async def _stream_responses_generate(
        self,
        payload: dict[str, Any],
        *,
        request_url: str,
        allow_reasoning_retry: bool = True,
    ) -> AsyncIterator[StreamChunk]:
        """執行 Responses API 串流生成。"""
        emitted_final = False
        tool_call_buffers: dict[int, dict[str, str]] = {}
        try:
            async with self._client.stream(
                "POST",
                self._request_url,
                json=payload,
                headers=self._build_headers(),
            ) as resp:
                await self._read_stream_error_body_before_raise(resp)
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

                    if _looks_like_chat_completions_response(data):
                        self._enable_responses_alias(
                            reason="responses_stream_returned_chat_completions_delta",
                            metadata={"stage": "stream_generate"},
                        )
                        choices: list[dict[str, Any]] = data.get("choices", [])
                        if not choices:
                            continue

                        choice0 = choices[0]
                        delta_obj = choice0.get("delta", {})

                        text_delta, thinking_delta = self._normalize_chat_delta(delta_obj)

                        tool_call_delta = self._parse_tool_call_delta(
                            delta_obj.get("tool_calls"),
                            tool_call_buffers,
                        )

                        finish_reason = choice0.get("finish_reason")
                        is_final = finish_reason is not None
                        if is_final:
                            emitted_final = True

                        if text_delta or thinking_delta or tool_call_delta is not None or is_final:
                            yield StreamChunk(
                                delta=text_delta,
                                thinking_delta=thinking_delta,
                                tool_call_delta=tool_call_delta,
                                is_final=is_final,
                                finish_reason=finish_reason,
                            )
                        continue

                    event_type = str(data.get("type", ""))
                    if event_type == "response.output_text.delta":
                        delta = data.get("delta")
                        if isinstance(delta, str) and delta:
                            yield StreamChunk(delta=delta)
                        continue

                    if event_type in {
                        "response.reasoning.delta",
                        "response.reasoning_text.delta",
                        "response.reasoning_summary_text.delta",
                    }:
                        delta = data.get("delta")
                        if isinstance(delta, str) and delta:
                            yield StreamChunk(thinking_delta=delta)
                        continue

                    if event_type in {"response.completed", "response.incomplete", "response.failed"}:
                        emitted_final = True
                        finish_reason = "stop" if event_type == "response.completed" else event_type
                        yield StreamChunk(is_final=True, finish_reason=finish_reason)
                        break
        except httpx.HTTPStatusError as exc:
            response_text = await self._read_response_text(exc.response)
            logger.error(
                f"OpenAI-compatible responses stream API error {exc.response.status_code}: {response_text}"
            )
            retry_payload = self._build_responses_reasoning_retry_payload(exc, payload)
            if retry_payload is not None and allow_reasoning_retry:
                async for chunk in self._stream_responses_generate(
                    retry_payload,
                    request_url=request_url,
                    allow_reasoning_retry=False,
                ):
                    yield chunk
                return
            chat_payload = self._build_chat_completions_payload(
                self._prepare_messages_from_responses_payload(payload),
                self._extract_chat_tools_from_responses_payload(payload),
                temperature=self._extract_float(payload.get("temperature"), 0.7),
                max_tokens=self._extract_int(payload.get("max_output_tokens"), 4096),
                top_p=self._extract_float(payload.get("top_p"), 1.0),
                frequency_penalty=self._extract_float(payload.get("frequency_penalty"), 0.0),
                presence_penalty=self._extract_float(payload.get("presence_penalty"), 0.0),
                reasoning_effort=self._extract_reasoning_effort_from_responses_payload(payload),
                stream=True,
            )
            alias_result = await self._retry_chat_completions_after_responses_error(
                exc,
                chat_payload,
                responses_url=request_url,
                chat_url=self._chat_completions_url,
                stage="stream_generate",
            )
            if alias_result is not None:
                if alias_result.content:
                    yield StreamChunk(delta=alias_result.content)
                for index, tool_call in enumerate(alias_result.tool_calls):
                    yield StreamChunk(
                        tool_call_delta=ToolCall(
                            id=tool_call.id,
                            name=tool_call.name,
                            arguments=tool_call.arguments,
                            index=tool_call.index if tool_call.index is not None else index,
                        )
                    )
                yield StreamChunk(
                    is_final=True,
                    finish_reason="tool_calls" if alias_result.tool_calls else alias_result.finish_reason,
                )
                return
            raise self._wrap_request_error(exc, stage="stream_generate", request_url=request_url) from exc
        except httpx.RequestError as exc:
            logger.error(f"OpenAI-compatible responses stream connection error: {exc}")
            raise self._wrap_request_error(exc, stage="stream_generate", request_url=request_url) from exc

        if not emitted_final:
            yield StreamChunk(is_final=True, finish_reason="stop")

    def _build_chat_completions_payload(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        temperature: float,
        max_tokens: int,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
        reasoning_effort: str | None,
        stream: bool,
    ) -> dict[str, Any]:
        """建立 Chat Completions payload。"""
        capabilities = resolve_model_inference_capabilities(self.get_model_info())
        supported = set(capabilities.supported_inference_parameters)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [m.to_dict() for m in messages],
            "stream": stream,
        }
        if "temperature" in supported:
            payload["temperature"] = temperature
        if "max_tokens" in supported:
            payload["max_tokens"] = max_tokens
        if "top_p" in supported:
            payload["top_p"] = top_p
        if "frequency_penalty" in supported:
            payload["frequency_penalty"] = frequency_penalty
        if "presence_penalty" in supported:
            payload["presence_penalty"] = presence_penalty
        if tools and self._tool_calling_enabled:
            payload["tools"] = [t.to_dict() for t in tools]
            payload["tool_choice"] = "auto"
        if (
            "reasoning_effort" in supported
            and reasoning_effort in capabilities.supported_reasoning_efforts
        ):
            payload["reasoning_effort"] = cast(ReasoningEffort, reasoning_effort)
        return payload

    def _build_responses_payload(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        temperature: float,
        max_tokens: int,
        top_p: float,
        frequency_penalty: float,
        presence_penalty: float,
        reasoning_effort: str | None,
        stream: bool,
    ) -> dict[str, Any]:
        """建立 Responses API payload。"""
        instructions = "\n\n".join(m.content for m in messages if m.role == "system" and m.content)
        capabilities = resolve_model_inference_capabilities(self.get_model_info())
        supported = set(capabilities.supported_inference_parameters)
        input_state = self._build_responses_input(messages)
        self._responses_last_continuity_mode = input_state.continuity_mode
        self._responses_last_replayed_items = input_state.replayed_items
        payload: dict[str, Any] = {
            "model": self.model,
            "input": input_state.input_items,
            "stream": stream,
        }
        if input_state.previous_response_id:
            payload["previous_response_id"] = input_state.previous_response_id
        if "temperature" in supported:
            payload["temperature"] = temperature
        if "max_tokens" in supported:
            payload["max_output_tokens"] = max_tokens
        if "top_p" in supported:
            payload["top_p"] = top_p
        if "frequency_penalty" in supported:
            payload["frequency_penalty"] = frequency_penalty
        if "presence_penalty" in supported:
            payload["presence_penalty"] = presence_penalty
        if instructions:
            payload["instructions"] = instructions
        if tools and self._tool_calling_enabled:
            payload["tools"] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                }
                for tool in tools
            ]
        reasoning_payload: dict[str, Any] = {}
        if (
            "reasoning_effort" in supported
            and reasoning_effort in capabilities.supported_reasoning_efforts
        ):
            reasoning_payload["effort"] = cast(ReasoningEffort, reasoning_effort)
        if self._should_request_reasoning_summary():
            reasoning_payload["summary"] = "concise"
        if reasoning_payload:
            payload["reasoning"] = reasoning_payload
        if self._should_include_encrypted_reasoning_content():
            payload["include"] = ["reasoning.encrypted_content"]
        return payload

    def _build_responses_input(self, messages: list[Message]) -> _ResponsesInputState:
        """將內部 message history 轉成 Responses API input items。"""
        previous_response_id, anchor_index = self._resolve_previous_response_id_anchor(messages)
        if previous_response_id is not None and anchor_index is not None:
            return _ResponsesInputState(
                input_items=self._build_responses_input_items(messages[anchor_index + 1 :]),
                previous_response_id=previous_response_id,
                continuity_mode="previous_response_id",
                replayed_items=0,
            )

        input_items: list[dict[str, Any]] = []
        continuity_mode: ResponsesContinuityMode = "degraded"
        replayed_items = 0
        for message in messages:
            if message.role == "system":
                continue

            replay = message.responses_replay if message.role == "assistant" else None
            if replay is not None and replay.assistant_output_items:
                input_items.extend(copy.deepcopy(replay.assistant_output_items))
                replayed_items += len(replay.assistant_output_items)
                if replay.encrypted_reasoning_content:
                    continuity_mode = "manual_encrypted"
                elif continuity_mode != "manual_encrypted":
                    continuity_mode = "manual_items"
                continue

            input_items.extend(self._build_responses_input_items([message]))

        if replayed_items == 0:
            continuity_mode = "degraded"

        return _ResponsesInputState(
            input_items=input_items,
            previous_response_id=None,
            continuity_mode=continuity_mode,
            replayed_items=replayed_items,
        )

    def _build_responses_input_items(self, messages: list[Message]) -> list[dict[str, Any]]:
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

    def _resolve_previous_response_id_anchor(
        self,
        messages: list[Message],
    ) -> tuple[str | None, int | None]:
        if not self._should_use_responses_transport():
            return None, None
        for index in range(len(messages) - 1, -1, -1):
            message = messages[index]
            if message.role != "assistant" or message.responses_replay is None:
                continue
            response_id = message.responses_replay.response_id
            if isinstance(response_id, str) and response_id:
                return response_id, index
        return None, None

    def _parse_responses_result(self, data: dict[str, Any]) -> GenerationResult:
        """解析 Responses API 回應。"""
        content = _coerce_text(data.get("output_text"))
        output = data.get("output")
        tool_calls = self._parse_responses_tool_calls(output)
        thinking = self._parse_responses_reasoning(output)
        replay_state = self._build_responses_replay_state(data, thinking)
        self._responses_reasoning_summary_received = bool(replay_state and replay_state.summary_text)
        if self._responses_reasoning_summary_received and self._responses_reasoning_summary_supported is None:
            self._responses_reasoning_summary_supported = True
        if replay_state and replay_state.encrypted_reasoning_content and self._responses_encrypted_include_supported is None:
            self._responses_encrypted_include_supported = True
        if not content and isinstance(output, list):
            output_items = cast(list[Any], output)
            for raw_item in output_items:
                if not isinstance(raw_item, dict):
                    continue
                item = cast(dict[str, Any], raw_item)
                content = self._extract_responses_output_text(item)
                if content:
                    break

        usage = data.get("usage", {})
        finish_reason = data.get("finish_reason")
        if not isinstance(finish_reason, str) or not finish_reason:
            choices = data.get("choices", [])
            if isinstance(choices, list) and choices:
                choice0 = choices[0]
                if isinstance(choice0, dict):
                    raw_finish_reason = choice0.get("finish_reason")
                    if isinstance(raw_finish_reason, str) and raw_finish_reason:
                        finish_reason = raw_finish_reason
            if not isinstance(finish_reason, str) or not finish_reason:
                finish_reason = "tool_calls" if tool_calls else "stop"

        return GenerationResult(
            content=content,
            thinking=thinking,
            tool_calls=tool_calls,
            input_tokens=self._as_int(usage.get("input_tokens") or usage.get("prompt_tokens")),
            output_tokens=self._as_int(usage.get("output_tokens") or usage.get("completion_tokens")),
            model=data.get("model", self.model),
            finish_reason=finish_reason,
            responses_replay=replay_state,
        )

    def _build_responses_replay_state(
        self,
        data: dict[str, Any],
        summary_text: str,
    ) -> ResponsesReplayState | None:
        output = data.get("output")
        if not isinstance(output, list):
            return None

        assistant_output_items: list[dict[str, Any]] = []
        encrypted_reasoning_content: str | None = None
        for raw_item in output:
            if not isinstance(raw_item, dict):
                continue
            item = cast(dict[str, Any], raw_item)
            if not self._is_replay_safe_responses_output_item(item):
                continue
            assistant_output_items.append(copy.deepcopy(item))
            if encrypted_reasoning_content is None and item.get("type") == "reasoning":
                candidate = item.get("encrypted_content")
                if isinstance(candidate, str) and candidate:
                    encrypted_reasoning_content = candidate

        if not assistant_output_items and not summary_text and not encrypted_reasoning_content:
            return None

        return ResponsesReplayState(
            response_id=data.get("id") if isinstance(data.get("id"), str) and data.get("id") else None,
            assistant_output_items=assistant_output_items,
            encrypted_reasoning_content=encrypted_reasoning_content,
            summary_text=summary_text,
            continuity_mode=(
                "manual_encrypted"
                if assistant_output_items and encrypted_reasoning_content
                else "manual_items"
                if assistant_output_items
                else "none"
            ),
        )

    def _is_replay_safe_responses_output_item(self, item: dict[str, Any]) -> bool:
        item_type = item.get("type")
        if item_type in {"reasoning", "function_call", "message"}:
            return True
        return item.get("role") == "assistant"

    def _extract_responses_output_text(self, item: dict[str, Any]) -> str:
        content = item.get("content")
        if isinstance(content, list):
            parts: list[str] = []
            for raw_part in content:
                if not isinstance(raw_part, dict):
                    text = _coerce_text(raw_part)
                else:
                    part = cast(dict[str, Any], raw_part)
                    text = _coerce_text(
                        part.get("text")
                        or part.get("output_text")
                        or part.get("refusal")
                        or part.get("content")
                    )
                if text:
                    parts.append(text)
            return "\n".join(parts)
        return _coerce_text(content)

    def _parse_responses_reasoning(self, output: Any) -> str:
        if not isinstance(output, list):
            return ""
        parts: list[str] = []
        for raw_item in output:
            if not isinstance(raw_item, dict):
                continue
            item = cast(dict[str, Any], raw_item)
            if item.get("type") != "reasoning":
                continue
            summary = item.get("summary")
            if isinstance(summary, list):
                for raw_summary in summary:
                    if isinstance(raw_summary, dict):
                        text = _coerce_text(raw_summary.get("text") or raw_summary.get("content"))
                    else:
                        text = _coerce_text(raw_summary)
                    if text:
                        parts.append(text)
            else:
                text = _coerce_text(item.get("text") or item.get("content"))
                if text:
                    parts.append(text)
        return "\n\n".join(parts)

    def _build_headers(self) -> dict[str, str]:
        """建立 HTTP 標頭。"""
        headers = {"Accept": "application/json", "Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _should_request_reasoning_summary(self) -> bool:
        return self._should_use_responses_transport() and self._responses_reasoning_summary_supported is not False

    def _should_include_encrypted_reasoning_content(self) -> bool:
        return self._should_use_responses_transport() and self._responses_encrypted_include_supported is not False

    def _build_responses_reasoning_retry_payload(
        self,
        exc: httpx.HTTPStatusError,
        payload: dict[str, Any],
    ) -> dict[str, Any] | None:
        if "input" not in payload:
            return None

        response_text = self._safe_response_text(exc.response).lower()
        retry_payload = copy.deepcopy(payload)
        removed_fields: list[str] = []

        reasoning_payload = retry_payload.get("reasoning")
        if isinstance(reasoning_payload, dict) and "summary" in reasoning_payload and any(
            marker in response_text
            for marker in (
                "reasoning.summary",
                '"summary"',
                "unknown field summary",
                "unsupported field summary",
            )
        ):
            reasoning_payload.pop("summary", None)
            if not reasoning_payload:
                retry_payload.pop("reasoning", None)
            removed_fields.append("reasoning.summary")
            self._responses_reasoning_summary_supported = False

        include_payload = retry_payload.get("include")
        if isinstance(include_payload, list) and any(
            marker in response_text
            for marker in (
                "reasoning.encrypted_content",
                "encrypted_content",
                '"include"',
                "unsupported include",
            )
        ):
            filtered_include = [item for item in include_payload if item != "reasoning.encrypted_content"]
            if filtered_include:
                retry_payload["include"] = filtered_include
            else:
                retry_payload.pop("include", None)
            removed_fields.append("include.reasoning.encrypted_content")
            self._responses_encrypted_include_supported = False

        if not removed_fields:
            return None

        append_fallback_diagnostic(
            self._fallback_diagnostics,
            category="reasoning",
            name="responses_reasoning_retry",
            reason="provider_rejected_responses_reasoning_fields",
            kind="retry",
            severity="info",
            metadata={
                "provider": self.provider,
                "model": self.model,
                "http_status": exc.response.status_code,
                "removed_fields": removed_fields,
            },
        )
        return retry_payload

    def _should_fallback_responses_http_error_to_chat(
        self,
        exc: httpx.HTTPStatusError,
    ) -> bool:
        if self._responses_chat_completions_alias:
            return False
        status_code = exc.response.status_code
        if status_code not in {400, 404, 405, 415, 422}:
            return False
        response_text = self._safe_response_text(exc.response).lower()
        if any(
            marker in response_text
            for marker in (
                "reasoning.summary",
                "reasoning.encrypted_content",
                "encrypted_content",
            )
        ):
            return False
        return any(
            marker in response_text
            for marker in (
                "/v1/responses",
                "responses",
                "chat/completions",
                "chat completions",
                "messages",
                "unknown field",
                "unrecognized field",
                "unsupported field",
                "invalid input",
                "invalid request body",
                "bad request",
                "not found",
                "method not allowed",
                "unsupported media type",
                "unprocessable",
            )
        )

    def _prepare_messages_from_responses_payload(self, payload: dict[str, Any]) -> list[Message]:
        messages: list[Message] = []
        instructions = payload.get("instructions")
        if isinstance(instructions, str) and instructions:
            messages.append(Message(role="system", content=instructions))

        input_items = payload.get("input")
        if not isinstance(input_items, list):
            return messages

        for raw_item in input_items:
            if not isinstance(raw_item, dict):
                continue
            item = cast(dict[str, Any], raw_item)
            item_type = item.get("type")
            if item_type == "function_call":
                messages.append(
                    Message(
                        role="assistant",
                        content="",
                        tool_calls=[
                            ToolCall(
                                id=str(item.get("call_id") or item.get("id") or ""),
                                name=_coerce_text(item.get("name")),
                                arguments=self._load_tool_arguments(item.get("arguments")),
                            )
                        ],
                    )
                )
                continue
            if item_type == "function_call_output":
                messages.append(
                    Message(
                        role="tool",
                        content=_coerce_text(item.get("output")),
                        tool_call_id=_coerce_text(item.get("call_id")) or None,
                    )
                )
                continue
            role = item.get("role")
            if role == "assistant":
                messages.append(Message(role="assistant", content=_coerce_text(item.get("content"))))
            elif role == "user":
                messages.append(Message(role="user", content=_coerce_text(item.get("content"))))

        return messages

    def _extract_chat_tools_from_responses_payload(
        self,
        payload: dict[str, Any],
    ) -> list[ToolSchema] | None:
        raw_tools = payload.get("tools")
        if not isinstance(raw_tools, list):
            return None

        tools: list[ToolSchema] = []
        for raw_tool in raw_tools:
            if not isinstance(raw_tool, dict):
                continue
            tool = cast(dict[str, Any], raw_tool)
            if tool.get("type") != "function":
                continue
            name = _coerce_text(tool.get("name"))
            if not name:
                continue
            parameters = tool.get("parameters")
            tools.append(
                ToolSchema(
                    name=name,
                    description=_coerce_text(tool.get("description")),
                    parameters=parameters if isinstance(parameters, dict) else {},
                )
            )
        return tools or None

    def _extract_reasoning_effort_from_responses_payload(self, payload: dict[str, Any]) -> str | None:
        reasoning = payload.get("reasoning")
        if not isinstance(reasoning, dict):
            return None
        effort = reasoning.get("effort")
        return effort if isinstance(effort, str) and effort else None

    def _extract_float(self, value: Any, default: float) -> float:
        if isinstance(value, int | float):
            return float(value)
        return default

    def _extract_int(self, value: Any, default: int) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        return default

    def _load_tool_arguments(self, value: Any) -> dict[str, Any]:
        if isinstance(value, dict):
            return value
        if isinstance(value, str) and value:
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return {}
            return parsed if isinstance(parsed, dict) else {}
        return {}

    def _wrap_request_error(
        self,
        exc: httpx.HTTPError,
        *,
        stage: str,
        request_url: str | None = None,
    ) -> BackendRequestError:
        metadata: dict[str, Any] = {
            "backend_name": "openai_compat",
            "api_mode": self._api_mode,
            "request_url": request_url or self._request_url,
            "stage": stage,
            "model": self.model,
        }
        if isinstance(exc, httpx.HTTPStatusError):
            metadata["status_code"] = exc.response.status_code
            metadata["response_text"] = self._safe_response_text(exc.response)
        return BackendRequestError(str(exc), metadata=metadata)

    async def _read_response_text(self, response: httpx.Response) -> str:
        """Read an error body safely, including inside httpx streaming responses."""
        try:
            return response.text
        except httpx.ResponseNotRead:
            try:
                await response.aread()
            except Exception as exc:  # pragma: no cover - defensive diagnostics only
                return f"<response body unavailable: {exc}>"
            return self._safe_response_text(response)
        except Exception as exc:  # pragma: no cover - defensive diagnostics only
            return f"<response body unavailable: {exc}>"

    async def _read_stream_error_body_before_raise(self, response: httpx.Response) -> None:
        if not self._is_http_error_response(response):
            return
        try:
            await response.aread()
        except Exception:
            return

    @staticmethod
    def _is_http_error_response(response: httpx.Response) -> bool:
        status_code = getattr(response, "status_code", 0)
        return isinstance(status_code, int) and status_code >= 400

    def _safe_response_text(self, response: httpx.Response) -> str:
        try:
            return response.text
        except httpx.ResponseNotRead:
            return "<streaming response body was not read>"
        except Exception as exc:  # pragma: no cover - defensive diagnostics only
            return f"<response body unavailable: {exc}>"

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
            index=index,
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

    def _has_generation_output(self, result: GenerationResult) -> bool:
        return bool(result.tool_calls or result.content.strip() or result.thinking.strip())

    def _as_int(self, value: Any) -> int:
        """將任意值轉成 int，失敗則回傳 0。"""
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    async def close(self) -> None:
        """關閉 HTTP client 連線。"""
        await self._client.aclose()

    def _should_retry_without_tools(
        self,
        exc: httpx.HTTPStatusError,
        payload: dict[str, Any],
    ) -> bool:
        return self._native_tool_fallback_for_error(exc, payload) is not None

    def _native_tool_fallback_for_error(
        self,
        exc: httpx.HTTPStatusError,
        payload: dict[str, Any],
    ) -> tuple[str, str] | None:
        if not payload.get("tools"):
            return None
        response_text = self._safe_response_text(exc.response).lower()
        if self.provider == "vllm" and exc.response.status_code == 400 and (
            "tool choice requires --enable-auto-tool-choice" in response_text
            or "--tool-call-parser" in response_text
        ):
            return (
                "rejected_missing_parser",
                "vLLM rejected native auto tool choice. Enable --enable-auto-tool-choice and --tool-call-parser.",
            )
        if exc.response.status_code not in {403, 429}:
            return None
        if not any(
            marker in response_text
            for marker in (
                "insufficient_quota",
                "permission_error",
                "tool",
                "function",
            )
        ):
            return None
        return (
            "native_tools_rejected_by_provider",
            "Provider rejected native tool-calling for this request; falling back to prompt-simulated tool calls.",
        )

    def _build_simulated_tool_retry_payload(self, payload: dict[str, Any]) -> dict[str, Any]:
        retry_payload = dict(payload)
        raw_tools = retry_payload.pop("tools", None)
        retry_payload.pop("tool_choice", None)
        tool_schemas = self._tool_schemas_from_payload(raw_tools)
        if not tool_schemas:
            return retry_payload

        if isinstance(retry_payload.get("messages"), list):
            retry_payload["messages"] = self._inject_simulated_tools_into_chat_messages(
                retry_payload["messages"],
                tool_schemas,
            )
        elif isinstance(retry_payload.get("input"), list):
            instructions = _coerce_text(retry_payload.get("instructions"))
            retry_payload["instructions"] = self._tool_call_simulator.inject_tools_into_prompt(
                instructions,
                tool_schemas,
            )
        return retry_payload

    def _inject_simulated_tools_into_chat_messages(
        self,
        raw_messages: list[Any],
        tools: list[ToolSchema],
    ) -> list[Any]:
        messages = [dict(item) if isinstance(item, dict) else item for item in raw_messages]
        for message in messages:
            if not isinstance(message, dict) or message.get("role") != "system":
                continue
            message["content"] = self._tool_call_simulator.inject_tools_into_prompt(
                _coerce_text(message.get("content")),
                tools,
            )
            return messages

        return [
            {
                "role": "system",
                "content": self._tool_call_simulator.inject_tools_into_prompt("", tools).strip(),
            },
            *messages,
        ]

    @staticmethod
    def _tool_schemas_from_payload(raw_tools: Any) -> list[ToolSchema]:
        if not isinstance(raw_tools, list):
            return []
        schemas: list[ToolSchema] = []
        for item in raw_tools:
            if not isinstance(item, dict):
                continue
            function = item.get("function")
            if isinstance(function, dict):
                name = function.get("name")
                description = function.get("description")
                parameters = function.get("parameters")
            else:
                name = item.get("name")
                description = item.get("description")
                parameters = item.get("parameters")
            if not isinstance(name, str) or not name.strip():
                continue
            schemas.append(
                ToolSchema(
                    name=name.strip(),
                    description=description if isinstance(description, str) else "",
                    parameters=parameters if isinstance(parameters, dict) else {},
                )
            )
        return schemas


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
            chat_completions_url=input_url,
            responses_url=f"{models_base}/responses",
            api_mode="chat_completions",
            health_urls=(f"{models_base}/models",),
            transport_locked=True,
        )

    if normalized_path.endswith("/responses"):
        models_base = _url_without_path_suffix(parts, "/responses")
        return _OpenAICompatEndpoint(
            input_url=input_url,
            request_url=input_url,
            chat_completions_url=f"{models_base}/chat/completions",
            responses_url=input_url,
            api_mode="responses",
            health_urls=(f"{models_base}/models",),
            transport_locked=True,
        )

    if normalized_path.endswith("/v1"):
        return _OpenAICompatEndpoint(
            input_url=input_url,
            request_url=f"{input_url}/chat/completions",
            chat_completions_url=f"{input_url}/chat/completions",
            responses_url=f"{input_url}/responses",
            api_mode="chat_completions",
            health_urls=(f"{input_url}/models",),
            transport_locked=False,
        )

    return _OpenAICompatEndpoint(
        input_url=input_url,
        request_url=f"{input_url}/chat/completions",
        chat_completions_url=f"{input_url}/chat/completions",
        responses_url=f"{input_url}/responses",
        api_mode="chat_completions",
        health_urls=(f"{input_url}/models", f"{input_url}/v1/models"),
        transport_locked=False,
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


def _looks_like_chat_completions_response(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    return isinstance(value.get("choices"), list)
