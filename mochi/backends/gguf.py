"""GGUF（llama-cpp-python）後端。"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any, Literal, cast

from mochi.backends.base import BaseLLMBackend
from mochi.backends.tool_call_simulator import ToolCallSimulator
from mochi.backends.types import (
    GenerationResult,
    Message,
    ModelInfo,
    StreamChunk,
    ToolCall,
    ToolSchema,
)

GGUFModelLoader = Callable[[], Any]
_STREAM_END = object()
_ToolCallMode = Literal["plain_chat", "structured_native", "flattened_text"]
_TOOL_AWARE_CHAT_FORMAT_ALLOWLIST = frozenset(
    {
        "functionary",
        "functionary-v1",
        "functionary-v2",
        "chatml-function-calling",
    }
)


class GGUFBackend(BaseLLMBackend):
    """GGUF 本地模型後端（llama-cpp-python）。"""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        n_threads: int | None = None,
        n_batch: int = 512,
        n_ubatch: int = 512,
        n_threads_batch: int | None = None,
        flash_attn: bool = False,
        offload_kqv: bool = True,
        use_mmap: bool = True,
        use_mlock: bool = False,
        llama_cpp_lib_path: str | None = None,
        *,
        model_loader: GGUFModelLoader | None = None,
        tool_call_simulator: ToolCallSimulator | None = None,
    ) -> None:
        """初始化 GGUF 後端。

        Args:
            model_path: GGUF 模型檔案路徑。
            n_ctx: 上下文長度。
            n_gpu_layers: GPU 層數。
            n_threads: CPU 執行緒數。
            model_loader: 可注入模型載入器，便於測試或替換 runtime。
            tool_call_simulator: 工具呼叫模擬器。
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.n_threads = n_threads
        self.n_batch = n_batch
        self.n_ubatch = n_ubatch
        self.n_threads_batch = n_threads_batch
        self.flash_attn = flash_attn
        self.offload_kqv = offload_kqv
        self.use_mmap = use_mmap
        self.use_mlock = use_mlock
        self.llama_cpp_lib_path = llama_cpp_lib_path
        self._model_loader = model_loader
        self._tool_call_simulator = tool_call_simulator or ToolCallSimulator()
        self._model: Any | None = None
        self._dependency_error: str | None = self._probe_dependency_error()
        self._tool_call_strategy = "flattened_text"
        self._tool_call_strategy_reason = "conservative default: tools not requested yet"
        self._detected_chat_format = "unknown"

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
        """執行生成。"""
        if stream:
            return self._stream_generate(
                messages,
                tools,
                temperature,
                max_tokens,
                top_p,
                min_p,
                top_k,
                frequency_penalty,
                presence_penalty,
                repeat_penalty,
            )

        return await self._generate_nonstream(
            messages,
            tools,
            temperature,
            max_tokens,
            top_p,
            min_p,
            top_k,
            frequency_penalty,
            presence_penalty,
            repeat_penalty,
        )

    async def _generate_nonstream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        temperature: float,
        max_tokens: int,
        top_p: float,
        min_p: float,
        top_k: int,
        frequency_penalty: float,
        presence_penalty: float,
        repeat_penalty: float,
    ) -> GenerationResult:
        """執行 non-stream 生成。"""
        model = await self._ensure_model_loaded()
        tool_call_mode = self._resolve_tool_call_mode(tools, model)
        prepared_messages = self._prepare_messages(messages, tools, strategy=tool_call_mode)
        request_payload = self._build_request_payload(
            messages=prepared_messages,
            tools=tools,
            strategy=tool_call_mode,
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
        raw_result = await asyncio.to_thread(
            model.create_chat_completion,
            **request_payload,
        )
        return self._parse_generation_result(
            raw_result=raw_result,
            strategy=tool_call_mode,
            prepared_messages=prepared_messages,
            model=model,
            max_tokens=max_tokens,
        )

    async def _stream_generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        temperature: float,
        max_tokens: int,
        top_p: float,
        min_p: float,
        top_k: int,
        frequency_penalty: float,
        presence_penalty: float,
        repeat_penalty: float,
    ) -> AsyncIterator[StreamChunk]:
        """執行最小可用 stream 生成。"""
        model = await self._ensure_model_loaded()
        tool_call_mode = self._resolve_tool_call_mode(tools, model)
        prepared_messages = self._prepare_messages(messages, tools, strategy=tool_call_mode)
        request_payload = self._build_request_payload(
            messages=prepared_messages,
            tools=tools,
            strategy=tool_call_mode,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            min_p=min_p,
            top_k=top_k,
            frequency_penalty=frequency_penalty,
            presence_penalty=presence_penalty,
            repeat_penalty=repeat_penalty,
            stream=True,
        )
        iterator = await asyncio.to_thread(
            model.create_chat_completion,
            **request_payload,
        )

        emitted_final = False
        while True:
            item = await asyncio.to_thread(self._next_stream_item, iterator)
            if item is _STREAM_END:
                break

            delta, finish_reason = self._parse_stream_item(item)
            if delta or finish_reason is not None:
                emitted_final = emitted_final or finish_reason is not None
                yield StreamChunk(
                    delta=delta,
                    is_final=finish_reason is not None,
                    finish_reason=finish_reason,
                )

        if not emitted_final:
            yield StreamChunk(is_final=True, finish_reason="stop")

    def supports_tool_calling(self) -> bool:
        """回報 GGUF 後端不支援原生 tool calling。"""
        return False

    def get_model_info(self) -> ModelInfo:
        """回傳 GGUF 後端模型資訊。"""
        return ModelInfo(
            name=self.model_path,
            backend_type="gguf",
            provider="local",
            context_length=self.n_ctx,
            supports_tool_calling=False,
            metadata={
                "model_path": self.model_path,
                "dependency_ready": self._dependency_error is None,
                "dependency_error": self._dependency_error,
                "loaded": self._model is not None,
                "n_gpu_layers": self.n_gpu_layers,
                "n_threads": self.n_threads,
                "n_batch": self.n_batch,
                "n_ubatch": self.n_ubatch,
                "n_threads_batch": self.n_threads_batch,
                "flash_attn": self.flash_attn,
                "offload_kqv": self.offload_kqv,
                "use_mmap": self.use_mmap,
                "use_mlock": self.use_mlock,
                "llama_cpp_lib_path": self.llama_cpp_lib_path,
                "tool_call_strategy": self._tool_call_strategy,
                "tool_call_strategy_reason": self._tool_call_strategy_reason,
                "detected_chat_format": self._detected_chat_format,
            },
        )

    async def health_check(self) -> bool:
        """檢查依賴與模型檔案是否可用。"""
        return self._dependency_error is None and Path(self.model_path).is_file()

    async def close(self) -> None:
        """釋放後端資源。"""
        self._model = None

    async def _ensure_model_loaded(self) -> Any:
        """確保模型已載入。"""
        if self._dependency_error is not None:
            raise self._build_generate_error("dependency_missing", self._dependency_error)

        model_file = Path(self.model_path)
        if not model_file.is_file():
            raise self._build_generate_error(
                "model_path_missing",
                f"GGUF model file not found: {self.model_path}",
            )

        if self._model is not None:
            return self._model

        loader = self._model_loader or self._default_model_loader
        try:
            self._model = await asyncio.to_thread(loader)
        except Exception as exc:
            raise self._build_generate_error(
                "model_load_failed",
                str(exc),
            ) from exc
        return self._model

    def _default_model_loader(self) -> Any:
        """使用 llama-cpp-python 載入模型。"""
        Llama = self._load_llama_class()

        kwargs: dict[str, Any] = {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "n_batch": self.n_batch,
            "n_ubatch": self.n_ubatch,
            "flash_attn": self.flash_attn,
            "offload_kqv": self.offload_kqv,
            "use_mmap": self.use_mmap,
            "use_mlock": self.use_mlock,
            "verbose": False,
        }
        if self.n_threads is not None:
            kwargs["n_threads"] = self.n_threads
        if self.n_threads_batch is not None:
            kwargs["n_threads_batch"] = self.n_threads_batch
        return Llama(**kwargs)

    def _load_llama_class(self) -> Any:
        """Load the installed llama-cpp-python runtime for GGUF inference."""
        from llama_cpp import Llama  # type: ignore[import-not-found]
        from llama_cpp import llama_cpp as lib  # type: ignore[import-not-found]

        supports_gpu_offload = getattr(lib, "llama_supports_gpu_offload", None)
        if (
            self.n_gpu_layers != 0
            and callable(supports_gpu_offload)
            and supports_gpu_offload() is False
        ):
            loaded_base = getattr(lib, "_base_path", None)
            configured_path = str(loaded_base) if loaded_base is not None else "unknown"
            raise RuntimeError(
                "Installed llama-cpp-python runtime does not support GPU offload. "
                "GGUF inference must use a CUDA/HIP-capable llama-cpp-python build that matches "
                f"its bundled llama runtime. Loaded library path: {configured_path}"
            )
        return Llama

    def _prepare_messages(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        *,
        strategy: _ToolCallMode,
    ) -> list[Message]:
        """必要時將工具說明注入 system prompt。"""
        prepared = [Message(**message.__dict__) for message in messages]
        if strategy in {"plain_chat", "structured_native"} or not tools:
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

    def _build_request_payload(
        self,
        *,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        strategy: _ToolCallMode,
        temperature: float,
        max_tokens: int,
        top_p: float,
        min_p: float,
        top_k: int,
        frequency_penalty: float,
        presence_penalty: float,
        repeat_penalty: float,
        stream: bool,
    ) -> dict[str, Any]:
        """Build llama-cpp chat-completion payload for the selected tool-call mode."""
        payload: dict[str, Any] = {
            "messages": [message.to_dict() for message in messages],
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
        if strategy == "structured_native" and tools:
            payload["tools"] = [tool.to_dict() for tool in tools]
        return payload

    def _resolve_tool_call_mode(self, tools: list[ToolSchema] | None, model: Any | None) -> _ToolCallMode:
        """Resolve one tool-call mode per request and keep metadata consistent."""
        if not tools:
            self._tool_call_strategy = "plain_chat"
            self._tool_call_strategy_reason = "tools not requested"
            self._detected_chat_format = "unknown"
            return "plain_chat"
        return cast(_ToolCallMode, self._resolve_tool_call_strategy(model))

    def _flatten_simulated_tool_messages(self, messages: list[Message]) -> list[Message]:
        """將 OpenAI-style tool message 壓平成一般聊天內容，避免 llama.cpp template 失敗。"""
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
                flattened.append(
                    Message(
                        role="user",
                        content=tool_text,
                    )
                )
                continue

            flattened.append(Message(**message.__dict__))
        return flattened

    def _resolve_tool_call_strategy(self, model: Any | None) -> str:
        """Conservatively choose GGUF tool message strategy from runtime chat format."""
        detected_chat_format, detect_reason = self._detect_chat_format(model)
        self._detected_chat_format = detected_chat_format or "unknown"

        if detected_chat_format and detected_chat_format in _TOOL_AWARE_CHAT_FORMAT_ALLOWLIST:
            self._tool_call_strategy = "structured_native"
            self._tool_call_strategy_reason = (
                f"detected chat format '{detected_chat_format}' is in allowlist"
            )
            return self._tool_call_strategy

        self._tool_call_strategy = "flattened_text"
        if detected_chat_format:
            self._tool_call_strategy_reason = (
                f"detected chat format '{detected_chat_format}' not in allowlist"
            )
        else:
            self._tool_call_strategy_reason = detect_reason
        return self._tool_call_strategy

    def _detect_chat_format(self, model: Any | None) -> tuple[str | None, str]:
        """Best-effort chat format inspection with safe fallback."""
        if model is None:
            return None, "unable to inspect chat format (model unavailable)"

        candidate_sources: list[tuple[Any, str]] = [(model, "chat_format"), (model, "_chat_format")]
        try:
            chat_handler = getattr(model, "chat_handler", None)
        except Exception as exc:
            return None, f"unable to inspect chat format ({exc.__class__.__name__})"

        if chat_handler is not None:
            candidate_sources.extend(
                [
                    (chat_handler, "chat_format"),
                    (chat_handler, "name"),
                ]
            )

        for source, attr_name in candidate_sources:
            try:
                value = getattr(source, attr_name, None)
            except Exception as exc:
                return None, f"unable to inspect chat format ({exc.__class__.__name__})"
            if isinstance(value, str):
                normalized = value.strip().lower()
                if normalized:
                    return normalized, f"detected via {attr_name}"

        return None, "unable to inspect chat format"

    def _parse_generation_result(
        self,
        raw_result: Any,
        strategy: _ToolCallMode,
        prepared_messages: list[Message],
        model: Any,
        max_tokens: int,
    ) -> GenerationResult:
        """解析 llama-cpp-python 回傳結果。"""
        choices = raw_result.get("choices", []) if isinstance(raw_result, dict) else []
        first_choice = choices[0] if choices else {}
        message = first_choice.get("message", {}) if isinstance(first_choice, dict) else {}

        content = ""
        if isinstance(message, dict):
            content_value = message.get("content", "")
            if isinstance(content_value, str):
                content = content_value

        usage = raw_result.get("usage", {}) if isinstance(raw_result, dict) else {}
        finish_reason = first_choice.get("finish_reason", "stop") if isinstance(first_choice, dict) else "stop"
        tool_calls: list[ToolCall] = []
        input_text = "\n".join(message.content for message in prepared_messages)

        if strategy == "structured_native":
            raw_tool_calls = message.get("tool_calls", []) if isinstance(message, dict) else []
            tool_calls = self._parse_native_tool_calls(raw_tool_calls)
            if tool_calls:
                finish_reason = "tool_calls"
        elif strategy == "flattened_text":
            tool_calls = self._tool_call_simulator.parse_tool_calls(content)
            content = self._tool_call_simulator.extract_text_response(content)
            if tool_calls:
                finish_reason = "tool_calls"

        input_tokens = self._resolve_token_count(
            usage_value=usage.get("prompt_tokens"),
            text=input_text,
            model=model,
            upper_bound=self.n_ctx,
        )
        output_tokens = self._resolve_token_count(
            usage_value=usage.get("completion_tokens"),
            text=content,
            model=model,
            upper_bound=max_tokens,
        )

        return GenerationResult(
            content=content,
            tool_calls=tool_calls,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            model=raw_result.get("model", self.model_path) if isinstance(raw_result, dict) else self.model_path,
            finish_reason=finish_reason,
        )

    def _parse_native_tool_calls(self, raw_tool_calls: Any) -> list[ToolCall]:
        """Parse OpenAI-style assistant tool_calls from llama-cpp chat completion output."""
        if not isinstance(raw_tool_calls, list):
            return []

        parsed: list[ToolCall] = []
        raw_items = cast(list[Any], raw_tool_calls)
        for index, raw_item in enumerate(raw_items):
            if not isinstance(raw_item, dict):
                continue
            item = cast(dict[str, Any], raw_item)
            fn = item.get("function", {})
            if not isinstance(fn, dict):
                fn = {}
            fn = cast(dict[str, Any], fn)

            name = fn.get("name", "")
            if not isinstance(name, str):
                name = ""

            raw_args = fn.get("arguments", {})
            arguments: dict[str, Any]
            if isinstance(raw_args, str):
                try:
                    decoded = json.loads(raw_args) if raw_args else {}
                except json.JSONDecodeError:
                    decoded = {}
                arguments = cast(dict[str, Any], decoded) if isinstance(decoded, dict) else {}
            elif isinstance(raw_args, dict):
                arguments = cast(dict[str, Any], raw_args)
            else:
                arguments = {}

            call_id = item.get("id")
            if not isinstance(call_id, str) or not call_id:
                call_id = f"gguf-tool-call-{index + 1}"

            parsed.append(ToolCall(id=call_id, name=name, arguments=arguments))

        return parsed

    def _parse_stream_item(self, item: Any) -> tuple[str, str | None]:
        """解析單個 stream chunk。"""
        if not isinstance(item, dict):
            return "", None

        choices = item.get("choices", [])
        first_choice = choices[0] if isinstance(choices, list) and choices else {}
        if not isinstance(first_choice, dict):
            first_choice = {}

        delta = ""
        delta_obj = first_choice.get("delta", {})
        if isinstance(delta_obj, dict):
            content_value = delta_obj.get("content", "")
            if isinstance(content_value, str):
                delta = content_value

        if not delta:
            message_obj = first_choice.get("message", {})
            if isinstance(message_obj, dict):
                content_value = message_obj.get("content", "")
                if isinstance(content_value, str):
                    delta = content_value

        if not delta:
            text_value = first_choice.get("text", "")
            if isinstance(text_value, str):
                delta = text_value

        finish_reason = first_choice.get("finish_reason")
        if not isinstance(finish_reason, str):
            finish_reason = None
        return delta, finish_reason

    @staticmethod
    def _next_stream_item(iterator: Any) -> Any:
        """從同步 iterator 取出下一個 chunk。"""
        try:
            return next(iterator)
        except StopIteration:
            return _STREAM_END

    def _probe_dependency_error(self) -> str | None:
        """檢查 llama-cpp-python 是否可用。"""
        try:
            import llama_cpp  # type: ignore[import-not-found]  # noqa: F401
        except Exception as exc:
            return (
                "llama-cpp-python is not available. "
                f"Install with `uv sync --extra gguf`. ({exc.__class__.__name__}: {exc})"
            )
        return None

    def _build_generate_error(self, code: str, detail: str) -> RuntimeError:
        """建立一致的 generate 錯誤語義。"""
        return RuntimeError(f"gguf generate unavailable [{code}]: {detail}")

    def _as_int(self, value: Any) -> int:
        """將任意值轉為 int。"""
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _resolve_token_count(
        self,
        usage_value: Any,
        text: str,
        model: Any,
        upper_bound: int | None = None,
    ) -> int:
        """優先採用 runtime usage，缺失時回退至可重現估算。"""
        explicit_count = self._as_int(usage_value)
        if explicit_count > 0:
            return explicit_count

        runtime_count = self._count_tokens_with_runtime(model=model, text=text)
        if runtime_count is not None and runtime_count > 0:
            if upper_bound is None:
                return runtime_count
            return min(runtime_count, upper_bound)

        heuristic_count = self._estimate_tokens(text)
        if upper_bound is None:
            return heuristic_count
        return min(heuristic_count, upper_bound)

    def _count_tokens_with_runtime(self, model: Any, text: str) -> int | None:
        """若 runtime 提供 tokenize，優先使用其 token 計數。"""
        tokenize = getattr(model, "tokenize", None)
        if not callable(tokenize):
            return None

        payload = text.encode("utf-8")
        tokens: Any
        try:
            tokens = tokenize(payload, add_bos=False)
        except TypeError:
            try:
                tokens = tokenize(payload)
            except Exception:
                return None
        except Exception:
            return None

        try:
            return len(tokens)
        except TypeError:
            return None

    def _estimate_tokens(self, text: str) -> int:
        """小型、可測試且可重現的 token 啟發式估算。"""
        normalized = " ".join(text.split())
        if not normalized:
            return 0
        word_estimate = len(normalized.split(" "))
        char_estimate = (len(normalized) + 3) // 4
        return max(1, word_estimate, char_estimate)
