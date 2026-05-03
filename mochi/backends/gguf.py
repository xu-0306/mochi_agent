"""GGUF（llama-cpp-python）後端。"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

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


class GGUFBackend(BaseLLMBackend):
    """GGUF 本地模型後端（llama-cpp-python）。"""

    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1,
        n_threads: int | None = None,
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
        self._model_loader = model_loader
        self._tool_call_simulator = tool_call_simulator or ToolCallSimulator()
        self._model: Any | None = None
        self._dependency_error: str | None = self._probe_dependency_error()

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        stream: bool = False,
    ) -> GenerationResult | AsyncIterator[StreamChunk]:
        """執行生成。"""
        if stream:
            return self._stream_generate(messages, tools, temperature, max_tokens)

        return await self._generate_nonstream(messages, tools, temperature, max_tokens)

    async def _generate_nonstream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        temperature: float,
        max_tokens: int,
    ) -> GenerationResult:
        """執行 non-stream 生成。"""
        model = await self._ensure_model_loaded()
        prepared_messages = self._prepare_messages(messages, tools)
        raw_result = await asyncio.to_thread(
            model.create_chat_completion,
            messages=[message.to_dict() for message in prepared_messages],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=False,
        )
        return self._parse_generation_result(
            raw_result=raw_result,
            tools=tools,
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
    ) -> AsyncIterator[StreamChunk]:
        """執行最小可用 stream 生成。"""
        model = await self._ensure_model_loaded()
        prepared_messages = self._prepare_messages(messages, tools)
        iterator = await asyncio.to_thread(
            model.create_chat_completion,
            messages=[message.to_dict() for message in prepared_messages],
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
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
            context_length=self.n_ctx,
            supports_tool_calling=False,
            metadata={
                "model_path": self.model_path,
                "dependency_ready": self._dependency_error is None,
                "loaded": self._model is not None,
                "n_gpu_layers": self.n_gpu_layers,
                "n_threads": self.n_threads,
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
        from llama_cpp import Llama  # type: ignore[import-not-found]

        kwargs: dict[str, Any] = {
            "model_path": self.model_path,
            "n_ctx": self.n_ctx,
            "n_gpu_layers": self.n_gpu_layers,
            "verbose": False,
        }
        if self.n_threads is not None:
            kwargs["n_threads"] = self.n_threads
        return Llama(**kwargs)

    def _prepare_messages(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
    ) -> list[Message]:
        """必要時將工具說明注入 system prompt。"""
        if not tools:
            return list(messages)

        prepared = [Message(**message.__dict__) for message in messages]
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

    def _parse_generation_result(
        self,
        raw_result: Any,
        tools: list[ToolSchema] | None,
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

        if tools:
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
