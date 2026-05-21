"""Safetensors / transformers family 後端。"""

from __future__ import annotations

import asyncio
import gc
import time
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

try:
    from loguru import logger
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal test envs
    import logging

    logger = logging.getLogger(__name__)

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

PipelineFactory = Callable[[], Any]


class SafetensorsBackend(BaseLLMBackend):
    """HuggingFace transformers family 後端。"""

    def __init__(
        self,
        model_dir: str,
        device: str = "auto",
        torch_dtype: str = "auto",
        *,
        pipeline_factory: PipelineFactory | None = None,
        tool_call_simulator: ToolCallSimulator | None = None,
    ) -> None:
        """初始化 Safetensors 後端。

        Args:
            model_dir: 模型目錄路徑。
            device: 推理設備（auto / cpu / cuda）。
            torch_dtype: Torch 資料型別。
            pipeline_factory: 可注入 pipeline 建立器，便於測試或替換 runtime。
            tool_call_simulator: 工具呼叫模擬器。
        """
        self.model_dir = model_dir
        self.device = device
        self.torch_dtype = torch_dtype
        self._pipeline_factory = pipeline_factory
        self._tool_call_simulator = tool_call_simulator or ToolCallSimulator()
        self._pipeline: Any | None = None
        self._dependency_error: str | None = self._probe_dependency_error()

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
                top_k,
                repeat_penalty,
            )

        return await self._generate_nonstream(
            messages,
            tools,
            temperature,
            max_tokens,
            top_p,
            top_k,
            repeat_penalty,
        )

    async def _generate_nonstream(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        temperature: float,
        max_tokens: int,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
    ) -> GenerationResult:
        """執行 non-stream 生成。"""
        pipeline = await self._ensure_pipeline_loaded()
        prompt = self._build_prompt(messages, tools)
        raw_result = await asyncio.to_thread(
            pipeline,
            prompt,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repeat_penalty,
            do_sample=temperature > 0,
            return_full_text=True,
        )
        return self._parse_generation_result(raw_result, prompt, tools)

    async def _stream_generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
        temperature: float,
        max_tokens: int,
        top_p: float,
        top_k: int,
        repeat_penalty: float,
    ) -> AsyncIterator[StreamChunk]:
        """執行最小可用 stream 生成。

        transformers family 在 MVP 先使用 pseudo-stream：
        先完成 non-stream 生成，再以單次 delta + final chunk 輸出。
        """
        result = await self._generate_nonstream(
            messages,
            tools,
            temperature,
            max_tokens,
            top_p,
            top_k,
            repeat_penalty,
        )

        if result.content:
            yield StreamChunk(delta=result.content)
        yield StreamChunk(is_final=True, finish_reason=result.finish_reason)

    def supports_tool_calling(self) -> bool:
        """回報 transformers family 後端不支援原生 tool calling。"""
        return False

    def get_model_info(self) -> ModelInfo:
        """回傳後端模型資訊。"""
        return ModelInfo(
            name=self.model_dir,
            backend_type="safetensors",
            context_length=4096,
            supports_tool_calling=False,
            metadata={
                "model_dir": self.model_dir,
                "dependency_ready": self._dependency_error is None,
                "dependency_error": self._dependency_error,
                "loaded": self._pipeline is not None,
                "device": self.device,
                "torch_dtype": self.torch_dtype,
                "idle_unloaded": self._pipeline is None,
            },
        )

    async def health_check(self) -> bool:
        """檢查依賴與模型目錄是否可用。"""
        return self._dependency_error is None and Path(self.model_dir).is_dir()

    async def close(self) -> None:
        """釋放後端資源。"""
        pipeline = self._pipeline
        self._pipeline = None
        if pipeline is None:
            return

        try:
            model = getattr(pipeline, "model", None)
            if model is not None:
                for attr_name in ("cpu",):
                    method = getattr(model, attr_name, None)
                    if callable(method):
                        try:
                            method()
                        except Exception as exc:
                            logger.debug("Safetensors model {}() during close failed: {}", attr_name, exc)
                        break
        finally:
            try:
                del pipeline
            except Exception:
                pass

        gc.collect()
        self._release_cuda_cache()

    async def _ensure_pipeline_loaded(self) -> Any:
        """確保 pipeline 已載入。"""
        if self._dependency_error is not None:
            raise self._build_generate_error("dependency_missing", self._dependency_error)

        model_dir = Path(self.model_dir)
        if not model_dir.is_dir():
            raise self._build_generate_error(
                "model_dir_missing",
                f"Model directory not found: {self.model_dir}",
            )

        if self._pipeline is not None:
            return self._pipeline

        factory = self._pipeline_factory or self._default_pipeline_factory
        started_at = time.perf_counter()
        logger.info(
            "Loading safetensors pipeline for '{}' (device={}, torch_dtype={})",
            self.model_dir,
            self.device,
            self.torch_dtype,
        )
        self._log_runtime_environment()
        try:
            self._pipeline = await asyncio.to_thread(factory)
        except Exception as exc:
            raise self._build_generate_error("model_load_failed", str(exc)) from exc
        elapsed = time.perf_counter() - started_at
        logger.info(
            "Loaded safetensors pipeline for '{}' in {:.2f}s",
            self.model_dir,
            elapsed,
        )
        self._log_loaded_pipeline_details(self._pipeline)
        return self._pipeline

    def _default_pipeline_factory(self) -> Any:
        """建立 transformers text-generation pipeline。"""
        from transformers import pipeline  # type: ignore[import-not-found]

        kwargs: dict[str, Any] = {
            "task": "text-generation",
            "model": self.model_dir,
            "tokenizer": self.model_dir,
        }
        if self.device != "auto":
            kwargs["device"] = self.device
        else:
            kwargs["device_map"] = "auto"
        torch_dtype = self._resolve_torch_dtype()
        if torch_dtype is not None:
            kwargs["torch_dtype"] = torch_dtype
        logger.info(
            "Initializing transformers pipeline for '{}' with kwargs={}",
            self.model_dir,
            self._summarize_pipeline_kwargs(kwargs),
        )
        return pipeline(**kwargs)

    def _resolve_torch_dtype(self) -> Any | None:
        """將字串 torch_dtype 解析為 torch 型別。"""
        if self.torch_dtype == "auto":
            return None
        try:
            import torch  # type: ignore[import-not-found]
        except Exception:
            return None
        return getattr(torch, self.torch_dtype, None)

    def _build_prompt(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None,
    ) -> str:
        """將 chat messages 轉為 pipeline 可接受的 prompt。"""
        prepared = [Message(**message.__dict__) for message in messages]
        if tools:
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

        tokenizer = self._resolve_chat_template_source()
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            try:
                rendered = tokenizer.apply_chat_template(
                    [{"role": m.role, "content": m.content} for m in prepared],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                if isinstance(rendered, str):
                    return rendered
            except Exception:
                pass

        parts = [f"{message.role}: {message.content}" for message in prepared]
        parts.append("assistant:")
        return "\n".join(parts)

    def _resolve_chat_template_source(self) -> Any | None:
        """取得可用的 chat-template 來源。"""
        pipeline = self._pipeline
        if pipeline is None:
            return None

        tokenizer = getattr(pipeline, "tokenizer", None)
        if tokenizer is not None:
            return tokenizer

        processor = getattr(pipeline, "processor", None)
        if processor is None:
            return None

        processor_tokenizer = getattr(processor, "tokenizer", None)
        if processor_tokenizer is not None:
            return processor_tokenizer
        return processor

    def _parse_generation_result(
        self,
        raw_result: Any,
        prompt: str,
        tools: list[ToolSchema] | None,
    ) -> GenerationResult:
        """解析 transformers pipeline 回傳結果。"""
        content = self._extract_generated_text(raw_result, prompt)
        finish_reason = "stop"
        tool_calls: list[ToolCall] = []
        if tools:
            tool_calls = self._tool_call_simulator.parse_tool_calls(content)
            content = self._tool_call_simulator.extract_text_response(content)
            if tool_calls:
                finish_reason = "tool_calls"

        return GenerationResult(
            content=content,
            tool_calls=tool_calls,
            input_tokens=self._resolve_input_tokens(raw_result, prompt),
            output_tokens=self._resolve_output_tokens(raw_result, content),
            model=self.model_dir,
            finish_reason=finish_reason,
        )

    def _extract_generated_text(self, raw_result: Any, prompt: str) -> str:
        """從 pipeline 結果抽取新生成文字。"""
        text = ""
        if isinstance(raw_result, str):
            text = raw_result
        elif isinstance(raw_result, list) and raw_result:
            first = raw_result[0]
            if isinstance(first, dict):
                candidate = first.get("generated_text", first.get("text", ""))
                if isinstance(candidate, str):
                    text = candidate

        if text.startswith(prompt):
            return text[len(prompt):].lstrip()
        return text

    def _probe_dependency_error(self) -> str | None:
        """檢查 transformers / accelerate 是否可用。"""
        missing: list[str] = []
        for module_name in ("transformers", "accelerate"):
            try:
                __import__(module_name)
            except Exception:
                missing.append(module_name)

        if missing:
            pkg_list = ", ".join(missing)
            return f"Missing dependencies: {pkg_list}. Install with `uv sync --extra hf`."
        return None

    def _release_cuda_cache(self) -> None:
        """盡量主動釋放 CUDA 快取，降低閒置 VRAM 佔用。"""
        try:
            import torch  # type: ignore[import-not-found]
        except Exception:
            return

        try:
            if not torch.cuda.is_available():
                return
            torch.cuda.empty_cache()
            ipc_collect = getattr(torch.cuda, "ipc_collect", None)
            if callable(ipc_collect):
                ipc_collect()
            logger.info("Released safetensors CUDA cache for '{}'", self.model_dir)
        except Exception as exc:
            logger.debug("Safetensors CUDA cache release failed for '{}': {}", self.model_dir, exc)

    def _build_generate_error(self, code: str, detail: str) -> RuntimeError:
        """建立一致的 generate 錯誤語義。"""
        return RuntimeError(f"safetensors generate unavailable [{code}]: {detail}")

    def _log_runtime_environment(self) -> None:
        """記錄載入前 runtime 診斷，便於確認 CUDA / dtype 狀態。"""
        try:
            import torch  # type: ignore[import-not-found]
        except Exception as exc:
            logger.info("Safetensors runtime torch import failed: {}", exc)
            return

        cuda_available = bool(torch.cuda.is_available())
        device_count = int(torch.cuda.device_count()) if cuda_available else 0
        device_name = (
            str(torch.cuda.get_device_name(0))
            if cuda_available and device_count > 0
            else "n/a"
        )
        logger.info(
            "Safetensors runtime env: torch={}, compiled_cuda={}, cuda_available={}, device_count={}, device0={}",
            getattr(torch, "__version__", "unknown"),
            getattr(torch.version, "cuda", None),
            cuda_available,
            device_count,
            device_name,
        )

    def _log_loaded_pipeline_details(self, pipeline: Any) -> None:
        """記錄 pipeline 建立後的模型放置資訊。"""
        model = getattr(pipeline, "model", None)
        if model is None:
            logger.info("Safetensors pipeline details unavailable: pipeline.model missing")
            return

        device = getattr(model, "device", None)
        hf_device_map = getattr(model, "hf_device_map", None)
        dtype = getattr(model, "dtype", None)
        logger.info(
            "Safetensors pipeline model placement: device={}, dtype={}, hf_device_map={}",
            str(device) if device is not None else "unknown",
            str(dtype) if dtype is not None else "unknown",
            self._summarize_device_map(hf_device_map),
        )

    def _summarize_pipeline_kwargs(self, kwargs: dict[str, Any]) -> dict[str, str]:
        """將 pipeline kwargs 壓成穩定、可讀的 log 內容。"""
        summary: dict[str, str] = {}
        for key, value in kwargs.items():
            if key in {"model", "tokenizer"}:
                summary[key] = str(value)
            else:
                summary[key] = str(value)
        return summary

    def _summarize_device_map(self, device_map: Any) -> str:
        """避免把巨大 device map 原樣灌進 log。"""
        if isinstance(device_map, dict):
            preview = list(device_map.items())[:8]
            suffix = "" if len(device_map) <= 8 else f" ... (+{len(device_map) - 8} more)"
            return f"{preview}{suffix}"
        return str(device_map)

    def _resolve_input_tokens(self, raw_result: Any, prompt: str) -> int:
        """解析輸入 token 數，缺值時回退到估算。"""
        usage = self._extract_usage(raw_result)
        usage_tokens = self._as_int_or_none(
            usage.get("prompt_tokens") if isinstance(usage, dict) else None
        )
        if usage_tokens is not None:
            return usage_tokens

        tokenizer = self._resolve_chat_template_source()
        tokenizer_tokens = self._count_tokens_with_tokenizer(prompt, tokenizer)
        if tokenizer_tokens is not None:
            return tokenizer_tokens
        return self._estimate_tokens(prompt)

    def _resolve_output_tokens(self, raw_result: Any, content: str) -> int:
        """解析輸出 token 數，缺值時回退到估算。"""
        usage = self._extract_usage(raw_result)
        usage_tokens = self._as_int_or_none(
            usage.get("completion_tokens") if isinstance(usage, dict) else None
        )
        if usage_tokens is not None:
            return usage_tokens

        tokenizer = self._resolve_chat_template_source()
        tokenizer_tokens = self._count_tokens_with_tokenizer(content, tokenizer)
        if tokenizer_tokens is not None:
            return tokenizer_tokens
        return self._estimate_tokens(content)

    def _extract_usage(self, raw_result: Any) -> dict[str, Any]:
        """從回傳中抽取 usage 物件。"""
        if isinstance(raw_result, dict):
            usage = raw_result.get("usage")
            if isinstance(usage, dict):
                return usage
            return {}

        if isinstance(raw_result, list) and raw_result:
            first = raw_result[0]
            if isinstance(first, dict):
                usage = first.get("usage")
                if isinstance(usage, dict):
                    return usage
        return {}

    def _count_tokens_with_tokenizer(self, text: str, tokenizer: Any | None) -> int | None:
        """使用 tokenizer 計算 token 數。"""
        if tokenizer is None:
            return None

        if hasattr(tokenizer, "encode"):
            try:
                encoded = tokenizer.encode(text, add_special_tokens=False)
                if isinstance(encoded, list):
                    return len(encoded)
            except Exception:
                pass

        try:
            encoded_dict = tokenizer(  # type: ignore[misc]
                text,
                add_special_tokens=False,
                return_attention_mask=False,
                return_token_type_ids=False,
            )
            if isinstance(encoded_dict, dict):
                input_ids = encoded_dict.get("input_ids")
                if isinstance(input_ids, list):
                    if input_ids and isinstance(input_ids[0], list):
                        return len(input_ids[0])
                    return len(input_ids)
        except Exception:
            pass

        if hasattr(tokenizer, "tokenize"):
            try:
                pieces = tokenizer.tokenize(text)
                if isinstance(pieces, list):
                    return len(pieces)
            except Exception:
                pass

        return None

    def _estimate_tokens(self, text: str) -> int:
        """Tokenizer 不可用時的最小估算。"""
        normalized = text.strip()
        if not normalized:
            return 0

        char_estimate = (len(normalized) + 3) // 4
        word_estimate = len(normalized.split())
        return max(1, min(len(normalized), max(char_estimate, word_estimate)))

    def _as_int_or_none(self, value: Any) -> int | None:
        """嘗試轉為 int，失敗回傳 None。"""
        try:
            return int(value)
        except (TypeError, ValueError):
            return None
