"""openai-whisper STT 包裝（bounded Phase 4）。"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from mochi.voice.base import BaseSTT, VoiceInfo


class OpenAIWhisperSTT(BaseSTT):
    """openai/whisper 本地模型 STT 最小封裝。"""

    def __init__(
        self,
        *,
        model: str = "base",
        device: str = "auto",
        language: str = "auto",
        model_cache_dir: Path | None = None,
        in_memory: bool = False,
        runtime: Any | None = None,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.language = language
        self.model_cache_dir = model_cache_dir
        self.in_memory = in_memory
        self._model_source = str(model)
        self._runtime = runtime
        self._model_factory = model_factory
        self._dependency_error: Exception | None = None

        if self._runtime is None and self._model_factory is None:
            try:
                import whisper  # type: ignore[import-not-found]
            except Exception as exc:
                self._dependency_error = exc
            else:
                self._model_factory = whisper.load_model

    async def transcribe(
        self,
        audio: bytes,
        *,
        sample_rate: int,
        language: str | None = None,
    ) -> str:
        del sample_rate  # openai/whisper 的 ndarray 路徑預設為 16kHz，不在本層做重採樣。
        if not audio:
            return ""

        try:
            return await asyncio.to_thread(self._transcribe_blocking, audio, language)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"openai-whisper transcribe unavailable [transcribe_failed]: {exc}"
            ) from exc

    def get_info(self) -> VoiceInfo:
        return VoiceInfo(
            kind="stt",
            family="openai-whisper",
            name=self.model,
            metadata={
                "device": self.device,
                "language": self.language,
                "loaded": self._runtime is not None,
                "dependency_ready": self._dependency_error is None,
                "model_source": self._model_source,
                "model_cache_dir": str(self.model_cache_dir) if self.model_cache_dir else None,
                "in_memory": self.in_memory,
            },
        )

    async def health_check(self) -> bool:
        return self._dependency_error is None

    async def close(self) -> None:
        self._runtime = None

    def _transcribe_blocking(self, audio: bytes, language: str | None) -> str:
        runtime = self._ensure_runtime()
        selected_language = language if language not in (None, "auto") else self.language
        if selected_language == "auto":
            selected_language = None

        raw_result = _call_with_supported_kwargs(
            runtime.transcribe,
            self._pcm16_to_float32(audio),
            language=selected_language,
            task="transcribe",
        )
        return self._extract_text(raw_result)

    def _ensure_runtime(self) -> Any:
        if self._runtime is not None:
            return self._runtime

        if self._dependency_error is not None:
            raise RuntimeError(
                "openai-whisper transcribe unavailable [dependency_missing]: "
                f"{self._dependency_error}"
            ) from self._dependency_error

        if self._model_factory is None:
            raise RuntimeError("openai-whisper transcribe unavailable [factory_missing]")

        try:
            factory_kwargs: dict[str, Any] = {}
            if self.device not in ("", "auto"):
                factory_kwargs["device"] = self.device
            if self.model_cache_dir is not None:
                factory_kwargs["download_root"] = str(self.model_cache_dir)
            if self.in_memory:
                factory_kwargs["in_memory"] = True
            self._runtime = _call_with_supported_kwargs(
                self._model_factory,
                self._model_source,
                **factory_kwargs,
            )
        except Exception as exc:
            raise RuntimeError(
                f"openai-whisper transcribe unavailable [runtime_init_failed]: {exc}"
            ) from exc
        return self._runtime

    @staticmethod
    def _pcm16_to_float32(audio: bytes) -> Any:
        try:
            import numpy as np  # type: ignore[import-not-found]
        except Exception as exc:
            raise RuntimeError(
                "openai-whisper transcribe unavailable [dependency_missing]: numpy is required"
            ) from exc
        return np.frombuffer(audio, dtype=np.int16).astype(np.float32) / 32768.0

    @staticmethod
    def _extract_text(raw_result: Any) -> str:
        if isinstance(raw_result, str):
            return raw_result.strip()
        if isinstance(raw_result, Mapping):
            text = raw_result.get("text", "")
            return str(text).strip() if text else ""

        text_attr = getattr(raw_result, "text", None)
        if isinstance(text_attr, str):
            return text_attr.strip()
        return ""


def _call_with_supported_kwargs(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(*args, **kwargs)

    if any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return func(*args, **kwargs)

    accepted_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return func(*args, **accepted_kwargs)
