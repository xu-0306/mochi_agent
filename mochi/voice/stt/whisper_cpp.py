"""whisper-cpp STT 包裝（bounded Phase 4）。"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from mochi.voice.base import BaseSTT, VoiceInfo


class WhisperCppSTT(BaseSTT):
    """whisper-cpp（whisper.cpp Python binding）最小封裝。"""

    def __init__(
        self,
        *,
        model: str = "base",
        model_path: str | Path | None = None,
        language: str = "auto",
        n_threads: int | None = None,
        runtime: Any | None = None,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.model = model
        self.model_path = Path(model_path).expanduser() if model_path is not None else None
        self.language = language
        self.n_threads = n_threads
        self._runtime = runtime
        self._model_factory = model_factory
        self._dependency_error: Exception | None = None
        self._model_source = str(self.model_path) if self.model_path is not None else self.model

        if self._runtime is None and self._model_factory is None:
            try:
                from whisper_cpp import Whisper  # type: ignore[import-not-found]
            except Exception as exc:
                self._dependency_error = exc
            else:
                self._model_factory = Whisper

    async def transcribe(
        self,
        audio: bytes,
        *,
        sample_rate: int,
        language: str | None = None,
    ) -> str:
        del sample_rate  # 本階段維持 mono PCM16 bytes 路徑，不做重採樣。
        if not audio:
            return ""

        try:
            return await asyncio.to_thread(self._transcribe_blocking, audio, language)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"whisper-cpp transcribe unavailable [transcribe_failed]: {exc}"
            ) from exc

    def get_info(self) -> VoiceInfo:
        return VoiceInfo(
            kind="stt",
            family="whisper-cpp",
            name=self.model,
            metadata={
                "language": self.language,
                "loaded": self._runtime is not None,
                "dependency_ready": self._dependency_error is None,
                "model_source": self._model_source,
                "n_threads": self.n_threads,
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
            audio,
            language=selected_language,
        )
        return self._extract_text(raw_result)

    def _ensure_runtime(self) -> Any:
        if self._runtime is not None:
            return self._runtime

        if self._dependency_error is not None:
            raise RuntimeError(
                "whisper-cpp transcribe unavailable [dependency_missing]: "
                f"{self._dependency_error}"
            ) from self._dependency_error

        if self._model_factory is None:
            raise RuntimeError("whisper-cpp transcribe unavailable [factory_missing]")

        try:
            factory_kwargs: dict[str, Any] = {"model_path": self._model_source}
            if self.n_threads is not None:
                factory_kwargs["n_threads"] = self.n_threads
            self._runtime = self._model_factory(**factory_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"whisper-cpp transcribe unavailable [runtime_init_failed]: {exc}"
            ) from exc
        return self._runtime

    @classmethod
    def _extract_text(cls, raw_result: Any) -> str:
        if raw_result is None:
            return ""
        if isinstance(raw_result, str):
            return raw_result.strip()
        if isinstance(raw_result, dict):
            text_value = raw_result.get("text", "")
            return str(text_value).strip() if text_value else ""
        if isinstance(raw_result, tuple) and raw_result:
            return cls._extract_text(raw_result[0])

        text_attr = getattr(raw_result, "text", None)
        if isinstance(text_attr, str):
            return text_attr.strip()

        if isinstance(raw_result, Iterable):
            parts: list[str] = []
            for segment in raw_result:
                segment_text = getattr(segment, "text", None)
                if segment_text:
                    parts.append(str(segment_text).strip())
            return " ".join(part for part in parts if part).strip()
        return ""


def _call_with_supported_kwargs(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """只傳入目標函式支援的 keyword 參數。"""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(*args, **kwargs)

    accepted_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return func(*args, **accepted_kwargs)
