"""faster-whisper STT 包裝（可注入 runtime/factory，便於測試）。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from mochi.voice.base import BaseSTT, VoiceInfo
from mochi.voice.model_manager import resolve_faster_whisper_source


class FasterWhisperSTT(BaseSTT):
    """faster-whisper 家族 STT 最小封裝。"""

    def __init__(
        self,
        *,
        model: str = "medium",
        device: str = "auto",
        language: str = "auto",
        model_cache_dir: Path | None = None,
        runtime: Any | None = None,
        model_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.model = model
        self.device = device
        self.language = language
        self.model_cache_dir = model_cache_dir
        self._model_source, self._uses_local_model_source = resolve_faster_whisper_source(
            model,
            model_cache_dir,
        )
        self._runtime = runtime
        self._model_factory = model_factory
        self._dependency_error: Exception | None = None

        if self._runtime is None and self._model_factory is None:
            try:
                from faster_whisper import WhisperModel  # type: ignore[import-not-found]
            except Exception as exc:
                self._dependency_error = exc
            else:
                self._model_factory = WhisperModel

    async def transcribe(
        self,
        audio: bytes,
        *,
        sample_rate: int,
        language: str | None = None,
    ) -> str:
        del sample_rate  # 本階段僅保留介面，不做重採樣。
        if not audio:
            return ""

        try:
            return await asyncio.to_thread(self._transcribe_blocking, audio, language)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"faster-whisper transcribe unavailable [transcribe_failed]: {exc}"
            ) from exc

    def get_info(self) -> VoiceInfo:
        return VoiceInfo(
            kind="stt",
            family="faster-whisper",
            name=self.model,
            metadata={
                "device": self.device,
                "language": self.language,
                "loaded": self._runtime is not None,
                "dependency_ready": self._dependency_error is None,
                "model_source": self._model_source,
                "uses_local_model_source": self._uses_local_model_source,
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

        segments, _ = runtime.transcribe(audio, language=selected_language, task="transcribe")
        return self._join_segments(segments)

    def _ensure_runtime(self) -> Any:
        if self._runtime is not None:
            return self._runtime

        if self._dependency_error is not None:
            raise RuntimeError(
                "faster-whisper transcribe unavailable [dependency_missing]: "
                f"{self._dependency_error}"
            ) from self._dependency_error

        if self._model_factory is None:
            raise RuntimeError("faster-whisper transcribe unavailable [factory_missing]")

        try:
            factory_kwargs: dict[str, Any] = {"device": self.device}
            if not self._uses_local_model_source and self.model_cache_dir is not None:
                factory_kwargs["download_root"] = str(self.model_cache_dir)
            self._runtime = self._model_factory(self._model_source, **factory_kwargs)
        except Exception as exc:
            raise RuntimeError(
                f"faster-whisper transcribe unavailable [runtime_init_failed]: {exc}"
            ) from exc
        return self._runtime

    @staticmethod
    def _join_segments(segments: Iterable[Any]) -> str:
        parts: list[str] = []
        for segment in segments:
            text = getattr(segment, "text", "")
            if text:
                parts.append(str(text).strip())
        return " ".join(part for part in parts if part).strip()
