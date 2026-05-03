"""Vosk STT 包裝（bounded Phase 4）。"""

from __future__ import annotations

import asyncio
import inspect
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from mochi.voice.base import BaseSTT, VoiceInfo


class VoskSTT(BaseSTT):
    """Vosk（KaldiRecognizer）最小封裝。"""

    def __init__(
        self,
        *,
        model: str | Path = "model",
        language: str = "auto",
        chunk_size: int = 4000,
        runtime: Any | None = None,
        model_factory: Callable[..., Any] | None = None,
        recognizer_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.model = str(model)
        self.language = language
        self.chunk_size = max(1, int(chunk_size))
        self._runtime = runtime
        self._model_factory = model_factory
        self._recognizer_factory = recognizer_factory
        self._dependency_error: Exception | None = None
        self._model_source = str(Path(model).expanduser())

        needs_model_factory = self._runtime is None and self._model_factory is None
        if needs_model_factory or self._recognizer_factory is None:
            try:
                from vosk import KaldiRecognizer, Model  # type: ignore[import-not-found]
            except Exception as exc:
                self._dependency_error = exc
            else:
                if self._model_factory is None:
                    self._model_factory = Model
                if self._recognizer_factory is None:
                    self._recognizer_factory = KaldiRecognizer

    async def transcribe(
        self,
        audio: bytes,
        *,
        sample_rate: int,
        language: str | None = None,
    ) -> str:
        del language  # Vosk 語言由模型決定，本層不做語言切換。
        if not audio:
            return ""

        try:
            return await asyncio.to_thread(self._transcribe_blocking, audio, sample_rate)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"vosk transcribe unavailable [transcribe_failed]: {exc}"
            ) from exc

    def get_info(self) -> VoiceInfo:
        return VoiceInfo(
            kind="stt",
            family="vosk",
            name=self.model,
            metadata={
                "language": self.language,
                "loaded": self._runtime is not None,
                "dependency_ready": self._dependency_error is None,
                "model_source": self._model_source,
                "chunk_size": self.chunk_size,
                "recognizer_ready": self._recognizer_factory is not None,
            },
        )

    async def health_check(self) -> bool:
        return (
            self._dependency_error is None
            and self._recognizer_factory is not None
            and (self._runtime is not None or self._model_factory is not None)
        )

    async def close(self) -> None:
        self._runtime = None

    def _transcribe_blocking(self, audio: bytes, sample_rate: int) -> str:
        model_runtime = self._ensure_runtime()
        recognizer = self._build_recognizer(model_runtime, sample_rate)

        parts: list[str] = []
        for start in range(0, len(audio), self.chunk_size):
            chunk = audio[start : start + self.chunk_size]
            if not chunk:
                continue
            accepted = bool(recognizer.AcceptWaveform(chunk))
            if accepted:
                text = self._extract_text(recognizer.Result())
                if text:
                    parts.append(text)

        final_text = self._extract_text(recognizer.FinalResult())
        if final_text:
            parts.append(final_text)
        return " ".join(parts).strip()

    def _ensure_runtime(self) -> Any:
        if self._runtime is not None:
            return self._runtime

        if self._dependency_error is not None:
            raise RuntimeError(
                "vosk transcribe unavailable [dependency_missing]: "
                f"{self._dependency_error}"
            ) from self._dependency_error

        if self._model_factory is None:
            raise RuntimeError("vosk transcribe unavailable [factory_missing]")

        try:
            self._runtime = _call_with_supported_kwargs(self._model_factory, self._model_source)
        except Exception as exc:
            raise RuntimeError(
                f"vosk transcribe unavailable [runtime_init_failed]: {exc}"
            ) from exc
        return self._runtime

    def _build_recognizer(self, model_runtime: Any, sample_rate: int) -> Any:
        if self._recognizer_factory is None:
            raise RuntimeError("vosk transcribe unavailable [factory_missing]")
        try:
            return _call_with_supported_kwargs(
                self._recognizer_factory,
                model_runtime,
                sample_rate,
            )
        except Exception as exc:
            raise RuntimeError(
                f"vosk transcribe unavailable [runtime_init_failed]: {exc}"
            ) from exc

    @staticmethod
    def _extract_text(raw_result: Any) -> str:
        parsed = _json_payload(raw_result)
        if isinstance(parsed, Mapping):
            text = parsed.get("text", "")
            return str(text).strip() if text else ""
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


def _json_payload(raw_result: Any) -> Any:
    if isinstance(raw_result, Mapping):
        return raw_result
    if isinstance(raw_result, str):
        try:
            return json.loads(raw_result)
        except json.JSONDecodeError:
            return {}
    return {}
