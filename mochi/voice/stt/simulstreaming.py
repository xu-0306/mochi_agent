"""SimulStreaming STT 包裝（bounded Phase 4）。"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Mapping
from typing import Any

from mochi.voice.base import BaseSTT, VoiceInfo


class SimulStreamingSTT(BaseSTT):
    """UFAL SimulStreaming 最小封裝。"""

    def __init__(
        self,
        *,
        model: str = "base",
        language: str = "auto",
        runtime: Any | None = None,
        runtime_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.model = model
        self.language = language
        self._runtime = runtime
        self._runtime_factory = runtime_factory
        self._dependency_error: Exception | None = None

        if self._runtime is None and self._runtime_factory is None:
            try:
                import whisper_streaming  # type: ignore[import-not-found]
            except Exception as exc:
                self._dependency_error = exc
            else:
                self._runtime_factory = _resolve_factory(whisper_streaming)

    async def transcribe(
        self,
        audio: bytes,
        *,
        sample_rate: int,
        language: str | None = None,
    ) -> str:
        if not audio:
            return ""
        try:
            return await asyncio.to_thread(self._transcribe_blocking, audio, sample_rate, language)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"simulstreaming transcribe unavailable [transcribe_failed]: {exc}"
            ) from exc

    def get_info(self) -> VoiceInfo:
        return VoiceInfo(
            kind="stt",
            family="simulstreaming",
            name=self.model,
            metadata={
                "language": self.language,
                "loaded": self._runtime is not None,
                "dependency_ready": self._dependency_error is None,
            },
        )

    async def health_check(self) -> bool:
        return self._dependency_error is None and (
            self._runtime is not None or self._runtime_factory is not None
        )

    async def close(self) -> None:
        self._runtime = None

    def _transcribe_blocking(self, audio: bytes, sample_rate: int, language: str | None) -> str:
        runtime = self._ensure_runtime()
        selected_language = language if language not in (None, "auto") else self.language
        if selected_language == "auto":
            selected_language = None
        raw_result = _call_candidate(
            runtime,
            audio=audio,
            sample_rate=sample_rate,
            language=selected_language,
        )
        return _extract_text(raw_result)

    def _ensure_runtime(self) -> Any:
        if self._runtime is not None:
            return self._runtime
        if self._dependency_error is not None:
            raise RuntimeError(
                "simulstreaming transcribe unavailable [dependency_missing]: "
                f"{self._dependency_error}"
            ) from self._dependency_error
        if self._runtime_factory is None:
            raise RuntimeError("simulstreaming transcribe unavailable [factory_missing]")
        try:
            self._runtime = _call_with_supported_kwargs(
                self._runtime_factory,
                model=self.model,
                language=self.language,
            )
        except Exception as exc:
            raise RuntimeError(
                f"simulstreaming transcribe unavailable [runtime_init_failed]: {exc}"
            ) from exc
        return self._runtime


def _resolve_factory(module: Any) -> Callable[..., Any] | None:
    for name in ("OnlineASRProcessor", "create_processor", "SimulStreaming"):
        factory = getattr(module, name, None)
        if callable(factory):
            return factory
    return None


def _call_candidate(runtime: Any, **kwargs: Any) -> Any:
    for name in ("transcribe", "process_iter", "infer", "__call__"):
        func = runtime if name == "__call__" and callable(runtime) else getattr(runtime, name, None)
        if callable(func):
            return _call_with_supported_kwargs(func, **kwargs)
    raise RuntimeError("simulstreaming transcribe unavailable [factory_missing]")


def _extract_text(raw_result: Any) -> str:
    if isinstance(raw_result, str):
        return raw_result.strip()
    if isinstance(raw_result, Mapping):
        for key in ("text", "transcript", "output_text"):
            value = raw_result.get(key, "")
            if value:
                return str(value).strip()
        return ""
    text_attr = getattr(raw_result, "text", None)
    if isinstance(text_attr, str):
        return text_attr.strip()
    if isinstance(raw_result, tuple):
        for item in reversed(raw_result):
            if isinstance(item, str) and item.strip():
                return item.strip()
            if isinstance(item, Mapping):
                text = _extract_text(item)
                if text:
                    return text
    return ""


def _call_with_supported_kwargs(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(*args, **kwargs)

    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return func(*args, **kwargs)

    accepted_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return func(*args, **accepted_kwargs)
