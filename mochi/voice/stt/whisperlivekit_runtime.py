"""WhisperLiveKit runtime/service abstraction."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, TypeVar, cast

from mochi.voice.events import (
    FinalTranscriptionEvent,
    PartialTranscriptionEvent,
    TranscriptionEvent,
)

T = TypeVar("T")


@dataclass(slots=True, frozen=True)
class WhisperLiveKitRuntimeOptions:
    """WhisperLiveKit engine 預設選項。"""

    model: str = "base"
    language: str = "auto"
    backend_policy: str = "localagreement"
    backend: str = "auto"
    min_chunk_size: float = 0.1
    buffer_trimming: str = "segment"
    buffer_trimming_sec: float = 15.0
    confidence_validation: bool = False
    pcm_input: bool = True
    vad: bool = True
    vac: bool = True
    vac_chunk_size: float = 0.04
    watchdog_enabled: bool = True
    watchdog_poll_interval: float = 0.5
    watchdog_no_result_timeout: float = 4.0
    watchdog_stall_timeout: float = 3.0
    watchdog_audio_idle_window: float = 1.5
    watchdog_reset_cooldown: float = 2.0
    model_cache_dir: str | None = None
    model_dir: str | None = None
    model_path: str | None = None
    lora_path: str | None = None

    def to_runtime_factory_kwargs(self) -> dict[str, Any]:
        """建立相容於不同 WhisperLiveKit factory 的 kwargs。"""
        return {
            "model": self.model,
            "model_size": self.model,
            "language": self.language,
            "lan": self.language,
            "backend_policy": self.backend_policy,
            "backend": self.backend,
            "min_chunk_size": self.min_chunk_size,
            "buffer_trimming": self.buffer_trimming,
            "buffer_trimming_sec": self.buffer_trimming_sec,
            "confidence_validation": self.confidence_validation,
            "pcm_input": self.pcm_input,
            "vad": self.vad,
            "vac": self.vac,
            "vac_chunk_size": self.vac_chunk_size,
            "model_cache_dir": self.model_cache_dir,
            "model_dir": self.model_dir,
            "model_path": self.model_path,
            "lora_path": self.lora_path,
            "target_language": "",
        }

    def resolve_language(self, language: str | None) -> str | None:
        """將 auto 語言設定正規化為 WhisperLiveKit 可接受的值。"""
        selected_language = language if language not in (None, "auto") else self.language
        return None if selected_language == "auto" else selected_language


@dataclass(slots=True, frozen=True)
class WhisperLiveKitRealtimeState:
    """Realtime session 狀態快照。"""

    first_audio_ts: float | None
    last_audio_ts: float
    last_result_ts: float
    result_seen: bool
    processor_builds: int
    watchdog_resets: int
    watchdog_runtime_rebuilds: int
    watchdog_last_reason: str | None


class WhisperLiveKitService:
    """封裝 WhisperLiveKit engine 與 realtime processor 建立流程。"""

    def __init__(
        self,
        *,
        options: WhisperLiveKitRuntimeOptions | None = None,
        runtime: Any | None = None,
        runtime_factory: Callable[..., Any] | None = None,
        audio_processor_factory: Callable[..., Any] | None = None,
    ) -> None:
        self._options = options or WhisperLiveKitRuntimeOptions()
        self._runtime = runtime
        self._runtime_factory = runtime_factory
        self._audio_processor_factory = audio_processor_factory
        self._runtime_module: Any | None = None
        self._dependency_error: Exception | None = None
        self._runtime_builds = 1 if runtime is not None else 0

        if self._runtime is None and self._runtime_factory is None:
            try:
                import whisperlivekit  # type: ignore[import-not-found]
            except Exception as exc:
                self._dependency_error = exc
            else:
                _patch_whisperlivekit_asrtoken(whisperlivekit)
                self._runtime_module = whisperlivekit
                self._runtime_factory = _resolve_runtime_factory(whisperlivekit)
                if self._audio_processor_factory is None:
                    self._audio_processor_factory = _resolve_audio_processor_factory(whisperlivekit)

    @property
    def dependency_error(self) -> Exception | None:
        return self._dependency_error

    @property
    def options(self) -> WhisperLiveKitRuntimeOptions:
        return self._options

    @property
    def runtime_loaded(self) -> bool:
        return self._runtime is not None

    @property
    def runtime_builds(self) -> int:
        return self._runtime_builds

    def health_check(self) -> bool:
        return self._dependency_error is None and (
            self._runtime is not None or self._runtime_factory is not None
        )

    def supports_realtime(self) -> bool:
        if self._audio_processor_factory is not None:
            return True
        if self._runtime is not None:
            factory = _resolve_audio_processor_factory(self._runtime)
            if factory is not None:
                return True
        return self._runtime_module is not None and _resolve_audio_processor_factory(
            self._runtime_module
        ) is not None

    def transcribe(
        self,
        audio: bytes,
        *,
        sample_rate: int,
        language: str | None = None,
    ) -> str:
        runtime = self.ensure_runtime()
        raw_result = _call_transcribe_candidate(
            runtime,
            audio=audio,
            sample_rate=sample_rate,
            language=self._options.resolve_language(language),
        )
        return _extract_text(raw_result)

    async def create_realtime_session(
        self,
        *,
        sample_rate: int,
        language: str | None = None,
    ) -> WhisperLiveKitRealtimeSession | None:
        factory = self._resolve_audio_processor_factory()
        if factory is None:
            return None

        processor, runtime = await self._build_audio_processor(
            sample_rate=sample_rate,
            language=language,
            rebuild_runtime=False,
        )
        return WhisperLiveKitRealtimeSession(
            service=self,
            options=self._options,
            runtime=runtime,
            processor=processor,
            sample_rate=sample_rate,
            language=language,
        )

    def ensure_runtime(self) -> Any:
        if self._runtime is not None:
            return self._runtime
        if self._dependency_error is not None:
            raise RuntimeError(
                "whisperlivekit transcribe unavailable [dependency_missing]: "
                f"{self._dependency_error}"
            ) from self._dependency_error
        if self._runtime_factory is None:
            raise RuntimeError("whisperlivekit transcribe unavailable [factory_missing]")
        try:
            self._runtime = _call_with_supported_kwargs(
                self._runtime_factory,
                **self._options.to_runtime_factory_kwargs(),
            )
        except Exception as exc:
            raise RuntimeError(
                f"whisperlivekit transcribe unavailable [runtime_init_failed]: {exc}"
            ) from exc
        self._runtime_builds += 1
        return self._runtime

    async def rebuild_runtime(self) -> Any:
        await self._close_runtime()
        return await asyncio.to_thread(self.ensure_runtime)

    async def close(self) -> None:
        await self._close_runtime()

    async def create_audio_processor(
        self,
        *,
        sample_rate: int,
        language: str | None = None,
        rebuild_runtime: bool = False,
    ) -> tuple[Any, Any]:
        return await self._build_audio_processor(
            sample_rate=sample_rate,
            language=language,
            rebuild_runtime=rebuild_runtime,
        )

    async def _build_audio_processor(
        self,
        *,
        sample_rate: int,
        language: str | None,
        rebuild_runtime: bool,
    ) -> tuple[Any, Any]:
        factory = self._resolve_audio_processor_factory()
        if factory is None:
            raise RuntimeError("whisperlivekit preview unavailable [processor_factory_missing]")

        runtime = (
            await self.rebuild_runtime()
            if rebuild_runtime
            else await asyncio.to_thread(self.ensure_runtime)
        )
        processor_kwargs = {
            "transcription_engine": runtime,
            "engine": runtime,
            "runtime": runtime,
            "sample_rate": sample_rate,
            "language": self._options.resolve_language(language),
            "enable_transcription": True,
            "transcription": True,
            "enable_diarization": False,
            "diarization": False,
        }

        try:
            processor = await _maybe_await(_call_with_supported_kwargs(factory, **processor_kwargs))
        except Exception as exc:
            raise RuntimeError(
                f"whisperlivekit preview unavailable [processor_init_failed]: {exc}"
            ) from exc
        return processor, runtime

    async def _close_runtime(self) -> None:
        runtime = self._runtime
        self._runtime = None
        if runtime is None:
            return
        await _close_component(runtime, ("cleanup", "close", "stop", "shutdown"))

    def _resolve_audio_processor_factory(self) -> Callable[..., Any] | None:
        if self._audio_processor_factory is not None:
            return self._audio_processor_factory
        if self._runtime is not None:
            factory = _resolve_audio_processor_factory(self._runtime)
            if factory is not None:
                return factory
        if self._runtime_module is not None:
            return _resolve_audio_processor_factory(self._runtime_module)
        return None


class WhisperLiveKitRealtimeSession:
    """WhisperLiveKit realtime session。"""

    def __init__(
        self,
        *,
        service: WhisperLiveKitService,
        options: WhisperLiveKitRuntimeOptions,
        runtime: Any,
        processor: Any,
        sample_rate: int,
        language: str | None,
    ) -> None:
        self._service = service
        self._options = options
        self._runtime = runtime
        self._processor = processor
        self._sample_rate = sample_rate
        self._language = language
        self._collector_task: asyncio.Task[None] | None = None
        self._watchdog_task: asyncio.Task[None] | None = None
        self._collector_error: Exception | None = None
        self._results_stream: Any | None = None
        self._queue: asyncio.Queue[Any] = asyncio.Queue()
        self._start_lock = asyncio.Lock()
        self._reset_lock = asyncio.Lock()
        self._closed = False
        self._last_preview_text = ""
        self._last_partial_text = ""
        self._seen_final_texts: set[str] = set()
        self._processor_builds = 1
        self._first_audio_ts: float | None = None
        self._watchdog_resets = 0
        self._watchdog_runtime_rebuilds = 0
        self._watchdog_last_reason: str | None = None
        self._last_watchdog_reset_ts = 0.0
        now = time.monotonic()
        self._last_audio_ts = now
        self._last_result_ts = now
        self._result_seen = False
        if self._options.watchdog_enabled:
            self._watchdog_task = asyncio.create_task(self._run_watchdog())

    async def append_audio(self, audio: bytes) -> list[TranscriptionEvent]:
        """餵入音訊並回傳目前可送出的 preview 事件。"""
        if self._closed or not audio:
            return []

        await self._ensure_started()

        process_audio = _resolve_processor_method(
            self._processor,
            ("process_audio", "append_audio", "feed_audio", "push_audio"),
        )
        if process_audio is None:
            raise RuntimeError("whisperlivekit preview unavailable [processor_feed_missing]")

        now = time.monotonic()
        if self._first_audio_ts is None:
            self._first_audio_ts = now
        self._last_audio_ts = now

        result = await _maybe_await(_call_with_supported_kwargs(
            process_audio,
            chunk=audio,
            data=audio,
            audio=audio,
        ))
        await self._capture_result(result)
        await asyncio.sleep(0)
        return self._drain_preview_events()

    async def push_audio(self, audio: bytes) -> list[str]:
        """legacy preview 介面：回傳純文字列表。"""
        if self._closed or not audio:
            return []

        await self._ensure_started()

        process_audio = _resolve_processor_method(
            self._processor,
            ("process_audio", "append_audio", "feed_audio", "push_audio"),
        )
        if process_audio is None:
            raise RuntimeError("whisperlivekit preview unavailable [processor_feed_missing]")

        now = time.monotonic()
        if self._first_audio_ts is None:
            self._first_audio_ts = now
        self._last_audio_ts = now

        result = await _maybe_await(_call_with_supported_kwargs(
            process_audio,
            chunk=audio,
            data=audio,
            audio=audio,
        ))
        await self._capture_result(result)
        await asyncio.sleep(0)
        return self._drain_preview_texts()

    async def drain_events(self) -> list[TranscriptionEvent]:
        """擷取目前已就緒但尚未送出的 preview 事件。"""
        if self._closed:
            return []
        if self._collector_task is None:
            await self._ensure_started()
        await asyncio.sleep(0)
        return self._drain_preview_events()

    async def drain_texts(self) -> list[str]:
        """legacy preview 介面：回傳純文字列表。"""
        if self._closed:
            return []
        if self._collector_task is None:
            await self._ensure_started()
        await asyncio.sleep(0)
        return self._drain_preview_texts()

    async def reset(self, *, rebuild_runtime: bool = False) -> None:
        """重建 audio processor，保留後續 realtime session。"""
        if self._closed:
            return

        async with self._reset_lock:
            await self._teardown_processor()
            processor, runtime = await self._service.create_audio_processor(
                sample_rate=self._sample_rate,
                language=self._language,
                rebuild_runtime=rebuild_runtime,
            )
            self._runtime = runtime
            self._processor = processor
            self._processor_builds += 1
            self._collector_error = None
            self._last_preview_text = ""
            self._last_partial_text = ""
            self._seen_final_texts.clear()
            self._first_audio_ts = None
            now = time.monotonic()
            self._last_audio_ts = now
            self._last_result_ts = now
            self._result_seen = False

    def get_state(self) -> WhisperLiveKitRealtimeState:
        """回傳 watchdog/reset 可用的狀態快照。"""
        return WhisperLiveKitRealtimeState(
            first_audio_ts=self._first_audio_ts,
            last_audio_ts=self._last_audio_ts,
            last_result_ts=self._last_result_ts,
            result_seen=self._result_seen,
            processor_builds=self._processor_builds,
            watchdog_resets=self._watchdog_resets,
            watchdog_runtime_rebuilds=self._watchdog_runtime_rebuilds,
            watchdog_last_reason=self._watchdog_last_reason,
        )

    async def close(self) -> None:
        """停止 collector 並釋放底層 processor。"""
        self._closed = True
        watchdog_task = self._watchdog_task
        self._watchdog_task = None
        if watchdog_task is not None:
            watchdog_task.cancel()
            with suppress(asyncio.CancelledError):
                await watchdog_task
        await self._teardown_processor()

    async def _run_watchdog(self) -> None:
        while not self._closed:
            await asyncio.sleep(self._options.watchdog_poll_interval)
            decision = self._get_watchdog_decision()
            if decision is None:
                continue
            reason, rebuild_runtime = decision
            try:
                await self.reset(rebuild_runtime=rebuild_runtime)
            except asyncio.CancelledError:
                raise
            except Exception:
                continue
            self._watchdog_resets += 1
            if rebuild_runtime:
                self._watchdog_runtime_rebuilds += 1
            self._watchdog_last_reason = reason
            self._last_watchdog_reset_ts = time.monotonic()

    def _get_watchdog_decision(self) -> tuple[str, bool] | None:
        if self._closed or self._reset_lock.locked():
            return None

        state = self.get_state()
        if state.first_audio_ts is None:
            return None

        now = time.monotonic()
        if now - self._last_watchdog_reset_ts < self._options.watchdog_reset_cooldown:
            return None
        if now - state.last_audio_ts > self._options.watchdog_audio_idle_window:
            return None

        if (
            not state.result_seen
            and now - state.first_audio_ts >= self._options.watchdog_no_result_timeout
        ):
            return ("no_result", True)

        if (
            state.result_seen
            and state.last_audio_ts > state.last_result_ts
            and now - state.last_result_ts >= self._options.watchdog_stall_timeout
        ):
            rebuild_runtime = (
                self._watchdog_last_reason == "stall"
                and self._watchdog_resets > self._watchdog_runtime_rebuilds
            )
            return ("stall", rebuild_runtime)

        return None

    async def _ensure_started(self) -> None:
        if self._closed or self._collector_task is not None:
            return

        async with self._start_lock:
            if self._closed or self._collector_task is not None:
                return

            create_tasks = getattr(self._processor, "create_tasks", None)
            if not callable(create_tasks):
                return

            result = await _maybe_await(_call_with_supported_kwargs(create_tasks))
            if result is None:
                return

            if hasattr(result, "__aiter__"):
                self._results_stream = result
                self._collector_task = asyncio.create_task(
                    self._collect_results(cast(AsyncIterator[Any], result))
                )
                return

            await self._capture_result(result)

    async def _collect_results(self, results: AsyncIterator[Any]) -> None:
        try:
            async for item in results:
                await self._queue.put(item)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._collector_error = exc

    async def _capture_result(self, result: Any) -> None:
        if result is None:
            return
        if hasattr(result, "__aiter__"):
            async for item in cast(AsyncIterator[Any], result):
                await self._queue.put(item)
            return
        await self._queue.put(result)

    def _drain_preview_events(self) -> list[TranscriptionEvent]:
        if self._collector_error is not None:
            exc = self._collector_error
            self._collector_error = None
            raise RuntimeError(
                f"whisperlivekit preview unavailable [collector_failed]: {exc}"
            ) from exc

        preview_events: list[TranscriptionEvent] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                self._queue.task_done()
            for candidate in _extract_preview_events(item):
                if not candidate.text:
                    continue
                if candidate.is_final:
                    if candidate.text in self._seen_final_texts:
                        continue
                    self._seen_final_texts.add(candidate.text)
                else:
                    if candidate.text == self._last_partial_text:
                        continue
                    self._last_partial_text = candidate.text
                preview_events.append(candidate)

        if preview_events:
            self._result_seen = True
            self._last_result_ts = time.monotonic()
        return preview_events

    def _drain_preview_texts(self) -> list[str]:
        if self._collector_error is not None:
            exc = self._collector_error
            self._collector_error = None
            raise RuntimeError(
                f"whisperlivekit preview unavailable [collector_failed]: {exc}"
            ) from exc

        preview_texts: list[str] = []
        while True:
            try:
                item = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            else:
                self._queue.task_done()
            for candidate in _extract_preview_texts(item):
                if not candidate or candidate == self._last_preview_text:
                    continue
                preview_texts.append(candidate)
                self._last_preview_text = candidate

        if preview_texts:
            self._result_seen = True
            self._last_result_ts = time.monotonic()
        return preview_texts

    async def _teardown_processor(self) -> None:
        collector_task = self._collector_task
        self._collector_task = None
        if collector_task is not None:
            collector_task.cancel()
            with suppress(asyncio.CancelledError):
                await collector_task

        results_stream = self._results_stream
        self._results_stream = None
        if results_stream is not None:
            await _close_results_stream(results_stream)

        processor = self._processor
        self._processor = None
        if processor is not None:
            await _close_component(processor, ("cleanup", "close", "stop", "shutdown"))


WhisperLiveKitPreviewSession = WhisperLiveKitRealtimeSession


def _patch_whisperlivekit_asrtoken(module: Any) -> None:
    try:
        la_backends = importlib.import_module(f"{module.__name__}.local_agreement.backends")
    except Exception:
        return
    asr_token = getattr(la_backends, "ASRToken", None)
    if asr_token is None:
        return
    try:
        signature = inspect.signature(asr_token.__init__)
    except (TypeError, ValueError):
        return
    if "probability" in signature.parameters:
        return

    original_init = asr_token.__init__

    def _init(self, *args: Any, probability: Any = None, **kwargs: Any) -> Any:  # noqa: ARG001
        return original_init(self, *args, **kwargs)

    asr_token.__init__ = _init


def _resolve_runtime_factory(module: Any) -> Callable[..., Any] | None:
    for name in ("TranscriptionEngine", "create_engine", "WhisperLiveKit"):
        factory = getattr(module, name, None)
        if callable(factory):
            return factory
    return None


def _resolve_audio_processor_factory(container: Any) -> Callable[..., Any] | None:
    for name in ("create_audio_processor", "create_processor", "AudioProcessor"):
        factory = getattr(container, name, None)
        if callable(factory):
            return factory
    return None


def _resolve_processor_method(processor: Any, names: tuple[str, ...]) -> Callable[..., Any] | None:
    for name in names:
        method = getattr(processor, name, None)
        if callable(method):
            return method
    return None


def _call_transcribe_candidate(runtime: Any, **kwargs: Any) -> Any:
    for name in ("transcribe", "process_file", "infer", "__call__"):
        func = runtime if name == "__call__" and callable(runtime) else getattr(runtime, name, None)
        if callable(func):
            return _call_with_supported_kwargs(func, **kwargs)
    raise RuntimeError("whisperlivekit transcribe unavailable [factory_missing]")


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
    return ""


def _extract_preview_events(raw_result: Any) -> list[TranscriptionEvent]:
    if raw_result is None:
        return []
    if isinstance(raw_result, str):
        text = raw_result.strip()
        return [PartialTranscriptionEvent(text=text)] if text else []
    if isinstance(raw_result, Mapping):
        events = _extract_preview_events_from_mapping(raw_result)
        if events:
            return events
        nested_results: list[TranscriptionEvent] = []
        for key in ("result", "response", "payload"):
            if key in raw_result:
                nested_results.extend(_extract_preview_events(raw_result.get(key)))
        return nested_results
    if isinstance(raw_result, (list, tuple)):
        preview_events: list[TranscriptionEvent] = []
        for item in raw_result:
            preview_events.extend(_extract_preview_events(item))
        return preview_events

    mapping_like = _coerce_mapping_like(raw_result)
    if mapping_like is not None:
        events = _extract_preview_events_from_mapping(mapping_like)
        if events:
            return events

    text = _extract_text(raw_result)
    return [PartialTranscriptionEvent(text=line) for line in text.splitlines() if line]


def _extract_preview_texts(raw_result: Any) -> list[str]:
    if raw_result is None:
        return []
    if isinstance(raw_result, str):
        text = raw_result.strip()
        return [text] if text else []
    if isinstance(raw_result, Mapping):
        composite = _build_preview_text_from_mapping(raw_result)
        if composite:
            return [composite]
        nested_results: list[str] = []
        for key in ("result", "response", "payload"):
            if key in raw_result:
                nested_results.extend(_extract_preview_texts(raw_result.get(key)))
        return nested_results
    if isinstance(raw_result, (list, tuple)):
        preview_texts: list[str] = []
        for item in raw_result:
            preview_texts.extend(_extract_preview_texts(item))
        return preview_texts

    mapping_like = _coerce_mapping_like(raw_result)
    if mapping_like is not None:
        composite = _build_preview_text_from_mapping(mapping_like)
        if composite:
            return [composite]

    fallback = _extract_text(raw_result)
    return fallback.splitlines() if fallback else []


def _extract_preview_events_from_mapping(payload: Mapping[str, Any]) -> list[TranscriptionEvent]:
    direct = _extract_text(payload)
    if direct:
        is_final = bool(payload.get("is_final", payload.get("final", False)))
        event_cls = FinalTranscriptionEvent if is_final else PartialTranscriptionEvent
        return [event_cls(text=direct)]

    events: list[TranscriptionEvent] = []
    lines = payload.get("lines")
    if isinstance(lines, list):
        for line in lines:
            line_text = _extract_line_text(line)
            if line_text:
                events.append(FinalTranscriptionEvent(text=line_text))

    buffer_text = payload.get("buffer_transcription")
    if not isinstance(buffer_text, str) or not buffer_text.strip():
        buffer_text = payload.get("buffer")
    if isinstance(buffer_text, str) and buffer_text.strip():
        events.append(PartialTranscriptionEvent(text=buffer_text.strip()))

    return events


def _build_preview_text_from_mapping(payload: Mapping[str, Any]) -> str:
    direct = _extract_text(payload)
    if direct:
        return direct

    segments: list[str] = []
    lines = payload.get("lines")
    if isinstance(lines, list):
        for line in lines:
            line_text = _extract_line_text(line)
            if line_text:
                segments.append(line_text)

    buffer_text = payload.get("buffer_transcription")
    if not isinstance(buffer_text, str) or not buffer_text.strip():
        buffer_text = payload.get("buffer")

    if isinstance(buffer_text, str) and buffer_text.strip():
        segments.append(buffer_text.strip())

    return " ".join(segment for segment in segments if segment).strip()


def _extract_line_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("text", "transcription", "transcript", "content"):
            candidate = value.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
        return ""
    for attr in ("text", "transcription", "transcript", "content"):
        candidate = getattr(value, attr, None)
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip()
    return ""


def _coerce_mapping_like(value: Any) -> Mapping[str, Any] | None:
    if isinstance(value, Mapping):
        return value

    mapping: dict[str, Any] = {}
    for attr in ("text", "transcript", "output_text", "lines", "buffer_transcription", "buffer"):
        candidate = getattr(value, attr, None)
        if candidate is not None:
            mapping[attr] = candidate
    return mapping or None


async def _maybe_await(value: T | Awaitable[T]) -> T:
    if inspect.isawaitable(value):
        return await cast(Awaitable[T], value)
    return cast(T, value)


async def _close_results_stream(results_stream: Any) -> None:
    closer = getattr(results_stream, "aclose", None)
    if not callable(closer):
        return
    with suppress(Exception):
        await closer()


async def _close_component(component: Any, method_names: tuple[str, ...]) -> None:
    for name in method_names:
        closer = getattr(component, name, None)
        if not callable(closer):
            continue
        with suppress(Exception):
            await _maybe_await(_call_with_supported_kwargs(closer))
        return


def _call_with_supported_kwargs(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(*args, **kwargs)

    if any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()):
        return func(*args, **kwargs)

    accepted_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
    return func(*args, **accepted_kwargs)
