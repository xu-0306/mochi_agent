"""單輪語音會話協調器。"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import asdict, is_dataclass
from typing import Any, Protocol, TypeVar, cast

from mochi.agents.events import AgentEvent, FinalAnswerEvent
from mochi.voice.events import (
    AgentFinalTextEvent,
    FinalTranscriptionEvent,
    PartialTranscriptionEvent,
    SynthesizedAudioChunkEvent,
    TranscriptionEvent,
    VoiceErrorEvent,
    VoiceEvent,
    VoiceStageEvent,
)

T = TypeVar("T")


class VADProtocol(Protocol):
    """VAD 介面。"""

    def detect_speech(self, audio: bytes) -> bool | Awaitable[bool]:
        """判定音訊是否有語音。"""


class STTProtocol(Protocol):
    """STT 介面。"""

    def transcribe(self, audio: bytes) -> str | Awaitable[str]:
        """將音訊轉寫為文字。"""


class TTSProtocol(Protocol):
    """TTS 介面。"""

    def synthesize(self, text: str) -> AsyncIterator[bytes]:
        """將文字轉為音訊分塊。"""


AgentChatFn = Callable[[str, str | None], AsyncIterator[AgentEvent]]


class TranscriptionPreviewSessionProtocol(Protocol):
    """增量轉寫 preview session 介面。

    正式 contract 優先使用：
    - `append_audio(...)`
    - `drain_events(...)`
    - `close()`

    為相容既有 bounded backend，仍接受 legacy：
    - `push_audio(...)`
    - `drain_texts(...)`
    """

    def append_audio(self, audio: bytes) -> object | Awaitable[object]:
        """追加新音訊並回傳可立即送出的 preview 事件。"""

    def drain_events(self) -> object | Awaitable[object]:
        """擷取目前累積的 preview 事件。"""

    def close(self) -> object | Awaitable[object]:
        """釋放 preview session 資源。"""


class VoiceSession:
    """語音會話流程協調器：支援單輪與可重用的 buffered turn。"""

    def __init__(
        self,
        vad: VADProtocol,
        stt: STTProtocol,
        tts: TTSProtocol,
        agent_chat: AgentChatFn,
        *,
        sample_rate: int = 16000,
    ) -> None:
        self._vad = vad
        self._stt = stt
        self._tts = tts
        self._agent_chat = agent_chat
        self._sample_rate = sample_rate
        self._input_buffer = bytearray()
        self._buffer_lock = asyncio.Lock()
        self._endpoint_prev_is_speech = False
        self._endpoint_saw_speech = False
        self._transcription_preview_session: object | None = None
        self._transcription_preview_lock = asyncio.Lock()
        self._transcription_preview_disabled = False

    async def append_audio_chunk(self, chunk: bytes) -> int:
        """追加一段音訊到內部緩衝，回傳當前累積位元組數。"""
        async with self._buffer_lock:
            self._input_buffer.extend(chunk)
            return len(self._input_buffer)

    async def append_audio_chunk_with_vad(
        self,
        chunk: bytes,
        session_id: str | None = None,
        *,
        include_vad_state: bool = False,
    ) -> bool | dict[str, object]:
        """追加音訊並回傳是否觸發 server-side VAD endpoint。"""
        await self.append_audio_chunk(chunk)
        endpoint, is_speech = await self._observe_audio_chunk_for_endpoint(
            chunk,
            session_id=session_id,
        )
        preview_events = await self._stream_transcription_preview(
            chunk,
            session_id=session_id,
        )
        if include_vad_state:
            payload: dict[str, bool | list[dict[str, object]]] = {"endpoint": endpoint}
            if is_speech is not None:
                payload["is_speech"] = is_speech
            if preview_events:
                payload["transcriptions"] = [
                    {"text": event.text, "is_final": event.is_final}
                    for event in preview_events
                ]
            return payload
        return endpoint

    async def consume_buffered_turn(
        self,
        session_id: str | None = None,
    ) -> AsyncIterator[VoiceEvent]:
        """在明確 end-of-turn 後消耗緩衝並執行單輪流程。"""
        async with self._buffer_lock:
            audio = bytes(self._input_buffer)
            self._input_buffer.clear()
        await self._reset_transcription_preview_session()
        await self.reset_server_vad_endpoint_state()

        if not audio:
            yield VoiceErrorEvent(message="No buffered audio to process.", code="EMPTY_AUDIO_BUFFER")
            return

        async for event in self._run_turn(audio, session_id=session_id):
            yield event

    async def interrupt_buffered_input(self) -> int:
        """安全中斷目前輸入，清空緩衝並回傳被清除的位元組數。"""
        async with self._buffer_lock:
            cleared = len(self._input_buffer)
            self._input_buffer.clear()
        await self._reset_transcription_preview_session()
        await self.reset_server_vad_endpoint_state()
        return cleared

    async def reset_server_vad_endpoint_state(self) -> None:
        """重置 server-side endpointing 狀態（含底層 VAD state）。"""
        self._endpoint_prev_is_speech = False
        self._endpoint_saw_speech = False

        reset = getattr(self._vad, "reset", None)
        if callable(reset):
            await _maybe_await(_call_with_supported_kwargs(reset))

    async def handle_turn(
        self,
        audio: bytes,
        session_id: str | None = None,
    ) -> AsyncIterator[VoiceEvent]:
        """執行單輪語音流程並輸出事件（相容既有 API）。"""
        async for event in self._run_turn(audio, session_id=session_id):
            yield event

    async def _run_turn(
        self,
        audio: bytes,
        session_id: str | None = None,
    ) -> AsyncIterator[VoiceEvent]:
        """執行單輪語音流程核心。"""
        try:
            has_speech = await self._detect_speech(audio)
        except Exception as exc:
            yield VoiceErrorEvent(message=str(exc), code="VAD_ERROR")
            return

        if not has_speech:
            yield VoiceErrorEvent(message="No speech detected.", code="NO_SPEECH")
            return

        yield VoiceStageEvent(stage="transcribing")

        try:
            transcription = await self._transcribe(audio)
        except Exception as exc:
            yield VoiceErrorEvent(message=str(exc), code="STT_ERROR")
            return

        if not transcription.strip():
            yield VoiceErrorEvent(message="Speech was detected but transcription is empty.", code="EMPTY_TRANSCRIPTION")
            return

        yield FinalTranscriptionEvent(text=transcription)

        final_text = ""
        yield VoiceStageEvent(stage="thinking")
        try:
            async for event in self._agent_chat(transcription, session_id):
                if isinstance(event, FinalAnswerEvent):
                    final_text = event.content
        except Exception as exc:
            yield VoiceErrorEvent(message=str(exc), code="AGENT_ERROR")
            return

        yield AgentFinalTextEvent(text=final_text)

        yield VoiceStageEvent(stage="synthesizing")
        try:
            async for chunk in self._synthesize(final_text):
                yield SynthesizedAudioChunkEvent(chunk=chunk)
        except Exception as exc:
            yield VoiceErrorEvent(message=str(exc), code="TTS_ERROR")

    async def _detect_speech(self, audio: bytes) -> bool:
        """同時相容 `detect_speech()` 與 `is_speech()`。"""
        detector = getattr(self._vad, "detect_speech", None)
        if callable(detector):
            return bool(await _maybe_await(detector(audio)))

        detector = getattr(self._vad, "is_speech", None)
        if callable(detector):
            return bool(await _maybe_await(_call_with_supported_kwargs(
                detector,
                audio,
                sample_rate=self._sample_rate,
            )))

        raise AttributeError("VAD must provide detect_speech() or is_speech().")

    async def _transcribe(self, audio: bytes) -> str:
        """同時相容不同 STT 方法簽名。"""
        transcribe = getattr(self._stt, "transcribe", None)
        if not callable(transcribe):
            raise AttributeError("STT must provide transcribe().")

        result = await _maybe_await(_call_with_supported_kwargs(
            transcribe,
            audio,
            sample_rate=self._sample_rate,
            language=None,
        ))
        return str(result)

    async def _synthesize(self, text: str) -> AsyncIterator[bytes]:
        """同時相容 bytes 與 AsyncIterator[bytes] 型別。"""
        synthesize = getattr(self._tts, "synthesize", None)
        if not callable(synthesize):
            raise AttributeError("TTS must provide synthesize().")

        result = await _maybe_await(_call_with_supported_kwargs(
            synthesize,
            text,
            voice=None,
            speed=None,
        ))

        if isinstance(result, (bytes, bytearray)):
            data = bytes(result)
            if data:
                yield data
            return

        if hasattr(result, "__aiter__"):
            async for chunk in cast(AsyncIterator[bytes], result):
                if chunk:
                    yield chunk
            return

        raise TypeError("TTS synthesize() must return bytes or AsyncIterator[bytes].")

    async def drain_transcription_preview(self) -> list[TranscriptionEvent]:
        """擷取目前已產生但尚未送出的 preview 轉寫。"""
        preview_session = await self._get_or_create_transcription_preview_session()
        if preview_session is None:
            return []

        try:
            raw_updates = await self._drain_transcription_preview_session(preview_session)
        except Exception:
            await self._disable_transcription_preview()
            return []

        return _coerce_transcription_preview_events(raw_updates)

    async def _observe_audio_chunk_for_endpoint(
        self,
        chunk: bytes,
        *,
        session_id: str | None = None,
    ) -> tuple[bool, bool | None]:
        endpoint, speech_state = await self._probe_endpoint_hook(chunk, session_id=session_id)
        if endpoint is not None:
            if speech_state is not None:
                if speech_state:
                    self._endpoint_saw_speech = True
                self._endpoint_prev_is_speech = speech_state
            return bool(endpoint), speech_state

        detector = getattr(self._vad, "is_speech", None)
        if not callable(detector):
            detector = getattr(self._vad, "detect_speech", None)
        if not callable(detector):
            return False, None

        is_speech = bool(await _maybe_await(_call_with_supported_kwargs(
            detector,
            chunk,
            sample_rate=self._sample_rate,
        )))
        if is_speech:
            self._endpoint_saw_speech = True

        endpoint = self._endpoint_saw_speech and self._endpoint_prev_is_speech and not is_speech
        self._endpoint_prev_is_speech = is_speech
        return endpoint, is_speech

    async def _probe_endpoint_hook(
        self,
        chunk: bytes,
        *,
        session_id: str | None = None,
    ) -> tuple[bool | None, bool | None]:
        for hook_name in ("observe_audio_chunk_for_endpoint", "observe_vad_chunk"):
            hook = getattr(self, hook_name, None)
            if not callable(hook):
                continue
            result = await _maybe_await(_call_with_supported_kwargs(
                hook,
                chunk,
                sample_rate=self._sample_rate,
                session_id=session_id,
            ))
            endpoint = _extract_vad_endpoint(result)
            speech_state = _extract_vad_speech_state(result)
            if endpoint is None and speech_state is None:
                continue
            return endpoint, speech_state
        return None, None

    async def _stream_transcription_preview(
        self,
        chunk: bytes,
        *,
        session_id: str | None = None,
    ) -> list[TranscriptionEvent]:
        preview_session = await self._get_or_create_transcription_preview_session(session_id=session_id)
        if preview_session is None:
            return []

        try:
            raw_updates = await self._append_preview_session_audio(
                preview_session,
                chunk,
                session_id=session_id,
            )
        except Exception:
            await self._disable_transcription_preview()
            return []

        return _coerce_transcription_preview_events(raw_updates)

    async def _get_or_create_transcription_preview_session(
        self,
        *,
        session_id: str | None = None,
    ) -> object | None:
        if self._transcription_preview_disabled:
            return None
        if self._transcription_preview_session is not None:
            return self._transcription_preview_session

        async with self._transcription_preview_lock:
            if self._transcription_preview_disabled:
                return None
            if self._transcription_preview_session is not None:
                return self._transcription_preview_session

            factory = self._resolve_transcription_preview_session_factory()
            if factory is None:
                self._transcription_preview_disabled = True
                return None

            try:
                preview_session = await _maybe_await(_call_with_supported_kwargs(
                    factory,
                    sample_rate=self._sample_rate,
                    session_id=session_id,
                    language=None,
                ))
            except Exception:
                self._transcription_preview_disabled = True
                return None

            if preview_session is None:
                self._transcription_preview_disabled = True
                return None

            self._transcription_preview_session = preview_session
            return self._transcription_preview_session

    def _resolve_transcription_preview_session_factory(self) -> Callable[..., Any] | None:
        """解析 STT 提供的 preview session factory。"""
        for name in ("create_transcription_preview_session", "create_preview_session"):
            factory = getattr(self._stt, name, None)
            if callable(factory):
                return cast(Callable[..., Any], factory)
        return None

    async def _append_preview_session_audio(
        self,
        preview_session: TranscriptionPreviewSessionProtocol | object,
        chunk: bytes,
        *,
        session_id: str | None = None,
    ) -> object:
        """將音訊送入 preview session，優先走正式 append contract。"""
        for method_name in ("append_audio", "push_audio"):
            append_audio = getattr(preview_session, method_name, None)
            if not callable(append_audio):
                continue
            try:
                return await _maybe_await(_call_with_supported_kwargs(
                    append_audio,
                    chunk,
                    sample_rate=self._sample_rate,
                    session_id=session_id,
                ))
            except TypeError:
                return await _maybe_await(_call_with_supported_kwargs(
                    append_audio,
                    sample_rate=self._sample_rate,
                    session_id=session_id,
                    audio=chunk,
                    chunk=chunk,
                    data=chunk,
                ))
        raise AttributeError("Transcription preview session must provide append_audio() or push_audio().")

    async def _drain_transcription_preview_session(
        self,
        preview_session: TranscriptionPreviewSessionProtocol | object,
    ) -> object:
        """從 preview session 擷取尚未送出的 preview 事件。"""
        for method_name in ("drain_events", "drain_transcriptions", "drain_texts"):
            drain = getattr(preview_session, method_name, None)
            if not callable(drain):
                continue
            return await _maybe_await(_call_with_supported_kwargs(drain))
        raise AttributeError(
            "Transcription preview session must provide drain_events(), "
            "drain_transcriptions(), or drain_texts()."
        )

    async def _reset_transcription_preview_session(self) -> None:
        preview_session = self._transcription_preview_session
        self._transcription_preview_session = None
        if preview_session is None:
            return

        close = getattr(preview_session, "close", None)
        if callable(close):
            await _maybe_await(_call_with_supported_kwargs(close))

    async def _disable_transcription_preview(self) -> None:
        self._transcription_preview_disabled = True
        await self._reset_transcription_preview_session()

    async def get_runtime_diagnostics(self) -> dict[str, Any]:
        """回傳 session 目前的最小 runtime 診斷摘要。"""
        async with self._buffer_lock:
            buffered_audio_bytes = len(self._input_buffer)

        preview_session = self._transcription_preview_session
        preview_state: dict[str, Any] | None = None
        preview_error: dict[str, str] | None = None
        if preview_session is not None:
            get_state = getattr(preview_session, "get_state", None)
            if callable(get_state):
                try:
                    raw_state = await _maybe_await(_call_with_supported_kwargs(get_state))
                except Exception as exc:
                    preview_error = {
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                else:
                    preview_state = _coerce_runtime_diagnostics_value(raw_state)

        return {
            "sample_rate": self._sample_rate,
            "buffered_audio_bytes": buffered_audio_bytes,
            "preview_disabled": self._transcription_preview_disabled,
            "preview_session": {
                "active": preview_session is not None,
                "type": (type(preview_session).__name__ if preview_session is not None else None),
                "state": preview_state,
                "state_error": preview_error,
            },
        }

    async def close(self) -> None:
        """釋放 session 內部暫存資源。"""
        await self._reset_transcription_preview_session()


def _extract_vad_endpoint(value: object) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, dict):
        for key in ("utterance_end", "is_endpoint", "endpoint", "should_end"):
            candidate = value.get(key)
            if isinstance(candidate, bool):
                return candidate
    for attr in ("utterance_end", "is_endpoint", "endpoint", "should_end"):
        candidate = getattr(value, attr, None)
        if isinstance(candidate, bool):
            return candidate
    return None


def _extract_vad_speech_state(value: object) -> bool | None:
    if isinstance(value, dict):
        for key in ("is_speech", "speech", "in_speech"):
            candidate = value.get(key)
            if isinstance(candidate, bool):
                return candidate
    for attr in ("is_speech", "speech", "in_speech"):
        candidate = getattr(value, attr, None)
        if isinstance(candidate, bool):
            return candidate
    return None


def _coerce_transcription_preview_events(value: object) -> list[TranscriptionEvent]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [PartialTranscriptionEvent(text=text)] if text else []
    if isinstance(value, TranscriptionEvent):
        return [value] if value.text.strip() else []
    if isinstance(value, dict):
        nested = value.get("transcriptions")
        if isinstance(nested, (list, tuple)):
            events: list[TranscriptionEvent] = []
            for item in nested:
                events.extend(_coerce_transcription_preview_events(item))
            return events
        text = value.get("text")
        if isinstance(text, str) and text.strip():
            is_final = bool(value.get("is_final", value.get("final", False)))
            event_cls = FinalTranscriptionEvent if is_final else PartialTranscriptionEvent
            return [event_cls(text=text.strip())]
        return []
    if isinstance(value, (list, tuple)):
        events: list[TranscriptionEvent] = []
        for item in value:
            events.extend(_coerce_transcription_preview_events(item))
        return events
    nested = getattr(value, "transcriptions", None)
    if isinstance(nested, (list, tuple)):
        events: list[TranscriptionEvent] = []
        for item in nested:
            events.extend(_coerce_transcription_preview_events(item))
        return events
    text = getattr(value, "text", None)
    if isinstance(text, str) and text.strip():
        is_final = bool(getattr(value, "is_final", getattr(value, "final", False)))
        event_cls = FinalTranscriptionEvent if is_final else PartialTranscriptionEvent
        return [event_cls(text=text.strip())]
    return []


async def _maybe_await(value: T | Awaitable[T]) -> T:
    """接受 sync/async 回傳值。"""
    if inspect.isawaitable(value):
        return await cast(Awaitable[T], value)
    return cast(T, value)


def _coerce_runtime_diagnostics_value(value: object) -> dict[str, Any]:
    """將 dataclass / mapping / object diagnostics 正規化為 dict。"""
    if is_dataclass(value):
        return cast(dict[str, Any], asdict(value))
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "__dict__"):
        return {
            key: candidate
            for key, candidate in vars(value).items()
            if not key.startswith("_")
        }
    return {"value": value}


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
