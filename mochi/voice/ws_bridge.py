"""`/v1/voice` websocket bridge。"""

from __future__ import annotations

import asyncio
import base64
import inspect
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from typing import Any, Protocol, TypeVar, cast

from fastapi import WebSocket, WebSocketDisconnect

from mochi.voice.events import (
    AgentFinalTextEvent,
    SynthesizedAudioChunkEvent,
    TranscriptionEvent,
    VoiceErrorEvent,
    VoiceStageEvent,
)

T = TypeVar("T")
logger = logging.getLogger(__name__)


class VoiceChatEngine(Protocol):
    """提供 `voice_chat` 的最小引擎介面。"""

    def voice_chat(
        self,
        audio: bytes,
        session_id: str | None = None,
    ) -> AsyncIterator[object]:
        """執行單輪語音對話。"""

    def get_or_create_voice_session(self) -> Awaitable[object] | object:
        """取得可重用語音會話（Phase 4 buffered contract）。"""


class VoiceWebSocketBridge:
    """處理單條 websocket 連線上的多輪語音互動。"""

    _DEFAULT_AUTO_END_IDLE_TIMEOUT_SECONDS = 0.35
    _MIN_AUTO_END_IDLE_TIMEOUT_SECONDS = 0.05
    _MAX_AUTO_END_IDLE_TIMEOUT_SECONDS = 2.0

    def __init__(
        self,
        *,
        engine: VoiceChatEngine,
        session_id: str | None = None,
        auto_end_idle_timeout_seconds: float = _DEFAULT_AUTO_END_IDLE_TIMEOUT_SECONDS,
    ) -> None:
        self._engine = engine
        self._session_id = session_id
        self._buffer = bytearray()
        self._buffered_session: object | None = None
        self._send_lock = asyncio.Lock()
        self._next_turn_id = 0
        self._active_turn_id: int | None = None
        self._cancelled_turn_ids: set[int] = set()
        self._turn_tasks: dict[int, asyncio.Task[None]] = {}
        self._turn_start_lock = asyncio.Lock()
        self._audio_input_seq = 0
        self._last_started_audio_seq = -1
        self._server_vad_prev_is_speech: bool | None = None
        self._auto_end_idle_timeout_seconds = self._normalize_idle_timeout(
            auto_end_idle_timeout_seconds,
        )
        self._auto_end_task: asyncio.Task[None] | None = None
        self._bridge_buffer_mode = False
        self._preview_append_failures = 0
        self._preview_flush_failures = 0
        self._preview_degraded_turns = 0
        self._last_preview_failure: dict[str, Any] | None = None

    async def serve(self, websocket: WebSocket) -> None:
        """處理 websocket 請求直到連線中斷。"""
        await websocket.accept()

        try:
            while True:
                try:
                    message = await websocket.receive_json()
                except WebSocketDisconnect:
                    return
                except Exception as exc:
                    await self._send_error(websocket, code="BAD_REQUEST", message=str(exc))
                    continue

                if not isinstance(message, dict):
                    await self._send_error(
                        websocket,
                        code="BAD_REQUEST",
                        message="Message payload must be a JSON object.",
                    )
                    continue

                msg_type = str(message.get("type", ""))

                if msg_type == "audio_chunk":
                    await self._on_audio_chunk(websocket, message)
                    continue
                if msg_type == "vad_end":
                    await self._on_vad_end(websocket)
                    continue
                if msg_type == "interrupt":
                    await self._on_interrupt(websocket)
                    continue

                await self._send_error(
                    websocket,
                    code="UNSUPPORTED_MESSAGE",
                    message=f"Unsupported message type: {msg_type!r}.",
                )
        finally:
            await self._cancel_auto_end_task()
            await self._shutdown_turn_tasks()

    async def _on_audio_chunk(self, websocket: WebSocket, message: dict[str, Any]) -> None:
        data = message.get("data")
        if not isinstance(data, str):
            await self._send_error(
                websocket,
                code="INVALID_AUDIO_CHUNK",
                message="audio_chunk.data must be a base64 string.",
            )
            return

        try:
            chunk = base64.b64decode(data, validate=True)
        except Exception:
            await self._send_error(
                websocket,
                code="INVALID_AUDIO_CHUNK",
                message="Invalid base64 audio data.",
            )
            return

        buffered_session = await self._get_buffered_session()
        if self._bridge_buffer_mode:
            self._buffer.extend(chunk)
            self._audio_input_seq += 1
            self._schedule_auto_end_task(websocket)
            return

        if buffered_session is not None:
            append_chunk_with_vad = getattr(buffered_session, "append_audio_chunk_with_vad", None)
            if callable(append_chunk_with_vad):
                try:
                    observation = await _maybe_await(_call_with_supported_kwargs(
                        append_chunk_with_vad,
                        chunk,
                        session_id=self._session_id,
                        include_vad_state=True,
                    ))
                except Exception as exc:
                    self._record_preview_failure(stage="append", error=exc)
                    await self._append_chunk_without_preview(buffered_session, chunk)
                    self._audio_input_seq += 1
                    self._schedule_auto_end_task(websocket)
                    return
                should_end, is_speech = _normalize_vad_observation(observation)
                self._audio_input_seq += 1
                await self._emit_server_vad_state_events(
                    websocket,
                    endpoint=should_end,
                    is_speech=is_speech,
                )
                await self._emit_preview_transcriptions(
                    websocket,
                    observation=observation,
                )
                if should_end:
                    await self._cancel_auto_end_task()
                    await self._start_turn_task(websocket)
                    return
                self._schedule_auto_end_task(websocket)
                return

            append_chunk = getattr(buffered_session, "append_audio_chunk", None)
            if callable(append_chunk):
                await _maybe_await(_call_with_supported_kwargs(append_chunk, chunk))
                self._audio_input_seq += 1
                self._schedule_auto_end_task(websocket)
                return

        self._buffer.extend(chunk)
        self._audio_input_seq += 1
        self._schedule_auto_end_task(websocket)

    async def _on_vad_end(self, websocket: WebSocket) -> None:
        await self._cancel_auto_end_task()
        await self._start_turn_task(websocket)

    async def _on_interrupt(self, websocket: WebSocket) -> None:
        await self._cancel_auto_end_task()
        interrupted_turn_id = self._active_turn_id
        if interrupted_turn_id is not None:
            self._cancelled_turn_ids.add(interrupted_turn_id)
            self._active_turn_id = None
        cleared_bytes = len(self._buffer)
        self._buffer.clear()
        buffered_session = await self._get_buffered_session()
        self._bridge_buffer_mode = False
        if buffered_session is not None:
            clear_buffer = getattr(buffered_session, "interrupt_buffered_input", None)
            if callable(clear_buffer):
                session_cleared = await _maybe_await(_call_with_supported_kwargs(clear_buffer))
                if isinstance(session_cleared, int):
                    cleared_bytes += session_cleared
        await self._reset_server_vad_endpoint_state(buffered_session=buffered_session)
        await self._safe_send_json(
            websocket,
            {
                "type": "interrupted",
                "cleared_bytes": cleared_bytes,
                "turn_id": interrupted_turn_id,
            },
        )

    def _schedule_auto_end_task(self, websocket: WebSocket) -> None:
        current = self._auto_end_task
        if current is not None and not current.done():
            current.cancel()
        self._auto_end_task = asyncio.create_task(self._auto_end_after_idle(websocket))

    async def _auto_end_after_idle(self, websocket: WebSocket) -> None:
        try:
            await asyncio.sleep(self._auto_end_idle_timeout_seconds)
            await self._start_turn_task(websocket)
        except asyncio.CancelledError:
            raise
        finally:
            current = asyncio.current_task()
            if current is not None and self._auto_end_task is current:
                self._auto_end_task = None

    async def _cancel_auto_end_task(self) -> None:
        task = self._auto_end_task
        self._auto_end_task = None
        if task is None:
            return
        current = asyncio.current_task()
        if task is current:
            return
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    async def _reset_server_vad_endpoint_state(self, *, buffered_session: object | None = None) -> None:
        self._server_vad_prev_is_speech = None
        session = buffered_session
        if session is None:
            session = await self._get_buffered_session()
        if session is None:
            return

        reset_hook = getattr(session, "reset_server_vad_endpoint_state", None)
        if callable(reset_hook):
            await _maybe_await(_call_with_supported_kwargs(reset_hook))

    async def _emit_server_vad_state_events(
        self,
        websocket: WebSocket,
        *,
        endpoint: bool,
        is_speech: bool | None,
    ) -> None:
        previous = self._server_vad_prev_is_speech
        speech_started = is_speech is True and previous is not True
        speech_ended = endpoint or (previous is True and is_speech is False)

        if speech_started:
            await self._safe_send_json(
                websocket,
                {
                    "type": "vad_state",
                    "state": "speech_started",
                    "is_speech": True,
                },
            )
        if speech_ended:
            await self._safe_send_json(
                websocket,
                {
                    "type": "vad_state",
                    "state": "speech_ended",
                    "is_speech": False,
                    "endpoint": endpoint,
                },
            )

        if is_speech is not None:
            self._server_vad_prev_is_speech = is_speech
        elif endpoint:
            self._server_vad_prev_is_speech = False

    async def _start_turn_task(self, websocket: WebSocket) -> None:
        async with self._turn_start_lock:
            if self._audio_input_seq <= self._last_started_audio_seq:
                return
            if not self._bridge_buffer_mode:
                await self._flush_session_preview_transcriptions(websocket)
            await self._reset_server_vad_endpoint_state()
            self._last_started_audio_seq = self._audio_input_seq
            turn_id = self._advance_turn()
            task = asyncio.create_task(self._run_turn(websocket, turn_id))
            self._turn_tasks[turn_id] = task
            task.add_done_callback(lambda _task, done_turn_id=turn_id: self._turn_tasks.pop(done_turn_id, None))

    def _advance_turn(self) -> int:
        """推進當前 turn id，並使舊 turn 失效。"""
        previous_turn_id = self._active_turn_id
        if previous_turn_id is not None:
            self._cancelled_turn_ids.add(previous_turn_id)
        self._next_turn_id += 1
        self._active_turn_id = self._next_turn_id
        return self._next_turn_id

    async def _run_turn(self, websocket: WebSocket, turn_id: int) -> None:
        """執行單輪語音處理，並透過 turn gate 避免 stale 輸出。"""
        buffered_session = None if self._bridge_buffer_mode else await self._get_buffered_session()
        try:
            process_turn: Callable[..., Any] | None = None
            if buffered_session is not None:
                process_turn = getattr(buffered_session, "consume_buffered_turn", None)

            if callable(process_turn):
                async for event in _to_async_iter(_call_with_supported_kwargs(
                    process_turn,
                    session_id=self._session_id,
                )):
                    if not self._is_turn_active(turn_id):
                        return
                    await self._forward_event(websocket, event, turn_id=turn_id)
                if self._is_turn_active(turn_id):
                    await self._safe_send_json(websocket, {"type": "done", "turn_id": turn_id})
                return

            audio = bytes(self._buffer)
            self._buffer.clear()
            if not audio:
                if self._is_turn_active(turn_id):
                    await self._send_error(
                        websocket,
                        code="EMPTY_AUDIO",
                        message="No audio buffered for this turn.",
                        turn_id=turn_id,
                    )
                    await self._safe_send_json(websocket, {"type": "done", "turn_id": turn_id})
                return

            async for event in self._engine.voice_chat(audio, session_id=self._session_id):
                if not self._is_turn_active(turn_id):
                    return
                await self._forward_event(websocket, event, turn_id=turn_id)
            if self._is_turn_active(turn_id):
                await self._safe_send_json(websocket, {"type": "done", "turn_id": turn_id})
        except Exception as exc:
            if self._is_turn_active(turn_id):
                await self._send_error(
                    websocket,
                    code="VOICE_BRIDGE_ERROR",
                    message=str(exc),
                    turn_id=turn_id,
                )
                await self._safe_send_json(websocket, {"type": "done", "turn_id": turn_id})
        finally:
            self._bridge_buffer_mode = False
            if self._active_turn_id == turn_id:
                self._active_turn_id = None
            self._cancelled_turn_ids.discard(turn_id)

    def _is_turn_active(self, turn_id: int) -> bool:
        """判斷 turn 是否仍為可發送狀態。"""
        return (
            self._active_turn_id == turn_id
            and turn_id not in self._cancelled_turn_ids
        )

    async def _shutdown_turn_tasks(self) -> None:
        """連線結束時取消未完成 turn task。"""
        tasks = list(self._turn_tasks.values())
        self._turn_tasks.clear()
        if not tasks:
            return
        for task in tasks:
            task.cancel()
        for task in tasks:
            with suppress(asyncio.CancelledError):
                await task

    async def _get_buffered_session(self) -> object | None:
        if self._buffered_session is not None:
            return self._buffered_session

        accessor_candidates: tuple[str, ...] = ("get_or_create_voice_session", "get_voice_session")
        for name in accessor_candidates:
            accessor = getattr(self._engine, name, None)
            if accessor is None:
                continue

            if callable(accessor):
                session = await _maybe_await(_call_with_supported_kwargs(
                    accessor,
                    session_id=self._session_id,
                ))
            else:
                session = accessor

            if session is None:
                continue
            self._buffered_session = session
            return self._buffered_session

        return None

    @classmethod
    def _normalize_idle_timeout(cls, timeout_seconds: float) -> float:
        try:
            value = float(timeout_seconds)
        except (TypeError, ValueError):
            return cls._DEFAULT_AUTO_END_IDLE_TIMEOUT_SECONDS
        if value < cls._MIN_AUTO_END_IDLE_TIMEOUT_SECONDS:
            return cls._MIN_AUTO_END_IDLE_TIMEOUT_SECONDS
        if value > cls._MAX_AUTO_END_IDLE_TIMEOUT_SECONDS:
            return cls._MAX_AUTO_END_IDLE_TIMEOUT_SECONDS
        return value

    async def _forward_event(
        self,
        websocket: WebSocket,
        event: object,
        *,
        turn_id: int | None = None,
    ) -> None:
        payload = _normalize_event_payload(event)
        if payload is None:
            return
        if turn_id is not None:
            payload["turn_id"] = turn_id
        await self._safe_send_json(websocket, payload)

    async def _send_error(
        self,
        websocket: WebSocket,
        *,
        code: str,
        message: str,
        turn_id: int | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "type": "error",
            "code": code,
            "message": message,
        }
        if turn_id is not None:
            payload["turn_id"] = turn_id
        await self._safe_send_json(websocket, payload)

    async def _safe_send_json(self, websocket: WebSocket, payload: dict[str, Any]) -> None:
        """序列化 websocket send，避免多 task 併發寫入。"""
        async with self._send_lock:
            await websocket.send_json(payload)

    def get_diagnostics(self) -> dict[str, Any]:
        """回傳 bridge 累積的最小診斷資訊。"""
        return {
            "preview_append_failures": self._preview_append_failures,
            "preview_flush_failures": self._preview_flush_failures,
            "preview_degraded_turns": self._preview_degraded_turns,
            "last_preview_failure": (
                None if self._last_preview_failure is None else dict(self._last_preview_failure)
            ),
        }

    async def _emit_preview_transcriptions(
        self,
        websocket: WebSocket,
        *,
        observation: object,
    ) -> None:
        for payload in _extract_transcription_payloads(observation):
            await self._safe_send_json(websocket, payload)

    async def _append_chunk_without_preview(
        self,
        buffered_session: object | None,
        chunk: bytes,
    ) -> None:
        if buffered_session is not None:
            append_chunk = getattr(buffered_session, "append_audio_chunk", None)
            if callable(append_chunk):
                try:
                    await _maybe_await(_call_with_supported_kwargs(append_chunk, chunk))
                    self._preview_degraded_turns += 1
                    return
                except Exception as exc:
                    logger.debug(
                        "Voice preview fallback append failed; degrading to local buffer. "
                        "session_id=%s error=%s",
                        self._session_id,
                        exc,
                        exc_info=exc,
                    )

        self._bridge_buffer_mode = True
        self._preview_degraded_turns += 1
        self._buffer.extend(chunk)

    def _record_preview_failure(self, *, stage: str, error: Exception) -> None:
        if stage == "append":
            self._preview_append_failures += 1
        elif stage == "flush":
            self._preview_flush_failures += 1

        self._last_preview_failure = {
            "stage": stage,
            "error_type": type(error).__name__,
            "message": str(error),
            "session_id": self._session_id,
        }
        logger.debug(
            "Voice preview %s failed on /v1/voice. session_id=%s error=%s",
            stage,
            self._session_id,
            error,
            exc_info=error,
        )

    async def _flush_session_preview_transcriptions(self, websocket: WebSocket) -> None:
        buffered_session = await self._get_buffered_session()
        if buffered_session is None:
            return

        drain = getattr(buffered_session, "drain_transcription_preview", None)
        if not callable(drain):
            return

        try:
            result = await _maybe_await(_call_with_supported_kwargs(drain))
        except Exception as exc:
            self._record_preview_failure(stage="flush", error=exc)
            return
        async for item in _iterate_items(result):
            payload = _normalize_event_payload(item)
            if payload is None:
                continue
            await self._safe_send_json(websocket, payload)


def _normalize_event_payload(event: object) -> dict[str, Any] | None:
    """將 voice event 正規化為 websocket JSON。"""
    if isinstance(event, TranscriptionEvent):
        return _build_transcription_payload(text=event.text, is_final=event.is_final)
    if isinstance(event, VoiceStageEvent):
        return _build_voice_stage_payload(stage=event.stage)
    if isinstance(event, AgentFinalTextEvent):
        return {"type": "text", "text": event.text}
    if isinstance(event, SynthesizedAudioChunkEvent):
        return {"type": "audio_chunk", "data": base64.b64encode(event.chunk).decode("ascii")}
    if isinstance(event, VoiceErrorEvent):
        return {"type": "error", "code": event.code, "message": event.message}

    if isinstance(event, dict):
        payload = dict(event)
        event_type = str(payload.get("type", ""))
        if event_type == "transcription":
            return _build_transcription_payload(
                text=str(payload.get("text", "")),
                is_final=payload.get("is_final", payload.get("final", True)),
            )
        if not event_type and "text" in payload and ("is_final" in payload or "final" in payload):
            return _build_transcription_payload(
                text=str(payload.get("text", "")),
                is_final=payload.get("is_final", payload.get("final", True)),
            )
        if event_type == "voice_stage":
            return _build_voice_stage_payload(stage=payload.get("stage", ""))
        if event_type == "agent_final_text":
            payload["type"] = "text"
        if payload.get("type") == "audio_chunk":
            data = payload.get("data")
            if isinstance(data, (bytes, bytearray)):
                payload["data"] = base64.b64encode(bytes(data)).decode("ascii")
        return payload

    event_type = getattr(event, "type", None)
    if event_type == "transcription":
        return _build_transcription_payload(
            text=str(getattr(event, "text", "")),
            is_final=getattr(event, "is_final", getattr(event, "final", True)),
        )
    if event_type == "voice_stage":
        return _build_voice_stage_payload(stage=getattr(event, "stage", ""))
    if event_type in {"agent_final_text", "text"}:
        return {"type": "text", "text": str(getattr(event, "text", ""))}
    if event_type in {"synthesized_audio_chunk", "audio_chunk"}:
        chunk = getattr(event, "chunk", None)
        if chunk is None:
            chunk = getattr(event, "data", b"")
        data = chunk if isinstance(chunk, str) else base64.b64encode(bytes(chunk)).decode("ascii")
        return {"type": "audio_chunk", "data": data}
    if event_type == "error":
        return {
            "type": "error",
            "code": str(getattr(event, "code", "VOICE_ERROR")),
            "message": str(getattr(event, "message", "Voice error.")),
        }
    return None


def _build_transcription_payload(*, text: str, is_final: Any) -> dict[str, Any]:
    """建立統一的 transcription websocket payload。"""
    return {
        "type": "transcription",
        "text": text,
        "is_final": bool(is_final),
    }


def _build_voice_stage_payload(*, stage: object) -> dict[str, Any]:
    """建立統一的 voice_stage websocket payload。"""
    return {
        "type": "voice_stage",
        "stage": str(stage),
    }


def _normalize_vad_observation(value: object) -> tuple[bool, bool | None]:
    if isinstance(value, bool):
        return value, None

    endpoint = _extract_vad_endpoint(value)
    speech_state = _extract_vad_speech_state(value)
    return bool(endpoint), speech_state


def _extract_vad_endpoint(value: object) -> bool | None:
    if isinstance(value, dict):
        for key in ("endpoint", "is_endpoint", "utterance_end", "should_end"):
            candidate = value.get(key)
            if isinstance(candidate, bool):
                return candidate
    for attr in ("endpoint", "is_endpoint", "utterance_end", "should_end"):
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


def _extract_transcription_payloads(value: object) -> list[dict[str, Any]]:
    raw_transcriptions: object | None = None
    if isinstance(value, dict):
        raw_transcriptions = value.get("transcriptions")
    else:
        raw_transcriptions = getattr(value, "transcriptions", None)

    if isinstance(raw_transcriptions, tuple):
        raw_transcriptions = list(raw_transcriptions)
    if not isinstance(raw_transcriptions, list):
        return []

    payloads: list[dict[str, Any]] = []
    for item in raw_transcriptions:
        payload: dict[str, Any] | None = None
        if isinstance(item, dict) and "type" not in item:
            payload = _build_transcription_payload(
                text=str(item.get("text", "")),
                is_final=item.get("is_final", False),
            )
        else:
            payload = _normalize_event_payload(item)
        if payload is None:
            continue
        payloads.append(payload)
    return payloads


async def _maybe_await(value: T | Awaitable[T]) -> T:
    """接受 sync/async 回傳值。"""
    if inspect.isawaitable(value):
        return await cast(Awaitable[T], value)
    return cast(T, value)


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


async def _to_async_iter(value: Any) -> AsyncIterator[object]:
    """將可能的非同步迭代結果標準化為 AsyncIterator。"""
    result = await _maybe_await(value)
    if hasattr(result, "__aiter__"):
        async for item in cast(AsyncIterator[object], result):
            yield item
        return
    raise TypeError("Buffered voice session must return an async iterator of events.")


async def _iterate_items(value: Any) -> AsyncIterator[object]:
    """將單一值、list 或 AsyncIterator 正規化為 AsyncIterator。"""
    result = await _maybe_await(value)
    if result is None:
        return
    if hasattr(result, "__aiter__"):
        async for item in cast(AsyncIterator[object], result):
            yield item
        return
    if isinstance(result, (list, tuple)):
        for item in result:
            yield item
        return
    yield cast(object, result)
