"""Discord 語音 runtime（D2/D3：join/leave/playback/ingest）。"""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, cast

from mochi.voice.events import (
    AgentFinalTextEvent,
    FinalTranscriptionEvent,
    PartialTranscriptionEvent,
    SynthesizedAudioChunkEvent,
    VoiceErrorEvent,
    VoiceStageEvent,
)

DiscordVoiceConnector = Callable[[int, int], Awaitable[object]]
DiscordVoiceDisconnect = Callable[[object], Awaitable[None]]
DiscordVoiceStopPlayback = Callable[[object], Awaitable[bool]]
DiscordVoicePlayAudio = Callable[[object, bytes], Awaitable[None]]
VoiceSessionFactory = Callable[[str], Awaitable[object]]
VoiceReplySynthesizer = Callable[[str], Awaitable[bytes]]


def _utc_now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


@dataclass
class DiscordVoiceRuntimeEvent:
    """Discord voice runtime 可觀測事件。"""

    type: str
    timestamp: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": self.type,
            "timestamp": self.timestamp,
            "payload": dict(self.payload),
        }


@dataclass
class DiscordVoiceRoomState:
    """單一 Discord 語音房間狀態。"""

    guild_id: str
    channel_id: str
    session_id: str
    participant_count: int = 0
    joined_at: str | None = None
    playback_state: str = "idle"
    current_text_preview: str | None = None
    listening_state: str = "idle"
    current_speaker_id: str | None = None
    buffered_audio_bytes: int = 0
    last_partial_transcript: str | None = None
    last_final_transcript: str | None = None
    last_agent_reply: str | None = None
    last_playback_started_at: str | None = None
    last_playback_finished_at: str | None = None
    last_turn_started_at: str | None = None
    last_turn_finished_at: str | None = None
    last_playback_error: str | None = None
    last_voice_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """轉為 API 可回傳的字典。"""
        return {
            "guild_id": self.guild_id,
            "channel_id": self.channel_id,
            "session_id": self.session_id,
            "participant_count": self.participant_count,
            "joined_at": self.joined_at,
            "playback_state": self.playback_state,
            "current_text_preview": self.current_text_preview,
            "listening_state": self.listening_state,
            "current_speaker_id": self.current_speaker_id,
            "buffered_audio_bytes": self.buffered_audio_bytes,
            "last_partial_transcript": self.last_partial_transcript,
            "last_final_transcript": self.last_final_transcript,
            "last_agent_reply": self.last_agent_reply,
            "last_playback_started_at": self.last_playback_started_at,
            "last_playback_finished_at": self.last_playback_finished_at,
            "last_turn_started_at": self.last_turn_started_at,
            "last_turn_finished_at": self.last_turn_finished_at,
            "last_playback_error": self.last_playback_error,
            "last_voice_error": self.last_voice_error,
        }


@dataclass
class DiscordVoiceRoom:
    """Guild 對應的一個 active voice room。"""

    state: DiscordVoiceRoomState
    voice_client: object
    voice_session: object | None = None
    playback_task: asyncio.Task[None] | None = None
    active_turn_task: asyncio.Task[None] | None = None
    recent_events: deque[DiscordVoiceRuntimeEvent] = field(default_factory=lambda: deque(maxlen=64))


class DiscordVoiceRuntime:
    """Discord 語音 runtime。

    支援：
    - join / leave
    - status
    - interrupt playback
    - 將現有 TTS PCM16 bytes 播放到 Discord voice client
    - ingest Discord remote audio chunk，走既有 VoiceSession buffered contract
    """

    def __init__(
        self,
        *,
        enabled: bool = False,
        stt_enabled: bool = True,
        tts_enabled: bool = True,
        connect_voice_channel: DiscordVoiceConnector | None = None,
        disconnect_voice_client: DiscordVoiceDisconnect | None = None,
        stop_voice_playback: DiscordVoiceStopPlayback | None = None,
        play_audio: DiscordVoicePlayAudio | None = None,
        voice_session_factory: VoiceSessionFactory | None = None,
        reply_synthesizer: VoiceReplySynthesizer | None = None,
        recent_event_limit: int = 64,
    ) -> None:
        self._enabled = enabled
        self._stt_enabled = stt_enabled
        self._tts_enabled = tts_enabled
        self._connect_voice_channel = connect_voice_channel
        self._disconnect_voice_client = disconnect_voice_client or self._default_disconnect_voice_client
        self._stop_voice_playback = stop_voice_playback or self._default_stop_voice_playback
        self._play_audio = play_audio or self._default_play_audio
        self._voice_session_factory = voice_session_factory
        self._reply_synthesizer = reply_synthesizer
        self._rooms: dict[str, DiscordVoiceRoom] = {}
        self._reconnect_count = 0
        self._last_error: str | None = None
        self._lock = asyncio.Lock()
        self._recent_event_limit = max(8, recent_event_limit)

    def get_status(self) -> dict[str, Any]:
        """回傳非敏感 runtime 狀態。"""
        return {
            "enabled": self._enabled,
            "stt_enabled": self._stt_enabled,
            "tts_enabled": self._tts_enabled,
            "phase": "d3_voice_ingest",
            "active_voice_rooms": [room.state.to_dict() for room in self._rooms.values()],
            "active_voice_room_count": len(self._rooms),
            "recent_events": [
                event.to_dict()
                for room in self._rooms.values()
                for event in list(room.recent_events)
            ][-self._recent_event_limit :],
            "reconnect_count": self._reconnect_count,
            "last_error": self._last_error,
        }

    def configure_integrations(
        self,
        *,
        voice_session_factory: VoiceSessionFactory | None = None,
        reply_synthesizer: VoiceReplySynthesizer | None = None,
    ) -> None:
        """注入 runtime 與 AgentEngine 的整合點。"""
        if voice_session_factory is not None:
            self._voice_session_factory = voice_session_factory
        if reply_synthesizer is not None:
            self._reply_synthesizer = reply_synthesizer

    async def join_voice_channel(self, guild_id: int, channel_id: int) -> DiscordVoiceRoomState:
        """加入指定 guild 的 voice channel；每個 guild 最多一間 active room。"""
        if not self._enabled:
            raise RuntimeError("Discord voice runtime is disabled.")
        if self._connect_voice_channel is None:
            raise RuntimeError("Discord voice connector is unavailable.")

        guild_key = str(guild_id)
        async with self._lock:
            existing = self._rooms.get(guild_key)
            if existing is not None:
                if existing.state.channel_id == str(channel_id):
                    return existing.state
                await self._leave_room(existing)

            try:
                voice_client = await self._connect_voice_channel(guild_id, channel_id)
            except Exception as exc:
                self._last_error = str(exc)
                raise

            room_state = DiscordVoiceRoomState(
                guild_id=guild_key,
                channel_id=str(channel_id),
                session_id=f"discord:voice:{guild_id}:{channel_id}",
                joined_at=_utc_now_iso(),
            )
            room = DiscordVoiceRoom(
                state=room_state,
                voice_client=voice_client,
                recent_events=deque(maxlen=self._recent_event_limit),
            )
            self._rooms[guild_key] = room
            self._record_room_event(
                room,
                event_type="discord_voice_room_state",
                payload={"action": "joined", "channel_id": str(channel_id)},
            )
            self._last_error = None
            return room_state

    async def leave_voice_channel(self, guild_id: int) -> bool:
        """離開指定 guild 的 active room。"""
        guild_key = str(guild_id)
        async with self._lock:
            room = self._rooms.get(guild_key)
            if room is None:
                return False
            await self._leave_room(room)
            return True

    async def interrupt_playback(self, guild_id: int | None = None) -> bool:
        """中斷指定 guild 或全部 guild 的播放。"""
        async with self._lock:
            if guild_id is None:
                rooms = list(self._rooms.values())
            else:
                room = self._rooms.get(str(guild_id))
                rooms = [room] if room is not None else []

        interrupted = False
        for room in rooms:
            if room is None:
                continue
            interrupted = await self._interrupt_room(room) or interrupted
        return interrupted

    async def speak_text(
        self,
        guild_id: int,
        text: str,
        *,
        synthesize_audio: Callable[[str], Awaitable[bytes]] | None = None,
    ) -> bool:
        """對指定 guild 的 active room 播放 TTS 音訊。"""
        if not self._tts_enabled:
            return False
        guild_key = str(guild_id)
        async with self._lock:
            room = self._rooms.get(guild_key)
            if room is None:
                return False
            await self._cancel_room_playback(room)
            room.state.playback_state = "synthesizing"
            room.state.current_text_preview = text[:160] if text else None
            room.state.last_playback_error = None
            room.playback_task = asyncio.create_task(
                self._run_room_playback(
                    room,
                    text=text,
                    synthesize_audio=synthesize_audio or self._reply_synthesizer,
                )
            )
            return True

    async def ingest_audio_chunk(
        self,
        guild_id: int,
        *,
        chunk: bytes,
        speaker_id: str | None = None,
        auto_end: bool = True,
    ) -> dict[str, Any]:
        """將 Discord remote audio chunk 送入 room voice session。"""
        if not self._enabled:
            raise RuntimeError("Discord voice runtime is disabled.")
        if not self._stt_enabled:
            raise RuntimeError("Discord voice STT is disabled.")
        if not chunk:
            return {"accepted": False, "endpoint": False, "transcriptions": []}

        room = await self._require_room(guild_id)
        session = await self._get_or_create_voice_session(room)
        room.state.current_speaker_id = speaker_id
        room.state.listening_state = "listening"

        append_chunk_with_vad = getattr(session, "append_audio_chunk_with_vad", None)
        append_chunk = getattr(session, "append_audio_chunk", None)
        observation: object
        if callable(append_chunk_with_vad):
            observation = await _maybe_await(_call_with_supported_kwargs(
                append_chunk_with_vad,
                chunk,
                session_id=room.state.session_id,
                include_vad_state=True,
            ))
        elif callable(append_chunk):
            await _maybe_await(_call_with_supported_kwargs(append_chunk, chunk))
            observation = {"endpoint": False}
        else:
            raise RuntimeError("Voice session does not support buffered ingest.")

        endpoint, is_speech = _normalize_vad_observation(observation)
        room.state.buffered_audio_bytes += len(chunk)
        transcriptions = _extract_transcription_payloads(observation)
        for payload in transcriptions:
            await self._apply_runtime_transcription_payload(room, payload)

        self._record_room_event(
            room,
            event_type="discord_voice_room_state",
            payload={
                "action": "audio_chunk",
                "bytes": len(chunk),
                "speaker_id": speaker_id,
                "endpoint": endpoint,
                "is_speech": is_speech,
            },
        )

        if endpoint and auto_end:
            await self._start_room_turn(room)

        return {
            "accepted": True,
            "endpoint": endpoint,
            "is_speech": is_speech,
            "transcriptions": transcriptions,
            "buffered_audio_bytes": room.state.buffered_audio_bytes,
        }

    async def end_listening_turn(self, guild_id: int) -> bool:
        """明確結束目前 buffered turn。"""
        room = await self._require_room(guild_id)
        return await self._start_room_turn(room)

    async def interrupt_listening(self, guild_id: int) -> int:
        """中斷目前 room 的 buffered input。"""
        room = await self._require_room(guild_id)
        session = room.voice_session
        if session is None:
            room.state.buffered_audio_bytes = 0
            room.state.listening_state = "idle"
            return 0
        clear_buffer = getattr(session, "interrupt_buffered_input", None)
        if not callable(clear_buffer):
            return 0
        cleared = await _maybe_await(_call_with_supported_kwargs(clear_buffer))
        room.state.buffered_audio_bytes = 0
        room.state.listening_state = "idle"
        self._record_room_event(
            room,
            event_type="discord_voice_room_state",
            payload={"action": "interrupt_listening", "cleared_bytes": int(cleared or 0)},
        )
        return int(cleared or 0)

    def set_last_error(self, message: str | None) -> None:
        """更新最近錯誤。"""
        self._last_error = message

    async def _get_or_create_voice_session(self, room: DiscordVoiceRoom) -> object:
        session = room.voice_session
        if session is not None:
            return session
        if self._voice_session_factory is None:
            raise RuntimeError("Discord voice session factory is unavailable.")
        room.voice_session = await self._voice_session_factory(room.state.session_id)
        return room.voice_session

    async def _start_room_turn(self, room: DiscordVoiceRoom) -> bool:
        if room.active_turn_task is not None and not room.active_turn_task.done():
            return False

        room.state.listening_state = "processing"
        room.state.last_turn_started_at = _utc_now_iso()
        room.active_turn_task = asyncio.create_task(self._run_room_turn(room))
        return True

    async def _run_room_turn(self, room: DiscordVoiceRoom) -> None:
        try:
            session = await self._get_or_create_voice_session(room)
            consume_turn = getattr(session, "consume_buffered_turn", None)
            if not callable(consume_turn):
                raise RuntimeError("Voice session does not support consume_buffered_turn().")

            async for event in _to_async_iter(_call_with_supported_kwargs(
                consume_turn,
                session_id=room.state.session_id,
            )):
                await self._handle_voice_event(room, event)
        except Exception as exc:
            message = str(exc)
            room.state.last_voice_error = message
            self._last_error = message
            self._record_room_event(
                room,
                event_type="discord_runtime_error",
                payload={"message": message, "code": "DISCORD_VOICE_TURN_ERROR"},
            )
        finally:
            room.state.buffered_audio_bytes = 0
            room.state.current_speaker_id = None
            room.state.listening_state = "idle"
            room.state.last_turn_finished_at = _utc_now_iso()
            room.active_turn_task = None

    async def _handle_voice_event(self, room: DiscordVoiceRoom, event: object) -> None:
        if isinstance(event, VoiceStageEvent):
            if event.stage == "transcribing":
                room.state.listening_state = "transcribing"
            elif event.stage == "thinking":
                room.state.listening_state = "thinking"
            elif event.stage == "synthesizing":
                room.state.listening_state = "synthesizing"
            self._record_room_event(
                room,
                event_type="discord_voice_room_state",
                payload={"action": "voice_stage", "stage": event.stage},
            )
            return

        if isinstance(event, PartialTranscriptionEvent):
            room.state.last_partial_transcript = event.text
            self._record_room_event(
                room,
                event_type="discord_voice_partial_transcript",
                payload={"text": event.text},
            )
            return

        if isinstance(event, FinalTranscriptionEvent):
            room.state.last_final_transcript = event.text
            room.state.last_partial_transcript = event.text
            self._record_room_event(
                room,
                event_type="discord_voice_final_transcript",
                payload={"text": event.text},
            )
            return

        if isinstance(event, AgentFinalTextEvent):
            room.state.last_agent_reply = event.text
            room.state.current_text_preview = event.text[:160] if event.text else None
            self._record_room_event(
                room,
                event_type="discord_voice_assistant_reply",
                payload={"text": event.text},
            )
            return

        if isinstance(event, SynthesizedAudioChunkEvent):
            if not self._tts_enabled:
                return
            room.state.playback_state = "playing"
            if room.state.last_playback_started_at is None:
                room.state.last_playback_started_at = _utc_now_iso()
            try:
                await self._play_audio(room.voice_client, event.chunk)
            except Exception as exc:
                message = str(exc)
                room.state.playback_state = "error"
                room.state.last_playback_error = message
                self._last_error = message
                self._record_room_event(
                    room,
                    event_type="discord_runtime_error",
                    payload={"message": message, "code": "DISCORD_VOICE_PLAYBACK_ERROR"},
                )
                return
            room.state.playback_state = "idle"
            room.state.last_playback_finished_at = _utc_now_iso()
            self._record_room_event(
                room,
                event_type="discord_voice_playback_state",
                payload={"state": "played_chunk", "bytes": len(event.chunk)},
            )
            return

        if isinstance(event, VoiceErrorEvent):
            room.state.last_voice_error = event.message
            self._last_error = event.message
            self._record_room_event(
                room,
                event_type="discord_runtime_error",
                payload={"message": event.message, "code": event.code},
            )
            return

        payload = _normalize_event_payload(event)
        if payload is None:
            return
        event_type = str(payload.get("type", ""))
        if event_type == "transcription":
            await self._apply_runtime_transcription_payload(room, payload)
            return
        if event_type == "voice_stage":
            stage = str(payload.get("stage", ""))
            await self._handle_voice_event(room, VoiceStageEvent(stage=stage))  # type: ignore[arg-type]
            return
        if event_type == "text":
            await self._handle_voice_event(
                room,
                AgentFinalTextEvent(text=str(payload.get("text", ""))),
            )
            return
        if event_type == "audio_chunk":
            data = payload.get("data", b"")
            if isinstance(data, str):
                data = data.encode()
            await self._handle_voice_event(
                room,
                SynthesizedAudioChunkEvent(chunk=bytes(data)),
            )
            return
        if event_type == "error":
            await self._handle_voice_event(
                room,
                VoiceErrorEvent(
                    message=str(payload.get("message", "")),
                    code=str(payload.get("code", "VOICE_ERROR")),
                ),
            )

    async def _apply_runtime_transcription_payload(
        self,
        room: DiscordVoiceRoom,
        payload: dict[str, Any],
    ) -> None:
        text = str(payload.get("text", ""))
        is_final = bool(payload.get("is_final", False))
        if is_final:
            room.state.last_final_transcript = text
            room.state.last_partial_transcript = text
            self._record_room_event(
                room,
                event_type="discord_voice_final_transcript",
                payload={"text": text},
            )
        else:
            room.state.last_partial_transcript = text
            self._record_room_event(
                room,
                event_type="discord_voice_partial_transcript",
                payload={"text": text},
            )

    async def _run_room_playback(
        self,
        room: DiscordVoiceRoom,
        *,
        text: str,
        synthesize_audio: Callable[[str], Awaitable[bytes]] | None,
    ) -> None:
        if synthesize_audio is None:
            room.state.playback_state = "error"
            room.state.last_playback_error = "Discord reply_synthesizer is unavailable."
            self._last_error = room.state.last_playback_error
            return

        room.state.last_playback_started_at = _utc_now_iso()
        try:
            audio = await synthesize_audio(text)
            room.state.playback_state = "playing"
            await self._play_audio(room.voice_client, audio)
            room.state.playback_state = "idle"
            room.state.last_playback_finished_at = _utc_now_iso()
            room.state.last_playback_error = None
            self._last_error = None
            self._record_room_event(
                room,
                event_type="discord_voice_playback_state",
                payload={"state": "spoken_reply", "bytes": len(audio)},
            )
        except asyncio.CancelledError:
            room.state.playback_state = "interrupted"
            room.state.last_playback_finished_at = _utc_now_iso()
            raise
        except Exception as exc:
            message = str(exc)
            room.state.playback_state = "error"
            room.state.last_playback_error = message
            room.state.last_playback_finished_at = _utc_now_iso()
            self._last_error = message
            self._record_room_event(
                room,
                event_type="discord_runtime_error",
                payload={"message": message, "code": "DISCORD_VOICE_REPLY_PLAYBACK_ERROR"},
            )
        finally:
            room.playback_task = None

    async def _leave_room(self, room: DiscordVoiceRoom) -> None:
        await self._cancel_room_playback(room)
        if room.active_turn_task is not None and not room.active_turn_task.done():
            room.active_turn_task.cancel()
            with suppress_cancelled():
                await room.active_turn_task
        await self._disconnect_voice_client(room.voice_client)
        self._record_room_event(
            room,
            event_type="discord_voice_room_state",
            payload={"action": "left"},
        )
        self._rooms.pop(room.state.guild_id, None)

    async def _interrupt_room(self, room: DiscordVoiceRoom) -> bool:
        cancelled = await self._cancel_room_playback(room)
        stopped = await self._stop_voice_playback(room.voice_client)
        if cancelled or stopped:
            room.state.playback_state = "interrupted"
            room.state.last_playback_finished_at = _utc_now_iso()
            self._record_room_event(
                room,
                event_type="discord_voice_playback_state",
                payload={"state": "interrupted"},
            )
        return cancelled or stopped

    async def _cancel_room_playback(self, room: DiscordVoiceRoom) -> bool:
        task = room.playback_task
        if task is None or task.done():
            return False
        task.cancel()
        with suppress_cancelled():
            await task
        return True

    async def _require_room(self, guild_id: int) -> DiscordVoiceRoom:
        async with self._lock:
            room = self._rooms.get(str(guild_id))
        if room is None:
            raise RuntimeError(f"Discord voice room is not active in guild {guild_id}.")
        return room

    def _record_room_event(
        self,
        room: DiscordVoiceRoom,
        *,
        event_type: str,
        payload: dict[str, Any],
    ) -> None:
        room.recent_events.append(
            DiscordVoiceRuntimeEvent(
                type=event_type,
                timestamp=_utc_now_iso(),
                payload=payload,
            )
        )

    @staticmethod
    async def _default_disconnect_voice_client(voice_client: object) -> None:
        disconnect = getattr(voice_client, "disconnect", None)
        if not callable(disconnect):
            raise RuntimeError("Discord voice client does not provide disconnect().")
        result = disconnect()
        if asyncio.iscoroutine(result):
            await cast(Awaitable[Any], result)

    @staticmethod
    async def _default_stop_voice_playback(voice_client: object) -> bool:
        is_playing = getattr(voice_client, "is_playing", None)
        stop = getattr(voice_client, "stop", None)
        if not callable(stop):
            return False
        currently_playing = bool(is_playing()) if callable(is_playing) else True
        if not currently_playing:
            return False
        stop()
        return True

    @staticmethod
    async def _default_play_audio(voice_client: object, audio: bytes) -> None:
        """播放 PCM16 bytes。"""
        play_pcm = getattr(voice_client, "play_pcm", None)
        if callable(play_pcm):
            result = play_pcm(audio)
            if asyncio.iscoroutine(result):
                await cast(Awaitable[Any], result)
            return
        raise RuntimeError("Discord voice playback bridge is unavailable.")


class suppress_cancelled:
    """最小 CancelledError suppress context manager。"""

    def __enter__(self) -> None:
        return None

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return exc_type is asyncio.CancelledError


async def _maybe_await(value: Any | Awaitable[Any]) -> Any:
    if asyncio.iscoroutine(value):
        return await value
    return value


def _call_with_supported_kwargs(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    import inspect

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
    result = await _maybe_await(value)
    if hasattr(result, "__aiter__"):
        async for item in cast(AsyncIterator[object], result):
            yield item
        return
    raise TypeError("Buffered voice session must return an async iterator of events.")


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
        payload = _normalize_event_payload(item)
        if payload is None:
            continue
        payloads.append(payload)
    return payloads


def _normalize_event_payload(event: object) -> dict[str, Any] | None:
    if isinstance(event, (PartialTranscriptionEvent, FinalTranscriptionEvent)):
        return {
            "type": "transcription",
            "text": event.text,
            "is_final": event.is_final,
        }
    if isinstance(event, dict):
        payload = dict(event)
        if "type" not in payload and "text" in payload and ("is_final" in payload or "final" in payload):
            return {
                "type": "transcription",
                "text": str(payload.get("text", "")),
                "is_final": bool(payload.get("is_final", payload.get("final", False))),
            }
        return payload
    event_type = getattr(event, "type", None)
    if event_type == "transcription":
        return {
            "type": "transcription",
            "text": str(getattr(event, "text", "")),
            "is_final": bool(getattr(event, "is_final", False)),
        }
    return None
