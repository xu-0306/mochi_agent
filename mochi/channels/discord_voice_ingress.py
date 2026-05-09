"""Discord 語音接收 ingress bridge。"""

from __future__ import annotations

import asyncio
import audioop
from collections.abc import Awaitable, Callable
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, cast

try:  # pragma: no cover - optional dependency is exercised via injection in tests.
    from discord.ext import voice_recv
except ImportError:  # pragma: no cover
    voice_recv = None  # type: ignore[assignment]


DiscordVoiceChunkHandler = Callable[[int, bytes, str | None], Awaitable[dict[str, Any]]]
DiscordVoiceTurnEndHandler = Callable[[int], Awaitable[bool]]
DiscordVoiceInterruptHandler = Callable[[int], Awaitable[int]]


@dataclass
class _IngressSession:
    guild_id: int
    voice_client: object
    sink: _BaseRuntimeSink


class DiscordVoiceIngress:
    """將 Discord remote voice 資料橋接到既有 VoiceSession buffered contract。"""

    def __init__(
        self,
        *,
        enabled: bool = False,
        sample_rate: int = 16000,
        inactivity_timeout_ms: int = 900,
        on_audio_chunk: DiscordVoiceChunkHandler,
        on_end_turn: DiscordVoiceTurnEndHandler,
        on_interrupt_input: DiscordVoiceInterruptHandler | None = None,
        recv_module: Any | None = None,
    ) -> None:
        self._enabled = enabled
        self._sample_rate = max(8_000, sample_rate)
        self._inactivity_timeout_ms = max(100, inactivity_timeout_ms)
        self._on_audio_chunk = on_audio_chunk
        self._on_end_turn = on_end_turn
        self._on_interrupt_input = on_interrupt_input
        self._recv_module = recv_module if recv_module is not None else voice_recv
        self._sessions: dict[int, _IngressSession] = {}
        self._last_error: str | None = None
        self._lock = asyncio.Lock()

    @property
    def enabled(self) -> bool:
        """回傳目前是否啟用 Discord receive ingress。"""
        return self._enabled

    def is_available(self) -> bool:
        """回傳接收擴充是否可用。"""
        return self._recv_module is not None

    def preferred_voice_client_cls(self) -> type[object] | None:
        """回傳應交給 `VoiceChannel.connect(cls=...)` 的 voice client class。"""
        if not self._enabled or self._recv_module is None:
            return None
        candidate = getattr(self._recv_module, "VoiceRecvClient", None)
        return candidate if isinstance(candidate, type) else None

    def get_status(self) -> dict[str, Any]:
        """回傳非敏感的 ingress 狀態。"""
        return {
            "enabled": self._enabled,
            "extension_available": self.is_available(),
            "active_guild_ids": sorted(self._sessions),
            "last_error": self._last_error,
            "sample_rate": self._sample_rate,
            "inactivity_timeout_ms": self._inactivity_timeout_ms,
        }

    def active_guild_ids(self) -> list[int]:
        """回傳目前已掛載 ingress 的 guild 列表。"""
        return sorted(self._sessions)

    async def attach(self, guild_id: int, voice_client: object) -> bool:
        """將 remote-audio sink 掛到指定 guild 的 voice client。"""
        if not self._enabled:
            return False
        if self._recv_module is None:
            self._last_error = (
                "discord-ext-voice-recv is not installed; Discord voice receive is unavailable."
            )
            return False

        async with self._lock:
            session = self._sessions.get(guild_id)
            if session is not None and session.voice_client is voice_client:
                if _voice_client_is_listening(voice_client):
                    return True
                session.sink.close(flush=False)
                self._sessions.pop(guild_id, None)
            elif session is not None:
                await self._detach_locked(guild_id, flush=False)

            listen = getattr(voice_client, "listen", None)
            if not callable(listen):
                self._last_error = (
                    "Discord voice client does not support receive/listen(); "
                    "connect with voice_recv.VoiceRecvClient first."
                )
                return False

            sink = _create_runtime_sink(
                recv_module=self._recv_module,
                guild_id=guild_id,
                sample_rate=self._sample_rate,
                inactivity_timeout_ms=self._inactivity_timeout_ms,
                on_audio_chunk=self._on_audio_chunk,
                on_end_turn=self._on_end_turn,
                on_error=self._set_last_error,
            )
            _stop_voice_client_listening(voice_client)
            listen(sink, after=sink.handle_listen_finished)
            self._sessions[guild_id] = _IngressSession(
                guild_id=guild_id,
                voice_client=voice_client,
                sink=sink,
            )
            self._last_error = None
            return True

    async def detach(self, guild_id: int, *, flush: bool = False) -> bool:
        """解除指定 guild 的 remote-audio sink。"""
        async with self._lock:
            return await self._detach_locked(guild_id, flush=flush)

    async def detach_all(self, *, flush: bool = False) -> int:
        """解除所有 active ingress session。"""
        async with self._lock:
            guild_ids = list(self._sessions)
            detached = 0
            for guild_id in guild_ids:
                detached += int(await self._detach_locked(guild_id, flush=flush))
            return detached

    async def _detach_locked(self, guild_id: int, *, flush: bool) -> bool:
        session = self._sessions.pop(guild_id, None)
        if session is None:
            return False
        session.sink.close(flush=flush)
        _stop_voice_client_listening(session.voice_client)
        if not flush and self._on_interrupt_input is not None:
            await self._on_interrupt_input(guild_id)
        return True

    def _set_last_error(self, message: str | None) -> None:
        self._last_error = message


class _BaseRuntimeSink:
    """最小 sink contract，供測試與 runtime bridge 共用。"""

    def close(self, *, flush: bool) -> None:
        raise NotImplementedError

    def handle_listen_finished(self, error: Exception | None) -> None:
        raise NotImplementedError


def _create_runtime_sink(
    *,
    recv_module: Any,
    guild_id: int,
    sample_rate: int,
    inactivity_timeout_ms: int,
    on_audio_chunk: DiscordVoiceChunkHandler,
    on_end_turn: DiscordVoiceTurnEndHandler,
    on_error: Callable[[str | None], None],
) -> _BaseRuntimeSink:
    base_cls = cast(type[object], recv_module.AudioSink)

    class _RuntimeSink(base_cls, _BaseRuntimeSink):  # type: ignore[misc]
        """將 Discord PCM callback 轉為 Mochi ingest/end-turn coroutine。"""

        def __init__(self) -> None:
            super().__init__()
            self._guild_id = guild_id
            self._sample_rate = sample_rate
            self._on_audio_chunk = on_audio_chunk
            self._on_end_turn = on_end_turn
            self._on_error = on_error
            self._loop = asyncio.get_running_loop()
            self._turn_timeout_seconds = inactivity_timeout_ms / 1000.0
            self._timeout_handle: asyncio.Handle | None = None
            self._active_speaker_id: str | None = None
            self._buffer_has_audio = False
            self._closed = False
            self._pending_tasks: set[asyncio.Task[Any]] = set()
            self._last_ingest_task: asyncio.Task[Any] | None = None

        def wants_opus(self) -> bool:
            return False

        def write(self, user: Any, data: Any) -> None:
            if self._closed:
                return

            pcm = getattr(data, "pcm", None)
            if isinstance(pcm, memoryview):
                pcm = pcm.tobytes()
            if not isinstance(pcm, (bytes, bytearray)):
                return

            normalized = _normalize_discord_receive_pcm(bytes(pcm), sample_rate=self._sample_rate)
            if not normalized:
                return

            speaker_id = _speaker_id_for_user(user)
            if self._active_speaker_id is None:
                self._active_speaker_id = speaker_id
            elif speaker_id is not None and self._active_speaker_id not in {None, speaker_id}:
                return

            self._buffer_has_audio = True
            self._schedule_timeout()
            self._loop.call_soon_threadsafe(self._spawn_ingest_task, normalized, speaker_id)

        def cleanup(self) -> None:
            self.close(flush=True)

        def close(self, *, flush: bool) -> None:
            if self._closed:
                return
            self._closed = True
            self._cancel_timeout()
            if flush and self._buffer_has_audio:
                self._loop.call_soon_threadsafe(self._spawn_end_turn_task)
            else:
                self._active_speaker_id = None
                self._buffer_has_audio = False

        def handle_listen_finished(self, error: Exception | None) -> None:
            if error is not None:
                self._on_error(str(error))
            self.close(flush=True)

        def _spawn_ingest_task(self, chunk: bytes, speaker_id: str | None) -> None:
            previous_task = self._last_ingest_task
            task = asyncio.create_task(
                self._forward_chunk(chunk, speaker_id, previous_task=previous_task)
            )
            self._last_ingest_task = task
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)

        def _spawn_end_turn_task(self) -> None:
            task = asyncio.create_task(self._end_turn(previous_task=self._last_ingest_task))
            self._pending_tasks.add(task)
            task.add_done_callback(self._pending_tasks.discard)

        async def _forward_chunk(
            self,
            chunk: bytes,
            speaker_id: str | None,
            *,
            previous_task: asyncio.Task[Any] | None,
        ) -> None:
            if previous_task is not None:
                with suppress(asyncio.CancelledError):
                    await previous_task
            try:
                result = await self._on_audio_chunk(self._guild_id, chunk, speaker_id)
            except Exception as exc:
                self._on_error(str(exc))
                return

            if bool(result.get("endpoint", False)):
                self._cancel_timeout()
                self._active_speaker_id = None
                self._buffer_has_audio = False

        async def _end_turn(self, *, previous_task: asyncio.Task[Any] | None) -> None:
            if previous_task is not None:
                with suppress(asyncio.CancelledError):
                    await previous_task
            if not self._buffer_has_audio:
                self._active_speaker_id = None
                return
            self._buffer_has_audio = False
            self._active_speaker_id = None
            try:
                await self._on_end_turn(self._guild_id)
            except Exception as exc:
                self._on_error(str(exc))

        def _schedule_timeout(self) -> None:
            self._cancel_timeout()
            self._timeout_handle = self._loop.call_later(
                self._turn_timeout_seconds,
                self._spawn_end_turn_task,
            )

        def _cancel_timeout(self) -> None:
            handle = self._timeout_handle
            if handle is None:
                return
            handle.cancel()
            self._timeout_handle = None

    return _RuntimeSink()


def _speaker_id_for_user(user: Any) -> str | None:
    raw_user_id = getattr(user, "id", None)
    if isinstance(raw_user_id, int):
        return str(raw_user_id)
    if isinstance(raw_user_id, str) and raw_user_id:
        return raw_user_id
    return None


def _voice_client_is_listening(voice_client: object) -> bool:
    is_listening = getattr(voice_client, "is_listening", None)
    return bool(is_listening()) if callable(is_listening) else False


def _stop_voice_client_listening(voice_client: object) -> bool:
    stop_listening = getattr(voice_client, "stop_listening", None)
    if callable(stop_listening):
        stop_listening()
        return True
    return False


def _normalize_discord_receive_pcm(audio: bytes, *, sample_rate: int) -> bytes:
    """將 Discord 48k PCM 轉為 VoiceSession 期望的 mono PCM16。"""
    if not audio:
        return b""

    normalized = audio
    if len(normalized) % 4 == 0:
        normalized = audioop.tomono(normalized, 2, 0.5, 0.5)
    elif len(normalized) % 2 != 0:
        normalized = normalized[: len(normalized) - 1]

    if sample_rate != 48_000:
        normalized, _ = audioop.ratecv(normalized, 2, 1, 48_000, sample_rate, None)
    return normalized
