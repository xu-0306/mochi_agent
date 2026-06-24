"""Discord voice runtime（D2）測試。"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from mochi.channels.discord_voice_runtime import DiscordVoiceRuntime
from mochi.voice.events import (
    AgentFinalTextEvent,
    FinalTranscriptionEvent,
    SynthesizedAudioChunkEvent,
    VoiceStageEvent,
)


class _FakeVoiceClient:
    def __init__(self) -> None:
        self.disconnected = False
        self.stop_calls = 0
        self.played: list[bytes] = []
        self._is_playing = False

    async def disconnect(self) -> None:
        self.disconnected = True

    def is_playing(self) -> bool:
        return self._is_playing

    def stop(self) -> None:
        self.stop_calls += 1
        self._is_playing = False

    async def play_pcm(self, audio: bytes) -> None:
        self.played.append(audio)
        self._is_playing = True
        await asyncio.sleep(0)
        self._is_playing = False


class _FakeBufferedVoiceSession:
    def __init__(self) -> None:
        self._buffer = bytearray()
        self.append_calls: list[bytes] = []
        self.consume_calls = 0
        self.interrupt_calls = 0

    async def append_audio_chunk_with_vad(
        self,
        chunk: bytes,
        session_id: str | None = None,  # noqa: ARG002
        *,
        include_vad_state: bool = False,
    ) -> bool | dict[str, object]:
        self.append_calls.append(chunk)
        if chunk != b"|":
            self._buffer.extend(chunk)
        payload: dict[str, object] = {
            "endpoint": chunk == b"|",
            "is_speech": chunk != b"|",
        }
        if include_vad_state and chunk != b"|":
            payload["transcriptions"] = [{"text": f"preview:{self._buffer.decode()}", "is_final": False}]
            return payload
        return chunk == b"|"

    async def consume_buffered_turn(self, session_id: str | None = None) -> AsyncIterator[object]:
        self.consume_calls += 1
        audio = bytes(self._buffer)
        self._buffer.clear()
        yield VoiceStageEvent(stage="transcribing")
        yield FinalTranscriptionEvent(text=f"heard:{audio.decode()}")
        yield VoiceStageEvent(stage="thinking")
        yield AgentFinalTextEvent(text=f"reply:{audio.decode()}")
        yield VoiceStageEvent(stage="synthesizing")
        yield SynthesizedAudioChunkEvent(chunk=f"tts:{audio.decode()}".encode())

    async def interrupt_buffered_input(self) -> int:
        self.interrupt_calls += 1
        size = len(self._buffer)
        self._buffer.clear()
        return size


@pytest.mark.asyncio
async def test_discord_voice_runtime_join_and_leave_room() -> None:
    connected: list[tuple[int, int]] = []
    voice_client = _FakeVoiceClient()

    async def _connect(guild_id: int, channel_id: int) -> object:
        connected.append((guild_id, channel_id))
        return voice_client

    runtime = DiscordVoiceRuntime(enabled=True, connect_voice_channel=_connect)

    room = await runtime.join_voice_channel(400, 500)
    left = await runtime.leave_voice_channel(400)

    assert connected == [(400, 500)]
    assert room.guild_id == "400"
    assert room.channel_id == "500"
    assert left is True
    assert voice_client.disconnected is True
    status = runtime.get_status()
    assert status["active_voice_room_count"] == 0


@pytest.mark.asyncio
async def test_discord_voice_runtime_speak_text_plays_tts_audio() -> None:
    voice_client = _FakeVoiceClient()

    async def _connect(guild_id: int, channel_id: int) -> object:  # noqa: ARG001
        return voice_client

    runtime = DiscordVoiceRuntime(enabled=True, connect_voice_channel=_connect)
    await runtime.join_voice_channel(400, 500)

    async def _synthesize(text: str) -> bytes:
        return f"tts:{text}".encode()

    spoken = await runtime.speak_text(400, "hello", synthesize_audio=_synthesize)
    await asyncio.sleep(0)

    assert spoken is True
    assert voice_client.played == [b"tts:hello"]
    status = runtime.get_status()
    room = status["active_voice_rooms"][0]
    assert room["playback_state"] in {"idle", "playing"}
    assert room["current_text_preview"] == "hello"


@pytest.mark.asyncio
async def test_discord_voice_runtime_interrupt_stops_active_playback() -> None:
    voice_client = _FakeVoiceClient()
    gate = asyncio.Event()

    async def _connect(guild_id: int, channel_id: int) -> object:  # noqa: ARG001
        return voice_client

    async def _play_audio(client: object, audio: bytes) -> None:  # noqa: ARG001
        typed = client
        assert isinstance(typed, _FakeVoiceClient)
        typed._is_playing = True
        typed.played.append(audio)
        await gate.wait()
        typed._is_playing = False

    runtime = DiscordVoiceRuntime(
        enabled=True,
        connect_voice_channel=_connect,
        play_audio=_play_audio,
    )
    await runtime.join_voice_channel(400, 500)

    async def _synthesize(text: str) -> bytes:
        return f"tts:{text}".encode()

    assert await runtime.speak_text(400, "hello", synthesize_audio=_synthesize) is True
    await asyncio.sleep(0)
    interrupted = await runtime.interrupt_playback(400)
    gate.set()
    await asyncio.sleep(0)

    assert interrupted is True
    assert voice_client.stop_calls == 1
    room = runtime.get_status()["active_voice_rooms"][0]
    assert room["playback_state"] == "interrupted"


@pytest.mark.asyncio
async def test_discord_voice_runtime_disabled_rejects_join() -> None:
    runtime = DiscordVoiceRuntime(enabled=False)

    with pytest.raises(RuntimeError, match="disabled"):
        await runtime.join_voice_channel(400, 500)


@pytest.mark.asyncio
async def test_discord_voice_runtime_ingest_audio_chunk_runs_buffered_turn_pipeline() -> None:
    voice_client = _FakeVoiceClient()
    session = _FakeBufferedVoiceSession()

    async def _connect(guild_id: int, channel_id: int) -> object:  # noqa: ARG001
        return voice_client

    async def _factory(session_id: str) -> object:
        assert session_id == "discord:voice:400:500"
        return session

    runtime = DiscordVoiceRuntime(
        enabled=True,
        stt_enabled=True,
        tts_enabled=True,
        connect_voice_channel=_connect,
        voice_session_factory=_factory,
    )
    await runtime.join_voice_channel(400, 500)

    observation = await runtime.ingest_audio_chunk(400, chunk=b"hel", speaker_id="u1")
    assert observation["accepted"] is True
    assert observation["endpoint"] is False
    assert observation["buffered_audio_bytes"] == 3

    observation = await runtime.ingest_audio_chunk(400, chunk=b"lo", speaker_id="u1")
    assert observation["accepted"] is True
    assert observation["endpoint"] is False
    assert observation["buffered_audio_bytes"] == 5

    observation = await runtime.ingest_audio_chunk(400, chunk=b"|", speaker_id="u1")
    assert observation["accepted"] is True
    assert observation["endpoint"] is True

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    status = runtime.get_status()
    room = status["active_voice_rooms"][0]
    assert room["last_partial_transcript"] == "heard:hello"
    assert room["last_final_transcript"] == "heard:hello"
    assert room["last_agent_reply"] == "reply:hello"
    assert voice_client.played == [b"tts:hello"]
    assert session.consume_calls == 1

    event_types = [event["type"] for event in status["recent_events"]]
    assert "discord_voice_partial_transcript" in event_types
    assert "discord_voice_final_transcript" in event_types
    assert "discord_voice_assistant_reply" in event_types
    assert "discord_voice_playback_state" in event_types


@pytest.mark.asyncio
async def test_discord_voice_runtime_manual_end_turn_processes_buffered_audio() -> None:
    voice_client = _FakeVoiceClient()
    session = _FakeBufferedVoiceSession()

    async def _connect(guild_id: int, channel_id: int) -> object:  # noqa: ARG001
        return voice_client

    async def _factory(session_id: str) -> object:  # noqa: ARG001
        return session

    runtime = DiscordVoiceRuntime(
        enabled=True,
        stt_enabled=True,
        tts_enabled=True,
        connect_voice_channel=_connect,
        voice_session_factory=_factory,
    )
    await runtime.join_voice_channel(400, 500)

    observation = await runtime.ingest_audio_chunk(
        400,
        chunk=b"hello",
        speaker_id="user-1",
        auto_end=False,
    )

    assert observation == {
        "accepted": True,
        "endpoint": False,
        "is_speech": True,
        "transcriptions": [
            {"type": "transcription", "text": "preview:hello", "is_final": False}
        ],
        "buffered_audio_bytes": 5,
    }
    assert session.consume_calls == 0

    status = runtime.get_status()
    room = status["active_voice_rooms"][0]
    assert room["last_partial_transcript"] == "preview:hello"
    assert room["last_final_transcript"] is None
    assert room["current_speaker_id"] == "user-1"
    assert room["listening_state"] == "listening"

    started = await runtime.end_listening_turn(400)
    assert started is True

    await asyncio.sleep(0)
    await asyncio.sleep(0)

    status = runtime.get_status()
    room = status["active_voice_rooms"][0]
    assert session.consume_calls == 1
    assert room["buffered_audio_bytes"] == 0
    assert room["current_speaker_id"] is None
    assert room["listening_state"] == "idle"
    assert room["last_final_transcript"] == "heard:hello"
    assert room["last_agent_reply"] == "reply:hello"
    assert voice_client.played == [b"tts:hello"]


@pytest.mark.asyncio
async def test_discord_voice_runtime_interrupt_listening_clears_buffer() -> None:
    voice_client = _FakeVoiceClient()
    session = _FakeBufferedVoiceSession()

    async def _connect(guild_id: int, channel_id: int) -> object:  # noqa: ARG001
        return voice_client

    async def _factory(session_id: str) -> object:  # noqa: ARG001
        return session

    runtime = DiscordVoiceRuntime(
        enabled=True,
        stt_enabled=True,
        connect_voice_channel=_connect,
        voice_session_factory=_factory,
    )
    await runtime.join_voice_channel(400, 500)
    await runtime.ingest_audio_chunk(400, chunk=b"hello")

    cleared = await runtime.interrupt_listening(400)

    assert cleared == 5
    status = runtime.get_status()
    room = status["active_voice_rooms"][0]
    assert room["buffered_audio_bytes"] == 0
    assert room["listening_state"] == "idle"
