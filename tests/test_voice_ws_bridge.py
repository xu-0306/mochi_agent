"""`/v1/voice` websocket bridge 測試。"""

from __future__ import annotations

import asyncio
import base64
import time
from collections.abc import AsyncIterator

from fastapi.testclient import TestClient

from mochi.api.server import app, create_app
from mochi.voice.events import (
    AgentFinalTextEvent,
    FinalTranscriptionEvent,
    PartialTranscriptionEvent,
    SynthesizedAudioChunkEvent,
    VoiceErrorEvent,
    VoiceStageEvent,
)


class _FakeVoiceSession:
    def __init__(self) -> None:
        self._buffer = bytearray()
        self.consumed_turns: list[tuple[bytes, str | None]] = []
        self.interrupt_calls = 0

    async def append_audio_chunk(self, chunk: bytes) -> int:
        self._buffer.extend(chunk)
        return len(self._buffer)

    async def consume_buffered_turn(
        self,
        session_id: str | None = None,
    ) -> AsyncIterator[object]:
        audio = bytes(self._buffer)
        self._buffer.clear()
        self.consumed_turns.append((audio, session_id))

        if not audio:
            yield VoiceErrorEvent(code="EMPTY_AUDIO_BUFFER", message="No buffered audio to process.")
            return

        text = audio.decode()
        yield FinalTranscriptionEvent(text=f"heard:{text}")
        yield AgentFinalTextEvent(text=f"reply:{text}")
        yield SynthesizedAudioChunkEvent(chunk=f"tts:{text}".encode())

    async def interrupt_buffered_input(self) -> int:
        self.interrupt_calls += 1
        size = len(self._buffer)
        self._buffer.clear()
        return size


class _PublicVadEndpointVoiceSession(_FakeVoiceSession):
    def __init__(self) -> None:
        super().__init__()
        self._endpoint_prev_is_speech = False
        self._endpoint_saw_speech = False
        self.reset_calls = 0

    @property
    def _vad(self) -> object:  # pragma: no cover - 應不被 bridge 存取
        raise AssertionError("bridge must not access _vad private field")

    @property
    def _sample_rate(self) -> int:  # pragma: no cover - 應不被 bridge 存取
        raise AssertionError("bridge must not access _sample_rate private field")

    async def append_audio_chunk_with_vad(
        self,
        chunk: bytes,
        session_id: str | None = None,  # noqa: ARG002
        *,
        include_vad_state: bool = False,
    ) -> bool | dict[str, bool]:
        is_speech = chunk != b"|"
        if is_speech:
            self._buffer.extend(chunk)
        previous_is_speech = self._endpoint_prev_is_speech
        if is_speech:
            self._endpoint_saw_speech = True
        endpoint = self._endpoint_saw_speech and self._endpoint_prev_is_speech and not is_speech
        self._endpoint_prev_is_speech = is_speech
        if include_vad_state:
            return {
                "is_speech": is_speech,
                "endpoint": endpoint,
                "speech_started": (not previous_is_speech and is_speech),
                "speech_ended": (previous_is_speech and not is_speech),
            }
        return endpoint

    async def reset_server_vad_endpoint_state(self) -> None:
        self.reset_calls += 1
        self._endpoint_prev_is_speech = False
        self._endpoint_saw_speech = False


class _DelayedVoiceSession(_FakeVoiceSession):
    def __init__(self, *, first_turn_delay_seconds: float) -> None:
        super().__init__()
        self._first_turn_delay_seconds = first_turn_delay_seconds

    async def consume_buffered_turn(
        self,
        session_id: str | None = None,
    ) -> AsyncIterator[object]:
        audio = bytes(self._buffer)
        self._buffer.clear()
        self.consumed_turns.append((audio, session_id))

        if not audio:
            yield VoiceErrorEvent(code="EMPTY_AUDIO_BUFFER", message="No buffered audio to process.")
            return

        text = audio.decode()
        yield FinalTranscriptionEvent(text=f"heard:{text}")
        if len(self.consumed_turns) == 1:
            await asyncio.sleep(self._first_turn_delay_seconds)
        yield AgentFinalTextEvent(text=f"reply:{text}")
        yield SynthesizedAudioChunkEvent(chunk=f"tts:{text}".encode())


class _PartialThenFinalVoiceSession(_FakeVoiceSession):
    async def consume_buffered_turn(
        self,
        session_id: str | None = None,
    ) -> AsyncIterator[object]:
        audio = bytes(self._buffer)
        self._buffer.clear()
        self.consumed_turns.append((audio, session_id))

        if not audio:
            yield VoiceErrorEvent(code="EMPTY_AUDIO_BUFFER", message="No buffered audio to process.")
            return

        text = audio.decode()
        yield PartialTranscriptionEvent(text=f"partial:{text}")
        yield FinalTranscriptionEvent(text=f"heard:{text}")
        yield AgentFinalTextEvent(text=f"reply:{text}")
        yield SynthesizedAudioChunkEvent(chunk=f"tts:{text}".encode())


class _PreviewingVoiceSession(_FakeVoiceSession):
    def __init__(self) -> None:
        super().__init__()
        self._preview_drain_count = 0

    async def append_audio_chunk_with_vad(
        self,
        chunk: bytes,
        session_id: str | None = None,  # noqa: ARG002
        *,
        include_vad_state: bool = False,
    ) -> bool | dict[str, object]:
        self._buffer.extend(chunk)
        text = self._buffer.decode()
        payload: dict[str, object] = {
            "endpoint": False,
            "is_speech": True,
        }
        if include_vad_state:
            payload["transcriptions"] = [
                {"text": f"preview:{text}", "is_final": False}
            ]
            return payload
        return False

    async def drain_transcription_preview(self) -> list[dict[str, object]]:
        self._preview_drain_count += 1
        if self._preview_drain_count == 1:
            return [{"text": f"flush:{self._buffer.decode()}", "is_final": False}]
        return []

    async def consume_buffered_turn(
        self,
        session_id: str | None = None,
    ) -> AsyncIterator[object]:
        audio = bytes(self._buffer)
        self._buffer.clear()
        self.consumed_turns.append((audio, session_id))

        if not audio:
            yield VoiceErrorEvent(code="EMPTY_AUDIO_BUFFER", message="No buffered audio to process.")
            return

        text = audio.decode()
        yield FinalTranscriptionEvent(text=f"heard:{text}")
        yield AgentFinalTextEvent(text=f"reply:{text}")
        yield SynthesizedAudioChunkEvent(chunk=f"tts:{text}".encode())


class _PreviewObservation:
    def __init__(
        self,
        *,
        endpoint: bool,
        is_speech: bool,
        transcriptions: list[object],
    ) -> None:
        self.endpoint = endpoint
        self.is_speech = is_speech
        self.transcriptions = transcriptions


class _FormalPreviewingVoiceSession(_FakeVoiceSession):
    def __init__(self) -> None:
        super().__init__()
        self._pending_flush_text: str | None = None
        self.reset_calls = 0

    async def append_audio_chunk_with_vad(
        self,
        chunk: bytes,
        session_id: str | None = None,  # noqa: ARG002
        *,
        include_vad_state: bool = False,
    ) -> bool | _PreviewObservation:
        self._buffer.extend(chunk)
        text = self._buffer.decode()
        self._pending_flush_text = f"flush:{text}"
        if not include_vad_state:
            return False
        return _PreviewObservation(
            endpoint=False,
            is_speech=True,
            transcriptions=[PartialTranscriptionEvent(text=f"preview:{text}")],
        )

    async def drain_transcription_preview(self) -> tuple[PartialTranscriptionEvent, ...]:
        if self._pending_flush_text is None:
            return ()
        text = self._pending_flush_text
        self._pending_flush_text = None
        return (PartialTranscriptionEvent(text=text),)

    async def interrupt_buffered_input(self) -> int:
        self._pending_flush_text = None
        return await super().interrupt_buffered_input()

    async def reset_server_vad_endpoint_state(self) -> None:
        self.reset_calls += 1


class _VoiceStageVoiceSession(_FakeVoiceSession):
    async def consume_buffered_turn(
        self,
        session_id: str | None = None,
    ) -> AsyncIterator[object]:
        audio = bytes(self._buffer)
        self._buffer.clear()
        self.consumed_turns.append((audio, session_id))

        if not audio:
            yield VoiceErrorEvent(code="EMPTY_AUDIO_BUFFER", message="No buffered audio to process.")
            return

        text = audio.decode()
        yield VoiceStageEvent(stage="transcribing")
        yield FinalTranscriptionEvent(text=f"heard:{text}")
        yield VoiceStageEvent(stage="thinking")
        yield AgentFinalTextEvent(text=f"reply:{text}")
        yield VoiceStageEvent(stage="synthesizing")
        yield SynthesizedAudioChunkEvent(chunk=f"tts:{text}".encode())


class _PreviewAppendFailureVoiceSession(_FakeVoiceSession):
    def __init__(self) -> None:
        super().__init__()
        self.preview_attempts = 0

    async def append_audio_chunk_with_vad(
        self,
        chunk: bytes,
        session_id: str | None = None,  # noqa: ARG002
        *,
        include_vad_state: bool = False,  # noqa: ARG002
    ) -> bool:
        self.preview_attempts += 1
        raise RuntimeError("preview append boom")


class _PreviewFlushFailureVoiceSession(_FormalPreviewingVoiceSession):
    async def drain_transcription_preview(self) -> tuple[PartialTranscriptionEvent, ...]:
        raise RuntimeError("preview flush boom")


class _FakeEngine:
    def __init__(self, *, voice_session: _FakeVoiceSession | None = None) -> None:
        self.voice_session = voice_session or _FakeVoiceSession()
        self.get_session_calls = 0
        self.closed = False

    async def get_or_create_voice_session(self) -> _FakeVoiceSession:
        self.get_session_calls += 1
        return self.voice_session

    async def close(self) -> None:
        self.closed = True


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def test_voice_ws_supports_repeated_turns_over_single_connection() -> None:
    """同一連線可連續執行多輪 audio_chunk -> vad_end。"""
    engine = _FakeEngine()
    app.state.engine = engine

    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/voice?session_id=voice-s1") as ws,
    ):
        ws.send_json({"type": "audio_chunk", "data": _b64(b"hel")})
        ws.send_json({"type": "audio_chunk", "data": _b64(b"lo")})
        ws.send_json({"type": "vad_end"})

        assert ws.receive_json() == {
            "type": "transcription",
            "text": "heard:hello",
            "is_final": True,
            "turn_id": 1,
        }
        assert ws.receive_json() == {"type": "text", "text": "reply:hello", "turn_id": 1}
        audio_message = ws.receive_json()
        assert audio_message["type"] == "audio_chunk"
        assert audio_message["turn_id"] == 1
        assert base64.b64decode(audio_message["data"]) == b"tts:hello"
        assert ws.receive_json() == {"type": "done", "turn_id": 1}

        ws.send_json({"type": "audio_chunk", "data": _b64(b"world")})
        ws.send_json({"type": "vad_end"})

        assert ws.receive_json() == {
            "type": "transcription",
            "text": "heard:world",
            "is_final": True,
            "turn_id": 2,
        }
        assert ws.receive_json() == {"type": "text", "text": "reply:world", "turn_id": 2}
        audio_message_2 = ws.receive_json()
        assert audio_message_2["type"] == "audio_chunk"
        assert audio_message_2["turn_id"] == 2
        assert base64.b64decode(audio_message_2["data"]) == b"tts:world"
        assert ws.receive_json() == {"type": "done", "turn_id": 2}

    assert engine.get_session_calls == 1
    assert engine.voice_session.consumed_turns == [(b"hello", "voice-s1"), (b"world", "voice-s1")]
    assert engine.closed is True


def test_voice_ws_forwards_voice_stage_in_pipeline_order() -> None:
    """voice_stage 事件應由 session 產生並由 bridge 依序轉發。"""
    engine = _FakeEngine(voice_session=_VoiceStageVoiceSession())
    app.state.engine = engine

    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/voice?session_id=voice-stage-s1") as ws,
    ):
        ws.send_json({"type": "audio_chunk", "data": _b64(b"hello")})
        ws.send_json({"type": "vad_end"})

        assert ws.receive_json() == {
            "type": "voice_stage",
            "stage": "transcribing",
            "turn_id": 1,
        }
        assert ws.receive_json() == {
            "type": "transcription",
            "text": "heard:hello",
            "is_final": True,
            "turn_id": 1,
        }
        assert ws.receive_json() == {
            "type": "voice_stage",
            "stage": "thinking",
            "turn_id": 1,
        }
        assert ws.receive_json() == {"type": "text", "text": "reply:hello", "turn_id": 1}
        assert ws.receive_json() == {
            "type": "voice_stage",
            "stage": "synthesizing",
            "turn_id": 1,
        }
        audio_message = ws.receive_json()
        assert audio_message["type"] == "audio_chunk"
        assert audio_message["turn_id"] == 1
        assert base64.b64decode(audio_message["data"]) == b"tts:hello"
        assert ws.receive_json() == {"type": "done", "turn_id": 1}


def test_voice_ws_interrupt_clears_buffered_audio() -> None:
    """收到 interrupt 後，後續 vad_end 應看到空緩衝錯誤。"""
    engine = _FakeEngine()
    app.state.engine = engine

    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/voice") as ws,
    ):
        ws.send_json({"type": "audio_chunk", "data": _b64(b"pending")})
        ws.send_json({"type": "interrupt"})
        assert ws.receive_json() == {"type": "interrupted", "cleared_bytes": 7, "turn_id": None}

        ws.send_json({"type": "vad_end"})
        assert ws.receive_json() == {
            "type": "error",
            "code": "EMPTY_AUDIO_BUFFER",
            "message": "No buffered audio to process.",
            "turn_id": 1,
        }
        assert ws.receive_json() == {"type": "done", "turn_id": 1}

    assert engine.voice_session.interrupt_calls == 1


def test_voice_ws_rejects_invalid_base64_audio_chunk() -> None:
    """audio_chunk 非法 base64 應回傳錯誤事件。"""
    engine = _FakeEngine()
    app.state.engine = engine

    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/voice") as ws,
    ):
        ws.send_json({"type": "audio_chunk", "data": "%%%not-base64%%%"})
        error_message = ws.receive_json()
        assert error_message == {
            "type": "error",
            "code": "INVALID_AUDIO_CHUNK",
            "message": "Invalid base64 audio data.",
        }


def test_voice_ws_interrupt_suppresses_inflight_stale_events() -> None:
    """interrupt 後，舊 turn 後續事件不得再回傳。"""
    engine = _FakeEngine(voice_session=_DelayedVoiceSession(first_turn_delay_seconds=0.2))
    app.state.engine = engine

    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/voice") as ws,
    ):
        ws.send_json({"type": "audio_chunk", "data": _b64(b"first")})
        ws.send_json({"type": "vad_end"})
        assert ws.receive_json() == {
            "type": "transcription",
            "text": "heard:first",
            "is_final": True,
            "turn_id": 1,
        }

        ws.send_json({"type": "interrupt"})
        assert ws.receive_json() == {"type": "interrupted", "cleared_bytes": 0, "turn_id": 1}

        ws.send_json({"type": "audio_chunk", "data": _b64(b"second")})
        ws.send_json({"type": "vad_end"})
        assert ws.receive_json() == {
            "type": "transcription",
            "text": "heard:second",
            "is_final": True,
            "turn_id": 2,
        }
        assert ws.receive_json() == {"type": "text", "text": "reply:second", "turn_id": 2}
        audio_message = ws.receive_json()
        assert audio_message["type"] == "audio_chunk"
        assert audio_message["turn_id"] == 2
        assert base64.b64decode(audio_message["data"]) == b"tts:second"
        assert ws.receive_json() == {"type": "done", "turn_id": 2}


def test_voice_ws_newer_turn_suppresses_previous_turn_tail_events() -> None:
    """新 turn 開始後，舊 turn 尚未送出的 tail event 不得再回傳。"""
    engine = _FakeEngine(voice_session=_DelayedVoiceSession(first_turn_delay_seconds=0.2))
    app.state.engine = engine

    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/voice") as ws,
    ):
        ws.send_json({"type": "audio_chunk", "data": _b64(b"first")})
        ws.send_json({"type": "vad_end"})
        assert ws.receive_json() == {
            "type": "transcription",
            "text": "heard:first",
            "is_final": True,
            "turn_id": 1,
        }

        ws.send_json({"type": "audio_chunk", "data": _b64(b"second")})
        ws.send_json({"type": "vad_end"})
        assert ws.receive_json() == {
            "type": "transcription",
            "text": "heard:second",
            "is_final": True,
            "turn_id": 2,
        }
        assert ws.receive_json() == {"type": "text", "text": "reply:second", "turn_id": 2}
        audio_message = ws.receive_json()
        assert audio_message["type"] == "audio_chunk"
        assert audio_message["turn_id"] == 2
        assert base64.b64decode(audio_message["data"]) == b"tts:second"
        assert ws.receive_json() == {"type": "done", "turn_id": 2}


def test_voice_ws_transcription_payload_distinguishes_partial_and_final() -> None:
    """transcription 事件應明確包含 is_final 狀態。"""
    engine = _FakeEngine(voice_session=_PartialThenFinalVoiceSession())
    app.state.engine = engine

    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/voice") as ws,
    ):
        ws.send_json({"type": "audio_chunk", "data": _b64(b"hello")})
        ws.send_json({"type": "vad_end"})

        assert ws.receive_json() == {
            "type": "transcription",
            "text": "partial:hello",
            "is_final": False,
            "turn_id": 1,
        }
        assert ws.receive_json() == {
            "type": "transcription",
            "text": "heard:hello",
            "is_final": True,
            "turn_id": 1,
        }
        assert ws.receive_json() == {"type": "text", "text": "reply:hello", "turn_id": 1}
        audio_message = ws.receive_json()
        assert audio_message["type"] == "audio_chunk"
        assert audio_message["turn_id"] == 1
        assert base64.b64decode(audio_message["data"]) == b"tts:hello"
        assert ws.receive_json() == {"type": "done", "turn_id": 1}


def test_voice_ws_auto_ends_turn_after_idle_timeout() -> None:
    """連續收到 audio_chunk 後，idle timeout 應自動觸發 turn。"""
    engine = _FakeEngine()
    app.state.engine = engine

    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/voice?idle_timeout_seconds=0.05") as ws,
    ):
        ws.send_json({"type": "audio_chunk", "data": _b64(b"hel")})
        ws.send_json({"type": "audio_chunk", "data": _b64(b"lo")})

        assert ws.receive_json() == {
            "type": "transcription",
            "text": "heard:hello",
            "is_final": True,
            "turn_id": 1,
        }
        assert ws.receive_json() == {"type": "text", "text": "reply:hello", "turn_id": 1}
        audio_message = ws.receive_json()
        assert audio_message["type"] == "audio_chunk"
        assert audio_message["turn_id"] == 1
        assert base64.b64decode(audio_message["data"]) == b"tts:hello"
        assert ws.receive_json() == {"type": "done", "turn_id": 1}

    assert engine.voice_session.consumed_turns == [(b"hello", None)]


def test_voice_ws_streams_preview_transcriptions_before_final_turn() -> None:
    """preview observation object 與 turn-start flush 應可被 bridge 正確轉發。"""
    engine = _FakeEngine(voice_session=_FormalPreviewingVoiceSession())
    app.state.engine = engine

    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/voice?session_id=preview-s1") as ws,
    ):
        ws.send_json({"type": "audio_chunk", "data": _b64(b"hel")})
        assert ws.receive_json() == {
            "type": "vad_state",
            "state": "speech_started",
            "is_speech": True,
        }
        assert ws.receive_json() == {
            "type": "transcription",
            "text": "preview:hel",
            "is_final": False,
        }

        ws.send_json({"type": "audio_chunk", "data": _b64(b"lo")})
        assert ws.receive_json() == {
            "type": "transcription",
            "text": "preview:hello",
            "is_final": False,
        }

        ws.send_json({"type": "vad_end"})
        assert ws.receive_json() == {
            "type": "transcription",
            "text": "flush:hello",
            "is_final": False,
        }
        assert ws.receive_json() == {
            "type": "transcription",
            "text": "heard:hello",
            "is_final": True,
            "turn_id": 1,
        }
        assert ws.receive_json() == {"type": "text", "text": "reply:hello", "turn_id": 1}
        audio_message = ws.receive_json()
        assert audio_message["type"] == "audio_chunk"
        assert audio_message["turn_id"] == 1
        assert base64.b64decode(audio_message["data"]) == b"tts:hello"
        assert ws.receive_json() == {"type": "done", "turn_id": 1}

    assert engine.voice_session.consumed_turns == [(b"hello", "preview-s1")]


def test_voice_ws_interrupt_clears_stale_preview_flush_state() -> None:
    """interrupt 後不得把舊 preview flush 到下一個 turn。"""
    voice_session = _FormalPreviewingVoiceSession()
    engine = _FakeEngine(voice_session=voice_session)
    app.state.engine = engine

    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/voice?session_id=preview-reset-s1") as ws,
    ):
        ws.send_json({"type": "audio_chunk", "data": _b64(b"old")})
        assert ws.receive_json() == {
            "type": "vad_state",
            "state": "speech_started",
            "is_speech": True,
        }
        assert ws.receive_json() == {
            "type": "transcription",
            "text": "preview:old",
            "is_final": False,
        }

        ws.send_json({"type": "interrupt"})
        assert ws.receive_json() == {"type": "interrupted", "cleared_bytes": 3, "turn_id": None}

        ws.send_json({"type": "audio_chunk", "data": _b64(b"new")})
        assert ws.receive_json() == {
            "type": "vad_state",
            "state": "speech_started",
            "is_speech": True,
        }
        assert ws.receive_json() == {
            "type": "transcription",
            "text": "preview:new",
            "is_final": False,
        }

        ws.send_json({"type": "vad_end"})
        assert ws.receive_json() == {
            "type": "transcription",
            "text": "flush:new",
            "is_final": False,
        }
        assert ws.receive_json() == {
            "type": "transcription",
            "text": "heard:new",
            "is_final": True,
            "turn_id": 1,
        }
        assert ws.receive_json() == {"type": "text", "text": "reply:new", "turn_id": 1}
        audio_message = ws.receive_json()
        assert audio_message["type"] == "audio_chunk"
        assert audio_message["turn_id"] == 1
        assert base64.b64decode(audio_message["data"]) == b"tts:new"
        assert ws.receive_json() == {"type": "done", "turn_id": 1}

    assert voice_session.consumed_turns == [(b"new", "preview-reset-s1")]
    assert voice_session.reset_calls >= 1


def test_voice_ws_server_vad_endpointing_uses_public_hook_only() -> None:
    """server-side endpointing 應透過 session public hook，且不探測 private 欄位。"""
    voice_session = _PublicVadEndpointVoiceSession()
    engine = _FakeEngine(voice_session=voice_session)
    app.state.engine = engine

    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/voice?idle_timeout_seconds=2.0") as ws,
    ):
        ws.send_json({"type": "audio_chunk", "data": _b64(b"hello")})
        ws.send_json({"type": "audio_chunk", "data": _b64(b"|")})

        assert ws.receive_json() == {
            "type": "vad_state",
            "state": "speech_started",
            "is_speech": True,
        }
        assert ws.receive_json() == {
            "type": "vad_state",
            "state": "speech_ended",
            "is_speech": False,
            "endpoint": True,
        }
        assert ws.receive_json() == {
            "type": "transcription",
            "text": "heard:hello",
            "is_final": True,
            "turn_id": 1,
        }
        assert ws.receive_json() == {"type": "text", "text": "reply:hello", "turn_id": 1}
        audio_message = ws.receive_json()
        assert audio_message["type"] == "audio_chunk"
        assert audio_message["turn_id"] == 1
        assert base64.b64decode(audio_message["data"]) == b"tts:hello"
        assert ws.receive_json() == {"type": "done", "turn_id": 1}

        ws.send_json({"type": "audio_chunk", "data": _b64(b"next")})
        assert ws.receive_json() == {
            "type": "vad_state",
            "state": "speech_started",
            "is_speech": True,
        }
        ws.send_json({"type": "interrupt"})
        assert ws.receive_json() == {"type": "interrupted", "cleared_bytes": 4, "turn_id": None}

    assert voice_session.consumed_turns == [(b"hello", None)]
    assert voice_session.reset_calls >= 2


def test_voice_ws_late_vad_end_after_auto_end_does_not_start_empty_turn() -> None:
    """auto-end 已觸發後，late vad_end 不得再開新 turn。"""
    engine = _FakeEngine()
    app.state.engine = engine

    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/voice?idle_timeout_seconds=0.05") as ws,
    ):
        ws.send_json({"type": "audio_chunk", "data": _b64(b"hello")})

        assert ws.receive_json() == {
            "type": "transcription",
            "text": "heard:hello",
            "is_final": True,
            "turn_id": 1,
        }
        assert ws.receive_json() == {"type": "text", "text": "reply:hello", "turn_id": 1}
        audio_message = ws.receive_json()
        assert audio_message["type"] == "audio_chunk"
        assert audio_message["turn_id"] == 1
        assert base64.b64decode(audio_message["data"]) == b"tts:hello"
        assert ws.receive_json() == {"type": "done", "turn_id": 1}

        ws.send_json({"type": "vad_end"})
        time.sleep(0.08)

    assert engine.voice_session.consumed_turns == [(b"hello", None)]


def test_voice_ws_interrupt_cancels_pending_auto_end_task() -> None:
    """interrupt 應取消 pending auto-end，避免 timeout 後誤觸發 turn。"""
    engine = _FakeEngine()
    app.state.engine = engine

    with (
        TestClient(app) as client,
        client.websocket_connect("/v1/voice?idle_timeout_seconds=0.05") as ws,
    ):
        ws.send_json({"type": "audio_chunk", "data": _b64(b"hello")})
        ws.send_json({"type": "interrupt"})
        assert ws.receive_json() == {"type": "interrupted", "cleared_bytes": 5, "turn_id": None}

        time.sleep(0.12)
        assert engine.voice_session.consumed_turns == []


def test_voice_ws_disconnect_cleans_pending_auto_end_task() -> None:
    """disconnect 時應清掉 pending auto-end task。"""
    engine = _FakeEngine()
    app.state.engine = engine

    with TestClient(app) as client:
        with client.websocket_connect("/v1/voice?idle_timeout_seconds=0.05") as ws:
            ws.send_json({"type": "audio_chunk", "data": _b64(b"bye")})
        time.sleep(0.12)

    assert engine.voice_session.consumed_turns == []


def test_voice_ws_preview_append_failure_degrades_to_bounded_flow(caplog) -> None:
    """preview append 失敗時應記錄診斷並維持 bounded turn。"""
    app_instance = create_app()
    engine = _FakeEngine(voice_session=_PreviewAppendFailureVoiceSession())
    app_instance.state.engine = engine

    with (
        caplog.at_level("DEBUG"),
        TestClient(app_instance) as client,
        client.websocket_connect("/v1/voice?session_id=preview-append-fail-s1") as ws,
    ):
        ws.send_json({"type": "audio_chunk", "data": _b64(b"hello")})
        ws.send_json({"type": "vad_end"})

        assert ws.receive_json() == {
            "type": "transcription",
            "text": "heard:hello",
            "is_final": True,
            "turn_id": 1,
        }
        assert ws.receive_json() == {"type": "text", "text": "reply:hello", "turn_id": 1}
        audio_message = ws.receive_json()
        assert audio_message["type"] == "audio_chunk"
        assert audio_message["turn_id"] == 1
        assert base64.b64decode(audio_message["data"]) == b"tts:hello"
        assert ws.receive_json() == {"type": "done", "turn_id": 1}

    diagnostics = app_instance.state.voice_bridge_diagnostics
    assert diagnostics["preview_append_failures"] == 1
    assert diagnostics["preview_flush_failures"] == 0
    assert diagnostics["preview_degraded_turns"] == 1
    assert diagnostics["last_preview_failure"] == {
        "stage": "append",
        "error_type": "RuntimeError",
        "message": "preview append boom",
        "session_id": "preview-append-fail-s1",
    }
    assert "Voice preview append failed on /v1/voice." in caplog.text
    assert engine.voice_session.consumed_turns == [(b"hello", "preview-append-fail-s1")]


def test_voice_ws_preview_flush_failure_is_recorded_without_breaking_turn(caplog) -> None:
    """preview flush 失敗應留下診斷，但 turn 仍完成。"""
    app_instance = create_app()
    engine = _FakeEngine(voice_session=_PreviewFlushFailureVoiceSession())
    app_instance.state.engine = engine

    with (
        caplog.at_level("DEBUG"),
        TestClient(app_instance) as client,
        client.websocket_connect("/v1/voice?session_id=preview-flush-fail-s1") as ws,
    ):
        ws.send_json({"type": "audio_chunk", "data": _b64(b"hello")})
        assert ws.receive_json() == {
            "type": "vad_state",
            "state": "speech_started",
            "is_speech": True,
        }
        assert ws.receive_json() == {
            "type": "transcription",
            "text": "preview:hello",
            "is_final": False,
        }

        ws.send_json({"type": "vad_end"})
        assert ws.receive_json() == {
            "type": "transcription",
            "text": "heard:hello",
            "is_final": True,
            "turn_id": 1,
        }
        assert ws.receive_json() == {"type": "text", "text": "reply:hello", "turn_id": 1}
        audio_message = ws.receive_json()
        assert audio_message["type"] == "audio_chunk"
        assert audio_message["turn_id"] == 1
        assert base64.b64decode(audio_message["data"]) == b"tts:hello"
        assert ws.receive_json() == {"type": "done", "turn_id": 1}

    diagnostics = app_instance.state.voice_bridge_diagnostics
    assert diagnostics["preview_append_failures"] == 0
    assert diagnostics["preview_flush_failures"] == 1
    assert diagnostics["preview_degraded_turns"] == 0
    assert diagnostics["last_preview_failure"] == {
        "stage": "flush",
        "error_type": "RuntimeError",
        "message": "preview flush boom",
        "session_id": "preview-flush-fail-s1",
    }
    assert "Voice preview flush failed on /v1/voice." in caplog.text
