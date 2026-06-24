"""VoiceSession 單元測試。"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from mochi.agents.events import FinalAnswerEvent
from mochi.voice.events import (
    AgentFinalTextEvent,
    SynthesizedAudioChunkEvent,
    TranscriptionEvent,
    VoiceErrorEvent,
    VoiceStageEvent,
)
from mochi.voice.voice_session import VoiceSession


class _FakeVAD:
    def __init__(self, has_speech: bool) -> None:
        self.has_speech = has_speech
        self.calls = 0

    def detect_speech(self, audio: bytes) -> bool:  # noqa: ARG002
        self.calls += 1
        return self.has_speech


class _SequenceSpeechVAD:
    def __init__(self, sequence: list[bool]) -> None:
        self._sequence = sequence
        self.calls = 0
        self.reset_calls = 0

    def is_speech(self, audio: bytes, *, sample_rate: int) -> bool:  # noqa: ARG002
        self.calls += 1
        if self.calls <= len(self._sequence):
            return self._sequence[self.calls - 1]
        return False

    def reset(self) -> None:
        self.reset_calls += 1


class _FakeSTT:
    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    def transcribe(self, audio: bytes) -> str:  # noqa: ARG002
        self.calls += 1
        return self.text


class _PreviewBatch:
    def __init__(self, transcriptions: list[object]) -> None:
        self.transcriptions = transcriptions


class _FormalPreviewSession:
    def __init__(self, session_index: int) -> None:
        self.session_index = session_index
        self.append_calls: list[bytes] = []
        self.close_calls = 0
        self._drain_calls = 0

    async def append_audio(
        self,
        audio: bytes,
        *,
        sample_rate: int,  # noqa: ARG002
        session_id: str | None = None,  # noqa: ARG002
    ) -> _PreviewBatch:
        self.append_calls.append(audio)
        text = audio.decode()
        return _PreviewBatch(
            [
                {"text": f"preview:{self.session_index}:{text}", "is_final": False},
                {"text": f"settled:{self.session_index}:{text}", "is_final": True},
            ]
        )

    async def drain_events(self) -> list[dict[str, object]]:
        self._drain_calls += 1
        if self._drain_calls == 1:
            return [
                {"text": f"flush:{self.session_index}", "is_final": False},
                {"text": f"flush-final:{self.session_index}", "is_final": True},
            ]
        return []

    async def close(self) -> None:
        self.close_calls += 1


class _FormalPreviewSTT(_FakeSTT):
    def __init__(self, text: str) -> None:
        super().__init__(text=text)
        self.preview_sessions: list[_FormalPreviewSession] = []

    async def create_preview_session(
        self,
        *,
        sample_rate: int,  # noqa: ARG002
        session_id: str | None = None,  # noqa: ARG002
        language: str | None = None,  # noqa: ARG002
    ) -> _FormalPreviewSession:
        preview_session = _FormalPreviewSession(session_index=len(self.preview_sessions) + 1)
        self.preview_sessions.append(preview_session)
        return preview_session


class _FakeTTS:
    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = chunks
        self.calls: list[str] = []

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        self.calls.append(text)
        for chunk in self.chunks:
            yield chunk


@pytest.mark.asyncio
async def test_voice_session_emits_minimal_pipeline_events() -> None:
    """應依序輸出語音階段與既有核心事件。"""
    vad = _FakeVAD(has_speech=True)
    stt = _FakeSTT(text="hello")
    tts = _FakeTTS(chunks=[b"a1", b"a2"])

    async def fake_agent_chat(message: str, session_id: str | None) -> AsyncIterator[FinalAnswerEvent]:
        assert message == "hello"
        assert session_id == "s-1"
        yield FinalAnswerEvent(content="world")

    session = VoiceSession(vad=vad, stt=stt, tts=tts, agent_chat=fake_agent_chat)

    events = [event async for event in session.handle_turn(b"audio", session_id="s-1")]

    assert len(events) == 7
    assert isinstance(events[0], VoiceStageEvent)
    assert events[0].stage == "transcribing"
    assert isinstance(events[1], TranscriptionEvent)
    assert events[1].text == "hello"
    assert events[1].is_final is True
    assert isinstance(events[2], VoiceStageEvent)
    assert events[2].stage == "thinking"
    assert isinstance(events[3], AgentFinalTextEvent)
    assert events[3].text == "world"
    assert isinstance(events[4], VoiceStageEvent)
    assert events[4].stage == "synthesizing"
    assert isinstance(events[5], SynthesizedAudioChunkEvent)
    assert events[5].chunk == b"a1"
    assert isinstance(events[6], SynthesizedAudioChunkEvent)
    assert events[6].chunk == b"a2"
    assert vad.calls == 1
    assert stt.calls == 1
    assert tts.calls == ["world"]


@pytest.mark.asyncio
async def test_voice_session_returns_no_speech_error_when_vad_rejects() -> None:
    """VAD 判定無語音時，應回傳 NO_SPEECH 並提前結束。"""
    vad = _FakeVAD(has_speech=False)
    stt = _FakeSTT(text="should-not-run")
    tts = _FakeTTS(chunks=[b"unused"])

    async def fake_agent_chat(message: str, session_id: str | None) -> AsyncIterator[FinalAnswerEvent]:  # noqa: ARG001
        yield FinalAnswerEvent(content="unused")

    session = VoiceSession(vad=vad, stt=stt, tts=tts, agent_chat=fake_agent_chat)

    events = [event async for event in session.handle_turn(b"silence")]

    assert len(events) == 1
    assert isinstance(events[0], VoiceErrorEvent)
    assert events[0].code == "NO_SPEECH"
    assert stt.calls == 0
    assert tts.calls == []


@pytest.mark.asyncio
async def test_voice_session_can_buffer_chunks_then_consume_on_end_of_turn() -> None:
    """應可先累積 chunk，再由 consume_buffered_turn 觸發單輪處理。"""
    vad = _FakeVAD(has_speech=True)
    stt = _FakeSTT(text="buffered hello")
    tts = _FakeTTS(chunks=[b"t1"])

    async def fake_agent_chat(message: str, session_id: str | None) -> AsyncIterator[FinalAnswerEvent]:
        assert message == "buffered hello"
        assert session_id == "voice-buffer-s1"
        yield FinalAnswerEvent(content="buffered world")

    session = VoiceSession(vad=vad, stt=stt, tts=tts, agent_chat=fake_agent_chat)

    size_1 = await session.append_audio_chunk(b"chunk-a")
    size_2 = await session.append_audio_chunk(b"chunk-b")
    events = [event async for event in session.consume_buffered_turn(session_id="voice-buffer-s1")]

    assert size_1 == len(b"chunk-a")
    assert size_2 == len(b"chunk-achunk-b")
    assert isinstance(events[0], VoiceStageEvent)
    assert events[0].stage == "transcribing"
    assert isinstance(events[1], TranscriptionEvent)
    assert events[1].text == "buffered hello"
    assert events[1].is_final is True
    assert isinstance(events[2], VoiceStageEvent)
    assert events[2].stage == "thinking"
    assert isinstance(events[3], AgentFinalTextEvent)
    assert events[3].text == "buffered world"
    assert isinstance(events[4], VoiceStageEvent)
    assert events[4].stage == "synthesizing"
    assert isinstance(events[5], SynthesizedAudioChunkEvent)
    assert events[5].chunk == b"t1"


@pytest.mark.asyncio
async def test_voice_session_interrupt_buffered_input_clears_pending_audio() -> None:
    """interrupt_buffered_input 應清空待處理音訊，後續 consume 回傳空緩衝錯誤。"""
    vad = _FakeVAD(has_speech=True)
    stt = _FakeSTT(text="unused")
    tts = _FakeTTS(chunks=[b"unused"])

    async def fake_agent_chat(message: str, session_id: str | None) -> AsyncIterator[FinalAnswerEvent]:  # noqa: ARG001
        yield FinalAnswerEvent(content="unused")

    session = VoiceSession(vad=vad, stt=stt, tts=tts, agent_chat=fake_agent_chat)

    await session.append_audio_chunk(b"chunk-a")
    await session.append_audio_chunk(b"chunk-b")
    cleared = await session.interrupt_buffered_input()
    events = [event async for event in session.consume_buffered_turn(session_id="voice-buffer-s2")]

    assert cleared == len(b"chunk-achunk-b")
    assert len(events) == 1
    assert isinstance(events[0], VoiceErrorEvent)
    assert events[0].code == "EMPTY_AUDIO_BUFFER"
    assert stt.calls == 0


@pytest.mark.asyncio
async def test_voice_session_append_audio_chunk_with_vad_resets_state_on_interrupt() -> None:
    """append_audio_chunk_with_vad 的 endpoint 狀態在 interrupt 後應重置。"""
    vad = _SequenceSpeechVAD(sequence=[True, False, False])
    stt = _FakeSTT(text="unused")
    tts = _FakeTTS(chunks=[b"unused"])

    async def fake_agent_chat(message: str, session_id: str | None) -> AsyncIterator[FinalAnswerEvent]:  # noqa: ARG001
        yield FinalAnswerEvent(content="unused")

    session = VoiceSession(vad=vad, stt=stt, tts=tts, agent_chat=fake_agent_chat)

    should_end_1 = await session.append_audio_chunk_with_vad(b"chunk-1")
    should_end_2 = await session.append_audio_chunk_with_vad(b"chunk-2")
    cleared = await session.interrupt_buffered_input()
    should_end_3 = await session.append_audio_chunk_with_vad(b"chunk-3")

    assert should_end_1 is False
    assert should_end_2 is True
    assert cleared == len(b"chunk-1chunk-2")
    assert vad.reset_calls == 1
    assert should_end_3 is False


@pytest.mark.asyncio
async def test_voice_session_preview_supports_formal_session_contract() -> None:
    """preview session 應支援正式 append/drain contract，且保留 final 狀態。"""
    vad = _SequenceSpeechVAD(sequence=[True])
    stt = _FormalPreviewSTT(text="unused")
    tts = _FakeTTS(chunks=[b"unused"])

    async def fake_agent_chat(message: str, session_id: str | None) -> AsyncIterator[FinalAnswerEvent]:  # noqa: ARG001
        yield FinalAnswerEvent(content="unused")

    session = VoiceSession(vad=vad, stt=stt, tts=tts, agent_chat=fake_agent_chat)

    observation = await session.append_audio_chunk_with_vad(
        b"chunk-a",
        session_id="preview-contract-s1",
        include_vad_state=True,
    )
    drained = await session.drain_transcription_preview()

    assert observation == {
        "endpoint": False,
        "is_speech": True,
        "transcriptions": [
            {"text": "preview:1:chunk-a", "is_final": False},
            {"text": "settled:1:chunk-a", "is_final": True},
        ],
    }
    assert [(event.text, event.is_final) for event in drained] == [
        ("flush:1", False),
        ("flush-final:1", True),
    ]


@pytest.mark.asyncio
async def test_voice_session_interrupt_recreates_preview_session() -> None:
    """interrupt 後應關閉舊 preview session，後續音訊建立新 session。"""
    vad = _SequenceSpeechVAD(sequence=[True, True])
    stt = _FormalPreviewSTT(text="unused")
    tts = _FakeTTS(chunks=[b"unused"])

    async def fake_agent_chat(message: str, session_id: str | None) -> AsyncIterator[FinalAnswerEvent]:  # noqa: ARG001
        yield FinalAnswerEvent(content="unused")

    session = VoiceSession(vad=vad, stt=stt, tts=tts, agent_chat=fake_agent_chat)

    await session.append_audio_chunk_with_vad(b"chunk-a")
    cleared = await session.interrupt_buffered_input()
    await session.append_audio_chunk_with_vad(b"chunk-b")

    assert cleared == len(b"chunk-a")
    assert len(stt.preview_sessions) == 2
    assert stt.preview_sessions[0].close_calls == 1
    assert stt.preview_sessions[1].append_calls == [b"chunk-b"]


@pytest.mark.asyncio
async def test_voice_session_consume_buffered_turn_resets_preview_session() -> None:
    """turn 消耗前應重置 preview session，避免跨 turn 殘留。"""
    vad = _FakeVAD(has_speech=True)
    stt = _FormalPreviewSTT(text="buffered hello")
    tts = _FakeTTS(chunks=[b"t1"])

    async def fake_agent_chat(message: str, session_id: str | None) -> AsyncIterator[FinalAnswerEvent]:
        yield FinalAnswerEvent(content="buffered world")

    session = VoiceSession(vad=vad, stt=stt, tts=tts, agent_chat=fake_agent_chat)

    await session.append_audio_chunk_with_vad(b"chunk-a")
    _ = [event async for event in session.consume_buffered_turn(session_id="voice-buffer-preview-s1")]
    await session.append_audio_chunk_with_vad(b"chunk-b")

    assert len(stt.preview_sessions) == 2
    assert stt.preview_sessions[0].close_calls == 1
    assert stt.preview_sessions[1].append_calls == [b"chunk-b"]
