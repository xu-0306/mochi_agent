"""Voice CLI 與 Audio I/O 測試。"""

from __future__ import annotations

import asyncio
import sys
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from typer.testing import CliRunner

from mochi.main import _voice_async, app
from mochi.voice.audio_io import SoundDeviceAudioIO
from mochi.voice.events import (
    AgentFinalTextEvent,
    SynthesizedAudioChunkEvent,
    TranscriptionEvent,
    VoiceErrorEvent,
    VoiceStageEvent,
)

runner = CliRunner()


def test_voice_command_uses_async_helper(monkeypatch) -> None:
    """`mochi voice` 應將 CLI 參數傳給 async helper。"""
    called: dict[str, object | None] = {}

    async def fake_voice_async(
        *,
        config_path: str | None,
        session_id: str | None,
        max_record_seconds: float,
        playback: bool,
        input_audio: str | None,
        output_audio: str | None,
        continuous: bool = False,
        chunk_seconds: float = 0.25,
        max_turns: int = 0,
        audio_io=None,  # noqa: ANN001
    ) -> None:
        called.update(
            {
                "config_path": config_path,
                "session_id": session_id,
                "max_record_seconds": max_record_seconds,
                "playback": playback,
                "input_audio": input_audio,
                "output_audio": output_audio,
                "continuous": continuous,
                "chunk_seconds": chunk_seconds,
                "max_turns": max_turns,
                "audio_io": audio_io,
            }
        )

    monkeypatch.setattr("mochi.main._voice_async", fake_voice_async)

    result = runner.invoke(
        app,
        [
            "voice",
            "--config",
            "voice.yaml",
            "--session-id",
            "s-voice",
            "--max-record-seconds",
            "3.5",
            "--no-playback",
            "--input-audio",
            "in.pcm",
            "--output-audio",
            "out.pcm",
        ],
    )

    assert result.exit_code == 0
    assert called == {
        "config_path": "voice.yaml",
        "session_id": "s-voice",
        "max_record_seconds": 3.5,
        "playback": False,
        "input_audio": "in.pcm",
        "output_audio": "out.pcm",
        "continuous": False,
        "chunk_seconds": 0.25,
        "max_turns": 0,
        "audio_io": None,
    }


@pytest.mark.asyncio
async def test_voice_async_reads_input_audio_and_plays_synthesized_output(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """`_voice_async` 應串起 input audio、voice events、輸出檔與播放。"""

    class _FakeAudioIO:
        def __init__(self) -> None:
            self.record_calls = 0
            self.play_calls: list[tuple[bytes, int, int]] = []

        async def record_once(self, *, sample_rate: int, channels: int, max_seconds: float) -> bytes:  # noqa: ARG002
            self.record_calls += 1
            return b"should-not-be-used"

        async def play_once(self, audio: bytes, *, sample_rate: int, channels: int) -> None:
            self.play_calls.append((audio, sample_rate, channels))

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001
            self.config = config
            self.initialized = False
            self.closed = False
            self.calls: list[tuple[bytes, str | None]] = []

        async def initialize(self) -> None:
            self.initialized = True

        async def voice_chat(self, audio: bytes, session_id: str | None = None):  # type: ignore[no-untyped-def]
            self.calls.append((audio, session_id))
            yield TranscriptionEvent(text="你好")
            yield AgentFinalTextEvent(text="世界")
            yield SynthesizedAudioChunkEvent(chunk=b"a1")
            yield SynthesizedAudioChunkEvent(chunk=b"a2")

        async def close(self) -> None:
            self.closed = True

    fake_cfg = SimpleNamespace(
        voice=SimpleNamespace(sample_rate=16000, channels=1),
    )
    input_path = tmp_path / "input.pcm"
    output_path = tmp_path / "output.pcm"
    input_path.write_bytes(b"input-audio")
    fake_audio_io = _FakeAudioIO()
    fake_engine = _FakeEngine(fake_cfg)

    monkeypatch.setattr("mochi.config.manager.load_config", lambda path=None: fake_cfg)
    monkeypatch.setattr("mochi.agents.engine.AgentEngine", lambda cfg: fake_engine)

    await _voice_async(
        config_path=None,
        session_id="voice-s1",
        max_record_seconds=5.0,
        playback=True,
        input_audio=str(input_path),
        output_audio=str(output_path),
        audio_io=fake_audio_io,
    )

    assert fake_audio_io.record_calls == 0
    assert fake_engine.initialized is True
    assert fake_engine.closed is True
    assert fake_engine.calls == [(b"input-audio", "voice-s1")]
    assert output_path.read_bytes() == b"a1a2"
    assert fake_audio_io.play_calls == [(b"a1a2", 16000, 1)]


@pytest.mark.asyncio
async def test_voice_async_prints_generic_voice_runtime_status_events(monkeypatch) -> None:
    """`_voice_async` 應以通用方式顯示 voice_stage / vad_state。"""

    class _FakeAudioIO:
        async def record_once(self, *, sample_rate: int, channels: int, max_seconds: float) -> bytes:  # noqa: ARG002
            return b"pcm-in"

        async def play_once(self, audio: bytes, *, sample_rate: int, channels: int) -> None:  # noqa: ARG002
            return None

    class _FakeEngine:
        async def initialize(self) -> None:
            return None

        async def voice_chat(self, audio: bytes, session_id: str | None = None):  # type: ignore[no-untyped-def, ARG002]
            yield {"type": "vad_state", "state": "speech_started", "is_speech": True}
            yield VoiceStageEvent(stage="transcribing")
            yield TranscriptionEvent(text="你好")
            yield VoiceStageEvent(stage="thinking")
            yield AgentFinalTextEvent(text="世界")
            yield VoiceStageEvent(stage="synthesizing")
            yield SynthesizedAudioChunkEvent(chunk=b"ok")

        async def close(self) -> None:
            return None

    fake_cfg = SimpleNamespace(voice=SimpleNamespace(sample_rate=16000, channels=1))
    printed: list[str] = []

    monkeypatch.setattr("mochi.config.manager.load_config", lambda path=None: fake_cfg)
    monkeypatch.setattr("mochi.agents.engine.AgentEngine", lambda cfg: _FakeEngine())
    monkeypatch.setattr("mochi.main.console.print", lambda *args, **kwargs: printed.append(str(args[0]) if args else ""))  # noqa: ARG005

    await _voice_async(
        config_path=None,
        session_id="voice-stage",
        max_record_seconds=2.0,
        playback=False,
        input_audio=None,
        output_audio=None,
        audio_io=_FakeAudioIO(),
    )

    assert any("VAD" in line and "speech_started" in line for line in printed)
    assert any("Stage" in line and "transcribing" in line for line in printed)
    assert any("Stage" in line and "thinking" in line for line in printed)
    assert any("Stage" in line and "synthesizing" in line for line in printed)
    assert any("STT" in line and "你好" in line for line in printed)
    assert any("Agent" in line and "世界" in line for line in printed)


@pytest.mark.asyncio
async def test_voice_async_accepts_wav_input(monkeypatch, tmp_path: Path) -> None:
    """`_voice_async` 應可讀取 mono WAV（不依賴 cfg.voice.channels）並傳遞 PCM16。"""

    class _FakeAudioIO:
        async def record_once(self, *, sample_rate: int, channels: int, max_seconds: float) -> bytes:  # noqa: ARG002
            return b"unused"

        async def play_once(self, audio: bytes, *, sample_rate: int, channels: int) -> None:  # noqa: ARG002
            return None

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001, ARG002
            self.calls: list[tuple[bytes, str | None]] = []

        async def initialize(self) -> None:
            return None

        async def voice_chat(self, audio: bytes, session_id: str | None = None):  # type: ignore[no-untyped-def]
            self.calls.append((audio, session_id))
            yield SynthesizedAudioChunkEvent(chunk=b"ok")

        async def close(self) -> None:
            return None

    fake_cfg = SimpleNamespace(voice=SimpleNamespace(sample_rate=16000, channels=2))
    input_wav = tmp_path / "input.wav"
    wav_payload = b"\x01\x00\xff\x7f\x00\x80"
    with wave.open(str(input_wav), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(wav_payload)

    fake_engine = _FakeEngine(fake_cfg)
    monkeypatch.setattr("mochi.config.manager.load_config", lambda path=None: fake_cfg)
    monkeypatch.setattr("mochi.agents.engine.AgentEngine", lambda cfg: fake_engine)

    await _voice_async(
        config_path=None,
        session_id="voice-wav",
        max_record_seconds=5.0,
        playback=False,
        input_audio=str(input_wav),
        output_audio=None,
        audio_io=_FakeAudioIO(),
    )

    assert fake_engine.calls == [(wav_payload, "voice-wav")]


@pytest.mark.asyncio
async def test_voice_async_writes_wav_output_for_wav_extension(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """`_voice_async` 在 `.wav` 輸出路徑應寫入 WAV 檔。"""

    class _FakeAudioIO:
        async def record_once(self, *, sample_rate: int, channels: int, max_seconds: float) -> bytes:  # noqa: ARG002
            return b"recorded"

        async def play_once(self, audio: bytes, *, sample_rate: int, channels: int) -> None:  # noqa: ARG002
            return None

    class _FakeEngine:
        async def initialize(self) -> None:
            return None

        async def voice_chat(self, audio: bytes, session_id: str | None = None):  # type: ignore[no-untyped-def, ARG002]
            yield SynthesizedAudioChunkEvent(chunk=b"\x34\x12\x78\x56")

        async def close(self) -> None:
            return None

    fake_cfg = SimpleNamespace(voice=SimpleNamespace(sample_rate=22050, channels=2))
    output_wav = tmp_path / "output.wav"

    monkeypatch.setattr("mochi.config.manager.load_config", lambda path=None: fake_cfg)
    monkeypatch.setattr("mochi.agents.engine.AgentEngine", lambda cfg: _FakeEngine())

    await _voice_async(
        config_path=None,
        session_id="voice-wav-out",
        max_record_seconds=2.0,
        playback=False,
        input_audio=None,
        output_audio=str(output_wav),
        audio_io=_FakeAudioIO(),
    )

    with wave.open(str(output_wav), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 22050
        assert wav_file.readframes(wav_file.getnframes()) == b"\x34\x12\x78\x56"


@pytest.mark.asyncio
async def test_voice_async_rejects_non_mono_wav_input(monkeypatch, tmp_path: Path) -> None:
    """`_voice_async` 對非 mono WAV 應失敗退出。"""
    fake_cfg = SimpleNamespace(voice=SimpleNamespace(sample_rate=16000, channels=1))
    stereo_wav = tmp_path / "stereo.wav"
    with wave.open(str(stereo_wav), "wb") as wav_file:
        wav_file.setnchannels(2)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00\x00\x00")

    monkeypatch.setattr("mochi.config.manager.load_config", lambda path=None: fake_cfg)

    class _FakeEngine:
        async def initialize(self) -> None:
            return None

        async def close(self) -> None:
            return None

    monkeypatch.setattr("mochi.agents.engine.AgentEngine", lambda cfg: _FakeEngine())

    with pytest.raises(SystemExit) as exc_info:
        await _voice_async(
            config_path=None,
            session_id="voice-stereo",
            max_record_seconds=2.0,
            playback=False,
            input_audio=str(stereo_wav),
            output_audio=None,
            audio_io=None,
        )

    assert exc_info.value.code == 1


@pytest.mark.asyncio
async def test_voice_async_exits_on_voice_error_event(monkeypatch) -> None:
    """語音流程若收到 VoiceErrorEvent，應以 exit code 1 結束。"""

    class _FakeAudioIO:
        def __init__(self) -> None:
            self.play_calls = 0

        async def record_once(self, *, sample_rate: int, channels: int, max_seconds: float) -> bytes:  # noqa: ARG002
            return b"recorded"

        async def play_once(self, audio: bytes, *, sample_rate: int, channels: int) -> None:  # noqa: ARG002
            self.play_calls += 1

    class _FakeEngine:
        async def initialize(self) -> None:
            return None

        async def voice_chat(self, audio: bytes, session_id: str | None = None):  # type: ignore[no-untyped-def, ARG002]
            yield VoiceErrorEvent(message="vad failed", code="VAD_ERROR")

        async def close(self) -> None:
            return None

    fake_cfg = SimpleNamespace(
        voice=SimpleNamespace(sample_rate=16000, channels=1),
    )
    fake_audio_io = _FakeAudioIO()

    monkeypatch.setattr("mochi.config.manager.load_config", lambda path=None: fake_cfg)
    monkeypatch.setattr("mochi.agents.engine.AgentEngine", lambda cfg: _FakeEngine())

    with pytest.raises(SystemExit, match="1"):
        await _voice_async(
            config_path=None,
            session_id="voice-s2",
            max_record_seconds=2.0,
            playback=True,
            input_audio=None,
            output_audio=None,
            audio_io=fake_audio_io,
        )

    assert fake_audio_io.play_calls == 0


@pytest.mark.asyncio
async def test_voice_async_continuous_mode_streams_chunks_and_processes_turns(monkeypatch) -> None:
    """連續模式應使用 record_stream，並在 endpoint 後處理 turn。"""

    class _FakeAudioIO:
        def __init__(self) -> None:
            self.play_calls: list[tuple[bytes, int, int]] = []

        async def record_once(self, *, sample_rate: int, channels: int, max_seconds: float) -> bytes:  # noqa: ARG002
            return b"unused"

        async def record_stream(
            self,
            *,
            sample_rate: int,  # noqa: ARG002
            channels: int,  # noqa: ARG002
            chunk_seconds: float,  # noqa: ARG002
            max_seconds: float,  # noqa: ARG002
        ):
            yield b"chunk-a"
            yield b"chunk-b"
            yield b"chunk-c"

        async def play_once(self, audio: bytes, *, sample_rate: int, channels: int) -> None:
            self.play_calls.append((audio, sample_rate, channels))

    class _FakeVoiceSession:
        def __init__(self) -> None:
            self.append_calls: list[bytes] = []

        async def append_audio_chunk_with_vad(
            self,
            chunk: bytes,
            session_id: str | None = None,  # noqa: ARG002
            *,
            include_vad_state: bool = False,  # noqa: ARG002
        ) -> dict[str, bool]:
            self.append_calls.append(chunk)
            if len(self.append_calls) == 1:
                return {"endpoint": False, "is_speech": True}
            if len(self.append_calls) == 2:
                return {"endpoint": True, "is_speech": False}
            return {"endpoint": False, "is_speech": False}

        async def interrupt_buffered_input(self) -> int:
            return 0

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001, ARG002
            self.voice_session = _FakeVoiceSession()
            self.closed = False
            self.voice_calls: list[bytes] = []

        async def initialize(self) -> None:
            return None

        async def get_or_create_voice_session(self, session_id: str | None = None) -> _FakeVoiceSession:  # noqa: ARG002
            return self.voice_session

        async def voice_chat(self, audio: bytes, session_id: str | None = None):  # type: ignore[no-untyped-def, ARG002]
            self.voice_calls.append(audio)
            yield TranscriptionEvent(text="你好")
            yield AgentFinalTextEvent(text="世界")
            yield SynthesizedAudioChunkEvent(chunk=b"turn-1")

        async def close(self) -> None:
            self.closed = True

    fake_cfg = SimpleNamespace(
        voice=SimpleNamespace(sample_rate=16000, channels=1),
    )
    fake_audio_io = _FakeAudioIO()
    fake_engine = _FakeEngine(fake_cfg)

    monkeypatch.setattr("mochi.config.manager.load_config", lambda path=None: fake_cfg)
    monkeypatch.setattr("mochi.agents.engine.AgentEngine", lambda cfg: fake_engine)

    await _voice_async(
        config_path=None,
        session_id="voice-cont",
        max_record_seconds=5.0,
        playback=True,
        input_audio=None,
        output_audio=None,
        continuous=True,
        chunk_seconds=0.2,
        max_turns=1,
        audio_io=fake_audio_io,
    )

    assert fake_engine.closed is True
    assert fake_engine.voice_session.append_calls == [b"chunk-a", b"chunk-b"]
    assert fake_engine.voice_calls == [b"chunk-achunk-b"]
    assert fake_audio_io.play_calls == [(b"turn-1", 16000, 1)]


@pytest.mark.asyncio
async def test_voice_async_continuous_mode_prefers_persistent_playback_session(monkeypatch) -> None:
    """continuous playback 應優先使用長生命週期 playback session，而非每 chunk 重開一次。"""

    class _FakeAudioIO:
        def __init__(self) -> None:
            self.play_calls: list[bytes] = []
            self.session_events: list[str] = []
            self.session_chunks: list[bytes] = []

        async def record_once(self, *, sample_rate: int, channels: int, max_seconds: float) -> bytes:  # noqa: ARG002
            return b"unused"

        async def record_stream(
            self,
            *,
            sample_rate: int,  # noqa: ARG002
            channels: int,  # noqa: ARG002
            chunk_seconds: float,  # noqa: ARG002
            max_seconds: float,  # noqa: ARG002
        ):
            yield b"chunk-a"
            yield b"chunk-b"

        async def play_once(self, audio: bytes, *, sample_rate: int, channels: int) -> None:  # noqa: ARG002
            self.play_calls.append(audio)
            raise AssertionError("continuous mode should use playback_session when available")

        @asynccontextmanager
        async def playback_session(self, *, sample_rate: int, channels: int):  # noqa: ARG002, ANN202
            self.session_events.append("enter")

            async def _play(audio: bytes) -> None:
                self.session_chunks.append(audio)

            try:
                yield _play
            finally:
                self.session_events.append("exit")

    class _FakeVoiceSession:
        def __init__(self) -> None:
            self.append_calls: list[bytes] = []

        async def append_audio_chunk_with_vad(
            self,
            chunk: bytes,
            session_id: str | None = None,  # noqa: ARG002
            *,
            include_vad_state: bool = False,  # noqa: ARG002
        ) -> dict[str, bool]:
            self.append_calls.append(chunk)
            if len(self.append_calls) == 1:
                return {"endpoint": False, "is_speech": True}
            return {"endpoint": True, "is_speech": False}

        async def interrupt_buffered_input(self) -> int:
            return 0

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001, ARG002
            self.voice_session = _FakeVoiceSession()
            self.closed = False

        async def initialize(self) -> None:
            return None

        async def get_or_create_voice_session(self, session_id: str | None = None) -> _FakeVoiceSession:  # noqa: ARG002
            return self.voice_session

        async def voice_chat(self, audio: bytes, session_id: str | None = None):  # type: ignore[no-untyped-def, ARG002]
            yield SynthesizedAudioChunkEvent(chunk=b"turn-1-a")
            yield SynthesizedAudioChunkEvent(chunk=b"turn-1-b")

        async def close(self) -> None:
            self.closed = True

    fake_cfg = SimpleNamespace(voice=SimpleNamespace(sample_rate=16000, channels=1))
    fake_audio_io = _FakeAudioIO()
    fake_engine = _FakeEngine(fake_cfg)

    monkeypatch.setattr("mochi.config.manager.load_config", lambda path=None: fake_cfg)
    monkeypatch.setattr("mochi.agents.engine.AgentEngine", lambda cfg: fake_engine)

    await _voice_async(
        config_path=None,
        session_id="voice-cont-persistent-playback",
        max_record_seconds=2.0,
        playback=True,
        input_audio=None,
        output_audio=None,
        continuous=True,
        chunk_seconds=0.2,
        max_turns=1,
        audio_io=fake_audio_io,
    )

    assert fake_engine.closed is True
    assert fake_audio_io.play_calls == []
    assert fake_audio_io.session_events == ["enter", "exit"]
    assert fake_audio_io.session_chunks == [b"turn-1-a", b"turn-1-b"]


@pytest.mark.asyncio
async def test_voice_async_continuous_mode_queues_next_turn_until_current_turn_done(monkeypatch) -> None:
    """使用者在舊回合播放中說話時，舊 TTS 應播完且新 utterance 排隊。"""

    class _FakeAudioIO:
        def __init__(self) -> None:
            self.play_calls: list[bytes] = []
            self.stop_calls = 0

        async def record_once(self, *, sample_rate: int, channels: int, max_seconds: float) -> bytes:  # noqa: ARG002
            return b"unused"

        async def record_stream(
            self,
            *,
            sample_rate: int,  # noqa: ARG002
            channels: int,  # noqa: ARG002
            chunk_seconds: float,  # noqa: ARG002
            max_seconds: float,  # noqa: ARG002
        ):
            yield b"u1-speech-start"
            yield b"u1-endpoint"
            await asyncio.sleep(0.05)
            yield b"u2-speech-start"
            yield b"u2-endpoint"
            await asyncio.sleep(0.05)

        async def play_once(self, audio: bytes, *, sample_rate: int, channels: int) -> None:  # noqa: ARG002
            self.play_calls.append(audio)
            await asyncio.sleep(0.03)

        async def stop_playback(self) -> bool:
            self.stop_calls += 1
            return True

    class _FakeVoiceSession:
        def __init__(self) -> None:
            self.append_calls = 0
            self.reset_calls = 0

        async def append_audio_chunk_with_vad(
            self,
            chunk: bytes,  # noqa: ARG002
            session_id: str | None = None,  # noqa: ARG002
            *,
            include_vad_state: bool = False,  # noqa: ARG002
        ) -> dict[str, bool]:
            self.append_calls += 1
            mapping = {
                1: {"endpoint": False, "is_speech": True},
                2: {"endpoint": True, "is_speech": False},
                3: {"endpoint": False, "is_speech": True},
                4: {"endpoint": True, "is_speech": False},
            }
            return mapping[self.append_calls]

        async def interrupt_buffered_input(self) -> int:
            self.reset_calls += 1
            return 0

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001, ARG002
            self.voice_session = _FakeVoiceSession()
            self.closed = False
            self.voice_calls: list[bytes] = []
            self.turn_start_play_lengths: list[int] = []

        async def initialize(self) -> None:
            return None

        async def get_or_create_voice_session(self, session_id: str | None = None) -> _FakeVoiceSession:  # noqa: ARG002
            return self.voice_session

        async def voice_chat(self, audio: bytes, session_id: str | None = None):  # type: ignore[no-untyped-def, ARG002]
            self.voice_calls.append(audio)
            self.turn_start_play_lengths.append(len(fake_io.play_calls))
            if len(self.voice_calls) == 1:
                await asyncio.sleep(0.03)
                yield TranscriptionEvent(text="old-transcript")
                yield AgentFinalTextEvent(text="old-agent")
                yield SynthesizedAudioChunkEvent(chunk=b"old-audio-1")
                yield SynthesizedAudioChunkEvent(chunk=b"old-audio-2")
                return
            yield TranscriptionEvent(text="new-transcript")
            yield AgentFinalTextEvent(text="new-agent")
            yield SynthesizedAudioChunkEvent(chunk=b"new-audio")

        async def close(self) -> None:
            self.closed = True

    fake_cfg = SimpleNamespace(voice=SimpleNamespace(sample_rate=16000, channels=1))
    fake_io = _FakeAudioIO()
    fake_engine = _FakeEngine(fake_cfg)
    printed: list[str] = []

    monkeypatch.setattr("mochi.config.manager.load_config", lambda path=None: fake_cfg)
    monkeypatch.setattr("mochi.agents.engine.AgentEngine", lambda cfg: fake_engine)
    monkeypatch.setattr(
        "mochi.main.console.print",
        lambda *args, **kwargs: printed.append(str(args[0]) if args else ""),  # noqa: ARG005
    )

    await _voice_async(
        config_path=None,
        session_id="voice-queue-next-turn",
        max_record_seconds=2.0,
        playback=True,
        input_audio=None,
        output_audio=None,
        continuous=True,
        chunk_seconds=0.1,
        max_turns=2,
        audio_io=fake_io,
    )

    assert fake_engine.closed is True
    assert fake_engine.voice_calls == [b"u1-speech-startu1-endpoint", b"u2-speech-startu2-endpoint"]
    assert fake_io.stop_calls == 0
    assert fake_io.play_calls == [b"old-audio-1", b"old-audio-2", b"new-audio"]
    assert fake_engine.turn_start_play_lengths == [0, 2]
    assert any("old-transcript" in line for line in printed)
    assert any("new-transcript" in line for line in printed)


@pytest.mark.asyncio
async def test_voice_async_continuous_mode_barge_in_does_not_stop_or_suppress_old_turn(
    monkeypatch,
) -> None:
    """barge-in 只排隊新 utterance，不應 stop/suppress 舊 turn TTS。"""

    class _FakeAudioIO:
        def __init__(self) -> None:
            self.play_calls: list[bytes] = []
            self.first_chunk_started = asyncio.Event()
            self.release_first_chunk = asyncio.Event()

        async def record_once(
            self,
            *,
            sample_rate: int,
            channels: int,
            max_seconds: float,
        ) -> bytes:  # noqa: ARG002
            return b"unused"

        async def record_stream(
            self,
            *,
            sample_rate: int,  # noqa: ARG002
            channels: int,  # noqa: ARG002
            chunk_seconds: float,  # noqa: ARG002
            max_seconds: float,  # noqa: ARG002
        ):
            yield b"u1-speech-start"
            yield b"u1-endpoint"
            await self.first_chunk_started.wait()
            yield b"u2-speech-start"
            yield b"u2-endpoint"
            self.release_first_chunk.set()
            await asyncio.sleep(0.01)

        async def play_once(self, audio: bytes, *, sample_rate: int, channels: int) -> None:  # noqa: ARG002
            self.play_calls.append(audio)
            if audio == b"old-audio-1":
                self.first_chunk_started.set()
                await self.release_first_chunk.wait()

    class _FakeVoiceSession:
        def __init__(self) -> None:
            self.append_calls = 0

        async def append_audio_chunk_with_vad(
            self,
            chunk: bytes,  # noqa: ARG002
            session_id: str | None = None,  # noqa: ARG002
            *,
            include_vad_state: bool = False,  # noqa: ARG002
        ) -> dict[str, bool]:
            self.append_calls += 1
            mapping = {
                1: {"endpoint": False, "is_speech": True},
                2: {"endpoint": True, "is_speech": False},
                3: {"endpoint": False, "is_speech": True},
                4: {"endpoint": True, "is_speech": False},
            }
            return mapping[self.append_calls]

        async def interrupt_buffered_input(self) -> int:
            return 0

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001, ARG002
            self.voice_session = _FakeVoiceSession()
            self.closed = False
            self.voice_calls: list[bytes] = []

        async def initialize(self) -> None:
            return None

        async def get_or_create_voice_session(
            self,
            session_id: str | None = None,
        ) -> _FakeVoiceSession:  # noqa: ARG002
            return self.voice_session

        async def voice_chat(self, audio: bytes, session_id: str | None = None):  # type: ignore[no-untyped-def, ARG002]
            self.voice_calls.append(audio)
            if len(self.voice_calls) == 1:
                yield SynthesizedAudioChunkEvent(chunk=b"old-audio-1")
                await fake_io.first_chunk_started.wait()
                yield TranscriptionEvent(text="old-transcript")
                yield AgentFinalTextEvent(text="old-agent")
                yield SynthesizedAudioChunkEvent(chunk=b"old-audio-2")
                return
            yield TranscriptionEvent(text="new-transcript")
            yield AgentFinalTextEvent(text="new-agent")
            yield SynthesizedAudioChunkEvent(chunk=b"new-audio")

        async def close(self) -> None:
            self.closed = True

    fake_cfg = SimpleNamespace(voice=SimpleNamespace(sample_rate=16000, channels=1))
    fake_io = _FakeAudioIO()
    fake_engine = _FakeEngine(fake_cfg)

    monkeypatch.setattr("mochi.config.manager.load_config", lambda path=None: fake_cfg)
    monkeypatch.setattr("mochi.agents.engine.AgentEngine", lambda cfg: fake_engine)

    await _voice_async(
        config_path=None,
        session_id="voice-barge-in-no-stop",
        max_record_seconds=2.0,
        playback=True,
        input_audio=None,
        output_audio=None,
        continuous=True,
        chunk_seconds=0.1,
        max_turns=2,
        audio_io=fake_io,
    )

    assert fake_engine.closed is True
    assert fake_engine.voice_calls == [b"u1-speech-startu1-endpoint", b"u2-speech-startu2-endpoint"]
    assert fake_io.play_calls == [b"old-audio-1", b"old-audio-2", b"new-audio"]


@pytest.mark.asyncio
async def test_voice_async_continuous_mode_short_insert_without_endpoint_keeps_old_turn(monkeypatch) -> None:
    """短暫插話若未形成 endpoint，不應作廢舊回合輸出。"""

    class _FakeAudioIO:
        def __init__(self) -> None:
            self.play_calls: list[bytes] = []
            self.stop_calls = 0

        async def record_once(self, *, sample_rate: int, channels: int, max_seconds: float) -> bytes:  # noqa: ARG002
            return b"unused"

        async def record_stream(
            self,
            *,
            sample_rate: int,  # noqa: ARG002
            channels: int,  # noqa: ARG002
            chunk_seconds: float,  # noqa: ARG002
            max_seconds: float,  # noqa: ARG002
        ):
            yield b"u1-speech-start"
            yield b"u1-endpoint"
            yield b"insert-speech-start-no-endpoint"
            await asyncio.sleep(0.05)

        async def play_once(self, audio: bytes, *, sample_rate: int, channels: int) -> None:  # noqa: ARG002
            self.play_calls.append(audio)

        async def stop_playback(self) -> bool:
            self.stop_calls += 1
            return True

    class _FakeVoiceSession:
        def __init__(self) -> None:
            self.append_calls = 0

        async def append_audio_chunk_with_vad(
            self,
            chunk: bytes,  # noqa: ARG002
            session_id: str | None = None,  # noqa: ARG002
            *,
            include_vad_state: bool = False,  # noqa: ARG002
        ) -> dict[str, bool]:
            self.append_calls += 1
            mapping = {
                1: {"endpoint": False, "is_speech": True},
                2: {"endpoint": True, "is_speech": False},
                3: {"endpoint": False, "is_speech": True},
            }
            return mapping[self.append_calls]

        async def interrupt_buffered_input(self) -> int:
            return 0

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001, ARG002
            self.voice_session = _FakeVoiceSession()
            self.closed = False
            self.voice_calls: list[bytes] = []

        async def initialize(self) -> None:
            return None

        async def get_or_create_voice_session(self, session_id: str | None = None) -> _FakeVoiceSession:  # noqa: ARG002
            return self.voice_session

        async def voice_chat(self, audio: bytes, session_id: str | None = None):  # type: ignore[no-untyped-def, ARG002]
            self.voice_calls.append(audio)
            yield TranscriptionEvent(text="old-transcript")
            yield AgentFinalTextEvent(text="old-agent")
            yield SynthesizedAudioChunkEvent(chunk=b"old-audio")

        async def close(self) -> None:
            self.closed = True

    fake_cfg = SimpleNamespace(voice=SimpleNamespace(sample_rate=16000, channels=1))
    fake_io = _FakeAudioIO()
    fake_engine = _FakeEngine(fake_cfg)
    printed: list[str] = []

    monkeypatch.setattr("mochi.config.manager.load_config", lambda path=None: fake_cfg)
    monkeypatch.setattr("mochi.agents.engine.AgentEngine", lambda cfg: fake_engine)
    monkeypatch.setattr(
        "mochi.main.console.print",
        lambda *args, **kwargs: printed.append(str(args[0]) if args else ""),  # noqa: ARG005
    )

    await _voice_async(
        config_path=None,
        session_id="voice-short-insert",
        max_record_seconds=2.0,
        playback=True,
        input_audio=None,
        output_audio=None,
        continuous=True,
        chunk_seconds=0.1,
        max_turns=1,
        audio_io=fake_io,
    )

    assert fake_engine.closed is True
    assert fake_engine.voice_calls == [b"u1-speech-startu1-endpoint"]
    assert fake_io.stop_calls == 0
    assert fake_io.play_calls == [b"old-audio"]
    assert any("old-transcript" in line for line in printed)


@pytest.mark.asyncio
async def test_voice_async_continuous_mode_two_endpoints_queue_two_turns_fifo(monkeypatch) -> None:
    """舊 turn 期間累積兩個 endpoint，應排成兩個 queued turn 且 FIFO。"""

    class _FakeAudioIO:
        async def record_once(self, *, sample_rate: int, channels: int, max_seconds: float) -> bytes:  # noqa: ARG002
            return b"unused"

        async def record_stream(
            self,
            *,
            sample_rate: int,  # noqa: ARG002
            channels: int,  # noqa: ARG002
            chunk_seconds: float,  # noqa: ARG002
            max_seconds: float,  # noqa: ARG002
        ):
            yield b"a1"
            yield b"a2"
            await asyncio.sleep(0.01)
            yield b"b1"
            yield b"b2"
            yield b"c1"
            yield b"c2"
            await asyncio.sleep(0.05)

        async def play_once(self, audio: bytes, *, sample_rate: int, channels: int) -> None:  # noqa: ARG002
            await asyncio.sleep(0.005)

    class _FakeVoiceSession:
        def __init__(self) -> None:
            self.append_calls = 0

        async def append_audio_chunk_with_vad(
            self,
            chunk: bytes,  # noqa: ARG002
            session_id: str | None = None,  # noqa: ARG002
            *,
            include_vad_state: bool = False,  # noqa: ARG002
        ) -> dict[str, bool]:
            self.append_calls += 1
            mapping = {
                1: {"endpoint": False, "is_speech": True},
                2: {"endpoint": True, "is_speech": False},
                3: {"endpoint": False, "is_speech": True},
                4: {"endpoint": True, "is_speech": False},
                5: {"endpoint": False, "is_speech": True},
                6: {"endpoint": True, "is_speech": False},
            }
            return mapping[self.append_calls]

        async def interrupt_buffered_input(self) -> int:
            return 0

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001, ARG002
            self.voice_session = _FakeVoiceSession()
            self.closed = False
            self.voice_calls: list[bytes] = []

        async def initialize(self) -> None:
            return None

        async def get_or_create_voice_session(self, session_id: str | None = None) -> _FakeVoiceSession:  # noqa: ARG002
            return self.voice_session

        async def voice_chat(self, audio: bytes, session_id: str | None = None):  # type: ignore[no-untyped-def, ARG002]
            self.voice_calls.append(audio)
            if len(self.voice_calls) == 1:
                await asyncio.sleep(0.03)
            yield SynthesizedAudioChunkEvent(chunk=b"ok")

        async def close(self) -> None:
            self.closed = True

    fake_cfg = SimpleNamespace(voice=SimpleNamespace(sample_rate=16000, channels=1))
    fake_io = _FakeAudioIO()
    fake_engine = _FakeEngine(fake_cfg)
    monkeypatch.setattr("mochi.config.manager.load_config", lambda path=None: fake_cfg)
    monkeypatch.setattr("mochi.agents.engine.AgentEngine", lambda cfg: fake_engine)

    await _voice_async(
        config_path=None,
        session_id="voice-queue-three",
        max_record_seconds=3.0,
        playback=True,
        input_audio=None,
        output_audio=None,
        continuous=True,
        chunk_seconds=0.1,
        max_turns=3,
        audio_io=fake_io,
    )

    assert fake_engine.closed is True
    assert fake_engine.voice_calls == [b"a1a2", b"b1b2", b"c1c2"]


@pytest.mark.asyncio
async def test_voice_async_continuous_mode_respects_max_turns_with_queue(monkeypatch) -> None:
    """queue 模式下 max_turns 仍限制總 turn 數。"""

    class _FakeAudioIO:
        async def record_once(self, *, sample_rate: int, channels: int, max_seconds: float) -> bytes:  # noqa: ARG002
            return b"unused"

        async def record_stream(
            self,
            *,
            sample_rate: int,  # noqa: ARG002
            channels: int,  # noqa: ARG002
            chunk_seconds: float,  # noqa: ARG002
            max_seconds: float,  # noqa: ARG002
        ):
            yield b"a1"
            yield b"a2"
            yield b"b1"
            yield b"b2"
            yield b"c1"
            yield b"c2"
            await asyncio.sleep(0.03)

        async def play_once(self, audio: bytes, *, sample_rate: int, channels: int) -> None:  # noqa: ARG002
            return None

    class _FakeVoiceSession:
        def __init__(self) -> None:
            self.append_calls = 0

        async def append_audio_chunk_with_vad(
            self,
            chunk: bytes,  # noqa: ARG002
            session_id: str | None = None,  # noqa: ARG002
            *,
            include_vad_state: bool = False,  # noqa: ARG002
        ) -> dict[str, bool]:
            self.append_calls += 1
            mapping = {
                1: {"endpoint": False, "is_speech": True},
                2: {"endpoint": True, "is_speech": False},
                3: {"endpoint": False, "is_speech": True},
                4: {"endpoint": True, "is_speech": False},
                5: {"endpoint": False, "is_speech": True},
                6: {"endpoint": True, "is_speech": False},
            }
            return mapping[self.append_calls]

        async def interrupt_buffered_input(self) -> int:
            return 0

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001, ARG002
            self.voice_session = _FakeVoiceSession()
            self.closed = False
            self.voice_calls: list[bytes] = []

        async def initialize(self) -> None:
            return None

        async def get_or_create_voice_session(self, session_id: str | None = None) -> _FakeVoiceSession:  # noqa: ARG002
            return self.voice_session

        async def voice_chat(self, audio: bytes, session_id: str | None = None):  # type: ignore[no-untyped-def, ARG002]
            self.voice_calls.append(audio)
            yield SynthesizedAudioChunkEvent(chunk=b"ok")

        async def close(self) -> None:
            self.closed = True

    fake_cfg = SimpleNamespace(voice=SimpleNamespace(sample_rate=16000, channels=1))
    fake_io = _FakeAudioIO()
    fake_engine = _FakeEngine(fake_cfg)
    monkeypatch.setattr("mochi.config.manager.load_config", lambda path=None: fake_cfg)
    monkeypatch.setattr("mochi.agents.engine.AgentEngine", lambda cfg: fake_engine)

    await _voice_async(
        config_path=None,
        session_id="voice-max-turns-queue",
        max_record_seconds=3.0,
        playback=True,
        input_audio=None,
        output_audio=None,
        continuous=True,
        chunk_seconds=0.1,
        max_turns=2,
        audio_io=fake_io,
    )

    assert fake_engine.closed is True
    assert fake_engine.voice_calls == [b"a1a2", b"b1b2"]


def test_sounddevice_audio_io_converts_recorded_audio_to_pcm16(monkeypatch) -> None:
    """SoundDeviceAudioIO 應將錄音 float32 轉為 PCM16 bytes。"""
    np = pytest.importorskip("numpy")
    fake_sounddevice = ModuleType("sounddevice")
    wait_calls: list[str] = []

    def fake_wait() -> None:
        wait_calls.append("wait")

    fake_sounddevice.rec = lambda *args, **kwargs: np.array(  # noqa: ARG005
        [[0.5, -0.5], [1.0, -1.0]],
        dtype=np.float32,
    )
    fake_sounddevice.play = lambda *args, **kwargs: None  # noqa: ARG005
    fake_sounddevice.wait = fake_wait
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    audio_io = SoundDeviceAudioIO()
    payload = audio_io._record_blocking(16000, 2, 1.0)  # noqa: SLF001

    assert payload == b"\x00\x00\x00\x00"
    assert wait_calls == ["wait"]


def test_sounddevice_audio_io_record_stream_uses_input_stream(monkeypatch) -> None:
    """連續錄音應優先使用 InputStream，避免 sounddevice 全域 rec/play 狀態互相干擾。"""
    np = pytest.importorskip("numpy")
    fake_sounddevice = ModuleType("sounddevice")
    stream_events: list[tuple[str, int | None]] = []

    class _FakeInputStream:
        def __init__(self, *, samplerate: int, channels: int, dtype: str) -> None:
            stream_events.append((f"open:{samplerate}:{channels}:{dtype}", None))

        def __enter__(self):  # noqa: ANN001
            stream_events.append(("enter", None))
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
            stream_events.append(("exit", None))

        def read(self, frames: int):  # noqa: ANN001
            stream_events.append(("read", frames))
            return np.array([[0.5], [1.0]], dtype=np.float32), True

    def fail_rec(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        raise AssertionError("record_stream should not call sd.rec when InputStream exists")

    async def collect_stream() -> tuple[list[bytes], dict[str, object]]:
        audio_io = SoundDeviceAudioIO()
        chunks = [
            chunk
            async for chunk in audio_io.record_stream(
                sample_rate=4,
                channels=1,
                chunk_seconds=0.5,
                max_seconds=1.0,
            )
        ]
        return chunks, audio_io.get_runtime_diagnostics()

    fake_sounddevice.InputStream = _FakeInputStream
    fake_sounddevice.OutputStream = None
    fake_sounddevice.rec = fail_rec
    fake_sounddevice.play = lambda *args, **kwargs: None  # noqa: ARG005
    fake_sounddevice.stop = lambda: None
    fake_sounddevice.wait = lambda: None
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    chunks, diagnostics = asyncio.run(collect_stream())
    expected = (np.array([0.5, 1.0]) * 32767.0).astype(np.int16).tobytes()

    assert chunks == [expected, expected]
    assert diagnostics["input_overflow_events"] == 2
    assert stream_events == [
        ("open:4:1:float32", None),
        ("enter", None),
        ("read", 2),
        ("read", 2),
        ("exit", None),
    ]


def test_sounddevice_audio_io_collects_runtime_diagnostics(monkeypatch) -> None:
    """SoundDeviceAudioIO 應提供裝置與執行時診斷資訊。"""
    np = pytest.importorskip("numpy")
    fake_sounddevice = ModuleType("sounddevice")
    check_calls: list[tuple[str, int, int, str]] = []

    class _DefaultSettings:
        device = (3, 7)
        samplerate = 48000

    def fake_check_input_settings(*, samplerate: int, channels: int, dtype: str) -> None:
        check_calls.append(("input", samplerate, channels, dtype))

    def fake_check_output_settings(*, samplerate: int, channels: int, dtype: str) -> None:
        check_calls.append(("output", samplerate, channels, dtype))

    fake_sounddevice.default = _DefaultSettings()
    fake_sounddevice.query_devices = lambda: [{"name": "Mic"}, {"name": "Speaker"}]
    fake_sounddevice.get_portaudio_version = lambda: (1900, "PortAudio V19")
    fake_sounddevice.check_input_settings = fake_check_input_settings
    fake_sounddevice.check_output_settings = fake_check_output_settings
    fake_sounddevice.rec = lambda *args, **kwargs: np.array([[0.0]], dtype=np.float32)  # noqa: ARG005
    fake_sounddevice.play = lambda *args, **kwargs: None  # noqa: ARG005
    fake_sounddevice.wait = lambda: None
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    audio_io = SoundDeviceAudioIO()
    asyncio.run(audio_io.record_once(sample_rate=16000, channels=1, max_seconds=0.01))
    asyncio.run(audio_io.play_once(b"\x00\x00", sample_rate=16000, channels=1))
    diagnostics = audio_io.get_runtime_diagnostics()

    assert diagnostics["backend"] == "sounddevice"
    assert diagnostics["default_input_device"] == 3
    assert diagnostics["default_output_device"] == 7
    assert diagnostics["default_samplerate"] == 48000
    assert diagnostics["device_count"] == 2
    assert diagnostics["portaudio_version"] == (1900, "PortAudio V19")
    assert diagnostics["last_input_settings_supported"] is True
    assert diagnostics["last_output_settings_supported"] is True
    assert check_calls == [
        ("input", 16000, 1, "float32"),
        ("output", 16000, 1, "float32"),
    ]


def test_sounddevice_audio_io_record_stream_closes_stream_when_read_fails(monkeypatch) -> None:
    """InputStream 讀取失敗時仍應確實退出 stream 並保留診斷。"""
    fake_sounddevice = ModuleType("sounddevice")
    stream_events: list[str] = []

    class _BrokenInputStream:
        def __init__(self, *, samplerate: int, channels: int, dtype: str) -> None:  # noqa: ARG002
            stream_events.append("open")

        def __enter__(self):  # noqa: ANN001
            stream_events.append("enter")
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
            stream_events.append("exit")

        def read(self, frames: int):  # noqa: ANN001, ARG002
            stream_events.append("read")
            raise RuntimeError("boom")

    async def consume_stream() -> None:
        audio_io = SoundDeviceAudioIO()
        with pytest.raises(RuntimeError, match="record_stream failed"):
            async for _ in audio_io.record_stream(
                sample_rate=4,
                channels=1,
                chunk_seconds=0.5,
                max_seconds=1.0,
            ):
                pass
        diagnostics = audio_io.get_runtime_diagnostics()
        assert "record_stream" in str(diagnostics["last_runtime_error"])
        assert "boom" in str(diagnostics["last_runtime_error"])

    fake_sounddevice.InputStream = _BrokenInputStream
    fake_sounddevice.OutputStream = None
    fake_sounddevice.rec = lambda *args, **kwargs: None  # noqa: ARG005
    fake_sounddevice.play = lambda *args, **kwargs: None  # noqa: ARG005
    fake_sounddevice.stop = lambda: None
    fake_sounddevice.wait = lambda: None
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    asyncio.run(consume_stream())

    assert stream_events == ["open", "enter", "read", "exit"]


def test_sounddevice_audio_io_expands_mono_pcm_for_multichannel_playback(monkeypatch) -> None:
    """SoundDeviceAudioIO 播放多聲道時應展開為對應 shape。"""
    np = pytest.importorskip("numpy")
    fake_sounddevice = ModuleType("sounddevice")
    played: list[tuple[object, int]] = []
    stopped: list[str] = []

    def fake_play(wave, *, samplerate: int) -> None:  # noqa: ANN001
        played.append((wave.copy(), samplerate))

    fake_sounddevice.rec = lambda *args, **kwargs: None  # noqa: ARG005
    fake_sounddevice.play = fake_play
    fake_sounddevice.stop = lambda: stopped.append("stop")
    fake_sounddevice.wait = lambda: None
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    audio_io = SoundDeviceAudioIO()
    audio_io._play_blocking(b"\xff\x7f\x01\x80", 22050, 2)  # noqa: SLF001
    assert asyncio.run(audio_io.stop_playback()) is True

    assert len(played) == 1
    wave, samplerate = played[0]
    assert samplerate == 22050
    assert wave.shape == (2, 2)
    assert np.isclose(wave[0, 0], 1.0)
    assert np.isclose(wave[1, 0], -1.0, atol=1e-4)
    assert stopped == ["stop"]


def test_sounddevice_audio_io_playback_uses_output_stream_when_available(monkeypatch) -> None:
    """播放應優先使用 OutputStream，避免 sd.play convenience API 影響錄音 stream。"""
    np = pytest.importorskip("numpy")
    fake_sounddevice = ModuleType("sounddevice")
    writes: list[tuple[object, int, int]] = []

    class _FakeOutputStream:
        def __init__(self, *, samplerate: int, channels: int, dtype: str) -> None:
            self.samplerate = samplerate
            self.channels = channels
            self.dtype = dtype

        def __enter__(self):  # noqa: ANN001
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
            return None

        def write(self, wave) -> None:  # noqa: ANN001
            writes.append((wave.copy(), self.samplerate, self.channels))

    def fail_play(*args, **kwargs):  # noqa: ANN002, ANN003, ARG001
        raise AssertionError("play_once should not call sd.play when OutputStream exists")

    fake_sounddevice.rec = lambda *args, **kwargs: None  # noqa: ARG005
    fake_sounddevice.play = fail_play
    fake_sounddevice.stop = lambda: None
    fake_sounddevice.wait = lambda: None
    fake_sounddevice.OutputStream = _FakeOutputStream
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    audio_io = SoundDeviceAudioIO()
    audio_io._play_blocking(b"\xff\x7f\x01\x80", 16000, 1)  # noqa: SLF001

    assert len(writes) == 1
    wave, samplerate, channels = writes[0]
    assert samplerate == 16000
    assert channels == 1
    assert wave.shape == (2, 1)
    assert np.isclose(wave[0, 0], 1.0)
    assert np.isclose(wave[1, 0], -1.0, atol=1e-4)


def test_sounddevice_audio_io_playback_session_reuses_single_output_stream(monkeypatch) -> None:
    """playback_session 應重用單一 OutputStream，避免每個 chunk 反覆重開。"""
    pytest.importorskip("numpy")
    fake_sounddevice = ModuleType("sounddevice")
    stream_events: list[str] = []
    writes: list[bytes] = []

    class _FakeOutputStream:
        def __init__(self, *, samplerate: int, channels: int, dtype: str) -> None:  # noqa: ARG002
            stream_events.append("open")

        def __enter__(self):  # noqa: ANN001
            stream_events.append("enter")
            return self

        def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
            stream_events.append("exit")

        def write(self, wave) -> bool:  # noqa: ANN001
            writes.append(wave.copy().tobytes())
            return False

    fake_sounddevice.rec = lambda *args, **kwargs: None  # noqa: ARG005
    fake_sounddevice.play = lambda *args, **kwargs: None  # noqa: ARG005
    fake_sounddevice.stop = lambda: None
    fake_sounddevice.wait = lambda: None
    fake_sounddevice.OutputStream = _FakeOutputStream
    monkeypatch.setitem(sys.modules, "sounddevice", fake_sounddevice)

    async def use_session() -> dict[str, object]:
        audio_io = SoundDeviceAudioIO()
        async with audio_io.playback_session(sample_rate=16000, channels=1) as play:
            await play(b"\x01\x00\x02\x00")
            await play(b"\x03\x00\x04\x00")
        return audio_io.get_runtime_diagnostics()

    diagnostics = asyncio.run(use_session())

    assert stream_events == ["open", "enter", "exit"]
    assert len(writes) == 2
    assert diagnostics["persistent_output_stream_supported"] is True
