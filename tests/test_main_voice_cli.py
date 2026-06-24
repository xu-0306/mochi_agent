"""voice CLI 測試。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from mochi.main import _voice_async, app
from mochi.voice.events import (
    AgentFinalTextEvent,
    SynthesizedAudioChunkEvent,
    TranscriptionEvent,
    VoiceErrorEvent,
)

runner = CliRunner()


def _patch_doctor_basics(monkeypatch, *, sample_rate: int = 16000, channels: int = 1) -> None:
    """設定 doctor 測試所需的最小依賴。"""
    class _FakeOllamaBackend:
        def __init__(self, model: str, base_url: str) -> None:  # noqa: ANN001, ARG002
            self.model = model
            self.base_url = base_url

        async def health_check(self) -> bool:
            return True

        async def close(self) -> None:
            return None

    def fake_load_config(config_path=None):  # noqa: ARG001
        return SimpleNamespace(
            model="ollama:qwen2.5",
            ollama=SimpleNamespace(base_url="http://localhost:11434"),
            voice=SimpleNamespace(sample_rate=sample_rate, channels=channels),
        )

    async def fake_inspect(model_spec: str, ollama_base_url: str):  # noqa: ARG001
        return (
            SimpleNamespace(
                name="configured-model",
                backend_type="ollama",
                context_length=4096,
                supports_tool_calling=False,
                metadata={},
            ),
            True,
            None,
        )

    monkeypatch.setattr("mochi.config.manager.load_config", fake_load_config)
    monkeypatch.setattr("mochi.backends.ollama.OllamaBackend", _FakeOllamaBackend)
    monkeypatch.setattr("mochi.main._inspect_configured_model", fake_inspect)
    monkeypatch.setattr("os.path.exists", lambda _: True)


def test_voice_command_calls_async_helper(monkeypatch) -> None:
    """voice 指令應將參數轉交給 async helper。"""
    called: dict[str, object] = {}

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
        audio_io=None,
    ) -> None:
        called["config_path"] = config_path
        called["session_id"] = session_id
        called["max_record_seconds"] = max_record_seconds
        called["playback"] = playback
        called["input_audio"] = input_audio
        called["output_audio"] = output_audio
        called["continuous"] = continuous
        called["chunk_seconds"] = chunk_seconds
        called["max_turns"] = max_turns
        called["audio_io"] = audio_io

    monkeypatch.setattr("mochi.main._voice_async", fake_voice_async)

    result = runner.invoke(
        app,
        [
            "voice",
            "--config",
            "cfg.yaml",
            "--session-id",
            "s-voice",
            "--max-record-seconds",
            "3.5",
            "--no-playback",
            "--input-audio",
            "input.pcm",
            "--output-audio",
            "output.pcm",
        ],
    )

    assert result.exit_code == 0
    assert called["config_path"] == "cfg.yaml"
    assert called["session_id"] == "s-voice"
    assert called["max_record_seconds"] == 3.5
    assert called["playback"] is False
    assert called["input_audio"] == "input.pcm"
    assert called["output_audio"] == "output.pcm"
    assert called["continuous"] is False
    assert called["chunk_seconds"] == 0.25
    assert called["max_turns"] == 0
    assert called["audio_io"] is None


@pytest.mark.asyncio
async def test_voice_async_records_runs_voice_chat_and_plays(monkeypatch) -> None:
    """_voice_async 成功時應錄音、彙整 TTS 並播放。"""
    class _FakeAudioIO:
        def __init__(self) -> None:
            self.record_calls: list[tuple[int, int, float]] = []
            self.play_calls: list[tuple[bytes, int, int]] = []

        async def record_once(self, *, sample_rate: int, channels: int, max_seconds: float) -> bytes:
            self.record_calls.append((sample_rate, channels, max_seconds))
            return b"pcm-in"

        async def play_once(self, audio: bytes, *, sample_rate: int, channels: int) -> None:
            self.play_calls.append((audio, sample_rate, channels))

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001
            self.config = config
            self.voice_calls: list[tuple[bytes, str | None]] = []
            self.closed = False

        async def initialize(self) -> None:
            return None

        async def voice_chat(self, audio: bytes, session_id: str | None = None) -> AsyncIterator[object]:
            self.voice_calls.append((audio, session_id))
            yield TranscriptionEvent(text="你好")
            yield AgentFinalTextEvent(text="世界")
            yield SynthesizedAudioChunkEvent(chunk=b"a")
            yield SynthesizedAudioChunkEvent(chunk=b"b")

        async def close(self) -> None:
            self.closed = True

    fake_engine_ref: dict[str, _FakeEngine] = {}

    def fake_load_config(config_path=None):  # noqa: ARG001
        return SimpleNamespace(voice=SimpleNamespace(sample_rate=16000, channels=1))

    def fake_engine_factory(config) -> _FakeEngine:  # noqa: ANN001
        engine = _FakeEngine(config)
        fake_engine_ref["engine"] = engine
        return engine

    monkeypatch.setattr("mochi.config.manager.load_config", fake_load_config)
    monkeypatch.setattr("mochi.agents.engine.AgentEngine", fake_engine_factory)

    io = _FakeAudioIO()
    await _voice_async(
        config_path=None,
        session_id="voice-s1",
        max_record_seconds=2.0,
        playback=True,
        input_audio=None,
        output_audio=None,
        audio_io=io,
    )

    assert io.record_calls == [(16000, 1, 2.0)]
    assert io.play_calls == [(b"ab", 16000, 1)]
    assert fake_engine_ref["engine"].voice_calls == [(b"pcm-in", "voice-s1")]
    assert fake_engine_ref["engine"].closed is True


@pytest.mark.asyncio
async def test_voice_async_exits_with_code_1_on_voice_error_event(monkeypatch) -> None:
    """語音流程發出錯誤事件時應以 exit code 1 結束。"""
    class _FakeAudioIO:
        async def record_once(self, *, sample_rate: int, channels: int, max_seconds: float) -> bytes:  # noqa: ARG002
            return b"pcm-in"

        async def play_once(self, audio: bytes, *, sample_rate: int, channels: int) -> None:  # noqa: ARG002
            raise AssertionError("play_once should not be called on error path")

    class _FakeEngine:
        def __init__(self, config) -> None:  # noqa: ANN001, ARG002
            self.closed = False

        async def initialize(self) -> None:
            return None

        async def voice_chat(self, audio: bytes, session_id: str | None = None) -> AsyncIterator[object]:  # noqa: ARG002
            yield VoiceErrorEvent(message="vad failed", code="VAD_ERROR")

        async def close(self) -> None:
            self.closed = True

    def fake_load_config(config_path=None):  # noqa: ARG001
        return SimpleNamespace(voice=SimpleNamespace(sample_rate=16000, channels=1))

    fake_engine = _FakeEngine(None)
    monkeypatch.setattr("mochi.config.manager.load_config", fake_load_config)
    monkeypatch.setattr("mochi.agents.engine.AgentEngine", lambda config: fake_engine)  # noqa: ARG005

    with pytest.raises(SystemExit) as exc_info:
        await _voice_async(
            config_path=None,
            session_id=None,
            max_record_seconds=2.0,
            playback=True,
            input_audio=None,
            output_audio=None,
            audio_io=_FakeAudioIO(),
        )

    assert exc_info.value.code == 1
    assert fake_engine.closed is True


def test_voice_command_calls_async_helper_with_continuous_mode(monkeypatch) -> None:
    """voice 指令啟用 continuous 時應傳遞相關參數。"""
    called: dict[str, object] = {}

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
        audio_io=None,
    ) -> None:
        called["config_path"] = config_path
        called["session_id"] = session_id
        called["max_record_seconds"] = max_record_seconds
        called["playback"] = playback
        called["input_audio"] = input_audio
        called["output_audio"] = output_audio
        called["continuous"] = continuous
        called["chunk_seconds"] = chunk_seconds
        called["max_turns"] = max_turns
        called["audio_io"] = audio_io

    monkeypatch.setattr("mochi.main._voice_async", fake_voice_async)

    result = runner.invoke(
        app,
        [
            "voice",
            "--continuous",
            "--chunk-seconds",
            "0.4",
            "--max-turns",
            "2",
            "--max-record-seconds",
            "8",
        ],
    )

    assert result.exit_code == 0
    assert called["continuous"] is True
    assert called["chunk_seconds"] == 0.4
    assert called["max_turns"] == 2
    assert called["max_record_seconds"] == 8.0


def test_doctor_reports_local_audio_diagnostics_from_runtime_helper(monkeypatch) -> None:
    """doctor 應輸出 bounded continuous 音訊診斷與 helper 回報內容。"""
    class _FakeAudioIO:
        async def record_once(self, *, sample_rate: int, channels: int, max_seconds: float) -> bytes:  # noqa: ARG002
            return b""

        async def play_once(self, audio: bytes, *, sample_rate: int, channels: int) -> None:  # noqa: ARG002
            return None

        async def record_stream(self, *, sample_rate: int, channels: int, chunk_seconds: float, max_seconds: float):  # noqa: ANN201, ARG002
            if False:  # pragma: no cover
                yield b""

        async def get_runtime_diagnostics(self) -> dict[str, object]:
            return {
                "input_device": "Mic-1",
                "output_device": "Speaker-1",
                "duplex_supported": True,
            }

    _patch_doctor_basics(monkeypatch)
    monkeypatch.setattr("mochi.main.create_default_audio_io", lambda: _FakeAudioIO())

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "Voice/Audio (bounded continuous)" in result.stdout
    assert "audio_io: _FakeAudioIO" in result.stdout
    assert "audio_config: 16000 Hz / 1 ch" in result.stdout
    assert "record_stream: available" in result.stdout
    assert "input_device: Mic-1" in result.stdout
    assert "output_device: Speaker-1" in result.stdout
    assert "duplex_supported: True" in result.stdout


def test_doctor_audio_diagnostics_degrade_gracefully_without_runtime_helper(monkeypatch) -> None:
    """若 audio_io 未提供 helper，doctor 也應完成並回報 fallback 訊息。"""
    class _FakeAudioIO:
        async def record_once(self, *, sample_rate: int, channels: int, max_seconds: float) -> bytes:  # noqa: ARG002
            return b""

        async def play_once(self, audio: bytes, *, sample_rate: int, channels: int) -> None:  # noqa: ARG002
            return None

        async def record_stream(self, *, sample_rate: int, channels: int, chunk_seconds: float, max_seconds: float):  # noqa: ANN201, ARG002
            if False:  # pragma: no cover
                yield b""

    _patch_doctor_basics(monkeypatch, sample_rate=22050, channels=2)
    monkeypatch.setattr("mochi.main.create_default_audio_io", lambda: _FakeAudioIO())

    result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0
    assert "Voice/Audio (bounded continuous)" in result.stdout
    assert "audio_config: 22050 Hz / 2 ch" in result.stdout
    assert "record_stream: available" in result.stdout
    assert "diagnostics: not exposed by audio_io backend" in result.stdout
