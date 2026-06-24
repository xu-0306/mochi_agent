"""Phase 4：Voice foundation 測試。"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from struct import pack
from typing import Any
from unittest.mock import patch

import pytest
from pydantic import SecretStr

from mochi.agents.engine import AgentEngine
from mochi.config.schema import MochiConfig, RegisteredTTSVoiceConfig, VoiceConfig
from mochi.voice.base import BaseSTT, BaseTTS, BaseVAD, VoiceInfo
from mochi.voice.router import VoiceRouter
from mochi.voice.silero_vad_iterator import FixedVADIterator
from mochi.voice.stt.faster_whisper import FasterWhisperSTT
from mochi.voice.stt.openai_api import OpenAIApiSTT
from mochi.voice.stt.openai_whisper import OpenAIWhisperSTT
from mochi.voice.stt.qwen_asr import QwenASRSTT
from mochi.voice.stt.vosk import VoskSTT
from mochi.voice.stt.whisper_cpp import WhisperCppSTT
from mochi.voice.stt.whisperlivekit import WhisperLiveKitSTT
from mochi.voice.stt.whisperlivekit_runtime import (
    WhisperLiveKitRuntimeOptions,
    WhisperLiveKitService,
)
from mochi.voice.tts.coqui_tts import CoquiTTS
from mochi.voice.tts.edge_tts import EdgeTTS
from mochi.voice.tts.kokoro_tts import KokoroTTS
from mochi.voice.tts.openai_tts import OpenAITTS
from mochi.voice.tts.piper import PiperTTS
from mochi.voice.vad import SileroIteratorVAD, SimpleEnergyVAD


def _pcm16_frame(value: int, samples: int = 1600) -> bytes:
    return b"".join(pack("<h", value) for _ in range(samples))


@dataclass
class _Segment:
    text: str


class _FakeWhisperRuntime:
    def transcribe(self, audio: bytes, language: str | None = None, task: str = "transcribe") -> tuple[list[_Segment], Any]:  # noqa: ARG002
        if not audio:
            return [], object()
        return [_Segment("你好"), _Segment("Mochi")], object()


class _FakeEdgeCommunicator:
    async def stream(self):  # type: ignore[no-untyped-def]
        yield {"type": "audio", "data": b"abc"}
        yield {"type": "word", "text": "ignored"}
        yield {"type": "audio", "data": b"123"}


class _FakeWLKRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def transcribe(
        self,
        *,
        audio: bytes,
        sample_rate: int,
        language: str | None = None,
    ) -> dict[str, str]:
        self.calls.append(
            {"audio": audio, "sample_rate": sample_rate, "language": language}
        )
        return {"text": "WLK text"}


class _SequencedWLKPreviewProcessor:
    def __init__(self, *, results: list[object | None]) -> None:
        self.results = list(results)
        self.feed_calls: list[bytes] = []
        self.closed = False
        self._queue: asyncio.Queue[object] = asyncio.Queue()
        self._stop = object()

    async def create_tasks(self):  # type: ignore[no-untyped-def]
        async def _results():  # type: ignore[no-untyped-def]
            while True:
                item = await self._queue.get()
                if item is self._stop:
                    return
                yield item

        return _results()

    async def process_audio(self, audio: bytes) -> None:
        self.feed_calls.append(audio)
        if not self.results:
            return
        item = self.results.pop(0)
        if item is not None:
            await self._queue.put(item)

    async def close(self) -> None:
        self.closed = True
        await self._queue.put(self._stop)


def _watchdog_options() -> WhisperLiveKitRuntimeOptions:
    return WhisperLiveKitRuntimeOptions(
        model="base",
        language="en",
        watchdog_enabled=True,
        watchdog_poll_interval=0.01,
        watchdog_no_result_timeout=0.05,
        watchdog_stall_timeout=0.05,
        watchdog_audio_idle_window=0.3,
        watchdog_reset_cooldown=0.02,
    )


class _DummySTT(BaseSTT):
    async def transcribe(self, audio: bytes, *, sample_rate: int, language: str | None = None) -> str:  # noqa: ARG002
        return "ok"

    def get_info(self) -> VoiceInfo:
        return VoiceInfo(kind="stt", family="dummy", name="dummy")

    async def health_check(self) -> bool:
        return True


class _DummyTTS(BaseTTS):
    async def synthesize(self, text: str, *, voice: str | None = None, speed: float | None = None) -> bytes:  # noqa: ARG002
        return b"ok"

    def get_info(self) -> VoiceInfo:
        return VoiceInfo(kind="tts", family="dummy", name="dummy")

    async def health_check(self) -> bool:
        return True


class _DummyVAD(BaseVAD):
    async def is_speech(self, audio: bytes, *, sample_rate: int) -> bool:  # noqa: ARG002
        return False

    def reset(self) -> None:
        return None

    def get_info(self) -> VoiceInfo:
        return VoiceInfo(kind="vad", family="dummy", name="dummy")


class _FakeSileroModel:
    def __init__(self, probs: list[float]) -> None:
        self._probs = probs
        self._index = 0
        self.reset_calls = 0

    def reset_states(self) -> None:
        self.reset_calls += 1

    def __call__(self, x: list[float], sampling_rate: int) -> float:  # noqa: ARG002
        assert len(x) == 512
        idx = min(self._index, len(self._probs) - 1)
        self._index += 1
        return self._probs[idx]


@pytest.mark.asyncio
async def test_faster_whisper_stt_with_injected_runtime() -> None:
    stt = FasterWhisperSTT(runtime=_FakeWhisperRuntime(), model="small", device="cpu")

    text = await stt.transcribe(b"\x01\x02\x03\x04", sample_rate=16000, language="zh")

    assert text == "你好 Mochi"
    assert stt.get_info().family == "faster-whisper"
    assert await stt.health_check() is True


@pytest.mark.asyncio
async def test_faster_whisper_stt_runtime_factory_failure_semantics() -> None:
    def _broken_factory(*args: Any, **kwargs: Any) -> Any:  # noqa: ARG001
        raise RuntimeError("load failed")

    stt = FasterWhisperSTT(model_factory=_broken_factory)

    with pytest.raises(RuntimeError, match=r"runtime_init_failed"):
        await stt.transcribe(b"\x01\x02", sample_rate=16000)


@pytest.mark.asyncio
async def test_faster_whisper_stt_uses_cached_local_model_directory(tmp_path: Path) -> None:
    """cache 下有同名資料夾時，應使用本地模型來源。"""
    local_dir = tmp_path / "small"
    local_dir.mkdir()
    seen: dict[str, Any] = {}

    def _factory(model_source: str, **kwargs: Any) -> _FakeWhisperRuntime:
        seen["model_source"] = model_source
        seen["kwargs"] = kwargs
        return _FakeWhisperRuntime()

    stt = FasterWhisperSTT(
        model="small",
        device="cpu",
        model_cache_dir=tmp_path,
        model_factory=_factory,
    )

    text = await stt.transcribe(b"\x01\x02\x03\x04", sample_rate=16000)

    assert text == "你好 Mochi"
    assert seen["model_source"] == str(local_dir)
    assert seen["kwargs"] == {"device": "cpu"}
    assert stt.get_info().metadata["uses_local_model_source"] is True


@pytest.mark.asyncio
async def test_faster_whisper_stt_uses_download_root_for_named_model(tmp_path: Path) -> None:
    """沒有本地模型時，應保留模型名並傳入 download_root。"""
    seen: dict[str, Any] = {}

    def _factory(model_source: str, **kwargs: Any) -> _FakeWhisperRuntime:
        seen["model_source"] = model_source
        seen["kwargs"] = kwargs
        return _FakeWhisperRuntime()

    stt = FasterWhisperSTT(
        model="small",
        device="cpu",
        model_cache_dir=tmp_path,
        model_factory=_factory,
    )

    text = await stt.transcribe(b"\x01\x02\x03\x04", sample_rate=16000)

    assert text == "你好 Mochi"
    assert seen["model_source"] == "small"
    assert seen["kwargs"] == {"device": "cpu", "download_root": str(tmp_path)}
    assert stt.get_info().metadata["uses_local_model_source"] is False


@pytest.mark.asyncio
async def test_edge_tts_with_injected_factory() -> None:
    tts = EdgeTTS(
        communicator_factory=lambda text, voice, rate: _FakeEdgeCommunicator(),  # noqa: ARG005
    )

    audio = await tts.synthesize("hello")

    assert audio == b"abc123"
    assert tts.get_info().family == "edge-tts"
    assert await tts.health_check() is True


@pytest.mark.asyncio
async def test_simple_energy_vad_detects_speech_then_resets_after_silence() -> None:
    vad = SimpleEnergyVAD(threshold=0.2, min_speech_ms=100, max_silence_ms=120)
    loud_frame = _pcm16_frame(22000)  # 約 100ms @ 16kHz
    silence_frame = _pcm16_frame(0)

    first = await vad.is_speech(loud_frame, sample_rate=16000)
    second = await vad.is_speech(loud_frame, sample_rate=16000)
    third = await vad.is_speech(silence_frame, sample_rate=16000)
    fourth = await vad.is_speech(silence_frame, sample_rate=16000)

    assert first is True
    assert second is True
    assert third is True
    assert fourth is False
    assert vad.get_info().kind == "vad"


@pytest.mark.asyncio
async def test_voice_router_loads_default_supported_backends(tmp_path: Path) -> None:
    config = VoiceConfig(
        stt_backend="faster-whisper",
        stt_model="medium",
        stt_language="auto",
        stt_device="auto",
        stt_model_cache_dir=tmp_path,
        tts_backend="edge-tts",
        tts_voice="en-US-JennyNeural",
        tts_speed=1.1,
        vad_threshold=0.5,
        vad_min_speech_ms=250,
        vad_max_silence_ms=700,
    )

    router = VoiceRouter(
        stt_factory=lambda cfg: _DummySTT(),  # noqa: ARG005
        tts_factory=lambda cfg: _DummyTTS(),  # noqa: ARG005
        vad_factory=lambda cfg: _DummyVAD(),  # noqa: ARG005
    )
    runtime = await router.load(config)

    assert isinstance(runtime.stt, _DummySTT)
    assert isinstance(runtime.tts, _DummyTTS)
    assert isinstance(runtime.vad, _DummyVAD)


@pytest.mark.asyncio
async def test_voice_router_rejects_unsupported_default_backends(tmp_path: Path) -> None:
    router = VoiceRouter()

    bad_stt = VoiceConfig.model_construct(
        enabled=False,
        stt_backend="invalid-stt",
        stt_model="medium",
        stt_language="auto",
        stt_device="auto",
        stt_model_cache_dir=tmp_path,
        stt_model_path=None,
        stt_openai_base_url=None,
        stt_openai_api_key=None,
        stt_openai_timeout=60.0,
        tts_backend="edge-tts",
        tts_model=None,
        tts_voice="zh-CN-XiaoxiaoNeural",
        tts_language=None,
        tts_speed=1.0,
        tts_use_gpu=False,
        tts_kokoro_lang_code="a",
        tts_split_pattern=r"\n+",
        tts_openai_base_url=None,
        tts_openai_api_key=None,
        tts_openai_timeout=60.0,
        tts_openai_response_format="pcm",
        vad_threshold=0.5,
        vad_min_speech_ms=250,
        vad_max_silence_ms=700,
        sample_rate=16000,
        channels=1,
    )
    with pytest.raises(ValueError, match="Unsupported STT backend"):
        await router.load(bad_stt)

    bad_tts = VoiceConfig.model_construct(
        enabled=False,
        stt_backend="faster-whisper",
        stt_model="medium",
        stt_language="auto",
        stt_device="auto",
        stt_model_cache_dir=tmp_path,
        stt_model_path=None,
        stt_openai_base_url=None,
        stt_openai_api_key=None,
        stt_openai_timeout=60.0,
        tts_backend="invalid-tts",
        tts_model=None,
        tts_voice="zh-CN-XiaoxiaoNeural",
        tts_language=None,
        tts_speed=1.0,
        tts_use_gpu=False,
        tts_kokoro_lang_code="a",
        tts_split_pattern=r"\n+",
        tts_openai_base_url=None,
        tts_openai_api_key=None,
        tts_openai_timeout=60.0,
        tts_openai_response_format="pcm",
        vad_threshold=0.5,
        vad_min_speech_ms=250,
        vad_max_silence_ms=700,
        sample_rate=16000,
        channels=1,
    )
    with pytest.raises(ValueError, match="Unsupported TTS backend"):
        await router.load(bad_tts)


@pytest.mark.asyncio
async def test_voice_router_uses_resolved_stt_model_source_for_factory(tmp_path: Path) -> None:
    """router 應先透過 model_manager 解析 STT 模型來源再交給 factory。"""
    local_model = tmp_path / "fw-small"
    local_model.mkdir()
    seen_models: list[str] = []

    def _capture_stt(cfg: VoiceConfig) -> BaseSTT:
        seen_models.append(cfg.stt_model)
        return _DummySTT()

    config = VoiceConfig(
        stt_backend="faster-whisper",
        stt_model="small",
        stt_model_cache_dir=tmp_path,
        tts_backend="edge-tts",
        stt_model_path=local_model,
    )

    router = VoiceRouter(
        stt_factory=_capture_stt,
        tts_factory=lambda cfg: _DummyTTS(),  # noqa: ARG005
        vad_factory=lambda cfg: _DummyVAD(),  # noqa: ARG005
    )
    await router.load(config)

    assert seen_models == [str(local_model)]


@pytest.mark.asyncio
async def test_voice_router_loads_whisper_cpp_and_piper_with_default_factories(tmp_path: Path) -> None:
    """bounded router 應可建立 whisper-cpp + piper runtime。"""
    config = VoiceConfig(
        stt_backend="whisper-cpp",
        stt_model="base",
        stt_model_path=tmp_path / "whisper.cpp" / "ggml-base.bin",
        tts_backend="piper",
        tts_voice="zh_TW-voice",
        stt_model_cache_dir=tmp_path,
    )

    runtime = await VoiceRouter().load(config)

    assert isinstance(runtime.stt, WhisperCppSTT)
    assert isinstance(runtime.tts, PiperTTS)


@pytest.mark.asyncio
async def test_voice_router_loads_external_api_stt_with_runtime_spec(tmp_path: Path) -> None:
    """bounded router 應可建立 external-api STT，並保留 runtime source 到 status surface。"""
    config = VoiceConfig(
        stt_backend="external-api",
        stt_model="whisper-1",
        stt_openai_base_url="http://api.example.com/v1",
        stt_openai_api_key=SecretStr("sk-test"),
        stt_model_cache_dir=tmp_path,
        tts_backend="edge-tts",
    )

    router = VoiceRouter()
    runtime = await router.load(config)
    router_state = router.describe_router_state()

    assert isinstance(runtime.stt, OpenAIApiSTT)
    assert runtime.stt.get_info().metadata["base_url"] == "http://api.example.com/v1"
    assert runtime.stt.get_info().metadata["api_key_configured"] is True
    assert router_state["stt_runtime_spec"] == {
        "backend": "external-api",
        "requested_model": "whisper-1",
        "model_source": "http://api.example.com/v1",
        "uses_local_model_source": False,
    }


@pytest.mark.asyncio
async def test_voice_router_loads_extended_phase4_backends(tmp_path: Path) -> None:
    """router 應可建立新增的 Phase 4 backend wrapper。"""
    runtime_openai_whisper = await VoiceRouter().load(
        VoiceConfig(
            stt_backend="openai-whisper",
            stt_model="base",
            stt_model_cache_dir=tmp_path,
            tts_backend="coqui-tts",
            tts_model="tts_models/en/ljspeech/tacotron2-DDC",
            tts_voice="speaker-1",
            tts_language="en",
            tts_use_gpu=True,
        )
    )
    assert isinstance(runtime_openai_whisper.stt, OpenAIWhisperSTT)
    assert isinstance(runtime_openai_whisper.tts, CoquiTTS)

    runtime_qwen = await VoiceRouter().load(
        VoiceConfig(
            stt_backend="qwen-asr",
            stt_model="qwen3-asr-0.6b",
            stt_model_cache_dir=tmp_path,
            stt_device="cpu",
            tts_backend="external-api",
            tts_model="gpt-4o-mini-tts",
            tts_voice="alloy",
            tts_openai_base_url="http://api.example.com/v1",
            tts_openai_api_key=SecretStr("sk-test"),
        )
    )
    assert isinstance(runtime_qwen.stt, QwenASRSTT)
    assert isinstance(runtime_qwen.tts, OpenAITTS)
    assert runtime_qwen.tts.get_info().metadata["api_key_configured"] is True


@pytest.mark.asyncio
async def test_voice_router_resolves_registered_tts_voice_id_for_custom_voice_pack(
    tmp_path: Path,
) -> None:
    custom_voice_path = tmp_path / "voices" / "custom-piper.onnx"
    custom_voice_path.parent.mkdir(parents=True, exist_ok=True)
    custom_voice_path.write_bytes(b"fake-voice")
    seen_tts_voice: dict[str, str] = {}

    def _tts_factory(cfg: VoiceConfig) -> _DummyTTS:
        seen_tts_voice["value"] = str(cfg.tts_voice)
        return _DummyTTS()

    runtime = await VoiceRouter(
        stt_factory=lambda cfg: _DummySTT(),  # noqa: ARG005
        tts_factory=_tts_factory,
        vad_factory=lambda cfg: _DummyVAD(),  # noqa: ARG005
    ).load(
        VoiceConfig(
            tts_backend="piper",
            tts_voice="custom-piper",
            stt_model_cache_dir=tmp_path,
            registered_tts_voices=[
                RegisteredTTSVoiceConfig(
                    id="custom-piper",
                    backend="piper",
                    path=custom_voice_path,
                    label="Custom Piper",
                    source="registered_path",
                )
            ],
        )
    )

    assert isinstance(runtime.tts, _DummyTTS)
    assert seen_tts_voice["value"] == str(custom_voice_path)


@pytest.mark.asyncio
async def test_voice_router_loads_vosk_and_whisperlivekit(tmp_path: Path) -> None:
    """router 應可建立其餘新增 STT backend wrapper。"""
    runtime_vosk = await VoiceRouter().load(
        VoiceConfig(
            stt_backend="vosk",
            stt_model=str(tmp_path / "vosk-model"),
            stt_model_cache_dir=tmp_path,
            tts_backend="kokoro-tts",
            tts_voice="af_heart",
            tts_kokoro_lang_code="a",
        )
    )
    assert isinstance(runtime_vosk.stt, VoskSTT)
    assert isinstance(runtime_vosk.tts, KokoroTTS)

    runtime_whisperlivekit = await VoiceRouter().load(
        VoiceConfig(
            stt_backend="whisperlivekit",
            stt_model="base",
            stt_model_cache_dir=tmp_path,
            tts_backend="edge-tts",
            sample_rate=24000,
        )
    )
    assert isinstance(runtime_whisperlivekit.stt, WhisperLiveKitSTT)
    assert isinstance(runtime_whisperlivekit.tts, EdgeTTS)
    whisperlivekit_info = runtime_whisperlivekit.stt.get_info()
    assert whisperlivekit_info.metadata["pcm_input"] is True
    assert whisperlivekit_info.metadata["audio_input_contract"] == {
        "transport": "websocket",
        "message_type": "audio_chunk",
        "message_field": "data",
        "payload_encoding": "base64",
        "encoding": "pcm16",
        "sample_format": "s16le",
        "endianness": "little",
        "channels": 1,
        "channel_layout": "mono",
        "sample_rate_hz": 24000,
        "pcm_input": True,
    }


def test_whisperlivekit_runtime_defaults_to_pcm_input_for_voice_ws_contract() -> None:
    """Mochi 的 WhisperLiveKit runtime 預設應對齊 `/v1/voice` raw PCM16 契約。"""
    options = WhisperLiveKitRuntimeOptions()

    assert options.pcm_input is True
    assert options.to_runtime_factory_kwargs()["pcm_input"] is True


def test_fixed_silero_vad_iterator_handles_non_512_input_and_emits_boundaries() -> None:
    model = _FakeSileroModel([0.9, 0.9, 0.1, 0.1, 0.1])
    iterator = FixedVADIterator(
        model,
        threshold=0.5,
        sampling_rate=16000,
        min_silence_duration_ms=64,  # 1024 samples
        speech_pad_ms=0,
    )

    start = iterator([0.0] * 600)
    no_end_yet = iterator([0.0] * 936)
    end = iterator([0.0] * 1024)

    assert start == {"start": 0}
    assert no_end_yet is None
    assert end == {"end": 1024}


def test_fixed_silero_vad_iterator_reset_clears_buffer_and_model_state() -> None:
    model = _FakeSileroModel([0.9])
    iterator = FixedVADIterator(model)
    assert model.reset_calls == 1

    iterator([0.0] * 100)
    assert len(iterator.buffer) == 100

    iterator.reset_states()

    assert len(iterator.buffer) == 0
    assert model.reset_calls == 2


@pytest.mark.asyncio
async def test_voice_router_default_vad_prefers_silero_when_loader_is_available(
    tmp_path: Path,
) -> None:
    config = VoiceConfig(
        stt_backend="faster-whisper",
        tts_backend="edge-tts",
        stt_model_cache_dir=tmp_path,
        sample_rate=16000,
    )

    with patch("mochi.voice.vad._load_silero_model", return_value=_FakeSileroModel([0.9])):
        runtime = await VoiceRouter(
            stt_factory=lambda cfg: _DummySTT(),  # noqa: ARG005
            tts_factory=lambda cfg: _DummyTTS(),  # noqa: ARG005
        ).load(config)

    assert isinstance(runtime.vad, SileroIteratorVAD)


@pytest.mark.asyncio
async def test_voice_router_default_vad_falls_back_to_energy_when_silero_loader_fails(
    tmp_path: Path,
) -> None:
    config = VoiceConfig(
        stt_backend="faster-whisper",
        tts_backend="edge-tts",
        stt_model_cache_dir=tmp_path,
        sample_rate=16000,
    )

    with patch("mochi.voice.vad._load_silero_model", side_effect=RuntimeError("boom")):
        runtime = await VoiceRouter(
            stt_factory=lambda cfg: _DummySTT(),  # noqa: ARG005
            tts_factory=lambda cfg: _DummyTTS(),  # noqa: ARG005
        ).load(config)

    assert isinstance(runtime.vad, SimpleEnergyVAD)


@pytest.mark.asyncio
async def test_engine_voice_runtime_status_reports_loaded_injected_components(tmp_path: Path) -> None:
    """engine 應提供可供 API 共用的 voice runtime status。"""
    config = MochiConfig(
        voice=VoiceConfig(
            enabled=True,
            stt_backend="faster-whisper",
            tts_backend="edge-tts",
            stt_model_cache_dir=tmp_path,
        )
    )
    engine = AgentEngine(
        config,
        voice_stt=_DummySTT(),
        voice_tts=_DummyTTS(),
        voice_vad=_DummyVAD(),
    )

    status = await engine.get_voice_runtime_status()

    assert status["type"] == "voice_runtime_status"
    assert status["phase"] == "bounded"
    assert status["loaded"] is True
    assert status["supported_backends"]["stt"] == [
        "auto",
        "external-api",
        "faster-whisper",
        "openai-api",
        "openai-whisper",
        "qwen-asr",
        "vosk",
        "whisper-cpp",
        "whisperlivekit",
    ]
    assert status["supported_backends"]["tts"] == [
        "auto",
        "coqui-tts",
        "edge-tts",
        "external-api",
        "kokoro-tts",
        "openai-tts",
        "piper",
    ]
    assert status["stt"]["configured_backend"] == "faster-whisper"
    assert status["tts"]["configured_backend"] == "edge-tts"
    assert status["stt"]["healthy"] is True
    assert status["tts"]["healthy"] is True
    assert status["configured"]["reply_model_mode"] == "inherit_active"
    assert status["configured"]["reply_model_id"] is None
    assert status["configured"]["session_mode"] == "append_current"
    assert status["configured"]["registered_tts_voices"] == []
    assert status["features"]["transcription_preview"] is False
    assert status["session_diagnostics"] == {
        "cached_session_count": 0,
        "active_preview_session_count": 0,
        "preview_disabled_session_count": 0,
        "watchdog": {
            "sessions_with_state": 0,
            "reset_total": 0,
            "runtime_rebuild_total": 0,
            "last_reason_counts": {},
        },
        "sessions": {},
    }

    await engine.close()


@pytest.mark.asyncio
async def test_engine_prepare_voice_runtime_prewarms_requested_session(tmp_path: Path) -> None:
    """prepare_voice_runtime 會先確保 runtime 可用，並建立指定 voice session。"""
    config = MochiConfig(
        voice=VoiceConfig(
            enabled=True,
            stt_backend="faster-whisper",
            tts_backend="edge-tts",
            stt_model_cache_dir=tmp_path,
        )
    )
    engine = AgentEngine(
        config,
        voice_stt=_DummySTT(),
        voice_tts=_DummyTTS(),
        voice_vad=_DummyVAD(),
    )

    status = await engine.prepare_voice_runtime(session_id="voice-session-1")

    assert status["loaded"] is True
    assert status["session_diagnostics"]["cached_session_count"] == 1
    assert "voice-session-1" in status["session_diagnostics"]["sessions"]

    await engine.close()


@pytest.mark.asyncio
async def test_engine_voice_runtime_status_exposes_whisperlivekit_pcm_input_contract(
    tmp_path: Path,
) -> None:
    """status surface 應揭露 WhisperLiveKit 對齊 `/v1/voice` 的 PCM16 契約。"""
    config = MochiConfig(
        voice=VoiceConfig(
            enabled=True,
            stt_backend="whisperlivekit",
            stt_model="base",
            stt_model_cache_dir=tmp_path,
            tts_backend="edge-tts",
            sample_rate=22050,
        )
    )
    engine = AgentEngine(
        config,
        voice_stt=WhisperLiveKitSTT(
            runtime=_FakeWhisperRuntime(),
            audio_input_contract=config.voice.voice_input_contract,
        ),
        voice_tts=_DummyTTS(),
        voice_vad=_DummyVAD(),
    )

    status = await engine.get_voice_runtime_status()

    assert status["configured"]["channels"] == 1
    assert status["configured"]["voice_input_contract"] == {
        "transport": "websocket",
        "message_type": "audio_chunk",
        "message_field": "data",
        "payload_encoding": "base64",
        "encoding": "pcm16",
        "sample_format": "s16le",
        "endianness": "little",
        "channels": 1,
        "channel_layout": "mono",
        "sample_rate_hz": 22050,
        "pcm_input": True,
    }
    assert status["configured"]["voice_input_channel_policy"] == {
        "mode": "mono-only",
        "configured_channels": 1,
        "supported_channels": [1],
        "validation_message": (
            "voice.channels must be 1 because /v1/voice only accepts mono "
            "PCM16 input (base64-encoded s16le)."
        ),
    }
    assert status["stt"]["configured_backend"] == "whisperlivekit"
    assert status["stt"]["info"]["metadata"]["pcm_input"] is True
    assert status["features"]["transcription_preview"] is True
    assert status["stt"]["info"]["metadata"]["audio_input_contract"] == {
        "transport": "websocket",
        "message_type": "audio_chunk",
        "message_field": "data",
        "payload_encoding": "base64",
        "encoding": "pcm16",
        "sample_format": "s16le",
        "endianness": "little",
        "channels": 1,
        "channel_layout": "mono",
        "sample_rate_hz": 22050,
        "pcm_input": True,
    }

    await engine.close()


@pytest.mark.asyncio
async def test_engine_voice_runtime_status_exposes_voice_reply_and_registry_metadata(
    tmp_path: Path,
) -> None:
    custom_voice_path = tmp_path / "packs" / "custom-piper.onnx"
    custom_voice_path.parent.mkdir(parents=True, exist_ok=True)
    custom_voice_path.write_bytes(b"voice")
    engine = AgentEngine(
        MochiConfig(
            voice=VoiceConfig(
                enabled=True,
                stt_backend="faster-whisper",
                tts_backend="piper",
                tts_voice="custom-piper",
                stt_model_cache_dir=tmp_path,
                reply_model_mode="configured_model",
                reply_model_id="voice-openai",
                session_mode="isolated_voice",
                voice_pack_dir=tmp_path / "voice-packs",
                registered_tts_voices=[
                    RegisteredTTSVoiceConfig(
                        id="custom-piper",
                        backend="piper",
                        path=custom_voice_path,
                        label="Custom Piper",
                        source="registered_path",
                    )
                ],
            )
        ),
        voice_stt=_DummySTT(),
        voice_tts=_DummyTTS(),
        voice_vad=_DummyVAD(),
    )

    status = await engine.get_voice_runtime_status()

    assert status["configured"]["reply_model_mode"] == "configured_model"
    assert status["configured"]["reply_model_id"] == "voice-openai"
    assert status["configured"]["session_mode"] == "isolated_voice"
    assert status["configured"]["voice_pack_dir"] == str(tmp_path / "voice-packs")
    assert status["configured"]["registered_tts_voices"] == [
        {
            "id": "custom-piper",
            "backend": "piper",
            "path": str(custom_voice_path),
            "label": "Custom Piper",
            "source": "registered_path",
        }
    ]

    await engine.close()


@pytest.mark.asyncio
async def test_engine_voice_runtime_status_exposes_preview_watchdog_summary(
    tmp_path: Path,
) -> None:
    """status surface 應可看見 cached session 的 preview watchdog 摘要。"""
    config = MochiConfig(
        voice=VoiceConfig(
            enabled=True,
            stt_backend="whisperlivekit",
            stt_model="base",
            stt_model_cache_dir=tmp_path,
            tts_backend="edge-tts",
        )
    )

    runtime_instances: list[_FakeWLKRuntime] = []
    processor_instances: list[_SequencedWLKPreviewProcessor] = []

    def _runtime_factory(**kwargs: Any) -> _FakeWLKRuntime:
        runtime = _FakeWLKRuntime()
        runtime.calls.append({"factory_kwargs": kwargs})
        runtime_instances.append(runtime)
        return runtime

    def _processor_factory(**kwargs: Any) -> _SequencedWLKPreviewProcessor:
        del kwargs
        results = [None] if not processor_instances else [{"lines": [{"text": "recovered"}]}]
        processor = _SequencedWLKPreviewProcessor(results=results)
        processor_instances.append(processor)
        return processor

    engine = AgentEngine(
        config,
        voice_stt=WhisperLiveKitSTT(
            service=WhisperLiveKitService(
                options=_watchdog_options(),
                runtime_factory=_runtime_factory,
                audio_processor_factory=_processor_factory,
            ),
            audio_input_contract=config.voice.voice_input_contract,
        ),
        voice_tts=_DummyTTS(),
        voice_vad=_DummyVAD(),
    )

    session = await engine.get_or_create_voice_session(session_id="watchdog-s1")
    await session.append_audio_chunk_with_vad(b"chunk-1", session_id="watchdog-s1")
    await asyncio.sleep(0.12)

    status = await engine.get_voice_runtime_status()

    assert len(runtime_instances) == 2
    assert len(processor_instances) == 2
    assert status["session_diagnostics"]["cached_session_count"] == 1
    assert status["session_diagnostics"]["active_preview_session_count"] == 1
    assert status["session_diagnostics"]["watchdog"] == {
        "sessions_with_state": 1,
        "reset_total": 1,
        "runtime_rebuild_total": 1,
        "last_reason_counts": {"no_result": 1},
    }
    preview_state = status["session_diagnostics"]["sessions"]["watchdog-s1"]["preview_session"]["state"]
    assert preview_state["first_audio_ts"] is None
    assert isinstance(preview_state["last_audio_ts"], float)
    assert isinstance(preview_state["last_result_ts"], float)
    assert preview_state["result_seen"] is False
    assert preview_state["processor_builds"] == 2
    assert preview_state["watchdog_resets"] == 1
    assert preview_state["watchdog_runtime_rebuilds"] == 1
    assert preview_state["watchdog_last_reason"] == "no_result"

    await engine.close()
