"""Voice router（Phase 4 bounded voice backend matrix）。"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from mochi.config.schema import VoiceConfig
from mochi.voice.base import BaseSTT, BaseTTS, BaseVAD
from mochi.voice.model_manager import STTRuntimeSpec, resolve_bounded_stt_runtime_spec
from mochi.voice.stt.faster_whisper import FasterWhisperSTT
from mochi.voice.stt.openai_api import OpenAIApiSTT
from mochi.voice.stt.openai_whisper import OpenAIWhisperSTT
from mochi.voice.stt.qwen_asr import QwenASRSTT
from mochi.voice.stt.vosk import VoskSTT
from mochi.voice.stt.whisper_cpp import WhisperCppSTT
from mochi.voice.stt.whisperlivekit import WhisperLiveKitSTT
from mochi.voice.tts.coqui_tts import CoquiTTS
from mochi.voice.tts.edge_tts import EdgeTTS
from mochi.voice.tts.kokoro_tts import KokoroTTS
from mochi.voice.tts.openai_tts import OpenAITTS
from mochi.voice.tts.piper import PiperTTS
from mochi.voice.vad import build_default_vad

SUPPORTED_STT_BACKENDS = frozenset(
    {
        "auto",
        "faster-whisper",
        "openai-api",
        "external-api",
        "openai-whisper",
        "qwen-asr",
        "vosk",
        "whisper-cpp",
        "whisperlivekit",
    }
)
SUPPORTED_TTS_BACKENDS = frozenset(
    {
        "auto",
        "coqui-tts",
        "edge-tts",
        "external-api",
        "kokoro-tts",
        "openai-tts",
        "piper",
    }
)


@dataclass(slots=True)
class VoiceRuntime:
    """已載入的語音執行個體集合。"""

    stt: BaseSTT
    tts: BaseTTS
    vad: BaseVAD


class VoiceRouter:
    """語音路由器（只支援 bounded Phase 4 路徑）。"""

    def __init__(
        self,
        *,
        stt_factory: Callable[[VoiceConfig], BaseSTT] | None = None,
        tts_factory: Callable[[VoiceConfig], BaseTTS] | None = None,
        vad_factory: Callable[[VoiceConfig], BaseVAD] | None = None,
    ) -> None:
        self._stt_factory = stt_factory or self._default_stt_factory
        self._tts_factory = tts_factory or self._default_tts_factory
        self._vad_factory = vad_factory or self._default_vad_factory
        self._active: VoiceRuntime | None = None
        self._last_stt_runtime_spec: STTRuntimeSpec | None = None
        self._last_load_error: str | None = None

    @property
    def active(self) -> VoiceRuntime:
        if self._active is None:
            raise RuntimeError("No voice runtime loaded. Call load() first.")
        return self._active

    @property
    def last_stt_runtime_spec(self) -> STTRuntimeSpec | None:
        """最近一次 load 的 STT runtime 解析結果。"""
        return self._last_stt_runtime_spec

    @property
    def last_load_error(self) -> str | None:
        """最近一次 load 失敗訊息。"""
        return self._last_load_error

    async def load(self, config: VoiceConfig) -> VoiceRuntime:
        self._validate_bounded_backends(config)
        stt_runtime_spec = resolve_bounded_stt_runtime_spec(
            stt_backend=config.stt_backend,
            stt_model=config.stt_model,
            stt_model_cache_dir=config.stt_model_cache_dir,
            stt_model_path=config.stt_model_path,
            stt_openai_base_url=config.stt_openai_base_url,
        )
        effective_config = config
        if (
            stt_runtime_spec.backend != "openai-api"
            and stt_runtime_spec.model_source
            and stt_runtime_spec.model_source != config.stt_model
        ):
            effective_config = config.model_copy(update={"stt_model": stt_runtime_spec.model_source})
        resolved_tts_voice = _resolve_registered_tts_voice_path(effective_config)
        if resolved_tts_voice != effective_config.tts_voice:
            effective_config = effective_config.model_copy(update={"tts_voice": resolved_tts_voice})

        try:
            runtime = VoiceRuntime(
                stt=self._stt_factory(effective_config),
                tts=self._tts_factory(effective_config),
                vad=self._vad_factory(effective_config),
            )
        except Exception as exc:
            self._last_load_error = str(exc)
            raise

        if self._active is not None:
            await self.close()
        self._active = runtime
        self._last_stt_runtime_spec = stt_runtime_spec
        self._last_load_error = None
        return runtime

    def create_vad(self, config: VoiceConfig) -> BaseVAD:
        """建立新的 VAD 實例（供每個 VoiceSession 獨立持有）。"""
        return self._vad_factory(config)

    async def close(self) -> None:
        if self._active is None:
            return
        await self._active.stt.close()
        await self._active.tts.close()
        self._active = None

    @staticmethod
    def _default_stt_factory(config: VoiceConfig) -> BaseSTT:
        backend = _normalize_backend_name(config.stt_backend)
        if backend in {"auto", "faster-whisper"}:
            return FasterWhisperSTT(
                model=config.stt_model,
                device=config.stt_device,
                language=config.stt_language,
                model_cache_dir=config.stt_model_cache_dir,
            )
        if backend == "whisper-cpp":
            return WhisperCppSTT(
                model=config.stt_model,
                model_path=config.stt_model_path,
                language=config.stt_language,
            )
        if backend == "openai-whisper":
            return OpenAIWhisperSTT(
                model=config.stt_model,
                device=config.stt_device,
                language=config.stt_language,
                model_cache_dir=config.stt_model_cache_dir,
            )
        if backend == "vosk":
            return VoskSTT(
                model=config.stt_model,
                language=config.stt_language,
            )
        if backend == "qwen-asr":
            return QwenASRSTT(
                model=config.stt_model,
                language=config.stt_language,
                device=config.stt_device,
            )
        if backend == "whisperlivekit":
            return WhisperLiveKitSTT(
                model=config.stt_model,
                language=config.stt_language,
                audio_input_contract=config.voice_input_contract,
            )
        if backend in {"openai-api", "external-api"}:
            return OpenAIApiSTT(
                model=config.stt_model,
                base_url=config.stt_openai_base_url,
                api_key=(
                    config.stt_openai_api_key.get_secret_value()
                    if config.stt_openai_api_key is not None
                    else ""
                ),
                language=config.stt_language,
                timeout=config.stt_openai_timeout,
            )
        raise ValueError(
            "Unsupported STT backend in bounded runtime: "
            f"{config.stt_backend!r}. Supported: {sorted(SUPPORTED_STT_BACKENDS)}."
        )

    @staticmethod
    def _default_tts_factory(config: VoiceConfig) -> BaseTTS:
        backend = _normalize_backend_name(config.tts_backend)
        if backend in {"auto", "edge-tts"}:
            return EdgeTTS(voice=config.tts_voice, speed=config.tts_speed)
        if backend == "piper":
            return PiperTTS(voice=config.tts_voice, speed=config.tts_speed)
        if backend == "coqui-tts":
            kwargs: dict[str, Any] = {
                "voice": config.tts_voice,
                "language": config.tts_language,
                "speed": config.tts_speed,
                "use_gpu": config.tts_use_gpu,
            }
            if config.tts_model:
                kwargs["model"] = config.tts_model
            return CoquiTTS(**kwargs)
        if backend == "kokoro-tts":
            return KokoroTTS(
                voice=config.tts_voice,
                lang_code=config.tts_kokoro_lang_code,
                speed=config.tts_speed,
                split_pattern=config.tts_split_pattern,
            )
        if backend in {"openai-tts", "external-api"}:
            kwargs: dict[str, Any] = {
                "voice": config.tts_voice,
                "speed": config.tts_speed,
                "base_url": config.tts_openai_base_url,
                "api_key": (
                    config.tts_openai_api_key.get_secret_value()
                    if config.tts_openai_api_key is not None
                    else ""
                ),
                "response_format": config.tts_openai_response_format,
                "timeout": config.tts_openai_timeout,
            }
            if config.tts_model:
                kwargs["model"] = config.tts_model
            return OpenAITTS(**kwargs)
        raise ValueError(
            "Unsupported TTS backend in bounded runtime: "
            f"{config.tts_backend!r}. Supported: {sorted(SUPPORTED_TTS_BACKENDS)}."
        )

    @staticmethod
    def _default_vad_factory(config: VoiceConfig) -> BaseVAD:
        return build_default_vad(
            sample_rate=config.sample_rate,
            threshold=config.vad_threshold,
            min_speech_ms=config.vad_min_speech_ms,
            max_silence_ms=config.vad_max_silence_ms,
        )

    @staticmethod
    def _validate_bounded_backends(config: VoiceConfig) -> None:
        stt_backend = _normalize_backend_name(config.stt_backend)
        if stt_backend not in SUPPORTED_STT_BACKENDS:
            raise ValueError(
                "Unsupported STT backend in bounded runtime: "
                f"{config.stt_backend!r}. Supported: {sorted(SUPPORTED_STT_BACKENDS)}."
            )

        tts_backend = _normalize_backend_name(config.tts_backend)
        if tts_backend not in SUPPORTED_TTS_BACKENDS:
            raise ValueError(
                "Unsupported TTS backend in bounded runtime: "
                f"{config.tts_backend!r}. Supported: {sorted(SUPPORTED_TTS_BACKENDS)}."
            )

    def describe_router_state(self) -> dict[str, Any]:
        """輸出 router 當前狀態摘要（供 runtime status surface 使用）。"""
        payload: dict[str, Any] = {
            "loaded": self._active is not None,
            "supported_stt_backends": sorted(SUPPORTED_STT_BACKENDS),
            "supported_tts_backends": sorted(SUPPORTED_TTS_BACKENDS),
        }
        if self._last_stt_runtime_spec is not None:
            payload["stt_runtime_spec"] = self._last_stt_runtime_spec.to_dict()
        if self._last_load_error:
            payload["last_load_error"] = self._last_load_error
        return payload


def _normalize_backend_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _resolve_registered_tts_voice_path(config: VoiceConfig) -> str:
    requested_voice = str(config.tts_voice)
    backend = _normalize_backend_name(config.tts_backend)
    for voice in config.registered_tts_voices:
        if voice.id != requested_voice:
            continue
        if voice.backend is not None and backend not in {"auto", _normalize_backend_name(voice.backend)}:
            continue
        return str(voice.path)
    return requested_voice
