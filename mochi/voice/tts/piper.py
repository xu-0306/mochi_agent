"""piper TTS 包裝（bounded Phase 4，輸出 mono PCM16）。"""

from __future__ import annotations

import asyncio
import inspect
import io
import wave
from collections.abc import Callable
from struct import iter_unpack, pack
from typing import Any

from mochi.voice.base import BaseTTS, VoiceInfo

PiperSynthesizeCallable = Callable[[str, str, float], bytes | bytearray]


class PiperTTS(BaseTTS):
    """piper 家族 TTS 最小封裝。"""

    def __init__(
        self,
        *,
        voice: str,
        speed: float = 1.0,
        speaker_id: int | None = None,
        synthesize_callable: PiperSynthesizeCallable | None = None,
    ) -> None:
        self.voice = voice
        self.speed = speed
        self.speaker_id = speaker_id
        self._synthesize_callable = synthesize_callable
        self._dependency_error: Exception | None = None

        if self._synthesize_callable is None:
            try:
                import piper  # type: ignore[import-not-found]
            except Exception as exc:
                self._dependency_error = exc
            else:
                try:
                    self._synthesize_callable = self._build_default_synthesize_callable(piper)
                except Exception as exc:
                    self._dependency_error = exc

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float | None = None,
    ) -> bytes:
        if not text:
            return b""

        if self._dependency_error is not None:
            raise RuntimeError(
                "piper synthesize unavailable [dependency_missing]: "
                f"{self._dependency_error}"
            ) from self._dependency_error
        if self._synthesize_callable is None:
            raise RuntimeError("piper synthesize unavailable [factory_missing]")

        selected_voice = voice or self.voice
        selected_speed = self.speed if speed is None else speed

        try:
            raw_audio = await asyncio.to_thread(
                self._synthesize_callable,
                text,
                selected_voice,
                selected_speed,
            )
            return self._normalize_to_pcm16_mono(raw_audio)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"piper synthesize unavailable [synthesize_failed]: {exc}") from exc

    def get_info(self) -> VoiceInfo:
        return VoiceInfo(
            kind="tts",
            family="piper",
            name=self.voice,
            metadata={
                "speed": self.speed,
                "speaker_id": self.speaker_id,
                "dependency_ready": self._dependency_error is None,
            },
        )

    async def health_check(self) -> bool:
        return self._dependency_error is None

    def _build_default_synthesize_callable(self, piper_module: Any) -> PiperSynthesizeCallable:
        voice_cls = getattr(piper_module, "PiperVoice", None)
        if voice_cls is None or not hasattr(voice_cls, "load"):
            raise RuntimeError("Unsupported piper runtime API: missing PiperVoice.load")

        cache: dict[str, Any] = {}

        def _synthesize(text: str, selected_voice: str, speed: float) -> bytes:
            runtime = cache.get(selected_voice)
            if runtime is None:
                runtime = voice_cls.load(selected_voice)
                cache[selected_voice] = runtime

            wav_bytes = io.BytesIO()
            with wave.open(wav_bytes, "wb") as wav_file:
                _call_with_supported_kwargs(
                    runtime.synthesize,
                    text,
                    wav_file,
                    speaker_id=self.speaker_id,
                    length_scale=self._speed_to_length_scale(speed),
                )
            return wav_bytes.getvalue()

        return _synthesize

    @staticmethod
    def _speed_to_length_scale(speed: float) -> float:
        if speed <= 0:
            return 1.0
        return 1.0 / speed

    def _normalize_to_pcm16_mono(self, audio: bytes | bytearray) -> bytes:
        """將輸入音訊正規化為 mono PCM16 bytes。"""
        data = bytes(audio)
        if not data:
            return b""

        if self._looks_like_wav(data):
            return self._wav_to_pcm16_mono(data)

        if len(data) % 2 != 0:
            raise RuntimeError(
                "piper synthesize unavailable [invalid_audio_format]: "
                "Expected PCM16 bytes length to be even."
            )
        return data

    @staticmethod
    def _looks_like_wav(data: bytes) -> bool:
        return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"

    def _wav_to_pcm16_mono(self, wav_data: bytes) -> bytes:
        try:
            with wave.open(io.BytesIO(wav_data), "rb") as wav_file:
                channels = wav_file.getnchannels()
                sample_width = wav_file.getsampwidth()
                frames = wav_file.readframes(wav_file.getnframes())
        except Exception as exc:
            raise RuntimeError(
                f"piper synthesize unavailable [invalid_audio_format]: {exc}"
            ) from exc

        if sample_width != 2:
            raise RuntimeError(
                "piper synthesize unavailable [invalid_audio_format]: "
                f"Expected 16-bit WAV, got sample width={sample_width}."
            )
        if channels <= 0:
            raise RuntimeError(
                "piper synthesize unavailable [invalid_audio_format]: invalid channel count."
            )
        if channels == 1:
            return frames

        return self._downmix_interleaved_pcm16(frames, channels)

    @staticmethod
    def _downmix_interleaved_pcm16(frames: bytes, channels: int) -> bytes:
        sample_count = len(frames) // 2
        if sample_count == 0:
            return b""

        trimmed = frames[: sample_count * 2]
        samples = [sample for (sample,) in iter_unpack("<h", trimmed)]
        frame_count = len(samples) // channels
        if frame_count == 0:
            return b""

        out = bytearray()
        for frame_idx in range(frame_count):
            offset = frame_idx * channels
            mixed = int(sum(samples[offset : offset + channels]) / channels)
            mixed = max(-32768, min(32767, mixed))
            out.extend(pack("<h", mixed))
        return bytes(out)


def _call_with_supported_kwargs(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """只傳入目標函式支援的 keyword 參數。"""
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(*args, **kwargs)

    accepts_var_keyword = any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if accepts_var_keyword:
        return func(*args, **kwargs)

    supported_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return func(*args, **supported_kwargs)
