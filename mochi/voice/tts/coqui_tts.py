"""Coqui TTS 包裝（bounded Phase 4，輸出 mono PCM16）。"""

from __future__ import annotations

import asyncio
import inspect
import io
import wave
from collections.abc import Callable, Sequence
from struct import iter_unpack, pack
from typing import Any

from mochi.voice.base import BaseTTS, VoiceInfo

CoquiSynthesizeCallable = Callable[[str, str | None, str | None, float], Any]


class CoquiTTS(BaseTTS):
    """coqui-ai/TTS 最小封裝。"""

    def __init__(
        self,
        *,
        model: str = "tts_models/en/ljspeech/tacotron2-DDC",
        voice: str | None = None,
        language: str | None = None,
        speed: float = 1.0,
        use_gpu: bool = False,
        runtime: Any | None = None,
        model_factory: Callable[..., Any] | None = None,
        synthesize_callable: CoquiSynthesizeCallable | None = None,
    ) -> None:
        self.model = model
        self.voice = voice
        self.language = language
        self.speed = speed
        self.use_gpu = use_gpu
        self._runtime = runtime
        self._model_factory = model_factory
        self._synthesize_callable = synthesize_callable
        self._dependency_error: Exception | None = None
        self._runtime_init_error: Exception | None = None

        if (
            self._runtime is None
            and self._synthesize_callable is None
            and self._model_factory is None
        ):
            try:
                from TTS.api import TTS as CoquiModel  # type: ignore[import-not-found]
            except Exception as exc:
                self._dependency_error = exc
            else:
                self._model_factory = CoquiModel

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float | None = None,
    ) -> bytes:
        if not text:
            return b""

        selected_voice = voice if voice is not None else self.voice
        selected_speed = self.speed if speed is None else speed
        selected_language = self.language

        try:
            raw_audio = await asyncio.to_thread(
                self._synthesize_blocking,
                text,
                selected_voice,
                selected_language,
                selected_speed,
            )
            return self._normalize_to_pcm16_mono(raw_audio)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"coqui-tts synthesize unavailable [synthesize_failed]: {exc}") from exc

    def get_info(self) -> VoiceInfo:
        return VoiceInfo(
            kind="tts",
            family="coqui-tts",
            name=self.model,
            metadata={
                "voice": self.voice,
                "language": self.language,
                "speed": self.speed,
                "use_gpu": self.use_gpu,
                "loaded": self._runtime is not None,
                "dependency_ready": self._dependency_error is None,
            },
        )

    async def health_check(self) -> bool:
        if self._dependency_error is not None or self._runtime_init_error is not None:
            return False
        return any(
            candidate is not None
            for candidate in (self._runtime, self._synthesize_callable, self._model_factory)
        )

    async def ensure_ready(self) -> None:
        await asyncio.to_thread(self._ensure_runtime)

    async def close(self) -> None:
        runtime = self._runtime
        self._runtime = None
        if runtime is None:
            return

        for name in ("aclose", "close"):
            closer = getattr(runtime, name, None)
            if not callable(closer):
                continue
            result = closer()
            if inspect.isawaitable(result):
                await result
            break

    def _synthesize_blocking(
        self,
        text: str,
        voice: str | None,
        language: str | None,
        speed: float,
    ) -> Any:
        if self._synthesize_callable is not None:
            return self._synthesize_callable(text, voice, language, speed)

        runtime = self._ensure_runtime()
        synthesize = getattr(runtime, "tts", None)
        if not callable(synthesize):
            raise RuntimeError("coqui-tts synthesize unavailable [factory_missing]")
        return _call_with_supported_kwargs(
            synthesize,
            text,
            speaker=voice,
            language=language,
            speed=speed,
        )

    def _ensure_runtime(self) -> Any:
        if self._runtime is not None:
            return self._runtime

        if self._dependency_error is not None:
            raise RuntimeError(
                "coqui-tts synthesize unavailable [dependency_missing]: "
                f"{self._dependency_error}"
            ) from self._dependency_error

        if self._runtime_init_error is not None:
            raise RuntimeError(
                "coqui-tts synthesize unavailable [runtime_init_failed]: "
                f"{self._runtime_init_error}"
            ) from self._runtime_init_error

        if self._model_factory is None:
            raise RuntimeError("coqui-tts synthesize unavailable [factory_missing]")

        try:
            self._runtime = _call_with_supported_kwargs(
                self._model_factory,
                model_name=self.model,
                progress_bar=False,
                gpu=self.use_gpu,
            )
        except Exception as exc:
            self._runtime_init_error = exc
            raise RuntimeError(
                f"coqui-tts synthesize unavailable [runtime_init_failed]: {exc}"
            ) from exc
        return self._runtime

    def _normalize_to_pcm16_mono(self, audio: Any) -> bytes:
        if isinstance(audio, (bytes, bytearray)):
            return self._normalize_bytes_audio(audio)
        return self._normalize_numeric_audio(audio)

    def _normalize_bytes_audio(self, audio: bytes | bytearray) -> bytes:
        data = bytes(audio)
        if not data:
            return b""
        if _looks_like_wav(data):
            return _wav_to_pcm16_mono("coqui-tts", data)
        if len(data) % 2 != 0:
            raise RuntimeError(
                "coqui-tts synthesize unavailable [invalid_audio_format]: "
                "Expected PCM16 bytes length to be even."
            )
        return data

    def _normalize_numeric_audio(self, audio: Any) -> bytes:
        normalized = _to_plain_audio_object(audio)
        if normalized is None:
            return b""
        if isinstance(normalized, (int, float)):
            return pack("<h", _to_pcm16_sample(normalized))
        if isinstance(normalized, Sequence) and not isinstance(normalized, (str, bytes, bytearray)):
            if not normalized:
                return b""
            first = normalized[0]
            if isinstance(first, Sequence) and not isinstance(first, (str, bytes, bytearray)):
                return _nested_numeric_to_pcm16_mono("coqui-tts", normalized)
            if _is_numeric(first):
                return b"".join(pack("<h", _to_pcm16_sample(value)) for value in normalized)
        raise RuntimeError("coqui-tts synthesize unavailable [invalid_audio_format]: unsupported audio type.")


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

    accepted_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return func(*args, **accepted_kwargs)


def _looks_like_wav(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def _wav_to_pcm16_mono(family: str, wav_data: bytes) -> bytes:
    try:
        with wave.open(io.BytesIO(wav_data), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.readframes(wav_file.getnframes())
    except Exception as exc:
        raise RuntimeError(
            f"{family} synthesize unavailable [invalid_audio_format]: {exc}"
        ) from exc

    if sample_width != 2:
        raise RuntimeError(
            f"{family} synthesize unavailable [invalid_audio_format]: "
            f"Expected 16-bit WAV, got sample width={sample_width}."
        )
    if channels <= 0:
        raise RuntimeError(f"{family} synthesize unavailable [invalid_audio_format]: invalid channel count.")
    if channels == 1:
        return frames
    return _downmix_interleaved_pcm16(frames, channels)


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


def _to_plain_audio_object(audio: Any) -> Any:
    value = audio

    detach = getattr(value, "detach", None)
    if callable(detach):
        value = detach()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    numpy_getter = getattr(value, "numpy", None)
    if callable(numpy_getter):
        value = numpy_getter()
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        value = to_list()

    return value


def _nested_numeric_to_pcm16_mono(family: str, samples_2d: Sequence[Any]) -> bytes:
    rows: list[list[float | int]] = []
    for row in samples_2d:
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise RuntimeError(
                f"{family} synthesize unavailable [invalid_audio_format]: mixed nested audio layout."
            )
        cast_row = list(row)
        if not cast_row:
            continue
        if not all(_is_numeric(item) for item in cast_row):
            raise RuntimeError(
                f"{family} synthesize unavailable [invalid_audio_format]: non-numeric samples."
            )
        rows.append(cast_row)

    if not rows:
        return b""

    row_len = len(rows[0])
    if any(len(row) != row_len for row in rows):
        raise RuntimeError(
            f"{family} synthesize unavailable [invalid_audio_format]: jagged audio matrix."
        )

    # frames x channels（常見）:
    if row_len <= 8:
        mono = [sum(row) / row_len for row in rows]
        return b"".join(pack("<h", _to_pcm16_sample(value)) for value in mono)

    # channels x frames（較少見）:
    if len(rows) <= 8:
        frame_count = row_len
        channel_count = len(rows)
        mono = []
        for frame_idx in range(frame_count):
            mixed = sum(rows[channel_idx][frame_idx] for channel_idx in range(channel_count)) / channel_count
            mono.append(mixed)
        return b"".join(pack("<h", _to_pcm16_sample(value)) for value in mono)

    raise RuntimeError(
        f"{family} synthesize unavailable [invalid_audio_format]: unsupported multi-channel layout."
    )


def _is_numeric(value: Any) -> bool:
    return isinstance(value, (int, float))


def _to_pcm16_sample(value: float | int) -> int:
    if isinstance(value, float):
        scaled = (
            int(round(value * 32767.0))
            if -1.0 <= value <= 1.0
            else int(round(value))
        )
        return max(-32768, min(32767, scaled))

    return max(-32768, min(32767, int(value)))
