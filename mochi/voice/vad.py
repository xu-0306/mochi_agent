"""VAD 實作（Silero iterator + 能量法回退）。"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from math import sqrt
from struct import iter_unpack
from typing import Any

from mochi.voice.base import BaseVAD, VoiceInfo
from mochi.voice.silero_vad_iterator import FixedVADIterator


class SimpleEnergyVAD(BaseVAD):
    """以 PCM16 能量估算語音活動的最小 VAD。"""

    def __init__(
        self,
        *,
        threshold: float = 0.5,
        min_speech_ms: int = 250,
        max_silence_ms: int = 700,
    ) -> None:
        self._threshold = max(0.0, min(1.0, threshold))
        self._min_speech_ms = max(0, min_speech_ms)
        self._max_silence_ms = max(0, max_silence_ms)
        self._speech_ms = 0.0
        self._silence_ms = 0.0

    async def is_speech(self, audio: bytes, *, sample_rate: int) -> bool:
        if sample_rate <= 0:
            raise ValueError("sample_rate must be > 0")
        if not audio:
            self._silence_ms += 0.0
            return False

        frame_ms = (len(audio) / 2.0 / float(sample_rate)) * 1000.0
        energy = self._rms_energy(audio)

        if energy >= self._threshold:
            self._speech_ms += frame_ms
            self._silence_ms = 0.0
        else:
            self._silence_ms += frame_ms

        if self._silence_ms > float(self._max_silence_ms):
            self.reset()
            return False

        return self._speech_ms >= float(self._min_speech_ms)

    def reset(self) -> None:
        self._speech_ms = 0.0
        self._silence_ms = 0.0

    def get_info(self) -> VoiceInfo:
        return VoiceInfo(
            kind="vad",
            family="energy",
            name="simple-energy-vad",
            metadata={
                "threshold": self._threshold,
                "min_speech_ms": self._min_speech_ms,
                "max_silence_ms": self._max_silence_ms,
            },
        )

    @staticmethod
    def _rms_energy(audio: bytes) -> float:
        if len(audio) < 2:
            return 0.0

        square_sum = 0.0
        count = 0
        for (sample,) in iter_unpack("<h", audio[: len(audio) - (len(audio) % 2)]):
            normalized = float(sample) / 32768.0
            square_sum += normalized * normalized
            count += 1

        if count == 0:
            return 0.0
        return sqrt(square_sum / float(count))


class SileroIteratorVAD(BaseVAD):
    """以移植進來的 Silero iterator 狀態機包裝 VAD。"""

    def __init__(
        self,
        model: Any,
        *,
        threshold: float = 0.5,
        sample_rate: int = 16000,
        min_silence_duration_ms: int = 500,
        speech_pad_ms: int = 100,
    ) -> None:
        self._sample_rate = sample_rate
        self._iterator = FixedVADIterator(
            model,
            threshold=threshold,
            sampling_rate=sample_rate,
            min_silence_duration_ms=min_silence_duration_ms,
            speech_pad_ms=speech_pad_ms,
        )

    async def is_speech(self, audio: bytes, *, sample_rate: int) -> bool:
        if sample_rate != self._sample_rate:
            raise ValueError(
                f"SileroIteratorVAD expected sample_rate={self._sample_rate}, got {sample_rate}"
            )

        event = self._iterator(_pcm16_to_float32(audio))
        if event is None:
            return bool(self._iterator.triggered)
        return "start" in event or "end" in event

    def reset(self) -> None:
        self._iterator.reset_states()

    def get_info(self) -> VoiceInfo:
        return VoiceInfo(
            kind="vad",
            family="silero-iterator",
            name="silero-vad-iterator",
            metadata={
                "sampling_rate": self._sample_rate,
                "threshold": self._iterator.threshold,
            },
        )


def build_default_vad(
    *,
    sample_rate: int,
    threshold: float,
    min_speech_ms: int,
    max_silence_ms: int,
    silero_model_loader: Callable[[], Any] | None = None,
) -> BaseVAD:
    """建立預設 VAD：優先 Silero iterator，失敗時回退能量法。"""
    if sample_rate in (8000, 16000):
        loader = silero_model_loader or _load_silero_model
        try:
            model = loader()
            return SileroIteratorVAD(
                model=model,
                threshold=threshold,
                sample_rate=sample_rate,
                min_silence_duration_ms=max_silence_ms,
                speech_pad_ms=min(100, max(0, min_speech_ms)),
            )
        except Exception:
            return SimpleEnergyVAD(
                threshold=threshold,
                min_speech_ms=min_speech_ms,
                max_silence_ms=max_silence_ms,
            )

    return SimpleEnergyVAD(
        threshold=threshold,
        min_speech_ms=min_speech_ms,
        max_silence_ms=max_silence_ms,
    )


def _load_silero_model() -> Any:
    """載入 silero-vad 模型（支援 tuple/list 回傳格式）。"""
    from silero_vad import load_silero_vad  # type: ignore[import-not-found]

    loaded = load_silero_vad()
    if isinstance(loaded, tuple):
        if not loaded:
            raise RuntimeError("silero model loader returned empty tuple")
        return loaded[0]
    if isinstance(loaded, list):
        if not loaded:
            raise RuntimeError("silero model loader returned empty list")
        return loaded[0]
    if inspect.isawaitable(loaded):
        raise TypeError("silero model loader must return synchronously")
    return loaded


def _pcm16_to_float32(audio: bytes) -> list[float]:
    """將 PCM16 little-endian bytes 轉為 [-1, 1] float list。"""
    if len(audio) < 2:
        return []

    values: list[float] = []
    for (sample,) in iter_unpack("<h", audio[: len(audio) - (len(audio) % 2)]):
        values.append(float(sample) / 32768.0)
    return values
