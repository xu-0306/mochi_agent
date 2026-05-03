"""Silero VAD iterator（從參考實作最小移植）。"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any


# Attribution:
# Adapted from:
# /mnt/g/_python/STT&TTS/SimulStreaming/whisper_streaming/silero_vad_iterator.py
# Original upstream notes in that file indicate it is copied from
# silero-vad utils_vad.py under MIT License:
# https://github.com/snakers4/silero-vad/blob/94811cbe1207ec24bc0f5370b895364b8934936f/src/silero_vad/utils_vad.py
class VADIterator:
    """模擬串流 VAD 的逐塊迭代器。"""

    def __init__(
        self,
        model: Any,
        threshold: float = 0.5,
        sampling_rate: int = 16000,
        min_silence_duration_ms: int = 500,
        speech_pad_ms: int = 100,
    ) -> None:
        self.model = model
        self.threshold = threshold
        self.sampling_rate = sampling_rate

        if sampling_rate not in (8000, 16000):
            raise ValueError("VADIterator does not support sampling rates other than [8000, 16000]")

        self.min_silence_samples = sampling_rate * min_silence_duration_ms / 1000
        self.speech_pad_samples = sampling_rate * speech_pad_ms / 1000
        self.reset_states()

    def reset_states(self) -> None:
        self.model.reset_states()
        self.triggered = False
        self.temp_end = 0
        self.current_sample = 0

    def __call__(
        self,
        x: Sequence[float] | Any,
        return_seconds: bool = False,
        time_resolution: int = 1,
    ) -> dict[str, int | float] | None:
        window_size_samples = _window_size(x)
        self.current_sample += window_size_samples

        speech_prob = float(_as_model_score(self.model(x, self.sampling_rate)))

        if (speech_prob >= self.threshold) and self.temp_end:
            self.temp_end = 0

        if (speech_prob >= self.threshold) and not self.triggered:
            self.triggered = True
            speech_start = max(0, self.current_sample - self.speech_pad_samples - window_size_samples)
            return {
                "start": (
                    int(speech_start)
                    if not return_seconds
                    else round(speech_start / self.sampling_rate, time_resolution)
                )
            }

        if (speech_prob < self.threshold - 0.15) and self.triggered:
            if not self.temp_end:
                self.temp_end = self.current_sample
            if self.current_sample - self.temp_end < self.min_silence_samples:
                return None
            speech_end = self.temp_end + self.speech_pad_samples - window_size_samples
            self.temp_end = 0
            self.triggered = False
            return {
                "end": (
                    int(speech_end)
                    if not return_seconds
                    else round(speech_end / self.sampling_rate, time_resolution)
                )
            }

        return None


class FixedVADIterator(VADIterator):
    """支援任意長度輸入，內部按 512-sample chunk 送入 VAD。"""

    def reset_states(self) -> None:
        super().reset_states()
        self.buffer: list[float] = []

    def __call__(self, x: Sequence[float], return_seconds: bool = False) -> dict[str, int | float] | None:
        self.buffer.extend(float(v) for v in x)
        ret: dict[str, int | float] | None = None

        while len(self.buffer) >= 512:
            r = super().__call__(self.buffer[:512], return_seconds=return_seconds)
            self.buffer = self.buffer[512:]
            if ret is None:
                ret = r
            elif r is not None:
                if "end" in r:
                    ret["end"] = r["end"]
                if ("start" in r) and ("end" in ret):
                    del ret["end"]

        return ret if ret != {} else None


def _window_size(x: Sequence[float] | Any) -> int:
    # 相容 tensor-like 2D 輸入：沿用來源實作的 x[0] 長度判斷。
    if hasattr(x, "dim") and callable(x.dim):
        if x.dim() == 2:
            return len(x[0])
        return len(x)
    if isinstance(x, Sequence) and x and isinstance(x[0], Sequence):
        return len(x[0])
    return len(x)


def _as_model_score(v: Any) -> float:
    if hasattr(v, "item") and callable(v.item):
        return float(v.item())
    return float(v)
