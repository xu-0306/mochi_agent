"""PiperTTS backend 測試。"""

from __future__ import annotations

import builtins
import io
import wave
from unittest.mock import patch

import pytest

from mochi.voice.tts.piper import PiperTTS


def _build_wav_pcm16(*, channels: int, sample_rate: int, samples: list[int]) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"".join(int(sample).to_bytes(2, "little", signed=True) for sample in samples))
    return buffer.getvalue()


@pytest.mark.asyncio
async def test_piper_tts_with_injected_callable_normalizes_wav_to_mono_pcm16() -> None:
    stereo_wav = _build_wav_pcm16(
        channels=2,
        sample_rate=22050,
        samples=[1000, -1000, 3000, 1000],
    )
    tts = PiperTTS(
        voice="fake-voice.onnx",
        synthesize_callable=lambda text, voice, speed: stereo_wav,  # noqa: ARG005
    )

    audio = await tts.synthesize("hello")

    assert audio == b"\x00\x00\xd0\x07"  # [0, 2000] in little-endian PCM16
    assert await tts.health_check() is True
    assert tts.get_info().family == "piper"


@pytest.mark.asyncio
async def test_piper_tts_with_injected_callable_accepts_raw_pcm16() -> None:
    raw_pcm = b"\x01\x00\x02\x00"
    tts = PiperTTS(
        voice="fake-voice.onnx",
        synthesize_callable=lambda text, voice, speed: raw_pcm,  # noqa: ARG005
    )

    audio = await tts.synthesize("hello")

    assert audio == raw_pcm


@pytest.mark.asyncio
async def test_piper_tts_dependency_missing_reports_health_and_error() -> None:
    original_import = builtins.__import__

    def _fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "piper":
            raise ImportError("piper is missing")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_fake_import):
        tts = PiperTTS(voice="missing.onnx")

    assert await tts.health_check() is False
    with pytest.raises(
        RuntimeError,
        match=r"piper synthesize unavailable \[dependency_missing\]",
    ):
        await tts.synthesize("hello")

