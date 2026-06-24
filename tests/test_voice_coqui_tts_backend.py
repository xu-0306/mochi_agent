"""CoquiTTS backend 測試。"""

from __future__ import annotations

import builtins
from unittest.mock import patch

import pytest

from mochi.voice.tts.coqui_tts import CoquiTTS


class _FakeCoquiRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.closed = False

    def tts(
        self,
        text: str,
        speaker: str | None = None,
        language: str | None = None,
        speed: float | None = None,
    ) -> list[float]:
        self.calls.append(
            {"text": text, "speaker": speaker, "language": language, "speed": speed}
        )
        return [0.0, 0.5, -0.5]

    def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_coqui_tts_with_injected_runtime_outputs_mono_pcm16() -> None:
    runtime = _FakeCoquiRuntime()
    tts = CoquiTTS(
        model="tts_models/en/ljspeech/tacotron2-DDC",
        runtime=runtime,
        voice="speaker-a",
        language="en",
        speed=1.2,
    )

    audio = await tts.synthesize("hello")

    assert audio == b"\x00\x00\x00@\x00\xc0"
    assert runtime.calls[0] == {
        "text": "hello",
        "speaker": "speaker-a",
        "language": "en",
        "speed": 1.2,
    }
    assert await tts.health_check() is True
    assert tts.get_info().family == "coqui-tts"

    await tts.close()
    assert runtime.closed is True


@pytest.mark.asyncio
async def test_coqui_tts_model_factory_failure_semantics() -> None:
    seen: dict[str, object] = {}

    def _broken_factory(**kwargs: object) -> object:
        seen["kwargs"] = kwargs
        raise RuntimeError("load failed")

    tts = CoquiTTS(
        model="tts_models/en/ljspeech/tacotron2-DDC",
        use_gpu=True,
        model_factory=_broken_factory,
    )

    with pytest.raises(RuntimeError, match=r"coqui-tts synthesize unavailable \[runtime_init_failed\]"):
        await tts.synthesize("hello")

    assert seen["kwargs"] == {
        "model_name": "tts_models/en/ljspeech/tacotron2-DDC",
        "progress_bar": False,
        "gpu": True,
    }
    assert await tts.health_check() is False


@pytest.mark.asyncio
async def test_coqui_tts_dependency_missing_reports_health_and_error() -> None:
    original_import = builtins.__import__

    def _fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name in {"TTS", "TTS.api"}:
            raise ImportError("coqui TTS missing")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_fake_import):
        tts = CoquiTTS(model="tts_models/en/ljspeech/tacotron2-DDC")

    assert await tts.health_check() is False
    with pytest.raises(
        RuntimeError,
        match=r"coqui-tts synthesize unavailable \[dependency_missing\]",
    ):
        await tts.synthesize("hello")
