"""KokoroTTS backend 測試。"""

from __future__ import annotations

import builtins
from unittest.mock import patch

import pytest

from mochi.voice.tts.kokoro_tts import KokoroTTS


class _FakeKokoroRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.closed = False

    def __call__(
        self,
        text: str,
        voice: str,
        speed: float,
        split_pattern: str,
    ) -> list[tuple[str, str, list[float]]]:
        self.calls.append(
            {
                "text": text,
                "voice": voice,
                "speed": speed,
                "split_pattern": split_pattern,
            }
        )
        return [
            ("g1", "p1", [0.0, 0.25]),
            ("g2", "p2", [-0.25, 1.0]),
        ]

    def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_kokoro_tts_with_injected_runtime_collects_chunks_and_outputs_pcm16() -> None:
    runtime = _FakeKokoroRuntime()
    tts = KokoroTTS(
        runtime=runtime,
        voice="af_heart",
        speed=1.1,
        split_pattern=r"\n+",
    )

    audio = await tts.synthesize("hello world")

    assert audio == b"\x00\x00\x00 \x00\xe0\xff\x7f"
    assert runtime.calls[0] == {
        "text": "hello world",
        "voice": "af_heart",
        "speed": 1.1,
        "split_pattern": r"\n+",
    }
    assert await tts.health_check() is True
    assert tts.get_info().family == "kokoro-tts"

    await tts.close()
    assert runtime.closed is True


@pytest.mark.asyncio
async def test_kokoro_tts_pipeline_factory_failure_semantics() -> None:
    seen: dict[str, object] = {}

    def _broken_pipeline_factory(**kwargs: object) -> object:
        seen["kwargs"] = kwargs
        raise RuntimeError("init failed")

    tts = KokoroTTS(
        lang_code="a",
        pipeline_factory=_broken_pipeline_factory,
    )

    with pytest.raises(RuntimeError, match=r"kokoro-tts synthesize unavailable \[runtime_init_failed\]"):
        await tts.synthesize("hello")

    assert seen["kwargs"] == {"lang_code": "a"}
    assert await tts.health_check() is False


@pytest.mark.asyncio
async def test_kokoro_tts_dependency_missing_reports_health_and_error() -> None:
    original_import = builtins.__import__

    def _fake_import(name: str, *args, **kwargs):  # type: ignore[no-untyped-def]
        if name == "kokoro":
            raise ImportError("kokoro missing")
        return original_import(name, *args, **kwargs)

    with patch("builtins.__import__", side_effect=_fake_import):
        tts = KokoroTTS()

    assert await tts.health_check() is False
    with pytest.raises(
        RuntimeError,
        match=r"kokoro-tts synthesize unavailable \[dependency_missing\]",
    ):
        await tts.synthesize("hello")
