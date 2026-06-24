"""qwen-asr STT backend 測試。"""

from __future__ import annotations

from typing import Any

import pytest

from mochi.voice.stt.qwen_asr import QwenASRSTT


class _FakeQwenRuntime:
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
        return {"text": "你好 Mochi"}


@pytest.mark.asyncio
async def test_qwen_asr_transcribe_with_injected_runtime() -> None:
    runtime = _FakeQwenRuntime()
    stt = QwenASRSTT(runtime=runtime, model="Qwen/Qwen3-ASR-0.6B", language="zh")

    text = await stt.transcribe(b"\x01\x00\x02\x00", sample_rate=16000, language="auto")

    assert text == "你好 Mochi"
    assert runtime.calls == [
        {"audio": b"\x01\x00\x02\x00", "sample_rate": 16000, "language": "zh"}
    ]
    assert stt.get_info().family == "qwen-asr"


@pytest.mark.asyncio
async def test_qwen_asr_model_factory_failure_semantics() -> None:
    def _broken_factory(**kwargs: Any) -> Any:
        raise RuntimeError("load failed")

    stt = QwenASRSTT(model_factory=_broken_factory)

    with pytest.raises(RuntimeError, match=r"runtime_init_failed"):
        await stt.transcribe(b"\x01\x00\x02\x00", sample_rate=16000)


@pytest.mark.asyncio
async def test_qwen_asr_dependency_missing_health_and_error_semantics() -> None:
    stt = QwenASRSTT(runtime=object())
    stt._runtime = None  # noqa: SLF001
    stt._model_factory = None  # noqa: SLF001
    stt._dependency_error = RuntimeError("missing qwen_asr")  # noqa: SLF001

    assert await stt.health_check() is False
    with pytest.raises(RuntimeError, match=r"qwen-asr transcribe unavailable \[dependency_missing\]"):
        await stt.transcribe(b"\x01\x00\x02\x00", sample_rate=16000)
