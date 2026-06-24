"""whisper-cpp STT backend 測試。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from mochi.voice.stt.whisper_cpp import WhisperCppSTT


@dataclass
class _Segment:
    text: str


class _FakeWhisperCppRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def transcribe(self, audio: bytes, language: str | None = None) -> list[_Segment]:
        self.calls.append({"audio": audio, "language": language})
        return [_Segment("你好"), _Segment("Mochi")]


@pytest.mark.asyncio
async def test_whisper_cpp_transcribe_with_injected_runtime() -> None:
    """可用注入 runtime 完成最小轉寫。"""
    runtime = _FakeWhisperCppRuntime()
    stt = WhisperCppSTT(runtime=runtime, model="base", language="auto")

    text = await stt.transcribe(b"\x01\x02\x03\x04", sample_rate=16000, language="zh")

    assert text == "你好 Mochi"
    assert runtime.calls[0]["audio"] == b"\x01\x02\x03\x04"
    assert runtime.calls[0]["language"] == "zh"
    assert stt.get_info().family == "whisper-cpp"
    assert stt.get_info().metadata["dependency_ready"] is True


@pytest.mark.asyncio
async def test_whisper_cpp_transcribe_uses_default_language_when_request_is_auto() -> None:
    """language=None/auto 時應回退到後端預設語言。"""
    runtime = _FakeWhisperCppRuntime()
    stt = WhisperCppSTT(runtime=runtime, language="ja")

    await stt.transcribe(b"\x01\x02", sample_rate=16000, language=None)

    assert runtime.calls[0]["language"] == "ja"


@pytest.mark.asyncio
async def test_whisper_cpp_model_factory_failure_semantics(tmp_path: Path) -> None:
    """runtime 初始化失敗時應回報一致語義。"""
    seen: dict[str, Any] = {}

    def _broken_factory(**kwargs: Any) -> Any:
        seen["kwargs"] = kwargs
        raise RuntimeError("load failed")

    stt = WhisperCppSTT(
        model="ggml-base.bin",
        model_path=tmp_path / "ggml-base.bin",
        n_threads=4,
        model_factory=_broken_factory,
    )

    with pytest.raises(RuntimeError, match=r"runtime_init_failed"):
        await stt.transcribe(b"\x01\x02", sample_rate=16000)

    assert seen["kwargs"]["model_path"] == str(tmp_path / "ggml-base.bin")
    assert seen["kwargs"]["n_threads"] == 4


@pytest.mark.asyncio
async def test_whisper_cpp_dependency_missing_health_and_error_semantics() -> None:
    """缺依賴時 health_check 應為 False 且 transcribe 回報 dependency_missing。"""
    stt = WhisperCppSTT(runtime=object())
    stt._runtime = None  # noqa: SLF001
    stt._model_factory = None  # noqa: SLF001
    stt._dependency_error = RuntimeError("missing whisper_cpp")  # noqa: SLF001

    assert await stt.health_check() is False
    with pytest.raises(RuntimeError, match=r"whisper-cpp transcribe unavailable \[dependency_missing\]"):
        await stt.transcribe(b"\x01\x02", sample_rate=16000)
