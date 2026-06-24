"""openai-whisper STT backend 測試。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mochi.voice.stt.openai_whisper import OpenAIWhisperSTT


class _FakeWhisperRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def transcribe(
        self,
        audio: Any,
        language: str | None = None,
        task: str = "transcribe",
    ) -> dict[str, str]:
        self.calls.append({"audio": audio, "language": language, "task": task})
        return {"text": "你好 Mochi"}


@pytest.mark.asyncio
async def test_openai_whisper_transcribe_with_injected_runtime() -> None:
    """可用 injected runtime 完成 bounded 轉寫。"""
    pytest.importorskip("numpy")
    runtime = _FakeWhisperRuntime()
    stt = OpenAIWhisperSTT(runtime=runtime, model="base", language="auto")

    text = await stt.transcribe(
        b"\x00\x80\x00\x00\xff\x7f",
        sample_rate=16000,
        language="zh",
    )

    assert text == "你好 Mochi"
    assert runtime.calls[0]["language"] == "zh"
    assert runtime.calls[0]["task"] == "transcribe"
    assert runtime.calls[0]["audio"].dtype.name == "float32"
    assert runtime.calls[0]["audio"].tolist() == pytest.approx([-1.0, 0.0, 32767 / 32768])
    assert stt.get_info().family == "openai-whisper"
    assert stt.get_info().metadata["dependency_ready"] is True


@pytest.mark.asyncio
async def test_openai_whisper_uses_default_language_when_request_is_auto() -> None:
    """language=None/auto 時應回退到後端預設語言。"""
    pytest.importorskip("numpy")
    runtime = _FakeWhisperRuntime()
    stt = OpenAIWhisperSTT(runtime=runtime, language="ja")

    await stt.transcribe(b"\x01\x00\x02\x00", sample_rate=16000, language=None)

    assert runtime.calls[0]["language"] == "ja"


@pytest.mark.asyncio
async def test_openai_whisper_model_factory_failure_semantics(tmp_path: Path) -> None:
    """runtime 初始化失敗時應回報一致語義。"""
    seen: dict[str, Any] = {}

    def _broken_factory(model_source: str, **kwargs: Any) -> Any:
        seen["model_source"] = model_source
        seen["kwargs"] = kwargs
        raise RuntimeError("load failed")

    stt = OpenAIWhisperSTT(
        model="small",
        device="cpu",
        model_cache_dir=tmp_path,
        in_memory=True,
        model_factory=_broken_factory,
    )

    with pytest.raises(RuntimeError, match=r"runtime_init_failed"):
        await stt.transcribe(b"\x01\x00\x02\x00", sample_rate=16000)

    assert seen["model_source"] == "small"
    assert seen["kwargs"] == {
        "device": "cpu",
        "download_root": str(tmp_path),
        "in_memory": True,
    }


@pytest.mark.asyncio
async def test_openai_whisper_dependency_missing_health_and_error_semantics() -> None:
    """缺依賴時 health_check 應為 False 且 transcribe 回報 dependency_missing。"""
    stt = OpenAIWhisperSTT(runtime=object())
    stt._runtime = None  # noqa: SLF001
    stt._model_factory = None  # noqa: SLF001
    stt._dependency_error = RuntimeError("missing whisper")  # noqa: SLF001

    assert await stt.health_check() is False
    with pytest.raises(RuntimeError, match=r"openai-whisper transcribe unavailable \[dependency_missing\]"):
        await stt.transcribe(b"\x01\x00\x02\x00", sample_rate=16000)
