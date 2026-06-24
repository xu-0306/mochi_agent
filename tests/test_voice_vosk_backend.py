"""vosk STT backend 測試。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from mochi.voice.stt.vosk import VoskSTT


class _FakeRecognizer:
    def __init__(self) -> None:
        self.accept_calls: list[bytes] = []
        self._accept_count = 0

    def AcceptWaveform(self, data: bytes) -> bool:  # noqa: N802
        self.accept_calls.append(data)
        self._accept_count += 1
        return self._accept_count == 2

    def Result(self) -> str:  # noqa: N802
        return '{"text": "你好"}'

    def FinalResult(self) -> str:  # noqa: N802
        return '{"text": "Mochi"}'


@pytest.mark.asyncio
async def test_vosk_transcribe_with_injected_runtime() -> None:
    """可用注入 runtime + recognizer factory 完成最小轉寫。"""
    runtime = object()
    seen: dict[str, Any] = {}

    def _recognizer_factory(model_runtime: Any, sample_rate: int) -> _FakeRecognizer:
        seen["model_runtime"] = model_runtime
        seen["sample_rate"] = sample_rate
        recognizer = _FakeRecognizer()
        seen["recognizer"] = recognizer
        return recognizer

    stt = VoskSTT(
        runtime=runtime,
        recognizer_factory=_recognizer_factory,
        chunk_size=4,
        model="vosk-model-small-cn",
    )

    text = await stt.transcribe(
        b"\x01\x00\x02\x00\x03\x00\x04\x00",
        sample_rate=16000,
        language="zh",
    )

    assert text == "你好 Mochi"
    assert seen["model_runtime"] is runtime
    assert seen["sample_rate"] == 16000
    assert seen["recognizer"].accept_calls == [
        b"\x01\x00\x02\x00",
        b"\x03\x00\x04\x00",
    ]
    assert stt.get_info().family == "vosk"
    assert stt.get_info().metadata["dependency_ready"] is True
    assert stt.get_info().metadata["recognizer_ready"] is True


@pytest.mark.asyncio
async def test_vosk_model_factory_failure_semantics(tmp_path: Path) -> None:
    """runtime 初始化失敗時應回報一致語義。"""
    seen: dict[str, Any] = {}

    def _broken_model_factory(model_source: str) -> Any:
        seen["model_source"] = model_source
        raise RuntimeError("load failed")

    stt = VoskSTT(
        model=tmp_path / "vosk-model",
        model_factory=_broken_model_factory,
        recognizer_factory=lambda model_runtime, sample_rate: _FakeRecognizer(),  # noqa: ARG005
    )

    with pytest.raises(RuntimeError, match=r"runtime_init_failed"):
        await stt.transcribe(b"\x01\x00\x02\x00", sample_rate=16000)

    assert seen["model_source"] == str(tmp_path / "vosk-model")


@pytest.mark.asyncio
async def test_vosk_dependency_missing_health_and_error_semantics() -> None:
    """缺依賴時 health_check 應為 False 且 transcribe 回報 dependency_missing。"""
    stt = VoskSTT(runtime=object(), recognizer_factory=lambda model_runtime, sample_rate: _FakeRecognizer())  # noqa: ARG005
    stt._runtime = None  # noqa: SLF001
    stt._model_factory = None  # noqa: SLF001
    stt._recognizer_factory = None  # noqa: SLF001
    stt._dependency_error = RuntimeError("missing vosk")  # noqa: SLF001

    assert await stt.health_check() is False
    with pytest.raises(RuntimeError, match=r"vosk transcribe unavailable \[dependency_missing\]"):
        await stt.transcribe(b"\x01\x00\x02\x00", sample_rate=16000)
