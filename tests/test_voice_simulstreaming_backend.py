"""simulstreaming STT backend 測試。"""

from __future__ import annotations

from typing import Any

import pytest

from mochi.voice.stt.simulstreaming import SimulStreamingSTT


class _FakeSimulRuntime:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def process_iter(
        self,
        *,
        audio: bytes,
        sample_rate: int,
        language: str | None = None,
    ) -> tuple[None, str, None]:
        self.calls.append(
            {"audio": audio, "sample_rate": sample_rate, "language": language}
        )
        return None, "Simul text", None


@pytest.mark.asyncio
async def test_simulstreaming_transcribe_with_injected_runtime() -> None:
    runtime = _FakeSimulRuntime()
    stt = SimulStreamingSTT(runtime=runtime, model="base", language="fr")

    text = await stt.transcribe(b"\x01\x00\x02\x00", sample_rate=16000)

    assert text == "Simul text"
    assert runtime.calls == [
        {"audio": b"\x01\x00\x02\x00", "sample_rate": 16000, "language": "fr"}
    ]
    assert stt.get_info().family == "simulstreaming"


@pytest.mark.asyncio
async def test_simulstreaming_runtime_factory_failure_semantics() -> None:
    def _broken_factory(**kwargs: Any) -> Any:
        raise RuntimeError("init failed")

    stt = SimulStreamingSTT(runtime_factory=_broken_factory)

    with pytest.raises(RuntimeError, match=r"runtime_init_failed"):
        await stt.transcribe(b"\x01\x00\x02\x00", sample_rate=16000)


@pytest.mark.asyncio
async def test_simulstreaming_dependency_missing_health_and_error_semantics() -> None:
    stt = SimulStreamingSTT(runtime=object())
    stt._runtime = None  # noqa: SLF001
    stt._runtime_factory = None  # noqa: SLF001
    stt._dependency_error = RuntimeError("missing whisper_streaming")  # noqa: SLF001

    assert await stt.health_check() is False
    with pytest.raises(RuntimeError, match=r"simulstreaming transcribe unavailable \[dependency_missing\]"):
        await stt.transcribe(b"\x01\x00\x02\x00", sample_rate=16000)
