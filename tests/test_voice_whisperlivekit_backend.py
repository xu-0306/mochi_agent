"""whisperlivekit STT backend 測試。"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from mochi.voice.stt.whisperlivekit import WhisperLiveKitSTT
from mochi.voice.stt.whisperlivekit_runtime import (
    WhisperLiveKitRuntimeOptions,
    WhisperLiveKitService,
)


class _FakeWLKRuntime:
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
        return {"text": "WLK text"}


class _FakeWLKPreviewProcessor:
    def __init__(self, name: str = "preview", buffer_text: str = "world") -> None:
        self.name = name
        self.buffer_text = buffer_text
        self.feed_calls: list[bytes] = []
        self.closed = False
        self._queue: asyncio.Queue[object] = asyncio.Queue()
        self._stop = object()

    async def create_tasks(self):  # type: ignore[no-untyped-def]
        async def _results():  # type: ignore[no-untyped-def]
            while True:
                item = await self._queue.get()
                if item is self._stop:
                    return
                yield item

        return _results()

    async def process_audio(self, audio: bytes) -> None:
        self.feed_calls.append(audio)
        if len(self.feed_calls) == 1:
            await self._queue.put(
                {"lines": [{"text": self.name}], "buffer_transcription": self.buffer_text[:3]}
            )
            return
        await self._queue.put(
            {"lines": [{"text": self.name}], "buffer_transcription": self.buffer_text}
        )

    async def close(self) -> None:
        self.closed = True
        await self._queue.put(self._stop)


class _SequencedWLKPreviewProcessor:
    def __init__(self, *, results: list[object | None]) -> None:
        self.results = list(results)
        self.feed_calls: list[bytes] = []
        self.closed = False
        self._queue: asyncio.Queue[object] = asyncio.Queue()
        self._stop = object()

    async def create_tasks(self):  # type: ignore[no-untyped-def]
        async def _results():  # type: ignore[no-untyped-def]
            while True:
                item = await self._queue.get()
                if item is self._stop:
                    return
                yield item

        return _results()

    async def process_audio(self, audio: bytes) -> None:
        self.feed_calls.append(audio)
        if not self.results:
            return
        item = self.results.pop(0)
        if item is not None:
            await self._queue.put(item)

    async def close(self) -> None:
        self.closed = True
        await self._queue.put(self._stop)


def _watchdog_options() -> WhisperLiveKitRuntimeOptions:
    return WhisperLiveKitRuntimeOptions(
        model="base",
        language="en",
        watchdog_enabled=True,
        watchdog_poll_interval=0.01,
        watchdog_no_result_timeout=0.05,
        watchdog_stall_timeout=0.05,
        watchdog_audio_idle_window=0.3,
        watchdog_reset_cooldown=0.02,
    )


@pytest.mark.asyncio
async def test_whisperlivekit_transcribe_with_injected_runtime() -> None:
    runtime = _FakeWLKRuntime()
    stt = WhisperLiveKitSTT(runtime=runtime, model="base", language="en")

    text = await stt.transcribe(b"\x01\x00\x02\x00", sample_rate=16000)

    assert text == "WLK text"
    assert runtime.calls == [
        {"audio": b"\x01\x00\x02\x00", "sample_rate": 16000, "language": "en"}
    ]
    assert stt.get_info().family == "whisperlivekit"


@pytest.mark.asyncio
async def test_whisperlivekit_runtime_factory_failure_semantics() -> None:
    def _broken_factory(**kwargs: Any) -> Any:
        raise RuntimeError("init failed")

    stt = WhisperLiveKitSTT(runtime_factory=_broken_factory)

    with pytest.raises(RuntimeError, match=r"runtime_init_failed"):
        await stt.transcribe(b"\x01\x00\x02\x00", sample_rate=16000)


@pytest.mark.asyncio
async def test_whisperlivekit_dependency_missing_health_and_error_semantics() -> None:
    service = WhisperLiveKitService(
        options=WhisperLiveKitRuntimeOptions(),
        runtime=object(),
    )
    service._runtime = None  # noqa: SLF001
    service._runtime_factory = None  # noqa: SLF001
    service._dependency_error = RuntimeError("missing whisperlivekit")  # noqa: SLF001
    stt = WhisperLiveKitSTT(service=service)

    assert await stt.health_check() is False
    with pytest.raises(RuntimeError, match=r"whisperlivekit transcribe unavailable \[dependency_missing\]"):
        await stt.transcribe(b"\x01\x00\x02\x00", sample_rate=16000)


@pytest.mark.asyncio
async def test_whisperlivekit_preview_session_streams_local_agreement_updates() -> None:
    runtime = _FakeWLKRuntime()
    processor = _FakeWLKPreviewProcessor(name="hello", buffer_text="world")
    processor_factory_calls: list[dict[str, Any]] = []

    def _processor_factory(**kwargs: Any) -> _FakeWLKPreviewProcessor:
        processor_factory_calls.append(kwargs)
        return processor

    stt = WhisperLiveKitSTT(
        runtime=runtime,
        model="base",
        language="en",
        audio_processor_factory=_processor_factory,
    )

    preview_session = await stt.create_transcription_preview_session(sample_rate=16000)

    assert preview_session is not None
    assert stt.supports_transcription_preview() is True
    assert processor_factory_calls == [
        {
            "transcription_engine": runtime,
            "engine": runtime,
            "runtime": runtime,
            "sample_rate": 16000,
            "language": "en",
            "enable_transcription": True,
            "transcription": True,
            "enable_diarization": False,
            "diarization": False,
        }
    ]

    assert await preview_session.push_audio(b"chunk-1") == ["hello wor"]
    assert await preview_session.push_audio(b"chunk-2") == ["hello world"]
    assert await preview_session.drain_texts() == []

    await preview_session.close()
    assert processor.feed_calls == [b"chunk-1", b"chunk-2"]
    assert processor.closed is True


@pytest.mark.asyncio
async def test_whisperlivekit_preview_session_supports_formal_event_contract() -> None:
    runtime = _FakeWLKRuntime()
    processor = _FakeWLKPreviewProcessor(name="hello", buffer_text="world")
    stt = WhisperLiveKitSTT(
        runtime=runtime,
        model="base",
        language="en",
        audio_processor_factory=lambda **kwargs: processor,  # noqa: ARG005
    )

    preview_session = await stt.create_transcription_preview_session(sample_rate=16000)

    assert preview_session is not None

    first_batch = await preview_session.append_audio(b"chunk-1")
    assert [(event.text, event.is_final) for event in first_batch] == [
        ("hello", True),
        ("wor", False),
    ]

    second_batch = await preview_session.append_audio(b"chunk-2")
    assert [(event.text, event.is_final) for event in second_batch] == [
        ("world", False),
    ]

    assert await preview_session.drain_events() == []
    await preview_session.close()


@pytest.mark.asyncio
async def test_whisperlivekit_preview_session_reset_rebuilds_runtime_and_processor() -> None:
    runtime_instances: list[_FakeWLKRuntime] = []
    processor_instances: list[_FakeWLKPreviewProcessor] = []

    def _runtime_factory(**kwargs: Any) -> _FakeWLKRuntime:
        runtime = _FakeWLKRuntime()
        runtime.calls.append({"factory_kwargs": kwargs})
        runtime_instances.append(runtime)
        return runtime

    def _processor_factory(**kwargs: Any) -> _FakeWLKPreviewProcessor:
        index = len(processor_instances) + 1
        processor = _FakeWLKPreviewProcessor(
            name=f"runtime-{index}",
            buffer_text=f"chunk-{index}",
        )
        processor_instances.append(processor)
        return processor

    stt = WhisperLiveKitSTT(
        model="base",
        language="en",
        runtime_factory=_runtime_factory,
        audio_processor_factory=_processor_factory,
    )

    preview_session = await stt.create_transcription_preview_session(sample_rate=16000)

    assert preview_session is not None
    assert await preview_session.push_audio(b"before-reset") == ["runtime-1 chu"]

    await preview_session.reset(rebuild_runtime=True)

    assert len(runtime_instances) == 2
    assert len(processor_instances) == 2
    assert processor_instances[0].closed is True
    assert await preview_session.push_audio(b"after-reset") == ["runtime-2 chu"]
    assert preview_session.get_state().processor_builds == 2

    await preview_session.close()
    assert processor_instances[1].closed is True


@pytest.mark.asyncio
async def test_whisperlivekit_watchdog_rebuilds_runtime_after_no_result() -> None:
    runtime_instances: list[_FakeWLKRuntime] = []
    processor_instances: list[_SequencedWLKPreviewProcessor] = []

    def _runtime_factory(**kwargs: Any) -> _FakeWLKRuntime:
        runtime = _FakeWLKRuntime()
        runtime.calls.append({"factory_kwargs": kwargs})
        runtime_instances.append(runtime)
        return runtime

    def _processor_factory(**kwargs: Any) -> _SequencedWLKPreviewProcessor:
        del kwargs
        results = [None] if not processor_instances else [{"lines": [{"text": "recovered"}]}]
        processor = _SequencedWLKPreviewProcessor(results=results)
        processor_instances.append(processor)
        return processor

    service = WhisperLiveKitService(
        options=_watchdog_options(),
        runtime_factory=_runtime_factory,
        audio_processor_factory=_processor_factory,
    )
    stt = WhisperLiveKitSTT(service=service)

    preview_session = await stt.create_transcription_preview_session(sample_rate=16000)

    assert preview_session is not None
    assert await preview_session.push_audio(b"chunk-1") == []

    await asyncio.sleep(0.12)

    state = preview_session.get_state()
    assert len(runtime_instances) == 2
    assert len(processor_instances) == 2
    assert processor_instances[0].closed is True
    assert state.processor_builds == 2
    assert state.watchdog_resets == 1
    assert state.watchdog_runtime_rebuilds == 1
    assert state.watchdog_last_reason == "no_result"
    assert await preview_session.push_audio(b"chunk-2") == ["recovered"]

    await preview_session.close()


@pytest.mark.asyncio
async def test_whisperlivekit_watchdog_resets_processor_after_stall() -> None:
    runtime_instances: list[_FakeWLKRuntime] = []
    processor_instances: list[_SequencedWLKPreviewProcessor] = []

    def _runtime_factory(**kwargs: Any) -> _FakeWLKRuntime:
        runtime = _FakeWLKRuntime()
        runtime.calls.append({"factory_kwargs": kwargs})
        runtime_instances.append(runtime)
        return runtime

    def _processor_factory(**kwargs: Any) -> _SequencedWLKPreviewProcessor:
        del kwargs
        results = (
            [{"lines": [{"text": "ready"}]}, None]
            if not processor_instances
            else [{"lines": [{"text": "recovered-after-stall"}]}]
        )
        processor = _SequencedWLKPreviewProcessor(results=results)
        processor_instances.append(processor)
        return processor

    service = WhisperLiveKitService(
        options=_watchdog_options(),
        runtime_factory=_runtime_factory,
        audio_processor_factory=_processor_factory,
    )
    stt = WhisperLiveKitSTT(service=service)

    preview_session = await stt.create_transcription_preview_session(sample_rate=16000)

    assert preview_session is not None
    assert await preview_session.push_audio(b"chunk-1") == ["ready"]
    assert await preview_session.push_audio(b"chunk-2") == []

    await asyncio.sleep(0.12)

    state = preview_session.get_state()
    assert len(runtime_instances) == 1
    assert len(processor_instances) == 2
    assert processor_instances[0].closed is True
    assert state.processor_builds == 2
    assert state.watchdog_resets == 1
    assert state.watchdog_runtime_rebuilds == 0
    assert state.watchdog_last_reason == "stall"
    assert await preview_session.push_audio(b"chunk-3") == ["recovered-after-stall"]

    await preview_session.close()


@pytest.mark.asyncio
async def test_whisperlivekit_watchdog_stops_after_close() -> None:
    runtime_instances: list[_FakeWLKRuntime] = []
    processor_instances: list[_SequencedWLKPreviewProcessor] = []

    def _runtime_factory(**kwargs: Any) -> _FakeWLKRuntime:
        runtime = _FakeWLKRuntime()
        runtime.calls.append({"factory_kwargs": kwargs})
        runtime_instances.append(runtime)
        return runtime

    def _processor_factory(**kwargs: Any) -> _SequencedWLKPreviewProcessor:
        del kwargs
        processor = _SequencedWLKPreviewProcessor(results=[None])
        processor_instances.append(processor)
        return processor

    service = WhisperLiveKitService(
        options=_watchdog_options(),
        runtime_factory=_runtime_factory,
        audio_processor_factory=_processor_factory,
    )
    stt = WhisperLiveKitSTT(service=service)

    preview_session = await stt.create_transcription_preview_session(sample_rate=16000)

    assert preview_session is not None
    assert await preview_session.push_audio(b"chunk-1") == []

    await preview_session.close()
    await asyncio.sleep(0.12)

    state = preview_session.get_state()
    assert len(runtime_instances) == 1
    assert len(processor_instances) == 1
    assert processor_instances[0].closed is True
    assert state.watchdog_resets == 0
