"""OpenAITTS backend 測試。"""

from __future__ import annotations

import io
import wave
from typing import Any

import pytest

from mochi.voice.tts.openai_tts import OpenAITTS


def _build_wav_pcm16(*, channels: int, sample_rate: int, samples: list[int]) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"".join(int(sample).to_bytes(2, "little", signed=True) for sample in samples))
    return buffer.getvalue()


class _FakeResponse:
    def __init__(self, content: bytes, status_code: int = 200) -> None:
        self.content = content
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


class _FakeOpenAIClient:
    def __init__(self, speech_content: bytes) -> None:
        self.post_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []
        self.closed = False
        self._speech_content = speech_content

    async def post(
        self,
        url: str,
        *,
        json: dict[str, Any],
        headers: dict[str, str],
    ) -> _FakeResponse:
        self.post_calls.append({"url": url, "json": json, "headers": headers})
        return _FakeResponse(self._speech_content)

    async def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        timeout: float | None = None,
    ) -> _FakeResponse:
        self.get_calls.append({"url": url, "headers": headers, "timeout": timeout})
        if url.endswith("/models"):
            return _FakeResponse(b"{}", status_code=200)
        return _FakeResponse(b"{}", status_code=404)

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_openai_tts_with_injected_client_returns_pcm16_and_sends_expected_payload() -> None:
    client = _FakeOpenAIClient(speech_content=b"\x01\x00\x02\x00")
    tts = OpenAITTS(
        client=client,
        model="gpt-4o-mini-tts",
        voice="alloy",
        speed=1.0,
        base_url="http://api.example.com/v1",
        api_key="sk-test",
        response_format="pcm",
    )

    audio = await tts.synthesize("hello", voice="nova", speed=1.25)

    assert audio == b"\x01\x00\x02\x00"
    assert client.post_calls[0]["url"] == "http://api.example.com/v1/audio/speech"
    assert client.post_calls[0]["json"] == {
        "model": "gpt-4o-mini-tts",
        "input": "hello",
        "voice": "nova",
        "speed": 1.25,
        "response_format": "pcm",
    }
    assert client.post_calls[0]["headers"] == {"Authorization": "Bearer sk-test"}
    assert tts.get_info().family == "openai-tts"
    assert tts.get_info().metadata["dependency_ready"] is True
    assert tts.get_info().metadata["client_ready"] is True

    await tts.close()
    assert client.closed is True


@pytest.mark.asyncio
async def test_openai_tts_normalizes_wav_to_mono_pcm16() -> None:
    stereo_wav = _build_wav_pcm16(
        channels=2,
        sample_rate=24000,
        samples=[1000, -1000, 3000, 1000],
    )
    client = _FakeOpenAIClient(speech_content=stereo_wav)
    tts = OpenAITTS(
        client=client,
        base_url="http://api.example.com/v1",
        response_format="wav",
    )

    audio = await tts.synthesize("hello")

    assert audio == b"\x00\x00\xd0\x07"


@pytest.mark.asyncio
async def test_openai_tts_health_check_uses_models_endpoint() -> None:
    client = _FakeOpenAIClient(speech_content=b"\x01\x00\x02\x00")
    tts = OpenAITTS(
        client=client,
        base_url="http://api.example.com",
        api_key="sk-test",
    )

    assert await tts.health_check() is True
    assert [call["url"] for call in client.get_calls] == [
        "http://api.example.com/models",
    ]


@pytest.mark.asyncio
async def test_openai_tts_runtime_init_failure_semantics() -> None:
    def _broken_factory(*, timeout: float) -> Any:  # noqa: ARG001
        raise RuntimeError("init failed")

    tts = OpenAITTS(
        model="gpt-4o-mini-tts",
        base_url="http://api.example.com/v1",
        client_factory=_broken_factory,
    )

    assert await tts.health_check() is False
    with pytest.raises(RuntimeError, match=r"openai-tts synthesize unavailable \[runtime_init_failed\]"):
        await tts.synthesize("hello")


@pytest.mark.asyncio
async def test_openai_tts_dependency_missing_health_and_error_semantics() -> None:
    tts = OpenAITTS(client=object(), base_url="http://api.example.com/v1")
    tts._client = None  # noqa: SLF001
    tts._client_factory = None  # noqa: SLF001
    tts._dependency_error = RuntimeError("missing httpx")  # noqa: SLF001

    assert await tts.health_check() is False
    with pytest.raises(RuntimeError, match=r"openai-tts synthesize unavailable \[dependency_missing\]"):
        await tts.synthesize("hello")
