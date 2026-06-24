"""openai-api STT backend 測試。"""

from __future__ import annotations

import io
import wave
from typing import Any

import pytest

from mochi.voice.stt.openai_api import OpenAIApiSTT


class _FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, Any]:
        return self._payload

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


class _FakeOpenAIClient:
    def __init__(self) -> None:
        self.post_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []
        self.closed = False

    async def post(
        self,
        url: str,
        *,
        data: dict[str, str],
        files: dict[str, tuple[str, bytes, str]],
        headers: dict[str, str],
    ) -> _FakeResponse:
        self.post_calls.append(
            {"url": url, "data": data, "files": files, "headers": headers}
        )
        return _FakeResponse({"text": "你好 Mochi"})

    async def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        timeout: float | None = None,
    ) -> _FakeResponse:
        self.get_calls.append({"url": url, "headers": headers, "timeout": timeout})
        if url.endswith("/v1/models"):
            return _FakeResponse({"data": []}, status_code=200)
        return _FakeResponse({"error": "missing"}, status_code=404)

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_openai_api_transcribe_with_injected_client() -> None:
    """可用 injected client 完成 bounded `/audio/transcriptions` 轉寫。"""
    client = _FakeOpenAIClient()
    stt = OpenAIApiSTT(
        client=client,
        model="whisper-1",
        base_url="http://api.example.com/v1",
        api_key="sk-test",
        language="auto",
    )

    text = await stt.transcribe(b"\x01\x00\x02\x00", sample_rate=16000, language="zh")

    assert text == "你好 Mochi"
    assert stt.get_info().family == "openai-api"
    assert stt.get_info().metadata["dependency_ready"] is True
    assert stt.get_info().metadata["client_ready"] is True
    assert client.post_calls[0]["url"] == "http://api.example.com/v1/audio/transcriptions"
    assert client.post_calls[0]["data"] == {"model": "whisper-1", "language": "zh"}
    assert client.post_calls[0]["headers"] == {"Authorization": "Bearer sk-test"}

    filename, wav_bytes, content_type = client.post_calls[0]["files"]["file"]
    assert filename == "audio.wav"
    assert content_type == "audio/wav"
    with wave.open(io.BytesIO(wav_bytes), "rb") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getsampwidth() == 2
        assert wav_file.getframerate() == 16000
        assert wav_file.readframes(2) == b"\x01\x00\x02\x00"

    await stt.close()
    assert client.closed is True


@pytest.mark.asyncio
async def test_openai_api_health_check_tries_models_endpoints() -> None:
    """health_check 應在有明確 HTTP 回應時視為 healthy。"""
    client = _FakeOpenAIClient()
    stt = OpenAIApiSTT(
        client=client,
        model="whisper-1",
        base_url="http://api.example.com",
        api_key="sk-test",
    )

    assert await stt.health_check() is True
    assert [call["url"] for call in client.get_calls] == [
        "http://api.example.com/models",
    ]


class _FakeOpenAIClientModels404(_FakeOpenAIClient):
    async def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        timeout: float | None = None,
    ) -> _FakeResponse:
        self.get_calls.append({"url": url, "headers": headers, "timeout": timeout})
        return _FakeResponse({"error": "not found"}, status_code=404)


class _FakeOpenAIClientConnectFails(_FakeOpenAIClient):
    async def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        timeout: float | None = None,
    ) -> _FakeResponse:
        self.get_calls.append({"url": url, "headers": headers, "timeout": timeout})
        raise RuntimeError("connection refused")


class _FakeOpenAIClientServerError(_FakeOpenAIClient):
    async def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        timeout: float | None = None,
    ) -> _FakeResponse:
        self.get_calls.append({"url": url, "headers": headers, "timeout": timeout})
        return _FakeResponse({"error": "server error"}, status_code=502)


@pytest.mark.asyncio
async def test_openai_api_health_check_does_not_require_models_200() -> None:
    """`/models` 不存在時，只要有明確 HTTP 回應仍不應誤判 unhealthy。"""
    client = _FakeOpenAIClientModels404()
    stt = OpenAIApiSTT(
        client=client,
        model="whisper-1",
        base_url="http://api.example.com/v1",
    )

    assert await stt.health_check() is True
    assert [call["url"] for call in client.get_calls] == [
        "http://api.example.com/v1/models",
    ]


@pytest.mark.asyncio
async def test_openai_api_health_check_returns_false_on_connection_failure() -> None:
    """明顯的連線失敗仍應視為 unhealthy。"""
    client = _FakeOpenAIClientConnectFails()
    stt = OpenAIApiSTT(
        client=client,
        model="whisper-1",
        base_url="http://api.example.com/v1",
    )

    assert await stt.health_check() is False


@pytest.mark.asyncio
async def test_openai_api_health_check_returns_false_on_server_error_response() -> None:
    """5xx 代表 server-side failure，不應直接視為 healthy。"""
    client = _FakeOpenAIClientServerError()
    stt = OpenAIApiSTT(
        client=client,
        model="whisper-1",
        base_url="http://api.example.com/v1",
    )

    assert await stt.health_check() is False


@pytest.mark.asyncio
async def test_openai_api_runtime_init_failure_semantics() -> None:
    """client 建立失敗時，應回報 runtime_init_failed。"""

    def _broken_factory(*, timeout: float) -> Any:  # noqa: ARG001
        raise RuntimeError("init failed")

    stt = OpenAIApiSTT(
        model="whisper-1",
        base_url="http://api.example.com/v1",
        client_factory=_broken_factory,
    )

    assert await stt.health_check() is False
    with pytest.raises(RuntimeError, match=r"openai-api transcribe unavailable \[runtime_init_failed\]"):
        await stt.transcribe(b"\x01\x02", sample_rate=16000)


@pytest.mark.asyncio
async def test_openai_api_dependency_missing_health_and_error_semantics() -> None:
    """缺依賴時 health_check 應為 False 且 transcribe 回報 dependency_missing。"""
    stt = OpenAIApiSTT(client=object(), base_url="http://api.example.com/v1")
    stt._client = None  # noqa: SLF001
    stt._client_factory = None  # noqa: SLF001
    stt._dependency_error = RuntimeError("missing httpx")  # noqa: SLF001

    assert await stt.health_check() is False
    with pytest.raises(RuntimeError, match=r"openai-api transcribe unavailable \[dependency_missing\]"):
        await stt.transcribe(b"\x01\x02", sample_rate=16000)
