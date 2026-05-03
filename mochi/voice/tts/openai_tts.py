"""OpenAI-compatible 語音合成 TTS 包裝（bounded Phase 4）。"""

from __future__ import annotations

import inspect
import io
import wave
from collections.abc import Callable
from struct import iter_unpack, pack
from typing import Any

from mochi.voice.base import BaseTTS, VoiceInfo

OpenAISpeechCallable = Callable[[str, dict[str, Any], dict[str, str]], Any]


class OpenAITTS(BaseTTS):
    """OpenAI-compatible `/audio/speech` 最小封裝。"""

    def __init__(
        self,
        *,
        model: str = "gpt-4o-mini-tts",
        voice: str = "alloy",
        speed: float = 1.0,
        base_url: str | None = None,
        api_key: str = "",
        response_format: str = "pcm",
        timeout: float = 60.0,
        client: Any | None = None,
        client_factory: Callable[..., Any] | None = None,
        speech_create_callable: OpenAISpeechCallable | None = None,
    ) -> None:
        self.model = model
        self.voice = voice
        self.speed = speed
        self.base_url = str(base_url or "").strip().rstrip("/")
        self.api_key = api_key
        self.response_format = response_format
        self.timeout = timeout
        self._client = client
        self._client_factory = client_factory
        self._speech_create_callable = speech_create_callable
        self._dependency_error: Exception | None = None
        self._runtime_init_error: Exception | None = None

        if (
            self._client is None
            and self._client_factory is None
            and self._speech_create_callable is None
        ):
            try:
                import httpx  # type: ignore[import-not-found]
            except Exception as exc:
                self._dependency_error = exc
            else:
                self._client_factory = lambda *, timeout: httpx.AsyncClient(timeout=timeout)

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float | None = None,
    ) -> bytes:
        if not text:
            return b""

        selected_voice = voice or self.voice
        selected_speed = self.speed if speed is None else speed

        try:
            payload = {
                "model": self.model,
                "input": text,
                "voice": selected_voice,
                "speed": selected_speed,
                "response_format": self.response_format,
            }
            headers = self._build_headers()
            raw = await self._create_speech(payload, headers)
            return self._normalize_to_pcm16_mono(raw, self.response_format)
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"openai-tts synthesize unavailable [synthesize_failed]: {exc}") from exc

    def get_info(self) -> VoiceInfo:
        client_ready = self._speech_create_callable is not None or (
            self._dependency_error is None
            and self._runtime_init_error is None
            and (
                self._client is not None
                or (self._client_factory is not None and bool(self.base_url))
            )
        )
        return VoiceInfo(
            kind="tts",
            family="openai-tts",
            name=self.model,
            metadata={
                "voice": self.voice,
                "speed": self.speed,
                "base_url": self.base_url,
                "response_format": self.response_format,
                "loaded": self._client is not None,
                "dependency_ready": self._dependency_error is None,
                "client_ready": client_ready,
                "api_key_configured": bool(self.api_key),
            },
        )

    async def health_check(self) -> bool:
        if self._speech_create_callable is not None:
            return self._dependency_error is None and self._runtime_init_error is None

        client = await self._get_client_for_health_check()
        if client is None:
            return False

        checker = getattr(client, "health_check", None)
        if callable(checker):
            try:
                result = checker()
                if inspect.isawaitable(result):
                    result = await result
                return bool(result)
            except Exception:
                return False

        getter = getattr(client, "get", None)
        if not callable(getter):
            return True

        for endpoint in self._health_check_urls():
            try:
                response = _call_with_supported_kwargs(
                    getter,
                    endpoint,
                    headers=self._build_headers(),
                    timeout=5.0,
                )
                if inspect.isawaitable(response):
                    response = await response
                status_code = getattr(response, "status_code", None)
                if isinstance(status_code, int) and 200 <= status_code < 500:
                    return True
            except Exception:
                continue
        return False

    async def close(self) -> None:
        client = self._client
        self._client = None
        if client is None:
            return
        closer = getattr(client, "aclose", None)
        if not callable(closer):
            closer = getattr(client, "close", None)
        if callable(closer):
            result = closer()
            if inspect.isawaitable(result):
                await result

    async def _create_speech(self, payload: dict[str, Any], headers: dict[str, str]) -> bytes:
        if self._speech_create_callable is not None:
            response = self._speech_create_callable(self._speech_url(), payload, headers)
            if inspect.isawaitable(response):
                response = await response
            if not isinstance(response, (bytes, bytearray, dict)):
                _raise_for_status(response)
            return await _extract_audio_bytes(response)

        client = await self._ensure_client()
        response = _call_with_supported_kwargs(
            client.post,
            self._speech_url(),
            json=payload,
            headers=headers,
        )
        if inspect.isawaitable(response):
            response = await response
        _raise_for_status(response)
        return await _extract_audio_bytes(response)

    async def _ensure_client(self) -> Any:
        client = await self._get_client_for_health_check()
        if client is not None:
            return client

        if self._dependency_error is not None:
            raise RuntimeError(
                "openai-tts synthesize unavailable [dependency_missing]: "
                f"{self._dependency_error}"
            ) from self._dependency_error
        if self._runtime_init_error is not None:
            raise RuntimeError(
                "openai-tts synthesize unavailable [runtime_init_failed]: "
                f"{self._runtime_init_error}"
            ) from self._runtime_init_error
        raise RuntimeError("openai-tts synthesize unavailable [factory_missing]")

    async def _get_client_for_health_check(self) -> Any | None:
        if self._client is not None:
            return self._client
        if self._dependency_error is not None or self._runtime_init_error is not None:
            return None
        if self._client_factory is None:
            return None
        if not self.base_url:
            self._runtime_init_error = RuntimeError("base_url is required")
            return None

        try:
            client = self._client_factory(timeout=self.timeout)
            if inspect.isawaitable(client):
                client = await client
        except Exception as exc:
            self._runtime_init_error = exc
            return None
        self._client = client
        return self._client

    def _build_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _speech_url(self) -> str:
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/audio/speech"
        if self.base_url:
            return f"{self.base_url}/v1/audio/speech"
        return "/v1/audio/speech"

    def _health_check_urls(self) -> tuple[str, ...]:
        if self.base_url.endswith("/v1"):
            return (f"{self.base_url}/models",)
        return (
            f"{self.base_url}/models",
            f"{self.base_url}/v1/models",
        )

    def _normalize_to_pcm16_mono(self, audio: bytes, response_format: str) -> bytes:
        if not audio:
            return b""
        fmt = response_format.lower()
        if fmt in {"pcm", "wav"}:
            if _looks_like_wav(audio):
                return _wav_to_pcm16_mono("openai-tts", audio)
            if len(audio) % 2 != 0:
                raise RuntimeError(
                    "openai-tts synthesize unavailable [invalid_audio_format]: "
                    "Expected PCM16 bytes length to be even."
                )
            return audio
        raise RuntimeError(
            "openai-tts synthesize unavailable [invalid_audio_format]: "
            f"Unsupported response_format='{response_format}' for PCM16 normalization."
        )


def _call_with_supported_kwargs(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(*args, **kwargs)

    accepts_var_keyword = any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in signature.parameters.values()
    )
    if accepts_var_keyword:
        return func(*args, **kwargs)

    accepted_kwargs = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return func(*args, **accepted_kwargs)


def _raise_for_status(response: Any) -> None:
    raise_for_status = getattr(response, "raise_for_status", None)
    if callable(raise_for_status):
        raise_for_status()


async def _extract_audio_bytes(response: Any) -> bytes:
    if isinstance(response, (bytes, bytearray)):
        return bytes(response)

    if isinstance(response, dict):
        data = response.get("audio", b"")
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)

    content = getattr(response, "content", None)
    if isinstance(content, (bytes, bytearray)):
        return bytes(content)

    aread = getattr(response, "aread", None)
    if callable(aread):
        data = aread()
        if inspect.isawaitable(data):
            data = await data
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)

    read = getattr(response, "read", None)
    if callable(read):
        data = read()
        if inspect.isawaitable(data):
            data = await data
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)

    return b""


def _looks_like_wav(data: bytes) -> bool:
    return len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def _wav_to_pcm16_mono(family: str, wav_data: bytes) -> bytes:
    try:
        with wave.open(io.BytesIO(wav_data), "rb") as wav_file:
            channels = wav_file.getnchannels()
            sample_width = wav_file.getsampwidth()
            frames = wav_file.readframes(wav_file.getnframes())
    except Exception as exc:
        raise RuntimeError(
            f"{family} synthesize unavailable [invalid_audio_format]: {exc}"
        ) from exc

    if sample_width != 2:
        raise RuntimeError(
            f"{family} synthesize unavailable [invalid_audio_format]: "
            f"Expected 16-bit WAV, got sample width={sample_width}."
        )
    if channels <= 0:
        raise RuntimeError(f"{family} synthesize unavailable [invalid_audio_format]: invalid channel count.")
    if channels == 1:
        return frames
    return _downmix_interleaved_pcm16(frames, channels)


def _downmix_interleaved_pcm16(frames: bytes, channels: int) -> bytes:
    sample_count = len(frames) // 2
    if sample_count == 0:
        return b""

    trimmed = frames[: sample_count * 2]
    samples = [sample for (sample,) in iter_unpack("<h", trimmed)]
    frame_count = len(samples) // channels
    if frame_count == 0:
        return b""

    out = bytearray()
    for frame_idx in range(frame_count):
        offset = frame_idx * channels
        mixed = int(sum(samples[offset : offset + channels]) / channels)
        mixed = max(-32768, min(32767, mixed))
        out.extend(pack("<h", mixed))
    return bytes(out)
