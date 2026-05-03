"""OpenAI-compatible 音訊轉寫 STT 包裝（bounded Phase 4）。"""

from __future__ import annotations

import inspect
import io
import wave
from collections.abc import Callable, Mapping
from typing import Any

from mochi.voice.base import BaseSTT, VoiceInfo


class OpenAIApiSTT(BaseSTT):
    """OpenAI-compatible `/audio/transcriptions` bounded 封裝。"""

    def __init__(
        self,
        *,
        model: str = "whisper-1",
        base_url: str | None = None,
        api_key: str = "",
        language: str = "auto",
        timeout: float = 60.0,
        client: Any | None = None,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.model = model
        self.base_url = str(base_url or "").strip().rstrip("/")
        self.api_key = api_key
        self.language = language
        self.timeout = timeout
        self._client = client
        self._client_factory = client_factory
        self._dependency_error: Exception | None = None
        self._runtime_init_error: Exception | None = None

        if self._client is None and self._client_factory is None:
            try:
                import httpx  # type: ignore[import-not-found]
            except Exception as exc:
                self._dependency_error = exc
            else:
                self._client_factory = lambda *, timeout: httpx.AsyncClient(timeout=timeout)

    async def transcribe(
        self,
        audio: bytes,
        *,
        sample_rate: int,
        language: str | None = None,
    ) -> str:
        if not audio:
            return ""

        try:
            client = await self._ensure_client()
            response = _call_with_supported_kwargs(
                client.post,
                self._transcriptions_url(),
                data=self._build_form_data(language),
                files={
                    "file": (
                        "audio.wav",
                        self._pcm16_to_wav(audio, sample_rate),
                        "audio/wav",
                    )
                },
                headers=self._build_headers(),
            )
            if inspect.isawaitable(response):
                response = await response
            _raise_for_status(response)
            return self._extract_text(_json_payload(response))
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"openai-api transcribe unavailable [transcribe_failed]: {exc}"
            ) from exc

    def get_info(self) -> VoiceInfo:
        client_ready = self._client is not None or (
            self._dependency_error is None
            and self._runtime_init_error is None
            and self._client_factory is not None
            and bool(self.base_url)
        )
        return VoiceInfo(
            kind="stt",
            family="openai-api",
            name=self.model,
            metadata={
                "language": self.language,
                "base_url": self.base_url,
                "loaded": self._client is not None,
                "dependency_ready": self._dependency_error is None,
                "client_ready": client_ready,
                "api_key_configured": bool(self.api_key),
            },
        )

    async def health_check(self) -> bool:
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
        if callable(closer):
            result = closer()
            if inspect.isawaitable(result):
                await result

    async def _ensure_client(self) -> Any:
        client = await self._get_client_for_health_check()
        if client is not None:
            return client

        if self._dependency_error is not None:
            raise RuntimeError(
                "openai-api transcribe unavailable [dependency_missing]: "
                f"{self._dependency_error}"
            ) from self._dependency_error
        if self._runtime_init_error is not None:
            raise RuntimeError(
                "openai-api transcribe unavailable [runtime_init_failed]: "
                f"{self._runtime_init_error}"
            ) from self._runtime_init_error
        raise RuntimeError("openai-api transcribe unavailable [factory_missing]")

    async def _get_client_for_health_check(self) -> Any | None:
        if self._client is not None:
            return self._client
        if self._dependency_error is not None or self._runtime_init_error is not None:
            return None
        if not self.base_url:
            self._runtime_init_error = RuntimeError("base_url is required")
            return None
        if self._client_factory is None:
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

    def _build_form_data(self, language: str | None) -> dict[str, str]:
        selected_language = language if language not in (None, "auto") else self.language
        payload = {"model": self.model}
        if selected_language not in (None, "auto", ""):
            payload["language"] = selected_language
        return payload

    def _transcriptions_url(self) -> str:
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/audio/transcriptions"
        return f"{self.base_url}/v1/audio/transcriptions"

    def _health_check_urls(self) -> tuple[str, ...]:
        if self.base_url.endswith("/v1"):
            return (f"{self.base_url}/models",)
        return (
            f"{self.base_url}/models",
            f"{self.base_url}/v1/models",
        )

    @staticmethod
    def _extract_text(payload: Any) -> str:
        if isinstance(payload, str):
            return payload.strip()
        if isinstance(payload, Mapping):
            text = payload.get("text", "")
            return str(text).strip() if text else ""
        return ""

    @staticmethod
    def _pcm16_to_wav(audio: bytes, sample_rate: int) -> bytes:
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio)
        return buffer.getvalue()


def _call_with_supported_kwargs(func: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
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


def _json_payload(response: Any) -> Any:
    json_getter = getattr(response, "json", None)
    if callable(json_getter):
        return json_getter()
    return response
