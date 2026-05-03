"""edge-tts TTS 包裝（可注入 runtime/factory，便於測試）。"""

from __future__ import annotations

from collections.abc import AsyncIterator, Callable
from typing import Any

from mochi.config.defaults import DEFAULT_TTS_VOICE
from mochi.voice.base import BaseTTS, VoiceInfo


class EdgeTTS(BaseTTS):
    """edge-tts 家族 TTS 最小封裝。"""

    def __init__(
        self,
        *,
        voice: str = DEFAULT_TTS_VOICE,
        speed: float = 1.0,
        communicator_factory: Callable[[str, str, str], Any] | None = None,
    ) -> None:
        self.voice = voice
        self.speed = speed
        self._communicator_factory = communicator_factory
        self._dependency_error: Exception | None = None

        if self._communicator_factory is None:
            try:
                import edge_tts  # type: ignore[import-not-found]
            except Exception as exc:
                self._dependency_error = exc
            else:
                self._communicator_factory = lambda text, selected_voice, rate: edge_tts.Communicate(  # noqa: E731
                    text=text,
                    voice=selected_voice,
                    rate=rate,
                )

    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float | None = None,
    ) -> bytes:
        if not text:
            return b""

        if self._dependency_error is not None:
            raise RuntimeError(
                "edge-tts synthesize unavailable [dependency_missing]: "
                f"{self._dependency_error}"
            ) from self._dependency_error
        if self._communicator_factory is None:
            raise RuntimeError("edge-tts synthesize unavailable [factory_missing]")

        selected_voice = voice or self.voice
        selected_speed = self.speed if speed is None else speed
        rate = self._speed_to_rate(selected_speed)

        try:
            communicator = self._communicator_factory(text, selected_voice, rate)
            return await self._collect_audio_chunks(communicator.stream())
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(f"edge-tts synthesize unavailable [synthesize_failed]: {exc}") from exc

    def get_info(self) -> VoiceInfo:
        return VoiceInfo(
            kind="tts",
            family="edge-tts",
            name=self.voice,
            metadata={
                "speed": self.speed,
                "dependency_ready": self._dependency_error is None,
            },
        )

    async def health_check(self) -> bool:
        return self._dependency_error is None

    @staticmethod
    def _speed_to_rate(speed: float) -> str:
        percent = int(round((speed - 1.0) * 100.0))
        return f"{percent:+d}%"

    @staticmethod
    async def _collect_audio_chunks(stream: AsyncIterator[Any]) -> bytes:
        chunks = bytearray()
        async for item in stream:
            if isinstance(item, (bytes, bytearray)):
                chunks.extend(item)
                continue
            if isinstance(item, dict) and item.get("type") == "audio":
                data = item.get("data", b"")
                if isinstance(data, (bytes, bytearray)):
                    chunks.extend(data)
        return bytes(chunks)
