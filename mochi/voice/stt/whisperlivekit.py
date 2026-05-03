"""WhisperLiveKit STT 包裝。"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import Any

from mochi.config.schema import VoiceConfig
from mochi.voice.base import BaseSTT, VoiceInfo

from .whisperlivekit_runtime import (
    WhisperLiveKitPreviewSession,
    WhisperLiveKitRuntimeOptions,
    WhisperLiveKitService,
)


class WhisperLiveKitSTT(BaseSTT):
    """WhisperLiveKit bounded STT 封裝。"""

    def __init__(
        self,
        *,
        model: str = "base",
        language: str = "auto",
        runtime: Any | None = None,
        runtime_factory: Callable[..., Any] | None = None,
        audio_processor_factory: Callable[..., Any] | None = None,
        service: WhisperLiveKitService | None = None,
        audio_input_contract: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.language = language
        self._audio_input_contract = dict(
            VoiceConfig().voice_input_contract
            if audio_input_contract is None
            else audio_input_contract
        )
        self._service = service or WhisperLiveKitService(
            options=WhisperLiveKitRuntimeOptions(model=model, language=language),
            runtime=runtime,
            runtime_factory=runtime_factory,
            audio_processor_factory=audio_processor_factory,
        )

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
            return await asyncio.to_thread(
                self._service.transcribe,
                audio,
                sample_rate=sample_rate,
                language=language,
            )
        except RuntimeError:
            raise
        except Exception as exc:
            raise RuntimeError(
                f"whisperlivekit transcribe unavailable [transcribe_failed]: {exc}"
            ) from exc

    def get_info(self) -> VoiceInfo:
        return VoiceInfo(
            kind="stt",
            family="whisperlivekit",
            name=self.model,
            metadata={
                "language": self.language,
                "loaded": self._service.runtime_loaded,
                "dependency_ready": self._service.dependency_error is None,
                "supports_preview": self._service.supports_realtime(),
                "pcm_input": self._service.options.pcm_input,
                "audio_input_contract": dict(self._audio_input_contract),
            },
        )

    async def health_check(self) -> bool:
        return self._service.health_check()

    async def close(self) -> None:
        await self._service.close()

    def supports_transcription_preview(self) -> bool:
        """回報是否可建立 WhisperLiveKit 即時 preview session。"""
        return self._service.supports_realtime()

    async def create_transcription_preview_session(
        self,
        *,
        sample_rate: int,
        language: str | None = None,
    ) -> WhisperLiveKitPreviewSession | None:
        """建立可增量餵音訊的 WhisperLiveKit preview session。"""
        return await self._service.create_realtime_session(
            sample_rate=sample_rate,
            language=language,
        )
