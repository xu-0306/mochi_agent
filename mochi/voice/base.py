"""語音子系統抽象介面（Phase 4 foundation）。"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True, frozen=True)
class VoiceInfo:
    """語音元件資訊。"""

    kind: str
    family: str
    name: str
    metadata: dict[str, Any] = field(default_factory=dict)


class BaseSTT(ABC):
    """STT（Speech-to-Text）抽象介面。"""

    @abstractmethod
    async def transcribe(
        self,
        audio: bytes,
        *,
        sample_rate: int,
        language: str | None = None,
    ) -> str:
        """將音訊位元組轉為文字。"""
        ...

    @abstractmethod
    def get_info(self) -> VoiceInfo:
        """取得 STT 端資訊。"""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """回報 STT 端是否可用。"""
        ...

    async def close(self) -> None:
        """釋放資源。"""
        return None


class BaseTTS(ABC):
    """TTS（Text-to-Speech）抽象介面。"""

    @abstractmethod
    async def synthesize(
        self,
        text: str,
        *,
        voice: str | None = None,
        speed: float | None = None,
    ) -> bytes:
        """將文字轉為音訊位元組。"""
        ...

    @abstractmethod
    def get_info(self) -> VoiceInfo:
        """取得 TTS 端資訊。"""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """回報 TTS 端是否可用。"""
        ...

    async def close(self) -> None:
        """釋放資源。"""
        return None


class BaseVAD(ABC):
    """VAD（Voice Activity Detection）抽象介面。"""

    @abstractmethod
    async def is_speech(self, audio: bytes, *, sample_rate: int) -> bool:
        """判斷該段音訊是否為語音活動。"""
        ...

    @abstractmethod
    def reset(self) -> None:
        """重置 VAD 狀態。"""
        ...

    @abstractmethod
    def get_info(self) -> VoiceInfo:
        """取得 VAD 端資訊。"""
        ...
