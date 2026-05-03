"""語音事件型別定義。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

VoiceStage = Literal["transcribing", "thinking", "synthesizing"]


@dataclass
class VoiceStageEvent:
    """語音流程階段事件。"""

    type: Literal["voice_stage"] = field(default="voice_stage", init=False)
    stage: VoiceStage = "transcribing"


@dataclass
class TranscriptionEvent:
    """STT 轉寫事件。"""

    type: Literal["transcription"] = field(default="transcription", init=False)
    text: str = ""
    is_final: bool = True


@dataclass
class PartialTranscriptionEvent(TranscriptionEvent):
    """STT 部分轉寫事件（預留未來串流 STT）。"""

    is_final: Literal[False] = False


@dataclass
class FinalTranscriptionEvent(TranscriptionEvent):
    """STT 最終轉寫事件。"""

    is_final: Literal[True] = True


@dataclass
class AgentFinalTextEvent:
    """Agent 最終文字回覆事件。"""

    type: Literal["agent_final_text"] = field(default="agent_final_text", init=False)
    text: str = ""


@dataclass
class SynthesizedAudioChunkEvent:
    """TTS 合成音訊分塊事件。"""

    type: Literal["synthesized_audio_chunk"] = field(
        default="synthesized_audio_chunk",
        init=False,
    )
    chunk: bytes = b""


@dataclass
class VoiceErrorEvent:
    """語音流程錯誤事件。"""

    type: Literal["error"] = field(default="error", init=False)
    message: str = ""
    code: str = "VOICE_ERROR"


VoiceEvent = (
    VoiceStageEvent
    | TranscriptionEvent
    | PartialTranscriptionEvent
    | FinalTranscriptionEvent
    | AgentFinalTextEvent
    | SynthesizedAudioChunkEvent
    | VoiceErrorEvent
)
