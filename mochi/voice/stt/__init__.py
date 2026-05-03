"""STT（語音轉文字）多後端子套件。"""

from __future__ import annotations

from .faster_whisper import FasterWhisperSTT
from .openai_api import OpenAIApiSTT
from .openai_whisper import OpenAIWhisperSTT
from .qwen_asr import QwenASRSTT
from .vosk import VoskSTT
from .whisper_cpp import WhisperCppSTT
from .whisperlivekit import WhisperLiveKitSTT

__all__ = [
    "FasterWhisperSTT",
    "OpenAIApiSTT",
    "OpenAIWhisperSTT",
    "QwenASRSTT",
    "VoskSTT",
    "WhisperCppSTT",
    "WhisperLiveKitSTT",
]
