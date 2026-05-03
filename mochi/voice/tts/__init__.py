"""TTS（文字轉語音）多後端子套件。"""

from __future__ import annotations

from .coqui_tts import CoquiTTS
from .edge_tts import EdgeTTS
from .kokoro_tts import KokoroTTS
from .openai_tts import OpenAITTS
from .piper import PiperTTS

__all__ = ["CoquiTTS", "EdgeTTS", "KokoroTTS", "OpenAITTS", "PiperTTS"]
