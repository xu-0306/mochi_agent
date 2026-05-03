"""預設設定值常數（供其他模組引用）。"""

from __future__ import annotations

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL_PROVIDER = "ollama"
DEFAULT_MODEL_NAME = "llama3.2"
DEFAULT_MODEL = f"{DEFAULT_MODEL_PROVIDER}:{DEFAULT_MODEL_NAME}"
DEFAULT_MODEL_SETUP_MODE = "configured_or_setup"
DEFAULT_MODEL_SETUP_REQUIRED = True
DEFAULT_MODEL_FALLBACK_CHAIN = [
    "user_config",
    "ollama_tags",
    "openai_compatible_provider",
]
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_REGION_PROFILE = "global"
DEFAULT_UI_LOCALE = "auto"
DEFAULT_UI_LOCALE_FALLBACK = "en-US"
DEFAULT_RESPONSE_LANGUAGE = "same_as_user"
DEFAULT_TIMEZONE = "auto"
DEFAULT_TTS_VOICE = "en-US-AriaNeural"
DEFAULT_EDGE_TTS_VOICE_PRESETS = [
    DEFAULT_TTS_VOICE,
    "en-US-JennyNeural",
    "zh-CN-XiaoxiaoNeural",
    "zh-TW-HsiaoChenNeural",
]
