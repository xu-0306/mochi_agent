"""預設設定值常數（供其他模組引用）。"""

from __future__ import annotations

import sys
from pathlib import Path

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
DEFAULT_TTS_VOICE = "af_heart"
DEFAULT_EDGE_TTS_VOICE_PRESETS = [
    "en-US-AriaNeural",
    "en-US-JennyNeural",
    "zh-CN-XiaoxiaoNeural",
    "zh-TW-HsiaoChenNeural",
]
WINDOWS_STATE_ROOT = ".mochi"


def running_on_windows() -> bool:
    """目前 Python runtime 是否為 Windows。"""
    return sys.platform.startswith("win")


def default_workspace_dir() -> str:
    """依平台回傳預設 workspace 目錄。"""
    return WINDOWS_STATE_ROOT if running_on_windows() else "~/.mochi"


def default_sessions_dir() -> str:
    """依平台回傳預設 sessions 目錄。"""
    return f"{WINDOWS_STATE_ROOT}/sessions" if running_on_windows() else "~/.mochi/sessions"


def default_skills_dir() -> str:
    """依平台回傳預設 skills 目錄。"""
    return f"{WINDOWS_STATE_ROOT}/skills" if running_on_windows() else "~/.mochi/skills"


def default_plugins_dir() -> str:
    """依平台回傳預設 plugins 目錄。"""
    return f"{WINDOWS_STATE_ROOT}/plugins" if running_on_windows() else "~/.mochi/plugins"


def default_memory_db_path() -> Path:
    """依平台回傳預設 memory SQLite 路徑。"""
    return Path(WINDOWS_STATE_ROOT) / "memory.db" if running_on_windows() else Path.home() / ".mochi" / "memory.db"


def default_config_path() -> Path:
    """依平台回傳預設使用者設定檔路徑。"""
    return Path(WINDOWS_STATE_ROOT) / "config.yaml" if running_on_windows() else Path.home() / ".mochi" / "config.yaml"


def repo_skills_dir() -> Path:
    """回傳 repo 內建 skills 目錄。"""
    return Path(__file__).resolve().parents[1] / "skills"
