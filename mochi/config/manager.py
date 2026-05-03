"""設定載入與管理器。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from loguru import logger
from pydantic import SecretStr

from mochi.config.schema import MochiConfig

PROJECT_DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


def user_config_path() -> Path:
    """回傳目前使用者的 Mochi YAML 設定檔路徑。"""
    return Path.home() / ".mochi" / "config.yaml"


def _coerce_mapping(value: Any) -> dict[str, Any]:
    """將任意值安全轉為 dict。"""
    if isinstance(value, dict):
        return dict(value)
    return {}


def _read_env(name: str) -> str | None:
    """讀取環境變數並忽略空白值。"""
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def read_env_cors_origins() -> list[str] | None:
    """讀取 CORS origins env override，供 config 與 app bootstrap 共用。"""
    cors_raw = _read_env("MOCHI_WEB_CORS_ORIGINS")
    if cors_raw is None:
        return None

    cors_origins = [item.strip() for item in cors_raw.split(",") if item.strip()]
    if cors_origins:
        return cors_origins

    logger.warning("Ignore empty MOCHI_WEB_CORS_ORIGINS override.")
    return None


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """將常用部署環境變數覆蓋到設定。"""
    merged = dict(raw)

    web_section = _coerce_mapping(merged.get("web"))
    host = _read_env("MOCHI_WEB_HOST")
    if host is not None:
        web_section["host"] = host

    port_raw = _read_env("MOCHI_WEB_PORT")
    if port_raw is not None:
        try:
            web_section["port"] = int(port_raw)
        except ValueError:
            logger.warning(
                "Ignore invalid MOCHI_WEB_PORT={!r}; expected integer.",
                port_raw,
            )

    cors_origins = read_env_cors_origins()
    if cors_origins is not None:
        web_section["cors_origins"] = cors_origins

    if web_section:
        merged["web"] = web_section

    locale_defaults_section = _coerce_mapping(merged.get("locale_defaults"))
    locale_env_map = {
        "MOCHI_REGION_PROFILE": "region_profile",
        "MOCHI_LOCALE": "ui_locale",
        "MOCHI_UI_LOCALE_FALLBACK": "ui_locale_fallback",
        "MOCHI_TIMEZONE": "timezone",
        "MOCHI_RESPONSE_LANGUAGE": "response_language",
        "MOCHI_DEFAULT_TTS_VOICE": "default_tts_voice",
    }
    for env_name, config_key in locale_env_map.items():
        value = _read_env(env_name)
        if value is not None:
            locale_defaults_section[config_key] = value

    if locale_defaults_section:
        merged["locale_defaults"] = locale_defaults_section

    default_tts_voice = _read_env("MOCHI_DEFAULT_TTS_VOICE")
    tts_voice = _read_env("MOCHI_TTS_VOICE") or default_tts_voice
    if tts_voice is not None:
        voice_section = _coerce_mapping(merged.get("voice"))
        voice_section["tts_voice"] = tts_voice
        merged["voice"] = voice_section

    ollama_base_url = _read_env("MOCHI_OLLAMA_BASE_URL")
    if ollama_base_url is not None:
        ollama_section = _coerce_mapping(merged.get("ollama"))
        ollama_section["base_url"] = ollama_base_url
        merged["ollama"] = ollama_section

    return merged


def load_config(config_path: str | Path | None = None) -> MochiConfig:
    """從 YAML 檔案載入設定，找不到時回傳預設值。

    Args:
        config_path: YAML 設定檔路徑；None 時依序嘗試
                     ~/.mochi/config.yaml → ./configs/default.yaml。

    Returns:
        解析後的 MochiConfig 實例。
    """
    search_paths: list[Path] = []

    if config_path is not None:
        search_paths.append(Path(config_path))
    else:
        search_paths.extend([
            user_config_path(),
            PROJECT_DEFAULT_CONFIG_PATH,
        ])

    for path in search_paths:
        if path.exists():
            logger.debug(f"Loading config from {path}")
            raw = _coerce_mapping(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
            return MochiConfig.model_validate(_apply_env_overrides(raw))

    logger.debug("No config file found, using defaults.")
    return MochiConfig.model_validate(_apply_env_overrides({}))


def save_config(config: MochiConfig, config_path: str | Path | None = None) -> Path:
    """將設定保存成 YAML，預設寫入使用者設定檔。

    `SecretStr` 會以原始值寫入本機檔案；呼叫端仍需避免將檔案內容回傳到 API。
    """
    path = Path(config_path) if config_path is not None else user_config_path()
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = _serialize_for_yaml(config.model_dump(mode="python"))
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
    logger.debug(f"Saved config to {path}")
    return path


def _serialize_for_yaml(value: Any) -> Any:
    """轉換 Pydantic dump 結果為 PyYAML 可安全輸出的基本型別。"""
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _serialize_for_yaml(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize_for_yaml(item) for item in value]
    return value
