"""Settings bounded API routes。"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from mochi.api.server import _get_config, _maybe_await
from mochi.config.defaults import DEFAULT_EDGE_TTS_VOICE_PRESETS
from mochi.config.manager import save_config
from mochi.config.schema import (
    ChannelsConfig,
    DiscordPlatformConfig,
    LearningConfig,
    LocaleDefaultsConfig,
    MemoryConfig,
    MochiConfig,
    TelegramPlatformConfig,
    VoiceConfig,
)
from mochi.voice.model_manager import (
    ensure_model_available,
    ensure_qwen_model_available,
    resolve_bounded_stt_runtime_spec,
)
from mochi.voice.router import SUPPORTED_STT_BACKENDS, SUPPORTED_TTS_BACKENDS

router = APIRouter(prefix="/v1", tags=["settings"])

WHISPER_MODEL_PRESETS = [
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large",
    "large-v1",
    "large-v2",
    "large-v3",
    "turbo",
]
FASTER_WHISPER_MODEL_PRESETS = [
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large-v1",
    "large-v2",
    "large-v3",
    "turbo",
    "distil-large-v3",
]
STT_MODEL_PRESETS_BY_BACKEND: dict[str, list[str]] = {
    "auto": FASTER_WHISPER_MODEL_PRESETS,
    "faster-whisper": FASTER_WHISPER_MODEL_PRESETS,
    "openai-whisper": WHISPER_MODEL_PRESETS,
    "whisperlivekit": WHISPER_MODEL_PRESETS,
    "whisper-cpp": WHISPER_MODEL_PRESETS,
    "openai-api": ["whisper-1"],
    "qwen-asr": ["qwen3-asr-0.6b", "qwen3-asr-1.7b"],
    "vosk": [
        "vosk-model-small-cn-0.22",
        "vosk-model-cn-0.22",
        "vosk-model-small-en-us-0.15",
    ],
}
TTS_MODEL_PRESETS_BY_BACKEND: dict[str, list[str]] = {
    "auto": ["none"],
    "edge-tts": ["none"],
    "openai-tts": ["gpt-4o-mini-tts", "tts-1", "tts-1-hd"],
    "piper": ["none"],
    "coqui-tts": [
        "tts_models/en/ljspeech/tacotron2-DDC",
        "tts_models/en/ljspeech/glow-tts",
        "tts_models/multilingual/multi-dataset/xtts_v2",
    ],
    "kokoro-tts": ["none"],
}
TTS_VOICE_PRESETS_BY_BACKEND: dict[str, list[str]] = {
    "auto": DEFAULT_EDGE_TTS_VOICE_PRESETS,
    "edge-tts": DEFAULT_EDGE_TTS_VOICE_PRESETS,
    "openai-tts": ["alloy", "verse", "aria", "coral", "sage", "nova", "shimmer"],
    "piper": ["zh_CN-huayan-medium", "en_US-lessac-medium"],
    "coqui-tts": ["default"],
    "kokoro-tts": ["af_heart", "af_bella", "bf_emma", "am_adam", "bm_george"],
}


class VoiceSettingsPatch(BaseModel):
    """可由 WebGUI 更新的非敏感 voice 設定。"""

    enabled: bool | None = None
    stt_backend: Literal[
        "auto",
        "faster-whisper",
        "openai-api",
        "openai-whisper",
        "qwen-asr",
        "vosk",
        "whisper-cpp",
        "whisperlivekit",
    ] | None = None
    stt_model: str | None = Field(default=None, min_length=1)
    stt_language: str | None = None
    stt_device: str | None = None
    stt_model_cache_dir: str | None = None
    stt_model_path: str | None = None
    tts_backend: Literal[
        "auto",
        "coqui-tts",
        "edge-tts",
        "kokoro-tts",
        "openai-tts",
        "piper",
    ] | None = None
    tts_model: str | None = None
    tts_voice: str | None = None
    tts_language: str | None = None
    tts_speed: float | None = Field(default=None, ge=0.5, le=2.0)
    tts_use_gpu: bool | None = None
    tts_kokoro_lang_code: str | None = None
    tts_openai_base_url: str | None = None
    tts_openai_response_format: Literal["pcm", "wav"] | None = None


class MemorySettingsPatch(BaseModel):
    """可由 WebGUI 更新的 memory 設定。"""

    db_path: str | None = None
    max_short_term_messages: int | None = Field(default=None, ge=1, le=500)
    fts_top_k: int | None = Field(default=None, ge=1, le=50)


class LearningSettingsPatch(BaseModel):
    """可由 WebGUI 更新的 learning 設定。"""

    enabled: bool | None = None
    auto_extract_skills: bool | None = None
    min_steps_for_extraction: int | None = Field(default=None, ge=1, le=100)
    trajectory_retention_days: int | None = Field(default=None, ge=1, le=3650)
    skill_improvement_threshold: float | None = Field(default=None, ge=0.0, le=1.0)
    max_skills: int | None = Field(default=None, ge=1, le=100_000)


class LocaleDefaultsSettingsPatch(BaseModel):
    """可由 WebGUI 更新的非敏感 locale/defaults 設定。"""

    region_profile: str | None = Field(default=None, min_length=1)
    ui_locale: str | None = Field(default=None, min_length=1)
    ui_locale_fallback: str | None = Field(default=None, min_length=1)
    response_language: str | None = Field(default=None, min_length=1)
    default_tts_voice: str | None = Field(default=None, min_length=1)
    timezone: str | None = Field(default=None, min_length=1)


class PathSettingsPatch(BaseModel):
    """可由 WebGUI 更新的頂層路徑設定。"""

    workspace_dir: str | None = None
    sessions_dir: str | None = None
    skills_dir: str | None = None
    plugins_dir: str | None = None


class DiscordChannelSettingsPatch(BaseModel):
    """可由 WebGUI 更新的非敏感 Discord 設定。"""

    enabled: bool | None = None
    text_enabled: bool | None = None
    voice_enabled: bool | None = None
    allowed_guild_ids: list[int] | None = None
    allowed_channel_ids: list[int] | None = None
    allowed_voice_channel_ids: list[int] | None = None
    allowed_user_ids: list[int] | None = None
    rate_limit_per_user: int | None = Field(default=None, ge=1, le=10_000)
    message_mode: Literal["all_messages", "mentions_only", "slash_only"] | None = None
    auto_join_policy: Literal["manual_only"] | None = None
    voice_auto_reply: bool | None = None
    voice_stt_enabled: bool | None = None
    voice_tts_enabled: bool | None = None


class TelegramChannelSettingsPatch(BaseModel):
    """可由 WebGUI 更新的非敏感 Telegram 設定。"""

    enabled: bool | None = None
    allowed_chat_ids: list[int] | None = None
    allowed_user_ids: list[int] | None = None
    rate_limit_per_user: int | None = Field(default=None, ge=1, le=10_000)


class ChannelsSettingsPatch(BaseModel):
    """可由 WebGUI 更新的非敏感 channel 設定。"""

    discord: DiscordChannelSettingsPatch | None = None
    telegram: TelegramChannelSettingsPatch | None = None


class UpdateSettingsRequest(BaseModel):
    """`PATCH /v1/settings` request payload。"""

    voice: VoiceSettingsPatch | None = None
    memory: MemorySettingsPatch | None = None
    learning: LearningSettingsPatch | None = None
    locale_defaults: LocaleDefaultsSettingsPatch | None = None
    paths: PathSettingsPatch | None = None
    channels: ChannelsSettingsPatch | None = None
    download_missing_models: bool = False
    reload_voice: bool = True
    persist: bool = True


def _stringify_path(value: Any) -> str | None:
    """將 Path-like 值轉為字串。"""
    if value is None:
        return None
    return str(value)


@router.get("/settings")
async def get_settings(request: Request) -> dict[str, Any]:
    """回傳非敏感設定摘要。"""
    app = request.app
    config = await _get_config(app)
    return _settings_payload(config)


@router.patch("/settings")
async def update_settings(request: Request, payload: UpdateSettingsRequest) -> dict[str, Any]:
    """更新 WebGUI 可編輯的非敏感 runtime 設定。"""
    config = await _get_config(request.app)
    updated = _apply_settings_patch(config, payload)
    _ensure_config_directories(updated)
    download_result = await _maybe_prepare_voice_models(updated.voice, payload.download_missing_models)

    request.app.state.config = updated
    engine = getattr(request.app.state, "engine", None)
    if engine is not None:
        apply_config = getattr(engine, "apply_config", None)
        if callable(apply_config):
            await _maybe_await(apply_config(updated, reload_voice=payload.reload_voice))

    persisted_path = _persist_config_if_enabled(request, updated, payload.persist)

    response = _settings_payload(updated)
    response["update"] = {
        "type": "settings_update",
        "download": download_result,
        "persisted": persisted_path is not None,
        "config_path": str(persisted_path) if persisted_path is not None else None,
    }
    return response


def _settings_payload(config: MochiConfig) -> dict[str, Any]:
    """建立 WebGUI 使用的非敏感設定 payload。"""
    trajectory_path = Path(config.workspace_dir).expanduser() / "trajectories.jsonl"
    skills_db_path = Path(config.skills_dir).expanduser() / "skills.db"
    return {
        "type": "settings",
        "model": config.model,
        "model_config": {
            "provider": _configured_provider(config),
            "ollama_base_url": config.ollama.base_url,
            "ollama_model": config.model.removeprefix("ollama:")
            if config.model.startswith("ollama:")
            else "",
            "openai_compat_provider": config.openai_compat.provider,
            "openai_compat_base_url": config.openai_compat.base_url,
            "openai_compat_model": config.openai_compat.model,
            "openai_compat_api_key_configured": config.openai_compat.api_key is not None,
        },
        "model_setup": {
            "mode": config.model_setup.mode,
            "default_provider": config.model_setup.default_provider,
            "default_model": config.model_setup.default_model,
            "default_model_spec": config.model_setup.default_model_spec,
            "setup_required": config.model_setup.setup_required,
            "fallback_chain": list(config.model_setup.fallback_chain),
        },
        "locale_defaults": {
            "region_profile": config.locale_defaults.region_profile,
            "ui_locale": config.locale_defaults.ui_locale,
            "ui_locale_fallback": config.locale_defaults.ui_locale_fallback,
            "response_language": config.locale_defaults.response_language,
            "default_tts_voice": config.locale_defaults.default_tts_voice,
            "timezone": config.locale_defaults.timezone,
        },
        "voice": {
            "enabled": config.voice.enabled,
            "stt_backend": config.voice.stt_backend,
            "stt_model": config.voice.stt_model,
            "stt_language": config.voice.stt_language,
            "stt_device": config.voice.stt_device,
            "stt_model_cache_dir": _stringify_path(config.voice.stt_model_cache_dir),
            "stt_model_path": _stringify_path(config.voice.stt_model_path),
            "supported_stt_backends": sorted(SUPPORTED_STT_BACKENDS),
            "supported_stt_models_by_backend": STT_MODEL_PRESETS_BY_BACKEND,
            "tts_backend": config.voice.tts_backend,
            "tts_model": config.voice.tts_model,
            "tts_voice": config.voice.tts_voice,
            "tts_language": config.voice.tts_language,
            "tts_speed": config.voice.tts_speed,
            "tts_use_gpu": config.voice.tts_use_gpu,
            "tts_kokoro_lang_code": config.voice.tts_kokoro_lang_code,
            "tts_openai_base_url": config.voice.tts_openai_base_url,
            "tts_openai_response_format": config.voice.tts_openai_response_format,
            "supported_tts_backends": sorted(SUPPORTED_TTS_BACKENDS),
            "supported_tts_models_by_backend": TTS_MODEL_PRESETS_BY_BACKEND,
            "supported_tts_voices_by_backend": TTS_VOICE_PRESETS_BY_BACKEND,
            "sample_rate": config.voice.sample_rate,
            "channels": config.voice.channels,
            "stt_runtime_spec": resolve_bounded_stt_runtime_spec(
                stt_backend=config.voice.stt_backend,
                stt_model=config.voice.stt_model,
                stt_model_cache_dir=config.voice.stt_model_cache_dir,
                stt_model_path=config.voice.stt_model_path,
                stt_openai_base_url=config.voice.stt_openai_base_url,
            ).to_dict(),
        },
        "memory": {
            "db_path": _stringify_path(config.memory.db_path),
            "max_short_term_messages": config.memory.max_short_term_messages,
            "fts_top_k": config.memory.fts_top_k,
        },
        "learning": {
            "enabled": config.learning.enabled,
            "auto_extract_skills": config.learning.auto_extract_skills,
            "min_steps_for_extraction": config.learning.min_steps_for_extraction,
            "trajectory_retention_days": config.learning.trajectory_retention_days,
            "skill_improvement_threshold": config.learning.skill_improvement_threshold,
            "max_skills": config.learning.max_skills,
            "trajectory_path": str(trajectory_path),
            "skills_db_path": str(skills_db_path),
        },
        "channels": {
            "discord": {
                "enabled": config.channels.discord.enabled,
                "text_enabled": config.channels.discord.text_enabled,
                "voice_enabled": config.channels.discord.voice_enabled,
                "bot_token_configured": config.channels.discord.bot_token is not None,
                "allowed_guild_ids": config.channels.discord.allowed_guild_ids,
                "allowed_channel_ids": config.channels.discord.allowed_channel_ids,
                "allowed_voice_channel_ids": config.channels.discord.allowed_voice_channel_ids,
                "allowed_user_ids": config.channels.discord.allowed_user_ids,
                "rate_limit_per_user": config.channels.discord.rate_limit_per_user,
                "message_mode": config.channels.discord.message_mode,
                "auto_join_policy": config.channels.discord.auto_join_policy,
                "voice_auto_reply": config.channels.discord.voice_auto_reply,
                "voice_stt_enabled": config.channels.discord.voice_stt_enabled,
                "voice_tts_enabled": config.channels.discord.voice_tts_enabled,
            },
            "telegram": {
                "enabled": config.channels.telegram.enabled,
                "allowed_chat_ids": config.channels.telegram.allowed_chat_ids,
                "allowed_user_ids": config.channels.telegram.allowed_user_ids,
                "rate_limit_per_user": config.channels.telegram.rate_limit_per_user,
            },
        },
        "web": {
            "host": config.web.host,
            "port": config.web.port,
        },
        "paths": {
            "workspace_dir": config.workspace_dir,
            "sessions_dir": config.sessions_dir,
            "skills_dir": config.skills_dir,
            "plugins_dir": config.plugins_dir,
        },
    }


def _apply_settings_patch(config: MochiConfig, payload: UpdateSettingsRequest) -> MochiConfig:
    updates: dict[str, Any] = {}
    if payload.voice is not None:
        voice_updates = payload.voice.model_dump(exclude_unset=True)
        voice_updates = _normalize_empty_paths(voice_updates, {"stt_model_path"})
        updates["voice"] = VoiceConfig.model_validate(
            {
                **config.voice.model_dump(),
                **voice_updates,
            }
        )

    if payload.memory is not None:
        updates["memory"] = MemoryConfig.model_validate(
            {
                **config.memory.model_dump(),
                **payload.memory.model_dump(exclude_unset=True),
            }
        )

    if payload.learning is not None:
        updates["learning"] = LearningConfig.model_validate(
            {
                **config.learning.model_dump(),
                **payload.learning.model_dump(exclude_unset=True),
            }
        )

    if payload.locale_defaults is not None:
        updates["locale_defaults"] = LocaleDefaultsConfig.model_validate(
            {
                **config.locale_defaults.model_dump(),
                **payload.locale_defaults.model_dump(exclude_unset=True),
            }
        )

    if payload.channels is not None:
        channels_updates: dict[str, Any] = {}
        if payload.channels.discord is not None:
            channels_updates["discord"] = DiscordPlatformConfig.model_validate(
                {
                    **config.channels.discord.model_dump(),
                    **payload.channels.discord.model_dump(exclude_unset=True),
                }
            )
        if payload.channels.telegram is not None:
            channels_updates["telegram"] = TelegramPlatformConfig.model_validate(
                {
                    **config.channels.telegram.model_dump(),
                    **payload.channels.telegram.model_dump(exclude_unset=True),
                }
            )
        if channels_updates:
            updates["channels"] = ChannelsConfig.model_validate(
                {
                    **config.channels.model_dump(),
                    **channels_updates,
                }
            )

    if payload.paths is not None:
        updates.update(_normalize_empty_paths(payload.paths.model_dump(exclude_unset=True), set()))

    return config.model_copy(update=updates)


def _configured_provider(config: MochiConfig) -> str:
    if config.model.startswith("ollama:"):
        return "ollama"
    if config.model.startswith(("http://", "https://")):
        return config.openai_compat.provider
    if config.model.lower().endswith(".gguf"):
        return "gguf"
    return "safetensors"


def _persist_config_if_enabled(
    request: Request,
    config: MochiConfig,
    persist: bool,
) -> Path | None:
    if not persist:
        return None
    config_path = getattr(request.app.state, "config_path", None)
    if config_path is None and getattr(request.app.state, "config_factory", None) is not None:
        return None
    return save_config(config, config_path)


def _normalize_empty_paths(values: dict[str, Any], nullable_path_keys: set[str]) -> dict[str, Any]:
    normalized = dict(values)
    for key, value in values.items():
        if isinstance(value, str) and not value.strip() and key in nullable_path_keys:
            normalized[key] = None
    return normalized


def _ensure_config_directories(config: MochiConfig) -> None:
    """建立設定指向的資料目錄，不建立模型檔本身。"""
    Path(config.workspace_dir).expanduser().mkdir(parents=True, exist_ok=True)
    Path(config.sessions_dir).expanduser().mkdir(parents=True, exist_ok=True)
    Path(config.skills_dir).expanduser().mkdir(parents=True, exist_ok=True)
    Path(config.plugins_dir).expanduser().mkdir(parents=True, exist_ok=True)
    Path(config.memory.db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
    Path(config.voice.stt_model_cache_dir).expanduser().mkdir(parents=True, exist_ok=True)
    if config.voice.stt_model_path is not None:
        Path(config.voice.stt_model_path).expanduser().parent.mkdir(parents=True, exist_ok=True)


async def _maybe_prepare_voice_models(
    voice: VoiceConfig,
    download_missing_models: bool,
) -> dict[str, Any]:
    backend = voice.stt_backend
    stt_cfg = {
        "model": voice.stt_model,
        "model_cache_dir": str(voice.stt_model_cache_dir),
        "model_path": str(voice.stt_model_path) if voice.stt_model_path is not None else None,
    }
    if not download_missing_models:
        return {"requested": False, "status": "skipped"}

    if backend == "openai-whisper":
        updated = await ensure_model_available(stt_cfg, download_fn=_download_file)
        return {
            "requested": True,
            "status": "ready" if updated.get("model_path") else "not_supported",
            "model_path": updated.get("model_path"),
        }

    if backend == "qwen-asr":
        updated = await ensure_qwen_model_available(stt_cfg, snapshot_fn=_snapshot_qwen_model)
        qwen_cfg = updated.get("qwen_asr", {})
        model_dir = qwen_cfg.get("model_dir") if isinstance(qwen_cfg, dict) else None
        return {
            "requested": True,
            "status": "ready" if model_dir else "not_supported",
            "model_path": model_dir,
        }

    if backend in {"auto", "faster-whisper"}:
        return {
            "requested": True,
            "status": "runtime_download_on_first_use",
            "model_cache_dir": str(voice.stt_model_cache_dir),
        }

    return {
        "requested": True,
        "status": "not_supported",
        "reason": f"Automatic model download is not implemented for STT backend {backend!r}.",
    }


async def _download_file(model_name: str, model_url: str, target: Path) -> None:
    """下載 Whisper 權重到指定路徑。"""
    del model_name
    import httpx

    target.parent.mkdir(parents=True, exist_ok=True)
    temp_target = target.with_suffix(target.suffix + ".part")
    async with (
        httpx.AsyncClient(timeout=None, follow_redirects=True) as client,
        client.stream("GET", model_url) as response,
    ):
        response.raise_for_status()
        with temp_target.open("wb") as handle:
            async for chunk in response.aiter_bytes():
                handle.write(chunk)
    temp_target.replace(target)


def _snapshot_qwen_model(repo: str, target: Path) -> None:
    """用 huggingface_hub 下載 Qwen ASR snapshot。"""
    try:
        from huggingface_hub import snapshot_download  # type: ignore[import-not-found]
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required to download Qwen ASR models. "
            "Install the hf extra or install huggingface_hub manually."
        ) from exc

    snapshot_download(
        repo_id=repo,
        local_dir=str(target),
        local_dir_use_symlinks=False,
    )
