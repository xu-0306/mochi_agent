"""Settings bounded API routes。"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel, Field, SecretStr

from mochi.api.server import _get_config, _maybe_await, _rebuild_channel_manager
from mochi.auth.openai_codex import OpenAICodexAuthService, normalize_openai_codex_base_url
from mochi.backends.inference_capabilities import ReasoningEffort
from mochi.config.defaults import DEFAULT_EDGE_TTS_VOICE_PRESETS
from mochi.config.manager import save_config
from mochi.config.schema import (
    AgentConfig,
    ChannelsConfig,
    DiscordPlatformConfig,
    GGUFConfig,
    InferencePreset,
    LearningConfig,
    LocalModelConfig,
    LocaleDefaultsConfig,
    MemoryConfig,
    MochiConfig,
    RegisteredTTSVoiceConfig,
    SecurityConfig,
    TelegramPlatformConfig,
    ToolsConfig,
    VLLMConfig,
    VoiceConfig,
)
from mochi.learning.skill_library_factory import resolve_skills_db_path
from mochi.security.policy import autonomy_mode_defaults
from mochi.tools.web_search_providers import build_web_search_provider_status_payload
from mochi.voice.model_manager import (
    ensure_model_available,
    ensure_qwen_model_available,
    ensure_tts_runtime_available,
    resolve_bounded_stt_runtime_spec,
)
from mochi.voice.presets import get_voice_recommendations_payload
from mochi.voice.router import SUPPORTED_STT_BACKENDS, SUPPORTED_TTS_BACKENDS

router = APIRouter(prefix="/v1", tags=["settings"])

_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^[A-Za-z]:[\\/]")

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
    "external-api": ["whisper-1"],
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
    "external-api": ["gpt-4o-mini-tts", "tts-1", "tts-1-hd"],
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
    "external-api": ["alloy", "verse", "aria", "coral", "sage", "nova", "shimmer"],
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
        "external-api",
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
    stt_openai_base_url: str | None = None
    stt_openai_api_key: SecretStr | None = None
    stt_openai_timeout: float | None = Field(default=None, ge=0.0)
    tts_backend: Literal[
        "auto",
        "coqui-tts",
        "edge-tts",
        "external-api",
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
    tts_openai_api_key: SecretStr | None = None
    tts_openai_timeout: float | None = Field(default=None, ge=0.0)
    tts_openai_response_format: Literal["pcm", "wav"] | None = None
    reply_model_mode: Literal["inherit_active", "configured_model"] | None = None
    reply_model_id: str | None = None
    session_mode: Literal["append_current", "isolated_voice"] | None = None
    voice_pack_dir: str | None = None
    registered_tts_voices: list[RegisteredTTSVoiceConfig] | None = None


class MemorySettingsPatch(BaseModel):
    """可由 WebGUI 更新的 memory 設定。"""

    db_path: str | None = None
    max_short_term_messages: int | None = Field(default=None, ge=1, le=500)
    semantic_compaction_enabled: bool | None = None
    semantic_summary_mode: Literal["deterministic", "hybrid"] | None = None
    max_short_term_tokens: int | None = Field(default=None, ge=256, le=1_000_000)
    semantic_keep_recent_messages: int | None = Field(default=None, ge=2, le=200)
    fts_top_k: int | None = Field(default=None, ge=1, le=50)


class LearningSettingsPatch(BaseModel):
    """可由 WebGUI 更新的 learning 設定。"""

    enabled: bool | None = None
    auto_extract_skills: bool | None = None
    auto_sync_filesystem_skills: bool | None = None
    min_steps_for_extraction: int | None = Field(default=None, ge=1, le=100)
    min_tool_calls_for_extraction: int | None = Field(default=None, ge=0, le=100)
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


class AgentSettingsPatch(BaseModel):
    """可由 WebGUI 更新的 agent 推理設定。"""

    system_prompt: str | None = None
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, ge=1, le=131072)
    top_p: float | None = Field(default=None, ge=0.0, le=1.0)
    min_p: float | None = Field(default=None, ge=0.0, le=1.0)
    top_k: int | None = Field(default=None, ge=0)
    frequency_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    presence_penalty: float | None = Field(default=None, ge=-2.0, le=2.0)
    repeat_penalty: float | None = Field(default=None, ge=0.0, le=2.0)
    reasoning_effort: ReasoningEffort | None = None
    show_token_stats: bool | None = None
    presets: list[InferencePreset] | None = None
    active_preset: str | None = None


class SecuritySettingsPatch(BaseModel):
    """可由 WebGUI 更新的非敏感安全設定。"""

    autonomy_mode: Literal["trusted_workspace", "strict", "high_autonomy", "auto_review"] | None = None
    require_approval_for_shell: bool | None = None
    require_approval_for_file_write: bool | None = None
    require_approval_for_exec: bool | None = None
    agent_run_default_max_wall_clock_sec: int | None = Field(default=None, ge=1, le=86_400)
    agent_run_default_heartbeat_timeout_sec: int | None = Field(default=None, ge=1, le=86_400)
    agent_run_default_checkpoint_interval_steps: int | None = Field(default=None, ge=1, le=10_000)
    agent_run_default_max_subagent_failures_per_role: int | None = Field(default=None, ge=0, le=100)
    agent_run_default_on_budget_exhausted: Literal["pause", "finalize_partial"] | None = None
    agent_run_default_on_subagent_disconnect: Literal["retry_then_degrade", "pause", "fail"] | None = None
    exec_default_timeout_sec: int | None = Field(default=None, ge=1, le=86_400)
    exec_session_output_limit: int | None = Field(default=None, ge=256, le=1_000_000)
    max_file_write_size_mb: float | None = Field(default=None, ge=0.0)
    file_ops_scope: Literal["workspace", "any"] | None = None
    file_undo_max_size_mb: float | None = Field(default=None, ge=0.0)


class ToolsSettingsPatch(BaseModel):
    """供 WebGUI / TUI 更新 web search / fetch 工具設定。"""

    web_search_engine: Literal[
        "tavily", "serper", "jina", "exa",
        "brave", "searxng", "duckduckgo", "duckduckgo_html",
    ] | None = None
    web_search_fallback_engines: list[str] | None = None
    web_search_tavily_api_key: SecretStr | None = None
    web_search_serper_api_key: SecretStr | None = None
    web_search_jina_api_key: SecretStr | None = None
    web_search_exa_api_key: SecretStr | None = None
    web_search_brave_api_key: SecretStr | None = None
    web_search_searxng_base_url: str | None = None
    web_search_language: str | None = None
    web_search_region: str | None = None
    web_fetch_extractor: Literal["trafilatura", "jina_reader", "htmlparser"] | None = None
    web_fetch_jina_api_key: SecretStr | None = None


class LocalModelSettingsPatch(BaseModel):
    """可由 WebGUI 更新的本地模型掛載/卸載設定。"""

    idle_unload_enabled: bool | None = None
    idle_unload_seconds: int | None = Field(default=None, ge=0, le=86_400)


class GGUFSettingsPatch(BaseModel):
    """GGUF runtime settings patch."""

    n_ctx: int | None = Field(default=None, ge=1, le=262_144)


class VLLMSettingsPatch(BaseModel):
    """vLLM runtime settings patch."""

    max_model_len: int | None = Field(default=None, ge=1, le=262_144)


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
    admin_user_ids: list[int] | None = None
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

    agent: AgentSettingsPatch | None = None
    voice: VoiceSettingsPatch | None = None
    memory: MemorySettingsPatch | None = None
    learning: LearningSettingsPatch | None = None
    locale_defaults: LocaleDefaultsSettingsPatch | None = None
    paths: PathSettingsPatch | None = None
    channels: ChannelsSettingsPatch | None = None
    security: SecuritySettingsPatch | None = None
    tools: ToolsSettingsPatch | None = None
    local_models: LocalModelSettingsPatch | None = None
    gguf: GGUFSettingsPatch | None = None
    vllm: VLLMSettingsPatch | None = None
    download_missing_models: bool = False
    reload_voice: bool = True
    persist: bool = True


class DiscordSetupRequest(BaseModel):
    """Discord 安全 onboarding request。"""

    bot_token: str | None = None
    enabled: bool = True
    text_enabled: bool = True
    voice_enabled: bool = True
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
    persist: bool = True
    reload_voice: bool = False


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
    download_result = await _maybe_prepare_voice_models(
        updated.voice,
        payload.download_missing_models,
        prepare_tts=_payload_updates_tts(payload.voice),
    )

    request.app.state.config = updated
    engine = getattr(request.app.state, "engine", None)
    if engine is not None:
        apply_config = getattr(engine, "apply_config", None)
        if callable(apply_config):
            await _maybe_await(apply_config(updated, reload_voice=payload.reload_voice))
    if payload.channels is not None and getattr(request.app.state, "channel_manager", None) is not None:
        await _rebuild_channel_manager(request.app)

    persisted_path = _persist_config_if_enabled(request, updated, payload.persist)

    response = _settings_payload(updated)
    response["update"] = {
        "type": "settings_update",
        "download": download_result,
        "persisted": persisted_path is not None,
        "config_path": str(persisted_path) if persisted_path is not None else None,
    }
    return response


@router.post("/setup/discord")
async def setup_discord(request: Request, payload: DiscordSetupRequest) -> dict[str, Any]:
    """以專用安全路徑設定 Discord token 與頻道選項。"""
    config = await _get_config(request.app)
    try:
        updated = _apply_discord_setup(config, payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    _ensure_config_directories(updated)

    request.app.state.config = updated
    engine = getattr(request.app.state, "engine", None)
    if engine is not None:
        apply_config = getattr(engine, "apply_config", None)
        if callable(apply_config):
            await _maybe_await(apply_config(updated, reload_voice=payload.reload_voice))
    if (
        getattr(request.app.state, "channel_manager", None) is not None
        or updated.channels.discord.enabled
    ):
        await _rebuild_channel_manager(request.app)

    persisted_path = _persist_config_if_enabled(request, updated, payload.persist)
    response = _settings_payload(updated)
    response["update"] = {
        "type": "discord_setup",
        "persisted": persisted_path is not None,
        "config_path": str(persisted_path) if persisted_path is not None else None,
        "discord": {
            "configured": True,
            "enabled": updated.channels.discord.enabled,
            "text_enabled": updated.channels.discord.text_enabled,
            "voice_enabled": updated.channels.discord.voice_enabled,
        },
    }
    return response


def _settings_payload(config: MochiConfig) -> dict[str, Any]:
    """建立 WebGUI 使用的非敏感設定 payload。"""
    trajectory_path = Path(config.workspace_dir).expanduser() / "trajectories.jsonl"
    skills_db_path = resolve_skills_db_path(
        skills_dir=config.skills_dir,
    )
    configured_provider = _configured_provider(config)
    active_openai_codex_profile_id = OpenAICodexAuthService(config.workspace_dir).resolve_profile_id(
        config.openai_codex.auth_profile_id
    )
    voice_recommendations = get_voice_recommendations_payload()
    return {
        "type": "settings",
        "model": config.model,
        "model_config": {
            "provider": configured_provider,
            "ollama_base_url": config.ollama.base_url,
            "ollama_model": config.model.removeprefix("ollama:")
            if configured_provider == "ollama"
            else "",
            "openai_compat_provider": config.openai_compat.provider,
            "openai_compat_base_url": config.openai_compat.base_url,
            "openai_compat_model": config.openai_compat.model,
            "openai_compat_api_key_configured": config.openai_compat.api_key is not None,
            "openai_codex_base_url": config.openai_codex.base_url,
            "openai_codex_model": config.openai_codex.model,
            "openai_codex_auth_profile_id": active_openai_codex_profile_id,
            "openai_codex_auth_configured": active_openai_codex_profile_id is not None,
            "local_model_path": config.model
            if configured_provider == "local"
            else "",
            "local_model_root": _local_model_root(config),
        },
        "model_setup": {
            "mode": config.model_setup.mode,
            "default_provider": config.model_setup.default_provider,
            "default_model": config.model_setup.default_model,
            "default_model_spec": config.model_setup.default_model_spec,
            "setup_required": config.model_setup.setup_required,
            "fallback_chain": list(config.model_setup.fallback_chain),
            "configured_models": [
                model.model_dump()
                for model in config.model_setup.configured_models
            ],
        },
        "locale_defaults": {
            "region_profile": config.locale_defaults.region_profile,
            "ui_locale": config.locale_defaults.ui_locale,
            "ui_locale_fallback": config.locale_defaults.ui_locale_fallback,
            "response_language": config.locale_defaults.response_language,
            "default_tts_voice": config.locale_defaults.default_tts_voice,
            "timezone": config.locale_defaults.timezone,
        },
        "agent": {
            "system_prompt": config.agent.system_prompt,
            "temperature": config.agent.temperature,
            "max_tokens": config.agent.max_tokens,
            "top_p": config.agent.top_p,
            "min_p": config.agent.min_p,
            "top_k": config.agent.top_k,
            "frequency_penalty": config.agent.frequency_penalty,
            "presence_penalty": config.agent.presence_penalty,
            "repeat_penalty": config.agent.repeat_penalty,
            "reasoning_effort": config.agent.reasoning_effort,
            "show_token_stats": config.agent.show_token_stats,
            "presets": [preset.model_dump() for preset in config.agent.presets],
            "active_preset": config.agent.active_preset,
        },
        "voice": {
            "enabled": config.voice.enabled,
            "stt_backend": config.voice.stt_backend,
            "stt_model": config.voice.stt_model,
            "stt_language": config.voice.stt_language,
            "stt_device": config.voice.stt_device,
            "stt_model_cache_dir": _stringify_path(config.voice.stt_model_cache_dir),
            "stt_model_path": _stringify_path(config.voice.stt_model_path),
            "stt_openai_base_url": config.voice.stt_openai_base_url,
            "stt_openai_api_key_configured": config.voice.stt_openai_api_key is not None,
            "stt_openai_timeout": config.voice.stt_openai_timeout,
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
            "tts_openai_api_key_configured": config.voice.tts_openai_api_key is not None,
            "tts_openai_timeout": config.voice.tts_openai_timeout,
            "tts_openai_response_format": config.voice.tts_openai_response_format,
            "reply_model_mode": config.voice.reply_model_mode,
            "reply_model_id": config.voice.reply_model_id,
            "session_mode": config.voice.session_mode,
            "voice_pack_dir": _stringify_path(config.voice.voice_pack_dir),
            "registered_tts_voices": [
                {
                    "id": voice.id,
                    "backend": voice.backend,
                    "path": str(voice.path),
                    "label": voice.label,
                    "source": voice.source,
                }
                for voice in config.voice.registered_tts_voices
            ],
            "supported_tts_backends": sorted(SUPPORTED_TTS_BACKENDS),
            "supported_tts_models_by_backend": TTS_MODEL_PRESETS_BY_BACKEND,
            "supported_tts_voices_by_backend": TTS_VOICE_PRESETS_BY_BACKEND,
            "recommended_local_tts_backends": voice_recommendations["recommended_local_tts_backends"],
            "external_api_tts_presets": voice_recommendations["external_api_tts_presets"],
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
            "semantic_compaction_enabled": config.memory.semantic_compaction_enabled,
            "semantic_summary_mode": config.memory.semantic_summary_mode,
            "max_short_term_tokens": config.memory.max_short_term_tokens,
            "semantic_keep_recent_messages": config.memory.semantic_keep_recent_messages,
            "fts_top_k": config.memory.fts_top_k,
        },
        "tools": {
            "web_search_engine": config.tools.web_search_engine,
            "web_search_fallback_engines": config.tools.web_search_fallback_engines,
            "web_search_searxng_base_url": config.tools.web_search_searxng_base_url,
            "web_search_language": config.tools.web_search_language,
            "web_search_region": config.tools.web_search_region,
            "web_fetch_extractor": config.tools.web_fetch_extractor,
            "web_fetch_jina_api_key_configured": config.tools.web_fetch_jina_api_key
            is not None,
            **build_web_search_provider_status_payload(config.tools),
        },
        "learning": {
            "enabled": config.learning.enabled,
            "auto_extract_skills": config.learning.auto_extract_skills,
            "auto_sync_filesystem_skills": config.learning.auto_sync_filesystem_skills,
            "min_steps_for_extraction": config.learning.min_steps_for_extraction,
            "min_tool_calls_for_extraction": config.learning.min_tool_calls_for_extraction,
            "trajectory_retention_days": config.learning.trajectory_retention_days,
            "skill_improvement_threshold": config.learning.skill_improvement_threshold,
            "max_skills": config.learning.max_skills,
            "trajectory_path": str(trajectory_path),
            "skills_db_path": str(skills_db_path),
        },
        "local_models": {
            "idle_unload_enabled": config.local_models.idle_unload_enabled,
            "idle_unload_seconds": config.local_models.idle_unload_seconds,
        },
        "gguf": {
            "n_ctx": config.gguf.n_ctx,
        },
        "vllm": {
            "max_model_len": config.vllm.max_model_len,
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
                "admin_user_ids": config.channels.discord.admin_user_ids,
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
        "security": {
            "autonomy_mode": config.security.autonomy_mode,
            "require_approval_for_shell": config.security.require_approval_for_shell,
            "require_approval_for_file_write": config.security.require_approval_for_file_write,
            "require_approval_for_exec": config.security.require_approval_for_exec,
            "agent_run_default_max_wall_clock_sec": config.security.agent_run_default_max_wall_clock_sec,
            "agent_run_default_heartbeat_timeout_sec": config.security.agent_run_default_heartbeat_timeout_sec,
            "agent_run_default_checkpoint_interval_steps": config.security.agent_run_default_checkpoint_interval_steps,
            "agent_run_default_max_subagent_failures_per_role": config.security.agent_run_default_max_subagent_failures_per_role,
            "agent_run_default_on_budget_exhausted": config.security.agent_run_default_on_budget_exhausted,
            "agent_run_default_on_subagent_disconnect": config.security.agent_run_default_on_subagent_disconnect,
            "exec_default_timeout_sec": config.security.exec_default_timeout_sec,
            "exec_session_output_limit": config.security.exec_session_output_limit,
            "max_file_write_size_mb": config.security.max_file_write_size_mb,
            "file_ops_scope": config.security.file_ops_scope,
            "file_undo_max_size_mb": config.security.file_undo_max_size_mb,
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
    if payload.agent is not None:
        updates["agent"] = AgentConfig.model_validate(
            {
                **config.agent.model_dump(),
                **payload.agent.model_dump(exclude_unset=True),
            }
        )
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

    if payload.local_models is not None:
        updates["local_models"] = LocalModelConfig.model_validate(
            {
                **config.local_models.model_dump(),
                **payload.local_models.model_dump(exclude_unset=True),
            }
        )

    if payload.gguf is not None:
        updates["gguf"] = GGUFConfig.model_validate(
            {
                **config.gguf.model_dump(),
                **payload.gguf.model_dump(exclude_unset=True),
            }
        )

    if payload.vllm is not None:
        updates["vllm"] = VLLMConfig.model_validate(
            {
                **config.vllm.model_dump(),
                **payload.vllm.model_dump(exclude_unset=True),
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

    if payload.security is not None:
        security_updates = payload.security.model_dump(exclude_unset=True)
        requested_mode = security_updates.get("autonomy_mode")
        if isinstance(requested_mode, str):
            mode_defaults = autonomy_mode_defaults(requested_mode)
            for key in (
                "require_approval_for_shell",
                "require_approval_for_file_write",
                "require_approval_for_exec",
                "file_ops_scope",
            ):
                security_updates.setdefault(key, mode_defaults[key])
        updates["security"] = SecurityConfig.model_validate(
            {
                **config.security.model_dump(),
                **security_updates,
            }
        )

    if payload.tools is not None:
        updates["tools"] = ToolsConfig.model_validate(
            {
                **config.tools.model_dump(),
                **payload.tools.model_dump(exclude_unset=True),
            }
        )

    return config.model_copy(update=updates)


def _apply_discord_setup(config: MochiConfig, payload: DiscordSetupRequest) -> MochiConfig:
    """套用 Discord 專用 onboarding 設定。"""
    discord_updates = payload.model_dump(exclude={"bot_token", "persist", "reload_voice"}, exclude_unset=True)
    existing_token = config.channels.discord.bot_token
    normalized_token = payload.bot_token.strip() if isinstance(payload.bot_token, str) else ""
    if normalized_token:
        secret_token = SecretStr(normalized_token)
    elif existing_token is not None:
        secret_token = existing_token
    else:
        raise ValueError("Discord bot token is required for initial setup.")
    merged_discord = {
        **config.channels.discord.model_dump(),
        **discord_updates,
        "bot_token": secret_token,
    }
    updated_discord = DiscordPlatformConfig.model_validate(merged_discord)
    updated_channels = ChannelsConfig.model_validate(
        {
            **config.channels.model_dump(),
            "discord": updated_discord,
        }
    )
    return config.model_copy(update={"channels": updated_channels})


def _configured_provider(config: MochiConfig) -> str:
    if config.model.startswith("ollama:"):
        return "ollama"
    if config.model.startswith(("http://", "https://")):
        try:
            normalized_codex_base_url = normalize_openai_codex_base_url(config.openai_codex.base_url)
        except ValueError:
            normalized_codex_base_url = None
        active_openai_codex_profile_id = OpenAICodexAuthService(config.workspace_dir).resolve_profile_id(
            config.openai_codex.auth_profile_id
        )
        if (
            normalized_codex_base_url is not None
            and config.model.rstrip("/") == normalized_codex_base_url
            and active_openai_codex_profile_id is not None
        ):
            return "openai_codex"
        return config.openai_compat.provider
    if _looks_like_local_model_path(config.model):
        return "local"
    return "ollama"


def _looks_like_local_model_path(value: str) -> bool:
    """用路徑形狀判斷是否為本地模型 spec，避免把 Ollama 裸模型名誤判成 local。"""
    raw = value.strip()
    if not raw or raw.startswith(("ollama:", "http://", "https://")):
        return False
    normalized = raw.replace("\\", "/")
    return (
        raw.startswith(("/", "~/", "./", "../", "\\\\"))
        or bool(_WINDOWS_ABSOLUTE_PATH_RE.match(raw))
        or normalized.startswith("/mnt/")
        or "/" in raw
        or "\\" in raw
    )


def _local_model_root(config: MochiConfig) -> str:
    """回傳目前 local model 的表單 root path。"""
    if _configured_provider(config) != "local":
        return ""
    model_path = Path(config.model).expanduser()
    if config.model.lower().endswith(".gguf"):
        return str(model_path.parent)
    return str(model_path.parent if model_path.name else model_path)


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
    _mkdir_if_possible(Path(config.workspace_dir).expanduser())
    _mkdir_if_possible(Path(config.sessions_dir).expanduser())
    _mkdir_if_possible(Path(config.skills_dir).expanduser())
    _mkdir_if_possible(Path(config.plugins_dir).expanduser())
    _mkdir_if_possible(Path(config.memory.db_path).expanduser().parent)
    _mkdir_if_possible(Path(config.voice.stt_model_cache_dir).expanduser())
    _mkdir_if_possible(Path(config.voice.voice_pack_dir).expanduser())
    if config.voice.stt_model_path is not None:
        _mkdir_if_possible(Path(config.voice.stt_model_path).expanduser().parent)


def _mkdir_if_possible(path: Path) -> None:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except PermissionError:
        return


async def _maybe_prepare_voice_models(
    voice: VoiceConfig,
    download_missing_models: bool,
    *,
    prepare_tts: bool = False,
) -> dict[str, Any]:
    backend = voice.stt_backend
    stt_cfg = {
        "model": voice.stt_model,
        "model_cache_dir": str(voice.stt_model_cache_dir),
        "model_path": str(voice.stt_model_path) if voice.stt_model_path is not None else None,
    }
    if not download_missing_models:
        return {"requested": False, "status": "skipped"}

    stt_result: dict[str, Any]
    if backend == "openai-whisper":
        updated = await ensure_model_available(stt_cfg, download_fn=_download_file)
        stt_result = {
            "requested": True,
            "status": "ready" if updated.get("model_path") else "not_supported",
            "model_path": updated.get("model_path"),
        }
    elif backend == "qwen-asr":
        updated = await ensure_qwen_model_available(stt_cfg, snapshot_fn=_snapshot_qwen_model)
        qwen_cfg = updated.get("qwen_asr", {})
        model_dir = qwen_cfg.get("model_dir") if isinstance(qwen_cfg, dict) else None
        stt_result = {
            "requested": True,
            "status": "ready" if model_dir else "not_supported",
            "model_path": model_dir,
        }
    elif backend in {"auto", "faster-whisper"}:
        stt_result = {
            "requested": True,
            "status": "runtime_download_on_first_use",
            "model_cache_dir": str(voice.stt_model_cache_dir),
        }
    else:
        stt_result = {
            "requested": True,
            "status": "not_supported",
            "reason": f"Automatic model download is not implemented for STT backend {backend!r}.",
        }

    tts_result = (
        await ensure_tts_runtime_available(voice)
        if prepare_tts
        else {
            "requested": False,
            "backend": str(voice.tts_backend),
            "status": "skipped",
        }
    )
    combined_status = _combine_voice_prepare_status(stt_result, tts_result)
    payload: dict[str, Any] = {
        "requested": bool(stt_result.get("requested")) or bool(tts_result.get("requested")),
        "status": combined_status,
        "stt": stt_result,
        "tts": tts_result,
    }
    if combined_status == "attention_required":
        payload["message"] = _voice_prepare_attention_message(stt_result, tts_result)
    return payload


def _payload_updates_tts(payload: VoiceSettingsPatch | None) -> bool:
    if payload is None:
        return False
    return any(
        field_name.startswith("tts_")
        or field_name in {"voice_pack_dir", "registered_tts_voices"}
        for field_name in payload.model_fields_set
    )


def _combine_voice_prepare_status(
    stt_result: dict[str, Any],
    tts_result: dict[str, Any],
) -> str:
    statuses = [
        str(stt_result.get("status", "")).strip(),
        str(tts_result.get("status", "")).strip(),
    ]
    if "prepare_failed" in statuses:
        return "attention_required"
    if "runtime_download_on_first_use" in statuses:
        return "runtime_download_on_first_use"
    if "ready" in statuses:
        return "ready"
    for candidate in statuses:
        if candidate and candidate != "skipped":
            return candidate
    return "skipped"


def _voice_prepare_attention_message(
    stt_result: dict[str, Any],
    tts_result: dict[str, Any],
) -> str:
    for label, result in (("STT", stt_result), ("TTS", tts_result)):
        if str(result.get("status", "")).strip() != "prepare_failed":
            continue
        error = str(result.get("error", "")).strip()
        backend = str(result.get("backend", "")).strip()
        if backend:
            return f"{label} backend {backend} could not be prepared automatically: {error}"
        return f"{label} backend could not be prepared automatically: {error}"
    return "One or more voice backends could not be prepared automatically."


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
