"""Voice runtime status helpers."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable
from typing import Any, cast

from mochi.config.schema import VoiceConfig
from mochi.voice.base import VoiceInfo


async def build_voice_runtime_status(
    *,
    config: VoiceConfig,
    supported_stt_backends: list[str],
    supported_tts_backends: list[str],
    stt_component: object | None,
    tts_component: object | None,
    vad_component: object | None,
    has_vad_factory: bool,
    stt_runtime_spec: dict[str, Any] | None = None,
    last_load_error: str | None = None,
    session_diagnostics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a shared voice runtime status payload."""
    stt_status = await _describe_component(
        stt_component,
        configured_backend=config.stt_backend,
        configured_name=config.stt_model,
    )
    if stt_runtime_spec is not None:
        stt_status["runtime"] = dict(stt_runtime_spec)

    tts_status = await _describe_component(
        tts_component,
        configured_backend=config.tts_backend,
        configured_name=config.tts_voice,
    )
    vad_status = await _describe_component(vad_component, configured_name="default")
    vad_status["factory_ready"] = has_vad_factory

    loaded = (
        stt_status["loaded"]
        and tts_status["loaded"]
        and (vad_status["loaded"] or has_vad_factory)
    )

    payload: dict[str, Any] = {
        "type": "voice_runtime_status",
        "phase": "bounded",
        "enabled": config.enabled,
        "loaded": loaded,
        "features": {
            "transcription_preview": (
                config.stt_backend == "whisperlivekit"
                or _supports_transcription_preview(stt_component)
            ),
        },
        "supported_backends": {
            "stt": list(supported_stt_backends),
            "tts": list(supported_tts_backends),
        },
        "configured": {
            "stt_backend": config.stt_backend,
            "stt_model": config.stt_model,
            "stt_model_path": str(config.stt_model_path) if config.stt_model_path else None,
            "stt_openai_base_url": config.stt_openai_base_url,
            "stt_openai_api_key_configured": config.stt_openai_api_key is not None,
            "tts_backend": config.tts_backend,
            "tts_model": config.tts_model,
            "tts_voice": config.tts_voice,
            "tts_language": config.tts_language,
            "tts_use_gpu": config.tts_use_gpu,
            "tts_kokoro_lang_code": config.tts_kokoro_lang_code,
            "tts_split_pattern": config.tts_split_pattern,
            "tts_openai_base_url": config.tts_openai_base_url,
            "tts_openai_api_key_configured": config.tts_openai_api_key is not None,
            "tts_openai_timeout": config.tts_openai_timeout,
            "tts_openai_response_format": config.tts_openai_response_format,
            "reply_model_mode": config.reply_model_mode,
            "reply_model_id": config.reply_model_id,
            "session_mode": config.session_mode,
            "voice_pack_dir": str(config.voice_pack_dir),
            "registered_tts_voices": [
                {
                    "id": voice.id,
                    "backend": voice.backend,
                    "path": str(voice.path),
                    "label": voice.label,
                    "source": voice.source,
                }
                for voice in config.registered_tts_voices
            ],
            "sample_rate": config.sample_rate,
            "channels": config.channels,
            "voice_input_contract": config.voice_input_contract,
            "voice_input_channel_policy": config.voice_input_channel_policy,
        },
        "stt": stt_status,
        "tts": tts_status,
        "vad": vad_status,
    }
    if last_load_error:
        payload["last_load_error"] = last_load_error
    if session_diagnostics is not None:
        payload["session_diagnostics"] = dict(session_diagnostics)
    return payload


async def _describe_component(
    component: object | None,
    *,
    configured_backend: str | None = None,
    configured_name: str | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"loaded": component is not None}
    if configured_backend is not None:
        payload["configured_backend"] = configured_backend
    if configured_name is not None:
        payload["configured_name"] = configured_name

    if component is None:
        return payload

    info = _safe_get_info(component)
    if info is not None:
        payload["info"] = _voice_info_to_dict(info)

    health, health_error = await _safe_health_check(component)
    payload["healthy"] = health
    if health_error is not None:
        payload["health_error"] = health_error
    return payload


def _safe_get_info(component: object) -> VoiceInfo | None:
    getter = getattr(component, "get_info", None)
    if not callable(getter):
        return None
    try:
        info = getter()
    except Exception:
        return None
    if isinstance(info, VoiceInfo):
        return info
    return None


def _voice_info_to_dict(info: VoiceInfo) -> dict[str, Any]:
    return {
        "kind": info.kind,
        "family": info.family,
        "name": info.name,
        "metadata": dict(info.metadata),
    }


async def _safe_health_check(component: object) -> tuple[bool | None, str | None]:
    checker = getattr(component, "health_check", None)
    if not callable(checker):
        return None, None
    try:
        result = checker()
        if inspect.isawaitable(result):
            result = await cast(Awaitable[bool], result)
        return bool(result), None
    except Exception as exc:  # pragma: no cover
        return False, str(exc)


def _supports_transcription_preview(component: object | None) -> bool:
    if component is None:
        return False
    checker = getattr(component, "supports_transcription_preview", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            return False
    info = _safe_get_info(component)
    if info is None:
        return False
    return bool(info.metadata.get("supports_preview", False))
