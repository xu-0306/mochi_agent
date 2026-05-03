"""語音能力描述：供 API / CLI / channel adapter 共用。"""

from __future__ import annotations

from typing import Any, get_args

from mochi.config.schema import VoiceConfig
from mochi.voice.events import VoiceStage
from mochi.voice.ws_bridge import VoiceWebSocketBridge

VOICE_CLIENT_MESSAGE_TYPES: tuple[str, ...] = (
    "audio_chunk",
    "vad_end",
    "interrupt",
)

VOICE_SERVER_EVENT_TYPES: tuple[str, ...] = (
    "vad_state",
    "voice_stage",
    "transcription",
    "text",
    "audio_chunk",
    "error",
    "done",
    "interrupted",
)

VOICE_VAD_STATES: tuple[str, ...] = (
    "speech_started",
    "speech_ended",
)


def get_voice_capabilities() -> dict[str, Any]:
    """回傳目前 bounded voice surface 的共享能力描述。"""
    voice_config = VoiceConfig()
    audio_contract = voice_config.voice_input_contract
    channel_policy = voice_config.voice_input_channel_policy
    return {
        "transport": "websocket",
        "path": "/v1/voice",
        "query_params": {
            "idle_timeout_seconds": {
                "default": VoiceWebSocketBridge._DEFAULT_AUTO_END_IDLE_TIMEOUT_SECONDS,
                "min": VoiceWebSocketBridge._MIN_AUTO_END_IDLE_TIMEOUT_SECONDS,
                "max": VoiceWebSocketBridge._MAX_AUTO_END_IDLE_TIMEOUT_SECONDS,
            },
        },
        "audio": {
            "encoding": audio_contract["encoding"],
            "sample_format": audio_contract["sample_format"],
            "endianness": audio_contract["endianness"],
            "channels": audio_contract["channels"],
            "channel_layout": audio_contract["channel_layout"],
            "sample_rate_hz": audio_contract["sample_rate_hz"],
            "payload": audio_contract["payload_encoding"],
            "message_type": audio_contract["message_type"],
            "message_field": audio_contract["message_field"],
            "pcm_input": audio_contract["pcm_input"],
        },
        "audio_input_contract": dict(audio_contract),
        "audio_input_channel_policy": dict(channel_policy),
        "client_messages": list(VOICE_CLIENT_MESSAGE_TYPES),
        "server_events": list(VOICE_SERVER_EVENT_TYPES),
        "states": {
            "vad_state": list(VOICE_VAD_STATES),
            "voice_stage": list(get_args(VoiceStage)),
        },
        "features": {
            "explicit_vad_end": True,
            "server_side_vad_state": True,
            "voice_stage_events": True,
            "server_side_endpointing": True,
            "idle_auto_end": True,
            "transcription_is_final": True,
            "interrupt_clears_buffer": True,
            "interrupt_suppresses_stale_output": True,
            "raw_pcm16_input_required": True,
            "mono_only_input": True,
            "whisperlivekit_pcm_input_default": True,
        },
    }
