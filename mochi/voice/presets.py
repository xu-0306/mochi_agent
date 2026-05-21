"""Voice preset metadata shared by API surfaces and UI clients."""

from __future__ import annotations

from typing import Any

LOCAL_TTS_RECOMMENDATIONS: list[dict[str, Any]] = [
    {
        "id": "kokoro-default",
        "backend": "kokoro-tts",
        "label": "Kokoro",
        "default_voice": "af_heart",
        "default_model": None,
        "local": True,
        "priority": 1,
        "summary": "Lightweight local default with good quality and fast startup.",
        "notes": [
            "Best default for local voice chat.",
            "Works well without a remote service.",
        ],
    },
    {
        "id": "piper-default",
        "backend": "piper",
        "label": "Piper",
        "default_voice": "en_US-lessac-medium",
        "default_model": None,
        "local": True,
        "priority": 2,
        "summary": "Simple offline path with downloadable ONNX voice packs.",
        "notes": [
            "Good for fully offline deployment.",
            "Custom uploaded voice packs fit naturally here.",
        ],
    },
    {
        "id": "coqui-default",
        "backend": "coqui-tts",
        "label": "Coqui TTS",
        "default_voice": "default",
        "default_model": "tts_models/multilingual/multi-dataset/xtts_v2",
        "local": True,
        "priority": 3,
        "summary": "Flexible local stack when you need broader model choices.",
        "notes": [
            "Heavier than Kokoro or Piper.",
            "Useful when multilingual model selection matters.",
        ],
    },
]

EXTERNAL_API_TTS_PRESETS: list[dict[str, Any]] = [
    {
        "id": "higgs-audio",
        "backend": "external-api",
        "label": "Higgs Audio",
        "compatibility": "openai-compatible",
        "model": "bosonai/higgs-audio-v2-generation-3B-base",
        "voice": "alloy",
        "summary": "High-end remote TTS option that fits the current external API path.",
        "requires_base_url": True,
        "requires_api_key": False,
        "apply_supported": True,
        "notes": [
            "Point this at an OpenAI-compatible Higgs deployment.",
            "Leave the base URL empty until your server endpoint is ready.",
        ],
    },
    {
        "id": "voicebox",
        "backend": "external-api",
        "planned_backend": "voicebox-bridge",
        "label": "Voicebox",
        "compatibility": "adapter-required",
        "model": None,
        "voice": None,
        "summary": "Promising extra integration target, but it needs a dedicated adapter first.",
        "requires_base_url": True,
        "requires_api_key": False,
        "apply_supported": False,
        "notes": [
            "Not directly compatible with the current OpenAI-style TTS adapter.",
            "Plan this as a future dedicated backend or bridge service.",
        ],
    },
]


def _clone_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cloned: list[dict[str, Any]] = []
    for item in items:
        copied = dict(item)
        notes = copied.get("notes")
        if isinstance(notes, list):
            copied["notes"] = list(notes)
        cloned.append(copied)
    return cloned


def get_voice_recommendations_payload() -> dict[str, list[dict[str, Any]]]:
    """Return API-ready voice recommendation metadata with independent containers."""
    return {
        "recommended_local_tts_backends": _clone_items(LOCAL_TTS_RECOMMENDATIONS),
        "external_api_tts_presets": _clone_items(EXTERNAL_API_TTS_PRESETS),
    }
