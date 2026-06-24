"""Voice preset/recommendation metadata tests."""

from __future__ import annotations

from mochi.voice.presets import (
    EXTERNAL_API_TTS_PRESETS,
    LOCAL_TTS_RECOMMENDATIONS,
    get_voice_recommendations_payload,
)


def test_local_tts_recommendations_prioritize_kokoro_first() -> None:
    """Local-first TTS recommendations should prefer Kokoro as the default path."""
    assert LOCAL_TTS_RECOMMENDATIONS[0]["backend"] == "kokoro-tts"
    assert LOCAL_TTS_RECOMMENDATIONS[0]["default_voice"] == "af_heart"


def test_external_api_tts_presets_include_higgs_and_voicebox_modes() -> None:
    """External presets should distinguish OpenAI-compatible vs adapter-required targets."""
    ids = {item["id"] for item in EXTERNAL_API_TTS_PRESETS}
    assert {"higgs-audio", "voicebox"}.issubset(ids)

    higgs = next(item for item in EXTERNAL_API_TTS_PRESETS if item["id"] == "higgs-audio")
    assert higgs["compatibility"] == "openai-compatible"
    assert higgs["backend"] == "external-api"
    assert higgs["model"] == "bosonai/higgs-audio-v2-generation-3B-base"

    voicebox = next(item for item in EXTERNAL_API_TTS_PRESETS if item["id"] == "voicebox")
    assert voicebox["compatibility"] == "adapter-required"
    assert voicebox["backend"] == "external-api"


def test_voice_recommendations_payload_is_api_ready() -> None:
    """Combined payload should expose both local recommendations and external presets."""
    payload = get_voice_recommendations_payload()
    assert payload["recommended_local_tts_backends"][0]["backend"] == "kokoro-tts"
    assert any(item["id"] == "higgs-audio" for item in payload["external_api_tts_presets"])
