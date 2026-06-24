"""Voice capability API 測試。"""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from mochi.api.server import create_app
from mochi.config.schema import MochiConfig, RegisteredTTSVoiceConfig, VoiceConfig
from mochi.voice.capabilities import get_voice_capabilities


def test_voice_capabilities_helper_returns_independent_payloads() -> None:
    """helper 每次都應回傳可安全修改的獨立 payload。"""
    payload_1 = get_voice_capabilities()
    payload_2 = get_voice_capabilities()

    payload_1["client_messages"].append("mutated")
    payload_1["audio_input_contract"]["encoding"] = "mutated"

    assert "mutated" not in payload_2["client_messages"]
    assert payload_2["audio_input_contract"]["encoding"] == "pcm16"


def test_voice_capabilities_expose_explicit_pcm16_input_contract() -> None:
    """capabilities 應明確揭露 `/v1/voice` 的 raw PCM16/base64 輸入契約。"""
    payload = get_voice_capabilities()
    voice_config = VoiceConfig()
    expected_contract = voice_config.voice_input_contract

    assert payload["audio_input_contract"] == expected_contract
    assert payload["audio_input_channel_policy"] == voice_config.voice_input_channel_policy
    assert payload["audio"] == {
        "encoding": "pcm16",
        "sample_format": "s16le",
        "endianness": "little",
        "channels": 1,
        "channel_layout": "mono",
        "sample_rate_hz": 16000,
        "payload": "base64",
        "message_type": "audio_chunk",
        "message_field": "data",
        "pcm_input": True,
    }
    assert payload["features"]["raw_pcm16_input_required"] is True
    assert payload["features"]["mono_only_input"] is True
    assert payload["features"]["whisperlivekit_pcm_input_default"] is True


def test_voice_capabilities_route_returns_shared_capabilities() -> None:
    """API 應暴露與共享 helper 一致的 voice capability 描述。"""
    app = create_app()

    with TestClient(app) as client:
        response = client.get("/v1/voice/capabilities")

    assert response.status_code == 200
    assert response.json() == get_voice_capabilities()


def test_voice_status_route_uses_engine_runtime_status_surface() -> None:
    """API 應可讀取 engine 的共享 voice runtime status。"""
    app = create_app()
    expected = {
        "type": "voice_runtime_status",
        "phase": "bounded",
        "loaded": True,
        "enabled": True,
    }

    class _FakeEngine:
        async def get_voice_runtime_status(self) -> dict[str, object]:
            return expected

    app.state.engine_factory = lambda: _FakeEngine()
    app.state.voice_bridge_diagnostics = {
        "preview_append_failures": 2,
        "preview_flush_failures": 1,
        "preview_degraded_turns": 3,
        "last_preview_failure": {
            "stage": "flush",
            "error_type": "RuntimeError",
            "message": "preview flush boom",
            "session_id": "s-1",
        },
    }

    with TestClient(app) as client:
        response = client.get("/v1/voice/status")

    assert response.status_code == 200
    assert response.json() == {
        **expected,
        "bridge_diagnostics": {
            "preview_append_failures": 2,
            "preview_flush_failures": 1,
            "preview_degraded_turns": 3,
            "last_preview_failure": {
                "stage": "flush",
                "error_type": "RuntimeError",
                "message": "preview flush boom",
                "session_id": "s-1",
            },
        },
    }
    assert expected == {
        "type": "voice_runtime_status",
        "phase": "bounded",
        "loaded": True,
        "enabled": True,
    }


def test_voice_status_route_returns_fallback_when_engine_has_no_status_method() -> None:
    """若 engine 未提供 status surface，API 應回傳相容 fallback。"""
    app = create_app()

    class _EngineWithoutStatus:
        pass

    app.state.engine_factory = lambda: _EngineWithoutStatus()

    with TestClient(app) as client:
        response = client.get("/v1/voice/status")

    assert response.status_code == 200
    assert response.json() == {
        "type": "voice_runtime_status",
        "phase": "bounded",
        "loaded": False,
        "error": "Engine does not provide get_voice_runtime_status().",
        "bridge_diagnostics": {
            "preview_append_failures": 0,
            "preview_flush_failures": 0,
            "preview_degraded_turns": 0,
            "last_preview_failure": None,
        },
    }


def test_voice_prepare_route_preloads_runtime_for_requested_session() -> None:
    """prepare route 會在開始錄音前先要求 engine 預載 runtime。"""
    app = create_app()
    prepared_sessions: list[str | None] = []
    expected = {
        "type": "voice_runtime_status",
        "phase": "bounded",
        "loaded": True,
        "enabled": True,
    }

    class _FakeEngine:
        async def prepare_voice_runtime(self, session_id: str | None = None) -> dict[str, object]:
            prepared_sessions.append(session_id)
            return expected

    app.state.engine_factory = lambda: _FakeEngine()

    with TestClient(app) as client:
        response = client.post("/v1/voice/prepare", json={"session_id": "voice-session-1"})

    assert response.status_code == 200
    assert prepared_sessions == ["voice-session-1"]
    assert response.json() == {
        **expected,
        "bridge_diagnostics": {
            "preview_append_failures": 0,
            "preview_flush_failures": 0,
            "preview_degraded_turns": 0,
            "last_preview_failure": None,
        },
    }


def test_voice_voices_route_lists_registered_registry_and_presets(tmp_path: Path) -> None:
    custom_voice_path = tmp_path / "voices" / "custom-piper.onnx"
    custom_voice_path.parent.mkdir(parents=True, exist_ok=True)
    custom_voice_path.write_bytes(b"voice")
    app = create_app()
    app.state.config_factory = lambda: MochiConfig(
        voice=VoiceConfig(
            voice_pack_dir=tmp_path / "voice-packs",
            registered_tts_voices=[
                RegisteredTTSVoiceConfig(
                    id="custom-piper",
                    backend="piper",
                    path=custom_voice_path,
                    label="Custom Piper",
                    source="registered_path",
                )
            ],
        )
    )

    with TestClient(app) as client:
        response = client.get("/v1/voice/voices")

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "voice_voices"
    assert payload["voice_pack_dir"] == str(tmp_path / "voice-packs")
    assert "piper" in payload["presets_by_backend"]
    assert payload["items"] == [
        {
            "id": "custom-piper",
            "backend": "piper",
            "path": str(custom_voice_path),
            "label": "Custom Piper",
            "source": "registered_path",
            "is_available": True,
        }
    ]


def test_voice_voices_register_path_updates_shared_config(tmp_path: Path) -> None:
    custom_voice_path = tmp_path / "voices" / "registered.onnx"
    custom_voice_path.parent.mkdir(parents=True, exist_ok=True)
    custom_voice_path.write_bytes(b"voice")
    app = create_app()
    app.state.config_factory = lambda: MochiConfig(voice=VoiceConfig(voice_pack_dir=tmp_path / "packs"))

    with TestClient(app) as client:
        response = client.post(
            "/v1/voice/voices/register-path",
            json={
                "path": str(custom_voice_path),
                "backend": "piper",
                "voice_id": "registered-piper",
                "label": "Registered Piper",
            },
        )
        assert response.status_code == 200
        payload = response.json()
        assert payload["voice"]["id"] == "registered-piper"
        assert payload["voice"]["backend"] == "piper"
        assert payload["voice"]["path"] == str(custom_voice_path)

        followup = client.get("/v1/voice/voices")

    assert followup.status_code == 200
    assert followup.json()["items"] == [
        {
            "id": "registered-piper",
            "backend": "piper",
            "path": str(custom_voice_path),
            "label": "Registered Piper",
            "source": "registered_path",
            "is_available": True,
        }
    ]


def test_voice_voices_upload_registers_uploaded_pack(tmp_path: Path) -> None:
    app = create_app()
    app.state.config_factory = lambda: MochiConfig(voice=VoiceConfig(voice_pack_dir=tmp_path / "packs"))

    with TestClient(app) as client:
        response = client.post(
            "/v1/voice/voices/upload",
            data={
                "backend": "piper",
                "voice_id": "uploaded-piper",
                "label": "Uploaded Piper",
            },
            files={"files": ("uploaded-piper.onnx", b"voice-bytes", "application/octet-stream")},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["type"] == "voice_voice_registration"
    assert payload["voice"]["id"] == "uploaded-piper"
    assert payload["voice"]["backend"] == "piper"
    assert payload["voice"]["source"] == "upload"
    assert Path(payload["voice"]["path"]).is_file()
    assert Path(payload["voice"]["path"]).read_bytes() == b"voice-bytes"


def test_voice_voices_delete_unregisters_and_removes_uploaded_pack(tmp_path: Path) -> None:
    uploaded_voice_path = tmp_path / "packs" / "browser-uploads" / "pkg" / "uploaded-piper.onnx"
    uploaded_voice_path.parent.mkdir(parents=True, exist_ok=True)
    uploaded_voice_path.write_bytes(b"voice")
    app = create_app()
    app.state.config_factory = lambda: MochiConfig(
        voice=VoiceConfig(
            voice_pack_dir=tmp_path / "packs",
            registered_tts_voices=[
                RegisteredTTSVoiceConfig(
                    id="uploaded-piper",
                    backend="piper",
                    path=uploaded_voice_path,
                    label="Uploaded Piper",
                    source="upload",
                )
            ],
        )
    )

    with TestClient(app) as client:
        response = client.request("DELETE", "/v1/voice/voices/uploaded-piper")
        assert response.status_code == 200
        payload = response.json()
        assert payload["deleted"] is True
        assert payload["removed_path"] == str(uploaded_voice_path)

        followup = client.get("/v1/voice/voices")

    assert uploaded_voice_path.exists() is False
    assert followup.status_code == 200
    assert followup.json()["items"] == []
