"""AgentEngine Phase 4（語音編排）測試。"""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from mochi.agents.engine import AgentEngine
from mochi.agents.events import FinalAnswerEvent
from mochi.agents.invocation import AgentInvocationResult
from mochi.config.schema import MochiConfig
from mochi.voice.events import (
    AgentFinalTextEvent,
    SynthesizedAudioChunkEvent,
    TranscriptionEvent,
    VoiceStageEvent,
)


class _FakeVAD:
    def detect_speech(self, audio: bytes) -> bool:  # noqa: ARG002
        return True


class _FakeSTT:
    def transcribe(self, audio: bytes) -> str:  # noqa: ARG002
        return "請說明今天任務"


class _FakeTTS:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def synthesize(self, text: str) -> AsyncIterator[bytes]:
        self.calls.append(text)
        yield b"chunk-1"
        yield b"chunk-2"


class _FakeManagedVLLMRuntimeManager:
    def __init__(self, *, base_url: str = "http://127.0.0.1:8200/v1") -> None:
        self.base_url = base_url
        self.start_calls: list[dict[str, Any]] = []

    async def start(self, **kwargs: Any) -> dict[str, Any]:
        self.start_calls.append(dict(kwargs))
        return {
            "state": "running",
            "running": True,
            "launch_mode": "managed",
            "active_model_id": kwargs.get("model_id"),
            "active_model_spec": kwargs.get("model_spec"),
            "base_url": self.base_url,
            "message": "ready",
        }


@pytest.mark.asyncio
async def test_engine_voice_chat_lazy_session_and_forwards_session_id(tmp_path: Path) -> None:
    """voice_chat 應 lazy 建立 VoiceSession，並依 session_id 隔離。"""
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
        }
    )

    tts = _FakeTTS()
    engine = AgentEngine(
        config,
        voice_vad=_FakeVAD(),
        voice_stt=_FakeSTT(),
        voice_tts=tts,
    )

    seen_calls: list[tuple[str, str | None]] = []

    async def fake_chat(message: str, session_id: str | None = None) -> AsyncIterator[FinalAnswerEvent]:
        seen_calls.append((message, session_id))
        yield FinalAnswerEvent(content="收到，這是今天任務摘要。")

    engine.chat = fake_chat  # type: ignore[method-assign]

    default_session = await engine.get_or_create_voice_session()
    same_default_session = await engine.get_or_create_voice_session(session_id=None)
    voice_session_s1 = await engine.get_or_create_voice_session(session_id="voice-s1")
    events = [event async for event in engine.voice_chat(b"audio-bytes", session_id="voice-s1")]
    same_session_s1 = await engine.get_or_create_voice_session(session_id="voice-s1")

    assert isinstance(events[0], VoiceStageEvent)
    assert events[0].stage == "transcribing"
    assert isinstance(events[1], TranscriptionEvent)
    assert isinstance(events[2], VoiceStageEvent)
    assert events[2].stage == "thinking"
    assert isinstance(events[3], AgentFinalTextEvent)
    assert events[3].text == "收到，這是今天任務摘要。"
    assert isinstance(events[4], VoiceStageEvent)
    assert events[4].stage == "synthesizing"
    assert isinstance(events[5], SynthesizedAudioChunkEvent)
    assert isinstance(events[6], SynthesizedAudioChunkEvent)
    assert seen_calls == [("請說明今天任務", "voice-s1")]
    assert tts.calls == ["收到，這是今天任務摘要。"]
    assert same_default_session is default_session
    assert same_session_s1 is voice_session_s1
    assert voice_session_s1 is not default_session

    _ = [event async for event in engine.voice_chat(b"audio-bytes-2", session_id="voice-s2")]
    voice_session_s2 = await engine.get_or_create_voice_session(session_id="voice-s2")
    assert await engine.get_or_create_voice_session() is default_session
    assert voice_session_s2 is not voice_session_s1
    assert seen_calls[-1] == ("請說明今天任務", "voice-s2")


@pytest.mark.asyncio
async def test_engine_voice_sessions_use_distinct_vad_instances_per_session(tmp_path: Path) -> None:
    """不同 session 的 VoiceSession 應持有不同 VAD，但可共享 STT/TTS。"""
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
        }
    )
    shared_stt = _FakeSTT()
    shared_tts = _FakeTTS()
    injected_vad = _FakeVAD()
    engine = AgentEngine(
        config,
        voice_vad=injected_vad,
        voice_stt=shared_stt,
        voice_tts=shared_tts,
    )

    default_session = await engine.get_or_create_voice_session()
    session_s1 = await engine.get_or_create_voice_session(session_id="s1")
    session_s2 = await engine.get_or_create_voice_session(session_id="s2")

    assert default_session is await engine.get_or_create_voice_session()
    assert session_s1 is await engine.get_or_create_voice_session(session_id="s1")
    assert session_s2 is await engine.get_or_create_voice_session(session_id="s2")

    assert default_session._vad is not session_s1._vad  # type: ignore[attr-defined]
    assert session_s1._vad is not session_s2._vad  # type: ignore[attr-defined]
    assert default_session._vad is not session_s2._vad  # type: ignore[attr-defined]
    assert default_session._vad is not injected_vad  # type: ignore[attr-defined]
    assert default_session._stt is shared_stt  # type: ignore[attr-defined]
    assert default_session._tts is shared_tts  # type: ignore[attr-defined]
    assert session_s1._stt is shared_stt  # type: ignore[attr-defined]
    assert session_s1._tts is shared_tts  # type: ignore[attr-defined]
    assert session_s2._stt is shared_stt  # type: ignore[attr-defined]
    assert session_s2._tts is shared_tts  # type: ignore[attr-defined]


@pytest.mark.asyncio
async def test_engine_voice_chat_can_build_components_from_voice_router(tmp_path: Path) -> None:
    """未注入 voice 元件時，應能透過 VoiceRouter lazy 建立。"""
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
        }
    )
    engine = AgentEngine(config)

    seen_calls: list[tuple[str, str | None]] = []

    async def fake_chat(message: str, session_id: str | None = None) -> AsyncIterator[FinalAnswerEvent]:
        seen_calls.append((message, session_id))
        yield FinalAnswerEvent(content="router 建立成功")

    class _FakeVoiceRouter:
        def __init__(self) -> None:
            self.closed = False
            self.create_vad_calls = 0

        async def load(self, config) -> object:  # noqa: ANN001
            return type(
                "_Runtime",
                (),
                {
                    "vad": _FakeVAD(),
                    "stt": _FakeSTT(),
                    "tts": _FakeTTS(),
                },
            )()

        def create_vad(self, config) -> _FakeVAD:  # noqa: ANN001
            self.create_vad_calls += 1
            return _FakeVAD()

        async def close(self) -> None:
            self.closed = True

    engine.chat = fake_chat  # type: ignore[method-assign]

    with patch("mochi.agents.engine.VoiceRouter", _FakeVoiceRouter):
        voice_session = await engine.get_or_create_voice_session(session_id="router-s1")
        events = [event async for event in engine.voice_chat(b"audio-router", session_id="router-s1")]

    assert voice_session is await engine.get_or_create_voice_session(session_id="router-s1")
    assert isinstance(events[0], VoiceStageEvent)
    assert events[0].stage == "transcribing"
    assert isinstance(events[1], TranscriptionEvent)
    assert isinstance(events[-1], SynthesizedAudioChunkEvent)
    assert seen_calls == [("請說明今天任務", "router-s1")]


@pytest.mark.asyncio
async def test_engine_release_voice_session_discards_cached_session(tmp_path: Path) -> None:
    """release_voice_session 應可依 session_id 釋放快取並允許重建。"""
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
        }
    )
    engine = AgentEngine(
        config,
        voice_vad=_FakeVAD(),
        voice_stt=_FakeSTT(),
        voice_tts=_FakeTTS(),
    )

    session_s1 = await engine.get_or_create_voice_session(session_id="release-s1")
    default_session = await engine.get_or_create_voice_session()

    assert await engine.release_voice_session(session_id="release-s1") is True
    assert await engine.release_voice_session(session_id="release-s1") is False

    session_s1_recreated = await engine.get_or_create_voice_session(session_id="release-s1")
    default_session_after = await engine.get_or_create_voice_session()

    assert session_s1_recreated is not session_s1
    assert default_session_after is default_session


@pytest.mark.asyncio
async def test_engine_voice_chat_uses_isolated_agent_session_when_configured(tmp_path: Path) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
            "voice": {
                "session_mode": "isolated_voice",
            },
        }
    )

    tts = _FakeTTS()
    engine = AgentEngine(
        config,
        voice_vad=_FakeVAD(),
        voice_stt=_FakeSTT(),
        voice_tts=tts,
    )

    seen_calls: list[tuple[str, str | None]] = []

    async def fake_chat(message: str, session_id: str | None = None) -> AsyncIterator[FinalAnswerEvent]:
        seen_calls.append((message, session_id))
        yield FinalAnswerEvent(content="isolated reply")

    engine.chat = fake_chat  # type: ignore[method-assign]

    _ = [event async for event in engine.voice_chat(b"audio-bytes", session_id="voice-s1")]

    assert seen_calls == [("請說明今天任務", "voice::voice-s1")]


@pytest.mark.asyncio
async def test_engine_voice_chat_uses_dedicated_configured_model_when_selected(tmp_path: Path) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
            "voice": {
                "reply_model_mode": "configured_model",
                "reply_model_id": "voice-openai",
            },
            "model_setup": {
                "configured_models": [
                    {
                        "id": "voice-openai",
                        "provider": "openai_compat",
                        "model": "gpt-4o-mini",
                        "model_spec": "https://example.invalid/v1",
                        "base_url": "https://example.invalid/v1",
                        "label": "Voice OpenAI",
                        "backend_type": "openai_compat",
                    }
                ]
            },
        }
    )
    engine = AgentEngine(
        config,
        voice_vad=_FakeVAD(),
        voice_stt=_FakeSTT(),
        voice_tts=_FakeTTS(),
    )

    class _TempBackend:
        async def close(self) -> None:
            return None

    captured: list[object | None] = []

    async def fake_invoke_shared_runtime(request) -> AgentInvocationResult:  # type: ignore[no-untyped-def]
        captured.append(request.backend_override)
        captured.append(request.session_id)
        return AgentInvocationResult(
            content="voice reply",
            events=[FinalAnswerEvent(content="voice reply")],
        )

    async def fake_acquire_voice_reply_backend() -> _TempBackend:
        return _TempBackend()

    engine._invoke_shared_runtime = fake_invoke_shared_runtime  # type: ignore[method-assign]
    engine._acquire_voice_reply_backend = fake_acquire_voice_reply_backend  # type: ignore[method-assign]

    _ = [event async for event in engine.voice_chat(b"audio-bytes", session_id="voice-s1")]

    assert isinstance(captured[0], _TempBackend)
    assert captured[1] == "voice-s1"


@pytest.mark.asyncio
async def test_engine_voice_chat_configured_model_accepts_vllm_provider(tmp_path: Path) -> None:
    """voice configured model provider=vllm 時，voice flow 應可正常路由。"""
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
            "voice": {
                "reply_model_mode": "configured_model",
                "reply_model_id": "voice-vllm",
            },
            "model_setup": {
                "configured_models": [
                    {
                        "id": "voice-vllm",
                        "provider": "vllm",
                        "model": "qwen2.5-7b-instruct",
                        "model_spec": "http://localhost:8000/v1",
                        "base_url": "http://localhost:8000/v1",
                        "label": "Voice vLLM",
                        "backend_type": "openai_compat",
                    }
                ]
            },
        }
    )
    engine = AgentEngine(
        config,
        voice_vad=_FakeVAD(),
        voice_stt=_FakeSTT(),
        voice_tts=_FakeTTS(),
    )

    class _TempBackend:
        async def close(self) -> None:
            return None

    captured: list[object | None] = []

    async def fake_invoke_shared_runtime(request) -> AgentInvocationResult:  # type: ignore[no-untyped-def]
        captured.append(request.backend_override)
        captured.append(request.session_id)
        return AgentInvocationResult(
            content="voice vllm reply",
            events=[FinalAnswerEvent(content="voice vllm reply")],
        )

    async def fake_acquire_voice_reply_backend() -> _TempBackend:
        return _TempBackend()

    engine._invoke_shared_runtime = fake_invoke_shared_runtime  # type: ignore[method-assign]
    engine._acquire_voice_reply_backend = fake_acquire_voice_reply_backend  # type: ignore[method-assign]

    _ = [event async for event in engine.voice_chat(b"audio-bytes", session_id="voice-vllm-s1")]

    assert isinstance(captured[0], _TempBackend)
    assert captured[1] == "voice-vllm-s1"


@pytest.mark.asyncio
async def test_acquire_voice_reply_backend_starts_managed_vllm_and_uses_runtime_base_url(
    tmp_path: Path,
) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
            "voice": {
                "reply_model_mode": "configured_model",
                "reply_model_id": "voice-vllm-managed",
            },
            "model_setup": {
                "configured_models": [
                    {
                        "id": "voice-vllm-managed",
                        "provider": "vllm",
                        "model": "Qwen/Qwen2.5-7B-Instruct",
                        "model_spec": "Qwen/Qwen2.5-7B-Instruct",
                        "base_url": "http://localhost:8000/v1",
                        "launch_mode": "managed",
                        "label": "Voice vLLM managed",
                        "backend_type": "openai_compat",
                    }
                ]
            },
        }
    )
    manager = _FakeManagedVLLMRuntimeManager(base_url="http://127.0.0.1:9300/v1")
    engine = AgentEngine(
        config,
        voice_vad=_FakeVAD(),
        voice_stt=_FakeSTT(),
        voice_tts=_FakeTTS(),
        vllm_runtime_manager=manager,
    )

    class _TempBackend:
        async def close(self) -> None:
            return None

    captured: dict[str, Any] = {}

    async def fake_acquire_temporary_backend(**kwargs: Any) -> _TempBackend:
        captured.update(kwargs)
        return _TempBackend()

    engine._router.acquire_temporary_backend = fake_acquire_temporary_backend  # type: ignore[method-assign]

    backend = await engine._acquire_voice_reply_backend()

    assert isinstance(backend, _TempBackend)
    assert manager.start_calls == [
        {
            "model_id": "voice-vllm-managed",
            "model_spec": "Qwen/Qwen2.5-7B-Instruct",
            "base_url": "http://localhost:8000/v1",
            "launch_mode": "managed",
            "config": config,
        }
    ]
    assert captured == {
        "model_spec": "http://127.0.0.1:9300/v1",
        "model_name": "Qwen/Qwen2.5-7B-Instruct",
        "provider": "vllm",
        "base_url": "http://127.0.0.1:9300/v1",
        "api_key": "",
        "auth_profile_id": None,
    }


@pytest.mark.asyncio
async def test_acquire_voice_reply_backend_uses_configured_vllm_api_key(
    tmp_path: Path,
) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
            "voice": {
                "reply_model_mode": "configured_model",
                "reply_model_id": "voice-vllm-managed",
            },
            "vllm": {
                "api_key": "vllm-runtime-secret",
            },
            "model_setup": {
                "configured_models": [
                    {
                        "id": "voice-vllm-managed",
                        "provider": "vllm",
                        "model": "Qwen/Qwen2.5-7B-Instruct",
                        "model_spec": "Qwen/Qwen2.5-7B-Instruct",
                        "base_url": "http://localhost:8000/v1",
                        "launch_mode": "managed",
                        "label": "Voice vLLM managed",
                        "backend_type": "openai_compat",
                    }
                ]
            },
        }
    )
    manager = _FakeManagedVLLMRuntimeManager(base_url="http://127.0.0.1:9300/v1")
    engine = AgentEngine(
        config,
        voice_vad=_FakeVAD(),
        voice_stt=_FakeSTT(),
        voice_tts=_FakeTTS(),
        vllm_runtime_manager=manager,
    )

    class _TempBackend:
        async def close(self) -> None:
            return None

    captured: dict[str, Any] = {}

    async def fake_acquire_temporary_backend(**kwargs: Any) -> _TempBackend:
        captured.update(kwargs)
        return _TempBackend()

    engine._router.acquire_temporary_backend = fake_acquire_temporary_backend  # type: ignore[method-assign]

    backend = await engine._acquire_voice_reply_backend()

    assert isinstance(backend, _TempBackend)
    assert captured["api_key"] == "vllm-runtime-secret"


@pytest.mark.asyncio
async def test_acquire_voice_reply_backend_rejects_managed_vllm_gguf_model_spec(
    tmp_path: Path,
) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
            "voice": {
                "reply_model_mode": "configured_model",
                "reply_model_id": "voice-vllm-managed-gguf",
            },
            "model_setup": {
                "configured_models": [
                    {
                        "id": "voice-vllm-managed-gguf",
                        "provider": "vllm",
                        "model": "demo.gguf",
                        "model_spec": "demo.gguf",
                        "base_url": "http://localhost:8000/v1",
                        "launch_mode": "managed",
                        "label": "Voice vLLM managed gguf",
                        "backend_type": "openai_compat",
                    }
                ]
            },
        }
    )
    manager = _FakeManagedVLLMRuntimeManager()
    engine = AgentEngine(
        config,
        voice_vad=_FakeVAD(),
        voice_stt=_FakeSTT(),
        voice_tts=_FakeTTS(),
        vllm_runtime_manager=manager,
    )

    with pytest.raises(RuntimeError, match=r"\.gguf"):
        await engine._acquire_voice_reply_backend()

    assert manager.start_calls == []


@pytest.mark.asyncio
async def test_acquire_voice_reply_backend_keeps_external_vllm_behavior(
    tmp_path: Path,
) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
            "voice": {
                "reply_model_mode": "configured_model",
                "reply_model_id": "voice-vllm-external",
            },
            "model_setup": {
                "configured_models": [
                    {
                        "id": "voice-vllm-external",
                        "provider": "vllm",
                        "model": "qwen2.5-7b-instruct",
                        "model_spec": "http://localhost:8000/v1",
                        "base_url": "http://localhost:8000/v1",
                        "launch_mode": "external",
                        "label": "Voice vLLM external",
                        "backend_type": "openai_compat",
                    }
                ]
            },
        }
    )
    manager = _FakeManagedVLLMRuntimeManager()
    engine = AgentEngine(
        config,
        voice_vad=_FakeVAD(),
        voice_stt=_FakeSTT(),
        voice_tts=_FakeTTS(),
        vllm_runtime_manager=manager,
    )

    class _TempBackend:
        async def close(self) -> None:
            return None

    captured: dict[str, Any] = {}

    async def fake_acquire_temporary_backend(**kwargs: Any) -> _TempBackend:
        captured.update(kwargs)
        return _TempBackend()

    engine._router.acquire_temporary_backend = fake_acquire_temporary_backend  # type: ignore[method-assign]

    backend = await engine._acquire_voice_reply_backend()

    assert isinstance(backend, _TempBackend)
    assert manager.start_calls == []
    assert captured == {
        "model_spec": "http://localhost:8000/v1",
        "model_name": "qwen2.5-7b-instruct",
        "provider": "vllm",
        "base_url": "http://localhost:8000/v1",
        "api_key": "",
        "auth_profile_id": None,
    }


@pytest.mark.asyncio
async def test_engine_close_stops_owned_vllm_runtime_manager(tmp_path: Path) -> None:
    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
        }
    )
    manager = _FakeManagedVLLMRuntimeManager()
    stopped = {"value": False}

    async def fake_stop() -> dict[str, str]:
        stopped["value"] = True
        return {"state": "stopped"}

    manager.stop = fake_stop  # type: ignore[method-assign]
    engine = AgentEngine(
        config,
        voice_vad=_FakeVAD(),
        voice_stt=_FakeSTT(),
        voice_tts=_FakeTTS(),
        vllm_runtime_manager=manager,
    )

    await engine.close()

    assert stopped["value"] is True
