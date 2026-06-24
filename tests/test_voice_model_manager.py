"""Voice model manager 測試。"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from mochi.config.schema import VoiceConfig
from mochi.voice.model_manager import (
    ensure_model_available,
    ensure_tts_runtime_available,
    model_filename,
    resolve_bounded_stt_runtime_spec,
    resolve_faster_whisper_source,
    resolve_model_target,
    resolve_qwen_model_target,
    resolve_qwen_repo,
)


def test_model_filename_and_resolve_target(tmp_path: Path) -> None:
    """應能解析預設 Whisper 模型檔名與目標路徑。"""
    assert model_filename("medium") == "medium.pt"
    assert resolve_model_target("medium", tmp_path) == tmp_path / "medium.pt"
    assert resolve_model_target("medium", tmp_path, tmp_path / "custom.pt") == tmp_path / "custom.pt"


def test_resolve_qwen_repo_and_target(tmp_path: Path) -> None:
    """Qwen ASR repo 與目標目錄應可正確解析。"""
    cfg = {
        "model": "qwen3-asr-0.6b",
        "model_cache_dir": str(tmp_path),
    }
    assert resolve_qwen_repo(cfg) == "Qwen/Qwen3-ASR-0.6B"
    assert resolve_qwen_model_target(cfg) == tmp_path / "qwen-asr" / "Qwen__Qwen3-ASR-0.6B"


def test_resolve_faster_whisper_source_prefers_existing_local_path(tmp_path: Path) -> None:
    """faster-whisper 應優先使用既有本地路徑。"""
    local_dir = tmp_path / "fw-small"
    local_dir.mkdir()

    resolved, is_local = resolve_faster_whisper_source(str(local_dir), tmp_path / "cache")

    assert resolved == str(local_dir)
    assert is_local is True


def test_resolve_faster_whisper_source_can_use_cached_directory(tmp_path: Path) -> None:
    """cache 下同名資料夾存在時，應視為本地模型來源。"""
    cache_dir = tmp_path / "models"
    cached_dir = cache_dir / "small"
    cached_dir.mkdir(parents=True)

    resolved, is_local = resolve_faster_whisper_source("small", cache_dir)

    assert resolved == str(cached_dir)
    assert is_local is True


def test_resolve_faster_whisper_source_keeps_model_name_when_no_local_path(tmp_path: Path) -> None:
    """沒有本地路徑時，應保留原始 model name 給 faster-whisper 處理。"""
    resolved, is_local = resolve_faster_whisper_source("small", tmp_path / "cache")

    assert resolved == "small"
    assert is_local is False


def test_resolve_bounded_stt_runtime_spec_prefers_explicit_model_path(tmp_path: Path) -> None:
    """bounded runtime spec 應優先採用存在的 stt_model_path。"""
    local_path = tmp_path / "model-dir"
    local_path.mkdir()

    spec = resolve_bounded_stt_runtime_spec(
        stt_backend="faster-whisper",
        stt_model="small",
        stt_model_cache_dir=tmp_path / "cache",
        stt_model_path=local_path,
    )

    assert spec.model_source == str(local_path)
    assert spec.uses_local_model_source is True


def test_resolve_bounded_stt_runtime_spec_uses_cache_directory_when_available(tmp_path: Path) -> None:
    """bounded runtime spec 應能解析 cache 內已存在模型目錄。"""
    cache_dir = tmp_path / "models"
    (cache_dir / "small").mkdir(parents=True)

    spec = resolve_bounded_stt_runtime_spec(
        stt_backend="auto",
        stt_model="small",
        stt_model_cache_dir=cache_dir,
    )

    assert spec.model_source == str(cache_dir / "small")
    assert spec.uses_local_model_source is True


def test_resolve_bounded_stt_runtime_spec_supports_whisper_cpp_model_path(tmp_path: Path) -> None:
    """whisper-cpp backend 應可回傳顯式 model_path 作為 model_source。"""
    model_path = tmp_path / "ggml-base.bin"

    spec = resolve_bounded_stt_runtime_spec(
        stt_backend="whisper-cpp",
        stt_model="base",
        stt_model_path=model_path,
    )

    assert spec.backend == "whisper-cpp"
    assert spec.model_source == str(model_path)
    assert spec.uses_local_model_source is False


def test_resolve_bounded_stt_runtime_spec_supports_external_api_runtime_source() -> None:
    """external-api backend 應以 API base URL 作為 bounded runtime source。"""
    spec = resolve_bounded_stt_runtime_spec(
        stt_backend="external-api",
        stt_model="whisper-1",
        stt_openai_base_url="https://api.example.com/v1",
    )

    assert spec.backend == "external-api"
    assert spec.requested_model == "whisper-1"
    assert spec.model_source == "https://api.example.com/v1"
    assert spec.uses_local_model_source is False


def test_resolve_bounded_stt_runtime_spec_supports_openai_whisper_cached_directory(
    tmp_path: Path,
) -> None:
    """openai-whisper backend 應可沿用 cache 內現有模型目錄。"""
    cache_dir = tmp_path / "models"
    (cache_dir / "base").mkdir(parents=True)

    spec = resolve_bounded_stt_runtime_spec(
        stt_backend="openai-whisper",
        stt_model="base",
        stt_model_cache_dir=cache_dir,
    )

    assert spec.backend == "openai-whisper"
    assert spec.model_source == str(cache_dir / "base")
    assert spec.uses_local_model_source is True


def test_resolve_bounded_stt_runtime_spec_supports_qwen_repo_alias(tmp_path: Path) -> None:
    """qwen-asr backend 應可把短別名解析為 repo id。"""
    spec = resolve_bounded_stt_runtime_spec(
        stt_backend="qwen-asr",
        stt_model="qwen3-asr-0.6b",
        stt_model_cache_dir=tmp_path,
    )

    assert spec.backend == "qwen-asr"
    assert spec.model_source == "Qwen/Qwen3-ASR-0.6B"
    assert spec.uses_local_model_source is False


def test_resolve_bounded_stt_runtime_spec_supports_vosk_local_model_path(tmp_path: Path) -> None:
    """vosk backend 應可保留既有本地模型路徑。"""
    model_dir = tmp_path / "vosk-small"
    model_dir.mkdir()

    spec = resolve_bounded_stt_runtime_spec(
        stt_backend="vosk",
        stt_model=str(model_dir),
    )

    assert spec.backend == "vosk"
    assert spec.model_source == str(model_dir)
    assert spec.uses_local_model_source is True


@pytest.mark.asyncio
async def test_ensure_model_available_sets_model_path_when_file_exists(tmp_path: Path) -> None:
    """模型已存在時應只補回 model_path。"""
    cache_dir = tmp_path / "models"
    cache_dir.mkdir()
    target = cache_dir / "small.pt"
    target.write_bytes(b"ready")

    updated = await ensure_model_available(
        {"model": "small", "model_cache_dir": str(cache_dir)},
    )

    assert updated["model_path"] == str(target)


@pytest.mark.asyncio
async def test_ensure_model_available_can_use_injected_downloader(tmp_path: Path) -> None:
    """缺模型時應可透過注入 downloader 建立檔案。"""
    messages: list[str] = []
    cache_dir = tmp_path / "models"

    async def notify(message: str) -> None:
        messages.append(message)

    def fake_downloader(model_name: str, model_url: str, target: Path) -> None:  # noqa: ARG001
        target.write_bytes(b"downloaded")

    updated = await ensure_model_available(
        {"model": "base", "model_cache_dir": str(cache_dir)},
        notify_cb=notify,
        download_fn=fake_downloader,
        download_lock=asyncio.Lock(),
    )

    assert Path(updated["model_path"]).read_bytes() == b"downloaded"
    assert any("Downloading Whisper model: base" in message for message in messages)


@pytest.mark.asyncio
async def test_ensure_tts_runtime_available_reports_ready_for_local_backend() -> None:
    """本地 TTS 後端預熱成功時應回報 ready，而不是等到第一次合成才發現問題。"""

    class _FakeTTS:
        def __init__(self) -> None:
            self.ensure_ready_calls = 0
            self.close_calls = 0

        async def ensure_ready(self) -> None:
            self.ensure_ready_calls += 1

        async def close(self) -> None:
            self.close_calls += 1

    fake_tts = _FakeTTS()

    result = await ensure_tts_runtime_available(
        VoiceConfig(tts_backend="kokoro-tts", tts_voice="af_heart"),
        tts_factory=lambda config: fake_tts,  # noqa: ARG005
    )

    assert result == {
        "requested": True,
        "backend": "kokoro-tts",
        "status": "ready",
    }
    assert fake_tts.ensure_ready_calls == 1
    assert fake_tts.close_calls == 1


@pytest.mark.asyncio
async def test_ensure_tts_runtime_available_returns_prepare_failed_instead_of_raising() -> None:
    """本地 TTS 自動下載/初始化失敗時應回報錯誤資訊，而不是直接 crash。"""

    class _BrokenTTS:
        async def ensure_ready(self) -> None:
            raise RuntimeError("download failed")

        async def close(self) -> None:
            return None

    result = await ensure_tts_runtime_available(
        VoiceConfig(tts_backend="coqui-tts", tts_model="tts_models/multilingual/multi-dataset/xtts_v2"),
        tts_factory=lambda config: _BrokenTTS(),  # noqa: ARG005
    )

    assert result["requested"] is True
    assert result["backend"] == "coqui-tts"
    assert result["status"] == "prepare_failed"
    assert "download failed" in str(result["error"])
