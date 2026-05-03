"""語音模型管理工具（路徑解析與輕量確保邏輯）。"""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Attribution:
# Adapted from /mnt/g/_python/STT&TTS/backend/model_manager.py
# Scope intentionally bounded for Mochi: path resolution, model target selection,
# and lightweight ensure-model behavior with injectable download functions.

WHISPER_MODEL_URLS: dict[str, str] = {
    "tiny": "https://openaipublic.azureedge.net/main/whisper/models/65147644a518d12f04e32d6f3b26facc3f8dd46e5390956a9424a650c0ce22b9/tiny.pt",
    "base": "https://openaipublic.azureedge.net/main/whisper/models/ed3a0b6b1c0edf879ad9b11b1af5a0e6ab5db9205f891f668f8b0e6c6326e34e/base.pt",
    "small": "https://openaipublic.azureedge.net/main/whisper/models/9ecf779972d90ba49c06d968637d720dd632c55bbf19d441fb42bf17a411e794/small.pt",
    "medium": "https://openaipublic.azureedge.net/main/whisper/models/345ae4da62f9b3d59415adc60127b97c714f32e89e936602e85993674d08dcb1/medium.pt",
    "large-v3": "https://openaipublic.azureedge.net/main/whisper/models/e5b1a55b89c1367dacf97e3e19bfd829a01529dbfdeefa8caeb59b3f1b81dadb/large-v3.pt",
}

QWEN_ASR_MODEL_REPOS: dict[str, str] = {
    "qwen3-asr-0.6b": "Qwen/Qwen3-ASR-0.6B",
    "qwen3-asr-1.7b": "Qwen/Qwen3-ASR-1.7B",
}

NotifyCallback = Callable[[str], object]
WhisperDownloadFn = Callable[[str, str, Path], object]
QwenSnapshotFn = Callable[[str, Path], object]


@dataclass(slots=True, frozen=True)
class STTRuntimeSpec:
    """STT runtime 解析結果（供 router/status 使用）。"""

    backend: str
    requested_model: str
    model_source: str
    uses_local_model_source: bool

    def to_dict(self) -> dict[str, Any]:
        """轉為可序列化 dict。"""
        return {
            "backend": self.backend,
            "requested_model": self.requested_model,
            "model_source": self.model_source,
            "uses_local_model_source": self.uses_local_model_source,
        }


def model_filename(model_size: str) -> str:
    """依模型名稱產生 Whisper 權重檔名。"""
    return f"{model_size}.pt"


def resolve_model_target(
    model_name: str,
    model_cache_dir: str | Path | None,
    model_path: str | Path | None = None,
) -> Path | None:
    """解析模型目標位置，`model_path` 優先。"""
    if model_path:
        return Path(model_path)
    if not model_cache_dir:
        return None
    return Path(model_cache_dir) / model_filename(model_name)


def resolve_faster_whisper_source(
    model_name: str,
    model_cache_dir: str | Path | None = None,
) -> tuple[str, bool]:
    """解析 faster-whisper 的模型來源。

    優先順序：
    1. 若 `model_name` 本身是存在的本地路徑，直接使用該路徑
    2. 若 cache 目錄下存在同名資料夾，視為已快取的本地模型
    3. 否則回傳原始 model name，交由 faster-whisper 自行解析/下載
    """
    cleaned = model_name.strip()
    if not cleaned:
        return cleaned, False

    explicit_path = Path(cleaned).expanduser()
    if explicit_path.exists():
        return str(explicit_path), True

    if model_cache_dir:
        cached_dir = Path(model_cache_dir).expanduser() / cleaned
        if cached_dir.is_dir():
            return str(cached_dir), True

    return cleaned, False


def resolve_bounded_stt_runtime_spec(
    *,
    stt_backend: str,
    stt_model: str,
    stt_model_cache_dir: str | Path | None = None,
    stt_model_path: str | Path | None = None,
    stt_openai_base_url: str | None = None,
) -> STTRuntimeSpec:
    """解析 Phase 4 STT runtime 來源。"""
    backend = _normalize_backend_name(stt_backend)
    requested_model = str(stt_model).strip()

    if backend in {"auto", "faster-whisper", "whisper-cpp"} and stt_model_path:
        explicit_path = Path(stt_model_path).expanduser()
        if explicit_path.exists():
            return STTRuntimeSpec(
                backend=backend,
                requested_model=requested_model,
                model_source=str(explicit_path),
                uses_local_model_source=True,
            )

    if backend in {"auto", "faster-whisper"}:
        model_source, uses_local_model_source = resolve_faster_whisper_source(
            requested_model,
            stt_model_cache_dir,
        )
        return STTRuntimeSpec(
            backend=backend,
            requested_model=requested_model,
            model_source=model_source,
            uses_local_model_source=uses_local_model_source,
        )
    if backend == "whisper-cpp":
        if stt_model_path:
            explicit_path = Path(stt_model_path).expanduser()
            return STTRuntimeSpec(
                backend=backend,
                requested_model=requested_model,
                model_source=str(explicit_path),
                uses_local_model_source=explicit_path.exists(),
            )
        return STTRuntimeSpec(
            backend=backend,
            requested_model=requested_model,
            model_source=requested_model,
            uses_local_model_source=False,
        )
    if backend == "openai-api":
        return STTRuntimeSpec(
            backend=backend,
            requested_model=requested_model,
            model_source=str(stt_openai_base_url or "").strip(),
            uses_local_model_source=False,
        )
    if backend == "openai-whisper":
        model_source, uses_local_model_source = resolve_faster_whisper_source(
            requested_model,
            stt_model_cache_dir,
        )
        return STTRuntimeSpec(
            backend=backend,
            requested_model=requested_model,
            model_source=model_source,
            uses_local_model_source=uses_local_model_source,
        )
    if backend == "qwen-asr":
        qwen_cfg: dict[str, Any] = {
            "model": requested_model,
            "model_cache_dir": str(stt_model_cache_dir) if stt_model_cache_dir else None,
        }
        target = resolve_qwen_model_target(qwen_cfg)
        if target is not None and target.exists():
            return STTRuntimeSpec(
                backend=backend,
                requested_model=requested_model,
                model_source=str(target),
                uses_local_model_source=True,
            )
        repo = resolve_qwen_repo(qwen_cfg)
        return STTRuntimeSpec(
            backend=backend,
            requested_model=requested_model,
            model_source=repo or requested_model,
            uses_local_model_source=False,
        )
    if backend in {"vosk", "whisperlivekit"}:
        candidate = _resolve_existing_model_source(requested_model, stt_model_path)
        return STTRuntimeSpec(
            backend=backend,
            requested_model=requested_model,
            model_source=candidate[0],
            uses_local_model_source=candidate[1],
        )

    return STTRuntimeSpec(
        backend=backend,
        requested_model=requested_model,
        model_source=requested_model,
        uses_local_model_source=False,
    )


def resolve_stt_model_target(stt_cfg: Mapping[str, Any]) -> Path | None:
    """解析 STT（Whisper）模型目標檔案路徑。"""
    model_name = str(stt_cfg.get("model") or "").strip()
    if not model_name:
        return None
    return resolve_model_target(
        model_name=model_name,
        model_cache_dir=stt_cfg.get("model_cache_dir"),
        model_path=stt_cfg.get("model_path"),
    )


def resolve_qwen_repo(stt_cfg: Mapping[str, Any]) -> str | None:
    """解析 Qwen ASR repo id。"""
    qwen_cfg = stt_cfg.get("qwen_asr", {})
    if isinstance(qwen_cfg, Mapping):
        configured = str(qwen_cfg.get("model") or "").strip()
        if configured:
            return configured

    model_name = str(stt_cfg.get("model") or "").strip()
    if not model_name:
        return None
    lower_name = model_name.lower()
    if lower_name in QWEN_ASR_MODEL_REPOS:
        return QWEN_ASR_MODEL_REPOS[lower_name]
    if lower_name.startswith("qwen/"):
        return model_name
    return None


def resolve_qwen_model_target(stt_cfg: Mapping[str, Any]) -> Path | None:
    """解析 Qwen ASR 模型目錄。"""
    qwen_cfg = stt_cfg.get("qwen_asr", {})
    if not isinstance(qwen_cfg, Mapping):
        qwen_cfg = {}

    model_dir = str(qwen_cfg.get("model_dir") or "").strip()
    if model_dir:
        return Path(model_dir)

    cache_dir = str(qwen_cfg.get("model_cache_dir") or stt_cfg.get("model_cache_dir") or "").strip()
    if not cache_dir:
        return None

    repo = resolve_qwen_repo(stt_cfg)
    if not repo:
        return None
    return Path(cache_dir) / "qwen-asr" / repo.replace("/", "__")


async def ensure_model_available(
    stt_cfg: Mapping[str, Any],
    *,
    notify_cb: NotifyCallback | None = None,
    download_fn: WhisperDownloadFn | None = None,
    download_lock: asyncio.Lock | None = None,
) -> dict[str, Any]:
    """確保 Whisper 模型可用。

    預設不做任何網路下載；僅在提供 `download_fn` 時才會觸發下載。
    """
    updated = dict(stt_cfg)
    model_name = str(updated.get("model") or "").strip()
    if not model_name:
        return updated

    model_url = WHISPER_MODEL_URLS.get(model_name.lower())
    if not model_url:
        return updated

    target = resolve_stt_model_target(updated)
    if target is None:
        return updated
    if target.exists():
        updated["model_path"] = str(target)
        return updated

    if download_fn is None:
        return updated

    target.parent.mkdir(parents=True, exist_ok=True)
    lock = download_lock or asyncio.Lock()
    async with lock:
        if not target.exists():
            await _maybe_call(notify_cb, f"Downloading Whisper model: {model_name}")
            await _maybe_call_download(download_fn, model_name, model_url, target)
            await _maybe_call(notify_cb, f"Model downloaded: {model_name}")

    if target.exists():
        updated["model_path"] = str(target)
    return updated


async def ensure_qwen_model_available(
    stt_cfg: Mapping[str, Any],
    *,
    notify_cb: NotifyCallback | None = None,
    snapshot_fn: QwenSnapshotFn | None = None,
    download_lock: asyncio.Lock | None = None,
) -> dict[str, Any]:
    """確保 Qwen ASR 模型可用。

    預設不做任何網路下載；僅在提供 `snapshot_fn` 時才會觸發下載。
    """
    updated = dict(stt_cfg)
    repo = resolve_qwen_repo(updated)
    if not repo:
        return updated

    target = resolve_qwen_model_target(updated)
    if target is None:
        return updated
    if _qwen_model_present(target):
        return _with_qwen_model_dir(updated, target)

    if snapshot_fn is None:
        return updated

    target.mkdir(parents=True, exist_ok=True)
    lock = download_lock or asyncio.Lock()
    async with lock:
        if not _qwen_model_present(target):
            await _maybe_call(notify_cb, f"Downloading Qwen3-ASR model: {repo}")
            await _maybe_call_snapshot(snapshot_fn, repo, target)
            await _maybe_call(notify_cb, f"Qwen3-ASR model ready: {repo}")

    if _qwen_model_present(target):
        return _with_qwen_model_dir(updated, target)
    return updated


def _with_qwen_model_dir(stt_cfg: dict[str, Any], target: Path) -> dict[str, Any]:
    qwen_cfg_raw = stt_cfg.get("qwen_asr", {})
    qwen_cfg = dict(qwen_cfg_raw) if isinstance(qwen_cfg_raw, Mapping) else {}
    qwen_cfg["model_dir"] = str(target)
    updated = dict(stt_cfg)
    updated["qwen_asr"] = qwen_cfg
    return updated


def _qwen_model_present(target: Path) -> bool:
    if not target.exists() or not target.is_dir():
        return False
    if (target / "config.json").exists():
        return True
    for pattern in ("*.safetensors", "*.bin"):
        if any(target.glob(pattern)):
            return True
    return any(target.iterdir())


async def _maybe_call(callback: NotifyCallback | None, message: str) -> None:
    if callback is None:
        return
    result = callback(message)
    if inspect.isawaitable(result):
        await result


async def _maybe_call_download(
    callback: WhisperDownloadFn,
    model_name: str,
    model_url: str,
    target: Path,
) -> None:
    if inspect.iscoroutinefunction(callback):
        await callback(model_name, model_url, target)
        return
    await asyncio.to_thread(callback, model_name, model_url, target)


async def _maybe_call_snapshot(
    callback: QwenSnapshotFn,
    repo: str,
    target: Path,
) -> None:
    if inspect.iscoroutinefunction(callback):
        await callback(repo, target)
        return
    await asyncio.to_thread(callback, repo, target)


def _normalize_backend_name(name: str) -> str:
    return name.strip().lower().replace("_", "-")


def _resolve_existing_model_source(
    model_name: str,
    model_path: str | Path | None = None,
) -> tuple[str, bool]:
    if model_path:
        explicit_path = Path(model_path).expanduser()
        if explicit_path.exists():
            return str(explicit_path), True

    if model_name:
        model_candidate = Path(model_name).expanduser()
        if model_candidate.exists():
            return str(model_candidate), True

    return model_name, False
