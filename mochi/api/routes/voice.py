"""Voice registry API routes."""

from __future__ import annotations

import shutil
import time
from pathlib import Path, PurePosixPath
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, Field

from mochi.api.server import _get_config, _maybe_await
from mochi.config.schema import RegisteredTTSVoiceConfig, VoiceConfig

from .settings import (
    TTS_VOICE_PRESETS_BY_BACKEND,
    _ensure_config_directories,
    _persist_config_if_enabled,
)

router = APIRouter(prefix="/v1/voice", tags=["voice"])


class RegisterVoicePathRequest(BaseModel):
    path: str = Field(min_length=1)
    backend: str | None = None
    voice_id: str | None = Field(default=None, min_length=1)
    label: str | None = None
    persist: bool = True
    reload_voice: bool = True


def _voice_to_payload(voice: RegisteredTTSVoiceConfig) -> dict[str, Any]:
    return {
        "id": voice.id,
        "backend": voice.backend,
        "path": str(voice.path),
        "label": voice.label,
        "source": voice.source,
        "is_available": voice.path.expanduser().exists(),
    }


@router.get("/voices")
async def list_voice_registry(request: Request) -> dict[str, Any]:
    config = await _get_config(request.app)
    return {
        "type": "voice_voices",
        "voice_pack_dir": str(config.voice.voice_pack_dir),
        "presets_by_backend": TTS_VOICE_PRESETS_BY_BACKEND,
        "items": [_voice_to_payload(voice) for voice in config.voice.registered_tts_voices],
    }


@router.post("/voices/register-path")
async def register_voice_path(request: Request, payload: RegisterVoicePathRequest) -> dict[str, Any]:
    config = await _get_config(request.app)
    source_path = Path(payload.path).expanduser()
    if not source_path.exists():
        raise HTTPException(status_code=404, detail="Voice path not found")

    voice_id = (payload.voice_id or source_path.stem or source_path.name).strip()
    if not voice_id:
        raise HTTPException(status_code=400, detail="voice_id could not be derived")
    label = payload.label.strip() if isinstance(payload.label, str) and payload.label.strip() else None

    updated_voice = RegisteredTTSVoiceConfig(
        id=voice_id,
        backend=payload.backend,
        path=source_path,
        label=label,
        source="registered_path",
    )
    updated_config = _replace_registered_voice(config.voice, updated_voice)
    await _apply_updated_voice_config(
        request,
        updated_config,
        persist=payload.persist,
        reload_voice=payload.reload_voice,
    )

    return {
        "type": "voice_voice_registration",
        "action": "register_path",
        "voice": _voice_to_payload(updated_voice),
    }


@router.post("/voices/upload")
async def upload_voice_pack(
    request: Request,
    files: Annotated[list[UploadFile], File()],
    backend: Annotated[str | None, Form()] = None,
    voice_id: Annotated[str | None, Form()] = None,
    label: Annotated[str | None, Form()] = None,
    relative_paths: Annotated[list[str] | None, Form()] = None,
    persist: Annotated[bool, Form()] = True,
    reload_voice: Annotated[bool, Form()] = True,
) -> dict[str, Any]:
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    config = await _get_config(request.app)
    upload_root = Path(config.voice.voice_pack_dir).expanduser() / "browser-uploads"
    package_root = upload_root / f"{int(time.time())}-{_safe_name(voice_id or label or 'voice-pack')}"
    package_root.mkdir(parents=True, exist_ok=True)

    saved_files: list[Path] = []
    relative_values = relative_paths or []
    for index, upload in enumerate(files):
        fallback_name = upload.filename or f"file-{index + 1}"
        relative = _safe_relative_path(
            relative_values[index] if index < len(relative_values) else fallback_name,
            fallback_name,
        )
        target = package_root / relative
        await _write_upload_file(upload, target)
        saved_files.append(target)

    imported_path = saved_files[0] if len(saved_files) == 1 else package_root
    resolved_voice_id = (voice_id or imported_path.stem or imported_path.name).strip()
    if not resolved_voice_id:
        raise HTTPException(status_code=400, detail="voice_id could not be derived")

    updated_voice = RegisteredTTSVoiceConfig(
        id=resolved_voice_id,
        backend=backend,
        path=imported_path,
        label=label.strip() if isinstance(label, str) and label.strip() else None,
        source="upload",
    )
    updated_config = _replace_registered_voice(config.voice, updated_voice)
    await _apply_updated_voice_config(
        request,
        updated_config,
        persist=persist,
        reload_voice=reload_voice,
    )

    return {
        "type": "voice_voice_registration",
        "action": "upload",
        "voice": _voice_to_payload(updated_voice),
    }


@router.delete("/voices/{voice_id}")
async def delete_registered_voice(
    request: Request,
    voice_id: str,
    delete_files: bool = Query(default=True),
    persist: bool = Query(default=True),
    reload_voice: bool = Query(default=True),
) -> dict[str, Any]:
    config = await _get_config(request.app)
    existing = next((voice for voice in config.voice.registered_tts_voices if voice.id == voice_id), None)
    if existing is None:
        raise HTTPException(status_code=404, detail="Voice not found")

    removed_path: str | None = None
    if delete_files and existing.source == "upload":
        candidate = existing.path.expanduser()
        if _is_within_voice_pack_dir(candidate, config.voice.voice_pack_dir):
            if candidate.is_dir():
                shutil.rmtree(candidate, ignore_errors=True)
            elif candidate.exists():
                candidate.unlink(missing_ok=True)
            removed_path = str(candidate)

    remaining = [voice for voice in config.voice.registered_tts_voices if voice.id != voice_id]
    next_voice = config.voice.model_copy(update={"registered_tts_voices": remaining})
    await _apply_updated_voice_config(
        request,
        next_voice,
        persist=persist,
        reload_voice=reload_voice,
    )

    return {
        "type": "voice_voice_delete",
        "deleted": True,
        "voice_id": voice_id,
        "removed_path": removed_path,
    }


def _replace_registered_voice(
    current: VoiceConfig,
    voice: RegisteredTTSVoiceConfig,
) -> VoiceConfig:
    remaining = [item for item in current.registered_tts_voices if item.id != voice.id]
    return current.model_copy(update={"registered_tts_voices": [voice, *remaining]})


async def _apply_updated_voice_config(
    request: Request,
    voice: VoiceConfig,
    *,
    persist: bool,
    reload_voice: bool,
) -> None:
    current = await _get_config(request.app)
    updated = current.model_copy(update={"voice": voice})
    _ensure_config_directories(updated)
    request.app.state.config = updated
    engine = getattr(request.app.state, "engine", None)
    if engine is not None:
        apply_config = getattr(engine, "apply_config", None)
        if callable(apply_config):
            await _maybe_await(apply_config(updated, reload_voice=reload_voice))
    _persist_config_if_enabled(request, updated, persist)


async def _write_upload_file(upload: UploadFile, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)
    await upload.close()


def _safe_name(value: str) -> str:
    cleaned = "".join(ch for ch in value if ch.isalnum() or ch in {"-", "_", "."}).strip(".-_")
    return cleaned[:80] or "voice-pack"


def _safe_relative_path(value: str, fallback_name: str) -> Path:
    raw = (value or fallback_name).replace("\\", "/").strip("/")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        pure = PurePosixPath(Path(fallback_name).name)
    return Path(*pure.parts)


def _is_within_voice_pack_dir(candidate: Path, root: Path) -> bool:
    try:
        candidate_resolved = candidate.resolve(strict=False)
        root_resolved = root.expanduser().resolve(strict=False)
    except OSError:
        return False
    try:
        candidate_resolved.relative_to(root_resolved)
    except ValueError:
        return False
    return True
