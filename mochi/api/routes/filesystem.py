"""Filesystem metadata API routes。"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path, PurePosixPath
from string import ascii_uppercase
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, Query, Request, UploadFile

from mochi.api.server import _get_config  # pyright: ignore[reportPrivateUsage]

router = APIRouter(prefix="/v1/filesystem", tags=["filesystem"])
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^([A-Za-z]):[\\/]*(.*)$")
_SAFE_PACKAGE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _path_from_client(value: str) -> Path:
    """將前端傳入路徑轉成後端可理解的 Path。

    WSL/Linux 後端常會收到 Windows 形式 `H:\\...`；若 `/mnt/h` 存在，
    轉成 `/mnt/h/...` 以符合後端實際檔案系統。
    """
    raw = value.strip()
    match = _WINDOWS_ABSOLUTE_PATH_RE.match(raw)
    if match and os.name != "nt":
        drive = match.group(1).lower()
        remainder = match.group(2).replace("\\", "/").strip("/")
        mount_root = Path("/mnt") / drive
        if mount_root.exists():
            return mount_root / remainder if remainder else mount_root
    return Path(raw).expanduser()


def _add_root(
    items: list[dict[str, str]],
    seen: set[str],
    *,
    name: str,
    path: Path,
) -> None:
    """加入 root 候選，並依字串路徑去重。"""
    normalized = str(path.expanduser())
    if normalized in seen:
        return
    seen.add(normalized)
    items.append({"name": name, "path": normalized})


def _windows_drive_roots() -> list[Path]:
    """Windows 平台列舉可用磁碟根目錄。"""
    if os.name != "nt":
        return []

    roots: list[Path] = []
    for drive in ascii_uppercase:
        candidate = Path(f"{drive}:\\")
        try:
            if candidate.exists():
                roots.append(candidate)
        except OSError:
            continue
    return roots


def _coerce_browse_directory(path: str) -> tuple[Path, Path | None]:
    """將使用者輸入路徑轉成可列出的資料夾。

    若輸入是檔案、或是尚未存在的檔案路徑，改列 parent directory。
    回傳 `(directory_to_list, selected_path)`；`selected_path` 代表原本使用者
    指向的檔案/不存在路徑，可供 UI 未來做高亮。
    """
    requested = _path_from_client(path)
    if requested.exists():
        if requested.is_dir():
            return requested, None
        parent = requested.parent
        if parent.exists() and parent.is_dir():
            return parent, requested
        raise HTTPException(status_code=400, detail="Path is not a directory")

    parent = requested.parent
    if str(parent) in {"", "."}:
        parent = Path.cwd()
    if parent.exists() and parent.is_dir():
        return parent, requested

    raise HTTPException(status_code=404, detail="Path not found")


def _safe_package_name(value: str | None) -> str:
    raw = (value or "upload").strip() or "upload"
    sanitized = _SAFE_PACKAGE_RE.sub("-", raw).strip(".-")
    return sanitized[:80] or "upload"


def _safe_relative_path(value: str, fallback_name: str) -> Path:
    raw = (value or fallback_name).replace("\\", "/").strip("/")
    pure = PurePosixPath(raw)
    if pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts):
        pure = PurePosixPath(Path(fallback_name).name)
    return Path(*pure.parts)


async def _write_upload_file(upload: UploadFile, target: Path) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with target.open("wb") as handle:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break
            total += len(chunk)
            handle.write(chunk)
    await upload.close()
    return total


@router.get("/roots")
async def list_filesystem_roots(request: Request) -> dict[str, Any]:
    """回傳 WebGUI 可瀏覽的預設根目錄集合。"""
    roots: list[dict[str, str]] = []
    seen_paths: set[str] = set()

    _add_root(roots, seen_paths, name="home", path=Path.home())
    _add_root(roots, seen_paths, name="cwd", path=Path.cwd())
    _add_root(roots, seen_paths, name="root", path=Path("/"))

    for drive_root in _windows_drive_roots():
        _add_root(roots, seen_paths, name=f"drive-{drive_root.drive}", path=drive_root)

    try:
        config = await _get_config(request.app)
    except Exception:
        config = None

    if config is not None:
        _add_root(roots, seen_paths, name="workspace", path=Path(config.workspace_dir).expanduser())
        _add_root(roots, seen_paths, name="sessions", path=Path(config.sessions_dir).expanduser())
        _add_root(roots, seen_paths, name="skills", path=Path(config.skills_dir).expanduser())
        _add_root(roots, seen_paths, name="plugins", path=Path(config.plugins_dir).expanduser())
        _add_root(
            roots,
            seen_paths,
            name="memory-parent",
            path=Path(config.memory.db_path).expanduser().parent,
        )
        _add_root(
            roots,
            seen_paths,
            name="stt-cache-parent",
            path=Path(config.voice.stt_model_cache_dir).expanduser().parent,
        )

    return {"type": "filesystem_roots", "items": roots}


@router.get("/list")
async def list_directory(path: str = Query(..., min_length=1)) -> dict[str, Any]:
    """列出單一資料夾的一層子項目。"""
    current, selected_path = _coerce_browse_directory(path)

    try:
        entries = list(current.iterdir())
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail="Permission denied") from exc
    except OSError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    items: list[dict[str, Any]] = []
    for entry in sorted(entries, key=lambda item: item.name.lower()):
        try:
            item = {
                "name": entry.name,
                "path": str(entry),
                "is_dir": entry.is_dir(),
                "is_file": entry.is_file(),
            }
        except (PermissionError, OSError):
            continue
        items.append(item)

    parent = current.parent if current.parent != current else None
    return {
        "type": "filesystem_list",
        "requested_path": path,
        "current_path": str(current),
        "parent_path": str(parent) if parent is not None else None,
        "selected_path": str(selected_path) if selected_path is not None else None,
        "items": items,
    }


@router.post("/import")
async def import_local_files(
    request: Request,
    files: Annotated[list[UploadFile], File()],
    relative_paths: Annotated[list[str] | None, Form()] = None,
    target_dir: Annotated[str | None, Form()] = None,
    package_name: Annotated[str | None, Form()] = None,
) -> dict[str, Any]:
    """匯入瀏覽器選取的本機檔案/資料夾到後端受控目錄。

    瀏覽器不提供可給後端直接使用的 client absolute path，因此這個 endpoint
    實作的是「選取後上傳」，並回傳後端實際保存路徑。
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    if target_dir:
        import_root = _path_from_client(target_dir)
    else:
        config = await _get_config(request.app)
        import_root = Path(config.voice.stt_model_cache_dir).expanduser()

    upload_root = import_root / "browser-imports"
    package_root = upload_root / f"{int(time.time())}-{_safe_package_name(package_name)}"
    package_root.mkdir(parents=True, exist_ok=True)

    saved_files: list[dict[str, Any]] = []
    total_bytes = 0
    upload_relative_paths = relative_paths or []
    for index, upload in enumerate(files):
        fallback_name = upload.filename or f"file-{index + 1}"
        relative = _safe_relative_path(
            upload_relative_paths[index] if index < len(upload_relative_paths) else fallback_name,
            fallback_name,
        )
        target = package_root / relative
        size = await _write_upload_file(upload, target)
        total_bytes += size
        saved_files.append({
            "name": fallback_name,
            "path": str(target),
            "relative_path": str(relative).replace("\\", "/"),
            "size": size,
        })

    imported_path = package_root
    if len(saved_files) == 1:
        imported_path = Path(saved_files[0]["path"])

    return {
        "type": "filesystem_import",
        "import_root": str(package_root),
        "imported_path": str(imported_path),
        "file_count": len(saved_files),
        "total_bytes": total_bytes,
        "files": saved_files,
    }
