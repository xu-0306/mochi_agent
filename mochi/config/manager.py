"""設定載入與管理器。"""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, cast

import yaml

try:
    from loguru import logger
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal test envs
    import logging

    logger = logging.getLogger(__name__)
from pydantic import SecretStr

from mochi.config import defaults
from mochi.config.schema import MochiConfig
from mochi.runtime.security_audit import register_known_secrets

if os.name == "nt":
    import ctypes
    from ctypes import wintypes

    class _WindowsOverlapped(ctypes.Structure):
        _fields_ = [
            ("Internal", ctypes.c_size_t),
            ("InternalHigh", ctypes.c_size_t),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        ]
else:
    _WindowsOverlapped = object  # type: ignore[misc,assignment]

PROJECT_DEFAULT_CONFIG_PATH = Path("configs/default.yaml")
EMPTY_CONFIG_REVISION = hashlib.sha256(b"").hexdigest()


@dataclass(frozen=True, slots=True)
class ConfigSnapshot:
    """Validated config paired with the exact on-disk byte revision."""

    config: MochiConfig
    revision: str
    path: Path
    exists: bool


class ConfigRevisionConflict(RuntimeError):
    """Raised when a compare-and-swap config write observes stale bytes."""

    def __init__(self, *, expected_revision: str, current_revision: str, path: Path) -> None:
        self.expected_revision = expected_revision
        self.current_revision = current_revision
        self.path = path
        super().__init__(f"Config revision conflict for {path}.")


def _safe_debug_log(message: str, *args: Any) -> None:
    """Best-effort debug logging that never breaks config loading on Windows consoles."""
    try:
        logger.debug(message, *args)
    except (OSError, UnicodeEncodeError):
        return


def user_config_path() -> Path:
    """回傳目前使用者的 Mochi YAML 設定檔路徑。"""
    if defaults.running_on_windows():
        return defaults.default_config_path()
    return _legacy_home_config_path()


def _legacy_home_config_path() -> Path:
    """Return the historic home-scoped config path for backward compatibility."""
    home = os.getenv("HOME")
    if home:
        return Path(home).expanduser() / ".mochi" / "config.yaml"
    return Path.home() / ".mochi" / "config.yaml"


def _coerce_mapping(value: Any) -> dict[str, Any]:
    """將任意值安全轉為 dict。"""
    if isinstance(value, dict):
        return dict(cast(dict[str, Any], value))
    return {}


def _read_env(name: str) -> str | None:
    """讀取環境變數並忽略空白值。"""
    value = os.getenv(name)
    if value is None:
        return None
    stripped = value.strip()
    return stripped if stripped else None


def read_env_cors_origins() -> list[str] | None:
    """讀取 CORS origins env override，供 config 與 app bootstrap 共用。"""
    cors_raw = _read_env("MOCHI_WEB_CORS_ORIGINS")
    if cors_raw is None:
        return None

    cors_origins = [item.strip() for item in cors_raw.split(",") if item.strip()]
    if cors_origins:
        return cors_origins

    logger.warning("Ignore empty MOCHI_WEB_CORS_ORIGINS override.")
    return None


def _apply_env_overrides(raw: dict[str, Any]) -> dict[str, Any]:
    """將常用部署環境變數覆蓋到設定。"""
    merged = dict(raw)

    web_section = _coerce_mapping(merged.get("web"))
    host = _read_env("MOCHI_WEB_HOST")
    if host is not None:
        web_section["host"] = host

    port_raw = _read_env("MOCHI_WEB_PORT")
    if port_raw is not None:
        try:
            web_section["port"] = int(port_raw)
        except ValueError:
            logger.warning(
                "Ignore invalid MOCHI_WEB_PORT={!r}; expected integer.",
                port_raw,
            )

    cors_origins = read_env_cors_origins()
    if cors_origins is not None:
        web_section["cors_origins"] = cors_origins

    if web_section:
        merged["web"] = web_section

    locale_defaults_section = _coerce_mapping(merged.get("locale_defaults"))
    locale_env_map = {
        "MOCHI_REGION_PROFILE": "region_profile",
        "MOCHI_LOCALE": "ui_locale",
        "MOCHI_UI_LOCALE_FALLBACK": "ui_locale_fallback",
        "MOCHI_TIMEZONE": "timezone",
        "MOCHI_RESPONSE_LANGUAGE": "response_language",
        "MOCHI_DEFAULT_TTS_VOICE": "default_tts_voice",
    }
    for env_name, config_key in locale_env_map.items():
        value = _read_env(env_name)
        if value is not None:
            locale_defaults_section[config_key] = value

    if locale_defaults_section:
        merged["locale_defaults"] = locale_defaults_section

    default_tts_voice = _read_env("MOCHI_DEFAULT_TTS_VOICE")
    tts_voice = _read_env("MOCHI_TTS_VOICE") or default_tts_voice
    if tts_voice is not None:
        voice_section = _coerce_mapping(merged.get("voice"))
        voice_section["tts_voice"] = tts_voice
        merged["voice"] = voice_section

    ollama_base_url = _read_env("MOCHI_OLLAMA_BASE_URL")
    if ollama_base_url is not None:
        ollama_section = _coerce_mapping(merged.get("ollama"))
        ollama_section["base_url"] = ollama_base_url
        merged["ollama"] = ollama_section

    tools_section = _coerce_mapping(merged.get("tools"))
    for env_name, config_key in {
        "MOCHI_WEB_SEARCH_ENGINE": "web_search_engine",
        "MOCHI_WEB_SEARCH_SEARXNG_BASE_URL": "web_search_searxng_base_url",
        "MOCHI_BRAVE_API_KEY": "web_search_brave_api_key",
        "MOCHI_TAVILY_API_KEY": "web_search_tavily_api_key",
        "MOCHI_SERPER_API_KEY": "web_search_serper_api_key",
        "MOCHI_JINA_API_KEY": "web_search_jina_api_key",
        "MOCHI_EXA_API_KEY": "web_search_exa_api_key",
        "MOCHI_WEB_FETCH_JINA_API_KEY": "web_fetch_jina_api_key",
        "MOCHI_S2_API_KEY": "semantic_scholar_api_key",
        "MOCHI_PUBMED_API_KEY": "pubmed_api_key",
        "MOCHI_PUBMED_EMAIL": "pubmed_email",
        "MOCHI_CROSSREF_MAILTO": "crossref_mailto",
    }.items():
        value = _read_env(env_name)
        if value is not None:
            tools_section[config_key] = value
    if tools_section:
        merged["tools"] = tools_section

    for env_name, config_key in {
        "MOCHI_WORKSPACE_DIR": "workspace_dir",
        "MOCHI_SESSIONS_DIR": "sessions_dir",
        "MOCHI_SKILLS_DIR": "skills_dir",
        "MOCHI_PLUGINS_DIR": "plugins_dir",
    }.items():
        value = _read_env(env_name)
        if value is not None:
            merged[config_key] = value

    return merged


def _apply_platform_path_defaults(raw: dict[str, Any]) -> dict[str, Any]:
    """未明確設定路徑時套用平台感知預設。"""
    merged = dict(raw)
    path_defaults = {
        "workspace_dir": defaults.default_workspace_dir(),
        "sessions_dir": defaults.default_sessions_dir(),
        "skills_dir": defaults.default_skills_dir(),
        "plugins_dir": defaults.default_plugins_dir(),
    }
    for key, value in path_defaults.items():
        if key not in merged or merged[key] in (None, ""):
            merged[key] = value
    memory_section = _coerce_mapping(merged.get("memory"))
    if "db_path" not in memory_section or memory_section.get("db_path") in (None, ""):
        memory_section["db_path"] = defaults.default_memory_db_path()
    merged["memory"] = memory_section
    return merged


def _normalize_windows_runtime_paths(raw: dict[str, Any]) -> dict[str, Any]:
    """Map legacy Windows `~/.mochi/...` runtime paths into the project-local `.mochi/...` tree."""
    if not defaults.running_on_windows():
        return raw

    merged = dict(raw)
    for key in ("workspace_dir", "sessions_dir", "skills_dir", "plugins_dir"):
        merged[key] = _normalize_windows_runtime_path_value(merged.get(key))

    memory_section = _coerce_mapping(merged.get("memory"))
    memory_section["db_path"] = _normalize_windows_runtime_path_value(memory_section.get("db_path"))
    merged["memory"] = memory_section
    return merged


def _normalize_windows_runtime_path_value(value: Any) -> Any:
    if not isinstance(value, (str, Path)):
        return value

    raw = str(value).replace("\\", "/")
    target_root = defaults.default_workspace_dir().replace("\\", "/")
    for legacy_root in _legacy_windows_state_roots():
        if raw == legacy_root:
            return target_root
        prefix = f"{legacy_root}/"
        if raw.startswith(prefix):
            suffix = raw[len(prefix):]
            return f"{target_root}/{suffix}"
    return value


def _legacy_windows_state_roots() -> tuple[str, ...]:
    roots = ["~/.mochi"]
    expanded = _legacy_home_config_path().parent.as_posix()
    if expanded not in roots:
        roots.append(expanded)
    return tuple(roots)


def _should_persist_windows_migration(
    *,
    source_path: Path,
    original: dict[str, Any],
    normalized: dict[str, Any],
) -> bool:
    if not defaults.running_on_windows():
        return False

    try:
        normalized_source = source_path.expanduser().resolve(strict=False)
        legacy_source = _legacy_home_config_path().expanduser().resolve(strict=False)
        current_source = user_config_path().expanduser().resolve(strict=False)
    except OSError:
        return original != normalized

    return (
        normalized_source == legacy_source
        or (normalized_source == current_source and original != normalized)
    )


def load_config(config_path: str | Path | None = None) -> MochiConfig:
    """從 YAML 檔案載入設定，找不到時回傳預設值。

    Args:
        config_path: YAML 設定檔路徑；None 時依序嘗試
                     platform user config → ./configs/default.yaml。

    Returns:
        解析後的 MochiConfig 實例。
    """
    search_paths: list[Path] = []

    if config_path is not None:
        search_paths.append(Path(config_path))
    else:
        search_paths.extend([
            user_config_path(),
            _legacy_home_config_path(),
            PROJECT_DEFAULT_CONFIG_PATH,
        ])

    for path in search_paths:
        if path.exists():
            _safe_debug_log("Loading config from {}", path)
            raw = _coerce_mapping(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
            prepared = _apply_env_overrides(_apply_platform_path_defaults(raw))
            normalized = _normalize_windows_runtime_paths(prepared)
            config = MochiConfig.model_validate(normalized)
            register_known_secrets(config)
            if config_path is None and _should_persist_windows_migration(
                source_path=path,
                original=prepared,
                normalized=normalized,
            ):
                target = user_config_path()
                save_config(
                    config,
                    target,
                    expected_revision=config_revision(target),
                )
            return config

    _safe_debug_log("No config file found, using defaults.")
    config = MochiConfig.model_validate(
        _normalize_windows_runtime_paths(
            _apply_env_overrides(_apply_platform_path_defaults({}))
        )
    )
    register_known_secrets(config)
    return config


def load_config_snapshot(config_path: str | Path | None = None) -> ConfigSnapshot:
    """Load validated config plus SHA-256 of the exact selected file bytes."""

    if config_path is None:
        # Preserve the existing one-time Windows migration before selecting the
        # write target. Project/legacy defaults may seed the config value, but
        # compare-and-swap must bind to the user file that save_config(None) writes.
        fallback_config = load_config()
        path = user_config_path().expanduser()
    else:
        fallback_config = None
        path = Path(config_path).expanduser()
    try:
        raw = path.read_bytes()
        exists = True
    except FileNotFoundError:
        raw = b""
        exists = False
    config = _config_from_bytes(raw) if exists else fallback_config or _config_from_bytes(b"")
    register_known_secrets(config)
    return ConfigSnapshot(
        config=config,
        revision=_revision(raw),
        path=path,
        exists=exists,
    )


def config_revision(config_path: str | Path | None = None) -> str:
    path = _selected_config_path(config_path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raw = b""
    return _revision(raw)


def save_config(
    config: MochiConfig,
    config_path: str | Path | None = None,
    *,
    expected_revision: str,
) -> Path:
    """將設定保存成 YAML，預設寫入使用者設定檔。

    `SecretStr` 會以原始值寫入本機檔案；呼叫端仍需避免將檔案內容回傳到 API。
    """
    path = Path(config_path) if config_path is not None else user_config_path()
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    register_known_secrets(config)
    data = _serialize_for_yaml(config.model_dump(mode="python"))
    encoded = yaml.safe_dump(data, allow_unicode=True, sort_keys=False).encode("utf-8")
    with _config_path_lock(path):
        current_revision = config_revision(path)
        if current_revision != expected_revision:
            raise ConfigRevisionConflict(
                expected_revision=expected_revision,
                current_revision=current_revision,
                path=path,
            )
        _atomic_write_config(path, encoded)
    _safe_debug_log("Saved config to {}", path)
    return path


def _selected_config_path(config_path: str | Path | None) -> Path:
    if config_path is not None:
        return Path(config_path).expanduser()
    return user_config_path().expanduser()


def _revision(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _config_from_bytes(raw: bytes) -> MochiConfig:
    loaded = yaml.safe_load(raw.decode("utf-8")) if raw else {}
    prepared = _apply_env_overrides(
        _apply_platform_path_defaults(_coerce_mapping(loaded or {}))
    )
    normalized = _normalize_windows_runtime_paths(prepared)
    return MochiConfig.model_validate(normalized)


@contextmanager
def _config_path_lock(path: Path) -> Iterator[None]:
    lock_path = path.with_name(f".{path.name}.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    handle = lock_path.open("a+b")
    lock_token: object | None = None
    locked = False
    try:
        lock_token = _lock_file(handle)
        locked = True
        yield
    finally:
        try:
            if locked:
                _unlock_file(handle, lock_token)
        finally:
            handle.close()


def _lock_file(handle: BinaryIO) -> object | None:
    if os.name != "nt":
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        return None

    import ctypes
    import msvcrt
    from ctypes import wintypes

    overlapped = _WindowsOverlapped()
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    lock_file_ex = kernel32.LockFileEx
    lock_file_ex.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_WindowsOverlapped),
    ]
    lock_file_ex.restype = wintypes.BOOL
    os_handle = wintypes.HANDLE(msvcrt.get_osfhandle(handle.fileno()))
    if not lock_file_ex(os_handle, 0x2, 0, 1, 0, ctypes.byref(overlapped)):
        raise ctypes.WinError(ctypes.get_last_error())
    return overlapped


def _unlock_file(handle: BinaryIO, lock_token: object | None) -> None:
    if os.name != "nt":
        import fcntl

        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return

    import ctypes
    import msvcrt
    from ctypes import wintypes

    overlapped = (
        lock_token if isinstance(lock_token, _WindowsOverlapped) else _WindowsOverlapped()
    )
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    unlock_file_ex = kernel32.UnlockFileEx
    unlock_file_ex.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_WindowsOverlapped),
    ]
    unlock_file_ex.restype = wintypes.BOOL
    os_handle = wintypes.HANDLE(msvcrt.get_osfhandle(handle.fileno()))
    if not unlock_file_ex(os_handle, 0, 1, 0, ctypes.byref(overlapped)):
        raise ctypes.WinError(ctypes.get_last_error())


def _atomic_write_config(path: Path, encoded: bytes) -> None:
    existing_mode = path.stat().st_mode if path.exists() else None
    file_descriptor, temp_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(file_descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        if existing_mode is not None and os.name != "nt":
            os.chmod(temp_path, existing_mode)
        _atomic_replace(temp_path, path)
        _fsync_parent(path.parent)
    except BaseException:
        with suppress(OSError):
            temp_path.unlink(missing_ok=True)
        raise


def _atomic_replace(source: Path, target: Path) -> None:
    if os.name != "nt" or not target.exists():
        os.replace(source, target)
        return

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    replace_file = kernel32.ReplaceFileW
    replace_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.LPVOID,
    ]
    replace_file.restype = wintypes.BOOL
    if not replace_file(str(target), str(source), None, 0x1, None, None):
        raise ctypes.WinError(ctypes.get_last_error())


def _fsync_parent(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _serialize_for_yaml(value: Any) -> Any:
    """轉換 Pydantic dump 結果為 PyYAML 可安全輸出的基本型別。"""
    if isinstance(value, SecretStr):
        return value.get_secret_value()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        mapping = cast(dict[Any, Any], value)
        return {key: _serialize_for_yaml(item) for key, item in mapping.items()}
    if isinstance(value, list):
        items = cast(list[Any], value)
        return [_serialize_for_yaml(item) for item in items]
    return value
