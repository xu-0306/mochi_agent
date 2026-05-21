"""安全工具 — 命令白名單、路徑限制與寫入大小檢查。"""

from __future__ import annotations

import shlex
from pathlib import Path

_FORBIDDEN_SHELL_PATTERNS: tuple[str, ...] = (
    "&&",
    "||",
    ";",
    "|",
    "`",
    "$(",
    "\n",
    "\r",
    ">",
    "<",
)


def _normalize_cmd_name(raw: str) -> str:
    """將命令字串標準化為純命令名稱。"""
    return Path(raw.strip()).name

def is_safe_command(command: str, allowlist: list[str]) -> bool:
    """判斷 shell 命令是否可安全執行。

    Args:
        command: 欲執行的 shell 命令字串。
        allowlist: 允許自動執行的命令列表。

    Returns:
        命令安全且在白名單中則回傳 True。
    """
    stripped = command.strip()
    if not stripped:
        return False

    if any(pattern in stripped for pattern in _FORBIDDEN_SHELL_PATTERNS):
        return False

    try:
        tokens = shlex.split(stripped, posix=True)
    except ValueError:
        return False

    if not tokens:
        return False

    base_cmd = _normalize_cmd_name(tokens[0])
    allowlist_names = {
        _normalize_cmd_name(item) for item in allowlist if isinstance(item, str) and item.strip()
    }
    return base_cmd in allowlist_names


def normalize_workspace_dir(workspace_dir: str | Path) -> Path:
    """正規化 workspace 路徑。"""
    return Path(workspace_dir).expanduser().resolve(strict=False)


def is_path_within_workspace(path: str | Path, workspace_dir: str | Path) -> bool:
    """檢查目標路徑是否位於 workspace 內。"""
    workspace = normalize_workspace_dir(workspace_dir)
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate

    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(workspace)
    except ValueError:
        return False
    return True


def resolve_path_in_workspace(path: str | Path, workspace_dir: str | Path) -> Path:
    """將路徑解析為絕對路徑，並限制於 workspace 內。"""
    workspace = normalize_workspace_dir(workspace_dir)
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve(strict=False)

    if not is_path_within_workspace(resolved, workspace):
        raise ValueError(f"Path '{path}' is outside workspace '{workspace}'.")
    return resolved


def resolve_path_with_scope(
    path: str | Path,
    workspace_dir: str | Path,
    scope: str,
) -> Path:
    """依 scope 解析路徑。"""
    workspace = normalize_workspace_dir(workspace_dir)
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = workspace / candidate
    resolved = candidate.resolve(strict=False)

    if scope == "workspace" and not is_path_within_workspace(resolved, workspace):
        raise ValueError(f"Path '{path}' is outside workspace '{workspace}'.")
    return resolved


def content_size_bytes(content: str, encoding: str = "utf-8") -> int:
    """計算文字內容在指定編碼下的位元組大小。"""
    return len(content.encode(encoding))


def size_limit_bytes(max_size_mb: float) -> int:
    """將 MB 換算為位元組（bytes）。"""
    return max(0, int(max_size_mb * 1024 * 1024))


def is_within_write_size_limit(
    content: str,
    max_size_mb: float,
    encoding: str = "utf-8",
) -> bool:
    """檢查內容大小是否在可寫入限制內。"""
    return content_size_bytes(content, encoding=encoding) <= size_limit_bytes(max_size_mb)
