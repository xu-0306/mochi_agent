"""檔案工具 — 讀取與寫入（含 workspace 安全限制）。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from mochi.tools.base import BaseTool, ToolResult
from mochi.utils.security import (
    content_size_bytes,
    is_within_write_size_limit,
    normalize_workspace_dir,
    resolve_path_in_workspace,
    size_limit_bytes,
)

FileReader = Callable[[Path, str], Awaitable[str]]
FileWriter = Callable[[Path, str, bool, str], Awaitable[int]]


class FileReadTool(BaseTool):
    """受控檔案讀取工具。"""

    def __init__(
        self,
        *,
        workspace_dir: str | Path | None = None,
        default_encoding: str = "utf-8",
        max_read_bytes: int = 1024 * 1024,
        reader: FileReader | None = None,
    ) -> None:
        """初始化讀取工具。

        Args:
            workspace_dir: 限制讀取範圍的根目錄。
            default_encoding: 預設檔案編碼。
            max_read_bytes: 單次讀取最大位元組數。
            reader: 可注入讀取函式（便於測試與替換）。
        """
        self._workspace_dir = normalize_workspace_dir(workspace_dir or "~/.mochi")
        self._default_encoding = default_encoding
        self._max_read_bytes = max_read_bytes
        self._reader = reader or self._default_reader

    @property
    def name(self) -> str:
        """工具名稱。"""
        return "file_read"

    @property
    def description(self) -> str:
        """工具用途描述。"""
        return "Read text file contents from inside the workspace."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema 格式參數。"""
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path. Must be inside the workspace."},
                "encoding": {"type": "string", "default": "utf-8"},
                "max_bytes": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum bytes allowed for this read. Overrides the default limit.",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        *,
        path: str,
        encoding: str | None = None,
        max_bytes: int | None = None,
    ) -> ToolResult:
        """讀取檔案內容。"""
        if not path.strip():
            return ToolResult(error="`path` must not be empty.")

        try:
            target = resolve_path_in_workspace(path, self._workspace_dir)
        except ValueError as exc:
            return ToolResult(error=str(exc))

        if not target.exists():
            return ToolResult(error=f"File not found: {target}")
        if not target.is_file():
            return ToolResult(error=f"Path is not a file: {target}")

        effective_max_bytes = max_bytes if max_bytes is not None else self._max_read_bytes
        if effective_max_bytes <= 0:
            return ToolResult(error="`max_bytes` must be greater than 0.")

        file_size = await asyncio.to_thread(lambda: target.stat().st_size)
        if file_size > effective_max_bytes:
            return ToolResult(
                error=(
                    f"File is too large to read: {file_size} bytes exceeds limit "
                    f"{effective_max_bytes} bytes."
                ),
                metadata={"path": str(target), "size_bytes": file_size},
            )

        text = await self._reader(target, encoding or self._default_encoding)
        return ToolResult(
            output=text,
            metadata={"path": str(target), "size_bytes": file_size},
        )

    @staticmethod
    async def _default_reader(path: Path, encoding: str) -> str:
        """預設讀取實作（以 thread 包裝同步 I/O）。"""
        return await asyncio.to_thread(path.read_text, encoding=encoding)


class FileWriteTool(BaseTool):
    """受控檔案寫入工具。"""

    def __init__(
        self,
        *,
        workspace_dir: str | Path | None = None,
        require_approval: bool = True,
        max_write_size_mb: float = 10.0,
        default_encoding: str = "utf-8",
        writer: FileWriter | None = None,
    ) -> None:
        """初始化寫入工具。

        Args:
            workspace_dir: 限制寫入範圍的根目錄。
            require_approval: 是否要求 approved=True 才可寫入。
            max_write_size_mb: 單次最大可寫入大小（MB）。
            default_encoding: 預設寫入編碼。
            writer: 可注入寫入函式（便於測試與替換）。
        """
        self._workspace_dir = normalize_workspace_dir(workspace_dir or "~/.mochi")
        self._require_approval = require_approval
        self._max_write_size_mb = max_write_size_mb
        self._default_encoding = default_encoding
        self._writer = writer or self._default_writer

    @property
    def name(self) -> str:
        """工具名稱。"""
        return "file_write"

    @property
    def description(self) -> str:
        """工具用途描述。"""
        return "Write text to a file inside the workspace, either replacing or appending."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema 格式參數。"""
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path. Must be inside the workspace."},
                "content": {"type": "string", "description": "Text content to write."},
                "append": {"type": "boolean", "default": False, "description": "Append instead of replacing the file."},
                "encoding": {"type": "string", "default": "utf-8"},
                "approved": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether user approval has been granted. Required when require_approval is true.",
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        }

    @property
    def requires_approval(self) -> bool:
        """此工具是否預設需要審批。"""
        return self._require_approval

    async def execute(
        self,
        *,
        path: str,
        content: str,
        append: bool = False,
        encoding: str | None = None,
        approved: bool = False,
    ) -> ToolResult:
        """寫入檔案內容。"""
        if not path.strip():
            return ToolResult(error="`path` must not be empty.")

        if self._require_approval and not approved:
            return ToolResult(
                error="File write requires approval.",
                metadata={"requires_approval": True},
            )

        active_encoding = encoding or self._default_encoding
        if not is_within_write_size_limit(
            content=content,
            max_size_mb=self._max_write_size_mb,
            encoding=active_encoding,
        ):
            content_size = content_size_bytes(content, encoding=active_encoding)
            return ToolResult(
                error=(
                    f"Write content too large: {content_size} bytes exceeds limit "
                    f"{size_limit_bytes(self._max_write_size_mb)} bytes."
                ),
                metadata={"size_bytes": content_size},
            )

        try:
            target = resolve_path_in_workspace(path, self._workspace_dir)
        except ValueError as exc:
            return ToolResult(error=str(exc))

        bytes_written = await self._writer(target, content, append, active_encoding)
        return ToolResult(
            output=str(target),
            metadata={"path": str(target), "bytes_written": bytes_written, "append": append},
        )

    @staticmethod
    async def _default_writer(path: Path, content: str, append: bool, encoding: str) -> int:
        """預設寫入實作（以 thread 包裝同步 I/O）。"""

        def _sync_write() -> int:
            path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with path.open(mode=mode, encoding=encoding) as file:
                file.write(content)
            return len(content.encode(encoding))

        return await asyncio.to_thread(_sync_write)
