"""File tools with workspace boundaries and stale-write guards."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
import difflib
from pathlib import Path
from typing import Any

from mochi.tools.base import BaseTool, FileReadState, ToolExecutionContext, ToolResult
from mochi.utils.security import (
    content_size_bytes,
    is_within_write_size_limit,
    normalize_workspace_dir,
    resolve_path_with_scope,
    size_limit_bytes,
)

FileReader = Callable[[Path, str], Awaitable[str]]
FileWriter = Callable[[Path, str, bool, str], Awaitable[int]]


class FileReadTool(BaseTool):
    """Read a text file inside the allowed workspace scope."""

    def __init__(
        self,
        *,
        workspace_dir: str | Path | None = None,
        path_scope: str = "workspace",
        default_encoding: str = "utf-8",
        max_read_bytes: int = 1024 * 1024,
        reader: FileReader | None = None,
    ) -> None:
        self._workspace_dir = normalize_workspace_dir(workspace_dir or "~/.mochi")
        self._path_scope = path_scope
        self._default_encoding = default_encoding
        self._max_read_bytes = max_read_bytes
        self._reader = reader or self._default_reader

    @property
    def name(self) -> str:
        return "file_read"

    @property
    def description(self) -> str:
        return (
            "Read the contents of a text file from the workspace. Returns full text "
            "content. Use when you need to inspect code, configuration, or notes. "
            "Cannot read binary files, and the path must stay inside the workspace."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
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

    @property
    def is_read_only(self) -> bool:
        return True

    @property
    def is_concurrency_safe(self) -> bool:
        return True

    async def execute(
        self,
        *,
        path: str,
        encoding: str | None = None,
        max_bytes: int | None = None,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        if not path.strip():
            return ToolResult(error="`path` must not be empty.")

        try:
            target = resolve_path_with_scope(path, self._workspace_dir, self._path_scope)
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
                metadata={"path": str(target), "size_bytes": file_size, "partial": False},
            )

        active_encoding = encoding or self._default_encoding
        text = await self._reader(target, active_encoding)

        if context is not None:
            stat = await asyncio.to_thread(target.stat)
            context.read_state_cache[str(target)] = FileReadState(
                path=str(target),
                content=text,
                encoding=active_encoding,
                mtime_ns=getattr(stat, "st_mtime_ns", None),
                size_bytes=file_size,
                partial=False,
            )

        return ToolResult(
            output=text,
            metadata={"path": str(target), "size_bytes": file_size, "partial": False},
        )

    @staticmethod
    async def _default_reader(path: Path, encoding: str) -> str:
        return await asyncio.to_thread(path.read_text, encoding=encoding)


class FileWriteTool(BaseTool):
    """Write text to a workspace file with undo metadata."""

    def __init__(
        self,
        *,
        workspace_dir: str | Path | None = None,
        path_scope: str = "workspace",
        require_approval: bool = True,
        max_write_size_mb: float = 10.0,
        undo_max_size_mb: float = 2.0,
        default_encoding: str = "utf-8",
        writer: FileWriter | None = None,
    ) -> None:
        self._workspace_dir = normalize_workspace_dir(workspace_dir or "~/.mochi")
        self._path_scope = path_scope
        self._require_approval = require_approval
        self._max_write_size_mb = max_write_size_mb
        self._undo_max_size_mb = undo_max_size_mb
        self._default_encoding = default_encoding
        self._writer = writer or self._default_writer

    @property
    def name(self) -> str:
        return "file_write"

    @property
    def description(self) -> str:
        return (
            "Write text to a file inside the workspace, either replacing or appending. "
            "Use for controlled text or code updates after deciding on the content. "
            "Cannot write outside the workspace and may require approval."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
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
        return self._require_approval

    async def execute(
        self,
        *,
        path: str,
        content: str,
        append: bool = False,
        encoding: str | None = None,
        approved: bool = False,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
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
            target = resolve_path_with_scope(path, self._workspace_dir, self._path_scope)
        except ValueError as exc:
            return ToolResult(error=str(exc))

        existing_result = await _load_existing_content(target, active_encoding)
        if existing_result.error is not None:
            return existing_result
        existing_content = str(existing_result.output)
        existed_before = target.exists()

        guard_error = await _check_stale_write_guard(
            target=target,
            current_content=existing_content,
            context=context,
        )
        if guard_error is not None:
            return guard_error

        merged_content = existing_content + content if append else content
        bytes_written = await self._writer(target, content if append else merged_content, append, active_encoding)
        metadata = await _build_file_change_metadata(
            target=target,
            original_content=existing_content,
            new_content=merged_content,
            encoding=active_encoding,
            undo_max_size_mb=self._undo_max_size_mb,
            append=append,
            existed_before=existed_before,
        )

        await _refresh_read_state_cache(
            context=context,
            target=target,
            content=merged_content,
            encoding=active_encoding,
        )
        metadata["bytes_written"] = bytes_written
        metadata["append"] = append
        return ToolResult(output=str(target), metadata=metadata)

    @staticmethod
    async def _default_writer(path: Path, content: str, append: bool, encoding: str) -> int:
        def _sync_write() -> int:
            path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with path.open(mode=mode, encoding=encoding) as file:
                file.write(content)
            return len(content.encode(encoding))

        return await asyncio.to_thread(_sync_write)


class FileEditTool(FileWriteTool):
    """Edit a file using old_string/new_string replacement semantics."""

    @property
    def name(self) -> str:
        return "file_edit"

    @property
    def description(self) -> str:
        return (
            "Edit a previously read text file by replacing old_string with new_string. "
            "Use this for incremental code or document edits instead of full-file rewrites."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path. Must be inside the workspace."},
                "old_string": {"type": "string", "description": "Text to replace."},
                "new_string": {"type": "string", "description": "Replacement text."},
                "replace_all": {
                    "type": "boolean",
                    "default": False,
                    "description": "Replace every match instead of only the first one.",
                },
                "encoding": {"type": "string", "default": "utf-8"},
                "approved": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether user approval has been granted when required.",
                },
            },
            "required": ["path", "old_string", "new_string"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        *,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        encoding: str | None = None,
        approved: bool = False,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        if self._require_approval and not approved:
            return ToolResult(
                error="File edit requires approval.",
                metadata={"requires_approval": True},
            )
        if not old_string:
            return ToolResult(error="`old_string` must not be empty.")

        active_encoding = encoding or self._default_encoding
        try:
            target = resolve_path_with_scope(path, self._workspace_dir, self._path_scope)
        except ValueError as exc:
            return ToolResult(error=str(exc))

        existing_result = await _load_existing_content(target, active_encoding)
        if existing_result.error is not None:
            return existing_result
        existing_content = str(existing_result.output)

        guard_error = await _check_stale_write_guard(
            target=target,
            current_content=existing_content,
            context=context,
            require_prior_read=True,
        )
        if guard_error is not None:
            return guard_error

        if old_string not in existing_content:
            return ToolResult(
                error="`old_string` was not found in the file. Re-read the file before editing.",
                suggestion="Read the latest file contents and retry with an exact match.",
            )

        if replace_all:
            new_content = existing_content.replace(old_string, new_string)
        else:
            new_content = existing_content.replace(old_string, new_string, 1)

        bytes_written = await self._writer(target, new_content, False, active_encoding)
        metadata = await _build_file_change_metadata(
            target=target,
            original_content=existing_content,
            new_content=new_content,
            encoding=active_encoding,
            undo_max_size_mb=self._undo_max_size_mb,
            append=False,
            existed_before=True,
        )
        await _refresh_read_state_cache(
            context=context,
            target=target,
            content=new_content,
            encoding=active_encoding,
        )
        metadata["bytes_written"] = bytes_written
        metadata["append"] = False
        metadata["edit_type"] = "replace_all" if replace_all else "replace_first"
        return ToolResult(output=str(target), metadata=metadata)


async def _load_existing_content(target: Path, encoding: str) -> ToolResult:
    if not target.exists():
        return ToolResult(output="")
    if not target.is_file():
        return ToolResult(error=f"Path is not a file: {target}")
    try:
        text = await asyncio.to_thread(target.read_text, encoding=encoding)
    except UnicodeDecodeError:
        return ToolResult(error=f"File is not valid {encoding} text: {target}")
    return ToolResult(output=text)


async def _check_stale_write_guard(
    *,
    target: Path,
    current_content: str,
    context: ToolExecutionContext | None,
    require_prior_read: bool | None = None,
) -> ToolResult | None:
    if context is None:
        return None

    if require_prior_read is None:
        require_prior_read = target.exists()

    snapshot = context.read_state_cache.get(str(target))
    if snapshot is None:
        if require_prior_read:
            return ToolResult(
                error="File must be read before write/edit.",
                suggestion="Use file_read on the target file before modifying it.",
            )
        return None

    if snapshot.partial:
        return ToolResult(
            error="Partial reads cannot be used for write/edit. Re-read the full file first.",
            suggestion="Read the full file contents before modifying it.",
        )

    if target.exists():
        stat = await asyncio.to_thread(target.stat)
        current_mtime = getattr(stat, "st_mtime_ns", None)
        if snapshot.mtime_ns is not None and current_mtime != snapshot.mtime_ns:
            return ToolResult(
                error="File changed after it was read. Re-read before writing.",
                retryable=True,
                suggestion="Run file_read again to refresh the cached snapshot.",
            )
    elif require_prior_read:
        return ToolResult(
            error="File was removed after it was read. Re-read before writing.",
            retryable=True,
        )

    if snapshot.content != current_content:
        return ToolResult(
            error="File contents are stale compared with the cached read. Re-read before writing.",
            retryable=True,
        )

    return None


async def _build_file_change_metadata(
    *,
    target: Path,
    original_content: str,
    new_content: str,
    encoding: str,
    undo_max_size_mb: float,
    append: bool,
    existed_before: bool,
) -> dict[str, Any]:
    undo_limit_bytes = size_limit_bytes(undo_max_size_mb)
    diff_text: str | None = None
    undo_reason: str | None = None
    undo_available = False
    undo_action = "restore"
    original_value: str | None = None
    new_value: str | None = None

    if undo_limit_bytes > 0:
        original_size = content_size_bytes(original_content, encoding=encoding)
        new_size = content_size_bytes(new_content, encoding=encoding)
        if max(original_size, new_size) <= undo_limit_bytes:
            undo_available = True
            original_value = original_content
            new_value = new_content
            if not existed_before and not append:
                undo_action = "delete"
        else:
            undo_reason = "file_too_large"

    if undo_available:
        diff_lines = difflib.unified_diff(
            original_content.splitlines(),
            new_content.splitlines(),
            fromfile="before",
            tofile="after",
            lineterm="",
        )
        diff_text = "\n".join(diff_lines)
        if content_size_bytes(diff_text, encoding=encoding) > undo_limit_bytes:
            diff_text = None

    return {
        "path": str(target),
        "file_path": str(target),
        "original_content": original_value,
        "new_content": new_value,
        "undo_available": undo_available,
        "undo_action": undo_action if undo_available else None,
        "undo_reason": undo_reason,
        "diff": diff_text,
        "diff_available": diff_text is not None,
        "undo_size_limit_bytes": undo_limit_bytes,
    }


async def _refresh_read_state_cache(
    *,
    context: ToolExecutionContext | None,
    target: Path,
    content: str,
    encoding: str,
) -> None:
    if context is None:
        return
    stat = await asyncio.to_thread(target.stat)
    context.read_state_cache[str(target)] = FileReadState(
        path=str(target),
        content=content,
        encoding=encoding,
        mtime_ns=getattr(stat, "st_mtime_ns", None),
        size_bytes=getattr(stat, "st_size", content_size_bytes(content, encoding=encoding)),
        partial=False,
    )
