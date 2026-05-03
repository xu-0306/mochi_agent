"""Shell 工具 — 在安全限制下執行命令。"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from mochi.tools.base import BaseTool, ToolResult
from mochi.utils.security import (
    is_safe_command,
    normalize_workspace_dir,
    resolve_path_in_workspace,
)

ShellRunner = Callable[[str, Path, int], Awaitable[tuple[int, str, str]]]


class ShellTool(BaseTool):
    """受控 Shell 命令執行工具（deny-by-default）。"""

    def __init__(
        self,
        *,
        allowlist: list[str] | None = None,
        workspace_dir: str | Path | None = None,
        require_approval: bool = True,
        default_timeout_sec: int = 30,
        runner: ShellRunner | None = None,
    ) -> None:
        """初始化 Shell 工具。

        Args:
            allowlist: 允許執行的命令白名單（只比對 base command）。
            workspace_dir: 工作目錄根路徑，`cwd` 只能在此範圍內。
            require_approval: 是否要求傳入 approved=True 才執行。
            default_timeout_sec: 預設命令逾時秒數。
            runner: 可注入的命令執行器（便於測試或替換 runtime）。
        """
        self._allowlist = allowlist or ["ls", "cat", "pwd", "echo", "date", "which"]
        self._workspace_dir = normalize_workspace_dir(workspace_dir or "~/.mochi")
        self._require_approval = require_approval
        self._default_timeout_sec = default_timeout_sec
        self._runner = runner or self._default_runner

    @property
    def name(self) -> str:
        """工具名稱。"""
        return "shell"

    @property
    def description(self) -> str:
        """工具用途描述。"""
        return "Run a shell command under policy constraints (allowlist plus workspace)."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema 格式參數。"""
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Shell command to run. Only allowlisted commands are permitted.",
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory. Must be inside workspace_dir.",
                },
                "timeout_sec": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 300,
                    "default": 30,
                    "description": "Command timeout in seconds.",
                },
                "approved": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether user approval has been granted. Required when require_approval is true.",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        }

    @property
    def requires_approval(self) -> bool:
        """此工具是否預設需要審批。"""
        return self._require_approval

    async def execute(
        self,
        *,
        command: str,
        cwd: str | None = None,
        timeout_sec: int | None = None,
        approved: bool = False,
    ) -> ToolResult:
        """執行受控 shell 命令。"""
        if not command.strip():
            return ToolResult(error="`command` must not be empty.")

        if not is_safe_command(command, self._allowlist):
            return ToolResult(
                error=(
                    "Command denied by policy: only allowlist commands without shell chaining "
                    "syntax are allowed."
                ),
                metadata={"allowlist": sorted(set(self._allowlist))},
            )

        if self._require_approval and not approved:
            return ToolResult(
                error="Shell command requires approval.",
                metadata={"requires_approval": True},
            )

        try:
            working_dir = (
                resolve_path_in_workspace(cwd, self._workspace_dir)
                if cwd is not None
                else self._workspace_dir
            )
        except ValueError as exc:
            return ToolResult(error=str(exc))

        if not working_dir.exists() or not working_dir.is_dir():
            return ToolResult(error=f"Working directory does not exist: {working_dir}")

        effective_timeout = timeout_sec if timeout_sec is not None else self._default_timeout_sec
        if effective_timeout <= 0:
            return ToolResult(error="`timeout_sec` must be greater than 0.")

        try:
            returncode, stdout, stderr = await self._runner(command, working_dir, effective_timeout)
        except Exception as exc:  # pragma: no cover - 防禦性保護
            return ToolResult(
                error=f"Shell execution failed: {exc}",
                metadata={"command": command, "cwd": str(working_dir)},
            )

        metadata = {
            "command": command,
            "cwd": str(working_dir),
            "returncode": returncode,
        }
        if stderr:
            metadata["stderr"] = stderr

        if returncode != 0:
            return ToolResult(
                error=stderr or f"Command exited with non-zero status: {returncode}",
                output=stdout,
                metadata=metadata,
            )

        return ToolResult(output=stdout, metadata=metadata)

    @staticmethod
    async def _default_runner(command: str, cwd: Path, timeout_sec: int) -> tuple[int, str, str]:
        """預設 subprocess runner。"""
        process = await asyncio.create_subprocess_shell(
            command,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                process.communicate(), timeout=timeout_sec
            )
        except TimeoutError:
            process.kill()
            await process.communicate()
            return 124, "", f"Command timed out after {timeout_sec} seconds."

        return (
            process.returncode or 0,
            stdout_bytes.decode("utf-8", errors="replace"),
            stderr_bytes.decode("utf-8", errors="replace"),
        )
