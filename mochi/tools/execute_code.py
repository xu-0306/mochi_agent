"""execute_code 工具 — 受控執行 Python 程式碼（deny-by-default）。"""

from __future__ import annotations

import asyncio
import subprocess
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from mochi.tools.base import BaseTool, ToolResult
from mochi.utils.security import normalize_workspace_dir, resolve_path_in_workspace

CodeRunner = Callable[[str, Path, int, str], Awaitable[tuple[int, str, str]]]


class ExecuteCodeTool(BaseTool):
    """受控 Python 程式碼執行工具。"""

    def __init__(
        self,
        *,
        workspace_dir: str | Path | None = None,
        require_approval: bool = True,
        default_timeout_sec: int = 10,
        python_executable: str | None = None,
        runner: CodeRunner | None = None,
    ) -> None:
        """初始化 execute_code 工具。

        Args:
            workspace_dir: 允許執行的工作目錄根路徑。
            require_approval: 是否要求傳入 approved=True 才執行。
            default_timeout_sec: 預設逾時秒數。
            python_executable: 指定 Python 執行檔路徑。
            runner: 可注入執行器（便於測試）。
        """
        self._workspace_dir = normalize_workspace_dir(workspace_dir or "~/.mochi")
        self._require_approval = require_approval
        self._default_timeout_sec = default_timeout_sec
        self._python_executable = python_executable or sys.executable
        self._runner = runner or self._default_runner

    @property
    def name(self) -> str:
        """工具名稱。"""
        return "execute_code"

    @property
    def description(self) -> str:
        """工具用途描述。"""
        return "Run Python code in a controlled subprocess with timeout and approval controls."

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema 格式參數。"""
        return {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code string to run."},
                "cwd": {
                    "type": "string",
                    "description": "Working directory. Must be inside workspace_dir.",
                },
                "timeout_sec": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 300,
                    "default": 10,
                    "description": "Execution timeout in seconds.",
                },
                "approved": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether user approval has been granted. Required when require_approval is true.",
                },
            },
            "required": ["code"],
            "additionalProperties": False,
        }

    @property
    def requires_approval(self) -> bool:
        """此工具是否預設需要審批。"""
        return self._require_approval

    async def execute(
        self,
        *,
        code: str,
        cwd: str | None = None,
        timeout_sec: int | None = None,
        approved: bool = False,
    ) -> ToolResult:
        """執行 Python 程式碼。"""
        if not code.strip():
            return ToolResult(error="`code` must not be empty.")

        if self._require_approval and not approved:
            return ToolResult(
                error="Code execution requires approval.",
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
            returncode, stdout, stderr = await self._runner(
                code,
                working_dir,
                effective_timeout,
                self._python_executable,
            )
        except Exception as exc:  # pragma: no cover - 防禦性保護
            return ToolResult(
                error=f"Code execution failed: {exc}",
                metadata={"cwd": str(working_dir)},
            )

        metadata = {
            "cwd": str(working_dir),
            "returncode": returncode,
        }
        if stderr:
            metadata["stderr"] = stderr

        if returncode != 0:
            return ToolResult(
                error=stderr or f"Process exited with non-zero status: {returncode}",
                output=stdout,
                metadata=metadata,
            )

        return ToolResult(output=stdout, metadata=metadata)

    @staticmethod
    async def _default_runner(
        code: str,
        cwd: Path,
        timeout_sec: int,
        python_executable: str,
    ) -> tuple[int, str, str]:
        """預設 subprocess 執行器。"""
        try:
            process = await asyncio.create_subprocess_exec(
                python_executable,
                "-I",
                "-c",
                code,
                cwd=str(cwd),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout_bytes, stderr_bytes = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout_sec,
                )
            except TimeoutError:
                process.kill()
                await process.communicate()
                return 124, "", f"Execution timed out after {timeout_sec} seconds."

            return (
                process.returncode or 0,
                stdout_bytes.decode("utf-8", errors="replace"),
                stderr_bytes.decode("utf-8", errors="replace"),
            )
        except Exception:
            return await asyncio.to_thread(
                ExecuteCodeTool._run_sync_fallback,
                code,
                cwd,
                timeout_sec,
                python_executable,
            )

    @staticmethod
    def _run_sync_fallback(
        code: str,
        cwd: Path,
        timeout_sec: int,
        python_executable: str,
    ) -> tuple[int, str, str]:
        """當 async subprocess 不可用時的同步 fallback。"""
        try:
            completed = subprocess.run(
                [python_executable, "-I", "-c", code],
                cwd=str(cwd),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout_sec,
                check=False,
            )
            return completed.returncode, completed.stdout, completed.stderr
        except subprocess.TimeoutExpired:
            return 124, "", f"Execution timed out after {timeout_sec} seconds."
