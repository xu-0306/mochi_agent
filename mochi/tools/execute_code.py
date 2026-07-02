"""execute_code tool with approval, timeout, and background-process support."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from mochi.config import defaults
from mochi.security import require_approval_decision
from mochi.tools.base import BaseTool, ToolCancellationResult, ToolExecutionContext, ToolResult
from mochi.tools.process_service import ProcessService
from mochi.utils.security import normalize_workspace_dir, resolve_path_in_workspace

CodeRunner = Callable[[str, Path, int, str], Awaitable[tuple[int, str, str]]]


class ExecuteCodeTool(BaseTool):
    """Run Python code in a controlled subprocess."""

    def __init__(
        self,
        *,
        workspace_dir: str | Path | None = None,
        require_approval: bool = True,
        default_timeout_sec: int = 10,
        python_executable: str | None = None,
        runner: CodeRunner | None = None,
        process_service: ProcessService | None = None,
    ) -> None:
        self._workspace_dir = normalize_workspace_dir(workspace_dir or defaults.default_workspace_dir())
        self._require_approval = require_approval
        self._default_timeout_sec = default_timeout_sec
        self._python_executable = python_executable or sys.executable
        self._runner = runner or self._default_runner
        self._uses_default_runner = runner is None
        self._process_service = process_service

    @property
    def name(self) -> str:
        return "execute_code"

    @property
    def description(self) -> str:
        return "Run Python code in a controlled subprocess with timeout and approval controls."

    @property
    def parameters_schema(self) -> dict[str, Any]:
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
                "background": {
                    "type": "boolean",
                    "default": False,
                    "description": "Run code in background and return process metadata immediately.",
                },
                "process_label": {
                    "type": "string",
                    "description": "Optional label for background process tracking.",
                },
            },
            "required": ["code"],
            "additionalProperties": False,
        }

    @property
    def requires_approval(self) -> bool:
        return self._require_approval

    @property
    def is_cancellable(self) -> bool:
        return True

    async def execute(
        self,
        *,
        code: str,
        cwd: str | None = None,
        timeout_sec: int | None = None,
        approved: bool = False,
        background: bool = False,
        process_label: str | None = None,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        if not code.strip():
            return ToolResult(error="`code` must not be empty.")

        if self._require_approval and not approved:
            decision = require_approval_decision(
                reason="Code execution requires explicit approval.",
                approval_kind="other",
                approval_scope="dangerous_command",
                replay_safe=True,
                policy_source="execute_code_policy",
            )
            return ToolResult(
                error="Code execution requires approval.",
                metadata=decision.to_metadata(),
            )

        workspace_root = self._resolve_workspace_root(context)
        try:
            working_dir = (
                resolve_path_in_workspace(cwd, workspace_root)
                if cwd is not None
                else workspace_root
            )
        except ValueError as exc:
            return ToolResult(error=str(exc))

        if not working_dir.exists() or not working_dir.is_dir():
            return ToolResult(error=f"Working directory does not exist: {working_dir}")

        effective_timeout = timeout_sec if timeout_sec is not None else self._default_timeout_sec
        if effective_timeout <= 0:
            return ToolResult(error="`timeout_sec` must be greater than 0.")

        if background:
            if self._process_service is None:
                return ToolResult(error="Background process runtime is not configured.")
            try:
                payload = await self._process_service.start_python(
                    code=code,
                    cwd=working_dir,
                    python_executable=self._python_executable,
                    label=process_label,
                )
            except Exception as exc:  # pragma: no cover
                return ToolResult(
                    error=f"Code background launch failed: {exc}",
                    metadata={"cwd": str(working_dir)},
                )
            return ToolResult(output=payload, metadata=payload)

        try:
            if self._uses_default_runner:
                runner_result = await self._execute_default_runner(
                    code=code,
                    cwd=working_dir,
                    timeout_sec=effective_timeout,
                    python_executable=self._python_executable,
                    context=context,
                )
                if isinstance(runner_result, ToolResult):
                    return runner_result
                returncode, stdout, stderr = runner_result
            else:
                returncode, stdout, stderr = await self._runner(
                    code,
                    working_dir,
                    effective_timeout,
                    self._python_executable,
                )
        except Exception as exc:  # pragma: no cover
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

    def _resolve_workspace_root(self, context: ToolExecutionContext | None) -> Path:
        if context is not None:
            for candidate in (
                context.task_sandbox_dir,
                context.project_workspace,
                context.workspace_dir,
            ):
                if candidate:
                    return normalize_workspace_dir(candidate)
        return self._workspace_dir

    @staticmethod
    async def _default_runner(
        code: str,
        cwd: Path,
        timeout_sec: int,
        python_executable: str,
    ) -> tuple[int, str, str]:
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
    async def _execute_default_runner(
        *,
        code: str,
        cwd: Path,
        timeout_sec: int,
        python_executable: str,
        context: ToolExecutionContext | None,
    ) -> ToolResult | tuple[int, str, str]:
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
        except Exception:
            return await asyncio.to_thread(
                ExecuteCodeTool._run_sync_fallback,
                code,
                cwd,
                timeout_sec,
                python_executable,
            )

        cancellation_seen = False

        async def _cancel_active_process() -> ToolCancellationResult:
            nonlocal cancellation_seen
            if process.returncode is not None:
                return ToolCancellationResult(
                    cancelled=False,
                    reason="tool_already_completed",
                    metadata={"returncode": process.returncode},
                )
            cancellation_seen = True
            process.kill()
            await process.wait()
            return ToolCancellationResult(
                cancelled=True,
                reason="tool_cancelled",
                metadata={"returncode": process.returncode},
            )

        if context is not None and context.active_tool_controller is not None:
            await context.active_tool_controller.bind_cancel_callback(
                session_id=None,
                callback=_cancel_active_process,
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

        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        returncode = process.returncode or 0

        if cancellation_seen:
            metadata = {
                "cwd": str(cwd),
                "returncode": returncode,
                "status": "cancelled",
                "cancelled": True,
            }
            if stderr:
                metadata["stderr"] = stderr
            return ToolResult(output=stdout, metadata=metadata)

        return returncode, stdout, stderr

    @staticmethod
    def _run_sync_fallback(
        code: str,
        cwd: Path,
        timeout_sec: int,
        python_executable: str,
    ) -> tuple[int, str, str]:
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
