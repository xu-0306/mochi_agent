"""Legacy shell tool kept for backwards compatibility."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from mochi.config import defaults
from mochi.security import (
    deny_security_decision,
    require_approval_decision,
    with_task_isolation_scope,
)
from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult
from mochi.tools.process_service import ProcessService
from mochi.utils.security import (
    explain_unsafe_shell_command,
    normalize_workspace_dir,
    resolve_path_in_workspace,
)

ShellRunner = Callable[[str, Path, int], Awaitable[tuple[int, str, str]]]


class ShellTool(BaseTool):
    """Legacy compatibility shell tool with strict allowlist + approval checks."""

    def __init__(
        self,
        *,
        allowlist: list[str] | None = None,
        workspace_dir: str | Path | None = None,
        require_approval: bool = True,
        default_timeout_sec: int = 30,
        runner: ShellRunner | None = None,
        process_service: ProcessService | None = None,
    ) -> None:
        self._allowlist = allowlist or ["ls", "cat", "pwd", "echo", "date", "which"]
        self._workspace_dir = normalize_workspace_dir(workspace_dir or defaults.default_workspace_dir())
        self._require_approval = require_approval
        self._default_timeout_sec = default_timeout_sec
        self._runner = runner or self._default_runner
        self._process_service = process_service

    @property
    def name(self) -> str:
        return "shell"

    @property
    def description(self) -> str:
        return (
            "Legacy compatibility shell runner for simple allowlisted commands in the "
            "workspace. Prefer exec_command for general command execution and session-based "
            "flows. Commands outside the allowlist or workspace policy are rejected."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Legacy compatibility command. Only allowlisted commands are permitted.",
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
                "background": {
                    "type": "boolean",
                    "default": False,
                    "description": "Run command in background and return process metadata immediately.",
                },
                "process_label": {
                    "type": "string",
                    "description": "Optional label for background process tracking.",
                },
            },
            "required": ["command"],
            "additionalProperties": False,
        }

    @property
    def requires_approval(self) -> bool:
        return self._require_approval

    async def execute(
        self,
        *,
        command: str,
        cwd: str | None = None,
        timeout_sec: int | None = None,
        approved: bool = False,
        background: bool = False,
        process_label: str | None = None,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        if not command.strip():
            return ToolResult(error="`command` must not be empty.")

        shell_reason = explain_unsafe_shell_command(command, self._allowlist)
        if shell_reason is not None:
            decision = deny_security_decision(
                reason=shell_reason,
                approval_kind="shell",
                approval_scope="dangerous_command",
                replay_safe=False,
                policy_source="shell_policy",
            )
            return ToolResult(
                error=shell_reason,
                metadata={
                    "allowlist": sorted(set(self._allowlist)),
                    **decision.to_metadata(),
                },
            )

        if self._require_approval and not approved:
            decision = require_approval_decision(
                reason="Shell commands require explicit approval in the current autonomy mode.",
                approval_kind="shell",
                approval_scope="workspace",
                replay_safe=True,
                policy_source="runtime_policy",
            )
            decision = with_task_isolation_scope(
                decision,
                task_sandbox_dir=context.task_sandbox_dir if context is not None else None,
            )
            return ToolResult(
                error="Shell command requires approval.",
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
                payload = await self._process_service.start_shell(
                    command=command,
                    cwd=working_dir,
                    label=process_label,
                )
            except Exception as exc:  # pragma: no cover
                return ToolResult(
                    error=f"Shell background launch failed: {exc}",
                    metadata={"command": command, "cwd": str(working_dir)},
                )
            return ToolResult(output=payload, metadata=payload)

        try:
            returncode, stdout, stderr = await self._runner(command, working_dir, effective_timeout)
        except Exception as exc:  # pragma: no cover
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
    async def _default_runner(command: str, cwd: Path, timeout_sec: int) -> tuple[int, str, str]:
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
