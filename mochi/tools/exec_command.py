"""Exec runtime tool for foreground/background shell command execution."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from uuid import uuid4

from mochi.config import defaults
from mochi.runtime.approvals import InMemoryApprovalStore
from mochi.runtime.exec_runtime import ExecRuntime
from mochi.runtime.exec_sessions import SessionPollResult
from mochi.security import deny_security_decision
from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult
from mochi.utils.command_security import CommandSecurityPolicy
from mochi.utils.security import normalize_workspace_dir, resolve_path_in_workspace

_SHARED_RUNTIME: ExecRuntime | None = None
_SHARED_APPROVAL_STORE: InMemoryApprovalStore | None = None


def get_shared_exec_runtime() -> ExecRuntime:
    """Return process-wide shared exec runtime for exec tool family."""
    global _SHARED_RUNTIME
    if _SHARED_RUNTIME is None:
        _SHARED_RUNTIME = ExecRuntime()
    return _SHARED_RUNTIME


def get_shared_exec_approval_store() -> InMemoryApprovalStore:
    """Return process-wide shared approval store for exec tool family."""
    global _SHARED_APPROVAL_STORE
    if _SHARED_APPROVAL_STORE is None:
        _SHARED_APPROVAL_STORE = InMemoryApprovalStore()
    return _SHARED_APPROVAL_STORE


class ExecCommandTool(BaseTool):
    """Run shell commands through ExecRuntime with command policy checks."""

    def __init__(
        self,
        *,
        runtime: ExecRuntime | None = None,
        approval_store: InMemoryApprovalStore | None = None,
        workspace_dir: str | Path | None = None,
        allowlist: list[str] | None = None,
        allowed_env_vars: list[str] | None = None,
        require_approval: bool = False,
        default_timeout_sec: int = 30,
    ) -> None:
        self._runtime = runtime or get_shared_exec_runtime()
        self._approval_store = approval_store or get_shared_exec_approval_store()
        self._workspace_dir = normalize_workspace_dir(workspace_dir or defaults.default_workspace_dir())
        self._allowlist = list(allowlist) if allowlist is not None else []
        self._allowed_env_vars = [str(item) for item in (allowed_env_vars or [])]
        self._require_approval = bool(require_approval)
        self._default_timeout_sec = max(1, int(default_timeout_sec))

    @property
    def name(self) -> str:
        return "exec_command"

    @property
    def description(self) -> str:
        return (
            "Run a shell command using the exec runtime with foreground or background "
            "mode, command security checks, and approval-pending metadata when required."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "Command to execute.",
                },
                "shell": {
                    "type": "string",
                    "enum": ["powershell", "pwsh", "bash", "sh", "cmd"],
                    "description": "Shell provider alias.",
                },
                "workdir": {
                    "type": "string",
                    "description": "Working directory. Must stay within workspace boundaries.",
                },
                "env": {
                    "type": "object",
                    "description": "Optional environment overrides passed to the subprocess.",
                    "additionalProperties": {"type": "string"},
                },
                "timeout": {
                    "type": "number",
                    "minimum": 0.001,
                    "description": "Command timeout in seconds.",
                },
                "background": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to run in background and return session_id immediately.",
                },
                "tty": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether to request shell interactive mode when supported.",
                },
                "sandbox_permissions": {
                    "type": "string",
                    "enum": ["use_default", "require_escalated"],
                    "default": "use_default",
                    "description": "Escalation intent hint used by approval policy.",
                },
                "justification": {
                    "type": "string",
                    "description": "User-facing rationale for escalation requests.",
                },
                "prefix_rule": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Suggested future-approval command prefix.",
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
        shell: str | None = None,
        workdir: str | None = None,
        env: dict[str, str] | None = None,
        timeout: float | int | None = None,
        background: bool = False,
        tty: bool = False,
        sandbox_permissions: str = "use_default",
        justification: str | None = None,
        prefix_rule: list[str] | None = None,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        del prefix_rule
        if not command.strip():
            return ToolResult(error="`command` must not be empty.")
        if sandbox_permissions not in {"use_default", "require_escalated"}:
            return ToolResult(
                error="`sandbox_permissions` must be 'use_default' or 'require_escalated'."
            )
        if env is not None and not isinstance(env, dict):
            return ToolResult(error="`env` must be an object mapping environment key/value strings.")
        normalized_env = self._normalize_env(env)
        if isinstance(normalized_env, ToolResult):
            return normalized_env

        workspace_root = self._resolve_workspace_root(context)
        if workdir is None:
            resolved_cwd = workspace_root
        else:
            try:
                resolved_cwd = resolve_path_in_workspace(workdir, workspace_root)
            except ValueError as exc:
                return ToolResult(error=str(exc))

        security = CommandSecurityPolicy(
            allowlist=self._allowlist,
            workspace_dir=resolved_cwd,
            allowed_env_vars=self._allowed_env_vars,
            allow_dangerous_interpreters=True,
        )
        classification = security.classify(command, shell=shell, env=normalized_env)
        if classification.action == "deny":
            decision = deny_security_decision(
                reason=classification.reason,
                approval_kind="shell",
                approval_scope="dangerous_command",
                replay_safe=False,
                policy_source=f"exec_policy:{classification.rule_id}",
            )
            return ToolResult(
                error=classification.reason,
                metadata={"status": "denied", **decision.to_metadata()},
                retryable=False,
            )

        needs_approval = (
            self._require_approval
            or classification.action == "ask"
            or sandbox_permissions == "require_escalated"
        )
        if needs_approval:
            reason = (
                justification
                or classification.reason
                or "Exec command requires approval before execution."
            )
            approval_id = f"exec-approval-{uuid4().hex[:12]}"
            approval_payload = {
                "command": command,
                "shell": shell,
                "workdir": str(resolved_cwd),
                "env": normalized_env,
                "timeout_sec": (
                    float(timeout)
                    if timeout is not None and not isinstance(timeout, bool)
                    else float(self._default_timeout_sec)
                ),
                "background": background,
                "tty": tty,
                "approval_state": "approved",
            }
            request = self._approval_store.create(
                approval_id=approval_id,
                command=command,
                shell=(shell or "auto"),
                scope="dangerous_command",
                reason=reason,
                command_payload=approval_payload,
            )
            return ToolResult(
                error="Exec command requires approval.",
                metadata={
                    "status": "approval_pending",
                    "approval_id": request.approval_id,
                    "session_id": None,
                    "timed_out": False,
                    "requires_approval": True,
                    "security_decision": "require_approval",
                    "approval_kind": "shell",
                    "approval_scope": request.scope,
                    "reason": request.reason,
                },
                retryable=True,
            )

        timeout_sec: float | None
        if timeout is None:
            timeout_sec = float(self._default_timeout_sec)
        else:
            try:
                timeout_sec = float(timeout)
            except (TypeError, ValueError):
                return ToolResult(error="`timeout` must be a number.")
            if timeout_sec <= 0:
                return ToolResult(error="`timeout` must be greater than 0.")

        try:
            result = await self._runtime.start_command(
                command=command,
                shell=shell,
                cwd=resolved_cwd,
                env=normalized_env,
                timeout_sec=timeout_sec,
                background=background,
                tty=tty,
                approval_state="not_required",
            )
        except Exception as exc:
            return ToolResult(error=f"Exec command failed: {exc}")

        payload = _poll_to_payload(result)
        metadata = {
            "status": payload["status"],
            "session_id": payload["session_id"],
            "approval_id": None,
            "timed_out": payload["timed_out"],
            "exit_code": payload["exit_code"],
        }
        if result.status.value in {"failed", "timed_out"}:
            return ToolResult(
                error=result.stderr.strip() or f"Command exited with status: {result.status.value}",
                output=payload,
                metadata=metadata,
            )
        return ToolResult(output=payload, metadata=metadata)

    @staticmethod
    def _normalize_env(env: dict[str, str] | None) -> dict[str, str] | ToolResult | None:
        if env is None:
            return None
        normalized: dict[str, str] = {}
        for key, value in env.items():
            if not isinstance(key, str) or not isinstance(value, str):
                return ToolResult(error="`env` keys and values must be strings.")
            normalized[key] = value
        return normalized

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


def _poll_to_payload(result: SessionPollResult) -> dict[str, Any]:
    return {
        "session_id": result.session_id,
        "shell": result.shell,
        "status": result.status.value,
        "background": result.background,
        "tty": result.tty,
        "pid": result.pid,
        "exit_code": result.exit_code,
        "timed_out": result.timed_out,
        "approval_state": result.approval_state,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
