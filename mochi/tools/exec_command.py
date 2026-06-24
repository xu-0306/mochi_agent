"""Exec runtime tool for foreground/background shell command execution."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from mochi.config import defaults
from mochi.runtime.approvals import ApprovalStore, InMemoryApprovalStore
from mochi.runtime.exec_runtime import ExecRuntime
from mochi.runtime.exec_sessions import SessionPollResult
from mochi.security import deny_security_decision
from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult
from mochi.utils.command_security import CommandSecurityPolicy, CommandSecurityResult
from mochi.utils.security import build_policy_metadata, normalize_workspace_dir, resolve_path_in_workspace

_SHARED_RUNTIME: ExecRuntime | None = None
_SHARED_APPROVAL_STORE: ApprovalStore | None = None


def _build_suggested_rule(
    classification: CommandSecurityResult,
    *,
    shell: str | None,
) -> dict[str, Any] | None:
    if not classification.parsed_tokens:
        return None
    return {
        "tokens": list(classification.parsed_tokens),
        "decision": "allow",
        "match": "prefix",
        "shells": [shell] if isinstance(shell, str) and shell else [],
    }


def _build_classification_metadata(
    classification: CommandSecurityResult,
    *,
    shell: str | None,
    decision: object | None = None,
    policy_state: str | None = None,
    policy_reason: str | None = None,
) -> dict[str, Any]:
    metadata = build_policy_metadata(
        decision=decision,
        policy_state=policy_state,
        policy_reason=policy_reason or classification.reason,
    )
    metadata["rule_id"] = classification.rule_id
    metadata["suggested_rule"] = _build_suggested_rule(classification, shell=shell)
    return metadata


def get_shared_exec_runtime() -> ExecRuntime:
    """Return process-wide shared exec runtime for exec tool family."""
    global _SHARED_RUNTIME
    if _SHARED_RUNTIME is None:
        _SHARED_RUNTIME = _build_shared_exec_runtime()
    return _SHARED_RUNTIME


def get_shared_exec_approval_store() -> ApprovalStore:
    """Return process-wide shared approval store for exec tool family."""
    global _SHARED_APPROVAL_STORE
    if _SHARED_APPROVAL_STORE is None:
        _SHARED_APPROVAL_STORE = InMemoryApprovalStore()
    return _SHARED_APPROVAL_STORE


def _shared_exec_runtime_state_root() -> Path:
    return normalize_workspace_dir(Path(defaults.default_workspace_dir()) / "exec-runtime")


def _build_shared_exec_runtime() -> ExecRuntime:
    state_root = _shared_exec_runtime_state_root()
    kwargs: dict[str, Any] = {}
    try:
        signature = inspect.signature(ExecRuntime)
    except (TypeError, ValueError):
        signature = None
    if signature is not None and "state_root" in signature.parameters:
        kwargs["state_root"] = state_root
    return ExecRuntime(**kwargs)


class ExecCommandTool(BaseTool):
    """Run shell commands through ExecRuntime with command policy checks."""

    def __init__(
        self,
        *,
        runtime: ExecRuntime | None = None,
        approval_store: ApprovalStore | None = None,
        workspace_dir: str | Path | None = None,
        command_rules: list[dict[str, object]] | None = None,
        allowed_env_vars: list[str] | None = None,
        require_approval: bool = False,
        default_timeout_sec: int = 30,
    ) -> None:
        self._runtime = runtime or get_shared_exec_runtime()
        self._approval_store = approval_store or get_shared_exec_approval_store()
        self._workspace_dir = normalize_workspace_dir(workspace_dir or defaults.default_workspace_dir())
        self._command_rules = [dict(item) for item in (command_rules or []) if isinstance(item, dict)]
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
            "mode, explicit allow/ask/deny command policy checks, and approval-pending "
            "metadata when required."
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
        log_path: str | None = None,
        checkpoint_dir: str | None = None,
        detached_layout: Mapping[str, Any] | None = None,
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
        resolved_layout = _resolve_detached_layout(
            log_path=log_path,
            checkpoint_dir=checkpoint_dir,
            detached_layout=detached_layout,
        )

        workspace_root = self._resolve_workspace_root(context)
        if workdir is None:
            resolved_cwd = workspace_root
        else:
            try:
                resolved_cwd = resolve_path_in_workspace(workdir, workspace_root)
            except ValueError as exc:
                return ToolResult(error=str(exc))

        security = CommandSecurityPolicy(
            command_rules=self._command_rules,
            workspace_dir=resolved_cwd,
            allowed_env_vars=self._allowed_env_vars,
            allow_dangerous_interpreters=True,
        )
        classification = security.classify(command, shell=shell, env=normalized_env)
        if classification.action == "deny":
            decision = deny_security_decision(
                reason=classification.reason,
                approval_kind="exec",
                approval_scope="dangerous_command",
                replay_safe=False,
                policy_source=f"exec_policy:{classification.rule_id}",
            )
            return ToolResult(
                error=classification.reason,
                metadata={
                    "status": "denied",
                    **_build_classification_metadata(
                        classification,
                        shell=shell,
                        decision=decision,
                    ),
                },
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
            suggested_rule = _build_suggested_rule(classification, shell=shell)
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
                "log_path": resolved_layout.get("log_path"),
                "checkpoint_dir": resolved_layout.get("checkpoint_dir"),
                "detached_layout": dict(resolved_layout),
                "approval_state": "approved",
            }
            request = self._approval_store.create(
                approval_id=approval_id,
                command=command,
                shell=(shell or "auto"),
                scope="dangerous_command",
                reason=reason,
                metadata={
                    "policy_state": "ask",
                    "policy_reason": classification.reason,
                    "rule_id": classification.rule_id,
                    "suggested_rule": suggested_rule,
                },
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
                    "approval_kind": "exec",
                    "approval_scope": request.scope,
                    "reason": request.reason,
                    **_build_classification_metadata(
                        classification,
                        shell=shell,
                        policy_state="ask",
                        policy_reason=classification.reason,
                    ),
                    **_build_exec_session_metadata(
                        payload=None,
                        detached_layout=resolved_layout,
                        background=background,
                        tty=tty,
                        approval_state="pending",
                    ),
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
                log_path=resolved_layout.get("log_path"),
                checkpoint_dir=resolved_layout.get("checkpoint_dir"),
            )
        except Exception as exc:
            return ToolResult(error=f"Exec command failed: {exc}")

        effective_layout = _realize_detached_layout(
            session_id=result.session_id,
            detached=result.detached,
            detached_layout=resolved_layout,
        )
        payload = _poll_to_payload(
            result,
            detached_layout=effective_layout,
        )
        metadata = _build_exec_session_metadata(
            payload=payload,
            detached_layout=effective_layout,
            background=background,
            tty=tty,
            approval_state=result.approval_state,
        )
        metadata.update(
            _build_classification_metadata(
                classification,
                shell=shell,
                policy_state="allow",
                policy_reason=classification.reason,
            )
        )
        metadata["approval_id"] = None
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


def _poll_to_payload(
    result: SessionPollResult,
    *,
    detached_layout: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
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
        "detached": result.detached,
        "restored": result.restored,
        "supports_stdin": result.supports_stdin,
    }
    payload.update(_layout_metadata(detached_layout))
    return payload


def _resolve_detached_layout(
    *,
    log_path: str | None = None,
    checkpoint_dir: str | None = None,
    detached_layout: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    layout: dict[str, Any] = {}
    if detached_layout is not None:
        layout.update(detached_layout)
    if log_path:
        layout.setdefault("log_path", str(Path(log_path).resolve()))
    if checkpoint_dir:
        layout.setdefault("checkpoint_dir", str(Path(checkpoint_dir).resolve()))
    session_log_path = layout.get("session_log_path") or layout.get("log_path")
    if isinstance(session_log_path, str) and session_log_path.strip():
        resolved_session_log = str(Path(session_log_path).resolve())
        layout["session_log_path"] = resolved_session_log
        layout["log_path"] = resolved_session_log
    if isinstance(layout.get("checkpoint_dir"), str) and str(layout["checkpoint_dir"]).strip():
        layout["checkpoint_dir"] = str(Path(str(layout["checkpoint_dir"])).resolve())
    for key in ("root_dir", "manifest_path", "stdout_log_path", "stderr_log_path", "runtime_state_root"):
        value = layout.get(key)
        if isinstance(value, str) and value.strip():
            layout[key] = str(Path(value).resolve())
    return layout


def _layout_metadata(detached_layout: Mapping[str, Any] | None) -> dict[str, Any]:
    if detached_layout is None:
        return {
            "detached_layout": None,
            "root_dir": None,
            "log_path": None,
            "session_log_path": None,
            "checkpoint_dir": None,
            "manifest_path": None,
            "stdout_log_path": None,
            "stderr_log_path": None,
            "runtime_state_root": None,
        }
    layout = dict(detached_layout)
    return {
        "detached_layout": layout,
        "root_dir": layout.get("root_dir"),
        "log_path": layout.get("log_path"),
        "session_log_path": layout.get("session_log_path") or layout.get("log_path"),
        "checkpoint_dir": layout.get("checkpoint_dir"),
        "manifest_path": layout.get("manifest_path"),
        "stdout_log_path": layout.get("stdout_log_path"),
        "stderr_log_path": layout.get("stderr_log_path"),
        "runtime_state_root": layout.get("runtime_state_root"),
    }


def _realize_detached_layout(
    *,
    session_id: str | None,
    detached: bool,
    detached_layout: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    if detached_layout is None:
        return None
    layout = dict(detached_layout)
    runtime_state_root = layout.get("runtime_state_root")
    if (
        detached
        and session_id
        and isinstance(runtime_state_root, str)
        and runtime_state_root.strip()
    ):
        layout["manifest_path"] = str(
            (Path(runtime_state_root).resolve() / session_id / "manifest.json").resolve()
        )
    return layout


def _build_exec_session_metadata(
    *,
    payload: Mapping[str, Any] | None,
    detached_layout: Mapping[str, Any] | None,
    background: bool,
    tty: bool,
    approval_state: str | None,
) -> dict[str, Any]:
    payload_dict = dict(payload or {})
    metadata = {
        "status": payload_dict.get("status") or ("approval_pending" if approval_state == "pending" else "completed"),
        "session_id": payload_dict.get("session_id"),
        "timed_out": bool(payload_dict.get("timed_out", False)),
        "exit_code": payload_dict.get("exit_code"),
        "pid": payload_dict.get("pid"),
        "background": bool(payload_dict.get("background", background)),
        "tty": bool(payload_dict.get("tty", tty)),
        "approval_state": approval_state,
        "detached": bool(payload_dict.get("detached", payload_dict.get("background", background))),
        "restored": bool(payload_dict.get("restored", False)),
        "supports_stdin": bool(payload_dict.get("supports_stdin", not bool(payload_dict.get("detached", payload_dict.get("background", background))))),
        "reattach_supported": bool(payload_dict.get("background", background)),
        "recovery_supported": bool(payload_dict.get("background", background)),
        "lease_owner": "runtime_service" if bool(payload_dict.get("background", background)) else None,
    }
    metadata.update(_layout_metadata(detached_layout))
    return metadata
