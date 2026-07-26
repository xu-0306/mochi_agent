"""Exec runtime tool for foreground/background shell command execution."""

from __future__ import annotations

import hashlib
import inspect
import json
import logging
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from mochi.config import defaults
from mochi.runtime.approval_state_machine import derive_approval_binding
from mochi.runtime.approvals import (
    APPROVAL_OWNER_TASK_ID_KEY,
    ApprovalStore,
    InMemoryApprovalStore,
)
from mochi.runtime.exec_runtime import ExecRuntime
from mochi.runtime.exec_sessions import ExecSessionStatus, SessionPollResult
from mochi.runtime.sandbox import SandboxMode, SandboxPlan
from mochi.security import deny_security_decision
from mochi.security.auto_review import (
    AutoReviewDecision,
    AutoReviewFacts,
    AutoReviewVerificationError,
    auto_review_metadata,
    review_authorization_envelope,
    verify_auto_review_decision,
)
from mochi.security.file_contract import (
    AUTHORIZATION_ENVELOPE_SCHEMA_VERSION,
    AuthorizationContext,
    AuthorizationEnvelope,
    EnvVarHash,
    ExecRequest,
    ResourceLimits,
    capture_file_identity,
)
from mochi.security.policy import (
    effective_policy_snapshot_from_mapping,
    matching_tool_hard_deny,
    policy_projection_version,
)
from mochi.sessions.timeline_coordinator import (
    mark_context_side_effect_started,
    timeline_pending_operation_binding,
)
from mochi.tools.base import (
    BaseTool,
    ToolCancellationResult,
    ToolExecutionContext,
    ToolResult,
)
from mochi.utils.command_security import CommandSecurityPolicy, CommandSecurityResult
from mochi.utils.security import (
    build_policy_metadata,
    normalize_workspace_dir,
    resolve_path_in_workspace,
)

_SHARED_RUNTIME: ExecRuntime | None = None
_SHARED_APPROVAL_STORE: ApprovalStore | None = None
_logger = logging.getLogger(__name__)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _ordinary_chat_approval_context(
    context: ToolExecutionContext | None,
) -> dict[str, Any] | None:
    if context is None or not isinstance(context.state, Mapping):
        return None
    raw = context.state.get("ordinary_chat_approval_context")
    if not isinstance(raw, Mapping) or raw.get("source") != "ordinary_chat":
        return None
    return dict(raw)


async def _observe_ordinary_chat_approval(
    context: ToolExecutionContext | None,
    approval: Any,
) -> None:
    """Publish the post-commit ordinary-Chat observation when the Engine wired it."""

    if context is None or not isinstance(context.state, Mapping):
        return
    observer = context.state.get("tool_workflow_approval_observer")
    if not callable(observer):
        return
    try:
        result = observer(approval)
        if inspect.isawaitable(result):
            await result
    except Exception as exc:
        # The approval DB create committed before this optional projection.
        # Leave the caller's approval-pending result untouched; the runtime
        # reconciler can republish the durable row without executing a tool.
        _logger.warning(
            "ordinary-chat approval observation handoff requires repair: %s (%s)",
            getattr(approval, "approval_id", ""),
            type(exc).__name__,
        )


def _approval_payload_digest(value: Mapping[str, Any]) -> str:
    return _sha256(
        json.dumps(
            dict(value),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )


def _build_exec_authorization_envelope(
    *,
    command: str,
    shell: str | None,
    classification: CommandSecurityResult,
    workspace_root: Path,
    resolved_cwd: Path,
    normalized_env: dict[str, str] | None,
    sandbox_permissions: str,
    autonomy_mode: str,
    effective_require_approval: bool,
    effective_policy_version: str | None,
    context: ToolExecutionContext | None,
    owner_task_id: str | None,
    sandbox_plan: SandboxPlan,
) -> AuthorizationEnvelope:
    parsed_tokens = classification.parsed_tokens
    if not parsed_tokens:
        raise ValueError("canonical exec authorization requires parsed command tokens")
    workspace_identity = capture_file_identity(workspace_root)
    requester_id = (
        f"runtime-task:{owner_task_id}" if owner_task_id else "runtime-service"
    )
    authorization_context = AuthorizationContext(
        requester_id=requester_id,
        session_id=(
            str(context.session_id).strip()
            if context is not None and context.session_id
            else owner_task_id or "runtime-session"
        ),
        task_id=owner_task_id,
        workspace_root=str(workspace_root),
        workspace_identity=workspace_identity,
    )
    env_hashes = tuple(
        EnvVarHash(key=str(key), value_sha256=_sha256(value))
        for key, value in sorted((normalized_env or {}).items())
    )
    policy_version = effective_policy_version or policy_projection_version(
        "exec-policy-fallback",
        {
            "autonomy_mode": autonomy_mode,
            "classification_action": classification.action,
            "classification_rule_id": classification.rule_id,
            "require_approval_for_exec": effective_require_approval,
            "reviewer_contract": "deterministic-v1",
        },
    )
    return AuthorizationEnvelope(
        schema_version=AUTHORIZATION_ENVELOPE_SCHEMA_VERSION,
        kind="exec",
        context=authorization_context,
        policy_version=policy_version,
        file_request=None,
        exec_request=ExecRequest(
            command_utf8_sha256=_sha256(command),
            shell=shell,
            executable=sandbox_plan.executable,
            argv=sandbox_plan.argv,
            resolved_cwd=str(resolved_cwd),
            env=env_hashes,
            network_policy=sandbox_plan.network_policy,
            resource_limits=ResourceLimits(
                timeout_seconds=max(
                    1,
                    int(math.ceil(sandbox_plan.resource_limits.timeout_milliseconds / 1000)),
                ),
                memory_limit_mb=sandbox_plan.resource_limits.memory_limit_mb,
                output_limit_bytes=sandbox_plan.resource_limits.output_limit_bytes,
            ),
            requested_escalation=sandbox_permissions,
            sandbox_backend=sandbox_plan.backend,
            sandbox_capability_plan_digest=sandbox_plan.digest,
        ),
    )


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


def _sandbox_metadata(plan: SandboxPlan) -> dict[str, Any]:
    capabilities = plan.capabilities
    enforcement_active = bool(plan.mode != "off" and capabilities.complete)
    return {
        "sandbox_mode": plan.mode,
        "sandbox_backend": plan.backend,
        "sandbox_backend_version": plan.backend_version,
        "sandbox_plan_digest": plan.digest,
        "sandbox_enforcement_active": enforcement_active,
        "sandbox_degraded": bool(plan.mode == "preferred" and not enforcement_active),
        "sandbox_degraded_reason": capabilities.degraded_reason,
        "sandbox_capabilities": capabilities.to_dict(),
    }


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
        sandbox_mode: SandboxMode = "off",
    ) -> None:
        self._runtime = runtime or get_shared_exec_runtime()
        self._approval_store = approval_store or get_shared_exec_approval_store()
        self._workspace_dir = normalize_workspace_dir(workspace_dir or defaults.default_workspace_dir())
        self._command_rules = [dict(item) for item in (command_rules or []) if isinstance(item, dict)]
        self._allowed_env_vars = [str(item) for item in (allowed_env_vars or [])]
        self._require_approval = bool(require_approval)
        self._default_timeout_sec = max(1, int(default_timeout_sec))
        self._sandbox_mode: SandboxMode = sandbox_mode

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

    @property
    def is_cancellable(self) -> bool:
        return True

    @property
    def supports_timeline_side_effect_boundary(self) -> bool:
        return True

    @property
    def supports_timeline_approval_revocation(self) -> bool:
        return True

    @property
    def timeline_approval_mode(self) -> str:
        return "continuable"

    def revoke_timeline_approval(self, approval_id: str, *, reason: str) -> bool:
        return self._approval_store.supersede(approval_id, reason=reason).status == "superseded"

    def validates_timeline_approval_binding(
        self,
        approval_id: str,
        *,
        operation_id: str,
        arguments_digest: str,
        call_id: str,
    ) -> bool:
        approval = self._approval_store.get(approval_id)
        if approval is None or approval.status != "pending":
            return False
        payload = approval.command_payload
        checkpoint = (
            payload.get("ordinary_chat_checkpoint")
            if isinstance(payload, Mapping)
            else None
        )
        return bool(
            approval.metadata.get("operation_id") == operation_id
            and approval.metadata.get("arguments_digest") == arguments_digest
            and approval.metadata.get("timeline_call_id") == call_id
            and isinstance(payload, Mapping)
            and payload.get("operation_id") == operation_id
            and payload.get("arguments_digest") == arguments_digest
            and payload.get("timeline_call_id") == call_id
            and isinstance(checkpoint, Mapping)
            and checkpoint.get("source") == "ordinary_chat"
            and checkpoint.get("operation_id") == operation_id
            and checkpoint.get("arguments_digest") == arguments_digest
            and checkpoint.get("timeline_call_id") == call_id
        )

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
        permission_policy = (
            context.permission_policy
            if context is not None and isinstance(context.permission_policy, Mapping)
            else {}
        )
        effective_snapshot = effective_policy_snapshot_from_mapping(permission_policy)
        effective_require_approval = self._require_approval
        if isinstance(permission_policy.get("require_approval_for_exec"), bool):
            effective_require_approval = bool(permission_policy["require_approval_for_exec"])
        autonomy_mode = str(permission_policy.get("autonomy_mode") or "").strip().lower()
        policy_metadata = {
            "policy_snapshot_id": (
                effective_snapshot.policy_snapshot_id if effective_snapshot is not None else None
            ),
            "effective_policy_version": (
                effective_snapshot.policy_version if effective_snapshot is not None else None
            ),
            "effective_policy_source": (
                "effective_policy_snapshot"
                if effective_snapshot is not None
                else "constructor_or_legacy_context_fallback"
            ),
        }
        hard_deny = matching_tool_hard_deny(
            effective_snapshot or permission_policy,
            tool_name=self.name,
            capability="exec",
        )
        if hard_deny is not None:
            reason = f"Execution is prohibited by hard policy selector: {hard_deny}."
            decision = deny_security_decision(
                reason=reason,
                approval_kind="exec",
                approval_scope="dangerous_command",
                replay_safe=False,
                policy_source="effective_policy_hard_deny",
            )
            return ToolResult(
                error=reason,
                metadata={
                    "status": "denied",
                    "hard_deny": hard_deny,
                    "tool_name": self.name,
                    **policy_metadata,
                    **decision.to_metadata(),
                },
                retryable=False,
            )
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

        if timeout is None:
            timeout_sec = float(self._default_timeout_sec)
        else:
            try:
                timeout_sec = float(timeout)
            except (TypeError, ValueError):
                return ToolResult(error="`timeout` must be a number.")
            if timeout_sec <= 0:
                return ToolResult(error="`timeout` must be greater than 0.")

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
                    **policy_metadata,
                    **_build_classification_metadata(
                        classification,
                        shell=shell,
                        decision=decision,
                    ),
                },
                retryable=False,
            )

        try:
            sandbox_plan = self._runtime.build_sandbox_plan(
                command=command,
                mode=self._sandbox_mode,
                shell=shell,
                cwd=resolved_cwd,
                env=normalized_env,
                timeout_sec=timeout_sec,
                requested_escalation=sandbox_permissions,
                workspace_root=workspace_root,
                background=background,
                tty=tty,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            return ToolResult(
                error=f"Exec sandbox plan could not be created: {exc}",
                metadata={
                    "status": "denied",
                    "sandbox_mode": self._sandbox_mode,
                    "sandbox_enforcement_unavailable": True,
                    **policy_metadata,
                },
                retryable=False,
            )

        owner_task_id = str(
            permission_policy.get(APPROVAL_OWNER_TASK_ID_KEY) or ""
        ).strip() or None
        review_envelope: AuthorizationEnvelope | None = None
        review_decision: AutoReviewDecision | None = None
        if (
            autonomy_mode in {"auto_review", "high_autonomy"}
            and not effective_require_approval
        ):
            try:
                review_envelope = _build_exec_authorization_envelope(
                    command=command,
                    shell=shell,
                    classification=classification,
                    workspace_root=workspace_root,
                    resolved_cwd=resolved_cwd,
                    normalized_env=normalized_env,
                    sandbox_permissions=sandbox_permissions,
                    autonomy_mode=autonomy_mode,
                    effective_require_approval=effective_require_approval,
                    effective_policy_version=(
                        effective_snapshot.policy_version
                        if effective_snapshot is not None
                        else None
                    ),
                    context=context,
                    owner_task_id=owner_task_id,
                    sandbox_plan=sandbox_plan,
                )
                review_decision = review_authorization_envelope(
                    review_envelope,
                    facts=AutoReviewFacts(
                        policy_action=classification.action,
                        policy_rule_id=classification.rule_id,
                        protected_path=classification.rule_id == "protected_path",
                        workspace_escape=classification.rule_id == "workspace_escape",
                        unknown_shell_parse=classification.rule_id == "parse_error",
                    ),
                )
            except (OSError, RuntimeError, ValueError) as exc:
                return ToolResult(
                    error=f"Auto review could not bind the authorization envelope: {exc}",
                    metadata={
                        "status": "denied",
                        **policy_metadata,
                        "auto_review_decision": "deny",
                        "auto_review_reason_codes": ["authorization_envelope_invalid"],
                    },
                    retryable=False,
                )

        auto_review_policy_allow = bool(
            classification.action == "ask"
            and review_decision is not None
            and review_decision.decision == "allow"
        )
        if review_decision is not None and review_decision.decision == "deny":
            return ToolResult(
                error="Auto review denied the exec authorization envelope.",
                metadata={
                    "status": "denied",
                    **policy_metadata,
                    **auto_review_metadata(review_decision),
                    **_build_classification_metadata(classification, shell=shell),
                },
                retryable=False,
            )

        needs_approval = (
            effective_require_approval
            or (classification.action == "ask" and not auto_review_policy_allow)
            or sandbox_permissions == "require_escalated"
            or (
                review_decision is not None
                and review_decision.decision == "require_approval"
            )
        )
        if needs_approval:
            reason = (
                justification
                or classification.reason
                or "Exec command requires approval before execution."
            )
            approval_id = f"exec-approval-{uuid4().hex[:12]}"
            suggested_rule = _build_suggested_rule(classification, shell=shell)
            ordinary_chat_context = _ordinary_chat_approval_context(context)
            execution_arguments = {
                "command": command,
                "shell": shell,
                "workdir": str(resolved_cwd),
                "env": normalized_env,
                "timeout": timeout_sec,
                "background": background,
                "tty": tty,
                "sandbox_permissions": sandbox_permissions,
                "justification": justification,
                "log_path": resolved_layout.get("log_path"),
                "checkpoint_dir": resolved_layout.get("checkpoint_dir"),
                "detached_layout": dict(resolved_layout),
            }
            try:
                timeline_binding = timeline_pending_operation_binding(
                    context,
                    tool_name=self.name,
                )
            except Exception as exc:
                return ToolResult(
                    error=f"timeline_approval_binding_invalid: {exc}",
                    metadata={
                        "status": "timeline_approval_binding_invalid",
                        "tool_name": self.name,
                        "timeline_fail_closed": True,
                    },
                    retryable=False,
                )
            if timeline_binding is not None:
                if ordinary_chat_context is None:
                    return ToolResult(
                        error=(
                            "timeline_approval_binding_invalid: ordinary Chat approval "
                            "context is missing."
                        ),
                        metadata={
                            "status": "timeline_approval_binding_invalid",
                            "tool_name": self.name,
                            "timeline_fail_closed": True,
                        },
                        retryable=False,
                    )
                resume_cursor = ordinary_chat_context.get("resume_cursor")
                cursor_call_id = (
                    str(resume_cursor.get("tool_call_id") or "").strip()
                    if isinstance(resume_cursor, Mapping)
                    else ""
                )
                call_id = str(timeline_binding["call_id"])
                if cursor_call_id != call_id:
                    return ToolResult(
                        error=(
                            "timeline_approval_binding_invalid: ordinary Chat resume cursor "
                            "does not match the precommitted tool call."
                        ),
                        metadata={
                            "status": "timeline_approval_binding_invalid",
                            "tool_name": self.name,
                            "timeline_fail_closed": True,
                        },
                        retryable=False,
                    )
                approval_arguments = dict(timeline_binding["arguments"])
                arguments_digest = str(timeline_binding["arguments_digest"])
                operation_id = str(timeline_binding["operation_id"])
            else:
                approval_arguments = dict(execution_arguments)
                arguments_digest = _approval_payload_digest(
                    {"tool_name": self.name, "arguments": approval_arguments}
                )
                operation_id = f"exec-operation-{uuid4().hex}"
                call_id = None
            inventory_version = _approval_payload_digest(
                {"tool_name": self.name, "parameters_schema": self.parameters_schema}
            )
            approval_payload = {
                "command": command,
                "shell": shell,
                "workdir": str(resolved_cwd),
                "env": normalized_env,
                "timeout_sec": timeout_sec,
                "background": background,
                "tty": tty,
                "sandbox_permissions": sandbox_permissions,
                "log_path": resolved_layout.get("log_path"),
                "checkpoint_dir": resolved_layout.get("checkpoint_dir"),
                "detached_layout": dict(resolved_layout),
                "approval_state": "approved",
                "sandbox_plan": sandbox_plan.to_dict(),
                "tool_name": self.name,
                "policy_snapshot_id": policy_metadata["policy_snapshot_id"],
                "effective_policy_version": policy_metadata[
                    "effective_policy_version"
                ],
                "normalized_arguments": approval_arguments,
                "execution_arguments": execution_arguments,
                "arguments_digest": arguments_digest,
                "operation_id": operation_id,
                "call_id": call_id,
                "timeline_call_id": call_id,
                "tool_inventory_version": inventory_version,
                "workspace_dir": str(workspace_root),
                "session_id": context.session_id if context is not None else None,
                "permission_policy": dict(permission_policy),
            }
            if ordinary_chat_context is not None:
                resume_cursor = ordinary_chat_context.get("resume_cursor")
                approval_payload["ordinary_chat_checkpoint"] = {
                    "schema_version": 1,
                    "source": "ordinary_chat",
                    "session_id": ordinary_chat_context.get("session_id"),
                    "turn_id": ordinary_chat_context.get("turn_id"),
                    "resume_cursor": dict(resume_cursor) if isinstance(resume_cursor, Mapping) else {},
                    "resolved_workspace_dir": str(workspace_root),
                    "resolved_target": str(resolved_cwd),
                    "operation_id": operation_id,
                    "call_id": call_id,
                    "timeline_call_id": call_id,
                    "tool_name": self.name,
                    "normalized_arguments": approval_arguments,
                    "execution_arguments": execution_arguments,
                    "arguments_digest": arguments_digest,
                    "policy_snapshot_id": policy_metadata["policy_snapshot_id"],
                    "policy_version": policy_metadata["effective_policy_version"],
                    "inventory_version": inventory_version,
                    "sandbox_plan_digest": sandbox_plan.to_dict().get("plan_digest"),
                    "sandbox_authorization_digest": sandbox_plan.approval_digest,
                    "react_continuation": (
                        dict(ordinary_chat_context["react_continuation"])
                        if isinstance(ordinary_chat_context.get("react_continuation"), Mapping)
                        else None
                    ),
                }
            requester_id, request_digest, context_digest = derive_approval_binding(
                requester_id=(
                    f"runtime-task:{owner_task_id}"
                    if owner_task_id
                    else f"runtime-session:{context.session_id}"
                    if ordinary_chat_context is not None and context is not None and context.session_id
                    else "runtime-service"
                ),
                request={
                    "command": command,
                    "shell": shell or "auto",
                    "scope": "dangerous_command",
                    "command_payload": approval_payload,
                },
                authorization_context={
                    "source": "exec_command",
                    "owner_task_id": owner_task_id or None,
                    "workspace_root": str(workspace_root),
                    "resolved_cwd": str(resolved_cwd),
                    "policy_snapshot_id": policy_metadata["policy_snapshot_id"],
                    "policy_version": policy_metadata["effective_policy_version"],
                    "tool_inventory_version": inventory_version,
                    "call_id": call_id,
                    "timeline_call_id": call_id,
                    "operation_id": operation_id,
                },
            )
            request = self._approval_store.create(
                approval_id=approval_id,
                command=command,
                shell=(shell or "auto"),
                scope="dangerous_command",
                reason=reason,
                metadata={
                    "tool_name": self.name,
                    **policy_metadata,
                    "policy_state": "ask",
                    "policy_reason": classification.reason,
                    "rule_id": classification.rule_id,
                    "suggested_rule": suggested_rule,
                    "operation_id": operation_id,
                    "arguments_digest": arguments_digest,
                    "tool_inventory_version": inventory_version,
                    **(
                        {
                            "approval_source": "ordinary_chat",
                            "call_id": call_id,
                            "timeline_call_id": call_id,
                            "resume_cursor": (
                                dict(ordinary_chat_context.get("resume_cursor") or {})
                                if isinstance(ordinary_chat_context.get("resume_cursor"), Mapping)
                                else {}
                            ),
                            "resolved_workspace_dir": str(workspace_root),
                            "resolved_target": str(resolved_cwd),
                        }
                        if ordinary_chat_context is not None
                        else {}
                    ),
                    **(
                        auto_review_metadata(review_decision)
                        if review_decision is not None
                        else {}
                    ),
                    **(
                        {APPROVAL_OWNER_TASK_ID_KEY: owner_task_id}
                        if owner_task_id
                        else {}
                    ),
                },
                command_payload=approval_payload,
                requester_id=requester_id,
                request_digest=request_digest,
                context_digest=context_digest,
            )
            await _observe_ordinary_chat_approval(context, request)
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
                    "tool_name": self.name,
                    "operation_id": operation_id,
                    "call_id": call_id,
                    "timeline_call_id": call_id,
                    "arguments_digest": arguments_digest,
                    "tool_inventory_version": inventory_version,
                    **(
                        {
                            "resume_cursor": (
                                dict(ordinary_chat_context.get("resume_cursor") or {})
                                if isinstance(ordinary_chat_context.get("resume_cursor"), Mapping)
                                else {}
                            ),
                            "approval_source": "ordinary_chat",
                            "call_id": call_id,
                            "timeline_call_id": call_id,
                        }
                        if ordinary_chat_context is not None
                        else {}
                    ),
                    **policy_metadata,
                    **(
                        auto_review_metadata(review_decision)
                        if review_decision is not None
                        else {}
                    ),
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
                    **_sandbox_metadata(sandbox_plan),
                },
                retryable=True,
            )

        if review_decision is not None and review_envelope is not None:
            try:
                verify_auto_review_decision(
                    review_decision,
                    review_envelope,
                    current_workspace_identity=capture_file_identity(workspace_root),
                )
            except (AutoReviewVerificationError, OSError) as exc:
                return ToolResult(
                    error=f"Auto review verification failed before execution: {exc}",
                    metadata={
                        "status": "denied",
                        **policy_metadata,
                        **auto_review_metadata(review_decision),
                        "auto_review_execution_verified": False,
                    },
                    retryable=False,
                )

        try:
            async def _bind_active_session(poll: SessionPollResult) -> None:
                if context is None or context.active_tool_controller is None:
                    return
                session_id = str(poll.session_id or "").strip()
                if not session_id:
                    return

                async def _cancel_active_session() -> ToolCancellationResult:
                    cancelled_poll = await self._runtime.kill_session(session_id)
                    if cancelled_poll is None:
                        return ToolCancellationResult(
                            cancelled=False,
                            reason="tool_already_completed",
                            session_id=session_id,
                        )
                    if cancelled_poll.status == ExecSessionStatus.KILLED:
                        return ToolCancellationResult(
                            cancelled=True,
                            reason="tool_cancelled",
                            session_id=session_id,
                            metadata={"status": cancelled_poll.status.value},
                        )
                    return ToolCancellationResult(
                        cancelled=False,
                        reason="tool_already_completed",
                        session_id=session_id,
                        metadata={"status": cancelled_poll.status.value},
                    )

                await context.active_tool_controller.bind_cancel_callback(
                    session_id=session_id,
                    callback=_cancel_active_session,
                )

            await mark_context_side_effect_started(context)
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
                session_started_callback=_bind_active_session,
                sandbox_plan=sandbox_plan,
            )
        except Exception as exc:
            return ToolResult(
                error=f"Exec command failed: {exc}",
                metadata={
                    **policy_metadata,
                    **_sandbox_metadata(sandbox_plan),
                    "timeline_result_disposition": "unknown",
                },
            )

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
        metadata.update(policy_metadata)
        metadata.update(_sandbox_metadata(sandbox_plan))
        policy_reason = classification.reason
        if auto_review_policy_allow:
            policy_reason = (
                "Auto-review allowed a command that would normally require approval. "
                f"{classification.reason}"
            )
        metadata.update(
            _build_classification_metadata(
                classification,
                shell=shell,
                policy_state="allow",
                policy_reason=policy_reason,
            )
        )
        if review_decision is not None:
            metadata.update(auto_review_metadata(review_decision))
            metadata["auto_review_execution_verified"] = True
        metadata["auto_reviewed_policy_ask"] = auto_review_policy_allow
        metadata["approval_id"] = None
        metadata["timeline_result_disposition"] = (
            "failed"
            if result.status in {
                ExecSessionStatus.KILLED,
                ExecSessionStatus.FAILED,
                ExecSessionStatus.TIMED_OUT,
            }
            else "succeeded"
        )
        if result.status == ExecSessionStatus.KILLED:
            payload["runtime_status"] = result.status.value
            payload["status"] = "cancelled"
            metadata["runtime_status"] = result.status.value
            metadata["status"] = "cancelled"
            metadata["cancelled"] = True
            return ToolResult(output=payload, metadata=metadata)
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
