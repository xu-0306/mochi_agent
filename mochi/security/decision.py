"""Shared security decision helpers for tool/runtime policy handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

SecurityDecisionAction = Literal["allow", "deny", "require_approval"]
ApprovalKind = Literal["exec", "shell", "file_write", "file_edit", "apply_patch", "other"]
ApprovalScope = Literal[
    "workspace",
    "protected_path",
    "dangerous_command",
    "task_resume",
    "task_isolation",
    "sandbox",
]


@dataclass(frozen=True)
class SecurityDecision:
    """Structured policy outcome shared by tools, runtime, and UI surfaces."""

    action: SecurityDecisionAction
    reason: str | None = None
    approval_kind: ApprovalKind = "other"
    approval_scope: ApprovalScope = "workspace"
    replay_safe: bool = False
    policy_source: str = "static_policy"

    @property
    def requires_approval(self) -> bool:
        return self.action == "require_approval"

    def to_metadata(self) -> dict[str, Any]:
        """Serialize the decision into additive tool/runtime metadata."""
        payload: dict[str, Any] = {
            "security_decision": self.action,
            "approval_kind": self.approval_kind,
            "approval_scope": self.approval_scope,
            "replay_safe": self.replay_safe,
            "policy_source": self.policy_source,
        }
        if self.requires_approval:
            payload["requires_approval"] = True
        if self.reason:
            payload["reason"] = self.reason
        return payload

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any] | None) -> SecurityDecision | None:
        """Best-effort reconstruction from previously serialized metadata."""
        if not isinstance(metadata, dict):
            return None

        raw_action = metadata.get("security_decision")
        action: SecurityDecisionAction | None = None
        if raw_action in {"allow", "deny", "require_approval"}:
            action = raw_action
        elif metadata.get("requires_approval") is True:
            action = "require_approval"
        elif any(
            key in metadata
            for key in ("approval_kind", "approval_scope", "replay_safe", "reason", "policy_source")
        ):
            action = "deny"

        if action is None:
            return None

        approval_kind = metadata.get("approval_kind")
        if approval_kind not in {"exec", "shell", "file_write", "file_edit", "apply_patch", "other"}:
            approval_kind = "other"

        approval_scope = metadata.get("approval_scope")
        if approval_scope not in {
            "workspace",
            "protected_path",
            "dangerous_command",
            "task_resume",
            "task_isolation",
            "sandbox",
        }:
            approval_scope = "workspace"

        return cls(
            action=action,
            reason=metadata.get("reason") if isinstance(metadata.get("reason"), str) else None,
            approval_kind=approval_kind,
            approval_scope=approval_scope,
            replay_safe=bool(metadata.get("replay_safe", False)),
            policy_source=(
                metadata.get("policy_source")
                if isinstance(metadata.get("policy_source"), str)
                else "static_policy"
            ),
        )


def allow_security_decision(
    *,
    reason: str | None = None,
    approval_kind: ApprovalKind = "other",
    approval_scope: ApprovalScope = "workspace",
    replay_safe: bool = True,
    policy_source: str = "static_policy",
) -> SecurityDecision:
    return SecurityDecision(
        action="allow",
        reason=reason,
        approval_kind=approval_kind,
        approval_scope=approval_scope,
        replay_safe=replay_safe,
        policy_source=policy_source,
    )


def deny_security_decision(
    *,
    reason: str,
    approval_kind: ApprovalKind = "other",
    approval_scope: ApprovalScope = "workspace",
    replay_safe: bool = False,
    policy_source: str = "static_policy",
) -> SecurityDecision:
    return SecurityDecision(
        action="deny",
        reason=reason,
        approval_kind=approval_kind,
        approval_scope=approval_scope,
        replay_safe=replay_safe,
        policy_source=policy_source,
    )


def require_approval_decision(
    *,
    reason: str,
    approval_kind: ApprovalKind = "other",
    approval_scope: ApprovalScope = "workspace",
    replay_safe: bool = True,
    policy_source: str = "runtime_policy",
) -> SecurityDecision:
    return SecurityDecision(
        action="require_approval",
        reason=reason,
        approval_kind=approval_kind,
        approval_scope=approval_scope,
        replay_safe=replay_safe,
        policy_source=policy_source,
    )


def with_task_isolation_scope(
    decision: SecurityDecision,
    *,
    task_sandbox_dir: str | None,
) -> SecurityDecision:
    """Promote workspace-style approvals into task-isolation scope when sandboxed."""
    if not task_sandbox_dir:
        return decision
    if decision.approval_scope != "workspace":
        return decision
    return SecurityDecision(
        action=decision.action,
        reason=decision.reason,
        approval_kind=decision.approval_kind,
        approval_scope="task_isolation",
        replay_safe=decision.replay_safe,
        policy_source=decision.policy_source,
    )
