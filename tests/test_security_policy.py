from __future__ import annotations

from mochi.config.schema import SecurityConfig
from mochi.security import (
    SecurityDecision,
    deny_security_decision,
    require_approval_decision,
    with_task_isolation_scope,
)
from mochi.security.policy import resolve_runtime_permission_policy
from mochi.utils.security import build_policy_metadata


def test_runtime_policy_defaults_follow_autonomy_mode() -> None:
    security = SecurityConfig(
        autonomy_mode="auto_review",
    )

    policy = resolve_runtime_permission_policy(security)

    assert policy.autonomy_mode == "auto_review"
    assert policy.require_approval_for_file_write is False
    assert policy.require_approval_for_exec is False


def test_runtime_policy_preserves_legacy_explicit_booleans() -> None:
    security = SecurityConfig(
        autonomy_mode="strict",
        require_approval_for_file_write=False,
        require_approval_for_exec=False,
    )

    policy = resolve_runtime_permission_policy(security)

    assert policy.require_approval_for_file_write is False
    assert policy.require_approval_for_exec is False


def test_runtime_policy_allows_runtime_overrides() -> None:
    security = SecurityConfig(
        autonomy_mode="strict",
    )

    policy = resolve_runtime_permission_policy(
        security,
        overrides={
            "require_approval_for_exec": False,
            "approved_tool_calls": [
                {"tool_name": "exec_command", "arguments": {"command": "dir"}}
            ],
        },
    )

    assert policy.require_approval_for_file_write is True
    assert policy.require_approval_for_exec is False


def test_infer_autonomy_mode_for_legacy_strict_config() -> None:
    security = SecurityConfig.model_validate(
        {
            "require_approval_for_shell": True,
            "require_approval_for_file_write": True,
            "file_ops_scope": "workspace",
        }
    )

    assert security.autonomy_mode == "strict"


def test_require_approval_decision_serializes_metadata() -> None:
    decision = require_approval_decision(
        reason="Exec commands require explicit approval.",
        approval_kind="exec",
        approval_scope="workspace",
        replay_safe=True,
        policy_source="runtime_policy",
    )

    metadata = decision.to_metadata()

    assert metadata == {
        "security_decision": "require_approval",
        "approval_kind": "exec",
        "approval_scope": "workspace",
        "replay_safe": True,
        "policy_source": "runtime_policy",
        "requires_approval": True,
        "reason": "Exec commands require explicit approval.",
    }


def test_security_decision_roundtrips_from_metadata() -> None:
    original = deny_security_decision(
        reason="Protected path denied by security policy.",
        approval_scope="protected_path",
        replay_safe=False,
        policy_source="path_policy",
    )

    restored = SecurityDecision.from_metadata(original.to_metadata())

    assert restored == original


def test_task_isolation_scope_only_applies_to_workspace_decisions() -> None:
    workspace_decision = require_approval_decision(
        reason="Exec commands require explicit approval.",
        approval_kind="exec",
        approval_scope="workspace",
        replay_safe=True,
        policy_source="runtime_policy",
    )
    isolated = with_task_isolation_scope(
        workspace_decision,
        task_sandbox_dir="/tmp/task-sandbox",
    )
    assert isolated.approval_scope == "task_isolation"
    assert isolated.action == workspace_decision.action
    assert isolated.policy_source == workspace_decision.policy_source

    protected_decision = deny_security_decision(
        reason="Protected path denied by security policy.",
        approval_scope="protected_path",
        replay_safe=False,
        policy_source="path_policy",
    )
    assert (
        with_task_isolation_scope(
            protected_decision,
            task_sandbox_dir="/tmp/task-sandbox",
        )
        == protected_decision
    )


def test_build_policy_metadata_adds_explicit_allow_ask_deny_state() -> None:
    decision = require_approval_decision(
        reason="Exec command requires approval.",
        approval_kind="exec",
        approval_scope="workspace",
        replay_safe=True,
        policy_source="runtime_policy",
    )

    metadata = build_policy_metadata(
        decision=decision,
        legacy_tool=True,
        preferred_tool="exec_command",
    )

    assert metadata["security_decision"] == "require_approval"
    assert metadata["policy_state"] == "ask"
    assert metadata["policy_reason"] == "Exec command requires approval."
    assert metadata["legacy_tool"] is True
    assert metadata["preferred_tool"] == "exec_command"
