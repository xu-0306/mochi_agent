from __future__ import annotations

from pathlib import Path

from mochi.config.schema import SecurityConfig
from mochi.security import (
    EffectivePolicyResolver,
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


def test_session_auto_review_expands_over_global_strict_preset() -> None:
    snapshot = EffectivePolicyResolver().resolve(
        SecurityConfig(autonomy_mode="strict"),
        session_overrides={"autonomy_mode": "auto_review"},
    )

    assert snapshot.autonomy_mode == "auto_review"
    assert snapshot.require_approval_for_file_write is False
    assert snapshot.require_approval_for_exec is False
    assert snapshot.file_read_scope == "workspace"
    assert snapshot.file_write_scope == "workspace"
    assert snapshot.source_chain == ("security_config", "session_override")


def test_legacy_runtime_resolver_also_expands_session_mode_preset() -> None:
    policy = resolve_runtime_permission_policy(
        SecurityConfig(autonomy_mode="strict"),
        overrides={"autonomy_mode": "auto_review"},
    )

    assert policy.autonomy_mode == "auto_review"
    assert policy.require_approval_for_file_write is False
    assert policy.require_approval_for_exec is False


def test_session_strict_expands_over_global_permissive_policy() -> None:
    snapshot = EffectivePolicyResolver().resolve(
        SecurityConfig(
            autonomy_mode="high_autonomy",
            file_read_scope="any",
            file_write_scope="any",
        ),
        session_overrides={"autonomy_mode": "strict"},
    )

    assert snapshot.autonomy_mode == "strict"
    assert snapshot.require_approval_for_file_write is True
    assert snapshot.require_approval_for_exec is True
    assert snapshot.file_read_scope == "workspace"
    assert snapshot.file_write_scope == "workspace"


def test_effective_policy_snapshot_is_deterministic() -> None:
    resolver = EffectivePolicyResolver()
    security = SecurityConfig(autonomy_mode="strict")

    first = resolver.resolve(
        security,
        session_overrides={"autonomy_mode": "auto_review"},
        hard_constraints={"hard_denies": ["network_write", "protected_path"]},
    )
    second = resolver.resolve(
        security,
        session_overrides={"autonomy_mode": "auto_review"},
        hard_constraints={"hard_denies": ["protected_path", "network_write"]},
    )

    assert first == second
    assert first.policy_snapshot_id == second.policy_snapshot_id
    assert first.policy_version == second.policy_version
    assert first.source_chain == (
        "security_config",
        "session_override",
        "hard_constraint",
    )


def test_hard_constraints_cannot_be_relaxed_by_session_or_run_policy() -> None:
    snapshot = EffectivePolicyResolver().resolve(
        SecurityConfig(
            autonomy_mode="high_autonomy",
            file_read_scope="any",
            file_write_scope="any",
        ),
        session_overrides={
            "autonomy_mode": "high_autonomy",
            "file_read_scope": "any",
            "file_write_scope": "any",
        },
        run_restrictions={"autonomy_mode": "auto_review"},
        hard_constraints={
            "require_approval_for_exec": True,
            "file_read_scope": "workspace",
            "file_write_scope": "workspace",
            "hard_denies": ["protected_path", "exec:shutdown"],
        },
    )

    assert snapshot.require_approval_for_exec is True
    assert snapshot.file_read_scope == "workspace"
    assert snapshot.file_write_scope == "workspace"
    assert snapshot.hard_denies == ("exec:shutdown", "protected_path")
    assert snapshot.source_chain[-2:] == ("run_restriction", "hard_constraint")


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


def test_runtime_policy_resolves_read_and_write_scope_independently() -> None:
    security = SecurityConfig(
        file_read_scope="workspace",
        file_write_scope="any",
    )

    policy = resolve_runtime_permission_policy(
        security,
        overrides={"file_read_scope": "any", "file_write_scope": "workspace"},
    )

    assert policy.file_read_scope == "any"
    assert policy.file_write_scope == "workspace"
    assert "file_ops_scope" not in policy.to_dict()


def test_runtime_policy_accepts_legacy_scope_override_without_reexporting_it() -> None:
    policy = resolve_runtime_permission_policy(
        SecurityConfig(),
        overrides={"file_ops_scope": "any"},
    )

    assert policy.file_read_scope == "any"
    assert policy.file_write_scope == "any"
    assert "file_ops_scope" not in policy.to_dict()

def test_infer_autonomy_mode_for_legacy_strict_config() -> None:
    security = SecurityConfig.model_validate(
        {
            "require_approval_for_shell": True,
            "require_approval_for_file_write": True,
            "file_ops_scope": "workspace",
        }
    )

    assert security.autonomy_mode == "strict"


def test_production_code_has_no_legacy_scope_consumers() -> None:
    source_root = Path(__file__).resolve().parents[1] / "mochi"
    forbidden = (
        "runtime_policy.file_ops_scope",
        "config.security.file_ops_scope",
        "security.file_ops_scope",
    )

    consumers = [
        f"{path.relative_to(source_root)}:{pattern}"
        for path in source_root.rglob("*.py")
        for pattern in forbidden
        if pattern in path.read_text(encoding="utf-8")
    ]

    assert consumers == []

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
