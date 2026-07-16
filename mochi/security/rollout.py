"""Pure protected-workspace rollout status projection."""

from __future__ import annotations

from typing import Literal, TypedDict

from mochi.config.schema import SandboxConfig, SecurityConfig


class ChangeContractCapabilities(TypedDict):
    legacy_file_mutation: bool
    edited_patch_replay: bool
    contract_enforcement: bool


class ChangeContractRollout(TypedDict):
    mode: Literal["observe", "enforce"]
    backend: str
    contract_available: bool
    capabilities: ChangeContractCapabilities
    configured_policy_decision: str
    enforcement_active: bool
    effective_file_behavior: str
    effective_undo_behavior: str
    status: str
    degraded: bool
    degraded_reason: str | None
    shadow_decision: str | None


class SandboxCapabilities(TypedDict):
    exec_containment: bool


class SandboxRollout(TypedDict):
    mode: Literal["off", "preferred", "required"]
    backend: None
    backend_available: bool
    capabilities: SandboxCapabilities
    configured_policy_decision: str
    enforcement_active: bool
    effective_exec_behavior: str
    status: str
    degraded: bool
    degraded_reason: str | None
    host_execution_allowed: bool


class ProtectedWorkspaceRollout(TypedDict):
    session_id: str | None
    change_contract: ChangeContractRollout
    sandbox: SandboxRollout


def project_change_contract_rollout(security: SecurityConfig) -> ChangeContractRollout:
    """Project configured file-contract intent without implying enforcement."""
    mode = security.change_contract_mode
    observing = mode == "observe"
    return {
        "mode": mode,
        "backend": "legacy_file_mutation",
        "contract_available": False,
        "capabilities": {
            "legacy_file_mutation": True,
            "edited_patch_replay": False,
            "contract_enforcement": False,
        },
        "configured_policy_decision": (
            "allow_legacy" if observing else "reject_contract_unavailable"
        ),
        "enforcement_active": False,
        "effective_file_behavior": "legacy_mutation_allowed",
        "effective_undo_behavior": "legacy_undo_available",
        "status": "not_enforced" if observing else "configured_unavailable",
        "degraded": not observing,
        "degraded_reason": (
            None if observing else "file_contract_pipeline_not_connected"
        ),
        "shadow_decision": (
            "would_reject_contract_unavailable" if observing else None
        ),
    }


def project_sandbox_rollout(sandbox: SandboxConfig) -> SandboxRollout:
    """Project configured sandbox intent while host execution remains effective."""
    mode = sandbox.mode
    configured = mode != "off"
    return {
        "mode": mode,
        "backend": None,
        "backend_available": False,
        "capabilities": {"exec_containment": False},
        "configured_policy_decision": (
            "allow_host"
            if mode == "off"
            else "prefer_sandbox_backend"
            if mode == "preferred"
            else "reject_backend_unavailable"
        ),
        "enforcement_active": False,
        "effective_exec_behavior": "host_execution_available",
        "status": "configured_unavailable" if configured else "not_enforced",
        "degraded": configured,
        "degraded_reason": (
            "sandbox_backend_unavailable_and_pipeline_not_connected"
            if configured
            else None
        ),
        "host_execution_allowed": True,
    }


def project_protected_workspace_rollout(
    security: SecurityConfig,
    sandbox: SandboxConfig,
    session_id: str | None = None,
) -> ProtectedWorkspaceRollout:
    """Combine independent file-contract and sandbox rollout projections."""
    return {
        "session_id": session_id,
        "change_contract": project_change_contract_rollout(security),
        "sandbox": project_sandbox_rollout(sandbox),
    }
