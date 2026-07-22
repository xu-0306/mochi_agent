"""Pure protected-workspace rollout status projection."""

from __future__ import annotations

from typing import Literal, TypedDict

from mochi.config.schema import SandboxConfig, SecurityConfig
from mochi.runtime.sandbox.base import HostSandboxBackend
from mochi.runtime.sandbox.selector import observed_platform_capabilities


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


class SandboxCapabilityRollout(TypedDict):
    exec_containment: bool
    filesystem: bool
    process: bool
    network: bool
    detached: bool


class SandboxRollout(TypedDict):
    mode: Literal["off", "preferred", "required"]
    backend: str
    backend_version: str
    backend_available: bool
    capabilities: SandboxCapabilityRollout
    configured_policy_decision: str
    enforcement_active: bool
    effective_exec_behavior: str
    status: str
    degraded: bool
    degraded_reason: str | None
    host_execution_allowed: bool
    last_probe_at: str | None


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
    """Project configured intent using observed backend capabilities."""
    mode = sandbox.mode
    observed = (
        HostSandboxBackend().probe()
        if mode == "off"
        else observed_platform_capabilities()
    )
    containment_available = observed.complete
    enforcement_active = mode != "off" and containment_available
    host_execution_allowed = mode != "required"
    if mode == "off":
        status = "not_enforced"
        effective_behavior = "host_execution_available"
        degraded = False
        degraded_reason = None
    elif containment_available:
        status = "enforced"
        effective_behavior = (
            "sandbox_execution_active"
            if mode == "preferred"
            else "sandbox_execution_required"
        )
        degraded = False
        degraded_reason = None
    elif mode == "preferred":
        status = "degraded"
        effective_behavior = "host_execution_degraded"
        degraded = True
        degraded_reason = observed.degraded_reason or "sandbox_backend_incomplete"
    else:
        status = "configured_unavailable"
        effective_behavior = "execution_blocked"
        degraded = True
        degraded_reason = observed.degraded_reason or "sandbox_backend_incomplete"
    return {
        "mode": mode,
        "backend": observed.backend,
        "backend_version": observed.version,
        "backend_available": observed.available,
        "capabilities": {
            "exec_containment": containment_available,
            "filesystem": observed.filesystem,
            "process": observed.process,
            "network": observed.network,
            "detached": observed.detached,
        },
        "configured_policy_decision": (
            "allow_host"
            if mode == "off"
            else "prefer_sandbox_backend"
            if mode == "preferred"
            else "reject_backend_unavailable"
        ),
        "enforcement_active": enforcement_active,
        "effective_exec_behavior": effective_behavior,
        "status": status,
        "degraded": degraded,
        "degraded_reason": degraded_reason,
        "host_execution_allowed": host_execution_allowed,
        "last_probe_at": observed.last_probe_at,
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
