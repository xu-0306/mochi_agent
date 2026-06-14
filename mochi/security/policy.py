"""Centralized runtime permission-policy derivation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from mochi.config.schema import SecurityConfig

AutonomyMode = Literal["trusted_workspace", "strict", "high_autonomy", "auto_review"]

_AUTONOMY_DEFAULTS: dict[AutonomyMode, dict[str, Any]] = {
    "strict": {
        "autonomy_mode": "strict",
        "require_approval_for_file_write": True,
        "require_approval_for_exec": True,
        "file_ops_scope": "workspace",
    },
    "trusted_workspace": {
        "autonomy_mode": "trusted_workspace",
        "require_approval_for_file_write": False,
        "require_approval_for_exec": True,
        "file_ops_scope": "workspace",
    },
    "auto_review": {
        "autonomy_mode": "auto_review",
        "require_approval_for_file_write": False,
        "require_approval_for_exec": False,
        "file_ops_scope": "workspace",
    },
    "high_autonomy": {
        "autonomy_mode": "high_autonomy",
        "require_approval_for_file_write": False,
        "require_approval_for_exec": False,
        "file_ops_scope": "any",
    },
}


@dataclass(frozen=True)
class RuntimePermissionPolicy:
    """Resolved runtime permission flags for tool execution."""

    autonomy_mode: AutonomyMode
    require_approval_for_file_write: bool
    require_approval_for_exec: bool
    file_ops_scope: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "autonomy_mode": self.autonomy_mode,
            "require_approval_for_file_write": self.require_approval_for_file_write,
            "require_approval_for_exec": self.require_approval_for_exec,
            "file_ops_scope": self.file_ops_scope,
        }


def autonomy_mode_defaults(mode: AutonomyMode) -> dict[str, Any]:
    """Return the built-in defaults for one autonomy mode."""
    return dict(_AUTONOMY_DEFAULTS[mode])


def infer_autonomy_mode(
    *,
    require_approval_for_exec: bool,
    require_approval_for_file_write: bool,
    file_ops_scope: str,
) -> AutonomyMode:
    """Infer the closest autonomy preset for legacy configs without a mode."""
    if (
        require_approval_for_exec is False
        and require_approval_for_file_write is False
        and file_ops_scope == "any"
    ):
        return "high_autonomy"
    if (
        require_approval_for_exec is False
        and require_approval_for_file_write is False
        and file_ops_scope == "workspace"
    ):
        return "auto_review"
    if (
        require_approval_for_exec is True
        and require_approval_for_file_write is False
        and file_ops_scope == "workspace"
    ):
        return "trusted_workspace"
    return "strict"


def resolve_runtime_permission_policy(
    security: SecurityConfig,
    *,
    overrides: dict[str, Any] | None = None,
) -> RuntimePermissionPolicy:
    """Resolve effective runtime permission policy from security config."""
    effective_mode = security.autonomy_mode
    effective_file_write = security.require_approval_for_file_write
    effective_exec = security.require_approval_for_exec
    effective_scope = security.file_ops_scope

    if isinstance(overrides, dict):
        if isinstance(overrides.get("autonomy_mode"), str):
            effective_mode = overrides["autonomy_mode"]
        if isinstance(overrides.get("require_approval_for_file_write"), bool):
            effective_file_write = overrides["require_approval_for_file_write"]
        if isinstance(overrides.get("require_approval_for_exec"), bool):
            effective_exec = overrides["require_approval_for_exec"]
        if isinstance(overrides.get("file_ops_scope"), str):
            effective_scope = overrides["file_ops_scope"]

    return RuntimePermissionPolicy(
        autonomy_mode=effective_mode,
        require_approval_for_file_write=effective_file_write,
        require_approval_for_exec=effective_exec,
        file_ops_scope=effective_scope,
    )


def build_runtime_permission_policy_dict(
    security: SecurityConfig,
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convenience wrapper returning dict payload."""
    payload = resolve_runtime_permission_policy(security, overrides=overrides).to_dict()
    if isinstance(overrides, dict):
        for key, value in overrides.items():
            if key not in payload:
                payload[key] = value
    return payload
