"""Centralized runtime permission-policy derivation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Mapping

if TYPE_CHECKING:
    from mochi.config.schema import SecurityConfig

AutonomyMode = Literal["trusted_workspace", "strict", "high_autonomy", "auto_review"]
PolicyScope = Literal["workspace", "any"]

_AUTONOMY_MODES = frozenset(
    {"trusted_workspace", "strict", "high_autonomy", "auto_review"}
)
_POLICY_SCOPES = frozenset({"workspace", "any"})
_POLICY_FIELDS = frozenset(
    {
        "autonomy_mode",
        "require_approval_for_file_write",
        "require_approval_for_exec",
        "file_ops_scope",
        "file_read_scope",
        "file_write_scope",
        "hard_denies",
    }
)


def policy_projection_version(namespace: str, projection: dict[str, Any]) -> str:
    """Return a deterministic version for the exact policy inputs used by a decision."""

    payload = json.dumps(
        projection,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"{namespace}-v1:{hashlib.sha256(payload).hexdigest()}"

_AUTONOMY_DEFAULTS: dict[AutonomyMode, dict[str, Any]] = {
    "strict": {
        "autonomy_mode": "strict",
        "require_approval_for_file_write": True,
        "require_approval_for_exec": True,
        "file_read_scope": "workspace",
        "file_write_scope": "workspace",
    },
    "trusted_workspace": {
        "autonomy_mode": "trusted_workspace",
        "require_approval_for_file_write": False,
        "require_approval_for_exec": True,
        "file_read_scope": "workspace",
        "file_write_scope": "workspace",
    },
    "auto_review": {
        "autonomy_mode": "auto_review",
        "require_approval_for_file_write": False,
        "require_approval_for_exec": False,
        "file_read_scope": "workspace",
        "file_write_scope": "workspace",
    },
    "high_autonomy": {
        "autonomy_mode": "high_autonomy",
        "require_approval_for_file_write": False,
        "require_approval_for_exec": False,
        "file_read_scope": "workspace",
        "file_write_scope": "workspace",
    },
}


@dataclass(frozen=True)
class RuntimePermissionPolicy:
    """Resolved runtime permission flags for tool execution."""

    autonomy_mode: AutonomyMode
    require_approval_for_file_write: bool
    require_approval_for_exec: bool
    file_read_scope: str
    file_write_scope: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "autonomy_mode": self.autonomy_mode,
            "require_approval_for_file_write": self.require_approval_for_file_write,
            "require_approval_for_exec": self.require_approval_for_exec,
            "file_read_scope": self.file_read_scope,
            "file_write_scope": self.file_write_scope,
        }


@dataclass(frozen=True)
class EffectivePolicySnapshot:
    """Immutable, deterministic effective policy used by one runtime decision.

    ``source_chain`` is ordered from the base policy through increasingly
    authoritative overlays.  Session overlays may select a different autonomy
    preset.  Run restrictions and hard constraints are applied last and can
    only make the policy more restrictive.
    """

    policy_snapshot_id: str
    policy_version: str
    source_chain: tuple[str, ...]
    autonomy_mode: AutonomyMode
    require_approval_for_file_write: bool
    require_approval_for_exec: bool
    file_read_scope: PolicyScope
    file_write_scope: PolicyScope
    hard_denies: tuple[str, ...] = ()

    def to_runtime_policy(self) -> RuntimePermissionPolicy:
        """Return the legacy runtime view without snapshot metadata."""
        return RuntimePermissionPolicy(
            autonomy_mode=self.autonomy_mode,
            require_approval_for_file_write=self.require_approval_for_file_write,
            require_approval_for_exec=self.require_approval_for_exec,
            file_read_scope=self.file_read_scope,
            file_write_scope=self.file_write_scope,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "policy_snapshot_id": self.policy_snapshot_id,
            "policy_version": self.policy_version,
            "source_chain": list(self.source_chain),
            "autonomy_mode": self.autonomy_mode,
            "require_approval_for_file_write": self.require_approval_for_file_write,
            "require_approval_for_exec": self.require_approval_for_exec,
            "file_read_scope": self.file_read_scope,
            "file_write_scope": self.file_write_scope,
            "hard_denies": list(self.hard_denies),
        }


def effective_policy_snapshot_from_mapping(
    value: Mapping[str, Any] | None,
) -> EffectivePolicySnapshot | None:
    """Parse the complete server-issued policy snapshot carried by one tool call.

    Partial legacy policy mappings deliberately return ``None``. Callers may
    still consume individual legacy flags as compatibility fallbacks, but must
    not present those projections as a server-issued effective snapshot.
    """
    if not isinstance(value, Mapping):
        return None
    snapshot_id = value.get("policy_snapshot_id")
    policy_version = value.get("policy_version")
    source_chain = value.get("source_chain")
    autonomy_mode = _normalized_mode(value.get("autonomy_mode"))
    read_scope = _normalized_scope(value.get("file_read_scope"))
    write_scope = _normalized_scope(value.get("file_write_scope"))
    file_approval = value.get("require_approval_for_file_write")
    exec_approval = value.get("require_approval_for_exec")
    if not (
        isinstance(snapshot_id, str)
        and snapshot_id.strip()
        and isinstance(policy_version, str)
        and policy_version.strip()
        and isinstance(source_chain, (list, tuple))
        and all(isinstance(item, str) and item.strip() for item in source_chain)
        and autonomy_mode is not None
        and read_scope is not None
        and write_scope is not None
        and isinstance(file_approval, bool)
        and isinstance(exec_approval, bool)
    ):
        return None
    return EffectivePolicySnapshot(
        policy_snapshot_id=snapshot_id.strip(),
        policy_version=policy_version.strip(),
        source_chain=tuple(item.strip() for item in source_chain),
        autonomy_mode=autonomy_mode,
        require_approval_for_file_write=file_approval,
        require_approval_for_exec=exec_approval,
        file_read_scope=read_scope,
        file_write_scope=write_scope,
        hard_denies=_normalized_hard_denies(value.get("hard_denies")),
    )


def matching_tool_hard_deny(
    value: Mapping[str, Any] | EffectivePolicySnapshot | None,
    *,
    tool_name: str,
    capability: str | None = None,
) -> str | None:
    """Return the deterministic hard-deny selector matching a concrete tool."""
    if isinstance(value, EffectivePolicySnapshot):
        denies = value.hard_denies
    elif isinstance(value, Mapping):
        denies = _normalized_hard_denies(value.get("hard_denies"))
    else:
        denies = ()
    normalized_tool = tool_name.strip().lower()
    normalized_capability = (capability or "").strip().lower()
    selectors = {normalized_tool, f"tool:{normalized_tool}"}
    if normalized_capability:
        selectors.update(
            {
                normalized_capability,
                f"{normalized_capability}:*",
                f"tool:{normalized_capability}",
            }
        )
    for deny in denies:
        if deny.strip().lower() in selectors:
            return deny
    return None


def autonomy_mode_defaults(mode: AutonomyMode) -> dict[str, Any]:
    """Return the built-in defaults for one autonomy mode."""
    return dict(_AUTONOMY_DEFAULTS[mode])


def infer_autonomy_mode(
    *,
    require_approval_for_exec: bool,
    require_approval_for_file_write: bool,
    file_write_scope: str,
) -> AutonomyMode:
    """Infer the closest autonomy preset for legacy configs without a mode."""
    if (
        require_approval_for_exec is False
        and require_approval_for_file_write is False
        and file_write_scope == "any"
    ):
        return "high_autonomy"
    if (
        require_approval_for_exec is False
        and require_approval_for_file_write is False
        and file_write_scope == "workspace"
    ):
        return "auto_review"
    if (
        require_approval_for_exec is True
        and require_approval_for_file_write is False
        and file_write_scope == "workspace"
    ):
        return "trusted_workspace"
    return "strict"


def _normalized_mode(value: Any) -> AutonomyMode | None:
    if isinstance(value, str) and value in _AUTONOMY_MODES:
        return value  # type: ignore[return-value]
    return None


def _normalized_scope(value: Any) -> PolicyScope | None:
    if isinstance(value, str) and value in _POLICY_SCOPES:
        return value  # type: ignore[return-value]
    return None


def _normalized_hard_denies(value: Any) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return ()
    return tuple(
        sorted(
            {
                item.strip()
                for item in value
                if isinstance(item, str) and item.strip()
            }
        )
    )


def _has_policy_inputs(value: Mapping[str, Any] | None) -> bool:
    return bool(value and any(key in _POLICY_FIELDS for key in value))


def _overlay_policy(
    current: dict[str, Any],
    overlay: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply a normal policy overlay, expanding an autonomy preset first."""
    resolved = dict(current)
    mode = _normalized_mode(overlay.get("autonomy_mode"))
    if mode is not None:
        resolved.update(autonomy_mode_defaults(mode))

    if isinstance(overlay.get("require_approval_for_file_write"), bool):
        resolved["require_approval_for_file_write"] = overlay[
            "require_approval_for_file_write"
        ]
    if isinstance(overlay.get("require_approval_for_exec"), bool):
        resolved["require_approval_for_exec"] = overlay["require_approval_for_exec"]

    legacy_scope = _normalized_scope(overlay.get("file_ops_scope"))
    if legacy_scope is not None:
        resolved["file_read_scope"] = legacy_scope
        resolved["file_write_scope"] = legacy_scope
    read_scope = _normalized_scope(overlay.get("file_read_scope"))
    if read_scope is not None:
        resolved["file_read_scope"] = read_scope
    write_scope = _normalized_scope(overlay.get("file_write_scope"))
    if write_scope is not None:
        resolved["file_write_scope"] = write_scope
    return resolved


def _narrower_scope(left: PolicyScope, right: PolicyScope) -> PolicyScope:
    if "workspace" in {left, right}:
        return "workspace"
    return "any"


def _restrict_policy(
    current: dict[str, Any],
    restrictions: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply a restriction layer without allowing it to widen access."""
    requested = _overlay_policy(current, restrictions)
    resolved = dict(current)
    resolved["require_approval_for_file_write"] = bool(
        current["require_approval_for_file_write"]
        or requested["require_approval_for_file_write"]
    )
    resolved["require_approval_for_exec"] = bool(
        current["require_approval_for_exec"] or requested["require_approval_for_exec"]
    )
    resolved["file_read_scope"] = _narrower_scope(
        current["file_read_scope"], requested["file_read_scope"]
    )
    resolved["file_write_scope"] = _narrower_scope(
        current["file_write_scope"], requested["file_write_scope"]
    )

    requested_mode = _normalized_mode(restrictions.get("autonomy_mode"))
    if requested_mode is not None and all(
        resolved[key] == requested[key]
        for key in (
            "require_approval_for_file_write",
            "require_approval_for_exec",
            "file_read_scope",
            "file_write_scope",
        )
    ):
        resolved["autonomy_mode"] = requested_mode
    return resolved


class EffectivePolicyResolver:
    """Resolve the single server-side policy truth for a session or run.

    Precedence is ``security_config < workspace_policy < session_override``.
    Per-run restrictions and hard platform constraints are then intersected
    with that result, so neither layer can relax approval or filesystem scope.
    """

    def resolve(
        self,
        security: SecurityConfig,
        *,
        workspace_policy: Mapping[str, Any] | None = None,
        session_overrides: Mapping[str, Any] | None = None,
        run_restrictions: Mapping[str, Any] | None = None,
        hard_constraints: Mapping[str, Any] | None = None,
    ) -> EffectivePolicySnapshot:
        resolved: dict[str, Any] = {
            "autonomy_mode": security.autonomy_mode,
            "require_approval_for_file_write": security.require_approval_for_file_write,
            "require_approval_for_exec": security.require_approval_for_exec,
            "file_read_scope": security.file_read_scope,
            "file_write_scope": security.file_write_scope,
        }
        source_chain = ["security_config"]

        if _has_policy_inputs(workspace_policy):
            resolved = _overlay_policy(resolved, workspace_policy or {})
            source_chain.append("workspace_policy")
        if _has_policy_inputs(session_overrides):
            resolved = _overlay_policy(resolved, session_overrides or {})
            source_chain.append("session_override")
        if _has_policy_inputs(run_restrictions):
            resolved = _restrict_policy(resolved, run_restrictions or {})
            source_chain.append("run_restriction")
        if _has_policy_inputs(hard_constraints):
            resolved = _restrict_policy(resolved, hard_constraints or {})
            source_chain.append("hard_constraint")

        hard_denies = tuple(
            sorted(
                set(_normalized_hard_denies((run_restrictions or {}).get("hard_denies")))
                | set(_normalized_hard_denies((hard_constraints or {}).get("hard_denies")))
            )
        )
        projection = {
            "schema_version": 1,
            "source_chain": source_chain,
            "autonomy_mode": resolved["autonomy_mode"],
            "require_approval_for_file_write": resolved[
                "require_approval_for_file_write"
            ],
            "require_approval_for_exec": resolved["require_approval_for_exec"],
            "file_read_scope": resolved["file_read_scope"],
            "file_write_scope": resolved["file_write_scope"],
            "hard_denies": list(hard_denies),
        }
        policy_version = policy_projection_version("effective-policy", projection)
        policy_digest = policy_version.rsplit(":", maxsplit=1)[-1]
        return EffectivePolicySnapshot(
            policy_snapshot_id=f"policy-{policy_digest}",
            policy_version=policy_version,
            source_chain=tuple(source_chain),
            autonomy_mode=resolved["autonomy_mode"],
            require_approval_for_file_write=resolved[
                "require_approval_for_file_write"
            ],
            require_approval_for_exec=resolved["require_approval_for_exec"],
            file_read_scope=resolved["file_read_scope"],
            file_write_scope=resolved["file_write_scope"],
            hard_denies=hard_denies,
        )


def resolve_runtime_permission_policy(
    security: SecurityConfig,
    *,
    overrides: dict[str, Any] | None = None,
) -> RuntimePermissionPolicy:
    """Resolve the legacy runtime policy view using session override semantics."""
    return EffectivePolicyResolver().resolve(
        security,
        session_overrides=overrides,
    ).to_runtime_policy()


def build_runtime_permission_policy_dict(
    security: SecurityConfig,
    *,
    overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convenience wrapper returning dict payload."""
    payload = resolve_runtime_permission_policy(security, overrides=overrides).to_dict()
    if isinstance(overrides, dict):
        for key, value in overrides.items():
            if key != "file_ops_scope" and key not in payload:
                payload[key] = value
    return payload
