"""Independent verification for workspace mutation artifacts.

Tool results are treated as execution claims.  Completion evidence comes from
re-resolving and reading each target inside the configured workspace.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

from mochi.agents.events import ToolCallRequestEvent, ToolCallResultEvent
from mochi.tools.file_mutations import PatchValidationError, parse_apply_patch

ARTIFACT_RECEIPT_SCHEMA_VERSION = 3
_LEGACY_ARTIFACT_RECEIPT_SCHEMA_VERSION = 1
_SCOPE_EVIDENCE_RECEIPT_SCHEMA_VERSION = 2
ACCEPTANCE_CRITERION_SCHEMA_VERSION = 1
TOOL_EXECUTION_EVIDENCE_SCHEMA_VERSION = 1

ExecutionStatus = Literal["succeeded", "failed", "partial", "unknown"]
VerificationStatus = Literal["verified", "failed", "partial", "not_run"]
RetryDisposition = Literal[
    "none",
    "retryable",
    "requires_replan",
    "requires_approval",
    "terminal",
]

_PATH_METADATA_KEYS = ("resolved_path", "path", "file_path", "relative_path")
_FIRST_PARTY_MUTATION_TOOLS = frozenset(
    {"file_write", "file_edit", "file_delete", "apply_patch"}
)
_EXECUTION_STATUSES = frozenset({"succeeded", "failed", "partial", "unknown"})
_VERIFICATION_STATUSES = frozenset({"verified", "failed", "partial", "not_run"})
_RETRY_DISPOSITIONS = frozenset(
    {"none", "retryable", "requires_replan", "requires_approval", "terminal"}
)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_TOOL_EXECUTION_CRITERION_KINDS = frozenset({"test", "lint"})
_SINGLE_COMMAND_FORBIDDEN_CHARACTERS = frozenset(
    {
        "\r",
        "\n",
        "&",
        ";",
        "|",
        "<",
        ">",
        "`",
        "$",
        "(",
        ")",
        "^",
        "%",
        "!",
        "\x00",
    }
)
_SHELL_WRAPPER_EXECUTABLES = frozenset(
    {
        ".",
        "bash",
        "call",
        "cmd",
        "cmd.exe",
        "command",
        "dash",
        "env",
        "eval",
        "exec",
        "fish",
        "iex",
        "invoke-expression",
        "nohup",
        "powershell",
        "powershell.exe",
        "pwsh",
        "pwsh.exe",
        "sh",
        "source",
        "start",
        "sudo",
        "time",
        "wsl",
        "wsl.exe",
        "xargs",
        "zsh",
    }
)


@dataclass(frozen=True)
class ArtifactExpectation:
    """Authoritative expected state for one target after execution."""

    path: str
    expected_after_sha256: str | None = None
    expected_content: str | bytes | None = None
    acceptance_criteria: tuple[Any, ...] = ()
    must_exist: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "acceptance_criteria",
            tuple(_parse_acceptance_criterion(item) for item in self.acceptance_criteria),
        )


@dataclass(frozen=True)
class _AcceptanceCriterion:
    """A parsed criterion; invalid legacy/future payloads remain fail-closed."""

    display: str
    payload: Mapping[str, Any] | None
    kind: Literal["file", "tool_execution", "invalid"]
    check: str
    value: str | None = None
    tool_name: str | None = None
    profile_id: str | None = None
    call_id: str | None = None
    arguments_digest: str | None = None
    operation_id: str | None = None
    turn_id: str | None = None
    expected_exit_code: int | None = None
    error_code: str | None = None


@dataclass(frozen=True)
class ToolExecutionEvidence:
    """Immutable evidence for one already-executed, policy-mediated tool call."""

    call_id: str
    tool_name: str
    arguments_digest: str
    operation_id: str
    turn_id: str
    exit_code: int | None
    status: str
    approval_pending: bool
    error: str | None
    arguments: Mapping[str, Any] = field(repr=False)
    schema_version: int = field(
        default=TOOL_EXECUTION_EVIDENCE_SCHEMA_VERSION,
        init=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", MappingProxyType(dict(self.arguments)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "call_id": self.call_id,
            "tool_name": self.tool_name,
            "arguments_digest": self.arguments_digest,
            "operation_id": self.operation_id,
            "turn_id": self.turn_id,
            "exit_code": self.exit_code,
            "status": self.status,
            "approval_pending": self.approval_pending,
            "error": self.error,
        }


ValidationProfileMatcher = Callable[[str, Mapping[str, Any]], bool]


@dataclass(frozen=True)
class ValidationProfileRegistry:
    """Host-owned structural matchers for already-executed validation calls."""

    matchers: Mapping[str, ValidationProfileMatcher]

    def __post_init__(self) -> None:
        normalized: dict[str, ValidationProfileMatcher] = {}
        for profile_id, matcher in self.matchers.items():
            if not isinstance(profile_id, str) or not profile_id.strip() or not callable(matcher):
                raise ValueError("Validation profiles require non-empty ids and callable matchers.")
            normalized[profile_id] = matcher
        object.__setattr__(self, "matchers", MappingProxyType(normalized))

    def matches(
        self,
        *,
        profile_id: str,
        tool_name: str,
        arguments: Mapping[str, Any],
    ) -> bool:
        matcher = self.matchers.get(profile_id)
        if matcher is None:
            return False
        try:
            return matcher(tool_name, arguments) is True
        except Exception:
            return False


def default_validation_profile_registry() -> ValidationProfileRegistry:
    """Initial host profiles; deployments may inject project-specific matchers."""

    return ValidationProfileRegistry(
        {
            "pytest": _matches_pytest_command,
            "ruff": _matches_ruff_check_command,
        }
    )


@dataclass(frozen=True)
class AcceptanceCheck:
    criterion: str
    passed: bool
    code: str
    detail: str | None = None
    criterion_payload: Mapping[str, Any] | None = None
    evidence: Mapping[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion": self.criterion,
            "passed": self.passed,
            "code": self.code,
            "detail": self.detail,
            "criterion_payload": (
                dict(self.criterion_payload) if self.criterion_payload is not None else None
            ),
            "evidence": dict(self.evidence) if self.evidence is not None else None,
        }


@dataclass(frozen=True)
class ArtifactTargetReceipt:
    requested_path: str
    resolved_path: str | None
    expected_exists: bool
    exists: bool
    in_workspace: bool
    size_bytes: int | None
    before_sha256: str | None
    expected_after_sha256: str | None
    actual_after_sha256: str | None
    changed: bool | None
    verification_status: VerificationStatus
    acceptance_checks: tuple[AcceptanceCheck, ...] = ()
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_path": self.requested_path,
            "resolved_path": self.resolved_path,
            "expected_exists": self.expected_exists,
            "exists": self.exists,
            "in_workspace": self.in_workspace,
            "size_bytes": self.size_bytes,
            "before_sha256": self.before_sha256,
            "expected_after_sha256": self.expected_after_sha256,
            "actual_after_sha256": self.actual_after_sha256,
            "changed": self.changed,
            "verification_status": self.verification_status,
            "acceptance_checks": [check.to_dict() for check in self.acceptance_checks],
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class ArtifactUnexpectedChangedPath:
    """A structured mutation report that escaped its call's authorized scope."""

    call_id: str
    path: str
    resolved_path: str

    def to_dict(self) -> dict[str, str]:
        return {
            "call_id": self.call_id,
            "path": self.path,
            "resolved_path": self.resolved_path,
        }


@dataclass(frozen=True)
class ArtifactScopeEvidence:
    """Per-call authorization and observed structured mutation reports."""

    authorized_paths_by_call: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    observed_paths_by_call: Mapping[str, tuple[str, ...]] = field(
        default_factory=dict
    )
    unexpected_changed_paths: tuple[ArtifactUnexpectedChangedPath, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "authorized_paths_by_call",
            MappingProxyType(
                {
                    str(call_id): tuple(dict.fromkeys(paths))
                    for call_id, paths in self.authorized_paths_by_call.items()
                }
            ),
        )
        object.__setattr__(
            self,
            "observed_paths_by_call",
            MappingProxyType(
                {
                    str(call_id): tuple(dict.fromkeys(paths))
                    for call_id, paths in self.observed_paths_by_call.items()
                }
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "authorized_paths_by_call": {
                call_id: list(paths)
                for call_id, paths in self.authorized_paths_by_call.items()
            },
            "observed_paths_by_call": {
                call_id: list(paths)
                for call_id, paths in self.observed_paths_by_call.items()
            },
            "unexpected_changed_paths": [
                item.to_dict() for item in self.unexpected_changed_paths
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ArtifactScopeEvidence:
        return cls(
            authorized_paths_by_call=_path_map_from_dict(
                payload.get("authorized_paths_by_call")
            ),
            observed_paths_by_call=_path_map_from_dict(
                payload.get("observed_paths_by_call")
            ),
            unexpected_changed_paths=tuple(
                ArtifactUnexpectedChangedPath(
                    call_id=_required_string(item, "call_id"),
                    path=_required_string(item, "path"),
                    resolved_path=_required_string(item, "resolved_path"),
                )
                for item in _mapping_items(
                    payload.get("unexpected_changed_paths"),
                    "unexpected_changed_paths",
                )
            ),
        )


@dataclass(frozen=True)
class ArtifactReceipt:
    """Versioned evidence emitted for one logical mutation operation."""

    operation_id: str
    turn_id: str
    goal_id: str | None
    tool_call_ids: tuple[str, ...]
    resolved_targets: tuple[str, ...]
    changed_paths: tuple[str, ...]
    before_hashes: Mapping[str, str | None]
    after_hashes: Mapping[str, str | None]
    expected_after_hashes: Mapping[str, str | None]
    execution_status: ExecutionStatus
    verification_status: VerificationStatus
    retry_disposition: RetryDisposition
    targets: tuple[ArtifactTargetReceipt, ...]
    scope_evidence: ArtifactScopeEvidence = field(default_factory=ArtifactScopeEvidence)
    errors: tuple[str, ...] = ()
    recovery_plan: tuple[str, ...] = ()
    schema_version: int = field(default=ARTIFACT_RECEIPT_SCHEMA_VERSION, init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "before_hashes", MappingProxyType(dict(self.before_hashes))
        )
        object.__setattr__(
            self, "after_hashes", MappingProxyType(dict(self.after_hashes))
        )
        object.__setattr__(
            self,
            "expected_after_hashes",
            MappingProxyType(dict(self.expected_after_hashes)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "operation_id": self.operation_id,
            "turn_id": self.turn_id,
            "goal_id": self.goal_id,
            "tool_call_ids": list(self.tool_call_ids),
            "resolved_targets": list(self.resolved_targets),
            "changed_paths": list(self.changed_paths),
            "before_hashes": dict(self.before_hashes),
            "after_hashes": dict(self.after_hashes),
            "expected_after_hashes": dict(self.expected_after_hashes),
            "execution_status": self.execution_status,
            "verification_status": self.verification_status,
            "retry_disposition": self.retry_disposition,
            "targets": [target.to_dict() for target in self.targets],
            "scope_evidence": self.scope_evidence.to_dict(),
            "errors": list(self.errors),
            "recovery_plan": list(self.recovery_plan),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ArtifactReceipt:
        """Read legacy receipts without inventing scope or acceptance evidence."""
        schema_version = payload.get("schema_version", _LEGACY_ARTIFACT_RECEIPT_SCHEMA_VERSION)
        if type(schema_version) is not int:
            raise ValueError("Artifact receipt schema_version must be an integer.")
        if schema_version not in {
            _LEGACY_ARTIFACT_RECEIPT_SCHEMA_VERSION,
            _SCOPE_EVIDENCE_RECEIPT_SCHEMA_VERSION,
            ARTIFACT_RECEIPT_SCHEMA_VERSION,
        }:
            raise ValueError(f"Unsupported artifact receipt schema version: {schema_version!r}")
        scope_evidence = (
            ArtifactScopeEvidence()
            if schema_version == _LEGACY_ARTIFACT_RECEIPT_SCHEMA_VERSION
            else ArtifactScopeEvidence.from_dict(
                _required_mapping(payload.get("scope_evidence"), "scope_evidence")
            )
        )
        return cls(
            operation_id=_required_string(payload, "operation_id"),
            turn_id=_required_string(payload, "turn_id"),
            goal_id=_optional_string(payload.get("goal_id"), "goal_id"),
            tool_call_ids=_string_tuple(payload.get("tool_call_ids"), "tool_call_ids"),
            resolved_targets=_string_tuple(payload.get("resolved_targets"), "resolved_targets"),
            changed_paths=_string_tuple(payload.get("changed_paths"), "changed_paths"),
            before_hashes=_optional_string_map(payload.get("before_hashes"), "before_hashes"),
            after_hashes=_optional_string_map(payload.get("after_hashes"), "after_hashes"),
            expected_after_hashes=_optional_string_map(
                payload.get("expected_after_hashes"),
                "expected_after_hashes",
            ),
            execution_status=_required_literal(
                payload, "execution_status", _EXECUTION_STATUSES
            ),
            verification_status=_required_literal(
                payload, "verification_status", _VERIFICATION_STATUSES
            ),
            retry_disposition=_required_literal(
                payload, "retry_disposition", _RETRY_DISPOSITIONS
            ),
            targets=tuple(
            _target_receipt_from_dict(item, schema_version=schema_version)
                for item in _mapping_items(payload.get("targets"), "targets")
            ),
            scope_evidence=scope_evidence,
            errors=_string_tuple(payload.get("errors"), "errors"),
            recovery_plan=_string_tuple(payload.get("recovery_plan"), "recovery_plan"),
        )


def _required_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Artifact receipt field {field_name!r} must be an object.")
    return value


def _required_string(payload: Mapping[str, Any], field_name: str) -> str:
    value = payload.get(field_name)
    if not isinstance(value, str):
        raise ValueError(f"Artifact receipt field {field_name!r} must be a string.")
    return value


def _optional_string(value: Any, field_name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"Artifact receipt field {field_name!r} must be a string or null.")
    return value


def _required_boolean(payload: Mapping[str, Any], field_name: str) -> bool:
    value = payload.get(field_name)
    if not isinstance(value, bool):
        raise ValueError(f"Artifact receipt field {field_name!r} must be boolean.")
    return value


def _required_literal(
    payload: Mapping[str, Any],
    field_name: str,
    allowed: frozenset[str],
) -> str:
    value = _required_string(payload, field_name)
    if value not in allowed:
        raise ValueError(
            f"Artifact receipt field {field_name!r} has unsupported value {value!r}."
        )
    return value


def _string_tuple(value: Any, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"Artifact receipt field {field_name!r} must be a list of strings.")
    if not all(isinstance(item, str) for item in value):
        raise ValueError(f"Artifact receipt field {field_name!r} must be a list of strings.")
    return tuple(value)


def _optional_string_map(
    value: Any,
    field_name: str,
) -> dict[str, str | None]:
    payload = _required_mapping(value, field_name)
    if not all(
        isinstance(key, str) and (item is None or isinstance(item, str))
        for key, item in payload.items()
    ):
        raise ValueError(
            f"Artifact receipt field {field_name!r} must map strings to strings or null."
        )
    return dict(payload)


def _path_map_from_dict(value: Any) -> dict[str, tuple[str, ...]]:
    payload = _required_mapping(value, "scope path map")
    return {
        _required_string({"call_id": call_id}, "call_id"): _string_tuple(
            paths,
            f"scope paths for {call_id!r}",
        )
        for call_id, paths in payload.items()
    }


def _mapping_items(value: Any, field_name: str) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"Artifact receipt field {field_name!r} must be a list of objects.")
    return tuple(_required_mapping(item, field_name) for item in value)


def _target_receipt_from_dict(
    payload: Mapping[str, Any],
    *,
    schema_version: int,
) -> ArtifactTargetReceipt:
    changed = payload.get("changed")
    if changed is not None and not isinstance(changed, bool):
        raise ValueError("Artifact target receipt field 'changed' must be boolean or null.")
    size_bytes = payload.get("size_bytes")
    if size_bytes is not None and (not isinstance(size_bytes, int) or isinstance(size_bytes, bool)):
        raise ValueError("Artifact target receipt field 'size_bytes' must be integer or null.")
    return ArtifactTargetReceipt(
        requested_path=_required_string(payload, "requested_path"),
        resolved_path=_optional_string(payload.get("resolved_path"), "resolved_path"),
        expected_exists=_required_boolean(payload, "expected_exists"),
        exists=_required_boolean(payload, "exists"),
        in_workspace=_required_boolean(payload, "in_workspace"),
        size_bytes=size_bytes,
        before_sha256=_optional_string(payload.get("before_sha256"), "before_sha256"),
        expected_after_sha256=_optional_string(
            payload.get("expected_after_sha256"),
            "expected_after_sha256",
        ),
        actual_after_sha256=_optional_string(
            payload.get("actual_after_sha256"),
            "actual_after_sha256",
        ),
        changed=changed,
        verification_status=_required_literal(
            payload, "verification_status", _VERIFICATION_STATUSES
        ),
        acceptance_checks=tuple(
            AcceptanceCheck(
                criterion=_required_string(item, "criterion"),
                passed=_required_boolean(item, "passed"),
                code=_required_string(item, "code"),
                detail=_optional_string(item.get("detail"), "detail"),
                criterion_payload=(
                    _criterion_payload_from_dict(
                        item.get("criterion_payload"), "criterion_payload"
                    )
                    if schema_version >= ARTIFACT_RECEIPT_SCHEMA_VERSION
                    else None
                ),
                evidence=(
                    _tool_execution_evidence_payload_from_dict(
                        item.get("evidence"), "evidence"
                    )
                    if schema_version >= ARTIFACT_RECEIPT_SCHEMA_VERSION
                    else None
                ),
            )
            for item in _mapping_items(
                payload.get("acceptance_checks"),
                "acceptance_checks",
            )
        ),
        errors=_string_tuple(payload.get("errors"), "errors"),
    )


def _optional_mapping(value: Any, field_name: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return _required_mapping(value, field_name)


def _criterion_payload_from_dict(
    value: Any,
    field_name: str,
) -> Mapping[str, Any] | None:
    payload = _optional_mapping(value, field_name)
    if payload is None:
        return None
    criterion = _parse_acceptance_criterion(payload)
    if criterion.kind == "invalid" or criterion.payload is None:
        raise ValueError(
            f"Artifact receipt field {field_name!r} is not a supported acceptance criterion."
        )
    return criterion.payload


def _tool_execution_evidence_payload_from_dict(
    value: Any,
    field_name: str,
) -> Mapping[str, Any] | None:
    payload = _optional_mapping(value, field_name)
    if payload is None:
        return None
    required = {
        "schema_version",
        "call_id",
        "tool_name",
        "arguments_digest",
        "operation_id",
        "turn_id",
        "exit_code",
        "status",
        "approval_pending",
        "error",
        "profile_id",
    }
    if set(payload) != required:
        raise ValueError(
            f"Artifact receipt field {field_name!r} has unexpected execution evidence fields."
        )
    if payload.get("schema_version") != TOOL_EXECUTION_EVIDENCE_SCHEMA_VERSION:
        raise ValueError(
            f"Artifact receipt field {field_name!r} has unsupported execution evidence schema."
        )
    for key in (
        "call_id",
        "tool_name",
        "arguments_digest",
        "operation_id",
        "turn_id",
        "status",
        "profile_id",
    ):
        if not isinstance(payload.get(key), str) or not str(payload[key]).strip():
            raise ValueError(
                f"Artifact receipt execution evidence field {key!r} must be a non-empty string."
            )
    if _SHA256_RE.fullmatch(str(payload["arguments_digest"])) is None:
        raise ValueError("Artifact receipt execution evidence has an invalid arguments digest.")
    exit_code = payload.get("exit_code")
    if exit_code is not None and (
        type(exit_code) is not int
    ):
        raise ValueError("Artifact receipt execution evidence exit_code must be integer or null.")
    if type(payload.get("approval_pending")) is not bool:
        raise ValueError("Artifact receipt execution evidence approval_pending must be boolean.")
    if payload.get("error") is not None and not isinstance(payload.get("error"), str):
        raise ValueError("Artifact receipt execution evidence error must be string or null.")
    return MappingProxyType(dict(payload))


@dataclass(frozen=True)
class ArtifactVerificationResult:
    success: bool
    receipt: ArtifactReceipt


@dataclass
class _TargetClaim:
    requested_path: str
    expected_exists: bool = True
    before_sha256: str | None = None
    expected_after_sha256: str | None = None
    expected_content: str | bytes | None = None
    acceptance_criteria: tuple[_AcceptanceCriterion, ...] = ()
    changed_claimed: bool = False


def tool_arguments_digest(*, tool_name: str, arguments: Mapping[str, Any]) -> str:
    """Digest canonical tool-call arguments without executing or stringifying them."""
    from mochi.security.file_contract import tool_arguments_digest as _tool_arguments_digest

    return _tool_arguments_digest(tool_name=tool_name, arguments=arguments)


def _criterion_payload(**values: Any) -> Mapping[str, Any]:
    return MappingProxyType(values)


def _invalid_criterion(value: Any, code: str) -> _AcceptanceCriterion:
    display = value if isinstance(value, str) else repr(value)
    return _AcceptanceCriterion(
        display=display,
        payload=None,
        kind="invalid",
        check="invalid",
        error_code=code,
    )


def _parse_acceptance_criterion(value: Any) -> _AcceptanceCriterion:
    """Parse only declared data contracts; strings are legacy file checks, never commands."""

    if isinstance(value, str):
        return _parse_legacy_file_criterion(value)
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        return _invalid_criterion(value, "invalid_acceptance_criterion")
    payload = dict(value)
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int:
        return _invalid_criterion(value, "invalid_acceptance_criterion_schema")
    if schema_version != ACCEPTANCE_CRITERION_SCHEMA_VERSION:
        return _invalid_criterion(value, "unsupported_acceptance_criterion_schema")
    kind = payload.get("kind")
    if kind == "file":
        return _parse_file_criterion(payload)
    if kind == "tool_execution":
        return _parse_tool_execution_criterion(payload)
    return _invalid_criterion(value, "unsupported_acceptance_criterion_kind")


def _parse_legacy_file_criterion(value: str) -> _AcceptanceCriterion:
    normalized = value.strip()
    lowered = normalized.lower()
    if lowered in {"exists", "target exists", "target_exists"}:
        return _file_criterion(check="exists", display=value)
    if lowered in {"non-empty", "nonempty", "not empty"}:
        return _file_criterion(check="non_empty", display=value)
    if lowered.startswith("contains:"):
        needle = normalized.split(":", 1)[1]
        return _file_criterion(check="contains", value=needle, display=value)
    if lowered.startswith("contains "):
        needle = normalized[len("contains ") :].strip().strip("\"'")
        return _file_criterion(check="contains", value=needle, display=value)
    if lowered.startswith("sha256:"):
        return _file_criterion(
            check="sha256",
            value=lowered.split(":", 1)[1].strip(),
            display=value,
        )
    return _invalid_criterion(value, "unsupported_acceptance_criterion")


def _file_criterion(
    *,
    check: str,
    value: str | None = None,
    display: str | None = None,
) -> _AcceptanceCriterion:
    payload: dict[str, Any] = {
        "schema_version": ACCEPTANCE_CRITERION_SCHEMA_VERSION,
        "kind": "file",
        "check": check,
    }
    if value is not None:
        payload["value"] = value
    return _AcceptanceCriterion(
        display=display or json.dumps(payload, ensure_ascii=False, sort_keys=True),
        payload=_criterion_payload(**payload),
        kind="file",
        check=check,
        value=value,
    )


def _parse_file_criterion(payload: Mapping[str, Any]) -> _AcceptanceCriterion:
    check = payload.get("check")
    if not isinstance(check, str) or check not in {
        "exists",
        "non_empty",
        "contains",
        "sha256",
    }:
        return _invalid_criterion(payload, "unsupported_file_acceptance_check")
    allowed = {"schema_version", "kind", "check"}
    value: str | None = None
    if check in {"contains", "sha256"}:
        allowed.add("value")
        candidate = payload.get("value")
        if not isinstance(candidate, str) or not candidate:
            return _invalid_criterion(payload, "invalid_file_acceptance_value")
        value = candidate
    if set(payload) != allowed:
        return _invalid_criterion(payload, "invalid_file_acceptance_fields")
    if check == "sha256" and _SHA256_RE.fullmatch(value or "") is None:
        return _invalid_criterion(payload, "invalid_file_acceptance_digest")
    return _file_criterion(
        check=check,
        value=value,
        display=json.dumps(dict(payload), ensure_ascii=False, sort_keys=True),
    )


def _parse_tool_execution_criterion(
    payload: Mapping[str, Any],
) -> _AcceptanceCriterion:
    check = payload.get("check")
    tool_name = payload.get("tool_name")
    profile_id = payload.get("profile_id")
    arguments_digest = payload.get("arguments_digest")
    required = {
        "schema_version",
        "kind",
        "check",
        "tool_name",
        "profile_id",
    }
    optional = {
        "call_id",
        "arguments_digest",
        "operation_id",
        "turn_id",
        "expected_exit_code",
    }
    if set(payload) - required - optional:
        return _invalid_criterion(payload, "invalid_tool_execution_criterion_fields")
    if not isinstance(check, str) or check not in _TOOL_EXECUTION_CRITERION_KINDS:
        return _invalid_criterion(payload, "unsupported_tool_execution_check")
    if not isinstance(tool_name, str) or not tool_name.strip():
        return _invalid_criterion(payload, "invalid_tool_execution_tool")
    if not isinstance(profile_id, str) or not profile_id.strip():
        return _invalid_criterion(payload, "invalid_tool_execution_profile")
    if arguments_digest is not None and (
        not isinstance(arguments_digest, str)
        or _SHA256_RE.fullmatch(arguments_digest) is None
    ):
        return _invalid_criterion(payload, "invalid_tool_execution_arguments_digest")
    pins: dict[str, str | None] = {}
    for key in ("call_id", "operation_id", "turn_id"):
        candidate = payload.get(key)
        if candidate is not None and (not isinstance(candidate, str) or not candidate.strip()):
            return _invalid_criterion(payload, f"invalid_tool_execution_{key}")
        pins[key] = candidate
    expected_exit_code = payload.get("expected_exit_code", 0)
    if type(expected_exit_code) is not int:
        return _invalid_criterion(payload, "invalid_tool_execution_exit_code")
    canonical: dict[str, Any] = {
        "schema_version": ACCEPTANCE_CRITERION_SCHEMA_VERSION,
        "kind": "tool_execution",
        "check": check,
        "tool_name": tool_name,
        "profile_id": profile_id,
    }
    if arguments_digest is not None:
        canonical["arguments_digest"] = arguments_digest
    for key, candidate in pins.items():
        if candidate is not None:
            canonical[key] = candidate
    if "expected_exit_code" in payload:
        canonical["expected_exit_code"] = expected_exit_code
    return _AcceptanceCriterion(
        display=json.dumps(canonical, ensure_ascii=False, sort_keys=True),
        payload=_criterion_payload(**canonical),
        kind="tool_execution",
        check=check,
        tool_name=tool_name,
        profile_id=profile_id,
        call_id=pins["call_id"],
        arguments_digest=arguments_digest if isinstance(arguments_digest, str) else None,
        operation_id=pins["operation_id"],
        turn_id=pins["turn_id"],
        expected_exit_code=expected_exit_code,
    )


def _command_tokens(arguments: Mapping[str, Any]) -> list[str] | None:
    """Parse a direct validation command without permitting shell syntax.

    Profiles are evidence matchers, not command allowlists. Accepting a shell
    wrapper here would let a successful unrelated command launder evidence as
    pytest or ruff. This deliberately small grammar accepts direct executable
    invocation plus arguments only and is shared across POSIX, cmd, and
    PowerShell evidence.
    """

    command = arguments.get("command")
    if not isinstance(command, str) or not command.strip():
        return None
    if any(character in command for character in _SINGLE_COMMAND_FORBIDDEN_CHARACTERS):
        return None
    try:
        tokens = shlex.split(command, posix=True, comments=False)
    except ValueError:
        return None
    if not tokens:
        return None
    executable = Path(tokens[0]).name.lower()
    if not executable or executable in _SHELL_WRAPPER_EXECUTABLES:
        return None
    return tokens


def _matches_pytest_command(tool_name: str, arguments: Mapping[str, Any]) -> bool:
    if tool_name != "exec_command":
        return False
    tokens = _command_tokens(arguments)
    if not tokens:
        return False
    executable = Path(tokens[0]).name.lower()
    if executable in {"pytest", "pytest.exe"}:
        return True
    if executable in {"python", "python3", "python.exe", "py"} and len(tokens) >= 3:
        return tokens[1] == "-m" and tokens[2].lower() == "pytest"
    return False


def _matches_ruff_check_command(tool_name: str, arguments: Mapping[str, Any]) -> bool:
    if tool_name != "exec_command":
        return False
    tokens = _command_tokens(arguments)
    if not tokens:
        return False
    executable = Path(tokens[0]).name.lower()
    if executable in {"ruff", "ruff.exe"}:
        return len(tokens) >= 2 and tokens[1] == "check"
    if executable in {"python", "python3", "python.exe", "py"}:
        return len(tokens) >= 4 and tokens[1] == "-m" and tokens[2].lower() == "ruff" and tokens[3] == "check"
    return False


def _tool_execution_evidence(
    *,
    turn_id: str,
    requests: Sequence[ToolCallRequestEvent],
    results: Sequence[ToolCallResultEvent],
) -> tuple[ToolExecutionEvidence, ...]:
    result_by_identity: dict[tuple[str, str], ToolCallResultEvent] = {}
    duplicate_result_identities: set[tuple[str, str]] = set()
    for result in results:
        identity = (result.call_id, result.tool_name)
        if identity in result_by_identity:
            duplicate_result_identities.add(identity)
        result_by_identity[identity] = result

    evidence: list[ToolExecutionEvidence] = []
    seen_requests: set[tuple[str, str]] = set()
    for request in requests:
        identity = (request.call_id, request.tool_name)
        if not request.call_id or identity in seen_requests or identity in duplicate_result_identities:
            continue
        seen_requests.add(identity)
        result = result_by_identity.get(identity)
        if result is None:
            continue
        try:
            arguments_digest = tool_arguments_digest(
                tool_name=request.tool_name,
                arguments=request.arguments,
            )
        except (TypeError, ValueError):
            continue
        metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
        operation_id = metadata.get("operation_id")
        if not isinstance(operation_id, str) or not operation_id.strip():
            continue
        output = result.result if isinstance(result.result, Mapping) else {}
        exit_code = output.get("exit_code", metadata.get("exit_code"))
        if type(exit_code) is not int:
            exit_code = None
        status = metadata.get("status")
        if not isinstance(status, str):
            status = "completed" if result.error is None else "failed"
        evidence.append(
            ToolExecutionEvidence(
                call_id=request.call_id,
                tool_name=request.tool_name,
                arguments_digest=arguments_digest,
                operation_id=operation_id,
                turn_id=turn_id,
                exit_code=exit_code,
                status=status,
                approval_pending=_requires_approval(result),
                error=result.error,
                arguments=request.arguments,
            )
        )
    return tuple(evidence)


class ArtifactVerifier:
    """Verify mutation claims against the current workspace filesystem."""

    def __init__(
        self,
        *,
        validation_profiles: ValidationProfileRegistry | None = None,
    ) -> None:
        self._validation_profiles = (
            validation_profiles or default_validation_profile_registry()
        )

    def verify(
        self,
        *,
        workspace_root: str | Path,
        turn_id: str,
        requests: Sequence[ToolCallRequestEvent],
        results: Sequence[ToolCallResultEvent],
        expectations: Sequence[ArtifactExpectation] = (),
        goal_id: str | None = None,
        operation_id: str | None = None,
        evidence_requests: Sequence[ToolCallRequestEvent] | None = None,
        evidence_results: Sequence[ToolCallResultEvent] | None = None,
    ) -> ArtifactVerificationResult:
        root = Path(workspace_root).expanduser().resolve()
        normalized_requests = tuple(requests)
        normalized_results = tuple(results)
        normalized_evidence_requests = tuple(
            evidence_requests if evidence_requests is not None else normalized_requests
        )
        normalized_evidence_results = tuple(
            evidence_results if evidence_results is not None else normalized_results
        )
        effective_operation_id = operation_id or self.operation_id(
            turn_id=turn_id,
            requests=normalized_requests,
        )
        execution_status = _execution_status(normalized_requests, normalized_results)
        claims, scope_evidence, extraction_errors = _collect_target_claims(
            root=root,
            requests=normalized_requests,
            results=normalized_results,
            expectations=expectations,
        )
        execution_evidence = _tool_execution_evidence(
            turn_id=turn_id,
            requests=normalized_evidence_requests,
            results=normalized_evidence_results,
        )

        claimed_target_receipts = tuple(
            _verify_target(
                root=root,
                claim=claim,
                execution_evidence=execution_evidence,
                validation_profiles=self._validation_profiles,
            )
            for claim in claims.values()
        )
        target_receipts = (
            *claimed_target_receipts,
            *(
                _unexpected_changed_path_target(root=root, violation=violation)
                for violation in scope_evidence.unexpected_changed_paths
            ),
        )
        verification_status = (
            "failed"
            if scope_evidence.unexpected_changed_paths
            else _verification_status(target_receipts)
        )
        aggregate_errors = [*extraction_errors]
        aggregate_errors.extend(
            error
            for target in target_receipts
            for error in target.errors
        )
        if not target_receipts:
            aggregate_errors.append("No artifact targets could be resolved from the operation.")

        retry_disposition = _retry_disposition(
            execution_status=execution_status,
            targets=target_receipts,
            results=normalized_results,
            extraction_errors=extraction_errors,
        )
        resolved_targets = tuple(
            target.resolved_path
            for target in target_receipts
            if target.resolved_path is not None
        )
        changed_paths = tuple(
            target.resolved_path
            for claim, target in zip(
                claims.values(),
                claimed_target_receipts,
                strict=True,
            )
            if target.resolved_path is not None and claim.changed_claimed
        )
        receipt = ArtifactReceipt(
            operation_id=effective_operation_id,
            turn_id=turn_id,
            goal_id=goal_id,
            tool_call_ids=tuple(
                dict.fromkeys(
                    event.call_id
                    for event in (*normalized_requests, *normalized_results)
                    if event.call_id
                )
            ),
            resolved_targets=resolved_targets,
            changed_paths=changed_paths,
            before_hashes={
                target.resolved_path: target.before_sha256
                for target in target_receipts
                if target.resolved_path is not None
            },
            after_hashes={
                target.resolved_path: target.actual_after_sha256
                for target in target_receipts
                if target.resolved_path is not None
            },
            expected_after_hashes={
                target.resolved_path: target.expected_after_sha256
                for target in target_receipts
                if target.resolved_path is not None
            },
            execution_status=execution_status,
            verification_status=verification_status,
            retry_disposition=retry_disposition,
            targets=target_receipts,
            scope_evidence=scope_evidence,
            errors=tuple(dict.fromkeys(aggregate_errors)),
            recovery_plan=_recovery_plan(target_receipts),
        )
        return ArtifactVerificationResult(
            success=(
                execution_status == "succeeded"
                and verification_status == "verified"
                and not extraction_errors
            ),
            receipt=receipt,
        )

    @staticmethod
    def operation_id(
        *,
        turn_id: str,
        requests: Sequence[ToolCallRequestEvent],
    ) -> str:
        """Return a deterministic idempotency key for normalized call intent."""

        payload = {
            "turn_id": turn_id,
            "calls": [
                {
                    "call_id": request.call_id,
                    "tool_name": request.tool_name,
                    "arguments": request.arguments,
                }
                for request in requests
            ],
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
        return f"artifact-op-v1:{hashlib.sha256(encoded).hexdigest()}"


def _collect_target_claims(
    *,
    root: Path,
    requests: Sequence[ToolCallRequestEvent],
    results: Sequence[ToolCallResultEvent],
    expectations: Sequence[ArtifactExpectation],
) -> tuple[
    dict[str, _TargetClaim],
    ArtifactScopeEvidence,
    list[str],
]:
    claims: dict[str, _TargetClaim] = {}
    authorized_targets_by_call: dict[tuple[str, str], set[str]] = {}
    authorized_paths_by_call: dict[str, dict[str, str]] = {}
    observed_paths_by_call: dict[str, dict[str, str]] = {}
    request_tools_by_call: dict[str, str] = {}
    unexpected_changed_paths: list[ArtifactUnexpectedChangedPath] = []
    errors: list[str] = []

    for request in requests:
        request_paths = _request_paths(request, errors)
        if request.call_id:
            prior_tool_name = request_tools_by_call.setdefault(
                request.call_id,
                request.tool_name,
            )
            if prior_tool_name != request.tool_name or (
                request.call_id,
                request.tool_name,
            ) in authorized_targets_by_call:
                errors.append(f"duplicate_request_call_id:{request.call_id}")
                continue
        authorized_targets = {
            _path_key(root=root, path=path) for path in request_paths
        }
        if request.call_id:
            authorized_targets_by_call[(request.call_id, request.tool_name)] = authorized_targets
            authorized_paths = authorized_paths_by_call.setdefault(
                request.call_id,
                {},
            )
            for path in request_paths:
                authorized_paths.setdefault(
                    _path_key(root=root, path=path),
                    str(_resolved_path(root=root, path=path)),
                )
        for path in request_paths:
            claim = _claim_for(claims, root=root, path=path)
            if request.tool_name == "file_write" and request.arguments.get("append") is not True:
                content = request.arguments.get("content")
                if isinstance(content, (str, bytes)):
                    claim.expected_content = content

    for result in results:
        if result.tool_name not in _FIRST_PARTY_MUTATION_TOOLS:
            continue
        expected_tool_name = request_tools_by_call.get(result.call_id)
        if expected_tool_name is None:
            errors.append(f"unrecognized_result_call_id:{result.call_id}")
        elif expected_tool_name != result.tool_name:
            errors.append(
                f"result_tool_mismatch:{result.call_id}:{expected_tool_name}:{result.tool_name}"
            )
        authorized_targets = authorized_targets_by_call.get(
            (result.call_id, result.tool_name),
            set(),
        )
        reported_target_keys: set[str] = set()
        result_items = (
            _result_target_items(result)
            if result.error is None
            else _structured_file_change_items(result)
        )
        for item in result_items:
            result_path = _target_path(item)
            if result_path is None:
                continue
            result_key = _path_key(root=root, path=result_path)
            if result_key in reported_target_keys:
                continue
            reported_target_keys.add(result_key)
            observed_paths_by_call.setdefault(result.call_id, {}).setdefault(
                result_key,
                str(_resolved_path(root=root, path=result_path)),
            )
            if result_key not in authorized_targets:
                unexpected_changed_paths.append(
                    ArtifactUnexpectedChangedPath(
                        call_id=result.call_id,
                        path=result_path,
                        resolved_path=str(_resolved_path(root=root, path=result_path)),
                    )
                )
                continue
            claim = _claim_for(claims, root=root, path=result_path)
            claim.changed_claimed = True
            claim.before_sha256 = _first_digest(
                item.get("before_sha256"),
                item.get("base_sha256"),
                claim.before_sha256,
            )
            claim.expected_after_sha256 = _first_digest(
                item.get("after_sha256"),
                claim.expected_after_sha256,
            )
            change_type = str(item.get("change_type") or item.get("operation") or "")
            if change_type == "delete":
                claim.expected_exists = False
            new_content = item.get("new_content")
            if isinstance(new_content, (str, bytes)):
                claim.expected_content = new_content

    for expectation in expectations:
        path = str(expectation.path).strip()
        if not path:
            errors.append("An artifact expectation contained an empty target path.")
            continue
        claim = _claim_for(claims, root=root, path=path)
        claim.expected_exists = expectation.must_exist
        claim.expected_after_sha256 = expectation.expected_after_sha256
        claim.expected_content = expectation.expected_content
        # Several required deliverables may intentionally bind the same file.
        # Every acceptance criterion remains authoritative; later entries must
        # not overwrite earlier criteria merely because their target matches.
        claim.acceptance_criteria = _dedupe_acceptance_criteria(
            (*claim.acceptance_criteria, *expectation.acceptance_criteria)
        )

    return (
        claims,
        ArtifactScopeEvidence(
            authorized_paths_by_call={
                call_id: tuple(paths.values())
                for call_id, paths in authorized_paths_by_call.items()
            },
            observed_paths_by_call={
                call_id: tuple(paths.values())
                for call_id, paths in observed_paths_by_call.items()
            },
            unexpected_changed_paths=tuple(unexpected_changed_paths),
        ),
        errors,
    )


def _dedupe_acceptance_criteria(
    criteria: Sequence[_AcceptanceCriterion],
) -> tuple[_AcceptanceCriterion, ...]:
    deduplicated: list[_AcceptanceCriterion] = []
    seen: set[str] = set()
    for criterion in criteria:
        key = json.dumps(
            dict(criterion.payload) if criterion.payload is not None else {
                "invalid": criterion.display,
                "code": criterion.error_code,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if key not in seen:
            seen.add(key)
            deduplicated.append(criterion)
    return tuple(deduplicated)


def _claim_for(
    claims: dict[str, _TargetClaim],
    *,
    root: Path,
    path: str,
) -> _TargetClaim:
    key = _path_key(root=root, path=path)
    return claims.setdefault(key, _TargetClaim(requested_path=path))


def _request_paths(
    request: ToolCallRequestEvent,
    errors: list[str],
) -> list[str]:
    if request.tool_name not in _FIRST_PARTY_MUTATION_TOOLS:
        return []

    paths: list[str] = []
    if request.tool_name in {"file_write", "file_edit", "file_delete"}:
        path = request.arguments.get("path")
        if isinstance(path, str) and path.strip():
            paths.append(path.strip())
    if request.tool_name == "apply_patch":
        patch = request.arguments.get("patch") or request.arguments.get("patch_text")
        if isinstance(patch, str) and patch.strip():
            try:
                paths.extend(operation.path for operation in parse_apply_patch(patch))
            except PatchValidationError as exc:
                errors.append(f"Could not parse apply_patch targets: {exc}")
    return list(dict.fromkeys(paths))


def _path_key(*, root: Path, path: str) -> str:
    return os.path.normcase(str(_resolved_path(root=root, path=path)))


def _resolved_path(*, root: Path, path: str) -> Path:
    candidate = Path(path).expanduser()
    return (candidate if candidate.is_absolute() else root / candidate).resolve()


def _result_target_items(result: ToolCallResultEvent) -> list[dict[str, Any]]:
    candidates: list[Mapping[str, Any]] = []
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    if isinstance(metadata.get("file_changes"), list):
        candidates.extend(
            item for item in metadata["file_changes"] if isinstance(item, Mapping)
        )
    candidates.append(metadata)
    if isinstance(result.result, Mapping):
        if isinstance(result.result.get("file_changes"), list):
            candidates.extend(
                item
                for item in result.result["file_changes"]
                if isinstance(item, Mapping)
            )
        candidates.append(result.result)
    return [dict(item) for item in candidates]


def _structured_file_change_items(result: ToolCallResultEvent) -> list[dict[str, Any]]:
    """Return only completed-change evidence from a failed or partial call."""
    candidates: list[Mapping[str, Any]] = []
    metadata = result.metadata if isinstance(result.metadata, Mapping) else {}
    if isinstance(metadata.get("file_changes"), list):
        candidates.extend(
            item for item in metadata["file_changes"] if isinstance(item, Mapping)
        )
    if isinstance(result.result, Mapping) and isinstance(
        result.result.get("file_changes"), list
    ):
        candidates.extend(
            item
            for item in result.result["file_changes"]
            if isinstance(item, Mapping)
        )
    return [dict(item) for item in candidates]


def _target_path(item: Mapping[str, Any]) -> str | None:
    for key in _PATH_METADATA_KEYS:
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _unexpected_changed_path_target(
    *,
    root: Path,
    violation: ArtifactUnexpectedChangedPath,
) -> ArtifactTargetReceipt:
    requested = Path(violation.path).expanduser()
    candidate = requested if requested.is_absolute() else root / requested
    try:
        resolved = candidate.resolve()
    except OSError as exc:
        return ArtifactTargetReceipt(
            requested_path=violation.path,
            resolved_path=None,
            expected_exists=True,
            exists=False,
            in_workspace=False,
            size_bytes=None,
            before_sha256=None,
            expected_after_sha256=None,
            actual_after_sha256=None,
            changed=True,
            verification_status="failed",
            acceptance_checks=(
                AcceptanceCheck(
                    "authorized_mutation_target",
                    False,
                    "unexpected_changed_path",
                    f"call_id={violation.call_id}; resolution_error={exc}",
                ),
            ),
            errors=("unexpected_changed_path",),
        )

    return ArtifactTargetReceipt(
        requested_path=violation.path,
        resolved_path=str(resolved),
        expected_exists=True,
        exists=resolved.is_file(),
        in_workspace=_is_within(resolved, root),
        size_bytes=None,
        before_sha256=None,
        expected_after_sha256=None,
        actual_after_sha256=None,
        changed=True,
        verification_status="failed",
        acceptance_checks=(
            AcceptanceCheck(
                "authorized_mutation_target",
                False,
                "unexpected_changed_path",
                f"call_id={violation.call_id}; path={resolved}",
            ),
        ),
        errors=("unexpected_changed_path",),
    )


def _verify_target(
    *,
    root: Path,
    claim: _TargetClaim,
    execution_evidence: Sequence[ToolExecutionEvidence],
    validation_profiles: ValidationProfileRegistry,
) -> ArtifactTargetReceipt:
    requested = Path(claim.requested_path).expanduser()
    candidate = requested if requested.is_absolute() else root / requested
    try:
        resolved = candidate.resolve()
    except OSError as exc:
        return ArtifactTargetReceipt(
            requested_path=claim.requested_path,
            resolved_path=None,
            expected_exists=claim.expected_exists,
            exists=False,
            in_workspace=False,
            size_bytes=None,
            before_sha256=claim.before_sha256,
            expected_after_sha256=claim.expected_after_sha256,
            actual_after_sha256=None,
            changed=None,
            verification_status="failed",
            errors=(f"Target resolution failed: {exc}",),
        )

    in_workspace = _is_within(resolved, root)
    if not in_workspace:
        return ArtifactTargetReceipt(
            requested_path=claim.requested_path,
            resolved_path=str(resolved),
            expected_exists=claim.expected_exists,
            exists=resolved.exists(),
            in_workspace=False,
            size_bytes=None,
            before_sha256=claim.before_sha256,
            expected_after_sha256=claim.expected_after_sha256,
            actual_after_sha256=None,
            changed=None,
            verification_status="failed",
            errors=("Resolved target escapes the configured workspace.",),
        )

    exists = resolved.is_file()
    if not claim.expected_exists:
        check = AcceptanceCheck(
            criterion="target_absent",
            passed=not resolved.exists(),
            code="target_absent" if not resolved.exists() else "unexpected_target_exists",
        )
        return ArtifactTargetReceipt(
            requested_path=claim.requested_path,
            resolved_path=str(resolved),
            expected_exists=False,
            exists=exists,
            in_workspace=True,
            size_bytes=None,
            before_sha256=claim.before_sha256,
            expected_after_sha256=None,
            actual_after_sha256=None,
            changed=(not resolved.exists()),
            verification_status="verified" if check.passed else "failed",
            acceptance_checks=(check,),
            errors=() if check.passed else ("Deleted target still exists.",),
        )

    if not exists:
        return ArtifactTargetReceipt(
            requested_path=claim.requested_path,
            resolved_path=str(resolved),
            expected_exists=True,
            exists=False,
            in_workspace=True,
            size_bytes=None,
            before_sha256=claim.before_sha256,
            expected_after_sha256=claim.expected_after_sha256,
            actual_after_sha256=None,
            changed=None,
            verification_status="failed",
            acceptance_checks=(
                AcceptanceCheck("target_exists", False, "target_missing"),
            ),
            errors=("Expected artifact target does not exist as a file.",),
        )

    try:
        content = resolved.read_bytes()
    except OSError as exc:
        return ArtifactTargetReceipt(
            requested_path=claim.requested_path,
            resolved_path=str(resolved),
            expected_exists=True,
            exists=True,
            in_workspace=True,
            size_bytes=None,
            before_sha256=claim.before_sha256,
            expected_after_sha256=claim.expected_after_sha256,
            actual_after_sha256=None,
            changed=None,
            verification_status="failed",
            errors=(f"Artifact read failed: {exc}",),
        )

    actual_digest = _sha256(content)
    expected_content = _expected_bytes(claim.expected_content)
    expected_digest = claim.expected_after_sha256
    if expected_digest is None and expected_content is not None:
        expected_digest = _sha256(expected_content)

    checks = [AcceptanceCheck("target_exists", True, "target_exists")]
    errors: list[str] = []
    if expected_digest is not None:
        normalized_digest = expected_digest.strip().lower()
        valid_digest = _SHA256_RE.fullmatch(normalized_digest) is not None
        digest_matches = valid_digest and actual_digest == normalized_digest
        checks.append(
            AcceptanceCheck(
                "expected_after_sha256",
                digest_matches,
                (
                    "digest_match"
                    if digest_matches
                    else "invalid_expected_digest"
                    if not valid_digest
                    else "digest_mismatch"
                ),
                None if digest_matches else f"actual={actual_digest}",
            )
        )
        if not digest_matches:
            errors.append("Artifact after digest does not match the expected digest.")
    if expected_content is not None:
        content_matches = content == expected_content
        checks.append(
            AcceptanceCheck(
                "expected_content",
                content_matches,
                "content_match" if content_matches else "content_mismatch",
            )
        )
        if not content_matches:
            errors.append("Artifact content does not match the expected content.")

    for criterion in claim.acceptance_criteria:
        check = _evaluate_acceptance(
            criterion,
            content,
            actual_digest,
            execution_evidence=execution_evidence,
            validation_profiles=validation_profiles,
        )
        checks.append(check)
        if not check.passed:
            errors.append(
                f"Acceptance criterion failed ({criterion}): {check.code}."
            )

    changed = actual_digest != claim.before_sha256 if claim.before_sha256 else None
    return ArtifactTargetReceipt(
        requested_path=claim.requested_path,
        resolved_path=str(resolved),
        expected_exists=True,
        exists=True,
        in_workspace=True,
        size_bytes=len(content),
        before_sha256=claim.before_sha256,
        expected_after_sha256=expected_digest,
        actual_after_sha256=actual_digest,
        changed=changed,
        verification_status="verified" if not errors else "failed",
        acceptance_checks=tuple(checks),
        errors=tuple(errors),
    )


def _evaluate_acceptance(
    criterion: _AcceptanceCriterion,
    content: bytes,
    actual_digest: str,
    *,
    execution_evidence: Sequence[ToolExecutionEvidence],
    validation_profiles: ValidationProfileRegistry,
) -> AcceptanceCheck:
    if criterion.kind == "invalid":
        return AcceptanceCheck(
            criterion.display,
            False,
            criterion.error_code or "invalid_acceptance_criterion",
        )
    if criterion.kind == "tool_execution":
        return _evaluate_tool_execution_criterion(
            criterion,
            execution_evidence=execution_evidence,
            validation_profiles=validation_profiles,
        )
    if criterion.check == "exists":
        return AcceptanceCheck(
            criterion.display,
            True,
            "target_exists",
            criterion_payload=criterion.payload,
        )
    if criterion.check == "non_empty":
        passed = bool(content)
        return AcceptanceCheck(
            criterion.display,
            passed,
            "non_empty" if passed else "empty_artifact",
            criterion_payload=criterion.payload,
        )
    if criterion.check == "contains":
        needle = (criterion.value or "").encode("utf-8")
        passed = bool(needle) and needle in content
        return AcceptanceCheck(
            criterion.display,
            passed,
            "contains_match" if passed else "contains_mismatch",
            criterion_payload=criterion.payload,
        )
    if criterion.check == "sha256":
        expected = criterion.value or ""
        passed = _SHA256_RE.fullmatch(expected) is not None and expected == actual_digest
        return AcceptanceCheck(
            criterion.display,
            passed,
            "digest_match" if passed else "digest_mismatch",
            None if passed else f"actual={actual_digest}",
            criterion_payload=criterion.payload,
        )
    return AcceptanceCheck(
        criterion.display,
        False,
        "unsupported_file_acceptance_check",
        criterion_payload=criterion.payload,
    )


def _evaluate_tool_execution_criterion(
    criterion: _AcceptanceCriterion,
    *,
    execution_evidence: Sequence[ToolExecutionEvidence],
    validation_profiles: ValidationProfileRegistry,
) -> AcceptanceCheck:
    assert criterion.profile_id is not None
    assert criterion.tool_name is not None
    if criterion.profile_id not in validation_profiles.matchers:
        return AcceptanceCheck(
            criterion.display,
            False,
            "unknown_validation_profile",
            criterion_payload=criterion.payload,
        )
    candidates = [
        evidence
        for evidence in execution_evidence
        if evidence.tool_name == criterion.tool_name
        and (
            criterion.arguments_digest is None
            or evidence.arguments_digest == criterion.arguments_digest
        )
        and (criterion.call_id is None or evidence.call_id == criterion.call_id)
        and (criterion.operation_id is None or evidence.operation_id == criterion.operation_id)
        and (criterion.turn_id is None or evidence.turn_id == criterion.turn_id)
    ]
    if not candidates:
        return AcceptanceCheck(
            criterion.display,
            False,
            "tool_execution_evidence_missing",
            criterion_payload=criterion.payload,
        )
    profile_matched = [
        evidence
        for evidence in candidates
        if validation_profiles.matches(
            profile_id=criterion.profile_id,
            tool_name=evidence.tool_name,
            arguments=evidence.arguments,
        )
    ]
    if not profile_matched:
        return AcceptanceCheck(
            criterion.display,
            False,
            "validation_profile_mismatch",
            criterion_payload=criterion.payload,
        )
    if len(profile_matched) != 1:
        return AcceptanceCheck(
            criterion.display,
            False,
            "tool_execution_evidence_ambiguous",
            criterion_payload=criterion.payload,
        )
    evidence = profile_matched[0]
    evidence_payload = {**evidence.to_dict(), "profile_id": criterion.profile_id}
    if evidence.approval_pending:
        return AcceptanceCheck(
            criterion.display,
            False,
            "tool_execution_requires_approval",
            criterion_payload=criterion.payload,
            evidence=evidence_payload,
        )
    expected_exit_code = criterion.expected_exit_code if criterion.expected_exit_code is not None else 0
    passed = (
        evidence.error is None
        and evidence.status == "completed"
        and evidence.exit_code == expected_exit_code
    )
    return AcceptanceCheck(
        criterion.display,
        passed,
        "tool_execution_verified" if passed else "tool_execution_failed",
        None
        if passed
        else f"status={evidence.status}; exit_code={evidence.exit_code}; error={evidence.error}",
        criterion_payload=criterion.payload,
        evidence=evidence_payload,
    )


def _execution_status(
    requests: Sequence[ToolCallRequestEvent],
    results: Sequence[ToolCallResultEvent],
) -> ExecutionStatus:
    if not results:
        return "unknown"
    requested_ids = {request.call_id for request in requests if request.call_id}
    result_by_id = {result.call_id: result for result in results if result.call_id}
    considered = (
        [result_by_id[call_id] for call_id in requested_ids if call_id in result_by_id]
        if requested_ids
        else list(results)
    )
    missing = requested_ids - set(result_by_id)
    successes = sum(result.error is None for result in considered)
    failures = sum(result.error is not None for result in considered) + len(missing)
    if successes and failures:
        return "partial"
    if failures:
        return "failed"
    return "succeeded" if successes else "unknown"


def _verification_status(
    targets: Sequence[ArtifactTargetReceipt],
) -> VerificationStatus:
    if not targets:
        return "not_run"
    verified = sum(target.verification_status == "verified" for target in targets)
    failed = len(targets) - verified
    if verified and failed:
        return "partial"
    return "verified" if verified else "failed"


def _retry_disposition(
    *,
    execution_status: ExecutionStatus,
    targets: Sequence[ArtifactTargetReceipt],
    results: Sequence[ToolCallResultEvent],
    extraction_errors: Sequence[str],
) -> RetryDisposition:
    if any(not target.in_workspace for target in targets):
        return "terminal"
    if any(
        check.code == "tool_execution_requires_approval"
        for target in targets
        for check in target.acceptance_checks
    ):
        return "requires_approval"
    if any(_requires_approval(result) for result in results):
        return "requires_approval"
    if execution_status == "unknown":
        # Absence of a durable result cannot prove that a side effect did not
        # happen. Only reconciliation may clear this state; retrying could
        # execute the same mutation twice.
        return "terminal"
    if execution_status == "failed" and any(
        result.metadata.get("retryable") is True for result in results
    ):
        # Tool metadata may describe a transient failure, but it cannot mint a
        # second execution attempt. The host recovery coordinator must replan
        # with a new operation identity after the failed result is durable.
        return "requires_replan"
    if (
        extraction_errors
        or not targets
        or any(target.verification_status != "verified" for target in targets)
    ):
        return "requires_replan"
    if execution_status in {"failed", "partial"}:
        return "requires_replan"
    return "none"


def _requires_approval(result: ToolCallResultEvent) -> bool:
    return bool(
        result.metadata.get("requires_approval") is True
        or result.metadata.get("approval_status") == "pending"
    )


def _recovery_plan(
    targets: Sequence[ArtifactTargetReceipt],
) -> tuple[str, ...]:
    failed = [
        target.resolved_path or target.requested_path
        for target in targets
        if target.verification_status != "verified"
    ]
    verified = [
        target.resolved_path or target.requested_path
        for target in targets
        if target.verification_status == "verified"
    ]
    plan: list[str] = []
    if verified and failed:
        plan.append(
            "Preserve verified targets and do not replay their completed mutations."
        )
    if failed:
        plan.append("Replan or retry only failed targets: " + ", ".join(failed))
    return tuple(plan)


def _first_digest(*values: Any) -> str | None:
    for value in values:
        if isinstance(value, str) and value.strip():
            return value.strip().lower()
    return None


def _expected_bytes(value: str | bytes | None) -> bytes | None:
    if isinstance(value, bytes):
        return value
    if isinstance(value, str):
        return value.encode("utf-8")
    return None


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_within(candidate: Path, root: Path) -> bool:
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True
