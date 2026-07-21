"""Deterministic, digest-bound review decisions for authorization envelopes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from mochi.security.file_contract import (
    AuthorizationEnvelope,
    FileIdentity,
)

AUTO_REVIEWER_VERSION = "deterministic-v1"

_PROTECTED_COMPONENTS = frozenset({".git", ".mochi", ".vscode", ".idea"})
_PROTECTED_FILES = frozenset(
    {
        ".env",
        ".env.local",
        ".env.production",
        "credentials.json",
        "secrets.json",
    }
)
_CREDENTIAL_ENV_MARKERS = (
    "API_KEY",
    "ACCESS_KEY",
    "AUTH",
    "CREDENTIAL",
    "PASSWORD",
    "PRIVATE_KEY",
    "SECRET",
    "TOKEN",
)
_RISK_ORDER = (
    "protected_path",
    "workspace_escape",
    "require_escalated",
    "unknown_shell_parse",
    "identity_mismatch",
    "stale_base",
    "network_credential_exposure",
    "policy_denied",
    "policy_requires_approval",
)


class AutoReviewDecision(BaseModel):
    """Stable reviewer output bound to one canonical authorization digest."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    decision: Literal["allow", "require_approval", "deny"]
    input_digest: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_version: str = Field(min_length=1)
    reviewer_version: str = Field(min_length=1)
    risk_factors: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AutoReviewFacts:
    """Structured facts produced by policy and pre-execution validation."""

    policy_action: Literal["allow", "ask", "deny"]
    policy_rule_id: str
    protected_path: bool = False
    workspace_escape: bool = False
    unknown_shell_parse: bool = False
    identity_mismatch: bool = False
    stale_base: bool = False


class AutoReviewVerificationError(RuntimeError):
    """Raised when an allow decision no longer matches execution state."""


def _ordered(values: set[str]) -> tuple[str, ...]:
    known = [value for value in _RISK_ORDER if value in values]
    unknown = sorted(values.difference(_RISK_ORDER))
    return tuple((*known, *unknown))


def _file_path_risks(envelope: AuthorizationEnvelope) -> set[str]:
    request = envelope.file_request
    if request is None:
        return set()
    risks: set[str] = set()
    for entry in request.entries:
        normalized = entry.relative_path.replace("\\", "/")
        path = PurePosixPath(normalized)
        parts = tuple(part.lower() for part in path.parts)
        has_windows_drive = len(normalized) >= 3 and normalized[1:3] == ":/"
        if path.is_absolute() or normalized.startswith("//") or has_windows_drive or ".." in parts:
            risks.add("workspace_escape")
        if any(part in _PROTECTED_COMPONENTS for part in parts):
            risks.add("protected_path")
        if parts and parts[-1] in _PROTECTED_FILES:
            risks.add("protected_path")
    return risks


def _exec_request_risks(envelope: AuthorizationEnvelope) -> set[str]:
    request = envelope.exec_request
    if request is None:
        return set()
    risks: set[str] = set()
    if request.requested_escalation not in {"none", "use_default"}:
        risks.add("require_escalated")
    if request.network_policy == "allow" and any(
        any(marker in item.key.upper() for marker in _CREDENTIAL_ENV_MARKERS)
        for item in request.env
    ):
        risks.add("network_credential_exposure")
    return risks


def review_authorization_envelope(
    envelope: AuthorizationEnvelope,
    *,
    facts: AutoReviewFacts,
) -> AutoReviewDecision:
    """Review one canonical envelope without consulting a model or mutable policy input."""

    if not isinstance(envelope, AuthorizationEnvelope):
        raise TypeError("envelope must be AuthorizationEnvelope")
    if not isinstance(facts, AutoReviewFacts):
        raise TypeError("facts must be AutoReviewFacts")

    risks = _file_path_risks(envelope) | _exec_request_risks(envelope)
    for enabled, code in (
        (facts.protected_path, "protected_path"),
        (facts.workspace_escape, "workspace_escape"),
        (facts.unknown_shell_parse, "unknown_shell_parse"),
        (facts.identity_mismatch, "identity_mismatch"),
        (facts.stale_base, "stale_base"),
    ):
        if enabled:
            risks.add(code)

    hard_denials = {
        "protected_path",
        "workspace_escape",
        "unknown_shell_parse",
        "identity_mismatch",
        "stale_base",
    }
    approval_gates = {"require_escalated", "network_credential_exposure"}
    reason_codes = set(risks)
    if facts.policy_action == "deny":
        risks.add("policy_denied")
        reason_codes.add("policy_denied")
        decision: Literal["allow", "require_approval", "deny"] = "deny"
    elif risks & hard_denials:
        decision = "deny"
    elif risks & approval_gates:
        decision = "require_approval"
    elif facts.policy_action == "ask":
        decision = "allow"
        reason_codes.add("reviewed_allow")
    else:
        decision = "allow"
        reason_codes.add("policy_auto_allow")

    return AutoReviewDecision(
        decision=decision,
        input_digest=envelope.request_digest,
        policy_version=envelope.policy_version,
        reviewer_version=AUTO_REVIEWER_VERSION,
        risk_factors=_ordered(risks),
        reason_codes=_ordered(reason_codes),
    )


def verify_auto_review_decision(
    decision: AutoReviewDecision,
    envelope: AuthorizationEnvelope,
    *,
    current_workspace_identity: FileIdentity | None = None,
) -> None:
    """Re-bind a reviewed allow to the exact envelope and current workspace identity."""

    if decision.decision != "allow":
        raise AutoReviewVerificationError("auto review decision does not allow execution")
    digest = envelope.request_digest
    if decision.input_digest != digest:
        raise AutoReviewVerificationError("auto review input digest changed before execution")
    if decision.policy_version != envelope.policy_version:
        raise AutoReviewVerificationError("auto review policy version changed before execution")
    if decision.reviewer_version != AUTO_REVIEWER_VERSION:
        raise AutoReviewVerificationError("auto reviewer version is not executable")
    if (
        current_workspace_identity is not None
        and current_workspace_identity != envelope.context.workspace_identity
    ):
        raise AutoReviewVerificationError("workspace identity changed before execution")


def auto_review_metadata(decision: AutoReviewDecision) -> dict[str, object]:
    """Return a stable API-safe projection for runtime and approval metadata."""

    source = None
    if "reviewed_allow" in decision.reason_codes:
        source = "reviewed_allow"
    elif "policy_auto_allow" in decision.reason_codes:
        source = "policy_auto_allow"
    return {
        "auto_review_decision": decision.decision,
        "auto_review_input_digest": decision.input_digest,
        "auto_review_policy_version": decision.policy_version,
        "auto_review_reviewer_version": decision.reviewer_version,
        "auto_review_risk_factors": list(decision.risk_factors),
        "auto_review_reason_codes": list(decision.reason_codes),
        "auto_review_source": source,
    }


__all__ = [
    "AUTO_REVIEWER_VERSION",
    "AutoReviewDecision",
    "AutoReviewFacts",
    "AutoReviewVerificationError",
    "auto_review_metadata",
    "review_authorization_envelope",
    "verify_auto_review_decision",
]
