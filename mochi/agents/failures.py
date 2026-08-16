"""Versioned, transport-neutral failures for agent event consumers."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

FAILURE_ENVELOPE_SCHEMA_VERSION = "1.0"
FAILURE_ENVELOPE_REQUIRED_FIELDS = frozenset(
    {
        "schema_version",
        "kind",
        "origin",
        "recoverability",
        "retry_policy",
        "terminal",
        "inject_into_model_context",
        "telemetry_key",
        "ui_hint",
        "diagnostics_ref",
    }
)
FAILURE_KINDS = frozenset(
    {
        "backend_error",
        "tool_error",
        "tool_denied",
        "runtime_steering",
        "context_overflow",
        "output_truncated",
        "empty_response",
        "invalid_tool_call",
        "cancelled",
    }
)
FAILURE_ORIGINS = frozenset({"agent", "backend", "runtime", "tool", "session"})
RETRY_POLICIES = frozenset({"automatic", "manual", "none"})
_SCHEMA_VERSION = re.compile(r"^(?P<major>[0-9]+)(?:\.(?P<minor>[0-9]+))?$")


class FailureEnvelopeValidationError(ValueError):
    """A failure payload cannot safely participate in the v1 contract."""


class FailureEnvelopeVersionError(FailureEnvelopeValidationError):
    """A consumer encountered a failure-envelope major it does not support."""


@dataclass(frozen=True)
class FailureEnvelope:
    """The stable failure vocabulary shared by runtime, persistence, and clients.

    ``extensions`` preserves unknown minor-version fields verbatim. It is not
    part of the public v1 field set, but makes a v1 consumer forward-compatible
    without silently accepting a new major version.
    """

    schema_version: str
    kind: str
    origin: str
    recoverability: str
    retry_policy: str
    terminal: bool
    inject_into_model_context: bool
    telemetry_key: str
    ui_hint: str
    diagnostics_ref: str | None
    extensions: Mapping[str, Any] = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        _validate_schema_version(self.schema_version)
        if self.kind not in FAILURE_KINDS:
            raise FailureEnvelopeValidationError(f"unsupported failure kind: {self.kind!r}")
        if self.origin not in FAILURE_ORIGINS:
            raise FailureEnvelopeValidationError(f"unsupported failure origin: {self.origin!r}")
        if not isinstance(self.recoverability, str) or not self.recoverability.strip():
            raise FailureEnvelopeValidationError("recoverability must be a non-empty string")
        if self.retry_policy not in RETRY_POLICIES:
            raise FailureEnvelopeValidationError("retry_policy is unsupported")
        if type(self.terminal) is not bool:
            raise FailureEnvelopeValidationError("terminal must be a boolean")
        if type(self.inject_into_model_context) is not bool:
            raise FailureEnvelopeValidationError("inject_into_model_context must be a boolean")
        if not isinstance(self.telemetry_key, str) or not self.telemetry_key.strip():
            raise FailureEnvelopeValidationError("telemetry_key must be a non-empty string")
        if not isinstance(self.ui_hint, str) or not self.ui_hint.strip():
            raise FailureEnvelopeValidationError("ui_hint must be a non-empty string")
        if self.diagnostics_ref is not None and (
            not isinstance(self.diagnostics_ref, str) or not self.diagnostics_ref.strip()
        ):
            raise FailureEnvelopeValidationError("diagnostics_ref must be null or a non-empty string")
        if not isinstance(self.extensions, Mapping):
            raise FailureEnvelopeValidationError("extensions must be an object")
        forbidden_extensions = FAILURE_ENVELOPE_REQUIRED_FIELDS.intersection(self.extensions)
        if forbidden_extensions:
            names = ", ".join(sorted(forbidden_extensions))
            raise FailureEnvelopeValidationError(f"extensions overwrite required fields: {names}")
        object.__setattr__(self, "extensions", MappingProxyType(dict(self.extensions)))

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> FailureEnvelope:
        if not isinstance(payload, Mapping):
            raise FailureEnvelopeValidationError("failure envelope must be an object")
        missing = FAILURE_ENVELOPE_REQUIRED_FIELDS.difference(payload)
        if missing:
            raise FailureEnvelopeValidationError(
                f"failure envelope missing required fields: {', '.join(sorted(missing))}"
            )
        extensions = {
            key: value
            for key, value in payload.items()
            if key not in FAILURE_ENVELOPE_REQUIRED_FIELDS
        }
        return cls(
            schema_version=payload["schema_version"],
            kind=payload["kind"],
            origin=payload["origin"],
            recoverability=payload["recoverability"],
            retry_policy=payload["retry_policy"],
            terminal=payload["terminal"],
            inject_into_model_context=payload["inject_into_model_context"],
            telemetry_key=payload["telemetry_key"],
            ui_hint=payload["ui_hint"],
            diagnostics_ref=payload["diagnostics_ref"],
            extensions=extensions,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "origin": self.origin,
            "recoverability": self.recoverability,
            "retry_policy": self.retry_policy,
            "terminal": self.terminal,
            "inject_into_model_context": self.inject_into_model_context,
            "telemetry_key": self.telemetry_key,
            "ui_hint": self.ui_hint,
            "diagnostics_ref": self.diagnostics_ref,
            **dict(self.extensions),
        }


def coerce_event_failure(
    failure: FailureEnvelope | Mapping[str, Any] | None,
    *,
    event_type: str,
    error: str | None = None,
    code: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    terminal: bool | None = None,
) -> FailureEnvelope | None:
    """Return an explicit envelope or deterministically adapt a legacy event."""

    if isinstance(failure, FailureEnvelope):
        return failure
    if failure is not None:
        return FailureEnvelope.from_mapping(failure)
    return legacy_failure_envelope(
        event_type=event_type,
        error=error,
        code=code,
        metadata=metadata,
        terminal=terminal,
    )


def legacy_failure_envelope(
    *,
    event_type: str,
    error: str | None = None,
    code: str | None = None,
    metadata: Mapping[str, Any] | None = None,
    terminal: bool | None = None,
) -> FailureEnvelope | None:
    """Map legacy event shape and taxonomy into the single v1 vocabulary."""

    metadata = metadata if isinstance(metadata, Mapping) else {}
    normalized_event_type = event_type.strip().lower()
    normalized_error_type = str(metadata.get("error_type") or "").strip().lower()
    normalized_status = str(metadata.get("status") or "").strip().lower()
    normalized_category = str(metadata.get("runtime_category") or "").strip().lower()
    normalized_code = (code or "").strip().lower()

    kind: str | None = None
    if _is_cancelled(normalized_status, normalized_error_type, normalized_code, metadata):
        kind = "cancelled"
    elif normalized_status == "runtime_steering" or normalized_category == "runtime_steering":
        kind = "runtime_steering"
    elif _is_tool_denial(normalized_status, metadata):
        kind = "tool_denied"
    elif "output_truncat" in normalized_error_type or normalized_category == "truncation":
        kind = "output_truncated"
    elif "empty" in normalized_error_type and "response" in normalized_error_type:
        kind = "empty_response"
    elif "context" in normalized_error_type and (
        "overflow" in normalized_error_type or "limit" in normalized_error_type
    ):
        kind = "context_overflow"
    elif "invalid_tool" in normalized_error_type or "tool_call_invalid" in normalized_error_type:
        kind = "invalid_tool_call"
    elif normalized_event_type in {"tool_call_result", "tool_call_completed"} and (
        bool(error) or bool(normalized_error_type)
    ):
        kind = "tool_error"
    elif normalized_event_type == "error":
        kind = "backend_error"

    if kind is None:
        return None

    recoverability = str(metadata.get("recoverability") or _default_recoverability(kind))
    return FailureEnvelope(
        schema_version=FAILURE_ENVELOPE_SCHEMA_VERSION,
        kind=kind,
        origin=_origin_for(kind, normalized_event_type, normalized_error_type, metadata),
        recoverability=recoverability,
        retry_policy=_retry_policy_for(kind, recoverability),
        terminal=_event_terminal(
            kind=kind,
            event_type=normalized_event_type,
            metadata=metadata,
            explicit_terminal=terminal,
        ),
        inject_into_model_context=kind in {"tool_error", "tool_denied", "runtime_steering"},
        telemetry_key=f"failure.{kind}",
        ui_hint=_ui_hint_for(kind),
        diagnostics_ref=_diagnostics_ref(metadata),
    )


def normalize_persisted_failure_event(event: Mapping[str, Any]) -> dict[str, Any]:
    """Upgrade a serialized event without changing legacy compatibility fields."""

    if not isinstance(event, Mapping):
        raise TypeError("persisted event must be an object")
    normalized = dict(event)
    payload = normalized.get("payload")
    if isinstance(payload, Mapping):
        normalized["payload"] = normalize_persisted_failure_event(payload)
    return _normalize_serialized_failure(normalized)


def _normalize_serialized_failure(event: dict[str, Any]) -> dict[str, Any]:
    metadata = event.get("metadata")
    metadata_mapping = metadata if isinstance(metadata, Mapping) else None
    event_type = str(event.get("type") or "")
    failure = coerce_event_failure(
        event.get("failure"),
        event_type=event_type,
        error=_as_optional_text(event.get("error") or event.get("message")),
        code=_as_optional_text(event.get("code")),
        metadata=metadata_mapping,
        terminal=True if event_type == "error" else None,
    )
    if failure is not None:
        event["failure"] = failure.to_dict()
    return event


def _validate_schema_version(value: object) -> None:
    if not isinstance(value, str):
        raise FailureEnvelopeValidationError("schema_version must be a string")
    match = _SCHEMA_VERSION.fullmatch(value)
    if match is None:
        raise FailureEnvelopeValidationError("schema_version is malformed")
    if int(match.group("major")) != 1:
        raise FailureEnvelopeVersionError(f"unsupported failure-envelope major: {value}")


def _is_cancelled(status: str, error_type: str, code: str, metadata: Mapping[str, Any]) -> bool:
    return (
        status == "cancelled"
        or metadata.get("cancelled") is True
        or "cancel" in error_type
        or "cancel" in code
    )


def _is_tool_denial(status: str, metadata: Mapping[str, Any]) -> bool:
    return (
        status in {"approval_pending", "denied", "rejected", "blocked"}
        or metadata.get("requires_approval") is True
        or isinstance(metadata.get("approval_id"), str)
    )


def _origin_for(
    kind: str,
    event_type: str,
    error_type: str,
    metadata: Mapping[str, Any],
) -> str:
    if kind in {"tool_error", "tool_denied"}:
        return "tool"
    if kind == "runtime_steering":
        return "runtime"
    if kind == "backend_error":
        if "backend" in error_type or isinstance(metadata.get("backend"), Mapping):
            return "backend"
        return "runtime" if event_type == "error" else "backend"
    if kind in {"context_overflow", "invalid_tool_call", "cancelled"}:
        return "runtime"
    if kind == "empty_response":
        return "backend"
    return "agent"


def _default_recoverability(kind: str) -> str:
    return {
        "backend_error": "manual_retry",
        "tool_error": "manual_retry",
        "tool_denied": "requires_approval",
        "runtime_steering": "recovered",
        "context_overflow": "requires_replanning",
        "output_truncated": "partial",
        "empty_response": "manual_retry",
        "invalid_tool_call": "retryable_after_repair",
        "cancelled": "manual_resume",
    }[kind]


def _event_terminal(
    *,
    kind: str,
    event_type: str,
    metadata: Mapping[str, Any],
    explicit_terminal: bool | None,
) -> bool:
    if explicit_terminal is not None:
        return explicit_terminal
    if isinstance(metadata.get("terminal"), bool):
        return bool(metadata["terminal"])
    return kind == "cancelled" or event_type in {"error", "final_answer"}


def _retry_policy_for(kind: str, recoverability: str) -> str:
    if kind == "runtime_steering":
        return "none"
    if kind in {"tool_denied", "cancelled"}:
        return "manual"
    if recoverability in {"retrying", "retryable", "retryable_after_repair"}:
        return "automatic"
    return "manual"


def _ui_hint_for(kind: str) -> str:
    return {
        "backend_error": "retry",
        "tool_error": "retry_tool",
        "tool_denied": "request_approval",
        "runtime_steering": "continue",
        "context_overflow": "reduce_context",
        "output_truncated": "continue",
        "empty_response": "retry",
        "invalid_tool_call": "retry",
        "cancelled": "resume",
    }[kind]


def _diagnostics_ref(metadata: Mapping[str, Any]) -> str | None:
    value = metadata.get("diagnostics_ref")
    return value.strip() if isinstance(value, str) and value.strip() else None


def _as_optional_text(value: object) -> str | None:
    return value if isinstance(value, str) else None
