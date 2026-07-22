"""Central redaction and persistence contracts for security audit data."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from threading import RLock
from typing import Any, Literal, cast

from pydantic import BaseModel, SecretStr

REDACTED = "[REDACTED]"
REDACTED_PATH = "[REDACTED_PATH]"

SecurityAuditEventType = Literal[
    "manifest_prepared",
    "approval_created",
    "approval_resolved",
    "review_decided",
    "mutation_applied",
    "mutation_conflicted",
    "undo_requested",
    "undo_applied",
    "path_denied",
    "sandbox_denied",
    "rule_persistence_failed",
    "rule_persistence_delivered",
    "rule_persistence_retrying",
]

ALLOWED_SECURITY_AUDIT_EVENT_TYPES: frozenset[str] = frozenset(
    {
        "manifest_prepared",
        "approval_created",
        "approval_resolved",
        "review_decided",
        "mutation_applied",
        "mutation_conflicted",
        "undo_requested",
        "undo_applied",
        "path_denied",
        "sandbox_denied",
        "rule_persistence_failed",
        "rule_persistence_delivered",
        "rule_persistence_retrying",
    }
)

_SENSITIVE_KEY_RE = re.compile(
    r"(?:api[_-]?key|authorization|cookie|credential|env(?:ironment)?|password|"
    r"private[_-]?key|refresh[_-]?token|secret|token|"
    r"before[_-]?content|after[_-]?content|original[_-]?content|"
    r"patch(?:[_-]?text)?|old[_-]?text|new[_-]?text)",
    re.IGNORECASE,
)
_SENSITIVE_PATH_PART_RE = re.compile(
    r"(?:\.env(?:\.|$)|credentials?|id_(?:rsa|ed25519)|private[_-]?key|secrets?)",
    re.IGNORECASE,
)
_SUBJECT_TYPE_RE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class KnownSecretRegistry:
    """Process-local registry used to redact exact known secret values."""

    def __init__(self) -> None:
        self._values: set[str] = set()
        self._lock = RLock()

    def register(self, value: str | bytes | None) -> None:
        text = _secret_text(value)
        if text:
            with self._lock:
                self._values.add(text)

    def discard(self, value: str | bytes | None) -> None:
        text = _secret_text(value)
        if text:
            with self._lock:
                self._values.discard(text)

    def clear(self) -> None:
        with self._lock:
            self._values.clear()

    def snapshot(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._values, key=len, reverse=True))


known_secrets = KnownSecretRegistry()


@dataclass(frozen=True, slots=True)
class SecurityAuditEvent:
    event_type: SecurityAuditEventType
    subject_type: str
    subject_id: str | None = None
    request_digest: str | None = None
    outcome: str | None = None
    details: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.event_type not in ALLOWED_SECURITY_AUDIT_EVENT_TYPES:
            raise ValueError(f"Unsupported security audit event type: {self.event_type}")
        if not _SUBJECT_TYPE_RE.fullmatch(self.subject_type):
            raise ValueError("Security audit subject_type must be a bounded identifier.")
        if self.request_digest is not None and not _SHA256_RE.fullmatch(
            self.request_digest
        ):
            raise ValueError("Security audit request_digest must be a SHA-256 digest.")


def redact_for_persistence(
    value: Any,
    *,
    registry: KnownSecretRegistry = known_secrets,
    max_string_length: int = 4096,
    max_depth: int = 12,
) -> Any:
    """Recursively redact observational data before it reaches a durable store.

    Authoritative file blobs must not use this helper: they are content-addressed,
    access-restricted data used by apply/undo and must remain byte exact.
    """

    secrets = registry.snapshot()
    return _redact(
        value,
        secrets=secrets,
        max_string_length=max(64, max_string_length),
        remaining_depth=max(0, max_depth),
        key=None,
    )


def audit_details(value: Mapping[str, Any] | None) -> dict[str, Any]:
    redacted = redact_for_persistence(dict(value or {}))
    return cast(dict[str, Any], redacted) if isinstance(redacted, dict) else {}


def security_audit_projection(event: SecurityAuditEvent) -> dict[str, Any]:
    """Return the complete allowlisted/redacted persistent event projection."""

    mutable = redact_for_persistence(
        {
            "subject_id": event.subject_id,
            "request_digest": event.request_digest,
            "outcome": event.outcome,
            "details": dict(event.details or {}),
        }
    )
    safe = cast(dict[str, Any], mutable) if isinstance(mutable, dict) else {}
    return {
        "event_type": event.event_type,
        "subject_type": event.subject_type,
        "subject_id": safe.get("subject_id"),
        "request_digest": safe.get("request_digest"),
        "outcome": safe.get("outcome"),
        "details": safe.get("details") if isinstance(safe.get("details"), dict) else {},
    }


def register_known_secrets(value: Any, *, max_depth: int = 12) -> None:
    """Register only explicit ``SecretStr`` values found in a config/object tree."""

    _register_known_secrets(value, remaining_depth=max(0, max_depth), seen=set())


def file_content_observation(content: str | bytes, *, reason_code: str) -> dict[str, Any]:
    """Return non-content metadata suitable for transcript/audit persistence."""

    raw = content.encode("utf-8") if isinstance(content, str) else bytes(content)
    return {
        "sha256": hashlib.sha256(raw).hexdigest(),
        "byte_count": len(raw),
        "reason_code": reason_code,
    }


def security_audit_digest(value: Any) -> str | None:
    """Return a canonical SHA-256 digest or ``None`` for legacy/untrusted values."""

    return value if isinstance(value, str) and _SHA256_RE.fullmatch(value) else None


def _redact(
    value: Any,
    *,
    secrets: tuple[str, ...],
    max_string_length: int,
    remaining_depth: int,
    key: str | None,
) -> Any:
    if key is not None and _SENSITIVE_KEY_RE.search(key):
        return REDACTED
    if remaining_depth <= 0:
        return "[TRUNCATED_DEPTH]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, bytes):
        return {
            "sha256": hashlib.sha256(value).hexdigest(),
            "byte_count": len(value),
            "content": REDACTED,
        }
    if isinstance(value, Path):
        return _redact_text(
            str(value), secrets=secrets, max_string_length=max_string_length, path_hint=True
        )
    if isinstance(value, str):
        return _redact_text(
            value,
            secrets=secrets,
            max_string_length=max_string_length,
            path_hint=bool(key and "path" in key.lower()),
        )
    if isinstance(value, Mapping):
        projected = {
            str(item_key): _redact(
                item_value,
                secrets=secrets,
                max_string_length=max_string_length,
                remaining_depth=remaining_depth - 1,
                key=str(item_key),
            )
            for item_key, item_value in islice(value.items(), 1000)
        }
        if len(value) > 1000:
            projected["__truncated_items__"] = len(value) - 1000
        return projected
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        projected_items = [
            _redact(
                item,
                secrets=secrets,
                max_string_length=max_string_length,
                remaining_depth=remaining_depth - 1,
                key=key,
            )
            for item in islice(value, 1000)
        ]
        if len(value) > 1000:
            projected_items.append({"__truncated_items__": len(value) - 1000})
        return projected_items
    return _redact_text(
        str(value), secrets=secrets, max_string_length=max_string_length, path_hint=False
    )


def _redact_text(
    value: str,
    *,
    secrets: tuple[str, ...],
    max_string_length: int,
    path_hint: bool,
) -> str:
    output = value
    if path_hint and _SENSITIVE_PATH_PART_RE.search(output.replace("\\", "/")):
        output = REDACTED_PATH
    else:
        for secret in secrets:
            if secret:
                output = output.replace(secret, REDACTED)
    if len(output) > max_string_length:
        output = f"{output[:max_string_length]}...[TRUNCATED]"
    return output


def _secret_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore").strip()
    return value.strip()


def _register_known_secrets(
    value: Any,
    *,
    remaining_depth: int,
    seen: set[int],
) -> None:
    if remaining_depth <= 0 or value is None:
        return
    if isinstance(value, SecretStr):
        known_secrets.register(value.get_secret_value())
        return
    if isinstance(value, (str, bytes, bool, int, float, Path)):
        return
    identity = id(value)
    if identity in seen:
        return
    seen.add(identity)
    if isinstance(value, BaseModel):
        for field_name in type(value).model_fields:
            _register_known_secrets(
                getattr(value, field_name, None),
                remaining_depth=remaining_depth - 1,
                seen=seen,
            )
        return
    if isinstance(value, Mapping):
        for item in value.values():
            _register_known_secrets(
                item,
                remaining_depth=remaining_depth - 1,
                seen=seen,
            )
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in islice(value, 1000):
            _register_known_secrets(
                item,
                remaining_depth=remaining_depth - 1,
                seen=seen,
            )


__all__ = [
    "ALLOWED_SECURITY_AUDIT_EVENT_TYPES",
    "KnownSecretRegistry",
    "REDACTED",
    "SecurityAuditEvent",
    "SecurityAuditEventType",
    "audit_details",
    "file_content_observation",
    "known_secrets",
    "redact_for_persistence",
    "register_known_secrets",
    "security_audit_digest",
    "security_audit_projection",
]
