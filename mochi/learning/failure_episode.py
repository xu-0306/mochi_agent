"""Redacted, versioned failure-learning candidates.

Failure candidates are telemetry records, not trajectories.  The constructor
normalizes the signature and redacts bounded feedback before anything is
durably appended, so callers cannot accidentally persist prompts, secrets, or
hidden reasoning as learning input.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

FAILURE_EPISODE_VERSION = "failure-episode-v1"
_MAX_ID_CHARS = 128
_MAX_SESSION_HASH_CHARS = 64
_MAX_TAGS = 16
_MAX_REASON_CODES = 16
_MAX_FEEDBACK = 8
_MAX_FEEDBACK_CHARS = 400
_MAX_SIGNATURE_CHARS = 128

_SECRET_PATTERNS = (
    (
        re.compile(
            r"(?i)\b(?:sk|pk|api[_ -]?key|access[_ -]?token|auth(?:orization)?|secret|token)"
            r"\s*[:=]\s*[^\s,;]+"
        ),
        "[REDACTED_SECRET]",
    ),
    (re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]+"), "Bearer [REDACTED_SECRET]"),
    (re.compile(r"\b[0-9]{13,19}\b"), "[REDACTED_PAYMENT]"),
    (re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_CONTACT]"),
    (re.compile(r"(?<!\d)(?:\+?\d[\d ()-]{7,}\d)(?!\d)"), "[REDACTED_CONTACT]"),
)


class FailureEpisodeError(ValueError):
    """Invalid or unsafe failure-learning data."""


def _clean_text(value: object, *, field_name: str, max_chars: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise FailureEpisodeError(f"{field_name} must be a non-empty string")
    value = value.strip()
    if len(value) > max_chars:
        raise FailureEpisodeError(f"{field_name} exceeds {max_chars} characters")
    return value


def _clean_optional_text(value: object, *, field_name: str, max_chars: int) -> str | None:
    if value is None:
        return None
    return _clean_text(value, field_name=field_name, max_chars=max_chars)


def _clean_bool(value: object, *, field_name: str) -> bool:
    if type(value) is not bool:
        raise FailureEpisodeError(f"{field_name} must be a boolean")
    return value


def _clean_tuple(
    value: object,
    *,
    field_name: str,
    max_items: int,
    max_chars: int,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise FailureEpisodeError(f"{field_name} must be a sequence")
    items = cast(Sequence[object], value)
    if len(items) > max_items:
        raise FailureEpisodeError(f"{field_name} exceeds {max_items} items")
    items = tuple(
        _clean_text(item, field_name=field_name, max_chars=max_chars) for item in items
    )
    if len(set(items)) != len(items):
        raise FailureEpisodeError(f"{field_name} must contain unique values")
    return items


def _require_exact_keys(
    payload: Mapping[str, Any], *, expected: frozenset[str], field_name: str
) -> None:
    actual = set(payload)
    missing = sorted(expected - actual)
    unexpected = sorted(actual - expected)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing fields: {missing}")
        if unexpected:
            details.append(f"unexpected fields: {unexpected}")
        raise FailureEpisodeError(f"{field_name} " + "; ".join(details))


def redact_failure_text(value: object, *, max_chars: int = _MAX_FEEDBACK_CHARS) -> str:
    """Return a bounded summary with common secret/contact forms removed."""

    if type(value) is not str:
        raise FailureEpisodeError("failure text must be a string")
    redacted = value
    for pattern, replacement in _SECRET_PATTERNS:
        redacted = pattern.sub(replacement, redacted)
    redacted = re.sub(r"\s+", " ", redacted).strip()
    return redacted[:max_chars]


def _normalize_signature(
    *,
    failure_signature: str,
    capability_tags: tuple[str, ...],
    tool_name: str | None,
    reason_codes: tuple[str, ...],
) -> str:
    normalized_text = redact_failure_text(failure_signature, max_chars=800).lower()
    material = "|".join(
        (
            ";".join(sorted(capability_tags)),
            tool_name or "",
            ";".join(sorted(reason_codes)),
            normalized_text,
        )
    )
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return f"failure:v1:{digest}"


@dataclass(frozen=True)
class FailureEpisode:
    """A redacted post-turn learning candidate."""

    episode_version: str
    episode_id: str
    idempotency_key: str
    session_id_hash: str
    turn_id: str
    capability_tags: tuple[str, ...]
    tool_name: str | None
    failure_signature: str
    reason_codes: tuple[str, ...]
    verifier_feedback: tuple[str, ...]
    correction_attempted: bool
    correction_verified: bool
    created_at: str

    def __post_init__(self) -> None:
        if self.episode_version != FAILURE_EPISODE_VERSION:
            raise FailureEpisodeError(
                f"unsupported episode_version: {self.episode_version!r}"
            )
        for field_name in ("episode_id", "idempotency_key", "turn_id"):
            object.__setattr__(
                self,
                field_name,
                _clean_text(getattr(self, field_name), field_name=field_name, max_chars=_MAX_ID_CHARS),
            )
        session_hash = _clean_text(
            self.session_id_hash,
            field_name="session_id_hash",
            max_chars=_MAX_SESSION_HASH_CHARS,
        )
        if not re.fullmatch(r"[0-9a-f]{64}", session_hash):
            raise FailureEpisodeError("session_id_hash must be a lowercase SHA-256 digest")
        object.__setattr__(self, "session_id_hash", session_hash)
        object.__setattr__(
            self,
            "capability_tags",
            _clean_tuple(
                self.capability_tags,
                field_name="capability_tags",
                max_items=_MAX_TAGS,
                max_chars=64,
            ),
        )
        object.__setattr__(
            self,
            "tool_name",
            _clean_optional_text(self.tool_name, field_name="tool_name", max_chars=128),
        )
        signature = _clean_text(
            self.failure_signature,
            field_name="failure_signature",
            max_chars=_MAX_SIGNATURE_CHARS,
        )
        if not re.fullmatch(r"failure:v1:[0-9a-f]{64}", signature):
            raise FailureEpisodeError("failure_signature must be a normalized digest")
        object.__setattr__(self, "failure_signature", signature)
        object.__setattr__(
            self,
            "reason_codes",
            _clean_tuple(
                self.reason_codes,
                field_name="reason_codes",
                max_items=_MAX_REASON_CODES,
                max_chars=128,
            ),
        )
        feedback = _clean_tuple(
            self.verifier_feedback,
            field_name="verifier_feedback",
            max_items=_MAX_FEEDBACK,
            max_chars=_MAX_FEEDBACK_CHARS,
        )
        object.__setattr__(
            self,
            "verifier_feedback",
            tuple(redact_failure_text(item) for item in feedback),
        )
        object.__setattr__(
            self,
            "correction_attempted",
            _clean_bool(self.correction_attempted, field_name="correction_attempted"),
        )
        object.__setattr__(
            self,
            "correction_verified",
            _clean_bool(self.correction_verified, field_name="correction_verified"),
        )
        object.__setattr__(
            self,
            "created_at",
            _clean_text(self.created_at, field_name="created_at", max_chars=64),
        )

    @classmethod
    def candidate(
        cls,
        *,
        session_id: str,
        turn_id: str,
        capability_tags: Sequence[str],
        tool_name: str | None,
        failure_signature: str,
        reason_codes: Sequence[str],
        verifier_feedback: Sequence[str],
        correction_attempted: bool,
        correction_verified: bool,
        episode_id: str,
        idempotency_key: str,
        created_at: str | None = None,
    ) -> FailureEpisode:
        if type(session_id) is not str or not session_id.strip():
            raise FailureEpisodeError("session_id must be a non-empty string")
        tags = _clean_tuple(
            capability_tags,
            field_name="capability_tags",
            max_items=_MAX_TAGS,
            max_chars=64,
        )
        reasons = _clean_tuple(
            reason_codes,
            field_name="reason_codes",
            max_items=_MAX_REASON_CODES,
            max_chars=128,
        )
        return cls(
            episode_version=FAILURE_EPISODE_VERSION,
            episode_id=episode_id,
            idempotency_key=idempotency_key,
            session_id_hash=hashlib.sha256(session_id.encode("utf-8")).hexdigest(),
            turn_id=turn_id,
            capability_tags=tags,
            tool_name=tool_name,
            failure_signature=_normalize_signature(
                failure_signature=failure_signature,
                capability_tags=tags,
                tool_name=tool_name,
                reason_codes=reasons,
            ),
            reason_codes=reasons,
            verifier_feedback=tuple(verifier_feedback),
            correction_attempted=correction_attempted,
            correction_verified=correction_verified,
            created_at=created_at or datetime.now(tz=UTC).isoformat(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episode_version": self.episode_version,
            "episode_id": self.episode_id,
            "idempotency_key": self.idempotency_key,
            "session_id_hash": self.session_id_hash,
            "turn_id": self.turn_id,
            "capability_tags": list(self.capability_tags),
            "tool_name": self.tool_name,
            "failure_signature": self.failure_signature,
            "reason_codes": list(self.reason_codes),
            "verifier_feedback": list(self.verifier_feedback),
            "correction_attempted": self.correction_attempted,
            "correction_verified": self.correction_verified,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> FailureEpisode:
        expected = frozenset(
            {
                "episode_version",
                "episode_id",
                "idempotency_key",
                "session_id_hash",
                "turn_id",
                "capability_tags",
                "tool_name",
                "failure_signature",
                "reason_codes",
                "verifier_feedback",
                "correction_attempted",
                "correction_verified",
                "created_at",
            }
        )
        _require_exact_keys(payload, expected=expected, field_name="failure episode")
        return cls(
            episode_version=payload["episode_version"],
            episode_id=payload["episode_id"],
            idempotency_key=payload["idempotency_key"],
            session_id_hash=payload["session_id_hash"],
            turn_id=payload["turn_id"],
            capability_tags=tuple(payload["capability_tags"]),
            tool_name=payload["tool_name"],
            failure_signature=payload["failure_signature"],
            reason_codes=tuple(payload["reason_codes"]),
            verifier_feedback=tuple(payload["verifier_feedback"]),
            correction_attempted=payload["correction_attempted"],
            correction_verified=payload["correction_verified"],
            created_at=payload["created_at"],
        )
