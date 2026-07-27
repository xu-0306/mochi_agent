"""Typed outcome verification for ordinary-Chat completion gating."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, Literal, Protocol, cast

from mochi.agents.artifact_verifier import (
    ArtifactReceipt,
    RetryDisposition,
    ToolExecutionEvidence,
)
from mochi.agents.plan_ledger import PlanItem
from mochi.agents.turn_intent_contract import DeliverableContract

CriterionKind = Literal[
    "artifact",
    "tool_execution",
    "state",
    "response_shape",
    "semantic",
    "manual",
]
CriterionVerdict = Literal["verified", "failed", "unverified", "not_applicable"]

VERIFICATION_RECEIPT_VERSION = "verification-receipt-v1"
VERIFICATION_RECEIPT_EVENT = "ordinary_chat_verification_receipt_recorded"
VERIFICATION_RECEIPT_EVENT_SCHEMA_VERSION = 1

_CRITERION_KINDS = frozenset(
    {"artifact", "tool_execution", "state", "response_shape", "semantic", "manual"}
)
_CRITERION_VERDICTS = frozenset(
    {"verified", "failed", "unverified", "not_applicable"}
)
_RETRY_DISPOSITIONS = frozenset(
    {"none", "retryable", "requires_replan", "requires_approval", "terminal"}
)
_SEMANTIC_VERIFIER_ID = "semantic_judge"
_MANUAL_VERIFIER_ID = "manual_review"
_RETRY_PRECEDENCE = {
    "terminal": 4,
    "requires_approval": 3,
    "requires_replan": 2,
    "retryable": 1,
    "none": 0,
}
_LEGACY_SHA256_LENGTH = 64


def _clean_text(
    value: Any,
    *,
    field_name: str,
    allow_empty: bool = False,
    max_chars: int = 1_000,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    cleaned = " ".join(value.split())
    if not cleaned and not allow_empty:
        raise ValueError(f"{field_name} must not be empty")
    if len(cleaned) > max_chars:
        raise ValueError(f"{field_name} exceeds {max_chars} characters")
    return cleaned


def _clean_bool(value: Any, *, field_name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be a boolean")
    return value


def _clean_confidence(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number or null")
    cleaned = float(value)
    if cleaned < 0.0 or cleaned > 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    return cleaned


def _clean_text_tuple(
    value: Any,
    *,
    field_name: str,
    min_items: int = 0,
    max_items: int = 16,
    max_chars: int = 240,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a list")
    cleaned: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        entry = _clean_text(
            item,
            field_name=f"{field_name}[{index}]",
            max_chars=max_chars,
        )
        if entry in seen:
            continue
        seen.add(entry)
        cleaned.append(entry)
    if len(cleaned) < min_items:
        raise ValueError(f"{field_name} must contain at least {min_items} item(s)")
    if len(cleaned) > max_items:
        raise ValueError(f"{field_name} exceeds {max_items} items")
    return tuple(cleaned)


def _clone_json(value: Any, *, field_name: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, list):
        return [_clone_json(item, field_name=field_name) for item in value]
    if isinstance(value, tuple):
        return [_clone_json(item, field_name=field_name) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{field_name} keys must be strings")
        return {
            key: _clone_json(item, field_name=field_name)
            for key, item in value.items()
        }
    raise TypeError(f"{field_name} must contain JSON-compatible values")


def _frozen_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    cloned = _clone_json(value, field_name=field_name)
    if not isinstance(cloned, dict):
        raise TypeError(f"{field_name} must be an object")
    return MappingProxyType(cloned)


def _require_exact_keys(
    payload: Mapping[str, Any],
    *,
    expected: frozenset[str],
    field_name: str,
) -> None:
    actual = frozenset(payload)
    unexpected = sorted(actual - expected)
    missing = sorted(expected - actual)
    details: list[str] = []
    if unexpected:
        details.append(f"unexpected keys: {unexpected}")
    if missing:
        details.append(f"missing keys: {missing}")
    if details:
        raise ValueError(f"{field_name} " + "; ".join(details))


def _require_literal(
    value: Any,
    *,
    allowed: frozenset[str],
    field_name: str,
) -> str:
    cleaned = _clean_text(value, field_name=field_name, max_chars=128)
    if cleaned not in allowed:
        raise ValueError(f"unsupported {field_name}: {cleaned!r}")
    return cleaned


def _retry_disposition_max(values: Sequence[str]) -> str:
    if not values:
        return "none"
    return max(values, key=lambda item: _RETRY_PRECEDENCE[item])


def _criterion_identity(
    prefix: str,
    source_turn_ids: tuple[str, ...],
    index: int,
    *,
    owner_identity: Mapping[str, Any] | None = None,
    target_hint: str | None = None,
) -> str:
    seed = json.dumps(
        {
            "prefix": prefix,
            "source_turn_ids": list(source_turn_ids),
            "index": index,
            "owner_identity": dict(owner_identity or {}),
            "target_hint": target_hint,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{prefix}:{hashlib.sha256(seed.encode('utf-8')).hexdigest()[:12]}"


def _normalized_artifact_target(value: str) -> str:
    normalized = value.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized


def _artifact_target_evidence_ref(receipt_id: str, requested_path: str) -> str:
    target_digest = hashlib.sha256(
        _normalized_artifact_target(requested_path).encode("utf-8")
    ).hexdigest()[:12]
    return f"{receipt_id}:target:{target_digest}"


def _legacy_artifact_payload(value: str) -> Mapping[str, Any] | None:
    if value == "exists":
        return MappingProxyType({"check": "exists"})
    if value in {"non-empty", "non_empty"}:
        return MappingProxyType({"check": "non_empty"})
    if value.startswith("contains:"):
        needle = value.partition(":")[2].strip()
        if needle:
            return MappingProxyType({"check": "contains", "value": needle})
    if value.startswith("sha256:"):
        digest = value.partition(":")[2].strip().lower()
        if len(digest) == _LEGACY_SHA256_LENGTH and all(
            character in "0123456789abcdef" for character in digest
        ):
            return MappingProxyType({"check": "sha256", "value": digest})
    return None


@dataclass(frozen=True)
class VerificationCriterion:
    criterion_id: str
    kind: CriterionKind
    required: bool
    description: str
    source_turn_ids: tuple[str, ...]
    verifier_id: str | None
    payload: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "criterion_id",
            _clean_text(self.criterion_id, field_name="criterion_id", max_chars=128),
        )
        if self.kind not in _CRITERION_KINDS:
            raise ValueError(f"unsupported criterion kind: {self.kind!r}")
        object.__setattr__(
            self,
            "required",
            _clean_bool(self.required, field_name="required"),
        )
        object.__setattr__(
            self,
            "description",
            _clean_text(self.description, field_name="description", max_chars=400),
        )
        object.__setattr__(
            self,
            "source_turn_ids",
            _clean_text_tuple(
                self.source_turn_ids,
                field_name="source_turn_ids",
                min_items=1,
                max_items=16,
                max_chars=128,
            ),
        )
        if self.verifier_id is not None:
            object.__setattr__(
                self,
                "verifier_id",
                _clean_text(self.verifier_id, field_name="verifier_id", max_chars=128),
            )
        object.__setattr__(
            self,
            "payload",
            _frozen_mapping(self.payload, field_name="payload"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "kind": self.kind,
            "required": self.required,
            "description": self.description,
            "source_turn_ids": list(self.source_turn_ids),
            "verifier_id": self.verifier_id,
            "payload": dict(self.payload),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VerificationCriterion:
        expected = frozenset(
            {
                "criterion_id",
                "kind",
                "required",
                "description",
                "source_turn_ids",
                "verifier_id",
                "payload",
            }
        )
        _require_exact_keys(payload, expected=expected, field_name="verification criterion")
        return cls(
            criterion_id=payload.get("criterion_id"),
            kind=cast(CriterionKind, payload.get("kind")),
            required=payload.get("required"),
            description=payload.get("description"),
            source_turn_ids=tuple(payload.get("source_turn_ids", ())),
            verifier_id=payload.get("verifier_id"),
            payload=payload.get("payload"),
        )


@dataclass(frozen=True)
class CriterionReceipt:
    criterion_id: str
    verdict: CriterionVerdict
    verifier_id: str
    evidence_refs: tuple[str, ...]
    reason_code: str
    retry_disposition: str
    confidence: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "criterion_id",
            _clean_text(self.criterion_id, field_name="criterion_id", max_chars=128),
        )
        if self.verdict not in _CRITERION_VERDICTS:
            raise ValueError(f"unsupported criterion verdict: {self.verdict!r}")
        object.__setattr__(
            self,
            "verifier_id",
            _clean_text(self.verifier_id, field_name="verifier_id", max_chars=128),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _clean_text_tuple(
                self.evidence_refs,
                field_name="evidence_refs",
                max_items=16,
                max_chars=128,
            ),
        )
        object.__setattr__(
            self,
            "reason_code",
            _clean_text(self.reason_code, field_name="reason_code", max_chars=128),
        )
        object.__setattr__(
            self,
            "retry_disposition",
            _require_literal(
                self.retry_disposition,
                allowed=_RETRY_DISPOSITIONS,
                field_name="retry_disposition",
            ),
        )
        object.__setattr__(
            self,
            "confidence",
            _clean_confidence(self.confidence, field_name="confidence"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "criterion_id": self.criterion_id,
            "verdict": self.verdict,
            "verifier_id": self.verifier_id,
            "evidence_refs": list(self.evidence_refs),
            "reason_code": self.reason_code,
            "retry_disposition": self.retry_disposition,
            "confidence": self.confidence,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CriterionReceipt:
        expected = frozenset(
            {
                "criterion_id",
                "verdict",
                "verifier_id",
                "evidence_refs",
                "reason_code",
                "retry_disposition",
                "confidence",
            }
        )
        _require_exact_keys(payload, expected=expected, field_name="criterion receipt")
        return cls(
            criterion_id=payload.get("criterion_id"),
            verdict=cast(CriterionVerdict, payload.get("verdict")),
            verifier_id=payload.get("verifier_id"),
            evidence_refs=tuple(payload.get("evidence_refs", ())),
            reason_code=payload.get("reason_code"),
            retry_disposition=payload.get("retry_disposition"),
            confidence=payload.get("confidence"),
        )


@dataclass(frozen=True)
class VerificationReceipt:
    receipt_version: str
    receipt_id: str
    turn_id: str
    goal_id: str | None
    verdict: CriterionVerdict
    criteria: tuple[CriterionReceipt, ...]
    hard_failure: bool
    retry_disposition: str

    def __post_init__(self) -> None:
        if self.receipt_version != VERIFICATION_RECEIPT_VERSION:
            raise ValueError(f"unsupported receipt_version: {self.receipt_version!r}")
        object.__setattr__(
            self,
            "receipt_id",
            _clean_text(self.receipt_id, field_name="receipt_id", max_chars=128),
        )
        object.__setattr__(
            self,
            "turn_id",
            _clean_text(self.turn_id, field_name="turn_id", max_chars=128),
        )
        if self.goal_id is not None:
            object.__setattr__(
                self,
                "goal_id",
                _clean_text(self.goal_id, field_name="goal_id", max_chars=128),
            )
        if self.verdict not in _CRITERION_VERDICTS:
            raise ValueError(f"unsupported aggregate verdict: {self.verdict!r}")
        if not isinstance(self.criteria, tuple):
            raise TypeError("criteria must be a tuple")
        object.__setattr__(
            self,
            "hard_failure",
            _clean_bool(self.hard_failure, field_name="hard_failure"),
        )
        object.__setattr__(
            self,
            "retry_disposition",
            _require_literal(
                self.retry_disposition,
                allowed=_RETRY_DISPOSITIONS,
                field_name="retry_disposition",
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "receipt_version": self.receipt_version,
            "receipt_id": self.receipt_id,
            "turn_id": self.turn_id,
            "goal_id": self.goal_id,
            "verdict": self.verdict,
            "criteria": [item.to_dict() for item in self.criteria],
            "hard_failure": self.hard_failure,
            "retry_disposition": self.retry_disposition,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> VerificationReceipt:
        expected = frozenset(
            {
                "receipt_version",
                "receipt_id",
                "turn_id",
                "goal_id",
                "verdict",
                "criteria",
                "hard_failure",
                "retry_disposition",
            }
        )
        _require_exact_keys(payload, expected=expected, field_name="verification receipt")
        raw_criteria = payload.get("criteria")
        if not isinstance(raw_criteria, list):
            raise TypeError("verification receipt criteria must be a list")
        return cls(
            receipt_version=payload.get("receipt_version"),
            receipt_id=payload.get("receipt_id"),
            turn_id=payload.get("turn_id"),
            goal_id=payload.get("goal_id"),
            verdict=cast(CriterionVerdict, payload.get("verdict")),
            criteria=tuple(CriterionReceipt.from_dict(item) for item in raw_criteria),
            hard_failure=payload.get("hard_failure"),
            retry_disposition=payload.get("retry_disposition"),
        )


VerificationReceiptLoadStatus = Literal[
    "missing",
    "loaded",
    "invalid",
    "unsupported_version",
]
VerificationReceiptSaveStatus = Literal["saved", "conflict", "invalid"]


class VerificationReceiptSessionStore(Protocol):
    async def load_session(self, session_id: str) -> list[dict[str, Any]]: ...

    async def append_event_if(
        self,
        session_id: str,
        event: dict[str, Any],
        predicate: Any,
    ) -> bool: ...


@dataclass(frozen=True)
class VerificationReceiptLoadResult:
    status: VerificationReceiptLoadStatus
    receipt: VerificationReceipt | None = None
    revision: int = 0
    event_index: int | None = None
    message: str | None = None


@dataclass(frozen=True)
class VerificationReceiptSaveResult:
    status: VerificationReceiptSaveStatus
    expected_revision: int
    current_revision: int
    receipt: VerificationReceipt | None = None
    message: str | None = None
    idempotent_replay: bool = False


class VerificationReceiptRepository:
    """Strict, idempotent aggregate verification receipt persistence."""

    _EVENT_FIELDS = frozenset(
        {
            "type",
            "event",
            "schema_version",
            "session_id",
            "turn_id",
            "receipt_revision",
            "idempotency_key",
            "verification_receipt",
            "timestamp",
        }
    )

    def __init__(self, session_store: VerificationReceiptSessionStore) -> None:
        self._session_store = session_store

    async def load(self, session_id: str, turn_id: str) -> VerificationReceiptLoadResult:
        normalized_session_id = _clean_text(
            session_id,
            field_name="session_id",
            max_chars=256,
        )
        normalized_turn_id = _clean_text(turn_id, field_name="turn_id", max_chars=128)
        events = await self._session_store.load_session(normalized_session_id)
        return self._load_from_events(
            session_id=normalized_session_id,
            turn_id=normalized_turn_id,
            events=events,
        )

    async def save(
        self,
        session_id: str,
        receipt: VerificationReceipt,
        *,
        expected_revision: int,
        idempotency_key: str,
        timestamp: str | None = None,
    ) -> VerificationReceiptSaveResult:
        normalized_session_id = _clean_text(
            session_id,
            field_name="session_id",
            max_chars=256,
        )
        if not isinstance(receipt, VerificationReceipt):
            raise TypeError("receipt must be a VerificationReceipt")
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("expected_revision must be a non-negative integer")
        normalized_idempotency_key = _clean_text(
            idempotency_key,
            field_name="idempotency_key",
            max_chars=256,
        )
        current = await self.load(normalized_session_id, receipt.turn_id)
        if current.status in {"invalid", "unsupported_version"}:
            return VerificationReceiptSaveResult(
                status="invalid",
                expected_revision=expected_revision,
                current_revision=current.revision,
                message=current.message or "cannot append after invalid verification receipt state",
            )
        next_revision = expected_revision + 1
        event = {
            "type": "session_meta",
            "event": VERIFICATION_RECEIPT_EVENT,
            "schema_version": VERIFICATION_RECEIPT_EVENT_SCHEMA_VERSION,
            "session_id": normalized_session_id,
            "turn_id": receipt.turn_id,
            "receipt_revision": next_revision,
            "idempotency_key": normalized_idempotency_key,
            "verification_receipt": receipt.to_dict(),
            "timestamp": timestamp or datetime.now(tz=UTC).isoformat(),
        }
        outcome: VerificationReceiptSaveResult | None = None

        def predicate(events: list[dict[str, Any]]) -> bool:
            nonlocal outcome
            loaded = self._load_from_events(
                session_id=normalized_session_id,
                turn_id=receipt.turn_id,
                events=events,
            )
            if loaded.status in {"invalid", "unsupported_version"}:
                outcome = VerificationReceiptSaveResult(
                    status="invalid",
                    expected_revision=expected_revision,
                    current_revision=loaded.revision,
                    message=loaded.message,
                )
                return False
            duplicate = self._find_idempotent_event(
                events,
                session_id=normalized_session_id,
                turn_id=receipt.turn_id,
                idempotency_key=normalized_idempotency_key,
            )
            if duplicate is not None:
                if duplicate.get("verification_receipt") == receipt.to_dict():
                    outcome = VerificationReceiptSaveResult(
                        status="saved",
                        expected_revision=expected_revision,
                        current_revision=int(duplicate["receipt_revision"]),
                        receipt=receipt,
                        message="idempotent replay",
                        idempotent_replay=True,
                    )
                    return False
                outcome = VerificationReceiptSaveResult(
                    status="invalid",
                    expected_revision=expected_revision,
                    current_revision=loaded.revision,
                    message="idempotency key was reused for a different verification receipt",
                )
                return False
            if loaded.revision != expected_revision:
                outcome = VerificationReceiptSaveResult(
                    status="conflict",
                    expected_revision=expected_revision,
                    current_revision=loaded.revision,
                    receipt=loaded.receipt,
                    message="verification receipt revision changed before append",
                )
                return False
            return True

        appended = await self._session_store.append_event_if(
            normalized_session_id,
            event,
            predicate,
        )
        if appended:
            return VerificationReceiptSaveResult(
                status="saved",
                expected_revision=expected_revision,
                current_revision=next_revision,
                receipt=receipt,
            )
        if outcome is not None:
            return outcome
        reloaded = await self.load(normalized_session_id, receipt.turn_id)
        return VerificationReceiptSaveResult(
            status="conflict" if reloaded.status == "loaded" else "invalid",
            expected_revision=expected_revision,
            current_revision=reloaded.revision,
            receipt=reloaded.receipt,
            message=reloaded.message or "verification receipt append lost the SessionStore CAS",
        )

    @staticmethod
    def _find_idempotent_event(
        events: Sequence[Mapping[str, Any]],
        *,
        session_id: str,
        turn_id: str,
        idempotency_key: str,
    ) -> Mapping[str, Any] | None:
        for event in reversed(events):
            if (
                event.get("event") == VERIFICATION_RECEIPT_EVENT
                and event.get("session_id") == session_id
                and event.get("turn_id") == turn_id
                and event.get("idempotency_key") == idempotency_key
            ):
                return event
        return None

    @classmethod
    def _load_from_events(
        cls,
        *,
        session_id: str,
        turn_id: str,
        events: Sequence[Mapping[str, Any]],
    ) -> VerificationReceiptLoadResult:
        for reverse_index, event in enumerate(reversed(events)):
            if (
                event.get("event") != VERIFICATION_RECEIPT_EVENT
                or event.get("turn_id") != turn_id
            ):
                continue
            event_index = len(events) - 1 - reverse_index
            version = event.get("schema_version")
            if type(version) is not int:
                return VerificationReceiptLoadResult(
                    status="invalid",
                    event_index=event_index,
                    message="verification receipt event schema_version must be an integer",
                )
            if version != VERIFICATION_RECEIPT_EVENT_SCHEMA_VERSION:
                return VerificationReceiptLoadResult(
                    status=(
                        "unsupported_version"
                        if version > VERIFICATION_RECEIPT_EVENT_SCHEMA_VERSION
                        else "invalid"
                    ),
                    event_index=event_index,
                    message=f"unsupported verification receipt event schema version: {version}",
                )
            try:
                _require_exact_keys(
                    event,
                    expected=cls._EVENT_FIELDS,
                    field_name="verification receipt event",
                )
                if event.get("type") != "session_meta":
                    raise ValueError("verification receipt event type must be session_meta")
                if event.get("session_id") != session_id:
                    raise ValueError(
                        "verification receipt event session_id does not match request"
                    )
                revision = event.get("receipt_revision")
                if type(revision) is not int or revision <= 0:
                    raise ValueError("receipt_revision must be a positive integer")
                _clean_text(
                    event.get("idempotency_key"),
                    field_name="idempotency_key",
                    max_chars=256,
                )
                _clean_text(event.get("timestamp"), field_name="timestamp", max_chars=128)
                payload = event.get("verification_receipt")
                if not isinstance(payload, Mapping):
                    raise TypeError("verification_receipt must be an object")
                receipt = VerificationReceipt.from_dict(payload)
                if receipt.turn_id != turn_id:
                    raise ValueError("verification receipt payload turn_id does not match envelope")
            except Exception as exc:
                return VerificationReceiptLoadResult(
                    status="invalid",
                    event_index=event_index,
                    message=f"invalid verification receipt event: {exc}",
                )
            return VerificationReceiptLoadResult(
                status="loaded",
                receipt=receipt,
                revision=revision,
                event_index=event_index,
            )
        return VerificationReceiptLoadResult(status="missing")


@dataclass(frozen=True)
class VerificationEvidence:
    artifact_receipts: Mapping[str, ArtifactReceipt] = field(default_factory=dict)
    tool_execution_evidence: tuple[ToolExecutionEvidence, ...] = ()
    state: Mapping[str, Any] = field(default_factory=dict)
    response_json: Mapping[str, Any] | None = None
    response_text: str | None = None

    def __post_init__(self) -> None:
        if any(not isinstance(key, str) for key in self.artifact_receipts):
            raise TypeError("artifact_receipts keys must be strings")
        object.__setattr__(
            self,
            "artifact_receipts",
            MappingProxyType(dict(self.artifact_receipts)),
        )
        if not isinstance(self.tool_execution_evidence, tuple):
            raise TypeError("tool_execution_evidence must be a tuple")
        object.__setattr__(
            self,
            "state",
            _frozen_mapping(self.state, field_name="state"),
        )
        if self.response_json is not None:
            object.__setattr__(
                self,
                "response_json",
                _frozen_mapping(self.response_json, field_name="response_json"),
            )
        if self.response_text is not None:
            object.__setattr__(
                self,
                "response_text",
                _clean_text(
                    self.response_text,
                    field_name="response_text",
                    allow_empty=True,
                    max_chars=12_000,
                ),
            )

    def recognized_evidence_refs(self) -> frozenset[str]:
        refs = set(self.artifact_receipts)
        for receipt_id, receipt in self.artifact_receipts.items():
            refs.update(
                _artifact_target_evidence_ref(receipt_id, target.requested_path)
                for target in receipt.targets
            )
        for item in self.tool_execution_evidence:
            refs.add(item.call_id)
            if item.operation_id:
                refs.add(item.operation_id)
        if self.response_text is not None or self.response_json is not None:
            refs.add("response")
        return frozenset(refs)


class SemanticJudge(Protocol):
    async def judge(
        self,
        criterion: VerificationCriterion,
        evidence: VerificationEvidence,
    ) -> Mapping[str, Any]:
        """Return a strict JSON-like verification verdict."""


class OutcomeVerifier(Protocol):
    verifier_id: str

    def supports(self, criterion: VerificationCriterion) -> bool:
        """Whether this verifier handles the criterion."""

    async def verify(
        self,
        criterion: VerificationCriterion,
        evidence: VerificationEvidence,
    ) -> CriterionReceipt:
        """Verify one criterion against trusted evidence."""


class ArtifactVerifierAdapter:
    verifier_id = "artifact"

    def supports(self, criterion: VerificationCriterion) -> bool:
        return criterion.kind == "artifact"

    async def verify(
        self,
        criterion: VerificationCriterion,
        evidence: VerificationEvidence,
    ) -> CriterionReceipt:
        receipt_id = criterion.payload.get("receipt_id")
        if not isinstance(receipt_id, str) or not receipt_id.strip():
            return CriterionReceipt(
                criterion_id=criterion.criterion_id,
                verdict="unverified",
                verifier_id=self.verifier_id,
                evidence_refs=(),
                reason_code="artifact_receipt_id_missing",
                retry_disposition="requires_replan",
            )
        receipt = evidence.artifact_receipts.get(receipt_id)
        if receipt is None:
            return CriterionReceipt(
                criterion_id=criterion.criterion_id,
                verdict="unverified",
                verifier_id=self.verifier_id,
                evidence_refs=(),
                reason_code="artifact_receipt_missing",
                retry_disposition="requires_replan",
            )

        target_hint = criterion.payload.get("target_hint")
        target = None
        if isinstance(target_hint, str) and target_hint.strip():
            normalized_hint = _normalized_artifact_target(target_hint)
            matches = [
                candidate
                for candidate in receipt.targets
                if _normalized_artifact_target(candidate.requested_path)
                == normalized_hint
            ]
            if len(matches) != 1:
                return CriterionReceipt(
                    criterion_id=criterion.criterion_id,
                    verdict="unverified",
                    verifier_id=self.verifier_id,
                    evidence_refs=(receipt_id,),
                    reason_code=(
                        "artifact_target_missing"
                        if not matches
                        else "artifact_target_ambiguous"
                    ),
                    retry_disposition="requires_replan",
                )
            target = matches[0]

        verification_status = (
            target.verification_status
            if target is not None
            else receipt.verification_status
        )
        verdict: CriterionVerdict = (
            "verified" if verification_status == "verified" else "failed"
        )
        evidence_ref = (
            _artifact_target_evidence_ref(receipt_id, target.requested_path)
            if target is not None
            else receipt_id
        )
        reason_scope = "artifact_target" if target is not None else "artifact"
        reason_code = (
            f"{reason_scope}_verified"
            if verdict == "verified"
            else f"{reason_scope}_{verification_status}"
        )
        return CriterionReceipt(
            criterion_id=criterion.criterion_id,
            verdict=verdict,
            verifier_id=self.verifier_id,
            evidence_refs=(evidence_ref,),
            reason_code=reason_code,
            retry_disposition=receipt.retry_disposition,
        )


class ToolExecutionVerifier:
    verifier_id = "tool_execution"

    def supports(self, criterion: VerificationCriterion) -> bool:
        return criterion.kind == "tool_execution"

    async def verify(
        self,
        criterion: VerificationCriterion,
        evidence: VerificationEvidence,
    ) -> CriterionReceipt:
        payload = criterion.payload
        expected_tool = payload.get("tool_name")
        if not isinstance(expected_tool, str) or not expected_tool.strip():
            return CriterionReceipt(
                criterion_id=criterion.criterion_id,
                verdict="unverified",
                verifier_id=self.verifier_id,
                evidence_refs=(),
                reason_code="tool_name_missing",
                retry_disposition="requires_replan",
            )
        expected_exit_code = payload.get("expected_exit_code", 0)
        if type(expected_exit_code) is not int:
            return CriterionReceipt(
                criterion_id=criterion.criterion_id,
                verdict="unverified",
                verifier_id=self.verifier_id,
                evidence_refs=(),
                reason_code="expected_exit_code_invalid",
                retry_disposition="requires_replan",
            )
        matches = [
            item
            for item in evidence.tool_execution_evidence
            if item.tool_name == expected_tool
            and (
                payload.get("call_id") is None
                or item.call_id == payload.get("call_id")
            )
            and (
                payload.get("arguments_digest") is None
                or item.arguments_digest == payload.get("arguments_digest")
            )
            and (
                payload.get("operation_id") is None
                or item.operation_id == payload.get("operation_id")
            )
            and (
                payload.get("turn_id") is None
                or item.turn_id == payload.get("turn_id")
            )
        ]
        if not matches:
            return CriterionReceipt(
                criterion_id=criterion.criterion_id,
                verdict="unverified",
                verifier_id=self.verifier_id,
                evidence_refs=(),
                reason_code="tool_execution_evidence_missing",
                retry_disposition="requires_replan",
            )
        if len(matches) > 1:
            return CriterionReceipt(
                criterion_id=criterion.criterion_id,
                verdict="unverified",
                verifier_id=self.verifier_id,
                evidence_refs=tuple(item.call_id for item in matches),
                reason_code="tool_execution_evidence_ambiguous",
                retry_disposition="requires_replan",
            )
        item = matches[0]
        if item.approval_pending:
            return CriterionReceipt(
                criterion_id=criterion.criterion_id,
                verdict="failed",
                verifier_id=self.verifier_id,
                evidence_refs=(item.call_id,),
                reason_code="tool_execution_requires_approval",
                retry_disposition="requires_approval",
            )
        succeeded = (
            item.error is None
            and item.exit_code == expected_exit_code
            and item.status in {"completed", "succeeded"}
        )
        return CriterionReceipt(
            criterion_id=criterion.criterion_id,
            verdict="verified" if succeeded else "failed",
            verifier_id=self.verifier_id,
            evidence_refs=(item.call_id,),
            reason_code="tool_execution_verified" if succeeded else "tool_execution_failed",
            retry_disposition="none" if succeeded else "requires_replan",
        )


class StateVerifier:
    verifier_id = "state"

    def supports(self, criterion: VerificationCriterion) -> bool:
        return criterion.kind == "state"

    async def verify(
        self,
        criterion: VerificationCriterion,
        evidence: VerificationEvidence,
    ) -> CriterionReceipt:
        payload = criterion.payload
        path = payload.get("path")
        if not isinstance(path, str) or not path.strip():
            return CriterionReceipt(
                criterion_id=criterion.criterion_id,
                verdict="unverified",
                verifier_id=self.verifier_id,
                evidence_refs=(),
                reason_code="state_path_missing",
                retry_disposition="requires_replan",
            )
        resolved, found = _resolve_state_path(evidence.state, path)
        if not found:
            return CriterionReceipt(
                criterion_id=criterion.criterion_id,
                verdict="unverified",
                verifier_id=self.verifier_id,
                evidence_refs=(),
                reason_code="state_path_missing",
                retry_disposition="requires_replan",
            )
        if "equals" in payload:
            passed = resolved == payload.get("equals")
            return CriterionReceipt(
                criterion_id=criterion.criterion_id,
                verdict="verified" if passed else "failed",
                verifier_id=self.verifier_id,
                evidence_refs=(f"state:{path}",),
                reason_code="state_equals_match" if passed else "state_equals_mismatch",
                retry_disposition="none" if passed else "requires_replan",
            )
        if "contains" in payload:
            container = resolved
            expected = payload.get("contains")
            passed = False
            if isinstance(container, str) and isinstance(expected, str):
                passed = expected in container
            elif isinstance(container, Sequence) and not isinstance(container, (str, bytes)):
                passed = expected in container
            return CriterionReceipt(
                criterion_id=criterion.criterion_id,
                verdict="verified" if passed else "failed",
                verifier_id=self.verifier_id,
                evidence_refs=(f"state:{path}",),
                reason_code="state_contains_match" if passed else "state_contains_mismatch",
                retry_disposition="none" if passed else "requires_replan",
            )
        return CriterionReceipt(
            criterion_id=criterion.criterion_id,
            verdict="unverified",
            verifier_id=self.verifier_id,
            evidence_refs=(f"state:{path}",),
            reason_code="state_verifier_payload_unsupported",
            retry_disposition="requires_replan",
        )


class ResponseShapeVerifier:
    verifier_id = "response_shape"

    def supports(self, criterion: VerificationCriterion) -> bool:
        return criterion.kind == "response_shape"

    async def verify(
        self,
        criterion: VerificationCriterion,
        evidence: VerificationEvidence,
    ) -> CriterionReceipt:
        payload = criterion.payload
        if evidence.response_json is None and evidence.response_text is None:
            return CriterionReceipt(
                criterion_id=criterion.criterion_id,
                verdict="unverified",
                verifier_id=self.verifier_id,
                evidence_refs=(),
                reason_code="response_missing",
                retry_disposition="requires_replan",
            )
        required_keys = payload.get("required_keys", ())
        if not isinstance(required_keys, (list, tuple)):
            required_keys = ()
        missing_keys = [
            key
            for key in required_keys
            if not isinstance(evidence.response_json, Mapping) or key not in evidence.response_json
        ]
        required_sections = payload.get("required_sections", ())
        if not isinstance(required_sections, (list, tuple)):
            required_sections = ()
        missing_sections = [
            section
            for section in required_sections
            if not isinstance(section, str)
            or not isinstance(evidence.response_text, str)
            or section not in evidence.response_text
        ]
        if missing_keys or missing_sections:
            return CriterionReceipt(
                criterion_id=criterion.criterion_id,
                verdict="failed",
                verifier_id=self.verifier_id,
                evidence_refs=("response",),
                reason_code="response_shape_mismatch",
                retry_disposition="requires_replan",
            )
        return CriterionReceipt(
            criterion_id=criterion.criterion_id,
            verdict="verified",
            verifier_id=self.verifier_id,
            evidence_refs=("response",),
            reason_code="response_shape_verified",
            retry_disposition="none",
        )


class SemanticJudgeVerifier:
    verifier_id = _SEMANTIC_VERIFIER_ID

    def __init__(
        self,
        *,
        judge: SemanticJudge,
        timeout_seconds: float = 20.0,
        max_criteria: int = 6,
    ) -> None:
        if isinstance(timeout_seconds, bool) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if type(max_criteria) is not int or max_criteria < 0:
            raise ValueError("max_criteria must be a non-negative integer")
        self._judge = judge
        self._timeout_seconds = timeout_seconds
        self._max_criteria = max_criteria
        self._criteria_used = 0

    def supports(self, criterion: VerificationCriterion) -> bool:
        return criterion.kind == "semantic"

    async def verify(
        self,
        criterion: VerificationCriterion,
        evidence: VerificationEvidence,
    ) -> CriterionReceipt:
        if self._criteria_used >= self._max_criteria:
            return CriterionReceipt(
                criterion_id=criterion.criterion_id,
                verdict="unverified",
                verifier_id=self.verifier_id,
                evidence_refs=(),
                reason_code="semantic_judge_budget_exhausted",
                retry_disposition="requires_replan",
            )
        self._criteria_used += 1
        try:
            raw = await asyncio.wait_for(
                self._judge.judge(criterion, evidence),
                timeout=self._timeout_seconds,
            )
        except asyncio.TimeoutError:
            return CriterionReceipt(
                criterion_id=criterion.criterion_id,
                verdict="unverified",
                verifier_id=self.verifier_id,
                evidence_refs=(),
                reason_code="semantic_judge_timeout",
                retry_disposition="requires_replan",
            )
        except Exception:
            return CriterionReceipt(
                criterion_id=criterion.criterion_id,
                verdict="unverified",
                verifier_id=self.verifier_id,
                evidence_refs=(),
                reason_code="semantic_judge_error",
                retry_disposition="requires_replan",
            )
        try:
            expected = frozenset(
                {"verdict", "evidence_refs", "reason_code", "retry_disposition", "confidence"}
            )
            if not isinstance(raw, Mapping):
                raise TypeError("semantic judge result must be an object")
            _require_exact_keys(raw, expected=expected, field_name="semantic judge result")
            receipt = CriterionReceipt(
                criterion_id=criterion.criterion_id,
                verdict=cast(CriterionVerdict, raw.get("verdict")),
                verifier_id=self.verifier_id,
                evidence_refs=tuple(raw.get("evidence_refs", ())),
                reason_code=raw.get("reason_code"),
                retry_disposition=raw.get("retry_disposition"),
                confidence=raw.get("confidence"),
            )
            recognized_refs = evidence.recognized_evidence_refs()
            if any(ref not in recognized_refs for ref in receipt.evidence_refs):
                raise ValueError("semantic judge referenced evidence not recognized by the host")
            if receipt.verdict == "verified" and not receipt.evidence_refs:
                raise ValueError("verified semantic verdict requires host evidence")
        except Exception:
            return CriterionReceipt(
                criterion_id=criterion.criterion_id,
                verdict="unverified",
                verifier_id=self.verifier_id,
                evidence_refs=(),
                reason_code="semantic_judge_malformed",
                retry_disposition="requires_replan",
            )
        return receipt


class ManualVerifier:
    verifier_id = _MANUAL_VERIFIER_ID

    def supports(self, criterion: VerificationCriterion) -> bool:
        return criterion.kind == "manual"

    async def verify(
        self,
        criterion: VerificationCriterion,
        evidence: VerificationEvidence,
    ) -> CriterionReceipt:
        del evidence
        return CriterionReceipt(
            criterion_id=criterion.criterion_id,
            verdict="unverified",
            verifier_id=self.verifier_id,
            evidence_refs=(),
            reason_code="manual_review_required",
            retry_disposition="requires_replan",
        )


class DeterministicVerifierRegistry:
    """Resolve criteria to bounded verifiers and aggregate their receipts."""

    def __init__(
        self,
        verifiers: Sequence[OutcomeVerifier] | None = None,
    ) -> None:
        resolved = tuple(
            verifiers
            or (
                ArtifactVerifierAdapter(),
                ToolExecutionVerifier(),
                StateVerifier(),
                ResponseShapeVerifier(),
                ManualVerifier(),
            )
        )
        verifier_ids = [verifier.verifier_id for verifier in resolved]
        if len(verifier_ids) != len(set(verifier_ids)):
            raise ValueError("verifier ids must be unique")
        self._verifiers = resolved

    async def verify_all(
        self,
        criteria: Sequence[VerificationCriterion],
        evidence: VerificationEvidence,
        *,
        receipt_id: str,
        turn_id: str,
        goal_id: str | None,
    ) -> VerificationReceipt:
        receipts: list[CriterionReceipt] = []
        required_count = 0
        required_verified_count = 0
        hard_failure = False
        for criterion in criteria:
            receipt = await self.verify_one(criterion, evidence)
            receipts.append(receipt)
            if criterion.required:
                required_count += 1
                if receipt.verdict == "verified":
                    required_verified_count += 1
                elif receipt.verdict == "failed" and criterion.kind != "semantic":
                    hard_failure = True
        verdict = self._aggregate_verdict(
            criteria,
            tuple(receipts),
            required_count,
            required_verified_count,
        )
        retry_disposition = _retry_disposition_max(
            [receipt.retry_disposition for receipt in receipts]
        )
        return VerificationReceipt(
            receipt_version=VERIFICATION_RECEIPT_VERSION,
            receipt_id=_clean_text(receipt_id, field_name="receipt_id", max_chars=128),
            turn_id=_clean_text(turn_id, field_name="turn_id", max_chars=128),
            goal_id=(
                _clean_text(goal_id, field_name="goal_id", max_chars=128)
                if goal_id is not None
                else None
            ),
            verdict=verdict,
            criteria=tuple(receipts),
            hard_failure=hard_failure,
            retry_disposition=retry_disposition,
        )

    async def verify_one(
        self,
        criterion: VerificationCriterion,
        evidence: VerificationEvidence,
    ) -> CriterionReceipt:
        for verifier in self._verifiers:
            if verifier.supports(criterion):
                if (
                    criterion.verifier_id is not None
                    and verifier.verifier_id != criterion.verifier_id
                ):
                    continue
                return await verifier.verify(criterion, evidence)
        return CriterionReceipt(
            criterion_id=criterion.criterion_id,
            verdict="unverified",
            verifier_id=criterion.verifier_id or "unsupported",
            evidence_refs=(),
            reason_code="unsupported_criterion",
            retry_disposition="requires_replan",
        )

    @staticmethod
    def _aggregate_verdict(
        criteria: Sequence[VerificationCriterion],
        receipts: tuple[CriterionReceipt, ...],
        required_count: int,
        required_verified_count: int,
    ) -> CriterionVerdict:
        if not criteria:
            return "not_applicable"
        paired = list(zip(criteria, receipts, strict=True))
        if any(
            criterion.required and receipt.verdict == "failed"
            for criterion, receipt in paired
        ):
            return "failed"
        if any(
            criterion.required and receipt.verdict in {"unverified", "not_applicable"}
            for criterion, receipt in paired
        ):
            return "unverified"
        if required_count == 0:
            if any(receipt.verdict == "verified" for receipt in receipts):
                return "verified"
            if any(receipt.verdict == "failed" for receipt in receipts):
                return "failed"
            if any(receipt.verdict == "unverified" for receipt in receipts):
                return "unverified"
            return "not_applicable"
        if required_verified_count == 0:
            return "unverified"
        return "verified"


class VerificationPlanCompiler:
    """Compile trusted contracts into typed verification criteria."""

    def __init__(self, *, semantic_fallback_enabled: bool = True) -> None:
        self._semantic_fallback_enabled = semantic_fallback_enabled

    def compile(
        self,
        *,
        deliverables: Sequence[DeliverableContract] = (),
        plan_items: Sequence[PlanItem] = (),
        artifact_obligations: Sequence[Mapping[str, Any]] = (),
        response_shape: Mapping[str, Any] | None = None,
    ) -> tuple[VerificationCriterion, ...]:
        criteria: list[VerificationCriterion] = []
        for deliverable_index, deliverable in enumerate(deliverables):
            deliverable_identity = {
                "scope": "deliverable",
                "ordinal": deliverable_index,
                "kind": deliverable.kind,
                "target_hint": deliverable.target_hint,
            }
            for index, acceptance in enumerate(deliverable.acceptance_criteria):
                criteria.append(
                    self._compile_acceptance_criterion(
                        acceptance=acceptance,
                        description=f"{deliverable.kind} acceptance {index + 1}",
                        required=deliverable.required,
                        source_turn_ids=deliverable.source_turn_ids,
                        ordinal=index,
                        owner_identity=deliverable_identity,
                        target_hint=deliverable.target_hint,
                    )
                )
        for obligation_index, obligation in enumerate(artifact_obligations):
            payload = _frozen_mapping(obligation, field_name="artifact_obligation")
            source_turn_ids = tuple(payload.get("source_turn_ids", ()))
            criteria.append(
                VerificationCriterion(
                    criterion_id=_criterion_identity(
                        "artifact",
                        source_turn_ids or ("artifact",),
                        obligation_index,
                        owner_identity={
                            "scope": "artifact_obligation",
                            "ordinal": obligation_index,
                        },
                        target_hint=str(
                            payload.get("target_hint") or payload.get("path") or ""
                        )
                        or None,
                    ),
                    kind="artifact",
                    required=bool(payload.get("required", True)),
                    description=_clean_text(
                        payload.get("description", "artifact obligation"),
                        field_name="artifact obligation description",
                        max_chars=400,
                    ),
                    source_turn_ids=(
                        _clean_text_tuple(
                            source_turn_ids or ("artifact",),
                            field_name="artifact obligation source_turn_ids",
                            min_items=1,
                            max_items=16,
                            max_chars=128,
                        )
                    ),
                    verifier_id="artifact",
                    payload=payload,
                )
            )
        for item_index, item in enumerate(plan_items):
            for criterion_index, success_criterion in enumerate(item.success_criteria):
                criteria.append(
                    self._compile_acceptance_criterion(
                        acceptance=success_criterion,
                        description=f"{item.title} success criterion {criterion_index + 1}",
                        required=True,
                        source_turn_ids=item.source_turn_ids,
                        ordinal=criterion_index,
                        owner_identity={
                            "scope": "plan_item",
                            "ordinal": item_index,
                            "item_id": item.item_id,
                        },
                        target_hint=None,
                    )
                )
        if response_shape is not None:
            payload = _frozen_mapping(response_shape, field_name="response_shape")
            criteria.append(
                VerificationCriterion(
                    criterion_id=_criterion_identity("response_shape", ("response",), 0),
                    kind="response_shape",
                    required=bool(payload.get("required", True)),
                    description=_clean_text(
                        payload.get("description", "response shape"),
                        field_name="response_shape.description",
                        max_chars=400,
                    ),
                    source_turn_ids=("response",),
                    verifier_id="response_shape",
                    payload=payload,
                )
            )
        return tuple(criteria)

    def _compile_acceptance_criterion(
        self,
        *,
        acceptance: Any,
        description: str,
        required: bool,
        source_turn_ids: tuple[str, ...],
        ordinal: int,
        owner_identity: Mapping[str, Any],
        target_hint: str | None,
    ) -> VerificationCriterion:
        if isinstance(acceptance, str):
            legacy_artifact = _legacy_artifact_payload(acceptance)
            if legacy_artifact is not None:
                artifact_payload = dict(legacy_artifact)
                if target_hint is not None:
                    artifact_payload["target_hint"] = target_hint
                return VerificationCriterion(
                    criterion_id=_criterion_identity(
                        "artifact",
                        source_turn_ids,
                        ordinal,
                        owner_identity=owner_identity,
                        target_hint=target_hint,
                    ),
                    kind="artifact",
                    required=required,
                    description=description,
                    source_turn_ids=source_turn_ids,
                    verifier_id="artifact",
                    payload=artifact_payload,
                )
            criterion_kind: CriterionKind = (
                "semantic" if self._semantic_fallback_enabled else "manual"
            )
            verifier_id = (
                _SEMANTIC_VERIFIER_ID if criterion_kind == "semantic" else _MANUAL_VERIFIER_ID
            )
            return VerificationCriterion(
                criterion_id=_criterion_identity(
                    criterion_kind,
                    source_turn_ids,
                    ordinal,
                    owner_identity=owner_identity,
                    target_hint=target_hint,
                ),
                kind=criterion_kind,
                required=required,
                description=description,
                source_turn_ids=source_turn_ids,
                verifier_id=verifier_id,
                payload={"rubric": acceptance, "target_hint": target_hint},
            )
        if isinstance(acceptance, Mapping):
            kind = acceptance.get("kind")
            if kind == "file":
                artifact_payload = dict(acceptance)
                if target_hint is not None and "target_hint" not in artifact_payload:
                    artifact_payload["target_hint"] = target_hint
                return VerificationCriterion(
                    criterion_id=_criterion_identity(
                        "artifact",
                        source_turn_ids,
                        ordinal,
                        owner_identity=owner_identity,
                        target_hint=target_hint,
                    ),
                    kind="artifact",
                    required=required,
                    description=description,
                    source_turn_ids=source_turn_ids,
                    verifier_id="artifact",
                    payload=artifact_payload,
                )
            if kind == "tool_execution":
                return VerificationCriterion(
                    criterion_id=_criterion_identity(
                        "tool_execution",
                        source_turn_ids,
                        ordinal,
                        owner_identity=owner_identity,
                        target_hint=target_hint,
                    ),
                    kind="tool_execution",
                    required=required,
                    description=description,
                    source_turn_ids=source_turn_ids,
                    verifier_id="tool_execution",
                    payload=acceptance,
                )
        criterion_kind = "manual"
        return VerificationCriterion(
            criterion_id=_criterion_identity(
                criterion_kind,
                source_turn_ids,
                ordinal,
                owner_identity=owner_identity,
                target_hint=target_hint,
            ),
            kind=criterion_kind,
            required=required,
            description=description,
            source_turn_ids=source_turn_ids,
            verifier_id=_MANUAL_VERIFIER_ID,
            payload={"reason": "unsupported_acceptance_shape"},
        )


def _resolve_state_path(state: Mapping[str, Any], path: str) -> tuple[Any, bool]:
    current: Any = state
    for segment in path.split("."):
        if not isinstance(current, Mapping) or segment not in current:
            return None, False
        current = current[segment]
    return current, True


__all__ = [
    "ArtifactVerifierAdapter",
    "CriterionKind",
    "CriterionReceipt",
    "CriterionVerdict",
    "DeterministicVerifierRegistry",
    "ManualVerifier",
    "OutcomeVerifier",
    "ResponseShapeVerifier",
    "SemanticJudge",
    "SemanticJudgeVerifier",
    "StateVerifier",
    "ToolExecutionVerifier",
    "VERIFICATION_RECEIPT_EVENT",
    "VERIFICATION_RECEIPT_EVENT_SCHEMA_VERSION",
    "VERIFICATION_RECEIPT_VERSION",
    "VerificationCriterion",
    "VerificationEvidence",
    "VerificationPlanCompiler",
    "VerificationReceipt",
    "VerificationReceiptLoadResult",
    "VerificationReceiptRepository",
    "VerificationReceiptSaveResult",
]
