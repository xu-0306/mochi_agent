"""Versioned contracts for conversation-resolved turn intent.

The types in this module deliberately describe user intent and task state without
granting tool access or security authorization.  They are safe inputs to a later
capability planner, but they are not policy decisions.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, cast

TurnOperation = Literal[
    "conversation",
    "open_world_lookup",
    "literature_research",
    "workspace_read",
    "workspace_write",
    "execution",
    "tool_discovery",
]
MutationRequirement = Literal["required", "forbidden", "unknown"]
SpeechAct = Literal[
    "request_information",
    "request_execution",
    "clarification",
    "constraint",
    "task_update",
    "side_question",
    "cancel",
    "unknown",
]
ReferenceStatus = Literal["resolved", "unresolved"]
DeliverableStatus = Literal["pending", "in_progress", "satisfied", "cancelled"]
ActiveTaskStatus = Literal["active", "paused", "blocked", "completed", "cancelled"]
EvidenceSource = Literal[
    "current_turn", "recent_history", "summary", "active_task", "interpreter"
]

TURN_INTENT_CONTRACT_VERSION = "turn-intent-v1"

_ALLOWED_OPERATIONS = frozenset(
    {
        "conversation",
        "open_world_lookup",
        "literature_research",
        "workspace_read",
        "workspace_write",
        "execution",
        "tool_discovery",
    }
)
_ALLOWED_MUTATION_REQUIREMENTS = frozenset({"required", "forbidden", "unknown"})
_ALLOWED_SPEECH_ACTS = frozenset(
    {
        "request_information",
        "request_execution",
        "clarification",
        "constraint",
        "task_update",
        "side_question",
        "cancel",
        "unknown",
    }
)
_ALLOWED_REFERENCE_STATUSES = frozenset({"resolved", "unresolved"})
_ALLOWED_DELIVERABLE_STATUSES = frozenset(
    {"pending", "in_progress", "satisfied", "cancelled"}
)
_ALLOWED_ACTIVE_TASK_STATUSES = frozenset(
    {"active", "paused", "blocked", "completed", "cancelled"}
)
_ALLOWED_EVIDENCE_SOURCES = frozenset(
    {"current_turn", "recent_history", "summary", "active_task", "interpreter"}
)
_ACCEPTANCE_CRITERION_SCHEMA_VERSION = 1
_ACCEPTANCE_FILE_CHECKS = frozenset({"exists", "non_empty", "contains", "sha256"})
_ACCEPTANCE_TOOL_EXECUTION_CHECKS = frozenset({"test", "lint"})
_ACCEPTANCE_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _unsupported_operations(values: frozenset[TurnOperation]) -> set[str]:
    return {str(value) for value in values if value not in _ALLOWED_OPERATIONS}


def _clean_text(value: str, *, field_name: str, allow_empty: bool = False) -> str:
    normalized = " ".join(str(value or "").split())
    if not normalized and not allow_empty:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _clean_turn_ids(
    values: tuple[str, ...], *, field_name: str = "source_turn_ids"
) -> tuple[str, ...]:
    cleaned = tuple(
        dict.fromkeys(_clean_text(value, field_name=field_name) for value in values)
    )
    if not cleaned:
        raise ValueError(f"{field_name} must contain at least one turn id")
    return cleaned


def _clean_text_tuple(values: tuple[str, ...], *, field_name: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(_clean_text(value, field_name=field_name) for value in values)
    )


def _clone_json_value(value: Any, *, field_name: str) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    if isinstance(value, list):
        return [_clone_json_value(item, field_name=field_name) for item in value]
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError(f"{field_name} object keys must be strings")
        return {
            key: _clone_json_value(item, field_name=field_name)
            for key, item in value.items()
        }
    raise TypeError(f"{field_name} must contain JSON-compatible values")


def _clean_acceptance_criteria(
    values: tuple[Any, ...],
    *,
    field_name: str,
) -> tuple[Any, ...]:
    normalized: list[Any] = []
    seen: set[str] = set()
    for index, value in enumerate(values):
        item_field_name = f"{field_name}[{index}]"
        if isinstance(value, str):
            item: Any = _clean_text(value, field_name=item_field_name)
        elif isinstance(value, Mapping):
            item = _validate_structured_acceptance_criterion(
                value,
                field_name=item_field_name,
            )
        else:
            raise TypeError(f"{field_name} entries must be strings or objects")
        identity = json.dumps(
            dict(item) if isinstance(item, Mapping) else item,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        if identity not in seen:
            seen.add(identity)
            normalized.append(item)
    return tuple(normalized)


def _validate_structured_acceptance_criterion(
    value: Mapping[str, Any],
    *,
    field_name: str,
) -> Mapping[str, Any]:
    """Validate the v1 criterion envelope before it reaches a verifier.

    Legacy strings remain data-only file checks.  Structured criteria are a
    small, versioned contract shared by model output and durable turn state;
    they contain profile identifiers and evidence pins, never shell commands.
    """

    payload = _clone_json_value(value, field_name=field_name)
    if not isinstance(payload, dict):  # Defensive: mappings clone to dicts.
        raise TypeError(f"{field_name} must be an object")
    schema_version = payload.get("schema_version")
    if type(schema_version) is not int:
        raise TypeError(f"{field_name}.schema_version must be an integer")
    if schema_version != _ACCEPTANCE_CRITERION_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported {field_name}.schema_version: {schema_version!r}"
        )
    kind = payload.get("kind")
    if kind == "file":
        _validate_file_acceptance_criterion(payload, field_name=field_name)
    elif kind == "tool_execution":
        _validate_tool_execution_acceptance_criterion(payload, field_name=field_name)
    else:
        raise ValueError(f"unsupported {field_name}.kind: {kind!r}")
    return MappingProxyType(payload)


def _validate_file_acceptance_criterion(
    payload: Mapping[str, Any],
    *,
    field_name: str,
) -> None:
    check = payload.get("check")
    if not isinstance(check, str) or check not in _ACCEPTANCE_FILE_CHECKS:
        raise ValueError(f"unsupported {field_name}.check: {check!r}")
    fields = {"schema_version", "kind", "check"}
    if check in {"contains", "sha256"}:
        fields.add("value")
        value = payload.get("value")
        if not isinstance(value, str) or not value:
            raise TypeError(f"{field_name}.value must be a non-empty string")
        if check == "sha256" and _ACCEPTANCE_SHA256_RE.fullmatch(value) is None:
            raise ValueError(f"{field_name}.value must be a lower-case SHA-256 digest")
    _require_exact_acceptance_fields(payload, fields, field_name=field_name)


def _validate_tool_execution_acceptance_criterion(
    payload: Mapping[str, Any],
    *,
    field_name: str,
) -> None:
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
    _require_exact_acceptance_fields(
        payload,
        required | optional,
        field_name=field_name,
        allow_missing=optional,
    )
    check = payload.get("check")
    if not isinstance(check, str) or check not in _ACCEPTANCE_TOOL_EXECUTION_CHECKS:
        raise ValueError(f"unsupported {field_name}.check: {check!r}")
    for key in ("tool_name", "profile_id"):
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise TypeError(f"{field_name}.{key} must be a non-empty string")
    for key in ("call_id", "operation_id", "turn_id"):
        if key in payload:
            value = payload[key]
            if not isinstance(value, str) or not value.strip():
                raise TypeError(f"{field_name}.{key} must be a non-empty string")
    if "arguments_digest" in payload:
        digest = payload["arguments_digest"]
        if not isinstance(digest, str) or _ACCEPTANCE_SHA256_RE.fullmatch(digest) is None:
            raise ValueError(f"{field_name}.arguments_digest must be a SHA-256 digest")
    if "expected_exit_code" in payload and type(payload["expected_exit_code"]) is not int:
        raise TypeError(f"{field_name}.expected_exit_code must be an integer")


def _require_exact_acceptance_fields(
    payload: Mapping[str, Any],
    fields: set[str],
    *,
    field_name: str,
    allow_missing: set[str] | None = None,
) -> None:
    allowed_missing = allow_missing or set()
    missing = sorted(fields - set(payload) - allowed_missing)
    unexpected = sorted(set(payload) - fields)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append(f"missing fields: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected fields: {', '.join(unexpected)}")
        raise ValueError(f"{field_name} must have exact fields ({'; '.join(details)})")


def _acceptance_criteria_to_list(values: tuple[Any, ...]) -> list[Any]:
    return [dict(item) if isinstance(item, Mapping) else item for item in values]


def _strict_acceptance_criteria(value: Any, *, field_name: str) -> tuple[Any, ...]:
    return _clean_acceptance_criteria(
        tuple(_strict_list(value, field_name=field_name)),
        field_name=field_name,
    )


def _validate_confidence(value: float) -> float:
    normalized = float(value)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")
    return normalized


def _strict_mapping(
    value: Any,
    *,
    field_name: str,
    fields: frozenset[str],
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{field_name} keys must be strings")
    keys = set(value)
    missing = sorted(fields - keys)
    unexpected = sorted(keys - fields)
    if missing:
        raise ValueError(f"{field_name} missing required fields: {missing}")
    if unexpected:
        raise ValueError(f"{field_name} has unexpected fields: {unexpected}")
    return value


def _strict_string(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    return value


def _strict_optional_string(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _strict_string(value, field_name=field_name)


def _strict_bool(value: Any, *, field_name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be a boolean")
    return value


def _strict_number(value: Any, *, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number")
    return float(value)


def _strict_list(value: Any, *, field_name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be an array")
    return value


def _strict_string_tuple(value: Any, *, field_name: str) -> tuple[str, ...]:
    return tuple(
        _strict_string(item, field_name=f"{field_name}[]")
        for item in _strict_list(value, field_name=field_name)
    )


def _strict_operations(value: Any, *, field_name: str) -> frozenset[TurnOperation]:
    operations = frozenset(_strict_string_tuple(value, field_name=field_name))
    invalid = {item for item in operations if item not in _ALLOWED_OPERATIONS}
    if invalid:
        raise ValueError(f"unsupported {field_name}: {sorted(invalid)}")
    return cast(frozenset[TurnOperation], operations)


@dataclass(frozen=True)
class ResolvedReference:
    """A surface reference and the context evidence used to resolve it."""

    surface: str
    resolved_to: str | None
    source_turn_ids: tuple[str, ...]
    status: ReferenceStatus = "resolved"

    def __post_init__(self) -> None:
        if self.status not in _ALLOWED_REFERENCE_STATUSES:
            raise ValueError(f"unsupported reference status: {self.status!r}")
        object.__setattr__(
            self, "surface", _clean_text(self.surface, field_name="surface")
        )
        object.__setattr__(
            self,
            "source_turn_ids",
            _clean_turn_ids(self.source_turn_ids),
        )
        resolved_to = (
            _clean_text(self.resolved_to, field_name="resolved_to")
            if self.resolved_to is not None
            else None
        )
        if self.status == "resolved" and resolved_to is None:
            raise ValueError("resolved references require resolved_to")
        if self.status == "unresolved" and resolved_to is not None:
            raise ValueError("unresolved references cannot provide resolved_to")
        object.__setattr__(self, "resolved_to", resolved_to)

    def to_dict(self) -> dict[str, Any]:
        return {
            "surface": self.surface,
            "resolved_to": self.resolved_to,
            "source_turn_ids": list(self.source_turn_ids),
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ResolvedReference:
        value = _strict_mapping(
            payload,
            field_name="resolved_reference",
            fields=frozenset(
                {"surface", "resolved_to", "source_turn_ids", "status"}
            ),
        )
        return cls(
            surface=_strict_string(value["surface"], field_name="surface"),
            resolved_to=_strict_optional_string(
                value["resolved_to"], field_name="resolved_to"
            ),
            source_turn_ids=_strict_string_tuple(
                value["source_turn_ids"], field_name="source_turn_ids"
            ),
            status=_strict_string(value["status"], field_name="status"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class IntentConstraint:
    """A positive or negative user constraint with provenance."""

    text: str
    source_turn_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "text", _clean_text(self.text, field_name="constraint.text")
        )
        object.__setattr__(
            self, "source_turn_ids", _clean_turn_ids(self.source_turn_ids)
        )

    def to_dict(self) -> dict[str, Any]:
        return {"text": self.text, "source_turn_ids": list(self.source_turn_ids)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> IntentConstraint:
        value = _strict_mapping(
            payload,
            field_name="intent_constraint",
            fields=frozenset({"text", "source_turn_ids"}),
        )
        return cls(
            text=_strict_string(value["text"], field_name="constraint.text"),
            source_turn_ids=_strict_string_tuple(
                value["source_turn_ids"], field_name="source_turn_ids"
            ),
        )


@dataclass(frozen=True)
class DeliverableContract:
    """One requested outcome and its independently testable acceptance criteria."""

    kind: str
    source_turn_ids: tuple[str, ...]
    target_hint: str | None = None
    required: bool = True
    acceptance_criteria: tuple[Any, ...] = ()
    status: DeliverableStatus = "pending"

    def __post_init__(self) -> None:
        if self.status not in _ALLOWED_DELIVERABLE_STATUSES:
            raise ValueError(f"unsupported deliverable status: {self.status!r}")
        object.__setattr__(
            self, "kind", _clean_text(self.kind, field_name="deliverable.kind")
        )
        object.__setattr__(
            self, "source_turn_ids", _clean_turn_ids(self.source_turn_ids)
        )
        if self.target_hint is not None:
            object.__setattr__(
                self,
                "target_hint",
                _clean_text(self.target_hint, field_name="deliverable.target_hint"),
            )
        object.__setattr__(
            self,
            "acceptance_criteria",
            _clean_acceptance_criteria(
                self.acceptance_criteria, field_name="acceptance_criteria"
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "target_hint": self.target_hint,
            "required": self.required,
            "acceptance_criteria": _acceptance_criteria_to_list(self.acceptance_criteria),
            "status": self.status,
            "source_turn_ids": list(self.source_turn_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DeliverableContract:
        value = _strict_mapping(
            payload,
            field_name="deliverable",
            fields=frozenset(
                {
                    "kind",
                    "target_hint",
                    "required",
                    "acceptance_criteria",
                    "status",
                    "source_turn_ids",
                }
            ),
        )
        return cls(
            kind=_strict_string(value["kind"], field_name="deliverable.kind"),
            target_hint=_strict_optional_string(
                value["target_hint"], field_name="deliverable.target_hint"
            ),
            required=_strict_bool(value["required"], field_name="deliverable.required"),
            acceptance_criteria=_strict_acceptance_criteria(
                value["acceptance_criteria"], field_name="acceptance_criteria"
            ),
            status=_strict_string(value["status"], field_name="deliverable.status"),  # type: ignore[arg-type]
            source_turn_ids=_strict_string_tuple(
                value["source_turn_ids"], field_name="source_turn_ids"
            ),
        )


@dataclass(frozen=True)
class ClarificationRequest:
    """A bounded clarification required before capability planning can proceed."""

    question: str
    missing_fields: tuple[str, ...]
    source_turn_ids: tuple[str, ...]
    reason_code: str = "material_ambiguity"

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "question", _clean_text(self.question, field_name="question")
        )
        object.__setattr__(
            self,
            "missing_fields",
            _clean_text_tuple(self.missing_fields, field_name="missing_fields"),
        )
        if not self.missing_fields:
            raise ValueError("clarification must identify at least one missing field")
        object.__setattr__(
            self, "source_turn_ids", _clean_turn_ids(self.source_turn_ids)
        )
        object.__setattr__(
            self, "reason_code", _clean_text(self.reason_code, field_name="reason_code")
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "missing_fields": list(self.missing_fields),
            "source_turn_ids": list(self.source_turn_ids),
            "reason_code": self.reason_code,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ClarificationRequest:
        value = _strict_mapping(
            payload,
            field_name="clarification",
            fields=frozenset(
                {"question", "missing_fields", "source_turn_ids", "reason_code"}
            ),
        )
        return cls(
            question=_strict_string(value["question"], field_name="question"),
            missing_fields=_strict_string_tuple(
                value["missing_fields"], field_name="missing_fields"
            ),
            source_turn_ids=_strict_string_tuple(
                value["source_turn_ids"], field_name="source_turn_ids"
            ),
            reason_code=_strict_string(
                value["reason_code"], field_name="reason_code"
            ),
        )


@dataclass(frozen=True)
class IntentEvidence:
    """Auditable evidence supporting one semantic interpretation."""

    statement: str
    source: EvidenceSource
    source_turn_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.source not in _ALLOWED_EVIDENCE_SOURCES:
            raise ValueError(f"unsupported evidence source: {self.source!r}")
        object.__setattr__(
            self,
            "statement",
            _clean_text(self.statement, field_name="evidence.statement"),
        )
        object.__setattr__(
            self, "source_turn_ids", _clean_turn_ids(self.source_turn_ids)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "statement": self.statement,
            "source": self.source,
            "source_turn_ids": list(self.source_turn_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> IntentEvidence:
        value = _strict_mapping(
            payload,
            field_name="intent_evidence",
            fields=frozenset({"statement", "source", "source_turn_ids"}),
        )
        return cls(
            statement=_strict_string(
                value["statement"], field_name="evidence.statement"
            ),
            source=_strict_string(value["source"], field_name="evidence.source"),  # type: ignore[arg-type]
            source_turn_ids=_strict_string_tuple(
                value["source_turn_ids"], field_name="source_turn_ids"
            ),
        )


@dataclass(frozen=True)
class IntentAdvisory:
    """Non-authoritative classifier telemetry attached after resolution."""

    label: str
    confidence: float | None
    rationale: str
    recommended_operations: frozenset[TurnOperation] = field(default_factory=frozenset)
    source_turn_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "label", _clean_text(self.label, field_name="advisory.label")
        )
        object.__setattr__(
            self,
            "rationale",
            _clean_text(self.rationale, field_name="advisory.rationale"),
        )
        if self.confidence is not None:
            object.__setattr__(
                self, "confidence", _validate_confidence(self.confidence)
            )
        invalid = _unsupported_operations(self.recommended_operations)
        if invalid:
            raise ValueError(f"unsupported advisory operations: {sorted(invalid)}")
        if self.source_turn_ids:
            object.__setattr__(
                self, "source_turn_ids", _clean_turn_ids(self.source_turn_ids)
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": self.confidence,
            "rationale": self.rationale,
            "recommended_operations": sorted(self.recommended_operations),
            "source_turn_ids": list(self.source_turn_ids),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> IntentAdvisory:
        value = _strict_mapping(
            payload,
            field_name="intent_advisory",
            fields=frozenset(
                {
                    "label",
                    "confidence",
                    "rationale",
                    "recommended_operations",
                    "source_turn_ids",
                }
            ),
        )
        confidence = value["confidence"]
        return cls(
            label=_strict_string(value["label"], field_name="advisory.label"),
            confidence=(
                _strict_number(confidence, field_name="advisory.confidence")
                if confidence is not None
                else None
            ),
            rationale=_strict_string(
                value["rationale"], field_name="advisory.rationale"
            ),
            recommended_operations=_strict_operations(
                value["recommended_operations"],
                field_name="recommended_operations",
            ),
            source_turn_ids=_strict_string_tuple(
                value["source_turn_ids"], field_name="source_turn_ids"
            ),
        )


@dataclass(frozen=True)
class ActiveTaskState:
    """Durable cross-turn state; never contains per-turn execution state."""

    goal_id: str
    objective: str
    status: ActiveTaskStatus = "active"
    operations: frozenset[TurnOperation] = field(default_factory=frozenset)
    mutation_requirement: MutationRequirement = "unknown"
    deliverables: tuple[DeliverableContract, ...] = ()
    positive_constraints: tuple[IntentConstraint, ...] = ()
    negative_constraints: tuple[IntentConstraint, ...] = ()
    decisions: tuple[str, ...] = ()
    source_turn_ids: tuple[str, ...] = ()
    updated_turn_id: str | None = None

    def __post_init__(self) -> None:
        if self.status not in _ALLOWED_ACTIVE_TASK_STATUSES:
            raise ValueError(f"unsupported active-task status: {self.status!r}")
        if self.mutation_requirement not in _ALLOWED_MUTATION_REQUIREMENTS:
            raise ValueError(
                f"unsupported task mutation requirement: {self.mutation_requirement!r}"
            )
        object.__setattr__(
            self, "goal_id", _clean_text(self.goal_id, field_name="goal_id")
        )
        object.__setattr__(
            self, "objective", _clean_text(self.objective, field_name="objective")
        )
        invalid = _unsupported_operations(self.operations)
        if invalid:
            raise ValueError(f"unsupported task operations: {sorted(invalid)}")
        if (
            self.mutation_requirement == "forbidden"
            and "workspace_write" in self.operations
        ):
            raise ValueError("workspace_write conflicts with forbidden task mutation")
        if (
            self.mutation_requirement == "required"
            and "workspace_write" not in self.operations
        ):
            raise ValueError("required task mutation must include workspace_write")
        object.__setattr__(
            self, "decisions", _clean_text_tuple(self.decisions, field_name="decisions")
        )
        if self.source_turn_ids:
            object.__setattr__(
                self, "source_turn_ids", _clean_turn_ids(self.source_turn_ids)
            )
        if self.updated_turn_id is not None:
            object.__setattr__(
                self,
                "updated_turn_id",
                _clean_text(self.updated_turn_id, field_name="updated_turn_id"),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal_id": self.goal_id,
            "objective": self.objective,
            "status": self.status,
            "operations": sorted(self.operations),
            "mutation_requirement": self.mutation_requirement,
            "deliverables": [item.to_dict() for item in self.deliverables],
            "positive_constraints": [
                item.to_dict() for item in self.positive_constraints
            ],
            "negative_constraints": [
                item.to_dict() for item in self.negative_constraints
            ],
            "decisions": list(self.decisions),
            "source_turn_ids": list(self.source_turn_ids),
            "updated_turn_id": self.updated_turn_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ActiveTaskState:
        value = _strict_mapping(
            payload,
            field_name="active_task_state",
            fields=frozenset(
                {
                    "goal_id",
                    "objective",
                    "status",
                    "operations",
                    "mutation_requirement",
                    "deliverables",
                    "positive_constraints",
                    "negative_constraints",
                    "decisions",
                    "source_turn_ids",
                    "updated_turn_id",
                }
            ),
        )
        return cls(
            goal_id=_strict_string(value["goal_id"], field_name="goal_id"),
            objective=_strict_string(value["objective"], field_name="objective"),
            status=_strict_string(value["status"], field_name="status"),  # type: ignore[arg-type]
            operations=_strict_operations(value["operations"], field_name="operations"),
            mutation_requirement=_strict_string(
                value["mutation_requirement"], field_name="mutation_requirement"
            ),  # type: ignore[arg-type]
            deliverables=tuple(
                DeliverableContract.from_dict(item)
                for item in _strict_list(
                    value["deliverables"], field_name="deliverables"
                )
            ),
            positive_constraints=tuple(
                IntentConstraint.from_dict(item)
                for item in _strict_list(
                    value["positive_constraints"], field_name="positive_constraints"
                )
            ),
            negative_constraints=tuple(
                IntentConstraint.from_dict(item)
                for item in _strict_list(
                    value["negative_constraints"], field_name="negative_constraints"
                )
            ),
            decisions=_strict_string_tuple(value["decisions"], field_name="decisions"),
            source_turn_ids=_strict_string_tuple(
                value["source_turn_ids"], field_name="source_turn_ids"
            ),
            updated_turn_id=_strict_optional_string(
                value["updated_turn_id"], field_name="updated_turn_id"
            ),
        )


@dataclass(frozen=True)
class TurnIntentContract:
    """Authoritative semantic contract consumed by future capability planning."""

    turn_id: str
    active_goal_id: str | None
    objective: str
    current_speech_act: SpeechAct
    operations: frozenset[TurnOperation]
    deliverables: tuple[DeliverableContract, ...]
    resolved_references: tuple[ResolvedReference, ...]
    positive_constraints: tuple[IntentConstraint, ...]
    negative_constraints: tuple[IntentConstraint, ...]
    mutation_requirement: MutationRequirement
    clarification: ClarificationRequest | None
    supersedes_previous_goal: bool
    cancels_active_goal: bool
    modifies_active_task: bool
    confidence: float
    evidence: tuple[IntentEvidence, ...]
    advisories: tuple[IntentAdvisory, ...] = ()
    contract_version: str = TURN_INTENT_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.current_speech_act not in _ALLOWED_SPEECH_ACTS:
            raise ValueError(f"unsupported speech act: {self.current_speech_act!r}")
        if self.mutation_requirement not in _ALLOWED_MUTATION_REQUIREMENTS:
            raise ValueError(
                f"unsupported turn mutation requirement: {self.mutation_requirement!r}"
            )
        object.__setattr__(
            self, "turn_id", _clean_text(self.turn_id, field_name="turn_id")
        )
        if self.active_goal_id is not None:
            object.__setattr__(
                self,
                "active_goal_id",
                _clean_text(self.active_goal_id, field_name="active_goal_id"),
            )
        object.__setattr__(
            self,
            "objective",
            _clean_text(self.objective, field_name="objective", allow_empty=True),
        )
        invalid = _unsupported_operations(self.operations)
        if invalid:
            raise ValueError(f"unsupported turn operations: {sorted(invalid)}")
        if (
            self.mutation_requirement == "forbidden"
            and "workspace_write" in self.operations
        ):
            raise ValueError("workspace_write conflicts with forbidden mutation")
        if (
            self.mutation_requirement == "required"
            and "workspace_write" not in self.operations
        ):
            raise ValueError("required mutation must include workspace_write")
        if self.mutation_requirement == "required" and any(
            deliverable.required
            and deliverable.status == "satisfied"
            and self.turn_id in deliverable.source_turn_ids
            for deliverable in self.deliverables
        ):
            raise ValueError(
                "current-turn required mutation deliverables cannot already be satisfied"
            )
        if self.cancels_active_goal and self.supersedes_previous_goal:
            raise ValueError(
                "a turn cannot cancel and supersede the active goal simultaneously"
            )
        object.__setattr__(self, "confidence", _validate_confidence(self.confidence))
        object.__setattr__(
            self,
            "contract_version",
            _clean_text(self.contract_version, field_name="contract_version"),
        )
        if self.contract_version != TURN_INTENT_CONTRACT_VERSION:
            raise ValueError(
                f"unsupported turn intent contract version: {self.contract_version!r}"
            )

    @property
    def clarification_needed(self) -> bool:
        return self.clarification is not None

    @property
    def source_turn_ids(self) -> tuple[str, ...]:
        values: list[str] = [self.turn_id]
        for reference in self.resolved_references:
            values.extend(reference.source_turn_ids)
        for deliverable in self.deliverables:
            values.extend(deliverable.source_turn_ids)
        for constraint in (*self.positive_constraints, *self.negative_constraints):
            values.extend(constraint.source_turn_ids)
        for evidence in self.evidence:
            values.extend(evidence.source_turn_ids)
        if self.clarification is not None:
            values.extend(self.clarification.source_turn_ids)
        return tuple(dict.fromkeys(values))

    def with_advisories(
        self, advisories: tuple[IntentAdvisory, ...]
    ) -> TurnIntentContract:
        """Return telemetry-enriched output without changing semantic fields."""

        return TurnIntentContract(
            turn_id=self.turn_id,
            active_goal_id=self.active_goal_id,
            objective=self.objective,
            current_speech_act=self.current_speech_act,
            operations=self.operations,
            deliverables=self.deliverables,
            resolved_references=self.resolved_references,
            positive_constraints=self.positive_constraints,
            negative_constraints=self.negative_constraints,
            mutation_requirement=self.mutation_requirement,
            clarification=self.clarification,
            supersedes_previous_goal=self.supersedes_previous_goal,
            cancels_active_goal=self.cancels_active_goal,
            modifies_active_task=self.modifies_active_task,
            confidence=self.confidence,
            evidence=self.evidence,
            advisories=advisories,
            contract_version=self.contract_version,
        )

    def semantic_projection(self) -> dict[str, Any]:
        """Return fields that an advisory classifier is forbidden to change."""

        payload = self.to_dict()
        payload.pop("advisories", None)
        return payload

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "turn_id": self.turn_id,
            "active_goal_id": self.active_goal_id,
            "objective": self.objective,
            "current_speech_act": self.current_speech_act,
            "resolved_references": [
                item.to_dict() for item in self.resolved_references
            ],
            "operations": sorted(self.operations),
            "deliverables": [item.to_dict() for item in self.deliverables],
            "positive_constraints": [
                item.to_dict() for item in self.positive_constraints
            ],
            "negative_constraints": [
                item.to_dict() for item in self.negative_constraints
            ],
            "mutation_requirement": self.mutation_requirement,
            "supersedes_previous_goal": self.supersedes_previous_goal,
            "cancels_active_goal": self.cancels_active_goal,
            "modifies_active_task": self.modifies_active_task,
            "clarification_needed": self.clarification_needed,
            "clarification": (
                self.clarification.to_dict() if self.clarification else None
            ),
            "confidence": self.confidence,
            "evidence": [item.to_dict() for item in self.evidence],
            "source_turn_ids": list(self.source_turn_ids),
            "advisories": [item.to_dict() for item in self.advisories],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TurnIntentContract:
        value = _strict_mapping(
            payload,
            field_name="turn_intent_contract",
            fields=frozenset(
                {
                    "contract_version",
                    "turn_id",
                    "active_goal_id",
                    "objective",
                    "current_speech_act",
                    "resolved_references",
                    "operations",
                    "deliverables",
                    "positive_constraints",
                    "negative_constraints",
                    "mutation_requirement",
                    "supersedes_previous_goal",
                    "cancels_active_goal",
                    "modifies_active_task",
                    "clarification_needed",
                    "clarification",
                    "confidence",
                    "evidence",
                    "source_turn_ids",
                    "advisories",
                }
            ),
        )
        contract_version = _strict_string(
            value["contract_version"], field_name="contract_version"
        )
        if contract_version != TURN_INTENT_CONTRACT_VERSION:
            raise ValueError(
                f"unsupported turn intent contract version: {contract_version!r}"
            )
        raw_clarification = value["clarification"]
        clarification = (
            ClarificationRequest.from_dict(raw_clarification)
            if raw_clarification is not None
            else None
        )
        result = cls(
            contract_version=contract_version,
            turn_id=_strict_string(value["turn_id"], field_name="turn_id"),
            active_goal_id=_strict_optional_string(
                value["active_goal_id"], field_name="active_goal_id"
            ),
            objective=_strict_string(value["objective"], field_name="objective"),
            current_speech_act=_strict_string(
                value["current_speech_act"], field_name="current_speech_act"
            ),  # type: ignore[arg-type]
            resolved_references=tuple(
                ResolvedReference.from_dict(item)
                for item in _strict_list(
                    value["resolved_references"], field_name="resolved_references"
                )
            ),
            operations=_strict_operations(value["operations"], field_name="operations"),
            deliverables=tuple(
                DeliverableContract.from_dict(item)
                for item in _strict_list(
                    value["deliverables"], field_name="deliverables"
                )
            ),
            positive_constraints=tuple(
                IntentConstraint.from_dict(item)
                for item in _strict_list(
                    value["positive_constraints"], field_name="positive_constraints"
                )
            ),
            negative_constraints=tuple(
                IntentConstraint.from_dict(item)
                for item in _strict_list(
                    value["negative_constraints"], field_name="negative_constraints"
                )
            ),
            mutation_requirement=_strict_string(
                value["mutation_requirement"], field_name="mutation_requirement"
            ),  # type: ignore[arg-type]
            supersedes_previous_goal=_strict_bool(
                value["supersedes_previous_goal"],
                field_name="supersedes_previous_goal",
            ),
            cancels_active_goal=_strict_bool(
                value["cancels_active_goal"], field_name="cancels_active_goal"
            ),
            modifies_active_task=_strict_bool(
                value["modifies_active_task"], field_name="modifies_active_task"
            ),
            clarification=clarification,
            confidence=_strict_number(value["confidence"], field_name="confidence"),
            evidence=tuple(
                IntentEvidence.from_dict(item)
                for item in _strict_list(value["evidence"], field_name="evidence")
            ),
            advisories=tuple(
                IntentAdvisory.from_dict(item)
                for item in _strict_list(value["advisories"], field_name="advisories")
            ),
        )
        clarification_needed = _strict_bool(
            value["clarification_needed"], field_name="clarification_needed"
        )
        if clarification_needed != result.clarification_needed:
            raise ValueError("clarification_needed does not match clarification")
        serialized_source_turn_ids = _strict_string_tuple(
            value["source_turn_ids"], field_name="source_turn_ids"
        )
        if serialized_source_turn_ids != result.source_turn_ids:
            raise ValueError("source_turn_ids do not match contract provenance")
        return result
