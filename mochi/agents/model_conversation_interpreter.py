"""Model-backed adapter for bounded conversation interpretation.

The adapter performs no tool selection, capability grant, or policy decision.
It asks the configured backend for a structured semantic proposal and converts
that proposal into the deterministic resolver's ``IntentInterpretation`` type.
Malformed or unsupported output raises an exception so that
``ConversationResolver`` can use its existing fail-closed clarification path.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import replace
from typing import Any, TypeVar, cast

from mochi.agents.conversation_resolver import (
    BoundedConversationContext,
    IntentInterpretation,
    TaskRelation,
)
from mochi.agents.turn_intent_contract import (
    ClarificationRequest,
    DeliverableContract,
    DeliverableStatus,
    EvidenceSource,
    IntentConstraint,
    IntentEvidence,
    MutationRequirement,
    ReferenceStatus,
    ResolvedReference,
    SpeechAct,
    TurnOperation,
)
from mochi.backends.base import BaseLLMBackend
from mochi.backends.types import GenerationResult, Message

_T = TypeVar("_T", bound=str)

_SPEECH_ACTS = frozenset(
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
_TASK_RELATIONS = frozenset(
    {"continue", "side_question", "start", "supersede", "cancel", "standalone"}
)
_OPERATIONS = frozenset(
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
_MUTATION_REQUIREMENTS = frozenset({"required", "forbidden", "unknown"})
_DELIVERABLE_STATUSES = frozenset({"pending", "in_progress", "satisfied", "cancelled"})
_REFERENCE_STATUSES = frozenset({"resolved", "unresolved"})
_EVIDENCE_SOURCES = frozenset(
    {"current_turn", "recent_history", "summary", "active_task", "interpreter"}
)

INTERPRETATION_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "additionalProperties": False,
    "required": [
        "current_speech_act",
        "task_relation",
        "objective",
        "operations",
        "deliverables",
        "resolved_references",
        "positive_constraints",
        "negative_constraints",
        "mutation_requirement",
        "clarification",
        "confidence",
        "evidence",
    ],
    "properties": {
        "current_speech_act": {"type": "string", "enum": sorted(_SPEECH_ACTS)},
        "task_relation": {"type": "string", "enum": sorted(_TASK_RELATIONS)},
        "objective": {"type": ["string", "null"]},
        "operations": {
            "type": "array",
            "items": {"type": "string", "enum": sorted(_OPERATIONS)},
            "uniqueItems": True,
        },
        "deliverables": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "kind",
                    "target_hint",
                    "required",
                    "acceptance_criteria",
                    "status",
                    "source_turn_ids",
                ],
                "properties": {
                    "kind": {"type": "string", "minLength": 1},
                    "target_hint": {"type": ["string", "null"]},
                    "required": {"type": "boolean"},
                    "acceptance_criteria": {
                        "type": "array",
                        "items": {
                            "oneOf": [
                                {"type": "string", "minLength": 1},
                                {"$ref": "#/$defs/file_acceptance_criterion_v1"},
                                {
                                    "$ref": "#/$defs/tool_execution_acceptance_criterion_v1"
                                },
                            ]
                        },
                    },
                    "status": {
                        "type": "string",
                        "enum": sorted(_DELIVERABLE_STATUSES),
                    },
                    "source_turn_ids": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "minItems": 1,
                        "uniqueItems": True,
                    },
                },
            },
        },
        "resolved_references": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "surface",
                    "resolved_to",
                    "status",
                    "source_turn_ids",
                ],
                "properties": {
                    "surface": {"type": "string", "minLength": 1},
                    "resolved_to": {"type": ["string", "null"]},
                    "status": {
                        "type": "string",
                        "enum": sorted(_REFERENCE_STATUSES),
                    },
                    "source_turn_ids": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "minItems": 1,
                        "uniqueItems": True,
                    },
                },
            },
        },
        "positive_constraints": {"$ref": "#/$defs/constraints"},
        "negative_constraints": {"$ref": "#/$defs/constraints"},
        "mutation_requirement": {
            "type": "string",
            "enum": sorted(_MUTATION_REQUIREMENTS),
        },
        "clarification": {
            "oneOf": [
                {"type": "null"},
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "question",
                        "missing_fields",
                        "source_turn_ids",
                        "reason_code",
                    ],
                    "properties": {
                        "question": {"type": "string", "minLength": 1},
                        "missing_fields": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                            "minItems": 1,
                            "uniqueItems": True,
                        },
                        "source_turn_ids": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                            "minItems": 1,
                            "uniqueItems": True,
                        },
                        "reason_code": {"type": "string", "minLength": 1},
                    },
                },
            ]
        },
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["statement", "source", "source_turn_ids"],
                "properties": {
                    "statement": {"type": "string", "minLength": 1},
                    "source": {
                        "type": "string",
                        "enum": sorted(_EVIDENCE_SOURCES),
                    },
                    "source_turn_ids": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "minItems": 1,
                        "uniqueItems": True,
                    },
                },
            },
        },
    },
    "$defs": {
        "constraints": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["text", "source_turn_ids"],
                "properties": {
                    "text": {"type": "string", "minLength": 1},
                    "source_turn_ids": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                        "minItems": 1,
                        "uniqueItems": True,
                    },
                },
            },
        },
        "file_acceptance_criterion_v1": {
            "oneOf": [
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["schema_version", "kind", "check"],
                    "properties": {
                        "schema_version": {"const": 1},
                        "kind": {"const": "file"},
                        "check": {"enum": ["exists", "non_empty"]},
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["schema_version", "kind", "check", "value"],
                    "properties": {
                        "schema_version": {"const": 1},
                        "kind": {"const": "file"},
                        "check": {"const": "contains"},
                        "value": {"type": "string", "minLength": 1},
                    },
                },
                {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["schema_version", "kind", "check", "value"],
                    "properties": {
                        "schema_version": {"const": 1},
                        "kind": {"const": "file"},
                        "check": {"const": "sha256"},
                        "value": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                    },
                },
            ]
        },
        "tool_execution_acceptance_criterion_v1": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "schema_version",
                "kind",
                "check",
                "tool_name",
                "profile_id",
            ],
            "properties": {
                "schema_version": {"const": 1},
                "kind": {"const": "tool_execution"},
                "check": {"enum": ["test", "lint"]},
                "tool_name": {"type": "string", "minLength": 1},
                "profile_id": {"type": "string", "minLength": 1},
                "call_id": {"type": "string", "minLength": 1},
                "arguments_digest": {
                    "type": "string",
                    "pattern": "^[0-9a-f]{64}$",
                },
                "operation_id": {"type": "string", "minLength": 1},
                "turn_id": {"type": "string", "minLength": 1},
                "expected_exit_code": {"type": "integer"},
            },
        },
    },
}

_SYSTEM_PROMPT = f"""
You are Mochi's bounded conversation interpreter. Convert the supplied structured
conversation context into exactly one semantic interpretation.

This is language understanding only. Do not select tools, grant capabilities,
authorize actions, relax policy, or infer sandbox access.

Rules:
- Use the current turn, recent history, durable summary, and active task together.
- Resolve ellipsis and references only when supported by supplied context.
- task_relation describes how the current turn relates to the durable active task:
  continue, side_question, start, supersede, cancel, or standalone.
- operations is a set and may contain multiple independent operations.
- mutation_requirement is required only for an intended persistent workspace
  change, forbidden when mutation is explicitly ruled out, and unknown otherwise.
- A required deliverable for a persistent workspace change requested by the
  current turn must be pending or in_progress. Use satisfied only for a durable
  deliverable that was already completed before the current turn and is not being
  reopened by this request.
- acceptance_criteria may be a legacy file-check string or a version 1
  structured file/tool_execution criterion. A tool_execution criterion names
  a host-owned profile and already-observed tool evidence; it must never contain
  a command, shell, executable, or environment. Natural language such as
  "tests pass" is only a description and never instructs command execution.
- A side question must describe only the side question's current operations and
  deliverables; do not silently continue the active task in the same turn.
- Preserve positive and negative constraints separately.
- Ask for clarification only when missing information would materially change the
  outcome. Otherwise make the narrowest supported interpretation.
- Every deliverable, reference, constraint, clarification, and evidence item must
  cite only source_turn_ids listed in available_source_turn_ids.
- An unresolved reference has status unresolved and resolved_to null. A resolved
  reference has status resolved and a non-null resolved_to.
- Return one JSON object only. Do not use Markdown or explanatory prose.

The output must conform exactly to this JSON Schema:
{json.dumps(INTERPRETATION_JSON_SCHEMA, ensure_ascii=False, sort_keys=True)}
""".strip()

_ROOT_KEYS = frozenset(INTERPRETATION_JSON_SCHEMA["required"])
_DELIVERABLE_KEYS = frozenset(
    {
        "kind",
        "target_hint",
        "required",
        "acceptance_criteria",
        "status",
        "source_turn_ids",
    }
)
_REFERENCE_KEYS = frozenset({"surface", "resolved_to", "status", "source_turn_ids"})
_CONSTRAINT_KEYS = frozenset({"text", "source_turn_ids"})
_CLARIFICATION_KEYS = frozenset(
    {"question", "missing_fields", "source_turn_ids", "reason_code"}
)
_EVIDENCE_KEYS = frozenset({"statement", "source", "source_turn_ids"})


class ModelConversationInterpreter:
    """Provider-agnostic ``ConversationInterpreter`` backed by ``generate``."""

    def __init__(self, backend: BaseLLMBackend, *, max_tokens: int = 2_400) -> None:
        if max_tokens < 256:
            raise ValueError("max_tokens must be at least 256")
        self._backend = backend
        self._max_tokens = max_tokens

    async def interpret(
        self, context: BoundedConversationContext
    ) -> IntentInterpretation:
        payload = conversation_context_payload(context)
        result = await self._backend.generate(
            [
                Message(role="system", content=_SYSTEM_PROMPT),
                Message(
                    role="user",
                    content=json.dumps(payload, ensure_ascii=False, sort_keys=True),
                ),
            ],
            tools=None,
            temperature=0.0,
            max_tokens=self._max_tokens,
            top_p=1.0,
            stream=False,
        )
        if not isinstance(result, GenerationResult):
            raise RuntimeError(
                "Conversation interpreter expected a non-stream backend response."
            )
        interpretation = parse_model_interpretation(result.content)
        return _normalize_current_turn_artifact_status(
            interpretation,
            current_turn_id=context.current_turn.turn_id,
        )


def conversation_context_payload(
    context: BoundedConversationContext,
) -> dict[str, Any]:
    """Serialize only the bounded resolver context required for interpretation."""

    return {
        "current_turn": {
            "turn_id": context.current_turn.turn_id,
            "role": context.current_turn.role,
            "content": context.current_turn.content,
        },
        "recent_history": [
            {"turn_id": turn.turn_id, "role": turn.role, "content": turn.content}
            for turn in context.recent_history
        ],
        "summary": (
            {
                "content": context.summary.content,
                "source_turn_ids": list(context.summary.source_turn_ids),
            }
            if context.summary is not None
            else None
        ),
        "active_task": (
            context.active_task.to_dict() if context.active_task is not None else None
        ),
        "available_source_turn_ids": sorted(context.available_source_turn_ids),
        "omitted_history_count": context.omitted_history_count,
        "truncated_fields": list(context.truncated_fields),
    }


def parse_model_interpretation(text: str) -> IntentInterpretation:
    """Parse and structurally validate one model-produced interpretation."""

    payload = _extract_json_object(text)
    _require_exact_keys(payload, _ROOT_KEYS, label="interpretation")

    operations = frozenset(
        _enum(value, _OPERATIONS, label="operations item")
        for value in _string_list(payload, "operations")
    )
    deliverables = tuple(
        _parse_deliverable(item, index=index)
        for index, item in enumerate(_object_list(payload, "deliverables"))
    )
    references = tuple(
        _parse_reference(item, index=index)
        for index, item in enumerate(_object_list(payload, "resolved_references"))
    )
    positive_constraints = tuple(
        _parse_constraint(item, label=f"positive_constraints[{index}]")
        for index, item in enumerate(_object_list(payload, "positive_constraints"))
    )
    negative_constraints = tuple(
        _parse_constraint(item, label=f"negative_constraints[{index}]")
        for index, item in enumerate(_object_list(payload, "negative_constraints"))
    )
    evidence = tuple(
        _parse_evidence(item, index=index)
        for index, item in enumerate(_object_list(payload, "evidence"))
    )

    raw_confidence = payload["confidence"]
    if isinstance(raw_confidence, bool) or not isinstance(raw_confidence, (int, float)):
        raise ValueError("confidence must be a number")
    confidence = float(raw_confidence)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be between 0.0 and 1.0")

    raw_objective = payload["objective"]
    if raw_objective is not None and not isinstance(raw_objective, str):
        raise ValueError("objective must be a string or null")
    objective = raw_objective.strip() if isinstance(raw_objective, str) else None
    if objective == "":
        objective = None

    return IntentInterpretation(
        current_speech_act=cast(
            SpeechAct,
            _enum(
                payload["current_speech_act"],
                _SPEECH_ACTS,
                label="current_speech_act",
            ),
        ),
        task_relation=cast(
            TaskRelation,
            _enum(payload["task_relation"], _TASK_RELATIONS, label="task_relation"),
        ),
        objective=objective,
        operations=cast(frozenset[TurnOperation], operations),
        deliverables=deliverables,
        resolved_references=references,
        positive_constraints=positive_constraints,
        negative_constraints=negative_constraints,
        mutation_requirement=cast(
            MutationRequirement,
            _enum(
                payload["mutation_requirement"],
                _MUTATION_REQUIREMENTS,
                label="mutation_requirement",
            ),
        ),
        clarification=_parse_clarification(payload["clarification"]),
        confidence=confidence,
        evidence=evidence,
    )


def _normalize_current_turn_artifact_status(
    interpretation: IntentInterpretation,
    *,
    current_turn_id: str,
) -> IntentInterpretation:
    """Keep model-declared completion from bypassing this turn's write obligation."""

    if interpretation.mutation_requirement != "required":
        return interpretation
    deliverables = tuple(
        replace(deliverable, status="pending")
        if deliverable.required
        and deliverable.status == "satisfied"
        and current_turn_id in deliverable.source_turn_ids
        else deliverable
        for deliverable in interpretation.deliverables
    )
    if deliverables == interpretation.deliverables:
        return interpretation
    return replace(interpretation, deliverables=deliverables)


def _parse_deliverable(payload: dict[str, Any], *, index: int) -> DeliverableContract:
    label = f"deliverables[{index}]"
    _require_exact_keys(payload, _DELIVERABLE_KEYS, label=label)
    raw_required = payload["required"]
    if not isinstance(raw_required, bool):
        raise ValueError(f"{label}.required must be a boolean")
    return DeliverableContract(
        kind=_required_text(payload["kind"], label=f"{label}.kind"),
        target_hint=_optional_text(
            payload["target_hint"], label=f"{label}.target_hint"
        ),
        required=raw_required,
        acceptance_criteria=_acceptance_criteria(
            payload["acceptance_criteria"],
            label=label,
        ),
        status=cast(
            DeliverableStatus,
            _enum(payload["status"], _DELIVERABLE_STATUSES, label=f"{label}.status"),
        ),
        source_turn_ids=tuple(_source_turn_ids(payload, label=label)),
    )


def _acceptance_criteria(value: Any, *, label: str) -> tuple[Any, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{label}.acceptance_criteria must be an array")
    criteria: list[Any] = []
    for index, item in enumerate(value):
        item_label = f"{label}.acceptance_criteria[{index}]"
        if isinstance(item, str):
            criteria.append(_required_text(item, label=item_label))
            continue
        if not isinstance(item, Mapping) or any(
            not isinstance(key, str) for key in item
        ):
            raise ValueError(f"{item_label} must be a string or object")
        # DeliverableContract owns the shared strict v1 criterion validation
        # used by both persisted state and downstream artifact verification.
        criteria.append(dict(item))
    return tuple(criteria)


def _parse_reference(payload: dict[str, Any], *, index: int) -> ResolvedReference:
    label = f"resolved_references[{index}]"
    _require_exact_keys(payload, _REFERENCE_KEYS, label=label)
    return ResolvedReference(
        surface=_required_text(payload["surface"], label=f"{label}.surface"),
        resolved_to=_optional_text(
            payload["resolved_to"], label=f"{label}.resolved_to"
        ),
        status=cast(
            ReferenceStatus,
            _enum(payload["status"], _REFERENCE_STATUSES, label=f"{label}.status"),
        ),
        source_turn_ids=tuple(_source_turn_ids(payload, label=label)),
    )


def _parse_constraint(payload: dict[str, Any], *, label: str) -> IntentConstraint:
    _require_exact_keys(payload, _CONSTRAINT_KEYS, label=label)
    return IntentConstraint(
        text=_required_text(payload["text"], label=f"{label}.text"),
        source_turn_ids=tuple(_source_turn_ids(payload, label=label)),
    )


def _parse_clarification(value: Any) -> ClarificationRequest | None:
    if value is None:
        return None
    payload = _mapping(value, label="clarification")
    _require_exact_keys(payload, _CLARIFICATION_KEYS, label="clarification")
    return ClarificationRequest(
        question=_required_text(payload["question"], label="clarification.question"),
        missing_fields=tuple(
            _string_list(payload, "missing_fields", label="clarification")
        ),
        source_turn_ids=tuple(_source_turn_ids(payload, label="clarification")),
        reason_code=_required_text(
            payload["reason_code"], label="clarification.reason_code"
        ),
    )


def _parse_evidence(payload: dict[str, Any], *, index: int) -> IntentEvidence:
    label = f"evidence[{index}]"
    _require_exact_keys(payload, _EVIDENCE_KEYS, label=label)
    return IntentEvidence(
        statement=_required_text(payload["statement"], label=f"{label}.statement"),
        source=cast(
            EvidenceSource,
            _enum(payload["source"], _EVIDENCE_SOURCES, label=f"{label}.source"),
        ),
        source_turn_ids=tuple(_source_turn_ids(payload, label=label)),
    )


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    if not stripped:
        raise ValueError("model returned an empty interpretation")

    candidates = [stripped]
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", stripped, flags=re.IGNORECASE)
    if fenced is not None:
        candidates.append(fenced.group(1).strip())

    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return cast(dict[str, Any], value)

    for index, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return cast(dict[str, Any], value)
    raise ValueError("model did not return a valid JSON object")


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise ValueError(f"{label} keys must be strings")
    return {str(key): item for key, item in value.items()}


def _require_exact_keys(
    payload: Mapping[str, Any], expected: frozenset[str], *, label: str
) -> None:
    actual = set(payload)
    missing = expected - actual
    extra = actual - expected
    if missing or extra:
        raise ValueError(
            f"{label} fields do not match schema; "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )


def _object_list(payload: Mapping[str, Any], key: str) -> list[dict[str, Any]]:
    value = payload[key]
    if not isinstance(value, list):
        raise ValueError(f"{key} must be an array")
    return [_mapping(item, label=f"{key}[{index}]") for index, item in enumerate(value)]


def _string_list(
    payload: Mapping[str, Any], key: str, *, label: str | None = None
) -> list[str]:
    value = payload[key]
    field_label = f"{label}.{key}" if label else key
    if not isinstance(value, list):
        raise ValueError(f"{field_label} must be an array")
    result = [
        _required_text(item, label=f"{field_label}[{index}]")
        for index, item in enumerate(value)
    ]
    if len(result) != len(set(result)):
        raise ValueError(f"{field_label} must not contain duplicates")
    return result


def _source_turn_ids(payload: Mapping[str, Any], *, label: str) -> list[str]:
    values = _string_list(payload, "source_turn_ids", label=label)
    if not values:
        raise ValueError(f"{label}.source_turn_ids must not be empty")
    return values


def _required_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, *, label: str) -> str | None:
    if value is None:
        return None
    return _required_text(value, label=label)


def _enum(value: Any, allowed: frozenset[str], *, label: str) -> str:
    if not isinstance(value, str) or value not in allowed:
        raise ValueError(f"unsupported {label}: {value!r}")
    return value
