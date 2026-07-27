"""Deterministic planning gate for ordinary-Chat task complexity."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol, cast

from mochi.agents.turn_intent_contract import TurnIntentContract

ComplexityDecisionKind = Literal[
    "no_plan",
    "plan_required",
    "continue_existing_plan",
    "preserve_existing_plan",
    "blocked_for_clarification",
]
ComplexityTaskRelation = Literal[
    "continue",
    "side_question",
    "start",
    "supersede",
    "cancel",
    "standalone",
]
AdvisorOutcomeStatus = Literal["accepted", "timeout", "malformed", "skipped"]

COMPLEXITY_DECISION_VERSION = "complexity-decision-v1"
COMPLEXITY_ADVISOR_REQUEST_VERSION = "complexity-advisor-request-v1"
COMPLEXITY_ADVISOR_RESPONSE_VERSION = "complexity-advisor-response-v1"

_DECISION_KINDS = frozenset(
    {
        "no_plan",
        "plan_required",
        "continue_existing_plan",
        "preserve_existing_plan",
        "blocked_for_clarification",
    }
)
_TASK_RELATIONS = frozenset(
    {"continue", "side_question", "start", "supersede", "cancel", "standalone"}
)
_ADVISOR_STATUSES = frozenset({"accepted", "timeout", "malformed", "skipped"})
_EFFECTFUL_OPERATIONS = frozenset({"workspace_write", "execution"})
_READ_ONLY_COMPLEXITY_OPERATIONS = frozenset(
    {"conversation", "open_world_lookup", "literature_research", "workspace_read"}
)
_MAX_REASON_CODES = 16
_MAX_REASON_CODE_CHARS = 64
_MAX_OBJECTIVE_CHARS = 2_000
_MAX_OPERATION_COUNT = 16
_MAX_SOFT_SCORE = 100


def _clean_text(
    value: Any,
    *,
    field_name: str,
    allow_empty: bool = False,
    max_chars: int = _MAX_OBJECTIVE_CHARS,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    cleaned = " ".join(value.split())
    if not cleaned and not allow_empty:
        raise ValueError(f"{field_name} must not be empty")
    if len(cleaned) > max_chars:
        raise ValueError(f"{field_name} exceeds {max_chars} characters")
    return cleaned


def _clean_non_negative_int(value: Any, *, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _clean_score(value: Any) -> int:
    score = _clean_non_negative_int(value, field_name="score")
    if score > _MAX_SOFT_SCORE:
        raise ValueError("score must be between 0 and 100")
    return score


def _clean_reason_codes(value: Any, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a list")
    cleaned: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        code = _clean_text(
            item,
            field_name=f"{field_name}[{index}]",
            max_chars=_MAX_REASON_CODE_CHARS,
        )
        if code in seen:
            continue
        seen.add(code)
        cleaned.append(code)
    if len(cleaned) > _MAX_REASON_CODES:
        raise ValueError(f"{field_name} exceeds {_MAX_REASON_CODES} items")
    return tuple(cleaned)


def _require_exact_keys(payload: Mapping[str, Any], *, expected: frozenset[str], field_name: str) -> None:
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


def _clean_confidence(value: Any, *, field_name: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field_name} must be a number or null")
    cleaned = float(value)
    if cleaned < 0.0 or cleaned > 1.0:
        raise ValueError(f"{field_name} must be between 0.0 and 1.0")
    return cleaned


def _clean_bool(value: Any, *, field_name: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{field_name} must be a boolean")
    return value


def _clean_summary_tuple(
    value: Any,
    *,
    field_name: str,
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
    if len(cleaned) > max_items:
        raise ValueError(f"{field_name} exceeds {max_items} items")
    return tuple(cleaned)


def _count_required_deliverables(contract: TurnIntentContract) -> int:
    return sum(1 for deliverable in contract.deliverables if deliverable.required)


def _acceptance_criteria_count(contract: TurnIntentContract) -> int:
    return sum(len(deliverable.acceptance_criteria) for deliverable in contract.deliverables)


@dataclass(frozen=True)
class ComplexityDecision:
    decision_version: str
    turn_id: str
    kind: ComplexityDecisionKind
    score: int
    hard_reason_codes: tuple[str, ...]
    soft_reason_codes: tuple[str, ...]
    advisor_used: bool
    advisor_confidence: float | None
    effectful_action_requires_plan: bool
    dynamic_recheck_after_iterations: int

    def __post_init__(self) -> None:
        if self.decision_version != COMPLEXITY_DECISION_VERSION:
            raise ValueError(f"unsupported decision_version: {self.decision_version!r}")
        object.__setattr__(self, "turn_id", _clean_text(self.turn_id, field_name="turn_id", max_chars=128))
        if self.kind not in _DECISION_KINDS:
            raise ValueError(f"unsupported decision kind: {self.kind!r}")
        object.__setattr__(self, "score", _clean_score(self.score))
        object.__setattr__(
            self,
            "hard_reason_codes",
            _clean_reason_codes(self.hard_reason_codes, field_name="hard_reason_codes"),
        )
        object.__setattr__(
            self,
            "soft_reason_codes",
            _clean_reason_codes(self.soft_reason_codes, field_name="soft_reason_codes"),
        )
        if not isinstance(self.advisor_used, bool):
            raise TypeError("advisor_used must be a boolean")
        object.__setattr__(
            self,
            "advisor_confidence",
            _clean_confidence(self.advisor_confidence, field_name="advisor_confidence"),
        )
        object.__setattr__(
            self,
            "effectful_action_requires_plan",
            _clean_bool(
                self.effectful_action_requires_plan,
                field_name="effectful_action_requires_plan",
            ),
        )
        if self.kind == "blocked_for_clarification" and self.effectful_action_requires_plan:
            raise ValueError("clarification blockers cannot require plan execution")
        if self.kind in {"continue_existing_plan", "preserve_existing_plan"} and self.score != 0:
            raise ValueError(f"{self.kind} decisions must keep score at 0")
        if type(self.dynamic_recheck_after_iterations) is not int or self.dynamic_recheck_after_iterations < 0:
            raise ValueError("dynamic_recheck_after_iterations must be a non-negative integer")

    def to_dict(self) -> dict[str, Any]:
        return {
            "decision_version": self.decision_version,
            "turn_id": self.turn_id,
            "kind": self.kind,
            "score": self.score,
            "hard_reason_codes": list(self.hard_reason_codes),
            "soft_reason_codes": list(self.soft_reason_codes),
            "advisor_used": self.advisor_used,
            "advisor_confidence": self.advisor_confidence,
            "effectful_action_requires_plan": self.effectful_action_requires_plan,
            "dynamic_recheck_after_iterations": self.dynamic_recheck_after_iterations,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ComplexityDecision:
        expected = frozenset(
            {
                "decision_version",
                "turn_id",
                "kind",
                "score",
                "hard_reason_codes",
                "soft_reason_codes",
                "advisor_used",
                "advisor_confidence",
                "effectful_action_requires_plan",
                "dynamic_recheck_after_iterations",
            }
        )
        _require_exact_keys(payload, expected=expected, field_name="complexity decision")
        return cls(
            decision_version=_clean_text(
                payload.get("decision_version"),
                field_name="decision_version",
                max_chars=64,
            ),
            turn_id=payload.get("turn_id"),
            kind=cast(ComplexityDecisionKind, payload.get("kind")),
            score=payload.get("score"),
            hard_reason_codes=_clean_reason_codes(
                payload.get("hard_reason_codes"),
                field_name="hard_reason_codes",
            ),
            soft_reason_codes=_clean_reason_codes(
                payload.get("soft_reason_codes"),
                field_name="soft_reason_codes",
            ),
            advisor_used=payload.get("advisor_used"),
            advisor_confidence=payload.get("advisor_confidence"),
            effectful_action_requires_plan=payload.get("effectful_action_requires_plan"),
            dynamic_recheck_after_iterations=payload.get("dynamic_recheck_after_iterations"),
        )


@dataclass(frozen=True)
class ComplexityCapabilitySummary:
    requires_user_approval: bool = False
    destructive_tool_available: bool = False
    effectful_tool_count: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.requires_user_approval, bool):
            raise TypeError("requires_user_approval must be a boolean")
        if not isinstance(self.destructive_tool_available, bool):
            raise TypeError("destructive_tool_available must be a boolean")
        object.__setattr__(
            self,
            "effectful_tool_count",
            _clean_non_negative_int(self.effectful_tool_count, field_name="effectful_tool_count"),
        )


@dataclass(frozen=True)
class ComplexityActivePlanSummary:
    ledger_id: str
    status: Literal["active", "completed", "blocked", "cancelled"]
    revision: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "ledger_id", _clean_text(self.ledger_id, field_name="ledger_id", max_chars=128))
        if self.status not in {"active", "completed", "blocked", "cancelled"}:
            raise ValueError(f"unsupported active plan status: {self.status!r}")
        object.__setattr__(self, "revision", _clean_non_negative_int(self.revision, field_name="revision"))

    @property
    def is_terminal(self) -> bool:
        return self.status in {"completed", "cancelled"}


@dataclass(frozen=True)
class ComplexityGateRequest:
    turn_intent: TurnIntentContract
    task_relation: ComplexityTaskRelation = "standalone"
    capability_summary: ComplexityCapabilitySummary = field(default_factory=ComplexityCapabilitySummary)
    active_plan: ComplexityActivePlanSummary | None = None
    completed_iterations: int = 0

    def __post_init__(self) -> None:
        if self.task_relation not in _TASK_RELATIONS:
            raise ValueError(f"unsupported task relation: {self.task_relation!r}")
        object.__setattr__(
            self,
            "completed_iterations",
            _clean_non_negative_int(
                self.completed_iterations,
                field_name="completed_iterations",
            ),
        )


@dataclass(frozen=True)
class ComplexityAdvisorRequest:
    request_version: str
    turn_id: str
    task_relation: ComplexityTaskRelation
    objective: str
    speech_act: str
    operation_names: tuple[str, ...]
    deliverable_summaries: tuple[str, ...]
    constraint_summaries: tuple[str, ...]
    capability_risk_summary: tuple[str, ...]
    existing_plan_summary: str | None
    deterministic_score: int
    hard_reason_codes: tuple[str, ...]
    soft_reason_codes: tuple[str, ...]
    effectful_action_requires_plan: bool

    def __post_init__(self) -> None:
        if self.request_version != COMPLEXITY_ADVISOR_REQUEST_VERSION:
            raise ValueError(f"unsupported request_version: {self.request_version!r}")
        object.__setattr__(self, "turn_id", _clean_text(self.turn_id, field_name="turn_id", max_chars=128))
        if self.task_relation not in _TASK_RELATIONS:
            raise ValueError(f"unsupported task relation: {self.task_relation!r}")
        object.__setattr__(self, "speech_act", _clean_text(self.speech_act, field_name="speech_act", max_chars=64))
        object.__setattr__(self, "objective", _clean_text(self.objective, field_name="objective"))
        object.__setattr__(
            self,
            "operation_names",
            _clean_summary_tuple(
                self.operation_names,
                field_name="operation_names",
                max_items=_MAX_OPERATION_COUNT,
                max_chars=64,
            ),
        )
        object.__setattr__(
            self,
            "deliverable_summaries",
            _clean_summary_tuple(
                self.deliverable_summaries,
                field_name="deliverable_summaries",
            ),
        )
        object.__setattr__(
            self,
            "constraint_summaries",
            _clean_summary_tuple(
                self.constraint_summaries,
                field_name="constraint_summaries",
            ),
        )
        object.__setattr__(
            self,
            "capability_risk_summary",
            _clean_summary_tuple(
                self.capability_risk_summary,
                field_name="capability_risk_summary",
            ),
        )
        object.__setattr__(self, "deterministic_score", _clean_score(self.deterministic_score))
        object.__setattr__(
            self,
            "hard_reason_codes",
            _clean_reason_codes(self.hard_reason_codes, field_name="hard_reason_codes"),
        )
        object.__setattr__(
            self,
            "soft_reason_codes",
            _clean_reason_codes(self.soft_reason_codes, field_name="soft_reason_codes"),
        )
        object.__setattr__(
            self,
            "effectful_action_requires_plan",
            _clean_bool(
                self.effectful_action_requires_plan,
                field_name="effectful_action_requires_plan",
            ),
        )
        if self.existing_plan_summary is not None:
            object.__setattr__(
                self,
                "existing_plan_summary",
                _clean_text(
                    self.existing_plan_summary,
                    field_name="existing_plan_summary",
                    max_chars=64,
                ),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_version": self.request_version,
            "turn_id": self.turn_id,
            "task_relation": self.task_relation,
            "objective": self.objective,
            "speech_act": self.speech_act,
            "operation_names": list(self.operation_names),
            "deliverable_summaries": list(self.deliverable_summaries),
            "constraint_summaries": list(self.constraint_summaries),
            "capability_risk_summary": list(self.capability_risk_summary),
            "existing_plan_summary": self.existing_plan_summary,
            "deterministic_score": self.deterministic_score,
            "hard_reason_codes": list(self.hard_reason_codes),
            "soft_reason_codes": list(self.soft_reason_codes),
            "effectful_action_requires_plan": self.effectful_action_requires_plan,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ComplexityAdvisorRequest:
        expected = frozenset(
            {
                "request_version",
                "turn_id",
                "task_relation",
                "objective",
                "speech_act",
                "operation_names",
                "deliverable_summaries",
                "constraint_summaries",
                "capability_risk_summary",
                "existing_plan_summary",
                "deterministic_score",
                "hard_reason_codes",
                "soft_reason_codes",
                "effectful_action_requires_plan",
            }
        )
        _require_exact_keys(payload, expected=expected, field_name="complexity advisor request")
        return cls(
            request_version=_clean_text(
                payload.get("request_version"),
                field_name="request_version",
                max_chars=64,
            ),
            turn_id=payload.get("turn_id"),
            task_relation=cast(ComplexityTaskRelation, payload.get("task_relation")),
            objective=payload.get("objective"),
            speech_act=payload.get("speech_act"),
            operation_names=tuple(payload.get("operation_names", ())),
            deliverable_summaries=tuple(payload.get("deliverable_summaries", ())),
            constraint_summaries=tuple(payload.get("constraint_summaries", ())),
            capability_risk_summary=tuple(payload.get("capability_risk_summary", ())),
            existing_plan_summary=payload.get("existing_plan_summary"),
            deterministic_score=payload.get("deterministic_score"),
            hard_reason_codes=_clean_reason_codes(
                payload.get("hard_reason_codes"),
                field_name="hard_reason_codes",
            ),
            soft_reason_codes=_clean_reason_codes(
                payload.get("soft_reason_codes"),
                field_name="soft_reason_codes",
            ),
            effectful_action_requires_plan=payload.get("effectful_action_requires_plan"),
        )


@dataclass(frozen=True)
class ComplexityAdvisorResponse:
    response_version: str
    plan_recommended: bool
    estimated_distinct_actions: int
    dependency_count: int
    confidence: float
    reason_codes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.response_version != COMPLEXITY_ADVISOR_RESPONSE_VERSION:
            raise ValueError(f"unsupported response_version: {self.response_version!r}")
        object.__setattr__(
            self,
            "plan_recommended",
            _clean_bool(self.plan_recommended, field_name="plan_recommended"),
        )
        object.__setattr__(
            self,
            "estimated_distinct_actions",
            _clean_non_negative_int(
                self.estimated_distinct_actions,
                field_name="estimated_distinct_actions",
            ),
        )
        object.__setattr__(
            self,
            "dependency_count",
            _clean_non_negative_int(
                self.dependency_count,
                field_name="dependency_count",
            ),
        )
        object.__setattr__(self, "confidence", _clean_confidence(self.confidence, field_name="confidence"))
        if self.confidence is None:
            raise ValueError("confidence must not be null")
        object.__setattr__(
            self,
            "reason_codes",
            _clean_reason_codes(self.reason_codes, field_name="reason_codes"),
        )

    @property
    def recommended_kind(self) -> ComplexityDecisionKind:
        return "plan_required" if self.plan_recommended else "no_plan"

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ComplexityAdvisorResponse:
        expected = frozenset(
            {
                "response_version",
                "plan_recommended",
                "estimated_distinct_actions",
                "dependency_count",
                "confidence",
                "reason_codes",
            }
        )
        _require_exact_keys(payload, expected=expected, field_name="complexity advisor response")
        return cls(
            response_version=_clean_text(
                payload.get("response_version"),
                field_name="response_version",
                max_chars=64,
            ),
            plan_recommended=payload.get("plan_recommended"),
            estimated_distinct_actions=payload.get("estimated_distinct_actions"),
            dependency_count=payload.get("dependency_count"),
            confidence=payload.get("confidence"),
            reason_codes=_clean_reason_codes(payload.get("reason_codes"), field_name="reason_codes"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "response_version": self.response_version,
            "plan_recommended": self.plan_recommended,
            "estimated_distinct_actions": self.estimated_distinct_actions,
            "dependency_count": self.dependency_count,
            "confidence": self.confidence,
            "reason_codes": list(self.reason_codes),
        }


@dataclass(frozen=True)
class ComplexityAdvisorOutcome:
    status: AdvisorOutcomeStatus
    response: ComplexityAdvisorResponse | None = None
    reason_code: str | None = None

    def __post_init__(self) -> None:
        if self.status not in _ADVISOR_STATUSES:
            raise ValueError(f"unsupported advisor status: {self.status!r}")
        if self.status == "accepted" and self.response is None:
            raise ValueError("accepted advisor outcome requires a response")
        if self.status != "accepted" and self.response is not None:
            raise ValueError("non-accepted advisor outcome cannot include a response")
        if self.reason_code is not None:
            object.__setattr__(
                self,
                "reason_code",
                _clean_text(
                    self.reason_code,
                    field_name="reason_code",
                    max_chars=_MAX_REASON_CODE_CHARS,
                ),
            )


class ComplexityAdvisor(Protocol):
    async def advise(
        self,
        request: ComplexityAdvisorRequest,
    ) -> Mapping[str, Any] | ComplexityAdvisorResponse:
        """Return a structured grey-zone recommendation."""


@dataclass(frozen=True)
class ComplexityGateConfig:
    no_plan_max_score: int = 2
    plan_required_min_score: int = 6
    advisor_enabled: bool = True
    advisor_timeout_seconds: float = 10.0
    dynamic_recheck_after_iterations: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "no_plan_max_score",
            _clean_non_negative_int(self.no_plan_max_score, field_name="no_plan_max_score"),
        )
        object.__setattr__(
            self,
            "plan_required_min_score",
            _clean_non_negative_int(
                self.plan_required_min_score,
                field_name="plan_required_min_score",
            ),
        )
        if self.plan_required_min_score <= self.no_plan_max_score:
            raise ValueError("plan_required_min_score must be greater than no_plan_max_score")
        if not isinstance(self.advisor_enabled, bool):
            raise TypeError("advisor_enabled must be a boolean")
        if isinstance(self.advisor_timeout_seconds, bool) or self.advisor_timeout_seconds <= 0:
            raise ValueError("advisor_timeout_seconds must be positive")
        object.__setattr__(
            self,
            "dynamic_recheck_after_iterations",
            _clean_non_negative_int(
                self.dynamic_recheck_after_iterations,
                field_name="dynamic_recheck_after_iterations",
            ),
        )


class ComplexityGate:
    """Evaluate whether a turn requires durable plan state."""

    def __init__(
        self,
        *,
        advisor: ComplexityAdvisor | None = None,
        config: ComplexityGateConfig | None = None,
    ) -> None:
        self._advisor = advisor
        self._config = config or ComplexityGateConfig()

    async def evaluate(self, request: ComplexityGateRequest) -> ComplexityDecision:
        deterministic = self.evaluate_deterministic(request)
        if not self._should_consult_advisor(request, deterministic):
            return deterministic
        advisor_request = self.build_advisor_request(request, deterministic)
        advisor_outcome = await self._run_advisor(advisor_request)
        if advisor_outcome.status == "accepted" and advisor_outcome.response is not None:
            response = advisor_outcome.response
            dynamic_recheck = (
                self._config.dynamic_recheck_after_iterations
                if response.recommended_kind == "no_plan"
                and deterministic.effectful_action_requires_plan
                else 0
            )
            return replace(
                deterministic,
                kind=cast(ComplexityDecisionKind, response.recommended_kind),
                advisor_used=True,
                advisor_confidence=response.confidence,
                soft_reason_codes=tuple(
                    dict.fromkeys(deterministic.soft_reason_codes + response.reason_codes)
                ),
                dynamic_recheck_after_iterations=dynamic_recheck,
            )
        soft_reason_codes = deterministic.soft_reason_codes
        if advisor_outcome.reason_code is not None:
            soft_reason_codes = tuple(dict.fromkeys(soft_reason_codes + (advisor_outcome.reason_code,)))
        fallback_kind: ComplexityDecisionKind = (
            "plan_required"
            if deterministic.effectful_action_requires_plan
            else deterministic.kind
        )
        return replace(
            deterministic,
            kind=fallback_kind,
            advisor_used=True,
            advisor_confidence=None,
            soft_reason_codes=soft_reason_codes,
            dynamic_recheck_after_iterations=(
                0
                if fallback_kind == "plan_required"
                else deterministic.dynamic_recheck_after_iterations
            ),
        )

    def evaluate_deterministic(self, request: ComplexityGateRequest) -> ComplexityDecision:
        contract = request.turn_intent
        active_plan = request.active_plan
        if contract.clarification_needed or contract.current_speech_act == "clarification":
            return ComplexityDecision(
                decision_version=COMPLEXITY_DECISION_VERSION,
                turn_id=contract.turn_id,
                kind="blocked_for_clarification",
                score=0,
                hard_reason_codes=("clarification_needed",),
                soft_reason_codes=(),
                advisor_used=False,
                advisor_confidence=None,
                effectful_action_requires_plan=False,
                dynamic_recheck_after_iterations=0,
            )
        if request.task_relation == "side_question" and active_plan is not None and not active_plan.is_terminal:
            return ComplexityDecision(
                decision_version=COMPLEXITY_DECISION_VERSION,
                turn_id=contract.turn_id,
                kind="preserve_existing_plan",
                score=0,
                hard_reason_codes=(),
                soft_reason_codes=("side_question",),
                advisor_used=False,
                advisor_confidence=None,
                effectful_action_requires_plan=False,
                dynamic_recheck_after_iterations=0,
            )
        if request.task_relation == "cancel" or contract.current_speech_act == "cancel":
            preserve = active_plan is not None and not active_plan.is_terminal
            return ComplexityDecision(
                decision_version=COMPLEXITY_DECISION_VERSION,
                turn_id=contract.turn_id,
                kind="preserve_existing_plan" if preserve else "no_plan",
                score=0,
                hard_reason_codes=("cancel_turn",),
                soft_reason_codes=(),
                advisor_used=False,
                advisor_confidence=None,
                effectful_action_requires_plan=False,
                dynamic_recheck_after_iterations=0,
            )
        if active_plan is not None and not active_plan.is_terminal:
            return ComplexityDecision(
                decision_version=COMPLEXITY_DECISION_VERSION,
                turn_id=contract.turn_id,
                kind="continue_existing_plan",
                score=0,
                hard_reason_codes=("active_plan_present",),
                soft_reason_codes=(),
                advisor_used=False,
                advisor_confidence=None,
                effectful_action_requires_plan=False,
                dynamic_recheck_after_iterations=0,
            )

        score = 0
        soft_reason_codes: list[str] = []
        hard_reason_codes: list[str] = []
        effectful = bool(contract.operations & _EFFECTFUL_OPERATIONS)

        if request.capability_summary.destructive_tool_available:
            hard_reason_codes.append("destructive_tool_available")
        if request.capability_summary.requires_user_approval:
            hard_reason_codes.append("approval_likely")
        if effectful:
            score += 1
            soft_reason_codes.append("effectful_operation")
        required_deliverables = _count_required_deliverables(contract)
        if required_deliverables >= 2:
            score += 2
            soft_reason_codes.append("multiple_deliverables")
        elif required_deliverables == 1 and effectful:
            soft_reason_codes.append("single_effectful_deliverable")
        non_conversation_operations = tuple(
            operation for operation in contract.operations if operation != "conversation"
        )
        if len(non_conversation_operations) >= 2:
            score += 2
            soft_reason_codes.append("multiple_requested_operations")
        if {"workspace_write", "execution"} <= contract.operations:
            score += 2
            soft_reason_codes.append("write_with_execution_dependency")
        if _acceptance_criteria_count(contract) >= 2:
            score += 1
            soft_reason_codes.append("multi_criterion_acceptance")
        if any(reference.status == "unresolved" for reference in contract.resolved_references):
            score += 1
            soft_reason_codes.append("unresolved_reference")
        if effectful and _requires_verifier(contract):
            score += 1
            soft_reason_codes.append("verifier_required")
        if request.capability_summary.effectful_tool_count >= 2:
            score += 1
            soft_reason_codes.append("multiple_effectful_tools")
        if request.task_relation in {"start", "supersede"} and (
            required_deliverables > 0 or effectful
        ):
            score += 1
            soft_reason_codes.append("new_effectful_task")
        if contract.operations and contract.operations <= _READ_ONLY_COMPLEXITY_OPERATIONS and not effectful:
            soft_reason_codes.append("read_only_request")
        score = min(score, _MAX_SOFT_SCORE)

        kind: ComplexityDecisionKind
        if hard_reason_codes:
            kind = "plan_required"
        elif score >= self._config.plan_required_min_score:
            kind = "plan_required"
        else:
            kind = "no_plan"
        dynamic_recheck = (
            self._config.dynamic_recheck_after_iterations
            if kind == "no_plan" and effectful
            else 0
        )
        return ComplexityDecision(
            decision_version=COMPLEXITY_DECISION_VERSION,
            turn_id=contract.turn_id,
            kind=kind,
            score=score,
            hard_reason_codes=tuple(hard_reason_codes),
            soft_reason_codes=tuple(soft_reason_codes),
            advisor_used=False,
            advisor_confidence=None,
            effectful_action_requires_plan=effectful,
            dynamic_recheck_after_iterations=dynamic_recheck,
        )

    def build_advisor_request(
        self,
        request: ComplexityGateRequest,
        decision: ComplexityDecision,
    ) -> ComplexityAdvisorRequest:
        contract = request.turn_intent
        return ComplexityAdvisorRequest(
            request_version=COMPLEXITY_ADVISOR_REQUEST_VERSION,
            turn_id=contract.turn_id,
            task_relation=request.task_relation,
            objective=contract.objective,
            speech_act=contract.current_speech_act,
            operation_names=tuple(sorted(contract.operations)),
            deliverable_summaries=_deliverable_summaries(contract),
            constraint_summaries=_constraint_summaries(contract),
            capability_risk_summary=_capability_risk_summary(request),
            existing_plan_summary=_existing_plan_summary(request.active_plan),
            deterministic_score=decision.score,
            hard_reason_codes=decision.hard_reason_codes,
            soft_reason_codes=decision.soft_reason_codes,
            effectful_action_requires_plan=decision.effectful_action_requires_plan,
        )

    def should_recheck(
        self,
        decision: ComplexityDecision,
        *,
        completed_iterations: int,
    ) -> bool:
        if decision.kind in {
            "blocked_for_clarification",
            "continue_existing_plan",
            "preserve_existing_plan",
        }:
            return False
        if decision.dynamic_recheck_after_iterations <= 0:
            return False
        return completed_iterations >= decision.dynamic_recheck_after_iterations

    async def recheck(
        self,
        request: ComplexityGateRequest,
        *,
        prior_decision: ComplexityDecision,
        completed_iterations: int,
    ) -> ComplexityDecision | None:
        if not self.should_recheck(prior_decision, completed_iterations=completed_iterations):
            return None
        next_request = replace(request, completed_iterations=completed_iterations)
        return await self.evaluate(next_request)

    def _should_consult_advisor(
        self,
        request: ComplexityGateRequest,
        decision: ComplexityDecision,
    ) -> bool:
        if not self._config.advisor_enabled or self._advisor is None:
            return False
        if decision.hard_reason_codes:
            return False
        if decision.kind != "no_plan":
            return False
        if decision.score <= self._config.no_plan_max_score:
            return False
        if decision.score >= self._config.plan_required_min_score:
            return False
        if request.task_relation in {"cancel", "side_question"}:
            return False
        return True

    async def _run_advisor(
        self,
        request: ComplexityAdvisorRequest,
    ) -> ComplexityAdvisorOutcome:
        if self._advisor is None:
            return ComplexityAdvisorOutcome(status="skipped", reason_code="advisor_unavailable")
        try:
            raw_response = await asyncio.wait_for(
                self._advisor.advise(request),
                timeout=self._config.advisor_timeout_seconds,
            )
        except asyncio.TimeoutError:
            return ComplexityAdvisorOutcome(status="timeout", reason_code="advisor_timeout")
        except Exception:
            return ComplexityAdvisorOutcome(status="malformed", reason_code="advisor_error")
        try:
            if isinstance(raw_response, ComplexityAdvisorResponse):
                response = raw_response
            elif isinstance(raw_response, Mapping):
                response = ComplexityAdvisorResponse.from_dict(raw_response)
            else:
                raise TypeError("advisor response must be a mapping")
        except Exception:
            return ComplexityAdvisorOutcome(status="malformed", reason_code="advisor_malformed")
        return ComplexityAdvisorOutcome(status="accepted", response=response)


def _requires_verifier(contract: TurnIntentContract) -> bool:
    for deliverable in contract.deliverables:
        for criterion in deliverable.acceptance_criteria:
            if isinstance(criterion, Mapping) and criterion.get("kind") == "tool_execution":
                return True
    return False


def _deliverable_summaries(contract: TurnIntentContract) -> tuple[str, ...]:
    summaries: list[str] = []
    for deliverable in contract.deliverables:
        required_label = "required" if deliverable.required else "optional"
        target = f":{deliverable.target_hint}" if deliverable.target_hint is not None else ""
        summaries.append(
            f"{deliverable.kind}{target} [{required_label}] criteria={len(deliverable.acceptance_criteria)}"
        )
    return tuple(dict.fromkeys(summaries))


def _constraint_summaries(contract: TurnIntentContract) -> tuple[str, ...]:
    summaries = [
        f"positive_constraints={len(contract.positive_constraints)}",
        f"negative_constraints={len(contract.negative_constraints)}",
        f"resolved_references={len(contract.resolved_references)}",
        f"evidence_count={len(contract.evidence)}",
    ]
    unresolved = sum(1 for reference in contract.resolved_references if reference.status == "unresolved")
    if unresolved:
        summaries.append(f"unresolved_references={unresolved}")
    return tuple(summaries)


def _capability_risk_summary(request: ComplexityGateRequest) -> tuple[str, ...]:
    summary = [
        f"approval_likely={request.capability_summary.requires_user_approval}",
        f"destructive_tool_available={request.capability_summary.destructive_tool_available}",
        f"effectful_tool_count={request.capability_summary.effectful_tool_count}",
        f"completed_iterations={request.completed_iterations}",
    ]
    return tuple(summary)


def _existing_plan_summary(
    active_plan: ComplexityActivePlanSummary | None,
) -> str | None:
    if active_plan is None:
        return None
    return f"{active_plan.status}:revision={active_plan.revision}"
