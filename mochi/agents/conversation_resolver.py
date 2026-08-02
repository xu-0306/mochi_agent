"""Bounded, interpreter-driven conversation resolution.

This module performs deterministic context bounding, validation and task-state
merging.  Language understanding is injected through ``ConversationInterpreter``;
the resolver intentionally contains no language-specific keyword router.
"""

from __future__ import annotations

from collections.abc import Callable, Hashable
from dataclasses import dataclass, replace
from typing import Any, Literal, Protocol, TypeVar

from mochi.agents.turn_intent_contract import (
    ActiveTaskState,
    ClarificationRequest,
    DeliverableContract,
    IntentAdvisory,
    IntentConstraint,
    IntentEvidence,
    MutationRequirement,
    ResolvedReference,
    SpeechAct,
    TurnIntentContract,
    TurnOperation,
)
from mochi.backends.base import BackendRequestError

TurnRole = Literal["user", "assistant", "system", "tool"]
TaskRelation = Literal[
    "continue", "side_question", "start", "supersede", "cancel", "standalone"
]
ResolutionSource = Literal["interpreter", "fallback"]
_MergedItem = TypeVar("_MergedItem")
_TURN_ROLES = frozenset({"user", "assistant", "system", "tool"})
_TASK_RELATIONS = frozenset(
    {"continue", "side_question", "start", "supersede", "cancel", "standalone"}
)


@dataclass(frozen=True)
class ConversationTurn:
    turn_id: str
    role: TurnRole
    content: str


@dataclass(frozen=True)
class ConversationSummary:
    content: str
    source_turn_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class BoundedConversationContext:
    current_turn: ConversationTurn
    recent_history: tuple[ConversationTurn, ...]
    summary: ConversationSummary | None
    active_task: ActiveTaskState | None
    omitted_history_count: int
    truncated_fields: tuple[str, ...]

    @property
    def available_source_turn_ids(self) -> frozenset[str]:
        values = {self.current_turn.turn_id}
        values.update(item.turn_id for item in self.recent_history)
        if self.summary is not None:
            values.update(self.summary.source_turn_ids)
        if self.active_task is not None:
            values.update(self.active_task.source_turn_ids)
            if self.active_task.updated_turn_id:
                values.add(self.active_task.updated_turn_id)
        return frozenset(values)


@dataclass(frozen=True)
class IntentInterpretation:
    """Semantic proposal returned by an injected interpreter."""

    current_speech_act: SpeechAct
    task_relation: TaskRelation
    objective: str | None = None
    operations: frozenset[TurnOperation] = frozenset()
    deliverables: tuple[DeliverableContract, ...] = ()
    resolved_references: tuple[ResolvedReference, ...] = ()
    positive_constraints: tuple[IntentConstraint, ...] = ()
    negative_constraints: tuple[IntentConstraint, ...] = ()
    mutation_requirement: MutationRequirement = "unknown"
    clarification: ClarificationRequest | None = None
    confidence: float = 0.0
    evidence: tuple[IntentEvidence, ...] = ()


class ConversationInterpreter(Protocol):
    async def interpret(
        self, context: BoundedConversationContext
    ) -> IntentInterpretation:
        """Interpret language into a semantic proposal without granting permissions."""
        ...


class AdvisoryIntentClassifier(Protocol):
    async def classify(
        self,
        context: BoundedConversationContext,
        contract: TurnIntentContract,
    ) -> IntentAdvisory | None:
        """Return telemetry or routing advice; the result cannot mutate the contract."""
        ...


@dataclass(frozen=True)
class ConversationResolution:
    contract: TurnIntentContract
    next_active_task: ActiveTaskState | None
    context: BoundedConversationContext
    diagnostics: dict[str, Any]
    resolution_source: ResolutionSource


class ConversationResolver:
    """Resolve one turn using bounded context and deterministic state semantics."""

    def __init__(
        self,
        *,
        interpreter: ConversationInterpreter | None = None,
        advisory_classifier: AdvisoryIntentClassifier | None = None,
        max_recent_turns: int = 8,
        max_chars_per_turn: int = 4_000,
        max_summary_chars: int = 8_000,
    ) -> None:
        if max_recent_turns < 0:
            raise ValueError("max_recent_turns must be non-negative")
        if max_chars_per_turn < 1 or max_summary_chars < 1:
            raise ValueError("context character limits must be positive")
        self._interpreter = interpreter
        self._advisory_classifier = advisory_classifier
        self._max_recent_turns = max_recent_turns
        self._max_chars_per_turn = max_chars_per_turn
        self._max_summary_chars = max_summary_chars

    async def resolve(
        self,
        *,
        current_turn: ConversationTurn,
        recent_history: list[ConversationTurn] | tuple[ConversationTurn, ...] = (),
        summary: ConversationSummary | None = None,
        active_task: ActiveTaskState | None = None,
    ) -> ConversationResolution:
        context = self._bound_context(
            current_turn=current_turn,
            recent_history=recent_history,
            summary=summary,
            active_task=active_task,
        )
        diagnostics: dict[str, Any] = {
            "resolver_version": "conversation-resolver-v1",
            "recent_history_count": len(context.recent_history),
            "omitted_history_count": context.omitted_history_count,
            "truncated_fields": list(context.truncated_fields),
            "interpreter_status": "unavailable",
            "advisory_status": "unavailable",
        }

        contract: TurnIntentContract | None = None
        next_active_task: ActiveTaskState | None = None
        resolution_source: ResolutionSource = "fallback"
        if self._interpreter is not None:
            try:
                candidate = await self._interpreter.interpret(context)
                candidate = self._normalize_conversation_only_interpretation(
                    candidate
                )
                interpretation = self._validate_interpretation(
                    candidate, context=context
                )
                contract, next_active_task = self._merge_interpretation(
                    context=context,
                    interpretation=interpretation,
                )
                diagnostics["interpreter_status"] = "accepted"
                resolution_source = "interpreter"
            except BackendRequestError as exc:
                # Intent interpretation is an optimization for capability
                # selection, never a prerequisite for an ordinary response.
                # Fall back to a capability-denying conversation contract so
                # the main model can answer while mutation remains blocked.
                diagnostics["interpreter_status"] = "unavailable"
                diagnostics["interpreter_error"] = f"{type(exc).__name__}: {exc}"
            except Exception as exc:
                diagnostics["interpreter_status"] = "rejected"
                diagnostics["interpreter_error"] = f"{type(exc).__name__}: {exc}"

        if contract is None:
            contract, next_active_task = self._fallback_resolution(context)

        if self._advisory_classifier is not None:
            semantic_before = contract.semantic_projection()
            try:
                advisory_candidate: Any = await self._advisory_classifier.classify(
                    context, contract
                )
                if advisory_candidate is not None:
                    if not isinstance(advisory_candidate, IntentAdvisory):
                        raise TypeError(
                            "advisory classifier must return IntentAdvisory"
                        )
                    advisory = advisory_candidate
                    self._validate_source_ids(advisory.source_turn_ids, context=context)
                    contract = contract.with_advisories(
                        (*contract.advisories, advisory)
                    )
                    diagnostics["advisory_status"] = "attached"
                else:
                    diagnostics["advisory_status"] = "empty"
            except Exception as exc:
                diagnostics["advisory_status"] = "rejected"
                diagnostics["advisory_error"] = f"{type(exc).__name__}: {exc}"
            if contract.semantic_projection() != semantic_before:
                raise RuntimeError(
                    "advisory classifier changed authoritative contract fields"
                )

        return ConversationResolution(
            contract=contract,
            next_active_task=next_active_task,
            context=context,
            diagnostics=diagnostics,
            resolution_source=resolution_source,
        )

    def _bound_context(
        self,
        *,
        current_turn: ConversationTurn,
        recent_history: list[ConversationTurn] | tuple[ConversationTurn, ...],
        summary: ConversationSummary | None,
        active_task: ActiveTaskState | None,
    ) -> BoundedConversationContext:
        current = self._normalize_turn(current_turn, label="current_turn")
        if current.role != "user":
            raise ValueError("current_turn must have role='user'")

        normalized_history = [
            self._normalize_turn(item, label="recent_history")
            for item in recent_history
        ]
        history_turn_ids = [item.turn_id for item in normalized_history]
        if len(history_turn_ids) != len(set(history_turn_ids)):
            raise ValueError("recent_history turn ids must be unique")
        if any(item.turn_id == current.turn_id for item in normalized_history):
            raise ValueError("current turn must not also appear in recent_history")
        selected = (
            normalized_history[-self._max_recent_turns :]
            if self._max_recent_turns
            else []
        )
        truncated_fields: list[str] = []
        if current.content != current_turn.content.strip():
            truncated_fields.append("current_turn.content")
        for index, (original, bounded) in enumerate(
            zip(
                recent_history[-len(selected) :] if selected else (),
                selected,
                strict=True,
            )
        ):
            if original.content.strip() != bounded.content:
                truncated_fields.append(f"recent_history[{index}].content")

        bounded_summary = None
        if summary is not None and summary.content.strip():
            content = summary.content.strip()
            if len(content) > self._max_summary_chars:
                content = content[: self._max_summary_chars]
                truncated_fields.append("summary.content")
            bounded_summary = ConversationSummary(
                content=content,
                source_turn_ids=tuple(
                    dict.fromkeys(
                        item.strip() for item in summary.source_turn_ids if item.strip()
                    )
                ),
            )

        return BoundedConversationContext(
            current_turn=current,
            recent_history=tuple(selected),
            summary=bounded_summary,
            active_task=active_task,
            omitted_history_count=max(0, len(normalized_history) - len(selected)),
            truncated_fields=tuple(truncated_fields),
        )

    def _normalize_turn(
        self, turn: ConversationTurn, *, label: str
    ) -> ConversationTurn:
        turn_id = str(turn.turn_id or "").strip()
        content = str(turn.content or "").strip()
        if not turn_id:
            raise ValueError(f"{label}.turn_id must not be empty")
        if not content:
            raise ValueError(f"{label}.content must not be empty")
        if turn.role not in _TURN_ROLES:
            raise ValueError(f"{label}.role is unsupported: {turn.role!r}")
        return ConversationTurn(
            turn_id=turn_id,
            role=turn.role,
            content=content[: self._max_chars_per_turn],
        )

    def _validate_interpretation(
        self,
        interpretation: object,
        *,
        context: BoundedConversationContext,
    ) -> IntentInterpretation:
        if not isinstance(interpretation, IntentInterpretation):
            raise TypeError("interpreter must return IntentInterpretation")
        if interpretation.task_relation not in _TASK_RELATIONS:
            raise ValueError(
                f"unsupported task relation: {interpretation.task_relation!r}"
            )
        if not 0.0 <= float(interpretation.confidence) <= 1.0:
            raise ValueError("interpretation confidence must be between 0.0 and 1.0")
        if (
            interpretation.mutation_requirement == "forbidden"
            and "workspace_write" in interpretation.operations
        ):
            raise ValueError("forbidden mutation conflicts with workspace_write")
        if (
            interpretation.task_relation in {"continue", "side_question", "cancel"}
            and context.active_task is None
        ):
            raise ValueError(
                f"task relation {interpretation.task_relation!r} requires an active task"
            )
        if interpretation.task_relation == "start" and context.active_task is not None:
            raise ValueError(
                "start cannot replace an existing active task; use supersede"
            )
        if interpretation.task_relation == "supersede" and context.active_task is None:
            raise ValueError("supersede requires an active task; use start")
        if (
            interpretation.task_relation in {"start", "supersede"}
            and not str(interpretation.objective or "").strip()
        ):
            raise ValueError(
                f"task relation {interpretation.task_relation!r} requires an objective"
            )
        if (
            interpretation.task_relation == "cancel"
            and interpretation.current_speech_act != "cancel"
        ):
            raise ValueError("cancel relation requires cancel speech act")

        for deliverable in interpretation.deliverables:
            self._validate_source_ids(deliverable.source_turn_ids, context=context)
        for reference in interpretation.resolved_references:
            self._validate_source_ids(reference.source_turn_ids, context=context)
        for constraint in (
            *interpretation.positive_constraints,
            *interpretation.negative_constraints,
        ):
            self._validate_source_ids(constraint.source_turn_ids, context=context)
        for evidence in interpretation.evidence:
            self._validate_source_ids(evidence.source_turn_ids, context=context)
        if interpretation.clarification is not None:
            self._validate_source_ids(
                interpretation.clarification.source_turn_ids, context=context
            )
        return interpretation

    @staticmethod
    def _normalize_conversation_only_interpretation(
        interpretation: object,
    ) -> object:
        """Recover a structurally empty task proposal as ordinary conversation.

        Some backends emit a task relation even for a turn that carries no
        objective, deliverable, operation, or mutation obligation.  Treating
        that malformed shape as a task would turn a harmless conversation into
        a validation failure.  This is deliberately structural rather than a
        phrase or language based classifier.
        """
        if not isinstance(interpretation, IntentInterpretation):
            return interpretation
        if (
            interpretation.task_relation in {"start", "supersede"}
            and not str(interpretation.objective or "").strip()
            and not interpretation.operations
            and not interpretation.deliverables
            and interpretation.mutation_requirement == "unknown"
            and interpretation.clarification is None
        ):
            return replace(
                interpretation,
                current_speech_act="request_information",
                task_relation="standalone",
                operations=frozenset({"conversation"}),
                mutation_requirement="forbidden",
            )
        return interpretation

    @staticmethod
    def _validate_source_ids(
        source_turn_ids: tuple[str, ...],
        *,
        context: BoundedConversationContext,
    ) -> None:
        unknown = set(source_turn_ids) - context.available_source_turn_ids
        if unknown:
            raise ValueError(
                f"semantic output cited unavailable source turns: {sorted(unknown)}"
            )

    def _merge_interpretation(
        self,
        *,
        context: BoundedConversationContext,
        interpretation: IntentInterpretation,
    ) -> tuple[TurnIntentContract, ActiveTaskState | None]:
        active = context.active_task
        relation = interpretation.task_relation
        inherit_task = relation == "continue" and active is not None

        operations: set[TurnOperation] = set(
            active.operations if inherit_task and active else ()
        )
        operations.update(interpretation.operations)
        if interpretation.mutation_requirement == "required":
            operations.add("workspace_write")
        if interpretation.mutation_requirement == "forbidden":
            operations.discard("workspace_write")

        deliverables = self._merge_items(
            active.deliverables if inherit_task and active else (),
            interpretation.deliverables,
            key=lambda item: (item.kind, item.target_hint),
        )
        positive_constraints = self._merge_items(
            active.positive_constraints if inherit_task and active else (),
            interpretation.positive_constraints,
            key=lambda item: item.text,
        )
        negative_constraints = self._merge_items(
            active.negative_constraints if inherit_task and active else (),
            interpretation.negative_constraints,
            key=lambda item: item.text,
        )

        if relation == "cancel":
            mutation_requirement: MutationRequirement = "forbidden"
            operations = set()
            deliverables = tuple(
                DeliverableContract(
                    kind=item.kind,
                    target_hint=item.target_hint,
                    required=item.required,
                    acceptance_criteria=item.acceptance_criteria,
                    status=(
                        "cancelled"
                        if item.status in {"pending", "in_progress"}
                        else item.status
                    ),
                    source_turn_ids=item.source_turn_ids,
                )
                for item in (active.deliverables if active else ())
            )
        else:
            mutation_requirement = (
                active.mutation_requirement
                if inherit_task
                and active is not None
                and interpretation.mutation_requirement == "unknown"
                else interpretation.mutation_requirement
            )

        objective = str(interpretation.objective or "").strip()
        if (
            not objective
            and active is not None
            and relation in {"continue", "side_question", "cancel"}
        ):
            objective = active.objective

        active_goal_id = (
            active.goal_id
            if active is not None
            and relation in {"continue", "side_question", "cancel"}
            else None
        )
        if relation in {"start", "supersede"}:
            active_goal_id = f"goal:{context.current_turn.turn_id}"

        evidence = list(interpretation.evidence)
        if inherit_task and active is not None and active.source_turn_ids:
            evidence.append(
                IntentEvidence(
                    statement="Inherited unresolved active-task state.",
                    source="active_task",
                    source_turn_ids=active.source_turn_ids,
                )
            )
        if not evidence:
            evidence.append(
                IntentEvidence(
                    statement="Semantic interpretation of the current turn.",
                    source="current_turn",
                    source_turn_ids=(context.current_turn.turn_id,),
                )
            )

        contract = TurnIntentContract(
            turn_id=context.current_turn.turn_id,
            active_goal_id=active_goal_id,
            objective=objective,
            current_speech_act=interpretation.current_speech_act,
            operations=frozenset(operations),
            deliverables=deliverables,
            resolved_references=interpretation.resolved_references,
            positive_constraints=positive_constraints,
            negative_constraints=negative_constraints,
            mutation_requirement=mutation_requirement,
            clarification=interpretation.clarification,
            supersedes_previous_goal=relation == "supersede",
            cancels_active_goal=relation == "cancel",
            modifies_active_task=relation
            in {"continue", "start", "supersede", "cancel"},
            confidence=float(interpretation.confidence),
            evidence=tuple(evidence),
        )
        return contract, self._next_task_state(
            context=context,
            contract=contract,
            relation=relation,
        )

    def _next_task_state(
        self,
        *,
        context: BoundedConversationContext,
        contract: TurnIntentContract,
        relation: TaskRelation,
    ) -> ActiveTaskState | None:
        active = context.active_task
        if relation in {"side_question", "standalone"}:
            return active
        if relation == "cancel":
            assert active is not None
            return ActiveTaskState(
                goal_id=active.goal_id,
                objective=active.objective,
                status="cancelled",
                operations=frozenset(),
                mutation_requirement="forbidden",
                deliverables=contract.deliverables,
                positive_constraints=active.positive_constraints,
                negative_constraints=active.negative_constraints,
                decisions=active.decisions,
                source_turn_ids=tuple(
                    dict.fromkeys((*active.source_turn_ids, contract.turn_id))
                ),
                updated_turn_id=contract.turn_id,
            )
        if relation in {"continue", "start", "supersede"}:
            source_ids = tuple(
                dict.fromkeys(
                    (
                        *(
                            active.source_turn_ids
                            if relation == "continue" and active
                            else ()
                        ),
                        *contract.source_turn_ids,
                    )
                )
            )
            return ActiveTaskState(
                goal_id=contract.active_goal_id or f"goal:{contract.turn_id}",
                objective=contract.objective,
                status="active",
                operations=contract.operations,
                mutation_requirement=contract.mutation_requirement,
                deliverables=contract.deliverables,
                positive_constraints=contract.positive_constraints,
                negative_constraints=contract.negative_constraints,
                decisions=active.decisions if relation == "continue" and active else (),
                source_turn_ids=source_ids,
                updated_turn_id=contract.turn_id,
            )
        return active

    def _fallback_resolution(
        self,
        context: BoundedConversationContext,
    ) -> tuple[TurnIntentContract, ActiveTaskState | None]:
        active = context.active_task
        contract = TurnIntentContract(
            turn_id=context.current_turn.turn_id,
            active_goal_id=active.goal_id if active else None,
            objective=active.objective if active else "",
            current_speech_act="request_information",
            # The baseline permits an answer and controlled capability
            # discovery.  It intentionally grants neither mutation nor
            # execution; those still require a validated semantic contract.
            operations=frozenset({"conversation", "tool_discovery"}),
            deliverables=(),
            resolved_references=(),
            positive_constraints=(),
            negative_constraints=(),
            mutation_requirement="forbidden",
            clarification=None,
            supersedes_previous_goal=False,
            cancels_active_goal=False,
            modifies_active_task=False,
            confidence=0.0,
            evidence=(
                IntentEvidence(
                    statement="No validated semantic interpretation was available.",
                    source="current_turn",
                    source_turn_ids=(context.current_turn.turn_id,),
                ),
            ),
        )
        return contract, active

    @staticmethod
    def _merge_items(
        base: tuple[_MergedItem, ...],
        updates: tuple[_MergedItem, ...],
        *,
        key: Callable[[_MergedItem], Hashable],
    ) -> tuple[_MergedItem, ...]:
        merged: dict[Hashable, _MergedItem] = {key(item): item for item in base}
        for item in updates:
            merged[key(item)] = item
        return tuple(merged.values())
