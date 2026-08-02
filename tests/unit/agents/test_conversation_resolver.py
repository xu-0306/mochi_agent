from __future__ import annotations

import pytest

from mochi.agents.conversation_resolver import (
    BoundedConversationContext,
    ConversationResolver,
    ConversationSummary,
    ConversationTurn,
    IntentInterpretation,
)
from mochi.agents.turn_intent_contract import (
    ActiveTaskState,
    DeliverableContract,
    IntentAdvisory,
    IntentConstraint,
    IntentEvidence,
    ResolvedReference,
    TurnIntentContract,
)
from mochi.backends.base import BackendRequestError


class _Interpreter:
    def __init__(self, result: IntentInterpretation) -> None:
        self.result = result
        self.context: BoundedConversationContext | None = None

    async def interpret(
        self, context: BoundedConversationContext
    ) -> IntentInterpretation:
        self.context = context
        return self.result


class _Classifier:
    def __init__(self, advisory: IntentAdvisory | None) -> None:
        self.advisory = advisory

    async def classify(self, context, contract):  # type: ignore[no-untyped-def]
        del context, contract
        return self.advisory


def _active_task() -> ActiveTaskState:
    return ActiveTaskState(
        goal_id="goal-existing",
        objective="Build the selected general-purpose project",
        operations=frozenset({"workspace_read", "workspace_write"}),
        mutation_requirement="required",
        deliverables=(
            DeliverableContract(
                kind="workspace_artifact",
                target_hint="project directory",
                source_turn_ids=("turn-choice",),
            ),
        ),
        positive_constraints=(
            IntentConstraint(text="general-purpose", source_turn_ids=("turn-choice",)),
        ),
        source_turn_ids=("turn-choice",),
        updated_turn_id="turn-choice",
    )


@pytest.mark.asyncio
async def test_resolver_uses_bounded_history_summary_and_durable_task_state() -> None:
    interpretation = IntentInterpretation(
        current_speech_act="request_execution",
        task_relation="continue",
        operations=frozenset({"workspace_write", "execution"}),
        mutation_requirement="required",
        resolved_references=(
            ResolvedReference(
                surface="option B",
                resolved_to="the selected general-purpose project",
                source_turn_ids=("turn-choice",),
            ),
        ),
        confidence=0.93,
        evidence=(
            IntentEvidence(
                statement="The active task resolves the abbreviated choice.",
                source="active_task",
                source_turn_ids=("turn-choice",),
            ),
        ),
    )
    interpreter = _Interpreter(interpretation)
    resolver = ConversationResolver(
        interpreter=interpreter,
        max_recent_turns=2,
        max_chars_per_turn=12,
        max_summary_chars=10,
    )

    result = await resolver.resolve(
        current_turn=ConversationTurn("turn-now", "user", "Proceed with option B"),
        recent_history=[
            ConversationTurn("turn-old", "user", "old context"),
            ConversationTurn("turn-choice", "assistant", "Option B is general purpose"),
            ConversationTurn("turn-side", "user", "side question answered"),
        ],
        summary=ConversationSummary("summary that is deliberately long", ("turn-old",)),
        active_task=_active_task(),
    )

    assert interpreter.context is not None
    assert [turn.turn_id for turn in interpreter.context.recent_history] == [
        "turn-choice",
        "turn-side",
    ]
    assert all(len(turn.content) <= 12 for turn in interpreter.context.recent_history)
    assert interpreter.context.summary is not None
    assert interpreter.context.summary.content == "summary th"
    assert result.diagnostics["omitted_history_count"] == 1
    assert result.contract.operations == frozenset(
        {"workspace_read", "workspace_write", "execution"}
    )
    assert result.contract.mutation_requirement == "required"
    assert result.contract.active_goal_id == "goal-existing"
    assert result.contract.deliverables == _active_task().deliverables
    assert "turn-choice" in result.contract.source_turn_ids
    assert result.next_active_task is not None
    assert result.next_active_task.updated_turn_id == "turn-now"


@pytest.mark.asyncio
async def test_side_question_preserves_active_task_without_inheriting_its_deliverables() -> (
    None
):
    active = _active_task()
    resolver = ConversationResolver(
        interpreter=_Interpreter(
            IntentInterpretation(
                current_speech_act="side_question",
                task_relation="side_question",
                objective="Explain available tools",
                operations=frozenset({"tool_discovery"}),
                mutation_requirement="forbidden",
                confidence=0.9,
            )
        )
    )

    result = await resolver.resolve(
        current_turn=ConversationTurn("turn-side", "user", "question"),
        active_task=active,
    )

    assert result.contract.operations == frozenset({"tool_discovery"})
    assert result.contract.deliverables == ()
    assert result.contract.modifies_active_task is False
    assert result.next_active_task is active
    assert result.resolution_source == "interpreter"


@pytest.mark.asyncio
async def test_continuation_inherits_durable_mutation_requirement_when_unspecified() -> (
    None
):
    resolver = ConversationResolver(
        interpreter=_Interpreter(
            IntentInterpretation(
                current_speech_act="request_execution",
                task_relation="continue",
                confidence=0.86,
            )
        )
    )

    result = await resolver.resolve(
        current_turn=ConversationTurn("turn-next", "user", "continue"),
        active_task=_active_task(),
    )

    assert result.contract.mutation_requirement == "required"
    assert "workspace_write" in result.contract.operations
    assert result.contract.deliverables == _active_task().deliverables


@pytest.mark.asyncio
async def test_negative_mutation_constraint_is_authoritative() -> None:
    resolver = ConversationResolver(
        interpreter=_Interpreter(
            IntentInterpretation(
                current_speech_act="constraint",
                task_relation="standalone",
                objective="Compare the approaches",
                operations=frozenset({"workspace_read"}),
                negative_constraints=(
                    IntentConstraint("Do not modify files", ("turn-now",)),
                ),
                mutation_requirement="forbidden",
                confidence=0.98,
            )
        )
    )

    result = await resolver.resolve(
        current_turn=ConversationTurn("turn-now", "user", "compare only"),
    )

    assert result.contract.mutation_requirement == "forbidden"
    assert "workspace_write" not in result.contract.operations
    assert result.contract.negative_constraints[0].source_turn_ids == ("turn-now",)


@pytest.mark.asyncio
async def test_cancel_updates_durable_state_and_cancels_pending_deliverables() -> None:
    active = _active_task()
    resolver = ConversationResolver(
        interpreter=_Interpreter(
            IntentInterpretation(
                current_speech_act="cancel",
                task_relation="cancel",
                confidence=0.99,
            )
        )
    )

    result = await resolver.resolve(
        current_turn=ConversationTurn("turn-cancel", "user", "cancel it"),
        active_task=active,
    )

    assert result.contract.cancels_active_goal is True
    assert result.contract.mutation_requirement == "forbidden"
    assert result.contract.operations == frozenset()
    assert result.next_active_task is not None
    assert result.next_active_task.status == "cancelled"
    assert result.next_active_task.deliverables[0].status == "cancelled"


@pytest.mark.asyncio
async def test_supersession_replaces_inherited_operations_and_deliverables() -> None:
    deliverable = DeliverableContract(
        kind="analysis",
        source_turn_ids=("turn-new",),
        acceptance_criteria=("comparison included",),
    )
    resolver = ConversationResolver(
        interpreter=_Interpreter(
            IntentInterpretation(
                current_speech_act="task_update",
                task_relation="supersede",
                objective="Compare the alternatives",
                operations=frozenset({"workspace_read"}),
                deliverables=(deliverable,),
                mutation_requirement="forbidden",
                confidence=0.95,
            )
        )
    )

    result = await resolver.resolve(
        current_turn=ConversationTurn("turn-new", "user", "new task"),
        active_task=_active_task(),
    )

    assert result.contract.supersedes_previous_goal is True
    assert result.contract.active_goal_id == "goal:turn-new"
    assert result.contract.operations == frozenset({"workspace_read"})
    assert result.contract.deliverables == (deliverable,)
    assert result.next_active_task is not None
    assert result.next_active_task.goal_id == "goal:turn-new"


@pytest.mark.asyncio
async def test_missing_interpreter_fails_open_to_a_safe_conversation_contract() -> None:
    active = _active_task()
    result = await ConversationResolver().resolve(
        current_turn=ConversationTurn("turn-now", "user", "ambiguous"),
        active_task=active,
    )

    assert result.contract.clarification_needed is False
    assert result.contract.mutation_requirement == "forbidden"
    assert result.contract.operations == frozenset({"conversation", "tool_discovery"})
    assert result.contract.modifies_active_task is False
    assert result.next_active_task is active


@pytest.mark.asyncio
async def test_invalid_or_fabricated_interpreter_evidence_falls_back_without_mutation() -> None:
    resolver = ConversationResolver(
        interpreter=_Interpreter(
            IntentInterpretation(
                current_speech_act="request_execution",
                task_relation="standalone",
                operations=frozenset({"workspace_write"}),
                mutation_requirement="required",
                confidence=0.9,
                evidence=(
                    IntentEvidence("fabricated", "recent_history", ("missing-turn",)),
                ),
            )
        )
    )

    result = await resolver.resolve(
        current_turn=ConversationTurn("turn-now", "user", "do it"),
    )

    assert result.diagnostics["interpreter_status"] == "rejected"
    assert result.contract.operations == frozenset({"conversation", "tool_discovery"})
    assert result.contract.mutation_requirement == "forbidden"
    assert result.contract.clarification is None
    assert result.resolution_source == "fallback"


@pytest.mark.asyncio
async def test_invalid_interpreter_contract_fields_fall_back_without_mutation() -> None:
    interpretation = IntentInterpretation(
        current_speech_act="request_execution",
        task_relation="standalone",
        operations=frozenset({"unsupported_operation"}),  # type: ignore[arg-type]
        confidence=0.9,
    )
    result = await ConversationResolver(
        interpreter=_Interpreter(interpretation)
    ).resolve(
        current_turn=ConversationTurn("turn-now", "user", "request"),
    )

    assert result.diagnostics["interpreter_status"] == "rejected"
    assert result.contract.clarification_needed is False
    assert result.contract.operations == frozenset({"conversation", "tool_discovery"})


@pytest.mark.asyncio
async def test_backend_interpreter_failure_falls_open_without_mutation() -> None:
    class _UnavailableInterpreter:
        async def interpret(self, context):  # type: ignore[no-untyped-def]
            del context
            raise BackendRequestError("provider unavailable", metadata={"status_code": 503})

    result = await ConversationResolver(interpreter=_UnavailableInterpreter()).resolve(
        current_turn=ConversationTurn("turn-now", "user", "你好"),
    )

    assert result.diagnostics["interpreter_status"] == "unavailable"
    assert result.contract.operations == frozenset({"conversation", "tool_discovery"})
    assert result.contract.mutation_requirement == "forbidden"
    assert result.resolution_source == "fallback"


@pytest.mark.asyncio
async def test_empty_task_shape_normalizes_to_language_agnostic_conversation() -> None:
    result = await ConversationResolver(
        interpreter=_Interpreter(
            IntentInterpretation(
                current_speech_act="unknown",
                task_relation="start",
                confidence=0.6,
            )
        )
    ).resolve(current_turn=ConversationTurn("turn-now", "user", "hola"))

    assert result.diagnostics["interpreter_status"] == "accepted"
    assert result.contract.current_speech_act == "request_information"
    assert result.contract.operations == frozenset({"conversation"})
    assert result.contract.mutation_requirement == "forbidden"


@pytest.mark.asyncio
async def test_accepted_interpretation_sets_resolution_source_to_interpreter() -> None:
    result = await ConversationResolver(
        interpreter=_Interpreter(
            IntentInterpretation(
                current_speech_act="request_information",
                task_relation="standalone",
                operations=frozenset({"conversation"}),
                mutation_requirement="forbidden",
                confidence=0.9,
            )
        )
    ).resolve(current_turn=ConversationTurn("turn-now", "user", "question"))

    assert result.resolution_source == "interpreter"


@pytest.mark.asyncio
async def test_resolution_source_is_fallback_when_interpreter_is_absent() -> None:
    result = await ConversationResolver().resolve(
        current_turn=ConversationTurn("turn-now", "user", "question")
    )

    assert result.resolution_source == "fallback"


@pytest.mark.asyncio
async def test_resolution_source_is_fallback_when_interpreter_rejects() -> None:
    result = await ConversationResolver(
        interpreter=_Interpreter(
            IntentInterpretation(
                current_speech_act="request_execution",
                task_relation="standalone",
                operations=frozenset({"unsupported_operation"}),  # type: ignore[arg-type]
                confidence=0.9,
            )
        )
    ).resolve(current_turn=ConversationTurn("turn-now", "user", "question"))

    assert result.resolution_source == "fallback"


@pytest.mark.asyncio
async def test_resolution_source_is_fallback_when_interpreter_is_unavailable() -> None:
    class _UnavailableInterpreter:
        async def interpret(self, context):  # type: ignore[no-untyped-def]
            del context
            raise BackendRequestError("unavailable")

    result = await ConversationResolver(interpreter=_UnavailableInterpreter()).resolve(
        current_turn=ConversationTurn("turn-now", "user", "question")
    )

    assert result.resolution_source == "fallback"


@pytest.mark.asyncio
async def test_classifier_is_advisory_and_cannot_override_contract() -> None:
    interpreter = _Interpreter(
        IntentInterpretation(
            current_speech_act="constraint",
            task_relation="standalone",
            objective="Analysis only",
            operations=frozenset({"workspace_read"}),
            mutation_requirement="forbidden",
            confidence=0.97,
        )
    )
    classifier = _Classifier(
        IntentAdvisory(
            label="workspace_write",
            confidence=0.99,
            rationale="Latest-message classifier disagrees.",
            recommended_operations=frozenset({"workspace_write"}),
            source_turn_ids=("turn-now",),
        )
    )
    resolver = ConversationResolver(
        interpreter=interpreter, advisory_classifier=classifier
    )

    result = await resolver.resolve(
        current_turn=ConversationTurn("turn-now", "user", "analyze only"),
    )

    assert result.contract.operations == frozenset({"workspace_read"})
    assert result.contract.mutation_requirement == "forbidden"
    assert result.contract.advisories[0].recommended_operations == frozenset(
        {"workspace_write"}
    )


@pytest.mark.asyncio
async def test_classifier_failure_does_not_remove_required_deliverable() -> None:
    class _FailingClassifier:
        async def classify(self, context, contract):  # type: ignore[no-untyped-def]
            del context, contract
            raise RuntimeError("classifier unavailable")

    deliverable = DeliverableContract(
        "workspace_artifact", ("turn-now",), target_hint="README.md"
    )
    resolver = ConversationResolver(
        interpreter=_Interpreter(
            IntentInterpretation(
                current_speech_act="request_execution",
                task_relation="standalone",
                operations=frozenset({"workspace_write"}),
                deliverables=(deliverable,),
                mutation_requirement="required",
                confidence=0.96,
            )
        ),
        advisory_classifier=_FailingClassifier(),
    )

    result = await resolver.resolve(
        current_turn=ConversationTurn("turn-now", "user", "save it"),
    )

    assert result.contract.deliverables == (deliverable,)
    assert result.contract.operations == frozenset({"workspace_write"})
    assert result.diagnostics["advisory_status"] == "rejected"


def test_turn_contract_rejects_conflicting_mutation_semantics() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        TurnIntentContract(
            turn_id="turn-now",
            active_goal_id=None,
            objective="Invalid contract",
            current_speech_act="request_execution",
            operations=frozenset({"workspace_write"}),
            deliverables=(),
            resolved_references=(),
            positive_constraints=(),
            negative_constraints=(),
            mutation_requirement="forbidden",
            clarification=None,
            supersedes_previous_goal=False,
            cancels_active_goal=False,
            modifies_active_task=False,
            confidence=1.0,
            evidence=(),
        )


def test_turn_contract_rejects_current_turn_satisfied_write_deliverable() -> None:
    with pytest.raises(ValueError, match="cannot already be satisfied"):
        TurnIntentContract(
            turn_id="turn-now",
            active_goal_id="goal-now",
            objective="Create report.md",
            current_speech_act="request_execution",
            operations=frozenset({"workspace_write"}),
            deliverables=(
                DeliverableContract(
                    kind="workspace_artifact",
                    target_hint="report.md",
                    status="satisfied",
                    source_turn_ids=("turn-now",),
                ),
            ),
            resolved_references=(),
            positive_constraints=(),
            negative_constraints=(),
            mutation_requirement="required",
            clarification=None,
            supersedes_previous_goal=False,
            cancels_active_goal=False,
            modifies_active_task=True,
            confidence=1.0,
            evidence=(),
        )


def test_turn_contract_preserves_previously_satisfied_durable_deliverable() -> None:
    contract = TurnIntentContract(
        turn_id="turn-next",
        active_goal_id="goal-existing",
        objective="Continue the existing task",
        current_speech_act="request_execution",
        operations=frozenset({"workspace_write"}),
        deliverables=(
            DeliverableContract(
                kind="workspace_artifact",
                target_hint="report.md",
                status="satisfied",
                source_turn_ids=("turn-prior",),
            ),
        ),
        resolved_references=(),
        positive_constraints=(),
        negative_constraints=(),
        mutation_requirement="required",
        clarification=None,
        supersedes_previous_goal=False,
        cancels_active_goal=False,
        modifies_active_task=True,
        confidence=1.0,
        evidence=(),
    )

    assert contract.deliverables[0].status == "satisfied"
