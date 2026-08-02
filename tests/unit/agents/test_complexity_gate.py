from __future__ import annotations

import asyncio

import pytest

from mochi.agents.complexity_gate import (
    COMPLEXITY_ADVISOR_REQUEST_VERSION,
    COMPLEXITY_ADVISOR_RESPONSE_VERSION,
    COMPLEXITY_DECISION_VERSION,
    ComplexityActivePlanSummary,
    ComplexityAdvisorRequest,
    ComplexityAdvisorResponse,
    ComplexityCapabilitySummary,
    ComplexityDecision,
    ComplexityGate,
    ComplexityGateConfig,
    ComplexityGateRequest,
)
from mochi.agents.turn_intent_contract import (
    DeliverableContract,
    IntentEvidence,
    ResolvedReference,
    TurnIntentContract,
)


def _deliverable(
    *,
    kind: str = "workspace_artifact",
    target_hint: str = "output/report.md",
    required: bool = True,
    acceptance_criteria: tuple[object, ...] = ("artifact exists",),
    source_turn_ids: tuple[str, ...] = ("turn-source",),
) -> DeliverableContract:
    return DeliverableContract(
        kind=kind,
        target_hint=target_hint,
        required=required,
        acceptance_criteria=acceptance_criteria,
        source_turn_ids=source_turn_ids,
    )


def _tool_execution_check() -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "tool_execution",
        "check": "test",
        "tool_name": "exec_command",
        "profile_id": "pytest",
        "expected_exit_code": 0,
    }


def _turn_contract(
    *,
    turn_id: str = "turn-1",
    objective: str = "Update the workspace artifact",
    speech_act: str = "request_execution",
    operations: frozenset[str] | None = None,
    deliverables: tuple[DeliverableContract, ...] = (),
    resolved_references: tuple[ResolvedReference, ...] = (),
) -> TurnIntentContract:
    normalized_operations = operations or frozenset({"workspace_write"})
    mutation_requirement = (
        "required" if "workspace_write" in normalized_operations else "forbidden"
    )
    return TurnIntentContract(
        turn_id=turn_id,
        active_goal_id="goal-1",
        objective=objective,
        current_speech_act=speech_act,  # type: ignore[arg-type]
        operations=normalized_operations,  # type: ignore[arg-type]
        deliverables=deliverables,
        resolved_references=resolved_references,
        positive_constraints=(),
        negative_constraints=(),
        mutation_requirement=mutation_requirement,  # type: ignore[arg-type]
        clarification=None,
        supersedes_previous_goal=False,
        cancels_active_goal=(speech_act == "cancel"),
        modifies_active_task=bool(deliverables or "workspace_write" in normalized_operations),
        confidence=0.93,
        evidence=(
            IntentEvidence(
                statement="Validated semantic contract.",
                source="current_turn",
                source_turn_ids=(turn_id,),
            ),
        ),
        advisories=(),
    )


class _Advisor:
    def __init__(
        self,
        *,
        raw_response: object,
        delay_seconds: float = 0.0,
        error: Exception | None = None,
    ) -> None:
        self.raw_response = raw_response
        self.delay_seconds = delay_seconds
        self.error = error
        self.requests: list[ComplexityAdvisorRequest] = []

    async def advise(self, request: ComplexityAdvisorRequest) -> object:
        self.requests.append(request)
        if self.delay_seconds:
            await asyncio.sleep(self.delay_seconds)
        if self.error is not None:
            raise self.error
        return self.raw_response


def test_complexity_decision_round_trip_and_strict_future_version_rejection() -> None:
    decision = ComplexityDecision(
        decision_version=COMPLEXITY_DECISION_VERSION,
        turn_id="turn-1",
        kind="no_plan",
        score=2,
        hard_reason_codes=(),
        soft_reason_codes=("effectful_operation",),
        advisor_used=False,
        advisor_confidence=None,
        effectful_action_requires_plan=True,
        dynamic_recheck_after_iterations=1,
    )

    assert ComplexityDecision.from_dict(decision.to_dict()) == decision

    invalid_version = decision.to_dict()
    invalid_version["decision_version"] = "complexity-decision-v2"
    with pytest.raises(ValueError, match="unsupported"):
        ComplexityDecision.from_dict(invalid_version)

    invalid_bool = decision.to_dict()
    invalid_bool["effectful_action_requires_plan"] = "true"
    with pytest.raises(TypeError, match="effectful_action_requires_plan"):
        ComplexityDecision.from_dict(invalid_bool)


def test_advisor_request_and_response_round_trip_strictly() -> None:
    request = ComplexityAdvisorRequest(
        request_version=COMPLEXITY_ADVISOR_REQUEST_VERSION,
        turn_id="turn-2",
        task_relation="start",
        objective="Plan a bounded mutation",
        speech_act="request_execution",
        operation_names=("workspace_write", "execution"),
        deliverable_summaries=("artifact [required] criteria=2",),
        constraint_summaries=("positive_constraints=1",),
        capability_risk_summary=("approval_likely=False",),
        existing_plan_summary=None,
        deterministic_score=4,
        hard_reason_codes=(),
        soft_reason_codes=("write_with_execution_dependency",),
        effectful_action_requires_plan=True,
    )
    response = ComplexityAdvisorResponse(
        response_version=COMPLEXITY_ADVISOR_RESPONSE_VERSION,
        plan_recommended=True,
        estimated_distinct_actions=3,
        dependency_count=1,
        confidence=0.78,
        reason_codes=("cross_tool_dependency",),
    )

    assert ComplexityAdvisorRequest.from_dict(request.to_dict()) == request
    assert ComplexityAdvisorResponse.from_dict(response.to_dict()) == response


def test_temporal_lookup_is_read_only_and_planless() -> None:
    decision = ComplexityGate().evaluate_deterministic(
        ComplexityGateRequest(
            turn_intent=_turn_contract(
                turn_id="turn-temporal",
                objective="Report the current time",
                speech_act="request_information",
                operations=frozenset({"temporal_lookup"}),
            ),
        )
    )

    assert decision.kind == "no_plan"
    assert decision.effectful_action_requires_plan is False
    assert "read_only_request" in decision.soft_reason_codes
    assert "effectful_operation" not in decision.soft_reason_codes


@pytest.mark.asyncio
async def test_simple_information_request_stays_planless_without_advisor_call() -> None:
    advisor = _Advisor(
        raw_response={
            "response_version": COMPLEXITY_ADVISOR_RESPONSE_VERSION,
            "plan_recommended": True,
            "estimated_distinct_actions": 3,
            "dependency_count": 1,
            "confidence": 0.7,
            "reason_codes": ["unused"],
        }
    )
    gate = ComplexityGate(advisor=advisor)
    request = ComplexityGateRequest(
        turn_intent=_turn_contract(
            turn_id="turn-info",
            objective="Explain the current architecture",
            speech_act="request_information",
            operations=frozenset({"conversation", "open_world_lookup"}),
            deliverables=(),
        ),
        task_relation="standalone",
    )

    decision = await gate.evaluate(request)

    assert decision.kind == "no_plan"
    assert decision.advisor_used is False
    assert advisor.requests == []


@pytest.mark.asyncio
async def test_single_safe_edit_remains_planless_and_recheckable() -> None:
    gate = ComplexityGate()
    request = ComplexityGateRequest(
        turn_intent=_turn_contract(
            turn_id="turn-edit",
            deliverables=(_deliverable(),),
        ),
        task_relation="start",
    )

    decision = await gate.evaluate(request)

    assert decision.kind == "no_plan"
    assert decision.effectful_action_requires_plan is True
    assert decision.dynamic_recheck_after_iterations == 1
    assert gate.should_recheck(decision, completed_iterations=0) is False
    assert gate.should_recheck(decision, completed_iterations=1) is True
    rechecked = await gate.recheck(
        request,
        prior_decision=decision,
        completed_iterations=1,
    )
    assert rechecked is not None
    assert rechecked.kind == "plan_required"
    assert rechecked.advisor_used is False
    assert "dynamic_iteration_threshold" in rechecked.soft_reason_codes


@pytest.mark.asyncio
async def test_dynamic_recheck_is_deterministic_and_never_reinvokes_grey_zone_advisor() -> None:
    advisor = _Advisor(
        raw_response={
            "response_version": COMPLEXITY_ADVISOR_RESPONSE_VERSION,
            "plan_recommended": False,
            "estimated_distinct_actions": 1,
            "dependency_count": 0,
            "confidence": 0.8,
            "reason_codes": ["single_safe_edit"],
        }
    )
    gate = ComplexityGate(advisor=advisor)
    request = ComplexityGateRequest(
        turn_intent=_turn_contract(
            turn_id="turn-dynamic-advisor",
            deliverables=(_deliverable(),),
        ),
        task_relation="start",
        capability_summary=ComplexityCapabilitySummary(effectful_tool_count=2),
    )

    initial = await gate.evaluate(request)
    assert initial.kind == "no_plan"
    assert len(advisor.requests) == 1

    rechecked = await gate.recheck(
        request,
        prior_decision=initial,
        completed_iterations=1,
        signals=("third_distinct_tool", "read_to_effectful"),
    )

    assert rechecked is not None
    assert rechecked.kind == "plan_required"
    assert rechecked.advisor_used is False
    assert len(advisor.requests) == 1
    assert "dynamic_third_distinct_tool" in rechecked.soft_reason_codes
    assert "dynamic_read_to_effectful" in rechecked.soft_reason_codes


@pytest.mark.asyncio
async def test_multiple_deliverables_and_execution_dependency_require_plan() -> None:
    gate = ComplexityGate()
    request = ComplexityGateRequest(
        turn_intent=_turn_contract(
            turn_id="turn-complex",
            operations=frozenset({"workspace_write", "execution"}),
            deliverables=(
                _deliverable(
                    target_hint="output/report.md",
                    acceptance_criteria=("report exists", _tool_execution_check()),
                ),
                _deliverable(
                    target_hint="output/tests.txt",
                    acceptance_criteria=("tests summarized", "failures explained"),
                ),
            ),
        ),
        task_relation="start",
    )

    decision = await gate.evaluate(request)

    assert decision.kind == "plan_required"
    assert decision.score >= 6
    assert "multiple_deliverables" in decision.soft_reason_codes
    assert "write_with_execution_dependency" in decision.soft_reason_codes


@pytest.mark.asyncio
async def test_approval_likely_capability_is_a_hard_plan_signal() -> None:
    gate = ComplexityGate()
    request = ComplexityGateRequest(
        turn_intent=_turn_contract(deliverables=(_deliverable(),)),
        capability_summary=ComplexityCapabilitySummary(requires_user_approval=True),
    )

    decision = await gate.evaluate(request)

    assert decision.kind == "plan_required"
    assert decision.hard_reason_codes == ("approval_likely",)


@pytest.mark.asyncio
async def test_side_question_and_cancel_preserve_but_do_not_advance_active_plan() -> None:
    gate = ComplexityGate()
    active_plan = ComplexityActivePlanSummary(
        ledger_id="plan-1",
        status="active",
        revision=3,
    )
    side_question = ComplexityGateRequest(
        turn_intent=_turn_contract(
            turn_id="turn-side",
            objective="One quick side question",
            speech_act="side_question",
            operations=frozenset({"conversation"}),
            deliverables=(),
        ),
        task_relation="side_question",
        active_plan=active_plan,
    )
    cancel_request = ComplexityGateRequest(
        turn_intent=_turn_contract(
            turn_id="turn-cancel",
            objective="Stop this task",
            speech_act="cancel",
            operations=frozenset({"conversation"}),
            deliverables=(),
        ),
        task_relation="cancel",
        active_plan=active_plan,
    )

    side_decision = await gate.evaluate(side_question)
    cancel_decision = await gate.evaluate(cancel_request)

    assert side_decision.kind == "preserve_existing_plan"
    assert cancel_decision.kind == "preserve_existing_plan"


@pytest.mark.asyncio
async def test_cancel_turn_never_creates_a_dynamic_plan_decision() -> None:
    gate = ComplexityGate()
    request = ComplexityGateRequest(
        turn_intent=_turn_contract(
            turn_id="turn-cancel-dynamic",
            objective="Stop the current task",
            speech_act="cancel",
            operations=frozenset({"conversation"}),
            deliverables=(),
        ),
        task_relation="cancel",
    )
    prior = ComplexityGate().evaluate_deterministic(
        ComplexityGateRequest(
            turn_intent=_turn_contract(deliverables=(_deliverable(),)),
            task_relation="start",
        )
    )

    assert await gate.recheck(
        request,
        prior_decision=prior,
        completed_iterations=1,
        signals=("read_to_effectful",),
    ) is None


@pytest.mark.asyncio
async def test_active_unfinished_plan_continues_existing_ledger() -> None:
    gate = ComplexityGate()
    request = ComplexityGateRequest(
        turn_intent=_turn_contract(deliverables=(_deliverable(),)),
        task_relation="continue",
        active_plan=ComplexityActivePlanSummary(
            ledger_id="plan-continue",
            status="active",
            revision=2,
        ),
    )

    decision = await gate.evaluate(request)

    assert decision.kind == "continue_existing_plan"
    assert decision.score == 0


@pytest.mark.asyncio
async def test_advisor_can_raise_a_grey_zone_request_to_plan_required() -> None:
    advisor = _Advisor(
        raw_response={
            "response_version": COMPLEXITY_ADVISOR_RESPONSE_VERSION,
            "plan_recommended": True,
            "estimated_distinct_actions": 4,
            "dependency_count": 2,
            "confidence": 0.81,
            "reason_codes": ["cross_tool_dependency"],
        }
    )
    gate = ComplexityGate(advisor=advisor)
    request = ComplexityGateRequest(
        turn_intent=_turn_contract(
            turn_id="turn-grey",
            operations=frozenset({"workspace_write"}),
            deliverables=(
                _deliverable(target_hint="one.md"),
                _deliverable(target_hint="two.md"),
            ),
        ),
        task_relation="start",
    )

    decision = await gate.evaluate(request)

    assert decision.kind == "plan_required"
    assert decision.advisor_used is True
    assert advisor.requests
    assert ComplexityAdvisorRequest.from_dict(advisor.requests[0].to_dict()) == advisor.requests[0]


@pytest.mark.asyncio
async def test_advisor_timeout_fails_closed_for_effectful_grey_zone() -> None:
    advisor = _Advisor(
        raw_response={
            "response_version": COMPLEXITY_ADVISOR_RESPONSE_VERSION,
            "plan_recommended": False,
            "estimated_distinct_actions": 2,
            "dependency_count": 1,
            "confidence": 0.55,
            "reason_codes": ["would_not_be_used"],
        },
        delay_seconds=0.05,
    )
    gate = ComplexityGate(
        advisor=advisor,
        config=ComplexityGateConfig(advisor_timeout_seconds=0.01),
    )
    request = ComplexityGateRequest(
        turn_intent=_turn_contract(
            turn_id="turn-timeout",
            operations=frozenset({"workspace_write"}),
            deliverables=(
                _deliverable(target_hint="one.md"),
                _deliverable(target_hint="two.md"),
            ),
        ),
    )

    decision = await gate.evaluate(request)

    assert decision.kind == "plan_required"
    assert decision.advisor_used is True
    assert "advisor_timeout" in decision.soft_reason_codes


@pytest.mark.asyncio
async def test_advisor_malformed_result_stays_planless_for_read_only_grey_zone() -> None:
    advisor = _Advisor(raw_response={"unexpected": True})
    gate = ComplexityGate(advisor=advisor)
    request = ComplexityGateRequest(
        turn_intent=_turn_contract(
            turn_id="turn-readonly-grey",
            objective="Research two bounded topics",
            speech_act="request_information",
            operations=frozenset({"open_world_lookup", "literature_research"}),
            deliverables=(
                _deliverable(target_hint="topic-a.md"),
                _deliverable(target_hint="topic-b.md"),
            ),
        ),
    )

    decision = await gate.evaluate(request)

    assert decision.kind == "no_plan"
    assert decision.advisor_used is True
    assert "advisor_malformed" in decision.soft_reason_codes


@pytest.mark.asyncio
async def test_same_semantics_with_different_wording_produce_same_deterministic_decision() -> None:
    gate = ComplexityGate()
    request_one = ComplexityGateRequest(
        turn_intent=_turn_contract(
            turn_id="turn-words-1",
            objective="Please update the selected report in the workspace.",
            deliverables=(_deliverable(),),
        ),
    )
    request_two = ComplexityGateRequest(
        turn_intent=_turn_contract(
            turn_id="turn-words-2",
            objective="請修改同一份工作區報告。",
            deliverables=(_deliverable(),),
        ),
    )

    decision_one = gate.evaluate_deterministic(request_one)
    decision_two = gate.evaluate_deterministic(request_two)

    assert decision_one.kind == decision_two.kind
    assert decision_one.score == decision_two.score
    assert decision_one.hard_reason_codes == decision_two.hard_reason_codes
    assert decision_one.soft_reason_codes == decision_two.soft_reason_codes
