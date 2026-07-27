from __future__ import annotations

import asyncio
from types import MappingProxyType

import pytest

from mochi.agents.artifact_verifier import ArtifactReceipt, ToolExecutionEvidence
from mochi.agents.outcome_verifier import (
    ArtifactVerifierAdapter,
    CriterionReceipt,
    DeterministicVerifierRegistry,
    ResponseShapeVerifier,
    SemanticJudgeVerifier,
    StateVerifier,
    ToolExecutionVerifier,
    VERIFICATION_RECEIPT_VERSION,
    VerificationCriterion,
    VerificationEvidence,
    VerificationPlanCompiler,
    VerificationReceipt,
)
from mochi.agents.plan_ledger import PlanItem
from mochi.agents.turn_intent_contract import DeliverableContract


def _artifact_receipt(*, verification_status: str, retry_disposition: str = "none") -> ArtifactReceipt:
    return ArtifactReceipt(
        operation_id="op-1",
        turn_id="turn-1",
        goal_id="goal-1",
        tool_call_ids=("call-1",),
        resolved_targets=("file.txt",),
        changed_paths=("file.txt",),
        before_hashes={"file.txt": None},
        after_hashes={"file.txt": "a" * 64},
        expected_after_hashes={"file.txt": "a" * 64},
        execution_status="succeeded",
        verification_status=verification_status,
        retry_disposition=retry_disposition,
        targets=(),
    )


def test_verification_criterion_round_trip_rejects_extra_fields() -> None:
    criterion = VerificationCriterion(
        criterion_id="criterion-1",
        kind="artifact",
        required=True,
        description="check file",
        source_turn_ids=("turn-1",),
        verifier_id="artifact",
        payload={"check": "exists"},
    )
    payload = criterion.to_dict()
    assert VerificationCriterion.from_dict(payload) == criterion
    with pytest.raises(ValueError, match="unexpected keys"):
        VerificationCriterion.from_dict({**payload, "extra": True})


def test_verification_receipt_round_trip() -> None:
    receipt = VerificationReceipt(
        receipt_version=VERIFICATION_RECEIPT_VERSION,
        receipt_id="receipt-1",
        turn_id="turn-1",
        goal_id="goal-1",
        verdict="verified",
        criteria=(
            CriterionReceipt(
                criterion_id="criterion-1",
                verdict="verified",
                verifier_id="artifact",
                evidence_refs=("artifact:1",),
                reason_code="artifact_verified",
                retry_disposition="none",
                confidence=0.75,
            ),
        ),
        hard_failure=False,
        retry_disposition="none",
    )
    assert VerificationReceipt.from_dict(receipt.to_dict()) == receipt


@pytest.mark.asyncio
async def test_registry_fails_when_required_deterministic_criterion_fails() -> None:
    criteria = (
        VerificationCriterion(
            criterion_id="artifact-1",
            kind="artifact",
            required=True,
            description="artifact proof",
            source_turn_ids=("turn-1",),
            verifier_id="artifact",
            payload={"receipt_id": "artifact:1"},
        ),
        VerificationCriterion(
            criterion_id="semantic-1",
            kind="semantic",
            required=True,
            description="semantic check",
            source_turn_ids=("turn-1",),
            verifier_id="semantic_judge",
            payload={"rubric": "must be helpful"},
        ),
    )

    class Judge:
        async def judge(self, criterion: VerificationCriterion, evidence: VerificationEvidence) -> dict[str, object]:
            del criterion, evidence
            return {
                "verdict": "verified",
                "evidence_refs": ["semantic:1"],
                "reason_code": "semantic_ok",
                "retry_disposition": "none",
                "confidence": 0.8,
            }

    registry = DeterministicVerifierRegistry(
        (
            ArtifactVerifierAdapter(),
            SemanticJudgeVerifier(judge=Judge()),
        )
    )
    evidence = VerificationEvidence(
        artifact_receipts={"artifact:1": _artifact_receipt(verification_status="failed", retry_disposition="requires_replan")}
    )
    receipt = await registry.verify_all(
        criteria,
        evidence,
        receipt_id="receipt-1",
        turn_id="turn-1",
        goal_id="goal-1",
    )
    assert receipt.verdict == "failed"
    assert receipt.hard_failure is True
    assert receipt.retry_disposition == "requires_replan"


@pytest.mark.asyncio
async def test_registry_marks_missing_required_support_unverified() -> None:
    registry = DeterministicVerifierRegistry((StateVerifier(),))
    criteria = (
        VerificationCriterion(
            criterion_id="manual-1",
            kind="manual",
            required=True,
            description="needs human",
            source_turn_ids=("turn-1",),
            verifier_id="manual_review",
            payload={"reason": "free-form"},
        ),
    )
    receipt = await registry.verify_all(
        criteria,
        VerificationEvidence(),
        receipt_id="receipt-1",
        turn_id="turn-1",
        goal_id=None,
    )
    assert receipt.verdict == "unverified"
    assert receipt.hard_failure is False


@pytest.mark.asyncio
async def test_registry_required_semantic_failure_blocks_verification_without_hard_failure() -> None:
    criterion = VerificationCriterion(
        criterion_id="semantic-1",
        kind="semantic",
        required=True,
        description="semantic check",
        source_turn_ids=("turn-1",),
        verifier_id="semantic_judge",
        payload={"rubric": "must directly answer"},
    )

    class Judge:
        async def judge(
            self,
            criterion: VerificationCriterion,
            evidence: VerificationEvidence,
        ) -> dict[str, object]:
            del criterion, evidence
            return {
                "verdict": "failed",
                "evidence_refs": ["semantic:1"],
                "reason_code": "semantic_mismatch",
                "retry_disposition": "requires_replan",
                "confidence": 0.4,
            }

    receipt = await DeterministicVerifierRegistry(
        (SemanticJudgeVerifier(judge=Judge()),)
    ).verify_all(
        (criterion,),
        VerificationEvidence(),
        receipt_id="receipt-1",
        turn_id="turn-1",
        goal_id="goal-1",
    )

    assert receipt.verdict == "failed"
    assert receipt.hard_failure is False
    assert receipt.retry_disposition == "requires_replan"
    assert receipt.criteria[0].reason_code == "semantic_mismatch"


def test_compiler_maps_legacy_and_structured_acceptance_criteria() -> None:
    compiler = VerificationPlanCompiler(semantic_fallback_enabled=True)
    deliverable = DeliverableContract(
        kind="patch",
        source_turn_ids=("turn-1",),
        acceptance_criteria=(
            "exists",
            "contains:hello",
            MappingProxyType(
                {
                    "schema_version": 1,
                    "kind": "tool_execution",
                    "check": "test",
                    "tool_name": "shell",
                    "profile_id": "pytest",
                }
            ),
            "explain the result",
        ),
    )
    item = PlanItem(
        item_id="item-1",
        title="Finish patch",
        status="pending",
        dependencies=(),
        success_criteria=("non-empty",),
        source_turn_ids=("turn-1",),
    )
    criteria = compiler.compile(
        deliverables=(deliverable,),
        plan_items=(item,),
        response_shape={"required": True, "required_sections": ["## Result"]},
    )
    assert [criterion.kind for criterion in criteria] == [
        "artifact",
        "artifact",
        "tool_execution",
        "semantic",
        "artifact",
        "response_shape",
    ]


@pytest.mark.asyncio
async def test_tool_execution_and_response_shape_verifiers() -> None:
    tool_criterion = VerificationCriterion(
        criterion_id="tool-1",
        kind="tool_execution",
        required=True,
        description="pytest passed",
        source_turn_ids=("turn-1",),
        verifier_id="tool_execution",
        payload={"tool_name": "shell", "expected_exit_code": 0},
    )
    response_criterion = VerificationCriterion(
        criterion_id="response-1",
        kind="response_shape",
        required=True,
        description="response sections",
        source_turn_ids=("turn-1",),
        verifier_id="response_shape",
        payload={"required_sections": ["## Result"], "required_keys": ["summary"]},
    )
    evidence = VerificationEvidence(
        tool_execution_evidence=(
            ToolExecutionEvidence(
                call_id="call-1",
                tool_name="shell",
                arguments_digest="a" * 64,
                operation_id="op-1",
                turn_id="turn-1",
                exit_code=0,
                status="completed",
                approval_pending=False,
                error=None,
                arguments={"command": "pytest"},
            ),
        ),
        response_json={"summary": "done"},
        response_text="## Result\nEverything passed.",
    )
    tool_receipt = await ToolExecutionVerifier().verify(tool_criterion, evidence)
    response_receipt = await ResponseShapeVerifier().verify(response_criterion, evidence)
    assert tool_receipt.verdict == "verified"
    assert response_receipt.verdict == "verified"


@pytest.mark.asyncio
async def test_semantic_judge_fails_closed_on_timeout_and_malformed_payload() -> None:
    criterion = VerificationCriterion(
        criterion_id="semantic-1",
        kind="semantic",
        required=True,
        description="judge text",
        source_turn_ids=("turn-1",),
        verifier_id="semantic_judge",
        payload={"rubric": "be correct"},
    )

    class SlowJudge:
        async def judge(self, criterion: VerificationCriterion, evidence: VerificationEvidence) -> dict[str, object]:
            del criterion, evidence
            await asyncio.sleep(0.05)
            return {
                "verdict": "verified",
                "evidence_refs": [],
                "reason_code": "ok",
                "retry_disposition": "none",
                "confidence": 0.9,
            }

    class BadJudge:
        async def judge(self, criterion: VerificationCriterion, evidence: VerificationEvidence) -> dict[str, object]:
            del criterion, evidence
            return {"bad": True}

    timeout_receipt = await SemanticJudgeVerifier(
        judge=SlowJudge(),
        timeout_seconds=0.001,
    ).verify(criterion, VerificationEvidence())
    malformed_receipt = await SemanticJudgeVerifier(
        judge=BadJudge(),
        timeout_seconds=1.0,
    ).verify(criterion, VerificationEvidence())
    assert timeout_receipt.verdict == "unverified"
    assert timeout_receipt.reason_code == "semantic_judge_timeout"
    assert malformed_receipt.verdict == "unverified"
    assert malformed_receipt.reason_code == "semantic_judge_malformed"
