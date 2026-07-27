from __future__ import annotations

import asyncio
from types import MappingProxyType

import pytest

from mochi.agents.artifact_verifier import (
    ArtifactReceipt,
    ArtifactTargetReceipt,
    ToolExecutionEvidence,
)
from mochi.agents.outcome_verifier import (
    ArtifactVerifierAdapter,
    CriterionReceipt,
    DeterministicVerifierRegistry,
    ResponseShapeVerifier,
    SemanticJudgeVerifier,
    StateVerifier,
    ToolExecutionVerifier,
    VERIFICATION_RECEIPT_EVENT,
    VERIFICATION_RECEIPT_EVENT_SCHEMA_VERSION,
    VERIFICATION_RECEIPT_VERSION,
    VerificationCriterion,
    VerificationEvidence,
    VerificationPlanCompiler,
    VerificationReceipt,
    VerificationReceiptRepository,
)
from mochi.agents.plan_ledger import PlanItem
from mochi.agents.turn_intent_contract import DeliverableContract
from mochi.sessions.store import SessionStore


def _artifact_receipt(
    *,
    verification_status: str,
    retry_disposition: str = "none",
) -> ArtifactReceipt:
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


def _artifact_target(
    requested_path: str,
    *,
    verification_status: str,
) -> ArtifactTargetReceipt:
    return ArtifactTargetReceipt(
        requested_path=requested_path,
        resolved_path=requested_path,
        expected_exists=True,
        exists=verification_status == "verified",
        in_workspace=True,
        size_bytes=1 if verification_status == "verified" else None,
        before_sha256=None,
        expected_after_sha256=None,
        actual_after_sha256=("a" * 64 if verification_status == "verified" else None),
        changed=True if verification_status == "verified" else None,
        verification_status=verification_status,  # type: ignore[arg-type]
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
async def test_artifact_verifier_binds_each_criterion_to_its_target_evidence() -> None:
    receipt = ArtifactReceipt(
        operation_id="op-multi",
        turn_id="turn-1",
        goal_id="goal-1",
        tool_call_ids=("call-1",),
        resolved_targets=("first.txt", "second.txt"),
        changed_paths=("first.txt",),
        before_hashes={"first.txt": None, "second.txt": None},
        after_hashes={"first.txt": "a" * 64, "second.txt": None},
        expected_after_hashes={"first.txt": None, "second.txt": None},
        execution_status="succeeded",
        verification_status="failed",
        retry_disposition="requires_replan",
        targets=(
            _artifact_target("first.txt", verification_status="verified"),
            _artifact_target("second.txt", verification_status="failed"),
        ),
    )
    criteria = tuple(
        VerificationCriterion(
            criterion_id=f"artifact-{index}",
            kind="artifact",
            required=True,
            description=f"verify {target}",
            source_turn_ids=("turn-1",),
            verifier_id="artifact",
            payload={
                "receipt_id": "artifact:multi",
                "check": "exists",
                "target_hint": target,
            },
        )
        for index, target in enumerate(("first.txt", "second.txt"), start=1)
    )

    aggregate = await DeterministicVerifierRegistry().verify_all(
        criteria,
        VerificationEvidence(artifact_receipts={"artifact:multi": receipt}),
        receipt_id="verification:turn-1",
        turn_id="turn-1",
        goal_id="goal-1",
    )

    assert [item.verdict for item in aggregate.criteria] == ["verified", "failed"]
    assert aggregate.criteria[0].evidence_refs != aggregate.criteria[1].evidence_refs
    assert aggregate.verdict == "failed"


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
        async def judge(
            self,
            criterion: VerificationCriterion,
            evidence: VerificationEvidence,
        ) -> dict[str, object]:
            del criterion, evidence
            return {
                "verdict": "verified",
                "evidence_refs": ["response"],
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
        artifact_receipts={
            "artifact:1": _artifact_receipt(
                verification_status="failed",
                retry_disposition="requires_replan",
            )
        },
        response_text="The response is available as host evidence.",
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
async def test_registry_required_semantic_failure_blocks_verification_without_hard_failure(
) -> None:
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
                "evidence_refs": ["response"],
                "reason_code": "semantic_mismatch",
                "retry_disposition": "requires_replan",
                "confidence": 0.4,
            }

    receipt = await DeterministicVerifierRegistry(
        (SemanticJudgeVerifier(judge=Judge()),)
    ).verify_all(
        (criterion,),
        VerificationEvidence(response_text="A response that does not satisfy the rubric."),
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


def test_compiler_assigns_unique_target_bound_ids_for_multiple_deliverables() -> None:
    compiler = VerificationPlanCompiler(semantic_fallback_enabled=True)
    criteria = compiler.compile(
        deliverables=(
            DeliverableContract(
                kind="file",
                target_hint="first.txt",
                source_turn_ids=("turn-1",),
                acceptance_criteria=("exists", "explain first"),
            ),
            DeliverableContract(
                kind="file",
                target_hint="second.txt",
                source_turn_ids=("turn-1",),
                acceptance_criteria=("exists", "explain second"),
            ),
        )
    )

    assert len({criterion.criterion_id for criterion in criteria}) == len(criteria)
    assert [criterion.payload.get("target_hint") for criterion in criteria] == [
        "first.txt",
        "first.txt",
        "second.txt",
        "second.txt",
    ]
    assert compiler.compile(
        deliverables=(
            DeliverableContract(
                kind="file",
                target_hint="first.txt",
                source_turn_ids=("turn-1",),
                acceptance_criteria=("exists", "explain first"),
            ),
            DeliverableContract(
                kind="file",
                target_hint="second.txt",
                source_turn_ids=("turn-1",),
                acceptance_criteria=("exists", "explain second"),
            ),
        )
    ) == criteria


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
        async def judge(
            self,
            criterion: VerificationCriterion,
            evidence: VerificationEvidence,
        ) -> dict[str, object]:
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
        async def judge(
            self,
            criterion: VerificationCriterion,
            evidence: VerificationEvidence,
        ) -> dict[str, object]:
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


@pytest.mark.asyncio
async def test_semantic_judge_rejects_model_invented_evidence_refs() -> None:
    criterion = VerificationCriterion(
        criterion_id="semantic-1",
        kind="semantic",
        required=True,
        description="judge text",
        source_turn_ids=("turn-1",),
        verifier_id="semantic_judge",
        payload={"rubric": "be correct"},
    )

    class InventingJudge:
        async def judge(self, criterion, evidence):  # type: ignore[no-untyped-def]
            del criterion, evidence
            return {
                "verdict": "verified",
                "evidence_refs": ["model-invented-proof"],
                "reason_code": "looks_good",
                "retry_disposition": "none",
                "confidence": 0.99,
            }

    receipt = await SemanticJudgeVerifier(judge=InventingJudge()).verify(
        criterion,
        VerificationEvidence(response_text="Host-owned response evidence."),
    )

    assert receipt.verdict == "unverified"
    assert receipt.evidence_refs == ()
    assert receipt.reason_code == "semantic_judge_malformed"


def _aggregate_receipt(*, verdict: str = "verified") -> VerificationReceipt:
    return VerificationReceipt(
        receipt_version=VERIFICATION_RECEIPT_VERSION,
        receipt_id="verification:turn-1",
        turn_id="turn-1",
        goal_id="goal-1",
        verdict=verdict,  # type: ignore[arg-type]
        criteria=(
            CriterionReceipt(
                criterion_id="criterion-1",
                verdict=verdict,  # type: ignore[arg-type]
                verifier_id="artifact",
                evidence_refs=("artifact:1",),
                reason_code="artifact_verified",
                retry_disposition="none",
            ),
        ),
        hard_failure=False,
        retry_disposition="none",
    )


@pytest.mark.asyncio
async def test_verification_receipt_repository_is_idempotent_and_restart_safe(tmp_path) -> None:
    sessions_dir = tmp_path / "sessions"
    first = VerificationReceiptRepository(SessionStore(sessions_dir))
    receipt = _aggregate_receipt()

    saved = await first.save(
        "session-1",
        receipt,
        expected_revision=0,
        idempotency_key="verification:turn-1:verified",
    )
    replay = await first.save(
        "session-1",
        receipt,
        expected_revision=0,
        idempotency_key="verification:turn-1:verified",
        timestamp="2026-07-27T01:00:00+00:00",
    )
    restarted = await VerificationReceiptRepository(
        SessionStore(sessions_dir)
    ).load("session-1", "turn-1")

    assert saved.status == "saved"
    assert replay.status == "saved"
    assert replay.idempotent_replay is True
    assert restarted.status == "loaded"
    assert restarted.receipt == receipt
    events = await SessionStore(sessions_dir).load_session("session-1")
    assert [event.get("event") for event in events].count(VERIFICATION_RECEIPT_EVENT) == 1


@pytest.mark.asyncio
async def test_verification_receipt_repository_latest_malformed_and_future_fail_closed(
    tmp_path,
) -> None:
    store = SessionStore(tmp_path / "sessions")
    repository = VerificationReceiptRepository(store)
    receipt = _aggregate_receipt()
    await repository.save(
        "session-1",
        receipt,
        expected_revision=0,
        idempotency_key="verification:valid",
    )
    malformed = {
        "type": "session_meta",
        "event": VERIFICATION_RECEIPT_EVENT,
        "schema_version": VERIFICATION_RECEIPT_EVENT_SCHEMA_VERSION,
        "session_id": "session-1",
        "turn_id": "turn-1",
        "receipt_revision": 2,
        "idempotency_key": "verification:malformed",
        "verification_receipt": {"receipt_version": "verification-receipt-v99"},
        "timestamp": "2026-07-27T02:00:00+00:00",
    }
    await store.save_event("session-1", malformed)
    assert (await repository.load("session-1", "turn-1")).status == "invalid"

    future_store = SessionStore(tmp_path / "future-sessions")
    await future_store.save_event(
        "session-2",
        {
            **malformed,
            "session_id": "session-2",
            "schema_version": VERIFICATION_RECEIPT_EVENT_SCHEMA_VERSION + 1,
        },
    )
    future = await VerificationReceiptRepository(future_store).load(
        "session-2",
        "turn-1",
    )
    assert future.status == "unsupported_version"
