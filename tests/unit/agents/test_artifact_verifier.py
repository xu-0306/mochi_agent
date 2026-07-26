from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from mochi.agents.artifact_verifier import (
    ARTIFACT_RECEIPT_SCHEMA_VERSION,
    ArtifactExpectation,
    ArtifactReceipt,
    ArtifactVerifier,
    ValidationProfileRegistry,
    tool_arguments_digest,
)
from mochi.agents.events import ToolCallRequestEvent, ToolCallResultEvent


def _request(
    path: str,
    *,
    call_id: str = "call-1",
    content: str = "verified content\n",
) -> ToolCallRequestEvent:
    return ToolCallRequestEvent(
        call_id=call_id,
        tool_name="file_write",
        arguments={"path": path, "content": content},
    )


def _success(
    path: Path,
    *,
    call_id: str = "call-1",
    file_changes: list[dict[str, object]] | None = None,
) -> ToolCallResultEvent:
    metadata: dict[str, object] = {"resolved_path": str(path)}
    if file_changes is not None:
        metadata["file_changes"] = file_changes
    return ToolCallResultEvent(
        call_id=call_id,
        tool_name="file_write",
        result=str(path),
        metadata=metadata,
    )


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _exec_request(
    command: str,
    *,
    call_id: str = "exec-call-1",
) -> ToolCallRequestEvent:
    return ToolCallRequestEvent(
        call_id=call_id,
        tool_name="exec_command",
        arguments={"command": command, "shell": "powershell"},
    )


def _exec_result(
    *,
    call_id: str = "exec-call-1",
    operation_id: str = "exec-operation-1",
    exit_code: int | None = 0,
    error: str | None = None,
    metadata: dict[str, object] | None = None,
) -> ToolCallResultEvent:
    result_metadata: dict[str, object] = {
        "operation_id": operation_id,
        "status": "completed" if error is None else "failed",
    }
    if metadata:
        result_metadata.update(metadata)
    return ToolCallResultEvent(
        call_id=call_id,
        tool_name="exec_command",
        result={"exit_code": exit_code},
        error=error,
        metadata=result_metadata,
    )


def _tool_execution_criterion(
    *,
    profile_id: str = "pytest",
    check: str = "test",
    tool_name: str = "exec_command",
    **pins: object,
) -> dict[str, object]:
    return {
        "schema_version": 1,
        "kind": "tool_execution",
        "check": check,
        "profile_id": profile_id,
        "tool_name": tool_name,
        **pins,
    }


def test_verifier_rereads_successful_file_and_builds_structured_receipt(
    tmp_path: Path,
) -> None:
    target = tmp_path / "output" / "report.md"
    target.parent.mkdir()
    target.write_bytes(b"verified content\n")

    result = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-success",
        goal_id="goal-success",
        requests=[_request("output/report.md")],
        results=[_success(target)],
        expectations=[
            ArtifactExpectation(
                path="output/report.md",
                acceptance_criteria=("exists", "non-empty", "contains:verified"),
            )
        ],
    )

    assert result.success is True
    assert result.receipt.schema_version == ARTIFACT_RECEIPT_SCHEMA_VERSION
    assert result.receipt.execution_status == "succeeded"
    assert result.receipt.verification_status == "verified"
    assert result.receipt.retry_disposition == "none"
    assert result.receipt.resolved_targets == (str(target.resolve()),)
    assert result.receipt.changed_paths == (str(target.resolve()),)
    assert result.receipt.after_hashes[str(target.resolve())] == _sha256(
        b"verified content\n"
    )
    assert result.receipt.goal_id == "goal-success"
    assert result.receipt.to_dict()["operation_id"].startswith("artifact-op-v1:")


def test_v2_receipt_round_trip_preserves_scope_evidence(tmp_path: Path) -> None:
    target = tmp_path / "report.md"
    target.write_text("verified content\n", encoding="utf-8")
    result = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-v2-round-trip",
        requests=[_request("report.md")],
        results=[_success(target)],
    )

    restored = ArtifactReceipt.from_dict(result.receipt.to_dict())

    assert restored.schema_version == ARTIFACT_RECEIPT_SCHEMA_VERSION == 3
    assert restored.scope_evidence == result.receipt.scope_evidence
    assert restored.scope_evidence.authorized_paths_by_call == {
        "call-1": (str(target.resolve()),)
    }
    assert restored.scope_evidence.observed_paths_by_call == {
        "call-1": (str(target.resolve()),)
    }


def test_v1_receipt_migrates_to_v2_without_fabricating_scope_evidence(
    tmp_path: Path,
) -> None:
    target = tmp_path / "report.md"
    target.write_text("verified content\n", encoding="utf-8")
    result = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-v1-migration",
        requests=[_request("report.md")],
        results=[_success(target)],
    )
    legacy_payload = result.receipt.to_dict()
    legacy_payload["schema_version"] = 1
    legacy_payload.pop("scope_evidence")

    migrated = ArtifactReceipt.from_dict(legacy_payload)

    assert migrated.schema_version == 3
    assert migrated.operation_id == result.receipt.operation_id
    assert migrated.scope_evidence.authorized_paths_by_call == {}
    assert migrated.scope_evidence.observed_paths_by_call == {}
    assert migrated.scope_evidence.unexpected_changed_paths == ()


def test_receipt_reader_rejects_unknown_future_schema(tmp_path: Path) -> None:
    target = tmp_path / "report.md"
    target.write_text("verified content\n", encoding="utf-8")
    result = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-future-schema",
        requests=[_request("report.md")],
        results=[_success(target)],
    )
    payload = result.receipt.to_dict()
    payload["schema_version"] = 99

    with pytest.raises(ValueError, match="Unsupported artifact receipt schema"):
        ArtifactReceipt.from_dict(payload)


def test_required_multi_deliverable_expectations_fail_closed_for_missing_sibling(
    tmp_path: Path,
) -> None:
    report = tmp_path / "output" / "report.md"
    report.parent.mkdir()
    report.write_text("accepted report\n", encoding="utf-8")

    result = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-multi",
        requests=[_request("output/report.md", content="accepted report\n")],
        results=[_success(report)],
        expectations=(
            ArtifactExpectation(
                path="output/report.md",
                acceptance_criteria=("contains:accepted",),
            ),
            ArtifactExpectation(
                path="output/summary.md",
                acceptance_criteria=("contains:summary",),
            ),
        ),
    )

    assert result.success is False
    receipts = {item.requested_path: item for item in result.receipt.targets}
    assert receipts["output/report.md"].verification_status == "verified"
    assert receipts["output/summary.md"].verification_status == "failed"


def test_unsupported_acceptance_criterion_requires_replan(tmp_path: Path) -> None:
    target = tmp_path / "output.md"
    target.write_text("content\n", encoding="utf-8")
    result = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-unsupported",
        requests=[_request("output.md", content="content\n")],
        results=[_success(target)],
        expectations=(ArtifactExpectation(path="output.md", acceptance_criteria=("valid markdown",)),),
    )
    assert result.success is False
    assert "unsupported_acceptance_criterion" in str(result.receipt.to_dict())


def test_same_target_deliverables_merge_all_acceptance_criteria(tmp_path: Path) -> None:
    target = tmp_path / "report.md"
    target.write_text("first requirement only\n", encoding="utf-8")
    failing = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="duplicate-target-fail",
        requests=[_request("report.md", content="first requirement only\n")],
        results=[_success(target)],
        expectations=(
            ArtifactExpectation(path="report.md", acceptance_criteria=("contains:first",)),
            ArtifactExpectation(path="report.md", acceptance_criteria=("contains:second",)),
        ),
    )
    assert failing.success is False
    assert {check.criterion for check in failing.receipt.targets[0].acceptance_checks} >= {
        "contains:first", "contains:second"
    }

    target.write_text("first and second requirements\n", encoding="utf-8")
    passing = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="duplicate-target-pass",
        requests=[_request("report.md", content="first and second requirements\n")],
        results=[_success(target)],
        expectations=(
            ArtifactExpectation(path="report.md", acceptance_criteria=("contains:first",)),
            ArtifactExpectation(path="report.md", acceptance_criteria=("contains:second",)),
        ),
    )
    assert passing.success is True


def test_tool_claims_success_but_missing_file_fails_verification(tmp_path: Path) -> None:
    target = tmp_path / "missing.txt"

    result = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-missing",
        requests=[_request("missing.txt")],
        results=[_success(target)],
    )

    assert result.success is False
    assert result.receipt.execution_status == "succeeded"
    assert result.receipt.verification_status == "failed"
    assert result.receipt.retry_disposition == "requires_replan"
    assert result.receipt.targets[0].exists is False
    assert "does not exist" in result.receipt.targets[0].errors[0]


@pytest.mark.parametrize(
    ("expectation", "failure_code"),
    [
        (
            ArtifactExpectation(
                path="artifact.txt",
                expected_after_sha256="0" * 64,
            ),
            "digest_mismatch",
        ),
        (
            ArtifactExpectation(
                path="artifact.txt",
                expected_content="different content",
            ),
            "digest_mismatch",
        ),
    ],
)
def test_digest_or_content_mismatch_is_not_completion(
    tmp_path: Path,
    expectation: ArtifactExpectation,
    failure_code: str,
) -> None:
    target = tmp_path / "artifact.txt"
    target.write_text("actual content", encoding="utf-8")

    result = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-mismatch",
        requests=[_request("artifact.txt", content="actual content")],
        results=[_success(target)],
        expectations=[expectation],
    )

    assert result.success is False
    assert result.receipt.verification_status == "failed"
    assert result.receipt.retry_disposition == "requires_replan"
    assert failure_code in {
        check.code for check in result.receipt.targets[0].acceptance_checks
    }


def test_acceptance_criteria_pass_and_fail_are_explicit(tmp_path: Path) -> None:
    target = tmp_path / "report.md"
    target.write_text("# Summary\nready\n", encoding="utf-8")
    request = _request("report.md", content="# Summary\nready\n")
    tool_result = _success(target)

    passed = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-criteria-pass",
        requests=[request],
        results=[tool_result],
        expectations=[
            ArtifactExpectation(
                path="report.md",
                acceptance_criteria=("non-empty", "contains:Summary"),
            )
        ],
    )
    failed = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-criteria-fail",
        requests=[request],
        results=[tool_result],
        expectations=[
            ArtifactExpectation(
                path="report.md",
                acceptance_criteria=("contains:Missing section",),
            )
        ],
    )

    assert passed.success is True
    assert failed.success is False
    assert failed.receipt.targets[0].acceptance_checks[-1].code == "contains_mismatch"


def test_unsupported_acceptance_criterion_fails_closed(tmp_path: Path) -> None:
    target = tmp_path / "report.md"
    target.write_text("ready", encoding="utf-8")

    result = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-unsupported",
        requests=[_request("report.md", content="ready")],
        results=[_success(target)],
        expectations=[
            ArtifactExpectation(
                path="report.md",
                acceptance_criteria=("lint passes",),
            )
        ],
    )

    assert result.success is False
    assert (
        result.receipt.targets[0].acceptance_checks[-1].code
        == "unsupported_acceptance_criterion"
    )
    assert result.receipt.retry_disposition == "requires_replan"


def test_workspace_escape_is_terminal_even_when_file_exists(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    outside.write_text("outside", encoding="utf-8")

    result = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-escape",
        requests=[_request(str(outside), content="outside")],
        results=[_success(outside)],
    )

    assert result.success is False
    assert result.receipt.verification_status == "failed"
    assert result.receipt.retry_disposition == "terminal"
    assert result.receipt.targets[0].in_workspace is False


def test_unexpected_sibling_changed_path_fails_closed_without_scope_expansion(
    tmp_path: Path,
) -> None:
    allowed = tmp_path / "allowed.txt"
    sibling = tmp_path / "sibling.txt"
    allowed.write_text("allowed\n", encoding="utf-8")
    sibling.write_text("sibling\n", encoding="utf-8")

    result = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-unexpected-sibling",
        requests=[_request("allowed.txt", content="allowed\n")],
        results=[
            ToolCallResultEvent(
                call_id="call-1",
                tool_name="file_write",
                metadata={
                    "file_changes": [
                        {"path": str(allowed)},
                        {"path": str(sibling)},
                    ]
                },
            )
        ],
        expectations=(ArtifactExpectation(path="sibling.txt"),),
    )

    assert result.success is False
    assert result.receipt.verification_status == "failed"
    assert result.receipt.retry_disposition == "requires_replan"
    assert result.receipt.errors.count("unexpected_changed_path") == 1
    assert result.receipt.scope_evidence.authorized_paths_by_call == {
        "call-1": (str(allowed.resolve()),)
    }
    assert result.receipt.scope_evidence.observed_paths_by_call == {
        "call-1": (str(allowed.resolve()), str(sibling.resolve()))
    }
    assert result.receipt.scope_evidence.unexpected_changed_paths[0].path == str(sibling)


def test_cross_call_changed_path_laundering_fails_closed(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first\n", encoding="utf-8")
    second.write_text("second\n", encoding="utf-8")

    result = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-cross-call",
        requests=[
            _request("first.txt", call_id="call-first", content="first\n"),
            _request("second.txt", call_id="call-second", content="second\n"),
        ],
        results=[
            ToolCallResultEvent(
                call_id="call-first",
                tool_name="file_write",
                metadata={"resolved_path": str(second)},
            ),
            ToolCallResultEvent(
                call_id="call-second",
                tool_name="file_write",
                metadata={"resolved_path": str(second)},
            ),
        ],
    )

    assert result.success is False
    assert result.receipt.execution_status == "succeeded"
    assert result.receipt.verification_status == "failed"
    assert result.receipt.retry_disposition == "requires_replan"
    assert len(result.receipt.scope_evidence.unexpected_changed_paths) == 1
    violation = result.receipt.scope_evidence.unexpected_changed_paths[0]
    assert violation.call_id == "call-first"
    assert violation.resolved_path == str(second.resolve())


def test_relative_absolute_and_windows_case_aliases_share_authorized_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "alias.txt"
    target.write_bytes(b"alias\n")
    resolved_alias = str(target).upper() if os.name == "nt" else str(target.resolve())

    result = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-alias",
        requests=[_request("alias.txt", content="alias\n")],
        results=[
            ToolCallResultEvent(
                call_id="call-1",
                tool_name="file_write",
                metadata={"resolved_path": resolved_alias},
            )
        ],
    )

    assert result.success is True
    assert result.receipt.scope_evidence.unexpected_changed_paths == ()
    assert result.receipt.changed_paths == (str(target.resolve()),)


def test_valid_multi_file_patch_reports_only_its_authorized_targets(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_bytes(b"first\n")
    second.write_bytes(b"second\n")
    patch = """*** Begin Patch
*** Add File: first.txt
+first
*** Add File: second.txt
+second
*** End Patch
"""

    result = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-patch-scope",
        requests=[
            ToolCallRequestEvent(
                call_id="call-patch",
                tool_name="apply_patch",
                arguments={"patch": patch},
            )
        ],
        results=[
            ToolCallResultEvent(
                call_id="call-patch",
                tool_name="apply_patch",
                metadata={
                    "file_changes": [
                        {"path": str(first), "new_content": "first\n"},
                        {"path": str(second), "new_content": "second\n"},
                    ]
                },
            )
        ],
    )

    assert result.success is True
    assert result.receipt.changed_paths == (
        str(first.resolve()),
        str(second.resolve()),
    )
    assert result.receipt.scope_evidence.unexpected_changed_paths == ()


def test_file_edit_path_remains_authorized(tmp_path: Path) -> None:
    target = tmp_path / "edited.txt"
    target.write_text("edited\n", encoding="utf-8")

    result = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-file-edit",
        requests=[
            ToolCallRequestEvent(
                call_id="call-edit",
                tool_name="file_edit",
                arguments={"path": "edited.txt"},
            )
        ],
        results=[
            ToolCallResultEvent(
                call_id="call-edit",
                tool_name="file_edit",
                metadata={"resolved_path": str(target)},
            )
        ],
    )

    assert result.success is True
    assert result.receipt.scope_evidence.unexpected_changed_paths == ()


def test_unexpected_workspace_escape_is_terminal(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed.txt"
    outside = tmp_path.parent / f"{tmp_path.name}-outside.txt"
    allowed.write_text("allowed\n", encoding="utf-8")
    outside.write_text("outside\n", encoding="utf-8")

    result = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-unexpected-escape",
        requests=[_request("allowed.txt", content="allowed\n")],
        results=[
            ToolCallResultEvent(
                call_id="call-1",
                tool_name="file_write",
                metadata={"resolved_path": str(outside)},
            )
        ],
    )

    assert result.success is False
    assert result.receipt.retry_disposition == "terminal"
    assert "unexpected_changed_path" in result.receipt.errors


def test_multi_file_operation_reports_partial_state_and_recovery(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_bytes(b"first\n")
    patch = """*** Begin Patch
*** Add File: first.txt
+first
*** Add File: second.txt
+second
*** End Patch
"""
    request = ToolCallRequestEvent(
        call_id="call-patch",
        tool_name="apply_patch",
        arguments={"patch": patch},
    )
    tool_result = ToolCallResultEvent(
        call_id="call-patch",
        tool_name="apply_patch",
        result={"paths": [str(first), str(second)], "change_count": 2},
        metadata={
            "file_changes": [
                {
                    "path": str(first),
                    "change_type": "add",
                    "new_content": "first\n",
                },
                {
                    "path": str(second),
                    "change_type": "add",
                    "new_content": "second\n",
                },
            ]
        },
    )

    result = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-partial",
        requests=[request],
        results=[tool_result],
    )

    assert result.success is False
    assert result.receipt.execution_status == "succeeded"
    assert result.receipt.verification_status == "partial"
    assert [target.verification_status for target in result.receipt.targets] == [
        "verified",
        "failed",
    ]
    assert result.receipt.retry_disposition == "requires_replan"
    assert "do not replay" in result.receipt.recovery_plan[0]
    assert str(second.resolve()) in result.receipt.recovery_plan[1]


def test_approval_pending_is_classified_without_retrying_execution(tmp_path: Path) -> None:
    target = tmp_path / "approval.txt"
    result = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-approval",
        requests=[_request("approval.txt")],
        results=[
            ToolCallResultEvent(
                call_id="call-1",
                tool_name="file_write",
                error="File write requires approval.",
                metadata={
                    "resolved_path": str(target),
                    "requires_approval": True,
                    "approval_status": "pending",
                },
            )
        ],
    )

    assert result.success is False
    assert result.receipt.execution_status == "failed"
    assert result.receipt.retry_disposition == "requires_approval"


def test_delete_is_verified_from_the_absent_target(tmp_path: Path) -> None:
    target = tmp_path / "deleted.txt"
    request = ToolCallRequestEvent(
        call_id="call-delete",
        tool_name="file_delete",
        arguments={"path": "deleted.txt"},
    )
    tool_result = ToolCallResultEvent(
        call_id="call-delete",
        tool_name="file_delete",
        result="deleted",
        metadata={
            "file_changes": [
                {"path": str(target), "change_type": "delete"},
            ]
        },
    )

    result = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-delete",
        requests=[request],
        results=[tool_result],
    )

    assert result.success is True
    assert result.receipt.verification_status == "verified"
    assert result.receipt.targets[0].expected_exists is False
    assert result.receipt.targets[0].acceptance_checks[0].code == "target_absent"


def test_operation_id_is_deterministic_and_changes_with_normalized_arguments() -> None:
    verifier = ArtifactVerifier()
    requests = [_request("same.txt")]

    first = verifier.operation_id(turn_id="turn-a", requests=requests)
    duplicate = verifier.operation_id(turn_id="turn-a", requests=requests)
    reordered_arguments = verifier.operation_id(
        turn_id="turn-a",
        requests=[
            ToolCallRequestEvent(
                call_id="call-1",
                tool_name="file_write",
                arguments={"content": "verified content\n", "path": "same.txt"},
            )
        ],
    )
    other_turn = verifier.operation_id(turn_id="turn-b", requests=requests)
    changed_arguments = verifier.operation_id(
        turn_id="turn-a",
        requests=[_request("same.txt", content="different content")],
    )

    assert first == duplicate
    assert first == reordered_arguments
    assert first != other_turn
    assert first != changed_arguments


@pytest.mark.parametrize(
    ("tool_call_request", "expected_error"),
    [
        (
            ToolCallRequestEvent(
                call_id="call-targetless",
                tool_name="file_write",
                arguments={"content": "missing a target"},
            ),
            "No artifact targets",
        ),
        (
            ToolCallRequestEvent(
                call_id="call-malformed-patch",
                tool_name="apply_patch",
                arguments={"patch": "not an apply patch"},
            ),
            "Could not parse apply_patch targets",
        ),
    ],
)
def test_malformed_or_targetless_mutation_events_require_replan(
    tmp_path: Path,
    tool_call_request: ToolCallRequestEvent,
    expected_error: str,
) -> None:
    result = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-malformed",
        requests=[tool_call_request],
        results=[
            ToolCallResultEvent(
                call_id=tool_call_request.call_id,
                tool_name=tool_call_request.tool_name,
                result="claimed success",
            )
        ],
    )

    assert result.success is False
    assert result.receipt.verification_status == "not_run"
    assert result.receipt.retry_disposition == "requires_replan"
    assert any(expected_error in error for error in result.receipt.errors)


def test_retryable_failed_tool_result_requires_host_replan(tmp_path: Path) -> None:
    target = tmp_path / "retry.txt"

    result = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-retryable",
        requests=[_request("retry.txt")],
        results=[
            ToolCallResultEvent(
                call_id="call-1",
                tool_name="file_write",
                error="temporary filesystem error",
                metadata={"resolved_path": str(target), "retryable": True},
            )
        ],
    )

    assert result.success is False
    assert result.receipt.execution_status == "failed"
    assert result.receipt.retry_disposition == "requires_replan"


def test_unknown_execution_result_is_terminal_not_retryable(tmp_path: Path) -> None:
    result = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-unknown",
        requests=[_request("unknown.txt")],
        results=[],
    )

    assert result.success is False
    assert result.receipt.execution_status == "unknown"
    assert result.receipt.retry_disposition == "terminal"


def test_tool_reported_digest_is_checked_against_disk(tmp_path: Path) -> None:
    target = tmp_path / "claimed.txt"
    target.write_text("actual", encoding="utf-8")
    claimed = _sha256(b"claimed")

    result = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-untrusted-result",
        requests=[_request("claimed.txt", content="actual")],
        results=[
            _success(
                target,
                file_changes=[
                    {
                        "path": str(target),
                        "change_type": "add",
                        "after_sha256": claimed,
                    }
                ],
            )
        ],
    )

    assert result.success is False
    assert result.receipt.expected_after_hashes[str(target.resolve())] == claimed
    assert result.receipt.after_hashes[str(target.resolve())] == _sha256(b"actual")
    assert "digest_mismatch" in {
        check.code for check in result.receipt.targets[0].acceptance_checks
    }


def test_failed_result_reports_unexpected_sibling_change_fail_closed(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed.txt"
    sibling = tmp_path / "sibling.txt"
    allowed.write_text("allowed\n", encoding="utf-8")
    sibling.write_text("sibling\n", encoding="utf-8")

    result = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-failed-sibling",
        requests=[_request("allowed.txt", content="allowed\n")],
        results=[
            ToolCallResultEvent(
                call_id="call-1",
                tool_name="file_write",
                error="partial failure",
                metadata={"file_changes": [{"path": str(sibling)}]},
            )
        ],
    )

    assert result.success is False
    assert result.receipt.verification_status == "failed"
    assert result.receipt.errors.count("unexpected_changed_path") == 1
    assert result.receipt.scope_evidence.observed_paths_by_call == {
        "call-1": (str(sibling.resolve()),)
    }


def test_partial_result_keeps_legal_file_changes_for_recovery(tmp_path: Path) -> None:
    target = tmp_path / "partial.txt"
    target.write_text("written\n", encoding="utf-8")

    result = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-partial-legal",
        requests=[_request("partial.txt", content="written\n")],
        results=[
            ToolCallResultEvent(
                call_id="call-1",
                tool_name="file_write",
                error="later operation failed",
                metadata={"file_changes": [{"path": str(target)}]},
            )
        ],
    )

    assert result.success is False
    assert result.receipt.changed_paths == (str(target.resolve()),)
    assert result.receipt.scope_evidence.unexpected_changed_paths == ()
    assert result.receipt.retry_disposition == "requires_replan"


def test_duplicate_request_call_id_does_not_union_authorized_paths(tmp_path: Path) -> None:
    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_text("first\n", encoding="utf-8")
    second.write_text("second\n", encoding="utf-8")

    result = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-duplicate-call",
        requests=[
            _request("first.txt", call_id="same", content="first\n"),
            _request("second.txt", call_id="same", content="second\n"),
        ],
        results=[
            ToolCallResultEvent(
                call_id="same",
                tool_name="file_write",
                metadata={"resolved_path": str(second)},
            )
        ],
    )

    assert result.success is False
    assert "duplicate_request_call_id:same" in result.receipt.errors
    assert result.receipt.errors.count("unexpected_changed_path") == 1
    assert result.receipt.scope_evidence.authorized_paths_by_call == {
        "same": (str(first.resolve()),)
    }


def test_result_tool_mismatch_cannot_use_request_scope(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("content\n", encoding="utf-8")

    result = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-tool-mismatch",
        requests=[_request("target.txt", content="content\n")],
        results=[
            ToolCallResultEvent(
                call_id="call-1",
                tool_name="file_edit",
                metadata={"resolved_path": str(target)},
            )
        ],
    )

    assert result.success is False
    assert "result_tool_mismatch:call-1:file_write:file_edit" in result.receipt.errors
    assert result.receipt.errors.count("unexpected_changed_path") == 1


def test_receipt_reader_rejects_boolean_and_literal_coercion(tmp_path: Path) -> None:
    target = tmp_path / "report.md"
    target.write_text("content\n", encoding="utf-8")
    receipt = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-corrupt-reader",
        requests=[_request("report.md", content="content\n")],
        results=[_success(target)],
    ).receipt.to_dict()

    corrupted_payloads = []
    invalid_boolean = dict(receipt)
    invalid_boolean["targets"] = [dict(receipt["targets"][0], expected_exists="false")]
    corrupted_payloads.append(invalid_boolean)
    invalid_check = dict(receipt)
    invalid_check["targets"] = [
        dict(
            receipt["targets"][0],
            acceptance_checks=[
                dict(receipt["targets"][0]["acceptance_checks"][0], passed="false")
            ],
        )
    ]
    corrupted_payloads.append(invalid_check)
    invalid_status = dict(receipt, execution_status="completed")
    corrupted_payloads.append(invalid_status)

    for payload in corrupted_payloads:
        with pytest.raises(ValueError):
            ArtifactReceipt.from_dict(payload)


def test_natural_language_test_criterion_fails_closed_without_running_a_matcher(
    tmp_path: Path,
) -> None:
    target = tmp_path / "report.md"
    target.write_text("ready\n", encoding="utf-8")
    matcher_called = False

    def _never_match(_: str, __: object) -> bool:
        nonlocal matcher_called
        matcher_called = True
        return True

    result = ArtifactVerifier(
        validation_profiles=ValidationProfileRegistry({"project-test": _never_match})
    ).verify(
        workspace_root=tmp_path,
        turn_id="turn-natural-language",
        requests=[_request("report.md", content="ready\n")],
        results=[_success(target)],
        expectations=(
            ArtifactExpectation(
                path="report.md",
                acceptance_criteria=("tests pass",),
            ),
        ),
    )

    assert result.success is False
    assert matcher_called is False
    assert result.receipt.targets[0].acceptance_checks[-1].code == (
        "unsupported_acceptance_criterion"
    )


def test_profile_bound_pytest_evidence_completes_and_round_trips(tmp_path: Path) -> None:
    target = tmp_path / "report.md"
    target.write_text("ready\n", encoding="utf-8")
    test_request = _exec_request("python -m pytest -q tests/unit")

    result = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-pytest",
        requests=[_request("report.md", content="ready\n")],
        results=[_success(target)],
        evidence_requests=[_request("report.md", content="ready\n"), test_request],
        evidence_results=[_success(target), _exec_result()],
        expectations=(
            ArtifactExpectation(
                path="report.md",
                acceptance_criteria=(_tool_execution_criterion(),),
            ),
        ),
    )

    assert result.success is True
    check = result.receipt.targets[0].acceptance_checks[-1]
    assert check.code == "tool_execution_verified"
    assert check.evidence is not None
    assert check.evidence["call_id"] == "exec-call-1"
    assert check.evidence["operation_id"] == "exec-operation-1"
    assert check.evidence["turn_id"] == "turn-pytest"
    assert check.evidence["arguments_digest"] == tool_arguments_digest(
        tool_name="exec_command",
        arguments=test_request.arguments,
    )
    restored = ArtifactReceipt.from_dict(result.receipt.to_dict())
    assert restored.targets[0].acceptance_checks[-1].evidence == check.evidence


@pytest.mark.parametrize(
    ("command", "pins", "expected_code"),
    [
        ("echo 0", {}, "validation_profile_mismatch"),
        (
            "python -m pytest -q tests/unit",
            {"arguments_digest": "0" * 64},
            "tool_execution_evidence_missing",
        ),
        (
            "python -m pytest -q tests/unit",
            {"call_id": "other-call"},
            "tool_execution_evidence_missing",
        ),
        (
            "python -m pytest -q tests/unit",
            {"turn_id": "other-turn"},
            "tool_execution_evidence_missing",
        ),
    ],
    ids=["wrong-command", "wrong-digest", "wrong-call", "wrong-turn"],
)
def test_tool_execution_criterion_rejects_non_exact_or_non_profile_evidence(
    tmp_path: Path,
    command: str,
    pins: dict[str, str],
    expected_code: str,
) -> None:
    target = tmp_path / "report.md"
    target.write_text("ready\n", encoding="utf-8")
    test_request = _exec_request(command)
    result = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-exact-binding",
        requests=[_request("report.md", content="ready\n")],
        results=[_success(target)],
        evidence_requests=[_request("report.md", content="ready\n"), test_request],
        evidence_results=[_success(target), _exec_result()],
        expectations=(
            ArtifactExpectation(
                path="report.md",
                acceptance_criteria=(_tool_execution_criterion(**pins),),
            ),
        ),
    )

    assert result.success is False
    assert result.receipt.retry_disposition == "requires_replan"
    assert result.receipt.targets[0].acceptance_checks[-1].code == expected_code


@pytest.mark.parametrize(
    ("profile_id", "check", "command"),
    [
        ("pytest", "test", "pytest -q\necho laundered"),
        ("pytest", "test", "pytest -q & echo laundered"),
        ("pytest", "test", "pytest -q > proof.txt"),
        ("pytest", "test", 'powershell -Command "pytest -q"'),
        ("pytest", "test", 'cmd /c "pytest -q"'),
        ("ruff", "lint", 'bash -c "ruff check ."'),
    ],
    ids=[
        "newline",
        "ampersand",
        "redirection",
        "powershell-wrapper",
        "cmd-wrapper",
        "bash-wrapper",
    ],
)
def test_default_validation_profiles_reject_shell_laundering(
    tmp_path: Path,
    profile_id: str,
    check: str,
    command: str,
) -> None:
    target = tmp_path / "report.md"
    target.write_text("ready\n", encoding="utf-8")
    test_request = _exec_request(command)

    result = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-shell-laundering",
        requests=[_request("report.md", content="ready\n")],
        results=[_success(target)],
        evidence_requests=[_request("report.md", content="ready\n"), test_request],
        evidence_results=[_success(target), _exec_result()],
        expectations=(
            ArtifactExpectation(
                path="report.md",
                acceptance_criteria=(
                    _tool_execution_criterion(profile_id=profile_id, check=check),
                ),
            ),
        ),
    )

    assert result.success is False
    assert result.receipt.targets[0].acceptance_checks[-1].code == (
        "validation_profile_mismatch"
    )


def test_tool_execution_criterion_waiting_for_approval_requires_approval(
    tmp_path: Path,
) -> None:
    target = tmp_path / "report.md"
    target.write_text("ready\n", encoding="utf-8")
    test_request = _exec_request("pytest -q")
    pending = _exec_result(
        error="Exec command requires approval.",
        metadata={
            "status": "approval_pending",
            "requires_approval": True,
            "approval_status": "pending",
        },
    )

    result = ArtifactVerifier().verify(
        workspace_root=tmp_path,
        turn_id="turn-approval-pending",
        requests=[_request("report.md", content="ready\n")],
        results=[_success(target)],
        evidence_requests=[_request("report.md", content="ready\n"), test_request],
        evidence_results=[_success(target), pending],
        expectations=(
            ArtifactExpectation(
                path="report.md",
                acceptance_criteria=(_tool_execution_criterion(),),
            ),
        ),
    )

    assert result.success is False
    assert result.receipt.retry_disposition == "requires_approval"
    assert result.receipt.targets[0].acceptance_checks[-1].code == (
        "tool_execution_requires_approval"
    )


def test_host_can_register_a_project_validation_profile(tmp_path: Path) -> None:
    target = tmp_path / "report.md"
    target.write_text("ready\n", encoding="utf-8")
    test_request = _exec_request("project-check --verify")
    registry = ValidationProfileRegistry(
        {
            "project-check": lambda tool_name, arguments: (
                tool_name == "exec_command"
                and arguments.get("command") == "project-check --verify"
            ),
        }
    )

    result = ArtifactVerifier(validation_profiles=registry).verify(
        workspace_root=tmp_path,
        turn_id="turn-custom-profile",
        requests=[_request("report.md", content="ready\n")],
        results=[_success(target)],
        evidence_requests=[_request("report.md", content="ready\n"), test_request],
        evidence_results=[_success(target), _exec_result()],
        expectations=(
            ArtifactExpectation(
                path="report.md",
                acceptance_criteria=(
                    _tool_execution_criterion(
                        profile_id="project-check",
                        check="test",
                    ),
                ),
            ),
        ),
    )

    assert result.success is True
