from __future__ import annotations

import hashlib
from dataclasses import replace

import pytest

from mochi.security.auto_review import (
    AUTO_REVIEWER_VERSION,
    AutoReviewFacts,
    AutoReviewVerificationError,
    review_authorization_envelope,
    verify_auto_review_decision,
)
from mochi.security.file_contract import (
    AuthorizationContext,
    AuthorizationEnvelope,
    ChangeEntry,
    EnvVarHash,
    ExecRequest,
    FileChangeRequest,
    FileIdentity,
    ResourceLimits,
    authorization_request_digest,
)


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _identity(file_id: str = "workspace-1") -> FileIdentity:
    return FileIdentity("windows", "volume-1", file_id, 1, False)


def _context() -> AuthorizationContext:
    return AuthorizationContext(
        requester_id="runtime-task:task-1",
        session_id="session-1",
        task_id="task-1",
        workspace_root="C:/workspace",
        workspace_identity=_identity(),
    )


def _exec_envelope(
    *,
    escalation: str = "use_default",
    network: str = "deny",
    env: tuple[EnvVarHash, ...] = (),
) -> AuthorizationEnvelope:
    return AuthorizationEnvelope(
        schema_version=1,
        kind="exec",
        context=_context(),
        policy_version="exec-policy-v1:test",
        file_request=None,
        exec_request=ExecRequest(
            command_utf8_sha256=_sha("tool --version"),
            shell="powershell",
            executable="tool",
            argv=("--version",),
            resolved_cwd="C:/workspace",
            env=env,
            network_policy=network,  # type: ignore[arg-type]
            resource_limits=ResourceLimits(30, 0, 1_048_576),
            requested_escalation=escalation,
            sandbox_backend="host",
            sandbox_capability_plan_digest=_sha("host-plan"),
        ),
    )


def _file_envelope(path: str = "src/app.py") -> AuthorizationEnvelope:
    entry = ChangeEntry(
        entry_id=_sha(path),
        relative_path=path,
        operation="update",
        base_sha256=_sha("before"),
        after_sha256=_sha("after"),
        base_identity=_identity("file-1"),
        before_blob_id="before-blob",
        after_blob_id="after-blob",
        mode_before=0o644,
        mode_after=0o644,
        base_metadata_sha256=None,
        after_metadata_sha256=None,
        rename_source=None,
        dependency_group=None,
    )
    return AuthorizationEnvelope(
        schema_version=1,
        kind="file_change",
        context=_context(),
        policy_version="file-policy-v1:test",
        file_request=FileChangeRequest(entries=(entry,), patch_sha256=_sha("patch")),
        exec_request=None,
    )


def _ask_facts(**overrides: bool) -> AutoReviewFacts:
    return AutoReviewFacts(
        policy_action="ask",
        policy_rule_id="unknown_requires_approval",
        **overrides,
    )


def test_reviewed_allow_is_deterministic_and_digest_bound() -> None:
    envelope = _exec_envelope()

    first = review_authorization_envelope(envelope, facts=_ask_facts())
    second = review_authorization_envelope(envelope, facts=_ask_facts())

    assert first == second
    assert first.decision == "allow"
    assert first.input_digest == authorization_request_digest(envelope)
    assert first.input_digest == envelope.request_digest
    assert first.policy_version == envelope.policy_version
    assert first.reviewer_version == AUTO_REVIEWER_VERSION
    assert first.risk_factors == ()
    assert first.reason_codes == ("reviewed_allow",)


@pytest.mark.parametrize(
    ("envelope", "facts", "risk"),
    [
        (_file_envelope(".git/config"), _ask_facts(), "protected_path"),
        (_file_envelope("../outside.txt"), _ask_facts(), "workspace_escape"),
        (_file_envelope("C:/outside.txt"), _ask_facts(), "workspace_escape"),
        (_exec_envelope(escalation="require_escalated"), _ask_facts(), "require_escalated"),
        (_exec_envelope(), _ask_facts(unknown_shell_parse=True), "unknown_shell_parse"),
        (_exec_envelope(), _ask_facts(identity_mismatch=True), "identity_mismatch"),
        (_file_envelope(), _ask_facts(stale_base=True), "stale_base"),
        (
            _exec_envelope(
                network="allow",
                env=(EnvVarHash("API_TOKEN", _sha("secret")),),
            ),
            _ask_facts(),
            "network_credential_exposure",
        ),
    ],
)
def test_fail_closed_risks_never_auto_allow(
    envelope: AuthorizationEnvelope,
    facts: AutoReviewFacts,
    risk: str,
) -> None:
    decision = review_authorization_envelope(envelope, facts=facts)

    assert decision.decision != "allow"
    assert risk in decision.risk_factors
    assert risk in decision.reason_codes


def test_policy_allow_is_distinguished_from_reviewed_allow() -> None:
    decision = review_authorization_envelope(
        _exec_envelope(),
        facts=AutoReviewFacts(policy_action="allow", policy_rule_id="read_only"),
    )

    assert decision.decision == "allow"
    assert decision.reason_codes == ("policy_auto_allow",)


def test_execution_rejects_digest_or_workspace_identity_mismatch() -> None:
    envelope = _exec_envelope()
    decision = review_authorization_envelope(envelope, facts=_ask_facts())
    changed_request = replace(envelope.exec_request, argv=("--help",))
    changed_envelope = replace(envelope, exec_request=changed_request)

    with pytest.raises(AutoReviewVerificationError, match="digest changed"):
        verify_auto_review_decision(decision, changed_envelope)
    with pytest.raises(AutoReviewVerificationError, match="identity changed"):
        verify_auto_review_decision(
            decision,
            envelope,
            current_workspace_identity=_identity("workspace-2"),
        )


def test_decision_model_rejects_unknown_output_fields() -> None:
    decision = review_authorization_envelope(_exec_envelope(), facts=_ask_facts())
    payload = decision.model_dump(mode="python")
    payload["explanation"] = "mutable prose must not enter the contract"

    with pytest.raises(ValueError):
        type(decision).model_validate(payload)
