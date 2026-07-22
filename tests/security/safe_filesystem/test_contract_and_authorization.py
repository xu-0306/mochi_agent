from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    pass

from tests.security.safe_filesystem._support import (
    _authorization,
    _entry,
    _exec_authorization,
    _manifest,
    _sha,
)


def test_contract_roundtrips_complete_file_request_and_identity() -> None:
    from mochi.security.file_contract import AuthorizationEnvelope, FileIdentity

    envelope = _authorization()
    assert AuthorizationEnvelope.from_dict(envelope.to_dict()) == envelope
    identity = FileIdentity("windows", "volume", "file", 1, True)
    assert FileIdentity.from_dict(identity.to_dict()) == identity
    assert identity.is_reparse_point is True



def test_file_request_digest_is_deterministic_and_binds_authorization() -> None:
    from mochi.security.file_contract import (
        FileChangeRequest,
        FileIdentity,
        authorization_request_digest,
    )

    envelope = _authorization()
    reversed_request = FileChangeRequest(
        entries=tuple(reversed(envelope.file_request.entries)),
        patch_sha256=envelope.file_request.patch_sha256,
    )
    assert authorization_request_digest(envelope) == authorization_request_digest(
        replace(envelope, file_request=reversed_request)
    )

    changed = [
        replace(envelope, context=replace(envelope.context, requester_id="requester-2")),
        replace(envelope, context=replace(envelope.context, session_id="session-2")),
        replace(envelope, context=replace(envelope.context, task_id=None)),
        replace(envelope, context=replace(envelope.context, workspace_root="other")),
        replace(envelope, context=replace(envelope.context, workspace_identity=FileIdentity("posix", "10", "21", 1, False))),
        replace(envelope, policy_version="policy-v2"),
        replace(
            envelope,
            file_request=replace(envelope.file_request, patch_sha256=_sha("other-patch")),
        ),
    ]
    original = authorization_request_digest(envelope)
    assert all(authorization_request_digest(item) != original for item in changed)



def test_envelope_requires_exactly_one_matching_request() -> None:
    from mochi.security.file_contract import (
        EnvVarHash,
        ExecRequest,
        ResourceLimits,
    )

    envelope = _authorization()
    exec_request = ExecRequest(
        command_utf8_sha256=_sha("command"),
        shell="powershell",
        executable="pwsh.exe",
        argv=("-NoProfile", "-Command", "Get-Date"),
        resolved_cwd="workspace",
        env=(EnvVarHash("PATH", _sha("path-value")),),
        network_policy="deny",
        resource_limits=ResourceLimits(30, 1024, 4096),
        requested_escalation="none",
        sandbox_backend="windows-native",
        sandbox_capability_plan_digest=_sha("sandbox-plan"),
    )
    with pytest.raises(ValueError, match="exactly one"):
        replace(envelope, exec_request=exec_request)
    with pytest.raises(ValueError, match="kind"):
        replace(envelope, kind="exec")



def test_contract_rejects_unknown_fields_at_every_model_layer() -> None:
    from mochi.security.file_contract import (
        ChangeManifest,
        EnvVarHash,
        ExecRequest,
        FileIdentity,
        ResourceLimits,
    )

    objects = [
        FileIdentity("posix", "1", "2", 1, False),
        _authorization().context,
        _entry(),
        _authorization().file_request,
        EnvVarHash("PATH", _sha("hash")),
        ResourceLimits(30, 1024, 4096),
        ExecRequest(
            _sha("command"), None, "tool", (), "workspace", (), "deny",
            ResourceLimits(30, 1024, 4096), "none", "native", _sha("plan")
        ),
        _authorization(),
        ChangeManifest(
            version=1,
            change_set_id="change-set",
            workspace_root="workspace",
            workspace_identity=FileIdentity("posix", "10", "20", 1, False),
            tool_name="write_file",
            intent="mutate",
            entries=(_entry(),),
            patch_sha256=_sha("patch"),
            policy_version="policy-v1",
            created_at="2026-01-01T00:00:00Z",
            expires_at="2026-01-01T01:00:00Z",
            request_digest=_sha("request"),
            ui_metadata={"label": "preview"},
        ),
    ]
    for obj in objects:
        payload = obj.to_dict()
        payload["unknown"] = "fail-closed"
        with pytest.raises(ValueError, match="unknown"):
            type(obj).from_dict(payload)



def test_canonical_json_rejects_ambiguous_non_json_types() -> None:
    from mochi.security.file_contract import canonical_json

    values = (1.25, float("nan"), Path("relative"), datetime(2026, 1, 1), {1: "x"})
    for value in values:
        with pytest.raises(TypeError):
            canonical_json({"value": value})



def test_manifest_projection_only_excludes_explicit_top_level_volatile_fields() -> None:
    from mochi.security.file_contract import (
        ChangeManifest,
        FileIdentity,
        manifest_digest_projection,
    )

    manifest = ChangeManifest(
        version=1,
        change_set_id="volatile",
        workspace_root="workspace",
        workspace_identity=FileIdentity("posix", "10", "20", 1, False),
        tool_name="write_file",
        intent="mutate",
        entries=(_entry(),),
        patch_sha256=_sha("patch"),
        policy_version="policy-v1",
        created_at="created",
        expires_at="expires",
        request_digest=_sha("request"),
        ui_metadata={"created_at": "nested-security-relevant"},
    )
    projection = manifest_digest_projection(manifest)

    assert "change_set_id" not in projection
    assert "created_at" not in projection
    assert "expires_at" not in projection
    assert "request_digest" not in projection
    assert "ui_metadata" not in projection
    assert manifest.ui_metadata["created_at"] == "nested-security-relevant"



def test_preview_idempotency_key_is_context_scoped_composite() -> None:
    from mochi.security.file_contract import preview_idempotency_key

    envelope = _authorization()
    key = preview_idempotency_key(envelope)
    assert key == preview_idempotency_key(envelope)
    assert key != preview_idempotency_key(
        replace(envelope, context=replace(envelope.context, session_id="different"))
    )
    assert key != preview_idempotency_key(replace(envelope, policy_version="different"))
    assert key != preview_idempotency_key(
        replace(envelope, file_request=replace(envelope.file_request, patch_sha256=_sha("different")))
    )



def test_authorization_envelope_supports_current_and_migratable_schema_versions() -> None:
    envelope = _authorization()

    assert replace(envelope, schema_version=1).schema_version == 1
    with pytest.raises(ValueError, match="schema_version"):
        replace(envelope, schema_version=3)



def test_file_request_and_manifest_reject_duplicate_entry_path_keys() -> None:
    from mochi.security.file_contract import FileChangeRequest

    duplicate = replace(_entry(), operation="delete")
    with pytest.raises(ValueError, match="duplicate"):
        FileChangeRequest(
            entries=(_entry(), duplicate),
            patch_sha256=_sha("patch"),
        )
    with pytest.raises(ValueError, match="duplicate"):
        _manifest(entries=(_entry(), duplicate))



def test_all_explicit_digest_fields_require_lowercase_sha256_hex() -> None:
    from mochi.security.file_contract import (
        EnvVarHash,
        ExecRequest,
        FileChangeRequest,
        ResourceLimits,
    )

    exec_request = ExecRequest(
        command_utf8_sha256=_sha("command"),
        shell=None,
        executable="tool",
        argv=(),
        resolved_cwd="workspace",
        env=(EnvVarHash("PATH", _sha("env")),),
        network_policy="deny",
        resource_limits=ResourceLimits(30, 1024, 4096),
        requested_escalation="none",
        sandbox_backend="native",
        sandbox_capability_plan_digest=_sha("sandbox-plan"),
    )
    manifest = _manifest()
    cases: tuple[Callable[[str], object], ...] = (
        lambda value: replace(_entry(), base_sha256=value),
        lambda value: replace(_entry(), after_sha256=value),
        lambda value: replace(_entry(), base_metadata_sha256=value),
        lambda value: replace(_entry(), after_metadata_sha256=value),
        lambda value: FileChangeRequest(
            entries=(_entry(),), patch_sha256=value
        ),
        lambda value: EnvVarHash("PATH", value),
        lambda value: replace(exec_request, command_utf8_sha256=value),
        lambda value: replace(
            exec_request, sandbox_capability_plan_digest=value
        ),
        lambda value: replace(manifest, patch_sha256=value),
        lambda value: replace(manifest, request_digest=value),
    )
    invalid_digests = ("A" * 64, "g" * 64, "a" * 63)

    for invalid in invalid_digests:
        for construct in cases:
            with pytest.raises(ValueError, match="lowercase hexadecimal"):
                construct(invalid)



def test_manifest_projection_fully_validates_raw_mappings() -> None:
    from mochi.security.file_contract import manifest_digest_projection

    payload = _manifest().to_dict()
    payload["patch_sha256"] = "not-a-digest"

    with pytest.raises(ValueError, match="lowercase hexadecimal"):
        manifest_digest_projection(payload)



def test_authorized_file_binding_captures_all_content_and_metadata_hashes() -> None:
    from mochi.security.safe_filesystem import (
        resolve_authorized_file_binding,
    )

    authorization = _authorization()
    entry = authorization.file_request.entries[0]

    binding = resolve_authorized_file_binding(
        canonical_relative_path="safe/note.txt",
        authorization=authorization,
        captured_identity=entry.base_identity,
        canonicalize_authorized_path=lambda path: path,
    )

    assert binding.base_sha256 == entry.base_sha256
    assert binding.after_sha256 == entry.after_sha256
    assert binding.base_metadata_sha256 == entry.base_metadata_sha256
    assert binding.after_metadata_sha256 == entry.after_metadata_sha256



def test_authorized_file_binding_rejects_tampered_or_alternate_entry_substitution() -> None:
    from mochi.security.file_contract import FileChangeRequest
    from mochi.security.safe_filesystem import (
        resolve_authorized_file_binding,
    )

    original = _authorization()
    first_entry = original.file_request.entries[0]
    alternate_entry = replace(
        first_entry,
        entry_id="0002",
        relative_path="safe/other.txt",
        base_sha256=_sha("alternate-base"),
        after_sha256=_sha("alternate-after"),
        base_metadata_sha256=_sha("alternate-base-metadata"),
        after_metadata_sha256=_sha("alternate-after-metadata"),
    )
    authorization = replace(
        original,
        file_request=FileChangeRequest(
            entries=(first_entry, alternate_entry),
            patch_sha256=original.file_request.patch_sha256,
        ),
    )

    def resolve(path: str, envelope=authorization):
        return resolve_authorized_file_binding(
            canonical_relative_path=path,
            authorization=envelope,
            captured_identity=first_entry.base_identity,
            canonicalize_authorized_path=lambda value: value,
        )

    expected = resolve("safe/note.txt")
    alternate = resolve("safe/other.txt")
    tampered_entry = replace(
        first_entry,
        after_sha256=_sha("tampered-after"),
        after_metadata_sha256=_sha("tampered-after-metadata"),
    )
    tampered_authorization = replace(
        original,
        file_request=FileChangeRequest(
            entries=(tampered_entry,),
            patch_sha256=original.file_request.patch_sha256,
        ),
    )
    tampered = resolve("safe/note.txt", tampered_authorization)

    assert alternate != expected
    assert tampered != expected
    assert alternate.after_sha256 == alternate_entry.after_sha256
    assert tampered.after_sha256 == tampered_entry.after_sha256
    assert tampered.after_metadata_sha256 == (
        tampered_entry.after_metadata_sha256
    )
    assert tampered.authorization_digest != expected.authorization_digest
    hash_only_alternate = replace(
        alternate,
        entry_id=expected.entry_id,
        canonical_relative_path=expected.canonical_relative_path,
    )
    assert hash_only_alternate != expected



def test_exec_digest_binds_every_authorization_and_execution_field() -> None:
    from mochi.security.file_contract import (
        EnvVarHash,
        FileIdentity,
        ResourceLimits,
        authorization_request_digest,
    )

    envelope = _exec_authorization()
    request = envelope.exec_request
    assert request is not None
    changed = [
        replace(
            envelope,
            context=replace(
                envelope.context, requester_id="requester-2"
            ),
        ),
        replace(
            envelope,
            context=replace(
                envelope.context, session_id="session-2"
            ),
        ),
        replace(
            envelope,
            context=replace(envelope.context, task_id=None),
        ),
        replace(
            envelope,
            context=replace(
                envelope.context, workspace_root="C:/other"
            ),
        ),
        replace(
            envelope,
            context=replace(
                envelope.context,
                workspace_identity=FileIdentity(
                    "windows", "10", "21", 1, False
                ),
            ),
        ),
        replace(envelope, policy_version="policy-v2"),
        replace(
            envelope,
            exec_request=replace(
                request, command_utf8_sha256=_sha("other-command")
            ),
        ),
        replace(
            envelope,
            exec_request=replace(request, shell=None),
        ),
        replace(
            envelope,
            exec_request=replace(request, executable="other.exe"),
        ),
        replace(
            envelope,
            exec_request=replace(request, argv=("changed",)),
        ),
        replace(
            envelope,
            exec_request=replace(
                request, resolved_cwd="C:/workspace/other"
            ),
        ),
        replace(
            envelope,
            exec_request=replace(
                request,
                env=(EnvVarHash("PATH", _sha("other-value")),),
            ),
        ),
        replace(
            envelope,
            exec_request=replace(request, network_policy="allow"),
        ),
        replace(
            envelope,
            exec_request=replace(
                request,
                resource_limits=ResourceLimits(31, 1024, 4096),
            ),
        ),
        replace(
            envelope,
            exec_request=replace(
                request, requested_escalation="require_escalated"
            ),
        ),
        replace(
            envelope,
            exec_request=replace(
                request, sandbox_backend="other-backend"
            ),
        ),
        replace(
            envelope,
            exec_request=replace(
                request,
                sandbox_capability_plan_digest=_sha("other-plan"),
            ),
        ),
    ]
    original = authorization_request_digest(envelope)

    assert all(
        authorization_request_digest(item) != original
        for item in changed
    )


def test_change_manifest_ui_metadata_is_immutable() -> None:
    from mochi.security.file_contract import ChangeManifest

    manifest = ChangeManifest(
        version=1,
        change_set_id="c",
        workspace_root="workspace",
        workspace_identity=_authorization().context.workspace_identity,
        tool_name="write_file",
        intent="mutate",
        entries=(_entry(),),
        patch_sha256=_sha("patch"),
        policy_version="p",
        created_at="created",
        expires_at="expires",
        request_digest=_sha("request"),
        ui_metadata={"label": "preview"},
    )

    with pytest.raises(TypeError):
        manifest.ui_metadata["label"] = "changed"
