from __future__ import annotations

import hashlib
import inspect
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from typing import TYPE_CHECKING, get_type_hints

import pytest

if TYPE_CHECKING:
    from mochi.security.file_contract import FileIdentity
    from mochi.security.safe_filesystem import (
        AuthorizedFileBinding,
        SafeTarget,
        StagedTemp,
    )
    from mochi.tools.file_transaction import FileMetadataSnapshot


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_label(label: str) -> str:
    return _sha_bytes(label.encode("utf-8"))


_BASE_BYTES = b"original bytes"
_AFTER_BYTES = b"replacement bytes"


class _EqualBindingImpostor:
    def __init__(self, payload: object) -> None:
        self.payload = payload

    def __eq__(self, other: object) -> bool:
        return True


class _UnsafeCleanupError(RuntimeError):
    def __str__(self) -> str:
        raise RuntimeError("cleanup formatting failed")


class _UnsafeAddNoteError(OSError):
    def add_note(self, note: str) -> None:
        raise RuntimeError("overridden add_note failed")


def _binding() -> AuthorizedFileBinding:
    from mochi.security.file_contract import FileIdentity
    from mochi.security.safe_filesystem import AuthorizedFileBinding

    metadata_sha256 = _sha_label("preserved-security-metadata")
    return AuthorizedFileBinding(
        entry_id="0001",
        canonical_relative_path="safe/note.txt",
        operation="update",
        base_identity=FileIdentity("posix", "1", "41", 1, False),
        base_sha256=_sha_bytes(_BASE_BYTES),
        after_sha256=_sha_bytes(_AFTER_BYTES),
        base_metadata_sha256=metadata_sha256,
        after_metadata_sha256=metadata_sha256,
        authorization_digest=_sha_label("authorization"),
    )


def _snapshot(binding: AuthorizedFileBinding) -> FileMetadataSnapshot:
    from mochi.tools.file_transaction import FileMetadataSnapshot

    return FileMetadataSnapshot(
        kind="existing_file",
        identity=binding.base_identity,
        binding=binding,
        canonical_metadata_sha256=binding.after_metadata_sha256,
    )


class _BehavioralTransactionOwner:
    """In-memory capability owner with complete transaction side effects."""

    def __init__(
        self,
        *,
        write_plan: list[int | BaseException] | None = None,
        failures: dict[str, BaseException] | None = None,
    ) -> None:
        from mochi.security.file_contract import FileIdentity
        from mochi.security.safe_filesystem import SafeTarget

        self.binding = _binding()
        self.events: list[str] = []
        self.binding_calls = 0
        self.write_plan = list(write_plan or [])
        self.failures = dict(failures or {})
        self.transaction_binding_result: object = self.binding
        self.temp_tamper: str | None = None
        self.replace_consumption = "both"
        self.committed_replace_failure: BaseException | None = None
        self.discard_attempts = 0
        self.original_bytes = _BASE_BYTES
        self.original_metadata_sha256 = self.binding.base_metadata_sha256
        self.original_identity = self.binding.base_identity
        self.original_live = True
        self.target_resource_open = True
        self.temp_resource_open = False
        self.temp_bytes = bytearray()
        self.temp_metadata_sha256: str | None = None
        self.temp_live = False
        self.temp_flushed = False
        self.issued_temp: StagedTemp | None = None
        self.discarded_temp: StagedTemp | None = None
        self.successor_identity = FileIdentity("posix", "1", "99", 1, False)
        self.replace_result: object = self.successor_identity
        self.temp_parent = object()
        self.target = SafeTarget._create(
            basename="note.txt",
            identity=self.binding.base_identity,
            authorization_digest=self.binding.authorization_digest,
            owner=self,
            parent=object(),
        )

    def _fail(self, phase: str) -> None:
        failure = self.failures.get(phase)
        if failure is not None:
            raise failure

    def transaction_binding(
        self, target: SafeTarget
    ) -> AuthorizedFileBinding:
        self.binding_calls += 1
        assert target is self.target and not target.closed
        return self.transaction_binding_result

    def create_temp(self, target: SafeTarget) -> StagedTemp:
        from mochi.security.safe_filesystem import StagedTemp

        assert target is self.target and not self.temp_live
        self.events.append("create_temp")
        temp = StagedTemp._create(
            basename=".mochi-note.tmp",
            identity=self.successor_identity,
            binding=self.binding,
            owner=self,
            parent=self.temp_parent,
        )
        self.issued_temp = temp
        self.temp_live = self.temp_resource_open = True
        if self.temp_tamper == "wrong_type":
            return object()
        if self.temp_tamper == "closed":
            temp._mark_closed()
        elif self.temp_tamper == "authenticity":
            object.__setattr__(temp, "_seal", object())
        elif self.temp_tamper == "foreign_owner":
            object.__setattr__(temp, "_owner", object())
        elif self.temp_tamper == "binding":
            object.__setattr__(
                temp,
                "_binding",
                replace(self.binding, entry_id="different-entry"),
            )
        elif self.temp_tamper == "authorization_digest":
            object.__setattr__(
                temp, "_authorization_digest", _sha_label("different-auth")
            )
        return temp

    def write_temp(self, temp: StagedTemp, data: memoryview) -> int:
        assert temp is self.issued_temp
        assert self.temp_live and self.temp_resource_open
        self.events.append("write")
        action: int | BaseException = (
            self.write_plan.pop(0) if self.write_plan else len(data)
        )
        if isinstance(action, BaseException):
            raise action
        written = min(action, len(data))
        self.temp_bytes.extend(data[:written])
        return written

    def apply_metadata_snapshot(
        self, temp: StagedTemp, snapshot: FileMetadataSnapshot
    ) -> None:
        assert temp is self.issued_temp
        assert bytes(self.temp_bytes) == _AFTER_BYTES
        self.events.append("apply_snapshot")
        self._fail("apply_snapshot")
        self.temp_metadata_sha256 = snapshot.canonical_metadata_sha256

    def verify_staged(
        self, temp: StagedTemp, snapshot: FileMetadataSnapshot
    ) -> None:
        assert temp is self.issued_temp
        self.events.append("verify_staged")
        self._fail("verify_staged")
        assert _sha_bytes(bytes(self.temp_bytes)) == self.binding.after_sha256
        assert (
            self.temp_metadata_sha256
            == snapshot.canonical_metadata_sha256
        )

    def flush_temp(self, temp: StagedTemp) -> None:
        assert temp is self.issued_temp
        self.events.append("flush_temp")
        self._fail("flush_temp")
        self.temp_flushed = True

    def revalidate_base(
        self, target: SafeTarget, snapshot: FileMetadataSnapshot
    ) -> None:
        assert target is self.target and self.temp_flushed
        self.events.append("revalidate_base")
        self._fail("revalidate_base")
        assert self.original_identity == self.binding.base_identity
        assert _sha_bytes(self.original_bytes) == self.binding.base_sha256
        assert (
            self.original_metadata_sha256
            == self.binding.base_metadata_sha256
        )
        assert snapshot.binding == self.binding

    def replace(
        self, source: StagedTemp, destination: SafeTarget
    ) -> FileIdentity:
        assert source is self.issued_temp and destination is self.target
        assert self.temp_flushed
        self.events.append("replace")
        self._fail("replace")
        self.original_bytes = bytes(self.temp_bytes)
        self.original_metadata_sha256 = self.temp_metadata_sha256
        self.original_identity = self.successor_identity
        self.temp_live = self.temp_resource_open = False
        self.target_resource_open = False
        if self.replace_consumption != "source_open":
            source._mark_closed()
        if self.replace_consumption != "destination_open":
            destination._mark_closed()
        if self.committed_replace_failure is not None:
            raise self.committed_replace_failure
        return self.replace_result

    def discard_temp(self, temp: StagedTemp) -> None:
        self.discard_attempts += 1
        assert temp is self.issued_temp and self.temp_live
        self.events.append("discard_temp")
        self.discarded_temp = temp
        self.temp_live = self.temp_resource_open = False
        temp._mark_closed()
        self._fail("discard_temp")

    def release_temp(self, temp: StagedTemp) -> None:
        assert temp is self.issued_temp
        self.temp_resource_open = False
        temp._mark_closed()

    def release_target(self, target: SafeTarget) -> None:
        assert target is self.target
        self.target_resource_open = False
        target._mark_closed()


def _transaction(**options):
    owner = _BehavioralTransactionOwner(**options)
    return owner, owner.target, _snapshot(owner.binding)


def test_snapshot_is_frozen_discriminated_and_digest_validated() -> None:
    owner, _, snapshot = _transaction()

    assert snapshot.kind == "existing_file"
    assert snapshot.identity == owner.binding.base_identity
    assert snapshot.binding == owner.binding
    assert (
        snapshot.canonical_metadata_sha256
        == owner.binding.after_metadata_sha256
    )
    with pytest.raises(FrozenInstanceError):
        snapshot.kind = "other"
    with pytest.raises(ValueError, match="kind"):
        replace(snapshot, kind="add")
    with pytest.raises(ValueError, match="lowercase hexadecimal"):
        replace(snapshot, canonical_metadata_sha256="not-a-digest")


def test_after_bytes_hash_mismatch_rejects_before_creating_temp() -> None:
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget
    from mochi.tools.file_transaction import atomic_write_bytes

    owner, target, snapshot = _transaction()
    with pytest.raises(UnsafeFilesystemTarget, match="after_sha256"):
        atomic_write_bytes(target, b"unauthorized bytes", snapshot)

    assert owner.events == []
    assert owner.issued_temp is None
    assert owner.original_live and owner.original_bytes == _BASE_BYTES
    assert not target.closed


@pytest.mark.parametrize(
    "tamper",
    ["closed", "authenticity", "authorization_digest", "identity"],
)
def test_invalid_target_capability_rejects_before_owner_operation(
    tamper: str,
) -> None:
    from mochi.security.file_contract import FileIdentity
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget
    from mochi.tools.file_transaction import atomic_write_bytes

    owner, target, snapshot = _transaction()
    if tamper == "closed":
        target._mark_closed()
    elif tamper == "authenticity":
        object.__setattr__(target, "_seal", object())
    elif tamper == "authorization_digest":
        object.__setattr__(
            target, "_authorization_digest", _sha_label("other-authorization")
        )
    else:
        object.__setattr__(
            target,
            "_identity",
            FileIdentity("posix", "1", "42", 1, False),
        )

    with pytest.raises(UnsafeFilesystemTarget, match="target capability"):
        atomic_write_bytes(target, _AFTER_BYTES, snapshot)

    assert owner.binding_calls == 0
    assert owner.events == []
    assert owner.issued_temp is None


@pytest.mark.parametrize(
    "tamper",
    [
        "identity",
        "entry",
        "base_content",
        "after_content",
        "base_metadata",
        "after_metadata",
        "canonical_metadata",
    ],
)
def test_snapshot_mismatch_rejects_before_creating_temp(tamper: str) -> None:
    from mochi.security.file_contract import FileIdentity
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget
    from mochi.tools.file_transaction import atomic_write_bytes

    owner, target, snapshot = _transaction()
    if tamper == "identity":
        snapshot = replace(
            snapshot,
            identity=FileIdentity("posix", "1", "42", 1, False),
        )
    elif tamper == "canonical_metadata":
        snapshot = replace(
            snapshot,
            canonical_metadata_sha256=_sha_label("wrong-metadata"),
        )
    else:
        binding_field = {
            "entry": "entry_id",
            "base_content": "base_sha256",
            "after_content": "after_sha256",
            "base_metadata": "base_metadata_sha256",
            "after_metadata": "after_metadata_sha256",
        }[tamper]
        value = (
            "alternate-entry"
            if binding_field == "entry_id"
            else _sha_label(f"wrong-{tamper}")
        )
        snapshot = replace(
            snapshot,
            binding=replace(snapshot.binding, **{binding_field: value}),
        )

    with pytest.raises(UnsafeFilesystemTarget, match="metadata snapshot"):
        atomic_write_bytes(target, _AFTER_BYTES, snapshot)

    assert owner.events == []
    assert owner.issued_temp is None
    assert owner.original_live and owner.original_bytes == _BASE_BYTES
    assert not target.closed


@pytest.mark.parametrize("payload", ["matching", "altered"])
def test_owner_binding_requires_exact_authorized_binding(
    payload: str,
) -> None:
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget
    from mochi.tools.file_transaction import atomic_write_bytes

    owner, target, snapshot = _transaction()
    impersonated = (
        owner.binding
        if payload == "matching"
        else replace(owner.binding, entry_id="different-entry")
    )
    owner.transaction_binding_result = _EqualBindingImpostor(impersonated)

    with pytest.raises(UnsafeFilesystemTarget, match="owner binding"):
        atomic_write_bytes(target, _AFTER_BYTES, snapshot)

    assert owner.events == []
    assert owner.issued_temp is None
    assert owner.original_bytes == _BASE_BYTES
    assert not target.closed


def test_distinct_parent_owner_temp_completes_transaction() -> None:
    from mochi.tools.file_transaction import atomic_write_bytes

    owner, target, snapshot = _transaction()
    assert owner.temp_parent is not target._parent

    result = atomic_write_bytes(target, _AFTER_BYTES, snapshot)

    assert result.successor_identity == owner.successor_identity
    assert owner.original_bytes == _AFTER_BYTES
    assert owner.issued_temp is not None and owner.issued_temp.closed
    assert target.closed


@pytest.mark.parametrize(
    "tamper",
    ["binding", "authorization_digest"],
)
def test_semantic_temp_mismatch_is_discarded_after_owner_claim(
    tamper: str,
) -> None:
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget
    from mochi.tools.file_transaction import atomic_write_bytes

    owner, target, snapshot = _transaction()
    owner.temp_tamper = tamper

    with pytest.raises(UnsafeFilesystemTarget, match="staged temp"):
        atomic_write_bytes(target, _AFTER_BYTES, snapshot)

    assert owner.events == ["create_temp", "discard_temp"]
    assert owner.discard_attempts == 1
    assert owner.discarded_temp is owner.issued_temp
    assert owner.issued_temp is not None and owner.issued_temp.closed
    assert not owner.temp_live and not owner.temp_resource_open
    assert owner.original_bytes == _BASE_BYTES
    assert owner.original_identity == owner.binding.base_identity
    assert owner.target_resource_open and not target.closed

@pytest.mark.parametrize(
    "tamper",
    [
        "wrong_type",
        "closed",
        "authenticity",
        "foreign_owner",
    ],
)
def test_untrusted_temp_candidate_is_rejected_before_staging(
    tamper: str,
) -> None:
    from mochi.security.safe_filesystem import UnsafeFilesystemTarget
    from mochi.tools.file_transaction import atomic_write_bytes

    owner, target, snapshot = _transaction()
    owner.temp_tamper = tamper

    with pytest.raises(UnsafeFilesystemTarget, match="staged temp"):
        atomic_write_bytes(target, _AFTER_BYTES, snapshot)

    assert owner.events == ["create_temp"]
    assert owner.discard_attempts == 0
    assert owner.temp_bytes == b""
    assert owner.original_bytes == _BASE_BYTES
    assert owner.original_identity == owner.binding.base_identity
    assert owner.target_resource_open and not target.closed


def test_short_writes_and_interruptions_are_retried_until_complete() -> None:
    from mochi.tools.file_transaction import atomic_write_bytes

    owner, target, snapshot = _transaction(
        write_plan=[InterruptedError(), 3, 2, 100]
    )
    result = atomic_write_bytes(target, _AFTER_BYTES, snapshot)

    assert result.bytes_written == len(_AFTER_BYTES)
    assert owner.original_bytes == _AFTER_BYTES
    assert owner.events.count("write") == 4
    assert target.closed


def test_zero_byte_write_fails_closed_and_discards_temp() -> None:
    from mochi.tools.file_transaction import atomic_write_bytes

    owner, target, snapshot = _transaction(write_plan=[0])
    with pytest.raises(OSError, match="progress|zero bytes"):
        atomic_write_bytes(target, _AFTER_BYTES, snapshot)

    assert owner.events == ["create_temp", "write", "discard_temp"]
    assert owner.discarded_temp is owner.issued_temp
    assert owner.issued_temp is not None and owner.issued_temp.closed
    assert not owner.temp_live and not owner.temp_resource_open
    assert owner.original_live and owner.original_bytes == _BASE_BYTES
    assert owner.target_resource_open and not target.closed


def test_mid_write_error_discards_issued_temp_and_preserves_target() -> None:
    from mochi.tools.file_transaction import atomic_write_bytes

    failure = OSError("device write failed")
    owner, target, snapshot = _transaction(write_plan=[4, failure])
    with pytest.raises(OSError) as raised:
        atomic_write_bytes(target, _AFTER_BYTES, snapshot)

    assert raised.value is failure
    assert owner.discarded_temp is owner.issued_temp
    assert owner.issued_temp is not None and owner.issued_temp.closed
    assert not owner.temp_live and not owner.temp_resource_open
    assert owner.original_live and owner.original_bytes == _BASE_BYTES
    assert owner.target_resource_open and not target.closed


def test_temp_flush_failure_discards_temp_and_preserves_target() -> None:
    from mochi.tools.file_transaction import atomic_write_bytes

    failure = OSError("temp flush failed")
    owner, target, snapshot = _transaction(
        failures={"flush_temp": failure}
    )
    with pytest.raises(OSError) as raised:
        atomic_write_bytes(target, _AFTER_BYTES, snapshot)

    assert raised.value is failure
    assert owner.events[-2:] == ["flush_temp", "discard_temp"]
    assert owner.discarded_temp is owner.issued_temp
    assert not owner.temp_live
    assert owner.original_live and owner.original_bytes == _BASE_BYTES
    assert not target.closed


def test_base_revalidation_failure_after_flush_preserves_target() -> None:
    from mochi.tools.file_transaction import atomic_write_bytes

    failure = OSError("base changed")
    owner, target, snapshot = _transaction(
        failures={"revalidate_base": failure}
    )
    with pytest.raises(OSError) as raised:
        atomic_write_bytes(target, _AFTER_BYTES, snapshot)

    assert raised.value is failure
    assert owner.events[-3:] == [
        "flush_temp",
        "revalidate_base",
        "discard_temp",
    ]
    assert owner.temp_flushed
    assert owner.discarded_temp is owner.issued_temp
    assert not owner.temp_live
    assert owner.original_live and owner.original_bytes == _BASE_BYTES
    assert not target.closed


def test_replace_precommit_failure_discards_temp_and_preserves_target() -> None:
    from mochi.tools.file_transaction import atomic_write_bytes

    failure = OSError("replace rejected")
    owner, target, snapshot = _transaction(failures={"replace": failure})
    with pytest.raises(OSError) as raised:
        atomic_write_bytes(target, _AFTER_BYTES, snapshot)

    assert raised.value is failure
    assert owner.events[-2:] == ["replace", "discard_temp"]
    assert owner.discarded_temp is owner.issued_temp
    assert not owner.temp_live
    assert owner.original_live and owner.original_bytes == _BASE_BYTES
    assert owner.target_resource_open and not target.closed


def test_committed_replace_failure_is_not_discarded_or_wrapped() -> None:
    from mochi.security.safe_filesystem import (
        CommittedFilesystemMutationError,
    )
    from mochi.tools.file_transaction import atomic_write_bytes

    committed = CommittedFilesystemMutationError(
        phase="release_target",
        cause=OSError("target release failed after replace"),
    )
    owner, target, snapshot = _transaction()
    owner.committed_replace_failure = committed

    with pytest.raises(CommittedFilesystemMutationError) as raised:
        atomic_write_bytes(target, _AFTER_BYTES, snapshot)

    assert raised.value is committed
    assert owner.discard_attempts == 0
    assert getattr(committed, "__notes__", ()) == ()
    assert owner.events[-1] == "replace"
    assert owner.original_bytes == _AFTER_BYTES
    assert owner.original_identity == owner.successor_identity
    assert owner.issued_temp is not None and owner.issued_temp.closed
    assert target.closed
    assert not owner.temp_live and not owner.temp_resource_open
    assert not owner.target_resource_open


@pytest.mark.parametrize(
    "violation",
    ["return_type", "source_open", "destination_open"],
)
def test_replace_success_contract_violation_is_committed(
    violation: str,
) -> None:
    from mochi.security.safe_filesystem import (
        CommittedFilesystemMutationError,
    )
    from mochi.tools.file_transaction import atomic_write_bytes

    owner, target, snapshot = _transaction()
    if violation == "return_type":
        owner.replace_result = object()
    else:
        owner.replace_consumption = violation

    with pytest.raises(CommittedFilesystemMutationError) as raised:
        atomic_write_bytes(target, _AFTER_BYTES, snapshot)

    assert raised.value.phase == "validate_successor"
    assert owner.discard_attempts == 0
    assert owner.events[-1] == "replace"
    assert owner.original_bytes == _AFTER_BYTES
    assert owner.original_identity == owner.successor_identity
    assert owner.issued_temp is not None
    if violation == "source_open":
        assert not owner.issued_temp.closed
    else:
        assert owner.issued_temp.closed
    if violation == "destination_open":
        assert not target.closed
    else:
        assert target.closed


def test_success_runs_exact_foundation_sequence_and_consumes_operands() -> None:
    from mochi.tools.file_transaction import (
        AtomicWriteResult,
        atomic_write_bytes,
    )

    owner, target, snapshot = _transaction()
    result = atomic_write_bytes(target, _AFTER_BYTES, snapshot)

    assert owner.events == [
        "create_temp",
        "write",
        "apply_snapshot",
        "verify_staged",
        "flush_temp",
        "revalidate_base",
        "replace",
    ]
    assert result == AtomicWriteResult(
        successor_identity=owner.successor_identity,
        bytes_written=len(_AFTER_BYTES),
    )
    assert owner.original_live and owner.original_bytes == _AFTER_BYTES
    assert owner.original_identity == owner.successor_identity
    assert owner.issued_temp is not None and owner.issued_temp.closed
    assert target.closed
    assert not owner.temp_live and not owner.temp_resource_open
    assert not owner.target_resource_open
    with pytest.raises(FrozenInstanceError):
        result.bytes_written = 0


def test_cleanup_str_failure_does_not_mask_primary_error() -> None:
    from mochi.tools.file_transaction import atomic_write_bytes

    primary = OSError("replace failed first")
    cleanup = _UnsafeCleanupError()
    owner, target, snapshot = _transaction(
        failures={"replace": primary, "discard_temp": cleanup}
    )

    with pytest.raises(BaseException) as raised:
        atomic_write_bytes(target, _AFTER_BYTES, snapshot)

    assert raised.value is primary
    assert owner.discard_attempts == 1
    assert owner.original_bytes == _BASE_BYTES
    assert not target.closed


def test_overridden_add_note_does_not_mask_primary_error() -> None:
    from mochi.tools.file_transaction import atomic_write_bytes

    primary = _UnsafeAddNoteError("replace failed first")
    cleanup = OSError("discard failed second")
    owner, target, snapshot = _transaction(
        failures={"replace": primary, "discard_temp": cleanup}
    )

    with pytest.raises(BaseException) as raised:
        atomic_write_bytes(target, _AFTER_BYTES, snapshot)

    assert raised.value is primary
    assert any(
        "discard failed second" in note
        for note in getattr(primary, "__notes__", ())
    )
    assert owner.original_bytes == _BASE_BYTES
    assert not target.closed


def test_cleanup_failure_is_attached_without_masking_precommit_error() -> None:
    from mochi.tools.file_transaction import atomic_write_bytes

    primary = OSError("replace failed first")
    cleanup = OSError("discard failed second")
    owner, target, snapshot = _transaction(
        failures={"replace": primary, "discard_temp": cleanup}
    )
    with pytest.raises(OSError) as raised:
        atomic_write_bytes(target, _AFTER_BYTES, snapshot)

    assert raised.value is primary
    assert any(
        "discard failed second" in note
        for note in getattr(primary, "__notes__", ())
    )
    assert owner.discarded_temp is owner.issued_temp
    assert not owner.temp_live
    assert owner.original_bytes == _BASE_BYTES
    assert not target.closed


@pytest.mark.parametrize(
    ("successor_identity", "bytes_written", "error"),
    [
        (object(), 0, TypeError),
        ("valid", True, TypeError),
        ("valid", 1.5, TypeError),
        ("valid", -1, ValueError),
    ],
)
def test_atomic_write_result_validates_exact_fields(
    successor_identity: object,
    bytes_written: object,
    error: type[Exception],
) -> None:
    from mochi.security.file_contract import FileIdentity
    from mochi.tools.file_transaction import AtomicWriteResult

    identity = (
        FileIdentity("posix", "1", "99", 1, False)
        if successor_identity == "valid"
        else successor_identity
    )
    with pytest.raises(error):
        AtomicWriteResult(
            successor_identity=identity,
            bytes_written=bytes_written,
        )


def test_public_contract_exposes_no_path_or_raw_resource_api() -> None:
    import mochi.tools.file_transaction as transaction
    from mochi.security.safe_filesystem import SafeTarget
    from mochi.tools.file_transaction import (
        AtomicWriteResult,
        FileMetadataSnapshot,
        atomic_write_bytes,
    )

    signature = inspect.signature(atomic_write_bytes)
    assert list(signature.parameters) == [
        "target",
        "data",
        "metadata_snapshot",
    ]
    assert all(
        item.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
        for item in signature.parameters.values()
    )
    assert get_type_hints(atomic_write_bytes) == {
        "target": SafeTarget,
        "data": bytes,
        "metadata_snapshot": FileMetadataSnapshot,
        "return": AtomicWriteResult,
    }
    forbidden = ("path", "fd", "handle", "descriptor")
    public_names = tuple(transaction.__all__)
    contract_fields = tuple(
        item.name
        for contract in (FileMetadataSnapshot, AtomicWriteResult)
        for item in fields(contract)
    )
    assert not any(
        token in name.casefold()
        for name in public_names + contract_fields
        for token in forbidden
    )

def test_undo_cas_rejects_same_content_with_replaced_identity() -> None:
    from mochi.security.file_contract import FileIdentity
    from mochi.tools.file_transaction import (
        UndoCASConflict,
        UndoCASObservation,
        validate_undo_cas,
    )

    digest = _sha_label("same-content")
    metadata = _sha_label("same-metadata")
    expected = UndoCASObservation(
        identity=FileIdentity("posix", "1", "41", 1, False),
        content_sha256=digest,
        metadata_sha256=metadata,
    )
    replaced = UndoCASObservation(
        identity=FileIdentity("posix", "1", "99", 1, False),
        content_sha256=digest,
        metadata_sha256=metadata,
    )

    with pytest.raises(UndoCASConflict, match="undo_target_changed"):
        validate_undo_cas(expected, replaced)

def test_undo_staging_failure_cleans_previously_staged_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import mochi.tools.file_transaction as transaction
    from mochi.tools.file_transaction import (
        UndoMutation,
        execute_authoritative_undo,
        observe_undo_target,
    )

    first = tmp_path / "first.txt"
    second = tmp_path / "second.txt"
    first.write_bytes(b"after-one")
    second.write_bytes(b"after-two")
    mutations = (
        UndoMutation(
            entry_id="entry-one",
            relative_name="first.txt",
            operation="restore",
            expected=observe_undo_target(first),
            restore_content=b"before-one",
        ),
        UndoMutation(
            entry_id="entry-two",
            relative_name="second.txt",
            operation="restore",
            expected=observe_undo_target(second),
            restore_content=b"before-two",
        ),
    )
    real_stage = transaction._stage_undo_bytes
    stage_calls = 0

    def fail_second_stage(target: Path, content: bytes, mode: int | None) -> Path:
        nonlocal stage_calls
        stage_calls += 1
        if stage_calls == 2:
            raise OSError("injected staging failure")
        return real_stage(target, content, mode)

    monkeypatch.setattr(transaction, "_stage_undo_bytes", fail_second_stage)
    with pytest.raises(OSError, match="injected staging failure"):
        execute_authoritative_undo(tmp_path, mutations)

    assert list(tmp_path.glob(".mochi-undo-*")) == []
    assert first.read_bytes() == b"after-one"
    assert second.read_bytes() == b"after-two"