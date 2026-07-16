from __future__ import annotations

import hashlib
import inspect
from dataclasses import FrozenInstanceError, fields, replace
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
        return self.binding

    def create_temp(self, target: SafeTarget) -> StagedTemp:
        from mochi.security.safe_filesystem import StagedTemp

        assert target is self.target and not self.temp_live
        self.events.append("create_temp")
        temp = StagedTemp._create(
            basename=".mochi-note.tmp",
            identity=self.successor_identity,
            binding=self.binding,
            owner=self,
            parent=target._parent,
        )
        self.issued_temp = temp
        self.temp_live = self.temp_resource_open = True
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
        source._mark_closed()
        destination._mark_closed()
        return self.successor_identity

    def discard_temp(self, temp: StagedTemp) -> None:
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
