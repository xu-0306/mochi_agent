"""Capability-only orchestration for an existing-file atomic write."""

from __future__ import annotations

import hashlib as _hashlib
import re as _re
from dataclasses import dataclass as _dataclass
from typing import Literal as _Literal
from typing import Protocol as _Protocol

from ..security.file_contract import FileIdentity as _FileIdentity
from ..security.safe_filesystem import (
    AuthorizedFileBinding as _AuthorizedFileBinding,
)
from ..security.safe_filesystem import (
    CommittedFilesystemMutationError as _CommittedFilesystemMutationError,
)
from ..security.safe_filesystem import (
    SafeTarget as _SafeTarget,
)
from ..security.safe_filesystem import (
    StagedTemp as _StagedTemp,
)
from ..security.safe_filesystem import (
    UnsafeFilesystemTarget as _UnsafeFilesystemTarget,
)

_SHA256_PATTERN = _re.compile(r"[0-9a-f]{64}\Z")


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256_PATTERN.fullmatch(value) is not None


@_dataclass(frozen=True, slots=True)
class FileMetadataSnapshot:
    """Authorized metadata state for an existing-file replacement."""

    kind: _Literal["existing_file"]
    identity: _FileIdentity
    binding: _AuthorizedFileBinding
    canonical_metadata_sha256: str

    def __post_init__(self) -> None:
        if self.kind != "existing_file":
            raise ValueError("kind must be exactly 'existing_file'")
        if not _is_sha256(self.canonical_metadata_sha256):
            raise ValueError(
                "canonical_metadata_sha256 must be lowercase hexadecimal SHA-256"
            )


@_dataclass(frozen=True, slots=True)
class AtomicWriteResult:
    """Result returned after the capability owner commits the replace."""

    successor_identity: _FileIdentity
    bytes_written: int

    def __post_init__(self) -> None:
        if type(self.successor_identity) is not _FileIdentity:
            raise TypeError("successor_identity must be exactly FileIdentity")
        if type(self.bytes_written) is not int:
            raise TypeError("bytes_written must be exactly int")
        if self.bytes_written < 0:
            raise ValueError("bytes_written must be non-negative")


class _TransactionOwner(_Protocol):
    def transaction_binding(
        self, target: _SafeTarget
    ) -> _AuthorizedFileBinding: ...

    def create_temp(self, target: _SafeTarget) -> _StagedTemp: ...

    def write_temp(self, temp: _StagedTemp, data: memoryview) -> int: ...

    def apply_metadata_snapshot(
        self, temp: _StagedTemp, snapshot: FileMetadataSnapshot
    ) -> None: ...

    def verify_staged(
        self, temp: _StagedTemp, snapshot: FileMetadataSnapshot
    ) -> None: ...

    def flush_temp(self, temp: _StagedTemp) -> None: ...

    def revalidate_base(
        self, target: _SafeTarget, snapshot: FileMetadataSnapshot
    ) -> None: ...

    def replace(
        self, source: _StagedTemp, destination: _SafeTarget
    ) -> _FileIdentity: ...

    def discard_temp(self, temp: _StagedTemp) -> None: ...


def _reject(message: str) -> None:
    raise _UnsafeFilesystemTarget(message)


def _validate_snapshot(
    target: _SafeTarget,
    data: bytes,
    metadata_snapshot: FileMetadataSnapshot,
) -> _TransactionOwner:
    if (
        not isinstance(target, _SafeTarget)
        or target.closed
        or not target._is_authentic()
    ):
        _reject("invalid target capability")
    if not isinstance(data, bytes):
        raise TypeError("data must be bytes")
    if not isinstance(metadata_snapshot, FileMetadataSnapshot):
        raise TypeError("metadata_snapshot must be FileMetadataSnapshot")

    binding = metadata_snapshot.binding
    if not isinstance(binding, _AuthorizedFileBinding):
        _reject("metadata snapshot has an invalid authorization binding")

    if (
        target.authorization_digest != binding.authorization_digest
        or target.identity != binding.base_identity
    ):
        _reject("target capability does not match its authorization binding")

    digest_fields = (
        binding.authorization_digest,
        binding.base_sha256,
        binding.after_sha256,
        binding.base_metadata_sha256,
        binding.after_metadata_sha256,
        metadata_snapshot.canonical_metadata_sha256,
    )
    if not all(_is_sha256(value) for value in digest_fields):
        _reject("metadata snapshot requires lowercase hexadecimal SHA-256 values")
    if binding.operation not in {"update", "rename"}:
        _reject("metadata snapshot is not an existing-file write operation")
    if (
        metadata_snapshot.identity != binding.base_identity
        or metadata_snapshot.identity != target.identity
        or metadata_snapshot.canonical_metadata_sha256
        != binding.after_metadata_sha256
    ):
        _reject("metadata snapshot does not match the target binding")
    if _hashlib.sha256(data).hexdigest() != binding.after_sha256:
        _reject("metadata snapshot data does not match authorized after_sha256")

    owner: _TransactionOwner = target._owner
    owner_binding = owner.transaction_binding(target)
    if (
        type(owner_binding) is not _AuthorizedFileBinding
        or owner_binding != binding
    ):
        _reject("metadata snapshot does not match the exact owner binding")
    return owner


def _validate_temp_candidate(
    candidate: object,
    owner: _TransactionOwner,
    target: _SafeTarget,
    binding: _AuthorizedFileBinding,
) -> _StagedTemp:
    if (
        type(candidate) is not _StagedTemp
        or candidate.closed
        or not candidate._is_authentic()
        or candidate._owner is not owner
        or type(candidate.binding) is not _AuthorizedFileBinding
        or candidate.binding != binding
        or candidate.authorization_digest != binding.authorization_digest
        or candidate._parent is not target._parent
    ):
        _reject("owner returned an invalid staged temp capability")
    return candidate


def _write_all(
    owner: _TransactionOwner, temp: _StagedTemp, data: bytes
) -> int:
    view = memoryview(data)
    total = 0
    while total < len(view):
        try:
            written = owner.write_temp(temp, view[total:])
        except InterruptedError:
            continue
        if type(written) is not int or written <= 0:
            raise OSError("temporary file write made no progress (zero bytes)")
        if written > len(view) - total:
            raise OSError("temporary file write exceeded the requested bytes")
        total += written
    return total


def _attach_cleanup_failure(
    primary: BaseException, cleanup: BaseException
) -> None:
    try:
        try:
            cleanup_message = str(cleanup)
        except BaseException:
            cleanup_message = f"<{type(cleanup).__name__} could not be formatted>"
        note = f"discard_temp cleanup failed: {cleanup_message}"
        BaseException.add_note(primary, note)
    except BaseException:
        pass


def _validate_successor(
    successor_identity: object,
    source: _StagedTemp,
    destination: _SafeTarget,
) -> _FileIdentity:
    cause: BaseException | None = None
    if type(successor_identity) is not _FileIdentity:
        cause = TypeError("replace must return exactly FileIdentity")
    elif not source.closed or not destination.closed:
        cause = RuntimeError(
            "replace must consume both source and destination capabilities"
        )
    if cause is not None:
        error = _CommittedFilesystemMutationError(
            phase="validate_successor",
            cause=cause,
        )
        raise error from cause
    return successor_identity


def atomic_write_bytes(
    target: _SafeTarget,
    data: bytes,
    metadata_snapshot: FileMetadataSnapshot,
) -> AtomicWriteResult:
    """Stage, verify, flush, revalidate, and atomically replace one file."""

    owner = _validate_snapshot(target, data, metadata_snapshot)
    temp: _StagedTemp | None = None
    try:
        candidate = owner.create_temp(target)
        temp = _validate_temp_candidate(
            candidate,
            owner,
            target,
            metadata_snapshot.binding,
        )
        bytes_written = _write_all(owner, temp, data)
        owner.apply_metadata_snapshot(temp, metadata_snapshot)
        owner.verify_staged(temp, metadata_snapshot)
        owner.flush_temp(temp)
        owner.revalidate_base(target, metadata_snapshot)
        successor_identity = owner.replace(temp, target)
        successor_identity = _validate_successor(
            successor_identity, temp, target
        )
    except _CommittedFilesystemMutationError:
        raise
    except BaseException as primary:
        if temp is not None:
            try:
                owner.discard_temp(temp)
            except BaseException as cleanup:
                _attach_cleanup_failure(primary, cleanup)
        raise

    return AtomicWriteResult(
        successor_identity=successor_identity,
        bytes_written=bytes_written,
    )


__all__ = [
    "AtomicWriteResult",
    "FileMetadataSnapshot",
    "atomic_write_bytes",
]
