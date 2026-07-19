"""Capability-only orchestration for an existing-file atomic write."""

from __future__ import annotations

import asyncio as _asyncio
import hashlib as _hashlib
import re as _re
from contextlib import suppress as _suppress
from dataclasses import dataclass as _dataclass
from typing import Literal as _Literal
from typing import Protocol as _Protocol
from typing import cast as _cast

from ..runtime.change_sets import (
    ChangeSetStore as _ChangeSetStore,
)
from ..runtime.change_sets import (
    JournalEntryRecord as _JournalEntryRecord,
)
from ..runtime.change_sets import (
    JournalEntryState as _JournalEntryState,
)
from ..security.file_contract import ChangeEntry as _ChangeEntry
from ..security.file_contract import ChangeManifest as _ChangeManifest
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


def _runtime_object(value: object) -> object:
    return value


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
        identity_value = _runtime_object(self.successor_identity)
        bytes_value = _runtime_object(self.bytes_written)
        if type(identity_value) is not _FileIdentity:
            raise TypeError("successor_identity must be exactly FileIdentity")
        if type(bytes_value) is not int:
            raise TypeError("bytes_written must be exactly int")
        if bytes_value < 0:
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
    target_value = _runtime_object(target)
    data_value = _runtime_object(data)
    snapshot_value = _runtime_object(metadata_snapshot)
    if not isinstance(target_value, _SafeTarget):
        _reject("invalid target capability")
    safe_target = _cast(_SafeTarget, target_value)
    # This layer must inspect opaque capability seals and owners to claim them.
    if (
        safe_target.closed
        or not safe_target._is_authentic()  # pyright: ignore[reportPrivateUsage]
    ):
        _reject("invalid target capability")
    if not isinstance(data_value, bytes):
        raise TypeError("data must be bytes")
    safe_data = data_value
    if not isinstance(snapshot_value, FileMetadataSnapshot):
        raise TypeError("metadata_snapshot must be FileMetadataSnapshot")
    snapshot = snapshot_value

    binding_value = _runtime_object(snapshot.binding)
    if not isinstance(binding_value, _AuthorizedFileBinding):
        _reject("metadata snapshot has an invalid authorization binding")
    binding = _cast(_AuthorizedFileBinding, binding_value)

    if (
        safe_target.authorization_digest != binding.authorization_digest
        or safe_target.identity != binding.base_identity
    ):
        _reject("target capability does not match its authorization binding")

    digest_fields = (
        binding.authorization_digest,
        binding.base_sha256,
        binding.after_sha256,
        binding.base_metadata_sha256,
        binding.after_metadata_sha256,
        snapshot.canonical_metadata_sha256,
    )
    if not all(_is_sha256(value) for value in digest_fields):
        _reject("metadata snapshot requires lowercase hexadecimal SHA-256 values")
    if binding.operation not in {"update", "rename"}:
        _reject("metadata snapshot is not an existing-file write operation")
    if (
        snapshot.identity != binding.base_identity
        or snapshot.identity != safe_target.identity
        or snapshot.canonical_metadata_sha256
        != binding.after_metadata_sha256
    ):
        _reject("metadata snapshot does not match the target binding")
    if _hashlib.sha256(safe_data).hexdigest() != binding.after_sha256:
        _reject("metadata snapshot data does not match authorized after_sha256")

    owner_value: object = safe_target._owner  # pyright: ignore[reportPrivateUsage]
    owner = _cast(_TransactionOwner, owner_value)
    owner_binding = owner.transaction_binding(safe_target)
    if (
        type(owner_binding) is not _AuthorizedFileBinding
        or owner_binding != binding
    ):
        _reject("metadata snapshot does not match the exact owner binding")
    return owner

def _claim_temp_candidate(
    candidate: object,
    owner: _TransactionOwner,
) -> _StagedTemp:
    if type(candidate) is not _StagedTemp:
        _reject("owner returned an invalid staged temp capability")
    temp = _cast(_StagedTemp, candidate)
    # Claiming requires package-private seal and owner inspection.
    if (
        temp.closed
        or not temp._is_authentic()  # pyright: ignore[reportPrivateUsage]
        or temp._owner is not owner  # pyright: ignore[reportPrivateUsage]
    ):
        _reject("owner returned an invalid staged temp capability")
    return temp


def _validate_temp_semantics(
    temp: _StagedTemp,
    binding: _AuthorizedFileBinding,
) -> None:
    binding_value: object = temp.binding
    if (
        type(binding_value) is not _AuthorizedFileBinding
        or binding_value != binding
        or temp.authorization_digest != binding.authorization_digest
    ):
        _reject("owner returned an invalid staged temp capability")

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
    return _cast(_FileIdentity, successor_identity)


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
        claimed_temp = _claim_temp_candidate(candidate, owner)
        temp = claimed_temp
        _validate_temp_semantics(temp, metadata_snapshot.binding)
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
    "DurableMutationParticipant",
    "FileMetadataSnapshot",
    "FileTransactionError",
    "JournalRecoveryAdapter",
    "RecoveryObservation",
    "StagedMutation",
    "StagedRollback",
    "atomic_write_bytes",
    "classify_journal_entry",
    "execute_durable_file_transaction",
    "recover_incomplete_file_transactions",
]



@_dataclass(frozen=True, slots=True)
class RecoveryObservation:
    """Identity, content, and metadata observed through a pinned successor."""

    identity: _FileIdentity
    content_sha256: str | None
    metadata_sha256: str | None


class JournalRecoveryAdapter(_Protocol):
    """Platform owner used to inspect and reconcile durable journal entries."""

    def observe(
        self,
        journal_id: str,
        entry: _JournalEntryRecord,
    ) -> RecoveryObservation: ...

    def discard_staged(
        self,
        journal_id: str,
        entry: _JournalEntryRecord,
    ) -> None: ...

    def stage_rollback(
        self,
        journal_id: str,
        entry: _JournalEntryRecord,
        rollback_content: bytes,
    ) -> StagedRollback: ...

    def rollback(
        self,
        journal_id: str,
        entry: _JournalEntryRecord,
        rollback_content: bytes,
    ) -> RecoveryObservation: ...


def classify_journal_entry(
    entry: _JournalEntryRecord,
    observation: RecoveryObservation,
    *,
    after_metadata_sha256: str | None,
) -> _JournalEntryState:
    """Classify a crash state only when identity, content, and metadata agree."""

    base_metadata = entry.base_metadata_blob_id
    base_matches = (
        entry.base_identity is not None
        and entry.base_sha256 is not None
        and base_metadata is not None
        and observation.content_sha256 == entry.base_sha256
        and observation.metadata_sha256 == base_metadata
    )
    if (
        base_matches
        and observation.identity
        in (
            entry.rollback_staged_identity,
            entry.rollback_successor_identity,
        )
    ):
        return "rolled_back"
    if (
        entry.staged_identity is not None
        and entry.after_sha256 is not None
        and after_metadata_sha256 is not None
        and observation.identity == entry.staged_identity
        and observation.content_sha256 == entry.after_sha256
        and observation.metadata_sha256 == after_metadata_sha256
    ):
        return "applied"
    if observation.identity == entry.base_identity and base_matches:
        return "staged"
    return "interference"


async def _observe_recovery_entry(
    adapter: JournalRecoveryAdapter,
    journal_id: str,
    entry: _JournalEntryRecord,
) -> RecoveryObservation:
    observation = _runtime_object(
        await _asyncio.to_thread(
            adapter.observe,
            journal_id,
            entry,
        )
    )
    if not isinstance(observation, RecoveryObservation):
        raise TypeError("recovery adapter must return RecoveryObservation")
    return observation



async def recover_incomplete_file_transactions(
    change_store: _ChangeSetStore,
    adapter: JournalRecoveryAdapter,
) -> tuple[dict[str, object], ...]:
    """Reconcile pending journals before accepting new file mutations."""

    journals = await change_store.list_incomplete_journals()
    recovered: list[dict[str, object]] = []
    for journal in journals:
        journal_id = str(journal["id"])
        change_set_id = str(journal["change_set_id"])
        change_set = await change_store.get_change_set(change_set_id)
        if change_set is None:
            updated = await change_store.update_journal(
                journal_id,
                status="interference",
                phase="recovery",
                error="change set is missing",
            )
            recovered.append(updated)
            continue
        manifest = change_set["manifest"]
        entry_contracts = {item.entry_id: item for item in manifest.entries}
        entries = tuple(journal["entries"])
        classifications: dict[str, _JournalEntryState] = {}
        observations: dict[str, RecoveryObservation] = {}
        recovery_error: BaseException | None = None
        for entry in entries:
            contract = entry_contracts.get(entry.entry_id)
            if contract is None:
                classifications[entry.entry_id] = "interference"
                continue
            try:
                observation = await _observe_recovery_entry(
                    adapter,
                    journal_id,
                    entry,
                )
            except BaseException as exc:
                recovery_error = exc
                classifications[entry.entry_id] = "interference"
                continue
            observations[entry.entry_id] = observation
            classifications[entry.entry_id] = classify_journal_entry(
                entry,
                observation,
                after_metadata_sha256=contract.after_metadata_sha256,
            )

        if recovery_error is not None or any(
            state == "interference" for state in classifications.values()
        ):
            for entry in entries:
                if classifications.get(entry.entry_id) == "interference":
                    await change_store.update_journal_entry(
                        journal_id,
                        entry.entry_id,
                        state="interference",
                        last_error=(
                            "recovery observation could not prove a known state"
                        ),
                    )
            updated = await change_store.finalize_journal(
                journal_id,
                change_set_id=change_set_id,
                status="interference",
                phase="recovery",
                error=(
                    str(recovery_error)
                    if recovery_error is not None
                    else "identity/content/metadata mismatch"
                ),
                release_references=False,
            )
            recovered.append(updated)
            continue

        if classifications and all(
            state == "applied" for state in classifications.values()
        ):
            for entry in entries:
                observation = observations[entry.entry_id]
                contract = entry_contracts[entry.entry_id]
                await change_store.update_journal_entry(
                    journal_id,
                    entry.entry_id,
                    state="applied",
                )
                await change_store.record_applied_entry(
                    change_set_id=change_set_id,
                    entry_id=entry.entry_id,
                    applied_sha256=observation.content_sha256,
                    applied_identity=observation.identity,
                    applied_metadata_sha256=contract.after_metadata_sha256,
                )
            updated = await change_store.finalize_journal(
                journal_id,
                change_set_id=change_set_id,
                status="applied",
                phase="verified",
                error=None,
                release_references=True,
            )
            recovered.append(updated)
            continue

        if not any(
            state == "applied" for state in classifications.values()
        ):
            cleanup_errors: list[BaseException] = []
            for entry in entries:
                try:
                    if classifications[entry.entry_id] == "staged":
                        await _asyncio.to_thread(
                            adapter.discard_staged,
                            journal_id,
                            entry,
                        )
                    await change_store.update_journal_entry(
                        journal_id,
                        entry.entry_id,
                        state="rolled_back",
                    )
                except BaseException as exc:
                    cleanup_errors.append(exc)
                    await change_store.update_journal_entry(
                        journal_id,
                        entry.entry_id,
                        state="rollback_failed",
                        last_error=str(exc),
                    )
            status = "rollback_failed" if cleanup_errors else "rolled_back"
            updated = await change_store.finalize_journal(
                journal_id,
                change_set_id=change_set_id,
                status=status,
                phase=(
                    "recovery_cleanup"
                    if cleanup_errors
                    else "recovered"
                ),
                error=str(cleanup_errors[0]) if cleanup_errors else None,
                release_references=not cleanup_errors,
            )
            recovered.append(updated)
            continue


        rollback_failed: BaseException | None = None
        for entry in sorted(
            entries,
            key=lambda item: item.ordinal,
            reverse=True,
        ):
            state = classifications[entry.entry_id]
            try:
                if state == "applied":
                    if entry.rollback_blob_id is None:
                        raise RuntimeError(
                            "applied journal entry has no rollback blob"
                        )
                    rollback_content = await change_store.get_blob(
                        entry.rollback_blob_id
                    )
                    if rollback_content is None:
                        raise RuntimeError("rollback blob is unavailable")
                    rollback_staged_value = _runtime_object(
                        await _asyncio.to_thread(
                            adapter.stage_rollback,
                            journal_id,
                            entry,
                            rollback_content,
                        )
                    )
                    if not isinstance(
                        rollback_staged_value,
                        StagedRollback,
                    ):
                        raise TypeError(
                            "rollback staging must return StagedRollback"
                        )
                    _validate_staged_rollback(
                        entry_contracts[entry.entry_id],
                        rollback_staged_value,
                    )
                    updated_entry = (
                        await change_store.update_journal_entry(
                            journal_id,
                            entry.entry_id,
                            state="rolling_back",
                            rollback_staged_name=(
                                rollback_staged_value.staged_name
                            ),
                            rollback_staged_identity=(
                                rollback_staged_value.staged_identity
                            ),
                        )
                    )
                    observation = await _asyncio.to_thread(
                        adapter.rollback,
                        journal_id,
                        updated_entry,
                        rollback_content,
                    )
                    observation_value = _runtime_object(observation)
                    if not isinstance(
                        observation_value,
                        RecoveryObservation,
                    ):
                        raise TypeError(
                            "rollback must return RecoveryObservation"
                        )
                    observation = observation_value
                    updated_entry = await change_store.update_journal_entry(
                        journal_id,
                        entry.entry_id,
                        rollback_successor_identity=observation.identity,
                    )
                    contract = entry_contracts[entry.entry_id]
                    if classify_journal_entry(
                        updated_entry,
                        observation,
                        after_metadata_sha256=contract.after_metadata_sha256,
                    ) != "rolled_back":
                        raise RuntimeError(
                            "rollback successor failed "
                            "identity/content/metadata verification"
                        )
                    await change_store.update_journal_entry(
                        journal_id,
                        entry.entry_id,
                        state="rolled_back",
                    )
                else:
                    await _asyncio.to_thread(
                        adapter.discard_staged,
                        journal_id,
                        entry,
                    )
                    await change_store.update_journal_entry(
                        journal_id,
                        entry.entry_id,
                        state="rolled_back",
                    )
            except BaseException as exc:
                rollback_failed = exc
                await change_store.update_journal_entry(
                    journal_id,
                    entry.entry_id,
                    state="rollback_failed",
                    last_error=str(exc),
                )
                break
        terminal_status = (
            "rollback_failed"
            if rollback_failed is not None
            else "rolled_back"
        )
        updated = await change_store.finalize_journal(
            journal_id,
            change_set_id=change_set_id,
            status=terminal_status,
            phase="recovery_rollback",
            error=None if rollback_failed is None else str(rollback_failed),
            release_references=rollback_failed is None,
        )
        recovered.append(updated)
    return tuple(recovered)



class FileTransactionError(RuntimeError):
    """A durable multi-entry transaction reached a terminal failure state."""

    def __init__(
        self,
        message: str,
        *,
        journal_id: str,
        status: str,
    ) -> None:
        super().__init__(message)
        self.journal_id = journal_id
        self.status = status


@_dataclass(frozen=True, slots=True)
class StagedMutation:
    """Owner-issued staged state plus authoritative rollback material."""

    entry_id: str
    staged_name: str
    staged_identity: _FileIdentity
    rollback_content: bytes
    base_metadata: bytes


@_dataclass(frozen=True, slots=True)
class StagedRollback:
    """Owner-issued rollback temp identity persisted before replacement."""

    entry_id: str
    staged_name: str
    staged_identity: _FileIdentity


class DurableMutationParticipant(_Protocol):
    entry_id: str

    def stage(self) -> StagedMutation: ...

    def validate(self, staged: StagedMutation) -> None: ...

    def commit(self, staged: StagedMutation) -> RecoveryObservation: ...

    def stage_rollback(
        self,
        staged: StagedMutation,
        rollback_content: bytes,
    ) -> StagedRollback: ...

    def rollback(
        self,
        staged: StagedMutation,
        rollback_staged: StagedRollback,
        rollback_content: bytes,
    ) -> RecoveryObservation: ...

    def discard(self, staged: StagedMutation) -> None: ...


def _content_digest(content: bytes) -> str:
    return _hashlib.sha256(content).hexdigest()


def _validate_staged_mutation(
    entry: _ChangeEntry,
    staged: StagedMutation,
) -> None:
    if staged.entry_id != entry.entry_id:
        raise _UnsafeFilesystemTarget(
            "participant staged another manifest entry"
        )
    if _content_digest(staged.rollback_content) != entry.base_sha256:
        raise _UnsafeFilesystemTarget(
            "rollback content does not match manifest base_sha256"
        )
    if _content_digest(staged.base_metadata) != entry.base_metadata_sha256:
        raise _UnsafeFilesystemTarget(
            "rollback metadata does not match manifest base metadata"
        )


def _validate_staged_rollback(
    entry: _ChangeEntry,
    staged: StagedRollback,
) -> None:
    if staged.entry_id != entry.entry_id:
        raise _UnsafeFilesystemTarget(
            "participant staged rollback for another manifest entry"
        )


def _validate_commit_observation(
    entry: _ChangeEntry,
    staged: StagedMutation,
    observation: RecoveryObservation,
) -> None:
    observation_value = _runtime_object(observation)
    if (
        not isinstance(observation_value, RecoveryObservation)
        or observation_value.identity != staged.staged_identity
        or observation_value.content_sha256 != entry.after_sha256
        or observation_value.metadata_sha256 != entry.after_metadata_sha256
    ):
        raise _UnsafeFilesystemTarget(
            "committed successor failed identity/content/metadata verification"
        )


def _validate_rollback_observation(
    entry: _ChangeEntry,
    rollback_staged: StagedRollback,
    observation: RecoveryObservation,
) -> None:
    observation_value = _runtime_object(observation)
    if (
        not isinstance(observation_value, RecoveryObservation)
        or observation_value.identity != rollback_staged.staged_identity
        or observation_value.content_sha256 != entry.base_sha256
        or observation_value.metadata_sha256 != entry.base_metadata_sha256
    ):
        raise _UnsafeFilesystemTarget(
            "rollback successor failed identity/content/metadata verification"
        )




async def execute_durable_file_transaction(
    change_store: _ChangeSetStore,
    manifest: _ChangeManifest,
    *,
    journal_id: str,
    participants: tuple[DurableMutationParticipant, ...],
) -> dict[str, object]:
    """Stage all, validate all, commit, and reverse-rollback on failure."""

    persisted = await change_store.get_change_set(manifest.change_set_id)
    if persisted is None or persisted["manifest"] != manifest:
        raise _UnsafeFilesystemTarget(
            "transaction requires an exact persisted immutable manifest"
        )
    participant_map = {
        participant.entry_id: participant for participant in participants
    }
    if (
        len(participant_map) != len(participants)
        or set(participant_map) != {entry.entry_id for entry in manifest.entries}
    ):
        raise _UnsafeFilesystemTarget(
            "transaction participants do not match manifest entries"
        )

    staged_by_id: dict[str, StagedMutation] = {}
    journal_entries: list[_JournalEntryRecord] = []
    try:
        for ordinal, entry in enumerate(manifest.entries):
            participant = participant_map[entry.entry_id]
            staged_value = _runtime_object(
                await _asyncio.to_thread(participant.stage)
            )
            if not isinstance(staged_value, StagedMutation):
                raise TypeError("participant stage must return StagedMutation")
            staged = staged_value
            _validate_staged_mutation(entry, staged)
            rollback_blob_id = await change_store.put_blob(
                staged.rollback_content
            )
            metadata_blob_id = await change_store.put_blob(
                staged.base_metadata
            )
            if (
                entry.before_blob_id is not None
                and entry.before_blob_id != rollback_blob_id
            ):
                raise _UnsafeFilesystemTarget(
                    "rollback blob does not match manifest before_blob_id"
                )
            staged_by_id[entry.entry_id] = staged
            journal_entries.append(
                _JournalEntryRecord(
                    entry_id=entry.entry_id,
                    ordinal=ordinal,
                    state="staged",
                    base_sha256=entry.base_sha256,
                    after_sha256=entry.after_sha256,
                    base_identity=entry.base_identity,
                    staged_name=staged.staged_name,
                    staged_identity=staged.staged_identity,
                    rollback_blob_id=rollback_blob_id,
                    rollback_staged_name=None,
                    rollback_staged_identity=None,
                    rollback_successor_identity=None,
                    base_metadata_blob_id=metadata_blob_id,
                    last_error=None,
                )
            )
    except BaseException as exc:
        for entry_id, staged in reversed(tuple(staged_by_id.items())):
            with _suppress(BaseException):
                await _asyncio.to_thread(
                    participant_map[entry_id].discard,
                    staged,
                )
        raise FileTransactionError(
            "file transaction staging failed",
            journal_id=journal_id,
            status="staging_failed",
        ) from exc

    durable_entries = tuple(journal_entries)
    try:
        await change_store.create_journal(
            journal_id=journal_id,
            change_set_id=manifest.change_set_id,
            entries=durable_entries,
        )
    except BaseException as exc:
        for entry_id, staged in reversed(tuple(staged_by_id.items())):
            with _suppress(BaseException):
                await _asyncio.to_thread(
                    participant_map[entry_id].discard,
                    staged,
                )
        raise FileTransactionError(
            "file transaction journal creation failed",
            journal_id=journal_id,
            status="journal_failed",
        ) from exc

    try:
        for entry in manifest.entries:
            await _asyncio.to_thread(
                participant_map[entry.entry_id].validate,
                staged_by_id[entry.entry_id],
            )
    except BaseException as exc:
        cleanup_errors: list[BaseException] = []
        for entry in reversed(manifest.entries):
            try:
                await _asyncio.to_thread(
                    participant_map[entry.entry_id].discard,
                    staged_by_id[entry.entry_id],
                )
                await change_store.update_journal_entry(
                    journal_id,
                    entry.entry_id,
                    state="rolled_back",
                )
            except BaseException as cleanup:
                cleanup_errors.append(cleanup)
                await change_store.update_journal_entry(
                    journal_id,
                    entry.entry_id,
                    state="rollback_failed",
                    last_error=str(cleanup),
                )
        status = "rollback_failed" if cleanup_errors else "rolled_back"
        await change_store.finalize_journal(
            journal_id,
            change_set_id=manifest.change_set_id,
            status=status,
            phase="validate_all",
            error=str(cleanup_errors[0] if cleanup_errors else exc),
            release_references=not cleanup_errors,
        )
        raise FileTransactionError(
            (
                "file transaction validation failed and was rolled back"
                if not cleanup_errors
                else "file transaction validation rollback failed"
            ),
            journal_id=journal_id,
            status=status,
        ) from exc

    await change_store.update_journal(
        journal_id,
        status="applying",
        phase="validate_all",
    )

    committed_entry_ids: list[str] = []
    current_entry_id: str | None = None
    commit_returned = False
    try:
        for entry in manifest.entries:
            current_entry_id = entry.entry_id
            commit_returned = False
            staged = staged_by_id[entry.entry_id]
            await change_store.update_journal_entry(
                journal_id,
                entry.entry_id,
                state="applying",
            )
            observation = await _asyncio.to_thread(
                participant_map[entry.entry_id].commit,
                staged,
            )
            commit_returned = True
            _validate_commit_observation(entry, staged, observation)
            committed_entry_ids.append(entry.entry_id)
            await change_store.record_applied_entry(
                change_set_id=manifest.change_set_id,
                entry_id=entry.entry_id,
                applied_sha256=observation.content_sha256,
                applied_identity=observation.identity,
                applied_metadata_sha256=observation.metadata_sha256,
            )
            await change_store.update_journal_entry(
                journal_id,
                entry.entry_id,
                state="applied",
            )
    except BaseException as exc:
        committed_ids = set(committed_entry_ids)
        uncertain_commit = (
            isinstance(exc, _CommittedFilesystemMutationError)
            or (
                commit_returned
                and current_entry_id is not None
                and current_entry_id not in committed_ids
            )
        )
        if uncertain_commit:
            if current_entry_id is not None:
                await change_store.update_journal_entry(
                    journal_id,
                    current_entry_id,
                    state="interference",
                    last_error=str(exc),
                )
            await change_store.finalize_journal(
                journal_id,
                change_set_id=manifest.change_set_id,
                status="interference",
                phase="commit_verification",
                error=str(exc),
                release_references=False,
            )
            raise FileTransactionError(
                "file transaction commit outcome requires recovery",
                journal_id=journal_id,
                status="interference",
            ) from exc

        rollback_errors: list[BaseException] = []
        for entry in reversed(manifest.entries):
            if entry.entry_id in committed_ids:
                continue
            try:
                await _asyncio.to_thread(
                    participant_map[entry.entry_id].discard,
                    staged_by_id[entry.entry_id],
                )
                await change_store.update_journal_entry(
                    journal_id,
                    entry.entry_id,
                    state="rolled_back",
                )
            except BaseException as cleanup:
                rollback_errors.append(cleanup)
                await change_store.update_journal_entry(
                    journal_id,
                    entry.entry_id,
                    state="rollback_failed",
                    last_error=str(cleanup),
                )

        entries_by_id = {
            entry.entry_id: entry for entry in manifest.entries
        }
        for entry_id in reversed(committed_entry_ids):
            entry = entries_by_id[entry_id]
            staged = staged_by_id[entry_id]
            try:
                rollback_blob_id = next(
                    item.rollback_blob_id
                    for item in durable_entries
                    if item.entry_id == entry_id
                )
                if rollback_blob_id is None:
                    raise RuntimeError(
                        "committed entry has no rollback blob"
                    )
                rollback_content = await change_store.get_blob(
                    rollback_blob_id
                )
                if rollback_content is None:
                    raise RuntimeError("rollback blob is unavailable")
                rollback_staged_value = _runtime_object(
                    await _asyncio.to_thread(
                        participant_map[entry_id].stage_rollback,
                        staged,
                        rollback_content,
                    )
                )
                if not isinstance(
                    rollback_staged_value,
                    StagedRollback,
                ):
                    raise TypeError(
                        "rollback staging must return StagedRollback"
                    )
                _validate_staged_rollback(
                    entry,
                    rollback_staged_value,
                )
                await change_store.update_journal_entry(
                    journal_id,
                    entry_id,
                    state="rolling_back",
                    rollback_staged_name=(
                        rollback_staged_value.staged_name
                    ),
                    rollback_staged_identity=(
                        rollback_staged_value.staged_identity
                    ),
                )
                observation = await _asyncio.to_thread(
                    participant_map[entry_id].rollback,
                    staged,
                    rollback_staged_value,
                    rollback_content,
                )
                _validate_rollback_observation(
                    entry,
                    rollback_staged_value,
                    observation,
                )
                await change_store.update_journal_entry(
                    journal_id,
                    entry_id,
                    state="rolled_back",
                    rollback_successor_identity=observation.identity,
                )
            except BaseException as rollback:
                rollback_errors.append(rollback)
                await change_store.update_journal_entry(
                    journal_id,
                    entry_id,
                    state="rollback_failed",
                    last_error=str(rollback),
                )

        status = "rollback_failed" if rollback_errors else "rolled_back"
        error = rollback_errors[0] if rollback_errors else exc
        await change_store.finalize_journal(
            journal_id,
            change_set_id=manifest.change_set_id,
            status=status,
            phase="rollback",
            error=str(error),
            release_references=not rollback_errors,
        )
        raise FileTransactionError(
            (
                "file transaction commit failed and was rolled back"
                if not rollback_errors
                else "file transaction rollback failed"
            ),
            journal_id=journal_id,
            status=status,
        ) from exc

    return await change_store.finalize_journal(
        journal_id,
        change_set_id=manifest.change_set_id,
        status="applied",
        phase="verified",
        error=None,
        release_references=True,
    )
