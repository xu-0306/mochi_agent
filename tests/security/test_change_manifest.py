"""Durable change-set, blob, and file-transaction journal tests."""

from __future__ import annotations

import asyncio
import hashlib
import sqlite3
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from mochi.runtime.change_sets import (
    ChangeSetConflict,
    ChangeSetStore,
    JournalEntryRecord,
)
from mochi.runtime.store import RuntimeStore
from mochi.security.file_contract import (
    AuthorizationContext,
    AuthorizationEnvelope,
    ChangeEntry,
    ChangeManifest,
    FileChangeRequest,
    FileIdentity,
    authorization_request_digest,
)
from mochi.tools.file_transaction import (
    FileTransactionError,
    RecoveryObservation,
    StagedMutation,
    StagedRollback,
    classify_journal_entry,
    execute_durable_file_transaction,
    recover_incomplete_file_transactions,
)


def _identity(file_id: str) -> FileIdentity:
    return FileIdentity(
        platform="windows",
        volume_id="volume",
        file_id=file_id,
        link_count=1,
        is_reparse_point=False,
    )


def _envelope(*, after_sha256: str | None = None) -> AuthorizationEnvelope:
    zero = hashlib.sha256(b"").hexdigest()
    entry = ChangeEntry(
        entry_id="entry-1",
        relative_path="safe/note.txt",
        operation="update",
        base_sha256=zero,
        after_sha256=after_sha256 or hashlib.sha256(b"after").hexdigest(),
        base_identity=_identity("base"),
        before_blob_id=None,
        after_blob_id=None,
        mode_before=None,
        mode_after=None,
        base_metadata_sha256=zero,
        after_metadata_sha256=zero,
        rename_source=None,
        dependency_group=None,
    )
    return AuthorizationEnvelope(
        schema_version=1,
        kind="file_change",
        context=AuthorizationContext(
            requester_id="requester",
            session_id="session",
            task_id="task",
            workspace_root="C:/workspace",
            workspace_identity=_identity("workspace"),
        ),
        policy_version="policy-v1",
        file_request=FileChangeRequest(entries=(entry,), patch_sha256=None),
        exec_request=None,
    )


def _manifest(
    envelope: AuthorizationEnvelope,
    *,
    change_set_id: str = "change-1",
) -> ChangeManifest:
    request = envelope.file_request
    assert request is not None
    return ChangeManifest(
        version=1,
        change_set_id=change_set_id,
        workspace_root=envelope.context.workspace_root,
        workspace_identity=envelope.context.workspace_identity,
        tool_name="apply_patch",
        intent="mutate",
        entries=request.entries,
        patch_sha256=request.patch_sha256,
        policy_version=envelope.policy_version,
        created_at="2026-07-18T00:00:00+00:00",
        expires_at="2026-07-19T00:00:00+00:00",
        request_digest=authorization_request_digest(envelope),
    )


def test_runtime_store_migrates_task3_schema_idempotently(tmp_path: Path) -> None:
    db_path = tmp_path / "runtime.db"
    runtime_store = RuntimeStore(db_path)
    asyncio.run(runtime_store.initialize())

    expected_tables = {
        "change_sets",
        "change_entries",
        "applied_change_entries",
        "change_blobs",
        "blob_references",
        "undo_retention",
        "file_transaction_journal",
        "file_transaction_entries",
    }
    with sqlite3.connect(db_path) as conn:
        tables = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }
        assert expected_tables <= tables
        indexes = {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            )
        }
        assert "change_set_idempotency" in indexes

    runtime_store._initialized = False  # pyright: ignore[reportPrivateUsage]
    asyncio.run(runtime_store.initialize())
    with sqlite3.connect(db_path) as conn:
        assert expected_tables <= {
            str(row[0])
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
        }


def test_manifest_persistence_is_scoped_idempotent_and_conflict_safe(
    tmp_path: Path,
) -> None:
    runtime_store = RuntimeStore(tmp_path / "runtime.db")
    change_store = ChangeSetStore(runtime_store)
    envelope = _envelope()
    manifest = _manifest(envelope)

    first = asyncio.run(change_store.persist_manifest(manifest, envelope))
    second = asyncio.run(change_store.persist_manifest(manifest, envelope))
    assert first == second
    assert first["status"] == "prepared"
    assert first["manifest"] == manifest

    conflicting_envelope = _envelope(
        after_sha256=hashlib.sha256(b"different").hexdigest()
    )
    with pytest.raises(ChangeSetConflict, match="immutable"):
        asyncio.run(
            change_store.persist_manifest(
                _manifest(conflicting_envelope, change_set_id=manifest.change_set_id),
                conflicting_envelope,
            )
        )

    asyncio.run(
        change_store.mark_change_set_status(
            manifest.change_set_id,
            status="applied",
        )
    )
    replay = asyncio.run(
        change_store.persist_manifest(
            replace(manifest, change_set_id="change-replay"),
            envelope,
        )
    )
    assert replay["id"] == manifest.change_set_id
    assert replay["status"] == "applied"


def test_blob_references_and_gc_preserve_live_data(tmp_path: Path) -> None:
    runtime_store = RuntimeStore(tmp_path / "runtime.db")
    change_store = ChangeSetStore(runtime_store)
    now = datetime(2026, 7, 18, tzinfo=UTC)
    blob_id = asyncio.run(change_store.put_blob(b"authoritative-before"))
    assert blob_id == hashlib.sha256(b"authoritative-before").hexdigest()
    assert asyncio.run(change_store.put_blob(b"authoritative-before")) == blob_id

    asyncio.run(
        change_store.add_blob_reference(
            blob_id=blob_id,
            owner_type="change_entry",
            owner_id="change-1:entry-1",
            purpose="undo",
            retained_until=(now + timedelta(hours=1)).isoformat(),
        )
    )
    assert asyncio.run(change_store.collect_garbage(now=now.isoformat())) == ()
    assert asyncio.run(change_store.get_blob(blob_id)) == b"authoritative-before"

    expired_at = (now + timedelta(hours=2)).isoformat()
    assert asyncio.run(
        change_store.expire_blob_references(now=expired_at)
    ) == 1
    assert asyncio.run(
        change_store.collect_garbage(now=expired_at)
    ) == (blob_id,)
    assert asyncio.run(change_store.get_blob(blob_id)) is None


def test_journal_entries_round_trip_durable_identity_state(tmp_path: Path) -> None:
    runtime_store = RuntimeStore(tmp_path / "runtime.db")
    change_store = ChangeSetStore(runtime_store)
    envelope = _envelope()
    manifest = _manifest(envelope)
    asyncio.run(change_store.persist_manifest(manifest, envelope))

    entry = JournalEntryRecord(
        entry_id="entry-1",
        ordinal=0,
        state="staged",
        base_sha256=manifest.entries[0].base_sha256,
        after_sha256=manifest.entries[0].after_sha256,
        base_identity=manifest.entries[0].base_identity,
        staged_name=".mochi-stage-1",
        staged_identity=_identity("staged"),
        rollback_blob_id=None,
        rollback_staged_name=None,
        rollback_staged_identity=None,
        rollback_successor_identity=None,
        base_metadata_blob_id=None,
        last_error=None,
    )
    asyncio.run(
        change_store.create_journal(
            journal_id="journal-1",
            change_set_id=manifest.change_set_id,
            entries=(entry,),
        )
    )
    asyncio.run(
        change_store.update_journal_entry(
            "journal-1",
            "entry-1",
            state="applying",
        )
    )
    journal = asyncio.run(change_store.get_journal("journal-1"))
    assert journal is not None
    assert journal["status"] == "pending"
    assert journal["phase"] == "staged"
    assert journal["entries"][0].state == "applying"

    pending = asyncio.run(change_store.list_incomplete_journals())
    assert [item["id"] for item in pending] == ["journal-1"]


class _RecoveryAdapter:
    def __init__(
        self,
        observations: dict[str, RecoveryObservation],
    ) -> None:
        self.observations = observations
        self.discarded: list[str] = []
        self.rollback_staged: list[str] = []
        self.rolled_back: list[str] = []

    def observe(
        self,
        journal_id: str,
        entry: JournalEntryRecord,
    ) -> RecoveryObservation:
        return self.observations[entry.entry_id]

    def discard_staged(
        self,
        journal_id: str,
        entry: JournalEntryRecord,
    ) -> None:
        self.discarded.append(entry.entry_id)

    def stage_rollback(
        self,
        journal_id: str,
        entry: JournalEntryRecord,
        rollback_content: bytes,
    ) -> StagedRollback:
        self.rollback_staged.append(entry.entry_id)
        return StagedRollback(
            entry_id=entry.entry_id,
            staged_name=f".rollback-{entry.entry_id}",
            staged_identity=_identity(f"rollback-stage-{entry.entry_id}"),
        )

    def rollback(
        self,
        journal_id: str,
        entry: JournalEntryRecord,
        rollback_content: bytes,
    ) -> RecoveryObservation:
        self.rolled_back.append(entry.entry_id)
        assert entry.rollback_staged_identity is not None
        return RecoveryObservation(
            identity=entry.rollback_staged_identity,
            content_sha256=entry.base_sha256,
            metadata_sha256=entry.base_metadata_blob_id,
        )


def _recovery_entry(manifest: ChangeManifest) -> JournalEntryRecord:
    item = manifest.entries[0]
    return JournalEntryRecord(
        entry_id=item.entry_id,
        ordinal=0,
        state="applying",
        base_sha256=item.base_sha256,
        after_sha256=item.after_sha256,
        base_identity=item.base_identity,
        staged_name=".mochi-stage-1",
        staged_identity=_identity("staged"),
        rollback_blob_id=None,
        rollback_staged_name=None,
        rollback_staged_identity=None,
        rollback_successor_identity=_identity("rollback"),
        base_metadata_blob_id=item.base_metadata_sha256,
        last_error=None,
    )


@pytest.mark.parametrize(
    ("observation", "expected"),
    [
        (
            RecoveryObservation(
                identity=_identity("base"),
                content_sha256=hashlib.sha256(b"").hexdigest(),
                metadata_sha256=hashlib.sha256(b"").hexdigest(),
            ),
            "staged",
        ),
        (
            RecoveryObservation(
                identity=_identity("staged"),
                content_sha256=hashlib.sha256(b"after").hexdigest(),
                metadata_sha256=hashlib.sha256(b"").hexdigest(),
            ),
            "applied",
        ),
        (
            RecoveryObservation(
                identity=_identity("rollback"),
                content_sha256=hashlib.sha256(b"").hexdigest(),
                metadata_sha256=hashlib.sha256(b"").hexdigest(),
            ),
            "rolled_back",
        ),
        (
            RecoveryObservation(
                identity=_identity("foreign"),
                content_sha256=hashlib.sha256(b"after").hexdigest(),
                metadata_sha256=hashlib.sha256(b"").hexdigest(),
            ),
            "interference",
        ),
    ],
)
def test_recovery_classifies_identity_content_and_metadata_together(
    observation: RecoveryObservation,
    expected: str,
) -> None:
    envelope = _envelope()
    manifest = _manifest(envelope)
    assert classify_journal_entry(
        _recovery_entry(manifest),
        observation,
        after_metadata_sha256=manifest.entries[0].after_metadata_sha256,
    ) == expected


@pytest.mark.parametrize(
    ("identity_name", "content", "expected_status"),
    [
        ("base", b"", "rolled_back"),
        ("staged", b"after", "applied"),
        ("foreign", b"after", "interference"),
    ],
)
def test_recover_incomplete_transactions_converges_only_proven_states(
    tmp_path: Path,
    identity_name: str,
    content: bytes,
    expected_status: str,
) -> None:
    runtime_store = RuntimeStore(tmp_path / "runtime.db")
    change_store = ChangeSetStore(runtime_store)
    envelope = _envelope()
    manifest = _manifest(envelope)
    asyncio.run(change_store.persist_manifest(manifest, envelope))
    asyncio.run(change_store.put_blob(b""))
    entry = _recovery_entry(manifest)
    asyncio.run(
        change_store.create_journal(
            journal_id="journal-recovery",
            change_set_id=manifest.change_set_id,
            entries=(entry,),
        )
    )
    adapter = _RecoveryAdapter(
        {
            entry.entry_id: RecoveryObservation(
                identity=_identity(identity_name),
                content_sha256=hashlib.sha256(content).hexdigest(),
                metadata_sha256=hashlib.sha256(b"").hexdigest(),
            )
        }
    )

    recovered = asyncio.run(
        recover_incomplete_file_transactions(change_store, adapter)
    )
    assert recovered[0]["status"] == expected_status
    persisted = asyncio.run(change_store.get_journal("journal-recovery"))
    assert persisted is not None
    assert persisted["status"] == expected_status
    if expected_status == "rolled_back":
        assert adapter.discarded == ["entry-1"]



def test_runtime_service_runs_file_recovery_before_other_startup_work(
    tmp_path: Path,
) -> None:
    from mochi.runtime.service import RuntimeService

    order: list[str] = []

    class _ExecRuntime:
        async def recover_detached_sessions(self) -> None:
            order.append("exec")

    async def recover_files() -> None:
        order.append("files")

    async def scenario() -> None:
        service = RuntimeService(
            engine=object(),
            store=RuntimeStore(tmp_path / "runtime.db"),
            exec_runtime=_ExecRuntime(),  # type: ignore[arg-type]
            file_transaction_recovery=recover_files,
        )
        await service.start()
        await service.close()

    asyncio.run(scenario())
    assert order[:2] == ["files", "exec"]



class _TransactionParticipant:
    def __init__(
        self,
        entry: ChangeEntry,
        events: list[str],
        *,
        fail_commit: bool = False,
        bad_commit_observation: bool = False,
        fail_rollback: bool = False,
    ) -> None:
        self.entry = entry
        self.entry_id = entry.entry_id
        self.events = events
        self.fail_commit = fail_commit
        self.bad_commit_observation = bad_commit_observation
        self.fail_rollback = fail_rollback

    def stage(self) -> StagedMutation:
        self.events.append(f"stage:{self.entry.entry_id}")
        return StagedMutation(
            entry_id=self.entry.entry_id,
            staged_name=f".stage-{self.entry.entry_id}",
            staged_identity=_identity(f"staged-{self.entry.entry_id}"),
            rollback_content=b"",
            base_metadata=b"",
        )

    def validate(self, staged: StagedMutation) -> None:
        self.events.append(f"validate:{self.entry.entry_id}")

    def commit(self, staged: StagedMutation) -> RecoveryObservation:
        self.events.append(f"commit:{self.entry.entry_id}")
        if self.fail_commit:
            raise OSError(f"commit failed for {self.entry.entry_id}")
        return RecoveryObservation(
            identity=(
                _identity(f"foreign-{self.entry.entry_id}")
                if self.bad_commit_observation
                else staged.staged_identity
            ),
            content_sha256=self.entry.after_sha256,
            metadata_sha256=self.entry.after_metadata_sha256,
        )

    def stage_rollback(
        self,
        staged: StagedMutation,
        rollback_content: bytes,
    ) -> StagedRollback:
        self.events.append(f"stage_rollback:{self.entry.entry_id}")
        return StagedRollback(
            entry_id=self.entry.entry_id,
            staged_name=f".rollback-{self.entry.entry_id}",
            staged_identity=_identity(
                f"rollback-stage-{self.entry.entry_id}"
            ),
        )

    def rollback(
        self,
        staged: StagedMutation,
        rollback_staged: StagedRollback,
        rollback_content: bytes,
    ) -> RecoveryObservation:
        self.events.append(f"rollback:{self.entry.entry_id}")
        if self.fail_rollback:
            raise OSError(f"rollback failed for {self.entry.entry_id}")
        return RecoveryObservation(
            identity=rollback_staged.staged_identity,
            content_sha256=self.entry.base_sha256,
            metadata_sha256=self.entry.base_metadata_sha256,
        )

    def discard(self, staged: StagedMutation) -> None:
        self.events.append(f"discard:{self.entry.entry_id}")


def _two_entry_manifest(
    *,
    change_set_id: str,
) -> tuple[AuthorizationEnvelope, ChangeManifest]:
    envelope = _envelope()
    request = envelope.file_request
    assert request is not None
    first = request.entries[0]
    second = replace(
        first,
        entry_id="entry-2",
        relative_path="safe/second.txt",
        base_identity=_identity("base-2"),
    )
    request = replace(request, entries=(first, second))
    envelope = replace(envelope, file_request=request)
    return envelope, _manifest(envelope, change_set_id=change_set_id)


def test_durable_transaction_stages_validates_then_commits_all(
    tmp_path: Path,
) -> None:
    runtime_store = RuntimeStore(tmp_path / "runtime.db")
    change_store = ChangeSetStore(runtime_store)
    envelope, manifest = _two_entry_manifest(change_set_id="change-success")
    asyncio.run(change_store.persist_manifest(manifest, envelope))
    events: list[str] = []
    participants = tuple(
        _TransactionParticipant(entry, events) for entry in manifest.entries
    )

    result = asyncio.run(
        execute_durable_file_transaction(
            change_store,
            manifest,
            journal_id="journal-success",
            participants=participants,
        )
    )
    assert result["status"] == "applied"
    assert events == [
        "stage:entry-1",
        "stage:entry-2",
        "validate:entry-1",
        "validate:entry-2",
        "commit:entry-1",
        "commit:entry-2",
    ]


def test_durable_transaction_rolls_back_committed_prefix_in_reverse(
    tmp_path: Path,
) -> None:
    runtime_store = RuntimeStore(tmp_path / "runtime.db")
    change_store = ChangeSetStore(runtime_store)
    envelope, manifest = _two_entry_manifest(change_set_id="change-failure")
    asyncio.run(change_store.persist_manifest(manifest, envelope))
    events: list[str] = []
    participants = (
        _TransactionParticipant(manifest.entries[0], events),
        _TransactionParticipant(
            manifest.entries[1],
            events,
            fail_commit=True,
        ),
    )

    with pytest.raises(FileTransactionError, match="rolled back"):
        asyncio.run(
            execute_durable_file_transaction(
                change_store,
                manifest,
                journal_id="journal-failure",
                participants=participants,
            )
        )
    journal = asyncio.run(change_store.get_journal("journal-failure"))
    assert journal is not None
    assert journal["status"] == "rolled_back"
    rolled_back_entry = journal["entries"][0]
    assert rolled_back_entry.rollback_staged_name == ".rollback-entry-1"
    assert rolled_back_entry.rollback_staged_identity == _identity(
        "rollback-stage-entry-1"
    )
    assert events[-3:] == [
        "discard:entry-2",
        "stage_rollback:entry-1",
        "rollback:entry-1",
    ]


def test_durable_transaction_rejects_journal_replay_without_reactivating_refs(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "runtime.db"
    runtime_store = RuntimeStore(db_path)
    change_store = ChangeSetStore(runtime_store)
    envelope, manifest = _two_entry_manifest(change_set_id="change-replay")
    asyncio.run(change_store.persist_manifest(manifest, envelope))
    first_events: list[str] = []
    asyncio.run(
        execute_durable_file_transaction(
            change_store,
            manifest,
            journal_id="journal-replay",
            participants=tuple(
                _TransactionParticipant(entry, first_events)
                for entry in manifest.entries
            ),
        )
    )

    second_events: list[str] = []
    with pytest.raises(FileTransactionError, match="journal"):
        asyncio.run(
            execute_durable_file_transaction(
                change_store,
                manifest,
                journal_id="journal-replay",
                participants=tuple(
                    _TransactionParticipant(entry, second_events)
                    for entry in manifest.entries
                ),
            )
        )
    assert not any(event.startswith("commit:") for event in second_events)
    with sqlite3.connect(db_path) as conn:
        states = {
            str(row[0])
            for row in conn.execute(
                "SELECT state FROM blob_references "
                "WHERE owner_type='file_transaction'"
            )
        }
    assert states == {"released"}


def test_commit_verification_mismatch_fails_closed_as_interference(
    tmp_path: Path,
) -> None:
    runtime_store = RuntimeStore(tmp_path / "runtime.db")
    change_store = ChangeSetStore(runtime_store)
    envelope, manifest = _two_entry_manifest(change_set_id="change-interference")
    asyncio.run(change_store.persist_manifest(manifest, envelope))
    events: list[str] = []
    participants = (
        _TransactionParticipant(manifest.entries[0], events),
        _TransactionParticipant(
            manifest.entries[1],
            events,
            bad_commit_observation=True,
        ),
    )

    with pytest.raises(FileTransactionError) as raised:
        asyncio.run(
            execute_durable_file_transaction(
                change_store,
                manifest,
                journal_id="journal-interference",
                participants=participants,
            )
        )
    assert raised.value.status == "interference"
    assert not any(
        event.startswith(("discard:", "rollback:")) for event in events
    )
    journal = asyncio.run(change_store.get_journal("journal-interference"))
    assert journal is not None
    assert journal["status"] == "interference"
    assert journal["entries"][1].state == "interference"


def test_terminal_finalize_sets_applied_at_and_releases_all_transaction_refs(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "runtime.db"
    runtime_store = RuntimeStore(db_path)
    change_store = ChangeSetStore(runtime_store)
    envelope, manifest = _two_entry_manifest(change_set_id="change-finalize")
    asyncio.run(change_store.persist_manifest(manifest, envelope))
    asyncio.run(
        execute_durable_file_transaction(
            change_store,
            manifest,
            journal_id="journal-finalize",
            participants=tuple(
                _TransactionParticipant(entry, [])
                for entry in manifest.entries
            ),
        )
    )

    persisted = asyncio.run(change_store.get_change_set(manifest.change_set_id))
    assert persisted is not None
    assert persisted["status"] == "applied"
    assert persisted["applied_at"] is not None
    with sqlite3.connect(db_path) as conn:
        assert {
            str(row[0])
            for row in conn.execute(
                "SELECT state FROM blob_references "
                "WHERE owner_type='file_transaction'"
            )
        } == {"released"}


def test_rollback_failure_retains_transaction_refs_for_recovery(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "runtime.db"
    runtime_store = RuntimeStore(db_path)
    change_store = ChangeSetStore(runtime_store)
    envelope, manifest = _two_entry_manifest(
        change_set_id="change-rollback-failure"
    )
    asyncio.run(change_store.persist_manifest(manifest, envelope))
    participants = (
        _TransactionParticipant(
            manifest.entries[0],
            [],
            fail_rollback=True,
        ),
        _TransactionParticipant(
            manifest.entries[1],
            [],
            fail_commit=True,
        ),
    )

    with pytest.raises(FileTransactionError) as raised:
        asyncio.run(
            execute_durable_file_transaction(
                change_store,
                manifest,
                journal_id="journal-rollback-failure",
                participants=participants,
            )
        )
    assert raised.value.status == "rollback_failed"
    with sqlite3.connect(db_path) as conn:
        assert {
            str(row[0])
            for row in conn.execute(
                "SELECT state FROM blob_references "
                "WHERE owner_type='file_transaction'"
            )
        } == {"active"}
