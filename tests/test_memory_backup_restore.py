"""SQLite/FTS qualification for versioned MemoryStore backup and restore."""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any, TypeVar

import pytest

from mochi.memory.store import MemoryEntry, MemoryStore

T = TypeVar("T")


def _run(coro: Coroutine[Any, Any, T]) -> T:
    return asyncio.run(coro)


def _copy_backup(backup: dict[str, Any]) -> dict[str, Any]:
    return json.loads(json.dumps(backup))


def _active_store_snapshot(db_path: Path) -> tuple[bytes, tuple[Any, ...], tuple[Any, ...], tuple[Any, ...]]:
    """Capture physical and canonical state after every rejected restore."""

    with sqlite3.connect(db_path) as connection:
        records = tuple(
            connection.execute(
                """
                SELECT id, content, category, metadata_json, created_at, namespace,
                       provenance_json, revision, supersedes_id, updated_at, checksum
                FROM memories
                ORDER BY id ASC
                """
            )
        )
        history = tuple(
            connection.execute(
                """
                SELECT memory_id, revision, record_json, recorded_at
                FROM memory_revisions
                ORDER BY memory_id ASC, revision ASC
                """
            )
        )
        profile = tuple(
            connection.execute("SELECT id, content, updated_at FROM user_profile ORDER BY id ASC")
        )
    return db_path.read_bytes(), records, history, profile


def _versioned_backup(store: MemoryStore) -> tuple[dict[str, Any], str]:
    memory_id = _run(
        store.save(
            content="quasar source record",
            category="research",
            metadata={"source": "backup-fixture"},
            namespace="research",
            provenance={"run_id": "run-backup"},
        )
    )
    _run(store.update(memory_id, content="quasar source record revised"))
    _run(store.update_user_profile("profile captured with backup"))
    return _run(store.backup()), memory_id


def _assert_real_fts(store: MemoryStore) -> None:
    # This package qualifies the FTS-derived path rather than the LIKE fallback.
    _run(store.export())
    assert store._supports_fts5 is True


def test_v2_checksum_backup_dry_run_reports_counts_without_mutating_sqlite(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    store = MemoryStore(db_path=db_path)
    _assert_real_fts(store)
    backup, source_id = _versioned_backup(store)

    assert backup["schema_version"] == 2
    assert backup["records"][0]["id"] == source_id
    assert len(backup["records"][0]["checksum"]) == 64
    assert backup["history"][-1]["checksum"] == backup["records"][0]["checksum"]

    _run(
        store.save(
            content="active drift must survive dry run",
            category="active",
            metadata={"source": "active-store"},
        )
    )
    _run(store.update_user_profile("active profile tail"))
    before = _active_store_snapshot(db_path)

    report = _run(store.restore(backup, dry_run=True))

    assert report == {"status": "validated", "records": 1, "history": 2, "user_profile": True}
    assert _active_store_snapshot(db_path) == before


def test_tampered_persisted_checksum_rejects_backup_without_mutation(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    store = MemoryStore(db_path=db_path)
    _assert_real_fts(store)
    _, memory_id = _versioned_backup(store)

    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE memories SET checksum = ? WHERE id = ?", ("0" * 64, memory_id))
    before = _active_store_snapshot(db_path)

    with pytest.raises(ValueError, match="checksum"):
        _run(store.backup())

    assert _active_store_snapshot(db_path) == before


@pytest.mark.parametrize(
    ("name", "mutate", "error"),
    [
        (
            "checksum-corruption",
            lambda backup: backup["records"][0].__setitem__("checksum", "0" * 64),
            "checksum",
        ),
        (
            "malformed-provenance",
            lambda backup: backup["records"][0].__setitem__("provenance", []),
            "metadata and provenance",
        ),
        (
            "unknown-schema",
            lambda backup: backup.__setitem__("schema_version", 3),
            "schema version",
        ),
        (
            "duplicate-id-conflict",
            lambda backup: backup["records"].append(_copy_backup(backup)["records"][0]),
            "duplicate ids",
        ),
    ],
    ids=lambda case: case if isinstance(case, str) else None,
)
def test_corrupt_or_conflicting_backup_leaves_sqlite_bytes_and_rows_unchanged(
    tmp_path: Path,
    name: str,
    mutate: Callable[[dict[str, Any]], None],
    error: str,
) -> None:
    db_path = tmp_path / f"{name}.db"
    store = MemoryStore(db_path=db_path)
    _assert_real_fts(store)
    backup, _ = _versioned_backup(store)
    _run(
        store.save(
            content="active state before rejected restore",
            category="active",
            metadata={"fixture": name},
        )
    )
    _run(store.update_user_profile("active profile after backup"))
    before = _active_store_snapshot(db_path)
    invalid_backup = _copy_backup(backup)
    mutate(invalid_backup)

    with pytest.raises(ValueError, match=error):
        _run(store.restore(invalid_backup))

    assert _active_store_snapshot(db_path) == before


def test_legacy_sqlite_row_migrates_to_checksummed_v2_backup(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy-memory.db"
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE memories (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                category TEXT NOT NULL,
                metadata_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO memories (id, content, category, metadata_json, created_at)
            VALUES ('legacy-id', 'legacy quasar archive', 'legacy', '{"origin":"v1"}',
                    '2026-08-01T00:00:00+00:00')
            """
        )

    store = MemoryStore(db_path=db_path)
    _assert_real_fts(store)
    backup = _run(store.backup())

    assert backup["schema_version"] == 2
    assert backup["records"] == backup["history"]
    record = backup["records"][0]
    assert record["namespace"] == "default"
    assert record["provenance"] == {}
    assert record["revision"] == 1
    assert record["updated_at"] == record["created_at"]
    assert len(record["checksum"]) == 64


def test_pre_swap_insert_failure_rolls_back_active_sqlite_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "memory.db"
    store = MemoryStore(db_path=db_path)
    _assert_real_fts(store)
    backup, _ = _versioned_backup(store)
    _run(
        store.save(
            content="second backup row forces a mid-transaction insertion failure",
            category="research",
            metadata={"source": "backup-fixture"},
        )
    )
    backup = _run(store.backup())
    _run(
        store.save(
            content="active row that must be restored after rollback",
            category="active",
            metadata={"fixture": "pre-swap"},
        )
    )
    _run(store.update_user_profile("active profile before injected failure"))
    before = _active_store_snapshot(db_path)
    original_insert = store._insert_memory_row
    insert_count = 0

    def fail_before_second_insert(connection: sqlite3.Connection, entry: MemoryEntry) -> None:
        nonlocal insert_count
        insert_count += 1
        if insert_count == 2:
            raise sqlite3.OperationalError("injected pre-swap insertion failure")
        original_insert(connection, entry)

    monkeypatch.setattr(store, "_insert_memory_row", fail_before_second_insert)

    with pytest.raises(sqlite3.OperationalError, match="pre-swap"):
        _run(store.restore(backup))

    assert insert_count == 2
    assert _active_store_snapshot(db_path) == before


def test_fts_rebuild_failure_rolls_back_active_sqlite_store(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    db_path = tmp_path / "memory.db"
    store = MemoryStore(db_path=db_path)
    _assert_real_fts(store)
    backup, _ = _versioned_backup(store)
    drift_id = _run(
        store.save(
            content="active drift must survive an FTS rebuild failure",
            category="active",
            metadata={"fixture": "fts-rebuild-failure"},
        )
    )
    _run(store.update_user_profile("active profile before FTS failure"))
    before = _active_store_snapshot(db_path)
    original_rebuild = store._rebuild_fts_sync
    rebuild_fail_closed: list[bool] = []

    def drop_fts_during_rebuild(connection: sqlite3.Connection, *, fail_closed: bool = False) -> None:
        rebuild_fail_closed.append(fail_closed)
        connection.execute("DROP TABLE memories_fts")
        original_rebuild(connection, fail_closed=fail_closed)

    monkeypatch.setattr(store, "_rebuild_fts_sync", drop_fts_during_rebuild)

    with pytest.raises(sqlite3.OperationalError, match="no such table"):
        _run(store.restore(backup))

    assert rebuild_fail_closed == [True]
    assert _active_store_snapshot(db_path) == before
    assert [record["id"] for record in _run(store.search("active drift"))] == [drift_id]


def test_successful_restore_replaces_canonical_rows_and_rebuilds_fts(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    store = MemoryStore(db_path=db_path)
    _assert_real_fts(store)
    backup, source_id = _versioned_backup(store)
    stale_id = _run(
        store.save(
            content="obsolete meteor record",
            category="active",
            metadata={"fixture": "stale"},
        )
    )

    report = _run(store.restore(backup))

    assert report == {"status": "restored", "records": 1, "history": 2, "user_profile": True}
    assert [record["id"] for record in _run(store.export())] == [source_id]
    assert _run(store.get_user_profile()) == "profile captured with backup"
    assert all(record["id"] != stale_id for record in _run(store.search("obsolete meteor")))
    assert [record["id"] for record in _run(store.search("quasar revised"))] == [source_id]

    with sqlite3.connect(db_path) as connection:
        fts_ids = [
            row[0]
            for row in connection.execute(
                "SELECT id FROM memories_fts WHERE memories_fts MATCH ?", ('"quasar"',)
            )
        ]
    assert fts_ids == [source_id]
