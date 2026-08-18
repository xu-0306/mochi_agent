"""MemoryStore 單元測試。"""

from __future__ import annotations

import asyncio
import json
import sqlite3
from collections.abc import Coroutine
from pathlib import Path
from typing import Any, TypeVar

import pytest

from mochi.memory.store import MemoryStore

T = TypeVar("T")


def _run(coro: Coroutine[Any, Any, T]) -> T:
    """同步測試中執行 async coroutine。"""
    return asyncio.run(coro)


def test_save_and_search_returns_structured_entries(tmp_path: Path) -> None:
    """save/search 應回傳包含必要欄位的結構化資料。"""
    store = MemoryStore(db_path=tmp_path / "memory.db")

    expected_id = _run(
        store.save(
            content="我喜歡烏龍茶，早上會喝。",
            category="preference",
            metadata={"source": "chat", "tags": ["drink", "tea"]},
        )
    )
    _run(
        store.save(
            content="明天要買牛奶和雞蛋。",
            category="todo",
            metadata={"source": "chat", "priority": "high"},
        )
    )

    results = _run(store.search("烏龍", top_k=5))
    assert results, "搜尋結果不應為空"

    first = results[0]
    assert {"id", "content", "category", "metadata", "created_at"} <= set(first)
    assert first["id"] == expected_id
    assert first["category"] == "preference"
    assert isinstance(first["metadata"], dict)
    assert first["metadata"]["source"] == "chat"


def test_search_supports_category_and_metadata_queries(tmp_path: Path) -> None:
    """category 與 metadata 內容也應可被搜尋到。"""
    store = MemoryStore(db_path=tmp_path / "memory.db")
    expected_id = _run(
        store.save(
            content="晚餐想做咖哩飯。",
            category="food-note",
            metadata={"project": "alpha-mochi"},
        )
    )

    category_hits = _run(store.search("food-note", top_k=5))
    assert any(item["id"] == expected_id for item in category_hits)

    metadata_hits = _run(store.search("alpha-mochi", top_k=5))
    assert any(item["id"] == expected_id for item in metadata_hits)


def test_user_profile_update_and_get(tmp_path: Path) -> None:
    """使用者模型應可讀寫並保留更新內容。"""
    store = MemoryStore(db_path=tmp_path / "memory.db")

    assert _run(store.get_user_profile()) == ""

    _run(store.update_user_profile("偏好簡潔回答。"))
    _run(store.update_user_profile("主要使用繁體中文。"))

    profile = _run(store.get_user_profile())
    assert "偏好簡潔回答。" in profile
    assert "主要使用繁體中文。" in profile
    assert profile.index("偏好簡潔回答。") < profile.index("主要使用繁體中文。")


def test_db_path_persists_between_instances(tmp_path: Path) -> None:
    """同一路徑的不同 MemoryStore 實例應能共享資料。"""
    db_path = tmp_path / "memory.db"
    first_store = MemoryStore(db_path=db_path)
    expected_id = _run(
        first_store.save(
            content="記得每週檢查備份。",
            category="ops",
            metadata={"team": "platform"},
        )
    )

    second_store = MemoryStore(db_path=db_path)
    results = _run(second_store.search("備份", top_k=3))
    assert any(item["id"] == expected_id for item in results)


def test_default_db_path_reads_from_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """未傳入 db_path 時應使用 config.memory.db_path。"""
    expected_path = tmp_path / "memory-from-config.db"
    monkeypatch.setattr(
        MemoryStore,
        "_resolve_default_db_path",
        lambda _self: expected_path,
    )

    store = MemoryStore()
    _run(
        store.save(
            content="設定來源測試",
            category="config",
            metadata={"origin": "config"},
        )
    )

    assert store._db_path == expected_path.expanduser()
    assert store._db_path.exists()


def test_fallback_to_like_when_fts5_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """FTS5 不可用時，仍應能透過 LIKE fallback 搜尋。"""

    def fake_create_fts_table(_self: MemoryStore, _conn: sqlite3.Connection) -> None:
        raise sqlite3.OperationalError("no such module: fts5")

    monkeypatch.setattr(MemoryStore, "_create_fts_table", fake_create_fts_table)

    store = MemoryStore(db_path=tmp_path / "memory.db")
    expected_id = _run(
        store.save(
            content="這筆資料要走 fallback 查詢。",
            category="fallback",
            metadata={"mode": "like"},
        )
    )

    assert store._supports_fts5 is False
    hits = _run(store.search("fallback", top_k=5))
    assert any(item["id"] == expected_id for item in hits)


def test_update_delete_and_export_roundtrip(tmp_path: Path) -> None:
    store = MemoryStore(db_path=tmp_path / "memory.db")
    entry_id = _run(
        store.save(
            content="draft memory",
            category="notes",
            metadata={"source": "test"},
        )
    )
    _run(
        store.save(
            content="second memory",
            category="other",
            metadata={"source": "test"},
        )
    )

    updated = _run(
        store.update(
            entry_id,
            content="final memory",
            category="archive",
            metadata={"source": "updated"},
        )
    )
    assert updated is not None
    assert updated["content"] == "final memory"
    assert updated["category"] == "archive"
    assert updated["metadata"]["source"] == "updated"

    exported = _run(store.export(category="archive"))
    assert len(exported) == 1
    assert exported[0]["id"] == entry_id

    deleted = _run(store.delete(entry_id))
    assert deleted is True

    exported_after_delete = _run(store.export())
    assert all(item["id"] != entry_id for item in exported_after_delete)


def test_v2_memory_records_preserve_governance_and_history(tmp_path: Path) -> None:
    store = MemoryStore(db_path=tmp_path / "memory.db")
    memory_id = _run(
        store.save(
            content="initial governed memory",
            category="note",
            metadata={"source": "test"},
            namespace="project-alpha",
            provenance={"run_id": "run-1"},
        )
    )

    initial = _run(store.export())[0]
    assert initial["namespace"] == "project-alpha"
    assert initial["provenance"] == {"run_id": "run-1"}
    assert initial["revision"] == 1
    assert initial["supersedes_id"] is None
    assert len(initial["checksum"]) == 64

    updated = _run(store.update(memory_id, content="revised governed memory"))
    assert updated is not None
    assert updated["revision"] == 2
    history = _run(store.get_history(memory_id))
    assert [item["revision"] for item in history] == [1, 2]
    assert history[0]["content"] == "initial governed memory"
    assert history[1]["content"] == "revised governed memory"


def test_backup_restore_validates_before_replacing_active_memory(tmp_path: Path) -> None:
    store = MemoryStore(db_path=tmp_path / "memory.db")
    first_id = _run(
        store.save(
            content="first backup record",
            category="note",
            metadata={"source": "test"},
        )
    )
    _run(store.update_user_profile("profile before backup"))
    backup = _run(store.backup())

    _run(
        store.save(
            content="later record",
            category="note",
            metadata={"source": "test"},
        )
    )
    dry_run = _run(store.restore(backup, dry_run=True))
    assert dry_run["status"] == "validated"
    assert len(_run(store.export())) == 2
    before_invalid_restore_ids = {item["id"] for item in _run(store.export())}

    invalid_backup = json.loads(json.dumps(backup))
    invalid_backup["records"][0]["checksum"] = "0" * 64
    with pytest.raises(ValueError, match="checksum"):
        _run(store.restore(invalid_backup))
    assert {item["id"] for item in _run(store.export())} == before_invalid_restore_ids
    assert _run(store.get_user_profile()) == "profile before backup"

    restored = _run(store.restore(backup))
    assert restored["status"] == "restored"
    restored_records = _run(store.export())
    assert [item["id"] for item in restored_records] == [first_id]
    assert _run(store.get_user_profile()) == "profile before backup"


def test_backup_rejects_tampered_persisted_record_checksum(tmp_path: Path) -> None:
    db_path = tmp_path / "memory.db"
    store = MemoryStore(db_path=db_path)
    memory_id = _run(
        store.save(
            content="canonical record",
            category="note",
            metadata={"source": "test"},
        )
    )

    with sqlite3.connect(db_path) as connection:
        connection.execute("UPDATE memories SET checksum = ? WHERE id = ?", ("0" * 64, memory_id))

    with pytest.raises(ValueError, match="checksum"):
        _run(store.backup())


def test_restore_rejects_active_record_that_conflicts_with_its_retained_history(tmp_path: Path) -> None:
    store = MemoryStore(db_path=tmp_path / "memory.db")
    _run(store.save(content="canonical record", category="note", metadata={"source": "test"}))
    backup = _run(store.backup())
    conflicting_history = backup["history"][0]
    conflicting_history["content"] = "different retained snapshot"
    conflicting_history["checksum"] = store._checksum(conflicting_history)

    with pytest.raises(ValueError, match="conflicts"):
        _run(store.restore(backup))


def test_restore_rolls_back_when_fts_rebuild_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = MemoryStore(db_path=tmp_path / "memory.db")
    original_id = _run(store.save(content="original record", category="note", metadata={"source": "test"}))
    backup = _run(store.backup())
    drift_id = _run(store.save(content="active drift", category="note", metadata={"source": "test"}))

    def fail_rebuild(_connection: sqlite3.Connection, *, fail_closed: bool = False) -> None:
        assert fail_closed is True
        raise sqlite3.OperationalError("injected fts rebuild failure")

    monkeypatch.setattr(store, "_rebuild_fts_sync", fail_rebuild)
    with pytest.raises(sqlite3.OperationalError, match="fts rebuild"):
        _run(store.restore(backup))

    assert {item["id"] for item in _run(store.export())} == {original_id, drift_id}


def test_legacy_memory_schema_migrates_to_v2_records(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy-memory.db"
    connection = sqlite3.connect(db_path)
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
        VALUES ('legacy-id', 'legacy content', 'legacy', '{"origin":"old"}', '2026-08-01T00:00:00+00:00')
        """
    )
    connection.commit()
    connection.close()

    store = MemoryStore(db_path=db_path)
    migrated = _run(store.export())

    assert len(migrated) == 1
    assert migrated[0]["id"] == "legacy-id"
    assert migrated[0]["namespace"] == "default"
    assert migrated[0]["provenance"] == {}
    assert migrated[0]["revision"] == 1
    assert migrated[0]["updated_at"] == migrated[0]["created_at"]
    assert len(migrated[0]["checksum"]) == 64
