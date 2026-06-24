"""MemoryStore 單元測試。"""

from __future__ import annotations

import asyncio
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
