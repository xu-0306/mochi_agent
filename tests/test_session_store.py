"""SessionStore 單元測試。"""

from __future__ import annotations

import asyncio
import json

from mochi.sessions.store import SessionStore


def test_save_and_load_session_round_trip(tmp_path) -> None:
    """save_event() 與 load_session() 應可正確往返資料。"""
    store = SessionStore(tmp_path / "sessions")
    session_id = "test-session"

    asyncio.run(store.save_event(session_id, {"type": "user", "content": "hello"}))
    asyncio.run(store.save_event(session_id, {"type": "assistant", "content": "world"}))

    events = asyncio.run(store.load_session(session_id))
    assert events == [
        {"type": "user", "content": "hello"},
        {"type": "assistant", "content": "world"},
    ]


def test_save_event_creates_directory_automatically(tmp_path) -> None:
    """save_event() 應自動建立不存在的 sessions 目錄。"""
    sessions_dir = tmp_path / "nested" / "sessions"
    store = SessionStore(sessions_dir)

    asyncio.run(store.save_event("abc", {"ok": True}))

    assert sessions_dir.exists()
    loaded = asyncio.run(store.load_session("abc"))
    assert loaded == [{"ok": True}]


def test_load_session_tolerates_bad_jsonl_lines(tmp_path) -> None:
    """load_session() 應跳過壞行與非 dict JSON。"""
    store = SessionStore(tmp_path / "sessions")
    session_id = "broken-data"
    path = store._session_path(session_id)  # noqa: SLF001
    path.parent.mkdir(parents=True, exist_ok=True)

    valid_1 = json.dumps({"type": "user", "content": "ok-1"}, ensure_ascii=False)
    invalid = '{"type": "assistant", bad-json'
    non_dict = json.dumps(["not", "object"], ensure_ascii=False)
    valid_2 = json.dumps({"type": "assistant", "content": "ok-2"}, ensure_ascii=False)
    path.write_text(f"{valid_1}\n{invalid}\n{non_dict}\n{valid_2}\n", encoding="utf-8")

    events = asyncio.run(store.load_session(session_id))
    assert events == [
        {"type": "user", "content": "ok-1"},
        {"type": "assistant", "content": "ok-2"},
    ]


def test_load_session_returns_empty_when_file_missing(tmp_path) -> None:
    """load_session() 對不存在 session 檔案應回傳空列表。"""
    store = SessionStore(tmp_path / "sessions")
    events = asyncio.run(store.load_session("missing"))
    assert events == []


def test_session_exists_and_delete_session(tmp_path) -> None:
    """session_exists() 與 delete_session() 應反映檔案狀態。"""
    store = SessionStore(tmp_path / "sessions")
    asyncio.run(store.save_event("delete-me", {"type": "message", "content": "x"}))

    assert asyncio.run(store.session_exists("delete-me")) is True
    assert asyncio.run(store.delete_session("delete-me")) is True
    assert asyncio.run(store.session_exists("delete-me")) is False
    assert asyncio.run(store.load_session("delete-me")) == []
    assert asyncio.run(store.delete_session("delete-me")) is False
