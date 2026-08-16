"""SessionStore 單元測試。"""

from __future__ import annotations

import asyncio
import errno
import json
import os

import pytest

from mochi.agents.failures import FailureEnvelopeVersionError
from mochi.sessions.index import SessionSearchIndexUnavailableError
from mochi.sessions.store import SessionIdentityConflictError, SessionStore


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


def test_append_event_if_flushes_and_returns_after_a_successful_durable_append(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    event = {"type": "message", "content": "claimed"}

    assert asyncio.run(store.append_event_if("conditional", event, lambda events: not events)) is True
    assert asyncio.run(store.load_session("conditional")) == [event]
    assert asyncio.run(store.append_event_if("conditional", event, lambda _events: False)) is False


def test_session_store_refreshes_rebuildable_search_index_after_source_commits(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session_id = "indexed-session"

    asyncio.run(store.save_event(session_id, {"type": "user", "turn_id": "first", "content": "old phrase"}))
    assert [hit.turn_id for hit in asyncio.run(store.session_search_index.search("old phrase"))] == [
        "first"
    ]

    asyncio.run(store.replace_session(session_id, [{"type": "assistant", "turn_id": "second", "content": "new phrase"}]))
    assert asyncio.run(store.session_search_index.search("old phrase")) == ()
    assert [hit.turn_id for hit in asyncio.run(store.session_search_index.search("new phrase"))] == [
        "second"
    ]

    assert asyncio.run(store.delete_session(session_id)) is True
    assert asyncio.run(store.session_search_index.search("new phrase")) == ()


def test_strict_cas_commit_refreshes_the_same_derived_index(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session_id = "strict-indexed"
    before = asyncio.run(store.load_strict_snapshot(session_id))

    result = asyncio.run(
        store.append_strict_batch_if_revision(
            session_id,
            expected_history_revision=before.history_revision,
            events=({"type": "assistant", "turn_id": "strict-turn", "content": "strict phrase"},),
        )
    )

    assert result.status == "appended"
    assert [hit.turn_id for hit in asyncio.run(store.session_search_index.search("strict phrase"))] == [
        "strict-turn"
    ]


def test_rebuild_session_search_index_uses_existing_strict_jsonl_sources(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    asyncio.run(store.save_event("alpha", {"type": "user", "content": "first rebuild token"}))
    asyncio.run(store.save_event("beta", {"type": "assistant", "content": "second rebuild token"}))
    store.session_search_index.database_path.unlink()

    report = asyncio.run(store.rebuild_session_search_index())

    assert report.session_count == 2
    assert [hit.session_id for hit in asyncio.run(store.session_search_index.search("rebuild token"))] == [
        "alpha",
        "beta",
    ]


def test_index_failure_never_rolls_back_a_successful_jsonl_source_commit(tmp_path, monkeypatch) -> None:
    store = SessionStore(tmp_path / "sessions")

    async def fail_refresh(*_args, **_kwargs) -> None:
        raise RuntimeError("simulated index I/O failure")

    monkeypatch.setattr(store.session_search_index, "replace_snapshot", fail_refresh)
    asyncio.run(store.save_event("source-wins", {"type": "user", "content": "durable source"}))

    assert asyncio.run(store.load_session("source-wins")) == [
        {"type": "user", "content": "durable source"},
    ]
    with pytest.raises(SessionSearchIndexUnavailableError):
        asyncio.run(store.session_search_index.search("durable source"))


def test_session_store_versions_legacy_terminal_errors_without_removing_legacy_fields(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    event = {
        "type": "turn_event",
        "turn_id": "turn-failure",
        "payload": {
            "type": "error",
            "error": "restricted provider detail",
            "code": "MODEL_REQUEST_FAILED",
            "metadata": {"error_type": "backend_request_error"},
        },
    }

    asyncio.run(store.save_event("failure-session", event))

    persisted = asyncio.run(store.load_session("failure-session"))[0]
    payload = persisted["payload"]
    assert payload["error"] == "restricted provider detail"
    assert payload["code"] == "MODEL_REQUEST_FAILED"
    assert payload["failure"]["schema_version"] == "1.0"
    assert payload["failure"]["kind"] == "backend_error"
    assert payload["failure"]["terminal"] is True


def test_session_store_rejects_an_unknown_failure_major(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    unknown_major = {
        "type": "error",
        "failure": {
            "schema_version": "2.0",
            "kind": "backend_error",
            "origin": "backend",
            "recoverability": "manual_retry",
            "retry_policy": "manual",
            "terminal": True,
            "inject_into_model_context": False,
            "telemetry_key": "failure.backend_error",
            "ui_hint": "retry",
            "diagnostics_ref": None,
        },
    }

    with pytest.raises(FailureEnvelopeVersionError):
        asyncio.run(store.save_event("failure-session", unknown_major))


def test_atomic_replace_retries_a_transient_destination_lock(tmp_path, monkeypatch) -> None:
    store = SessionStore(tmp_path / "sessions")
    session_id = "transient-lock"
    asyncio.run(store.save_event(session_id, {"type": "user", "content": "before"}))
    real_replace = os.replace
    attempts = 0
    delays: list[float] = []

    def replace_after_transient_lock(source, target) -> None:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise PermissionError(13, "destination is temporarily locked", str(target))
        real_replace(source, target)

    monkeypatch.setattr("mochi.sessions.store.os.replace", replace_after_transient_lock)
    monkeypatch.setattr("mochi.sessions.store.time.sleep", delays.append)

    replacement = [{"type": "assistant", "content": "after"}]
    asyncio.run(store.replace_session(session_id, replacement))

    assert attempts == 3
    assert delays == [0.01, 0.02]
    assert asyncio.run(store.load_session(session_id)) == replacement


@pytest.mark.skipif(os.name != "nt", reason="Windows sidecar locking behavior")
def test_windows_sidecar_lock_retries_nonblocking_contention(tmp_path, monkeypatch) -> None:
    import msvcrt

    lock_path = tmp_path / "sidecar.lock"
    lock_path.write_bytes(b"0")
    calls: list[tuple[int, int]] = []
    delays: list[float] = []

    def lock_after_brief_contention(file_descriptor: int, mode: int, size: int) -> None:
        calls.append((mode, size))
        if len(calls) < 3:
            raise OSError(errno.EACCES, "sidecar is temporarily locked")

    monkeypatch.setattr(msvcrt, "locking", lock_after_brief_contention)
    monkeypatch.setattr("mochi.sessions.store.time.sleep", delays.append)

    with lock_path.open("a+b") as lock_file:
        SessionStore._lock_file(lock_file)  # noqa: SLF001

    assert calls == [(msvcrt.LK_NBLCK, 1)] * 3
    assert delays == [0.01, 0.01]


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
    valid_first = {"type": "user", "content": "ok-1"}
    asyncio.run(store.save_event(session_id, valid_first))
    path = store._session_path(session_id)  # noqa: SLF001

    invalid = '{"type": "assistant", bad-json'
    non_dict = json.dumps(["not", "object"], ensure_ascii=False)
    valid_2 = json.dumps({"type": "assistant", "content": "ok-2"}, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"{invalid}\n{non_dict}\n{valid_2}\n")

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


def test_special_session_ids_coexist_without_lossy_filename_collisions(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    first_id = "a:b"
    second_id = "a?b"

    asyncio.run(store.save_event(first_id, {"type": "message", "content": "first"}))
    asyncio.run(store.save_event(second_id, {"type": "message", "content": "second"}))

    assert store._session_path(first_id) != store._session_path(second_id)  # noqa: SLF001
    assert asyncio.run(store.load_session(first_id)) == [{"type": "message", "content": "first"}]
    assert asyncio.run(store.load_session(second_id)) == [{"type": "message", "content": "second"}]
    assert set(asyncio.run(store.list_session_ids())) == {first_id, second_id}


def test_v2_filename_is_safe_on_case_insensitive_windows_volumes(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    # These were ``YcOA`` and ``YcOa`` in the rejected Base64 scheme, which
    # collide after Windows case-folding.  V2 filenames are lowercase hashes
    # and are verified by exact-ID sidecars.
    first_id = "a\u00c0"
    second_id = "a\u00da"

    first_path = store._session_path(first_id)  # noqa: SLF001
    second_path = store._session_path(second_id)  # noqa: SLF001
    assert first_path.name.casefold() != second_path.name.casefold()

    asyncio.run(store.save_event(first_id, {"type": "message", "content": "first"}))
    asyncio.run(store.save_event(second_id, {"type": "message", "content": "second"}))

    assert asyncio.run(store.load_session(first_id))[0]["content"] == "first"
    assert asyncio.run(store.load_session(second_id))[0]["content"] == "second"


def test_long_session_id_uses_fixed_length_filename_and_round_trips(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session_id = "long:" + ("\U0001f642" * 1024)
    path = store._session_path(session_id)  # noqa: SLF001

    assert len(path.name) < 255
    asyncio.run(store.save_event(session_id, {"type": "message", "content": "long"}))

    assert path.exists()
    assert asyncio.run(store.load_session(session_id)) == [{"type": "message", "content": "long"}]
    assert session_id in asyncio.run(store.list_session_ids())


def test_identity_sidecar_probes_hash_collisions_without_data_aliasing(tmp_path, monkeypatch) -> None:
    store = SessionStore(tmp_path / "sessions")
    monkeypatch.setattr(
        SessionStore,
        "_session_id_digest",
        staticmethod(lambda _session_id: "f" * 64),
    )
    first_id = "collision:first"
    second_id = "collision:second"

    asyncio.run(store.save_event(first_id, {"type": "message", "content": "first"}))
    asyncio.run(store.save_event(second_id, {"type": "message", "content": "second"}))

    assert store._find_v2_path(first_id) != store._find_v2_path(second_id)  # noqa: SLF001
    assert asyncio.run(store.load_session(first_id))[0]["content"] == "first"
    assert asyncio.run(store.load_session(second_id))[0]["content"] == "second"

    assert asyncio.run(store.delete_session(first_id)) is True
    # The first slot's identity tombstone keeps the second colliding slot
    # addressable after deletion instead of aliasing it back to slot zero.
    assert asyncio.run(store.load_session(second_id))[0]["content"] == "second"


def test_verified_legacy_special_session_is_read_listed_and_migrated_on_replace(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session_id = "legacy:session"
    legacy_path = store._legacy_session_path(session_id)  # noqa: SLF001
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(
        json.dumps({"type": "session_meta", "session_id": session_id, "event": "created"}) + "\n",
        encoding="utf-8",
    )

    assert asyncio.run(store.load_session(session_id))[0]["session_id"] == session_id
    assert session_id in asyncio.run(store.list_session_ids())

    replacement = [{"type": "message", "content": "preserve opaque identity in filename"}]
    asyncio.run(store.replace_session(session_id, replacement))

    assert not legacy_path.exists()
    assert store._session_path(session_id).exists()  # noqa: SLF001
    assert asyncio.run(store.load_session(session_id)) == replacement


def test_verified_legacy_special_session_can_be_deleted(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    session_id = "delete:legacy"
    legacy_path = store._legacy_session_path(session_id)  # noqa: SLF001
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(json.dumps({"session_id": session_id, "type": "message"}) + "\n", encoding="utf-8")

    assert asyncio.run(store.delete_session(session_id)) is True
    assert not legacy_path.exists()


def test_conflicting_legacy_sanitized_identity_fails_closed(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    claimed_id = "a?b"
    conflicting_request_id = "a:b"
    legacy_path = store._legacy_session_path(claimed_id)  # noqa: SLF001
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(
        json.dumps({"type": "message", "session_id": claimed_id, "content": "claimed"}) + "\n",
        encoding="utf-8",
    )

    assert asyncio.run(store.load_session(claimed_id))[0]["content"] == "claimed"
    with pytest.raises(SessionIdentityConflictError):
        asyncio.run(store.load_session(conflicting_request_id))
    with pytest.raises(SessionIdentityConflictError):
        asyncio.run(store.replace_session(conflicting_request_id, [{"type": "message"}]))


@pytest.mark.skipif(os.name != "nt", reason="Windows case-insensitive filename lookup")
def test_identity_free_legacy_filename_requires_exact_preserved_case(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    legacy_path = store._legacy_session_path("alpha")  # noqa: SLF001
    legacy_path.parent.mkdir(parents=True, exist_ok=True)
    legacy_path.write_text(json.dumps({"type": "message", "content": "lower"}) + "\n", encoding="utf-8")

    assert asyncio.run(store.load_session("alpha"))[0]["content"] == "lower"
    with pytest.raises(SessionIdentityConflictError):
        asyncio.run(store.load_session("ALPHA"))
