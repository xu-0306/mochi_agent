"""Contract coverage for canonical, fail-closed session lineage resolution."""

from __future__ import annotations

import asyncio

import pytest

from mochi.sessions.lineage import SessionLineageError, build_session_lineage_envelope
from mochi.sessions.search import SessionSearchService
from mochi.sessions.store import SessionStore


def _created_event(
    store: SessionStore,
    session_id: str,
    *,
    parent_session_id: str | None = None,
    parent_storage_id: str | None = None,
    fork_until_turn_id: str | None = None,
) -> dict[str, object]:
    return {
        "type": "session_meta",
        "event": "created",
        "session_id": session_id,
        "timestamp": "2026-08-08T00:00:00Z",
        "lineage": build_session_lineage_envelope(
            storage_id=store.storage_id,
            session_id=session_id,
            parent_session_id=parent_session_id,
            parent_storage_id=parent_storage_id,
            fork_until_turn_id=fork_until_turn_id,
        ),
    }


def _save_created(
    store: SessionStore,
    session_id: str,
    *,
    parent_session_id: str | None = None,
    parent_storage_id: str | None = None,
    fork_until_turn_id: str | None = None,
) -> None:
    asyncio.run(
        store.save_event(
            session_id,
            _created_event(
                store,
                session_id,
                parent_session_id=parent_session_id,
                parent_storage_id=parent_storage_id,
                fork_until_turn_id=fork_until_turn_id,
            ),
        )
    )


def test_lineage_resolves_a_same_store_chain_and_legacy_history(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    _save_created(store, "root")
    _save_created(
        store,
        "child",
        parent_session_id="root",
        parent_storage_id=store.storage_id,
        fork_until_turn_id="turn-1",
    )
    _save_created(
        store,
        "grandchild",
        parent_session_id="child",
        parent_storage_id=store.storage_id,
        fork_until_turn_id="turn-2",
    )
    asyncio.run(store.save_event("legacy", {"type": "message", "content": "legacy"}))

    resolved = asyncio.run(store.resolve_session_lineage("grandchild"))
    legacy = asyncio.run(store.resolve_session_lineage("legacy"))

    assert (resolved.storage_id, resolved.session_id, resolved.root_session_id) == (
        store.storage_id,
        "grandchild",
        "root",
    )
    assert (legacy.session_id, legacy.root_session_id) == ("legacy", "legacy")
    assert asyncio.run(store.resolve_session_root("root")) == "root"


@pytest.mark.parametrize(
    ("parent_session_id", "parent_storage_id", "expected_reason"),
    [
        ("missing", None, "dangling_parent"),
        ("self", None, "self_parent"),
        ("root", "storage:v1:" + "f" * 32, "cross_store_parent"),
    ],
)
def test_lineage_rejects_invalid_parent_metadata(
    tmp_path,
    parent_session_id: str,
    parent_storage_id: str | None,
    expected_reason: str,
) -> None:
    store = SessionStore(tmp_path / "sessions")
    parent_storage = parent_storage_id or store.storage_id
    target = "self" if expected_reason == "self_parent" else "child"
    if expected_reason == "cross_store_parent":
        event = _created_event(
            store,
            target,
            parent_session_id=parent_session_id,
            parent_storage_id=store.storage_id,
            fork_until_turn_id="turn-1",
        )
        event["lineage"]["parent_storage_id"] = parent_storage  # type: ignore[index]
        asyncio.run(store.save_event(target, event))
    else:
        _save_created(
            store,
            target,
            parent_session_id=parent_session_id,
            parent_storage_id=parent_storage,
            fork_until_turn_id="turn-1",
        )

    with pytest.raises(SessionLineageError) as captured:
        asyncio.run(store.resolve_session_lineage(target))

    assert captured.value.reason == expected_reason


def test_lineage_rejects_cycles_and_conflicting_envelopes(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    _save_created(
        store,
        "a",
        parent_session_id="b",
        parent_storage_id=store.storage_id,
        fork_until_turn_id="turn-a",
    )
    _save_created(
        store,
        "b",
        parent_session_id="a",
        parent_storage_id=store.storage_id,
        fork_until_turn_id="turn-b",
    )

    with pytest.raises(SessionLineageError) as cycle:
        asyncio.run(store.resolve_session_lineage("a"))
    assert cycle.value.reason == "cycle"

    _save_created(store, "conflicted")
    asyncio.run(store.save_event("conflicted", _created_event(store, "conflicted")))
    with pytest.raises(SessionLineageError) as conflict:
        asyncio.run(store.resolve_session_lineage("conflicted"))
    assert conflict.value.reason == "conflicting_record"


def test_replacing_a_lineage_enabled_history_must_preserve_its_envelope(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    _save_created(store, "root")
    original = asyncio.run(store.load_session("root"))

    with pytest.raises(SessionLineageError) as omitted:
        asyncio.run(store.replace_session("root", [{"type": "message", "content": "replacement"}]))
    assert omitted.value.reason == "conflicting_record"

    preserved = [*original, {"type": "message", "content": "replacement"}]
    asyncio.run(store.replace_session("root", preserved))
    assert asyncio.run(store.resolve_session_root("root")) == "root"


def test_initial_lineage_history_is_created_once_as_a_complete_batch(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    initial = [
        _created_event(store, "root"),
        {"type": "message", "role": "assistant", "turn_id": "turn-1", "content": "ready"},
    ]

    assert asyncio.run(store.create_session_if_absent("root", events=initial)) is True
    persisted = asyncio.run(store.load_session("root"))
    assert persisted == initial
    assert asyncio.run(store.resolve_session_root("root")) == "root"

    replacement_attempt = [
        _created_event(
            store,
            "root",
            parent_session_id="other",
            parent_storage_id=store.storage_id,
            fork_until_turn_id="turn-2",
        )
    ]
    assert asyncio.run(store.create_session_if_absent("root", events=replacement_attempt)) is False
    assert asyncio.run(store.load_session("root")) == persisted


def test_concurrent_initial_creators_commit_exactly_one_complete_lineage_history(tmp_path) -> None:
    sessions_dir = tmp_path / "sessions"
    first = SessionStore(sessions_dir)
    second = SessionStore(sessions_dir)
    first_initial = [
        _created_event(first, "raced"),
        {"type": "message", "content": "first complete batch"},
    ]
    second_initial = [
        _created_event(second, "raced"),
        {"type": "message", "content": "second complete batch"},
    ]

    async def create_from_both_stores() -> tuple[bool, bool]:
        first_result, second_result = await asyncio.gather(
            first.create_session_if_absent("raced", events=first_initial),
            second.create_session_if_absent("raced", events=second_initial),
        )
        return first_result, second_result

    outcomes = asyncio.run(create_from_both_stores())

    assert outcomes.count(True) == 1
    assert outcomes.count(False) == 1
    assert asyncio.run(first.load_session("raced")) in (first_initial, second_initial)
    assert asyncio.run(first.resolve_session_root("raced")) == "raced"


def test_store_root_resolver_excludes_the_entire_lineage_from_search(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    _save_created(store, "root")
    _save_created(
        store,
        "child",
        parent_session_id="root",
        parent_storage_id=store.storage_id,
        fork_until_turn_id="turn-1",
    )
    _save_created(store, "other")
    for session_id in ("root", "child", "other"):
        asyncio.run(
            store.save_event(
                session_id,
                {"type": "message", "turn_id": f"{session_id}-turn", "content": "lineage needle"},
            )
        )

    service = SessionSearchService(store.session_search_index, lineage_resolver=store.resolve_session_root)
    results = asyncio.run(service.search("lineage needle", current_session_id="child"))

    assert [(result.session_id, result.root_session_id) for result in results] == [("other", "other")]
