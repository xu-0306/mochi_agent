from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from mochi.agents.tool_discovery_state import (
    TOOL_DISCOVERY_EVENT,
    TOOL_DISCOVERY_STATE_VERSION,
    ToolDiscoveryObservation,
    ToolDiscoveryState,
    ToolDiscoveryStateRepository,
)
from mochi.sessions.store import SessionStore


def _observation(
    *,
    tool_name: str = "file_write",
    query_hash: str = "a" * 64,
    turn_id: str = "turn-1",
    turn_index: int = 1,
    catalog_fingerprint: str = "f" * 64,
    catalog_generation: int = 1,
    capability_risk_class: str = "destructive",
) -> ToolDiscoveryObservation:
    return ToolDiscoveryObservation(
        tool_name=tool_name,
        source_query_hash=query_hash,
        turn_id=turn_id,
        turn_index=turn_index,
        catalog_fingerprint=catalog_fingerprint,
        catalog_generation=catalog_generation,
        capability_risk_class=capability_risk_class,
    )


def test_tool_discovery_state_round_trip_is_strict_and_future_version_rejected() -> None:
    state = ToolDiscoveryState.empty(
        "session-1",
        catalog_generation=1,
        catalog_fingerprint="f" * 64,
    ).record_observations(
        [_observation()],
        current_turn_index=1,
        max_entries=20,
        ttl_turns=20,
        catalog_generation=1,
        catalog_fingerprint="f" * 64,
    )

    assert ToolDiscoveryState.from_dict(state.to_dict()) == state

    invalid_extra = state.to_dict()
    invalid_extra["unexpected"] = True
    with pytest.raises(ValueError, match="unexpected"):
        ToolDiscoveryState.from_dict(invalid_extra)

    invalid_version = state.to_dict()
    invalid_version["state_version"] = "tool-discovery-state-v99"
    with pytest.raises(ValueError, match="unsupported"):
        ToolDiscoveryState.from_dict(invalid_version)


def test_record_observations_prunes_by_ttl_lru_and_catalog_invalidation() -> None:
    state = ToolDiscoveryState.empty(
        "session-ttl",
        catalog_generation=1,
        catalog_fingerprint="f" * 64,
    )
    state = state.record_observations(
        [_observation(turn_id="turn-1", turn_index=1)],
        current_turn_index=1,
        max_entries=10,
        ttl_turns=20,
        catalog_generation=1,
        catalog_fingerprint="f" * 64,
    )
    state = state.record_observations(
        [_observation(turn_id="turn-5", turn_index=5)],
        current_turn_index=5,
        max_entries=10,
        ttl_turns=20,
        catalog_generation=1,
        catalog_fingerprint="f" * 64,
    )
    entry = state.entries[0]
    assert entry.discovered_turn_index == 1
    assert entry.last_used_turn_index == 5

    state = state.record_observations(
        [
            _observation(
                tool_name="file_read",
                query_hash="b" * 64,
                turn_id="turn-6",
                turn_index=6,
                capability_risk_class="read_only",
            ),
            _observation(
                tool_name="web_search",
                query_hash="c" * 64,
                turn_id="turn-7",
                turn_index=7,
                capability_risk_class="open_world",
            ),
        ],
        current_turn_index=7,
        max_entries=2,
        ttl_turns=20,
        catalog_generation=1,
        catalog_fingerprint="f" * 64,
    )
    assert [item.tool_name for item in state.entries] == ["web_search", "file_read"]

    state = state.record_observations(
        [
            _observation(
                tool_name="file_write",
                query_hash="a" * 64,
                turn_id="turn-15",
                turn_index=15,
                catalog_generation=2,
                catalog_fingerprint="e" * 64,
            )
        ],
        current_turn_index=15,
        max_entries=5,
        ttl_turns=10,
        catalog_generation=2,
        catalog_fingerprint="e" * 64,
    )
    assert len(state.entries) == 1
    assert state.entries[0].tool_name == "file_write"
    assert state.entries[0].catalog_generation == 2
    assert state.entries[0].catalog_fingerprint == "e" * 64


@pytest.mark.asyncio
async def test_repository_idempotent_replay_does_not_append_twice(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    repository = ToolDiscoveryStateRepository(store)
    result = await repository.record_observations(
        session_id="session-replay",
        turn_id="turn-1",
        current_turn_index=1,
        catalog_generation=1,
        catalog_fingerprint="f" * 64,
        observations=[_observation()],
        idempotency_key="tool-discovery:replay",
    )
    replay = await repository.record_observations(
        session_id="session-replay",
        turn_id="turn-1",
        current_turn_index=1,
        catalog_generation=1,
        catalog_fingerprint="f" * 64,
        observations=[_observation()],
        idempotency_key="tool-discovery:replay",
        timestamp="2026-07-26T01:00:00+00:00",
    )

    assert result.status == "saved"
    assert replay.status == "saved"
    assert replay.idempotent_replay is True
    events = await store.load_session("session-replay")
    assert [event.get("event") for event in events].count(TOOL_DISCOVERY_EVENT) == 1


@pytest.mark.asyncio
async def test_repository_has_one_cas_winner_across_instances(tmp_path) -> None:
    sessions_dir = tmp_path / "sessions"
    first = ToolDiscoveryStateRepository(SessionStore(sessions_dir))
    second = ToolDiscoveryStateRepository(SessionStore(sessions_dir))

    results = await asyncio.gather(
        first.record_observations(
            session_id="session-cas",
            turn_id="turn-a",
            current_turn_index=1,
            catalog_generation=1,
            catalog_fingerprint="f" * 64,
            observations=[_observation(turn_id="turn-a")],
            idempotency_key="tool-discovery:a",
        ),
        second.record_observations(
            session_id="session-cas",
            turn_id="turn-b",
            current_turn_index=1,
            catalog_generation=1,
            catalog_fingerprint="f" * 64,
            observations=[_observation(turn_id="turn-b", query_hash="b" * 64)],
            idempotency_key="tool-discovery:b",
        ),
    )

    assert [result.status for result in results].count("saved") == 1
    assert [result.status for result in results].count("conflict") == 1


@pytest.mark.asyncio
async def test_repository_latest_invalid_event_fails_closed(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    repository = ToolDiscoveryStateRepository(store)
    saved = await repository.record_observations(
        session_id="session-invalid",
        turn_id="turn-1",
        current_turn_index=1,
        catalog_generation=1,
        catalog_fingerprint="f" * 64,
        observations=[_observation()],
        idempotency_key="tool-discovery:ok",
    )
    assert saved.status == "saved"

    await store.save_event(
        "session-invalid",
        {
            "type": "session_meta",
            "event": TOOL_DISCOVERY_EVENT,
            "schema_version": 1,
            "session_id": "session-invalid",
            "state_revision": 2,
            "turn_id": "turn-2",
            "catalog_generation": 1,
            "idempotency_key": "tool-discovery:bad",
            "tool_discovery_state": {"state_version": "tool-discovery-state-v99"},
            "timestamp": "2026-07-26T02:00:00+00:00",
        },
    )

    loaded = await repository.load("session-invalid")

    assert loaded.status == "invalid"
