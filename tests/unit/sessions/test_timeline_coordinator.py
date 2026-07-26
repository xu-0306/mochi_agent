from __future__ import annotations

import asyncio
from datetime import datetime

import pytest

from mochi.sessions.store import SessionStore
from mochi.sessions.timeline_coordinator import TimelineCoordinator


@pytest.mark.asyncio
async def test_heartbeat_renews_claimed_lane_without_holding_external_work(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    coordinator = TimelineCoordinator(
        session_store=store,
        session_id="heartbeat-session",
        turn_id="turn-one",
        lease_seconds=6,
    )
    await coordinator.admit_user_message(
        {
            "type": "message",
            "session_id": "heartbeat-session",
            "turn_id": "turn-one",
            "role": "user",
            "content": "keep the lane alive",
        }
    )
    await coordinator.claim()
    before = await coordinator._repository.load("heartbeat-session")  # noqa: SLF001
    assert before.timeline is not None
    first_expiry = before.timeline.lane_lease_expires_at
    assert first_expiry is not None

    await coordinator.start_heartbeat()
    await asyncio.sleep(2.1)
    after = await coordinator._repository.load("heartbeat-session")  # noqa: SLF001
    assert after.timeline is not None
    renewed_expiry = after.timeline.lane_lease_expires_at
    assert renewed_expiry is not None
    assert datetime.fromisoformat(renewed_expiry) > datetime.fromisoformat(first_expiry)
    await coordinator.finish()


@pytest.mark.asyncio
async def test_linearized_history_keeps_legacy_prefix_and_compatibility_tail(
    tmp_path,
) -> None:
    store = SessionStore(tmp_path / "sessions")
    for role, content in (
        ("user", "legacy request"),
        ("assistant", "legacy response"),
    ):
        await store.save_event(
            "compatibility-history",
            {
                "type": "message",
                "session_id": "compatibility-history",
                "role": role,
                "content": content,
            },
        )

    first = TimelineCoordinator(
        session_store=store,
        session_id="compatibility-history",
        turn_id="turn-one",
    )
    await first.admit_user_message(
        {
            "type": "message",
            "session_id": "compatibility-history",
            "turn_id": "turn-one",
            "role": "user",
            "content": "managed request",
        }
    )
    first_history = await first.claim()
    assert [event.get("content") for event in first_history] == [
        "legacy request",
        "legacy response",
    ]
    await first.finish(
        companion_events=(
            {
                "type": "message",
                "session_id": "compatibility-history",
                "turn_id": "turn-one",
                "role": "assistant",
                "content": "managed response",
            },
        )
    )
    await store.save_event(
        "compatibility-history",
        {
            "type": "message",
            "session_id": "compatibility-history",
            "turn_id": "turn-one:approval:approval-one",
            "role": "assistant",
            "content": "approval continuation response",
        },
    )

    second = TimelineCoordinator(
        session_store=store,
        session_id="compatibility-history",
        turn_id="turn-two",
    )
    await second.admit_user_message(
        {
            "type": "message",
            "session_id": "compatibility-history",
            "turn_id": "turn-two",
            "role": "user",
            "content": "next request",
        }
    )
    second_history = await second.claim()
    assert [event.get("content") for event in second_history] == [
        "legacy request",
        "legacy response",
        "managed request",
        "managed response",
        "approval continuation response",
    ]
    await second.finish()
