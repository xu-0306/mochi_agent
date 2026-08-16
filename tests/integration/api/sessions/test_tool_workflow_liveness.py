from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import httpx
import pytest

from mochi.api.server import create_app
from mochi.config.schema import MochiConfig
from mochi.sessions.store import SessionStore


@pytest.mark.asyncio
async def test_tool_workflow_fanout_preserves_health_and_worker_liveness(
    tmp_path: Path,
) -> None:
    """Many turn snapshots share one read without starving unrelated requests."""

    sessions_dir = tmp_path / "sessions"
    config = MochiConfig.model_validate({"sessions_dir": str(sessions_dir)})
    config.agent.tool_observability_v1 = True
    store = SessionStore(sessions_dir)
    session_id = "session-with-many-turns"
    turn_ids = [f"turn-{index}" for index in range(64)]
    events: list[dict[str, object]] = [
        {
            "type": "session_meta",
            "event": "created",
            "session_id": session_id,
        }
    ]
    events.extend(
        {
            "type": "turn_event",
            "schema_version": 1,
            "event_id": f"{turn_id}:1",
            "turn_id": turn_id,
            "seq": 1,
            "phase": "final_answer",
            "payload": {"content": "x" * 1_024},
        }
        for turn_id in turn_ids
    )
    assert await store.create_session_if_absent(session_id, events=events)

    session_path = await asyncio.to_thread(store._resolve_existing_path, session_id)  # noqa: SLF001
    assert session_path.stat().st_size >= 64 * 1_024

    original_read = store._read_strict_snapshot_locked  # noqa: SLF001
    read_counter_lock = threading.Lock()
    release_reads = threading.Event()
    sidecar_held = threading.Event()
    read_calls = 0

    def counting_read(path: Path, requested_session_id: str):
        nonlocal read_calls
        with read_counter_lock:
            read_calls += 1
        return original_read(path, requested_session_id)

    def hold_real_sidecar_lock() -> None:
        # This is the same lock and worker-thread path used by a real strict
        # snapshot read.  It models a brief overlapping writer without
        # replacing SessionStore's persistence behavior with an async stub.
        with store._sidecar_lock(session_path):  # noqa: SLF001
            sidecar_held.set()
            release_reads.wait(5)

    store._read_strict_snapshot_locked = counting_read  # type: ignore[method-assign]  # noqa: SLF001
    app = create_app()
    app.state.config = config
    app.state.session_store = store
    transport = httpx.ASGITransport(app=app)

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        lock_holder = asyncio.create_task(asyncio.to_thread(hold_real_sidecar_lock))
        assert await asyncio.to_thread(sidecar_held.wait, 2)
        turn_requests = [
            asyncio.create_task(
                client.get(
                    f"/v1/sessions/{session_id}/turns/{turn_id}/tool-workflow",
                )
            )
            for turn_id in turn_ids
        ]
        try:
            # Give every request time to reach the shared repository.  Without
            # route caching and single-flight loading these reads fill the
            # executor and the worker probe cannot be scheduled.
            await asyncio.sleep(0.1)
            health = await asyncio.wait_for(client.get("/health"), timeout=0.5)
            worker_probe = await asyncio.wait_for(
                asyncio.to_thread(lambda: "responsive"),
                timeout=0.5,
            )
        finally:
            release_reads.set()
            await asyncio.wait_for(lock_holder, timeout=2)

        responses = await asyncio.wait_for(asyncio.gather(*turn_requests), timeout=5)

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert worker_probe == "responsive"
    assert read_calls == 1
    assert all(response.status_code == 200 for response in responses)
