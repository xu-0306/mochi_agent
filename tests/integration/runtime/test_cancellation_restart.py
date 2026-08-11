"""Durable Agent Run cancellation boundary integration tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

from mochi.runtime.service import RuntimeService
from mochi.runtime.store import RuntimeStore


def test_agent_run_cancellation_before_commit_records_no_completion_effect(tmp_path: Path) -> None:
    async def _exercise() -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
        store = RuntimeStore(tmp_path / "runtime.db")
        await store.create_agent_run(
            run_id="run-cancel-before-commit",
            protocol_id="teacher_student_distill",
            title="Cancellation before commit",
            topic="cancellation",
        )
        await store.update_agent_run_status("run-cancel-before-commit", "running")
        service = RuntimeService(engine=object(), store=store)

        response = await service.cancel_agent_run("run-cancel-before-commit")
        run = await store.get_agent_run("run-cancel-before-commit")
        events = await store.get_agent_run_events("run-cancel-before-commit")
        assert response is not None
        assert run is not None
        return response, run, events

    response, run, events = asyncio.run(_exercise())

    assert response["status"] == "cancelled"
    assert response["cancellation"] == {
        "capability": "immediate",
        "durable_outcome": "cancelled_pre_commit",
    }
    assert run["status"] == "cancelled"
    assert not [event for event in events if event["type"] == "run_completion_committed"]


def test_agent_run_restart_does_not_replay_committed_effect(tmp_path: Path) -> None:
    async def _exercise() -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
        store = RuntimeStore(tmp_path / "runtime.db")
        await store.create_agent_run(
            run_id="run-commit-restart",
            protocol_id="teacher_student_distill",
            title="Committed run",
            topic="restart",
        )
        await store.update_agent_run_status("run-commit-restart", "running")
        await store.append_agent_run_event(
            "run-commit-restart",
            {
                "type": "run_completion_committed",
                "completion_status": "succeeded",
                "durable_outcome": "committed",
            },
        )

        restarted_service = RuntimeService(engine=object(), store=store)
        response = await restarted_service.cancel_agent_run("run-commit-restart")
        run = await store.get_agent_run("run-commit-restart")
        events = await store.get_agent_run_events("run-commit-restart")
        assert response is not None
        assert run is not None
        return response, run, events

    response, run, events = asyncio.run(_exercise())

    assert response["status"] == "running"
    assert response["cancellation"] == {
        "capability": "immediate",
        "durable_outcome": "already_committed",
    }
    assert run["status"] == "running"
    assert [event["type"] for event in events].count("run_completion_committed") == 1
