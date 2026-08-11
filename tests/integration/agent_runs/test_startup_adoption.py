"""Restart qualification for standalone AgentRun startup adoption."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock

from mochi.runtime.service import RuntimeService
from mochi.runtime.store import RuntimeStore


def test_concurrent_startup_adopts_one_verified_run_and_rejects_unverifiable_runs(
    tmp_path: Path,
) -> None:
    async def _exercise() -> tuple[list[dict[str, object]], list[dict[str, object]]]:
        store = RuntimeStore(tmp_path / "runtime.db")
        await store.create_agent_run(
            run_id="verified-run",
            protocol_id="teacher_student_distill",
            title="Verified startup run",
            topic="restart qualification",
        )
        await store.update_agent_run_status("verified-run", "running")
        await store.create_agent_run(
            run_id="foreign-lease-run",
            protocol_id="teacher_student_distill",
            title="Foreign lease run",
            topic="restart qualification",
        )
        await store.update_agent_run_status("foreign-lease-run", "running")
        await store.acquire_agent_run_lease(
            run_id="foreign-lease-run",
            owner_id="other-runtime",
            expires_at=(datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
        )
        await store.create_agent_run(
            run_id="orphaned-job-run",
            protocol_id="controlled_subagent_execution",
            title="Orphaned detached job",
            topic="restart qualification",
            artifacts=[
                {
                    "artifact_id": "orphaned-job-run:detached",
                    "artifact_type": "detached_exec_jobs",
                    "title": "Detached exec jobs",
                    "uri": "agent-run://orphaned-job-run/artifacts/detached",
                    "mime_type": "application/json",
                    "metadata": {
                        "content": {
                            "items": [
                                {
                                    "session_id": "unknown-detached-session",
                                    "status": "orphaned",
                                    "identity_state": "unknown",
                                }
                            ]
                        }
                    },
                }
            ],
        )
        await store.update_agent_run_status("orphaned-job-run", "running")

        first = RuntimeService(engine=object(), store=RuntimeStore(store.database_path))
        second = RuntimeService(engine=object(), store=RuntimeStore(store.database_path))
        first_schedule = AsyncMock()
        second_schedule = AsyncMock()
        first._ensure_agent_run_job = first_schedule  # type: ignore[method-assign]
        second._ensure_agent_run_job = second_schedule  # type: ignore[method-assign]

        await asyncio.gather(
            first._reconcile_agent_runs_on_startup(),  # type: ignore[attr-defined]
            second._reconcile_agent_runs_on_startup(),  # type: ignore[attr-defined]
        )

        assert first_schedule.await_count + second_schedule.await_count == 1
        verified_events = await store.get_agent_run_events("verified-run")
        foreign_events = await store.get_agent_run_events("foreign-lease-run")
        orphaned_events = await store.get_agent_run_events("orphaned-job-run")
        assert [event["type"] for event in verified_events].count("run_adopted") == 1
        assert not [event for event in foreign_events if event["type"] == "run_adopted"]
        assert not [event for event in orphaned_events if event["type"] == "run_adopted"]
        return verified_events, orphaned_events

    verified_events, orphaned_events = asyncio.run(_exercise())

    adopted = next(event for event in verified_events if event["type"] == "run_adopted")
    assert adopted["source"] == "startup_reconciler"
    assert adopted["active_runs"] == 3
    assert adopted["detached_execs"] == 0
    assert orphaned_events == []


def test_startup_projects_committed_completion_once(tmp_path: Path) -> None:
    async def _exercise() -> tuple[dict[str, object], list[dict[str, object]]]:
        store = RuntimeStore(tmp_path / "runtime.db")
        await store.create_agent_run(
            run_id="completion-before-crash",
            protocol_id="teacher_student_distill",
            title="Committed before restart",
            topic="restart qualification",
        )
        await store.update_agent_run_status("completion-before-crash", "running")
        await store.append_agent_run_event(
            "completion-before-crash",
            {
                "type": "run_completion_committed",
                "completion_status": "succeeded",
                "durable_outcome": "committed",
            },
        )

        service = RuntimeService(engine=object(), store=store)
        await service.start()
        await service.close()

        restarted = RuntimeService(engine=object(), store=RuntimeStore(store.database_path))
        await restarted.start()
        await restarted.close()
        run = await store.get_agent_run("completion-before-crash")
        assert run is not None
        events = await store.get_agent_run_events("completion-before-crash")
        return run, events

    run, events = asyncio.run(_exercise())

    assert run["status"] == "succeeded"
    assert [event["type"] for event in events].count("run_completion_committed") == 1
    assert not [event for event in events if event["type"] == "run_adopted"]
