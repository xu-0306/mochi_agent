"""Integration coverage for Main-60 startup supervision."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock

from mochi.runtime.service import RuntimeService
from mochi.runtime.store import RuntimeStore


def test_start_reconciles_agent_runs_before_starting_scheduler(tmp_path: Path) -> None:
    async def _exercise() -> None:
        service = RuntimeService(engine=object(), store=RuntimeStore(tmp_path / "runtime.db"))
        reconcile = AsyncMock()
        service._reconcile_agent_runs_on_startup = reconcile  # type: ignore[attr-defined]

        await service.start()
        try:
            reconcile.assert_awaited_once_with()
            assert service._scheduler_task is not None
        finally:
            await service.close()

    asyncio.run(_exercise())


def test_only_one_runtime_adopts_a_standalone_run(tmp_path: Path) -> None:
    async def _exercise() -> None:
        store = RuntimeStore(tmp_path / "runtime.db")
        await store.create_agent_run(
            run_id="standalone-restart-run",
            protocol_id="teacher_student_distill",
            title="Standalone restart",
            topic="startup adoption",
        )
        await store.update_agent_run_status("standalone-restart-run", "running")

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
        lease = await store.get_agent_run_lease("standalone-restart-run")
        assert lease is not None
        assert lease["owner_id"] in {first._runtime_owner_id, second._runtime_owner_id}

    asyncio.run(_exercise())


def test_startup_does_not_adopt_run_with_orphaned_detached_job(tmp_path: Path) -> None:
    async def _exercise() -> None:
        store = RuntimeStore(tmp_path / "runtime.db")
        await store.create_agent_run(
            run_id="orphaned-detached-run",
            protocol_id="controlled_subagent_execution",
            title="Orphaned detached job",
            topic="startup adoption",
        )
        await store.update_agent_run_status("orphaned-detached-run", "running")
        await store.append_agent_run_artifact(
            "orphaned-detached-run",
            artifact_id="orphaned-detached-run:detached",
            artifact_type="detached_exec_jobs",
            title="Detached exec jobs",
            uri="agent-run://orphaned-detached-run/artifacts/detached",
            mime_type="application/json",
            metadata={
                "content": {
                    "items": [
                        {
                            "session_id": "orphaned-session",
                            "status": "orphaned",
                            "identity_state": "unknown",
                        }
                    ]
                }
            },
        )
        service = RuntimeService(engine=object(), store=store)
        schedule = AsyncMock()
        service._ensure_agent_run_job = schedule  # type: ignore[method-assign]

        await service._reconcile_agent_runs_on_startup()  # type: ignore[attr-defined]

        schedule.assert_not_awaited()

    asyncio.run(_exercise())
