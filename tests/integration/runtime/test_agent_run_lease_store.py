"""Integration coverage for fenced standalone AgentRun ownership leases."""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

from mochi.runtime.store import RuntimeStore


def test_agent_run_lease_store_serializes_owners_and_fences_stale_writers(tmp_path: Path) -> None:
    async def _exercise() -> None:
        database_path = tmp_path / "runtime.db"
        first_runtime = RuntimeStore(database_path)
        second_runtime = RuntimeStore(database_path)
        await first_runtime.create_agent_run(
            run_id="standalone-run-1",
            protocol_id="teacher_student_distill",
            title="Fenced ownership",
            topic="restart adoption",
        )

        expires_at = (datetime.now(UTC) + timedelta(minutes=5)).isoformat()
        first_claim, second_claim = await asyncio.gather(
            first_runtime.acquire_agent_run_lease(
                run_id="standalone-run-1",
                owner_id="runtime-a",
                expires_at=expires_at,
            ),
            second_runtime.acquire_agent_run_lease(
                run_id="standalone-run-1",
                owner_id="runtime-b",
                expires_at=expires_at,
            ),
        )

        claims = [first_claim, second_claim]
        winners = [claim for claim in claims if claim["status"] == "acquired"]
        blocked = [claim for claim in claims if claim["status"] == "held_by_other"]
        assert len(winners) == 1
        assert len(blocked) == 1
        first_owner = str(winners[0]["owner_id"])
        first_epoch = int(winners[0]["lease_epoch"])
        next_owner = "runtime-b" if first_owner == "runtime-a" else "runtime-a"

        current_renewal = await first_runtime.renew_agent_run_lease(
            run_id="standalone-run-1",
            owner_id=first_owner,
            lease_epoch=first_epoch,
            expires_at=(datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
        )
        assert current_renewal["status"] == "renewed"
        assert current_renewal["lease_epoch"] == first_epoch

        stale_expiry = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
        expired_lease = await first_runtime.acquire_agent_run_lease(
            run_id="standalone-run-1",
            owner_id=first_owner,
            expires_at=stale_expiry,
        )
        assert expired_lease["status"] == "renewed"
        assert expired_lease["lease_epoch"] == first_epoch

        takeover = await second_runtime.acquire_agent_run_lease(
            run_id="standalone-run-1",
            owner_id=next_owner,
            expires_at=(datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
        )
        assert takeover["status"] == "taken_over"
        assert takeover["owner_id"] == next_owner
        assert takeover["lease_epoch"] == first_epoch + 1

        stale_renewal = await first_runtime.renew_agent_run_lease(
            run_id="standalone-run-1",
            owner_id=first_owner,
            lease_epoch=first_epoch,
            expires_at=(datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
        )
        stale_release = await first_runtime.release_agent_run_lease(
            run_id="standalone-run-1",
            owner_id=first_owner,
            lease_epoch=first_epoch,
        )
        stale_write = await first_runtime.update_agent_run_status_if_lease_current(
            "standalone-run-1",
            "running",
            owner_id=first_owner,
            lease_epoch=first_epoch,
        )

        assert stale_renewal["status"] == "stale_fence"
        assert stale_release["status"] == "stale_fence"
        assert stale_write is False
        current = await second_runtime.get_agent_run_lease("standalone-run-1")
        run = await second_runtime.get_agent_run("standalone-run-1")
        assert current is not None
        assert run is not None
        assert current["owner_id"] == next_owner
        assert current["lease_epoch"] == first_epoch + 1
        assert run["status"] == "created"

        current_release = await second_runtime.release_agent_run_lease(
            run_id="standalone-run-1",
            owner_id=next_owner,
            lease_epoch=first_epoch + 1,
        )
        assert current_release["status"] == "released"
        assert await second_runtime.get_agent_run_lease("standalone-run-1") is None

        reacquired = await first_runtime.acquire_agent_run_lease(
            run_id="standalone-run-1",
            owner_id=first_owner,
            expires_at=(datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
        )
        assert reacquired["status"] == "acquired"
        assert reacquired["lease_epoch"] == first_epoch + 2

        replayed_write = await first_runtime.update_agent_run_status_if_lease_current(
            "standalone-run-1",
            "succeeded",
            owner_id=first_owner,
            lease_epoch=first_epoch,
        )
        assert replayed_write is False
        run_after_replay = await first_runtime.get_agent_run("standalone-run-1")
        assert run_after_replay is not None
        assert run_after_replay["status"] == "created"

    asyncio.run(_exercise())


def test_agent_run_lease_epoch_migration_fences_pre_counter_tokens(tmp_path: Path) -> None:
    async def _exercise() -> None:
        database_path = tmp_path / "legacy-runtime.db"
        legacy_runtime = RuntimeStore(database_path)
        await legacy_runtime.create_agent_run(
            run_id="legacy-standalone-run",
            protocol_id="teacher_student_distill",
            title="Migrated fenced ownership",
            topic="restart adoption",
        )
        original = await legacy_runtime.acquire_agent_run_lease(
            run_id="legacy-standalone-run",
            owner_id="runtime-legacy",
            expires_at=(datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
        )
        original_epoch = int(original["lease_epoch"])
        released = await legacy_runtime.release_agent_run_lease(
            run_id="legacy-standalone-run",
            owner_id="runtime-legacy",
            lease_epoch=original_epoch,
        )
        assert released["status"] == "released"

        with sqlite3.connect(database_path) as conn:
            conn.execute("DROP TABLE agent_run_lease_epochs")
            conn.commit()

        migrated_runtime = RuntimeStore(database_path)
        reacquired = await migrated_runtime.acquire_agent_run_lease(
            run_id="legacy-standalone-run",
            owner_id="runtime-legacy",
            expires_at=(datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
        )
        assert int(reacquired["lease_epoch"]) > original_epoch
        assert await migrated_runtime.update_agent_run_status_if_lease_current(
            "legacy-standalone-run",
            "succeeded",
            owner_id="runtime-legacy",
            lease_epoch=original_epoch,
        ) is False

    asyncio.run(_exercise())
