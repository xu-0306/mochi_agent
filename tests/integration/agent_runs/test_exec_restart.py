"""Restart qualification for detached AgentRun exec sessions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from mochi.runtime.exec_sessions import ExecSessionStatus, SessionPollResult
from mochi.runtime.service import RuntimeService
from mochi.runtime.store import RuntimeStore


def _poll(session_id: str, status: ExecSessionStatus) -> SessionPollResult:
    return SessionPollResult(
        session_id=session_id,
        shell="test",
        status=status,
        background=True,
        tty=False,
        pid=12345,
        exit_code=None,
        timed_out=False,
        approval_state="not_required",
        stdout="",
        stderr="",
        detached=True,
        restored=True,
    )


@dataclass
class _RecoveredExecRuntime:
    read_result: SessionPollResult
    kill_result: SessionPollResult
    kill_calls: int = 0

    async def read_session(self, session_id: str, *, yield_time_ms: int | None = None) -> SessionPollResult:
        del session_id, yield_time_ms
        return self.read_result

    async def kill_session(self, session_id: str) -> SessionPollResult:
        del session_id
        self.kill_calls += 1
        return self.kill_result


def _detached_job_artifact(session_id: str) -> dict[str, object]:
    return {
        "artifact_id": f"{session_id}:detached",
        "artifact_type": "detached_exec_jobs",
        "title": "Detached exec jobs",
        "uri": f"agent-run://exec-restart/artifacts/{session_id}",
        "mime_type": "application/json",
        "metadata": {
            "content": {
                "items": [
                    {
                        "session_id": session_id,
                        "request_id": "request-1",
                        "status": "running",
                        "log_path": "logs/session.log",
                        "checkpoint_dir": "checkpoints/session",
                    }
                ]
            }
        },
    }


def test_only_verified_running_session_is_reattached(tmp_path: Path) -> None:
    async def _exercise() -> tuple[dict[str, object], dict[str, object], list[dict[str, object]]]:
        store = RuntimeStore(tmp_path / "runtime.db")
        await store.create_agent_run(
            run_id="exec-restart",
            protocol_id="controlled_subagent_execution",
            title="Detached exec restart",
            topic="restart qualification",
            artifacts=[
                _detached_job_artifact("verified-session"),
                _detached_job_artifact("orphaned-session"),
            ],
        )
        service = RuntimeService(engine=object(), store=store)
        runtime = _RecoveredExecRuntime(
            read_result=_poll("verified-session", ExecSessionStatus.RUNNING),
            kill_result=_poll("verified-session", ExecSessionStatus.KILLED),
        )
        service._exec_runtime = runtime  # type: ignore[assignment]

        verified = await service.reattach_agent_run_exec_session("exec-restart", "verified-session")
        runtime.read_result = _poll("orphaned-session", ExecSessionStatus.ORPHANED)
        orphaned = await service.reattach_agent_run_exec_session("exec-restart", "orphaned-session")
        events = await store.get_agent_run_events("exec-restart")
        assert isinstance(verified, dict)
        assert isinstance(orphaned, dict)
        return verified, orphaned, events

    verified, orphaned, events = asyncio.run(_exercise())

    assert verified["reattached"] is True
    assert verified["session"]["status"] == "running"
    assert orphaned["reattached"] is False
    assert orphaned["reattach_status"] == "orphaned"
    assert [event["type"] for event in events] == ["detached_exec_reattached"]


def test_stop_is_idempotent_after_recovery(tmp_path: Path) -> None:
    async def _exercise() -> tuple[dict[str, object], dict[str, object], list[dict[str, object]], int]:
        store = RuntimeStore(tmp_path / "runtime.db")
        await store.create_agent_run(
            run_id="exec-stop-restart",
            protocol_id="controlled_subagent_execution",
            title="Detached exec stop restart",
            topic="restart qualification",
            artifacts=[_detached_job_artifact("verified-session")],
        )
        service = RuntimeService(engine=object(), store=store)
        runtime = _RecoveredExecRuntime(
            read_result=_poll("verified-session", ExecSessionStatus.KILLED),
            kill_result=_poll("verified-session", ExecSessionStatus.KILLED),
        )
        service._exec_runtime = runtime  # type: ignore[assignment]

        first = await service.stop_agent_run_exec_session("exec-stop-restart", "verified-session")
        second = await service.stop_agent_run_exec_session("exec-stop-restart", "verified-session")
        events = await store.get_agent_run_events("exec-stop-restart")
        assert isinstance(first, dict)
        assert isinstance(second, dict)
        return first, second, events, runtime.kill_calls

    first, second, events, kill_calls = asyncio.run(_exercise())

    assert first["stop_status"] == "killed"
    assert first["cancellation"]["durable_outcome"] == "committed"
    assert second["stop_status"] == "already_committed"
    assert second["cancellation"]["durable_outcome"] == "already_committed"
    assert kill_calls == 1
    assert [event["type"] for event in events] == ["detached_exec_stop"]
