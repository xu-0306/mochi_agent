"""Runtime store tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

from mochi.runtime.store import RuntimeStore


def test_runtime_store_persists_tasks_events_and_approvals(tmp_path: Path) -> None:
    store = RuntimeStore(tmp_path / "sessions" / "runtime.db")
    task = asyncio.run(
        store.create_task_run(
            task_id="task-1",
            input_text="hello",
            session_id="s1",
            project_id=None,
            workspace_dir=None,
            inference_overrides={"temperature": 0.1},
        )
    )
    assert task["id"] == "task-1"
    assert task["status"] == "queued"

    asyncio.run(store.append_task_event("task-1", {"type": "thinking", "content": "..." }))
    events = asyncio.run(store.get_task_events("task-1"))
    assert events == [{"type": "thinking", "content": "..."}]

    approval = asyncio.run(
        store.create_approval_request(
            approval_id="approval-1",
            task_id="task-1",
            call_id="call-1",
            tool_name="shell",
            arguments={"cmd": "ls"},
        )
    )
    assert approval["status"] == "pending"
    assert approval["arguments"] == {"cmd": "ls"}

    resolved = asyncio.run(
        store.resolve_approval_request(
            "approval-1",
            approved=True,
            reason="ok",
        )
    )
    assert resolved is not None
    assert resolved["status"] == "approved"
    assert resolved["reason"] == "ok"

    asyncio.run(store.update_task_status("task-1", "succeeded"))
    saved_task = asyncio.run(store.get_task_run("task-1"))
    assert saved_task is not None
    assert saved_task["status"] == "succeeded"
