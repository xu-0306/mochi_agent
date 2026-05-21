"""Runtime task/approval API tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path
import time
from typing import Any

from fastapi.testclient import TestClient

from mochi.agents.events import FinalAnswerEvent, ThinkingEvent, ToolCallRequestEvent, ToolCallResultEvent
from mochi.api.server import create_app
from mochi.config.schema import MochiConfig


class _RuntimeFakeEngine:
    def __init__(self) -> None:
        self.permission_policy_calls: list[dict[str, Any] | None] = []
        self._run_count = 0

    async def chat(
        self,
        message: str,
        session_id: str | None = None,
        inference_overrides: dict[str, Any] | None = None,
        project_id: str | None = None,
        workspace_dir: str | None = None,
        permission_policy: dict[str, Any] | None = None,
    ) -> AsyncIterator[object]:
        _ = (message, session_id, inference_overrides, project_id, workspace_dir)
        self.permission_policy_calls.append(permission_policy)
        self._run_count += 1
        if self._run_count == 1:
            yield ThinkingEvent(content="thinking")
            yield ToolCallRequestEvent(call_id="call-1", tool_name="shell", arguments={"cmd": "dir"})
            yield ToolCallResultEvent(
                call_id="call-1",
                tool_name="shell",
                result=None,
                metadata={"requires_approval": True},
            )
            return
        yield FinalAnswerEvent(content="done", trajectory_id="traj-1")


def _wait_until(
    client: TestClient,
    task_id: str,
    statuses: set[str],
    *,
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    steps = max(1, int(timeout_seconds / 0.05))
    payload: dict[str, Any] = {}
    for _ in range(steps):
        response = client.get(f"/v1/tasks/{task_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in statuses:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"Task did not reach statuses {statuses}: {payload}")


def test_task_and_approval_flow_with_resume(tmp_path: Path) -> None:
    app = create_app()
    engine = _RuntimeFakeEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/tasks",
            json={"input_message": "run something", "session_id": "runtime-s1"},
        )
        assert create_response.status_code == 200
        task_id = create_response.json()["task_id"]

        task_payload = _wait_until(client, task_id, {"awaiting_approval"})
        assert task_payload["pending_approval"] is not None
        assert task_payload["events"][0]["type"] == "thinking"
        assert task_payload["events"][1]["type"] == "tool_call_request"
        assert task_payload["events"][2]["type"] == "tool_call_result"

        approvals_response = client.get("/v1/approvals?status=pending")
        assert approvals_response.status_code == 200
        approvals = approvals_response.json()
        assert len(approvals) == 1
        approval_id = approvals[0]["approval_id"]

        resolve_response = client.post(
            f"/v1/approvals/{approval_id}/resolve",
            json={"approved": True, "reason": "allowed"},
        )
        assert resolve_response.status_code == 200

        done_payload = _wait_until(client, task_id, {"succeeded"})
        assert done_payload["events"][-1]["type"] == "final_answer"

    assert engine.permission_policy_calls[0] == {
        "autonomy_mode": "trusted_workspace",
        "require_approval_for_shell": True,
        "require_approval_for_file_write": False,
        "file_ops_scope": "workspace",
    }
    assert engine.permission_policy_calls[1] == {
        "autonomy_mode": "trusted_workspace",
        "require_approval_for_shell": True,
        "require_approval_for_file_write": False,
        "file_ops_scope": "workspace",
        "approved_tool_calls": [
            {
                "tool_name": "shell",
                "arguments": {"cmd": "dir"},
            }
        ]
    }
