"""Goal runtime API tests."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
import sqlite3
import sys
import time
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

from mochi.agents.multi_agent.orchestrator import MultiAgentRunResult
from mochi.api.server import create_app
from mochi.config.schema import MochiConfig
from mochi.runtime.goal_strategy_registry import (
    GoalStrategyRegistryEntryData,
    registered_goal_strategy_entries_for_test,
)
from mochi.runtime.approvals import InMemoryApprovalStore
from mochi.runtime.exec_runtime import ExecRuntime
from mochi.runtime.service import RuntimeService
from mochi.runtime.store import RuntimeStore
from mochi.utils.shell_providers import BaseShellProvider, SubprocessSpec
from tests.support.app_factories import create_runtime_test_app
from tests.support.exec_providers import PythonDirectProvider as _GoalApiPythonDirectProvider
from tests.support.polling import wait_for_status

def _create_goal_test_client(tmp_path: Path) -> TestClient:
    app, _runtime_service = create_runtime_test_app(tmp_path / "sessions")
    return TestClient(app)

def _create_goal_test_app(
    tmp_path: Path,
    *,
    active_goal_turn_selector: Any | None = None,
) -> tuple[Any, RuntimeService]:
    return create_runtime_test_app(
        tmp_path / "sessions",
        active_goal_turn_selector=active_goal_turn_selector,
    )

def _create_goal_exec_test_client(
    *,
    sessions_dir: Path,
    exec_approval_store: InMemoryApprovalStore,
) -> TestClient:
    app, _runtime_service = create_runtime_test_app(
        sessions_dir,
        exec_approval_store=exec_approval_store,
        exec_runtime=ExecRuntime(
            providers={"test": _GoalApiPythonDirectProvider()},
            default_shell="test",
        ),
    )
    return TestClient(app)

def _wait_goal_until(
    client: TestClient,
    goal_id: str,
    statuses: set[str],
    *,
    timeout_seconds: float = 4.0,
) -> dict[str, Any]:
    return wait_for_status(
        client,
        f"/v1/goals/{goal_id}",
        statuses,
        timeout_seconds=timeout_seconds,
        resource_label=f"Goal {goal_id}",
    )

def _set_goal_started_at(db_path: Path, goal_id: str, started_at: str) -> None:
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE goals SET started_at=?, updated_at=? WHERE id=?",
            (started_at, started_at, goal_id),
        )
        conn.commit()

def _create_goal_audit_finding(
    runtime_service: RuntimeService,
    *,
    goal_id: str,
    objective: str,
    finding_code: str,
    summary: str,
    details: dict[str, Any] | None = None,
    status: str = "open",
) -> dict[str, Any]:
    asyncio.run(
        runtime_service._store.create_goal(
            goal_id=goal_id,
            objective=objective,
            summary={"phase": "operator_review"},
        )
    )
    return asyncio.run(
        runtime_service._store.upsert_goal_audit_finding(
            goal_id=goal_id,
            finding_code=finding_code,
            summary=summary,
            details=details or {},
            status=status,
        )
    )

def _build_goal_linked_exec_approval_orchestrator(
    *,
    approval_id: str,
    workdir: Path,
    final_answer: str,
) -> Any:
    async def _approval_then_success_run(self: Any, request: Any) -> MultiAgentRunResult:
        approval = self._exec_approval_store.get(approval_id)
        if approval is None or approval.status == "pending":
            if approval is None:
                self._exec_approval_store.create(
                    approval_id=approval_id,
                    command="print('goal restart approval')",
                    shell="test",
                    scope="dangerous_command",
                    reason="Exec command requires approval.",
                    command_payload={
                        "command": "print('goal restart approval')",
                        "shell": "test",
                        "workdir": str(workdir),
                        "env": None,
                        "timeout_sec": 5.0,
                        "background": False,
                        "tty": False,
                        "approval_state": "approved",
                    },
                )
            role_task_snapshot = {
                "roles": {
                    "planner": {"role_id": "planner", "status": "completed"},
                    "executor": {"role_id": "executor", "status": "completed"},
                    "controller": {
                        "role_id": "controller",
                        "status": "waiting_approval",
                        "stage": "controlled_execution_exec:req-1",
                    },
                },
                "tasks": {
                    "controlled_execution_planner": {
                        "task_key": "controlled_execution_planner",
                        "role_id": "planner",
                        "status": "completed",
                        "result_summary": {"planner_output": {"content": "plan"}},
                    },
                    "controlled_execution_executor": {
                        "task_key": "controlled_execution_executor",
                        "role_id": "executor",
                        "status": "completed",
                        "result_summary": {"execution_requests": [{"request_id": "req-1"}]},
                    },
                    "controlled_execution_controller:req-1": {
                        "task_key": "controlled_execution_controller:req-1",
                        "role_id": "controller",
                        "status": "completed",
                        "result_summary": {
                            "controller_decision": {
                                "request_id": "req-1",
                                "status": "approved",
                                "command": "print('goal restart approval')",
                                "shell": "test",
                            }
                        },
                    },
                },
            }
            resume_payload = {
                "version": 1,
                "executor": "continue_from_checkpoint",
                "strategy_default": "continue_from_checkpoint",
                "stage": "controlled_execution_exec:req-1",
                "checkpoint": {
                    "checkpoint_index": 4,
                    "stage": "controlled_execution_controller:req-1",
                },
                "guidance_messages": [],
                "role_guidance_messages": {},
                "metadata_state": {},
                "precomputed_artifacts": {},
                "protocol_artifacts": {},
                "candidates": [],
                "evidence_packets": [],
                "verifications": [],
                "role_task_snapshot": role_task_snapshot,
            }
            return MultiAgentRunResult(
                run_id=request.run_id,
                protocol="controlled_subagent_execution",
                state="awaiting_approval",
                task_input=request.task_input,
                candidates=[],
                selected_candidate_id=None,
                evaluation={},
                artifacts={
                    "final_answer": None,
                    "controlled_execution_runtime": {"approval_pending_count": 1},
                },
                events=[],
                metadata={
                    "approval_state": {
                        "status": "awaiting_approval",
                        "pending_count": 1,
                        "approval_ids": [approval_id],
                        "pending_approvals": [
                            {
                                "approval_id": approval_id,
                                "tool_name": "exec_command",
                                "request_id": "req-1",
                                "task_key": "controlled_execution_exec:req-1",
                                "stage": "controlled_execution_exec:req-1",
                                "role_id": "controller",
                                "source": "controlled_execution",
                            }
                        ],
                    },
                    "recovery_state": {
                        "status": "awaiting_approval",
                        "action": "await_approval",
                        "reason": "Execution approval required",
                        "stage": "controlled_execution_exec:req-1",
                        "checkpoint": {
                            "checkpoint_index": 4,
                            "stage": "controlled_execution_controller:req-1",
                        },
                        "role_task_snapshot": role_task_snapshot,
                        "resume_payload": resume_payload,
                    },
                },
            )
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="controlled_subagent_execution",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": final_answer},
            events=[],
            metadata={},
        )

    return _approval_then_success_run

__all__ = ['asyncio', 'json', 'UTC', 'datetime', 'timedelta', 'Path', 'sqlite3', 'sys', 'time', 'SimpleNamespace', 'Any', 'pytest', 'TestClient', 'MultiAgentRunResult', 'create_app', 'MochiConfig', 'GoalStrategyRegistryEntryData', 'registered_goal_strategy_entries_for_test', 'InMemoryApprovalStore', 'ExecRuntime', 'RuntimeService', 'RuntimeStore', 'BaseShellProvider', 'SubprocessSpec', '_create_goal_test_client', '_create_goal_test_app', '_create_goal_exec_test_client', '_GoalApiPythonDirectProvider', '_wait_goal_until', '_set_goal_started_at', '_create_goal_audit_finding', '_build_goal_linked_exec_approval_orchestrator']
