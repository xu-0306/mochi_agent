"""Runtime API support helpers."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from mochi.agents.events import (
    FinalAnswerEvent,
    ThinkingEvent,
    ToolCallRequestEvent,
    ToolCallResultEvent,
)
from mochi.agents.multi_agent.orchestrator import MultiAgentRunResult
from mochi.backends.base import BackendRequestError
from mochi.backends.types import GenerationResult, Message
from mochi.runtime.approvals import InMemoryApprovalStore
from mochi.runtime.exec_runtime import ExecRuntime
from tests.support.app_factories import create_runtime_test_app
from tests.support.exec_providers import PythonDirectProvider as _ApiRuntimePythonDirectProvider
from tests.support.polling import wait_for_status


class _RuntimeFakeEngine:
    def __init__(self) -> None:
        self.permission_policy_calls: list[dict[str, Any] | None] = []
        self.task_workspace_calls: list[str | None] = []
        self._run_count = 0

    async def chat(
        self,
        message: str,
        session_id: str | None = None,
        inference_overrides: dict[str, Any] | None = None,
        project_id: str | None = None,
        workspace_dir: str | None = None,
        task_workspace_dir: str | None = None,
        permission_policy: dict[str, Any] | None = None,
    ) -> AsyncIterator[object]:
        _ = (
            message,
            session_id,
            inference_overrides,
            project_id,
            workspace_dir,
        )
        self.permission_policy_calls.append(permission_policy)
        self.task_workspace_calls.append(task_workspace_dir)
        self._run_count += 1
        if self._run_count == 1:
            yield ThinkingEvent(content="thinking")
            yield ToolCallRequestEvent(
                call_id="call-1",
                tool_name="exec_command",
                arguments={"command": "dir", "shell": "cmd"},
            )
            yield ToolCallResultEvent(
                call_id="call-1",
                tool_name="exec_command",
                result=None,
                metadata={
                    "requires_approval": True,
                    "approval_kind": "exec",
                    "approval_scope": "dangerous_command",
                    "replay_safe": False,
                    "reason": "Command requires approval by policy.",
                },
            )
            return
        yield FinalAnswerEvent(content="done", trajectory_id="traj-1")

class _RuntimeFileMutationFakeEngine:
    def __init__(self) -> None:
        self.permission_policy_calls: list[dict[str, Any] | None] = []
        self.task_workspace_calls: list[str | None] = []
        self._run_count = 0

    async def chat(
        self,
        message: str,
        session_id: str | None = None,
        inference_overrides: dict[str, Any] | None = None,
        project_id: str | None = None,
        workspace_dir: str | None = None,
        task_workspace_dir: str | None = None,
        permission_policy: dict[str, Any] | None = None,
    ) -> AsyncIterator[object]:
        _ = (
            message,
            session_id,
            inference_overrides,
            project_id,
            workspace_dir,
        )
        self.permission_policy_calls.append(permission_policy)
        self.task_workspace_calls.append(task_workspace_dir)
        self._run_count += 1
        if self._run_count == 1:
            patch_text = "\n".join(
                [
                    "*** Begin Patch",
                    "*** Update File: notes.py",
                    "@@",
                    "-print('alpha')",
                    "+print('beta')",
                    "*** End Patch",
                ]
            )
            yield ToolCallRequestEvent(
                call_id="call-patch-1",
                tool_name="apply_patch",
                arguments={"patch": patch_text},
            )
            yield ToolCallResultEvent(
                call_id="call-patch-1",
                tool_name="apply_patch",
                result=None,
                error="Patch application requires approval.",
                metadata={
                    "requires_approval": True,
                    "approval_kind": "apply_patch",
                    "approval_scope": "workspace",
                    "replay_safe": True,
                    "reason": "Patch application requires explicit approval in the current autonomy mode.",
                    "change_count": 1,
                    "paths": [str(Path(workspace_dir or ".") / "notes.py")],
                    "diff_available": True,
                    "file_changes": [
                        {
                            "tool_name": "apply_patch",
                            "path": str(Path(workspace_dir or ".") / "notes.py"),
                            "file_path": str(Path(workspace_dir or ".") / "notes.py"),
                            "relative_path": "notes.py",
                            "change_type": "update",
                            "diff": "--- a/notes.py\n+++ b/notes.py\n@@ -1 +1 @@\n-print('alpha')\n+print('beta')",
                            "diff_available": True,
                            "undo_available": True,
                            "undo_action": "restore",
                        }
                    ],
                },
            )
            return
        yield FinalAnswerEvent(content="done", trajectory_id="traj-file")

class _RuntimeExecLinkedFakeEngine:
    def __init__(self, *, exec_approval_store: InMemoryApprovalStore) -> None:
        self._exec_approval_store = exec_approval_store
        self._run_count = 0
        self.linked_exec_approval_id: str | None = None
        self.second_run_started = False

    async def chat(
        self,
        message: str,
        session_id: str | None = None,
        inference_overrides: dict[str, Any] | None = None,
        project_id: str | None = None,
        workspace_dir: str | None = None,
        task_workspace_dir: str | None = None,
        permission_policy: dict[str, Any] | None = None,
    ) -> AsyncIterator[object]:
        _ = (
            message,
            session_id,
            inference_overrides,
            project_id,
            workspace_dir,
            task_workspace_dir,
            permission_policy,
        )
        self._run_count += 1
        if self._run_count == 1:
            resolved_workdir = (
                str(Path(task_workspace_dir).resolve())
                if isinstance(task_workspace_dir, str) and task_workspace_dir
                else None
            )
            approval = self._exec_approval_store.create(
                approval_id="exec-approval-linked-runtime-1",
                command="print('linked exec approved')",
                shell="test",
                scope="dangerous_command",
                reason="Exec command requires approval.",
                command_payload={
                    "command": "print('linked exec approved')",
                    "shell": "test",
                    "workdir": resolved_workdir,
                    "env": None,
                    "timeout_sec": 5.0,
                    "background": False,
                    "tty": False,
                    "approval_state": "approved",
                },
            )
            self.linked_exec_approval_id = approval.approval_id
            yield ToolCallRequestEvent(
                call_id="call-exec-1",
                tool_name="exec_command",
                arguments={"command": "print('linked exec approved')", "shell": "test"},
            )
            yield ToolCallResultEvent(
                call_id="call-exec-1",
                tool_name="exec_command",
                result=None,
                error="Exec command requires approval.",
                metadata={
                    "status": "approval_pending",
                    "approval_id": approval.approval_id,
                    "session_id": None,
                    "timed_out": False,
                    "requires_approval": True,
                    "security_decision": "require_approval",
                    "approval_kind": "exec",
                    "approval_scope": "dangerous_command",
                    "reason": "Exec command requires approval.",
                    "policy_source": "exec_runtime",
                },
            )
            return
        self.second_run_started = True
        yield FinalAnswerEvent(content="done after exec approval", trajectory_id="traj-exec")

class _RuntimeExecLinkedBackgroundFakeEngine:
    def __init__(self, *, exec_approval_store: InMemoryApprovalStore) -> None:
        self._exec_approval_store = exec_approval_store
        self._run_count = 0
        self.linked_exec_approval_id: str | None = None

    async def chat(
        self,
        message: str,
        session_id: str | None = None,
        inference_overrides: dict[str, Any] | None = None,
        project_id: str | None = None,
        workspace_dir: str | None = None,
        task_workspace_dir: str | None = None,
        permission_policy: dict[str, Any] | None = None,
    ) -> AsyncIterator[object]:
        _ = (
            message,
            session_id,
            inference_overrides,
            project_id,
            workspace_dir,
            task_workspace_dir,
            permission_policy,
        )
        self._run_count += 1
        if self._run_count == 1:
            resolved_workdir = (
                str(Path(task_workspace_dir).resolve())
                if isinstance(task_workspace_dir, str) and task_workspace_dir
                else None
            )
            approval = self._exec_approval_store.create(
                approval_id="exec-approval-linked-runtime-bg-1",
                command=_BACKGROUND_SMOKE_COMMAND,
                shell="test",
                scope="dangerous_command",
                reason="Exec command requires approval.",
                command_payload={
                    "command": _BACKGROUND_SMOKE_COMMAND,
                    "shell": "test",
                    "workdir": resolved_workdir,
                    "env": None,
                    "timeout_sec": 30.0,
                    "background": True,
                    "tty": False,
                    "approval_state": "approved",
                },
            )
            self.linked_exec_approval_id = approval.approval_id
            yield ToolCallRequestEvent(
                call_id="call-exec-bg-1",
                tool_name="exec_command",
                arguments={"command": _BACKGROUND_SMOKE_COMMAND, "shell": "test"},
            )
            yield ToolCallResultEvent(
                call_id="call-exec-bg-1",
                tool_name="exec_command",
                result=None,
                error="Exec command requires approval.",
                metadata={
                    "status": "approval_pending",
                    "approval_id": approval.approval_id,
                    "session_id": None,
                    "timed_out": False,
                    "requires_approval": True,
                    "security_decision": "require_approval",
                    "approval_kind": "exec",
                    "approval_scope": "dangerous_command",
                    "reason": "Exec command requires approval.",
                    "policy_source": "exec_runtime",
                },
            )
            return
        yield FinalAnswerEvent(content="done after background exec approval", trajectory_id="traj-exec-bg")

_BACKGROUND_SMOKE_COMMAND = (
    "(__import__('sys').stdout.write('bg-start\\\\n'), "
    "__import__('sys').stdout.flush(), "
    "__import__('time').sleep(5))"
)

_BACKGROUND_SMOKE_COMMAND_RULE = {
    "tokens": [
        "n),",
        "__import__(sys).stdout.flush(),",
        "__import__(time).sleep(5))",
    ],
    "decision": "allow",
    "match": "exact",
}

_CONTROLLED_SMOKE_COMMAND_RULE = {
    "tokens": ["echo", "controlled-ok"],
    "decision": "allow",
    "match": "exact",
    "shells": ["powershell"],
}

class _AgentRunModelBackedEngine:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.evidence_calls: list[dict[str, Any]] = []

    async def generate_with_configured_model(
        self,
        *,
        model_id: str,
        messages: list[Message],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        reasoning_effort: str | None = None,
    ) -> GenerationResult:
        self.calls.append(
            {
                "model_id": model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "reasoning_effort": reasoning_effort,
            }
        )
        if model_id == "teacher-model":
            return GenerationResult(content="Teacher evidence-backed draft.", model=model_id)
        if model_id == "student-model":
            return GenerationResult(content="Student concise final answer.", model=model_id)
        if model_id == "proposer-model":
            return GenerationResult(
                content=(
                    '{"tasks":[{"task_id":"task-1","question":"Find the supported deployment answer.",'
                    '"difficulty":"medium","rationale":"Requires source checking"}]}'
                ),
                model=model_id,
            )
        if model_id == "solver-model":
            return GenerationResult(content="Solver supported deployment answer.", model=model_id)
        if model_id == "planner-model":
            return GenerationResult(
                content=(
                    '{"subquestions":["What is the strongest direct source?","What challenges the current answer?"],'
                    '"evidence_requirements":["Use attributable sources","Mark weak claims"],'
                    '"exclusion_rules":["No unsupported speculation"],'
                    '"evidence_queries":["approved deployment note","deployment contradiction"]}'
                ),
                model=model_id,
            )
        if model_id == "debater-a-model":
            return GenerationResult(content="Argument A cites source src-1 and favors the first path.", model=model_id)
        if model_id == "debater-b-model":
            return GenerationResult(content="Argument B cites source src-1 and challenges unsupported assumptions.", model=model_id)
        if model_id == "local-worker-model":
            return GenerationResult(
                content="Local worker note: summarize evidence, inspect counter-evidence, and keep only verified dataset candidates.",
                model=model_id,
            )
        if model_id == "verifier-model":
            if any(
                isinstance(message, Message) and "candidate_id=solver" in message.content
                for message in messages
            ):
                return GenerationResult(
                    content=(
                        '{"candidate_verifications":['
                        '{"candidate_id":"solver_1_1","status":"verified","rationale":"supported by source","citations":[{"evidence_id":"src-1","summary":"solver matches source"}],"issues":[]}'
                        ']}'
                    ),
                    model=model_id,
                )
            return GenerationResult(
                content=(
                    '{"candidate_verifications":['
                    '{"candidate_id":"teacher","status":"failed","rationale":"unsupported claim","citations":[{"evidence_id":"src-1","summary":"teacher conflicts with source"}],"issues":["unsupported claim"]},'
                    '{"candidate_id":"student","status":"verified","rationale":"supported by source","citations":[{"evidence_id":"src-1","summary":"student matches source"}],"issues":[]},'
                    '{"candidate_id":"debater_a","status":"failed","rationale":"source disagrees","citations":[{"evidence_id":"src-1","summary":"source weakens argument A"}],"issues":["unsupported claim"]},'
                    '{"candidate_id":"debater_b","status":"verified","rationale":"supported by source","citations":[{"evidence_id":"src-1","summary":"source supports argument B"}],"issues":[]}'
                    ']}'
                ),
                model=model_id,
            )
        if model_id == "judge-model":
            if any(
                isinstance(message, Message) and "candidate_id=solver" in message.content
                for message in messages
            ):
                return GenerationResult(
                    content=(
                        '{"selected_candidate_id":"solver_1_1","scores":[{"candidate_id":"solver_1_1","score":0.94,"rationale":"verified solver rollout","evidence_gate":{"status":"skipped"}}]}'
                    ),
                    model=model_id,
                )
            content = (
                '{"selected_candidate_id":"teacher","scores":[{"candidate_id":"teacher","score":0.96,"rationale":"looks comprehensive","evidence_gate":{"status":"skipped"}},{"candidate_id":"student","score":0.82,"rationale":"clear but shorter","evidence_gate":{"status":"skipped"}}]}'
            )
            if any(
                isinstance(message, Message) and "candidate_id=debater_b" in message.content
                for message in messages
            ):
                content = (
                    '{"selected_candidate_id":"debater_b","scores":[{"candidate_id":"debater_a","score":0.67,"rationale":"weaker evidence fit","evidence_gate":{"status":"skipped"}},{"candidate_id":"debater_b","score":0.91,"rationale":"best evidence-grounded argument","evidence_gate":{"status":"skipped"}}]}'
                )
            return GenerationResult(
                content=content,
                model=model_id,
            )
        if model_id == "synth-model":
            return GenerationResult(
                content="# Research Brief\n\n## Summary\nDeployment review.\n\n## Findings\nArgument B is best supported.\n\n## Evidence Quality\nOne attributable source.\n\n## Claim Status\nVerified claims retained.\n\n## Open Gaps\nNeed more source diversity.",
                model=model_id,
            )
        if model_id == "controlled-planner-model":
            return GenerationResult(content="Plan: run one safe command.", model=model_id)
        if model_id == "controlled-executor-model":
            return GenerationResult(
                content=(
                    '{"execution_requests":[{"request_id":"req-1","command":"echo controlled-ok",'
                    '"shell":"powershell","timeout":30,"rationale":"smoke test",'
                    '"expected_artifacts":["stdout"],"success_metric":"stdout contains controlled-ok"}]}'
                ),
                model=model_id,
            )
        if model_id == "controlled-controller-model":
            return GenerationResult(
                content=(
                    '{"status":"approved","reason":"bounded smoke command",'
                    '"command":"echo controlled-ok","shell":"powershell","timeout":30}'
                ),
                model=model_id,
            )
        if model_id == "controlled-evaluator-model":
            return GenerationResult(content="Execution finished successfully.", model=model_id)
        if model_id == "controlled-judge-model":
            return GenerationResult(
                content=(
                    '{"selected_candidate_id":"evaluator","scores":['
                    '{"candidate_id":"planner","score":0.7,"rationale":"planned","evidence_gate":{"status":"skipped"}},'
                    '{"candidate_id":"executor","score":0.7,"rationale":"requested","evidence_gate":{"status":"skipped"}},'
                    '{"candidate_id":"evaluator","score":0.95,"rationale":"summarized","evidence_gate":{"status":"skipped"}}]}'
                ),
                model=model_id,
            )
        if model_id == "controlled-verifier-model":
            return GenerationResult(
                content=(
                    '{"candidate_verifications":['
                    '{"candidate_id":"planner","status":"verified","rationale":"ok","citations":[],"issues":[]},'
                    '{"candidate_id":"executor","status":"verified","rationale":"ok","citations":[],"issues":[]},'
                    '{"candidate_id":"evaluator","status":"verified","rationale":"ok","citations":[],"issues":[]}'
                    ']}'
                ),
                model=model_id,
            )
        raise AssertionError(f"Unexpected model_id: {model_id}")

    async def collect_agent_run_evidence(
        self,
        *,
        queries: list[str],
        metadata: dict[str, Any] | None = None,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        self.evidence_calls.append(
            {
                "queries": list(queries),
                "metadata": dict(metadata or {}),
            }
        )
        return (
            [
                {
                    "evidence_id": "src-1",
                    "title": "Deployment note",
                    "content": "The approved note matches the student answer and rejects the teacher claim.",
                    "url": "https://example.com/deployment-note",
                    "source_type": "web_fetch",
                    "query": queries[0] if queries else "",
                }
            ],
            {
                "query_count": len(queries),
                "collected_packet_count": 1,
                "provider_counts": {"stub-search": 1},
                "queries": [{"query": queries[0] if queries else "", "packet_count": 1}],
            },
        )

class _SlowAgentRunModelBackedEngine(_AgentRunModelBackedEngine):
    def __init__(self, *, delay_seconds: float = 0.55) -> None:
        super().__init__()
        self.delay_seconds = delay_seconds

    async def generate_with_configured_model(
        self,
        *,
        model_id: str,
        messages: list[Message],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        reasoning_effort: str | None = None,
    ) -> GenerationResult:
        await asyncio.sleep(self.delay_seconds)
        return await super().generate_with_configured_model(
            model_id=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )

class _QuotaExhaustedAgentRunEngine(_AgentRunModelBackedEngine):
    async def generate_with_configured_model(
        self,
        *,
        model_id: str,
        messages: list[Message],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        reasoning_effort: str | None = None,
    ) -> GenerationResult:
        del model_id, messages, temperature, max_tokens, reasoning_effort
        raise BackendRequestError(
            "OpenAI quota exhausted for this organization.",
            metadata={
                "backend_name": "openai_compat",
                "status_code": 429,
                "response_text": '{"error":{"code":"insufficient_quota","message":"Please check your plan and billing details."}}',
                "request_url": "https://api.openai.test/v1/chat/completions",
                "model": "gpt-test",
            },
        )

class _LocalModelUnavailableAgentRunEngine(_AgentRunModelBackedEngine):
    async def generate_with_configured_model(
        self,
        *,
        model_id: str,
        messages: list[Message],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        reasoning_effort: str | None = None,
    ) -> GenerationResult:
        del messages, temperature, max_tokens, reasoning_effort
        raise BackendRequestError(
            f"Configured model {model_id!r} is not available.",
            metadata={
                "backend_name": "ollama",
                "model": model_id,
            },
        )

class _BackgroundControlledExecAgentRunEngine(_AgentRunModelBackedEngine):
    async def generate_with_configured_model(
        self,
        *,
        model_id: str,
        messages: list[Message],
        temperature: float = 0.2,
        max_tokens: int = 1024,
        reasoning_effort: str | None = None,
    ) -> GenerationResult:
        if model_id == "controlled-executor-model":
            return GenerationResult(
                content=(
                    '{"execution_requests":[{"request_id":"req-bg-1",'
                    f'"command":"{_BACKGROUND_SMOKE_COMMAND}",'
                    '"shell":"test","timeout":30,"background":true,'
                    '"rationale":"background smoke test",'
                    '"expected_artifacts":["stdout"],'
                    '"success_metric":"session stays readable"}]}'
                ),
                model=model_id,
            )
        if model_id == "controlled-controller-model":
            return GenerationResult(
                content=(
                    '{"status":"approved","reason":"background smoke command",'
                    f'"command":"{_BACKGROUND_SMOKE_COMMAND}",'
                    '"shell":"test","timeout":30,"background":true}'
                ),
                model=model_id,
            )
        return await super().generate_with_configured_model(
            model_id=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            reasoning_effort=reasoning_effort,
        )

def _agent_run_response_payload(
    run_id: str,
    *,
    status: str = "running",
    recovery_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    timestamp = datetime.now(UTC).isoformat()
    return {
        "run_id": run_id,
        "protocol_id": "teacher_student_distill",
        "title": "Resume test run",
        "topic": "resume behavior",
        "status": status,
        "selected_models_roles": {},
        "evaluation_policy": {},
        "run_policy": {},
        "schedule": {},
        "summary": {},
        "recovery_state": dict(recovery_state or {}),
        "degraded": False,
        "latest_error": None,
        "evidence_status": {},
        "artifacts": [],
        "created_at": timestamp,
        "updated_at": timestamp,
        "started_at": None,
        "finished_at": None,
        "events": [],
    }

class _ResumeStrategyRecordingService:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def update_security_config(self, security: Any) -> None:
        del security

    def bind_app_config(self, *, config: Any, config_path: Any) -> None:
        del config, config_path

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def resume_agent_run(
        self,
        run_id: str,
        strategy: str = "continue_from_checkpoint",
    ) -> dict[str, Any] | None:
        checkpoint = (
            {
                "checkpoint_index": 2,
                "stage": "teacher_generation",
            }
            if strategy == "continue_from_checkpoint"
            else None
        )
        self.calls.append({"run_id": run_id, "strategy": strategy, "checkpoint_used": checkpoint is not None})
        return _agent_run_response_payload(
            run_id,
            recovery_state={
                "resume_strategy": strategy,
                "checkpoint": checkpoint,
            },
        )

def _wait_until(
    client: TestClient,
    task_id: str,
    statuses: set[str],
    *,
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    return wait_for_status(
        client,
        f"/v1/tasks/{task_id}",
        statuses,
        timeout_seconds=timeout_seconds,
        resource_label=f"Task {task_id}",
    )

def _wait_agent_run_until(
    client: TestClient,
    run_id: str,
    statuses: set[str],
    *,
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    return wait_for_status(
        client,
        f"/v1/agent-runs/{run_id}",
        statuses,
        timeout_seconds=timeout_seconds,
        resource_label=f"Agent run {run_id}",
    )

def _create_agent_run_exec_test_client(
    *,
    sessions_dir: Path,
    exec_approval_store: InMemoryApprovalStore,
) -> TestClient:
    app, _runtime_service = create_runtime_test_app(
        sessions_dir,
        exec_approval_store=exec_approval_store,
        exec_runtime=ExecRuntime(
            providers={"test": _ApiRuntimePythonDirectProvider()},
            default_shell="test",
        ),
    )
    return TestClient(app)

def _build_linked_agent_run_exec_approval_orchestrator(
    *,
    approval_id: str,
    workdir: Path,
    final_answer: str,
    resumed_payloads: list[dict[str, Any]] | None = None,
) -> Any:
    async def _approval_then_success_run(self: Any, request: Any) -> MultiAgentRunResult:
        approval = self._exec_approval_store.get(approval_id)
        if approval is None or approval.status == "pending":
            if approval is None:
                self._exec_approval_store.create(
                    approval_id=approval_id,
                    command="print('agent run restart approval')",
                    shell="test",
                    scope="dangerous_command",
                    reason="Exec command requires approval.",
                    command_payload={
                        "command": "print('agent run restart approval')",
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
                                "command": "print('agent run restart approval')",
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
        resume_payload = (
            dict(request.metadata.get("resume_payload"))
            if isinstance(request.metadata.get("resume_payload"), dict)
            else {}
        )
        if resumed_payloads is not None:
            resumed_payloads.append(resume_payload)
        role_task_snapshot = (
            dict(resume_payload.get("role_task_snapshot"))
            if isinstance(resume_payload.get("role_task_snapshot"), dict)
            else {}
        )
        tasks = (
            dict(role_task_snapshot.get("tasks"))
            if isinstance(role_task_snapshot.get("tasks"), dict)
            else {}
        )
        exec_task = (
            dict(tasks.get("controlled_execution_exec:req-1"))
            if isinstance(tasks.get("controlled_execution_exec:req-1"), dict)
            else {}
        )
        exec_result = (
            dict(exec_task.get("result_summary", {}).get("execution_result"))
            if isinstance(exec_task.get("result_summary"), dict)
            and isinstance(exec_task.get("result_summary", {}).get("execution_result"), dict)
            else {}
        )
        assert exec_result["status"] == "completed"
        assert exec_result["request_id"] == "req-1"
        assert isinstance(exec_result.get("session_id"), str)
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

__all__ = [name for name in globals() if not name.startswith("__")]
