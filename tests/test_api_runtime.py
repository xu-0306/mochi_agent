"""Runtime task/approval API tests."""

from __future__ import annotations

from collections.abc import AsyncIterator
import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
import threading
import time
from typing import Any
import sys

from fastapi.testclient import TestClient

from mochi.agents.events import FinalAnswerEvent, ThinkingEvent, ToolCallRequestEvent, ToolCallResultEvent
from mochi.agents.multi_agent.orchestrator import (
    MultiAgentOrchestrator,
    MultiAgentRunEvent,
    MultiAgentRunResult,
)
from mochi.backends.base import BackendRequestError
from mochi.backends.types import GenerationResult, Message
from mochi.api.server import create_app
from mochi.config.schema import MochiConfig
from mochi.runtime.approvals import InMemoryApprovalStore, PersistentApprovalStore
from mochi.runtime.exec_runtime import ExecRuntime
from mochi.runtime.service import (
    RuntimeService,
    _ensure_agent_run_resume_payload,
    _resolve_agent_run_resume_strategy,
)
from mochi.runtime.store import RuntimeStore
from mochi.utils.shell_providers import BaseShellProvider, SubprocessSpec


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


class _ApiRuntimePythonDirectProvider(BaseShellProvider):
    @property
    def canonical_name(self) -> str:
        return "test"

    @property
    def aliases(self) -> tuple[str, ...]:
        return ("test",)

    def build_subprocess_spec(self, command: str, *, tty: bool = False) -> SubprocessSpec:
        del tty
        return SubprocessSpec(executable=sys.executable, args=("-c", command))


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


def _wait_agent_run_until(
    client: TestClient,
    run_id: str,
    statuses: set[str],
    *,
    timeout_seconds: float = 2.0,
) -> dict[str, Any]:
    steps = max(1, int(timeout_seconds / 0.05))
    payload: dict[str, Any] = {}
    for _ in range(steps):
        response = client.get(f"/v1/agent-runs/{run_id}")
        assert response.status_code == 200
        payload = response.json()
        if payload["status"] in statuses:
            return payload
        time.sleep(0.05)
    raise AssertionError(f"Agent run did not reach statuses {statuses}: {payload}")


def _create_agent_run_exec_test_client(
    *,
    sessions_dir: Path,
    exec_approval_store: InMemoryApprovalStore,
) -> TestClient:
    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(sessions_dir / "runtime.db"),
        exec_approval_store=exec_approval_store,
        exec_runtime=ExecRuntime(
            providers={"test": _ApiRuntimePythonDirectProvider()},
            default_shell="test",
        ),
    )
    runtime_service.set_scheduler_poll_interval(0.05)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(sessions_dir)}
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


def test_agent_run_resume_defaults_to_checkpoint_continue_strategy(tmp_path: Path) -> None:
    app = create_app()
    service = _ResumeStrategyRecordingService()
    app.state.runtime_service = service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        response = client.post("/v1/agent-runs/run-resume-default/resume")
        assert response.status_code == 200
        payload = response.json()

    assert service.calls == [
        {
            "run_id": "run-resume-default",
            "strategy": "continue_from_checkpoint",
            "checkpoint_used": True,
        }
    ]
    assert payload["recovery_state"]["resume_strategy"] == "continue_from_checkpoint"
    assert payload["recovery_state"]["checkpoint"]["checkpoint_index"] == 2


def test_agent_run_resume_restart_attempt_ignores_checkpoint_payload(tmp_path: Path) -> None:
    app = create_app()
    service = _ResumeStrategyRecordingService()
    app.state.runtime_service = service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        response = client.post(
            "/v1/agent-runs/run-resume-restart/resume",
            json={"strategy": "restart_attempt"},
        )
        assert response.status_code == 200
        payload = response.json()

    assert service.calls == [
        {
            "run_id": "run-resume-restart",
            "strategy": "restart_attempt",
            "checkpoint_used": False,
        }
    ]
    assert payload["recovery_state"]["resume_strategy"] == "restart_attempt"
    assert payload["recovery_state"]["checkpoint"] is None


def test_resolve_agent_run_resume_strategy_reads_structured_resume_executor() -> None:
    run = {
        "summary": {
            "recovery_state": {
                "resume_payload": {
                    "executor": "restart_attempt",
                    "stage": "protocol_completed",
                    "checkpoint": {
                        "checkpoint_index": 3,
                        "stage": "protocol_completed",
                    },
                }
            }
        }
    }

    assert _resolve_agent_run_resume_strategy(None, run) == "restart_attempt"


def test_ensure_agent_run_resume_payload_upgrades_legacy_payload_shape() -> None:
    run = {
        "protocol_id": "teacher_student_distill",
        "task": "Summarize the deployment risks.",
        "events": [
            {
                "type": "role_output",
                "payload": {
                    "candidate_id": "student",
                    "role_id": "student",
                    "content": "Student answer",
                    "model_id": "student-model",
                    "round_index": 1,
                },
            },
            {
                "type": "evaluation",
                "payload": {
                    "policy": {"mode": "judge"},
                    "selected_candidate_id": "student",
                    "scores": [{"candidate_id": "student", "score": 0.9}],
                },
            },
        ],
        "artifacts": [
            {
                "artifact_type": "run_summary",
                "metadata": {
                    "content": {
                        "protocol": "teacher_student_distill",
                        "selected_candidate_id": "student",
                        "final_answer": "Student answer",
                        "candidate_count": 1,
                        "debate_transcript": {"items": []},
                    }
                },
            },
            {
                "artifact_type": "evidence_summary",
                "metadata": {
                    "content": {
                        "evidence_packets": [
                            {
                                "evidence_id": "src-1",
                                "title": "Deployment note",
                            }
                        ],
                        "collected_packet_count": 1,
                    }
                },
            },
            {
                "artifact_type": "verification_summary",
                "metadata": {
                    "content": {
                        "verified_candidate_ids": ["student"],
                        "verifications": [
                            {
                                "candidate_id": "student",
                                "status": "verified",
                                "summary": "Matches evidence.",
                                "citations": [],
                            }
                        ],
                    }
                },
            },
            {
                "artifact_type": "role_task_snapshot",
                "metadata": {
                    "content": {
                        "roles": {
                            "teacher": {
                                "role_id": "teacher",
                                "status": "completed",
                                "stage": "teacher_generation",
                                "assigned_model_id": "teacher-model",
                                "candidate": {
                                    "candidate_id": "teacher",
                                    "role_id": "teacher",
                                    "content": "Teacher answer",
                                    "metadata": {"model_id": "teacher-model"},
                                },
                            }
                        }
                    }
                },
            },
        ],
        "summary": {
            "evidence_packets": [
                {
                    "evidence_id": "src-1",
                    "title": "Deployment note",
                }
            ]
        },
    }
    summary = {
        "recovery_state": {
            "status": "awaiting_resources",
            "action": "pause",
            "reason": "budget exhausted",
            "stage": "evaluation_completed",
            "checkpoint": {
                "checkpoint_index": 7,
                "stage": "evaluation_completed",
            },
            "resume_payload": {
                "strategy_default": "continue_from_checkpoint",
                "state": {"legacy": True},
            },
        }
    }

    upgraded = _ensure_agent_run_resume_payload(
        run=run,
        summary=summary,
        strategy="continue_from_checkpoint",
    )

    payload = upgraded["recovery_state"]["resume_payload"]
    assert payload["stage"] == "evaluation_completed"
    assert payload["executor"] == "continue_from_checkpoint"
    assert payload["checkpoint"]["checkpoint_index"] == 7
    assert payload["candidates"][0]["candidate_id"] == "student"
    assert payload["evaluation"]["selected_candidate_id"] == "student"
    assert payload["verification_summary"]["verified_candidate_ids"] == ["student"]
    assert payload["role_task_snapshot"]["roles"]["teacher"]["status"] == "completed"


def test_ensure_agent_run_resume_payload_includes_live_collector_shard_artifacts() -> None:
    run = {
        "protocol_id": "teacher_student_distill",
        "events": [],
        "artifacts": [
            {
                "artifact_type": "collector_shard_manifest",
                "artifact_id": "collector-live-1",
                "metadata": {
                    "attempt_id": "attempt-1",
                    "content": {
                        "shards": [
                            {
                                "shard_id": "discourse-topic-274354",
                                "adapter_name": "discourse_topic_adapter",
                                "status": "running",
                                "updated_at": "2026-06-24T00:00:00+00:00",
                                "progress": {
                                    "cursor": "101",
                                    "items_collected": 2,
                                    "items_emitted": 2,
                                },
                            }
                        ]
                    },
                },
            },
            {
                "artifact_type": "dataset_record",
                "artifact_id": "collector-record-live-1",
                "metadata": {
                    "attempt_id": "attempt-1",
                    "record": {
                        "input": "Topic: API examples",
                        "target": {"answer": "First collected post"},
                        "metadata": {
                            "capability_family": "dataset_collection",
                            "collector_provenance": {
                                "source_url": "https://forum.example/t/api-examples/274354/1",
                                "source_id": "topic:274354:post:1",
                                "adapter_name": "discourse_topic_adapter",
                                "tool_name": "discourse_topic_collect",
                                "policy_disposition": "allow",
                                "shard_id": "discourse-topic-274354",
                            },
                        },
                    },
                },
            },
        ],
        "summary": {},
        "schedule": {"current_attempt_id": "attempt-1"},
    }
    summary = {
        "recovery_state": {
            "status": "awaiting_resources",
            "action": "pause",
            "reason": "collector interrupted",
            "stage": "protocol_completed",
            "checkpoint": {
                "checkpoint_index": 3,
                "stage": "protocol_completed",
            },
        }
    }

    upgraded = _ensure_agent_run_resume_payload(
        run=run,
        summary=summary,
        strategy="continue_from_checkpoint",
    )

    payload = upgraded["recovery_state"]["resume_payload"]["protocol_artifacts"]
    assert payload["collector_shard_manifests"]["shards"][0]["shard_id"] == "discourse-topic-274354"
    assert payload["collector_shard_manifests"]["shards"][0]["progress"]["cursor"] == "101"
    assert payload["collector_dataset_records"]["records"][0]["target"]["answer"] == (
        "First collected post"
    )
    assert payload["collector_record_provenance"]["records"][0]["source_id"] == (
        "topic:274354:post:1"
    )


def test_agent_run_resume_duplicate_requests_do_not_enqueue_parallel_recovery_jobs(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    run_id = "run-resume-dedupe"
    recovery_summary = {
        "task_input": "resume the latest checkpoint",
        "recovery_state": {
            "status": "awaiting_resources",
            "action": "pause",
            "reason": "quota exhausted",
            "stage": "teacher_generation",
            "checkpoint": {
                "checkpoint_index": 2,
                "stage": "teacher_generation",
                "task_input": "resume the latest checkpoint",
            },
            "unfinished_steps": ["resume from the latest checkpoint"],
            "recommended_resume_conditions": ["restore quota before resuming"],
        },
    }
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id=run_id,
            protocol_id="teacher_student_distill",
            title="Resume dedupe run",
            topic="avoid duplicate recovery jobs",
            summary=recovery_summary,
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status(run_id, "awaiting_resources"))

    resume_started = threading.Event()
    release_resume = threading.Event()
    resume_call_count = 0

    async def _blocking_run_agent_run(self: RuntimeService, *, run_id: str) -> None:
        nonlocal resume_call_count
        del self, run_id
        resume_call_count += 1
        resume_started.set()
        await asyncio.to_thread(release_resume.wait, 1.0)

    monkeypatch.setattr(RuntimeService, "_run_agent_run", _blocking_run_agent_run)

    with TestClient(app) as client:
        first_response = client.post(f"/v1/agent-runs/{run_id}/resume")
        assert first_response.status_code == 200
        assert resume_started.wait(timeout=1.0)

        second_response = client.post(f"/v1/agent-runs/{run_id}/resume")
        assert second_response.status_code == 200
        assert second_response.json()["status"] == "running"

        time.sleep(0.1)
        release_resume.set()

    assert resume_call_count == 1


def test_agent_run_resume_upgrades_legacy_payload_before_recovery_execution(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured_requests: list[Any] = []

    async def _capture_resumed_request(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        captured_requests.append(request)
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Legacy non-goal resume was upgraded before execution."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _capture_resumed_request)

    app = create_app()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    runtime_service.set_scheduler_poll_interval(0.05)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    run_id = "run-legacy-resume-upgrade-1"
    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id=run_id,
            protocol_id="teacher_student_distill",
            title="Legacy non-goal resume",
            topic="upgrade non-goal resume payload",
            summary={
                "task_input": "Resume the latest checkpoint without a linked goal.",
                "evidence_packets": [
                    {
                        "evidence_id": "src-1",
                        "title": "Deployment note",
                    }
                ],
                "recovery_state": {
                    "status": "awaiting_resources",
                    "action": "pause",
                    "reason": "budget exhausted",
                    "stage": "research_context_prepared",
                    "checkpoint": {
                        "checkpoint_index": 7,
                        "stage": "research_context_prepared",
                    },
                    "resume_payload": {
                        "strategy_default": "continue_from_checkpoint",
                        "state": {"legacy": True},
                    },
                },
            },
        )
    )
    asyncio.run(runtime_service._store.update_agent_run_status(run_id, "awaiting_resources"))
    asyncio.run(
        runtime_service._store.append_agent_run_event(
            run_id,
            {
                "type": "role_output",
                "payload": {
                    "candidate_id": "student",
                    "role_id": "student",
                    "content": "Student answer",
                    "model_id": "student-model",
                    "round_index": 1,
                },
            },
        )
    )
    asyncio.run(
        runtime_service._store.append_agent_run_event(
            run_id,
            {
                "type": "evaluation",
                "payload": {
                    "policy": {"mode": "judge"},
                    "selected_candidate_id": "student",
                    "scores": [{"candidate_id": "student", "score": 0.9}],
                },
            },
        )
    )
    asyncio.run(
        runtime_service._store.append_agent_run_artifact(
            run_id,
            artifact_id="run-legacy-resume-upgrade-1:run-summary",
            artifact_type="run_summary",
            title="Run Summary",
            uri=f"agent-run://{run_id}/artifacts/run-summary",
            metadata={
                "content": {
                    "protocol": "teacher_student_distill",
                    "selected_candidate_id": "student",
                    "final_answer": "Student answer",
                    "candidate_count": 1,
                    "debate_transcript": {"items": []},
                }
            },
        )
    )
    asyncio.run(
        runtime_service._store.append_agent_run_artifact(
            run_id,
            artifact_id="run-legacy-resume-upgrade-1:evidence-summary",
            artifact_type="evidence_summary",
            title="Evidence Summary",
            uri=f"agent-run://{run_id}/artifacts/evidence-summary",
            metadata={
                "content": {
                    "evidence_packets": [
                        {
                            "evidence_id": "src-1",
                            "title": "Deployment note",
                        }
                    ],
                    "collected_packet_count": 1,
                }
            },
        )
    )
    asyncio.run(
        runtime_service._store.append_agent_run_artifact(
            run_id,
            artifact_id="run-legacy-resume-upgrade-1:verification-summary",
            artifact_type="verification_summary",
            title="Verification Summary",
            uri=f"agent-run://{run_id}/artifacts/verification-summary",
            metadata={
                "content": {
                    "verified_candidate_ids": ["student"],
                    "verifications": [
                        {
                            "candidate_id": "student",
                            "status": "verified",
                            "summary": "Matches evidence.",
                            "citations": [],
                        }
                    ],
                }
            },
        )
    )
    asyncio.run(
        runtime_service._store.append_agent_run_artifact(
            run_id,
            artifact_id="run-legacy-resume-upgrade-1:role-task-snapshot",
            artifact_type="role_task_snapshot",
            title="Role Task Snapshot",
            uri=f"agent-run://{run_id}/artifacts/role-task-snapshot",
            metadata={
                "content": {
                    "roles": {
                        "teacher": {
                            "role_id": "teacher",
                            "status": "completed",
                            "stage": "teacher_generation",
                            "assigned_model_id": "teacher-model",
                            "candidate": {
                                "candidate_id": "teacher",
                                "role_id": "teacher",
                                "content": "Teacher answer",
                                "metadata": {"model_id": "teacher-model"},
                            },
                        }
                    }
                }
            },
        )
    )

    with TestClient(app) as client:
        resume_response = client.post(f"/v1/agent-runs/{run_id}/resume")
        assert resume_response.status_code == 200

        completed = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)

    assert completed["summary"]["final_answer"] == (
        "Legacy non-goal resume was upgraded before execution."
    )
    assert len(captured_requests) == 1
    assert captured_requests[0].metadata["resume_strategy"] == "continue_from_checkpoint"
    resume_payload = captured_requests[0].metadata["resume_payload"]
    assert resume_payload["stage"] == "research_context_prepared"
    assert resume_payload["executor"] == "continue_from_checkpoint"
    assert resume_payload["supported_actions"] == ["restart_attempt", "continue_from_checkpoint"]
    assert resume_payload["checkpoint"]["checkpoint_index"] == 7


def test_agent_run_resume_endpoint_resolves_exec_approval_before_resuming(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    approval_id = "exec-approval-agent-run-resume-1"

    async def _approval_then_success_run(self: Any, request: Any) -> MultiAgentRunResult:
        approval = self._exec_approval_store.get(approval_id)
        if approval is None or approval.status == "pending":
            if approval is None:
                self._exec_approval_store.create(
                    approval_id=approval_id,
                    command="print('agent run approval resume')",
                    shell="test",
                    scope="dangerous_command",
                    reason="Exec command requires approval.",
                    command_payload={
                        "command": "print('agent run approval resume')",
                        "shell": "test",
                        "workdir": str(tmp_path),
                        "env": None,
                        "timeout_sec": 5.0,
                        "background": False,
                        "tty": False,
                        "approval_state": "approved",
                    },
                )
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
            artifacts={"final_answer": "Approved agent run completed."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _approval_then_success_run)

    app = create_app()
    exec_approval_store = InMemoryApprovalStore()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
        exec_approval_store=exec_approval_store,
        exec_runtime=ExecRuntime(
            providers={"test": _ApiRuntimePythonDirectProvider()},
            default_shell="test",
        ),
    )
    runtime_service.set_scheduler_poll_interval(0.05)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "controlled_subagent_execution",
                "title": "Approval resume run",
                "topic": "resume after approval",
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        waiting = _wait_agent_run_until(client, run_id, {"awaiting_approval"}, timeout_seconds=4.0)
        assert waiting["summary"]["approval_state"]["approval_ids"] == [approval_id]

        resume_response = client.post(
            f"/v1/agent-runs/{run_id}/resume",
            json={
                "approval_id": approval_id,
                "decision": "approve_once",
                "reason": "approved for resume",
                "strategy": "continue_from_checkpoint",
            },
        )
        assert resume_response.status_code == 200
        assert resume_response.json()["run_id"] == run_id

        completed = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        assert completed["summary"]["final_answer"] == "Approved agent run completed."

    resolved = exec_approval_store.get(approval_id)
    assert resolved is not None
    assert resolved.status == "approved_once"
    assert resolved.execution_result is not None


def test_approval_resolve_endpoint_auto_resumes_linked_agent_run(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    approval_id = "exec-approval-agent-run-auto-resume-1"
    resumed_payloads: list[dict[str, Any]] = []

    async def _approval_then_success_run(self: Any, request: Any) -> MultiAgentRunResult:
        approval = self._exec_approval_store.get(approval_id)
        if approval is None or approval.status == "pending":
            if approval is None:
                self._exec_approval_store.create(
                    approval_id=approval_id,
                    command="print('agent run auto resume')",
                    shell="test",
                    scope="dangerous_command",
                    reason="Exec command requires approval.",
                    command_payload={
                        "command": "print('agent run auto resume')",
                        "shell": "test",
                        "workdir": str(tmp_path),
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
                                "command": "print('agent run auto resume')",
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
        resumed_payloads.append(resume_payload)
        role_task_snapshot = (
            dict(resume_payload.get("role_task_snapshot"))
            if isinstance(resume_payload.get("role_task_snapshot"), dict)
            else {}
        )
        tasks = dict(role_task_snapshot.get("tasks")) if isinstance(role_task_snapshot.get("tasks"), dict) else {}
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
            artifacts={"final_answer": "Approval auto-resume completed."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _approval_then_success_run)

    app = create_app()
    exec_approval_store = InMemoryApprovalStore()
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
        exec_approval_store=exec_approval_store,
        exec_runtime=ExecRuntime(
            providers={"test": _ApiRuntimePythonDirectProvider()},
            default_shell="test",
        ),
    )
    runtime_service.set_scheduler_poll_interval(0.05)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: object()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "controlled_subagent_execution",
                "title": "Approval auto-resume run",
                "topic": "generic approval resolve auto-resume",
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        waiting = _wait_agent_run_until(client, run_id, {"awaiting_approval"}, timeout_seconds=4.0)
        assert waiting["summary"]["approval_state"]["approval_ids"] == [approval_id]

        resolve_response = client.post(
            f"/v1/approvals/{approval_id}/resolve",
            json={"decision": "approve_once", "reason": "allow linked auto resume"},
        )
        assert resolve_response.status_code == 200
        assert resolve_response.json()["status"] == "approved_once"

        completed = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        assert completed["summary"]["final_answer"] == "Approval auto-resume completed."

    assert resumed_payloads
    resolved = exec_approval_store.get(approval_id)
    assert resolved is not None
    assert resolved.status == "approved_once"
    assert resolved.execution_result is not None


def test_linked_agent_run_exec_approval_remains_listed_after_runtime_restart(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    approval_id = "exec-approval-agent-run-restart-list-1"
    sessions_dir = tmp_path / "sessions"
    monkeypatch.setattr(
        "mochi.runtime.service.MultiAgentOrchestrator.run",
        _build_linked_agent_run_exec_approval_orchestrator(
            approval_id=approval_id,
            workdir=tmp_path,
            final_answer="Approval restart listing completed.",
        ),
    )

    with _create_agent_run_exec_test_client(
        sessions_dir=sessions_dir,
        exec_approval_store=InMemoryApprovalStore(),
    ) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "controlled_subagent_execution",
                "title": "Approval restart listing run",
                "topic": "list linked approval after restart",
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        waiting = _wait_agent_run_until(client, run_id, {"awaiting_approval"}, timeout_seconds=4.0)
        assert waiting["summary"]["approval_state"]["approval_ids"] == [approval_id]

    with _create_agent_run_exec_test_client(
        sessions_dir=sessions_dir,
        exec_approval_store=InMemoryApprovalStore(),
    ) as restarted_client:
        approvals_response = restarted_client.get("/v1/approvals?status=pending")
        assert approvals_response.status_code == 200
        approvals = approvals_response.json()
        linked = [item for item in approvals if item["approval_id"] == approval_id]
        assert len(linked) == 1
        assert linked[0]["status"] == "pending"
        assert linked[0]["tool_name"] == "exec_command"


def test_approval_resolve_endpoint_after_restart_auto_resumes_linked_agent_run(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    approval_id = "exec-approval-agent-run-restart-resume-1"
    resumed_payloads: list[dict[str, Any]] = []
    sessions_dir = tmp_path / "sessions"
    monkeypatch.setattr(
        "mochi.runtime.service.MultiAgentOrchestrator.run",
        _build_linked_agent_run_exec_approval_orchestrator(
            approval_id=approval_id,
            workdir=tmp_path,
            final_answer="Approval restart auto-resume completed.",
            resumed_payloads=resumed_payloads,
        ),
    )

    with _create_agent_run_exec_test_client(
        sessions_dir=sessions_dir,
        exec_approval_store=InMemoryApprovalStore(),
    ) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "controlled_subagent_execution",
                "title": "Approval restart auto-resume run",
                "topic": "resolve linked approval after restart",
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        waiting = _wait_agent_run_until(client, run_id, {"awaiting_approval"}, timeout_seconds=4.0)
        assert waiting["summary"]["approval_state"]["approval_ids"] == [approval_id]

    restarted_exec_approval_store = InMemoryApprovalStore()
    with _create_agent_run_exec_test_client(
        sessions_dir=sessions_dir,
        exec_approval_store=restarted_exec_approval_store,
    ) as restarted_client:
        resolve_response = restarted_client.post(
            f"/v1/approvals/{approval_id}/resolve",
            json={"decision": "approve_once", "reason": "allow linked restart auto resume"},
        )
        assert resolve_response.status_code == 200
        assert resolve_response.json()["status"] == "approved_once"

        completed = _wait_agent_run_until(
            restarted_client,
            run_id,
            {"succeeded"},
            timeout_seconds=4.0,
        )
        assert completed["summary"]["final_answer"] == "Approval restart auto-resume completed."

    assert resumed_payloads
    resolved = restarted_exec_approval_store.get(approval_id)
    assert resolved is not None
    assert resolved.status == "approved_once"
    assert resolved.execution_result is not None


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
            json={
                "input_message": "run something",
                "session_id": "runtime-s1",
                "workspace_dir": str(tmp_path / "project-workspace"),
            },
        )
        assert create_response.status_code == 200
        created = create_response.json()
        task_id = created["task_id"]
        assert created["project_workspace_dir"] == str(tmp_path / "project-workspace")
        expected_task_workspace = (
            tmp_path / "sessions" / "runtime-tasks" / task_id / "workspace"
        ).resolve()
        assert created["task_workspace_dir"] == str(expected_task_workspace)
        assert expected_task_workspace.is_dir()

        task_payload = _wait_until(client, task_id, {"awaiting_approval"})
        assert task_payload["project_workspace_dir"] == str(tmp_path / "project-workspace")
        assert task_payload["task_workspace_dir"] == str(expected_task_workspace)
        assert task_payload["pending_approval"] is not None
        assert task_payload["events"][0]["type"] == "thinking"
        assert task_payload["events"][1]["type"] == "tool_call_request"
        assert task_payload["events"][2]["type"] == "tool_call_result"

        list_response = client.get("/v1/tasks")
        assert list_response.status_code == 200
        listed = list_response.json()
        assert len(listed) == 1
        assert listed[0]["project_workspace_dir"] == str(tmp_path / "project-workspace")
        assert listed[0]["task_workspace_dir"] == str(expected_task_workspace)

        approvals_response = client.get("/v1/approvals?status=pending")
        assert approvals_response.status_code == 200
        approvals = approvals_response.json()
        assert len(approvals) == 1
        assert approvals[0]["approval_kind"] == "exec"
        assert approvals[0]["approval_scope"] == "dangerous_command"
        assert approvals[0]["requires_approval"] is True
        assert approvals[0]["replay_safe"] is False
        assert approvals[0]["security_decision"] == "require_approval"
        assert approvals[0]["policy_source"] == "static_policy"
        assert approvals[0]["policy_reason"] == "Command requires approval by policy."
        approval_id = approvals[0]["approval_id"]

        resolve_response = client.post(
            f"/v1/approvals/{approval_id}/resolve",
            json={"decision": "approve_once", "reason": "allowed"},
        )
        assert resolve_response.status_code == 200
        resolved = resolve_response.json()
        assert resolved["reason"] == "allowed"
        assert resolved["approval_kind"] == "exec"
        assert resolved["security_decision"] == "require_approval"

        done_payload = _wait_until(client, task_id, {"succeeded"})
        assert done_payload["events"][-1]["type"] == "final_answer"

    assert engine.permission_policy_calls[0] == {
        "autonomy_mode": "trusted_workspace",
        "require_approval_for_file_write": False,
        "require_approval_for_exec": True,
        "file_ops_scope": "workspace",
    }
    assert engine.permission_policy_calls[1] == {
        "autonomy_mode": "trusted_workspace",
        "require_approval_for_file_write": False,
        "require_approval_for_exec": True,
        "file_ops_scope": "workspace",
        "approved_tool_calls": [
            {
                "tool_name": "exec_command",
                "arguments": {"command": "dir", "shell": "cmd"},
            }
        ]
    }
    assert engine.task_workspace_calls == [
        str(expected_task_workspace),
        str(expected_task_workspace),
    ]


def test_file_mutation_approval_uses_replay_override_and_downgrades_save_rule(tmp_path: Path) -> None:
    app = create_app()
    engine = _RuntimeFileMutationFakeEngine()
    sessions_dir = tmp_path / "sessions"
    project_workspace = tmp_path / "project-workspace"
    project_workspace.mkdir(parents=True)
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "sessions_dir": str(sessions_dir),
            "workspace_dir": str(project_workspace),
        }
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/tasks",
            json={
                "input_message": "edit notes",
                "session_id": "runtime-s1",
                "workspace_dir": str(project_workspace),
            },
        )
        assert create_response.status_code == 200
        task_id = create_response.json()["task_id"]
        task_workspace = sessions_dir / "runtime-tasks" / task_id / "workspace"
        task_workspace.mkdir(parents=True, exist_ok=True)
        (task_workspace / "notes.py").write_text("print('alpha')\n", encoding="utf-8")

        waiting = _wait_until(client, task_id, {"awaiting_approval"})
        approval_id = waiting["pending_approval"]["id"]

        approvals_response = client.get("/v1/approvals?status=pending")
        assert approvals_response.status_code == 200
        approval = next(item for item in approvals_response.json() if item["approval_id"] == approval_id)
        assert approval["tool_name"] == "apply_patch"
        assert approval["approval_kind"] == "apply_patch"
        assert approval["allowed_decisions"] == ["approve_once", "reject"]
        assert approval["change_count"] == 1
        assert approval["file_changes"][0]["relative_path"] == "notes.py"

        edited_patch = "\n".join(
            [
                "*** Begin Patch",
                "*** Update File: notes.py",
                "@@",
                "-print('alpha')",
                "+print('gamma')",
                "*** End Patch",
            ]
        )
        preview_response = client.post(
            "/v1/workspace/patch/preview",
            json={
                "approval_id": approval_id,
                "patch": edited_patch,
            },
        )
        assert preview_response.status_code == 200
        preview_payload = preview_response.json()
        assert preview_payload["valid"] is True
        assert preview_payload["workspace_dir"] == str(task_workspace.resolve())
        assert preview_payload["file_changes"][0]["relative_path"] == "notes.py"

        resolve_response = client.post(
            f"/v1/approvals/{approval_id}/resolve",
            json={
                "decision": "approve_and_save_rule",
                "reason": "apply edited patch",
                "replay_override": {
                    "tool_name": "apply_patch",
                    "arguments": {"patch": edited_patch},
                },
            },
        )
        assert resolve_response.status_code == 200
        resolved = resolve_response.json()
        assert resolved["status"] == "approved_once"
        assert resolved["decision"] == "approve_once"

        done_payload = _wait_until(client, task_id, {"succeeded"})
        assert done_payload["status"] == "succeeded"
        assert done_payload["final_answer"] == "Applied approved file change to notes.py."

    assert (task_workspace / "notes.py").read_text(encoding="utf-8") == "print('gamma')\n"
    assert len(engine.permission_policy_calls) == 1


def test_controlled_subagent_execution_task_runs_in_task_workspace(tmp_path: Path) -> None:
    app = create_app()
    engine = _AgentRunModelBackedEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "sessions_dir": str(tmp_path / "sessions"),
                "security": {
                    "require_approval_for_exec": False,
                    "command_rules": [_CONTROLLED_SMOKE_COMMAND_RULE],
                },
            }
        )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/tasks",
            json={
                "input_message": "Run a controlled smoke command.",
                "task_type": "controlled_subagent_execution",
                "workspace_dir": str(tmp_path / "project-workspace"),
                "metadata": {
                    "protocol_config": {
                        "max_execution_requests": 1,
                        "max_commands_per_request": 1,
                        "default_timeout_sec": 30,
                        "background_allowed": False,
                    },
                    "selected_models_roles": {
                        "by_role": {
                            "planner": "controlled-planner-model",
                            "executor": "controlled-executor-model",
                            "controller": "controlled-controller-model",
                            "evaluator": "controlled-evaluator-model",
                            "judge": "controlled-judge-model",
                            "verifier": "controlled-verifier-model",
                        }
                    },
                },
            },
        )
        assert create_response.status_code == 200
        created = create_response.json()
        assert created["task_type"] == "controlled_subagent_execution"
        task_id = created["task_id"]
        expected_task_workspace = (
            tmp_path / "sessions" / "runtime-tasks" / task_id / "workspace"
        ).resolve()
        assert created["task_workspace_dir"] == str(expected_task_workspace)
        assert expected_task_workspace.is_dir()

        done_payload = _wait_until(client, task_id, {"succeeded"})
        assert done_payload["task_type"] == "controlled_subagent_execution"
        artifact_events = [
            event for event in done_payload["events"] if event.get("type") == "artifact"
        ]
        assert any(
            event.get("payload", {}).get("name") == "controlled_execution_runtime"
            for event in artifact_events
        )
        assert "Execution finished" in str(done_payload["final_answer"])


def test_delegated_multi_agent_task_runs_with_explicit_protocol(tmp_path: Path) -> None:
    app = create_app()
    engine = _AgentRunModelBackedEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "sessions_dir": str(tmp_path / "sessions"),
                "security": {
                    "require_approval_for_exec": False,
                    "command_rules": [_CONTROLLED_SMOKE_COMMAND_RULE],
                },
            }
        )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/tasks",
            json={
                "input_message": "Compare two deployment answers and select the stronger one.",
                "task_type": "delegated_multi_agent",
                "workspace_dir": str(tmp_path / "project-workspace"),
                "metadata": {
                    "protocol": "multi_agent_debate",
                    "protocol_config": {"rounds": 2},
                    "execution_policy": {"mode": "disabled"},
                    "selected_models_roles": {
                        "by_role": {
                            "debater_a": "debater-a-model",
                            "debater_b": "debater-b-model",
                            "judge": "judge-model",
                            "verifier": "verifier-model",
                        }
                    },
                },
            },
        )
        assert create_response.status_code == 200
        created = create_response.json()
        assert created["task_type"] == "delegated_multi_agent"

        done_payload = _wait_until(client, created["task_id"], {"succeeded"})
        assert done_payload["task_type"] == "delegated_multi_agent"
        assert "Argument B" in str(done_payload["final_answer"])
        artifact_events = [
            event for event in done_payload["events"] if event.get("type") == "artifact"
        ]
        assert any(
            event.get("payload", {}).get("name") == "debate_state"
            for event in artifact_events
        )


def test_create_delegated_subagent_task_defaults_to_generic_protocol_with_execution_policy(
    tmp_path: Path,
) -> None:
    runtime_service = RuntimeService(
        engine=_RuntimeFakeEngine(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )

    created = asyncio.run(
        runtime_service.create_delegated_subagent_task(
            objective="Summarize the strongest deployment guidance.",
            session_id="chat-session-42",
            suggested_roles=["teacher", "student"],
            suggested_models={"teacher": "teacher-model", "student": "student-model"},
            execution_budget={
                "max_execution_requests": 2,
                "max_commands_per_request": 1,
                "default_timeout_sec": 45,
                "background_allowed": False,
            },
        )
    )
    task = asyncio.run(runtime_service._store.get_task_run(created["task_id"]))

    assert task is not None
    assert created["task_type"] == "delegated_multi_agent"
    assert created["status"] == "queued"
    assert isinstance(created["display_name"], str)
    assert created["display_name"]
    assert created["display_name"] != "teacher"
    assert created["display_name"] != "Teacher"
    assert created["role"] == "teacher"
    assert created["instruction"] == "Summarize the strongest deployment guidance."
    assert created["objective"] == "Summarize the strongest deployment guidance."
    assert created["parent_session_id"] == "chat-session-42"
    display_name = created["display_name"]

    metadata = task["metadata"]
    assert metadata["protocol"] == "teacher_student_distill"
    assert metadata["protocol_config"] == {}
    assert metadata["execution_policy"]["mode"] == "controlled"
    assert metadata["execution_policy"]["max_execution_requests"] == 2
    assert metadata["execution_policy"]["default_timeout_sec"] == 45
    assert metadata["execution_policy"]["background_allowed"] is False
    assert metadata["delegated_subagent"] == {
        "display_name": display_name,
        "role": "teacher",
        "instruction": "Summarize the strongest deployment guidance.",
        "objective": "Summarize the strongest deployment guidance.",
        "parent_session_id": "chat-session-42",
        "status": "queued",
    }

    payload = asyncio.run(runtime_service.get_task(created["task_id"]))
    assert payload is not None
    assert payload["metadata"]["protocol"] == "teacher_student_distill"
    assert payload["metadata"]["selected_models_roles"]["by_role"] == {
        "teacher": "teacher-model",
        "student": "student-model",
    }
    assert payload["delegated_subagent"] == metadata["delegated_subagent"]
    assert payload["events"][0]["type"] == "delegated_subagent_created"
    assert payload["events"][0]["display_name"] == display_name
    assert payload["events"][0]["role"] == "teacher"
    assert payload["events"][0]["instruction"] == "Summarize the strongest deployment guidance."
    assert payload["events"][0]["parent_session_id"] == "chat-session-42"

    asyncio.run(runtime_service._store.update_task_status(created["task_id"], "running"))
    refreshed_task = asyncio.run(runtime_service._store.get_task_run(created["task_id"]))
    assert refreshed_task is not None
    assert refreshed_task["metadata"]["delegated_subagent"]["status"] == "running"


def test_delegated_task_message_appends_guidance_only_transcript(tmp_path: Path) -> None:
    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/tasks",
            json={
                "input_message": "Summarize the strongest deployment guidance.",
                "session_id": "chat-session-42",
                "task_type": "delegated_multi_agent",
                "metadata": {
                    "delegated_subagent": {
                        "display_name": "Ada",
                        "role": "teacher",
                        "instruction": "Summarize the strongest deployment guidance.",
                        "objective": "Summarize the strongest deployment guidance.",
                        "parent_session_id": "chat-session-42",
                        "status": "queued",
                    }
                },
            },
        )
        assert create_response.status_code == 200
        task_id = create_response.json()["task_id"]

        message_response = client.post(
            f"/v1/tasks/{task_id}/messages",
            json={
                "content": "Please narrow the answer to verified deployment blockers only.",
                "metadata": {"channel": "subagent-pane", "author": "user"},
            },
        )
        assert message_response.status_code == 200
        payload = message_response.json()
        assert payload["task_id"] == task_id

        message_events = [
            event
            for event in payload["events"]
            if event.get("type") == "delegated_subagent_message"
        ]
        assert len(message_events) == 1
        event = message_events[0]
        assert event["content"] == "Please narrow the answer to verified deployment blockers only."
        assert event["task_id"] == task_id
        assert event["display_name"] == "Ada"
        assert event["role"] == "teacher"
        assert event["parent_session_id"] == "chat-session-42"
        assert event["metadata"] == {"channel": "subagent-pane", "author": "user"}
        assert event["created_context"] == {
            "source": "task_message_api",
            "delivery": "guidance_only",
        }
        assert payload["delegated_subagent"]["display_name"] == "Ada"
        assert payload["delegated_subagent"]["role"] == "teacher"


def test_task_message_endpoint_rejects_non_delegated_tasks(tmp_path: Path) -> None:
    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/tasks",
            json={
                "input_message": "run something",
                "session_id": "runtime-s1",
            },
        )
        assert create_response.status_code == 200
        task_id = create_response.json()["task_id"]

        message_response = client.post(
            f"/v1/tasks/{task_id}/messages",
            json={"content": "Follow this up in the subagent pane."},
        )
        assert message_response.status_code == 400
        assert message_response.json()["detail"] == (
            "Task messages are only supported for delegated_multi_agent tasks."
        )

        task_payload = client.get(f"/v1/tasks/{task_id}")
        assert task_payload.status_code == 200
        assert all(
            event.get("type") != "delegated_subagent_message"
            for event in task_payload.json()["events"]
        )


def test_approvals_api_includes_and_resolves_standalone_exec_approvals(tmp_path: Path) -> None:
    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    exec_approval_store = InMemoryApprovalStore()
    created = exec_approval_store.create(
        approval_id="exec-approval-standalone-1",
        command="print('approved exec')",
        shell="test",
        scope="dangerous_command",
        reason="requires review",
        command_payload={
            "command": "print('approved exec')",
            "shell": "test",
            "workdir": str(tmp_path),
            "env": None,
            "timeout_sec": 5.0,
            "background": False,
            "tty": False,
            "approval_state": "approved",
        },
    )
    exec_runtime = ExecRuntime(
        providers={"test": _ApiRuntimePythonDirectProvider()},
        default_shell="test",
    )
    runtime_service = RuntimeService(
        engine=_RuntimeFakeEngine(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
        exec_approval_store=exec_approval_store,
        exec_runtime=exec_runtime,
    )
    app.state.runtime_service = runtime_service

    with TestClient(app) as client:
        pending_response = client.get("/v1/approvals?status=pending")
        assert pending_response.status_code == 200
        pending_items = pending_response.json()
        standalone = [
            item
            for item in pending_items
            if item["approval_id"] == created.approval_id
        ]
        assert len(standalone) == 1
        assert standalone[0]["tool_name"] == "exec_command"
        assert standalone[0]["task_id"] is None
        assert standalone[0]["source"] == "exec_runtime"
        assert standalone[0]["requires_approval"] is True
        assert standalone[0]["exec_approval_id"] == created.approval_id
        assert standalone[0]["exec_status"] == "pending"
        assert standalone[0]["exec_session_id"] is None
        assert standalone[0]["execution_result"] is None

        resolve_response = client.post(
            f"/v1/approvals/{created.approval_id}/resolve",
            json={"decision": "approve_once", "reason": "allowed"},
        )
        assert resolve_response.status_code == 200
        resolved = resolve_response.json()
        assert resolved["approval_id"] == created.approval_id
        assert resolved["status"] == "approved_once"
        assert resolved["reason"] == "allowed"
        assert resolved["source"] == "exec_runtime"
        assert resolved["execution_result"]["status"] == "completed"
        assert "approved exec" in resolved["execution_result"]["stdout"]
        assert resolved["exec_session_id"] is not None

        approved_response = client.get("/v1/approvals?status=approved_once")
        assert approved_response.status_code == 200
        approved_items = approved_response.json()
        assert any(item["approval_id"] == created.approval_id for item in approved_items)

    updated = exec_approval_store.get(created.approval_id)
    assert updated is not None
    assert updated.status == "approved_once"
    assert updated.execution_result is not None
    assert updated.execution_result["status"] == "completed"


def test_approvals_api_persists_standalone_exec_approvals_across_runtime_restart(
    tmp_path: Path,
) -> None:
    sessions_dir = tmp_path / "sessions"
    exec_approval_id = "exec-approval-standalone-restart-1"

    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(sessions_dir)}
    )

    initial_exec_store = PersistentApprovalStore(sessions_dir / "exec-approvals.db")
    initial_exec_store.create(
        approval_id=exec_approval_id,
        command="print('approved exec')",
        shell="test",
        scope="dangerous_command",
        reason="requires review",
        command_payload={
            "command": "print('approved exec')",
            "shell": "test",
            "workdir": str(tmp_path),
            "env": None,
            "timeout_sec": 5.0,
            "background": False,
            "tty": False,
            "approval_state": "approved",
        },
    )
    initial_runtime_service = RuntimeService(
        engine=_RuntimeFakeEngine(),
        store=RuntimeStore(sessions_dir / "runtime.db"),
        exec_approval_store=initial_exec_store,
        exec_runtime=ExecRuntime(
            providers={"test": _ApiRuntimePythonDirectProvider()},
            default_shell="test",
        ),
    )
    app.state.runtime_service = initial_runtime_service

    with TestClient(app) as client:
        pending_response = client.get("/v1/approvals?status=pending")
        assert pending_response.status_code == 200
        pending_items = pending_response.json()
        assert any(item["approval_id"] == exec_approval_id for item in pending_items)

    restarted_app = create_app()
    restarted_app.state.engine_factory = lambda: _RuntimeFakeEngine()
    restarted_app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(sessions_dir)}
    )
    restarted_runtime_service = RuntimeService(
        engine=_RuntimeFakeEngine(),
        store=RuntimeStore(sessions_dir / "runtime.db"),
        exec_approval_store=PersistentApprovalStore(sessions_dir / "exec-approvals.db"),
        exec_runtime=ExecRuntime(
            providers={"test": _ApiRuntimePythonDirectProvider()},
            default_shell="test",
        ),
    )
    restarted_app.state.runtime_service = restarted_runtime_service

    with TestClient(restarted_app) as restarted_client:
        restarted_pending = restarted_client.get("/v1/approvals?status=pending")
        assert restarted_pending.status_code == 200
        assert any(
            item["approval_id"] == exec_approval_id
            for item in restarted_pending.json()
        )

        resolve_response = restarted_client.post(
            f"/v1/approvals/{exec_approval_id}/resolve",
            json={"decision": "approve_once", "reason": "allowed"},
        )
        assert resolve_response.status_code == 200
        resolved = resolve_response.json()
        assert resolved["approval_id"] == exec_approval_id
        assert resolved["status"] == "approved_once"
        assert resolved["execution_result"]["status"] == "completed"
        assert resolved["exec_session_id"] is not None

    final_app = create_app()
    final_app.state.engine_factory = lambda: _RuntimeFakeEngine()
    final_app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(sessions_dir)}
    )
    final_runtime_service = RuntimeService(
        engine=_RuntimeFakeEngine(),
        store=RuntimeStore(sessions_dir / "runtime.db"),
        exec_approval_store=PersistentApprovalStore(sessions_dir / "exec-approvals.db"),
        exec_runtime=ExecRuntime(
            providers={"test": _ApiRuntimePythonDirectProvider()},
            default_shell="test",
        ),
    )
    final_app.state.runtime_service = final_runtime_service

    with TestClient(final_app) as final_client:
        approved_response = final_client.get("/v1/approvals?status=approved_once")
        assert approved_response.status_code == 200
        approved_items = approved_response.json()
        approved = next(item for item in approved_items if item["approval_id"] == exec_approval_id)
        assert approved["exec_status"] == "approved_once"
        assert approved["exec_session_id"] is not None
        assert approved["execution_result"]["status"] == "completed"
        assert "approved exec" in approved["execution_result"]["stdout"]


def test_runtime_task_resolve_syncs_linked_exec_approval(tmp_path: Path) -> None:
    app = create_app()
    exec_approval_store = InMemoryApprovalStore()
    engine = _RuntimeExecLinkedFakeEngine(exec_approval_store=exec_approval_store)
    exec_runtime = ExecRuntime(
        providers={"test": _ApiRuntimePythonDirectProvider()},
        default_shell="test",
    )
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )
    runtime_service = RuntimeService(
        engine=engine,
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
        exec_approval_store=exec_approval_store,
        exec_runtime=exec_runtime,
    )
    app.state.runtime_service = runtime_service

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/tasks",
            json={"input_message": "run exec command", "workspace_dir": str(tmp_path / "workspace")},
        )
        assert create_response.status_code == 200
        task_id = create_response.json()["task_id"]

        waiting = _wait_until(client, task_id, {"awaiting_approval"})
        assert waiting["pending_approval"] is not None

        approvals_response = client.get("/v1/approvals?status=pending")
        assert approvals_response.status_code == 200
        approvals = approvals_response.json()
        assert len(approvals) == 1
        runtime_approval = approvals[0]
        assert runtime_approval["tool_name"] == "exec_command"
        assert runtime_approval["source"] == "task_runtime"
        assert runtime_approval["exec_approval_id"] == "exec-approval-linked-runtime-1"
        assert runtime_approval["exec_status"] == "pending"
        assert runtime_approval["exec_session_id"] is None
        assert runtime_approval["execution_result"] is None

        resolve_response = client.post(
            f"/v1/approvals/{runtime_approval['approval_id']}/resolve",
            json={"decision": "approve_once", "reason": "allow linked exec"},
        )
        assert resolve_response.status_code == 200
        resolved = resolve_response.json()
        assert resolved["status"] == "approved_once"
        assert resolved["exec_approval_id"] == "exec-approval-linked-runtime-1"
        assert resolved["execution_result"]["status"] == "completed"
        assert "linked exec approved" in resolved["execution_result"]["stdout"]
        assert resolved["exec_session_id"] is not None

        done_payload = _wait_until(client, task_id, {"succeeded"})
        assert done_payload["status"] == "succeeded"
        assert "linked exec approved" in (done_payload["final_answer"] or "")
        assert done_payload["events"][-1]["type"] == "final_answer"
        assert "linked exec approved" in done_payload["events"][-1]["content"]

    linked = exec_approval_store.get(engine.linked_exec_approval_id or "")
    assert linked is not None
    assert linked.status == "approved_once"
    assert linked.execution_result is not None
    assert linked.execution_result["status"] == "completed"
    assert "linked exec approved" in linked.execution_result["stdout"]
    assert engine.second_run_started is False


def test_task_exec_approval_remains_resolvable_after_runtime_restart(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"
    approval_id: str
    task_id: str

    app = create_app()
    initial_exec_approval_store = InMemoryApprovalStore()
    initial_engine = _RuntimeExecLinkedFakeEngine(exec_approval_store=initial_exec_approval_store)
    initial_runtime_service = RuntimeService(
        engine=initial_engine,
        store=RuntimeStore(sessions_dir / "runtime.db"),
        exec_approval_store=initial_exec_approval_store,
        exec_runtime=ExecRuntime(
            providers={"test": _ApiRuntimePythonDirectProvider()},
            default_shell="test",
        ),
    )
    initial_runtime_service.set_scheduler_poll_interval(0.05)
    app.state.runtime_service = initial_runtime_service
    app.state.engine_factory = lambda: initial_engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(sessions_dir)}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/tasks",
            json={"input_message": "run exec command", "workspace_dir": str(tmp_path / "workspace")},
        )
        assert create_response.status_code == 200
        task_id = create_response.json()["task_id"]

        waiting = _wait_until(client, task_id, {"awaiting_approval"})
        approval_id = waiting["pending_approval"]["id"]

    restarted_app = create_app()
    restarted_exec_approval_store = InMemoryApprovalStore()
    restarted_runtime_service = RuntimeService(
        engine=_RuntimeExecLinkedFakeEngine(exec_approval_store=restarted_exec_approval_store),
        store=RuntimeStore(sessions_dir / "runtime.db"),
        exec_approval_store=restarted_exec_approval_store,
        exec_runtime=ExecRuntime(
            providers={"test": _ApiRuntimePythonDirectProvider()},
            default_shell="test",
        ),
    )
    restarted_runtime_service.set_scheduler_poll_interval(0.05)
    restarted_app.state.runtime_service = restarted_runtime_service
    restarted_app.state.engine_factory = lambda: _RuntimeExecLinkedFakeEngine(
        exec_approval_store=restarted_exec_approval_store
    )
    restarted_app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(sessions_dir)}
    )

    with TestClient(restarted_app) as restarted_client:
        approvals_response = restarted_client.get("/v1/approvals?status=pending")
        assert approvals_response.status_code == 200
        restarted_approval = next(
            item for item in approvals_response.json() if item["approval_id"] == approval_id
        )
        assert restarted_approval["source"] == "task_runtime"
        assert restarted_approval["exec_approval_id"] == "exec-approval-linked-runtime-1"
        assert restarted_approval["exec_status"] == "pending"

        resolve_response = restarted_client.post(
            f"/v1/approvals/{approval_id}/resolve",
            json={"decision": "approve_once", "reason": "allow linked exec after restart"},
        )
        assert resolve_response.status_code == 200
        resolved = resolve_response.json()
        assert resolved["status"] == "approved_once"
        assert resolved["execution_result"]["status"] == "completed"
        assert "linked exec approved" in resolved["execution_result"]["stdout"]

        done_payload = _wait_until(restarted_client, task_id, {"succeeded"})
        assert done_payload["status"] == "succeeded"
        assert "linked exec approved" in (done_payload["final_answer"] or "")

    rehydrated = restarted_exec_approval_store.get("exec-approval-linked-runtime-1")
    assert rehydrated is not None
    assert rehydrated.status == "approved_once"
    assert rehydrated.execution_result is not None
    assert rehydrated.execution_result["status"] == "completed"


def test_task_exec_approval_result_stays_visible_after_runtime_restart(tmp_path: Path) -> None:
    sessions_dir = tmp_path / "sessions"

    app = create_app()
    exec_approval_store = InMemoryApprovalStore()
    engine = _RuntimeExecLinkedFakeEngine(exec_approval_store=exec_approval_store)
    runtime_service = RuntimeService(
        engine=engine,
        store=RuntimeStore(sessions_dir / "runtime.db"),
        exec_approval_store=exec_approval_store,
        exec_runtime=ExecRuntime(
            providers={"test": _ApiRuntimePythonDirectProvider()},
            default_shell="test",
        ),
    )
    runtime_service.set_scheduler_poll_interval(0.05)
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(sessions_dir)}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/tasks",
            json={"input_message": "run exec command", "workspace_dir": str(tmp_path / "workspace")},
        )
        assert create_response.status_code == 200
        task_id = create_response.json()["task_id"]

        waiting = _wait_until(client, task_id, {"awaiting_approval"})
        approval_id = waiting["pending_approval"]["id"]

        resolve_response = client.post(
            f"/v1/approvals/{approval_id}/resolve",
            json={"decision": "approve_once", "reason": "allow linked exec"},
        )
        assert resolve_response.status_code == 200

        done_payload = _wait_until(client, task_id, {"succeeded"})
        assert done_payload["status"] == "succeeded"

    restarted_app = create_app()
    restarted_runtime_service = RuntimeService(
        engine=_RuntimeExecLinkedFakeEngine(exec_approval_store=InMemoryApprovalStore()),
        store=RuntimeStore(sessions_dir / "runtime.db"),
        exec_approval_store=InMemoryApprovalStore(),
        exec_runtime=ExecRuntime(
            providers={"test": _ApiRuntimePythonDirectProvider()},
            default_shell="test",
        ),
    )
    restarted_app.state.runtime_service = restarted_runtime_service
    restarted_app.state.engine_factory = lambda: _RuntimeExecLinkedFakeEngine(
        exec_approval_store=InMemoryApprovalStore()
    )
    restarted_app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(sessions_dir)}
    )

    with TestClient(restarted_app) as restarted_client:
        approved_response = restarted_client.get("/v1/approvals?status=approved_once")
        assert approved_response.status_code == 200
        approved = next(item for item in approved_response.json() if item["approval_id"] == approval_id)
        assert approved["source"] == "task_runtime"
        assert approved["exec_approval_id"] == "exec-approval-linked-runtime-1"
        assert approved["exec_status"] == "approved_once"
        assert approved["exec_session_id"] is not None
        assert approved["execution_result"]["status"] == "completed"
        assert "linked exec approved" in approved["execution_result"]["stdout"]


def test_approval_exec_session_endpoints_support_live_standalone_exec_sessions(tmp_path: Path) -> None:
    app = create_app()
    exec_approval_store = InMemoryApprovalStore()
    created = exec_approval_store.create(
        approval_id="exec-approval-live-session-1",
        command=_BACKGROUND_SMOKE_COMMAND,
        shell="test",
        scope="dangerous_command",
        reason="requires review",
        command_payload={
            "command": _BACKGROUND_SMOKE_COMMAND,
            "shell": "test",
            "workdir": str(tmp_path),
            "env": None,
            "timeout_sec": 30.0,
            "background": True,
            "tty": False,
            "approval_state": "approved",
        },
    )
    exec_runtime = ExecRuntime(
        providers={"test": _ApiRuntimePythonDirectProvider()},
        default_shell="test",
    )
    runtime_service = RuntimeService(
        engine=_RuntimeFakeEngine(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
        exec_approval_store=exec_approval_store,
        exec_runtime=exec_runtime,
    )
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        resolve_response = client.post(
            f"/v1/approvals/{created.approval_id}/resolve",
            json={"decision": "approve_once", "reason": "allowed"},
        )
        assert resolve_response.status_code == 200
        resolved = resolve_response.json()
        session_id = resolved["exec_session_id"]
        assert resolved["exec_status"] == "approved_once"
        assert session_id is not None

        session_response = client.get(
            f"/v1/approvals/{created.approval_id}/exec-session",
            params={"yield_time_ms": 50},
        )
        assert session_response.status_code == 200
        session_payload = session_response.json()
        assert session_payload["approval_id"] == created.approval_id
        assert session_payload["source"] == "exec_runtime"
        assert session_payload["exec_approval_id"] == created.approval_id
        assert session_payload["session_id"] == session_id
        assert session_payload["status"] == "approved_once"
        assert session_payload["live_status"] == "available"
        assert session_payload["session"]["status"] in {"running", "completed"}

        stop_response = client.post(f"/v1/approvals/{created.approval_id}/exec-session/stop")
        assert stop_response.status_code == 200
        stop_payload = stop_response.json()
        assert stop_payload["approval_id"] == created.approval_id
        assert stop_payload["session_id"] == session_id
        assert stop_payload["stop_status"] in {"killed", "completed"}
        assert stop_payload["session"]["status"] in {"killed", "completed"}

        approved_response = client.get("/v1/approvals?status=approved_once")
        assert approved_response.status_code == 200
        approved_items = approved_response.json()
        updated = next(item for item in approved_items if item["approval_id"] == created.approval_id)
        assert updated["execution_result"]["status"] == stop_payload["session"]["status"]
        assert updated["exec_session_id"] == session_id


def test_approval_exec_session_endpoints_reject_non_exec_approvals(tmp_path: Path) -> None:
    app = create_app()
    engine = _RuntimeFakeEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/tasks",
            json={"input_message": "run shell approval", "workspace_dir": str(tmp_path / "workspace")},
        )
        assert create_response.status_code == 200
        task_id = create_response.json()["task_id"]

        waiting = _wait_until(client, task_id, {"awaiting_approval"})
        approval_id = waiting["pending_approval"]["id"]

        session_response = client.get(f"/v1/approvals/{approval_id}/exec-session")
        assert session_response.status_code == 404

        stop_response = client.post(f"/v1/approvals/{approval_id}/exec-session/stop")
        assert stop_response.status_code == 404


def test_approval_exec_session_endpoints_return_conflict_without_live_session(tmp_path: Path) -> None:
    app = create_app()
    exec_approval_store = InMemoryApprovalStore()
    engine = _RuntimeExecLinkedFakeEngine(exec_approval_store=exec_approval_store)
    exec_runtime = ExecRuntime(
        providers={"test": _ApiRuntimePythonDirectProvider()},
        default_shell="test",
    )
    runtime_service = RuntimeService(
        engine=engine,
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
        exec_approval_store=exec_approval_store,
        exec_runtime=exec_runtime,
    )
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/tasks",
            json={"input_message": "run exec command", "workspace_dir": str(tmp_path / "workspace")},
        )
        assert create_response.status_code == 200
        task_id = create_response.json()["task_id"]

        waiting = _wait_until(client, task_id, {"awaiting_approval"})
        approval_id = waiting["pending_approval"]["id"]

        session_response = client.get(f"/v1/approvals/{approval_id}/exec-session")
        assert session_response.status_code == 409

        stop_response = client.post(f"/v1/approvals/{approval_id}/exec-session/stop")
        assert stop_response.status_code == 409


def test_linked_runtime_approval_stop_persists_latest_exec_state(tmp_path: Path) -> None:
    app = create_app()
    exec_approval_store = InMemoryApprovalStore()
    engine = _RuntimeExecLinkedBackgroundFakeEngine(exec_approval_store=exec_approval_store)
    exec_runtime = ExecRuntime(
        providers={"test": _ApiRuntimePythonDirectProvider()},
        default_shell="test",
    )
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )
    runtime_service = RuntimeService(
        engine=engine,
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
        exec_approval_store=exec_approval_store,
        exec_runtime=exec_runtime,
    )
    app.state.runtime_service = runtime_service

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/tasks",
            json={"input_message": "run background exec command", "workspace_dir": str(tmp_path / "workspace")},
        )
        assert create_response.status_code == 200
        task_id = create_response.json()["task_id"]

        waiting = _wait_until(client, task_id, {"awaiting_approval"})
        approval_id = waiting["pending_approval"]["id"]

        resolve_response = client.post(
            f"/v1/approvals/{approval_id}/resolve",
            json={"decision": "approve_once", "reason": "allow linked background exec"},
        )
        assert resolve_response.status_code == 200
        resolved = resolve_response.json()
        session_id = resolved["exec_session_id"]
        assert resolved["execution_result"]["status"] in {"running", "completed"}
        assert session_id is not None

        stop_response = client.post(f"/v1/approvals/{approval_id}/exec-session/stop")
        assert stop_response.status_code == 200
        stop_payload = stop_response.json()
        assert stop_payload["session_id"] == session_id
        assert stop_payload["session"]["status"] in {"killed", "completed"}

        approved_response = client.get("/v1/approvals?status=approved_once")
        assert approved_response.status_code == 200
        approved_items = approved_response.json()
        updated = next(item for item in approved_items if item["approval_id"] == approval_id)
        assert updated["exec_approval_id"] == "exec-approval-linked-runtime-bg-1"
        assert updated["execution_result"]["status"] == stop_payload["session"]["status"]
        assert updated["exec_session_id"] == session_id


def test_standalone_exec_approval_save_rule_without_valid_rule_downgrades_to_approve_once(
    tmp_path: Path,
) -> None:
    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    exec_approval_store = InMemoryApprovalStore()
    created = exec_approval_store.create(
        approval_id="exec-approval-no-rule-1",
        command="print('approved exec')",
        shell="test",
        scope="dangerous_command",
        reason="requires review",
        command_payload={
            "command": "print('approved exec')",
            "shell": "test",
            "workdir": str(tmp_path),
            "env": None,
            "timeout_sec": 5.0,
            "background": False,
            "tty": False,
            "approval_state": "approved",
        },
    )
    runtime_service = RuntimeService(
        engine=_RuntimeFakeEngine(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
        exec_approval_store=exec_approval_store,
        exec_runtime=ExecRuntime(
            providers={"test": _ApiRuntimePythonDirectProvider()},
            default_shell="test",
        ),
    )
    app.state.runtime_service = runtime_service

    with TestClient(app) as client:
        resolve_response = client.post(
            f"/v1/approvals/{created.approval_id}/resolve",
            json={"decision": "approve_and_save_rule", "reason": "allow but no rule"},
        )
        assert resolve_response.status_code == 200
        resolved = resolve_response.json()
        assert resolved["status"] == "approved_once"
        assert resolved["decision"] == "approve_once"

    updated = exec_approval_store.get(created.approval_id)
    assert updated is not None
    assert updated.status == "approved_once"
    assert updated.metadata.get("saved_rule") is None


def test_task_approval_save_rule_without_valid_rule_downgrades_to_approve_once(tmp_path: Path) -> None:
    app = create_app()
    engine = _RuntimeFakeEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/tasks",
            json={"input_message": "run shell approval", "workspace_dir": str(tmp_path / "workspace")},
        )
        assert create_response.status_code == 200
        task_id = create_response.json()["task_id"]

        waiting = _wait_until(client, task_id, {"awaiting_approval"})
        approval_id = waiting["pending_approval"]["id"]

        resolve_response = client.post(
            f"/v1/approvals/{approval_id}/resolve",
            json={"decision": "approve_and_save_rule", "reason": "allow but no rule"},
        )
        assert resolve_response.status_code == 200
        resolved = resolve_response.json()
        assert resolved["status"] == "approved_once"
        assert resolved["decision"] == "approve_once"


def test_agent_runs_api_flow(tmp_path: Path) -> None:
    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "research-protocol-v1",
                "title": "Cancer literature review",
                "topic": "oncology",
                "reasoning_effort": "high",
                "selected_models_roles": {
                    "planner": {"model": "qwen3", "role": "lead"},
                    "worker": {"model": "llama", "role": "analysis"},
                },
                "evaluation_policy": {"strict": True},
                "schedule": {"cron": "0 9 * * 1", "timezone": "Asia/Taipei"},
                "summary": {"objective": "collect evidence"},
                "evidence_status": {"task": "pending", "rag": "pending"},
                "artifacts": [
                    {
                        "artifact_id": "artifact-1",
                        "artifact_type": "report",
                        "title": "Initial brief",
                        "uri": "memory://artifact-1",
                    }
                ],
            },
        )
        assert create_response.status_code == 200
        created = create_response.json()
        run_id = created["run_id"]
        assert created["status"] == "created"
        assert created["reasoning_effort"] == "high"
        assert created["schedule"] == {"cron": "0 9 * * 1", "timezone": "Asia/Taipei"}
        assert created["artifacts"][0]["artifact_id"] == "artifact-1"

        list_response = client.get("/v1/agent-runs")
        assert list_response.status_code == 200
        listed = list_response.json()
        assert len(listed) == 1
        assert listed[0]["run_id"] == run_id
        assert listed[0]["status"] == "created"
        assert listed[0]["reasoning_effort"] == "high"

        get_response = client.get(f"/v1/agent-runs/{run_id}")
        assert get_response.status_code == 200
        fetched = get_response.json()
        assert fetched["protocol_id"] == "research-protocol-v1"
        assert fetched["reasoning_effort"] == "high"
        assert fetched["events"][0]["type"] == "run_created"

        guidance_response = client.post(
            f"/v1/agent-runs/{run_id}/guidance",
            json={
                "guidance": "Prioritize RCT sources first.",
                "author": "user",
                "metadata": {"priority": "high"},
            },
        )
        assert guidance_response.status_code == 200
        guidance_payload = guidance_response.json()
        guidance_events = [event for event in guidance_payload["events"] if event["type"] == "guidance"]
        assert guidance_events[-1]["guidance"] == "Prioritize RCT sources first."
        assert guidance_events[-1]["author"] == "user"

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200
        assert start_response.json()["status"] in {"running", "succeeded"}

        final_payload = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        assert final_payload["finished_at"] is not None
        assert len(final_payload["artifacts"]) >= 1
        assert final_payload["summary"]["final_answer"]
        assert final_payload["evidence_status"]["selected_candidate_id"]
        event_types = [event["type"] for event in final_payload["events"]]
        assert "run_started" in event_types
        assert "role_output" in event_types
        assert "evaluation" in event_types
        assert "artifact" in event_types


def test_agent_run_pause_cancels_background_job(tmp_path: Path, monkeypatch: Any) -> None:
    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    async def _slow_run(self: Any, request: Any) -> MultiAgentRunResult:
        await asyncio.sleep(5)
        return MultiAgentRunResult(
            run_id=request.run_id or "slow-run",
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _slow_run)

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Pause test",
                "topic": "verify pause behavior",
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200
        assert start_response.json()["status"] == "running"

        pause_response = client.post(f"/v1/agent-runs/{run_id}/pause")
        assert pause_response.status_code == 200
        assert pause_response.json()["status"] == "paused"

        paused_payload = _wait_agent_run_until(client, run_id, {"paused"})
        assert paused_payload["status"] == "paused"
        assert paused_payload["finished_at"] is None
        assert "run_paused" in [event["type"] for event in paused_payload["events"]]


def test_agent_runs_api_flow_uses_configured_model_generation(tmp_path: Path) -> None:
    app = create_app()
    engine = _AgentRunModelBackedEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Model-backed run",
                "topic": "deployment summary",
                "reasoning_effort": "medium",
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                        "verifier": "verifier-model",
                        "judge": "judge-model",
                    }
                },
                "summary": {
                    "evidence_packets": [
                        {
                            "evidence_id": "src-1",
                            "title": "Deployment note",
                            "content": "The approved note matches the student answer and rejects the teacher claim.",
                        }
                    ]
                }
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        assert final_payload["summary"]["final_answer"] == "Student concise final answer."
        assert final_payload["summary"]["selected_candidate_id"] == "student"
        assert final_payload["reasoning_effort"] == "medium"
        assert final_payload["evidence_status"]["evidence_gate_counts"] == {
            "verified": 1,
            "skipped": 0,
            "failed": 1,
        }
        artifact_types = [artifact["artifact_type"] for artifact in final_payload["artifacts"]]
        assert "run_summary" in artifact_types
        assert "verification_summary" in artifact_types
        assert "subagent_runtime" in artifact_types

    assert [call["model_id"] for call in engine.calls] == [
        "teacher-model",
        "student-model",
        "verifier-model",
        "judge-model",
    ]
    assert all(call["reasoning_effort"] == "medium" for call in engine.calls)


def test_agent_runs_api_flow_supports_dr_zero_dataset_package(tmp_path: Path) -> None:
    app = create_app()
    engine = _AgentRunModelBackedEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "dr_zero_self_evolve",
                "title": "Dr.Zero run",
                "topic": "deployment search self-evolution",
                "selected_models_roles": {
                    "by_role": {
                        "proposer": "proposer-model",
                        "solver": "solver-model",
                        "verifier": "verifier-model",
                        "judge": "judge-model",
                    }
                },
                "summary": {
                    "protocol_config": {
                        "proposal_sample_size": 1,
                        "solver_rollouts_per_task": 1,
                    },
                    "evidence_packets": [
                        {
                            "evidence_id": "src-1",
                            "title": "Deployment note",
                            "content": "The supported deployment answer matches the solver.",
                        }
                    ],
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        assert final_payload["protocol_id"] == "dr_zero_self_evolve"
        assert final_payload["summary"]["selected_candidate_id"] == "solver_1_1"
        artifact_types = [artifact["artifact_type"] for artifact in final_payload["artifacts"]]
        assert "dataset_record" in artifact_types
        assert "dataset_package_snapshot" in artifact_types
        assert "synthetic_tasks" in artifact_types
        assert "solver_rollouts" in artifact_types
        assert "reward_summary" in artifact_types

        dataset_response = client.get(f"/v1/agent-runs/{run_id}/packages/dataset")
        assert dataset_response.status_code == 200
        dataset_package = dataset_response.json()
        assert dataset_package["protocol_id"] == "dr_zero_self_evolve"
        assert dataset_package["training_ready_count"] == 1
        record = dataset_package["training_ready_records"][0]["record"]
        assert record["metadata"]["dataset_mode"] == "self_evolve_search"
        assert record["synthetic_tasks"]["tasks"][0]["task_id"] == "task-1"

    assert [call["model_id"] for call in engine.calls] == [
        "proposer-model",
        "solver-model",
        "verifier-model",
        "judge-model",
    ]


def test_agent_runs_api_flow_supports_controlled_execution_dataset_package(tmp_path: Path) -> None:
    app = create_app()
    engine = _AgentRunModelBackedEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "sessions_dir": str(tmp_path / "sessions"),
                "security": {
                    "require_approval_for_exec": False,
                    "command_rules": [_CONTROLLED_SMOKE_COMMAND_RULE],
                },
            }
        )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "controlled_subagent_execution",
                "title": "Controlled execution run",
                "topic": "run a controlled smoke command",
                "selected_models_roles": {
                    "by_role": {
                        "planner": "controlled-planner-model",
                        "executor": "controlled-executor-model",
                        "controller": "controlled-controller-model",
                        "evaluator": "controlled-evaluator-model",
                        "judge": "controlled-judge-model",
                        "verifier": "controlled-verifier-model",
                    }
                },
                "summary": {
                    "protocol_config": {
                        "max_execution_requests": 1,
                        "default_timeout_sec": 30,
                        "background_allowed": False,
                    },
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        assert final_payload["protocol_id"] == "controlled_subagent_execution"
        artifact_types = [artifact["artifact_type"] for artifact in final_payload["artifacts"]]
        assert "execution_requests" in artifact_types
        assert "controller_decisions" in artifact_types
        assert "execution_results" in artifact_types
        assert "controlled_execution_runtime" in artifact_types

        dataset_response = client.get(f"/v1/agent-runs/{run_id}/packages/dataset")
        assert dataset_response.status_code == 200
        dataset_package = dataset_response.json()
        assert dataset_package["protocol_id"] == "controlled_subagent_execution"
        record = dataset_package["all_records"][0]["record"]
        assert record["metadata"]["dataset_mode"] == "agentic_execution"
        assert record["supervision"]["type"] == "agentic_execution_trace"


def test_runtime_service_persists_live_collector_shard_snapshot_before_run_completion(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    run_id = "live-collector-run-1"
    live_snapshot_artifacts: list[dict[str, Any]] = []
    live_dataset_record_artifacts: list[dict[str, Any]] = []
    received_callback: list[object | None] = []

    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id=run_id,
            protocol_id="teacher_student_distill",
            title="Live collector snapshot run",
            topic="collector live snapshot",
            summary={
                "task_input": "Collect a Discourse topic into dataset records.",
                "objective": "Collect a Discourse topic into dataset records.",
            },
        )
    )

    original_append_agent_run_artifact = runtime_service._store.append_agent_run_artifact

    async def _capturing_append_agent_run_artifact(
        run_id: str,
        *,
        artifact_id: str | None,
        artifact_type: str,
        title: str | None = None,
        uri: str | None = None,
        mime_type: str | None = None,
        size_bytes: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await original_append_agent_run_artifact(
            run_id,
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            title=title,
            uri=uri,
            mime_type=mime_type,
            size_bytes=size_bytes,
            metadata=metadata,
        )
        if (
            artifact_type == "collector_shard_manifest"
            and isinstance(artifact_id, str)
            and "collector-shard-live" in artifact_id
        ):
            live_snapshot_artifacts.append(dict(metadata or {}))
        if (
            artifact_type == "dataset_record"
            and isinstance(artifact_id, str)
            and "collector-dataset-live" in artifact_id
        ):
            live_dataset_record_artifacts.append(dict(metadata or {}))

    monkeypatch.setattr(
        runtime_service._store,
        "append_agent_run_artifact",
        _capturing_append_agent_run_artifact,
    )

    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        received_callback.append(request.runtime_event_callback or request.event_callback)
        provenance_event = MultiAgentRunEvent(
            run_id=request.run_id or run_id,
            seq=5,
            type="artifact",
            payload={
                "name": "collector_record_provenance",
                "content": {
                    "records": [
                        {
                            "source_url": "https://forum.example/t/api-examples/274354/1",
                            "source_id": "topic:274354:post:1",
                            "adapter_name": "discourse_topic_adapter",
                            "tool_name": "discourse_topic_collect",
                            "policy_disposition": "allow",
                            "shard_id": "discourse-topic-274354",
                        }
                    ]
                },
            },
            timestamp="2026-06-24T00:00:00+00:00",
        )
        dataset_event = MultiAgentRunEvent(
            run_id=request.run_id or run_id,
            seq=6,
            type="artifact",
            payload={
                "name": "collector_dataset_records",
                "content": {
                    "records": [
                        {
                            "input": "Topic: API examples",
                            "target": {"answer": "First collected post"},
                            "metadata": {},
                        }
                    ]
                },
            },
            timestamp="2026-06-24T00:00:00+00:00",
        )
        shard_event = MultiAgentRunEvent(
            run_id=request.run_id or run_id,
            seq=7,
            type="artifact",
            payload={
                "name": "collector_shard_manifests",
                "content": {
                    "shards": [
                        {
                            "shard_id": "discourse-topic-274354",
                            "adapter_name": "discourse_topic_adapter",
                            "status": "running",
                            "source": {
                                "url": "https://forum.example/t/api-examples/274354",
                                "id": "topic:274354",
                            },
                            "progress": {
                                "cursor": "101",
                                "items_collected": 2,
                                "items_emitted": 2,
                            },
                        }
                    ]
                },
            },
            timestamp="2026-06-24T00:00:00+00:00",
        )
        callback = request.runtime_event_callback or request.event_callback
        if callback is not None:
            await callback(provenance_event)
            await callback(dataset_event)
            await callback(shard_event)
        assert live_snapshot_artifacts
        assert live_dataset_record_artifacts
        return MultiAgentRunResult(
            run_id=run_id,
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={
                "collector_dataset_records": {
                    "records": [
                        {
                            "input": "Topic: API examples",
                            "target": {"answer": "First collected post"},
                            "metadata": {
                                "collector_provenance": {
                                    "source_url": "https://forum.example/t/api-examples/274354/1",
                                    "source_id": "topic:274354:post:1",
                                    "adapter_name": "discourse_topic_adapter",
                                    "tool_name": "discourse_topic_collect",
                                    "policy_disposition": "allow",
                                    "shard_id": "discourse-topic-274354",
                                }
                            },
                        }
                    ]
                },
                "collector_record_provenance": {
                    "records": [
                        {
                            "source_url": "https://forum.example/t/api-examples/274354/1",
                            "source_id": "topic:274354:post:1",
                            "adapter_name": "discourse_topic_adapter",
                            "tool_name": "discourse_topic_collect",
                            "policy_disposition": "allow",
                            "shard_id": "discourse-topic-274354",
                        }
                    ]
                },
                "collector_shard_manifests": {
                    "shards": [
                        {
                            "shard_id": "discourse-topic-274354",
                            "adapter_name": "discourse_topic_adapter",
                            "status": "completed",
                            "source": {
                                "url": "https://forum.example/t/api-examples/274354",
                                "id": "topic:274354",
                            },
                            "progress": {
                                "cursor": "102",
                                "items_collected": 3,
                                "items_emitted": 3,
                            },
                        }
                    ]
                }
            },
            events=[provenance_event, dataset_event, shard_event],
            metadata={},
        )

    monkeypatch.setattr(MultiAgentOrchestrator, "run", _fake_run)

    asyncio.run(runtime_service._run_agent_run(run_id=run_id))

    assert received_callback and received_callback[0] is not None
    assert live_snapshot_artifacts
    assert live_snapshot_artifacts[0]["content"]["shards"][0]["shard_id"] == (
        "discourse-topic-274354"
    )
    assert live_snapshot_artifacts[0]["content"]["shards"][0]["status"] == "running"
    assert live_dataset_record_artifacts
    assert live_dataset_record_artifacts[0]["record"]["target"]["answer"] == "First collected post"
    assert live_dataset_record_artifacts[0]["record"]["metadata"]["collector_provenance"][
        "source_id"
    ] == "topic:274354:post:1"

    persisted_run = asyncio.run(runtime_service.get_agent_run(run_id))
    dataset_artifacts = [
        artifact
        for artifact in persisted_run["artifacts"]
        if artifact["artifact_type"] == "dataset_record"
    ]
    assert len(dataset_artifacts) == 1


def test_runtime_service_persists_live_subagent_runtime_snapshot_before_run_completion(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    runtime_service = RuntimeService(
        engine=object(),
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
    )
    run_id = "live-subagent-runtime-run-1"
    live_runtime_artifacts: list[dict[str, Any]] = []
    received_callback: list[object | None] = []

    asyncio.run(
        runtime_service._store.create_agent_run(
            run_id=run_id,
            protocol_id="teacher_student_distill",
            title="Live subagent runtime run",
            topic="subagent runtime live snapshot",
            summary={
                "task_input": "Summarize the deployment risks.",
                "objective": "Summarize the deployment risks.",
            },
        )
    )

    original_append_agent_run_artifact = runtime_service._store.append_agent_run_artifact

    async def _capturing_append_agent_run_artifact(
        run_id: str,
        *,
        artifact_id: str | None,
        artifact_type: str,
        title: str | None = None,
        uri: str | None = None,
        mime_type: str | None = None,
        size_bytes: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await original_append_agent_run_artifact(
            run_id,
            artifact_id=artifact_id,
            artifact_type=artifact_type,
            title=title,
            uri=uri,
            mime_type=mime_type,
            size_bytes=size_bytes,
            metadata=metadata,
        )
        if (
            artifact_type == "subagent_runtime"
            and isinstance(artifact_id, str)
            and "subagent-runtime-live" in artifact_id
        ):
            live_runtime_artifacts.append(dict(metadata or {}))

    monkeypatch.setattr(
        runtime_service._store,
        "append_agent_run_artifact",
        _capturing_append_agent_run_artifact,
    )

    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        received_callback.append(request.runtime_event_callback or request.event_callback)
        runtime_event = MultiAgentRunEvent(
            run_id=request.run_id or run_id,
            seq=4,
            type="artifact",
            payload={
                "name": "subagent_runtime",
                "content": {
                    "invocation_count": 1,
                    "completed_invocation_count": 1,
                    "token_tracked_invocation_count": 1,
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "total_tokens": 18,
                    "generation_time_ms": 12.5,
                    "finish_reason_counts": {"stop": 1},
                    "approval_pending": [],
                    "risky_tool_events": [],
                    "invocations": [],
                },
            },
            timestamp="2026-06-24T00:00:00+00:00",
        )
        callback = request.runtime_event_callback or request.event_callback
        if callback is not None:
            await callback(runtime_event)
        assert live_runtime_artifacts
        return MultiAgentRunResult(
            run_id=run_id,
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={
                "subagent_runtime": {
                    "invocation_count": 3,
                    "completed_invocation_count": 3,
                    "token_tracked_invocation_count": 3,
                    "input_tokens": 24,
                    "output_tokens": 15,
                    "total_tokens": 39,
                    "generation_time_ms": 37.5,
                    "finish_reason_counts": {"stop": 3},
                    "approval_pending": [],
                    "risky_tool_events": [],
                    "invocations": [],
                }
            },
            events=[runtime_event],
            metadata={},
        )

    monkeypatch.setattr(MultiAgentOrchestrator, "run", _fake_run)

    asyncio.run(runtime_service._run_agent_run(run_id=run_id))

    assert received_callback and received_callback[0] is not None
    assert live_runtime_artifacts
    assert live_runtime_artifacts[0]["content"]["total_tokens"] == 18
    assert live_runtime_artifacts[0]["content"]["finish_reason_counts"] == {"stop": 1}

    persisted_run = asyncio.run(runtime_service.get_agent_run(run_id))
    subagent_runtime_artifacts = [
        artifact
        for artifact in persisted_run["artifacts"]
        if artifact["artifact_type"] == "subagent_runtime"
    ]
    assert len(subagent_runtime_artifacts) >= 2


def test_agent_run_background_exec_persists_detached_job_and_supports_run_scoped_session_lookup(
    tmp_path: Path,
) -> None:
    app = create_app()
    engine = _BackgroundControlledExecAgentRunEngine()
    exec_runtime = ExecRuntime(
        providers={"test": _ApiRuntimePythonDirectProvider()},
        default_shell="test",
    )
    runtime_service = RuntimeService(
        engine=engine,
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
        exec_runtime=exec_runtime,
    )
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "sessions_dir": str(tmp_path / "sessions"),
            "security": {
                "require_approval_for_exec": False,
                "command_rules": [_BACKGROUND_SMOKE_COMMAND_RULE],
            },
        }
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "controlled_subagent_execution",
                "title": "Controlled background execution run",
                "topic": "run a detached smoke command",
                "selected_models_roles": {
                    "by_role": {
                        "planner": "controlled-planner-model",
                        "executor": "controlled-executor-model",
                        "controller": "controlled-controller-model",
                        "evaluator": "controlled-evaluator-model",
                        "judge": "controlled-judge-model",
                        "verifier": "controlled-verifier-model",
                    }
                },
                "summary": {
                    "protocol_config": {
                        "max_execution_requests": 1,
                        "default_timeout_sec": 30,
                        "background_allowed": True,
                    },
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        artifact_types = [artifact["artifact_type"] for artifact in final_payload["artifacts"]]
        assert "detached_exec_jobs" in artifact_types
        runtime_artifact = next(
            artifact
            for artifact in final_payload["artifacts"]
            if artifact["artifact_type"] == "controlled_execution_runtime"
        )
        assert runtime_artifact["metadata"]["content"]["detached_exec_job_count"] == 1
        detached_artifact = next(
            artifact
            for artifact in final_payload["artifacts"]
            if artifact["artifact_type"] == "detached_exec_jobs"
        )
        detached_jobs = detached_artifact["metadata"]["content"]["items"]
        assert len(detached_jobs) == 1
        session_id = detached_jobs[0]["session_id"]
        assert detached_jobs[0]["reattach_supported"] is True
        log_path = Path(detached_jobs[0]["log_path"])
        checkpoint_dir = Path(detached_jobs[0]["checkpoint_dir"])
        assert log_path.name == "session.log"
        assert checkpoint_dir.name == "checkpoints"
        assert checkpoint_dir.is_dir()

        session_response = client.get(
            f"/v1/agent-runs/{run_id}/exec/{session_id}",
            params={"yield_time_ms": 50},
        )
        assert session_response.status_code == 200
        session_payload = session_response.json()
        assert session_payload["run_id"] == run_id
        assert session_payload["session_id"] == session_id
        assert session_payload["associated"] is True
        assert "bg-start" in session_payload["lease"]["command"]
        assert "sleep(5)" in session_payload["lease"]["command"]
        assert Path(session_payload["lease"]["log_path"]) == log_path
        assert Path(session_payload["lease"]["checkpoint_dir"]) == checkpoint_dir
        assert session_payload["live_status"] in {"available", "unavailable"}
        if session_payload["session"] is not None:
            assert session_payload["session"]["status"] in {"running", "completed"}
        log_text = ""
        for _ in range(10):
            if log_path.exists():
                log_text = log_path.read_text(encoding="utf-8")
                if "bg-start" in log_text:
                    break
            time.sleep(0.05)
        assert "bg-start" in log_text


def test_agent_run_background_exec_supports_run_scoped_stop(tmp_path: Path) -> None:
    app = create_app()
    engine = _BackgroundControlledExecAgentRunEngine()
    exec_runtime = ExecRuntime(
        providers={"test": _ApiRuntimePythonDirectProvider()},
        default_shell="test",
    )
    runtime_service = RuntimeService(
        engine=engine,
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
        exec_runtime=exec_runtime,
    )
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "sessions_dir": str(tmp_path / "sessions"),
            "security": {
                "require_approval_for_exec": False,
                "command_rules": [_BACKGROUND_SMOKE_COMMAND_RULE],
            },
        }
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "controlled_subagent_execution",
                "title": "Controlled background execution stop run",
                "topic": "stop a detached smoke command",
                "selected_models_roles": {
                    "by_role": {
                        "planner": "controlled-planner-model",
                        "executor": "controlled-executor-model",
                        "controller": "controlled-controller-model",
                        "evaluator": "controlled-evaluator-model",
                        "judge": "controlled-judge-model",
                        "verifier": "controlled-verifier-model",
                    }
                },
                "summary": {
                    "protocol_config": {
                        "max_execution_requests": 1,
                        "default_timeout_sec": 30,
                        "background_allowed": True,
                    },
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        detached_artifact = next(
            artifact
            for artifact in final_payload["artifacts"]
            if artifact["artifact_type"] == "detached_exec_jobs"
        )
        session_id = detached_artifact["metadata"]["content"]["items"][0]["session_id"]

        stop_response = client.post(f"/v1/agent-runs/{run_id}/exec/{session_id}/stop")
        assert stop_response.status_code == 200
        stop_payload = stop_response.json()
        assert stop_payload["run_id"] == run_id
        assert stop_payload["session_id"] == session_id
        assert stop_payload["associated"] is True
        assert stop_payload["stop_status"] in {"killed", "completed"}
        if stop_payload["session"] is not None:
            assert stop_payload["session"]["status"] in {"killed", "completed"}

        updated = client.get(f"/v1/agent-runs/{run_id}")
        assert updated.status_code == 200
        stop_events = [event for event in updated.json()["events"] if event["type"] == "detached_exec_stop"]
        assert stop_events
        assert stop_events[-1]["session_id"] == session_id
        assert stop_events[-1]["status"] in {"killed", "completed"}


def test_agent_run_background_exec_supports_reattach(tmp_path: Path) -> None:
    app = create_app()
    engine = _BackgroundControlledExecAgentRunEngine()
    exec_runtime = ExecRuntime(
        providers={"test": _ApiRuntimePythonDirectProvider()},
        default_shell="test",
    )
    runtime_service = RuntimeService(
        engine=engine,
        store=RuntimeStore(tmp_path / "sessions" / "runtime.db"),
        exec_runtime=exec_runtime,
    )
    app.state.runtime_service = runtime_service
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "sessions_dir": str(tmp_path / "sessions"),
            "security": {
                "require_approval_for_exec": False,
                "command_rules": [_BACKGROUND_SMOKE_COMMAND_RULE],
            },
        }
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "controlled_subagent_execution",
                "title": "Controlled background execution reattach run",
                "topic": "reattach a detached smoke command",
                "selected_models_roles": {
                    "by_role": {
                        "planner": "controlled-planner-model",
                        "executor": "controlled-executor-model",
                        "controller": "controlled-controller-model",
                        "evaluator": "controlled-evaluator-model",
                        "judge": "controlled-judge-model",
                        "verifier": "controlled-verifier-model",
                    }
                },
                "summary": {
                    "protocol_config": {
                        "max_execution_requests": 1,
                        "default_timeout_sec": 30,
                        "background_allowed": True,
                    },
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        detached_artifact = next(
            artifact
            for artifact in final_payload["artifacts"]
            if artifact["artifact_type"] == "detached_exec_jobs"
        )
        session_id = detached_artifact["metadata"]["content"]["items"][0]["session_id"]

        reattach_response = client.post(
            f"/v1/agent-runs/{run_id}/reattach-exec/{session_id}",
            params={"yield_time_ms": 50},
        )
        assert reattach_response.status_code == 200
        reattach_payload = reattach_response.json()
        assert reattach_payload["run_id"] == run_id
        assert reattach_payload["session_id"] == session_id
        assert reattach_payload["associated"] is True
        assert reattach_payload["reattached"] is True
        assert reattach_payload["reattach_status"] in {"available", "unavailable"}

        updated = client.get(f"/v1/agent-runs/{run_id}")
        assert updated.status_code == 200
        reattach_events = [
            event for event in updated.json()["events"] if event["type"] == "detached_exec_reattached"
        ]
        assert reattach_events
        assert reattach_events[-1]["session_id"] == session_id


def test_agent_runs_api_flow_collects_evidence_queries_and_persists_artifacts(tmp_path: Path) -> None:
    app = create_app()
    engine = _AgentRunModelBackedEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Evidence-backed run",
                "topic": "deployment summary",
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                        "verifier": "verifier-model",
                        "judge": "judge-model",
                    }
                },
                "summary": {
                    "evidence_queries": ["approved deployment note"],
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        assert final_payload["summary"]["selected_candidate_id"] == "student"
        artifact_types = [artifact["artifact_type"] for artifact in final_payload["artifacts"]]
        assert "evidence_summary" in artifact_types
        assert "verification_summary" in artifact_types

    assert engine.evidence_calls
    assert engine.evidence_calls[0]["queries"] == ["approved deployment note"]


def test_agent_run_run_policy_pause_maps_to_awaiting_resources(tmp_path: Path) -> None:
    app = create_app()
    engine = _SlowAgentRunModelBackedEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Budget pause run",
                "topic": "deployment summary",
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                    }
                },
                "run_policy": {
                    "max_wall_clock_sec": 1,
                    "on_budget_exhausted": "pause",
                },
            },
        )
        assert create_response.status_code == 200
        created = create_response.json()
        run_id = created["run_id"]
        assert created["run_policy"]["max_wall_clock_sec"] == 1
        assert created["run_policy"]["on_budget_exhausted"] == "pause"

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(client, run_id, {"awaiting_resources"}, timeout_seconds=5.0)
        assert final_payload["status"] == "awaiting_resources"
        assert final_payload["finished_at"] is None
        assert final_payload["recovery_state"]["status"] == "awaiting_resources"
        assert final_payload["recovery_state"]["action"] == "pause"
        artifact_types = [artifact["artifact_type"] for artifact in final_payload["artifacts"]]
        assert "run_guard_policy" in artifact_types
        assert "recovery_checkpoint" in artifact_types
        assert "partial_summary" in artifact_types


def test_agent_run_heartbeat_timeout_maps_to_stalled(tmp_path: Path) -> None:
    app = create_app()
    engine = _SlowAgentRunModelBackedEngine(delay_seconds=1.2)
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Heartbeat stalled run",
                "topic": "deployment summary",
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                    }
                },
                "run_policy": {
                    "heartbeat_timeout_sec": 1,
                    "max_subagent_failures_per_role": 0,
                    "on_subagent_disconnect": "pause",
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(client, run_id, {"stalled"}, timeout_seconds=5.0)
        assert final_payload["status"] == "stalled"
        assert final_payload["finished_at"] is None
        assert final_payload["recovery_state"]["status"] == "stalled"
        assert final_payload["recovery_state"]["action"] == "pause"
        artifact_types = [artifact["artifact_type"] for artifact in final_payload["artifacts"]]
        assert "subagent_health_snapshot" in artifact_types


def test_agent_run_run_policy_finalize_partial_sets_partial_status(tmp_path: Path) -> None:
    app = create_app()
    engine = _SlowAgentRunModelBackedEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Partial finalize run",
                "topic": "deployment summary",
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                    }
                },
                "run_policy": {
                    "max_wall_clock_sec": 1,
                    "on_budget_exhausted": "finalize_partial",
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(client, run_id, {"partial"}, timeout_seconds=5.0)
        assert final_payload["status"] == "partial"
        assert final_payload["finished_at"] is not None
        assert final_payload["recovery_state"]["status"] == "partial"
        assert final_payload["recovery_state"]["action"] == "finalize_partial"
        artifact_types = [artifact["artifact_type"] for artifact in final_payload["artifacts"]]
        assert "partial_summary" in artifact_types


def test_agent_run_quota_exhaustion_maps_to_awaiting_resources(tmp_path: Path) -> None:
    app = create_app()
    engine = _QuotaExhaustedAgentRunEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Quota exhaustion run",
                "topic": "deployment summary",
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                    }
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(client, run_id, {"awaiting_resources"}, timeout_seconds=5.0)
        assert final_payload["status"] == "awaiting_resources"
        assert final_payload["recovery_state"]["status"] == "awaiting_resources"
        assert final_payload["recovery_state"]["resource_exhaustion_report"]["classification"] == "quota_exhausted"
        artifact_types = [artifact["artifact_type"] for artifact in final_payload["artifacts"]]
        assert "resource_exhaustion_report" in artifact_types


def test_agent_run_uses_security_default_run_policy_when_request_omits_it(tmp_path: Path) -> None:
    app = create_app()
    engine = _SlowAgentRunModelBackedEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "sessions_dir": str(tmp_path / "sessions"),
            "security": {
                "agent_run_default_max_wall_clock_sec": 1,
                "agent_run_default_heartbeat_timeout_sec": 45,
                "agent_run_default_checkpoint_interval_steps": 3,
                "agent_run_default_max_subagent_failures_per_role": 4,
                "agent_run_default_on_budget_exhausted": "pause",
                "agent_run_default_on_subagent_disconnect": "pause",
            },
        }
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Default run policy from settings",
                "topic": "verify settings-backed run guard",
            },
        )
        assert create_response.status_code == 200
        created = create_response.json()
        run_id = created["run_id"]
        assert created["run_policy"]["max_wall_clock_sec"] == 1
        assert created["run_policy"]["heartbeat_timeout_sec"] == 45
        assert created["run_policy"]["checkpoint_interval_steps"] == 3
        assert created["run_policy"]["max_subagent_failures_per_role"] == 4
        assert created["run_policy"]["on_budget_exhausted"] == "pause"
        assert created["run_policy"]["on_subagent_disconnect"] == "pause"

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(
            client,
            run_id,
            {"awaiting_resources", "partial", "succeeded"},
            timeout_seconds=5.0,
        )
        assert final_payload["run_policy"]["max_wall_clock_sec"] == 1
        assert final_payload["run_policy"]["on_budget_exhausted"] == "pause"


def test_agent_run_local_model_unavailable_maps_to_awaiting_resources(tmp_path: Path) -> None:
    app = create_app()
    engine = _LocalModelUnavailableAgentRunEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Local runtime unavailable run",
                "topic": "deployment summary",
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                    }
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(client, run_id, {"awaiting_resources"}, timeout_seconds=5.0)
        assert final_payload["status"] == "awaiting_resources"
        assert final_payload["recovery_state"]["resource_exhaustion_report"]["classification"] == "local_model_unavailable"
        assert final_payload["latest_error"] == "Configured model 'teacher-model' is not available."


def test_agent_runs_api_flow_persists_research_debate_artifacts(tmp_path: Path) -> None:
    app = create_app()
    engine = _AgentRunModelBackedEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "multi_agent_debate",
                "title": "Research debate run",
                "topic": "deployment review",
                "selected_models_roles": {
                    "by_role": {
                        "debater_a": "debater-a-model",
                        "debater_b": "debater-b-model",
                        "judge": "judge-model",
                        "verifier": "verifier-model",
                        "planner": "planner-model",
                        "local_worker": "local-worker-model",
                        "synthesizer": "synth-model",
                    }
                },
                "summary": {
                    "evidence_queries": ["approved deployment note"],
                    "protocol_config": {"rounds": 2},
                },
                "evaluation_policy": {
                    "research": {
                        "enabled": True,
                        "preset": "smart_judge_research_debate",
                        "output_targets": ["research_brief"],
                        "source_mode": "hybrid",
                        "citation_policy": "claim_level_required",
                        "local_worker_count": 2,
                        "local_worker_count_max": 6,
                        "max_research_queries": 4,
                        "max_sources_per_query": 3,
                        "debate_rounds": 2,
                    }
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        artifact_types = [artifact["artifact_type"] for artifact in final_payload["artifacts"]]
        assert "research_plan" in artifact_types
        assert "source_quality_table" in artifact_types
        assert "claim_evidence_map" in artifact_types
        assert "research_brief" in artifact_types
        assert "dataset_record" not in artifact_types


def test_agent_runs_api_flow_keeps_multi_agent_debate_protocol_with_controlled_execution_capability(
    tmp_path: Path,
) -> None:
    app = create_app()
    engine = _AgentRunModelBackedEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "sessions_dir": str(tmp_path / "sessions"),
            "security": {
                "require_approval_for_exec": False,
                "command_rules": [_CONTROLLED_SMOKE_COMMAND_RULE],
            },
        }
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "multi_agent_debate",
                "title": "Debate with controlled execution capability",
                "topic": "deployment review",
                "selected_models_roles": {
                    "by_role": {
                        "debater_a": "debater-a-model",
                        "debater_b": "debater-b-model",
                        "judge": "judge-model",
                        "verifier": "verifier-model",
                        "planner": "controlled-planner-model",
                        "executor": "controlled-executor-model",
                        "controller": "controlled-controller-model",
                        "evaluator": "controlled-evaluator-model",
                    }
                },
                "summary": {
                    "protocol_config": {"rounds": 2},
                    "execution_policy": {
                        "mode": "controlled",
                        "max_execution_requests": 1,
                        "max_commands_per_request": 1,
                        "default_timeout_sec": 30,
                        "background_allowed": False,
                    },
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        assert final_payload["protocol_id"] == "multi_agent_debate"
        artifact_types = [artifact["artifact_type"] for artifact in final_payload["artifacts"]]
        assert "controlled_execution_runtime" in artifact_types
        assert "execution_requests" in artifact_types
        runtime_artifact = next(
            artifact
            for artifact in final_payload["artifacts"]
            if artifact["artifact_type"] == "controlled_execution_runtime"
        )
        runtime_payload = runtime_artifact["metadata"]["content"]
        assert runtime_payload["primary_workflow"] is False
        assert (
            runtime_payload["execution_boundary"]
            == "subagents_propose_controller_approves_shared_runtime_executes"
        )


def test_agent_run_schedule_starts_due_run_automatically(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        RuntimeService,
        "_DEFAULT_SCHEDULER_POLL_INTERVAL_SECONDS",
        0.05,
    )
    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Scheduled run",
                "topic": "auto execute",
                "schedule": {
                    "enabled": True,
                    "interval_seconds": 3600,
                    "start_immediately": True,
                },
            },
        )
        assert create_response.status_code == 200
        created = create_response.json()
        run_id = created["run_id"]
        assert created["status"] == "created"

        final_payload = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        assert final_payload["summary"]["final_answer"]
        assert final_payload["schedule"]["last_scheduled_at"] is not None
        assert final_payload["schedule"]["last_completion_status"] == "succeeded"
        assert final_payload["schedule"]["next_run_at"] is not None
        assert final_payload["schedule"]["attempt_count"] == 1
        assert isinstance(final_payload["schedule"].get("recent_attempts"), list)
        latest_attempt = final_payload["schedule"]["recent_attempts"][0]
        assert latest_attempt["attempt_id"]
        assert latest_attempt["status"] == "succeeded"
        assert latest_attempt["selected_candidate_id"] == final_payload["summary"].get(
            "selected_candidate_id"
        )
        assert latest_attempt["completed_at"] == final_payload["schedule"]["last_completed_at"]
        assert latest_attempt["final_answer_preview"]
        artifact_attempt_ids = {
            artifact.get("metadata", {}).get("attempt_id")
            for artifact in final_payload["artifacts"]
            if isinstance(artifact.get("metadata"), dict)
        }
        assert latest_attempt["attempt_id"] in artifact_attempt_ids
        scheduled_events = [
            event
            for event in final_payload["events"]
            if event.get("type") in {"run_scheduled", "run_started"}
        ]
        assert scheduled_events
        assert all(event.get("attempt_id") == latest_attempt["attempt_id"] for event in scheduled_events)
        event_types = [event["type"] for event in final_payload["events"]]
        assert "run_scheduled" in event_types
        assert "run_started" in event_types


def test_agent_run_schedule_disabled_does_not_start(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        RuntimeService,
        "_DEFAULT_SCHEDULER_POLL_INTERVAL_SECONDS",
        0.05,
    )
    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    due_at = (datetime.now(UTC) - timedelta(minutes=1)).isoformat()
    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Disabled scheduled run",
                "topic": "stay idle",
                "schedule": {
                    "enabled": False,
                    "run_at": due_at,
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        time.sleep(0.3)
        get_response = client.get(f"/v1/agent-runs/{run_id}")
        assert get_response.status_code == 200
        payload = get_response.json()
        assert payload["status"] == "created"
        assert payload["schedule"].get("last_scheduled_at") is None
        assert "run_scheduled" not in [event["type"] for event in payload["events"]]


def test_agent_run_schedule_respects_max_runs(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        RuntimeService,
        "_DEFAULT_SCHEDULER_POLL_INTERVAL_SECONDS",
        0.05,
    )
    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Max runs policy",
                "topic": "stop after one run",
                "schedule": {
                    "enabled": True,
                    "interval_seconds": 1,
                    "start_immediately": True,
                    "max_runs": 1,
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        final_payload = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        assert final_payload["schedule"]["attempt_count"] == 1
        assert final_payload["schedule"]["enabled"] is False
        assert final_payload["schedule"]["next_run_at"] is None
        assert final_payload["schedule"]["schedule_status"] == "max_runs_reached"
        assert final_payload["schedule"]["health_status"] == "completed"

        time.sleep(0.3)
        follow_up = client.get(f"/v1/agent-runs/{run_id}")
        assert follow_up.status_code == 200
        follow_up_payload = follow_up.json()
        assert follow_up_payload["schedule"]["attempt_count"] == 1


def test_agent_run_schedule_auto_pauses_after_failure(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        RuntimeService,
        "_DEFAULT_SCHEDULER_POLL_INTERVAL_SECONDS",
        0.05,
    )
    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    async def _failing_run(self: Any, request: Any) -> MultiAgentRunResult:
        raise RuntimeError("synthetic scheduled failure")

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _failing_run)

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Auto pause on failure",
                "topic": "synthetic failure",
                "schedule": {
                    "enabled": True,
                    "interval_seconds": 60,
                    "start_immediately": True,
                    "auto_pause_on_failure": True,
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        final_payload = _wait_agent_run_until(client, run_id, {"failed"}, timeout_seconds=4.0)
        assert final_payload["schedule"]["attempt_count"] == 1
        assert final_payload["schedule"]["failure_streak"] == 1
        assert final_payload["schedule"]["enabled"] is False
        assert final_payload["schedule"]["next_run_at"] is None
        assert final_payload["schedule"]["schedule_status"] == "paused_after_failure"
        assert final_payload["schedule"]["health_status"] == "paused_after_failure"
        assert final_payload["latest_error"] == "synthetic scheduled failure"


def test_agent_run_operator_messages_are_collected_as_guidance_for_runtime(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured_request: dict[str, Any] = {}

    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        captured_request["guidance_messages"] = list(request.guidance_messages)
        captured_request["metadata"] = dict(request.metadata)
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Guided result."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Workflow-guided run",
                "topic": "deployment summary",
                "project_id": "project-guidance",
                "workspace_dir": str(tmp_path / "workspace-guidance"),
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        message_response = client.post(
            f"/v1/agent-runs/{run_id}/messages",
            json={
                "role": "operator",
                "content": "Prioritize a concise answer with verified claims only.",
            },
        )
        assert message_response.status_code == 200

        guidance_response = client.post(
            f"/v1/agent-runs/{run_id}/guidance",
            json={"guidance": "Mention any uncertainty explicitly."},
        )
        assert guidance_response.status_code == 200

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        assert final_payload["summary"]["final_answer"] == "Guided result."

    assert captured_request["guidance_messages"] == [
        "Prioritize a concise answer with verified claims only.",
        "Mention any uncertainty explicitly.",
    ]
    assert captured_request["metadata"]["project_id"] == "project-guidance"
    assert captured_request["metadata"]["workspace_dir"] == str(tmp_path / "workspace-guidance")


def test_agent_run_subagent_messages_do_not_change_global_guidance_collection(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    captured_request: dict[str, Any] = {}

    async def _fake_run(self: Any, request: Any) -> MultiAgentRunResult:
        del self
        captured_request["guidance_messages"] = list(request.guidance_messages)
        captured_request["role_guidance_messages"] = dict(request.role_guidance_messages)
        return MultiAgentRunResult(
            run_id=request.run_id,
            protocol="teacher_student_distill",
            state="succeeded",
            task_input=request.task_input,
            candidates=[],
            selected_candidate_id=None,
            evaluation={},
            artifacts={"final_answer": "Subagent guidance stays additive."},
            events=[],
            metadata={},
        )

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _fake_run)

    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Subagent-guided run",
                "topic": "deployment summary",
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        subagent_message_response = client.post(
            f"/v1/agent-runs/{run_id}/subagents/verifier/messages",
            json={
                "content": "Please inspect only claim 2 evidence.",
                "metadata": {"channel": "subagent-chat"},
            },
        )
        assert subagent_message_response.status_code == 200
        assert [
            event
            for event in subagent_message_response.json()["events"]
            if event["type"] == "subagent_message"
        ][-1]["target_role_id"] == "verifier"

        guidance_response = client.post(
            f"/v1/agent-runs/{run_id}/guidance",
            json={"guidance": "Keep the final answer concise."},
        )
        assert guidance_response.status_code == 200

        start_response = client.post(f"/v1/agent-runs/{run_id}/start")
        assert start_response.status_code == 200

        final_payload = _wait_agent_run_until(client, run_id, {"succeeded"}, timeout_seconds=4.0)
        assert final_payload["summary"]["final_answer"] == "Subagent guidance stays additive."

    assert captured_request["guidance_messages"] == [
        "Keep the final answer concise.",
    ]
    assert captured_request["role_guidance_messages"] == {
        "verifier": ["Please inspect only claim 2 evidence."],
    }


def test_agent_run_operator_message_preserves_structured_attachments(
    tmp_path: Path,
) -> None:
    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Attachment replay run",
                "topic": "structured attachments",
                "project_id": "project-attachments",
                "workspace_dir": str(tmp_path / "workspace-attachments"),
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        attachment = {
            "name": "app.py",
            "path": str(tmp_path / "workspace-attachments" / "app.py"),
            "source": "workspace_selection",
            "line_start": 10,
            "line_end": 14,
            "quote": "print('hello')",
            "note": "Investigate this block.",
        }
        message_response = client.post(
            f"/v1/agent-runs/{run_id}/messages",
            json={
                "role": "operator",
                "content": "Please inspect this selection.",
                "attachments": [attachment],
            },
        )
        assert message_response.status_code == 200
        payload = message_response.json()

    operator_events = [event for event in payload["events"] if event["type"] == "operator_message"]
    assert operator_events
    expected_attachment = {
        **attachment,
        "size": None,
        "content_type": None,
    }
    assert operator_events[-1]["attachments"] == [expected_attachment]
    assert operator_events[-1]["metadata"]["attachments"] == [expected_attachment]


def test_agent_run_operator_message_rejects_invalid_attachment_ranges(
    tmp_path: Path,
) -> None:
    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Invalid attachment run",
                "topic": "structured attachments",
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        response = client.post(
            f"/v1/agent-runs/{run_id}/messages",
            json={
                "role": "operator",
                "content": "Please inspect this selection.",
                "attachments": [
                    {
                        "name": "app.py",
                        "path": str(tmp_path / "app.py"),
                        "source": "workspace_selection",
                        "line_start": 12,
                        "line_end": 4,
                    }
                ],
            },
        )

    assert response.status_code == 422


def test_agent_run_operator_message_allows_attachment_only_payloads(
    tmp_path: Path,
) -> None:
    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Attachment only run",
                "topic": "attachment only workflow message",
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        response = client.post(
            f"/v1/agent-runs/{run_id}/messages",
            json={
                "role": "operator",
                "content": "",
                "attachments": [
                    {
                        "name": "notes.md",
                        "path": str(tmp_path / "notes.md"),
                        "source": "workspace_file",
                        "note": "Use this document as context.",
                    }
                ],
            },
        )

    assert response.status_code == 200
    payload = response.json()
    operator_event = [event for event in payload["events"] if event["type"] == "operator_message"][-1]
    assistant_event = [event for event in payload["events"] if event["type"] == "assistant_message"][-1]
    assert operator_event["content"] == ""
    assert operator_event["attachments"] == [
        {
            "name": "notes.md",
            "path": str(tmp_path / "notes.md"),
            "size": None,
            "content_type": None,
            "source": "workspace_file",
            "line_start": None,
            "line_end": None,
            "quote": None,
            "note": "Use this document as context.",
        }
    ]
    assert "message received" in assistant_event["content"]


def test_agent_run_schedule_failure_streak_persists_when_not_auto_paused(
    tmp_path: Path,
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        RuntimeService,
        "_DEFAULT_SCHEDULER_POLL_INTERVAL_SECONDS",
        0.05,
    )
    app = create_app()
    app.state.engine_factory = lambda: _RuntimeFakeEngine()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    async def _failing_run(self: Any, request: Any) -> MultiAgentRunResult:
        raise RuntimeError("non-pausing failure")

    monkeypatch.setattr("mochi.runtime.service.MultiAgentOrchestrator.run", _failing_run)

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/agent-runs",
            json={
                "protocol_id": "teacher_student_distill",
                "title": "Failure streak",
                "topic": "keep schedule active",
                "schedule": {
                    "enabled": True,
                    "interval_seconds": 60,
                    "start_immediately": True,
                    "auto_pause_on_failure": False,
                },
            },
        )
        assert create_response.status_code == 200
        run_id = create_response.json()["run_id"]

        final_payload = _wait_agent_run_until(client, run_id, {"failed"}, timeout_seconds=4.0)
        assert final_payload["schedule"]["attempt_count"] == 1
        assert final_payload["schedule"]["failure_streak"] == 1
        assert final_payload["schedule"]["enabled"] is True
        assert final_payload["schedule"]["next_run_at"] is not None
        assert final_payload["schedule"]["schedule_status"] == "scheduled"
        assert final_payload["schedule"]["health_status"] == "failing"


def test_task_resume_endpoint_applies_approval_override(tmp_path: Path) -> None:
    app = create_app()
    engine = _RuntimeFakeEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/tasks",
            json={
                "input_message": "run with resume endpoint",
                "workspace_dir": str(tmp_path / "workspace"),
            },
        )
        assert create_response.status_code == 200
        task_id = create_response.json()["task_id"]

        waiting_payload = _wait_until(client, task_id, {"awaiting_approval"})
        assert waiting_payload["pending_approval"] is not None

        resume_response = client.post(
            f"/v1/tasks/{task_id}/resume",
            json={"decision": "approve_once", "reason": "manual resume"},
        )
        assert resume_response.status_code == 200
        resumed = resume_response.json()
        assert resumed["status"] in {"resumed", "running"}

        done_payload = _wait_until(client, task_id, {"succeeded"})
        assert done_payload["status"] == "succeeded"
        assert done_payload["final_answer"] == "done"

    assert len(engine.permission_policy_calls) == 2
    assert engine.permission_policy_calls[1] == {
        "autonomy_mode": "trusted_workspace",
        "require_approval_for_file_write": False,
        "require_approval_for_exec": True,
        "file_ops_scope": "workspace",
        "approved_tool_calls": [
            {
                "tool_name": "exec_command",
                "arguments": {"command": "dir", "shell": "cmd"},
            }
        ],
    }
