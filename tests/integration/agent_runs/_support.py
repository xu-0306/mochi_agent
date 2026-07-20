from __future__ import annotations

import asyncio

from mochi.agents.events import FinalAnswerEvent, ToolCallRequestEvent, ToolCallResultEvent
from mochi.agents.invocation import (
    AgentInvocationDiagnostics,
    AgentInvocationRequest,
    AgentInvocationResult,
)
from mochi.backends.types import GenerationResult, Message


class _ConfiguredModelEngineStub:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []
        self.invoke_calls: list[dict[str, object]] = []
        self.evidence_calls: list[dict[str, object]] = []

    async def invoke(self, request: AgentInvocationRequest) -> AgentInvocationResult:
        self.invoke_calls.append(
            {
                "message": request.message,
                "session_id": request.session_id,
                "inference_overrides": dict(request.inference_overrides or {}),
                "tool_mode": request.tool_mode,
                "permission_policy": (
                    dict(request.permission_policy)
                    if isinstance(request.permission_policy, dict)
                    else None
                ),
                "tool_allowlist": (
                    list(request.tool_allowlist)
                    if isinstance(request.tool_allowlist, list)
                    else None
                ),
                "tool_denylist": (
                    list(request.tool_denylist)
                    if isinstance(request.tool_denylist, list)
                    else None
                ),
                "execution_profile": request.execution_profile,
                "system_prompt_addendum": request.system_prompt_addendum,
                "persist_session": request.persist_session,
            }
        )
        model_id = str((request.inference_overrides or {}).get("model") or "")
        if model_id == "teacher-model":
            content = "Teacher evidence-backed draft."
        elif model_id == "student-model":
            content = "Student concise final answer."
        elif model_id == "proposer-model":
            content = (
                '{"tasks":['
                '{"task_id":"task-1","question":"Find the verified deployment answer.","difficulty":"medium","rationale":"Needs evidence"},'
                '{"task_id":"task-2","question":"Identify the unsupported deployment claim.","difficulty":"hard","rationale":"Needs contrast"}'
                ']}'
            )
        elif model_id == "solver-model":
            content = "Solver verified answer with source-aware rationale."
        elif model_id == "debater-a-model":
            content = "Argument A prefers the first route."
        elif model_id == "debater-b-model":
            content = "Argument B rejects weak assumptions."
        elif model_id == "judge-model":
            selected_candidate_id = "student"
            if "candidate_id=debater_b" in request.message:
                selected_candidate_id = "debater_b"
            if "candidate_id=solver" in request.message:
                selected_candidate_id = "solver_1_1"
                content = (
                    '{"selected_candidate_id":"solver_1_1",'
                    '"scores":[{"candidate_id":"solver_1_1","score":0.93,"rationale":"best verified solver rollout","evidence_gate":{"status":"skipped"}}]}'
                )
                events = [FinalAnswerEvent(content=content)]
                return AgentInvocationResult(
                    content=content,
                    events=events,
                    diagnostics=AgentInvocationDiagnostics(
                        execution_profile=request.execution_profile,
                        tool_mode=request.tool_mode,
                        exposed_tools=["file_read"],
                        matched_tool_groups=["workspace"],
                    ),
                )
            content = (
                f'{{"selected_candidate_id":"{selected_candidate_id}",'
                '"scores":['
                '{"candidate_id":"teacher","score":0.61,"rationale":"good coverage","evidence_gate":{"status":"skipped"}},'
                '{"candidate_id":"student","score":0.94,"rationale":"clearer final answer","evidence_gate":{"status":"skipped"}}'
                ']}'
            )
        elif model_id == "planner-model":
            content = (
                '{"subquestions":["What does the primary source say?","What contradicts the current answer?"],'
                '"evidence_requirements":["Use attributable sources","Flag unsupported claims"],'
                '"exclusion_rules":["No unsupported speculation"],'
                '"evidence_queries":["approved deployment note","deployment contradiction"]}'
            )
        elif model_id == "local-worker-model":
            content = "Local worker note: prioritize attributable evidence and extract only verified dataset candidates."
        elif model_id == "synth-model":
            content = (
                "# Research Brief\n\n## Summary\nEvidence-backed deployment summary.\n\n"
                "## Findings\nArgument B remains stronger after evidence review.\n\n"
                "## Evidence Quality\nHigh-confidence fetched source available.\n\n"
                "## Claim Status\nOne supported claim.\n\n## Open Gaps\nNeed broader source diversity."
            )
        elif model_id == "verifier-model":
            if "candidate_id=solver" in request.message:
                content = (
                    '{"candidate_verifications":['
                    '{"candidate_id":"solver_1_1","status":"verified","rationale":"solver answer matches source","citations":[{"evidence_id":"src-1","summary":"Evidence supports solver"}],"issues":[]},'
                    '{"candidate_id":"solver_2_1","status":"failed","rationale":"unsupported contrast","citations":[],"issues":["unsupported claim"]}'
                    ']}'
                )
            elif "candidate_id=teacher" in request.message:
                content = (
                    '{"candidate_verifications":['
                    '{"candidate_id":"teacher","status":"failed","rationale":"conflicts with evidence","citations":[{"evidence_id":"src-1","summary":"Evidence contradicts teacher claim"}],"issues":["unsupported claim"]},'
                    '{"candidate_id":"student","status":"verified","rationale":"matches evidence","citations":[{"evidence_id":"src-1","summary":"Evidence supports student summary"}],"issues":[]}'
                    ']}'
                )
            else:
                content = '{"candidate_verifications":[]}'
        elif model_id == "controlled-planner-model":
            content = "Plan: run one safe command and inspect the output."
        elif model_id == "controlled-executor-model":
            content = (
                '{"execution_requests":[{"request_id":"req-1","command":"echo controlled-ok",'
                '"shell":"powershell","timeout":30,"rationale":"smoke test",'
                '"expected_artifacts":["stdout"],"success_metric":"stdout contains controlled-ok"}]}'
            )
        elif model_id == "controlled-controller-model":
            content = (
                '{"status":"approved","reason":"bounded smoke command",'
                '"command":"echo controlled-ok","shell":"powershell","timeout":30}'
            )
        elif model_id == "controlled-reject-controller-model":
            content = '{"status":"rejected","reason":"not necessary"}'
        elif model_id == "controlled-evaluator-model":
            content = "Execution finished; stdout should contain controlled-ok."
        elif model_id == "controlled-judge-model":
            content = (
                '{"selected_candidate_id":"evaluator","scores":['
                '{"candidate_id":"planner","score":0.7,"rationale":"planned","evidence_gate":{"status":"skipped"}},'
                '{"candidate_id":"executor","score":0.7,"rationale":"requested","evidence_gate":{"status":"skipped"}},'
                '{"candidate_id":"evaluator","score":0.95,"rationale":"summarized","evidence_gate":{"status":"skipped"}}]}'
            )
        elif model_id == "controlled-verifier-model":
            content = (
                '{"candidate_verifications":['
                '{"candidate_id":"planner","status":"verified","rationale":"ok","citations":[],"issues":[]},'
                '{"candidate_id":"executor","status":"verified","rationale":"ok","citations":[],"issues":[]},'
                '{"candidate_id":"evaluator","status":"verified","rationale":"ok","citations":[],"issues":[]}'
                ']}'
            )
        else:
            raise AssertionError(f"Unexpected invoke model_id: {model_id}")
        events = [FinalAnswerEvent(content=content)]
        if model_id in {"teacher-model", "debater-a-model"}:
            events.insert(
                0,
                ToolCallRequestEvent(
                    call_id=f"{model_id}-read-1",
                    tool_name="file_read",
                    arguments={"path": "notes.md"},
                ),
            )
            events.insert(
                1,
                ToolCallResultEvent(
                    call_id=f"{model_id}-read-1",
                    tool_name="file_read",
                    result="notes",
                    metadata={"status": "ok"},
                ),
            )
        return AgentInvocationResult(
            content=content,
            events=events,
            diagnostics=AgentInvocationDiagnostics(
                execution_profile=request.execution_profile,
                tool_mode=request.tool_mode,
                exposed_tools=["file_read"],
                matched_tool_groups=["workspace"],
            ),
        )

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
                    '{"tasks":['
                    '{"task_id":"task-1","question":"Find the verified deployment answer.","difficulty":"medium","rationale":"Needs evidence"},'
                    '{"task_id":"task-2","question":"Identify the unsupported deployment claim.","difficulty":"hard","rationale":"Needs contrast"}'
                    ']}'
                ),
                model=model_id,
            )
        if model_id == "solver-model":
            return GenerationResult(
                content="Solver verified answer with source-aware rationale.",
                model=model_id,
            )
        if model_id == "judge-model":
            selected_candidate_id = "student"
            if any(
                isinstance(message, Message) and "candidate_id=debater_b" in message.content
                for message in messages
            ):
                selected_candidate_id = "debater_b"
            if any(
                isinstance(message, Message) and "candidate_id=solver" in message.content
                for message in messages
            ):
                return GenerationResult(
                    content=(
                        '{"selected_candidate_id":"solver_1_1",'
                        '"scores":[{"candidate_id":"solver_1_1","score":0.93,"rationale":"best verified solver rollout","evidence_gate":{"status":"skipped"}}]}'
                    ),
                    model=model_id,
                )
            return GenerationResult(
                content=(
                    f'{{"selected_candidate_id":"{selected_candidate_id}",'
                    '"scores":['
                    '{"candidate_id":"teacher","score":0.61,"rationale":"good coverage","evidence_gate":{"status":"skipped"}},'
                    '{"candidate_id":"student","score":0.94,"rationale":"clearer final answer","evidence_gate":{"status":"skipped"}}'
                    ']}'
                ),
                model=model_id,
            )
        if model_id == "planner-model":
            return GenerationResult(
                content=(
                    '{"subquestions":["What does the primary source say?","What contradicts the current answer?"],'
                    '"evidence_requirements":["Use attributable sources","Flag unsupported claims"],'
                    '"exclusion_rules":["No unsupported speculation"],'
                    '"evidence_queries":["approved deployment note","deployment contradiction"]}'
                ),
                model=model_id,
            )
        if model_id == "debater-a-model":
            return GenerationResult(content="Argument A prefers the first route.", model=model_id)
        if model_id == "debater-b-model":
            return GenerationResult(content="Argument B rejects weak assumptions.", model=model_id)
        if model_id == "local-worker-model":
            return GenerationResult(
                content="Local worker note: prioritize attributable evidence and extract only verified dataset candidates.",
                model=model_id,
            )
        if model_id == "synth-model":
            return GenerationResult(
                content=(
                    "# Research Brief\n\n## Summary\nEvidence-backed deployment summary.\n\n"
                    "## Findings\nArgument B remains stronger after evidence review.\n\n"
                    "## Evidence Quality\nHigh-confidence fetched source available.\n\n"
                    "## Claim Status\nOne supported claim.\n\n## Open Gaps\nNeed broader source diversity."
                ),
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
                        '{"candidate_id":"solver_1_1","status":"verified","rationale":"solver answer matches source","citations":[{"evidence_id":"src-1","summary":"Evidence supports solver"}],"issues":[]},'
                        '{"candidate_id":"solver_2_1","status":"failed","rationale":"unsupported contrast","citations":[],"issues":["unsupported claim"]}'
                        ']}'
                    ),
                    model=model_id,
                )
            if any(
                isinstance(message, Message) and "candidate_id=teacher" in message.content
                for message in messages
            ):
                return GenerationResult(
                    content=(
                        '{"candidate_verifications":['
                        '{"candidate_id":"teacher","status":"failed","rationale":"conflicts with evidence","citations":[{"evidence_id":"src-1","summary":"Evidence contradicts teacher claim"}],"issues":["unsupported claim"]},'
                        '{"candidate_id":"student","status":"verified","rationale":"matches evidence","citations":[{"evidence_id":"src-1","summary":"Evidence supports student summary"}],"issues":[]}'
                        ']}'
                    ),
                    model=model_id,
                )
            return GenerationResult(
                content='{"candidate_verifications":[]}',
                model=model_id,
            )
        raise AssertionError(f"Unexpected model_id: {model_id}")

    async def collect_agent_run_evidence(
        self,
        *,
        queries: list[str],
        metadata: dict[str, object] | None = None,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        self.evidence_calls.append(
            {
                "queries": list(queries),
                "metadata": dict(metadata or {}),
            }
        )
        return (
            [
                {
                    "evidence_id": "collected-1",
                    "title": "Collected deployment note",
                    "content": "Only the student summary matches the collected deployment note.",
                    "url": "https://example.com/deployment-note",
                    "source_type": "web_fetch",
                    "query": queries[0] if queries else "",
                }
            ],
            {
                "query_count": len(queries),
                "collected_packet_count": 1,
                "provider_counts": {"stub-search": 1},
                "queries": [
                    {
                        "query": queries[0] if queries else "",
                        "packet_count": 1,
                    }
                ],
            },
        )


class _GenerateOnlyEngineStub(_ConfiguredModelEngineStub):
    invoke = None  # type: ignore[assignment]


class _DebaterBFailureEngine(_ConfiguredModelEngineStub):
    async def invoke(self, request: AgentInvocationRequest) -> AgentInvocationResult:
        model_id = str((request.inference_overrides or {}).get("model") or "")
        if model_id == "debater-b-model":
            raise RuntimeError("debater-b disconnected")
        return await super().invoke(request)


class _TeacherHeartbeatTimeoutEngine(_ConfiguredModelEngineStub):
    async def invoke(self, request: AgentInvocationRequest) -> AgentInvocationResult:
        model_id = str((request.inference_overrides or {}).get("model") or "")
        if model_id == "teacher-model":
            await asyncio.sleep(1.2)
        return await super().invoke(request)


class _CollectorToolEngine(_ConfiguredModelEngineStub):
    async def invoke(self, request: AgentInvocationRequest) -> AgentInvocationResult:
        model_id = str((request.inference_overrides or {}).get("model") or "")
        if model_id != "teacher-model":
            return await super().invoke(request)
        collector_payload = {
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
                        "status": "running",
                        "source": {
                            "url": "https://forum.example/t/api-examples/274354",
                            "id": "topic:274354",
                        },
                        "tool": {
                            "name": "discourse_topic_collect",
                            "arguments": {"base_url": "https://forum.example", "topic_id": "274354"},
                        },
                        "progress": {
                            "cursor": "101",
                            "items_collected": 2,
                            "items_emitted": 2,
                            "remaining_item_count": 1,
                        },
                    }
                ]
            },
        }
        return AgentInvocationResult(
            content="Teacher collected topic evidence.",
            events=[
                ToolCallRequestEvent(
                    call_id="collector-call-1",
                    tool_name="discourse_topic_collect",
                    arguments={"base_url": "https://forum.example", "topic_id": "274354"},
                ),
                ToolCallResultEvent(
                    call_id="collector-call-1",
                    tool_name="discourse_topic_collect",
                    result=collector_payload,
                    metadata={"status": "ok"},
                ),
                FinalAnswerEvent(content="Teacher collected topic evidence."),
            ],
            diagnostics=AgentInvocationDiagnostics(
                execution_profile=request.execution_profile,
                tool_mode=request.tool_mode,
                exposed_tools=["discourse_topic_collect"],
                matched_tool_groups=["collector"],
            ),
        )
