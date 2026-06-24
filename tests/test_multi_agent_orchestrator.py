from __future__ import annotations

import asyncio

import pytest

from mochi.agents.events import FinalAnswerEvent, ToolCallRequestEvent, ToolCallResultEvent
from mochi.agents.invocation import (
    AgentInvocationDiagnostics,
    AgentInvocationRequest,
    AgentInvocationResult,
)
from mochi.backends.types import GenerationResult, Message
from mochi.agents.multi_agent.orchestrator import (
    MultiAgentOrchestrator,
    MultiAgentRunRequest,
    RunPolicyStop,
)
from mochi.runtime.approvals import InMemoryApprovalStore


@pytest.mark.asyncio
async def test_orchestrator_smoke_emits_state_and_guidance_events() -> None:
    orchestrator = MultiAgentOrchestrator()
    request = MultiAgentRunRequest(
        task_input="Summarize this note",
        protocol={"protocol": "teacher_student_distill"},
        guidance_messages=["Prefer concise answers."],
    )

    result = await orchestrator.run(request)
    event_types = [event.type for event in result.events]

    assert result.state == "succeeded"
    assert event_types[0] == "state_changed"
    assert "guidance" in event_types
    assert "role_output" in event_types
    assert "evaluation" in event_types
    assert event_types[-1] == "state_changed"


@pytest.mark.asyncio
async def test_orchestrator_finalize_partial_when_goal_checkpoint_step_cadence_reached() -> None:
    engine = _ConfiguredModelEngineStub()

    result = await MultiAgentOrchestrator(engine=engine).run(
        MultiAgentRunRequest(
            task_input="Summarize this deployment note.",
            protocol={"protocol": "teacher_student_distill"},
            run_policy={
                "checkpoint_interval_steps": 2,
                "goal_checkpoint_cadence_owned": True,
                "goal_checkpoint_cadence_effective_steps": 2,
                "on_budget_exhausted": "finalize_partial",
            },
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                        "judge": "judge-model",
                    }
                }
            },
        )
    )

    assert result.state == "partial"
    recovery_state = dict(result.metadata["recovery_state"])
    assert recovery_state["status"] == "partial"
    assert recovery_state["action"] == "finalize_partial"
    assert recovery_state["stage"] == "protocol_completed"
    assert recovery_state["checkpoint"]["checkpoint_index"] == 2
    assert recovery_state["reason"] == "Run reached checkpoint_interval_steps=2 at checkpoint_index=2."
    assert [call["inference_overrides"]["model"] for call in engine.invoke_calls] == [
        "teacher-model",
        "student-model",
    ]


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


@pytest.mark.asyncio
async def test_orchestrator_uses_configured_models_for_teacher_student_protocol() -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(engine=engine)

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Summarize the deployment risks.",
            protocol={"protocol": "teacher_student_distill"},
            guidance_messages=["Ground every claim."],
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                        "judge": "judge-model",
                    },
                    "subagents": [
                        {"role": "teacher", "model_id": "teacher-model"},
                        {"role": "student", "model_id": "student-model"},
                        {"role": "judge", "model_id": "judge-model"},
                    ],
                }
            },
        )
    )

    assert result.selected_candidate_id == "student"
    assert result.artifacts["final_answer"] == "Student concise final answer."
    assert [call["inference_overrides"]["model"] for call in engine.invoke_calls] == [
        "teacher-model",
        "student-model",
        "judge-model",
    ]
    assert engine.calls == []
    assert engine.invoke_calls[0]["execution_profile"] == "subagent_readonly"
    assert engine.invoke_calls[0]["tool_mode"] == "auto"
    assert "Role identity: Teacher" in str(engine.invoke_calls[0]["system_prompt_addendum"])
    assert "Teacher evidence-backed draft." in str(engine.invoke_calls[1]["message"])
    teacher_event_types = [
        event.type
        for event in result.events
        if event.payload.get("role_id") == "teacher"
    ]
    assert "role_started" in teacher_event_types
    assert "role_completed" in teacher_event_types
    assert "role_output" in teacher_event_types
    diagnostics = result.candidates[0].metadata["diagnostics"]
    assert diagnostics["execution_profile"] == "subagent_readonly"
    assert diagnostics["tool_mode"] == "auto"
    assert diagnostics["exposed_tools"] == ["file_read"]
    runtime = result.artifacts["subagent_runtime"]
    assert runtime["execution_boundary"] == "multi_agent_subagents_research_only_main_agent_executes_code_and_training"
    assert runtime["tool_event_count"] == 2
    assert runtime["approval_pending_count"] == 0
    assert runtime["risky_tool_event_count"] == 0
    assert runtime["invocations"][0]["tool_events"][0]["tool_name"] == "file_read"


@pytest.mark.asyncio
async def test_orchestrator_passes_goal_capability_tool_allowlist_to_engine() -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(engine=engine)

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Summarize the deployment risks.",
            protocol={"protocol": "teacher_student_distill"},
            metadata={
                "goal_capability_policy": {
                    "allowed_tools": [" file_read ", "web_search", "file_read", ""],
                },
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                        "judge": "judge-model",
                    }
                },
            },
        )
    )

    assert result.state == "succeeded"
    assert len(engine.invoke_calls) == 3
    assert all(
        call["tool_allowlist"] == ["file_read", "web_search"]
        for call in engine.invoke_calls
    )
    assert all(call["tool_denylist"] is None for call in engine.invoke_calls)


@pytest.mark.asyncio
async def test_orchestrator_passes_goal_operator_tool_denylist_to_engine() -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(engine=engine)

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Summarize the deployment risks.",
            protocol={"protocol": "teacher_student_distill"},
            metadata={
                "goal_operator_controls": {
                    "tool_denylist": [" web_fetch ", "web_search", "web_fetch", ""],
                },
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                        "judge": "judge-model",
                    }
                },
            },
        )
    )

    assert result.state == "succeeded"
    assert len(engine.invoke_calls) == 3
    assert all(
        call["tool_denylist"] == ["web_fetch", "web_search"]
        for call in engine.invoke_calls
    )
    assert all(call["tool_allowlist"] is None for call in engine.invoke_calls)


@pytest.mark.asyncio
async def test_orchestrator_derives_blocked_web_domain_policy_for_engine_and_evidence() -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(engine=engine)

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Summarize the deployment risks.",
            protocol={"protocol": "teacher_student_distill"},
            metadata={
                "permission_policy": {
                    "blocked_web_domains": ["preexisting.test"],
                },
                "goal_operator_controls": {
                    "blocked_domains": [
                        " Example.com ",
                        "blocked.example.org",
                        "example.com",
                    ],
                },
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                        "judge": "judge-model",
                        "verifier": "verifier-model",
                    }
                },
                "summary": {
                    "evidence_queries": ["approved deployment note"],
                },
            },
        )
    )

    assert result.state == "succeeded"
    assert len(engine.invoke_calls) == 4
    assert all(
        call["permission_policy"] == {
            "blocked_web_domains": [
                "preexisting.test",
                "example.com",
                "blocked.example.org",
            ],
        }
        for call in engine.invoke_calls
    )
    assert len(engine.evidence_calls) == 1
    assert engine.evidence_calls[0]["metadata"]["permission_policy"] == {
        "blocked_web_domains": [
            "preexisting.test",
            "example.com",
            "blocked.example.org",
        ],
    }


@pytest.mark.asyncio
async def test_orchestrator_role_candidate_fallback_records_diagnostics() -> None:
    engine = _GenerateOnlyEngineStub()
    orchestrator = MultiAgentOrchestrator(engine=engine)

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Summarize the deployment risks.",
            protocol={"protocol": "teacher_student_distill"},
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                        "judge": "judge-model",
                    }
                }
            },
        )
    )

    assert result.selected_candidate_id == "student"
    assert engine.invoke_calls == []
    assert [call["model_id"] for call in engine.calls] == [
        "teacher-model",
        "student-model",
        "judge-model",
    ]
    teacher_diagnostics = result.candidates[0].metadata["diagnostics"]
    assert teacher_diagnostics["fallback_reason"] == "invoke_unavailable_used_generate_with_configured_model"


@pytest.mark.asyncio
async def test_orchestrator_uses_configured_models_for_dr_zero_protocol() -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(engine=engine)

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Self-evolve deployment search tasks.",
            protocol={
                "protocol": "dr_zero_self_evolve",
                "proposal_sample_size": 2,
                "solver_rollouts_per_task": 1,
            },
            guidance_messages=["Prefer verifiable tasks."],
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "proposer": "proposer-model",
                        "solver": "solver-model",
                        "verifier": "verifier-model",
                        "judge": "judge-model",
                    }
                },
                "summary": {
                    "evidence_packets": [
                        {
                            "evidence_id": "src-1",
                            "title": "Deployment note",
                            "content": "Evidence supports solver.",
                        }
                    ]
                },
            },
        )
    )

    assert result.protocol == "dr_zero_self_evolve"
    assert result.selected_candidate_id == "solver_1_1"
    assert result.artifacts["synthetic_tasks"]["parse_diagnostics"]["status"] == "parsed"
    assert len(result.artifacts["synthetic_tasks"]["tasks"]) == 2
    assert result.artifacts["solver_rollouts"]["rollout_count"] == 2
    assert result.artifacts["reward_summary"]["status"] == "pending_verification"
    assert result.artifacts["curriculum_state"]["iterations_configured"] == 1
    assert result.artifacts["drzero_iteration_summary"]["proposal_parse_status"] == "parsed"
    assert [call["inference_overrides"]["model"] for call in engine.invoke_calls] == [
        "proposer-model",
        "solver-model",
        "solver-model",
        "verifier-model",
        "judge-model",
    ]
    assert all(candidate.role_id == "solver" for candidate in result.candidates)
    assert result.candidates[0].metadata["synthetic_task_id"] == "task-1"


@pytest.mark.asyncio
async def test_orchestrator_role_guidance_stays_isolated_across_debate_roles() -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(engine=engine)
    directive = "Verifier-only directive."

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Assess the deployment summary for evidence-backed release guidance.",
            protocol={"protocol": "multi_agent_debate"},
            guidance_messages=["Keep responses concise."],
            role_guidance_messages={"verifier": [directive]},
            metadata={
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
                "evaluation_policy": {
                    "research": {
                        "enabled": True,
                        "preset": "smart_judge_research_debate",
                        "output_targets": ["research_brief"],
                        "source_mode": "hybrid",
                        "citation_policy": "claim_level_required",
                        "local_worker_count": 1,
                        "max_research_queries": 2,
                        "max_sources_per_query": 2,
                        "debate_rounds": 1,
                    }
                },
                "summary": {
                    "evidence_queries": ["approved deployment note"],
                    "evidence_packets": [
                        {
                            "evidence_id": "src-1",
                            "title": "Deployment note",
                            "content": "Approved route is supported.",
                        }
                    ],
                },
            },
        )
    )

    assert result.state == "succeeded"
    verifier_prompts = [
        str(call["message"])
        for call in engine.invoke_calls
        if call["inference_overrides"]["model"] == "verifier-model"
    ]
    assert any(directive in prompt for prompt in verifier_prompts)
    for model_id in {"planner-model", "debater-a-model", "debater-b-model", "judge-model"}:
        assert all(directive not in str(call["message"]) for call in engine.invoke_calls if call["inference_overrides"]["model"] == model_id)


@pytest.mark.asyncio
async def test_orchestrator_role_guidance_stays_isolated_across_dr_zero_solver_roles() -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(engine=engine)
    directive = "Verifier-only directive."

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Self-evolve deployment search tasks.",
            protocol={"protocol": "dr_zero_self_evolve", "proposal_sample_size": 1, "solver_rollouts_per_task": 1},
            guidance_messages=["Prefer verifiable tasks."],
            role_guidance_messages={"verifier": [directive]},
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "proposer": "proposer-model",
                        "solver": "solver-model",
                        "verifier": "verifier-model",
                        "judge": "judge-model",
                    }
                },
                "summary": {
                    "evidence_packets": [
                        {
                            "evidence_id": "src-1",
                            "title": "Deployment note",
                            "content": "Evidence supports solver.",
                        }
                    ]
                },
            },
        )
    )

    assert result.state == "succeeded"
    verifier_prompts = [
        str(call["message"])
        for call in engine.invoke_calls
        if call["inference_overrides"]["model"] == "verifier-model"
    ]
    assert any(directive in prompt for prompt in verifier_prompts)
    for model_id in {"proposer-model", "solver-model", "judge-model"}:
        assert all(directive not in str(call["message"]) for call in engine.invoke_calls if call["inference_overrides"]["model"] == model_id)


@pytest.mark.asyncio
async def test_orchestrator_dr_zero_falls_back_when_proposer_json_is_unparseable() -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(engine=engine)

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Create a fallback synthetic task.",
            protocol={"protocol": "dr_zero_self_evolve"},
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "proposer": "teacher-model",
                        "solver": "solver-model",
                        "judge": "judge-model",
                    }
                }
            },
        )
    )

    assert result.protocol == "dr_zero_self_evolve"
    assert result.artifacts["synthetic_tasks"]["parse_diagnostics"]["status"] == "fallback"
    assert result.artifacts["synthetic_tasks"]["tasks"][0]["metadata"]["fallback"] is True
    assert result.candidates[0].metadata["synthetic_task_id"] == "task-1"


@pytest.mark.asyncio
async def test_orchestrator_prefers_verified_candidate_and_emits_verification_outputs() -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(engine=engine)

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Summarize the deployment risks.",
            protocol={"protocol": "teacher_student_distill"},
            guidance_messages=["Ground every claim."],
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                        "judge": "judge-model",
                        "verifier": "verifier-model",
                    }
                },
                "summary": {
                    "evidence_packets": [
                        {
                            "evidence_id": "src-1",
                            "title": "Deployment note",
                            "content": "Only the student summary matches the approved deployment note.",
                        }
                    ]
                },
            },
        )
    )

    assert result.selected_candidate_id == "student"
    assert result.artifacts["verification"]["verified_candidate_ids"] == ["student"]
    assert result.artifacts["verification"]["failed_candidate_ids"] == ["teacher"]
    assert any(event.type == "verification" for event in result.events)
    scores_by_candidate = {
        score["candidate_id"]: score
        for score in result.evaluation["scores"]
    }
    assert scores_by_candidate["teacher"]["evidence_gate"]["status"] == "failed"
    assert scores_by_candidate["student"]["evidence_gate"]["status"] == "verified"
    assert [call["inference_overrides"]["model"] for call in engine.invoke_calls] == [
        "teacher-model",
        "student-model",
        "verifier-model",
        "judge-model",
    ]
    assert engine.calls == []


@pytest.mark.asyncio
async def test_orchestrator_collects_evidence_queries_before_verification() -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(engine=engine)

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Summarize the deployment risks.",
            protocol={"protocol": "teacher_student_distill"},
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                        "judge": "judge-model",
                        "verifier": "verifier-model",
                    }
                },
                "summary": {
                    "evidence_queries": ["approved deployment note"],
                },
            },
        )
    )

    assert result.selected_candidate_id == "student"
    assert len(engine.evidence_calls) == 1
    assert engine.evidence_calls[0]["queries"] == ["approved deployment note"]
    assert engine.evidence_calls[0]["metadata"]["selected_models_roles"] == result.metadata["selected_models_roles"]
    assert result.artifacts["evidence_collection"]["collected_packet_count"] == 1
    assert result.artifacts["evidence_collection"]["mode"] == "hybrid"
    assert result.artifacts["verification"]["evidence_packet_count"] == 1
    artifact_events = [event for event in result.events if event.type == "artifact"]
    assert any(event.payload["name"] == "evidence_summary" for event in artifact_events)


@pytest.mark.asyncio
async def test_orchestrator_does_not_collect_evidence_when_policy_disables_it() -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(engine=engine)

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Summarize the deployment risks.",
            protocol={"protocol": "teacher_student_distill"},
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                        "judge": "judge-model",
                        "verifier": "verifier-model",
                    }
                },
                "summary": {
                    "evidence_queries": ["approved deployment note"],
                },
                "evaluation_policy": {
                    "evidence_collection": {
                        "enabled": False,
                    }
                },
            },
        )
    )

    assert result.selected_candidate_id == "student"
    assert engine.evidence_calls == []
    assert result.artifacts["evidence_collection"]["status"] == "disabled"


@pytest.mark.asyncio
async def test_orchestrator_uses_configured_models_for_multi_agent_debate_protocol() -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(engine=engine)

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Choose the safer migration strategy.",
            protocol={"protocol": "multi_agent_debate"},
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "debater_a": "debater-a-model",
                        "debater_b": "debater-b-model",
                        "judge": "judge-model",
                    }
                }
            },
        )
    )

    assert result.selected_candidate_id == "student" or result.selected_candidate_id == "debater_b"
    assert [call["inference_overrides"]["model"] for call in engine.invoke_calls] == [
        "debater-a-model",
        "debater-b-model",
        "debater-a-model",
        "debater-b-model",
        "judge-model",
    ]
    assert engine.calls == []
    debate_events = [event for event in result.events if event.type == "role_output"]
    assert len(debate_events) == 4
    assert result.artifacts["debate_state"]["claim_cards"]
    assert result.artifacts["debate_context_snapshot"]["snapshots"]


@pytest.mark.asyncio
async def test_orchestrator_builds_research_debate_artifacts_and_honors_output_modes() -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(engine=engine)

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Assess the deployment summary for evidence-backed release guidance.",
            protocol={"protocol": "multi_agent_debate"},
            metadata={
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
                "evaluation_policy": {
                    "research": {
                        "enabled": True,
                        "preset": "smart_judge_research_debate",
                        "output_targets": ["research_brief"],
                        "source_mode": "hybrid",
                        "citation_policy": "claim_level_required",
                        "local_worker_count": 2,
                        "max_research_queries": 4,
                        "max_sources_per_query": 3,
                        "debate_rounds": 2,
                    }
                },
                "summary": {
                    "evidence_queries": ["approved deployment note"],
                },
            },
        )
    )

    assert "research_plan" in result.artifacts
    assert "source_quality_table" in result.artifacts
    assert "claim_evidence_map" in result.artifacts
    assert "research_brief" in result.artifacts
    assert result.artifacts["research_plan"]["worker_outputs"]["worker_count"] == 2
    assert result.artifacts["claim_evidence_map"]["claims"]
    assert result.artifacts["debate_state"]["claim_cards"][0]["support_status"] in {
        "supported",
        "contested",
        "needs_evidence",
        "refuted",
    }


@pytest.mark.asyncio
async def test_orchestrator_controlled_execution_approves_and_executes(tmp_path) -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(
        engine=engine,
        controlled_exec_require_approval=False,
    )

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Run a bounded smoke command.",
            protocol={"protocol": "controlled_subagent_execution"},
            metadata={
                "task_workspace_dir": str(tmp_path),
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
        )
    )

    runtime = result.artifacts["controlled_execution_runtime"]
    assert runtime["execution_boundary"] == "subagents_propose_controller_approves_shared_runtime_executes"
    assert result.artifacts["execution_requests"]["parse_diagnostics"]["status"] == "parsed"
    assert result.artifacts["controller_decisions"]["items"][0]["status"] == "approved"
    execution_result = result.artifacts["execution_results"]["items"][0]
    assert execution_result["status"] != "skipped"
    assert execution_result["command"] == "echo controlled-ok"
    if execution_result["status"] == "completed":
        assert "controlled-ok" in str(execution_result["stdout"])
    profiles = [call["execution_profile"] for call in engine.invoke_calls]
    assert "subagent_execution_request" in profiles
    assert "controller_exec" in profiles


@pytest.mark.asyncio
async def test_orchestrator_controlled_execution_approval_pending_returns_waiting_approval_state(tmp_path) -> None:
    engine = _ConfiguredModelEngineStub()
    approval_store = InMemoryApprovalStore()
    orchestrator = MultiAgentOrchestrator(
        engine=engine,
        exec_approval_store=approval_store,
        controlled_exec_require_approval=True,
    )

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Run a bounded smoke command after approval.",
            protocol={"protocol": "controlled_subagent_execution"},
            metadata={
                "task_workspace_dir": str(tmp_path),
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
        )
    )

    assert result.state == "awaiting_approval"
    assert result.artifacts["execution_results"]["items"][0]["status"] == "approval_pending"
    assert result.artifacts["controlled_execution_runtime"]["approval_pending_count"] == 1

    approval_state = result.metadata["approval_state"]
    recovery_state = result.metadata["recovery_state"]
    assert approval_state["status"] == "awaiting_approval"
    assert approval_state["pending_count"] == 1
    assert recovery_state["status"] == "awaiting_approval"
    assert recovery_state["action"] == "await_approval"
    assert recovery_state["stage"] == "controlled_execution_exec:req-1"
    assert recovery_state["approval_state"]["approval_ids"] == approval_state["approval_ids"]

    pending = approval_state["pending_approvals"][0]
    approval_id = pending["approval_id"]
    assert pending["source"] == "controlled_execution"
    assert pending["request_id"] == "req-1"
    assert pending["task_key"] == "controlled_execution_exec:req-1"
    assert pending["tool_name"] == "exec_command"
    approval_request = approval_store.get(approval_id)
    assert approval_request is not None
    assert pending["reason"] == approval_request.reason

    resume_snapshot = recovery_state["resume_payload"]["role_task_snapshot"]
    assert "controlled_execution_exec:req-1" not in resume_snapshot["tasks"]
    assert "controlled_execution_evaluator" not in resume_snapshot["tasks"]
    assert "evaluator" not in resume_snapshot["roles"]
    assert resume_snapshot["tasks"]["controlled_execution_controller:req-1"]["status"] == "completed"
    assert result.events[-1].payload["current_state"] == "awaiting_approval"
    assert result.events[-1].payload["reason"] == "approval_pending"


@pytest.mark.asyncio
async def test_orchestrator_controlled_execution_rejects_without_running(tmp_path) -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(
        engine=engine,
        controlled_exec_require_approval=False,
    )

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Review but do not run unnecessary command.",
            protocol={"protocol": "controlled_subagent_execution"},
            metadata={
                "task_workspace_dir": str(tmp_path),
                "selected_models_roles": {
                    "by_role": {
                        "planner": "controlled-planner-model",
                        "executor": "controlled-executor-model",
                        "controller": "controlled-reject-controller-model",
                        "evaluator": "controlled-evaluator-model",
                        "judge": "controlled-judge-model",
                        "verifier": "controlled-verifier-model",
                    }
                },
            },
        )
    )

    assert result.artifacts["controller_decisions"]["items"][0]["status"] == "rejected"
    assert result.artifacts["execution_results"]["items"][0]["status"] == "skipped"


@pytest.mark.asyncio
async def test_orchestrator_debate_can_attach_controlled_execution_capability(tmp_path) -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(
        engine=engine,
        controlled_exec_require_approval=False,
    )

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Debate the best deployment path and validate a bounded command if needed.",
            protocol={"protocol": "multi_agent_debate", "rounds": 2},
            metadata={
                "task_workspace_dir": str(tmp_path),
                "execution_policy": {
                    "mode": "controlled",
                    "max_execution_requests": 1,
                    "max_commands_per_request": 1,
                    "default_timeout_sec": 30,
                    "background_allowed": False,
                },
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
            },
        )
    )

    assert result.protocol == "multi_agent_debate"
    assert result.selected_candidate_id == "debater_b"
    runtime = result.artifacts["controlled_execution_runtime"]
    assert runtime["primary_workflow"] is False
    assert runtime["execution_boundary"] == "subagents_propose_controller_approves_shared_runtime_executes"
    profiles = [call["execution_profile"] for call in engine.invoke_calls]
    assert "controller_exec" in profiles
    assert any(
        "Controlled execution context summary" in str(call["message"])
        for call in engine.invoke_calls
        if call["execution_profile"] in {"subagent_readonly", "judge", "verifier"}
    )
    assert result.artifacts["controlled_execution_runtime"]["executed_count"] == 1


@pytest.mark.asyncio
async def test_orchestrator_degrades_when_one_debate_role_fails() -> None:
    engine = _DebaterBFailureEngine()
    orchestrator = MultiAgentOrchestrator(engine=engine)

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Compare two deployment approaches.",
            protocol={"protocol": "multi_agent_debate", "rounds": 2},
            run_policy={
                "max_subagent_failures_per_role": 0,
                "on_subagent_disconnect": "retry_then_degrade",
            },
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "debater_a": "debater-a-model",
                        "debater_b": "debater-b-model",
                        "judge": "judge-model",
                        "verifier": "verifier-model",
                    }
                }
            },
        )
    )

    assert result.state == "succeeded"
    assert result.metadata["degraded"] is True
    assert [candidate.role_id for candidate in result.candidates] == ["debater_a"]
    snapshot = result.artifacts["subagent_health_snapshot"]
    assert snapshot["degraded"] is True
    assert snapshot["degraded_role_ids"] == ["debater_b"]
    assert snapshot["failure_counts"]["debater_b"] == 1


@pytest.mark.asyncio
async def test_orchestrator_resume_continues_from_protocol_checkpoint_without_rerunning_protocol() -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(engine=engine)
    original = orchestrator._raise_if_run_policy_exhausted

    def _pause_after_protocol(**kwargs: object) -> None:
        original(**kwargs)
        if kwargs.get("stage") == "protocol_completed":
            raise RunPolicyStop(
                status="awaiting_resources",
                action="pause",
                reason="pause after protocol for resume test",
                stage="protocol_completed",
                checkpoint=orchestrator._latest_checkpoint,
            )

    orchestrator._raise_if_run_policy_exhausted = _pause_after_protocol  # type: ignore[method-assign]

    paused = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Summarize the deployment risks.",
            protocol={"protocol": "teacher_student_distill"},
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                        "judge": "judge-model",
                        "verifier": "verifier-model",
                    }
                },
                "summary": {
                    "evidence_packets": [
                        {
                            "evidence_id": "src-1",
                            "title": "Deployment note",
                            "content": "Only the student summary matches the approved deployment note.",
                        }
                    ]
                },
            },
        )
    )

    assert paused.state == "awaiting_resources"
    assert paused.metadata["recovery_state"]["resume_executor"] == "continue_from_checkpoint"

    resumed = await MultiAgentOrchestrator(engine=engine).run(
        MultiAgentRunRequest(
            run_id=paused.run_id,
            task_input="Summarize the deployment risks.",
            protocol={"protocol": "teacher_student_distill"},
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                        "judge": "judge-model",
                        "verifier": "verifier-model",
                    }
                },
                "summary": {
                    "evidence_packets": [
                        {
                            "evidence_id": "src-1",
                            "title": "Deployment note",
                            "content": "Only the student summary matches the approved deployment note.",
                        }
                    ],
                    "recovery_state": paused.metadata["recovery_state"],
                },
            },
        )
    )

    assert resumed.state == "succeeded"
    assert resumed.selected_candidate_id == "student"
    invoked_models = [call["inference_overrides"]["model"] for call in engine.invoke_calls]
    assert invoked_models.count("teacher-model") == 1
    assert invoked_models.count("student-model") == 1
    assert invoked_models[-1] == "judge-model"


@pytest.mark.asyncio
async def test_orchestrator_resume_reassigns_missing_role_and_exposes_role_task_snapshot() -> None:
    engine = _ConfiguredModelEngineStub()

    result = await MultiAgentOrchestrator(engine=engine).run(
        MultiAgentRunRequest(
            task_input="Summarize the deployment risks.",
            protocol={"protocol": "teacher_student_distill"},
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "judge": "judge-model",
                    }
                },
                "summary": {
                    "recovery_state": {
                        "resume_payload": {
                            "executor": "continue_from_checkpoint",
                            "stage": "student_generation",
                            "checkpoint": {
                                "checkpoint_index": 2,
                                "stage": "student_generation",
                            },
                            "role_task_snapshot": {
                                "roles": {
                                    "teacher": {
                                        "role_id": "teacher",
                                        "status": "completed",
                                        "stage": "teacher_generation",
                                        "assigned_model_id": "teacher-model",
                                        "original_model_id": "teacher-model",
                                        "candidate": {
                                            "candidate_id": "teacher",
                                            "role_id": "teacher",
                                            "content": "Teacher evidence-backed draft.",
                                            "metadata": {"model_id": "teacher-model"},
                                        },
                                    },
                                    "student": {
                                        "role_id": "student",
                                        "status": "running",
                                        "stage": "student_generation",
                                        "assigned_model_id": "student-model",
                                        "original_model_id": "student-model",
                                    },
                                }
                            },
                        }
                    }
                },
            },
        )
    )

    assert result.state == "succeeded"
    snapshot = result.artifacts["role_task_snapshot"]
    assert snapshot["roles"]["teacher"]["status"] == "completed"
    assert snapshot["roles"]["teacher"]["candidate"]["candidate_id"] == "teacher"
    assert snapshot["resume_plan"]["assignments"]["student"]["assigned_model_id"] == "teacher-model"
    assert snapshot["resume_plan"]["assignments"]["student"]["assignment_source"] == "reassigned_to_available_model"
    assert [call["inference_overrides"]["model"] for call in engine.invoke_calls] == [
        "teacher-model",
        "judge-model",
    ]


@pytest.mark.asyncio
async def test_orchestrator_resume_uses_same_task_snapshot_method_for_debate_protocol() -> None:
    engine = _ConfiguredModelEngineStub()

    result = await MultiAgentOrchestrator(engine=engine).run(
        MultiAgentRunRequest(
            task_input="Compare both deployment proposals.",
            protocol={"protocol": "multi_agent_debate", "rounds": 1},
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "debater_a": "debater-a-model",
                        "judge": "judge-model",
                    }
                },
                "summary": {
                    "recovery_state": {
                        "resume_payload": {
                            "executor": "continue_from_checkpoint",
                            "stage": "debate_round_1:debater_b",
                            "checkpoint": {
                                "checkpoint_index": 2,
                                "stage": "debate_round_1:debater_b",
                            },
                            "role_task_snapshot": {
                                "roles": {
                                    "debater_a": {
                                        "role_id": "debater_a",
                                        "status": "completed",
                                        "stage": "debate_round_1:debater_a",
                                        "assigned_model_id": "debater-a-model",
                                        "candidate": {
                                            "candidate_id": "debater_a",
                                            "role_id": "debater_a",
                                            "content": "Argument A prefers the first route.",
                                            "metadata": {"model_id": "debater-a-model"},
                                        },
                                    },
                                    "debater_b": {
                                        "role_id": "debater_b",
                                        "status": "running",
                                        "stage": "debate_round_1:debater_b",
                                        "assigned_model_id": "debater-b-model",
                                        "original_model_id": "debater-b-model",
                                    },
                                },
                                "tasks": {
                                    "debate_round_1:debater_a": {
                                        "task_key": "debate_round_1:debater_a",
                                        "role_id": "debater_a",
                                        "status": "completed",
                                        "stage": "debate_round_1:debater_a",
                                        "assigned_model_id": "debater-a-model",
                                        "candidate": {
                                            "candidate_id": "debater_a",
                                            "role_id": "debater_a",
                                            "content": "Argument A prefers the first route.",
                                            "metadata": {"model_id": "debater-a-model"},
                                        },
                                    }
                                },
                            },
                        }
                    }
                },
            },
        )
    )

    assert result.state == "succeeded"
    snapshot = result.artifacts["role_task_snapshot"]
    assert snapshot["tasks"]["debate_round_1:debater_a"]["status"] == "completed"
    assert snapshot["resume_plan"]["assignments"]["debater_b"]["assigned_model_id"] == "debater-a-model"
    assert snapshot["resume_plan"]["assignments"]["debater_b"]["assignment_source"] == "reassigned_to_available_model"
    assert [call["inference_overrides"]["model"] for call in engine.invoke_calls] == [
        "debater-a-model",
        "judge-model",
    ]


@pytest.mark.asyncio
async def test_orchestrator_resume_reuses_research_tasks_via_shared_task_snapshot() -> None:
    engine = _ConfiguredModelEngineStub()

    result = await MultiAgentOrchestrator(engine=engine).run(
        MultiAgentRunRequest(
            task_input="Compare both deployment proposals.",
            protocol={"protocol": "multi_agent_debate", "rounds": 1},
            metadata={
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
                "evaluation_policy": {
                    "research": {
                        "enabled": True,
                        "preset": "smart_judge_research_debate",
                        "output_targets": ["research_brief"],
                        "source_mode": "hybrid",
                        "citation_policy": "claim_level_required",
                        "local_worker_count": 2,
                        "max_research_queries": 4,
                        "max_sources_per_query": 3,
                        "debate_rounds": 1,
                    }
                },
                "summary": {
                    "evidence_queries": ["approved deployment note"],
                    "recovery_state": {
                        "resume_payload": {
                            "executor": "continue_from_checkpoint",
                            "stage": "research_synthesis",
                            "checkpoint": {
                                "checkpoint_index": 3,
                                "stage": "research_synthesis",
                            },
                            "metadata_state": {
                                "research_runtime": {
                                    "research_plan": {
                                        "subquestions": ["What does the source say?"],
                                        "evidence_queries": ["approved deployment note"],
                                    },
                                    "source_quality_table": {
                                        "claims": [],
                                        "sources": [],
                                    },
                                    "worker_outputs": {
                                        "worker_count": 2,
                                        "workers": [
                                            {"worker_id": "worker-1", "summary": "verified evidence"},
                                            {"worker_id": "worker-2", "summary": "contradiction note"},
                                        ],
                                    },
                                }
                            },
                            "evidence_packets": [
                                {
                                    "evidence_id": "src-1",
                                    "title": "Deployment note",
                                    "content": "Approved route is supported.",
                                }
                            ],
                            "role_task_snapshot": {
                                "roles": {
                                    "research_planner": {
                                        "role_id": "research_planner",
                                        "status": "completed",
                                        "stage": "research_planning",
                                    },
                                    "research_worker": {
                                        "role_id": "research_worker",
                                        "status": "completed",
                                        "stage": "research_workers",
                                    },
                                    "research_synthesizer": {
                                        "role_id": "research_synthesizer",
                                        "status": "completed",
                                        "stage": "research_synthesis",
                                    },
                                },
                                "tasks": {
                                    "research_planning": {
                                        "task_key": "research_planning",
                                        "role_id": "research_planner",
                                        "status": "completed",
                                        "stage": "research_planning",
                                        "result_summary": {
                                            "research_plan": {
                                                "subquestions": ["What does the source say?"],
                                                "evidence_queries": ["approved deployment note"],
                                            }
                                        },
                                    },
                                    "research_workers": {
                                        "task_key": "research_workers",
                                        "role_id": "research_worker",
                                        "status": "completed",
                                        "stage": "research_workers",
                                        "result_summary": {
                                            "worker_outputs": {
                                                "worker_count": 2,
                                                "workers": [
                                                    {"worker_id": "worker-1", "summary": "verified evidence"},
                                                    {"worker_id": "worker-2", "summary": "contradiction note"},
                                                ],
                                            }
                                        },
                                    },
                                    "research_synthesis": {
                                        "task_key": "research_synthesis",
                                        "role_id": "research_synthesizer",
                                        "status": "completed",
                                        "stage": "research_synthesis",
                                        "result_summary": {
                                            "research_brief": {
                                                "markdown": "# Research Brief\n\nReused brief.",
                                                "summary": "Reused brief.",
                                            }
                                        },
                                    },
                                },
                            },
                        }
                    },
                },
            },
        )
    )

    assert result.state == "succeeded"
    assert result.artifacts["research_brief"]["summary"] == "Reused brief."
    models = [call["inference_overrides"]["model"] for call in engine.invoke_calls]
    assert "planner-model" not in models
    assert "local-worker-model" not in models
    assert "synth-model" not in models


@pytest.mark.asyncio
async def test_orchestrator_resume_reuses_controlled_execution_tasks_via_shared_snapshot() -> None:
    engine = _ConfiguredModelEngineStub()

    result = await MultiAgentOrchestrator(
        engine=engine,
        controlled_exec_require_approval=False,
    ).run(
        MultiAgentRunRequest(
            task_input="Resume the bounded controlled execution workflow.",
            protocol={"protocol": "controlled_subagent_execution"},
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "planner": "controlled-evaluator-model",
                        "judge": "controlled-judge-model",
                        "verifier": "controlled-verifier-model",
                    }
                },
                "summary": {
                    "recovery_state": {
                        "resume_payload": {
                            "executor": "continue_from_checkpoint",
                            "stage": "controlled_execution_evaluator",
                            "checkpoint": {
                                "checkpoint_index": 4,
                                "stage": "controlled_execution_evaluator",
                            },
                            "role_task_snapshot": {
                                "roles": {
                                    "planner": {
                                        "role_id": "planner",
                                        "status": "completed",
                                        "stage": "controlled_execution_planner",
                                        "assigned_model_id": "controlled-planner-model",
                                        "candidate": {
                                            "candidate_id": "planner",
                                            "role_id": "planner",
                                            "content": "Plan: run one safe command and inspect the output.",
                                            "metadata": {"model_id": "controlled-planner-model"},
                                        },
                                    },
                                    "executor": {
                                        "role_id": "executor",
                                        "status": "completed",
                                        "stage": "controlled_execution_executor",
                                        "assigned_model_id": "controlled-executor-model",
                                        "candidate": {
                                            "candidate_id": "executor",
                                            "role_id": "executor",
                                            "content": (
                                                '{"execution_requests":[{"request_id":"req-1",'
                                                '"command":"echo controlled-ok","shell":"powershell",'
                                                '"timeout":30,"background":true,"rationale":"smoke test",'
                                                '"expected_artifacts":["stdout"],'
                                                '"success_metric":"stdout contains controlled-ok"}]}'
                                            ),
                                            "metadata": {"model_id": "controlled-executor-model"},
                                        },
                                    },
                                    "controller": {
                                        "role_id": "controller",
                                        "status": "completed",
                                        "stage": "controlled_execution_controller:req-1",
                                        "assigned_model_id": "controlled-controller-model",
                                    },
                                    "evaluator": {
                                        "role_id": "evaluator",
                                        "status": "running",
                                        "stage": "controlled_execution_evaluator",
                                        "assigned_model_id": "controlled-evaluator-model",
                                        "original_model_id": "controlled-evaluator-model",
                                    },
                                },
                                "tasks": {
                                    "controlled_execution_planner": {
                                        "task_key": "controlled_execution_planner",
                                        "role_id": "planner",
                                        "status": "completed",
                                        "stage": "controlled_execution_planner",
                                        "assigned_model_id": "controlled-planner-model",
                                        "result_summary": {
                                            "planner_output": {
                                                "candidate_id": "planner",
                                                "role_id": "planner",
                                                "content": "Plan: run one safe command and inspect the output.",
                                                "metadata": {"model_id": "controlled-planner-model"},
                                            }
                                        },
                                    },
                                    "controlled_execution_executor": {
                                        "task_key": "controlled_execution_executor",
                                        "role_id": "executor",
                                        "status": "completed",
                                        "stage": "controlled_execution_executor",
                                        "assigned_model_id": "controlled-executor-model",
                                        "result_summary": {
                                            "executor_output": {
                                                "candidate_id": "executor",
                                                "role_id": "executor",
                                                "content": (
                                                    '{"execution_requests":[{"request_id":"req-1",'
                                                    '"command":"echo controlled-ok","shell":"powershell",'
                                                    '"timeout":30,"background":true,"rationale":"smoke test",'
                                                    '"expected_artifacts":["stdout"],'
                                                    '"success_metric":"stdout contains controlled-ok"}]}'
                                                ),
                                                "metadata": {"model_id": "controlled-executor-model"},
                                            },
                                            "execution_requests": [
                                                {
                                                    "request_id": "req-1",
                                                    "command": "echo controlled-ok",
                                                    "shell": "powershell",
                                                    "timeout": 30,
                                                    "background": True,
                                                    "rationale": "smoke test",
                                                    "expected_artifacts": ["stdout"],
                                                    "success_metric": "stdout contains controlled-ok",
                                                }
                                            ],
                                            "request_parse_diagnostics": {
                                                "status": "parsed",
                                                "parsed_request_count": 1,
                                                "max_execution_requests": 1,
                                                "reason": None,
                                            },
                                        },
                                    },
                                    "controlled_execution_controller:req-1": {
                                        "task_key": "controlled_execution_controller:req-1",
                                        "role_id": "controller",
                                        "status": "completed",
                                        "stage": "controlled_execution_controller:req-1",
                                        "assigned_model_id": "controlled-controller-model",
                                        "result_summary": {
                                            "controller_decision": {
                                                "decision_id": "controller-decision-1",
                                                "request_id": "req-1",
                                                "status": "approved",
                                                "reason": "bounded smoke command",
                                                "command": "echo controlled-ok",
                                                "shell": "powershell",
                                                "timeout": 30,
                                                "background": True,
                                                "raw_content": (
                                                    '{"status":"approved","reason":"bounded smoke command",'
                                                    '"command":"echo controlled-ok","shell":"powershell",'
                                                    '"timeout":30,"background":true}'
                                                ),
                                                "diagnostics": {},
                                            }
                                        },
                                    },
                                    "controlled_execution_exec:req-1": {
                                        "task_key": "controlled_execution_exec:req-1",
                                        "role_id": "controller",
                                        "status": "completed",
                                        "stage": "controlled_execution_exec:req-1",
                                        "assigned_model_id": "controlled-controller-model",
                                        "result_summary": {
                                            "execution_result": {
                                                "request_id": "req-1",
                                                "status": "completed",
                                                "command": "echo controlled-ok",
                                                "shell": "powershell",
                                                "workdir": None,
                                                "timeout": 30,
                                                "background": True,
                                                "log_path": "H:/tmp/req-1/session.log",
                                                "session_log_path": "H:/tmp/req-1/session.log",
                                                "checkpoint_dir": "H:/tmp/req-1/checkpoints",
                                                "root_dir": "H:/tmp/req-1",
                                                "manifest_path": "H:/tmp/req-1/manifest.json",
                                                "stdout_log_path": "H:/tmp/req-1/stdout.log",
                                                "stderr_log_path": "H:/tmp/req-1/stderr.log",
                                                "stdout": "controlled-ok",
                                                "stderr": "",
                                                "error": None,
                                                "metadata": {
                                                    "session_id": "detached-req-1",
                                                    "status": "completed",
                                                    "detached": True,
                                                    "recovery_supported": True,
                                                    "detached_layout": {
                                                        "root_dir": "H:/tmp/req-1",
                                                        "log_path": "H:/tmp/req-1/session.log",
                                                        "session_log_path": "H:/tmp/req-1/session.log",
                                                        "checkpoint_dir": "H:/tmp/req-1/checkpoints",
                                                        "manifest_path": "H:/tmp/req-1/manifest.json",
                                                        "stdout_log_path": "H:/tmp/req-1/stdout.log",
                                                        "stderr_log_path": "H:/tmp/req-1/stderr.log",
                                                        "runtime_state_root": "H:/_python/agent_mochi/.mochi/exec-runtime",
                                                    },
                                                },
                                            }
                                        },
                                    },
                                },
                            },
                        }
                    }
                },
            },
        )
    )

    assert result.state == "succeeded"
    assert result.selected_candidate_id == "evaluator"
    assert result.artifacts["execution_requests"]["items"][0]["request_id"] == "req-1"
    assert result.artifacts["controller_decisions"]["items"][0]["status"] == "approved"
    assert result.artifacts["execution_results"]["items"][0]["metadata"]["session_id"] == "detached-req-1"
    assert result.artifacts["detached_exec_jobs"]["count"] == 1
    assert result.artifacts["controlled_execution_runtime"]["detached_exec_job_count"] == 1

    snapshot = result.artifacts["role_task_snapshot"]
    assert snapshot["tasks"]["controlled_execution_planner"]["status"] == "completed"
    assert snapshot["tasks"]["controlled_execution_executor"]["status"] == "completed"
    assert snapshot["tasks"]["controlled_execution_controller:req-1"]["status"] == "completed"
    assert snapshot["tasks"]["controlled_execution_exec:req-1"]["status"] == "completed"
    assert snapshot["tasks"]["controlled_execution_evaluator"]["status"] == "completed"
    assert snapshot["resume_plan"]["assignments"]["evaluator"]["assigned_model_id"] == "controlled-evaluator-model"
    assert snapshot["resume_plan"]["assignments"]["evaluator"]["assignment_source"] == "reassigned_to_available_model"

    models = [call["inference_overrides"]["model"] for call in engine.invoke_calls]
    assert "controlled-planner-model" not in models
    assert "controlled-executor-model" not in models
    assert "controlled-controller-model" not in models
    assert models.count("controlled-evaluator-model") == 1


@pytest.mark.asyncio
async def test_orchestrator_resume_reuses_saved_verification_and_evaluation_state() -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(engine=engine)
    original = orchestrator._raise_if_run_policy_exhausted

    def _pause_after_evaluation(**kwargs: object) -> None:
        original(**kwargs)
        if kwargs.get("stage") == "evaluation_completed":
            raise RunPolicyStop(
                status="awaiting_resources",
                action="pause",
                reason="pause after evaluation for resume test",
                stage="evaluation_completed",
                checkpoint=orchestrator._latest_checkpoint,
            )

    orchestrator._raise_if_run_policy_exhausted = _pause_after_evaluation  # type: ignore[method-assign]

    paused = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Summarize the deployment risks.",
            protocol={"protocol": "teacher_student_distill"},
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                        "judge": "judge-model",
                        "verifier": "verifier-model",
                    }
                },
                "summary": {
                    "evidence_packets": [
                        {
                            "evidence_id": "src-1",
                            "title": "Deployment note",
                            "content": "Only the student summary matches the approved deployment note.",
                        }
                    ]
                },
            },
        )
    )

    assert paused.state == "awaiting_resources"
    assert paused.metadata["recovery_state"]["resume_payload"]["evaluation"]["selected_candidate_id"] == "student"

    resumed = await MultiAgentOrchestrator(engine=engine).run(
        MultiAgentRunRequest(
            run_id=paused.run_id,
            task_input="Summarize the deployment risks.",
            protocol={"protocol": "teacher_student_distill"},
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                        "judge": "judge-model",
                        "verifier": "verifier-model",
                    }
                },
                "summary": {
                    "evidence_packets": [
                        {
                            "evidence_id": "src-1",
                            "title": "Deployment note",
                            "content": "Only the student summary matches the approved deployment note.",
                        }
                    ],
                    "recovery_state": paused.metadata["recovery_state"],
                },
            },
        )
    )

    assert resumed.state == "succeeded"
    assert resumed.selected_candidate_id == "student"
    assert resumed.artifacts["verification"]["verified_candidate_ids"] == ["student"]
    assert [call["inference_overrides"]["model"] for call in engine.invoke_calls] == [
        "teacher-model",
        "student-model",
        "verifier-model",
        "judge-model",
    ]


@pytest.mark.asyncio
async def test_orchestrator_resume_falls_back_to_restart_when_checkpoint_payload_is_incomplete() -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(engine=engine)

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Summarize the deployment risks.",
            protocol={"protocol": "teacher_student_distill"},
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                        "judge": "judge-model",
                    }
                },
                "summary": {
                    "recovery_state": {
                        "resume_payload": {
                            "executor": "continue_from_checkpoint",
                            "stage": "protocol_completed",
                            "checkpoint": {
                                "checkpoint_index": 3,
                                "stage": "protocol_completed",
                            },
                            "candidates": [],
                        }
                    }
                },
            },
        )
    )

    assert result.state == "succeeded"
    assert [call["inference_overrides"]["model"] for call in engine.invoke_calls] == [
        "teacher-model",
        "student-model",
        "judge-model",
    ]


@pytest.mark.asyncio
async def test_orchestrator_marks_run_stalled_on_heartbeat_timeout() -> None:
    engine = _TeacherHeartbeatTimeoutEngine()
    orchestrator = MultiAgentOrchestrator(engine=engine)

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Summarize the deployment note.",
            protocol={"protocol": "teacher_student_distill"},
            run_policy={
                "heartbeat_timeout_sec": 1,
                "max_subagent_failures_per_role": 0,
                "on_subagent_disconnect": "pause",
            },
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                    }
                }
            },
        )
    )

    assert result.state == "stalled"
    assert result.metadata["recovery_state"]["status"] == "stalled"
    assert result.metadata["recovery_state"]["action"] == "pause"
    snapshot = result.artifacts["subagent_health_snapshot"]
    assert snapshot["failure_counts"]["teacher"] == 1


@pytest.mark.asyncio
async def test_orchestrator_collects_collector_artifacts_and_emits_live_shard_events() -> None:
    engine = _CollectorToolEngine()
    observed_events = []

    async def _capture_event(event: object) -> None:
        observed_events.append(event)

    result = await MultiAgentOrchestrator(engine=engine).run(
        MultiAgentRunRequest(
            task_input="Collect a Discourse topic into dataset records.",
            protocol={"protocol": "teacher_student_distill"},
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                        "judge": "judge-model",
                        "verifier": "verifier-model",
                    }
                }
            },
            runtime_event_callback=_capture_event,
        )
    )

    live_event = next(
        event
        for event in observed_events
        if getattr(event, "type", None) == "artifact"
        and isinstance(getattr(event, "payload", None), dict)
        and event.payload.get("name") == "collector_shard_manifests"
    )

    assert live_event.payload["content"]["shards"][0]["shard_id"] == "discourse-topic-274354"
    assert result.artifacts["collector_shard_manifests"]["shards"][0]["status"] == "running"
    assert result.artifacts["collector_dataset_records"]["records"][0]["target"]["answer"] == (
        "First collected post"
    )
    assert result.artifacts["collector_record_provenance"]["records"][0]["source_id"] == (
        "topic:274354:post:1"
    )


@pytest.mark.asyncio
async def test_orchestrator_emits_live_subagent_runtime_artifacts() -> None:
    class _LiveRuntimeEngine:
        async def invoke(self, request: AgentInvocationRequest) -> AgentInvocationResult:
            model_id = str((request.inference_overrides or {}).get("model") or "")
            if model_id == "teacher-model":
                content = "Teacher evidence-backed draft."
                input_tokens = 11
                output_tokens = 7
            elif model_id == "student-model":
                content = "Student concise final answer."
                input_tokens = 9
                output_tokens = 5
            elif model_id == "judge-model":
                content = '{"selected_candidate_id":"student","scores":[]}'
                input_tokens = 4
                output_tokens = 3
            else:
                raise AssertionError(f"Unexpected invoke model_id: {model_id}")
            return AgentInvocationResult(
                content=content,
                events=[
                    FinalAnswerEvent(
                        content=content,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        generation_time_ms=12.5,
                        finish_reason="stop",
                    )
                ],
                diagnostics=AgentInvocationDiagnostics(
                    execution_profile=request.execution_profile,
                    tool_mode=request.tool_mode,
                    exposed_tools=[],
                    matched_tool_groups=[],
                ),
            )

    engine = _LiveRuntimeEngine()
    observed_events = []

    async def _capture_event(event: object) -> None:
        observed_events.append(event)

    result = await MultiAgentOrchestrator(engine=engine).run(
        MultiAgentRunRequest(
            task_input="Summarize the deployment risks.",
            protocol={"protocol": "teacher_student_distill"},
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                        "judge": "judge-model",
                    }
                }
            },
            runtime_event_callback=_capture_event,
        )
    )

    artifact_events = [
        event
        for event in observed_events
        if getattr(event, "type", None) == "artifact"
        and isinstance(getattr(event, "payload", None), dict)
    ]
    subagent_runtime_indices = [
        index
        for index, event in enumerate(observed_events)
        if getattr(event, "type", None) == "artifact"
        and isinstance(getattr(event, "payload", None), dict)
        and event.payload.get("name") == "subagent_runtime"
    ]
    evaluation_index = next(
        index
        for index, event in enumerate(observed_events)
        if getattr(event, "type", None) == "evaluation"
    )

    assert artifact_events
    assert subagent_runtime_indices
    assert subagent_runtime_indices[0] < evaluation_index
    live_runtime_payload = next(
        event.payload["content"]
        for event in artifact_events
        if event.payload.get("name") == "subagent_runtime"
    )
    assert live_runtime_payload["completed_invocation_count"] == 1
    assert live_runtime_payload["input_tokens"] == 11
    assert live_runtime_payload["output_tokens"] == 7
    assert live_runtime_payload["total_tokens"] == 18
    assert live_runtime_payload["generation_time_ms"] == 12.5
    assert result.artifacts["subagent_runtime"]["total_tokens"] == 39


@pytest.mark.asyncio
async def test_orchestrator_recovery_payload_keeps_collector_artifacts() -> None:
    engine = _CollectorToolEngine()
    orchestrator = MultiAgentOrchestrator(engine=engine)
    original = orchestrator._raise_if_run_policy_exhausted

    def _pause_after_protocol(**kwargs: object) -> None:
        original(**kwargs)
        if kwargs.get("stage") == "protocol_completed":
            raise RunPolicyStop(
                status="awaiting_resources",
                action="pause",
                reason="pause after collector tool execution",
                stage="protocol_completed",
                checkpoint=orchestrator._latest_checkpoint,
            )

    orchestrator._raise_if_run_policy_exhausted = _pause_after_protocol  # type: ignore[method-assign]

    paused = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Collect a Discourse topic into dataset records.",
            protocol={"protocol": "teacher_student_distill"},
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                        "judge": "judge-model",
                        "verifier": "verifier-model",
                    }
                }
            },
        )
    )

    assert paused.state == "awaiting_resources"
    resume_payload = paused.metadata["recovery_state"]["resume_payload"]["protocol_artifacts"]
    assert resume_payload["collector_shard_manifests"]["shards"][0]["shard_id"] == (
        "discourse-topic-274354"
    )
    assert resume_payload["collector_dataset_records"]["records"][0]["target"]["answer"] == (
        "First collected post"
    )
    assert resume_payload["collector_record_provenance"]["records"][0]["source_id"] == (
        "topic:274354:post:1"
    )
