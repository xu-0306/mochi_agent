from __future__ import annotations

import pytest

from mochi.agents.events import FinalAnswerEvent
from mochi.agents.invocation import (
    AgentInvocationDiagnostics,
    AgentInvocationRequest,
    AgentInvocationResult,
)
from mochi.agents.multi_agent.orchestrator import (
    MultiAgentOrchestrator,
    MultiAgentRunRequest,
    RunPolicyStop,
)

from ._support import _CollectorToolEngine


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
