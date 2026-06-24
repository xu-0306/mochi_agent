from __future__ import annotations

import pytest

from mochi.agents.multi_agent.orchestrator import (
    CandidateOutput,
    MultiAgentOrchestrator,
    MultiAgentRunRequest,
    MultiAgentRunResult,
)
from mochi.learning.dataset_exporter import export_run_to_dataset_records


@pytest.mark.asyncio
async def test_dataset_export_record_shape_from_multi_agent_run() -> None:
    orchestrator = MultiAgentOrchestrator()
    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Explain async in one paragraph",
            protocol={"protocol": "teacher_student_distill"},
            guidance_messages=["Prefer plain language."],
        )
    )

    records = export_run_to_dataset_records(result)
    assert len(records) == 1
    record = records[0]

    assert record["contract"] == "dataset_record"
    assert record["schema_version"] == "1.0"
    assert record["input"]
    assert "target" in record
    assert "supervision" in record
    assert record["metadata"]["source_provenance"]["run_id"] == result.run_id
    assert "evaluation" in record["evidence"]
    assert "events" in record["evidence"]
    assert "collected" not in record["evidence"]


def test_dataset_export_includes_collector_provenance_from_artifacts() -> None:
    result = MultiAgentRunResult(
        run_id="run-collector",
        protocol="teacher_student_distill",
        state="succeeded",
        task_input="Collect one forum thread",
        candidates=[
            CandidateOutput(
                candidate_id="student",
                role_id="student",
                content="Curated forum thread summary",
            )
        ],
        selected_candidate_id="student",
        evaluation={"scores": []},
        artifacts={
            "collector_record_provenance": {
                "source_url": "https://forum.example/thread-1",
                "source_id": "thread-1",
                "collected_at": "2026-06-24T00:00:00+00:00",
                "adapter_name": "forum_thread_adapter",
                "tool_name": "web_fetch",
                "tool_arguments": {"url": "https://forum.example/thread-1"},
                "license": "cc-by-sa-4.0",
                "policy_disposition": "allow",
                "dedupe_hash": "sha256:abc123",
                "shard_id": "forum-thread-1",
            }
        },
        events=[],
        metadata={},
    )

    records = export_run_to_dataset_records(result)

    provenance = records[0]["metadata"]["collector_provenance"]
    assert provenance["source_url"] == "https://forum.example/thread-1"
    assert provenance["adapter_name"] == "forum_thread_adapter"
    assert provenance["tool_name"] == "web_fetch"
    assert provenance["policy_disposition"] == "allow"
    assert provenance["dedupe_hash"] == "sha256:abc123"


def test_dataset_export_derives_collector_provenance_from_single_shard_manifest() -> None:
    result = MultiAgentRunResult(
        run_id="run-collector-shard",
        protocol="teacher_student_distill",
        state="succeeded",
        task_input="Collect one forum thread",
        candidates=[
            CandidateOutput(
                candidate_id="student",
                role_id="student",
                content="Curated forum thread summary",
            )
        ],
        selected_candidate_id="student",
        evaluation={"scores": []},
        artifacts={
            "collector_shards": [
                {
                    "shard_id": "forum-thread-1",
                    "adapter_name": "forum_thread_adapter",
                    "status": "completed",
                    "source": {
                        "url": "https://forum.example/thread-1",
                        "id": "thread-1",
                    },
                    "tool": {
                        "name": "web_fetch",
                        "arguments": {"url": "https://forum.example/thread-1"},
                    },
                    "policy": {
                        "license": "cc-by-sa-4.0",
                        "disposition": "allow",
                    },
                    "dedupe_hash": "sha256:def456",
                    "completed_at": "2026-06-24T00:00:00+00:00",
                }
            ]
        },
        events=[],
        metadata={},
    )

    records = export_run_to_dataset_records(result)

    provenance = records[0]["metadata"]["collector_provenance"]
    assert provenance["source_id"] == "thread-1"
    assert provenance["adapter_name"] == "forum_thread_adapter"
    assert provenance["tool_arguments"]["url"] == "https://forum.example/thread-1"
    assert provenance["license"] == "cc-by-sa-4.0"
    assert provenance["shard_id"] == "forum-thread-1"


def test_dataset_export_uses_collector_dataset_records_without_synthesizing_run_record() -> None:
    result = MultiAgentRunResult(
        run_id="run-collector-records",
        protocol="teacher_student_distill",
        state="succeeded",
        task_input="Collect a Discourse topic",
        candidates=[
            CandidateOutput(
                candidate_id="student",
                role_id="student",
                content="This synthetic candidate should not appear in collector dataset output.",
            )
        ],
        selected_candidate_id="student",
        evaluation={"scores": []},
        artifacts={
            "collector_dataset_records": [
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
                },
                {
                    "input": "Topic: API examples",
                    "target": {"answer": "Second collected post"},
                    "metadata": {
                        "source_provenance": {
                            "source_type": "discourse_topic_post",
                            "topic_id": 274354,
                            "post_id": 2,
                        },
                        "collector_provenance": {
                            "source_url": "https://forum.example/t/api-examples/274354/2",
                            "source_id": "topic:274354:post:2",
                            "adapter_name": "discourse_topic_adapter",
                            "tool_name": "discourse_topic_collect",
                            "policy_disposition": "allow",
                            "shard_id": "discourse-topic-274354",
                        },
                    },
                },
            ],
            "collector_record_provenance": [
                {"source_id": "fallback-record-1", "shard_id": "fallback-shard-1"},
                {"source_id": "fallback-record-2", "shard_id": "fallback-shard-2"},
            ],
        },
        events=[],
        metadata={},
    )

    records = export_run_to_dataset_records(result)

    assert len(records) == 2
    assert "candidates" not in records[0]
    assert records[0]["target"]["answer"] == "First collected post"
    assert records[1]["target"]["answer"] == "Second collected post"
    assert records[0]["metadata"]["collector_provenance"]["source_id"] == "topic:274354:post:1"
    assert records[1]["metadata"]["collector_provenance"]["source_id"] == "topic:274354:post:2"
    assert records[0]["metadata"]["source_provenance"]["source_type"] == "collector_dataset_record"
    assert records[1]["metadata"]["source_provenance"]["source_type"] == "discourse_topic_post"


@pytest.mark.asyncio
async def test_dataset_export_includes_collected_evidence_summary_when_present() -> None:
    orchestrator = MultiAgentOrchestrator()
    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Explain async in one paragraph",
            protocol={"protocol": "teacher_student_distill"},
        )
    )
    result.artifacts["evidence_collection"] = {
        "query_count": 1,
        "collected_packet_count": 2,
        "total_packet_count": 2,
    }

    records = export_run_to_dataset_records(result)

    assert records[0]["evidence"]["collected"]["query_count"] == 1
    assert records[0]["evidence"]["collected"]["collected_packet_count"] == 2


@pytest.mark.asyncio
async def test_dataset_export_marks_dr_zero_self_evolve_records() -> None:
    orchestrator = MultiAgentOrchestrator()
    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Generate and solve search tasks",
            protocol={"protocol": "dr_zero_self_evolve"},
        )
    )

    records = export_run_to_dataset_records(result)
    assert len(records) == 1
    record = records[0]

    assert record["metadata"]["dataset_mode"] == "self_evolve_search"
    assert record["metadata"]["supervision_shape"] == "solver_rollout_with_reward"
    assert record["metadata"]["dr_zero"]["synthetic_tasks"]["tasks"]
    assert record["synthetic_tasks"]["protocol"] == "dr_zero_self_evolve"
    assert record["solver_rollouts"]["rollout_count"] == 1
    assert record["reward_summary"]["status"] == "pending_verification"


def test_dataset_export_marks_controlled_subagent_execution_records() -> None:
    result = MultiAgentRunResult(
        run_id="run-controlled",
        protocol="controlled_subagent_execution",
        state="succeeded",
        task_input="Run a controlled task",
        candidates=[
            CandidateOutput(
                candidate_id="evaluator",
                role_id="evaluator",
                content="Execution summary",
            )
        ],
        selected_candidate_id="evaluator",
        evaluation={"scores": []},
        artifacts={
            "execution_plan": {"content": "plan"},
            "execution_requests": {"items": [{"request_id": "req-1"}]},
            "controller_decisions": {"items": [{"status": "approved"}]},
            "execution_results": {"items": [{"status": "completed"}]},
            "produced_artifacts": {"items": []},
            "evaluation_summary": {"status": "completed"},
            "controlled_execution_runtime": {
                "execution_boundary": "subagents_propose_controller_approves_shared_runtime_executes",
                "primary_workflow": True,
            },
        },
        events=[],
        metadata={},
    )

    records = export_run_to_dataset_records(result)
    record = records[0]

    assert record["metadata"]["dataset_mode"] == "agentic_execution"
    assert record["metadata"]["supervision_shape"] == "plan_execute_evaluate_trace"
    assert record["supervision"]["type"] == "agentic_execution_trace"
    assert record["controlled_execution_runtime"]["execution_boundary"] == (
        "subagents_propose_controller_approves_shared_runtime_executes"
    )


def test_dataset_export_uses_execution_policy_metadata_for_controlled_runs() -> None:
    result = MultiAgentRunResult(
        run_id="run-controlled-capability",
        protocol="teacher_student_distill",
        state="succeeded",
        task_input="Run a controlled capability task",
        candidates=[
            CandidateOutput(
                candidate_id="evaluator",
                role_id="evaluator",
                content="Execution summary",
            )
        ],
        selected_candidate_id="evaluator",
        evaluation={"scores": []},
        artifacts={
            "controlled_execution_runtime": {
                "execution_boundary": "subagents_propose_controller_approves_shared_runtime_executes",
                "primary_workflow": True,
            }
        },
        events=[],
        metadata={"execution_policy": {"mode": "controlled"}},
    )

    records = export_run_to_dataset_records(result)

    assert records[0]["metadata"]["dataset_mode"] == "agentic_execution"
    assert records[0]["metadata"]["supervision_shape"] == "plan_execute_evaluate_trace"
    assert records[0]["supervision"]["type"] == "agentic_execution_trace"


def test_dataset_export_keeps_debate_mode_when_controlled_execution_is_auxiliary() -> None:
    result = MultiAgentRunResult(
        run_id="run-debate-with-controlled-capability",
        protocol="multi_agent_debate",
        state="succeeded",
        task_input="Debate and use bounded execution context if needed",
        candidates=[
            CandidateOutput(
                candidate_id="debater_a",
                role_id="debater_a",
                content="Argument A",
            ),
            CandidateOutput(
                candidate_id="debater_b",
                role_id="debater_b",
                content="Argument B",
            ),
        ],
        selected_candidate_id="debater_b",
        evaluation={"scores": []},
        artifacts={
            "controlled_execution_runtime": {
                "execution_boundary": "subagents_propose_controller_approves_shared_runtime_executes",
                "primary_workflow": False,
            }
        },
        events=[],
        metadata={"execution_policy": {"mode": "controlled"}},
    )

    records = export_run_to_dataset_records(result)

    assert records[0]["metadata"]["dataset_mode"] == "preference_pair"
    assert records[0]["metadata"]["supervision_shape"] == "pairwise_preference"
    assert records[0]["supervision"]["type"] == "preference_pair"


@pytest.mark.asyncio
async def test_dataset_export_skips_records_when_research_targets_exclude_dataset_package() -> None:
    orchestrator = MultiAgentOrchestrator()
    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Explain async in one paragraph",
            protocol={"protocol": "multi_agent_debate"},
            metadata={
                "evaluation_policy": {
                    "research": {
                        "enabled": True,
                        "output_targets": ["research_brief"],
                    }
                }
            },
        )
    )

    records = export_run_to_dataset_records(result)

    assert records == []
