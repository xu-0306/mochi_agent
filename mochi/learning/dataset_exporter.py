"""Export multi-agent runs into dataset-record artifacts."""

from __future__ import annotations

from typing import Any

from mochi.agents.multi_agent.execution_policy import run_uses_controlled_execution
from mochi.agents.multi_agent.orchestrator import MultiAgentRunResult
from mochi.agents.multi_agent.research import policy_requests_dataset_output
from mochi.runtime.collector_contracts import (
    collector_dataset_records_from_result,
    collector_record_provenance_for_index,
    collector_shard_manifests_from_result,
)


def export_run_to_dataset_records(run: MultiAgentRunResult) -> list[dict[str, Any]]:
    """Convert one run into zero or more dataset records."""

    if not policy_requests_dataset_output(run.metadata):
        return []
    controlled_execution = run_uses_controlled_execution(
        run.protocol,
        metadata=run.metadata,
        artifacts=run.artifacts,
    )
    collector_records = collector_dataset_records_from_result(run.artifacts)
    if collector_records:
        records = [
            normalize_collector_dataset_record_for_run(
                record,
                run_id=run.run_id,
                protocol=run.protocol,
                state=run.state,
            )
            for record in collector_records
        ]
    else:
        record: dict[str, Any] = {
            "contract": "dataset_record",
            "schema_version": "1.0",
            "input": run.task_input,
            "candidates": [candidate.to_dict() for candidate in run.candidates],
            "metadata": {
                "dataset_mode": _dataset_mode_for_run(run.protocol, controlled_execution=controlled_execution),
                "capability_family": "multi_agent_orchestration",
                "supervision_shape": _supervision_shape_for_run(
                    run.protocol,
                    controlled_execution=controlled_execution,
                ),
                "teacher_role": "orchestrator",
                "source_provenance": {
                    "source_type": "multi_agent_run",
                    "run_id": run.run_id,
                    "protocol": run.protocol,
                },
                "run_state": run.state,
            },
        }

        selected = _selected_candidate(run)
        if selected is not None:
            record["target"] = {
                "candidate_id": selected["candidate_id"],
                "role_id": selected["role_id"],
                "answer": selected["content"],
            }

        record["supervision"] = _build_supervision_payload(run)

        if run.evaluation:
            record["evidence"] = {
                "evaluation": dict(run.evaluation),
                "events": [event.to_dict() for event in run.events],
            }
            if isinstance(run.artifacts.get("evidence_collection"), dict):
                record["evidence"]["collected"] = dict(run.artifacts["evidence_collection"])
            if isinstance(run.artifacts.get("verification"), dict):
                record["evidence"]["verification"] = dict(run.artifacts["verification"])
            if isinstance(run.artifacts.get("claim_evidence_map"), dict):
                record["evidence"]["claim_evidence_map"] = dict(run.artifacts["claim_evidence_map"])
            if isinstance(run.artifacts.get("source_quality_table"), dict):
                record["evidence"]["source_quality_table"] = dict(run.artifacts["source_quality_table"])

        if run.protocol == "dr_zero_self_evolve":
            dr_zero_metadata = {
                "synthetic_tasks": run.artifacts.get("synthetic_tasks"),
                "solver_rollouts": run.artifacts.get("solver_rollouts"),
                "reward_summary": run.artifacts.get("reward_summary"),
                "curriculum_state": run.artifacts.get("curriculum_state"),
                "iteration_summary": run.artifacts.get("drzero_iteration_summary"),
            }
            record["metadata"]["dr_zero"] = {
                key: value for key, value in dr_zero_metadata.items() if isinstance(value, dict)
            }
            if isinstance(run.artifacts.get("synthetic_tasks"), dict):
                record["synthetic_tasks"] = dict(run.artifacts["synthetic_tasks"])
            if isinstance(run.artifacts.get("solver_rollouts"), dict):
                record["solver_rollouts"] = dict(run.artifacts["solver_rollouts"])
            if isinstance(run.artifacts.get("reward_summary"), dict):
                record["reward_summary"] = dict(run.artifacts["reward_summary"])

        if controlled_execution:
            controlled_metadata = {
                "execution_plan": run.artifacts.get("execution_plan"),
                "execution_requests": run.artifacts.get("execution_requests"),
                "controller_decisions": run.artifacts.get("controller_decisions"),
                "execution_results": run.artifacts.get("execution_results"),
                "produced_artifacts": run.artifacts.get("produced_artifacts"),
                "evaluation_summary": run.artifacts.get("evaluation_summary"),
                "runtime": run.artifacts.get("controlled_execution_runtime"),
            }
            record["metadata"]["controlled_execution"] = {
                key: value for key, value in controlled_metadata.items() if isinstance(value, dict)
            }
            for key in (
                "execution_plan",
                "execution_requests",
                "controller_decisions",
                "execution_results",
                "produced_artifacts",
                "evaluation_summary",
                "controlled_execution_runtime",
            ):
                if isinstance(run.artifacts.get(key), dict):
                    record[key] = dict(run.artifacts[key])

        records = [record]

    collector_shard_manifests = collector_shard_manifests_from_result(run.artifacts)
    for index, dataset_record in enumerate(records):
        collector_provenance = collector_record_provenance_for_index(
            run.artifacts,
            index=index,
            shard_manifests=collector_shard_manifests,
        )
        if collector_provenance is not None:
            dataset_record.setdefault("metadata", {})
            dataset_record["metadata"].setdefault("collector_provenance", collector_provenance)

    return records


def normalize_collector_dataset_record_for_run(
    record: dict[str, Any],
    *,
    run_id: str,
    protocol: str,
    state: str,
) -> dict[str, Any]:
    metadata = dict(record.get("metadata") or {})
    source_provenance = (
        dict(metadata.get("source_provenance"))
        if isinstance(metadata.get("source_provenance"), dict)
        else {}
    )
    if not source_provenance:
        source_provenance = {
            "source_type": "collector_dataset_record",
            "run_id": run_id,
            "protocol": protocol,
        }
    metadata["source_provenance"] = source_provenance
    metadata.setdefault("dataset_mode", "source_capture")
    metadata.setdefault("capability_family", "dataset_collection")
    metadata.setdefault("supervision_shape", "source_capture")
    metadata.setdefault("run_state", state)

    normalized = dict(record)
    normalized["contract"] = "dataset_record"
    normalized["schema_version"] = str(record.get("schema_version") or "1.0")
    normalized["metadata"] = metadata
    normalized.setdefault("input", "")
    normalized.setdefault(
        "supervision",
        {
            "type": "source_capture",
            "loss_mask": "target_only",
        },
    )
    return normalized


def _dataset_mode_for_run(protocol: str, *, controlled_execution: bool) -> str:
    if protocol == "dr_zero_self_evolve":
        return "self_evolve_search"
    if controlled_execution:
        return "agentic_execution"
    if protocol == "multi_agent_debate":
        return "preference_pair"
    return "trajectory_distillation"


def _supervision_shape_for_run(protocol: str, *, controlled_execution: bool) -> str:
    if protocol == "dr_zero_self_evolve":
        return "solver_rollout_with_reward"
    if controlled_execution:
        return "plan_execute_evaluate_trace"
    if protocol == "multi_agent_debate":
        return "pairwise_preference"
    return "single_target"


def _build_supervision_payload(run: MultiAgentRunResult) -> dict[str, Any]:
    selected_id = run.selected_candidate_id
    controlled_execution = run_uses_controlled_execution(
        run.protocol,
        metadata=run.metadata,
        artifacts=run.artifacts,
    )
    if run.protocol == "multi_agent_debate":
        rejected_id = None
        for candidate in run.candidates:
            if candidate.candidate_id != selected_id:
                rejected_id = candidate.candidate_id
                break
        return {
            "type": "preference_pair",
            "chosen_candidate_id": selected_id,
            "rejected_candidate_id": rejected_id,
            "comparison_basis": "llm_first_policy",
        }
    if controlled_execution:
        return {
            "type": "agentic_execution_trace",
            "selected_candidate_id": selected_id,
            "execution_result_shape": "plan_execute_evaluate",
        }
    return {
        "type": "sft",
        "loss_mask": "target_only",
        "selected_candidate_id": selected_id,
    }


def _selected_candidate(run: MultiAgentRunResult) -> dict[str, Any] | None:
    if run.selected_candidate_id is None:
        return None
    for candidate in run.candidates:
        if candidate.candidate_id == run.selected_candidate_id:
            return candidate.to_dict()
    return None
