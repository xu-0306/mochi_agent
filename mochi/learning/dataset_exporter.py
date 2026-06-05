"""Export multi-agent runs into dataset-record artifacts."""

from __future__ import annotations

from typing import Any

from mochi.agents.multi_agent.orchestrator import MultiAgentRunResult
from mochi.agents.multi_agent.research import policy_requests_dataset_output


def export_run_to_dataset_records(run: MultiAgentRunResult) -> list[dict[str, Any]]:
    """Convert one run into zero or more dataset records."""

    if not policy_requests_dataset_output(run.metadata):
        return []

    record: dict[str, Any] = {
        "contract": "dataset_record",
        "schema_version": "1.0",
        "input": run.task_input,
        "candidates": [candidate.to_dict() for candidate in run.candidates],
        "metadata": {
            "dataset_mode": _dataset_mode_for_protocol(run.protocol),
            "capability_family": "multi_agent_orchestration",
            "supervision_shape": _supervision_shape_for_protocol(run.protocol),
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

    return [record]


def _dataset_mode_for_protocol(protocol: str) -> str:
    if protocol == "dr_zero_self_evolve":
        return "self_evolve_search"
    if protocol == "multi_agent_debate":
        return "preference_pair"
    return "trajectory_distillation"


def _supervision_shape_for_protocol(protocol: str) -> str:
    if protocol == "dr_zero_self_evolve":
        return "solver_rollout_with_reward"
    if protocol == "multi_agent_debate":
        return "pairwise_preference"
    return "single_target"


def _build_supervision_payload(run: MultiAgentRunResult) -> dict[str, Any]:
    selected_id = run.selected_candidate_id
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
