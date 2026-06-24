from __future__ import annotations

from mochi.agents.multi_agent.execution_policy import (
    LEGACY_CONTROLLED_SUBAGENT_PROTOCOL,
    parse_subagent_execution_policy,
)
from mochi.agents.multi_agent.protocols import (
    ControlledSubagentExecutionProtocol,
    DrZeroSelfEvolveProtocol,
    MultiAgentDebateProtocol,
    TeacherStudentDistillProtocol,
    parse_protocol_config,
)
from mochi.agents.multi_agent.research import ResearchDebatePolicy


def test_parse_teacher_student_protocol_defaults() -> None:
    resolved = parse_protocol_config({"protocol": "teacher_student_distill"})

    assert isinstance(resolved, TeacherStudentDistillProtocol)
    assert resolved.rounds == 1
    assert resolved.teacher_role_id == "teacher"
    assert resolved.student_role_id == "student"


def test_parse_multi_agent_debate_protocol_defaults() -> None:
    resolved = parse_protocol_config({"protocol": "multi_agent_debate"})

    assert isinstance(resolved, MultiAgentDebateProtocol)
    assert resolved.rounds == 2
    assert resolved.debater_a_role_id == "debater_a"
    assert resolved.debater_b_role_id == "debater_b"
    assert resolved.judge_role_id == "judge"


def test_parse_dr_zero_self_evolve_protocol_defaults() -> None:
    resolved = parse_protocol_config({"protocol": "dr_zero_self_evolve"})

    assert isinstance(resolved, DrZeroSelfEvolveProtocol)
    assert resolved.iterations == 1
    assert resolved.proposal_sample_size == 3
    assert resolved.solver_rollouts_per_task == 1
    assert resolved.proposer_role_id == "proposer"
    assert resolved.solver_role_id == "solver"
    assert resolved.verifier_role_id == "verifier"


def test_parse_dr_zero_self_evolve_protocol_custom_values_are_bounded() -> None:
    resolved = parse_protocol_config(
        {
            "protocol": "dr_zero_self_evolve",
            "iterations": 99,
            "proposal_sample_size": 2,
            "solver_rollouts_per_task": 2,
            "proposer_role_id": "challenger",
            "solver_role_id": "answerer",
            "verifier_role_id": "rewarder",
        }
    )

    assert isinstance(resolved, DrZeroSelfEvolveProtocol)
    assert resolved.iterations == 8
    assert resolved.proposal_sample_size == 2
    assert resolved.solver_rollouts_per_task == 2
    assert resolved.proposer_role_id == "challenger"
    assert resolved.solver_role_id == "answerer"
    assert resolved.verifier_role_id == "rewarder"


def test_parse_controlled_subagent_execution_protocol_defaults() -> None:
    resolved = parse_protocol_config({"protocol": "controlled_subagent_execution"})

    assert isinstance(resolved, ControlledSubagentExecutionProtocol)
    assert resolved.protocol == "controlled_subagent_execution"
    assert resolved.guidance_required is False


def test_parse_controlled_subagent_execution_protocol_is_only_a_legacy_marker() -> None:
    resolved = parse_protocol_config(
        {
            "protocol": "controlled_subagent_execution",
            "max_execution_requests": 99,
            "max_commands_per_request": 4,
            "default_timeout_sec": 999999,
            "background_allowed": False,
            "planner_role_id": "architect",
            "executor_role_id": "local_worker",
            "controller_role_id": "main_controller",
            "evaluator_role_id": "judge",
            "guidance_required": True,
        }
    )

    assert isinstance(resolved, ControlledSubagentExecutionProtocol)
    assert resolved.protocol == "controlled_subagent_execution"
    assert resolved.guidance_required is True


def test_parse_subagent_execution_policy_uses_legacy_protocol_defaults() -> None:
    resolved = parse_subagent_execution_policy(None, legacy_protocol=LEGACY_CONTROLLED_SUBAGENT_PROTOCOL)

    assert resolved.mode == "controlled"
    assert resolved.max_execution_requests == 5
    assert resolved.default_timeout_sec == 300
    assert resolved.allowed_roles == ("planner", "executor", "controller", "evaluator")


def test_parse_subagent_execution_policy_bounds_and_serializes_roles() -> None:
    resolved = parse_subagent_execution_policy(
        {
            "mode": "controlled",
            "allowed_roles": ["executor", "controller", "", "executor"],
            "max_execution_requests": 999,
            "max_commands_per_request": 4,
            "default_timeout_sec": 999999,
            "background_allowed": False,
            "planner_role_id": "architect",
            "executor_role_id": "runner",
            "controller_role_id": "reviewer",
            "evaluator_role_id": "reporter",
        }
    )

    assert resolved.mode == "controlled"
    assert resolved.allowed_roles == ("executor", "controller")
    assert resolved.max_execution_requests == 20
    assert resolved.max_commands_per_request == 4
    assert resolved.default_timeout_sec == 86_400
    assert resolved.background_allowed is False
    assert resolved.planner_role_id == "architect"
    assert resolved.executor_role_id == "runner"
    assert resolved.controller_role_id == "reviewer"
    assert resolved.evaluator_role_id == "reporter"


def test_parse_research_debate_policy_defaults() -> None:
    policy = ResearchDebatePolicy.from_metadata(
        {
            "evaluation_policy": {
                "research": {
                    "enabled": True,
                    "preset": "smart_judge_research_debate",
                    "output_targets": ["research_brief"],
                    "source_mode": "web_first",
                    "citation_policy": "claim_level_required",
                    "local_worker_count": 4,
                    "local_worker_count_max": 6,
                    "max_research_queries": 9,
                    "max_sources_per_query": 5,
                    "debate_rounds": 3,
                }
            }
        }
    )

    assert policy.enabled is True
    assert policy.output_targets == ("research_brief",)
    assert policy.evidence_collection_mode == "web"
    assert policy.local_worker_count == 4
    assert policy.max_research_queries == 9
    assert policy.debate_rounds == 3
