"""Multi-agent protocol 設定與解析。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

ProtocolName = Literal[
    "teacher_student_distill",
    "multi_agent_debate",
    "dr_zero_self_evolve",
    "controlled_subagent_execution",
]


@dataclass(frozen=True)
class TeacherStudentDistillProtocol:
    """Teacher-Student distillation 協議設定。"""

    protocol: Literal["teacher_student_distill"] = "teacher_student_distill"
    rounds: int = 1
    teacher_role_id: str = "teacher"
    student_role_id: str = "student"
    guidance_required: bool = False


@dataclass(frozen=True)
class MultiAgentDebateProtocol:
    """Multi-agent debate 協議設定。"""

    protocol: Literal["multi_agent_debate"] = "multi_agent_debate"
    rounds: int = 2
    debater_a_role_id: str = "debater_a"
    debater_b_role_id: str = "debater_b"
    judge_role_id: str = "judge"
    guidance_required: bool = False


@dataclass(frozen=True)
class DrZeroSelfEvolveProtocol:
    """Dr.Zero-style self-evolution protocol setting."""

    protocol: Literal["dr_zero_self_evolve"] = "dr_zero_self_evolve"
    iterations: int = 1
    proposal_sample_size: int = 3
    solver_rollouts_per_task: int = 1
    proposer_role_id: str = "proposer"
    solver_role_id: str = "solver"
    verifier_role_id: str = "verifier"
    guidance_required: bool = False


@dataclass(frozen=True)
class ControlledSubagentExecutionProtocol:
    """Legacy protocol marker kept only for backward compatibility."""

    protocol: Literal["controlled_subagent_execution"] = "controlled_subagent_execution"
    guidance_required: bool = False


ProtocolConfig = (
    TeacherStudentDistillProtocol
    | MultiAgentDebateProtocol
    | DrZeroSelfEvolveProtocol
    | ControlledSubagentExecutionProtocol
)


def parse_protocol_config(payload: Mapping[str, Any] | ProtocolConfig | None) -> ProtocolConfig:
    """將 API-style payload 正規化為協議 dataclass。"""
    if payload is None:
        return TeacherStudentDistillProtocol()
    if isinstance(
        payload,
        (
            TeacherStudentDistillProtocol,
            MultiAgentDebateProtocol,
            DrZeroSelfEvolveProtocol,
            ControlledSubagentExecutionProtocol,
        ),
    ):
        return payload

    raw_protocol = str(payload.get("protocol", "teacher_student_distill")).strip()
    if raw_protocol == "teacher_student_distill":
        return TeacherStudentDistillProtocol(
            rounds=_bounded_rounds(payload.get("rounds"), default=1),
            teacher_role_id=_clean_role_id(payload.get("teacher_role_id"), default="teacher"),
            student_role_id=_clean_role_id(payload.get("student_role_id"), default="student"),
            guidance_required=bool(payload.get("guidance_required", False)),
        )
    if raw_protocol == "multi_agent_debate":
        return MultiAgentDebateProtocol(
            rounds=_bounded_rounds(payload.get("rounds"), default=2),
            debater_a_role_id=_clean_role_id(payload.get("debater_a_role_id"), default="debater_a"),
            debater_b_role_id=_clean_role_id(payload.get("debater_b_role_id"), default="debater_b"),
            judge_role_id=_clean_role_id(payload.get("judge_role_id"), default="judge"),
            guidance_required=bool(payload.get("guidance_required", False)),
        )
    if raw_protocol == "dr_zero_self_evolve":
        return DrZeroSelfEvolveProtocol(
            iterations=_bounded_int(payload.get("iterations"), default=1, minimum=1, maximum=8),
            proposal_sample_size=_bounded_int(
                payload.get("proposal_sample_size"),
                default=3,
                minimum=1,
                maximum=20,
            ),
            solver_rollouts_per_task=_bounded_int(
                payload.get("solver_rollouts_per_task"),
                default=1,
                minimum=1,
                maximum=8,
            ),
            proposer_role_id=_clean_role_id(payload.get("proposer_role_id"), default="proposer"),
            solver_role_id=_clean_role_id(payload.get("solver_role_id"), default="solver"),
            verifier_role_id=_clean_role_id(payload.get("verifier_role_id"), default="verifier"),
            guidance_required=bool(payload.get("guidance_required", False)),
        )
    if raw_protocol == "controlled_subagent_execution":
        return ControlledSubagentExecutionProtocol(
            guidance_required=bool(payload.get("guidance_required", False)),
        )
    raise ValueError(f"Unsupported multi-agent protocol: {raw_protocol}")


def _bounded_rounds(value: Any, *, default: int) -> int:
    return _bounded_int(value, default=default, minimum=1, maximum=8)


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, minimum), maximum)


def _clean_role_id(value: Any, *, default: str) -> str:
    text = str(value or "").strip()
    return text or default
