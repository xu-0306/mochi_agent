"""Multi-agent protocol 設定與解析。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

ProtocolName = Literal["teacher_student_distill", "multi_agent_debate"]


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


ProtocolConfig = TeacherStudentDistillProtocol | MultiAgentDebateProtocol


def parse_protocol_config(payload: Mapping[str, Any] | ProtocolConfig | None) -> ProtocolConfig:
    """將 API-style payload 正規化為協議 dataclass。"""
    if payload is None:
        return TeacherStudentDistillProtocol()
    if isinstance(payload, (TeacherStudentDistillProtocol, MultiAgentDebateProtocol)):
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
    raise ValueError(f"Unsupported multi-agent protocol: {raw_protocol}")


def _bounded_rounds(value: Any, *, default: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, 1), 8)


def _clean_role_id(value: Any, *, default: str) -> str:
    text = str(value or "").strip()
    return text or default
