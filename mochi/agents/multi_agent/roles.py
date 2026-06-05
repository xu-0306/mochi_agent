"""Multi-agent 角色定義。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

RoleKind = Literal["teacher", "student", "debater", "judge"]


@dataclass(frozen=True)
class AgentRoleProfile:
    """單一 agent 角色設定。"""

    role_id: str
    kind: RoleKind
    title: str
    instruction: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """序列化為 JSON-safe dict。"""
        return {
            "role_id": self.role_id,
            "kind": self.kind,
            "title": self.title,
            "instruction": self.instruction,
            "metadata": dict(self.metadata),
        }


def build_teacher_student_roles(
    *,
    teacher_role_id: str = "teacher",
    student_role_id: str = "student",
) -> list[AgentRoleProfile]:
    """建立 teacher-student distill 協議的預設角色。"""
    return [
        AgentRoleProfile(
            role_id=teacher_role_id,
            kind="teacher",
            title="Teacher",
            instruction="Produce a strong reference answer with concise rationale.",
        ),
        AgentRoleProfile(
            role_id=student_role_id,
            kind="student",
            title="Student",
            instruction="Distill the teacher answer into a compact, usable response.",
        ),
    ]


def build_multi_agent_debate_roles(
    *,
    debater_a_role_id: str = "debater_a",
    debater_b_role_id: str = "debater_b",
    judge_role_id: str = "judge",
) -> list[AgentRoleProfile]:
    """建立 multi-agent debate 協議的預設角色。"""
    return [
        AgentRoleProfile(
            role_id=debater_a_role_id,
            kind="debater",
            title="Debater A",
            instruction="Argue for the strongest candidate solution first.",
        ),
        AgentRoleProfile(
            role_id=debater_b_role_id,
            kind="debater",
            title="Debater B",
            instruction="Challenge weak assumptions and provide an alternative.",
        ),
        AgentRoleProfile(
            role_id=judge_role_id,
            kind="judge",
            title="Judge",
            instruction="Pick the most robust argument based on correctness and clarity.",
        ),
    ]
