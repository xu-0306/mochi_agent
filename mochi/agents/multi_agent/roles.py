"""Multi-agent 角色定義。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

RoleKind = Literal[
    "teacher",
    "student",
    "debater",
    "judge",
    "proposer",
    "solver",
    "verifier",
    "planner",
    "executor",
    "controller",
    "evaluator",
]


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


def build_dr_zero_roles(
    *,
    proposer_role_id: str = "proposer",
    solver_role_id: str = "solver",
    verifier_role_id: str = "verifier",
) -> list[AgentRoleProfile]:
    """Build default roles for the Dr.Zero self-evolve protocol."""
    return [
        AgentRoleProfile(
            role_id=proposer_role_id,
            kind="proposer",
            title="Proposer",
            instruction=(
                "Generate diverse, hard-but-solvable synthetic tasks for a search-capable "
                "solver. Return JSON when possible."
            ),
        ),
        AgentRoleProfile(
            role_id=solver_role_id,
            kind="solver",
            title="Solver",
            instruction=(
                "Solve the proposed task using available read and research tools. "
                "Ground the answer in verifiable evidence."
            ),
        ),
        AgentRoleProfile(
            role_id=verifier_role_id,
            kind="verifier",
            title="Verifier",
            instruction="Verify solver rollouts and reward correctness, difficulty, and solvability.",
        ),
    ]


def build_controlled_execution_roles(
    *,
    planner_role_id: str = "planner",
    executor_role_id: str = "executor",
    controller_role_id: str = "controller",
    evaluator_role_id: str = "evaluator",
) -> list[AgentRoleProfile]:
    """Build roles for controller-gated subagent execution."""
    return [
        AgentRoleProfile(
            role_id=planner_role_id,
            kind="planner",
            title="Planner",
            instruction="Create a concise executable plan with expected artifacts and success criteria.",
        ),
        AgentRoleProfile(
            role_id=executor_role_id,
            kind="executor",
            title="Executor",
            instruction=(
                "Propose structured execution requests for the plan. Do not claim to run commands; "
                "return JSON requests for controller review."
            ),
        ),
        AgentRoleProfile(
            role_id=controller_role_id,
            kind="controller",
            title="Controller",
            instruction=(
                "Review execution requests for safety and task fit. Approve, reject, or rewrite "
                "commands before shared runtime execution."
            ),
        ),
        AgentRoleProfile(
            role_id=evaluator_role_id,
            kind="evaluator",
            title="Evaluator",
            instruction="Summarize execution results, produced artifacts, metrics, and next steps.",
        ),
    ]
