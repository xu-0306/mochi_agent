"""學習系統共用型別（Trajectory、Skill）。"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, cast

StepType = Literal["llm_call", "tool_call", "tool_result", "final_answer"]
TrajectoryOutcome = Literal["success", "failure", "partial", "unknown"]


@dataclass
class TrajectoryStep:
    """單一推理/執行步驟的記錄。"""

    step_id: int
    timestamp: float
    step_type: StepType
    input_data: dict
    output_data: dict
    tokens_used: int
    duration_ms: float
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """轉換為可 JSON 序列化的 dict。"""
        return {
            "step_id": self.step_id,
            "timestamp": self.timestamp,
            "step_type": self.step_type,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "tokens_used": self.tokens_used,
            "duration_ms": self.duration_ms,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrajectoryStep:
        """從 dict 還原步驟記錄。"""
        return cls(
            step_id=int(data["step_id"]),
            timestamp=float(data["timestamp"]),
            step_type=cast(StepType, data["step_type"]),
            input_data=dict(data["input_data"]),
            output_data=dict(data["output_data"]),
            tokens_used=int(data["tokens_used"]),
            duration_ms=float(data["duration_ms"]),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class Trajectory:
    """完整任務執行軌跡。"""

    trajectory_id: str
    task_description: str
    steps: list[TrajectoryStep]
    outcome: TrajectoryOutcome
    user_feedback: str | None = None
    total_tokens: int = 0
    total_duration_ms: float = 0

    def to_dict(self) -> dict[str, Any]:
        """轉換為可 JSON 序列化的 dict。"""
        return {
            "trajectory_id": self.trajectory_id,
            "task_description": self.task_description,
            "steps": [step.to_dict() for step in self.steps],
            "outcome": self.outcome,
            "user_feedback": self.user_feedback,
            "total_tokens": self.total_tokens,
            "total_duration_ms": self.total_duration_ms,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Trajectory:
        """從 dict 還原完整軌跡。"""
        return cls(
            trajectory_id=str(data["trajectory_id"]),
            task_description=str(data["task_description"]),
            steps=[TrajectoryStep.from_dict(step) for step in data.get("steps", [])],
            outcome=cast(TrajectoryOutcome, data["outcome"]),
            user_feedback=data.get("user_feedback"),
            total_tokens=int(data.get("total_tokens", 0)),
            total_duration_ms=float(data.get("total_duration_ms", 0)),
        )


@dataclass
class Skill:
    """可重用技能定義。"""

    skill_id: str
    name: str
    description: str
    trigger_keywords: list[str]
    preconditions: str
    steps: list[str]
    tools_used: list[str]
    source_trajectory_id: str
    times_used: int = 0
    success_rate: float = 1.0
    created_at: float = 0
    updated_at: float = 0
    version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """轉換為可 JSON 序列化的 dict。"""
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "trigger_keywords": self.trigger_keywords,
            "preconditions": self.preconditions,
            "steps": self.steps,
            "tools_used": self.tools_used,
            "source_trajectory_id": self.source_trajectory_id,
            "times_used": self.times_used,
            "success_rate": self.success_rate,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Skill:
        """從 dict 還原技能定義。"""
        return cls(
            skill_id=str(data["skill_id"]),
            name=str(data["name"]),
            description=str(data["description"]),
            trigger_keywords=list(data.get("trigger_keywords", [])),
            preconditions=str(data["preconditions"]),
            steps=list(data.get("steps", [])),
            tools_used=list(data.get("tools_used", [])),
            source_trajectory_id=str(data["source_trajectory_id"]),
            times_used=int(data.get("times_used", 0)),
            success_rate=float(data.get("success_rate", 1.0)),
            created_at=float(data.get("created_at", 0)),
            updated_at=float(data.get("updated_at", 0)),
            version=int(data.get("version", 1)),
        )
