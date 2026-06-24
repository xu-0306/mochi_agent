from __future__ import annotations

from dataclasses import dataclass

import pytest

from mochi.learning.evaluator import OutcomeEvaluator
from mochi.learning.extractor import SkillExtractor
from mochi.learning.improver import SkillImprover
from mochi.learning.types import Skill, Trajectory, TrajectoryStep


def make_trajectory(outcome: str = "success") -> Trajectory:
    return Trajectory(
        trajectory_id="traj-1",
        task_description="Summarize a log file with exec_command",
        steps=[
            TrajectoryStep(
                step_id=1,
                timestamp=1.0,
                step_type="tool_call",
                input_data={"tool_name": "exec_command", "command": "cat app.log"},
                output_data={},
                tokens_used=0,
                duration_ms=10,
            ),
            TrajectoryStep(
                step_id=2,
                timestamp=2.0,
                step_type="final_answer",
                input_data={},
                output_data={"content": "The log file contains two warnings."},
                tokens_used=12,
                duration_ms=20,
            ),
        ],
        outcome=outcome,
    )


def make_skill() -> Skill:
    return Skill(
        skill_id="skill-1",
        name="Summarize logs",
        description="Summarize log files.",
        trigger_keywords=["log", "summary"],
        preconditions="A log file exists.",
        steps=["Open the log file.", "Summarize notable entries."],
        tools_used=["exec_command"],
        source_trajectory_id="old-traj",
        version=3,
    )


class JsonBackend:
    def __init__(self, content: str) -> None:
        self.content = content
        self.messages: list[dict[str, str]] | None = None

    async def generate(self, messages: list, tools: object = None) -> str:
        self.messages = messages
        return self.content


@dataclass
class ContentResponse:
    content: str


@dataclass
class Message:
    content: str


@dataclass
class MessageResponse:
    message: Message


class ObjectContentBackend:
    def generate(self, messages: list, tools: object = None) -> ContentResponse:
        return ContentResponse(
            content='{"name":"Object Skill","description":"From object","steps":["Do it"]}'
        )


class EmptyBackend:
    async def generate(self, messages: list, tools: object = None) -> str:
        return ""


@pytest.mark.asyncio
async def test_evaluator_returns_existing_valid_outcome() -> None:
    evaluator = OutcomeEvaluator()

    assert await evaluator.evaluate(make_trajectory("success")) == "success"


@pytest.mark.asyncio
async def test_evaluator_unknown_uses_final_answer_and_error() -> None:
    trajectory = make_trajectory("unknown")

    assert await OutcomeEvaluator().evaluate(trajectory) == "success"

    trajectory.steps[0].metadata["error"] = "tool failed once"
    assert await OutcomeEvaluator().evaluate(trajectory) == "partial"


@pytest.mark.asyncio
async def test_extract_json_success() -> None:
    backend = JsonBackend(
        """
        Here is the skill:
        {
          "skill_id": "skill-json",
          "name": "Log summarizer",
          "description": "Summarize log files with exec_command output.",
          "trigger_keywords": ["log", "summarize"],
          "preconditions": "A readable log file is available.",
          "steps": ["Read the log.", "Summarize warnings."],
          "tools_used": ["exec_command"]
        }
        """
    )

    skill = await SkillExtractor().extract(make_trajectory(), backend)

    assert skill.skill_id == "skill-json"
    assert skill.name == "Log summarizer"
    assert skill.source_trajectory_id == "traj-1"
    assert skill.version == 1
    assert backend.messages is not None


@pytest.mark.asyncio
async def test_extract_rejects_non_success() -> None:
    with pytest.raises(ValueError):
        await SkillExtractor().extract(make_trajectory("partial"), JsonBackend("{}"))


@pytest.mark.asyncio
async def test_extract_parse_object_content_and_defaults() -> None:
    skill = await SkillExtractor().extract(make_trajectory(), ObjectContentBackend())

    assert skill.name == "Object Skill"
    assert skill.description == "From object"
    assert skill.steps == ["Do it"]
    assert skill.trigger_keywords == ["summarize", "log", "file", "with", "exec"]
    assert skill.tools_used == ["exec_command"]
    assert skill.source_trajectory_id == "traj-1"


@pytest.mark.asyncio
async def test_improve_json_increments_version_and_updates_source() -> None:
    backend = JsonBackend(
        """
        {"name":"Better log summary","description":"Improved description.",
         "steps":["Open the log file.","Group warnings by type."],
         "tools_used":["exec_command","python"]}
        """
    )

    improved = await SkillImprover().improve(make_skill(), make_trajectory(), backend)

    assert improved.name == "Better log summary"
    assert improved.description == "Improved description."
    assert improved.version == 4
    assert improved.source_trajectory_id == "traj-1"
    assert improved.tools_used == ["exec_command", "python"]


@pytest.mark.asyncio
async def test_improve_fallback_merge_when_backend_returns_empty() -> None:
    improved = await SkillImprover().improve(make_skill(), make_trajectory(), EmptyBackend())

    assert improved.name == "Summarize logs"
    assert improved.version == 4
    assert improved.source_trajectory_id == "traj-1"
    assert "Observed outcome: The log file contains two warnings." in improved.description
    assert "Run tool exec_command." in improved.steps
