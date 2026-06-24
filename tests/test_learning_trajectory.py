from __future__ import annotations

import json

import pytest

from mochi.learning.trajectory import TrajectoryLogger
from mochi.learning.types import Skill, Trajectory, TrajectoryStep


def make_step(step_id: int = 1) -> TrajectoryStep:
    return TrajectoryStep(
        step_id=step_id,
        timestamp=123.45 + step_id,
        step_type="tool_call",
        input_data={"tool": "search", "query": "mochi"},
        output_data={"status": "ok"},
        tokens_used=10 * step_id,
        duration_ms=25.5 * step_id,
        metadata={"phase": "test"},
    )


def test_start_log_finish_export() -> None:
    logger = TrajectoryLogger()
    trajectory_id = logger.start("測試 trajectory logger")

    logger.log_step(trajectory_id, make_step(1))
    logger.log_step(trajectory_id, make_step(2))
    logger.finish(trajectory_id, "success", feedback="works")

    trajectory = logger.export(trajectory_id)

    assert trajectory.trajectory_id == trajectory_id
    assert trajectory.task_description == "測試 trajectory logger"
    assert trajectory.outcome == "success"
    assert trajectory.user_feedback == "works"
    assert len(trajectory.steps) == 2
    assert trajectory.total_tokens == 30
    assert trajectory.total_duration_ms == 76.5


def test_jsonl_persistence_and_load(tmp_path) -> None:
    storage_path = tmp_path / "trajectories.jsonl"
    logger = TrajectoryLogger(storage_path)
    trajectory_id = logger.start("persist me")

    logger.log_step(trajectory_id, make_step())
    logger.finish(trajectory_id, "partial")

    lines = storage_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["trajectory_id"] == trajectory_id

    loaded = TrajectoryLogger(storage_path).load(trajectory_id)
    assert loaded.trajectory_id == trajectory_id
    assert loaded.outcome == "partial"
    assert loaded.total_tokens == 10
    assert loaded.steps[0].metadata == {"phase": "test"}


def test_from_dict_roundtrip() -> None:
    trajectory = Trajectory(
        trajectory_id="traj-1",
        task_description="roundtrip",
        steps=[make_step()],
        outcome="unknown",
        user_feedback=None,
        total_tokens=10,
        total_duration_ms=25.5,
    )
    skill = Skill(
        skill_id="skill-1",
        name="測試技能",
        description="roundtrip skill",
        trigger_keywords=["test"],
        preconditions="有測試資料",
        steps=["run test"],
        tools_used=["pytest"],
        source_trajectory_id="traj-1",
        created_at=1.0,
        updated_at=2.0,
    )

    assert Trajectory.from_dict(trajectory.to_dict()) == trajectory
    assert Skill.from_dict(skill.to_dict()) == skill
    assert TrajectoryStep.from_dict(make_step().to_dict()) == make_step()


def test_unknown_trajectory_id_errors() -> None:
    logger = TrajectoryLogger()

    with pytest.raises(KeyError, match="missing"):
        logger.log_step("missing", make_step())

    with pytest.raises(KeyError, match="missing"):
        logger.export("missing")

    with pytest.raises(KeyError, match="missing"):
        logger.finish("missing", "success")


def test_invalid_outcome_errors() -> None:
    logger = TrajectoryLogger()
    trajectory_id = logger.start("bad outcome")

    with pytest.raises(ValueError, match="Invalid trajectory outcome"):
        logger.finish(trajectory_id, "done")
