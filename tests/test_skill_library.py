from __future__ import annotations

import json

import pytest

from mochi.learning.skill_library import SkillLibrary
from mochi.learning.types import Skill, Trajectory, TrajectoryStep


def make_skill(skill_id: str = "") -> Skill:
    return Skill(
        skill_id=skill_id,
        name="Debug Python import errors",
        description="Resolve ModuleNotFoundError in a Python project",
        trigger_keywords=["python", "import", "ModuleNotFoundError"],
        preconditions="A failing Python command is available.",
        steps=[
            "Read the traceback",
            "Check the active environment",
            "Install or fix the missing package",
        ],
        tools_used=["exec_command", "pytest"],
        source_trajectory_id="traj-1",
    )


def make_trajectory(trajectory_id: str = "traj-2") -> Trajectory:
    return Trajectory(
        trajectory_id=trajectory_id,
        task_description="Fix failing pytest import",
        steps=[
            TrajectoryStep(
                step_id=1,
                timestamp=1.0,
                step_type="tool_call",
                input_data={"cmd": "pytest"},
                output_data={"status": "failed"},
                tokens_used=0,
                duration_ms=10,
            )
        ],
        outcome="success",
    )


@pytest.mark.asyncio
async def test_add_get_list_and_generated_defaults() -> None:
    library = SkillLibrary()

    skill_id = await library.add(make_skill())

    assert skill_id
    stored = await library.get(skill_id)
    assert stored is not None
    assert stored.skill_id == skill_id
    assert stored.created_at > 0
    assert stored.updated_at > 0
    assert await library.list() == [stored]


@pytest.mark.asyncio
async def test_search_uses_skill_text_and_empty_query_returns_recent() -> None:
    library = SkillLibrary()
    older = make_skill("old")
    older.name = "Clean CSV with pandas"
    older.description = "Normalize spreadsheet columns"
    older.trigger_keywords = ["csv", "pandas"]
    older.updated_at = 10
    newer = make_skill("new")
    newer.updated_at = 20
    await library.add(older)
    await library.add(newer)

    results = await library.search("ModuleNotFoundError", top_k=1)

    assert [skill.skill_id for skill in results] == ["new"]
    assert [skill.skill_id for skill in await library.search("", top_k=1)] == ["new"]


@pytest.mark.asyncio
async def test_search_falls_back_to_like_when_fts_is_disabled() -> None:
    library = SkillLibrary()
    skill_id = await library.add(make_skill("fallback"))
    library._fts_enabled = False

    results = await library.search("pytest", top_k=3)

    assert [skill.skill_id for skill in results] == [skill_id]


@pytest.mark.asyncio
async def test_update_validates_fields_and_missing_ids() -> None:
    library = SkillLibrary()
    skill_id = await library.add(make_skill("skill-1"))

    await library.update(skill_id, {"times_used": 3, "trigger_keywords": ["debug", "python"]})
    updated = await library.get(skill_id)
    assert updated is not None
    assert updated.times_used == 3
    assert updated.trigger_keywords == ["debug", "python"]

    with pytest.raises(ValueError):
        await library.update(skill_id, {"unknown": "value"})
    with pytest.raises(KeyError):
        await library.update("missing", {"name": "Nope"})


@pytest.mark.asyncio
async def test_delete_stats_and_export() -> None:
    library = SkillLibrary()
    skill_id = await library.add(make_skill("skill-1"))

    stats = await library.get_stats()
    assert stats["total_skills"] == 1
    assert stats["total_times_used"] == 0
    assert stats["max_version"] == 1
    assert "fts_enabled" in stats

    exported = await library.export()
    assert exported[0]["skill_id"] == skill_id
    assert json.loads(await library.export_json())[0]["skill_id"] == skill_id

    assert await library.delete(skill_id) is True
    assert await library.delete(skill_id) is False
    assert await library.get(skill_id) is None
    assert (await library.get_stats())["total_skills"] == 0


@pytest.mark.asyncio
async def test_merge_is_deterministic_version_update() -> None:
    library = SkillLibrary()
    skill_id = await library.add(make_skill("skill-1"))
    before = await library.get(skill_id)
    assert before is not None

    merged = await library.merge(skill_id, make_trajectory("traj-new"))

    assert merged.version == before.version + 1
    assert merged.source_trajectory_id == "traj-new"
    assert merged.updated_at >= before.updated_at


@pytest.mark.asyncio
async def test_temp_db_persistence(tmp_path) -> None:
    db_path = tmp_path / "skills.db"
    first = SkillLibrary(db_path)
    skill_id = await first.add(make_skill("persistent"))

    second = SkillLibrary(db_path)
    stored = await second.get(skill_id)

    assert stored is not None
    assert stored.name == "Debug Python import errors"
    assert [skill.skill_id for skill in await second.search("Python", top_k=3)] == ["persistent"]


@pytest.mark.asyncio
async def test_search_uses_filesystem_skill_body() -> None:
    library = SkillLibrary()
    skill = make_skill("filesystem-skill")
    skill.name = "Filesystem Skill"
    skill.description = "Loaded from SKILL.md"
    skill.trigger_keywords = ["filesystem"]
    skill.body = "# Skill Body\n\nUnique body phrase for installer scripts."
    await library.add(skill)

    results = await library.search("unique installer scripts", top_k=3)

    assert [item.skill_id for item in results] == ["filesystem-skill"]


@pytest.mark.asyncio
async def test_db_path_parent_directory_is_created(tmp_path) -> None:
    db_path = tmp_path / "nested" / "skills" / "skills.db"

    library = SkillLibrary(db_path)
    skill_id = await library.add(make_skill("nested"))

    assert db_path.exists()
    assert (await library.get(skill_id)) is not None
