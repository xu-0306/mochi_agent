"""Filesystem SKILL.md loader tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from mochi.learning.skill_library import SkillLibrary
from mochi.learning.skill_loader import SkillLoader, parse_skill_file


def write_skill(path: Path, *, description: str = "Install local skills.") -> Path:
    skill_dir = path / "skill-installer"
    skill_dir.mkdir(parents=True)
    skill_path = skill_dir / "SKILL.md"
    skill_path.write_text(
        f"""---
name: skill-installer
description: {description}
tags: [skills, install]
---

# Skill Installer

Use scripts in this directory to install skills.
""",
        encoding="utf-8",
    )
    return skill_path


def test_parse_skill_file_reads_frontmatter_and_body(tmp_path: Path) -> None:
    skill_path = write_skill(tmp_path)

    skill = parse_skill_file(skill_path, source_root=tmp_path)

    assert skill.skill_id == "skill-installer"
    assert skill.name == "skill-installer"
    assert skill.description == "Install local skills."
    assert skill.source_type == "filesystem"
    assert skill.source_path == str(skill_path)
    assert skill.content_hash
    assert skill.metadata["tags"] == ["skills", "install"]
    assert "# Skill Installer" in skill.body
    assert "skills" in skill.trigger_keywords


@pytest.mark.asyncio
async def test_skill_loader_syncs_add_update_and_delete(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills"
    db_path = tmp_path / "skills.db"
    skill_path = write_skill(skills_dir)
    library = SkillLibrary(db_path)
    loader = SkillLoader.from_paths(skills_dir, system_skills_dir=None)

    first = await loader.sync(library)

    assert first.scanned == 1
    assert first.added == 1
    assert [skill.name for skill in await library.search("install skills")] == ["skill-installer"]

    skill_path.write_text(
        skill_path.read_text(encoding="utf-8").replace(
            "Install local skills.",
            "Install updated local skills.",
        ),
        encoding="utf-8",
    )
    second = await loader.sync(library)
    updated = await library.get("skill-installer")

    assert second.updated == 1
    assert updated is not None
    assert updated.description == "Install updated local skills."

    skill_path.unlink()
    third = await loader.sync(library)

    assert third.removed == 1
    assert await library.get("skill-installer") is None


@pytest.mark.asyncio
async def test_skill_loader_skips_system_subdir_when_scanning_user_skills(tmp_path: Path) -> None:
    """掃描 user skills root 時不應把 `.system` 當 filesystem skill 重複索引。"""
    skills_dir = tmp_path / "skills"
    write_skill(skills_dir)
    system_skill_path = write_skill(skills_dir / ".system", description="Install system skills.")

    library = SkillLibrary(tmp_path / "skills.db")
    loader = SkillLoader.from_paths(skills_dir, system_skills_dir=skills_dir / ".system")

    result = await loader.sync(library)

    assert result.scanned == 2
    assert await library.get("skill-installer") is not None
    assert await library.get("system:skill-installer") is not None
    filesystem_skill = await library.get("skill-installer")
    system_skill = await library.get("system:skill-installer")
    assert filesystem_skill is not None
    assert system_skill is not None
    assert filesystem_skill.source_path != str(system_skill_path)
    assert system_skill.source_path == str(system_skill_path)
