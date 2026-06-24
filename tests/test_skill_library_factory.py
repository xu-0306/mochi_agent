from __future__ import annotations

from pathlib import Path

from mochi.learning.skill_library_factory import resolve_skills_db_path


def test_resolve_skills_db_path_uses_configured_skills_directory(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills-store"

    resolved = resolve_skills_db_path(skills_dir=skills_dir)

    assert resolved == (skills_dir / "skills.db").resolve()


def test_resolve_skills_db_path_resolves_project_local_relative_path(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    resolved = resolve_skills_db_path(skills_dir=".mochi/skills")

    assert resolved == (tmp_path / ".mochi" / "skills" / "skills.db").resolve()
