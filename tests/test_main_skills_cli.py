"""技能庫 CLI 測試。"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from typer.testing import CliRunner

from mochi.main import app

runner = CliRunner()


@dataclass
class _FakeSkill:
    skill_id: str
    name: str
    description: str = "Useful skill"
    preconditions: str = "Task matches the skill"
    steps: list[str] = field(default_factory=lambda: ["Inspect context", "Apply skill"])
    tools_used: list[str] = field(default_factory=lambda: ["rg"])
    version: int = 1
    times_used: int = 2
    success_rate: float = 0.75


class _FakeSkillLibrary:
    skills: dict[str, _FakeSkill] = {}
    last_db_path = None

    def __init__(self, db_path=None) -> None:  # noqa: ANN001
        self.__class__.last_db_path = db_path

    async def list(self) -> list[_FakeSkill]:
        return list(self.skills.values())

    async def get(self, skill_id: str) -> _FakeSkill | None:
        return self.skills.get(skill_id)

    async def delete(self, skill_id: str) -> None:
        self.skills.pop(skill_id, None)

    async def export(self) -> list[dict]:
        return [dict(skill.__dict__) for skill in self.skills.values()]


def _install_fake_library(monkeypatch, skills: list[_FakeSkill]) -> type[_FakeSkillLibrary]:
    class FakeSkillLibrary(_FakeSkillLibrary):
        pass

    FakeSkillLibrary.skills = {skill.skill_id: skill for skill in skills}
    FakeSkillLibrary.last_db_path = None
    monkeypatch.setattr("mochi.learning.skill_library.SkillLibrary", FakeSkillLibrary)
    return FakeSkillLibrary


def test_skills_list_uses_db_path_and_prints_summary(monkeypatch, tmp_path: Path) -> None:
    """skills list 應列出摘要欄位並使用 --db。"""
    fake_library = _install_fake_library(
        monkeypatch,
        [_FakeSkill(skill_id="skill-1", name="Debug Python", version=2)],
    )
    db_path = tmp_path / "skills.db"

    result = runner.invoke(app, ["skills", "list", "--db", str(db_path)])

    assert result.exit_code == 0
    assert fake_library.last_db_path == db_path
    assert "Debug Python" in result.stdout
    assert "skill-1" in result.stdout
    assert "0.75" in result.stdout


def test_skills_show_prints_details(monkeypatch, tmp_path: Path) -> None:
    """skills show 應顯示詳細欄位。"""
    _install_fake_library(
        monkeypatch,
        [_FakeSkill(skill_id="skill-1", name="Debug Python", tools_used=["rg", "pytest"])],
    )

    result = runner.invoke(app, ["skills", "show", "skill-1", "--db", str(tmp_path / "s.db")])

    assert result.exit_code == 0
    assert "Debug Python" in result.stdout
    assert "Useful skill" in result.stdout
    assert "Inspect context" in result.stdout
    assert "pytest" in result.stdout


def test_skills_delete_removes_existing_skill(monkeypatch, tmp_path: Path) -> None:
    """skills delete 應刪除既有技能。"""
    fake_library = _install_fake_library(
        monkeypatch,
        [_FakeSkill(skill_id="skill-1", name="Debug Python")],
    )

    result = runner.invoke(app, ["skills", "delete", "skill-1", "--db", str(tmp_path / "s.db")])

    assert result.exit_code == 0
    assert "Deleted skill" in result.stdout
    assert fake_library.skills == {}


def test_skills_delete_missing_skill_exits_nonzero(monkeypatch, tmp_path: Path) -> None:
    """刪除不存在技能時應清楚失敗。"""
    _install_fake_library(monkeypatch, [])

    result = runner.invoke(app, ["skills", "delete", "missing", "--db", str(tmp_path / "s.db")])

    assert result.exit_code == 1
    assert "Skill not found" in result.stdout
    assert "missing" in result.stdout


def test_skills_export_prints_json_to_stdout(monkeypatch, tmp_path: Path) -> None:
    """未指定 --output 時 export 應印出 JSON。"""
    _install_fake_library(
        monkeypatch,
        [_FakeSkill(skill_id="skill-1", name="Debug Python")],
    )

    result = runner.invoke(app, ["skills", "export", "--db", str(tmp_path / "s.db")])

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload[0]["skill_id"] == "skill-1"
    assert payload[0]["name"] == "Debug Python"


def test_skills_export_writes_json_file(monkeypatch, tmp_path: Path) -> None:
    """指定 --output 時 export 應寫入 JSON 檔。"""
    _install_fake_library(
        monkeypatch,
        [_FakeSkill(skill_id="skill-1", name="Debug Python")],
    )
    output_path = tmp_path / "export" / "skills.json"

    result = runner.invoke(
        app,
        ["skills", "export", "--db", str(tmp_path / "s.db"), "--output", str(output_path)],
    )

    assert result.exit_code == 0
    assert "Exported skill library" in result.stdout
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload[0]["skill_id"] == "skill-1"
