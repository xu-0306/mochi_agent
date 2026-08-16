"""Skills API routes tests."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi.testclient import TestClient

from mochi.api.server import create_app
from mochi.config.schema import MochiConfig
from mochi.learning.skill_library import SkillLibrary
from mochi.learning.types import Skill


def make_skill(
    *,
    skill_id: str,
    name: str,
    description: str,
    trigger_keywords: list[str],
    updated_at: float,
) -> Skill:
    return Skill(
        skill_id=skill_id,
        name=name,
        description=description,
        trigger_keywords=trigger_keywords,
        preconditions="A reproducible task is available.",
        steps=["Inspect inputs", "Apply fix", "Verify result"],
        tools_used=["exec_command", "pytest"],
        source_trajectory_id=f"traj-{skill_id}",
        updated_at=updated_at,
    )


def _create_app_with_skills_router():
    return create_app()


def _add_skills(library: SkillLibrary, *skills: Skill) -> None:
    for skill in skills:
        asyncio.run(library.add(skill))


class _FakeSkillLibrary:
    def __init__(self, skills: list[Skill]) -> None:
        self.skills = {skill.skill_id: skill for skill in skills}
        self.calls: list[tuple[str, Any]] = []

    async def list(self, limit: int | None = None) -> list[Skill]:
        self.calls.append(("list", limit))
        return list(self.skills.values())[:limit]

    async def search(self, query: str, top_k: int = 3) -> list[Skill]:
        self.calls.append(("search", query, top_k))
        return [
            skill for skill in self.skills.values()
            if query in skill.description or query in skill.name
        ][:top_k]

    async def get(self, skill_id: str) -> Skill | None:
        self.calls.append(("get", skill_id))
        return self.skills.get(skill_id)

    async def delete(self, skill_id: str) -> bool:
        self.calls.append(("delete", skill_id))
        return self.skills.pop(skill_id, None) is not None

    async def export(self) -> list[dict[str, Any]]:
        self.calls.append(("export", None))
        return [skill.to_dict() for skill in self.skills.values()]


class _PromotionConflictLibrary:
    async def promote(self, skill_id: str, version: int, evaluation_evidence: dict[str, Any]) -> Skill:
        raise ValueError("skill version conflict")


def test_skills_routes_prefer_app_state_skill_library() -> None:
    """skills routes 應優先使用 app.state.skill_library。"""
    app = _create_app_with_skills_router()
    first = make_skill(
        skill_id="skill-debug",
        name="Debug import failures",
        description="Resolve ModuleNotFoundError from pytest runs",
        trigger_keywords=["python", "pytest", "import"],
        updated_at=10,
    )
    second = make_skill(
        skill_id="skill-csv",
        name="Clean CSV columns",
        description="Normalize malformed spreadsheet headers",
        trigger_keywords=["csv", "pandas"],
        updated_at=20,
    )
    fake_library = _FakeSkillLibrary([second, first])
    app.state.skill_library = fake_library
    app.state.config_factory = lambda: (_ for _ in ()).throw(AssertionError("config should not be used"))

    with TestClient(app) as client:
        list_response = client.get("/v1/skills")
        search_response = client.get("/v1/skills", params={"q": "ModuleNotFoundError", "limit": 1})
        get_response = client.get("/v1/skills/skill-debug")
        export_response = client.get("/v1/skills/export")

    assert list_response.status_code == 200
    assert [item["skill_id"] for item in list_response.json()] == ["skill-csv", "skill-debug"]

    assert search_response.status_code == 200
    assert [item["skill_id"] for item in search_response.json()] == ["skill-debug"]

    assert get_response.status_code == 200
    assert get_response.json()["skill_id"] == "skill-debug"
    assert set(get_response.json()) == {
        "skill_id",
        "name",
        "description",
        "trigger_keywords",
        "preconditions",
        "steps",
        "tools_used",
        "source_trajectory_id",
        "times_used",
        "success_rate",
        "created_at",
        "updated_at",
        "version",
        "source_type",
        "source_path",
        "content_hash",
        "body",
        "metadata",
    }

    assert export_response.status_code == 200
    assert [item["skill_id"] for item in export_response.json()] == ["skill-csv", "skill-debug"]
    assert fake_library.calls == [
        ("list", 50),
        ("search", "ModuleNotFoundError", 1),
        ("get", "skill-debug"),
        ("export", None),
    ]


def test_skills_routes_with_real_library_cover_crud_search_export_and_404(tmp_path: Path) -> None:
    """使用 temp db + real SkillLibrary 覆蓋 list/search/get/delete/export 與 404。"""
    skills_dir = tmp_path / "skills-store"
    db_path = skills_dir / "skills.db"
    library = SkillLibrary(db_path)
    debug_skill = make_skill(
        skill_id="skill-debug",
        name="Debug import failures",
        description="Resolve ModuleNotFoundError from pytest runs",
        trigger_keywords=["python", "pytest", "import"],
        updated_at=10,
    )
    csv_skill = make_skill(
        skill_id="skill-csv",
        name="Clean CSV columns",
        description="Normalize malformed spreadsheet headers",
        trigger_keywords=["csv", "pandas"],
        updated_at=20,
    )
    config_skill = make_skill(
        skill_id="skill-from-config",
        name="Inspect HTTP payloads",
        description="Capture and compare API JSON payloads",
        trigger_keywords=["http", "json"],
        updated_at=30,
    )
    _add_skills(library, debug_skill, csv_skill, config_skill)

    app = _create_app_with_skills_router()
    app.state.config_factory = lambda: MochiConfig.model_validate({"skills_dir": str(skills_dir)})

    with TestClient(app) as client:
        list_response = client.get("/v1/skills")
        search_response = client.get("/v1/skills", params={"q": "ModuleNotFoundError", "limit": 1})
        get_response = client.get("/v1/skills/skill-from-config")
        export_response = client.get("/v1/skills/export")
        delete_response = client.delete("/v1/skills/skill-debug")
        missing_get_response = client.get("/v1/skills/missing")
        missing_delete_response = client.delete("/v1/skills/missing")
        deleted_get_response = client.get("/v1/skills/skill-debug")

    assert list_response.status_code == 200
    list_ids = [item["skill_id"] for item in list_response.json()]
    learned_list_ids = [item_id for item_id in list_ids if not item_id.startswith("system:")]
    assert learned_list_ids == ["skill-from-config", "skill-csv", "skill-debug"]

    assert search_response.status_code == 200
    assert [item["skill_id"] for item in search_response.json()] == ["skill-debug"]

    assert get_response.status_code == 200
    assert get_response.json()["name"] == "Inspect HTTP payloads"

    assert export_response.status_code == 200
    export_ids = [item["skill_id"] for item in export_response.json()]
    learned_export_ids = [item_id for item_id in export_ids if not item_id.startswith("system:")]
    assert learned_export_ids == ["skill-from-config", "skill-csv", "skill-debug"]

    assert delete_response.status_code == 200
    assert delete_response.json() == {"deleted": True}

    assert missing_get_response.status_code == 404
    assert missing_get_response.json() == {"detail": "Skill not found"}
    assert missing_delete_response.status_code == 404
    assert missing_delete_response.json() == {"detail": "Skill not found"}
    assert deleted_get_response.status_code == 404
    assert deleted_get_response.json() == {"detail": "Skill not found"}
    assert db_path.exists()


def test_skills_routes_auto_sync_filesystem_skills(tmp_path: Path) -> None:
    """`GET /skills` 應自動索引 skills_dir 底下的 SKILL.md。"""
    skills_dir = tmp_path / "skills-store"
    skill_dir = skills_dir / "skill-installer"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        """---
name: skill-installer
description: Add local skills by placing SKILL.md directories under skills_dir.
tags: [skills, install]
---

# Skill Installer

No manual import command is required.
""",
        encoding="utf-8",
    )

    app = _create_app_with_skills_router()
    app.state.config_factory = lambda: MochiConfig.model_validate({"skills_dir": str(skills_dir)})

    with TestClient(app) as client:
        list_response = client.get("/v1/skills", params={"q": "manual import"})

    assert list_response.status_code == 200
    payload = list_response.json()
    assert any(item["skill_id"] == "skill-installer" for item in payload)
    indexed = next(item for item in payload if item["skill_id"] == "skill-installer")
    assert indexed["source_type"] == "filesystem"
    assert indexed["body"].startswith("# Skill Installer")


def test_skill_version_routes_require_evidence_and_preserve_history_and_pins(tmp_path: Path) -> None:
    skills_dir = tmp_path / "skills-store"
    library = SkillLibrary(skills_dir / "skills.db")
    skill = make_skill(
        skill_id="skill-versioned",
        name="Versioned skill",
        description="Initial retained version",
        trigger_keywords=["version"],
        updated_at=10,
    )
    _add_skills(library, skill)
    asyncio.run(library.update(skill.skill_id, {"description": "Improved retained version"}))
    asyncio.run(library.pin_run_version("recorded-run", skill.skill_id, 1))

    app = _create_app_with_skills_router()
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "skills_dir": str(skills_dir),
            "learning": {"auto_sync_filesystem_skills": False},
        },
    )

    with TestClient(app) as client:
        history_response = client.get("/v1/skills/skill-versioned/versions")
        retained_response = client.get("/v1/skills/skill-versioned/versions/1")
        no_evidence_response = client.post(
            "/v1/skills/skill-versioned/versions/1/promote",
            json={"evaluation_evidence": {}},
        )
        no_evidence_id_response = client.post(
            "/v1/skills/skill-versioned/versions/1/promote",
            json={"evaluation_evidence": {"score": 0.9}},
        )
        promoted_response = client.post(
            "/v1/skills/skill-versioned/versions/1/promote",
            json={"evaluation_evidence": {"evidence_id": "eval-1", "score": 0.9}},
        )
        active_after_promotion = client.get("/v1/skills/skill-versioned")
        missing_version_response = client.post(
            "/v1/skills/skill-versioned/versions/99/promote",
            json={"evaluation_evidence": {"evidence_id": "eval-99"}},
        )
        blank_reason_response = client.post(
            "/v1/skills/skill-versioned/versions/2/rollback",
            json={"reason": "   "},
        )
        rollback_response = client.post(
            "/v1/skills/skill-versioned/versions/2/rollback",
            json={"reason": "regression observed"},
        )
        active_after_rollback = client.get("/v1/skills/skill-versioned")

    assert history_response.status_code == 200
    assert [item["version"] for item in history_response.json()] == [1, 2]
    assert retained_response.status_code == 200
    assert retained_response.json()["description"] == "Initial retained version"

    assert no_evidence_response.status_code == 422
    assert no_evidence_id_response.status_code == 422
    assert promoted_response.status_code == 200
    assert promoted_response.json()["skill"]["version"] == 1
    assert promoted_response.json()["promotion"]["evidence"]["evidence_id"] == "eval-1"
    assert active_after_promotion.json()["version"] == 1
    assert missing_version_response.status_code == 404
    assert missing_version_response.json() == {"detail": "Skill version not found"}

    assert blank_reason_response.status_code == 422
    assert rollback_response.status_code == 200
    assert rollback_response.json()["skill"]["version"] == 2
    assert rollback_response.json()["reason"] == "regression observed"
    assert active_after_rollback.json()["version"] == 2
    assert [item.version for item in asyncio.run(library.list_versions(skill.skill_id))] == [1, 2]
    assert asyncio.run(library.get_pinned_version("recorded-run")).version == 1


def test_skill_version_routes_map_conflicts_to_409() -> None:
    app = _create_app_with_skills_router()
    app.state.skill_library = _PromotionConflictLibrary()

    with TestClient(app) as client:
        response = client.post(
            "/v1/skills/skill-versioned/versions/1/promote",
            json={"evaluation_evidence": {"evidence_id": "eval-1"}},
        )

    assert response.status_code == 409
    assert response.json() == {"detail": "skill version conflict"}
