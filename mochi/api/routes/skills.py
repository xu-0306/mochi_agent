"""Skills API routes."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, cast

from fastapi import APIRouter, HTTPException, Query, Request

from mochi.api.server import _get_config  # pyright: ignore[reportPrivateUsage]
from mochi.learning.skill_library import SkillLibrary
from mochi.learning.skill_loader import SkillLoader, default_system_skills_dir

router = APIRouter(prefix="/v1")


class SupportsSkillLibrary(Protocol):
    """Skills route 需要的 library 介面。"""

    async def list(self, limit: int | None = None) -> list[Any]:
        """列出技能。"""
        ...

    async def search(self, query: str, top_k: int = 3) -> list[Any]:
        """搜尋技能。"""
        ...

    async def get(self, skill_id: str) -> Any | None:
        """取得單一技能。"""
        ...

    async def delete(self, skill_id: str) -> bool:
        """刪除技能。"""
        ...

    async def export(self) -> list[dict[str, Any]]:
        """匯出技能。"""
        ...


async def _get_skill_library(request: Request) -> SupportsSkillLibrary:
    """取得 skills route 使用的 SkillLibrary。"""
    existing = cast(SupportsSkillLibrary | None, getattr(request.app.state, "skill_library", None))
    if existing is not None:
        return existing

    config = await _get_config(request.app)
    db_path = getattr(config, "skills_dir", None)
    if db_path is None:
        raise RuntimeError("Config does not provide skills_dir")
    return SkillLibrary(db_path=Path(db_path).expanduser() / "skills.db")


async def _sync_filesystem_skills(request: Request, library: SupportsSkillLibrary) -> None:
    """同步 filesystem/system SKILL.md 到技能索引。"""
    if not isinstance(library, SkillLibrary):
        return
    config = await _get_config(request.app)
    if not config.learning.auto_sync_filesystem_skills:
        return
    loader = SkillLoader.from_paths(
        getattr(config, "skills_dir", None),
        system_skills_dir=default_system_skills_dir(),
    )
    await loader.sync(library)


def _skill_payload(skill: Any) -> dict[str, Any]:
    """限制輸出為 skill fields。"""
    return dict(skill.to_dict())


@router.get("/skills")
async def list_skills(
    request: Request,
    q: str | None = None,
    limit: int = Query(default=50, ge=1),
) -> list[dict[str, Any]]:
    """列出或搜尋技能。"""
    library = await _get_skill_library(request)
    await _sync_filesystem_skills(request, library)
    skills = await library.search(q, top_k=limit) if q else await library.list(limit=limit)
    return [_skill_payload(skill) for skill in skills]


@router.get("/skills/export")
async def export_skills(request: Request) -> list[dict[str, Any]]:
    """匯出所有技能。"""
    library = await _get_skill_library(request)
    await _sync_filesystem_skills(request, library)
    return await library.export()


@router.get("/skills/{skill_id}")
async def get_skill(request: Request, skill_id: str) -> dict[str, Any]:
    """依 ID 取得技能。"""
    library = await _get_skill_library(request)
    await _sync_filesystem_skills(request, library)
    skill = await library.get(skill_id)
    if skill is None:
        raise HTTPException(status_code=404, detail="Skill not found")
    return _skill_payload(skill)


@router.delete("/skills/{skill_id}")
async def delete_skill(request: Request, skill_id: str) -> dict[str, bool]:
    """刪除技能。"""
    library = await _get_skill_library(request)
    deleted = await library.delete(skill_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Skill not found")
    return {"deleted": True}
