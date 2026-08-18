"""Skills API routes."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, cast

from fastapi import APIRouter, HTTPException, Path, Query, Request
from pydantic import BaseModel, Field

from mochi.api.server import _get_config  # pyright: ignore[reportPrivateUsage]
from mochi.learning.skill_library import SkillLibrary
from mochi.learning.skill_library_factory import resolve_skills_db_path
from mochi.learning.skill_loader import SkillLoader, default_system_skills_dir

router = APIRouter(prefix="/v1")


class PromoteSkillVersionRequest(BaseModel):
    """Evidence required to select a retained skill version."""

    evaluation_evidence: dict[str, Any] = Field(min_length=1)


class RollbackSkillVersionRequest(BaseModel):
    """Reason required to select an earlier retained skill version."""

    reason: str = Field(min_length=1)


class PromotionResponse(BaseModel):
    """The active projection and the durable evidence that selected it."""

    skill: dict[str, Any]
    promotion: dict[str, Any]


class RollbackResponse(BaseModel):
    """The active projection selected by an auditable rollback."""

    skill: dict[str, Any]
    reason: str


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


    async def get_version(self, skill_id: str, version: int) -> Any | None:
        """Return one immutable retained skill version."""
        ...

    async def list_versions(self, skill_id: str) -> list[Any]:
        """Return immutable retained versions in ascending order."""
        ...

    async def promote(
        self,
        skill_id: str,
        version: int,
        evaluation_evidence: Mapping[str, Any],
    ) -> Any:
        """Select a retained version after validating evaluation evidence."""
        ...

    async def rollback(self, skill_id: str, version: int, *, reason: str) -> Any:
        """Select a retained version without changing history or run pins."""
        ...

    async def get_promotion(self, skill_id: str, version: int) -> dict[str, Any] | None:
        """Return retained evidence for a promoted version."""
        ...


async def _get_skill_library(request: Request) -> SupportsSkillLibrary:
    """取得 skills route 使用的 SkillLibrary。"""
    existing = cast(SupportsSkillLibrary | None, getattr(request.app.state, "skill_library", None))
    if existing is not None:
        return existing

    config = await _get_config(request.app)
    skills_dir = getattr(config, "skills_dir", None)
    if skills_dir is None:
        raise RuntimeError("Config does not provide skills_dir")

    library = SkillLibrary(
        db_path=resolve_skills_db_path(skills_dir=skills_dir),
    )
    request.app.state.skill_library = library
    return library


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


def _raise_skill_version_error(error: ValueError | KeyError) -> None:
    """Project governance failures to stable HTTP status classes."""

    if isinstance(error, KeyError):
        raise HTTPException(status_code=404, detail="Skill version not found") from error

    detail = str(error)
    if "conflict" in detail.lower() or "already pinned" in detail.lower():
        raise HTTPException(status_code=409, detail=detail) from error
    raise HTTPException(status_code=422, detail=detail) from error


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


@router.get("/skills/{skill_id}/versions")
async def list_skill_versions(request: Request, skill_id: str) -> list[dict[str, Any]]:
    """List immutable retained versions for a skill, including inactive versions."""

    library = await _get_skill_library(request)
    await _sync_filesystem_skills(request, library)
    versions = await library.list_versions(skill_id)
    if not versions:
        raise HTTPException(status_code=404, detail="Skill version not found")
    return [_skill_payload(skill) for skill in versions]


@router.get("/skills/{skill_id}/versions/{version}")
async def get_skill_version(
    request: Request,
    skill_id: str,
    version: int = Path(ge=1),
) -> dict[str, Any]:
    """Return one immutable retained version without changing active selection."""

    library = await _get_skill_library(request)
    await _sync_filesystem_skills(request, library)
    skill = await library.get_version(skill_id, version)
    if skill is None:
        raise HTTPException(status_code=404, detail="Skill version not found")
    return _skill_payload(skill)


@router.post("/skills/{skill_id}/versions/{version}/promote", response_model=PromotionResponse)
async def promote_skill_version(
    request: Request,
    skill_id: str,
    payload: PromoteSkillVersionRequest,
    version: int = Path(ge=1),
) -> PromotionResponse:
    """Select a retained version only with non-empty, identified evaluation evidence."""

    library = await _get_skill_library(request)
    try:
        skill = await library.promote(skill_id, version, payload.evaluation_evidence)
    except (KeyError, ValueError) as error:
        _raise_skill_version_error(error)

    promotion = await library.get_promotion(skill_id, version)
    if promotion is None:
        raise HTTPException(status_code=409, detail="Skill version promotion was not recorded")
    return PromotionResponse(skill=_skill_payload(skill), promotion=promotion)


@router.post("/skills/{skill_id}/versions/{version}/rollback", response_model=RollbackResponse)
async def rollback_skill_version(
    request: Request,
    skill_id: str,
    payload: RollbackSkillVersionRequest,
    version: int = Path(ge=1),
) -> RollbackResponse:
    """Select a retained version without deleting history or changing run pins."""

    library = await _get_skill_library(request)
    try:
        skill = await library.rollback(skill_id, version, reason=payload.reason)
    except (KeyError, ValueError) as error:
        _raise_skill_version_error(error)
    return RollbackResponse(skill=_skill_payload(skill), reason=payload.reason.strip())


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
