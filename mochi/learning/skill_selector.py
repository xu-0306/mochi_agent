"""Thin explicit-skill selection for prompt injection and tool preference."""

from __future__ import annotations

from dataclasses import dataclass

from mochi.learning.skill_library import SkillLibrary
from mochi.learning.skill_loader import SkillLoader
from mochi.learning.types import Skill


@dataclass(frozen=True)
class SkillSelection:
    """Selected skills for one turn."""

    explicit_skills: list[Skill]
    suggested_skills: list[Skill]
    preferred_tool_names: list[str]

    @property
    def all_skills(self) -> list[Skill]:
        """All selected skills in prompt order."""
        return [*self.explicit_skills, *self.suggested_skills]


class SkillSelector:
    """Resolve explicit and inferred skills without heavy planning."""

    def __init__(
        self,
        *,
        library: SkillLibrary,
        loader: SkillLoader | None = None,
        auto_sync: bool = True,
        max_skills: int = 3,
    ) -> None:
        self._library = library
        self._loader = loader
        self._auto_sync = auto_sync
        self._max_skills = max(0, max_skills)

    async def select(
        self,
        message: str,
        *,
        selected_skill_ids: list[str] | None = None,
    ) -> SkillSelection:
        """Select explicit skills first, then fill remaining slots from search."""
        if self._auto_sync and self._loader is not None:
            await self._loader.sync(self._library)

        explicit_ids = self._normalize_skill_ids(selected_skill_ids)
        explicit_skills: list[Skill] = []
        seen_ids: set[str] = set()
        for skill_id in explicit_ids:
            skill = await self._library.get(skill_id)
            if skill is None or skill.skill_id in seen_ids:
                continue
            explicit_skills.append(skill)
            seen_ids.add(skill.skill_id)

        remaining = max(0, self._max_skills - len(explicit_skills))
        suggested_skills: list[Skill] = []
        if remaining > 0:
            search_results = await self._library.search(message, top_k=max(self._max_skills * 2, remaining))
            for skill in search_results:
                if skill.skill_id in seen_ids:
                    continue
                suggested_skills.append(skill)
                seen_ids.add(skill.skill_id)
                if len(suggested_skills) >= remaining:
                    break

        preferred_tool_names = self._collect_tool_names([*explicit_skills, *suggested_skills])
        return SkillSelection(
            explicit_skills=explicit_skills,
            suggested_skills=suggested_skills,
            preferred_tool_names=preferred_tool_names,
        )

    @staticmethod
    def _normalize_skill_ids(selected_skill_ids: list[str] | None) -> list[str]:
        normalized: list[str] = []
        for skill_id in selected_skill_ids or []:
            if not isinstance(skill_id, str):
                continue
            compact = skill_id.strip()
            if compact and compact not in normalized:
                normalized.append(compact)
        return normalized

    @staticmethod
    def _collect_tool_names(skills: list[Skill]) -> list[str]:
        tool_names: list[str] = []
        for skill in skills:
            for tool_name in skill.tools_used:
                compact = str(tool_name).strip()
                if compact and compact not in tool_names:
                    tool_names.append(compact)
        return tool_names
