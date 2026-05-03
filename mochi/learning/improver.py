"""技能改進器。"""

from __future__ import annotations

import time
from dataclasses import replace

from mochi.backends.types import Message
from mochi.learning.extractor import (
    _as_text,
    _as_text_list,
    _call_backend,
    _default_steps,
    _final_answer,
    _parse_json_object,
    _response_content,
    _tools_used,
)
from mochi.learning.types import Skill, Trajectory


class SkillImprover:
    """Reflexion 模式技能改進器。"""

    async def improve(
        self,
        original_skill: Skill,
        new_trajectory: Trajectory,
        backend: object,
    ) -> Skill:
        """將新軌跡與原技能合併為改進版本。"""
        payload: dict[str, object] = {}
        try:
            response = await _call_backend(
                backend,
                _build_messages(original_skill, new_trajectory),
                fallback_method_names=("improve_skill", "improve"),
                original_skill=original_skill,
                new_trajectory=new_trajectory,
            )
            payload = _parse_json_object(_response_content(response))
        except ValueError:
            payload = {}

        if payload:
            return _improved_from_payload(original_skill, new_trajectory, payload)
        return _fallback_merge(original_skill, new_trajectory)


def _build_messages(original_skill: Skill, new_trajectory: Trajectory) -> list[Message]:
    return [
        Message(
            role="system",
            content=(
                "Improve the original skill with the new trajectory. "
                "Return only one JSON object."
            ),
        ),
        Message(
            role="user",
            content=(
                f"Original skill: {original_skill}\n"
                f"New trajectory: {new_trajectory.trajectory_id}\n"
                f"Task: {new_trajectory.task_description}\n"
                f"Final answer: {_final_answer(new_trajectory)}\n"
                f"Tools used: {_tools_used(new_trajectory)}"
            ),
        ),
    ]


def _improved_from_payload(
    original_skill: Skill,
    new_trajectory: Trajectory,
    payload: dict[str, object],
) -> Skill:
    now = time.time()
    return replace(
        original_skill,
        skill_id=_as_text(payload.get("skill_id")) or original_skill.skill_id,
        name=_as_text(payload.get("name")) or original_skill.name,
        description=_as_text(payload.get("description")) or original_skill.description,
        trigger_keywords=_as_text_list(payload.get("trigger_keywords"))
        or original_skill.trigger_keywords,
        preconditions=_as_text(payload.get("preconditions")) or original_skill.preconditions,
        steps=_as_text_list(payload.get("steps")) or original_skill.steps,
        tools_used=_merge_unique(
            _as_text_list(payload.get("tools_used")) or original_skill.tools_used,
            _tools_used(new_trajectory),
        ),
        source_trajectory_id=new_trajectory.trajectory_id,
        times_used=int(payload.get("times_used") or original_skill.times_used),
        success_rate=float(payload.get("success_rate") or original_skill.success_rate),
        created_at=float(payload.get("created_at") or original_skill.created_at),
        updated_at=float(payload.get("updated_at") or now),
        version=original_skill.version + 1,
    )


def _fallback_merge(original_skill: Skill, new_trajectory: Trajectory) -> Skill:
    now = time.time()
    final_answer = _final_answer(new_trajectory)
    description_parts = [original_skill.description]
    if final_answer and final_answer not in original_skill.description:
        description_parts.append(f"Observed outcome: {final_answer[:200]}")

    appended_steps = _default_steps(new_trajectory)
    return replace(
        original_skill,
        description="\n".join(part for part in description_parts if part),
        steps=_merge_unique(original_skill.steps, appended_steps),
        tools_used=_merge_unique(original_skill.tools_used, _tools_used(new_trajectory)),
        source_trajectory_id=new_trajectory.trajectory_id,
        updated_at=now,
        version=original_skill.version + 1,
    )


def _merge_unique(first: list[str], second: list[str]) -> list[str]:
    merged: list[str] = []
    for item in [*first, *second]:
        text = item.strip()
        if text and text not in merged:
            merged.append(text)
    return merged
