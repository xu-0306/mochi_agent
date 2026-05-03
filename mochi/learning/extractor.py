"""技能萃取器。"""

from __future__ import annotations

import inspect
import json
import time
from typing import Any
from uuid import uuid4

from mochi.backends.types import Message
from mochi.learning.types import Skill, Trajectory


class SkillExtractor:
    """用 LLM 將成功軌跡蒸餾為可重用 Skill。"""

    async def extract(self, trajectory: Trajectory, backend: object) -> Skill:
        """從成功軌跡萃取技能。"""
        if trajectory.outcome != "success":
            raise ValueError("Skill extraction requires a successful trajectory.")

        messages = _build_messages(trajectory)
        response = await _call_backend(
            backend,
            messages,
            fallback_method_names=("extract_skill", "extract"),
            trajectory=trajectory,
        )
        payload = _parse_json_object(_response_content(response))
        return _skill_from_payload(payload, trajectory)


def _build_messages(trajectory: Trajectory) -> list[Message]:
    return [
        Message(
            role="system",
            content=(
                "Extract one reusable skill from the successful trajectory. "
                "Return only one JSON object."
            ),
        ),
        Message(
            role="user",
            content=json.dumps(_trajectory_summary(trajectory), ensure_ascii=False),
        ),
    ]


async def _call_backend(
    backend: object,
    messages: list[Message],
    fallback_method_names: tuple[str, ...],
    **kwargs: object,
) -> object:
    method = getattr(backend, "generate", None)
    if callable(method):
        try:
            result = method(messages=messages, tools=None)
        except TypeError:
            result = method(messages, tools=None)
        if inspect.isawaitable(result):
            return await result
        return result

    for method_name in fallback_method_names:
        method = getattr(backend, method_name, None)
        if not callable(method):
            continue
        try:
            result = method(**kwargs)
        except TypeError:
            result = method(*kwargs.values())
        if inspect.isawaitable(result):
            return await result
        return result

    raise ValueError("Backend must provide generate() or a compatible fake backend method.")


def _response_content(response: object) -> str:
    if response is None:
        return ""
    if isinstance(response, str):
        return response

    content = getattr(response, "content", None)
    if isinstance(content, str):
        return content

    message = getattr(response, "message", None)
    message_content = getattr(message, "content", None)
    if isinstance(message_content, str):
        return message_content

    if isinstance(response, dict):
        content = response.get("content")
        if isinstance(content, str):
            return content
        message = response.get("message")
        if isinstance(message, dict) and isinstance(message.get("content"), str):
            return message["content"]

    return str(response)


def _parse_json_object(content: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    text = content.strip()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return {}


def _skill_from_payload(payload: dict[str, Any], trajectory: Trajectory) -> Skill:
    now = time.time()
    return Skill(
        skill_id=str(payload.get("skill_id") or uuid4()),
        name=_as_text(payload.get("name")) or _default_name(trajectory),
        description=_as_text(payload.get("description")) or _default_description(trajectory),
        trigger_keywords=_as_text_list(payload.get("trigger_keywords"))
        or _default_keywords(trajectory),
        preconditions=_as_text(payload.get("preconditions")) or "",
        steps=_as_text_list(payload.get("steps")) or _default_steps(trajectory),
        tools_used=_as_text_list(payload.get("tools_used")) or _tools_used(trajectory),
        source_trajectory_id=trajectory.trajectory_id,
        times_used=int(payload.get("times_used") or 0),
        success_rate=float(payload.get("success_rate") or 1.0),
        created_at=float(payload.get("created_at") or now),
        updated_at=float(payload.get("updated_at") or now),
        version=int(payload.get("version") or 1),
    )


def _trajectory_summary(trajectory: Trajectory) -> dict[str, object]:
    return {
        "trajectory_id": trajectory.trajectory_id,
        "task_description": trajectory.task_description,
        "outcome": trajectory.outcome,
        "final_answer": _final_answer(trajectory),
        "tools_used": _tools_used(trajectory),
        "steps": [
            {
                "step_id": step.step_id,
                "step_type": step.step_type,
                "input_data": step.input_data,
                "output_data": step.output_data,
                "metadata": step.metadata,
            }
            for step in trajectory.steps
        ],
    }


def _default_name(trajectory: Trajectory) -> str:
    text = trajectory.task_description.strip()
    return text[:60] if text else f"Skill from {trajectory.trajectory_id}"


def _default_description(trajectory: Trajectory) -> str:
    final_answer = _final_answer(trajectory)
    if final_answer:
        return final_answer[:200]
    return trajectory.task_description.strip() or "Reusable skill extracted from a trajectory."


def _default_keywords(trajectory: Trajectory) -> list[str]:
    words: list[str] = []
    for raw_word in trajectory.task_description.replace("_", " ").split():
        word = raw_word.strip(".,:;!?()[]{}\"'").lower()
        if len(word) >= 3 and word not in words:
            words.append(word)
        if len(words) == 5:
            break
    return words or ["trajectory", "skill"]


def _default_steps(trajectory: Trajectory) -> list[str]:
    steps: list[str] = []
    for step in trajectory.steps:
        if step.step_type == "tool_call":
            tool_name = _tool_name(step)
            steps.append(f"Run tool {tool_name}." if tool_name else "Run the required tool.")
        elif step.step_type == "final_answer":
            final_answer = _final_answer(trajectory)
            steps.append(f"Return the final answer: {final_answer[:120]}" if final_answer else "Return the final answer.")
    return steps or ["Review the task context.", "Perform the required actions.", "Return the result."]


def _tools_used(trajectory: Trajectory) -> list[str]:
    tools: list[str] = []
    for step in trajectory.steps:
        if step.step_type != "tool_call":
            continue
        tool_name = _tool_name(step)
        if tool_name and tool_name not in tools:
            tools.append(tool_name)
    return tools


def _tool_name(step: object) -> str:
    for container_name in ("input_data", "output_data", "metadata"):
        container = getattr(step, container_name, {})
        if not isinstance(container, dict):
            continue
        for key in ("tool", "tool_name", "name"):
            value = container.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return ""


def _final_answer(trajectory: Trajectory) -> str:
    for step in reversed(trajectory.steps):
        if step.step_type != "final_answer":
            continue
        for container in (step.output_data, step.metadata):
            text = _first_text(container)
            if text:
                return text
    return ""


def _first_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("content", "answer", "final_answer", "text", "message", "result"):
            text = _first_text(value.get(key))
            if text:
                return text
        for child in value.values():
            text = _first_text(child)
            if text:
                return text
    if isinstance(value, list):
        for child in value:
            text = _first_text(child)
            if text:
                return text
    return ""


def _as_text(value: object) -> str:
    return value.strip() if isinstance(value, str) else ""


def _as_text_list(value: object) -> list[str]:
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return []
