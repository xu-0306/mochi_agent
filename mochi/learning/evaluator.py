"""結果評估器。"""

from __future__ import annotations

from mochi.learning.types import Trajectory

VALID_OUTCOMES = {"success", "failure", "partial", "unknown"}


class OutcomeEvaluator:
    """評估任務結果，決定是否觸發技能萃取。"""

    async def evaluate(self, trajectory: Trajectory) -> str:
        """評估軌跡結果（success/failure/partial/unknown）。"""
        if trajectory.outcome in {"success", "failure", "partial"}:
            return trajectory.outcome
        if trajectory.outcome != "unknown":
            return "unknown"

        feedback_outcome = _classify_feedback(trajectory.user_feedback)
        if feedback_outcome != "unknown":
            return feedback_outcome

        has_final_answer = _has_final_answer(trajectory)
        has_error = _has_error(trajectory)

        if has_error and has_final_answer:
            return "partial"
        if has_error:
            return "failure"
        if has_final_answer:
            return "success"
        return "unknown"


def _classify_feedback(feedback: str | None) -> str:
    if not feedback:
        return "unknown"

    text = feedback.strip().lower()
    if not text:
        return "unknown"

    partial_markers = (
        "partial",
        "partially",
        "incomplete",
        "not complete",
        "almost",
        "部分",
        "一部分",
        "尚未完成",
        "不完整",
    )
    failure_markers = (
        "fail",
        "failed",
        "wrong",
        "incorrect",
        "error",
        "bad",
        "not solved",
        "失敗",
        "錯",
        "錯誤",
        "不對",
        "沒解決",
    )
    success_markers = (
        "success",
        "succeeded",
        "works",
        "worked",
        "correct",
        "done",
        "ok",
        "good",
        "成功",
        "完成",
        "正確",
        "可以",
    )

    if any(marker in text for marker in partial_markers):
        return "partial"
    if any(marker in text for marker in failure_markers):
        return "failure"
    if any(marker in text for marker in success_markers):
        return "success"
    return "unknown"


def _has_final_answer(trajectory: Trajectory) -> bool:
    for step in trajectory.steps:
        if step.step_type != "final_answer":
            continue
        if _contains_text(step.output_data) or _contains_text(step.metadata):
            return True
    return False


def _has_error(trajectory: Trajectory) -> bool:
    for step in trajectory.steps:
        if _dict_has_error(step.output_data) or _dict_has_error(step.metadata):
            return True
    return False


def _dict_has_error(value: object) -> bool:
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key).lower()
            if key_text in {"error", "exception", "traceback"} and child:
                return True
            if _dict_has_error(child):
                return True
    if isinstance(value, list):
        return any(_dict_has_error(item) for item in value)
    return False


def _contains_text(value: object) -> bool:
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, dict):
        return any(_contains_text(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_text(item) for item in value)
    return value is not None
