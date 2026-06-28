from __future__ import annotations

from mochi.goal_proposal_copy import (
    build_goal_card_chrome_copy,
    build_goal_card_status_label,
    build_goal_follow_up_message,
)
from mochi.terminal_goal_helpers import goal_card_from_summary


def test_goal_card_chrome_copy_localizes_traditional_chinese_labels() -> None:
    copy = build_goal_card_chrome_copy(user_message="\u958b\u59cb\u9019\u500b goal")

    assert copy.goal_status_label == "Goal \u72c0\u614b"
    assert copy.execution_label == "\u57f7\u884c\u65b9\u5f0f"
    assert build_goal_card_status_label(
        user_message="\u958b\u59cb\u9019\u500b goal",
        status="running",
    ) == "\u57f7\u884c\u4e2d"


def test_goal_follow_up_message_falls_back_to_chinese_when_summary_is_english() -> None:
    message = build_goal_follow_up_message(
        user_message="\u8acb\u7e7c\u7e8c\u8655\u7406",
        kind="manual_resolution_required",
        summary="The active goal needs approval handling before it can continue.",
        approval_count=1,
        tool_names=["exec_command"],
    )

    assert "The active goal needs approval handling" not in message
    assert "\u5f85\u6838\u51c6\u5de5\u5177" in message
    assert "Goal Console" in message


def test_terminal_goal_card_uses_copy_source_for_default_label() -> None:
    card = goal_card_from_summary(
        {
            "objective": "Ship the release",
            "execution_mode": "single_agent",
        },
        kind="started",
        copy_source="\u958b\u59cb",
    )

    assert card["label"] == "Goal \u5df2\u555f\u52d5"
    assert card["copySource"] == "\u958b\u59cb"
