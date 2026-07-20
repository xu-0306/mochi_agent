from __future__ import annotations

from mochi.goal_proposal_copy import (
    build_goal_chrome_copy,
    build_goal_command_help_message,
    build_goal_follow_up_message,
    build_goal_lifecycle_message,
    build_goal_proposal_assistant_copy_fallback,
    build_goal_status_label,
)
from mochi.terminal_goal_helpers import (
    build_goal_summary_from_goal,
    normalize_goal_session_state,
)


def test_goal_chrome_copy_localizes_traditional_chinese_labels() -> None:
    copy = build_goal_chrome_copy(user_message="\u958b\u59cb\u9019\u500b goal")

    assert copy.goal_status_label == "Goal \u72c0\u614b"
    assert copy.execution_label == "\u57f7\u884c\u65b9\u5f0f"
    assert build_goal_status_label(
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


def test_goal_follow_up_message_queued_after_resolution_mentions_queue_and_approval_gate() -> None:
    message = build_goal_follow_up_message(
        user_message="keep going after approval",
        kind="queued_after_resolution",
        approval_count=2,
    )

    assert "queued your new direction" in message
    assert "Resolve the pending approvals" in message
    assert "continue without you restating it" in message


def test_goal_follow_up_message_queued_after_resolution_traditional_chinese_mentions_queue_and_approval_gate() -> None:
    message = build_goal_follow_up_message(
        user_message="\u7b49\u6838\u51c6\u5b8c\u518d\u7e7c\u7e8c",
        kind="queued_after_resolution",
        approval_count=2,
    )

    assert "\u5148\u628a\u4f60\u7684\u65b0\u65b9\u5411\u8a18\u5230\u76ee\u524d\u7684 attempt" in message
    assert "\u5148\u8655\u7406\u5b8c 2 \u500b\u5f85\u6838\u51c6\u9805\u76ee\u5f8c" in message
    assert "\u4e0d\u9700\u8981\u4f60\u518d\u91cd\u8aaa\u4e00\u6b21" in message


def test_goal_proposal_fallback_copy_uses_goal_contract_strategy_language() -> None:
    message = build_goal_proposal_assistant_copy_fallback(
        user_message="Plan the migration",
        proposal_objective="Plan the migration",
        execution_mode="workflow",
        protocol_selection="controlled_subagent_execution",
        revision_index=0,
    )

    assert "goal draft" in message
    assert "contract for this task" in message
    assert "selected execution strategy is controlled_subagent_execution" in message
    assert "execution begins only after you confirm the start" in message
    assert "launch directly" not in message
    assert "workflow goal" not in message


def test_goal_proposal_fallback_copy_traditional_chinese_uses_goal_contract_strategy_language() -> None:
    message = build_goal_proposal_assistant_copy_fallback(
        user_message="\u8acb\u898f\u5283\u9077\u79fb",
        proposal_objective="\u8acb\u898f\u5283\u9077\u79fb",
        execution_mode="workflow",
        protocol_selection="controlled_subagent_execution",
        revision_index=0,
    )

    assert "\u8349\u7a3f" in message
    assert "\u57f7\u884c\u5951\u7d04" in message
    assert "controlled_subagent_execution" in message
    assert "\u78ba\u8a8d\u555f\u52d5\u5f8c\u624d\u6703\u958b\u59cb\u57f7\u884c" in message
    assert "\u53ef\u4ee5\u76f4\u63a5\u555f\u52d5" not in message
    assert "workflow goal" not in message


def test_goal_proposal_fallback_copy_simplified_chinese_uses_goal_contract_strategy_language() -> None:
    message = build_goal_proposal_assistant_copy_fallback(
        user_message="\u8bf7\u89c4\u5212\u8fc1\u79fb",
        proposal_objective="\u8bf7\u89c4\u5212\u8fc1\u79fb",
        execution_mode="workflow",
        protocol_selection="controlled_subagent_execution",
        revision_index=0,
    )

    assert "\u8349\u7a3f" in message
    assert "\u6267\u884c\u5951\u7ea6" in message
    assert "controlled_subagent_execution" in message
    assert "\u786e\u8ba4\u542f\u52a8\u540e\u624d\u4f1a\u5f00\u59cb\u6267\u884c" in message
    assert "\u53ef\u4ee5\u76f4\u63a5\u542f\u52a8" not in message
    assert "workflow goal" not in message


def test_goal_command_help_message_demotes_workflow_to_explicit_override() -> None:
    message = build_goal_command_help_message(user_message="help with goals")

    assert "Describe the task normally" in message
    assert "use `/goal <request>` to prepare a goal draft" in message
    assert "Use `/workflow <request>` only when you want to explicitly override the goal with a workflow strategy." in message
    assert "prepare a workflow goal" not in message


def test_goal_command_help_message_traditional_chinese_demotes_workflow_to_explicit_override() -> None:
    message = build_goal_command_help_message(user_message="\u8acb\u8aaa\u660e goal \u6307\u4ee4")

    assert "\u76f4\u63a5\u63cf\u8ff0\u60f3\u5b8c\u6210\u7684\u4efb\u52d9" in message
    assert "`/goal <request>`" in message
    assert "\u660e\u78ba\u6307\u5b9a workflow \u7b56\u7565" in message
    assert "\u9032\u968e\u8986\u5beb" in message
    assert "\u6e96\u5099 workflow goal" not in message


def test_goal_command_help_message_simplified_chinese_demotes_workflow_to_explicit_override() -> None:
    message = build_goal_command_help_message(user_message="\u8bf7\u8bf4\u660e goal \u6307\u4ee4")

    assert "\u76f4\u63a5\u63cf\u8ff0\u60f3\u5b8c\u6210\u7684\u4efb\u52a1" in message
    assert "`/goal <request>`" in message
    assert "\u660e\u786e\u6307\u5b9a workflow \u7b56\u7565" in message
    assert "\u8fdb\u9636\u8986\u5199" in message
    assert "\u51c6\u5907 workflow goal" not in message


def test_goal_lifecycle_copy_does_not_offer_workflow_as_primary_new_goal_entry() -> None:
    no_active_goal = build_goal_lifecycle_message(
        user_message="help",
        kind="no_active_goal",
    )
    pending_cleared = build_goal_lifecycle_message(
        user_message="help",
        kind="pending_cleared",
    )

    assert no_active_goal.endswith(
        "Describe the task normally, or use `/goal <request>` to prepare a new goal."
    )
    assert pending_cleared.endswith(
        "Describe the task normally, or use `/goal <request>` to prepare a new draft."
    )
    assert "/workflow <request>" not in no_active_goal
    assert "/workflow <request>" not in pending_cleared


def test_goal_lifecycle_copy_traditional_chinese_does_not_offer_workflow_as_primary_new_goal_entry() -> None:
    no_active_goal = build_goal_lifecycle_message(
        user_message="\u8acb\u5e6b\u6211",
        kind="no_active_goal",
    )
    pending_cleared = build_goal_lifecycle_message(
        user_message="\u8acb\u5e6b\u6211",
        kind="pending_cleared",
    )

    assert "\u76f4\u63a5\u63cf\u8ff0\u4efb\u52d9" in no_active_goal
    assert "\u76f4\u63a5\u63cf\u8ff0\u4efb\u52d9" in pending_cleared
    assert "`/goal <request>`" in no_active_goal
    assert "`/goal <request>`" in pending_cleared
    assert "/workflow <request>" not in no_active_goal
    assert "/workflow <request>" not in pending_cleared


def test_goal_lifecycle_copy_simplified_chinese_does_not_offer_workflow_as_primary_new_goal_entry() -> None:
    no_active_goal = build_goal_lifecycle_message(
        user_message="\u8bf7\u5e2e\u6211",
        kind="no_active_goal",
    )
    pending_cleared = build_goal_lifecycle_message(
        user_message="\u8bf7\u5e2e\u6211",
        kind="pending_cleared",
    )

    assert "\u76f4\u63a5\u63cf\u8ff0\u4efb\u52a1" in no_active_goal
    assert "\u76f4\u63a5\u63cf\u8ff0\u4efb\u52a1" in pending_cleared
    assert "`/goal <request>`" in no_active_goal
    assert "`/goal <request>`" in pending_cleared
    assert "/workflow <request>" not in no_active_goal
    assert "/workflow <request>" not in pending_cleared


def test_normalize_goal_session_state_preserves_backend_goal_metadata_without_synthetic_defaults() -> None:
    state = normalize_goal_session_state(
        {
            "active_goal_id": "goal-1",
            "active_goal_status": "running",
            "execution_mode": "single_agent",
            "interaction_mode": "goal",
            "execution_topology": "single_agent",
            "protocol_selection": "backend-selected",
            "selection_rationale": "Backend chose direct execution.",
            "default_route": "workflow",
            "last_goal_summary": {
                "goal_id": "goal-1",
                "objective": "Ship the release",
                "execution_mode": "single_agent",
                "interaction_mode": "goal",
                "execution_topology": "single_agent",
                "protocol_selection": "backend-selected",
                "selection_rationale": "Backend chose direct execution.",
                "status": "running",
            },
        }
    )

    assert state["interaction_mode"] == "goal"
    assert state["execution_topology"] == "single_agent"
    assert state["protocol_selection"] == "backend-selected"
    assert state["selection_rationale"] == "Backend chose direct execution."
    assert state["default_route"] == "workflow"
    assert state["last_goal_summary"]["interaction_mode"] == "goal"
    assert state["last_goal_summary"]["execution_topology"] == "single_agent"


def test_build_goal_summary_from_goal_does_not_invent_route_metadata_for_legacy_goal_rows() -> None:
    summary = build_goal_summary_from_goal(
        {
            "goal_id": "goal-2",
            "objective": "Legacy row",
            "execution_mode": "workflow",
            "status": "running",
        }
    )

    assert summary["execution_mode"] == "workflow"
    assert summary["interaction_mode"] is None
    assert summary["execution_topology"] is None
    assert summary["protocol_selection"] is None
    assert summary["selection_rationale"] is None
