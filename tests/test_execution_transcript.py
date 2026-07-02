"""Execution transcript normalization tests."""

from __future__ import annotations

from mochi.runtime.execution_transcript import normalize_subagent_event


def test_normalize_subagent_started_event_requires_identity() -> None:
    event = normalize_subagent_event(
        {
            "type": "role_started",
            "subagent_id": "sub-1",
            "role_id": "researcher",
            "model_id": "qwen",
            "current_action": "Researcher is preparing a response.",
        },
        parent_type="agent_run",
        parent_id="run-1",
    )

    assert event["type"] == "subagent_started"
    assert event["subagent_id"] == "sub-1"
    assert event["role_id"] == "researcher"
    assert event["parent_type"] == "agent_run"
    assert event["parent_id"] == "run-1"
    assert event["content"] == "Researcher is preparing a response."


def test_normalize_runtime_blocker_event_preserves_approval_metadata() -> None:
    event = normalize_subagent_event(
        {
            "type": "runtime_blocked",
            "blocker_type": "approval",
            "summary": "Goal is waiting on operator approval.",
            "approval_ids": ["approval-1"],
            "tool_names": ["exec_command"],
            "recommended_action": "resolve_approval",
        },
        parent_type="goal",
        parent_id="goal-1",
    )

    assert event["type"] == "runtime_blocked"
    assert event["blocker_type"] == "approval"
    assert event["approval_ids"] == ["approval-1"]
    assert event["tool_names"] == ["exec_command"]
    assert event["parent_type"] == "goal"
    assert event["parent_id"] == "goal-1"
    assert "subagent_id" not in event


def test_normalize_transcript_event_preserves_raw_source_contract_fields() -> None:
    event = normalize_subagent_event(
        {
            "type": "runtime_blocked",
            "event_id": "event-runtime-blocked-1",
            "dedupe_key": "approval-blocker:approval-1",
            "visibility": "visible",
            "durability": "durable",
            "projection_lane": "goal_surface",
            "blocker_type": "approval",
            "summary": "Goal is waiting on operator approval.",
            "approval_ids": ["approval-1"],
        },
        parent_type="goal",
        parent_id="goal-1",
    )

    assert event["event_id"] == "event-runtime-blocked-1"
    assert event["dedupe_key"] == "approval-blocker:approval-1"
    assert event["visibility"] == "visible"
    assert event["durability"] == "durable"
    assert event["projection_lane"] == "goal_surface"
    assert "event_id" not in event["metadata"]
    assert "dedupe_key" not in event["metadata"]


def test_normalize_transcript_event_derives_source_contract_defaults() -> None:
    first = normalize_subagent_event(
        {
            "type": "subagent_tool_result",
            "subagent_id": "sub-1",
            "tool_call_id": "tool-call-1",
            "tool_name": "exec_command",
            "status": "failed",
            "summary": "Command failed.",
            "created_at": "2026-07-02T01:00:00Z",
        },
        parent_type="agent_run",
        parent_id="run-1",
    )
    second = normalize_subagent_event(
        {
            "type": "subagent_tool_result",
            "subagent_id": "sub-1",
            "tool_call_id": "tool-call-1",
            "tool_name": "exec_command",
            "status": "failed",
            "summary": "Command failed.",
            "created_at": "2026-07-02T01:00:00Z",
        },
        parent_type="agent_run",
        parent_id="run-1",
    )

    assert first["event_id"].startswith("evt_")
    assert first["event_id"] == second["event_id"]
    assert first["dedupe_key"] == first["event_id"]
    assert first["visibility"] == "visible"
    assert first["durability"] == "transient"
    assert first["projection_lane"] == "goal_surface"


def test_normalize_repeated_identical_blockers_keep_distinct_backend_identity() -> None:
    first = normalize_subagent_event(
        {
            "type": "runtime_blocked",
            "event_id": "blocker-event-1",
            "blocker_type": "approval",
            "summary": "Goal is waiting on operator approval.",
            "approval_ids": ["approval-1"],
        },
        parent_type="goal",
        parent_id="goal-1",
    )
    second = normalize_subagent_event(
        {
            "type": "runtime_blocked",
            "event_id": "blocker-event-2",
            "blocker_type": "approval",
            "summary": "Goal is waiting on operator approval.",
            "approval_ids": ["approval-1"],
        },
        parent_type="goal",
        parent_id="goal-1",
    )

    assert first["event_id"] == "blocker-event-1"
    assert second["event_id"] == "blocker-event-2"
    assert first["dedupe_key"] == "blocker-event-1"
    assert second["dedupe_key"] == "blocker-event-2"


def test_normalize_runtime_blocker_event_preserves_real_subagent_identity() -> None:
    event = normalize_subagent_event(
        {
            "type": "runtime_blocked",
            "subagent_id": "sub-approval-1",
            "blocker_type": "approval",
            "summary": "Researcher is waiting on approval.",
        },
        parent_type="agent_run",
        parent_id="run-blocked-1",
    )

    assert event["type"] == "runtime_blocked"
    assert event["subagent_id"] == "sub-approval-1"


def test_normalize_role_error_maps_to_blocked_completion() -> None:
    event = normalize_subagent_event(
        {
            "type": "role_error",
            "role_id": "researcher",
            "stage": "role::researcher",
            "summary": "Waiting for approval before continuing.",
            "blocker_type": "approval",
            "approval_ids": ["approval-2"],
        },
        parent_type="agent_run",
        parent_id="run-2",
    )

    assert event["type"] == "subagent_completed"
    assert event["status"] == "blocked"
    assert event["approval_ids"] == ["approval-2"]
    assert event["subagent_id"] == "run-2:researcher:role-researcher"


def test_normalize_unknown_fields_are_filtered_into_metadata() -> None:
    event = normalize_subagent_event(
        {
            "type": "subagent_progress",
            "role_id": "researcher",
            "content": "Searching references.",
            "stdout": "very long command output",
            "api_key": "secret",
            "custom_flag": True,
        },
        parent_type="chat_turn",
        parent_id="turn-1",
    )

    assert event["type"] == "subagent_progress"
    assert event["content"] == "Searching references."
    assert event["metadata"]["custom_flag"] is True
    assert "stdout" not in event["metadata"]
    assert "api_key" not in event["metadata"]


def test_normalize_prompt_and_tool_events_preserve_safe_fields() -> None:
    prompt_event = normalize_subagent_event(
        {
            "type": "subagent_prompt",
            "subagent_id": "sub-3",
            "role_id": "researcher",
            "model_id": "gpt-test",
            "system_prompt": "System prompt",
            "user_prompt": "User prompt",
            "prompt_preview": "User prompt",
            "execution_profile": "subagent_readonly",
        },
        parent_type="agent_run",
        parent_id="run-3",
    )
    tool_event = normalize_subagent_event(
        {
            "type": "subagent_tool_call",
            "subagent_id": "sub-3",
            "tool_call_id": "call-1",
            "tool_name": "web_search",
            "arguments_preview": "query=...",
            "environment": {"API_TOKEN": "secret"},
        },
        parent_type="agent_run",
        parent_id="run-3",
    )

    assert prompt_event["type"] == "subagent_prompt"
    assert prompt_event["system_prompt"] == "System prompt"
    assert prompt_event["user_prompt"] == "User prompt"
    assert prompt_event["prompt_preview"] == "User prompt"
    assert prompt_event["metadata"]["execution_profile"] == "subagent_readonly"
    assert tool_event["type"] == "subagent_tool_call"
    assert tool_event["tool_call_id"] == "call-1"
    assert tool_event["tool_name"] == "web_search"
    assert tool_event["arguments_preview"] == "query=..."
    assert "environment" not in tool_event["metadata"]


def test_normalize_subagent_message_delivery_event_preserves_status() -> None:
    event = normalize_subagent_event(
        {
            "type": "subagent_message_deferred",
            "subagent_id": "sub-4",
            "role_id": "researcher",
            "message_id": "message-1",
            "delivery_mode": "inject_now",
            "delivery_status": "deferred",
            "reason": "generation_in_progress",
            "content": "Please narrow the scope.",
        },
        parent_type="delegated_task",
        parent_id="task-1",
    )

    assert event["type"] == "subagent_message_deferred"
    assert event["subagent_id"] == "sub-4"
    assert event["message_id"] == "message-1"
    assert event["delivery_mode"] == "inject_now"
    assert event["delivery_status"] == "deferred"
    assert event["reason"] == "generation_in_progress"


def test_normalize_subagent_control_events_preserve_interrupt_and_cancel_fields() -> None:
    event = normalize_subagent_event(
        {
            "type": "subagent_tool_cancel_deferred",
            "subagent_id": "sub-5",
            "role_id": "researcher",
            "message_id": "message-cancel-1",
            "delivery_mode": "inject_now",
            "delivery_status": "queued",
            "delivery_reason": "generation_in_progress",
            "reason": "generation_in_progress",
            "interrupt": True,
            "cancel_current_tool": True,
            "content": "Stop the current command.",
        },
        parent_type="delegated_task",
        parent_id="task-2",
    )

    assert event["type"] == "subagent_tool_cancel_deferred"
    assert event["status"] == "running"
    assert event["message_id"] == "message-cancel-1"
    assert event["delivery_mode"] == "inject_now"
    assert event["delivery_status"] == "queued"
    assert event["delivery_reason"] == "generation_in_progress"
    assert event["interrupt"] is True
    assert event["cancel_current_tool"] is True


def test_normalize_subagent_tool_cancelled_preserves_tool_identity() -> None:
    event = normalize_subagent_event(
        {
            "type": "subagent_tool_cancelled",
            "subagent_id": "sub-6",
            "role_id": "researcher",
            "message_id": "message-cancel-2",
            "tool_call_id": "tool-call-2",
            "tool_name": "exec_command",
            "delivery_mode": "inject_now",
            "delivery_reason": "tool_cancelled",
            "interrupt": True,
            "cancel_current_tool": True,
        },
        parent_type="delegated_task",
        parent_id="task-3",
    )

    assert event["type"] == "subagent_tool_cancelled"
    assert event["status"] == "cancelled"
    assert event["tool_call_id"] == "tool-call-2"
    assert event["tool_name"] == "exec_command"
    assert event["delivery_reason"] == "tool_cancelled"
