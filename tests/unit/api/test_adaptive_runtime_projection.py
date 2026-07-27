from __future__ import annotations

import json

from mochi.api.adaptive_runtime_projection import project_adaptive_runtime


def _plan_event(*, revision: int, title: str, sequence: int) -> dict:
    return {
        "type": "session_meta",
        "event": "ordinary_chat_plan_ledger_updated",
        "schema_version": 1,
        "session_id": "projection-session",
        "goal_id": "goal:turn-1",
        "ledger_id": "plan:1",
        "ledger_revision": revision,
        "turn_id": "turn-1",
        "idempotency_key": f"plan-update:{revision}",
        "sequence": sequence,
        "plan_ledger": {
            "ledger_version": "plan-ledger-v1",
            "ledger_id": "plan:1",
            "session_id": "projection-session",
            "goal_id": "goal:turn-1",
            "revision": revision,
            "status": "active" if revision == 1 else "blocked",
            "objective": title,
            "reason_codes": ["multiple_deliverables"],
            "items": [
                {
                    "item_id": "item-1",
                    "title": title,
                    "status": "blocked" if revision > 1 else "in_progress",
                    "dependencies": [],
                    "success_criteria": ["A bounded criterion"],
                    "source_turn_ids": ["turn-1"],
                    "evidence_refs": [],
                    "blocker_reason": "awaiting approval" if revision > 1 else None,
                    "attempts": 1,
                }
            ],
            "created_turn_id": "turn-1",
            "updated_turn_id": "turn-1",
        },
        "timestamp": f"2026-07-27T00:00:0{revision}+00:00",
    }


def test_projection_is_bounded_redacted_and_revision_aware() -> None:
    newer = _plan_event(
        revision=2,
        title="Publish token=sk-live-should-not-appear",
        sequence=20,
    )
    older = _plan_event(revision=1, title="Inspect workspace", sequence=10)
    checkpoint = {
        "type": "session_meta",
        "event": "turn_execution_checkpoint",
        "schema_version": 1,
        "session_id": "projection-session",
        "turn_id": "turn-1",
        "sequence": 30,
        "checkpoint": {
            "checkpoint_version": "turn-checkpoint-v1",
            "session_id": "projection-session",
            "turn_id": "turn-1",
            "revision": 3,
            "stage": "blocked",
            "turn_intent_contract": {"message": "secret=do-not-copy"},
            "capability_plan": {},
            "complexity_decision": {
                "decision_version": "complexity-decision-v1",
                "kind": "plan_required",
                "score": 8,
                "hard_reason_codes": ["multi_step"],
                "soft_reason_codes": [],
                "advisor_used": False,
                "advisor_confidence": None,
            },
            "inventory_snapshot": {
                "catalog_scope": "policy_eligible",
                "eligible_tool_names": ["file_read", "file_write"],
                "exposed_tool_names": ["file_read"],
                "activation_eligible_tool_names": [],
                "inventory_version": "sha256:inventory",
            },
            "recovery_budget": {
                "budget_version": "recovery-budget-v1",
                "remaining_attempts": 0,
                "remaining_extra_model_calls": 0,
                "remaining_extra_tool_calls": 0,
                "remaining_extra_wall_seconds": 0.0,
            },
            "verification_result": {
                "verification_status": "failed",
                "errors": ["raw output must never be projected"],
            },
            "blocker_reason": "verification_failed",
            "completion_reason": None,
        },
    }
    duplicate = dict(newer)

    projection = project_adaptive_runtime(
        "projection-session",
        [newer, older, duplicate, checkpoint],
        max_events=8,
    )

    turn = projection["turns"][0]
    assert turn["turn_id"] == "turn-1"
    assert turn["status"] == "blocked"
    assert turn["plan"]["revision"] == 2
    assert turn["retrieval"] == {
        "catalog_scope": "policy_eligible",
        "eligible_count": 2,
        "exposed_count": 1,
        "activation_eligible_count": 0,
        "inventory_version": "sha256:inventory",
    }
    assert projection["metrics"]["duplicate_event_count"] == 1
    serialized = json.dumps(projection, ensure_ascii=False)
    assert "sk-live" not in serialized
    assert "do-not-copy" not in serialized
    assert "raw output must never be projected" not in serialized

    no_sequence_projection = project_adaptive_runtime(
        "projection-session",
        [{key: value for key, value in newer.items() if key != "sequence"},
         {key: value for key, value in older.items() if key != "sequence"}],
    )
    assert no_sequence_projection["turns"][0]["plan"]["revision"] == 2


def test_projection_reorders_explicit_sequences_and_preserves_three_turn_order() -> None:
    events = [
        {"type": "message", "role": "assistant", "turn_id": "turn-3", "sequence": 30},
        {"type": "message", "role": "assistant", "turn_id": "turn-1", "sequence": 10},
        {"type": "message", "role": "assistant", "turn_id": "turn-2", "sequence": 20},
    ]

    projection = project_adaptive_runtime("projection-session", events)

    assert [item["turn_id"] for item in projection["turns"]] == [
        "turn-1",
        "turn-2",
        "turn-3",
    ]
    assert projection["latest_sequence"] == 30


def test_projection_marks_cancelled_turn_from_replayed_timeline() -> None:
    timeline = {
        "type": "session_meta",
        "event": "session_turn_timeline",
        "session_id": "projection-session",
        "sequence": 40,
        "timeline": {
            "timeline_version": "session-turn-timeline-v4",
            "session_id": "projection-session",
            "history_current_revision": 4,
            "turns": [
                {"turn_id": "turn-1", "status": "terminal", "terminal_outcome": "completed"},
                {
                    "turn_id": "turn-2",
                    "status": "cancelled",
                    "terminal_outcome": "cancelled",
                    "cancellation_outcome": "cancelled_queued",
                },
            ],
        },
    }

    projection = project_adaptive_runtime("projection-session", [timeline])

    assert [item["turn_id"] for item in projection["turns"]] == ["turn-1", "turn-2"]
    assert projection["turns"][1]["status"] == "cancelled"
    assert "cancelled_queued" in projection["turns"][1]["blockers"]
