from __future__ import annotations

import json

from mochi.agents.adaptive_diagnostics import (
    DIAGNOSTICS_EVENT,
    AdaptiveDiagnosticsAccumulator,
    AdaptiveDiagnosticsRecord,
)
from mochi.api.adaptive_runtime_projection import project_adaptive_runtime
from mochi.learning.failure_attribution import (
    FAILURE_ATTRIBUTION_EVENT,
    FailureAttributionRecord,
)


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


def _diagnostics_event(
    *,
    turn_id: str,
    classification: str = "simple",
    **counters: int,
) -> dict:
    if "model_calls" in counters:
        counters.setdefault(
            "model_usage_observed_calls",
            counters["model_calls"],
        )
        counters.setdefault(
            "model_wall_observed_calls",
            counters["model_calls"],
        )
    accumulator = AdaptiveDiagnosticsAccumulator(**counters)
    event = AdaptiveDiagnosticsRecord.create(
        session_id="projection-session",
        turn_id=turn_id,
        classification=classification,  # type: ignore[arg-type]
        accumulator=accumulator,
        timestamp="2026-07-28T12:34:56+00:00",
    ).to_event()
    return event


def _failure_attribution_event(
    *,
    candidate_id: str,
    turn_id: str,
    transition: str,
) -> dict:
    return FailureAttributionRecord.create(
        candidate_id=candidate_id,
        turn_id=turn_id,
        transition=transition,  # type: ignore[arg-type]
        timestamp="2026-07-28T12:34:56+00:00",
    ).to_event()


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
        [
            {key: value for key, value in newer.items() if key != "sequence"},
            {key: value for key, value in older.items() if key != "sequence"},
        ],
    )
    assert no_sequence_projection["turns"][0]["plan"]["revision"] == 2


def test_projection_bounds_and_redacts_metric_cardinality() -> None:
    events = []
    for index in range(6):
        events.append(
            {
                "type": "session_meta",
                "event": "turn_execution_checkpoint",
                "schema_version": 1,
                "session_id": "projection-session",
                "turn_id": f"turn-{index}",
                "sequence": index + 1,
                "checkpoint": {
                    "revision": 1,
                    "stage": "blocked",
                    "complexity_decision": {
                        "kind": "plan_required",
                        "score": 8,
                        "hard_reason_codes": [
                            f"reason-{index} secret=metric-{index}"
                        ],
                        "soft_reason_codes": [],
                        "advisor_used": False,
                        "advisor_confidence": None,
                    },
                    "blocker_reason": (
                        f"blocker-{index} token=private-{index}"
                    ),
                },
            }
        )

    projection = project_adaptive_runtime(
        "projection-session",
        events,
        max_turns=2,
        max_items=2,
    )

    assert len(projection["turns"]) == 2
    assert len(projection["metrics"]["gate"]["by_reason"]) == 2
    assert len(projection["metrics"]["complexity_decisions"]["by_reason"]) == 2
    assert len(projection["metrics"]["recovery"]["blocked_reasons"]) == 2
    serialized = json.dumps(projection, ensure_ascii=False)
    assert "metric-0" not in serialized
    assert "private-0" not in serialized
    assert "[REDACTED_SECRET]" in serialized


def test_projection_reorders_explicit_sequences_and_preserves_three_turn_order() -> (
    None
):
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
                {
                    "turn_id": "turn-1",
                    "status": "terminal",
                    "terminal_outcome": "completed",
                },
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


def test_projection_marks_terminal_plan_finalization_partial_and_redacts_payload() -> (
    None
):
    activation_denied = {
        "type": "turn_event",
        "phase": "tool_call_result",
        "turn_id": "turn-1",
        "seq": 20,
        "event_id": "turn-1:20",
        "payload": {
            "tool_name": "tool_activate",
            "arguments": {"tool_name": "get_current_time", "token": "sk-secret"},
            "result": "private tool result",
            "error": "private error text",
            "metadata": {
                "error_type": "tool_activation_denied",
                "reason": "not_discoverable",
                "recoverability": "requires_policy_change_or_replanning",
            },
        },
    }
    finalization_error = {
        "type": "turn_event",
        "phase": "final_answer",
        "turn_id": "turn-1",
        "seq": 30,
        "event_id": "turn-1:30",
        "payload": {
            "content": "private final content",
            "hidden_reasoning": "private chain of thought",
            "metadata": {
                "error_type": "plan_finalization_required",
                "recoverability": "partial",
                "reason": "plan_incomplete_at_finalization",
            },
        },
    }
    timeline = {
        "type": "session_meta",
        "event": "session_turn_timeline",
        "session_id": "projection-session",
        "sequence": 40,
        "timeline": {
            "history_current_revision": 4,
            "turns": [
                {
                    "turn_id": "turn-1",
                    "status": "terminal",
                    "terminal_outcome": "completed",
                }
            ],
        },
    }

    projection = project_adaptive_runtime(
        "projection-session",
        [
            _plan_event(revision=1, title="Inspect", sequence=10),
            activation_denied,
            finalization_error,
            timeline,
        ],
    )

    turn = projection["turns"][0]
    assert turn["status"] == "partial"
    assert "plan_finalization_required" in turn["blockers"]
    status_events = [
        event for event in projection["events"] if event["event"] == "turn_status_hint"
    ]
    assert status_events[-1]["payload"] == {
        "status": "partial",
        "blocker_code": "plan_finalization_required",
    }
    serialized = json.dumps(projection, ensure_ascii=False)
    assert "private final content" not in serialized
    assert "private chain of thought" not in serialized
    assert "private tool result" not in serialized
    assert "private error text" not in serialized
    assert "get_current_time" not in serialized
    assert "sk-secret" not in serialized


def test_projection_marks_unresolved_terminal_activation_denial_blocked() -> None:
    activation_denied = {
        "type": "turn_event",
        "phase": "tool_call_result",
        "turn_id": "turn-1",
        "seq": 10,
        "event_id": "turn-1:10",
        "payload": {
            "tool_name": "tool_activate",
            "metadata": {
                "error_type": "tool_activation_denied",
                "reason": "tool_mode_disabled",
                "recoverability": "requires_policy_change_or_replanning",
            },
        },
    }
    running_timeline = {
        "type": "session_meta",
        "event": "session_turn_timeline",
        "session_id": "projection-session",
        "sequence": 20,
        "timeline": {
            "history_current_revision": 2,
            "turns": [{"turn_id": "turn-1", "status": "running"}],
        },
    }
    terminal_timeline = {
        **running_timeline,
        "sequence": 30,
        "timeline": {
            "history_current_revision": 3,
            "turns": [
                {
                    "turn_id": "turn-1",
                    "status": "terminal",
                    "terminal_outcome": "completed",
                }
            ],
        },
    }

    running = project_adaptive_runtime(
        "projection-session", [activation_denied, running_timeline]
    )
    terminal = project_adaptive_runtime(
        "projection-session", [activation_denied, terminal_timeline]
    )

    assert running["turns"][0]["status"] == "running"
    assert terminal["turns"][0]["status"] == "blocked"
    assert "tool_activation_denied" in terminal["turns"][0]["blockers"]


def test_projection_terminal_completed_overrides_active_plan_without_failure() -> None:
    timeline = {
        "type": "session_meta",
        "event": "session_turn_timeline",
        "session_id": "projection-session",
        "sequence": 20,
        "timeline": {
            "history_current_revision": 2,
            "turns": [
                {
                    "turn_id": "turn-1",
                    "status": "terminal",
                    "terminal_outcome": "completed",
                }
            ],
        },
    }

    projection = project_adaptive_runtime(
        "projection-session",
        [_plan_event(revision=1, title="Inspect", sequence=10), timeline],
    )

    assert projection["turns"][0]["plan"]["status"] == "active"
    assert projection["turns"][0]["status"] == "completed"


def test_projection_later_activation_clears_denial_hint() -> None:
    denied = {
        "type": "turn_event",
        "phase": "tool_call_result",
        "turn_id": "turn-1",
        "seq": 10,
        "payload": {
            "tool_name": "tool_activate",
            "metadata": {"error_type": "tool_activation_denied"},
        },
    }
    activated = {
        "type": "turn_event",
        "phase": "tool_call_result",
        "turn_id": "turn-1",
        "seq": 20,
        "payload": {
            "tool_name": "tool_activate",
            "metadata": {
                "status": "tool_activated",
                "activation_authorizes_tool_call": False,
            },
        },
    }
    timeline = {
        "type": "session_meta",
        "event": "session_turn_timeline",
        "session_id": "projection-session",
        "sequence": 30,
        "timeline": {
            "history_current_revision": 3,
            "turns": [
                {
                    "turn_id": "turn-1",
                    "status": "terminal",
                    "terminal_outcome": "completed",
                }
            ],
        },
    }

    projection = project_adaptive_runtime(
        "projection-session", [denied, activated, timeline]
    )

    assert projection["turns"][0]["status"] == "completed"


def test_projection_aggregates_strict_redacted_diagnostics_metrics() -> None:
    simple = _diagnostics_event(
        turn_id="turn-1",
        classification="simple",
        model_calls=1,
        tool_calls=2,
        input_tokens=30,
        output_tokens=7,
        model_wall_ms=40,
        tool_wall_ms=8,
        effectful_plan_guard_blocks=1,
        retrieval_search_queries=2,
        retrieval_zero_match_queries=1,
        retrieval_candidates=3,
        retrieval_activations=1,
        retrieval_schema_count_before_total=5,
        retrieval_schema_count_after_total=7,
        retrieval_schema_token_estimate_before_total=50,
        retrieval_schema_token_estimate_after_total=70,
    )
    complex_event = _diagnostics_event(
        turn_id="turn-2",
        classification="complex",
        model_calls=3,
        tool_calls=1,
        input_tokens=90,
        output_tokens=20,
        model_wall_ms=120,
        tool_wall_ms=12,
        recovery_attempts=1,
        recovery_blocked=1,
        recovery_budget_exhaustions=1,
        recovery_model_calls=2,
        recovery_tool_calls=1,
        recovery_input_tokens=60,
        recovery_output_tokens=10,
        recovery_model_wall_ms=80,
        recovery_tool_wall_ms=12,
    )

    projection = project_adaptive_runtime(
        "projection-session",
        [simple, complex_event],
    )

    assert projection["turns"][0]["diagnostics"] == {
        "classification": "simple",
        "counters": simple["counters"],
    }
    public = [
        event for event in projection["events"] if event["event"] == DIAGNOSTICS_EVENT
    ]
    assert len(public) == 2
    assert public[0]["payload"] == {
        "classification": "simple",
        "counters": simple["counters"],
    }
    metrics = projection["metrics"]
    assert metrics["diagnostics"] == {
        "coverage": "complete",
        "observed_turns": 2,
        "expected_turns": 2,
    }
    for section in ("plan", "retrieval", "recovery"):
        assert metrics[section]["coverage"] == "partial"
        assert metrics[section]["diagnostics_coverage"] == "complete"
        assert metrics[section]["diagnostics_observed_turns"] == 2
        assert metrics[section]["diagnostics_expected_turns"] == 2
    assert metrics["plan"]["effectful_guard_blocks"] == 1
    assert metrics["retrieval"]["search_queries"] == 2
    assert metrics["retrieval"]["zero_match_queries"] == 1
    assert metrics["retrieval"]["candidates"] == 3
    assert metrics["retrieval"]["activations"] == 1
    assert metrics["retrieval"]["schema_count_before_total"] == 5
    assert metrics["retrieval"]["schema_count_after_total"] == 7
    assert metrics["retrieval"]["schema_token_estimate_before_total"] == 50
    assert metrics["retrieval"]["schema_token_estimate_after_total"] == 70
    assert metrics["recovery"]["attempts"] == 1
    assert metrics["recovery"]["blocked"] == 1
    assert metrics["recovery"]["budget_exhausted"] == 1
    assert metrics["recovery"]["extra_model_calls"] == 2
    assert metrics["recovery"]["extra_tool_calls"] == 1
    assert metrics["recovery"]["extra_input_tokens"] == 60
    assert metrics["recovery"]["extra_output_tokens"] == 10
    assert metrics["recovery"]["extra_model_wall_ms"] == 80
    assert metrics["recovery"]["extra_tool_wall_ms"] == 12
    assert metrics["cost"]["by_classification"]["simple"] == {
        "turns": 1,
        "model_calls": 1,
        "model_usage_observed_calls": 1,
        "model_wall_observed_calls": 1,
        "tool_calls": 2,
        "input_tokens": 30,
        "output_tokens": 7,
        "model_wall_ms": 40,
        "tool_wall_ms": 8,
    }
    assert metrics["cost"]["by_classification"]["complex"]["model_calls"] == 3
    assert metrics["cost"]["coverage"] == "complete"
    assert metrics["cost"]["token_coverage"] == "complete"
    assert metrics["cost"]["wall_coverage"] == "complete"


def test_projection_marks_cost_partial_when_model_usage_is_unobserved() -> None:
    projection = project_adaptive_runtime(
        "projection-session",
        [
            _diagnostics_event(
                turn_id="turn-1",
                model_calls=1,
                model_usage_observed_calls=0,
                model_wall_observed_calls=1,
                model_wall_ms=5,
            )
        ],
    )

    assert projection["metrics"]["diagnostics"]["coverage"] == "complete"
    assert projection["metrics"]["cost"]["token_coverage"] == "partial"
    assert projection["metrics"]["cost"]["wall_coverage"] == "complete"
    assert projection["metrics"]["cost"]["coverage"] == "partial"


def test_projection_deduplicates_diagnostics_and_keeps_latest_conflict() -> None:
    first = _diagnostics_event(
        turn_id="turn-1",
        classification="simple",
        model_calls=1,
    )
    conflicting = _diagnostics_event(
        turn_id="turn-1",
        classification="complex",
        model_calls=2,
    )

    projection = project_adaptive_runtime(
        "projection-session",
        [first, dict(first), conflicting],
    )

    assert projection["metrics"]["duplicate_event_count"] == 2
    assert len(projection["events"]) == 1
    assert projection["turns"][0]["diagnostics"]["classification"] == "complex"
    assert projection["metrics"]["cost"]["by_classification"] == {
        "complex": {
            "turns": 1,
            "model_calls": 2,
            "model_usage_observed_calls": 2,
            "model_wall_observed_calls": 2,
            "tool_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "model_wall_ms": 0,
            "tool_wall_ms": 0,
        }
    }


def test_projection_rejects_malformed_diagnostics_and_preserves_legacy_coverage() -> (
    None
):
    malformed = {
        **_diagnostics_event(turn_id="turn-1"),
        "raw_prompt": "secret=must-not-appear",
    }
    future = {
        **_diagnostics_event(turn_id="turn-2"),
        "schema_version": 999,
    }
    missing_counter = _diagnostics_event(turn_id="turn-2")
    del missing_counter["counters"]["model_calls"]
    legacy = _plan_event(revision=1, title="Legacy plan", sequence=30)
    legacy["turn_id"] = "turn-2"
    valid = _diagnostics_event(turn_id="turn-1")

    projection = project_adaptive_runtime(
        "projection-session",
        [malformed, future, missing_counter, legacy, valid],
    )

    assert projection["metrics"]["ignored_event_count"] == 3
    assert projection["metrics"]["diagnostics"] == {
        "coverage": "partial",
        "observed_turns": 1,
        "expected_turns": 2,
    }
    for section in ("plan", "retrieval", "recovery"):
        assert projection["metrics"][section]["coverage"] == "partial"
        assert projection["metrics"][section]["diagnostics_coverage"] == "partial"
        assert projection["metrics"][section]["diagnostics_observed_turns"] == 1
        assert projection["metrics"][section]["diagnostics_expected_turns"] == 2
    assert projection["turns"][0]["diagnostics"]["classification"] == "simple"
    assert projection["turns"][1]["diagnostics"] == {}
    serialized = json.dumps(projection, ensure_ascii=False)
    assert "must-not-appear" not in serialized

    legacy_only = project_adaptive_runtime(
        "projection-session",
        [legacy],
    )
    assert legacy_only["metrics"]["diagnostics"] == {
        "coverage": "partial",
        "observed_turns": 0,
        "expected_turns": 1,
    }


def test_projection_diagnostics_coverage_tracks_scanned_history_when_turns_truncate() -> (
    None
):
    projection = project_adaptive_runtime(
        "projection-session",
        [
            _diagnostics_event(
                turn_id="turn-1",
                classification="simple",
                model_calls=1,
            ),
            _diagnostics_event(
                turn_id="turn-2",
                classification="complex",
                model_calls=2,
            ),
        ],
        max_turns=1,
    )

    assert [turn["turn_id"] for turn in projection["turns"]] == ["turn-2"]
    assert projection["metrics"]["diagnostics"] == {
        "coverage": "complete",
        "observed_turns": 2,
        "expected_turns": 2,
    }
    assert (
        sum(
            bucket["turns"]
            for bucket in projection["metrics"]["cost"]["by_classification"].values()
        )
        == projection["metrics"]["diagnostics"]["observed_turns"]
    )
    for section in ("plan", "retrieval", "recovery"):
        assert projection["metrics"][section]["coverage"] == "partial"
        assert projection["metrics"][section]["diagnostics_coverage"] == "complete"


def test_projection_strictly_attributes_failure_learning_transitions() -> None:
    candidate = _failure_attribution_event(
        candidate_id="candidate-1",
        turn_id="turn-1",
        transition="candidate",
    )
    processed = _failure_attribution_event(
        candidate_id="candidate-1",
        turn_id="turn-1",
        transition="processed",
    )
    rejected = _failure_attribution_event(
        candidate_id="candidate-2",
        turn_id="turn-2",
        transition="rejected",
    )
    hint = _failure_attribution_event(
        candidate_id="candidate-1",
        turn_id="turn-2",
        transition="hint_selected",
    )

    projection = project_adaptive_runtime(
        "projection-session",
        [candidate, dict(candidate), processed, rejected, hint],
    )

    assert projection["metrics"]["duplicate_event_count"] == 1
    assert projection["metrics"]["failure_learning"] == {
        "coverage": "complete",
        "valid_transition_count": 4,
        "ignored_transition_count": 0,
        "candidates": 1,
        "processed": 1,
        "rejected": 1,
        "hints_selected": 1,
    }
    assert projection["turns"][0]["failure_learning"] == {
        "candidate_count": 1,
        "processed_count": 1,
        "rejected_count": 0,
        "hints_selected_count": 0,
    }
    assert projection["turns"][1]["failure_learning"] == {
        "candidate_count": 0,
        "processed_count": 0,
        "rejected_count": 1,
        "hints_selected_count": 1,
    }
    public = [
        event
        for event in projection["events"]
        if event["event"] == FAILURE_ATTRIBUTION_EVENT
    ]
    assert [event["payload"]["transition"] for event in public] == [
        "candidate",
        "processed",
        "rejected",
        "hint_selected",
    ]


def test_projection_rejects_malformed_attribution_and_global_outbox_payload() -> (
    None
):
    malformed = {
        **_failure_attribution_event(
            candidate_id="candidate-1",
            turn_id="turn-1",
            transition="candidate",
        ),
        "raw_prompt": "secret=must-not-appear",
    }
    global_outbox = {
        "type": "session_meta",
        "event": "failure_learning_candidate",
        "candidate_id": "candidate-global",
        "failure_episode": {
            "turn_id": "turn-global",
            "verifier_feedback": ["raw global payload"],
        },
    }

    projection = project_adaptive_runtime(
        "projection-session",
        [malformed, global_outbox],
    )

    assert projection["turns"] == []
    assert projection["metrics"]["ignored_event_count"] == 1
    assert projection["metrics"]["failure_learning"] == {
        "coverage": "partial",
        "valid_transition_count": 0,
        "ignored_transition_count": 1,
        "candidates": 0,
        "processed": 0,
        "rejected": 0,
        "hints_selected": 0,
    }
    serialized = json.dumps(projection, ensure_ascii=False)
    assert "must-not-appear" not in serialized
    assert "raw global payload" not in serialized
