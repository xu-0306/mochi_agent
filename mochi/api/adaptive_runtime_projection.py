"""Replay-safe public projection for ordinary Chat adaptive runtime state.

The session log remains the source of truth.  This module only reads known
durable events and produces a small, redacted display/API projection.  It is
deliberately independent from the model transcript: no assistant content,
tool arguments, tool output, prompts, or hidden reasoning is copied here.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from typing import Any

ADAPTIVE_RUNTIME_PROJECTION_VERSION = "ordinary-chat-adaptive-runtime-v1"
ADAPTIVE_RUNTIME_SCHEMA_VERSION = 1

_KNOWN_EVENTS = frozenset(
    {
        "ordinary_chat_plan_ledger_updated",
        "ordinary_chat_verification_receipt_recorded",
        "failure_learning_candidate",
        "failure_learning_processed",
        "turn_execution_checkpoint",
        "session_turn_timeline",
        "message",
    }
)
_REDACTION_PATTERNS = (
    (re.compile(r"(?i)\b(?:api[_ -]?key|access[_ -]?token|authorization|secret|token)\s*[:=]\s*[^\s,;]+"), "[REDACTED_SECRET]"),
    (re.compile(r"(?i)\bbearer\s+[a-z0-9._~+/=-]+"), "Bearer [REDACTED_SECRET]"),
    (re.compile(r"\b[0-9]{13,19}\b"), "[REDACTED_PAYMENT]"),
    (re.compile(r"\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_CONTACT]"),
)


def project_adaptive_runtime(
    session_id: str,
    events: Iterable[Mapping[str, Any]],
    *,
    max_turns: int = 12,
    max_items: int = 12,
    max_events: int = 128,
) -> dict[str, Any]:
    """Project a session event snapshot into bounded public runtime state.

    Durable aggregate revisions are preferred for ordering.  When an older
    installation has no explicit sequence, the source position is used as a
    deterministic fallback.  Duplicate aggregate updates are collapsed by
    identity while distinct revisions remain available to an SSE/replay
    consumer.
    """

    session_id = _required_text(session_id, "session_id", max_chars=256)
    max_turns = _bounded_limit(max_turns, "max_turns", upper=100)
    max_items = _bounded_limit(max_items, "max_items", upper=100)
    max_events = _bounded_limit(max_events, "max_events", upper=500)

    candidates: list[dict[str, Any]] = []
    ignored_event_count = 0
    for source_position, raw in enumerate(events, start=1):
        if not isinstance(raw, Mapping):
            ignored_event_count += 1
            continue
        timeline = raw.get("timeline")
        event_name = raw.get("event") or raw.get("type")
        if event_name == "session_turn_timeline" and isinstance(timeline, Mapping):
            timeline_turns = timeline.get("turns")
            if isinstance(timeline_turns, (list, tuple)) and not raw.get("turn_id"):
                for timeline_turn in timeline_turns:
                    if not isinstance(timeline_turn, Mapping):
                        continue
                    timeline_turn_id = _text_or_none(timeline_turn.get("turn_id"))
                    if timeline_turn_id is None:
                        continue
                    candidate = _candidate_from_event(
                        session_id=session_id,
                        raw={**raw, "turn_id": timeline_turn_id},
                        source_position=source_position,
                    )
                    if candidate is not None:
                        candidates.append(candidate)
                continue
        candidate = _candidate_from_event(
            session_id=session_id,
            raw=raw,
            source_position=source_position,
        )
        if candidate is not None:
            candidates.append(candidate)

    candidate_count = len(candidates)
    candidates = _deduplicate_candidates(candidates)
    duplicate_event_count = max(0, candidate_count - len(candidates))
    candidates.sort(key=lambda item: (item["order"], item["source_position"], item["identity"]))
    public_events: list[dict[str, Any]] = []
    for candidate in candidates:
        public_events.extend(_public_events_for_candidate(candidate))
    if len(public_events) > max_events:
        public_events = public_events[-max_events:]
        ignored_event_count += len(candidates) - max_events

    turns: dict[str, dict[str, Any]] = {}
    turn_first_order: dict[str, tuple[int, int]] = {}
    metrics: dict[str, Any] = {
        "adaptive_event_count": len(public_events),
        "ignored_event_count": ignored_event_count,
        "duplicate_event_count": duplicate_event_count,
        "gate": {
            "decisions": 0,
            "by_kind": {},
            "by_reason": {},
            "advisor_calls": 0,
            "advisor_timeouts": 0,
            "advisor_malformed": 0,
        },
        "complexity_decisions": {"total": 0, "by_kind": {}, "by_reason": {}},
        "plan": {
            "created": 0,
            "updated": 0,
            "cas_conflicts": 0,
            "effectful_call_guard_blocks": 0,
        },
        "plan_updates": 0,
        "plan_cas_conflicts": 0,
        "retrieval": {
            "turns_with_inventory": 0,
            "eligible_tools": 0,
            "exposed_tools": 0,
            "search_queries": 0,
            "zero_matches": 0,
            "candidates": 0,
            "activations": 0,
        },
        "verification": {
            "receipts": 0,
            "by_verdict": {},
            "semantic_judge_calls": 0,
            "semantic_judge_timeouts": 0,
            "semantic_judge_malformed": 0,
        },
        "recovery": {
            "decisions": 0,
            "attempts": 0,
            "blocked_reasons": {},
            "budget_exhausted": 0,
        },
        "failure_learning": {
            "candidates": 0,
            "processed": 0,
            "rejected": 0,
            "hints_selected": 0,
        },
        "cost": {
            "simple_turns": 0,
            "complex_turns": 0,
            "extra_model_calls": 0,
            "extra_tool_calls": 0,
            "extra_wall_seconds": 0.0,
        },
    }

    for candidate in candidates:
        turn_id = candidate.get("turn_id")
        if not isinstance(turn_id, str) or not turn_id:
            continue
        turn = turns.setdefault(turn_id, _new_turn(turn_id))
        order_key = (int(candidate["order"]), int(candidate["source_position"]))
        turn_first_order.setdefault(turn_id, order_key)
        turn["updated_sequence"] = max(int(turn["updated_sequence"]), int(candidate["order"]))
        _apply_candidate(turn, candidate, metrics, max_items=max_items)

    # Authoritative checkpoint/receipt/timeline state wins over a provisional
    # assistant final that may have streamed immediately before verification.
    for turn in turns.values():
        _finalize_turn_status(turn)
        turn["blockers"] = _bounded_unique(turn["blockers"], max_items=8, max_chars=160)
        turn.pop("_status_hints", None)
        turn.pop("_latest_checkpoint_order", None)
        turn.pop("_latest_receipt_order", None)
        turn.pop("_latest_timeline_order", None)
        turn.pop("_latest_plan_order", None)
        turn.pop("_latest_plan_revision", None)

    ordered_turn_ids = sorted(turns, key=lambda key: turn_first_order[key])
    if len(ordered_turn_ids) > max_turns:
        ordered_turn_ids = ordered_turn_ids[-max_turns:]
    projected_turns = [turns[turn_id] for turn_id in ordered_turn_ids]

    return {
        "projection_version": ADAPTIVE_RUNTIME_PROJECTION_VERSION,
        "schema_version": ADAPTIVE_RUNTIME_SCHEMA_VERSION,
        "session_id": session_id,
        "revision": max((int(item["order"]) for item in candidates), default=0),
        "latest_sequence": max((int(item["order"]) for item in candidates), default=0),
        "events": public_events,
        "turns": projected_turns,
        "metrics": metrics,
    }


def reduce_adaptive_runtime_events(
    session_id: str,
    events: Iterable[Mapping[str, Any]],
    **limits: int,
) -> dict[str, Any]:
    """Compatibility name for callers that treat this module as a reducer."""

    return project_adaptive_runtime(session_id, events, **limits)


def _candidate_from_event(
    *, session_id: str, raw: Mapping[str, Any], source_position: int
) -> dict[str, Any] | None:
    event_name = raw.get("event")
    event_type = raw.get("type")
    if event_name not in _KNOWN_EVENTS and event_type not in _KNOWN_EVENTS:
        return None

    if event_name in _KNOWN_EVENTS:
        kind = str(event_name)
    elif event_type in _KNOWN_EVENTS:
        kind = str(event_type)
    else:
        return None

    raw_session_id = raw.get("session_id")
    if isinstance(raw_session_id, str) and raw_session_id != session_id:
        return None

    payload: Mapping[str, Any] = raw
    turn_id = _text_or_none(raw.get("turn_id"))
    revision = _positive_or_zero_int(
        raw.get("sequence")
        if isinstance(raw.get("sequence"), int) and not isinstance(raw.get("sequence"), bool)
        else raw.get("seq")
    )

    if kind == "turn_execution_checkpoint":
        checkpoint = raw.get("checkpoint")
        if not isinstance(checkpoint, Mapping):
            return None
        payload = checkpoint
        turn_id = turn_id or _text_or_none(checkpoint.get("turn_id"))
        revision = _positive_or_zero_int(checkpoint.get("revision"))
    elif kind in {"ordinary_chat_plan_ledger_updated", "ordinary_chat_verification_receipt_recorded"}:
        nested_name = "plan_ledger" if kind.startswith("ordinary_chat_plan") else "verification_receipt"
        nested = raw.get(nested_name)
        if not isinstance(nested, Mapping):
            return None
        payload = nested
        turn_id = turn_id or _text_or_none(nested.get("turn_id"))
        revision = revision or _positive_or_zero_int(
            raw.get("ledger_revision") if kind.startswith("ordinary_chat_plan") else raw.get("receipt_revision")
        )
    elif kind in {"failure_learning_candidate", "failure_learning_processed"}:
        nested = raw.get("failure_episode")
        if kind == "failure_learning_processed" and not isinstance(nested, Mapping):
            nested = raw.get("candidate")
        if isinstance(nested, Mapping):
            payload = nested
            turn_id = turn_id or _text_or_none(nested.get("turn_id"))
        revision = revision or _positive_or_zero_int(raw.get("attempts"))
    elif kind == "session_turn_timeline":
        timeline = raw.get("timeline")
        if not isinstance(timeline, Mapping):
            return None
        payload = timeline
        turn_id = turn_id or _timeline_turn_id(timeline)
        revision = revision or _positive_or_zero_int(timeline.get("history_current_revision"))
    elif kind == "message":
        turn_id = turn_id or _text_or_none(raw.get("turn_id"))
        if turn_id is None:
            return None
        # A message is only a bounded status hint.  Its content and metadata
        # are deliberately never carried into the projection.
        payload = {
            "role": raw.get("role"),
            "event_type": raw.get("event_type") or raw.get("type"),
            "status": raw.get("status"),
            "error_code": raw.get("error_code"),
            "metadata": raw.get("metadata"),
        }

    if turn_id is None and kind not in {"failure_learning_candidate", "failure_learning_processed"}:
        return None
    explicit_order = _first_int(raw, "sequence", "event_sequence", "source_sequence", "seq")
    order = explicit_order or source_position
    identity_material = {
        "event": kind,
        "session_id": session_id,
        "turn_id": turn_id,
        "revision": revision,
        "idempotency_key": _text_or_none(raw.get("idempotency_key")),
        "event_id": _text_or_none(raw.get("event_id")),
        "candidate_id": _text_or_none(raw.get("candidate_id")),
        "source_position": source_position,
    }
    identity = hashlib.sha256(
        json.dumps(identity_material, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return {
        "kind": kind,
        "turn_id": turn_id,
        "revision": revision,
        "order": order,
        "source_position": source_position,
        "identity": identity,
        "raw": raw,
        "payload": payload,
    }


def _deduplicate_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        raw = candidate["raw"]
        kind = candidate["kind"]
        stable_key: tuple[Any, ...]
        if kind == "ordinary_chat_plan_ledger_updated":
            stable_key = (kind, _text_or_none(raw.get("ledger_id")), candidate["revision"])
        elif kind == "ordinary_chat_verification_receipt_recorded":
            stable_key = (kind, _text_or_none(raw.get("idempotency_key")) or _text_or_none(candidate["payload"].get("receipt_id")))
        elif kind == "turn_execution_checkpoint":
            stable_key = (kind, candidate["turn_id"], candidate["revision"])
        elif kind == "session_turn_timeline":
            stable_key = (kind, candidate["turn_id"], candidate["revision"])
        elif kind in {"failure_learning_candidate", "failure_learning_processed"}:
            stable_key = (kind, _text_or_none(raw.get("candidate_id")), candidate["revision"])
        else:
            stable_key = (kind, candidate["turn_id"], candidate["order"], candidate["source_position"])
        key = json.dumps(stable_key, ensure_ascii=False, separators=(",", ":"), default=str)
        prior = selected.get(key)
        if prior is not None:
            # Keep the candidate with the strongest explicit ordering and use
            # source position only as a deterministic tie breaker.
            if (candidate["order"], candidate["source_position"]) > (prior["order"], prior["source_position"]):
                selected[key] = candidate
        else:
            selected[key] = candidate
    result = list(selected.values())
    return result


def _new_turn(turn_id: str) -> dict[str, Any]:
    return {
        "turn_id": turn_id,
        "status": "running",
        "updated_sequence": 0,
        "complexity": {},
        "plan": None,
        "retrieval": {},
        "evidence": {"status": "not_observed", "receipts": []},
        "recovery": {},
        "failure_learning": {"candidate_count": 0, "processed_count": 0},
        "blockers": [],
        "_status_hints": [],
    }


def _apply_candidate(
    turn: dict[str, Any], candidate: Mapping[str, Any], metrics: dict[str, Any], *, max_items: int
) -> None:
    kind = candidate["kind"]
    raw = candidate["raw"]
    payload = candidate["payload"]
    revision = int(candidate["revision"])
    order = int(candidate["order"])

    if kind == "ordinary_chat_plan_ledger_updated":
        latest_revision = int(turn.get("_latest_plan_revision", -1))
        latest_order = int(turn.get("_latest_plan_order", -1))
        if revision < latest_revision or (revision == latest_revision and order < latest_order):
            return
        turn["_latest_plan_revision"] = revision
        turn["_latest_plan_order"] = order
        turn["plan"] = _public_plan(payload, max_items=max_items)
        metrics["plan_updates"] += 1
        metrics["plan"]["updated"] += 1
        if turn["plan"].get("revision") == 1:
            metrics["plan"]["created"] += 1
        if str(raw.get("idempotency_key") or "").startswith("plan-update:"):
            conflict = int("conflict" in str(raw.get("idempotency_key")))
            metrics["plan_cas_conflicts"] += conflict
            metrics["plan"]["cas_conflicts"] += conflict
        return

    if kind == "ordinary_chat_verification_receipt_recorded":
        receipt = _public_receipt(payload)
        evidence = turn["evidence"]
        receipts = [item for item in evidence.get("receipts", []) if item.get("receipt_id") != receipt.get("receipt_id")]
        receipts.append(receipt)
        evidence["receipts"] = receipts[-8:]
        evidence["status"] = receipt["verdict"]
        turn["_latest_receipt_order"] = max(order, int(turn.get("_latest_receipt_order", -1)))
        _add_blocker(turn, receipt["verdict"] if receipt["verdict"] in {"failed", "unverified"} else None)
        metrics["verification"]["receipts"] += 1
        verdicts = metrics["verification"]["by_verdict"]
        verdicts[receipt["verdict"]] = int(verdicts.get(receipt["verdict"], 0)) + 1
        return

    if kind == "turn_execution_checkpoint":
        if revision < int(turn.get("_latest_checkpoint_order", -1)):
            return
        turn["_latest_checkpoint_order"] = revision
        checkpoint = payload
        stage = _safe_enum(checkpoint.get("stage"), {"contract_resolved", "awaiting_approval", "executing", "verifying", "completed", "blocked"})
        if stage:
            turn["_status_hints"] = [stage]
        complexity = checkpoint.get("complexity_decision")
        if isinstance(complexity, Mapping):
            turn["complexity"] = _public_complexity(complexity)
            _count_complexity(metrics, turn["complexity"])
        ledger = checkpoint.get("plan_ledger_snapshot")
        if isinstance(ledger, Mapping):
            turn["plan"] = _public_plan(ledger, max_items=max_items)
        turn["retrieval"] = _public_retrieval(checkpoint.get("inventory_snapshot"))
        if turn["retrieval"]:
            metrics["retrieval"]["turns_with_inventory"] += 1
            metrics["retrieval"]["eligible_tools"] += int(turn["retrieval"].get("eligible_count", 0))
            metrics["retrieval"]["exposed_tools"] += int(turn["retrieval"].get("exposed_count", 0))
        turn["recovery"] = _public_recovery(checkpoint)
        if turn["recovery"]:
            attempts_used = int(turn["recovery"].get("attempts_used") or 0)
            metrics["recovery"]["decisions"] += int(attempts_used > 0)
            metrics["recovery"]["attempts"] += attempts_used
            metrics["recovery"]["budget_exhausted"] += int(bool(turn["recovery"].get("exhausted_reason")))
        verification = checkpoint.get("verification_result")
        if isinstance(verification, Mapping):
            _apply_checkpoint_verification(turn, verification)
        if stage == "blocked":
            _add_blocker(turn, checkpoint.get("blocker_reason"))
        if stage == "awaiting_approval":
            _add_blocker(turn, "awaiting_approval")
        return

    if kind == "session_turn_timeline":
        timeline_turn = _timeline_turn(payload, turn["turn_id"])
        if timeline_turn is None:
            return
        timeline_order = int(turn.get("_latest_timeline_order", -1))
        current_revision = _positive_or_zero_int(payload.get("history_current_revision"))
        if current_revision < timeline_order:
            return
        turn["_latest_timeline_order"] = current_revision
        status = _safe_enum(timeline_turn.get("status"), {"queued", "running", "terminal", "cancelled"})
        terminal_outcome = _text_or_none(timeline_turn.get("terminal_outcome"))
        cancellation_outcome = _text_or_none(timeline_turn.get("cancellation_outcome"))
        if status == "cancelled" or terminal_outcome == "cancelled" or cancellation_outcome:
            turn["_status_hints"] = ["cancelled"]
            _add_blocker(turn, cancellation_outcome or "cancelled")
        return

    if kind == "failure_learning_candidate":
        turn["failure_learning"]["candidate_count"] += 1
        metrics["failure_learning"]["candidates"] += 1
        for code in _safe_string_list(payload.get("reason_codes"), max_items=8, max_chars=120):
            _add_blocker(turn, code)
        return

    if kind == "failure_learning_processed":
        turn["failure_learning"]["processed_count"] += 1
        metrics["failure_learning"]["processed"] += 1
        status = _text_or_none(raw.get("status")) or _text_or_none(payload.get("status"))
        if status == "hint_selected":
            metrics["failure_learning"]["hints_selected"] += 1
        return

    if kind == "message":
        metadata = payload.get("metadata")
        if isinstance(metadata, Mapping):
            status_hint = _text_or_none(metadata.get("artifact_verification_status"))
            error_type = _text_or_none(metadata.get("error_type"))
            if status_hint in {"failed", "unverified", "blocked"}:
                _add_blocker(turn, status_hint)
            if error_type:
                _add_blocker(turn, error_type)
        error_code = _text_or_none(payload.get("error_code"))
        if error_code:
            _add_blocker(turn, error_code)
        return


def _apply_checkpoint_verification(turn: dict[str, Any], verification: Mapping[str, Any]) -> None:
    receipt = verification.get("aggregate_verification_receipt")
    if isinstance(receipt, Mapping):
        public_receipt = _public_receipt(receipt)
        receipts = [item for item in turn["evidence"].get("receipts", []) if item.get("receipt_id") != public_receipt.get("receipt_id")]
        receipts.append(public_receipt)
        turn["evidence"] = {"status": public_receipt["verdict"], "receipts": receipts[-8:]}
        if public_receipt["verdict"] in {"failed", "unverified"}:
            _add_blocker(turn, public_receipt["verdict"])
    verdict = _text_or_none(verification.get("aggregate_verdict")) or _text_or_none(verification.get("verification_status"))
    if verdict in {"failed", "unverified", "blocked"}:
        turn["evidence"]["status"] = verdict
        _add_blocker(turn, verdict)


def _finalize_turn_status(turn: dict[str, Any]) -> None:
    hints = turn.get("_status_hints", [])
    if "cancelled" in hints:
        turn["status"] = "cancelled"
        return
    evidence_status = turn.get("evidence", {}).get("status")
    if evidence_status == "failed":
        turn["status"] = "blocked"
        return
    if "blocked" in hints:
        turn["status"] = "blocked"
        return
    if evidence_status == "unverified":
        turn["status"] = "partial"
        return
    plan_status = turn.get("plan", {}).get("status") if isinstance(turn.get("plan"), Mapping) else None
    if plan_status == "cancelled":
        turn["status"] = "cancelled"
    elif plan_status == "blocked":
        turn["status"] = "blocked"
    elif plan_status == "completed" or "completed" in hints:
        turn["status"] = "completed"
    elif "awaiting_approval" in hints:
        turn["status"] = "awaiting_approval"
    else:
        turn["status"] = "running"


def _public_event(candidate: Mapping[str, Any]) -> dict[str, Any]:
    kind = str(candidate["kind"])
    payload = candidate["payload"]
    public_payload: dict[str, Any]
    if kind == "ordinary_chat_plan_ledger_updated":
        public_payload = {"plan": _public_plan(payload, max_items=12)}
    elif kind == "complexity_decision":
        public_payload = {"decision": _public_complexity(payload)}
    elif kind == "tool_retrieval_result":
        public_payload = {"retrieval": _public_retrieval(payload)}
    elif kind == "recovery_decision":
        public_payload = {"recovery": _public_recovery(payload)}
    elif kind == "turn_cancelled":
        public_payload = {
            "status": "cancelled",
            "cancellation_outcome": _safe_text(payload.get("cancellation_outcome"), max_chars=120),
        }
    elif kind == "ordinary_chat_verification_receipt_recorded":
        public_payload = {"receipt": _public_receipt(payload)}
    elif kind == "turn_execution_checkpoint":
        checkpoint = payload
        public_payload = {
            "stage": _safe_enum(checkpoint.get("stage"), {"contract_resolved", "awaiting_approval", "executing", "verifying", "completed", "blocked"}),
            "complexity": _public_complexity(checkpoint.get("complexity_decision")),
            "retrieval": _public_retrieval(checkpoint.get("inventory_snapshot")),
            "recovery": _public_recovery(checkpoint),
            "verification_status": _text_or_none((checkpoint.get("verification_result") or {}).get("verification_status")) if isinstance(checkpoint.get("verification_result"), Mapping) else None,
            "blocker_reason": _safe_text(checkpoint.get("blocker_reason"), max_chars=160),
        }
    elif kind == "session_turn_timeline":
        timeline_turn = _timeline_turn(payload, candidate.get("turn_id"))
        public_payload = {
            "status": _safe_enum(timeline_turn.get("status"), {"queued", "running", "terminal", "cancelled"}) if timeline_turn else None,
            "terminal_outcome": _safe_text(timeline_turn.get("terminal_outcome"), max_chars=80) if timeline_turn else None,
            "cancellation_outcome": _safe_text(timeline_turn.get("cancellation_outcome"), max_chars=120) if timeline_turn else None,
        }
    elif kind in {"failure_learning_candidate", "failure_learning_processed"}:
        public_payload = {
            "candidate_id": _safe_text(candidate["raw"].get("candidate_id"), max_chars=128),
            "reason_codes": _safe_string_list(payload.get("reason_codes"), max_items=8, max_chars=120),
            "status": _safe_text(candidate["raw"].get("status") or payload.get("status"), max_chars=80),
        }
    elif kind == "message":
        public_payload = {
            "role": _safe_enum(payload.get("role"), {"user", "assistant", "system"}),
            "event_type": _safe_text(payload.get("event_type"), max_chars=64),
            "status": _safe_text(payload.get("status"), max_chars=64),
            "error_code": _safe_text(payload.get("error_code"), max_chars=120),
        }
    else:
        public_payload = {}
    event_id = "adaptive:v1:" + hashlib.sha256(
        json.dumps(
            {"event": kind, "turn_id": candidate.get("turn_id"), "revision": candidate.get("revision"), "payload": public_payload},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()[:32]
    return {
        "event_id": event_id,
        "event": kind,
        "schema_version": ADAPTIVE_RUNTIME_SCHEMA_VERSION,
        "sequence": int(candidate["order"]),
        "revision": int(candidate["revision"]),
        "turn_id": candidate.get("turn_id"),
        "payload": public_payload,
    }


def _public_events_for_candidate(candidate: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Expose checkpoint-derived summaries as replayable projection events."""

    events = [_public_event(candidate)]
    if candidate["kind"] != "turn_execution_checkpoint":
        return events
    checkpoint = candidate["payload"]
    if not isinstance(checkpoint, Mapping):
        return events
    derived: list[tuple[str, Any]] = []
    complexity = checkpoint.get("complexity_decision")
    if isinstance(complexity, Mapping) and complexity:
        derived.append(("complexity_decision", complexity))
    inventory = checkpoint.get("inventory_snapshot")
    if isinstance(inventory, Mapping) and inventory:
        derived.append(("tool_retrieval_result", inventory))
    recovery = _public_recovery(checkpoint)
    if recovery and (
        recovery.get("attempts_used")
        or recovery.get("exhausted_reason")
        or recovery.get("reason_code")
    ):
        derived.append(("recovery_decision", checkpoint))
    for kind, payload in derived:
        derived_candidate = dict(candidate)
        derived_candidate["kind"] = kind
        derived_candidate["payload"] = payload
        events.append(_public_event(derived_candidate))
    return events


def _public_plan(payload: Mapping[str, Any], *, max_items: int) -> dict[str, Any]:
    items: list[dict[str, Any]] = []
    raw_items = payload.get("items")
    if isinstance(raw_items, (list, tuple)):
        for raw_item in raw_items[:max_items]:
            if not isinstance(raw_item, Mapping):
                continue
            item = {
                "item_id": _safe_text(raw_item.get("item_id"), max_chars=128),
                "title": _safe_text(raw_item.get("title"), max_chars=200),
                "status": _safe_enum(raw_item.get("status"), {"pending", "in_progress", "completed", "blocked", "cancelled"}),
                "dependencies": _safe_string_list(raw_item.get("dependencies"), max_items=8, max_chars=128),
                "success_criteria": _safe_string_list(raw_item.get("success_criteria"), max_items=8, max_chars=240),
                "evidence_refs": _safe_string_list(raw_item.get("evidence_refs"), max_items=8, max_chars=128),
                "blocker_reason": _safe_text(raw_item.get("blocker_reason"), max_chars=160),
                "attempts": _non_negative_int(raw_item.get("attempts")),
            }
            items.append(item)
    return {
        "ledger_id": _safe_text(payload.get("ledger_id"), max_chars=128),
        "revision": _non_negative_int(payload.get("revision")),
        "status": _safe_enum(payload.get("status"), {"active", "completed", "blocked", "cancelled"}),
        "objective": _safe_text(payload.get("objective"), max_chars=240),
        "reason_codes": _safe_string_list(payload.get("reason_codes"), max_items=8, max_chars=120),
        "items": items,
        "blockers": _bounded_unique(
            [item["blocker_reason"] for item in items if item.get("blocker_reason")],
            max_items=8,
            max_chars=160,
        ),
    }


def _public_complexity(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    return {
        "decision_version": _safe_text(payload.get("decision_version"), max_chars=64),
        "kind": _safe_enum(payload.get("kind"), {"no_plan", "plan_required", "continue_existing_plan", "preserve_existing_plan", "blocked_for_clarification"}),
        "score": _bounded_int(payload.get("score"), 0, 100),
        "hard_reason_codes": _safe_string_list(payload.get("hard_reason_codes"), max_items=8, max_chars=120),
        "soft_reason_codes": _safe_string_list(payload.get("soft_reason_codes"), max_items=8, max_chars=120),
        "advisor_used": payload.get("advisor_used") if type(payload.get("advisor_used")) is bool else None,
        "advisor_confidence": _bounded_float(payload.get("advisor_confidence"), 0.0, 1.0),
    }


def _public_retrieval(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        return {}
    eligible = _safe_string_list(payload.get("eligible_tool_names"), max_items=100, max_chars=128)
    exposed = _safe_string_list(payload.get("exposed_tool_names"), max_items=100, max_chars=128)
    activation = _safe_string_list(payload.get("activation_eligible_tool_names"), max_items=100, max_chars=128)
    return {
        "catalog_scope": _safe_text(payload.get("catalog_scope"), max_chars=80),
        "eligible_count": len(eligible),
        "exposed_count": len(exposed),
        "activation_eligible_count": len(activation),
        "inventory_version": _safe_text(payload.get("inventory_version"), max_chars=80),
    }


def _public_recovery(checkpoint: Mapping[str, Any]) -> dict[str, Any]:
    budget = checkpoint.get("recovery_budget")
    receipt = checkpoint.get("execution_receipt")
    controlled = receipt.get("controlled_recovery") if isinstance(receipt, Mapping) else None
    if not isinstance(budget, Mapping) and not isinstance(controlled, Mapping):
        return {}
    remaining_attempts = _non_negative_int(budget.get("remaining_attempts")) if isinstance(budget, Mapping) else None
    remaining_model = _non_negative_int(budget.get("remaining_extra_model_calls")) if isinstance(budget, Mapping) else None
    remaining_tools = _non_negative_int(budget.get("remaining_extra_tool_calls")) if isinstance(budget, Mapping) else None
    remaining_wall = _bounded_float(budget.get("remaining_extra_wall_seconds"), 0.0, 3600.0) if isinstance(budget, Mapping) else None
    exhausted = _safe_text(
        (controlled or {}).get("exhausted_reason") if isinstance(controlled, Mapping) else None,
        max_chars=120,
    )
    if exhausted is None and remaining_attempts == 0:
        exhausted = "recovery_attempt_budget_exhausted"
    return {
        "attempts_used": _non_negative_int((controlled or {}).get("replans_used")) if isinstance(controlled, Mapping) else None,
        "remaining_attempts": remaining_attempts,
        "remaining_model_calls": remaining_model,
        "remaining_tool_calls": remaining_tools,
        "remaining_wall_seconds": remaining_wall,
        "exhausted_reason": exhausted,
        "reason_code": _safe_text((controlled or {}).get("reason_code") if isinstance(controlled, Mapping) else None, max_chars=120),
    }


def _public_receipt(payload: Mapping[str, Any]) -> dict[str, Any]:
    criteria: list[dict[str, Any]] = []
    raw_criteria = payload.get("criteria")
    if isinstance(raw_criteria, (list, tuple)):
        for raw in raw_criteria[:12]:
            if not isinstance(raw, Mapping):
                continue
            criteria.append(
                {
                    "criterion_id": _safe_text(raw.get("criterion_id"), max_chars=128),
                    "verdict": _safe_enum(raw.get("verdict"), {"verified", "failed", "unverified", "not_applicable"}),
                    "verifier_id": _safe_text(raw.get("verifier_id"), max_chars=128),
                    "evidence_refs": _safe_string_list(raw.get("evidence_refs"), max_items=8, max_chars=128),
                    "reason_code": _safe_text(raw.get("reason_code"), max_chars=120),
                    "retry_disposition": _safe_text(raw.get("retry_disposition"), max_chars=80),
                }
            )
    return {
        "receipt_id": _safe_text(payload.get("receipt_id"), max_chars=128),
        "turn_id": _safe_text(payload.get("turn_id"), max_chars=128),
        "verdict": _safe_enum(payload.get("verdict"), {"verified", "failed", "unverified", "not_applicable"}) or "unverified",
        "hard_failure": payload.get("hard_failure") if type(payload.get("hard_failure")) is bool else False,
        "retry_disposition": _safe_text(payload.get("retry_disposition"), max_chars=80),
        "criteria": criteria,
    }


def _timeline_turn(timeline: Mapping[str, Any], turn_id: Any) -> Mapping[str, Any] | None:
    turns = timeline.get("turns")
    if not isinstance(turns, (list, tuple)):
        return None
    for item in turns:
        if isinstance(item, Mapping) and (turn_id is None or item.get("turn_id") == turn_id):
            return item
    return None


def _timeline_turn_id(timeline: Mapping[str, Any]) -> str | None:
    item = _timeline_turn(timeline, None)
    return _text_or_none(item.get("turn_id")) if item else None


def _count_complexity(metrics: dict[str, Any], complexity: Mapping[str, Any]) -> None:
    metrics["complexity_decisions"]["total"] += 1
    metrics["gate"]["decisions"] += 1
    kind = complexity.get("kind")
    if isinstance(kind, str):
        by_kind = metrics["complexity_decisions"]["by_kind"]
        by_kind[kind] = int(by_kind.get(kind, 0)) + 1
        gate_by_kind = metrics["gate"]["by_kind"]
        gate_by_kind[kind] = int(gate_by_kind.get(kind, 0)) + 1
    for reason in list(complexity.get("hard_reason_codes", [])) + list(complexity.get("soft_reason_codes", [])):
        if isinstance(reason, str):
            by_reason = metrics["complexity_decisions"]["by_reason"]
            by_reason[reason] = int(by_reason.get(reason, 0)) + 1
            gate_by_reason = metrics["gate"]["by_reason"]
            gate_by_reason[reason] = int(gate_by_reason.get(reason, 0)) + 1


def _add_blocker(turn: dict[str, Any], value: Any) -> None:
    text = _safe_text(value, max_chars=160)
    if text and text not in turn["blockers"]:
        turn["blockers"].append(text)


def _first_int(payload: Mapping[str, Any], *names: str) -> int | None:
    for name in names:
        value = payload.get(name)
        if type(value) is int and value > 0:
            return value
    return None


def _bounded_limit(value: Any, field_name: str, *, upper: int) -> int:
    if type(value) is not int or value < 1 or value > upper:
        raise ValueError(f"{field_name} must be between 1 and {upper}")
    return value


def _required_text(value: Any, field_name: str, *, max_chars: int) -> str:
    text = _safe_text(value, max_chars=max_chars)
    if not text:
        raise ValueError(f"{field_name} must be a non-empty string")
    return text


def _safe_text(value: Any, *, max_chars: int) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = re.sub(r"\s+", " ", value).strip()
    for pattern, replacement in _REDACTION_PATTERNS:
        text = pattern.sub(replacement, text)
    return text[:max_chars]


def _text_or_none(value: Any) -> str | None:
    return _safe_text(value, max_chars=256)


def _safe_string_list(value: Any, *, max_items: int, max_chars: int) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    result: list[str] = []
    for item in value:
        text = _safe_text(item, max_chars=max_chars)
        if text and text not in result:
            result.append(text)
        if len(result) >= max_items:
            break
    return result


def _bounded_unique(values: Iterable[Any], *, max_items: int, max_chars: int) -> list[str]:
    return _safe_string_list(list(values), max_items=max_items, max_chars=max_chars)


def _safe_enum(value: Any, allowed: set[str]) -> str | None:
    return value if isinstance(value, str) and value in allowed else None


def _positive_or_zero_int(value: Any) -> int:
    return value if type(value) is int and value >= 0 else 0


def _non_negative_int(value: Any) -> int | None:
    return value if type(value) is int and value >= 0 else None


def _bounded_int(value: Any, lower: int, upper: int) -> int | None:
    return value if type(value) is int and lower <= value <= upper else None


def _bounded_float(value: Any, lower: float, upper: float) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if lower <= float(value) <= upper:
        return float(value)
    return None


__all__ = [
    "ADAPTIVE_RUNTIME_PROJECTION_VERSION",
    "ADAPTIVE_RUNTIME_SCHEMA_VERSION",
    "project_adaptive_runtime",
    "reduce_adaptive_runtime_events",
]
