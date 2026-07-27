from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from mochi.agents.plan_ledger import (
    PLAN_LEDGER_EVENT,
    PLAN_LEDGER_VERSION,
    PlanItem,
    PlanLedger,
    PlanLedgerRepository,
    PlanLedgerTransitionValidator,
)
from mochi.sessions.store import SessionStore


def _item(
    *,
    item_id: str = "item-1",
    title: str = "Do the thing",
    status: str = "pending",
    dependencies: tuple[str, ...] = (),
    evidence_refs: tuple[str, ...] = (),
    blocker_reason: str | None = None,
    attempts: int = 0,
) -> PlanItem:
    return PlanItem(
        item_id=item_id,
        title=title,
        status=status,  # type: ignore[arg-type]
        dependencies=dependencies,
        success_criteria=("done",),
        source_turn_ids=("turn-1",),
        evidence_refs=evidence_refs,
        blocker_reason=blocker_reason,
        attempts=attempts,
    )


def _ledger(
    *,
    ledger_id: str = "plan-1",
    session_id: str = "session-1",
    goal_id: str = "goal-1",
    revision: int = 0,
    status: str = "active",
    items: tuple[PlanItem, ...] | None = None,
    created_turn_id: str = "turn-1",
    updated_turn_id: str = "turn-1",
) -> PlanLedger:
    return PlanLedger(
        ledger_version=PLAN_LEDGER_VERSION,
        ledger_id=ledger_id,
        session_id=session_id,
        goal_id=goal_id,
        revision=revision,
        status=status,  # type: ignore[arg-type]
        objective="Build the selected artifact",
        reason_codes=("multiple_deliverables",),
        items=items or (_item(),),
        created_turn_id=created_turn_id,
        updated_turn_id=updated_turn_id,
    )


def test_plan_item_and_ledger_round_trip_strictly() -> None:
    ledger = _ledger(
        items=(
            _item(item_id="item-1"),
            _item(item_id="item-2", dependencies=("item-1",)),
        )
    )

    assert PlanItem.from_dict(ledger.items[0].to_dict()) == ledger.items[0]
    assert PlanLedger.from_dict(ledger.to_dict()) == ledger

    invalid = ledger.to_dict()
    invalid["ledger_version"] = "plan-ledger-v2"
    with pytest.raises(ValueError, match="unsupported"):
        PlanLedger.from_dict(invalid)

    invalid_extra = ledger.to_dict()
    invalid_extra["unexpected"] = True
    with pytest.raises(ValueError, match="unexpected"):
        PlanLedger.from_dict(invalid_extra)


def test_plan_ledger_rejects_unknown_dependencies_cycles_and_multiple_in_progress() -> None:
    with pytest.raises(ValueError, match="unknown dependency"):
        _ledger(items=(_item(item_id="item-1", dependencies=("missing",)),))

    with pytest.raises(ValueError, match="acyclic"):
        _ledger(
            items=(
                _item(item_id="item-1", dependencies=("item-2",)),
                _item(item_id="item-2", dependencies=("item-1",)),
            )
        )

    with pytest.raises(ValueError, match="at most one"):
        _ledger(
            items=(
                _item(item_id="item-1", status="in_progress"),
                _item(item_id="item-2", status="in_progress"),
            )
        )


def test_plan_items_require_evidence_and_blocker_reason() -> None:
    with pytest.raises(ValueError, match="evidence_refs"):
        _item(status="completed")

    with pytest.raises(ValueError, match="blocker_reason"):
        _item(status="blocked")


def test_transition_validator_enforces_recognized_evidence_and_terminal_items() -> None:
    validator = PlanLedgerTransitionValidator(recognized_evidence_refs={"receipt-1", "receipt-2"})
    valid = _ledger(items=(_item(status="completed", evidence_refs=("receipt-1",)),))
    validator.validate_replacement(previous=None, proposed=valid)

    invalid = _ledger(items=(_item(status="completed", evidence_refs=("invented",)),))
    with pytest.raises(ValueError, match="host-recognized evidence_refs"):
        validator.validate_replacement(previous=None, proposed=invalid)

    prior = _ledger(
        items=(
            _item(item_id="item-1", status="completed", evidence_refs=("receipt-1",)),
            _item(item_id="item-2"),
        )
    )
    with pytest.raises(ValueError, match="terminal plan item"):
        validator.set_item_status(
            prior,
            item_id="item-1",
            status="completed",
            updated_turn_id="turn-2",
            evidence_refs=("receipt-2",),
        )


@pytest.mark.asyncio
async def test_repository_idempotent_replay_does_not_append_twice(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    repository = PlanLedgerRepository(store)
    ledger = _ledger(session_id="session-replay", goal_id="goal-replay")

    first = await repository.save(
        ledger,
        expected_revision=0,
        turn_id="turn-1",
        idempotency_key="plan-update:replay",
        timestamp="2026-07-26T01:00:00+00:00",
    )
    second = await repository.save(
        ledger,
        expected_revision=0,
        turn_id="turn-1",
        idempotency_key="plan-update:replay",
        timestamp="2026-07-26T01:00:01+00:00",
    )

    assert first.status == "saved"
    assert first.ledger is not None
    assert first.ledger.revision == 1
    assert second.status == "saved"
    assert second.idempotent_replay is True
    events = await store.load_session("session-replay")
    assert [event.get("event") for event in events].count(PLAN_LEDGER_EVENT) == 1


@pytest.mark.asyncio
async def test_repository_has_one_cas_winner_across_instances(tmp_path) -> None:
    sessions_dir = tmp_path / "sessions"
    first = PlanLedgerRepository(SessionStore(sessions_dir))
    second = PlanLedgerRepository(SessionStore(sessions_dir))
    base = _ledger(session_id="session-cas", goal_id="goal-cas")

    results = await asyncio.gather(
        first.save(
            base,
            expected_revision=0,
            turn_id="turn-a",
            idempotency_key="plan-update:a",
        ),
        second.save(
            replace(base, objective="Competing writer"),
            expected_revision=0,
            turn_id="turn-b",
            idempotency_key="plan-update:b",
        ),
    )

    assert [result.status for result in results].count("saved") == 1
    assert [result.status for result in results].count("conflict") == 1


@pytest.mark.asyncio
async def test_repository_latest_invalid_event_fails_closed(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    repository = PlanLedgerRepository(store)
    ledger = _ledger(session_id="session-invalid", goal_id="goal-invalid")

    saved = await repository.save(
        ledger,
        expected_revision=0,
        turn_id="turn-1",
        idempotency_key="plan-update:ok",
    )
    assert saved.status == "saved"

    await store.save_event(
        "session-invalid",
        {
            "type": "session_meta",
            "event": PLAN_LEDGER_EVENT,
            "schema_version": 1,
            "session_id": "session-invalid",
            "goal_id": "goal-invalid",
            "ledger_id": "plan-1",
            "ledger_revision": 2,
            "turn_id": "turn-2",
            "idempotency_key": "plan-update:bad",
            "plan_ledger": {"ledger_version": "plan-ledger-v999"},
            "timestamp": "2026-07-26T02:00:00+00:00",
        },
    )

    loaded = await repository.load("session-invalid", "goal-invalid", ledger_id="plan-1")

    assert loaded.status == "invalid"

