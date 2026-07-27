from __future__ import annotations

import asyncio
from dataclasses import replace

import pytest

from mochi.agents.conversation_state_store import (
    ACTIVE_TASK_STATE_EVENT_VERSION,
    TURN_CHECKPOINT_VERSION,
    ConversationStateRepository,
    TurnCheckpoint,
    TurnCheckpointRepository,
)
from mochi.agents.turn_intent_contract import (
    ActiveTaskState,
    DeliverableContract,
    IntentAdvisory,
    IntentConstraint,
    IntentEvidence,
    ResolvedReference,
    TurnIntentContract,
)
from mochi.sessions.store import SessionStore


def _active_task(*, status: str = "active") -> ActiveTaskState:
    deliverable_status = "cancelled" if status == "cancelled" else "pending"
    return ActiveTaskState(
        goal_id="goal-durable",
        objective="Build the selected workspace artifact",
        status=status,  # type: ignore[arg-type]
        operations=(
            frozenset()
            if status == "cancelled"
            else frozenset({"workspace_read", "workspace_write"})
        ),
        mutation_requirement="forbidden" if status == "cancelled" else "required",
        deliverables=(
            DeliverableContract(
                kind="workspace_artifact",
                target_hint="output/report.md",
                required=True,
                acceptance_criteria=("contains a summary",),
                status=deliverable_status,  # type: ignore[arg-type]
                source_turn_ids=("turn-start",),
            ),
        ),
        positive_constraints=(
            IntentConstraint("Use the selected format", ("turn-start",)),
        ),
        negative_constraints=(
            IntentConstraint("Do not edit unrelated files", ("turn-start",)),
        ),
        decisions=("Selected option B",),
        source_turn_ids=("turn-start", "turn-next"),
        updated_turn_id="turn-next",
    )


def _turn_contract() -> TurnIntentContract:
    deliverable = DeliverableContract(
        kind="workspace_artifact",
        target_hint="output/report.md",
        acceptance_criteria=(
            "contains a summary",
            {
                "schema_version": 1,
                "kind": "tool_execution",
                "check": "test",
                "tool_name": "exec_command",
                "profile_id": "pytest",
                "expected_exit_code": 0,
            },
        ),
        source_turn_ids=("turn-start",),
    )


    return TurnIntentContract(
        turn_id="turn-next",
        active_goal_id="goal-durable",
        objective="Build the selected workspace artifact",
        current_speech_act="request_execution",
        operations=frozenset({"workspace_read", "workspace_write"}),
        deliverables=(deliverable,),
        resolved_references=(
            ResolvedReference(
                surface="option B",
                resolved_to="the selected workspace artifact",
                source_turn_ids=("turn-start",),
            ),
        ),
        positive_constraints=(
            IntentConstraint("Use the selected format", ("turn-start",)),
        ),
        negative_constraints=(
            IntentConstraint("Do not edit unrelated files", ("turn-start",)),
        ),
        mutation_requirement="required",
        clarification=None,
        supersedes_previous_goal=False,
        cancels_active_goal=False,
        modifies_active_task=True,
        confidence=0.94,
        evidence=(
            IntentEvidence(
                statement="The durable task resolves option B.",
                source="active_task",
                source_turn_ids=("turn-start",),
            ),
        ),
        advisories=(
            IntentAdvisory(
                label="workspace_write",
                confidence=0.61,
                rationale="Classifier telemetry only.",
                recommended_operations=frozenset({"workspace_write"}),
                source_turn_ids=("turn-next",),
            ),
        ),
    )


def _checkpoint(
    *,
    session_id: str = "session-checkpoint",
    turn_id: str = "turn-checkpoint",
    stage: str = "contract_resolved",
    allowed_tool: str = "file_write",
) -> TurnCheckpoint:
    return TurnCheckpoint(
        session_id=session_id,
        turn_id=turn_id,
        revision=0,
        stage=stage,  # type: ignore[arg-type]
        turn_intent_contract=_turn_contract().to_dict(),
        capability_plan={
            "plan_version": "capability-plan-v1",
            "turn_id": turn_id,
            "exposed_tools": [allowed_tool],
        },
        active_goal_id="goal-durable",
        policy_snapshot={
            "policy_snapshot_id": "policy-test",
            "policy_version": "sha256:test",
        },
        inventory_snapshot={"inventory_version": "inventory-test"},
        activation_state={"allowed_tool_names": [allowed_tool]},
        complexity_decision={"kind": "no_plan", "score": 1},
        plan_ledger_snapshot={"ledger_id": "plan-1", "status": "active"},
        verification_plan={"criteria": [{"kind": "artifact"}]},
        recovery_budget={"remaining_attempts": 1, "remaining_extra_tool_calls": 4},
        resume_cursor={"turn_id": turn_id, "phase": "contract"},
        completion_reason=("verified" if stage == "completed" else None),
        blocker_reason=("blocked_for_test" if stage == "blocked" else None),
    )


def test_turn_contract_types_strictly_round_trip() -> None:
    state = _active_task()
    contract = _turn_contract()

    assert ActiveTaskState.from_dict(state.to_dict()) == state
    assert TurnIntentContract.from_dict(contract.to_dict()) == contract

    invalid_state = state.to_dict()
    invalid_state["unexpected_grant"] = "workspace_write"
    with pytest.raises(ValueError, match="unexpected fields"):
        ActiveTaskState.from_dict(invalid_state)

    invalid_contract = contract.to_dict()
    invalid_contract["modifies_active_task"] = "true"
    with pytest.raises(TypeError, match="modifies_active_task"):
        TurnIntentContract.from_dict(invalid_contract)

    invalid_criterion = contract.to_dict()
    invalid_criterion["deliverables"][0]["acceptance_criteria"][1]["unexpected"] = True
    with pytest.raises(ValueError, match="unexpected fields"):
        TurnIntentContract.from_dict(invalid_criterion)


@pytest.mark.asyncio
async def test_repository_round_trips_across_process_reload(tmp_path) -> None:
    sessions_dir = tmp_path / "sessions"
    first_repository = ConversationStateRepository(SessionStore(sessions_dir))
    state = _active_task()
    contract = _turn_contract()

    await first_repository.save(
        "session-round-trip",
        active_task=state,
        turn_intent=contract,
        timestamp="2026-07-24T01:02:03+00:00",
    )

    reloaded_repository = ConversationStateRepository(SessionStore(sessions_dir))
    result = await reloaded_repository.load("session-round-trip")

    assert result.active_task == state
    assert result.turn_intent == contract
    assert result.diagnostics.status == "loaded"
    assert result.diagnostics.event_schema_version == ACTIVE_TASK_STATE_EVENT_VERSION
    assert result.diagnostics.messages == ()


@pytest.mark.asyncio
@pytest.mark.parametrize("status", ["cancelled", "completed"])
async def test_repository_rebuilds_terminal_task_states(tmp_path, status: str) -> None:
    repository = ConversationStateRepository(SessionStore(tmp_path / "sessions"))
    state = _active_task(status=status)

    await repository.save("session-terminal", active_task=state)
    result = await ConversationStateRepository(
        SessionStore(tmp_path / "sessions")
    ).load("session-terminal")

    assert result.active_task == state
    assert result.active_task is not None
    assert result.active_task.status == status


@pytest.mark.asyncio
async def test_latest_corrupt_event_fails_closed_without_reviving_write_state(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    repository = ConversationStateRepository(store)
    await repository.save("session-corrupt", active_task=_active_task())
    corrupt_state = _active_task().to_dict()
    corrupt_state["mutation_requirement"] = "forbidden"
    await store.save_event(
        "session-corrupt",
        {
            "type": "session_meta",
            "event": "active_task_state_updated",
            "schema_version": ACTIVE_TASK_STATE_EVENT_VERSION,
            "session_id": "session-corrupt",
            "state_revision": 2,
            "active_task_state": corrupt_state,
            "turn_intent_contract": None,
            "timestamp": "2026-07-24T02:00:00+00:00",
        },
    )

    result = await repository.load("session-corrupt")

    assert result.active_task is None
    assert result.turn_intent is None
    assert result.diagnostics.status == "invalid"
    assert "workspace_write" in " ".join(result.diagnostics.messages)


@pytest.mark.asyncio
async def test_latest_future_event_version_fails_closed(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    repository = ConversationStateRepository(store)
    await repository.save("session-future", active_task=_active_task())
    await store.save_event(
        "session-future",
        {
            "type": "session_meta",
            "event": "active_task_state_updated",
            "schema_version": ACTIVE_TASK_STATE_EVENT_VERSION + 1,
            "session_id": "session-future",
            "active_task_state": _active_task().to_dict(),
            "turn_intent_contract": None,
            "timestamp": "2026-07-24T03:00:00+00:00",
        },
    )

    result = await repository.load("session-future")

    assert result.active_task is None
    assert result.diagnostics.status == "unsupported_version"
    assert result.diagnostics.event_schema_version == ACTIVE_TASK_STATE_EVENT_VERSION + 1


@pytest.mark.asyncio
async def test_repository_reports_missing_state(tmp_path) -> None:
    result = await ConversationStateRepository(
        SessionStore(tmp_path / "sessions")
    ).load("missing-session")

    assert result.active_task is None
    assert result.turn_intent is None
    assert result.diagnostics.status == "missing"


@pytest.mark.asyncio
async def test_advisory_payload_cannot_change_durable_mutation_semantics(tmp_path) -> None:
    read_only_state = ActiveTaskState(
        goal_id="goal-read-only",
        objective="Inspect the workspace without modification",
        operations=frozenset({"workspace_read"}),
        mutation_requirement="forbidden",
        source_turn_ids=("turn-read",),
        updated_turn_id="turn-read",
    )
    advisory_contract = replace(
        _turn_contract(),
        active_goal_id="goal-read-only",
        objective=read_only_state.objective,
        operations=frozenset({"workspace_read"}),
        deliverables=(),
        mutation_requirement="forbidden",
        advisories=(
            IntentAdvisory(
                label="workspace_write",
                confidence=1.0,
                rationale="Untrusted serialized advisory.",
                recommended_operations=frozenset({"workspace_write"}),
                source_turn_ids=("turn-read",),
            ),
        ),
    )
    repository = ConversationStateRepository(SessionStore(tmp_path / "sessions"))

    await repository.save(
        "session-read-only",
        active_task=read_only_state,
        turn_intent=advisory_contract,
    )
    result = await repository.load("session-read-only")

    assert result.active_task is not None
    assert result.active_task.operations == frozenset({"workspace_read"})
    assert result.active_task.mutation_requirement == "forbidden"
    assert result.turn_intent is not None
    assert result.turn_intent.operations == frozenset({"workspace_read"})
    assert result.turn_intent.advisories[0].recommended_operations == frozenset(
        {"workspace_write"}
    )


@pytest.mark.asyncio
async def test_corrupt_serialized_contract_does_not_invalidate_safe_active_state(tmp_path) -> None:
    store = SessionStore(tmp_path / "sessions")
    read_only_state = ActiveTaskState(
        goal_id="goal-safe",
        objective="Read only",
        operations=frozenset({"workspace_read"}),
        mutation_requirement="forbidden",
        source_turn_ids=("turn-safe",),
    )
    corrupt_contract = _turn_contract().to_dict()
    corrupt_contract["contract_version"] = "turn-intent-v999"
    await store.save_event(
        "session-safe",
        {
            "type": "session_meta",
            "event": "active_task_state_updated",
            "schema_version": ACTIVE_TASK_STATE_EVENT_VERSION,
            "session_id": "session-safe",
            "state_revision": 1,
            "active_task_state": read_only_state.to_dict(),
            "turn_intent_contract": corrupt_contract,
            "timestamp": "2026-07-24T04:00:00+00:00",
        },
    )

    result = await ConversationStateRepository(store).load("session-safe")

    assert result.active_task == read_only_state
    assert result.turn_intent is None
    assert result.diagnostics.status == "loaded"
    assert "turn_intent_contract" in " ".join(result.diagnostics.messages)


@pytest.mark.asyncio
async def test_active_task_state_compare_and_swap_rejects_stale_transition(tmp_path) -> None:
    repository = ConversationStateRepository(SessionStore(tmp_path / "sessions"))
    initial = await repository.save(
        "session-cas",
        active_task=_active_task(),
        turn_intent=_turn_contract(),
        expected_revision=0,
    )

    assert initial.status == "saved"
    assert initial.saved_revision == 1

    stale = await repository.save(
        "session-cas",
        active_task=replace(_active_task(), objective="stale overwrite"),
        expected_revision=0,
    )

    assert stale.status == "conflict"
    assert stale.current_revision == 1
    reloaded = await repository.load("session-cas")
    assert reloaded.state_revision == 1
    assert reloaded.active_task == _active_task()


@pytest.mark.asyncio
async def test_turn_checkpoint_cas_reconstructs_latest_state_after_restart(tmp_path) -> None:
    sessions_dir = tmp_path / "sessions"
    first_repository = TurnCheckpointRepository(SessionStore(sessions_dir))
    initial = _checkpoint()

    saved_initial = await first_repository.save(initial, expected_revision=0)
    assert saved_initial.status == "saved"
    assert saved_initial.checkpoint is not None
    assert saved_initial.checkpoint.revision == 1

    executing = replace(
        saved_initial.checkpoint,
        stage="executing",
        execution_receipt={"status": "started", "operation_ids": []},
        resume_cursor={"turn_id": "turn-checkpoint", "phase": "tool_call"},
    )
    saved_executing = await first_repository.save(executing, expected_revision=1)
    assert saved_executing.status == "saved"
    assert saved_executing.checkpoint is not None
    assert saved_executing.checkpoint.revision == 2

    stale = await first_repository.save(executing, expected_revision=1)
    assert stale.status == "conflict"
    assert stale.current_revision == 2

    reloaded = await TurnCheckpointRepository(SessionStore(sessions_dir)).load(
        "session-checkpoint",
        "turn-checkpoint",
    )
    assert reloaded.diagnostics.status == "loaded"
    assert reloaded.checkpoint == saved_executing.checkpoint


def test_turn_checkpoint_v1_payload_migrates_to_v2_defaults() -> None:
    payload = _checkpoint().to_dict()
    payload["checkpoint_version"] = "turn-checkpoint-v1"
    payload.pop("complexity_decision")
    payload.pop("plan_ledger_snapshot")
    payload.pop("verification_plan")
    payload.pop("recovery_budget")

    migrated = TurnCheckpoint.from_dict(payload)

    assert migrated.checkpoint_version == TURN_CHECKPOINT_VERSION
    assert migrated.complexity_decision == {}
    assert migrated.plan_ledger_snapshot is None
    assert migrated.verification_plan is None
    assert migrated.recovery_budget == {
        "remaining_attempts": 1,
        "remaining_extra_model_calls": 1,
        "remaining_extra_tool_calls": 4,
        "remaining_extra_wall_seconds": 120.0,
    }


def test_turn_checkpoint_future_version_fails_closed() -> None:
    payload = _checkpoint().to_dict()
    payload["checkpoint_version"] = "turn-checkpoint-v999"

    with pytest.raises(ValueError, match="unsupported checkpoint version"):
        TurnCheckpoint.from_dict(payload)


@pytest.mark.asyncio
async def test_turn_checkpoints_for_concurrent_session_turns_are_isolated(tmp_path) -> None:
    repository = TurnCheckpointRepository(SessionStore(tmp_path / "sessions"))
    first = _checkpoint(turn_id="turn-one", allowed_tool="file_write")
    second = _checkpoint(turn_id="turn-two", allowed_tool="exec_command")

    first_saved, second_saved = await asyncio.gather(
        repository.save(first, expected_revision=0),
        repository.save(second, expected_revision=0),
    )

    assert first_saved.status == "saved"
    assert second_saved.status == "saved"
    assert first_saved.checkpoint is not None
    assert second_saved.checkpoint is not None
    assert first_saved.checkpoint.turn_id == "turn-one"
    assert second_saved.checkpoint.turn_id == "turn-two"
    assert first_saved.checkpoint.activation_state == {
        "allowed_tool_names": ["file_write"]
    }
    assert second_saved.checkpoint.activation_state == {
        "allowed_tool_names": ["exec_command"]
    }


@pytest.mark.asyncio
async def test_independent_repository_instances_have_one_durable_cas_winner(tmp_path) -> None:
    sessions_dir = tmp_path / "sessions"
    first_state = ConversationStateRepository(SessionStore(sessions_dir))
    second_state = ConversationStateRepository(SessionStore(sessions_dir))
    state_results = await asyncio.gather(
        first_state.save("shared", active_task=_active_task(), expected_revision=0),
        second_state.save(
            "shared",
            active_task=replace(_active_task(), objective="other writer"),
            expected_revision=0,
        ),
    )
    assert [item.status for item in state_results].count("saved") == 1
    assert [item.status for item in state_results].count("conflict") == 1

    first_checkpoint = TurnCheckpointRepository(SessionStore(sessions_dir))
    second_checkpoint = TurnCheckpointRepository(SessionStore(sessions_dir))
    checkpoint_results = await asyncio.gather(
        first_checkpoint.save(_checkpoint(session_id="shared", turn_id="turn"), expected_revision=0),
        second_checkpoint.save(_checkpoint(session_id="shared", turn_id="turn"), expected_revision=0),
    )
    assert [item.status for item in checkpoint_results].count("saved") == 1
    assert [item.status for item in checkpoint_results].count("conflict") == 1
    pending = await first_checkpoint.list_nonterminal("shared")
    assert [item.turn_id for item in pending] == ["turn"]
