"""Durable conversation task state and versioned per-turn checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Any, Literal, Mapping, cast

from mochi.agents.turn_intent_contract import ActiveTaskState, TurnIntentContract
from mochi.sessions.store import SessionStore

ACTIVE_TASK_STATE_EVENT = "active_task_state_updated"
ACTIVE_TASK_STATE_EVENT_VERSION = 2
_LEGACY_ACTIVE_TASK_STATE_EVENT_VERSION = 1

TURN_CHECKPOINT_EVENT = "turn_execution_checkpoint"
TURN_CHECKPOINT_EVENT_VERSION = 1
TURN_CHECKPOINT_VERSION = "turn-checkpoint-v1"

ConversationStateLoadStatus = Literal[
    "loaded",
    "missing",
    "invalid",
    "unsupported_version",
]
ConversationStateSaveStatus = Literal["saved", "conflict", "invalid"]
TurnCheckpointLoadStatus = ConversationStateLoadStatus
TurnCheckpointSaveStatus = Literal["saved", "conflict", "invalid"]
TurnCheckpointStage = Literal[
    "contract_resolved",
    "awaiting_approval",
    "executing",
    "verifying",
    "completed",
    "blocked",
]

_ACTIVE_TASK_EVENT_FIELDS_V1 = frozenset(
    {
        "type",
        "event",
        "schema_version",
        "session_id",
        "active_task_state",
        "turn_intent_contract",
        "timestamp",
    }
)
_ACTIVE_TASK_EVENT_FIELDS_V2 = _ACTIVE_TASK_EVENT_FIELDS_V1 | {"state_revision"}
_TURN_CHECKPOINT_EVENT_FIELDS = frozenset(
    {
        "type",
        "event",
        "schema_version",
        "session_id",
        "turn_id",
        "checkpoint",
        "timestamp",
    }
)
_TERMINAL_CHECKPOINT_STAGES = frozenset({"completed", "blocked"})
_CHECKPOINT_TRANSITIONS: dict[str | None, frozenset[TurnCheckpointStage]] = {
    None: frozenset({"contract_resolved"}),
    "contract_resolved": frozenset({"contract_resolved", "awaiting_approval", "executing", "blocked"}),
    "awaiting_approval": frozenset({"awaiting_approval", "executing", "blocked"}),
    "executing": frozenset({"executing", "awaiting_approval", "verifying", "blocked"}),
    "verifying": frozenset({"verifying", "completed", "blocked"}),
    "completed": frozenset(),
    "blocked": frozenset(),
}


@dataclass(frozen=True)
class ConversationStateLoadDiagnostics:
    """Non-authoritative diagnostics for one reverse event scan."""

    status: ConversationStateLoadStatus
    event_schema_version: int | None = None
    event_index: int | None = None
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class ConversationStateLoadResult:
    """Latest validated durable state plus optional audit-only turn contract."""

    active_task: ActiveTaskState | None
    turn_intent: TurnIntentContract | None
    diagnostics: ConversationStateLoadDiagnostics
    state_revision: int = 0


@dataclass(frozen=True)
class ConversationStateSaveResult:
    """Outcome of one active-task state CAS transition."""

    status: ConversationStateSaveStatus
    expected_revision: int | None
    current_revision: int
    saved_revision: int | None = None
    message: str | None = None


@dataclass(frozen=True)
class TurnCheckpoint:
    """One durable, independently recoverable execution state for a turn."""

    session_id: str
    turn_id: str
    revision: int
    stage: TurnCheckpointStage
    turn_intent_contract: Mapping[str, Any]
    capability_plan: Mapping[str, Any]
    active_goal_id: str | None = None
    policy_snapshot: Mapping[str, Any] = field(default_factory=dict)
    inventory_snapshot: Mapping[str, Any] = field(default_factory=dict)
    activation_state: Mapping[str, Any] = field(default_factory=dict)
    pending_tool_call: Mapping[str, Any] | None = None
    approval_record: Mapping[str, Any] | None = None
    execution_receipt: Mapping[str, Any] | None = None
    verification_result: Mapping[str, Any] | None = None
    resume_cursor: Mapping[str, Any] | None = None
    completion_reason: str | None = None
    blocker_reason: str | None = None
    checkpoint_version: str = field(default=TURN_CHECKPOINT_VERSION, init=False)

    def __post_init__(self) -> None:
        _require_text(self.session_id, field_name="session_id")
        _require_text(self.turn_id, field_name="turn_id")
        if type(self.revision) is not int or self.revision < 0:
            raise ValueError("checkpoint revision must be a non-negative integer")
        if self.stage not in _CHECKPOINT_TRANSITIONS:
            raise ValueError(f"unsupported checkpoint stage: {self.stage!r}")
        if self.active_goal_id is not None:
            _require_text(self.active_goal_id, field_name="active_goal_id")
        for field_name in (
            "turn_intent_contract",
            "capability_plan",
            "policy_snapshot",
            "inventory_snapshot",
            "activation_state",
        ):
            object.__setattr__(
                self,
                field_name,
                _frozen_json_mapping(getattr(self, field_name), field_name=field_name),
            )
        for field_name in (
            "pending_tool_call",
            "approval_record",
            "execution_receipt",
            "verification_result",
            "resume_cursor",
        ):
            value = getattr(self, field_name)
            object.__setattr__(
                self,
                field_name,
                (
                    _frozen_json_mapping(value, field_name=field_name)
                    if value is not None
                    else None
                ),
            )
        for field_name in ("completion_reason", "blocker_reason"):
            value = getattr(self, field_name)
            if value is not None:
                _require_text(value, field_name=field_name)
        if self.stage == "completed" and not self.completion_reason:
            raise ValueError("completed checkpoint requires completion_reason")
        if self.stage == "blocked" and not self.blocker_reason:
            raise ValueError("blocked checkpoint requires blocker_reason")

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_version": self.checkpoint_version,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "revision": self.revision,
            "stage": self.stage,
            "turn_intent_contract": _json_clone(self.turn_intent_contract),
            "capability_plan": _json_clone(self.capability_plan),
            "active_goal_id": self.active_goal_id,
            "policy_snapshot": _json_clone(self.policy_snapshot),
            "inventory_snapshot": _json_clone(self.inventory_snapshot),
            "activation_state": _json_clone(self.activation_state),
            "pending_tool_call": _json_clone_or_none(self.pending_tool_call),
            "approval_record": _json_clone_or_none(self.approval_record),
            "execution_receipt": _json_clone_or_none(self.execution_receipt),
            "verification_result": _json_clone_or_none(self.verification_result),
            "resume_cursor": _json_clone_or_none(self.resume_cursor),
            "completion_reason": self.completion_reason,
            "blocker_reason": self.blocker_reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TurnCheckpoint:
        expected = {
            "checkpoint_version",
            "session_id",
            "turn_id",
            "revision",
            "stage",
            "turn_intent_contract",
            "capability_plan",
            "active_goal_id",
            "policy_snapshot",
            "inventory_snapshot",
            "activation_state",
            "pending_tool_call",
            "approval_record",
            "execution_receipt",
            "verification_result",
            "resume_cursor",
            "completion_reason",
            "blocker_reason",
        }
        missing = sorted(expected - set(value))
        unexpected = sorted(set(value) - expected)
        if missing or unexpected:
            details: list[str] = []
            if missing:
                details.append(f"missing fields: {missing}")
            if unexpected:
                details.append(f"unexpected fields: {unexpected}")
            raise ValueError("turn checkpoint " + "; ".join(details))
        if value.get("checkpoint_version") != TURN_CHECKPOINT_VERSION:
            raise ValueError(
                f"unsupported checkpoint version: {value.get('checkpoint_version')!r}"
            )
        raw_stage = value.get("stage")
        if not isinstance(raw_stage, str) or raw_stage not in _CHECKPOINT_TRANSITIONS:
            raise ValueError(f"unsupported checkpoint stage: {raw_stage!r}")
        stage = cast(TurnCheckpointStage, raw_stage)
        return cls(
            session_id=_require_text(value.get("session_id"), field_name="session_id"),
            turn_id=_require_text(value.get("turn_id"), field_name="turn_id"),
            revision=_require_non_negative_int(value.get("revision"), field_name="revision"),
            stage=stage,
            turn_intent_contract=_require_mapping(
                value.get("turn_intent_contract"), field_name="turn_intent_contract"
            ),
            capability_plan=_require_mapping(
                value.get("capability_plan"), field_name="capability_plan"
            ),
            active_goal_id=_optional_text(value.get("active_goal_id"), field_name="active_goal_id"),
            policy_snapshot=_require_mapping(
                value.get("policy_snapshot"), field_name="policy_snapshot"
            ),
            inventory_snapshot=_require_mapping(
                value.get("inventory_snapshot"), field_name="inventory_snapshot"
            ),
            activation_state=_require_mapping(
                value.get("activation_state"), field_name="activation_state"
            ),
            pending_tool_call=_optional_mapping(
                value.get("pending_tool_call"), field_name="pending_tool_call"
            ),
            approval_record=_optional_mapping(
                value.get("approval_record"), field_name="approval_record"
            ),
            execution_receipt=_optional_mapping(
                value.get("execution_receipt"), field_name="execution_receipt"
            ),
            verification_result=_optional_mapping(
                value.get("verification_result"), field_name="verification_result"
            ),
            resume_cursor=_optional_mapping(
                value.get("resume_cursor"), field_name="resume_cursor"
            ),
            completion_reason=_optional_text(
                value.get("completion_reason"), field_name="completion_reason"
            ),
            blocker_reason=_optional_text(
                value.get("blocker_reason"), field_name="blocker_reason"
            ),
        )


@dataclass(frozen=True)
class TurnCheckpointLoadDiagnostics:
    status: TurnCheckpointLoadStatus
    event_schema_version: int | None = None
    event_index: int | None = None
    messages: tuple[str, ...] = ()


@dataclass(frozen=True)
class TurnCheckpointLoadResult:
    checkpoint: TurnCheckpoint | None
    diagnostics: TurnCheckpointLoadDiagnostics


@dataclass(frozen=True)
class TurnCheckpointSaveResult:
    status: TurnCheckpointSaveStatus
    expected_revision: int
    current_revision: int
    checkpoint: TurnCheckpoint | None = None
    message: str | None = None


class ConversationStateRepository:
    """Append and rebuild active-task state with revision compare-and-swap."""

    def __init__(self, session_store: SessionStore) -> None:
        self._session_store = session_store

    async def save(
        self,
        session_id: str,
        *,
        active_task: ActiveTaskState,
        turn_intent: TurnIntentContract | None = None,
        timestamp: str | None = None,
        expected_revision: int | None = None,
    ) -> ConversationStateSaveResult:
        """Append one complete state snapshot, optionally guarded by a revision."""
        if expected_revision is not None:
            _require_non_negative_int(expected_revision, field_name="expected_revision")
        outcome: ConversationStateSaveResult | None = None

        def can_append(events: list[dict]) -> bool:
            nonlocal outcome
            current = self._load_from_events(session_id=session_id, events=events)
            if current.diagnostics.status in {"invalid", "unsupported_version"}:
                outcome = ConversationStateSaveResult(
                    status="invalid",
                    expected_revision=expected_revision,
                    current_revision=current.state_revision,
                    message="cannot transition from invalid durable active-task state",
                )
                return False
            if (
                expected_revision is not None
                and current.state_revision != expected_revision
            ):
                outcome = ConversationStateSaveResult(
                    status="conflict",
                    expected_revision=expected_revision,
                    current_revision=current.state_revision,
                    message="active-task state revision changed before transition",
                )
                return False
            next_revision = current.state_revision + 1
            event["state_revision"] = next_revision
            outcome = ConversationStateSaveResult(
                status="saved",
                expected_revision=expected_revision,
                current_revision=current.state_revision,
                saved_revision=next_revision,
            )
            return True

        event: dict[str, Any] = {
                "type": "session_meta",
                "event": ACTIVE_TASK_STATE_EVENT,
                "schema_version": ACTIVE_TASK_STATE_EVENT_VERSION,
                "session_id": session_id,
                "state_revision": 0,
                "active_task_state": active_task.to_dict(),
                "turn_intent_contract": (
                    turn_intent.to_dict() if turn_intent is not None else None
                ),
                "timestamp": timestamp or datetime.now(tz=UTC).isoformat(),
            }
        appended = await self._session_store.append_event_if(session_id, event, can_append)
        if outcome is not None:
            return outcome
        return ConversationStateSaveResult(
            status="conflict" if not appended else "invalid",
            expected_revision=expected_revision,
            current_revision=0,
            message="durable active-task transition did not produce an outcome",
        )

    async def load(self, session_id: str) -> ConversationStateLoadResult:
        """Load the newest state event, failing closed if that event is invalid."""
        events = await self._session_store.load_session(session_id)
        return self._load_from_events(session_id=session_id, events=events)

    def _load_from_events(
        self,
        *,
        session_id: str,
        events: list[dict],
    ) -> ConversationStateLoadResult:
        for event_index in range(len(events) - 1, -1, -1):
            event = events[event_index]
            if event.get("event") != ACTIVE_TASK_STATE_EVENT:
                continue
            return self._load_event(
                session_id=session_id,
                event=event,
                event_index=event_index,
            )
        return ConversationStateLoadResult(
            active_task=None,
            turn_intent=None,
            diagnostics=ConversationStateLoadDiagnostics(status="missing"),
            state_revision=0,
        )

    @staticmethod
    def _load_event(
        *,
        session_id: str,
        event: dict[str, Any],
        event_index: int,
    ) -> ConversationStateLoadResult:
        version = event.get("schema_version")
        if type(version) is not int:
            return _failed_state_result(
                status="invalid",
                event_index=event_index,
                message="active task event schema_version must be an integer",
            )
        if version not in {
            _LEGACY_ACTIVE_TASK_STATE_EVENT_VERSION,
            ACTIVE_TASK_STATE_EVENT_VERSION,
        }:
            return _failed_state_result(
                status="unsupported_version",
                event_schema_version=version,
                event_index=event_index,
                message=f"unsupported active task event schema version: {version}",
            )

        expected_fields = (
            _ACTIVE_TASK_EVENT_FIELDS_V1
            if version == _LEGACY_ACTIVE_TASK_STATE_EVENT_VERSION
            else _ACTIVE_TASK_EVENT_FIELDS_V2
        )
        unexpected = sorted(set(event) - expected_fields)
        missing = sorted(expected_fields - set(event))
        if missing or unexpected:
            details: list[str] = []
            if missing:
                details.append(f"missing fields: {missing}")
            if unexpected:
                details.append(f"unexpected fields: {unexpected}")
            return _failed_state_result(
                status="invalid",
                event_schema_version=version,
                event_index=event_index,
                message="active task event " + "; ".join(details),
            )
        if event.get("type") != "session_meta":
            return _failed_state_result(
                status="invalid",
                event_schema_version=version,
                event_index=event_index,
                message="active task event type must be session_meta",
            )
        if event.get("session_id") != session_id:
            return _failed_state_result(
                status="invalid",
                event_schema_version=version,
                event_index=event_index,
                message="active task event session_id does not match the requested session",
            )
        timestamp = event.get("timestamp")
        if not isinstance(timestamp, str) or not timestamp.strip():
            return _failed_state_result(
                status="invalid",
                event_schema_version=version,
                event_index=event_index,
                message="active task event timestamp must be a non-empty string",
            )
        state_revision = (
            0
            if version == _LEGACY_ACTIVE_TASK_STATE_EVENT_VERSION
            else _require_event_revision(event.get("state_revision"))
        )
        if state_revision is None:
            return _failed_state_result(
                status="invalid",
                event_schema_version=version,
                event_index=event_index,
                message="active task event state_revision must be a positive integer",
            )

        raw_active_task = event.get("active_task_state")
        if not isinstance(raw_active_task, Mapping):
            return _failed_state_result(
                status="invalid",
                event_schema_version=version,
                event_index=event_index,
                message="invalid active_task_state: expected an object",
            )
        try:
            active_task = ActiveTaskState.from_dict(raw_active_task)
        except (TypeError, ValueError) as exc:
            return _failed_state_result(
                status="invalid",
                event_schema_version=version,
                event_index=event_index,
                message=f"invalid active_task_state: {exc}",
            )

        messages: list[str] = []
        if version == _LEGACY_ACTIVE_TASK_STATE_EVENT_VERSION:
            messages.append("legacy active-task state has implicit revision 0")
        turn_intent: TurnIntentContract | None = None
        raw_contract = event.get("turn_intent_contract")
        if raw_contract is not None:
            try:
                candidate = TurnIntentContract.from_dict(raw_contract)
            except (TypeError, ValueError) as exc:
                messages.append(f"invalid turn_intent_contract ignored: {exc}")
            else:
                if (
                    candidate.active_goal_id is not None
                    and candidate.active_goal_id != active_task.goal_id
                ):
                    messages.append(
                        "turn_intent_contract goal does not match active_task_state; ignored"
                    )
                else:
                    turn_intent = candidate

        return ConversationStateLoadResult(
            active_task=active_task,
            turn_intent=turn_intent,
            diagnostics=ConversationStateLoadDiagnostics(
                status="loaded",
                event_schema_version=version,
                event_index=event_index,
                messages=tuple(messages),
            ),
            state_revision=state_revision,
        )


class TurnCheckpointRepository:
    """Append and reconstruct independent turn state with per-turn CAS."""

    def __init__(self, session_store: SessionStore) -> None:
        self._session_store = session_store

    async def load(self, session_id: str, turn_id: str) -> TurnCheckpointLoadResult:
        _require_text(session_id, field_name="session_id")
        _require_text(turn_id, field_name="turn_id")
        events = await self._session_store.load_session(session_id)
        return self._load_from_events(
            session_id=session_id,
            turn_id=turn_id,
            events=events,
        )

    async def list_nonterminal(self, session_id: str) -> tuple[TurnCheckpoint, ...]:
        """Discover recoverable turn work without replaying tool side effects."""
        _require_text(session_id, field_name="session_id")
        events = await self._session_store.load_session(session_id)
        newest_turn_ids: list[str] = []
        seen: set[str] = set()
        for event in reversed(events):
            if event.get("event") != TURN_CHECKPOINT_EVENT:
                continue
            turn_id = event.get("turn_id")
            if isinstance(turn_id, str) and turn_id and turn_id not in seen:
                seen.add(turn_id)
                newest_turn_ids.append(turn_id)
        checkpoints: list[TurnCheckpoint] = []
        for turn_id in newest_turn_ids:
            loaded = self._load_from_events(
                session_id=session_id,
                turn_id=turn_id,
                events=events,
            )
            if (
                loaded.diagnostics.status == "loaded"
                and loaded.checkpoint is not None
                and loaded.checkpoint.stage not in _TERMINAL_CHECKPOINT_STAGES
            ):
                checkpoints.append(loaded.checkpoint)
        return tuple(checkpoints)

    def _load_from_events(
        self,
        *,
        session_id: str,
        turn_id: str,
        events: list[dict],
    ) -> TurnCheckpointLoadResult:
        for event_index in range(len(events) - 1, -1, -1):
            event = events[event_index]
            if (
                event.get("event") != TURN_CHECKPOINT_EVENT
                or event.get("turn_id") != turn_id
            ):
                continue
            return self._load_event(
                session_id=session_id,
                turn_id=turn_id,
                event=event,
                event_index=event_index,
            )
        return TurnCheckpointLoadResult(
            checkpoint=None,
            diagnostics=TurnCheckpointLoadDiagnostics(status="missing"),
        )

    async def save(
        self,
        checkpoint: TurnCheckpoint,
        *,
        expected_revision: int,
        timestamp: str | None = None,
    ) -> TurnCheckpointSaveResult:
        """Append a legal checkpoint transition only if its revision still matches."""
        _require_non_negative_int(expected_revision, field_name="expected_revision")
        # Checkpoints are an aggregate authority.  Use the strict SessionStore
        # callback so an enabled aggregate outbox is appended in the same CAS
        # batch as this source transition.
        for _ in range(8):
            snapshot = await self._session_store.load_strict_snapshot(checkpoint.session_id)
            outcome: TurnCheckpointSaveResult | None = None

            def build_events(snapshot_under_lock: Any) -> tuple[dict[str, Any], ...] | None:
                nonlocal outcome
                current = self._load_from_events(
                    session_id=checkpoint.session_id,
                    turn_id=checkpoint.turn_id,
                    events=[dict(item) for item in snapshot_under_lock.events],
                )
                if current.diagnostics.status in {"invalid", "unsupported_version"}:
                    outcome = TurnCheckpointSaveResult(
                        status="invalid",
                        expected_revision=expected_revision,
                        current_revision=(
                            current.checkpoint.revision if current.checkpoint is not None else 0
                        ),
                        message="cannot transition from invalid durable turn checkpoint",
                    )
                    return None
                prior = current.checkpoint
                current_revision = prior.revision if prior is not None else 0
                if current_revision != expected_revision:
                    outcome = TurnCheckpointSaveResult(
                        status="conflict",
                        expected_revision=expected_revision,
                        current_revision=current_revision,
                        checkpoint=prior,
                        message="turn checkpoint revision changed before transition",
                    )
                    return None
                prior_stage = prior.stage if prior is not None else None
                if checkpoint.stage not in _CHECKPOINT_TRANSITIONS[prior_stage]:
                    outcome = TurnCheckpointSaveResult(
                        status="invalid",
                        expected_revision=expected_revision,
                        current_revision=current_revision,
                        checkpoint=prior,
                        message=(
                            "illegal turn checkpoint transition: "
                            f"{prior_stage or 'missing'} -> {checkpoint.stage}"
                        ),
                    )
                    return None
                next_checkpoint = replace(checkpoint, revision=current_revision + 1)
                outcome = TurnCheckpointSaveResult(
                    status="saved",
                    expected_revision=expected_revision,
                    current_revision=current_revision,
                    checkpoint=next_checkpoint,
                )
                return (
                    {
                        "type": "session_meta",
                        "event": TURN_CHECKPOINT_EVENT,
                        "schema_version": TURN_CHECKPOINT_EVENT_VERSION,
                        "session_id": checkpoint.session_id,
                        "turn_id": checkpoint.turn_id,
                        "checkpoint": next_checkpoint.to_dict(),
                        "timestamp": timestamp or datetime.now(tz=UTC).isoformat(),
                    },
                )

            result = await self._session_store.mutate_strict_snapshot(
                checkpoint.session_id,
                expected_history_revision=snapshot.history_revision,
                build_events=build_events,
            )
            if result.status == "rebase_required":
                continue
            if outcome is not None:
                return outcome
            return TurnCheckpointSaveResult(
                status="invalid",
                expected_revision=expected_revision,
                current_revision=0,
                message="durable turn checkpoint transition did not produce an outcome",
            )
        return TurnCheckpointSaveResult(
            status="conflict",
            expected_revision=expected_revision,
            current_revision=0,
            message="turn checkpoint repeatedly lost the SessionStore CAS",
        )

    @staticmethod
    def _load_event(
        *,
        session_id: str,
        turn_id: str,
        event: dict[str, Any],
        event_index: int,
    ) -> TurnCheckpointLoadResult:
        version = event.get("schema_version")
        if type(version) is not int:
            return _failed_checkpoint_result(
                status="invalid",
                event_index=event_index,
                message="turn checkpoint event schema_version must be an integer",
            )
        if version != TURN_CHECKPOINT_EVENT_VERSION:
            return _failed_checkpoint_result(
                status="unsupported_version",
                event_schema_version=version,
                event_index=event_index,
                message=f"unsupported turn checkpoint event schema version: {version}",
            )
        unexpected = sorted(set(event) - _TURN_CHECKPOINT_EVENT_FIELDS)
        missing = sorted(_TURN_CHECKPOINT_EVENT_FIELDS - set(event))
        if missing or unexpected:
            details: list[str] = []
            if missing:
                details.append(f"missing fields: {missing}")
            if unexpected:
                details.append(f"unexpected fields: {unexpected}")
            return _failed_checkpoint_result(
                status="invalid",
                event_schema_version=version,
                event_index=event_index,
                message="turn checkpoint event " + "; ".join(details),
            )
        if event.get("type") != "session_meta":
            return _failed_checkpoint_result(
                status="invalid",
                event_schema_version=version,
                event_index=event_index,
                message="turn checkpoint event type must be session_meta",
            )
        if event.get("session_id") != session_id or event.get("turn_id") != turn_id:
            return _failed_checkpoint_result(
                status="invalid",
                event_schema_version=version,
                event_index=event_index,
                message="turn checkpoint event does not match the requested session or turn",
            )
        timestamp = event.get("timestamp")
        if not isinstance(timestamp, str) or not timestamp.strip():
            return _failed_checkpoint_result(
                status="invalid",
                event_schema_version=version,
                event_index=event_index,
                message="turn checkpoint event timestamp must be a non-empty string",
            )
        payload = event.get("checkpoint")
        if not isinstance(payload, Mapping):
            return _failed_checkpoint_result(
                status="invalid",
                event_schema_version=version,
                event_index=event_index,
                message="turn checkpoint event checkpoint must be an object",
            )
        try:
            checkpoint = TurnCheckpoint.from_dict(payload)
        except (TypeError, ValueError) as exc:
            return _failed_checkpoint_result(
                status="invalid",
                event_schema_version=version,
                event_index=event_index,
                message=f"invalid turn checkpoint: {exc}",
            )
        if checkpoint.session_id != session_id or checkpoint.turn_id != turn_id:
            return _failed_checkpoint_result(
                status="invalid",
                event_schema_version=version,
                event_index=event_index,
                message="checkpoint identity does not match envelope identity",
            )
        return TurnCheckpointLoadResult(
            checkpoint=checkpoint,
            diagnostics=TurnCheckpointLoadDiagnostics(
                status="loaded",
                event_schema_version=version,
                event_index=event_index,
            ),
        )


def _failed_state_result(
    *,
    status: Literal["invalid", "unsupported_version"],
    event_index: int,
    message: str,
    event_schema_version: int | None = None,
) -> ConversationStateLoadResult:
    return ConversationStateLoadResult(
        active_task=None,
        turn_intent=None,
        diagnostics=ConversationStateLoadDiagnostics(
            status=status,
            event_schema_version=event_schema_version,
            event_index=event_index,
            messages=(message,),
        ),
        state_revision=0,
    )


def _failed_checkpoint_result(
    *,
    status: Literal["invalid", "unsupported_version"],
    event_index: int,
    message: str,
    event_schema_version: int | None = None,
) -> TurnCheckpointLoadResult:
    return TurnCheckpointLoadResult(
        checkpoint=None,
        diagnostics=TurnCheckpointLoadDiagnostics(
            status=status,
            event_schema_version=event_schema_version,
            event_index=event_index,
            messages=(message,),
        ),
    )


def _require_event_revision(value: Any) -> int | None:
    return value if type(value) is int and value > 0 else None


def _require_non_negative_int(value: Any, *, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _require_text(value: Any, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _optional_text(value: Any, *, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_text(value, field_name=field_name)


def _require_mapping(value: Any, *, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _optional_mapping(value: Any, *, field_name: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    return _require_mapping(value, field_name=field_name)


def _frozen_json_mapping(value: Mapping[str, Any], *, field_name: str) -> Mapping[str, Any]:
    normalized = _json_clone(_require_mapping(value, field_name=field_name))
    if not isinstance(normalized, dict):  # Defensive: mapping JSON must remain an object.
        raise ValueError(f"{field_name} must be a JSON object")
    return MappingProxyType(normalized)


def _json_clone(value: Any) -> Any:
    try:
        return json.loads(
            json.dumps(
                _json_compatible_value(value),
                ensure_ascii=False,
                sort_keys=True,
                allow_nan=False,
            )
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(f"checkpoint payload must be JSON serializable: {exc}") from exc


def _json_clone_or_none(value: Mapping[str, Any] | None) -> dict[str, Any] | None:
    return _json_clone(value) if value is not None else None


def _json_compatible_value(value: Any) -> Any:
    """Normalize immutable mappings before the strict JSON round trip."""
    if isinstance(value, Mapping):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError("checkpoint object keys must be strings")
            normalized[key] = _json_compatible_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_json_compatible_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported checkpoint JSON value: {type(value).__name__}")
