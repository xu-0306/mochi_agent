"""Runtime binding for the durable ordinary-Chat turn timeline.

The coordinator owns only short CAS transitions.  It deliberately never keeps
its state lock while a model, tool, or queue wait is in progress.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from mochi.security.file_contract import tool_arguments_digest
from mochi.sessions.store import SessionStore
from mochi.sessions.turn_timeline import (
    OperationDescriptor,
    SessionTurnTimelineRepository,
    TimelineMutationResult,
)


class TimelineCoordinatorError(RuntimeError):
    """The current Chat turn cannot continue without violating timeline safety."""


class TimelineTurnCancelled(TimelineCoordinatorError):
    """A queued or running turn was durably cancelled before more work began."""


@dataclass
class TimelineCoordinator:
    """Serialize one ordinary-Chat turn through the durable FIFO lane."""

    session_store: SessionStore
    session_id: str
    turn_id: str
    owner: str = field(default_factory=lambda: f"ordinary-chat:{uuid4().hex}")
    token: str = field(default_factory=lambda: f"timeline-lease:{uuid4().hex}")
    lease_seconds: int = 30
    poll_seconds: float = 0.02
    _repository: SessionTurnTimelineRepository = field(init=False, repr=False)
    _history_revision: str | None = field(init=False, default=None, repr=False)
    _state_lock: asyncio.Lock = field(init=False, default_factory=asyncio.Lock, repr=False)
    _operation_dispatch_lock: asyncio.Lock = field(
        init=False,
        default_factory=asyncio.Lock,
        repr=False,
    )
    _heartbeat_task: asyncio.Task[None] | None = field(init=False, default=None, repr=False)
    _cancel_requested: bool = field(init=False, default=False, repr=False)
    _unstarted_blocked: bool = field(init=False, default=False, repr=False)
    _current_operation_id: str | None = field(init=False, default=None, repr=False)
    _unknown_operation: bool = field(init=False, default=False, repr=False)
    _claimed: bool = field(init=False, default=False, repr=False)
    _closed: bool = field(init=False, default=False, repr=False)
    _admission_lease_expires_at: str | None = field(init=False, default=None, repr=False)
    _dispatched_operation_ids: set[str] = field(
        init=False,
        default_factory=set,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not self.session_id.strip() or not self.turn_id.strip():
            raise ValueError("session_id and turn_id must be non-empty")
        if self.lease_seconds < 6:
            raise ValueError("lease_seconds must be at least 6")
        self._repository = SessionTurnTimelineRepository(self.session_store)

    async def admit_user_message(self, event: Mapping[str, Any]) -> None:
        """Atomically append the user message and its queued turn identity."""
        if self._admission_lease_expires_at is None:
            self._admission_lease_expires_at = self._admission_lease_expiry().isoformat()
        result = await self._mutate(
            lambda revision: self._repository.admit(
                self.session_id,
                turn_id=self.turn_id,
                expected_history_revision=revision,
                companion_events=(event,),
                admission_owner=self.owner,
                admission_token=self.token,
                admission_lease_expires_at=self._admission_lease_expires_at,
                now=self._now(),
            )
        )
        if result.status not in {"admitted", "duplicate"}:
            raise TimelineCoordinatorError(
                f"unable to admit ordinary Chat turn: {result.status}: {result.message or ''}"
            )

    async def claim(self) -> Sequence[Mapping[str, Any]]:
        """Wait until this turn is FIFO head, then return its linearized history."""
        while True:
            if self._cancel_requested:
                raise TimelineTurnCancelled("ordinary Chat turn was cancelled before claim")
            loaded = await self._repository.load(self.session_id)
            if loaded.status in {"invalid", "unsupported_version"} or loaded.timeline is None:
                raise TimelineCoordinatorError(
                    f"unable to load ordinary Chat timeline: {loaded.status}: {loaded.message or ''}"
                )
            self._history_revision = loaded.history_revision
            turn = next((item for item in loaded.timeline.turns if item.turn_id == self.turn_id), None)
            if turn is None:
                raise TimelineCoordinatorError("admitted ordinary Chat turn disappeared")
            if turn.status == "terminal":
                raise TimelineTurnCancelled("ordinary Chat turn is already terminal")
            if turn.status == "queued" and self._admission_needs_renewal(turn):
                renewal = await self._mutate(
                    lambda revision: self._repository.renew_queued_admission(
                        self.session_id,
                        turn_id=self.turn_id,
                        expected_history_revision=revision,
                        owner=self.owner,
                        token=self.token,
                        admission_lease_expires_at=self._admission_lease_expiry().isoformat(),
                        now=self._now(),
                    )
                )
                if renewal.status == "admission_lease_renewed":
                    if renewal.timeline is not None:
                        renewed = next(
                            (item for item in renewal.timeline.turns if item.turn_id == self.turn_id),
                            None,
                        )
                        if renewed is not None:
                            self._admission_lease_expires_at = renewed.admission_lease_expires_at
                    continue
                if renewal.status == "admission_stale":
                    await self._recover_expired_admission(loaded)
                    continue
                raise TimelineCoordinatorError(
                    "unable to renew ordinary Chat admission lease: "
                    f"{renewal.status}: {renewal.message or ''}"
                )
            queued = next((item for item in loaded.timeline.turns if item.status == "queued"), None)
            if queued is None or queued.turn_id != self.turn_id:
                if loaded.timeline.lane_turn_id is not None and self._lease_is_stale(loaded.timeline):
                    await self._recover_stale_lane(loaded)
                elif self._admission_is_stale(queued):
                    await self._recover_expired_admission(loaded)
                else:
                    await asyncio.sleep(self.poll_seconds)
                continue
            if loaded.history_revision is None:
                raise TimelineCoordinatorError("ordinary Chat timeline has no history revision")
            result = await self._mutate(
                lambda revision: self._repository.claim_next(
                    self.session_id,
                    expected_history_revision=revision,
                    owner=self.owner,
                    token=self.token,
                    lease_expires_at=self._lease_expiry().isoformat(),
                    now=self._now(),
                )
            )
            if result.status == "claimed":
                if result.timeline is None or result.timeline.lane_turn_id != self.turn_id:
                    raise TimelineCoordinatorError("ordinary Chat claim selected a different turn")
                self._claimed = True
                self._admission_lease_expires_at = None
                return await self.linearized_history_events()
            if result.status in {"lane_busy", "admission_busy", "rebase_required"}:
                await asyncio.sleep(self.poll_seconds)
                continue
            if result.status == "lease_stale":
                await self._recover_stale_lane(result)
                continue
            if result.status == "admission_stale":
                await self._recover_expired_admission(result)
                continue
            raise TimelineCoordinatorError(
                f"unable to claim ordinary Chat lane: {result.status}: {result.message or ''}"
            )

    async def start_heartbeat(self) -> None:
        if not self._claimed or self._closed or self._heartbeat_task is not None:
            return
        self._heartbeat_task = asyncio.create_task(
            self._heartbeat(),
            name=f"ordinary-chat-timeline-lease-{self.turn_id}",
        )

    async def stop_heartbeat(self) -> None:
        task = self._heartbeat_task
        self._heartbeat_task = None
        if task is None or task.done():
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task

    async def linearized_history_events(self) -> Sequence[Mapping[str, Any]]:
        """Return terminal predecessor messages in durable FIFO turn order."""
        for _ in range(4):
            loaded = await self._repository.load(self.session_id)
            snapshot = await self.session_store.load_strict_snapshot(self.session_id)
            if loaded.history_revision != snapshot.history_revision or loaded.timeline is None:
                continue
            self._history_revision = snapshot.history_revision
            current = next(
                (item for item in loaded.timeline.turns if item.turn_id == self.turn_id),
                None,
            )
            if current is None or current.status != "running":
                raise TimelineCoordinatorError("ordinary Chat lane is not running while building history")
            predecessors = tuple(
                sorted(
                    (
                        item
                        for item in loaded.timeline.turns
                        if (
                            item.sequence < current.sequence
                            and item.status == "terminal"
                            and item.terminal_outcome in {"completed", "blocked", "unknown"}
                            and item.recovery_reason is None
                        )
                    ),
                    key=lambda item: item.sequence,
                )
            )
            predecessor_ids = {item.turn_id for item in predecessors}
            known_ids = {item.turn_id for item in loaded.timeline.turns}
            base_messages: list[Mapping[str, Any]] = []
            messages_by_turn: dict[str, list[Mapping[str, Any]]] = {
                item.turn_id: [] for item in predecessors
            }
            compatibility_messages: list[Mapping[str, Any]] = []
            for event_index, event in enumerate(snapshot.events):
                if event.get("type") != "message":
                    continue
                if event_index < loaded.timeline.history_base_revision:
                    base_messages.append(event)
                    continue
                event_turn_id = event.get("turn_id")
                if event_turn_id == self.turn_id:
                    continue
                if isinstance(event_turn_id, str) and event_turn_id in predecessor_ids:
                    messages_by_turn[event_turn_id].append(event)
                    continue
                if isinstance(event_turn_id, str) and event_turn_id in known_ids:
                    continue
                # Approval continuations and pre-timeline compatibility callers
                # can still append canonical messages without a timeline row.
                # They have no FIFO sequence to interleave with managed turns,
                # so retain them after all durable predecessor groups.
                compatibility_messages.append(event)
            messages: list[Mapping[str, Any]] = list(base_messages)
            for predecessor in predecessors:
                messages.extend(messages_by_turn[predecessor.turn_id])
            messages.extend(compatibility_messages)
            return tuple(messages)
        raise TimelineCoordinatorError("ordinary Chat history changed while constructing a prompt")

    async def precommit_mutation(
        self,
        *,
        tool_name: str,
        arguments: Mapping[str, Any],
        call_id: str,
    ) -> tuple[str, str]:
        """Durably bind the exact operation before a tool can reach its effect."""
        if self._cancel_requested:
            raise TimelineTurnCancelled("ordinary Chat cancellation blocks a new side effect")
        if self._unstarted_blocked:
            raise TimelineCoordinatorError("an unstarted blocked turn cannot begin a side effect")
        if self._unknown_operation:
            raise TimelineCoordinatorError("unknown operation state blocks later side effects")
        normalized_call_id = str(call_id or "").strip()
        if not normalized_call_id:
            raise TimelineCoordinatorError("side-effecting tool call has no stable call_id")
        arguments_digest = tool_arguments_digest(
            tool_name=tool_name,
            arguments=arguments,
        )
        operation_id = _operation_id(self.turn_id, normalized_call_id, arguments_digest)
        descriptor = OperationDescriptor(
            operation_id=operation_id,
            tool_name=str(tool_name),
            arguments_digest=arguments_digest,
            call_id=normalized_call_id,
        )
        async with self._operation_dispatch_lock:
            result = await self._mutate(
                lambda revision: self._repository.record_operation_precommit(
                    self.session_id,
                    turn_id=self.turn_id,
                    expected_history_revision=revision,
                    owner=self.owner,
                    token=self.token,
                    descriptor=descriptor,
                    now=self._now(),
                )
            )
            if result.status != "precommitted":
                raise TimelineCoordinatorError(
                    f"unable to precommit mutation: {result.status}: {result.message or ''}"
                )
            if operation_id in self._dispatched_operation_ids:
                raise TimelineCoordinatorError(
                    "the exact timeline operation was already dispatched"
                )
            self._dispatched_operation_ids.add(operation_id)
        return operation_id, arguments_digest

    async def mark_mutation_started(self, *, operation_id: str) -> None:
        """Record the last durable boundary before the physical side effect."""
        if self._cancel_requested:
            raise TimelineTurnCancelled("ordinary Chat cancellation blocks a side effect")
        if self._unstarted_blocked:
            raise TimelineCoordinatorError("an unstarted blocked turn cannot begin a side effect")
        result = await self._mutate(
            lambda revision: self._repository.mark_side_effect_boundary(
                self.session_id,
                turn_id=self.turn_id,
                expected_history_revision=revision,
                owner=self.owner,
                token=self.token,
                boundary="started",
                operation_id=operation_id,
                now=self._now(),
            )
        )
        if result.status != "boundary_updated":
            raise TimelineCoordinatorError(
                f"unable to start mutation boundary: {result.status}: {result.message or ''}"
            )
        self._current_operation_id = operation_id

    async def before_mutation(
        self,
        *,
        tool_name: str,
        arguments: Mapping[str, Any],
        call_id: str,
    ) -> str:
        """Compatibility helper for callers that own both lifecycle boundaries."""
        operation_id, _ = await self.precommit_mutation(
            tool_name=tool_name,
            arguments=arguments,
            call_id=call_id,
        )
        await self.mark_mutation_started(operation_id=operation_id)
        return operation_id

    async def block_unstarted_turn(self) -> None:
        """Prevent effects while retaining the lane for transcript terminalization."""
        if self._current_operation_id is not None:
            raise TimelineCoordinatorError("a started operation cannot be blocked as unstarted")
        self._unstarted_blocked = True

    async def fail_closed_unstarted_turn(self) -> None:
        """Compatibility alias; terminal release is owned by ``finish()``."""
        await self.block_unstarted_turn()

    async def persist_tool_result(
        self,
        *,
        operation_id: str,
        event_id: str,
        sequence: int,
        payload: Mapping[str, Any],
        error: str | None,
        unknown: bool = False,
        disposition: str | None = None,
    ) -> bool:
        """Durably persist the tool result and descriptor transition in one CAS."""
        if not operation_id:
            return False
        status = (
            disposition
            if disposition in {"succeeded", "failed", "unknown"}
            else "unknown" if unknown else ("failed" if error else "succeeded")
        )
        is_unknown = status == "unknown"
        companion = {
            "type": "turn_event",
            "schema_version": 1,
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "event_id": event_id,
            "seq": sequence,
            "phase": "tool_call_result",
            "timestamp": datetime.now(tz=UTC).isoformat(),
            "payload": dict(payload),
        }
        result = await self._mutate(
            lambda revision: self._repository.record_operation_result(
                self.session_id,
                turn_id=self.turn_id,
                expected_history_revision=revision,
                owner=self.owner,
                token=self.token,
                operation_id=operation_id,
                status=status,  # type: ignore[arg-type]
                result_digest=None if is_unknown else _digest(payload),
                receipt_reference=None if is_unknown else event_id,
                companion_events=(companion,),
                now=self._now(),
            )
        )
        if result.status != "operation_result":
            raise TimelineCoordinatorError(
                f"unable to persist mutation result: {result.status}: {result.message or ''}"
            )
        self._current_operation_id = None
        self._unknown_operation = is_unknown
        return True

    async def persist_approval_pending(
        self,
        *,
        operation_id: str,
        event_id: str,
        sequence: int,
        payload: Mapping[str, Any],
    ) -> None:
        """Commit the approval interrupt event without recording a tool result."""
        result = await self._mutate(
            lambda revision: self._repository.record_operation_approval_pending(
                self.session_id,
                turn_id=self.turn_id,
                expected_history_revision=revision,
                owner=self.owner,
                token=self.token,
                operation_id=operation_id,
                companion_events=(
                    {
                        "type": "turn_event",
                        "schema_version": 1,
                        "session_id": self.session_id,
                        "turn_id": self.turn_id,
                        "event_id": event_id,
                        "seq": sequence,
                        "phase": "tool_call_result",
                        "timestamp": datetime.now(tz=UTC).isoformat(),
                        "payload": dict(payload),
                    },
                ),
                now=self._now(),
            )
        )
        if result.status != "precommitted":
            raise TimelineCoordinatorError(
                "unable to persist pending approval: "
                f"{result.status}: {result.message or ''}"
            )

    async def abandon_pre_effect_operation(
        self,
        *,
        operation_id: str,
        event_id: str,
        sequence: int,
        payload: Mapping[str, Any],
    ) -> None:
        """Atomically receipt a known no-effect tool result before terminal blocking."""
        result = await self._mutate(
            lambda revision: self._repository.abandon_operation_pre_effect(
                self.session_id,
                turn_id=self.turn_id,
                expected_history_revision=revision,
                owner=self.owner,
                token=self.token,
                operation_id=operation_id,
                result_digest=_digest(payload),
                receipt_reference=event_id,
                companion_events=(
                    {
                        "type": "turn_event",
                        "schema_version": 1,
                        "session_id": self.session_id,
                        "turn_id": self.turn_id,
                        "event_id": event_id,
                        "seq": sequence,
                        "phase": "tool_call_result",
                        "timestamp": datetime.now(tz=UTC).isoformat(),
                        "payload": dict(payload),
                    },
                ),
                now=self._now(),
            )
        )
        if result.status != "operation_abandoned":
            raise TimelineCoordinatorError(
                f"unable to record pre-effect abandonment: {result.status}: {result.message or ''}"
            )

    async def request_cancel(self) -> None:
        """Cancel queued work now and defer a claimed lane to its worker exit."""
        self._cancel_requested = True
        loaded = await self._repository.load(self.session_id)
        if loaded.history_revision is None:
            return
        self._history_revision = loaded.history_revision
        turn = (
            next(
                (item for item in loaded.timeline.turns if item.turn_id == self.turn_id),
                None,
            )
            if loaded.timeline is not None
            else None
        )
        if turn is not None and turn.status == "running":
            # The worker still owns the model/tool boundary.  Releasing it
            # here would allow a following turn to start before cancellation
            # has actually stopped the in-flight work.
            return
        result = await self._mutate(
            lambda revision: self._repository.cancel(
                self.session_id,
                turn_id=self.turn_id,
                expected_history_revision=revision,
                owner=self.owner if self._claimed else None,
                token=self.token if self._claimed else None,
                now=self._now(),
            )
        )
        if result.status in {"terminal", "already_terminal", "missing", "lease_invalid", "lease_stale", "invalid"}:
            return
        raise TimelineCoordinatorError(
            f"unable to cancel ordinary Chat turn: {result.status}: {result.message or ''}"
        )

    async def finish(
        self,
        *,
        cancelled: bool = False,
        failed: bool = False,
        companion_events: Sequence[Mapping[str, Any]] = (),
    ) -> None:
        """Terminalize after transcript persistence; never release an unknown effect as known."""
        if self._closed:
            return
        await self.stop_heartbeat()
        if not self._claimed:
            self._closed = True
            return
        if self._current_operation_id is not None:
            await self._mark_current_unknown()
        if self._unknown_operation:
            outcome = "unknown"
            cancellation_outcome = None
        elif cancelled or self._cancel_requested:
            result = await self._mutate(
                lambda revision: self._repository.cancel(
                    self.session_id,
                    turn_id=self.turn_id,
                    expected_history_revision=revision,
                    owner=self.owner,
                    token=self.token,
                    now=self._now(),
                )
            )
            if result.status not in {"terminal", "already_terminal"}:
                raise TimelineCoordinatorError(
                    f"unable to terminally cancel ordinary Chat turn: {result.status}"
                )
            self._closed = True
            return
        else:
            outcome = "blocked" if (failed or self._unstarted_blocked) else "completed"
            cancellation_outcome = None
        result = await self._mutate(
            lambda revision: self._repository.terminal(
                self.session_id,
                turn_id=self.turn_id,
                expected_history_revision=revision,
                owner=self.owner,
                token=self.token,
                outcome=outcome,  # type: ignore[arg-type]
                cancellation_outcome=cancellation_outcome,
                companion_events=companion_events,
                now=self._now(),
            )
        )
        if result.status != "terminal":
            raise TimelineCoordinatorError(
                f"unable to terminalize ordinary Chat turn: {result.status}: {result.message or ''}"
            )
        self._closed = True

    async def _mark_current_unknown(self) -> None:
        operation_id = self._current_operation_id
        if operation_id is None:
            return
        result = await self._mutate(
            lambda revision: self._repository.record_operation_result(
                self.session_id,
                turn_id=self.turn_id,
                expected_history_revision=revision,
                owner=self.owner,
                token=self.token,
                operation_id=operation_id,
                status="unknown",
                now=self._now(),
            )
        )
        if result.status != "operation_result":
            raise TimelineCoordinatorError(
                f"unable to quarantine interrupted mutation: {result.status}: {result.message or ''}"
            )
        self._current_operation_id = None
        self._unknown_operation = True

    async def _heartbeat(self) -> None:
        try:
            while not self._closed:
                await asyncio.sleep(max(1.0, self.lease_seconds / 3))
                if self._closed:
                    return
                result = await self._mutate(
                    lambda revision: self._repository.renew_lease(
                        self.session_id,
                        turn_id=self.turn_id,
                        expected_history_revision=revision,
                        owner=self.owner,
                        token=self.token,
                        lease_expires_at=self._lease_expiry().isoformat(),
                        now=self._now(),
                    )
                )
                if result.status != "lease_renewed":
                    self._cancel_requested = True
                    return
        except asyncio.CancelledError:
            raise
        except Exception:
            self._cancel_requested = True

    async def _recover_stale_lane(self, loaded: Any) -> None:
        timeline = getattr(loaded, "timeline", None)
        revision = getattr(loaded, "history_revision", None)
        if timeline is None or not isinstance(revision, str) or not timeline.lane_turn_id:
            return
        active_turn = next(
            (item for item in timeline.turns if item.turn_id == timeline.lane_turn_id),
            None,
        )
        if active_turn is None:
            return
        self._history_revision = revision
        has_started = any(item.status in {"started", "unknown"} for item in active_turn.operation_descriptors)
        if has_started:
            await self._mutate(
                lambda expected: self._repository.recover_stale_started_operation(
                    self.session_id,
                    turn_id=active_turn.turn_id,
                    expected_history_revision=expected,
                    now=self._now(),
                )
            )
        else:
            await self._mutate(
                lambda expected: self._repository.recover_stale_unstarted_turn(
                    self.session_id,
                    turn_id=active_turn.turn_id,
                    expected_history_revision=expected,
                    now=self._now(),
                )
            )

    async def _recover_expired_admission(self, loaded: Any) -> None:
        timeline = getattr(loaded, "timeline", None)
        revision = getattr(loaded, "history_revision", None)
        if timeline is None or not isinstance(revision, str) or timeline.lane_turn_id is not None:
            return
        queued_head = next((item for item in timeline.turns if item.status == "queued"), None)
        if queued_head is None:
            return
        self._history_revision = revision
        await self._mutate(
            lambda expected: self._repository.recover_expired_queued_admission(
                self.session_id,
                turn_id=queued_head.turn_id,
                expected_history_revision=expected,
                now=self._now(),
            )
        )

    def _lease_is_stale(self, timeline: Any) -> bool:
        value = getattr(timeline, "lane_lease_expires_at", None)
        if not isinstance(value, str):
            return False
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")) <= self._now()
        except ValueError:
            return True

    def _admission_is_stale(self, turn: Any) -> bool:
        value = getattr(turn, "admission_lease_expires_at", None)
        if not isinstance(value, str):
            return False
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")) <= self._now()
        except ValueError:
            return False

    def _admission_needs_renewal(self, turn: Any) -> bool:
        value = getattr(turn, "admission_lease_expires_at", None)
        if not isinstance(value, str):
            return False
        try:
            expiry = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return False
        return expiry <= self._now() + timedelta(seconds=max(1.0, self.lease_seconds / 3))

    @staticmethod
    def _now() -> datetime:
        return datetime.now(tz=UTC)

    def _lease_expiry(self) -> datetime:
        return self._now() + timedelta(seconds=self.lease_seconds)

    def _admission_lease_expiry(self) -> datetime:
        return self._now() + timedelta(seconds=self.lease_seconds)

    async def _mutate(self, operation: Any) -> TimelineMutationResult:
        """Retry only pure CAS mutations; this lock never surrounds external work."""
        async with self._state_lock:
            for _ in range(8):
                if self._history_revision is None:
                    loaded = await self._repository.load(self.session_id)
                    if loaded.history_revision is None:
                        raise TimelineCoordinatorError("ordinary Chat timeline has no history revision")
                    self._history_revision = loaded.history_revision
                result = await operation(self._history_revision)
                if result.status == "rebase_required":
                    self._history_revision = result.history_revision
                    continue
                if result.history_revision is not None:
                    self._history_revision = result.history_revision
                return result
        raise TimelineCoordinatorError("ordinary Chat timeline CAS repeatedly rebased")


def _digest(value: Mapping[str, Any]) -> str:
    raw = json.dumps(
        dict(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _operation_id(turn_id: str, call_id: str, arguments_digest: str) -> str:
    return "ordinary-chat-operation-" + hashlib.sha256(
        f"{turn_id}\x00{call_id}\x00{arguments_digest}".encode()
    ).hexdigest()


async def mark_context_side_effect_started(context: Any) -> None:
    """Cross the effect boundary for a registry-precommitted tool operation."""
    state = getattr(context, "state", None)
    if not isinstance(state, dict):
        return
    lifecycle = state.get("timeline_tool_lifecycle")
    binding = state.get("timeline_pending_operation")
    if lifecycle is None or not isinstance(binding, Mapping):
        return
    operation_id = str(binding.get("operation_id") or "").strip()
    if not operation_id:
        raise TimelineCoordinatorError("timeline operation binding is missing operation_id")
    await lifecycle.mark_mutation_started(operation_id=operation_id)
    state["timeline_operation_started"] = operation_id


def timeline_operation_metadata(context: Any) -> dict[str, str]:
    """Return the host-owned identity carried with a tool result."""
    state = getattr(context, "state", None)
    if not isinstance(state, Mapping):
        return {}
    binding = state.get("timeline_pending_operation")
    if not isinstance(binding, Mapping):
        return {}
    operation_id = str(binding.get("operation_id") or "").strip()
    arguments_digest = str(binding.get("arguments_digest") or "").strip()
    call_id = str(binding.get("call_id") or "").strip()
    if not operation_id or not arguments_digest or not call_id:
        return {}
    return {
        "timeline_operation_id": operation_id,
        "operation_id": operation_id,
        "arguments_digest": arguments_digest,
        "call_id": call_id,
    }


def timeline_pending_operation_binding(
    context: Any,
    *,
    tool_name: str,
    arguments: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Read and validate the exact host-owned operation reserved by the timeline.

    Approval-producing tools must use this identity verbatim.  A missing
    binding means the call is outside ordinary-Chat timeline execution; a
    malformed or mismatched binding is fail-closed rather than silently
    creating a second operation identity.
    """
    state = getattr(context, "state", None)
    if not isinstance(state, Mapping):
        return None
    lifecycle = state.get("timeline_tool_lifecycle")
    binding = state.get("timeline_pending_operation")
    if binding is None:
        if lifecycle is None:
            return None
        raise TimelineCoordinatorError("timeline operation binding is missing")
    if not isinstance(binding, Mapping):
        raise TimelineCoordinatorError("timeline operation binding is invalid")
    operation_id = str(binding.get("operation_id") or "").strip()
    arguments_digest = str(binding.get("arguments_digest") or "").strip()
    call_id = str(binding.get("call_id") or "").strip()
    bound_tool_name = str(binding.get("tool_name") or "").strip()
    if not operation_id or not arguments_digest or not call_id or not bound_tool_name:
        raise TimelineCoordinatorError("timeline operation binding is incomplete")
    if bound_tool_name != tool_name:
        raise TimelineCoordinatorError("timeline operation binding tool does not match call")
    bound_arguments = binding.get("arguments")
    if not isinstance(bound_arguments, Mapping):
        raise TimelineCoordinatorError("timeline operation binding arguments are missing")
    normalized_arguments = dict(bound_arguments)
    actual_digest = tool_arguments_digest(
        tool_name=tool_name,
        arguments=normalized_arguments,
    )
    if arguments_digest != actual_digest:
        raise TimelineCoordinatorError("timeline operation binding digest is invalid")
    if arguments is not None and dict(arguments) != normalized_arguments:
        raise TimelineCoordinatorError("timeline operation binding arguments do not match call")
    return {
        "operation_id": operation_id,
        "arguments_digest": arguments_digest,
        "call_id": call_id,
        "tool_name": bound_tool_name,
        "arguments": normalized_arguments,
    }


__all__ = [
    "TimelineCoordinator",
    "TimelineCoordinatorError",
    "TimelineTurnCancelled",
    "mark_context_side_effect_started",
    "timeline_pending_operation_binding",
    "timeline_operation_metadata",
]
