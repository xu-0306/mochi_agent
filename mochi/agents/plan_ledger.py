"""Durable task-plan ledger contracts for ordinary Chat."""

from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Literal, Protocol, cast

PlanStatus = Literal["active", "completed", "blocked", "cancelled"]
PlanItemStatus = Literal["pending", "in_progress", "completed", "blocked", "cancelled"]
PlanLedgerLoadStatus = Literal["loaded", "missing", "invalid", "unsupported_version"]
PlanLedgerSaveStatus = Literal["saved", "conflict", "invalid"]

PLAN_LEDGER_VERSION = "plan-ledger-v1"
PLAN_LEDGER_EVENT = "ordinary_chat_plan_ledger_updated"
PLAN_LEDGER_EVENT_SCHEMA_VERSION = 1

_PLAN_STATUSES = frozenset({"active", "completed", "blocked", "cancelled"})
_PLAN_ITEM_STATUSES = frozenset(
    {"pending", "in_progress", "completed", "blocked", "cancelled"}
)
_LOAD_STATUSES = frozenset({"loaded", "missing", "invalid", "unsupported_version"})
_SAVE_STATUSES = frozenset({"saved", "conflict", "invalid"})
_TERMINAL_LEDGER_STATUSES = frozenset({"completed", "cancelled"})
_TERMINAL_ITEM_STATUSES = frozenset({"completed", "cancelled"})
_PLAN_LEDGER_EVENT_FIELDS = frozenset(
    {
        "type",
        "event",
        "schema_version",
        "session_id",
        "goal_id",
        "ledger_id",
        "ledger_revision",
        "turn_id",
        "idempotency_key",
        "plan_ledger",
        "timestamp",
    }
)


def _clean_text(
    value: Any,
    *,
    field_name: str,
    allow_empty: bool = False,
    max_chars: int = 1_000,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    cleaned = " ".join(value.split())
    if not cleaned and not allow_empty:
        raise ValueError(f"{field_name} must not be empty")
    if len(cleaned) > max_chars:
        raise ValueError(f"{field_name} exceeds {max_chars} characters")
    return cleaned


def _clean_non_negative_int(value: Any, *, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _require_exact_keys(
    payload: Mapping[str, Any],
    *,
    expected: frozenset[str],
    field_name: str,
) -> None:
    actual = frozenset(payload)
    unexpected = sorted(actual - expected)
    missing = sorted(expected - actual)
    details: list[str] = []
    if unexpected:
        details.append(f"unexpected keys: {unexpected}")
    if missing:
        details.append(f"missing keys: {missing}")
    if details:
        raise ValueError(f"{field_name} " + "; ".join(details))


def _clean_text_tuple(
    value: Any,
    *,
    field_name: str,
    min_items: int = 0,
    max_items: int,
    max_chars: int = 240,
) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field_name} must be a list")
    cleaned: list[str] = []
    seen: set[str] = set()
    for index, item in enumerate(value):
        entry = _clean_text(item, field_name=f"{field_name}[{index}]", max_chars=max_chars)
        if entry in seen:
            continue
        seen.add(entry)
        cleaned.append(entry)
    if len(cleaned) < min_items:
        raise ValueError(f"{field_name} must contain at least {min_items} item(s)")
    if len(cleaned) > max_items:
        raise ValueError(f"{field_name} exceeds {max_items} items")
    return tuple(cleaned)


def _clean_optional_text(
    value: Any,
    *,
    field_name: str,
    max_chars: int = 400,
) -> str | None:
    if value is None:
        return None
    return _clean_text(value, field_name=field_name, max_chars=max_chars)


@dataclass(frozen=True)
class PlanLedgerLimits:
    max_items: int = 12
    max_dependencies_per_item: int = 8
    max_success_criteria_per_item: int = 8
    max_source_turn_ids_per_item: int = 8
    max_evidence_refs_per_item: int = 8
    max_reason_codes: int = 8
    max_identifier_chars: int = 128
    max_title_chars: int = 200
    max_objective_chars: int = 1_000
    max_criterion_chars: int = 240
    max_blocker_reason_chars: int = 400

    def __post_init__(self) -> None:
        for field_name in (
            "max_items",
            "max_dependencies_per_item",
            "max_success_criteria_per_item",
            "max_source_turn_ids_per_item",
            "max_evidence_refs_per_item",
            "max_reason_codes",
            "max_identifier_chars",
            "max_title_chars",
            "max_objective_chars",
            "max_criterion_chars",
            "max_blocker_reason_chars",
        ):
            _clean_non_negative_int(getattr(self, field_name), field_name=field_name)


DEFAULT_PLAN_LEDGER_LIMITS = PlanLedgerLimits()


@dataclass(frozen=True)
class PlanItem:
    item_id: str
    title: str
    status: PlanItemStatus
    dependencies: tuple[str, ...]
    success_criteria: tuple[str, ...]
    source_turn_ids: tuple[str, ...]
    evidence_refs: tuple[str, ...] = ()
    blocker_reason: str | None = None
    attempts: int = 0

    def __post_init__(self) -> None:
        limits = DEFAULT_PLAN_LEDGER_LIMITS
        object.__setattr__(
            self,
            "item_id",
            _clean_text(
                self.item_id,
                field_name="item_id",
                max_chars=limits.max_identifier_chars,
            ),
        )
        object.__setattr__(
            self,
            "title",
            _clean_text(
                self.title,
                field_name="title",
                max_chars=limits.max_title_chars,
            ),
        )
        if self.status not in _PLAN_ITEM_STATUSES:
            raise ValueError(f"unsupported plan item status: {self.status!r}")
        object.__setattr__(
            self,
            "dependencies",
            _clean_text_tuple(
                self.dependencies,
                field_name="dependencies",
                max_items=limits.max_dependencies_per_item,
                max_chars=limits.max_identifier_chars,
            ),
        )
        if self.item_id in self.dependencies:
            raise ValueError("plan item cannot depend on itself")
        object.__setattr__(
            self,
            "success_criteria",
            _clean_text_tuple(
                self.success_criteria,
                field_name="success_criteria",
                min_items=1,
                max_items=limits.max_success_criteria_per_item,
                max_chars=limits.max_criterion_chars,
            ),
        )
        object.__setattr__(
            self,
            "source_turn_ids",
            _clean_text_tuple(
                self.source_turn_ids,
                field_name="source_turn_ids",
                min_items=1,
                max_items=limits.max_source_turn_ids_per_item,
                max_chars=limits.max_identifier_chars,
            ),
        )
        object.__setattr__(
            self,
            "evidence_refs",
            _clean_text_tuple(
                self.evidence_refs,
                field_name="evidence_refs",
                max_items=limits.max_evidence_refs_per_item,
                max_chars=limits.max_identifier_chars,
            ),
        )
        object.__setattr__(
            self,
            "blocker_reason",
            _clean_optional_text(
                self.blocker_reason,
                field_name="blocker_reason",
                max_chars=limits.max_blocker_reason_chars,
            ),
        )
        object.__setattr__(self, "attempts", _clean_non_negative_int(self.attempts, field_name="attempts"))
        if self.status == "blocked" and self.blocker_reason is None:
            raise ValueError("blocked plan items require blocker_reason")
        if self.status != "blocked" and self.blocker_reason is not None:
            raise ValueError("only blocked plan items may carry blocker_reason")
        if self.status == "completed" and not self.evidence_refs:
            raise ValueError("completed plan items require evidence_refs")
        if self.status != "completed" and self.evidence_refs:
            raise ValueError("only completed plan items may carry evidence_refs")

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "title": self.title,
            "status": self.status,
            "dependencies": list(self.dependencies),
            "success_criteria": list(self.success_criteria),
            "source_turn_ids": list(self.source_turn_ids),
            "evidence_refs": list(self.evidence_refs),
            "blocker_reason": self.blocker_reason,
            "attempts": self.attempts,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PlanItem:
        expected = frozenset(
            {
                "item_id",
                "title",
                "status",
                "dependencies",
                "success_criteria",
                "source_turn_ids",
                "evidence_refs",
                "blocker_reason",
                "attempts",
            }
        )
        _require_exact_keys(payload, expected=expected, field_name="plan item")
        return cls(
            item_id=payload.get("item_id"),
            title=payload.get("title"),
            status=cast(PlanItemStatus, payload.get("status")),
            dependencies=tuple(payload.get("dependencies", ())),
            success_criteria=tuple(payload.get("success_criteria", ())),
            source_turn_ids=tuple(payload.get("source_turn_ids", ())),
            evidence_refs=tuple(payload.get("evidence_refs", ())),
            blocker_reason=payload.get("blocker_reason"),
            attempts=payload.get("attempts"),
        )


@dataclass(frozen=True)
class PlanLedger:
    ledger_version: str
    ledger_id: str
    session_id: str
    goal_id: str
    revision: int
    status: PlanStatus
    objective: str
    reason_codes: tuple[str, ...]
    items: tuple[PlanItem, ...]
    created_turn_id: str
    updated_turn_id: str

    def __post_init__(self) -> None:
        limits = DEFAULT_PLAN_LEDGER_LIMITS
        if self.ledger_version != PLAN_LEDGER_VERSION:
            raise ValueError(f"unsupported ledger_version: {self.ledger_version!r}")
        for field_name in ("ledger_id", "session_id", "goal_id", "created_turn_id", "updated_turn_id"):
            object.__setattr__(
                self,
                field_name,
                _clean_text(
                    getattr(self, field_name),
                    field_name=field_name,
                    max_chars=limits.max_identifier_chars,
                ),
            )
        object.__setattr__(
            self,
            "revision",
            _clean_non_negative_int(self.revision, field_name="revision"),
        )
        if self.status not in _PLAN_STATUSES:
            raise ValueError(f"unsupported plan status: {self.status!r}")
        object.__setattr__(
            self,
            "objective",
            _clean_text(
                self.objective,
                field_name="objective",
                max_chars=limits.max_objective_chars,
            ),
        )
        object.__setattr__(
            self,
            "reason_codes",
            _clean_text_tuple(
                self.reason_codes,
                field_name="reason_codes",
                max_items=limits.max_reason_codes,
                max_chars=limits.max_identifier_chars,
            ),
        )
        if not isinstance(self.items, tuple):
            raise TypeError("items must be a tuple")
        if not self.items:
            raise ValueError("plan ledger must contain at least one item")
        if len(self.items) > limits.max_items:
            raise ValueError(f"plan ledger exceeds {limits.max_items} items")
        validate_plan_ledger_items(self.items)
        if self.status in _TERMINAL_LEDGER_STATUSES and self.updated_turn_id == self.created_turn_id and self.revision == 0:
            raise ValueError("terminal plan ledgers must be durably advanced")

    def to_dict(self) -> dict[str, Any]:
        return {
            "ledger_version": self.ledger_version,
            "ledger_id": self.ledger_id,
            "session_id": self.session_id,
            "goal_id": self.goal_id,
            "revision": self.revision,
            "status": self.status,
            "objective": self.objective,
            "reason_codes": list(self.reason_codes),
            "items": [item.to_dict() for item in self.items],
            "created_turn_id": self.created_turn_id,
            "updated_turn_id": self.updated_turn_id,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PlanLedger:
        expected = frozenset(
            {
                "ledger_version",
                "ledger_id",
                "session_id",
                "goal_id",
                "revision",
                "status",
                "objective",
                "reason_codes",
                "items",
                "created_turn_id",
                "updated_turn_id",
            }
        )
        _require_exact_keys(payload, expected=expected, field_name="plan ledger")
        raw_items = payload.get("items")
        if not isinstance(raw_items, list):
            raise TypeError("plan ledger items must be a list")
        return cls(
            ledger_version=_clean_text(
                payload.get("ledger_version"),
                field_name="ledger_version",
                max_chars=64,
            ),
            ledger_id=payload.get("ledger_id"),
            session_id=payload.get("session_id"),
            goal_id=payload.get("goal_id"),
            revision=payload.get("revision"),
            status=cast(PlanStatus, payload.get("status")),
            objective=payload.get("objective"),
            reason_codes=tuple(payload.get("reason_codes", ())),
            items=tuple(PlanItem.from_dict(item) for item in raw_items),
            created_turn_id=payload.get("created_turn_id"),
            updated_turn_id=payload.get("updated_turn_id"),
        )


@dataclass(frozen=True)
class PlanLedgerLoadResult:
    status: PlanLedgerLoadStatus
    ledger: PlanLedger | None = None
    event_index: int | None = None
    message: str | None = None

    def __post_init__(self) -> None:
        if self.status not in _LOAD_STATUSES:
            raise ValueError(f"unsupported load status: {self.status!r}")


@dataclass(frozen=True)
class PlanLedgerSaveResult:
    status: PlanLedgerSaveStatus
    expected_revision: int
    current_revision: int
    ledger: PlanLedger | None = None
    message: str | None = None
    idempotent_replay: bool = False

    def __post_init__(self) -> None:
        if self.status not in _SAVE_STATUSES:
            raise ValueError(f"unsupported save status: {self.status!r}")


class SessionStoreCompatible(Protocol):
    async def append_event_if(
        self,
        session_id: str,
        event: dict[str, Any],
        predicate: Callable[[list[dict[str, Any]]], bool],
    ) -> bool:
        """Append when the predicate accepts the current event history."""

    async def load_session(self, session_id: str) -> list[dict[str, Any]]:
        """Load current durable session events."""


def validate_plan_ledger_items(items: tuple[PlanItem, ...]) -> None:
    item_ids = {item.item_id for item in items}
    if len(item_ids) != len(items):
        raise ValueError("plan ledger item_ids must be unique")
    in_progress_count = sum(1 for item in items if item.status == "in_progress")
    if in_progress_count > 1:
        raise ValueError("at most one plan item may be in_progress")
    item_map = {item.item_id: item for item in items}
    for item in items:
        for dependency in item.dependencies:
            if dependency not in item_ids:
                raise ValueError(f"unknown dependency id: {dependency!r}")
        if item.status == "in_progress":
            missing = [
                dependency
                for dependency in item.dependencies
                if item_map[dependency].status != "completed"
            ]
            if missing:
                raise ValueError(
                    "in_progress items require completed dependencies: "
                    + ", ".join(sorted(missing))
                )

    visited: set[str] = set()
    visiting: set[str] = set()

    def visit(item_id: str) -> None:
        if item_id in visited:
            return
        if item_id in visiting:
            raise ValueError("plan ledger dependencies must be acyclic")
        visiting.add(item_id)
        for dependency in item_map[item_id].dependencies:
            visit(dependency)
        visiting.remove(item_id)
        visited.add(item_id)

    for item in items:
        visit(item.item_id)


class PlanLedgerTransitionValidator:
    """Host-enforced plan mutation rules independent from Engine wiring."""

    def __init__(
        self,
        *,
        recognized_evidence_refs: Collection[str] = (),
    ) -> None:
        self._recognized_evidence_refs = frozenset(
            _clean_text(value, field_name="recognized_evidence_ref", max_chars=128)
            for value in recognized_evidence_refs
        )

    def validate_replacement(
        self,
        *,
        previous: PlanLedger | None,
        proposed: PlanLedger,
    ) -> None:
        self._validate_recognized_evidence(proposed)
        if previous is None:
            return
        if previous.status in _TERMINAL_LEDGER_STATUSES:
            raise ValueError("terminal plan ledgers cannot be replaced")
        if (
            previous.ledger_id != proposed.ledger_id
            or previous.session_id != proposed.session_id
            or previous.goal_id != proposed.goal_id
        ):
            raise ValueError("replacement ledger must preserve durable identity")
        if previous.created_turn_id != proposed.created_turn_id:
            raise ValueError("replacement ledger must preserve created_turn_id")
        prior_items = {item.item_id: item for item in previous.items}
        next_items = {item.item_id: item for item in proposed.items}
        for item_id, prior_item in prior_items.items():
            if prior_item.status in _TERMINAL_ITEM_STATUSES:
                if next_items.get(item_id) != prior_item:
                    raise ValueError(
                        f"terminal plan item {item_id!r} must remain unchanged in replacements"
                    )

    def set_item_status(
        self,
        ledger: PlanLedger,
        *,
        item_id: str,
        status: PlanItemStatus,
        updated_turn_id: str,
        evidence_refs: Collection[str] = (),
        blocker_reason: str | None = None,
    ) -> PlanLedger:
        if ledger.status in _TERMINAL_LEDGER_STATUSES:
            raise ValueError("terminal plan ledgers cannot be mutated")
        if status not in _PLAN_ITEM_STATUSES:
            raise ValueError(f"unsupported plan item status: {status!r}")
        normalized_item_id = _clean_text(item_id, field_name="item_id", max_chars=128)
        normalized_turn_id = _clean_text(
            updated_turn_id,
            field_name="updated_turn_id",
            max_chars=128,
        )
        normalized_blocker = _clean_optional_text(
            blocker_reason,
            field_name="blocker_reason",
            max_chars=DEFAULT_PLAN_LEDGER_LIMITS.max_blocker_reason_chars,
        )
        recognized_evidence = self._recognized_subset(evidence_refs)
        items: list[PlanItem] = []
        found = False
        for item in ledger.items:
            if item.item_id != normalized_item_id:
                items.append(item)
                continue
            found = True
            if item.status in _TERMINAL_ITEM_STATUSES:
                if (
                    item.status != status
                    or tuple(item.evidence_refs) != tuple(recognized_evidence if status == "completed" else ())
                    or item.blocker_reason != (normalized_blocker if status == "blocked" else None)
                ):
                    raise ValueError(f"terminal plan item {item.item_id!r} cannot be mutated")
                items.append(item)
                continue
            if item.status in _TERMINAL_ITEM_STATUSES and item.status != status:
                raise ValueError(f"terminal plan item {item.item_id!r} cannot change status")
            if status == "in_progress":
                missing = self._incomplete_dependencies(ledger, item)
                if missing:
                    raise ValueError(
                        "cannot start item before dependencies complete: "
                        + ", ".join(sorted(missing))
                    )
                if any(
                    other.status == "in_progress" and other.item_id != item.item_id
                    for other in ledger.items
                ):
                    raise ValueError("at most one plan item may be in_progress")
            if status == "completed" and not recognized_evidence:
                raise ValueError("completed plan items require recognized evidence_refs")
            if status == "blocked" and normalized_blocker is None:
                raise ValueError("blocked plan items require blocker_reason")
            if status != "blocked" and normalized_blocker is not None:
                raise ValueError("only blocked plan items may carry blocker_reason")
            attempts = item.attempts + 1 if status == "in_progress" and item.status != "in_progress" else item.attempts
            items.append(
                PlanItem(
                    item_id=item.item_id,
                    title=item.title,
                    status=status,
                    dependencies=item.dependencies,
                    success_criteria=item.success_criteria,
                    source_turn_ids=item.source_turn_ids,
                    evidence_refs=recognized_evidence if status == "completed" else (),
                    blocker_reason=normalized_blocker if status == "blocked" else None,
                    attempts=attempts,
                )
            )
        if not found:
            raise ValueError(f"unknown plan item: {normalized_item_id!r}")
        return replace(
            ledger,
            items=tuple(items),
            updated_turn_id=normalized_turn_id,
        )

    def _recognized_subset(self, evidence_refs: Collection[str]) -> tuple[str, ...]:
        cleaned = tuple(
            dict.fromkeys(
                _clean_text(value, field_name="evidence_ref", max_chars=128)
                for value in evidence_refs
            )
        )
        if not cleaned:
            return ()
        recognized = tuple(ref for ref in cleaned if ref in self._recognized_evidence_refs)
        if len(recognized) != len(cleaned):
            raise ValueError("plan item completion requires host-recognized evidence_refs")
        return recognized

    def _validate_recognized_evidence(self, ledger: PlanLedger) -> None:
        for item in ledger.items:
            if item.status == "completed":
                self._recognized_subset(item.evidence_refs)

    @staticmethod
    def _incomplete_dependencies(ledger: PlanLedger, item: PlanItem) -> tuple[str, ...]:
        item_map = {entry.item_id: entry for entry in ledger.items}
        return tuple(
            dependency
            for dependency in item.dependencies
            if item_map[dependency].status != "completed"
        )


class PlanLedgerRepository:
    """CAS-protected event repository for durable plan ledgers."""

    def __init__(self, session_store: SessionStoreCompatible) -> None:
        self._session_store = session_store

    async def load(
        self,
        session_id: str,
        goal_id: str,
        *,
        ledger_id: str | None = None,
    ) -> PlanLedgerLoadResult:
        events = await self._session_store.load_session(session_id)
        return self._load_from_events(
            session_id=session_id,
            goal_id=goal_id,
            ledger_id=ledger_id,
            events=events,
        )

    async def load_active(
        self,
        session_id: str,
        goal_id: str,
        *,
        ledger_id: str | None = None,
    ) -> PlanLedgerLoadResult:
        loaded = await self.load(session_id, goal_id, ledger_id=ledger_id)
        if loaded.status != "loaded" or loaded.ledger is None:
            return loaded
        if loaded.ledger.status in _TERMINAL_LEDGER_STATUSES:
            return PlanLedgerLoadResult(status="missing")
        return loaded

    async def list_active(
        self,
        session_id: str,
        *,
        goal_id: str | None = None,
    ) -> tuple[PlanLedger, ...]:
        events = await self._session_store.load_session(session_id)
        latest_by_ledger: dict[str, PlanLedger] = {}
        seen_ledgers: set[str] = set()
        for reverse_index, event in enumerate(reversed(events)):
            if event.get("event") != PLAN_LEDGER_EVENT:
                continue
            candidate_ledger_id = event.get("ledger_id")
            if not isinstance(candidate_ledger_id, str) or candidate_ledger_id in seen_ledgers:
                continue
            seen_ledgers.add(candidate_ledger_id)
            loaded = self._load_event(
                session_id=session_id,
                goal_id=goal_id,
                ledger_id=candidate_ledger_id,
                event=event,
                event_index=len(events) - 1 - reverse_index,
            )
            if loaded.status != "loaded" or loaded.ledger is None:
                raise ValueError(loaded.message or "invalid latest plan ledger event")
            if loaded.ledger.status not in _TERMINAL_LEDGER_STATUSES:
                latest_by_ledger[candidate_ledger_id] = loaded.ledger
        return tuple(
            latest_by_ledger[item_id]
            for item_id in sorted(latest_by_ledger, key=lambda key: latest_by_ledger[key].revision)
        )

    async def save(
        self,
        ledger: PlanLedger,
        *,
        expected_revision: int,
        turn_id: str,
        idempotency_key: str,
        timestamp: str | None = None,
    ) -> PlanLedgerSaveResult:
        current = await self.load(
            ledger.session_id,
            ledger.goal_id,
            ledger_id=ledger.ledger_id,
        )
        if current.status in {"invalid", "unsupported_version"}:
            return PlanLedgerSaveResult(
                status="invalid",
                expected_revision=expected_revision,
                current_revision=current.ledger.revision if current.ledger is not None else 0,
                message=current.message or "cannot append from invalid durable plan ledger state",
            )
        current_revision = current.ledger.revision if current.ledger is not None else 0
        next_revision = expected_revision + 1
        next_ledger = replace(ledger, revision=next_revision)
        event = {
            "type": "session_meta",
            "event": PLAN_LEDGER_EVENT,
            "schema_version": PLAN_LEDGER_EVENT_SCHEMA_VERSION,
            "session_id": ledger.session_id,
            "goal_id": ledger.goal_id,
            "ledger_id": ledger.ledger_id,
            "ledger_revision": next_revision,
            "turn_id": _clean_text(turn_id, field_name="turn_id", max_chars=128),
            "idempotency_key": _clean_text(
                idempotency_key,
                field_name="idempotency_key",
                max_chars=256,
            ),
            "plan_ledger": next_ledger.to_dict(),
            "timestamp": timestamp or datetime.now(tz=UTC).isoformat(),
        }
        outcome: PlanLedgerSaveResult | None = None

        def predicate(events: list[dict[str, Any]]) -> bool:
            nonlocal outcome
            loaded = self._load_from_events(
                session_id=ledger.session_id,
                goal_id=ledger.goal_id,
                ledger_id=ledger.ledger_id,
                events=events,
            )
            if loaded.status in {"invalid", "unsupported_version"}:
                outcome = PlanLedgerSaveResult(
                    status="invalid",
                    expected_revision=expected_revision,
                    current_revision=loaded.ledger.revision if loaded.ledger is not None else 0,
                    message=loaded.message or "cannot append from invalid durable plan ledger state",
                )
                return False
            duplicate = self._find_idempotent_event(
                events,
                session_id=ledger.session_id,
                goal_id=ledger.goal_id,
                ledger_id=ledger.ledger_id,
                idempotency_key=cast(str, event["idempotency_key"]),
            )
            if duplicate is not None:
                if self._events_match_for_idempotency(duplicate, event):
                    duplicate_payload = duplicate.get("plan_ledger")
                    duplicate_ledger = (
                        PlanLedger.from_dict(duplicate_payload)
                        if isinstance(duplicate_payload, Mapping)
                        else next_ledger
                    )
                    outcome = PlanLedgerSaveResult(
                        status="saved",
                        expected_revision=expected_revision,
                        current_revision=expected_revision,
                        ledger=duplicate_ledger,
                        message="idempotent replay",
                        idempotent_replay=True,
                    )
                    return False
                outcome = PlanLedgerSaveResult(
                    status="invalid",
                    expected_revision=expected_revision,
                    current_revision=current_revision,
                    message="idempotency key was reused for a different plan mutation",
                )
                return False
            durable_revision = loaded.ledger.revision if loaded.ledger is not None else 0
            if durable_revision != expected_revision:
                outcome = PlanLedgerSaveResult(
                    status="conflict",
                    expected_revision=expected_revision,
                    current_revision=durable_revision,
                    ledger=loaded.ledger,
                    message="plan ledger revision changed before transition",
                )
                return False
            return True

        appended = await self._session_store.append_event_if(ledger.session_id, event, predicate)
        if appended:
            return PlanLedgerSaveResult(
                status="saved",
                expected_revision=expected_revision,
                current_revision=expected_revision,
                ledger=next_ledger,
            )
        if outcome is not None:
            return outcome
        reloaded = await self.load(ledger.session_id, ledger.goal_id, ledger_id=ledger.ledger_id)
        if reloaded.status == "loaded" and reloaded.ledger is not None:
            return PlanLedgerSaveResult(
                status="conflict",
                expected_revision=expected_revision,
                current_revision=reloaded.ledger.revision,
                ledger=reloaded.ledger,
                message="plan ledger append lost the SessionStore CAS",
            )
        return PlanLedgerSaveResult(
            status="invalid",
            expected_revision=expected_revision,
            current_revision=current_revision,
            message="plan ledger append did not produce a durable result",
        )

    @staticmethod
    def _find_idempotent_event(
        events: list[dict[str, Any]],
        *,
        session_id: str,
        goal_id: str,
        ledger_id: str,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        for event in reversed(events):
            if (
                event.get("event") == PLAN_LEDGER_EVENT
                and event.get("session_id") == session_id
                and event.get("goal_id") == goal_id
                and event.get("ledger_id") == ledger_id
                and event.get("idempotency_key") == idempotency_key
            ):
                return dict(event)
        return None

    @staticmethod
    def _events_match_for_idempotency(
        persisted: Mapping[str, Any],
        candidate: Mapping[str, Any],
    ) -> bool:
        comparable_fields = _PLAN_LEDGER_EVENT_FIELDS - {"timestamp"}
        if frozenset(persisted) != frozenset(candidate):
            return False
        return all(persisted.get(field_name) == candidate.get(field_name) for field_name in comparable_fields)

    @staticmethod
    def _load_from_events(
        *,
        session_id: str,
        goal_id: str,
        ledger_id: str | None,
        events: list[dict[str, Any]],
    ) -> PlanLedgerLoadResult:
        for reverse_index, event in enumerate(reversed(events)):
            if event.get("event") != PLAN_LEDGER_EVENT:
                continue
            candidate_goal_id = event.get("goal_id")
            candidate_ledger_id = event.get("ledger_id")
            if candidate_goal_id != goal_id:
                continue
            if ledger_id is not None and candidate_ledger_id != ledger_id:
                continue
            return PlanLedgerRepository._load_event(
                session_id=session_id,
                goal_id=goal_id,
                ledger_id=cast(str | None, candidate_ledger_id),
                event=event,
                event_index=len(events) - 1 - reverse_index,
            )
        return PlanLedgerLoadResult(status="missing")

    @staticmethod
    def _load_event(
        *,
        session_id: str,
        goal_id: str | None,
        ledger_id: str | None,
        event: dict[str, Any],
        event_index: int,
    ) -> PlanLedgerLoadResult:
        version = event.get("schema_version")
        if type(version) is not int:
            return PlanLedgerLoadResult(
                status="invalid",
                event_index=event_index,
                message="plan ledger event schema_version must be an integer",
            )
        if version != PLAN_LEDGER_EVENT_SCHEMA_VERSION:
            status: PlanLedgerLoadStatus = "unsupported_version" if version > PLAN_LEDGER_EVENT_SCHEMA_VERSION else "invalid"
            return PlanLedgerLoadResult(
                status=status,
                event_index=event_index,
                message=f"unsupported plan ledger event schema version: {version}",
            )
        try:
            _require_exact_keys(
                event,
                expected=_PLAN_LEDGER_EVENT_FIELDS,
                field_name="plan ledger event",
            )
            if event.get("type") != "session_meta":
                raise ValueError("plan ledger event type must be session_meta")
            if event.get("event") != PLAN_LEDGER_EVENT:
                raise ValueError("plan ledger event name is invalid")
            if event.get("session_id") != session_id:
                raise ValueError("plan ledger event session_id does not match request")
            if goal_id is not None and event.get("goal_id") != goal_id:
                raise ValueError("plan ledger event goal_id does not match request")
            if ledger_id is not None and event.get("ledger_id") != ledger_id:
                raise ValueError("plan ledger event ledger_id does not match request")
            timestamp = _clean_text(event.get("timestamp"), field_name="timestamp", max_chars=128)
            if not timestamp:
                raise ValueError("plan ledger event timestamp must not be empty")
            envelope_turn_id = _clean_text(event.get("turn_id"), field_name="turn_id", max_chars=128)
            envelope_idempotency_key = _clean_text(
                event.get("idempotency_key"),
                field_name="idempotency_key",
                max_chars=256,
            )
            envelope_revision = _clean_non_negative_int(
                event.get("ledger_revision"),
                field_name="ledger_revision",
            )
            del timestamp, envelope_turn_id, envelope_idempotency_key
            payload = event.get("plan_ledger")
            if not isinstance(payload, Mapping):
                raise TypeError("plan ledger event plan_ledger must be an object")
            ledger = PlanLedger.from_dict(payload)
            if ledger.session_id != session_id or ledger.goal_id != event.get("goal_id"):
                raise ValueError("plan ledger payload identity does not match its envelope")
            if ledger.ledger_id != event.get("ledger_id"):
                raise ValueError("plan ledger payload ledger_id does not match its envelope")
            if ledger.revision != envelope_revision:
                raise ValueError("plan ledger payload revision does not match its envelope")
        except Exception as exc:
            return PlanLedgerLoadResult(
                status="invalid",
                event_index=event_index,
                message=f"invalid plan ledger event: {exc}",
            )
        return PlanLedgerLoadResult(status="loaded", ledger=ledger, event_index=event_index)
