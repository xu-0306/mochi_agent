"""Durable bounded discovery metadata for ordinary Chat tool retrieval."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from typing import Any, Literal, Protocol

from mochi.sessions.store import SessionStore

ToolDiscoveryStateLoadStatus = Literal["loaded", "missing", "invalid", "unsupported_version"]
ToolDiscoveryStateSaveStatus = Literal["saved", "conflict", "invalid"]

TOOL_DISCOVERY_STATE_VERSION = "tool-discovery-state-v1"
TOOL_DISCOVERY_EVENT = "ordinary_chat_tool_discovery_updated"
TOOL_DISCOVERY_EVENT_SCHEMA_VERSION = 1

_ENTRY_FIELDS = frozenset(
    {
        "tool_name",
        "source_query_hash",
        "discovered_turn_id",
        "discovered_turn_index",
        "last_used_turn_id",
        "last_used_turn_index",
        "catalog_fingerprint",
        "catalog_generation",
        "capability_risk_class",
    }
)
_STATE_FIELDS = frozenset(
    {
        "state_version",
        "session_id",
        "revision",
        "catalog_generation",
        "catalog_fingerprint",
        "entries",
    }
)
_EVENT_FIELDS = frozenset(
    {
        "type",
        "event",
        "schema_version",
        "session_id",
        "state_revision",
        "turn_id",
        "catalog_generation",
        "idempotency_key",
        "tool_discovery_state",
        "timestamp",
    }
)


class SessionStoreCompatible(Protocol):
    async def load_session(self, session_id: str) -> list[dict[str, Any]]:
        ...

    async def append_event_if(
        self,
        session_id: str,
        event: dict[str, Any],
        predicate: Any,
    ) -> bool:
        ...


@dataclass(frozen=True)
class ToolDiscoveryObservation:
    """One discovered or re-used tool observation before persistence."""

    tool_name: str
    source_query_hash: str
    turn_id: str
    turn_index: int
    catalog_fingerprint: str
    catalog_generation: int
    capability_risk_class: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_name", _clean_text(self.tool_name, field_name="tool_name"))
        object.__setattr__(
            self,
            "source_query_hash",
            _clean_text(self.source_query_hash, field_name="source_query_hash", max_chars=128),
        )
        object.__setattr__(self, "turn_id", _clean_text(self.turn_id, field_name="turn_id"))
        object.__setattr__(
            self,
            "turn_index",
            _clean_non_negative_int(self.turn_index, field_name="turn_index"),
        )
        object.__setattr__(
            self,
            "catalog_fingerprint",
            _clean_text(
                self.catalog_fingerprint,
                field_name="catalog_fingerprint",
                max_chars=128,
            ),
        )
        object.__setattr__(
            self,
            "catalog_generation",
            _clean_non_negative_int(
                self.catalog_generation,
                field_name="catalog_generation",
            ),
        )
        object.__setattr__(
            self,
            "capability_risk_class",
            _clean_text(
                self.capability_risk_class,
                field_name="capability_risk_class",
                max_chars=128,
            ),
        )


@dataclass(frozen=True)
class DiscoveredToolEntry:
    """Strict persisted discovery metadata for one tool/query pair."""

    tool_name: str
    source_query_hash: str
    discovered_turn_id: str
    discovered_turn_index: int
    last_used_turn_id: str
    last_used_turn_index: int
    catalog_fingerprint: str
    catalog_generation: int
    capability_risk_class: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "tool_name", _clean_text(self.tool_name, field_name="tool_name"))
        object.__setattr__(
            self,
            "source_query_hash",
            _clean_text(self.source_query_hash, field_name="source_query_hash", max_chars=128),
        )
        object.__setattr__(
            self,
            "discovered_turn_id",
            _clean_text(self.discovered_turn_id, field_name="discovered_turn_id"),
        )
        object.__setattr__(
            self,
            "discovered_turn_index",
            _clean_non_negative_int(
                self.discovered_turn_index,
                field_name="discovered_turn_index",
            ),
        )
        object.__setattr__(
            self,
            "last_used_turn_id",
            _clean_text(self.last_used_turn_id, field_name="last_used_turn_id"),
        )
        object.__setattr__(
            self,
            "last_used_turn_index",
            _clean_non_negative_int(
                self.last_used_turn_index,
                field_name="last_used_turn_index",
            ),
        )
        if self.last_used_turn_index < self.discovered_turn_index:
            raise ValueError("last_used_turn_index must be greater than or equal to discovered_turn_index")
        object.__setattr__(
            self,
            "catalog_fingerprint",
            _clean_text(
                self.catalog_fingerprint,
                field_name="catalog_fingerprint",
                max_chars=128,
            ),
        )
        object.__setattr__(
            self,
            "catalog_generation",
            _clean_non_negative_int(
                self.catalog_generation,
                field_name="catalog_generation",
            ),
        )
        object.__setattr__(
            self,
            "capability_risk_class",
            _clean_text(
                self.capability_risk_class,
                field_name="capability_risk_class",
                max_chars=128,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool_name": self.tool_name,
            "source_query_hash": self.source_query_hash,
            "discovered_turn_id": self.discovered_turn_id,
            "discovered_turn_index": self.discovered_turn_index,
            "last_used_turn_id": self.last_used_turn_id,
            "last_used_turn_index": self.last_used_turn_index,
            "catalog_fingerprint": self.catalog_fingerprint,
            "catalog_generation": self.catalog_generation,
            "capability_risk_class": self.capability_risk_class,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DiscoveredToolEntry:
        _require_exact_keys(value, expected=_ENTRY_FIELDS, field_name="discovered tool entry")
        return cls(
            tool_name=value["tool_name"],
            source_query_hash=value["source_query_hash"],
            discovered_turn_id=value["discovered_turn_id"],
            discovered_turn_index=value["discovered_turn_index"],
            last_used_turn_id=value["last_used_turn_id"],
            last_used_turn_index=value["last_used_turn_index"],
            catalog_fingerprint=value["catalog_fingerprint"],
            catalog_generation=value["catalog_generation"],
            capability_risk_class=value["capability_risk_class"],
        )


@dataclass(frozen=True)
class ToolDiscoveryState:
    """Strict versioned persisted discovery state."""

    state_version: str
    session_id: str
    revision: int
    catalog_generation: int
    catalog_fingerprint: str
    entries: tuple[DiscoveredToolEntry, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.state_version != TOOL_DISCOVERY_STATE_VERSION:
            raise ValueError(f"unsupported tool discovery state version: {self.state_version!r}")
        object.__setattr__(self, "session_id", _clean_text(self.session_id, field_name="session_id"))
        object.__setattr__(self, "revision", _clean_non_negative_int(self.revision, field_name="revision"))
        object.__setattr__(
            self,
            "catalog_generation",
            _clean_non_negative_int(self.catalog_generation, field_name="catalog_generation"),
        )
        object.__setattr__(
            self,
            "catalog_fingerprint",
            _clean_text(
                self.catalog_fingerprint,
                field_name="catalog_fingerprint",
                max_chars=128,
            ),
        )
        normalized_entries: list[DiscoveredToolEntry] = []
        seen_keys: set[tuple[str, str]] = set()
        for entry in self.entries:
            if not isinstance(entry, DiscoveredToolEntry):
                raise TypeError("entries must contain DiscoveredToolEntry values")
            key = (entry.tool_name, entry.source_query_hash)
            if key in seen_keys:
                raise ValueError("duplicate discovery entry for tool_name/source_query_hash")
            seen_keys.add(key)
            normalized_entries.append(entry)
        object.__setattr__(self, "entries", tuple(normalized_entries))

    @classmethod
    def empty(
        cls,
        session_id: str,
        *,
        catalog_generation: int = 0,
        catalog_fingerprint: str = "catalog:empty",
    ) -> ToolDiscoveryState:
        return cls(
            state_version=TOOL_DISCOVERY_STATE_VERSION,
            session_id=session_id,
            revision=0,
            catalog_generation=catalog_generation,
            catalog_fingerprint=catalog_fingerprint,
            entries=(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_version": self.state_version,
            "session_id": self.session_id,
            "revision": self.revision,
            "catalog_generation": self.catalog_generation,
            "catalog_fingerprint": self.catalog_fingerprint,
            "entries": [entry.to_dict() for entry in self.entries],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ToolDiscoveryState:
        _require_exact_keys(value, expected=_STATE_FIELDS, field_name="tool discovery state")
        raw_entries = value["entries"]
        if not isinstance(raw_entries, list):
            raise TypeError("tool discovery state entries must be a list")
        return cls(
            state_version=value["state_version"],
            session_id=value["session_id"],
            revision=value["revision"],
            catalog_generation=value["catalog_generation"],
            catalog_fingerprint=value["catalog_fingerprint"],
            entries=tuple(DiscoveredToolEntry.from_dict(item) for item in raw_entries),
        )

    def prune(
        self,
        *,
        current_turn_index: int,
        max_entries: int,
        ttl_turns: int,
        catalog_generation: int,
        catalog_fingerprint: str,
    ) -> ToolDiscoveryState:
        current = _clean_non_negative_int(current_turn_index, field_name="current_turn_index")
        limit = _clean_non_negative_int(max_entries, field_name="max_entries")
        ttl = _clean_positive_int(ttl_turns, field_name="ttl_turns")
        current_generation = _clean_non_negative_int(
            catalog_generation,
            field_name="catalog_generation",
        )
        current_fingerprint = _clean_text(
            catalog_fingerprint,
            field_name="catalog_fingerprint",
            max_chars=128,
        )

        retained = [
            entry
            for entry in self.entries
            if entry.catalog_generation == current_generation
            and entry.catalog_fingerprint == current_fingerprint
            and current - entry.last_used_turn_index <= ttl
        ]
        retained.sort(
            key=lambda entry: (
                -entry.last_used_turn_index,
                -entry.discovered_turn_index,
                entry.tool_name,
                entry.source_query_hash,
            )
        )
        return replace(
            self,
            catalog_generation=current_generation,
            catalog_fingerprint=current_fingerprint,
            entries=tuple(retained[:limit]),
        )

    def record_observations(
        self,
        observations: Iterable[ToolDiscoveryObservation],
        *,
        current_turn_index: int,
        max_entries: int,
        ttl_turns: int,
        catalog_generation: int,
        catalog_fingerprint: str,
    ) -> ToolDiscoveryState:
        next_state = self.prune(
            current_turn_index=current_turn_index,
            max_entries=max_entries,
            ttl_turns=ttl_turns,
            catalog_generation=catalog_generation,
            catalog_fingerprint=catalog_fingerprint,
        )
        by_key = {
            (entry.tool_name, entry.source_query_hash): entry
            for entry in next_state.entries
        }
        for observation in observations:
            if not isinstance(observation, ToolDiscoveryObservation):
                raise TypeError("observations must contain ToolDiscoveryObservation values")
            key = (observation.tool_name, observation.source_query_hash)
            existing = by_key.get(key)
            if existing is None:
                by_key[key] = DiscoveredToolEntry(
                    tool_name=observation.tool_name,
                    source_query_hash=observation.source_query_hash,
                    discovered_turn_id=observation.turn_id,
                    discovered_turn_index=observation.turn_index,
                    last_used_turn_id=observation.turn_id,
                    last_used_turn_index=observation.turn_index,
                    catalog_fingerprint=observation.catalog_fingerprint,
                    catalog_generation=observation.catalog_generation,
                    capability_risk_class=observation.capability_risk_class,
                )
                continue
            by_key[key] = replace(
                existing,
                last_used_turn_id=observation.turn_id,
                last_used_turn_index=observation.turn_index,
                catalog_fingerprint=observation.catalog_fingerprint,
                catalog_generation=observation.catalog_generation,
                capability_risk_class=observation.capability_risk_class,
            )

        merged = replace(next_state, entries=tuple(by_key.values()))
        return merged.prune(
            current_turn_index=current_turn_index,
            max_entries=max_entries,
            ttl_turns=ttl_turns,
            catalog_generation=catalog_generation,
            catalog_fingerprint=catalog_fingerprint,
        )


@dataclass(frozen=True)
class ToolDiscoveryStateLoadResult:
    status: ToolDiscoveryStateLoadStatus
    state: ToolDiscoveryState | None = None
    event_index: int | None = None
    message: str | None = None


@dataclass(frozen=True)
class ToolDiscoveryStateSaveResult:
    status: ToolDiscoveryStateSaveStatus
    expected_revision: int
    current_revision: int
    state: ToolDiscoveryState | None = None
    message: str | None = None
    idempotent_replay: bool = False


class ToolDiscoveryStateRepository:
    """CAS/idempotent SessionStore repository for discovery metadata."""

    def __init__(self, session_store: SessionStoreCompatible | SessionStore) -> None:
        self._session_store = session_store

    async def load(self, session_id: str) -> ToolDiscoveryStateLoadResult:
        normalized_session_id = _clean_text(session_id, field_name="session_id")
        events = await self._session_store.load_session(normalized_session_id)
        return self._load_from_events(normalized_session_id, events)

    async def save(
        self,
        state: ToolDiscoveryState,
        *,
        expected_revision: int,
        turn_id: str,
        idempotency_key: str,
        timestamp: str | None = None,
    ) -> ToolDiscoveryStateSaveResult:
        current = await self.load(state.session_id)
        if current.status in {"invalid", "unsupported_version"}:
            return ToolDiscoveryStateSaveResult(
                status="invalid",
                expected_revision=expected_revision,
                current_revision=current.state.revision if current.state is not None else 0,
                message=current.message or "cannot transition from invalid durable discovery state",
            )

        next_revision = _clean_non_negative_int(expected_revision, field_name="expected_revision") + 1
        next_state = replace(state, revision=next_revision)
        event = {
            "type": "session_meta",
            "event": TOOL_DISCOVERY_EVENT,
            "schema_version": TOOL_DISCOVERY_EVENT_SCHEMA_VERSION,
            "session_id": state.session_id,
            "state_revision": next_revision,
            "turn_id": _clean_text(turn_id, field_name="turn_id"),
            "catalog_generation": next_state.catalog_generation,
            "idempotency_key": _clean_text(
                idempotency_key,
                field_name="idempotency_key",
                max_chars=256,
            ),
            "tool_discovery_state": next_state.to_dict(),
            "timestamp": timestamp or datetime.now(tz=UTC).isoformat(),
        }
        outcome: ToolDiscoveryStateSaveResult | None = None

        def predicate(events: list[dict[str, Any]]) -> bool:
            nonlocal outcome
            loaded = self._load_from_events(state.session_id, events)
            if loaded.status in {"invalid", "unsupported_version"}:
                outcome = ToolDiscoveryStateSaveResult(
                    status="invalid",
                    expected_revision=expected_revision,
                    current_revision=loaded.state.revision if loaded.state is not None else 0,
                    message=loaded.message or "cannot transition from invalid durable discovery state",
                )
                return False
            duplicate = self._find_idempotent_event(
                events,
                session_id=state.session_id,
                idempotency_key=event["idempotency_key"],
            )
            if duplicate is not None:
                duplicate_state_payload = duplicate.get("tool_discovery_state")
                if (
                    isinstance(duplicate_state_payload, Mapping)
                    and self._duplicate_matches_requested_state(
                        duplicate=duplicate,
                        requested_state=state,
                        requested_turn_id=turn_id,
                        requested_catalog_generation=next_state.catalog_generation,
                    )
                ):
                    duplicate_state = ToolDiscoveryState.from_dict(duplicate_state_payload)
                    outcome = ToolDiscoveryStateSaveResult(
                        status="saved",
                        expected_revision=expected_revision,
                        current_revision=duplicate_state.revision,
                        state=duplicate_state,
                        message="idempotent replay",
                        idempotent_replay=True,
                    )
                    return False
                outcome = ToolDiscoveryStateSaveResult(
                    status="invalid",
                    expected_revision=expected_revision,
                    current_revision=loaded.state.revision if loaded.state is not None else 0,
                    message="idempotency key was reused for a different discovery mutation",
                )
                return False
            durable_revision = loaded.state.revision if loaded.state is not None else 0
            if durable_revision != expected_revision:
                outcome = ToolDiscoveryStateSaveResult(
                    status="conflict",
                    expected_revision=expected_revision,
                    current_revision=durable_revision,
                    state=loaded.state,
                    message="tool discovery state revision changed before transition",
                )
                return False
            return True

        appended = await self._session_store.append_event_if(state.session_id, event, predicate)
        if appended:
            return ToolDiscoveryStateSaveResult(
                status="saved",
                expected_revision=expected_revision,
                current_revision=expected_revision,
                state=next_state,
            )
        if outcome is not None:
            return outcome
        reloaded = await self.load(state.session_id)
        if reloaded.status == "loaded" and reloaded.state is not None:
            return ToolDiscoveryStateSaveResult(
                status="conflict",
                expected_revision=expected_revision,
                current_revision=reloaded.state.revision,
                state=reloaded.state,
                message="tool discovery state append lost the SessionStore CAS",
            )
        return ToolDiscoveryStateSaveResult(
            status="invalid",
            expected_revision=expected_revision,
            current_revision=current.state.revision if current.state is not None else 0,
            message="tool discovery state append did not produce a durable result",
        )

    async def record_observations(
        self,
        *,
        session_id: str,
        turn_id: str,
        current_turn_index: int,
        catalog_generation: int,
        catalog_fingerprint: str,
        observations: Iterable[ToolDiscoveryObservation],
        idempotency_key: str,
        max_entries: int = 20,
        ttl_turns: int = 20,
        timestamp: str | None = None,
    ) -> ToolDiscoveryStateSaveResult:
        normalized_session_id = _clean_text(session_id, field_name="session_id")
        loaded = await self.load(normalized_session_id)
        if loaded.status in {"invalid", "unsupported_version"}:
            return ToolDiscoveryStateSaveResult(
                status="invalid",
                expected_revision=0,
                current_revision=loaded.state.revision if loaded.state is not None else 0,
                message=loaded.message or "cannot record observations into invalid durable state",
            )
        base_state = loaded.state or ToolDiscoveryState.empty(
            normalized_session_id,
            catalog_generation=catalog_generation,
            catalog_fingerprint=catalog_fingerprint,
        )
        next_state = base_state.record_observations(
            observations,
            current_turn_index=current_turn_index,
            max_entries=max_entries,
            ttl_turns=ttl_turns,
            catalog_generation=catalog_generation,
            catalog_fingerprint=catalog_fingerprint,
        )
        return await self.save(
            next_state,
            expected_revision=base_state.revision,
            turn_id=turn_id,
            idempotency_key=idempotency_key,
            timestamp=timestamp,
        )

    @staticmethod
    def _find_idempotent_event(
        events: list[dict[str, Any]],
        *,
        session_id: str,
        idempotency_key: str,
    ) -> dict[str, Any] | None:
        for event in reversed(events):
            if (
                event.get("event") == TOOL_DISCOVERY_EVENT
                and event.get("session_id") == session_id
                and event.get("idempotency_key") == idempotency_key
            ):
                return dict(event)
        return None

    @staticmethod
    def _events_match_for_idempotency(
        persisted: Mapping[str, Any],
        candidate: Mapping[str, Any],
    ) -> bool:
        comparable_fields = _EVENT_FIELDS - {"timestamp"}
        if frozenset(persisted) != frozenset(candidate):
            return False
        return all(persisted.get(field_name) == candidate.get(field_name) for field_name in comparable_fields)

    @staticmethod
    def _duplicate_matches_requested_state(
        *,
        duplicate: Mapping[str, Any],
        requested_state: ToolDiscoveryState,
        requested_turn_id: str,
        requested_catalog_generation: int,
    ) -> bool:
        try:
            duplicate_state = ToolDiscoveryState.from_dict(duplicate["tool_discovery_state"])
        except Exception:
            return False
        if duplicate.get("turn_id") != requested_turn_id:
            return False
        if duplicate.get("catalog_generation") != requested_catalog_generation:
            return False
        return duplicate_state.to_dict() == requested_state.to_dict()

    @staticmethod
    def _load_from_events(
        session_id: str,
        events: list[dict[str, Any]],
    ) -> ToolDiscoveryStateLoadResult:
        for reverse_index, event in enumerate(reversed(events)):
            if event.get("event") != TOOL_DISCOVERY_EVENT:
                continue
            return ToolDiscoveryStateRepository._load_event(
                session_id=session_id,
                event=event,
                event_index=len(events) - 1 - reverse_index,
            )
        return ToolDiscoveryStateLoadResult(status="missing")

    @staticmethod
    def _load_event(
        *,
        session_id: str,
        event: dict[str, Any],
        event_index: int,
    ) -> ToolDiscoveryStateLoadResult:
        version = event.get("schema_version")
        if type(version) is not int:
            return ToolDiscoveryStateLoadResult(
                status="invalid",
                event_index=event_index,
                message="tool discovery event schema_version must be an integer",
            )
        if version != TOOL_DISCOVERY_EVENT_SCHEMA_VERSION:
            status: ToolDiscoveryStateLoadStatus = (
                "unsupported_version" if version > TOOL_DISCOVERY_EVENT_SCHEMA_VERSION else "invalid"
            )
            return ToolDiscoveryStateLoadResult(
                status=status,
                event_index=event_index,
                message=f"unsupported tool discovery event schema version: {version}",
            )
        try:
            _require_exact_keys(event, expected=_EVENT_FIELDS, field_name="tool discovery event")
            if event.get("type") != "session_meta":
                raise ValueError("tool discovery event type must be session_meta")
            if event.get("event") != TOOL_DISCOVERY_EVENT:
                raise ValueError("tool discovery event name is invalid")
            if event.get("session_id") != session_id:
                raise ValueError("tool discovery event session_id does not match request")
            _clean_text(event.get("turn_id"), field_name="turn_id")
            envelope_revision = _clean_non_negative_int(
                event.get("state_revision"),
                field_name="state_revision",
            )
            envelope_generation = _clean_non_negative_int(
                event.get("catalog_generation"),
                field_name="catalog_generation",
            )
            _clean_text(event.get("timestamp"), field_name="timestamp", max_chars=128)
            _clean_text(event.get("idempotency_key"), field_name="idempotency_key", max_chars=256)
            payload = event.get("tool_discovery_state")
            if not isinstance(payload, Mapping):
                raise TypeError("tool discovery event tool_discovery_state must be an object")
            state = ToolDiscoveryState.from_dict(payload)
            if state.session_id != session_id:
                raise ValueError("tool discovery payload session_id does not match its envelope")
            if state.revision != envelope_revision:
                raise ValueError("tool discovery payload revision does not match its envelope")
            if state.catalog_generation != envelope_generation:
                raise ValueError("tool discovery payload catalog_generation does not match its envelope")
        except Exception as exc:
            return ToolDiscoveryStateLoadResult(
                status="invalid",
                event_index=event_index,
                message=f"invalid tool discovery event: {exc}",
            )
        return ToolDiscoveryStateLoadResult(
            status="loaded",
            state=state,
            event_index=event_index,
        )


def _clean_text(
    value: Any,
    *,
    field_name: str,
    max_chars: int = 256,
) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    cleaned = " ".join(value.split())
    if not cleaned:
        raise ValueError(f"{field_name} must not be empty")
    if len(cleaned) > max_chars:
        raise ValueError(f"{field_name} exceeds {max_chars} characters")
    return cleaned


def _clean_non_negative_int(value: Any, *, field_name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{field_name} must be a non-negative integer")
    return value


def _clean_positive_int(value: Any, *, field_name: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{field_name} must be a positive integer")
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
