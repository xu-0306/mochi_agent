"""Runtime-control tool for durable task-plan state."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast

from mochi.agents.plan_ledger import (
    DEFAULT_PLAN_LEDGER_LIMITS,
    PLAN_LEDGER_VERSION,
    PlanItem,
    PlanItemStatus,
    PlanLedger,
    PlanLedgerRepository,
    PlanLedgerTransitionValidator,
)
from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult

UpdatePlanAction = Literal["view", "create_or_replace", "set_status"]
_UPDATE_PLAN_ACTIONS = frozenset({"view", "create_or_replace", "set_status"})
_UPDATE_PLAN_ITEM_STATUSES = frozenset(
    {"pending", "in_progress", "completed", "blocked", "cancelled"}
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


def _require_exact_keys(payload: dict[str, Any], *, expected: frozenset[str]) -> None:
    actual = frozenset(payload)
    unexpected = sorted(actual - expected)
    missing = sorted(expected - actual)
    details: list[str] = []
    if unexpected:
        details.append(f"unexpected keys: {unexpected}")
    if missing:
        details.append(f"missing keys: {missing}")
    if details:
        raise ValueError("update_plan arguments " + "; ".join(details))


def _clean_text_tuple(
    value: Any,
    *,
    field_name: str,
    max_items: int = 8,
    max_chars: int = 128,
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
    if len(cleaned) > max_items:
        raise ValueError(f"{field_name} exceeds {max_items} items")
    return tuple(cleaned)


@dataclass(frozen=True)
class UpdatePlanRequest:
    action: UpdatePlanAction
    expected_revision: int
    items: tuple[PlanItem, ...] = ()
    item_id: str | None = None
    status: PlanItemStatus | None = None
    evidence_refs: tuple[str, ...] = ()
    blocker_reason: str | None = None

    @classmethod
    def from_tool_arguments(cls, arguments: dict[str, Any]) -> UpdatePlanRequest:
        expected = frozenset(
            {
                "action",
                "expected_revision",
                "items",
                "item_id",
                "status",
                "evidence_refs",
                "blocker_reason",
            }
        )
        _require_exact_keys(arguments, expected=expected)
        action = cast(UpdatePlanAction, arguments.get("action"))
        if action not in _UPDATE_PLAN_ACTIONS:
            raise ValueError(f"unsupported update_plan action: {action!r}")
        expected_revision = _clean_non_negative_int(
            arguments.get("expected_revision"),
            field_name="expected_revision",
        )
        if action == "view":
            request = cls(action=action, expected_revision=expected_revision)
            request._validate()
            return request

        raw_items = arguments.get("items")
        if action == "create_or_replace":
            if not isinstance(raw_items, list):
                raise TypeError("items must be a list")
            items = tuple(PlanItem.from_dict(item) for item in raw_items)
            request = cls(
                action=action,
                expected_revision=expected_revision,
                items=items,
            )
            request._validate()
            return request

        item_id = arguments.get("item_id")
        status = arguments.get("status")
        blocker_reason = arguments.get("blocker_reason")
        evidence_refs = _clean_text_tuple(
            arguments.get("evidence_refs"),
            field_name="evidence_refs",
            max_items=DEFAULT_PLAN_LEDGER_LIMITS.max_evidence_refs_per_item,
        )
        if item_id is not None:
            item_id = _clean_text(item_id, field_name="item_id", max_chars=128)
        if status is not None and status not in _UPDATE_PLAN_ITEM_STATUSES:
            raise ValueError(f"unsupported plan item status: {status!r}")
        if blocker_reason is not None:
            blocker_reason = _clean_text(
                blocker_reason,
                field_name="blocker_reason",
                max_chars=DEFAULT_PLAN_LEDGER_LIMITS.max_blocker_reason_chars,
            )
        request = cls(
            action=action,
            expected_revision=expected_revision,
            item_id=item_id,
            status=cast(PlanItemStatus | None, status),
            evidence_refs=evidence_refs,
            blocker_reason=blocker_reason,
        )
        request._validate()
        return request

    def _validate(self) -> None:
        if self.action == "view":
            return
        if self.action == "create_or_replace":
            if not self.items:
                raise ValueError("create_or_replace requires items")
            return
        if self.item_id is None or self.status is None:
            raise ValueError("set_status requires item_id and status")


@dataclass(frozen=True)
class UpdatePlanRuntimeContext:
    session_id: str
    goal_id: str
    ledger_id: str
    turn_id: str
    objective: str
    reason_codes: tuple[str, ...] = ()
    recognized_evidence_refs: frozenset[str] = field(default_factory=frozenset)

    def __post_init__(self) -> None:
        for field_name in ("session_id", "goal_id", "ledger_id", "turn_id"):
            object.__setattr__(
                self,
                field_name,
                _clean_text(
                    getattr(self, field_name),
                    field_name=field_name,
                    max_chars=128,
                ),
            )
        object.__setattr__(
            self,
            "objective",
            _clean_text(self.objective, field_name="objective", max_chars=1_000),
        )
        object.__setattr__(
            self,
            "reason_codes",
            tuple(
                dict.fromkeys(
                    _clean_text(reason_code, field_name="reason_codes", max_chars=128)
                    for reason_code in self.reason_codes
                )
            ),
        )
        object.__setattr__(
            self,
            "recognized_evidence_refs",
            frozenset(
                _clean_text(value, field_name="recognized_evidence_ref", max_chars=128)
                for value in self.recognized_evidence_refs
            ),
        )


class UpdatePlanController(Protocol):
    async def apply(self, request: UpdatePlanRequest) -> ToolResult:
        """Apply a scoped update_plan request."""


class ScopedPlanController:
    """Apply scoped plan mutations using only trusted runtime identifiers."""

    def __init__(
        self,
        *,
        repository: PlanLedgerRepository,
        runtime_context: UpdatePlanRuntimeContext,
    ) -> None:
        self._repository = repository
        self._runtime_context = runtime_context

    async def apply(self, request: UpdatePlanRequest) -> ToolResult:
        if request.action == "view":
            return await self._view()
        if request.action == "create_or_replace":
            return await self._create_or_replace(request)
        return await self._set_status(request)

    async def _view(self) -> ToolResult:
        loaded = await self._repository.load(
            self._runtime_context.session_id,
            self._runtime_context.goal_id,
            ledger_id=self._runtime_context.ledger_id,
        )
        output = {
            "status": loaded.status,
            "ledger": loaded.ledger.to_dict() if loaded.ledger is not None else None,
            "current_revision": loaded.ledger.revision if loaded.ledger is not None else 0,
        }
        metadata = self._metadata()
        metadata["load_status"] = loaded.status
        if loaded.status == "loaded":
            metadata["ledger_revision"] = cast(PlanLedger, loaded.ledger).revision
        return ToolResult(output=output, metadata=metadata)

    async def _create_or_replace(self, request: UpdatePlanRequest) -> ToolResult:
        loaded = await self._repository.load(
            self._runtime_context.session_id,
            self._runtime_context.goal_id,
            ledger_id=self._runtime_context.ledger_id,
        )
        if loaded.status in {"invalid", "unsupported_version"}:
            return self._error(
                loaded.message or "durable plan ledger is invalid",
                error_type="plan_ledger_invalid",
                retryable=False,
            )
        previous = loaded.ledger
        created_turn_id = previous.created_turn_id if previous is not None else self._runtime_context.turn_id
        proposed = PlanLedger(
            ledger_version=PLAN_LEDGER_VERSION,
            ledger_id=self._runtime_context.ledger_id,
            session_id=self._runtime_context.session_id,
            goal_id=self._runtime_context.goal_id,
            revision=previous.revision if previous is not None else 0,
            status="active",
            objective=self._runtime_context.objective,
            reason_codes=previous.reason_codes if previous is not None else self._runtime_context.reason_codes,
            items=request.items,
            created_turn_id=created_turn_id,
            updated_turn_id=self._runtime_context.turn_id,
        )
        try:
            validator = PlanLedgerTransitionValidator(
                recognized_evidence_refs=self._runtime_context.recognized_evidence_refs
            )
            validator.validate_replacement(previous=previous, proposed=proposed)
        except Exception as exc:
            return self._error(
                str(exc),
                error_type="plan_transition_invalid",
                retryable=False,
            )
        saved = await self._repository.save(
            proposed,
            expected_revision=request.expected_revision,
            turn_id=self._runtime_context.turn_id,
            idempotency_key=self._idempotency_key_for(request),
        )
        return self._save_result_to_tool_result(saved)

    async def _set_status(self, request: UpdatePlanRequest) -> ToolResult:
        loaded = await self._repository.load(
            self._runtime_context.session_id,
            self._runtime_context.goal_id,
            ledger_id=self._runtime_context.ledger_id,
        )
        if loaded.status != "loaded" or loaded.ledger is None:
            return self._error(
                loaded.message or "no active plan ledger is available",
                error_type="plan_ledger_missing",
                retryable=False,
            )
        try:
            validator = PlanLedgerTransitionValidator(
                recognized_evidence_refs=self._runtime_context.recognized_evidence_refs
            )
            proposed = validator.set_item_status(
                loaded.ledger,
                item_id=cast(str, request.item_id),
                status=cast(PlanItemStatus, request.status),
                updated_turn_id=self._runtime_context.turn_id,
                evidence_refs=request.evidence_refs,
                blocker_reason=request.blocker_reason,
            )
        except Exception as exc:
            return self._error(
                str(exc),
                error_type="plan_transition_invalid",
                retryable=False,
            )
        saved = await self._repository.save(
            proposed,
            expected_revision=request.expected_revision,
            turn_id=self._runtime_context.turn_id,
            idempotency_key=self._idempotency_key_for(request),
        )
        return self._save_result_to_tool_result(saved)

    def _save_result_to_tool_result(self, saved: Any) -> ToolResult:
        if saved.status == "saved":
            metadata = self._metadata()
            metadata.update(
                {
                    "save_status": "saved",
                    "idempotent_replay": bool(saved.idempotent_replay),
                    "ledger_revision": saved.ledger.revision if saved.ledger is not None else 0,
                }
            )
            return ToolResult(
                output={
                    "status": "saved",
                    "ledger": saved.ledger.to_dict() if saved.ledger is not None else None,
                    "current_revision": saved.current_revision,
                    "saved_revision": saved.ledger.revision if saved.ledger is not None else None,
                    "idempotent_replay": bool(saved.idempotent_replay),
                },
                metadata=metadata,
            )
        if saved.status == "conflict":
            return self._error(
                saved.message or "plan ledger revision changed before transition",
                error_type="stale_plan_revision",
                retryable=True,
                current_revision=saved.current_revision,
                ledger=saved.ledger,
            )
        return self._error(
            saved.message or "plan mutation is invalid",
            error_type="plan_mutation_invalid",
            retryable=False,
            current_revision=saved.current_revision,
            ledger=saved.ledger,
        )

    def _idempotency_key_for(self, request: UpdatePlanRequest) -> str:
        payload = {
            "session_id": self._runtime_context.session_id,
            "goal_id": self._runtime_context.goal_id,
            "ledger_id": self._runtime_context.ledger_id,
            "turn_id": self._runtime_context.turn_id,
            "action": request.action,
            "expected_revision": request.expected_revision,
            "items": [item.to_dict() for item in request.items],
            "item_id": request.item_id,
            "status": request.status,
            "evidence_refs": list(request.evidence_refs),
            "blocker_reason": request.blocker_reason,
        }
        digest = hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode(
                "utf-8"
            )
        ).hexdigest()
        return f"plan-update:{digest}"

    def _metadata(self) -> dict[str, Any]:
        return {
            "runtime_category": "task_planning",
            "session_id": self._runtime_context.session_id,
            "goal_id": self._runtime_context.goal_id,
            "ledger_id": self._runtime_context.ledger_id,
        }

    def _error(
        self,
        message: str,
        *,
        error_type: str,
        retryable: bool,
        current_revision: int | None = None,
        ledger: PlanLedger | None = None,
    ) -> ToolResult:
        metadata = self._metadata()
        metadata["error_type"] = error_type
        if current_revision is not None:
            metadata["current_revision"] = current_revision
        if ledger is not None:
            metadata["ledger_revision"] = ledger.revision
        return ToolResult(
            error=message,
            metadata=metadata,
            retryable=retryable,
        )


class UpdatePlanTool(BaseTool):
    """Expose one strictly scoped plan-state control surface."""

    @property
    def name(self) -> str:
        return "update_plan"

    @property
    def description(self) -> str:
        return (
            "View or update the current durable task plan. "
            "This tool is host-scoped to the current session goal and never grants "
            "workspace or tool execution authority."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["view", "create_or_replace", "set_status"],
                },
                "expected_revision": {"type": "integer", "minimum": 0},
                "items": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "item_id": {"type": "string"},
                            "title": {"type": "string"},
                            "status": {"type": "string"},
                            "dependencies": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "success_criteria": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "source_turn_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "evidence_refs": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "blocker_reason": {"type": ["string", "null"]},
                            "attempts": {"type": "integer", "minimum": 0},
                        },
                        "required": [
                            "item_id",
                            "title",
                            "status",
                            "dependencies",
                            "success_criteria",
                            "source_turn_ids",
                            "evidence_refs",
                            "blocker_reason",
                            "attempts",
                        ],
                        "additionalProperties": False,
                    },
                },
                "item_id": {"type": ["string", "null"]},
                "status": {
                    "type": ["string", "null"],
                    "enum": [None, "pending", "in_progress", "completed", "blocked", "cancelled"],
                },
                "evidence_refs": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "blocker_reason": {"type": ["string", "null"]},
            },
            "required": [
                "action",
                "expected_revision",
                "items",
                "item_id",
                "status",
                "evidence_refs",
                "blocker_reason",
            ],
            "additionalProperties": False,
        }

    @property
    def is_read_only(self) -> bool:
        return False

    async def execute(self, **kwargs: Any) -> ToolResult:
        raw_context = kwargs.get("context")
        if raw_context is not None and not isinstance(raw_context, ToolExecutionContext):
            return ToolResult(
                error="`context` must be a ToolExecutionContext when provided.",
                metadata={
                    "runtime_category": "task_planning",
                    "error_type": "plan_tool_invalid_context",
                },
            )
        context = cast(ToolExecutionContext | None, raw_context)
        controller = resolve_update_plan_controller(context)
        if controller is None:
            return ToolResult(
                error="update_plan requires a scoped plan controller in runtime context.",
                metadata={
                    "runtime_category": "task_planning",
                    "error_type": "plan_controller_missing",
                },
                retryable=True,
            )
        arguments = {
            "action": kwargs.get("action"),
            "expected_revision": kwargs.get("expected_revision"),
            "items": kwargs.get("items", []),
            "item_id": kwargs.get("item_id"),
            "status": kwargs.get("status"),
            "evidence_refs": kwargs.get("evidence_refs", []),
            "blocker_reason": kwargs.get("blocker_reason"),
        }
        try:
            request = UpdatePlanRequest.from_tool_arguments(arguments)
        except Exception as exc:
            return ToolResult(
                error=str(exc),
                metadata={
                    "runtime_category": "task_planning",
                    "error_type": "plan_tool_invalid_request",
                },
                retryable=False,
            )
        return await controller.apply(request)


def resolve_update_plan_controller(
    context: ToolExecutionContext | None,
) -> UpdatePlanController | None:
    if context is None:
        return None
    controller = context.state.get("update_plan_controller")
    if controller is None:
        return None
    return cast(UpdatePlanController, controller)
