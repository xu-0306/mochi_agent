"""Exec approval request primitives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

ApprovalStatus = Literal["pending", "approved", "rejected"]


def utc_now_iso() -> str:
    """回傳 UTC ISO8601 時間字串。"""
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class ExecApprovalRequest:
    """Exec approval request model."""

    approval_id: str
    status: ApprovalStatus
    reason: str | None
    command: str
    shell: str
    scope: str
    created_at: str
    command_payload: dict[str, Any] | None = None
    execution_result: dict[str, Any] | None = None
    resolved_at: str | None = None


class InMemoryApprovalStore:
    """In-memory approval request store primitive."""

    def __init__(self) -> None:
        self._items: dict[str, ExecApprovalRequest] = {}

    def create(
        self,
        *,
        approval_id: str,
        command: str,
        shell: str,
        scope: str,
        reason: str | None = None,
        command_payload: dict[str, Any] | None = None,
    ) -> ExecApprovalRequest:
        request = ExecApprovalRequest(
            approval_id=approval_id,
            status="pending",
            reason=reason,
            command=command,
            shell=shell,
            scope=scope,
            created_at=utc_now_iso(),
            command_payload=dict(command_payload) if isinstance(command_payload, dict) else None,
            execution_result=None,
            resolved_at=None,
        )
        self._items[approval_id] = request
        return request

    def get(self, approval_id: str) -> ExecApprovalRequest | None:
        return self._items.get(approval_id)

    def list(self, *, status: ApprovalStatus | None = None) -> list[ExecApprovalRequest]:
        items = list(self._items.values())
        if status is None:
            return sorted(items, key=lambda item: item.created_at, reverse=True)
        filtered = [item for item in items if item.status == status]
        return sorted(filtered, key=lambda item: item.created_at, reverse=True)

    def resolve(
        self,
        approval_id: str,
        *,
        approved: bool,
        reason: str | None = None,
        execution_result: dict[str, Any] | None = None,
    ) -> ExecApprovalRequest | None:
        current = self._items.get(approval_id)
        if current is None:
            return None
        next_status: ApprovalStatus = "approved" if approved else "rejected"
        resolved = ExecApprovalRequest(
            approval_id=current.approval_id,
            status=next_status,
            reason=reason if reason is not None else current.reason,
            command=current.command,
            shell=current.shell,
            scope=current.scope,
            created_at=current.created_at,
            command_payload=current.command_payload,
            execution_result=(
                dict(execution_result) if isinstance(execution_result, dict) else current.execution_result
            ),
            resolved_at=utc_now_iso(),
        )
        self._items[approval_id] = resolved
        return resolved
