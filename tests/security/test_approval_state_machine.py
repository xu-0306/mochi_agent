from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Protocol

import pytest

from mochi.runtime.approval_state_machine import (
    ApprovalConflict,
    ApprovalExpired,
    ApprovalLifecycleState,
    ApprovalRequesterMismatch,
    claim_approval,
    complete_approval,
    recover_approval,
    resolve_approval,
    supersede_approval,
)
from mochi.runtime.approvals import (
    ApprovalStore,
    ExecApprovalRequest,
    InMemoryApprovalStore,
    PersistentApprovalStore,
)
from mochi.runtime.store import RuntimeStore


def _future(seconds: int = 300) -> str:
    return (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat()


def _past() -> str:
    return (datetime.now(UTC) - timedelta(seconds=1)).isoformat()


@dataclass(frozen=True)
class _ApprovalView:
    status: str
    resolution_kind: str | None = None
    consume_lease_token: str | None = None


class _ContractAdapter(Protocol):
    async def create(self, approval_id: str, *, expires_at: str) -> _ApprovalView: ...
    async def get(self, approval_id: str) -> _ApprovalView: ...
    async def resolve(
        self, approval_id: str, *, requester_id: str = "requester-1",
        decision: str = "approve_once",
    ) -> _ApprovalView: ...
    async def consume(
        self, approval_id: str, *, context_digest: str = "b" * 64
    ) -> _ApprovalView: ...
    async def complete(
        self, approval_id: str, *, lease_token: str
    ) -> _ApprovalView: ...
    async def supersede(self, approval_id: str) -> _ApprovalView: ...


def _exec_view(item: ExecApprovalRequest) -> _ApprovalView:
    return _ApprovalView(
        status=item.status,
        resolution_kind=item.resolution_kind,
        consume_lease_token=item.consume_lease_token,
    )


class _ExecStoreAdapter:
    def __init__(self, store: ApprovalStore) -> None:
        self._store = store

    async def create(self, approval_id: str, *, expires_at: str) -> _ApprovalView:
        return _exec_view(
            self._store.create(
                approval_id=approval_id,
                command="echo ok",
                shell="powershell",
                scope="dangerous_command",
                requester_id="requester-1",
                request_digest="a" * 64,
                context_digest="b" * 64,
                expires_at=expires_at,
            )
        )

    async def get(self, approval_id: str) -> _ApprovalView:
        item = self._store.get(approval_id)
        assert item is not None
        return _exec_view(item)

    async def resolve(
        self,
        approval_id: str,
        *,
        requester_id: str = "requester-1",
        decision: str = "approve_once",
    ) -> _ApprovalView:
        item = self._store.resolve(
            approval_id,
            decision=decision,  # type: ignore[arg-type]
            requester_id=requester_id,
            request_digest="a" * 64,
            context_digest="b" * 64,
        )
        assert item is not None
        return _exec_view(item)

    async def consume(
        self,
        approval_id: str,
        *,
        context_digest: str = "b" * 64,
    ) -> _ApprovalView:
        return _exec_view(
            self._store.consume(
                approval_id,
                execution_idempotency_key=f"execution:{approval_id}",
                lease_owner="worker-1",
                requester_id="requester-1",
                request_digest="a" * 64,
                context_digest=context_digest,
            )
        )

    async def complete(
        self,
        approval_id: str,
        *,
        lease_token: str,
    ) -> _ApprovalView:
        return _exec_view(
            self._store.complete_consumption(
                approval_id,
                execution_idempotency_key=f"execution:{approval_id}",
                lease_owner="worker-1",
                lease_token=lease_token,
                execution_result={"status": "completed"},
            )
        )

    async def supersede(self, approval_id: str) -> _ApprovalView:
        return _exec_view(self._store.supersede(approval_id, reason="replacement"))


class _RuntimeStoreAdapter:
    def __init__(self, store: RuntimeStore, workspace: Path) -> None:
        self._store = store
        self._workspace = workspace

    async def initialize(self) -> None:
        await self._store.initialize()
        await self._store.create_task_run(
            task_id="contract-task",
            input_text="run",
            session_id=None,
            project_id=None,
            workspace_dir=str(self._workspace),
            project_workspace_dir=None,
            task_workspace_dir=None,
            inference_overrides={},
        )

    @staticmethod
    def _view(item: dict[str, object]) -> _ApprovalView:
        return _ApprovalView(
            status=str(item["status"]),
            resolution_kind=(
                str(item["resolution_kind"])
                if item.get("resolution_kind") is not None
                else None
            ),
            consume_lease_token=(
                str(item["consume_lease_token"])
                if item.get("consume_lease_token") is not None
                else None
            ),
        )

    async def create(self, approval_id: str, *, expires_at: str) -> _ApprovalView:
        item = await self._store.create_approval_request(
            approval_id=approval_id,
            task_id="contract-task",
            call_id=f"call:{approval_id}",
            tool_name="exec_command",
            arguments={"command": "echo ok"},
            requester_id="requester-1",
            request_digest="a" * 64,
            context_digest="b" * 64,
            expires_at=expires_at,
        )
        return self._view(item)

    async def get(self, approval_id: str) -> _ApprovalView:
        item = await self._store.get_approval_request(approval_id)
        assert item is not None
        return self._view(item)

    async def resolve(
        self,
        approval_id: str,
        *,
        requester_id: str = "requester-1",
        decision: str = "approve_once",
    ) -> _ApprovalView:
        item = await self._store.resolve_approval_request(
            approval_id,
            decision=decision,
            requester_id=requester_id,
            request_digest="a" * 64,
            context_digest="b" * 64,
        )
        assert item is not None
        return self._view(item)

    async def consume(
        self,
        approval_id: str,
        *,
        context_digest: str = "b" * 64,
    ) -> _ApprovalView:
        item = await self._store.consume_approval_request(
            approval_id,
            execution_idempotency_key=f"execution:{approval_id}",
            lease_owner="worker-1",
            requester_id="requester-1",
            request_digest="a" * 64,
            context_digest=context_digest,
        )
        return self._view(item)

    async def complete(
        self,
        approval_id: str,
        *,
        lease_token: str,
    ) -> _ApprovalView:
        item = await self._store.complete_approval_consumption(
            approval_id,
            execution_idempotency_key=f"execution:{approval_id}",
            lease_owner="worker-1",
            lease_token=lease_token,
            execution_result={"status": "completed"},
        )
        return self._view(item)

    async def supersede(self, approval_id: str) -> _ApprovalView:
        item = await self._store.supersede_approval_request(
            approval_id,
            reason="replacement",
        )
        return self._view(item)


async def _adapter(kind: str, tmp_path: Path) -> _ContractAdapter:
    if kind == "in-memory":
        return _ExecStoreAdapter(InMemoryApprovalStore())
    if kind == "persistent":
        return _ExecStoreAdapter(PersistentApprovalStore(tmp_path / "approvals.db"))
    runtime = _RuntimeStoreAdapter(RuntimeStore(tmp_path / "runtime.db"), tmp_path)
    await runtime.initialize()
    return runtime


@pytest.mark.parametrize("kind", ["in-memory", "persistent", "runtime"])
@pytest.mark.asyncio
async def test_all_adapters_share_resolve_consume_contract(
    kind: str,
    tmp_path: Path,
) -> None:
    adapter = await _adapter(kind, tmp_path)
    await adapter.create("approval-1", expires_at=_future())

    with pytest.raises(ApprovalRequesterMismatch):
        await adapter.resolve("approval-1", requester_id="attacker")
    assert (await adapter.get("approval-1")).status == "pending"

    resolved = await adapter.resolve("approval-1")
    assert resolved.status == "approved_once"
    assert resolved.resolution_kind == "approve_once"

    with pytest.raises(ApprovalConflict):
        await adapter.resolve("approval-1")
    with pytest.raises(ApprovalConflict):
        await adapter.consume("approval-1", context_digest="f" * 64)

    claim = await adapter.consume("approval-1")
    assert claim.status == "consuming"
    assert claim.consume_lease_token

    with pytest.raises(ApprovalConflict):
        await adapter.consume("approval-1")
    with pytest.raises(ApprovalConflict):
        await adapter.complete("approval-1", lease_token="wrong-token")

    completed = await adapter.complete(
        "approval-1",
        lease_token=claim.consume_lease_token or "",
    )
    assert completed.status == "consumed"

    with pytest.raises(ApprovalConflict):
        await adapter.complete(
            "approval-1",
            lease_token=claim.consume_lease_token or "",
        )


@pytest.mark.parametrize("kind", ["in-memory", "persistent", "runtime"])
@pytest.mark.asyncio
async def test_all_adapters_share_expiry_and_supersede_contract(
    kind: str,
    tmp_path: Path,
) -> None:
    adapter = await _adapter(kind, tmp_path)
    await adapter.create("expired", expires_at=_past())
    with pytest.raises(ApprovalExpired):
        await adapter.resolve("expired")
    assert (await adapter.get("expired")).status == "expired"

    await adapter.create("superseded", expires_at=_future())
    assert (await adapter.supersede("superseded")).status == "superseded"
    with pytest.raises(ApprovalConflict):
        await adapter.supersede("superseded")


@pytest.mark.parametrize("kind", ["in-memory", "persistent", "runtime"])
@pytest.mark.asyncio
async def test_all_adapters_reject_invalid_decisions_without_transition(
    kind: str,
    tmp_path: Path,
) -> None:
    adapter = await _adapter(kind, tmp_path)
    await adapter.create("invalid-decision", expires_at=_future())

    with pytest.raises(ValueError):
        await adapter.resolve("invalid-decision", decision="bogus")

    pending = await adapter.get("invalid-decision")
    assert pending.status == "pending"
    assert pending.resolution_kind is None

def _state(**changes: object) -> ApprovalLifecycleState:
    values: dict[str, object] = {
        "approval_id": "approval-1",
        "status": "pending",
        "requester_id": "requester-1",
        "request_digest": "a" * 64,
        "context_digest": "b" * 64,
        "expires_at": _future(),
    }
    values.update(changes)
    return ApprovalLifecycleState(**values)  # type: ignore[arg-type]


def test_shared_state_machine_owns_all_transition_decisions() -> None:
    resolved = resolve_approval(
        _state(),
        decision="approve_once",
        requester_id="requester-1",
        request_digest="a" * 64,
        context_digest="b" * 64,
    )
    assert resolved.status == "approved_once"

    claimed = claim_approval(
        _state(status="approved_once"),
        execution_idempotency_key="execution-1",
        lease_owner="worker-1",
    )
    assert claimed.status == "consuming"
    assert claimed.consume_lease_token

    consuming = _state(
        status="consuming",
        execution_idempotency_key="execution-1",
        consume_lease_owner="worker-1",
        consume_lease_token="token-1",
        consume_lease_expires_at=_past(),
    )
    assert (
        complete_approval(
            consuming,
            execution_idempotency_key="execution-1",
            lease_owner="worker-1",
            lease_token="token-1",
            succeeded=False,
        ).status
        == "execution_failed"
    )
    assert recover_approval(consuming, outcome="applied").status == "consumed"
    assert recover_approval(consuming, outcome="not_started").status == "approved_once"
    assert recover_approval(consuming, outcome="unknown").status == "execution_failed"
    assert supersede_approval(_state()).status == "superseded"
