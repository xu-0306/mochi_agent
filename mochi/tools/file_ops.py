"""File tools with workspace boundaries, stale-write guards, and patch support."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import inspect
import logging
import stat
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

from mochi.config import defaults
from mochi.config.schema import SecurityConfig
from mochi.runtime.approval_state_machine import derive_approval_binding
from mochi.runtime.approvals import APPROVAL_OWNER_TASK_ID_KEY, ApprovalStore
from mochi.runtime.change_sets import ChangeSetStore
from mochi.runtime.store import RuntimeStore
from mochi.security import require_approval_decision, with_task_isolation_scope
from mochi.security.auto_review import (
    AutoReviewDecision,
    AutoReviewFacts,
    AutoReviewVerificationError,
    auto_review_metadata,
    review_authorization_envelope,
    verify_auto_review_decision,
)
from mochi.security.file_contract import (
    AUTHORIZATION_ENVELOPE_SCHEMA_VERSION,
    AuthorizationContext,
    AuthorizationEnvelope,
    ChangeEntry,
    ChangeManifest,
    FileChangeRequest,
    authorization_request_digest,
    canonical_json,
    capture_file_identity,
    detect_content_fidelity,
    tool_arguments_digest,
)
from mochi.security.policy import policy_projection_version
from mochi.sessions.timeline_coordinator import (
    mark_context_side_effect_started,
    timeline_pending_operation_binding,
)
from mochi.tools.base import BaseTool, FileReadState, ToolExecutionContext, ToolResult
from mochi.tools.file_mutations import (
    PatchValidationError,
    build_file_change_entry,
    build_file_change_payload,
    prepare_apply_patch,
)
from mochi.utils.security import (
    check_file_tool_path,
    content_size_bytes,
    is_within_write_size_limit,
    normalize_workspace_dir,
    resolve_path_with_scope,
    size_limit_bytes,
)

FileReader = Callable[[Path, str], Awaitable[str]]
FileWriter = Callable[[Path, str, bool, str], Awaitable[int]]
_TOOL_RESULT_PATH_PREFIX = "tool-result://"
_logger = logging.getLogger(__name__)


async def _observe_ordinary_chat_approval(
    context: ToolExecutionContext | None,
    approval: Any,
) -> None:
    """Publish only the approval identity/revision handoff when the Engine wired it."""

    if context is None or not isinstance(context.state, Mapping):
        return
    observer = context.state.get("tool_workflow_approval_observer")
    if not callable(observer):
        return
    try:
        result = observer(approval)
        if inspect.isawaitable(result):
            await result
    except Exception as exc:
        # The approval transaction is already durable.  This cross-store
        # publication is repairable at startup and must never turn a pending
        # approval into a failed mutation result.
        _logger.warning(
            "ordinary-chat approval observation handoff requires repair: %s (%s)",
            getattr(approval, "approval_id", ""),
            type(exc).__name__,
        )


def file_mutation_tool_inventory_version(tool: BaseTool) -> str:
    """Return the replay contract version for one concrete mutation tool."""
    return policy_projection_version(
        "file-mutation-tool",
        {
            "tool_name": tool.name,
            "parameters_schema": tool.parameters_schema,
        },
    )


def file_mutation_arguments_digest(
    *,
    tool_name: str,
    arguments: Mapping[str, Any],
) -> str:
    """Digest normalized mutation arguments without lossy string projection."""
    return tool_arguments_digest(tool_name=tool_name, arguments=arguments)


def _file_base_state(
    *,
    target: Path,
    workspace_root: Path,
    before: bytes | None,
) -> dict[str, Any]:
    return {
        "relative_path": target.relative_to(workspace_root).as_posix(),
        "exists": before is not None,
        "sha256": _content_digest(before),
    }


@dataclass(frozen=True)
class _FileMutationCallPolicy:
    require_approval: bool
    path_scope: str
    autonomy_mode: str
    source: str
    policy_snapshot_id: str | None
    policy_version: str | None

    def metadata(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "path_scope": self.path_scope,
            "require_approval_for_file_write": self.require_approval,
            "file_permission_policy_source": self.source,
        }
        if self.policy_snapshot_id is not None:
            payload["policy_snapshot_id"] = self.policy_snapshot_id
        if self.policy_version is not None:
            payload["effective_policy_version"] = self.policy_version
        return payload


def _resolve_file_mutation_call_policy(
    *,
    context: ToolExecutionContext | None,
    legacy_require_approval: bool,
    legacy_path_scope: str,
) -> _FileMutationCallPolicy:
    """Resolve mutable file policy per call, with constructor values as fallback.

    A task sandbox is an immutable containment ceiling.  Even if a malformed or
    stale execution policy requests ``any``, a sandboxed call remains confined
    to its task workspace.  Protected-path checks are enforced separately for
    every call by ``check_file_tool_path`` / ``prepare_apply_patch``.
    """
    permission_policy: Mapping[str, Any] = {}
    if context is not None and isinstance(context.permission_policy, Mapping):
        permission_policy = context.permission_policy

    approval_value = permission_policy.get("require_approval_for_file_write")
    require_approval = (
        approval_value if isinstance(approval_value, bool) else legacy_require_approval
    )
    scope_value = permission_policy.get("file_write_scope")
    path_scope = (
        scope_value if scope_value in {"workspace", "any"} else legacy_path_scope
    )
    has_context_policy = isinstance(approval_value, bool) or scope_value in {
        "workspace",
        "any",
    }
    source = "execution_context" if has_context_policy else "constructor_fallback"
    if context is not None and context.task_sandbox_dir:
        path_scope = "workspace"
        source = f"{source}+task_sandbox_ceiling"

    autonomy_mode = str(permission_policy.get("autonomy_mode") or "").strip().lower()
    snapshot_value = permission_policy.get("policy_snapshot_id")
    version_value = permission_policy.get("policy_version")
    return _FileMutationCallPolicy(
        require_approval=bool(require_approval),
        path_scope=path_scope,
        autonomy_mode=autonomy_mode,
        source=source,
        policy_snapshot_id=(
            snapshot_value.strip()
            if isinstance(snapshot_value, str) and snapshot_value.strip()
            else None
        ),
        policy_version=(
            version_value.strip()
            if isinstance(version_value, str) and version_value.strip()
            else None
        ),
    )


def _create_file_mutation_approval(
    *,
    tool: BaseTool,
    approval_store: ApprovalStore,
    arguments: Mapping[str, Any],
    workspace_root: Path,
    base_states: list[dict[str, Any]],
    preview_metadata: Mapping[str, Any],
    reason: str,
    call_policy: _FileMutationCallPolicy,
    context: ToolExecutionContext | None,
) -> Any:
    timeline_binding = timeline_pending_operation_binding(
        context,
        tool_name=tool.name,
    )
    arguments_payload = (
        dict(timeline_binding["arguments"])
        if timeline_binding is not None
        else dict(arguments)
    )
    arguments_digest = (
        timeline_binding["arguments_digest"]
        if timeline_binding is not None
        else file_mutation_arguments_digest(
            tool_name=tool.name,
            arguments=arguments_payload,
        )
    )
    cache_key = ":".join(
        (
            tool.name,
            arguments_digest,
            call_policy.policy_snapshot_id or "",
            timeline_binding["operation_id"] if timeline_binding is not None else "",
        )
    )
    if context is not None:
        cached = context.state.setdefault("durable_file_approval_ids", {})
        if isinstance(cached, dict):
            approval_id = cached.get(cache_key)
            if isinstance(approval_id, str):
                existing = approval_store.get(approval_id)
                if existing is not None:
                    return existing

    operation_id = (
        timeline_binding["operation_id"]
        if timeline_binding is not None
        else f"file-operation-{uuid4().hex}"
    )
    approval_id = f"tool-approval-{uuid4().hex[:20]}"
    inventory_version = file_mutation_tool_inventory_version(tool)
    permission_policy = (
        dict(context.permission_policy)
        if context is not None and isinstance(context.permission_policy, Mapping)
        else {}
    )
    ordinary_chat_context = _ordinary_chat_approval_context(context)
    resume_cursor = (
        dict(ordinary_chat_context.get("resume_cursor") or {})
        if isinstance(ordinary_chat_context, Mapping)
        and isinstance(ordinary_chat_context.get("resume_cursor"), Mapping)
        else {}
    )
    if timeline_binding is not None:
        if ordinary_chat_context is None:
            raise ValueError(
                "timeline-bound file approval requires an ordinary-Chat continuation context"
            )
        cursor_call_id = str(resume_cursor.get("tool_call_id") or "").strip()
        cursor_tool_name = str(resume_cursor.get("tool_name") or "").strip()
        if cursor_call_id != timeline_binding["call_id"] or (
            cursor_tool_name and cursor_tool_name != tool.name
        ):
            raise ValueError(
                "ordinary-Chat approval cursor does not match the timeline-bound call"
            )
    owner_task_id = str(permission_policy.get(APPROVAL_OWNER_TASK_ID_KEY) or "").strip() or None
    requester_id, request_digest, context_digest = derive_approval_binding(
        requester_id=(
            f"runtime-task:{owner_task_id}"
            if owner_task_id
            else f"runtime-session:{context.session_id}"
            if context is not None and context.session_id
            else "runtime-service"
        ),
        request={
            "tool_name": tool.name,
            "arguments": arguments_payload,
            "arguments_digest": arguments_digest,
            "operation_id": operation_id,
            **(
                {"timeline_call_id": timeline_binding["call_id"]}
                if timeline_binding is not None
                else {}
            ),
        },
        authorization_context={
            "workspace_root": str(workspace_root),
            "session_id": context.session_id if context is not None else None,
            "policy_snapshot_id": call_policy.policy_snapshot_id,
            "policy_version": call_policy.policy_version,
            "tool_inventory_version": inventory_version,
        },
    )
    replay_payload = {
        "schema_version": 1,
        "tool_name": tool.name,
        "arguments": arguments_payload,
        "arguments_digest": arguments_digest,
        "operation_id": operation_id,
        **(
            {"timeline_call_id": timeline_binding["call_id"]}
            if timeline_binding is not None
            else {}
        ),
        "workspace_dir": str(workspace_root),
        "session_id": context.session_id if context is not None else None,
        "permission_policy": permission_policy,
        "policy_snapshot_id": call_policy.policy_snapshot_id,
        "effective_policy_version": call_policy.policy_version,
        "tool_inventory_version": inventory_version,
        "base_states": [dict(item) for item in base_states],
    }
    if ordinary_chat_context is not None:
        # A Chat approval is a durable interrupt, not an invitation for the
        # model to recreate a similar call on a later turn.  Keep the exact
        # normalized call plus the identity needed to safely resume it.
        replay_payload["ordinary_chat_checkpoint"] = {
            "schema_version": 1,
            "source": "ordinary_chat",
            "session_id": ordinary_chat_context.get("session_id"),
            "turn_id": ordinary_chat_context.get("turn_id"),
            "resume_cursor": resume_cursor,
            "resolved_workspace_dir": str(workspace_root),
            "resolved_targets": [dict(item) for item in base_states],
            "operation_id": operation_id,
            **(
                {"timeline_call_id": timeline_binding["call_id"]}
                if timeline_binding is not None
                else {}
            ),
            "tool_name": tool.name,
            "normalized_arguments": arguments_payload,
            "arguments_digest": arguments_digest,
            "policy_snapshot_id": call_policy.policy_snapshot_id,
            "policy_version": call_policy.policy_version,
            "inventory_version": inventory_version,
            "react_continuation": (
                dict(ordinary_chat_context["react_continuation"])
                if isinstance(ordinary_chat_context.get("react_continuation"), Mapping)
                else None
            ),
        }
    stored = approval_store.create(
        approval_id=approval_id,
        command=tool.name,
        shell="tool",
        scope="workspace",
        reason=reason,
        metadata={
            **dict(preview_metadata),
            "tool_name": tool.name,
            "arguments_digest": arguments_digest,
            "operation_id": operation_id,
            **(
                {"timeline_call_id": timeline_binding["call_id"]}
                if timeline_binding is not None
                else {}
            ),
            "policy_snapshot_id": call_policy.policy_snapshot_id,
            "effective_policy_version": call_policy.policy_version,
            "tool_inventory_version": inventory_version,
            **(
                {
                    "approval_source": "ordinary_chat",
                    "resume_cursor": resume_cursor,
                    "resolved_workspace_dir": str(workspace_root),
                    "resolved_targets": [dict(item) for item in base_states],
                }
                if ordinary_chat_context is not None
                else {}
            ),
            **(
                {APPROVAL_OWNER_TASK_ID_KEY: owner_task_id}
                if owner_task_id
                else {}
            ),
        },
        command_payload=replay_payload,
        requester_id=requester_id,
        request_digest=request_digest,
        context_digest=context_digest,
    )
    if context is not None:
        cached = context.state.setdefault("durable_file_approval_ids", {})
        if isinstance(cached, dict):
            cached[cache_key] = stored.approval_id
    return stored


def _ordinary_chat_approval_context(
    context: ToolExecutionContext | None,
) -> dict[str, Any] | None:
    """Return the engine-owned ordinary-Chat checkpoint context, if present."""
    if context is None or not isinstance(context.state, Mapping):
        return None
    raw = context.state.get("ordinary_chat_approval_context")
    if not isinstance(raw, Mapping) or raw.get("source") != "ordinary_chat":
        return None
    return dict(raw)


def _build_path_denial_metadata(
    *,
    path: str | Path,
    workspace_root: Path,
    path_scope: str,
    security_metadata: dict[str, Any],
) -> dict[str, Any]:
    """Expose the effective path boundary in structured write-denial diagnostics."""
    metadata = dict(security_metadata)
    metadata.update(
        {
            "workspace_dir": str(workspace_root),
            "requested_path": str(path),
            "path_scope": path_scope,
        }
    )
    with contextlib.suppress(OSError, RuntimeError, ValueError):
        metadata["resolved_path"] = str(resolve_path_with_scope(path, workspace_root, "any"))
    return metadata


class FileChangeContractConflict(RuntimeError):
    """Raised when an immutable patch preview no longer matches execution state."""


def file_change_policy_version(security: SecurityConfig) -> str:
    """Return a stable digest for file-authorization policy inputs."""

    projection = {
        "change_contract_mode": security.change_contract_mode,
        "autonomy_mode": security.autonomy_mode,
        "require_approval_for_file_write": security.require_approval_for_file_write,
        "file_write_scope": security.file_write_scope,
        "max_file_write_size_mb": str(security.max_file_write_size_mb),
        "file_undo_max_size_mb": str(security.file_undo_max_size_mb),
    }
    return policy_projection_version("file-policy", projection)


def _content_digest(content: bytes | None) -> str | None:
    return None if content is None else hashlib.sha256(content).hexdigest()


def _content_fidelity_projection(
    before: bytes | None,
    after: bytes | None,
) -> tuple[str | None, str | None, bool | None]:
    content = after if after is not None else before
    if content is None:
        return None, None, None
    fidelity = detect_content_fidelity(content)
    return fidelity.encoding, fidelity.newline_style, fidelity.eof_newline


def _context_digest(context: AuthorizationContext) -> str:
    payload = canonical_json(context.to_dict()).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


async def _prepare_file_auto_review(
    *,
    workspace_root: Path,
    tool_name: str,
    call_policy: _FileMutationCallPolicy,
    changes: list[tuple[Path, str, bytes | None, bytes | None]],
    patch_sha256: str | None,
    context: ToolExecutionContext | None,
) -> tuple[AutoReviewDecision, AuthorizationEnvelope] | None:
    permission_policy = context.permission_policy if context is not None else {}
    autonomy_mode = call_policy.autonomy_mode
    if autonomy_mode not in {"auto_review", "high_autonomy"}:
        return None

    owner_task_id = str(permission_policy.get("approval_owner_task_id") or "").strip() or None
    authorization_context = AuthorizationContext(
        requester_id=(
            f"runtime-task:{owner_task_id}" if owner_task_id else "runtime-service"
        ),
        session_id=(
            str(context.session_id).strip()
            if context is not None and context.session_id
            else owner_task_id or "runtime-session"
        ),
        task_id=owner_task_id,
        workspace_root=str(workspace_root),
        workspace_identity=await asyncio.to_thread(capture_file_identity, workspace_root),
    )
    entries: list[ChangeEntry] = []
    for ordinal, (target, operation, before, after) in enumerate(changes):
        entry_encoding, newline_style, eof_newline = _content_fidelity_projection(
            before,
            after,
        )
        relative_path = target.relative_to(workspace_root).as_posix()
        base_identity = (
            None
            if before is None
            else await asyncio.to_thread(capture_file_identity, target)
        )
        mode_before = None
        if before is not None:
            info = await asyncio.to_thread(target.stat)
            mode_before = stat.S_IMODE(info.st_mode)
        entry_seed = canonical_json(
            {
                "ordinal": ordinal,
                "operation": operation,
                "relative_path": relative_path,
                "base_sha256": _content_digest(before),
                "after_sha256": _content_digest(after),
            }
        )
        entries.append(
            ChangeEntry(
                entry_id=hashlib.sha256(entry_seed.encode("utf-8")).hexdigest(),
                relative_path=relative_path,
                operation=operation,  # type: ignore[arg-type]
                base_sha256=_content_digest(before),
                after_sha256=_content_digest(after),
                base_identity=base_identity,
                before_blob_id=None,
                after_blob_id=None,
                mode_before=mode_before,
                mode_after=mode_before if after is not None else None,
                base_metadata_sha256=None,
                after_metadata_sha256=None,
                rename_source=None,
                dependency_group=None,
                encoding=entry_encoding,
                newline_style=newline_style,  # type: ignore[arg-type]
                eof_newline=eof_newline,
            )
        )
    policy_version = policy_projection_version(
        "file-auto-review-policy",
        {
            "autonomy_mode": autonomy_mode,
            "file_write_scope": call_policy.path_scope,
            "require_approval_for_file_write": call_policy.require_approval,
            "effective_policy_version": call_policy.policy_version,
            "policy_source": call_policy.source,
            "tool_name": tool_name,
        },
    )
    envelope = AuthorizationEnvelope(
        schema_version=AUTHORIZATION_ENVELOPE_SCHEMA_VERSION,
        kind="file_change",
        context=authorization_context,
        policy_version=policy_version,
        file_request=FileChangeRequest(
            entries=tuple(entries),
            patch_sha256=patch_sha256,
        ),
        exec_request=None,
    )
    decision = review_authorization_envelope(
        envelope,
        facts=AutoReviewFacts(
            policy_action="ask",
            policy_rule_id="file_mutation_requires_review",
        ),
    )
    return decision, envelope


async def _verify_file_auto_review(
    decision: AutoReviewDecision,
    envelope: AuthorizationEnvelope,
) -> None:
    workspace_root = Path(envelope.context.workspace_root)
    current_identity = await asyncio.to_thread(capture_file_identity, workspace_root)
    verify_auto_review_decision(
        decision,
        envelope,
        current_workspace_identity=current_identity,
    )
    request = envelope.file_request
    if request is None:
        raise AutoReviewVerificationError("file review envelope lost its file request")
    for entry in request.entries:
        target = workspace_root / Path(entry.relative_path)
        exists = await asyncio.to_thread(target.exists)
        if entry.operation == "add":
            if exists:
                raise AutoReviewVerificationError("reviewed add target now exists")
            continue
        if not exists or not await asyncio.to_thread(target.is_file):
            raise AutoReviewVerificationError("reviewed file base is missing")
        current = await asyncio.to_thread(target.read_bytes)
        if _content_digest(current) != entry.base_sha256:
            raise AutoReviewVerificationError("reviewed file base content is stale")
        identity = await asyncio.to_thread(capture_file_identity, target)
        if identity != entry.base_identity:
            raise AutoReviewVerificationError("reviewed file base identity changed")


async def _enforce_file_auto_review(
    *,
    workspace_root: Path,
    tool_name: str,
    call_policy: _FileMutationCallPolicy,
    changes: list[tuple[Path, str, bytes | None, bytes | None]],
    patch_sha256: str | None,
    context: ToolExecutionContext | None,
    metadata: dict[str, Any],
    approved: bool,
) -> ToolResult | None:
    if approved:
        return None
    try:
        reviewed = await _prepare_file_auto_review(
            workspace_root=workspace_root,
            tool_name=tool_name,
            call_policy=call_policy,
            changes=changes,
            patch_sha256=patch_sha256,
            context=context,
        )
        if reviewed is None:
            return None
        decision, envelope = reviewed
        metadata.update(auto_review_metadata(decision))
        if decision.decision != "allow":
            metadata["status"] = "denied"
            metadata["requires_approval"] = decision.decision == "require_approval"
            return ToolResult(
                error=f"Auto review {decision.decision.replace('_', ' ')} for file mutation.",
                metadata=metadata,
                retryable=decision.decision == "require_approval",
            )
        await _verify_file_auto_review(decision, envelope)
        metadata["auto_review_execution_verified"] = True
        return None
    except (AutoReviewVerificationError, OSError, RuntimeError, ValueError) as exc:
        metadata.update(
            {
                "status": "denied",
                "auto_review_execution_verified": False,
                "auto_review_verification_error": str(exc),
            }
        )
        return ToolResult(
            error=f"Auto review verification failed before file mutation: {exc}",
            metadata=metadata,
            retryable=False,
        )


async def prepare_patch_change_contract(
    *,
    runtime_store: RuntimeStore,
    patch: str,
    workspace_dir: str | Path,
    security: SecurityConfig,
    requester_id: str,
    session_id: str,
    task_id: str | None,
    encoding: str = "utf-8",
) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    """Prepare and persist a canonical immutable patch manifest."""

    workspace_root = normalize_workspace_dir(workspace_dir)
    prepared, change_payload = await prepare_apply_patch(
        patch=patch,
        workspace_dir=workspace_root,
        path_scope=security.file_write_scope,
        encoding=encoding,
        undo_max_size_mb=security.file_undo_max_size_mb,
    )
    change_store = ChangeSetStore(runtime_store)
    workspace_identity = await asyncio.to_thread(capture_file_identity, workspace_root)
    policy_version = file_change_policy_version(security)
    patch_sha256 = hashlib.sha256(patch.encode("utf-8")).hexdigest()
    context = AuthorizationContext(
        requester_id=requester_id,
        session_id=session_id,
        task_id=task_id,
        workspace_root=str(workspace_root),
        workspace_identity=workspace_identity,
    )

    entries: list[ChangeEntry] = []
    for ordinal, item in enumerate(prepared):
        before = (
            None
            if not item.existed_before
            else await asyncio.to_thread(item.target.read_bytes)
        )
        after = None if item.new_content is None else item.new_content.encode(encoding)
        entry_encoding, newline_style, eof_newline = _content_fidelity_projection(
            before,
            after,
        )
        before_blob_id = None if before is None else await change_store.put_blob(before)
        after_blob_id = None if after is None else await change_store.put_blob(after)
        base_identity = (
            None
            if not item.existed_before
            else await asyncio.to_thread(capture_file_identity, item.target)
        )
        mode_before = None
        if item.existed_before:
            info = await asyncio.to_thread(item.target.stat)
            mode_before = stat.S_IMODE(info.st_mode)
        relative_path = item.target.relative_to(workspace_root).as_posix()
        entry_seed = canonical_json(
            {
                "context": context.to_dict(),
                "policy_version": policy_version,
                "patch_sha256": patch_sha256,
                "ordinal": ordinal,
                "relative_path": relative_path,
                "operation": item.operation.kind,
                "base_sha256": _content_digest(before),
                "after_sha256": _content_digest(after),
            }
        )
        entries.append(
            ChangeEntry(
                entry_id=hashlib.sha256(entry_seed.encode("utf-8")).hexdigest(),
                relative_path=relative_path,
                operation=item.operation.kind,
                base_sha256=_content_digest(before),
                after_sha256=_content_digest(after),
                base_identity=base_identity,
                before_blob_id=before_blob_id,
                after_blob_id=after_blob_id,
                mode_before=mode_before,
                mode_after=mode_before if after is not None else None,
                base_metadata_sha256=None,
                after_metadata_sha256=None,
                rename_source=None,
                dependency_group=None,
                encoding=entry_encoding,
                newline_style=newline_style,  # type: ignore[arg-type]
                eof_newline=eof_newline,
            )
        )

    file_request = FileChangeRequest(entries=tuple(entries), patch_sha256=patch_sha256)
    envelope = AuthorizationEnvelope(
        schema_version=AUTHORIZATION_ENVELOPE_SCHEMA_VERSION,
        kind="file_change",
        context=context,
        policy_version=policy_version,
        file_request=file_request,
        exec_request=None,
    )
    request_digest = authorization_request_digest(envelope)
    now = datetime.now(UTC)
    manifest = ChangeManifest(
        version=AUTHORIZATION_ENVELOPE_SCHEMA_VERSION,
        change_set_id=str(uuid4()),
        workspace_root=str(workspace_root),
        workspace_identity=workspace_identity,
        tool_name="apply_patch",
        intent="mutate",
        entries=file_request.entries,
        patch_sha256=patch_sha256,
        policy_version=policy_version,
        created_at=now.isoformat(),
        expires_at=(now + timedelta(minutes=15)).isoformat(),
        request_digest=request_digest,
        ui_metadata={
            "encoding": encoding,
            "change_count": change_payload.get("change_count", 0),
        },
    )
    persisted = await change_store.persist_manifest(manifest, envelope)
    stored_manifest = persisted["manifest"]
    contract = {
        "change_set_id": stored_manifest.change_set_id,
        "request_digest": stored_manifest.request_digest,
        "expires_at": stored_manifest.expires_at,
        "policy_version": stored_manifest.policy_version,
        "change_contract_mode": security.change_contract_mode,
        "context_digest": _context_digest(envelope.context),
    }
    return prepared, change_payload, contract


async def revalidate_patch_change_contract(
    *,
    runtime_store: RuntimeStore,
    approval: Mapping[str, Any],
    task: Mapping[str, Any],
    security: SecurityConfig,
) -> ChangeManifest:
    """Reload and compare a bound manifest immediately before mutation."""

    metadata = approval.get("metadata")
    metadata = metadata if isinstance(metadata, Mapping) else {}
    change_set_id = metadata.get("change_set_id")
    if not isinstance(change_set_id, str) or not change_set_id:
        raise FileChangeContractConflict("change_set_missing")

    change_store = ChangeSetStore(runtime_store)
    persisted = await change_store.get_change_set(change_set_id)
    if persisted is None:
        raise FileChangeContractConflict("change_set_missing")
    manifest = persisted["manifest"]
    envelope = persisted["envelope"]
    try:
        if persisted["status"] != "prepared":
            raise FileChangeContractConflict("change_set_not_prepared")
        if datetime.fromisoformat(manifest.expires_at) <= datetime.now(UTC):
            raise FileChangeContractConflict("change_set_expired")
        if approval.get("request_digest") != manifest.request_digest:
            raise FileChangeContractConflict("approval_digest_mismatch")
        if metadata.get("request_digest") != manifest.request_digest:
            raise FileChangeContractConflict("approval_metadata_digest_mismatch")
        if manifest.policy_version != file_change_policy_version(security):
            raise FileChangeContractConflict("policy_changed")
        if str(task.get("id") or "") != str(envelope.context.task_id or ""):
            raise FileChangeContractConflict("task_context_changed")
        if str(task.get("session_id") or "draft-session") != envelope.context.session_id:
            raise FileChangeContractConflict("session_context_changed")

        workspace_value = (
            task.get("task_workspace_dir")
            or task.get("project_workspace_dir")
            or task.get("workspace_dir")
        )
        if not isinstance(workspace_value, str):
            raise FileChangeContractConflict("workspace_context_missing")
        workspace_root = normalize_workspace_dir(workspace_value)
        if str(workspace_root) != manifest.workspace_root:
            raise FileChangeContractConflict("workspace_context_changed")
        current_workspace_identity = await asyncio.to_thread(
            capture_file_identity,
            workspace_root,
        )
        if current_workspace_identity != manifest.workspace_identity:
            raise FileChangeContractConflict("workspace_identity_changed")

        arguments = approval.get("arguments")
        arguments = arguments if isinstance(arguments, Mapping) else {}
        patch = arguments.get("patch")
        if not isinstance(patch, str):
            raise FileChangeContractConflict("server_patch_missing")
        if hashlib.sha256(patch.encode("utf-8")).hexdigest() != manifest.patch_sha256:
            raise FileChangeContractConflict("server_patch_digest_mismatch")

        for entry in manifest.entries:
            target = workspace_root / Path(entry.relative_path)
            exists = await asyncio.to_thread(target.exists)
            if entry.operation == "add":
                if exists:
                    raise FileChangeContractConflict("base_path_now_exists")
                continue
            if not exists or not await asyncio.to_thread(target.is_file):
                raise FileChangeContractConflict("base_path_missing")
            current = await asyncio.to_thread(target.read_bytes)
            if _content_digest(current) != entry.base_sha256:
                raise FileChangeContractConflict("base_content_changed")
            identity = await asyncio.to_thread(capture_file_identity, target)
            if identity != entry.base_identity:
                raise FileChangeContractConflict("base_identity_changed")
    except FileChangeContractConflict:
        await change_store.mark_change_set_status(change_set_id, status="conflicted")
        raise
    return manifest


class FileReadTool(BaseTool):
    """Read a text file from the local filesystem."""

    def __init__(
        self,
        *,
        workspace_dir: str | Path | None = None,
        path_scope: str = "workspace",
        default_encoding: str = "utf-8",
        max_read_bytes: int = 1024 * 1024,
        reader: FileReader | None = None,
    ) -> None:
        self._workspace_dir = normalize_workspace_dir(workspace_dir or defaults.default_workspace_dir())
        self._path_scope = path_scope
        self._default_encoding = default_encoding
        self._max_read_bytes = max_read_bytes
        self._reader = reader or self._default_reader

    @property
    def name(self) -> str:
        return "file_read"

    @property
    def description(self) -> str:
        return (
            "Read the contents of a local text file. Returns full text content. "
            "Use when you need to inspect code, configuration, or notes. "
            "Cannot read binary files."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Local file path to read."},
                "encoding": {"type": "string", "default": "utf-8"},
                "max_bytes": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum bytes allowed for this read. Overrides the default limit.",
                },
                "offset": {
                    "type": "integer",
                    "minimum": 1,
                    "default": 1,
                    "description": "Starting line number for partial reads. Uses 1-based indexing.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Maximum number of lines to return from the starting line.",
                },
                "line_numbers": {
                    "type": "boolean",
                    "default": True,
                    "description": "Whether to prefix returned lines with their line number.",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        }

    @property
    def is_read_only(self) -> bool:
        return True

    @property
    def is_concurrency_safe(self) -> bool:
        return True

    @property
    def allow_plain_text_result_for_model(self) -> bool:
        return True

    def _resolve_workspace_root(self, context: ToolExecutionContext | None) -> Path:
        if context is not None:
            for candidate in (
                context.task_sandbox_dir,
                context.project_workspace,
                context.workspace_dir,
            ):
                if candidate:
                    return normalize_workspace_dir(candidate)
        return self._workspace_dir

    async def execute(
        self,
        *,
        path: str,
        encoding: str | None = None,
        max_bytes: int | None = None,
        offset: int = 1,
        limit: int | None = None,
        line_numbers: bool = True,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        if not path.strip():
            return ToolResult(error="`path` must not be empty.")
        if offset <= 0:
            return ToolResult(error="`offset` must be greater than 0.")
        if limit is not None and limit <= 0:
            return ToolResult(error="`limit` must be greater than 0.")

        target: Path
        metadata_path = path
        reference_metadata: dict[str, Any] = {}
        if path.startswith(_TOOL_RESULT_PATH_PREFIX):
            target_result = self._resolve_tool_result_reference(path=path, context=context)
            if target_result.error is not None:
                return target_result
            target = Path(str(target_result.output))
            reference_metadata = dict(target_result.metadata)
        else:
            workspace_root = self._resolve_workspace_root(context)
            target, security_decision = check_file_tool_path(
                path,
                workspace_dir=workspace_root,
                scope=self._path_scope,
                access="read",
            )
            if security_decision is not None or target is None:
                return ToolResult(
                    error=security_decision.reason if security_decision is not None else "Path denied.",
                    metadata=security_decision.to_metadata() if security_decision is not None else {},
                )
            metadata_path = str(target)

        active_encoding = (
            str(reference_metadata.get("encoding"))
            if encoding is None and isinstance(reference_metadata.get("encoding"), str)
            else (encoding or self._default_encoding)
        )

        if not target.exists():
            return ToolResult(error=f"File not found: {target}")
        if not target.is_file():
            return ToolResult(error=f"Path is not a file: {target}")

        effective_max_bytes = max_bytes if max_bytes is not None else self._max_read_bytes
        if effective_max_bytes <= 0:
            return ToolResult(error="`max_bytes` must be greater than 0.")

        file_size = await asyncio.to_thread(lambda: target.stat().st_size)
        if file_size > effective_max_bytes:
            if limit is None:
                retry_call = (
                    f'file_read(path="{path}", offset=1, limit=200, line_numbers=True)'
                )
                message = (
                    f"File is larger than the current read limit ({file_size} bytes exceeds "
                    f"{effective_max_bytes} bytes). Retry with a bounded line chunk, for example: "
                    f"{retry_call}"
                )
                return ToolResult(
                    output=message,
                    metadata={
                        "path": metadata_path,
                        "size_bytes": file_size,
                        "partial": True,
                        "line_numbers": True,
                        "encoding": active_encoding,
                        **reference_metadata,
                    },
                    suggestion=retry_call,
                )

            chunk_result = await self._read_chunk_from_path(
                target=target,
                encoding=active_encoding,
                offset=offset,
                limit=limit,
                line_numbers=line_numbers,
            )
            chunk_result.metadata.update(
                {
                    "path": metadata_path,
                    "size_bytes": file_size,
                    "encoding": active_encoding,
                    **reference_metadata,
                }
            )
            if chunk_result.error is not None:
                return chunk_result
            return chunk_result

        text = await self._reader(target, active_encoding)
        lines = text.splitlines()
        if lines and offset > len(lines):
            return ToolResult(
                error=(
                    f"File exists but is shorter than the provided offset ({offset}). "
                    f"The file has {len(lines)} lines."
                ),
                metadata={
                    "path": metadata_path,
                    "total_lines": len(lines),
                    "partial": False,
                    "encoding": active_encoding,
                    **reference_metadata,
                },
            )

        rendered_text = text
        partial = False
        start_line = 1
        end_line = len(lines)
        if limit is not None or offset != 1:
            partial = True
            start_idx = offset - 1
            selected_lines = lines[start_idx:] if limit is None else lines[start_idx : start_idx + limit]
            start_line = offset
            end_line = offset + len(selected_lines) - 1 if selected_lines else offset - 1
            if line_numbers:
                rendered_text = "\n".join(
                    f"{line_no}: {line}"
                    for line_no, line in enumerate(selected_lines, start=offset)
                )
            else:
                rendered_text = "\n".join(selected_lines)

        if context is not None:
            stat = await asyncio.to_thread(target.stat)
            context.read_state_cache[str(target)] = FileReadState(
                path=str(target),
                content=text,
                encoding=active_encoding,
                mtime_ns=getattr(stat, "st_mtime_ns", None),
                size_bytes=file_size,
                partial=partial,
            )

        return ToolResult(
            output=rendered_text,
            metadata={
                "path": metadata_path,
                "size_bytes": file_size,
                "partial": partial,
                "start_line": start_line,
                "end_line": end_line,
                "total_lines": len(lines),
                "line_numbers": line_numbers,
                "encoding": active_encoding,
                **reference_metadata,
            },
        )

    @staticmethod
    async def _default_reader(path: Path, encoding: str) -> str:
        return await asyncio.to_thread(_read_text_preserving_newlines, path, encoding)

    def _resolve_tool_result_reference(
        self,
        *,
        path: str,
        context: ToolExecutionContext | None,
    ) -> ToolResult:
        if context is None:
            return ToolResult(error="`tool-result://` reads require an execution context.")

        reference_id = path[len(_TOOL_RESULT_PATH_PREFIX) :].strip()
        if not reference_id:
            return ToolResult(error="`tool-result://` path must include a reference id.")

        return self._resolve_tool_result_reference_id(reference_id=reference_id, context=context)

    def _resolve_tool_result_reference_id(
        self,
        *,
        reference_id: str,
        context: ToolExecutionContext | None,
    ) -> ToolResult:
        if context is None:
            return ToolResult(error="Tool result reads require an execution context.")

        if not reference_id.strip():
            return ToolResult(error="`reference_id` must not be empty.")

        reference = context.tool_result_references.get(reference_id)
        if not isinstance(reference, dict):
            return ToolResult(error=f"Unknown tool result reference: {reference_id}")

        artifact_path = reference.get("artifact_path")
        if not isinstance(artifact_path, str) or not artifact_path.strip():
            return ToolResult(error=f"Tool result reference is missing artifact_path: {reference_id}")

        source_path = reference.get("source_path")
        artifact_encoding = reference.get("artifact_encoding", self._default_encoding)
        continuation_target = artifact_path
        continuation_encoding = artifact_encoding
        if isinstance(source_path, str) and source_path.strip():
            source_candidate = Path(source_path)
            if source_candidate.exists() and source_candidate.is_file():
                continuation_target = source_path
                continuation_encoding = reference.get("encoding", self._default_encoding)

        return ToolResult(
            output=continuation_target,
            metadata={
                "reference_id": reference_id,
                "artifact_path": artifact_path,
                "artifact_encoding": artifact_encoding,
                "source_path": source_path,
                "tool_name": reference.get("tool_name"),
                "encoding": continuation_encoding,
            },
        )

    async def _read_chunk_from_path(
        self,
        *,
        target: Path,
        encoding: str,
        offset: int,
        limit: int,
        line_numbers: bool,
    ) -> ToolResult:
        def _sync_read() -> tuple[str, int, int, int]:
            selected_lines: list[str] = []
            total_lines = 0
            with target.open("r", encoding=encoding) as file:
                for line_no, raw_line in enumerate(file, start=1):
                    total_lines = line_no
                    if line_no < offset:
                        continue
                    if len(selected_lines) >= limit:
                        continue
                    selected_lines.append(raw_line.rstrip("\r\n"))

            if total_lines > 0 and offset > total_lines:
                raise ValueError(
                    f"File exists but is shorter than the provided offset ({offset}). "
                    f"The file has {total_lines} lines."
                )

            start_line = offset
            end_line = offset + len(selected_lines) - 1 if selected_lines else offset - 1
            if line_numbers:
                rendered = "\n".join(
                    f"{line_no}: {line}"
                    for line_no, line in enumerate(selected_lines, start=offset)
                )
            else:
                rendered = "\n".join(selected_lines)
            return rendered, start_line, end_line, total_lines

        try:
            rendered_text, start_line, end_line, total_lines = await asyncio.to_thread(_sync_read)
        except UnicodeDecodeError:
            return ToolResult(error=f"File is not valid {encoding} text: {target}")
        except ValueError as exc:
            return ToolResult(error=str(exc))

        return ToolResult(
            output=rendered_text,
            metadata={
                "partial": True,
                "start_line": start_line,
                "end_line": end_line,
                "total_lines": total_lines,
                "line_numbers": line_numbers,
            },
        )


class FileWriteTool(BaseTool):
    """Write text to a workspace file with undo metadata."""

    def __init__(
        self,
        *,
        workspace_dir: str | Path | None = None,
        path_scope: str = "workspace",
        require_approval: bool = True,
        max_write_size_mb: float = 10.0,
        undo_max_size_mb: float = 2.0,
        default_encoding: str = "utf-8",
        writer: FileWriter | None = None,
        approval_store: ApprovalStore | None = None,
    ) -> None:
        self._workspace_dir = normalize_workspace_dir(workspace_dir or defaults.default_workspace_dir())
        self._path_scope = path_scope
        self._require_approval = require_approval
        self._max_write_size_mb = max_write_size_mb
        self._undo_max_size_mb = undo_max_size_mb
        self._default_encoding = default_encoding
        self._writer = writer or self._default_writer
        self._approval_store = approval_store

    @property
    def name(self) -> str:
        return "file_write"

    @property
    def description(self) -> str:
        return (
            "Write text to a file inside the workspace, either replacing or appending. "
            "Use for controlled text or code updates after deciding on the content. "
            "Cannot write outside the workspace and may require approval."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path. Must be inside the workspace."},
                "content": {"type": "string", "description": "Text content to write."},
                "append": {"type": "boolean", "default": False, "description": "Append instead of replacing the file."},
                "encoding": {"type": "string", "default": "utf-8"},
                "approved": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether user approval has been granted. Required when require_approval is true.",
                },
            },
            "required": ["path", "content"],
            "additionalProperties": False,
        }

    @property
    def requires_approval(self) -> bool:
        return self._require_approval

    @property
    def supports_timeline_side_effect_boundary(self) -> bool:
        return True

    @property
    def supports_timeline_approval_revocation(self) -> bool:
        return True

    @property
    def timeline_approval_mode(self) -> str:
        return "continuable"

    def revoke_timeline_approval(self, approval_id: str, *, reason: str) -> bool:
        if self._approval_store is None:
            return False
        return self._approval_store.supersede(approval_id, reason=reason).status == "superseded"

    def validates_timeline_approval_binding(
        self,
        approval_id: str,
        *,
        operation_id: str,
        arguments_digest: str,
        call_id: str,
    ) -> bool:
        if self._approval_store is None:
            return False
        approval = self._approval_store.get(approval_id)
        if approval is None or approval.status != "pending":
            return False
        payload = approval.command_payload
        checkpoint = (
            payload.get("ordinary_chat_checkpoint")
            if isinstance(payload, Mapping)
            else None
        )
        return bool(
            approval.metadata.get("operation_id") == operation_id
            and approval.metadata.get("arguments_digest") == arguments_digest
            and approval.metadata.get("timeline_call_id") == call_id
            and isinstance(payload, Mapping)
            and payload.get("operation_id") == operation_id
            and payload.get("arguments_digest") == arguments_digest
            and payload.get("timeline_call_id") == call_id
            and isinstance(checkpoint, Mapping)
            and checkpoint.get("source") == "ordinary_chat"
            and checkpoint.get("operation_id") == operation_id
            and checkpoint.get("arguments_digest") == arguments_digest
            and checkpoint.get("timeline_call_id") == call_id
        )

    def _resolve_workspace_root(self, context: ToolExecutionContext | None) -> Path:
        if context is not None:
            for candidate in (
                context.task_sandbox_dir,
                context.project_workspace,
                context.workspace_dir,
            ):
                if candidate:
                    return normalize_workspace_dir(candidate)
        return self._workspace_dir

    async def execute(
        self,
        *,
        path: str,
        content: str,
        append: bool = False,
        encoding: str | None = None,
        approved: bool = False,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        if not path.strip():
            return ToolResult(error="`path` must not be empty.")

        workspace_root = self._resolve_workspace_root(context)
        call_policy = _resolve_file_mutation_call_policy(
            context=context,
            legacy_require_approval=self._require_approval,
            legacy_path_scope=self._path_scope,
        )
        active_encoding = encoding or self._default_encoding
        if not is_within_write_size_limit(
            content=content,
            max_size_mb=self._max_write_size_mb,
            encoding=active_encoding,
        ):
            content_size = content_size_bytes(content, encoding=active_encoding)
            return ToolResult(
                error=(
                    f"Write content too large: {content_size} bytes exceeds limit "
                    f"{size_limit_bytes(self._max_write_size_mb)} bytes."
                ),
                metadata={"size_bytes": content_size},
            )

        target, security_decision = check_file_tool_path(
            path,
            workspace_dir=workspace_root,
            scope=call_policy.path_scope,
        )
        if security_decision is not None or target is None:
            metadata = _build_path_denial_metadata(
                path=path,
                workspace_root=workspace_root,
                path_scope=call_policy.path_scope,
                security_metadata=(
                    security_decision.to_metadata() if security_decision is not None else {}
                ),
            )
            metadata.update(call_policy.metadata())
            if security_decision is not None:
                metadata.update(
                    {
                        "runtime_category": "permission",
                        "error_type": "file_path_denied",
                        "recoverability": "requires_changed_path_or_policy",
                    }
                )
            return ToolResult(
                error=security_decision.reason if security_decision is not None else "Path denied.",
                metadata=metadata,
            )

        existing_result = await _load_existing_content(target, active_encoding)
        if existing_result.error is not None:
            return existing_result
        existing_content = str(existing_result.output)
        existed_before = target.exists()

        guard_error = await _check_stale_write_guard(
            target=target,
            current_content=existing_content,
            context=context,
        )
        if guard_error is not None:
            return guard_error

        merged_content = existing_content + content if append else content
        file_change = build_file_change_entry(
            target=target,
            workspace_root=workspace_root,
            tool_name=self.name,
            change_type="add" if not existed_before and not append else "update",
            original_content=existing_content,
            new_content=merged_content,
            encoding=active_encoding,
            undo_max_size_mb=self._undo_max_size_mb,
            extra={"append": append},
        )
        metadata = build_file_change_payload([file_change])
        metadata.update(
            {
                "workspace_dir": str(workspace_root),
                "resolved_path": str(target),
                **call_policy.metadata(),
            }
        )
        approval_replay = bool(
            context is not None and context.state.get("approval_replay") is True
        )
        trusted_approved = bool(
            approved and (self._approval_store is None or approval_replay)
        )
        if call_policy.require_approval and not trusted_approved:
            decision = require_approval_decision(
                reason="File writes require explicit approval in the current autonomy mode.",
                approval_kind="file_write",
                approval_scope="workspace",
                replay_safe=True,
                policy_source="runtime_policy",
            )
            decision = with_task_isolation_scope(
                decision,
                task_sandbox_dir=context.task_sandbox_dir if context is not None else None,
            )
            metadata.update(decision.to_metadata())
            if self._approval_store is not None:
                before = (
                    await asyncio.to_thread(target.read_bytes)
                    if existed_before
                    else None
                )
                approval = _create_file_mutation_approval(
                    tool=self,
                    approval_store=self._approval_store,
                    arguments={
                        "path": target.relative_to(workspace_root).as_posix(),
                        "content": content,
                        "append": append,
                        "encoding": active_encoding,
                    },
                    workspace_root=workspace_root,
                    base_states=[
                        _file_base_state(
                            target=target,
                            workspace_root=workspace_root,
                            before=before,
                        )
                    ],
                    preview_metadata=metadata,
                    reason=decision.reason,
                    call_policy=call_policy,
                    context=context,
                )
                await _observe_ordinary_chat_approval(context, approval)
                metadata.update(
                    {
                        "status": "approval_pending",
                        "approval_id": approval.approval_id,
                        "operation_id": approval.metadata.get("operation_id"),
                        "arguments_digest": approval.metadata.get("arguments_digest"),
                        "tool_inventory_version": approval.metadata.get(
                            "tool_inventory_version"
                        ),
                    }
                )
            return ToolResult(
                error="File write requires approval.",
                metadata=metadata,
            )

        review_error = await _enforce_file_auto_review(
            workspace_root=workspace_root,
            tool_name=self.name,
            call_policy=call_policy,
            changes=[
                (
                    target,
                    "add" if not existed_before else "update",
                    await asyncio.to_thread(target.read_bytes) if existed_before else None,
                    merged_content.encode(active_encoding),
                )
            ],
            patch_sha256=None,
            context=context,
            metadata=metadata,
            approved=trusted_approved,
        )
        if review_error is not None:
            return review_error

        await mark_context_side_effect_started(context)
        bytes_written = await self._writer(target, content if append else merged_content, append, active_encoding)

        await _refresh_read_state_cache(
            context=context,
            target=target,
            content=merged_content,
            encoding=active_encoding,
        )
        metadata["bytes_written"] = bytes_written
        metadata["append"] = append
        metadata["timeline_result_disposition"] = "succeeded"
        return ToolResult(output=str(target), metadata=metadata)

    @staticmethod
    async def _default_writer(path: Path, content: str, append: bool, encoding: str) -> int:
        def _sync_write() -> int:
            path.parent.mkdir(parents=True, exist_ok=True)
            mode = "a" if append else "w"
            with path.open(mode=mode, encoding=encoding, newline="") as file:
                file.write(content)
            return len(content.encode(encoding))

        return await asyncio.to_thread(_sync_write)


class FileEditTool(FileWriteTool):
    """Edit a file using old_string/new_string replacement semantics."""

    @property
    def name(self) -> str:
        return "file_edit"

    @property
    def description(self) -> str:
        return (
            "Edit a previously read text file by replacing old_string with new_string. "
            "Use this for incremental code or document edits instead of full-file rewrites."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path. Must be inside the workspace."},
                "old_string": {"type": "string", "description": "Text to replace."},
                "new_string": {"type": "string", "description": "Replacement text."},
                "replace_all": {
                    "type": "boolean",
                    "default": False,
                    "description": "Replace every match instead of only the first one.",
                },
                "encoding": {"type": "string", "default": "utf-8"},
                "approved": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether user approval has been granted when required.",
                },
            },
            "required": ["path", "old_string", "new_string"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        *,
        path: str,
        old_string: str,
        new_string: str,
        replace_all: bool = False,
        encoding: str | None = None,
        approved: bool = False,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        if not old_string:
            return ToolResult(error="`old_string` must not be empty.")

        workspace_root = self._resolve_workspace_root(context)
        call_policy = _resolve_file_mutation_call_policy(
            context=context,
            legacy_require_approval=self._require_approval,
            legacy_path_scope=self._path_scope,
        )
        active_encoding = encoding or self._default_encoding
        target, security_decision = check_file_tool_path(
            path,
            workspace_dir=workspace_root,
            scope=call_policy.path_scope,
        )
        if security_decision is not None or target is None:
            metadata = _build_path_denial_metadata(
                path=path,
                workspace_root=workspace_root,
                path_scope=call_policy.path_scope,
                security_metadata=(
                    security_decision.to_metadata() if security_decision is not None else {}
                ),
            )
            metadata.update(call_policy.metadata())
            if security_decision is not None:
                metadata.update(
                    {
                        "runtime_category": "permission",
                        "error_type": "file_path_denied",
                        "recoverability": "requires_changed_path_or_policy",
                    }
                )
            return ToolResult(
                error=security_decision.reason if security_decision is not None else "Path denied.",
                metadata=metadata,
            )

        existing_result = await _load_existing_content(target, active_encoding)
        if existing_result.error is not None:
            return existing_result
        existing_content = str(existing_result.output)

        guard_error = await _check_stale_write_guard(
            target=target,
            current_content=existing_content,
            context=context,
            require_prior_read=True,
        )
        if guard_error is not None:
            return guard_error

        if old_string not in existing_content:
            return ToolResult(
                error="`old_string` was not found in the file. Re-read the file before editing.",
                suggestion="Read the latest file contents and retry with an exact match.",
            )

        if replace_all:
            new_content = existing_content.replace(old_string, new_string)
        else:
            new_content = existing_content.replace(old_string, new_string, 1)

        file_change = build_file_change_entry(
            target=target,
            workspace_root=workspace_root,
            tool_name=self.name,
            change_type="update",
            original_content=existing_content,
            new_content=new_content,
            encoding=active_encoding,
            undo_max_size_mb=self._undo_max_size_mb,
            extra={"append": False, "edit_type": "replace_all" if replace_all else "replace_first"},
        )
        metadata = build_file_change_payload([file_change])
        metadata.update(
            {
                "workspace_dir": str(workspace_root),
                "resolved_path": str(target),
                **call_policy.metadata(),
            }
        )
        approval_replay = bool(
            context is not None and context.state.get("approval_replay") is True
        )
        trusted_approved = bool(
            approved and (self._approval_store is None or approval_replay)
        )
        if call_policy.require_approval and not trusted_approved:
            decision = require_approval_decision(
                reason="File edits require explicit approval in the current autonomy mode.",
                approval_kind="file_edit",
                approval_scope="workspace",
                replay_safe=True,
                policy_source="runtime_policy",
            )
            decision = with_task_isolation_scope(
                decision,
                task_sandbox_dir=context.task_sandbox_dir if context is not None else None,
            )
            metadata.update(decision.to_metadata())
            if self._approval_store is not None:
                before = await asyncio.to_thread(target.read_bytes)
                approval = _create_file_mutation_approval(
                    tool=self,
                    approval_store=self._approval_store,
                    arguments={
                        "path": target.relative_to(workspace_root).as_posix(),
                        "old_string": old_string,
                        "new_string": new_string,
                        "replace_all": replace_all,
                        "encoding": active_encoding,
                    },
                    workspace_root=workspace_root,
                    base_states=[
                        _file_base_state(
                            target=target,
                            workspace_root=workspace_root,
                            before=before,
                        )
                    ],
                    preview_metadata=metadata,
                    reason=decision.reason,
                    call_policy=call_policy,
                    context=context,
                )
                await _observe_ordinary_chat_approval(context, approval)
                metadata.update(
                    {
                        "status": "approval_pending",
                        "approval_id": approval.approval_id,
                        "operation_id": approval.metadata.get("operation_id"),
                        "arguments_digest": approval.metadata.get("arguments_digest"),
                        "tool_inventory_version": approval.metadata.get(
                            "tool_inventory_version"
                        ),
                    }
                )
            return ToolResult(
                error="File edit requires approval.",
                metadata=metadata,
            )

        review_error = await _enforce_file_auto_review(
            workspace_root=workspace_root,
            tool_name=self.name,
            call_policy=call_policy,
            changes=[
                (
                    target,
                    "update",
                    await asyncio.to_thread(target.read_bytes),
                    new_content.encode(active_encoding),
                )
            ],
            patch_sha256=None,
            context=context,
            metadata=metadata,
            approved=trusted_approved,
        )
        if review_error is not None:
            return review_error

        await mark_context_side_effect_started(context)
        bytes_written = await self._writer(target, new_content, False, active_encoding)
        await _refresh_read_state_cache(
            context=context,
            target=target,
            content=new_content,
            encoding=active_encoding,
        )
        metadata["bytes_written"] = bytes_written
        metadata["timeline_result_disposition"] = "succeeded"
        return ToolResult(output=str(target), metadata=metadata)


class ApplyPatchTool(FileWriteTool):
    """Apply a strict multi-file patch inside the workspace."""

    @property
    def name(self) -> str:
        return "apply_patch"

    @property
    def description(self) -> str:
        return (
            "Apply a strict patch using *** Begin Patch / *** Add File / "
            "*** Update File / *** Delete File / *** End Patch blocks."
        )

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "patch": {"type": "string", "description": "Strict apply_patch payload."},
                "encoding": {"type": "string", "default": "utf-8"},
                "approved": {
                    "type": "boolean",
                    "default": False,
                    "description": "Whether user approval has been granted when required.",
                },
            },
            "required": ["patch"],
            "additionalProperties": False,
        }

    async def execute(
        self,
        *,
        patch: str,
        encoding: str | None = None,
        approved: bool = False,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        workspace_root = self._resolve_workspace_root(context)
        call_policy = _resolve_file_mutation_call_policy(
            context=context,
            legacy_require_approval=self._require_approval,
            legacy_path_scope=self._path_scope,
        )
        active_encoding = encoding or self._default_encoding
        try:
            prepared, metadata = await prepare_apply_patch(
                patch=patch,
                workspace_dir=workspace_root,
                path_scope=call_policy.path_scope,
                encoding=active_encoding,
                undo_max_size_mb=self._undo_max_size_mb,
                tool_name=self.name,
            )
        except PatchValidationError as exc:
            error_metadata = dict(getattr(exc, "metadata", {}) or {})
            error_metadata.update(call_policy.metadata())
            return ToolResult(
                error=str(exc),
                metadata=error_metadata,
            )

        metadata.update(call_policy.metadata())

        for item in prepared:
            if item.new_content is not None and not is_within_write_size_limit(
                content=item.new_content,
                max_size_mb=self._max_write_size_mb,
                encoding=active_encoding,
            ):
                size_bytes = content_size_bytes(item.new_content, encoding=active_encoding)
                return ToolResult(
                    error=(
                        f"Write content too large: {size_bytes} bytes exceeds limit "
                        f"{size_limit_bytes(self._max_write_size_mb)} bytes."
                    ),
                    metadata={"size_bytes": size_bytes, **metadata},
                )

            if item.original_content is not None:
                guard_error = await _check_stale_write_guard(
                    target=item.target,
                    current_content=item.original_content,
                    context=context,
                    require_prior_read=item.operation.kind != "add",
                )
                if guard_error is not None:
                    return guard_error

        approval_replay = bool(
            context is not None and context.state.get("approval_replay") is True
        )
        trusted_approved = bool(
            approved and (self._approval_store is None or approval_replay)
        )
        if call_policy.require_approval and not trusted_approved:
            decision = require_approval_decision(
                reason="Patch application requires explicit approval in the current autonomy mode.",
                approval_kind="apply_patch",
                approval_scope="workspace",
                replay_safe=True,
                policy_source="runtime_policy",
            )
            decision = with_task_isolation_scope(
                decision,
                task_sandbox_dir=context.task_sandbox_dir if context is not None else None,
            )
            metadata.update(decision.to_metadata())
            if self._approval_store is not None:
                base_states: list[dict[str, Any]] = []
                for item in prepared:
                    before = (
                        await asyncio.to_thread(item.target.read_bytes)
                        if await asyncio.to_thread(item.target.exists)
                        else None
                    )
                    base_states.append(
                        _file_base_state(
                            target=item.target,
                            workspace_root=workspace_root,
                            before=before,
                        )
                    )
                approval = _create_file_mutation_approval(
                    tool=self,
                    approval_store=self._approval_store,
                    arguments={"patch": patch, "encoding": active_encoding},
                    workspace_root=workspace_root,
                    base_states=base_states,
                    preview_metadata=metadata,
                    reason=decision.reason,
                    call_policy=call_policy,
                    context=context,
                )
                await _observe_ordinary_chat_approval(context, approval)
                metadata.update(
                    {
                        "status": "approval_pending",
                        "approval_id": approval.approval_id,
                        "operation_id": approval.metadata.get("operation_id"),
                        "arguments_digest": approval.metadata.get("arguments_digest"),
                        "tool_inventory_version": approval.metadata.get(
                            "tool_inventory_version"
                        ),
                    }
                )
            return ToolResult(
                error="Patch application requires approval.",
                metadata=metadata,
            )

        review_changes: list[tuple[Path, str, bytes | None, bytes | None]] = []
        for item in prepared:
            before = (
                await asyncio.to_thread(item.target.read_bytes)
                if await asyncio.to_thread(item.target.exists)
                else None
            )
            after = (
                item.new_content.encode(active_encoding)
                if item.new_content is not None
                else None
            )
            review_changes.append((item.target, item.operation.kind, before, after))
        review_error = await _enforce_file_auto_review(
            workspace_root=workspace_root,
            tool_name=self.name,
            call_policy=call_policy,
            changes=review_changes,
            patch_sha256=hashlib.sha256(patch.encode("utf-8")).hexdigest(),
            context=context,
            metadata=metadata,
            approved=trusted_approved,
        )
        if review_error is not None:
            return review_error

        await mark_context_side_effect_started(context)
        total_bytes_written = 0
        for item in prepared:
            if item.operation.kind == "delete":
                if await asyncio.to_thread(item.target.exists):
                    await asyncio.to_thread(item.target.unlink)
                if context is not None:
                    context.read_state_cache.pop(str(item.target), None)
                continue

            new_content = item.new_content or ""
            total_bytes_written += await self._writer(item.target, new_content, False, active_encoding)
            await _refresh_read_state_cache(
                context=context,
                target=item.target,
                content=new_content,
                encoding=active_encoding,
            )

        metadata["bytes_written"] = total_bytes_written
        metadata["timeline_result_disposition"] = "succeeded"
        return ToolResult(
            output={"paths": metadata.get("paths", []), "change_count": metadata.get("change_count", 0)},
            metadata=metadata,
        )


async def _load_existing_content(target: Path, encoding: str) -> ToolResult:
    if not target.exists():
        return ToolResult(output="")
    if not target.is_file():
        return ToolResult(error=f"Path is not a file: {target}")
    try:
        text = await asyncio.to_thread(_read_text_preserving_newlines, target, encoding)
    except UnicodeDecodeError:
        return ToolResult(error=f"File is not valid {encoding} text: {target}")
    return ToolResult(output=text)


def _read_text_preserving_newlines(path: Path, encoding: str) -> str:
    with path.open("r", encoding=encoding, errors="strict", newline="") as stream:
        return stream.read()


async def _check_stale_write_guard(
    *,
    target: Path,
    current_content: str,
    context: ToolExecutionContext | None,
    require_prior_read: bool | None = None,
) -> ToolResult | None:
    if context is None:
        return None
    if context.state.get("approval_replay") is True:
        return None

    if require_prior_read is None:
        require_prior_read = target.exists()

    snapshot = context.read_state_cache.get(str(target))
    if snapshot is None:
        if require_prior_read:
            return ToolResult(
                error="File must be read before write/edit.",
                suggestion="Use file_read on the target file before modifying it.",
            )
        return None

    if snapshot.partial:
        return ToolResult(
            error="Partial reads cannot be used for write/edit. Re-read the full file first.",
            suggestion="Read the full file contents before modifying it.",
        )

    if target.exists():
        stat = await asyncio.to_thread(target.stat)
        current_mtime = getattr(stat, "st_mtime_ns", None)
        if snapshot.mtime_ns is not None and current_mtime != snapshot.mtime_ns:
            return ToolResult(
                error="File changed after it was read. Re-read before writing.",
                retryable=True,
                suggestion="Run file_read again to refresh the cached snapshot.",
            )
    elif require_prior_read:
        return ToolResult(
            error="File was removed after it was read. Re-read before writing.",
            retryable=True,
        )

    if snapshot.content != current_content:
        return ToolResult(
            error="File contents are stale compared with the cached read. Re-read before writing.",
            retryable=True,
        )

    return None


async def _refresh_read_state_cache(
    *,
    context: ToolExecutionContext | None,
    target: Path,
    content: str,
    encoding: str,
) -> None:
    if context is None:
        return
    stat = await asyncio.to_thread(target.stat)
    context.read_state_cache[str(target)] = FileReadState(
        path=str(target),
        content=content,
        encoding=encoding,
        mtime_ns=getattr(stat, "st_mtime_ns", None),
        size_bytes=getattr(stat, "st_size", content_size_bytes(content, encoding=encoding)),
        partial=False,
    )
