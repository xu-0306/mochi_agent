"""Helpers for normalizing runtime transcript events into UI-safe payloads."""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha1
from typing import Any

from mochi.runtime.security_audit import redact_for_persistence

_SENSITIVE_KEY_PARTS = (
    "api_key",
    "authorization",
    "cookie",
    "env",
    "password",
    "secret",
    "stderr",
    "stdout",
    "token",
)
_SAFE_EVENT_FIELDS = {
    "allowed_decisions",
    "approval_id",
    "approval_ids",
    "approval_kind",
    "approval_scope",
    "approval_status",
    "approval_wait_elapsed_sec",
    "approval_wait_expires_at",
    "approval_wait_started_at",
    "approval_wait_timeout_sec",
    "approval_ids",
    "arguments_preview",
    "blocker_type",
    "content",
    "created_at",
    "delivery_mode",
    "delivery_reason",
    "delivery_status",
    "dedupeKey",
    "dedupe_key",
    "durability",
    "eventId",
    "event_id",
    "interrupt",
    "cancel_current_tool",
    "pending_approvals",
    "policy_source",
    "projectionLane",
    "projection_lane",
    "current_action",
    "model_id",
    "name",
    "parent_id",
    "parent_type",
    "reason",
    "replay_safe",
    "message_id",
    "prompt_preview",
    "recommended_action",
    "role_id",
    "security_decision",
    "status",
    "subagent_id",
    "summary",
    "system_prompt",
    "title",
    "tool_call_id",
    "tool_names",
    "tool_name",
    "type",
    "user_prompt",
    "visibility",
}
_GOAL_SURFACE_LIFECYCLE_EVENT_TYPES = {
    "runtime_blocked",
    "subagent_completed",
    "subagent_interrupted",
    "subagent_started",
}
_GOAL_SURFACE_TOOL_RESULT_STATUSES = {
    "approval_required",
    "awaiting_approval",
    "blocked",
    "cancelled",
    "error",
    "failed",
}
_ROLE_EVENT_TYPE_MAP = {
    "role_started": "subagent_started",
    "role_progress": "subagent_progress",
    "role_completed": "subagent_completed",
    "role_error": "subagent_completed",
}
_ALLOWED_EVENT_TYPES = {
    "runtime_blocked",
    "subagent_completed",
    "subagent_progress",
    "subagent_prompt",
    "subagent_started",
    "subagent_thinking",
    "subagent_message_accepted",
    "subagent_message_applied",
    "subagent_message_cancelled",
    "subagent_message_deferred",
    "subagent_message_queued",
    "subagent_interrupted",
    "subagent_tool_call",
    "subagent_tool_cancel_requested",
    "subagent_tool_cancelled",
    "subagent_tool_cancel_deferred",
    "subagent_tool_result",
}


def normalize_subagent_event(
    raw: Mapping[str, Any],
    *,
    parent_type: str,
    parent_id: str,
) -> dict[str, Any]:
    """Normalize legacy and current runtime events into a UI-safe transcript event."""

    raw_event = dict(raw)
    raw_type = str(raw_event.get("type") or "").strip()
    event_type = _ROLE_EVENT_TYPE_MAP.get(raw_type, raw_type)
    if event_type not in _ALLOWED_EVENT_TYPES:
        raise ValueError(f"unsupported transcript event type: {raw_type or '<missing>'}")

    role_id = _clean_optional_text(raw_event.get("role_id"))
    title = _clean_optional_text(raw_event.get("title")) or _clean_optional_text(
        raw_event.get("name")
    )
    created_at = _clean_optional_text(raw_event.get("created_at"))
    stage = _clean_optional_text(raw_event.get("stage")) or raw_type or event_type
    subagent_id = _clean_optional_text(raw_event.get("subagent_id"))
    if subagent_id is None and event_type != "runtime_blocked":
        subagent_id = _fallback_subagent_id(
            parent_id=parent_id,
            role_id=role_id,
            stage=stage,
        )

    normalized: dict[str, Any] = {
        "type": event_type,
        "parent_type": parent_type,
        "parent_id": parent_id,
        "subagent_id": subagent_id,
        "role_id": role_id,
        "title": title,
        "model_id": _clean_optional_text(raw_event.get("model_id")),
        "content": _clean_optional_text(raw_event.get("content"))
        or _clean_optional_text(raw_event.get("current_action")),
        "summary": _clean_optional_text(raw_event.get("summary")),
        "status": _clean_optional_text(raw_event.get("status")),
        "approval_id": _clean_optional_text(raw_event.get("approval_id")),
        "approval_kind": _clean_optional_text(raw_event.get("approval_kind")),
        "approval_scope": _clean_optional_text(raw_event.get("approval_scope")),
        "approval_status": _clean_optional_text(raw_event.get("approval_status")),
        "approval_ids": _normalize_string_list(raw_event.get("approval_ids")),
        "allowed_decisions": _normalize_string_list(raw_event.get("allowed_decisions")),
        "pending_approvals": _normalize_pending_approvals(
            raw_event.get("pending_approvals") or raw_event.get("pendingApprovals")
        ),
        "tool_names": _normalize_string_list(raw_event.get("tool_names")),
        "recommended_action": _clean_optional_text(raw_event.get("recommended_action")),
        "blocker_type": _clean_optional_text(raw_event.get("blocker_type")),
        "reason": _clean_optional_text(raw_event.get("reason")),
        "replay_safe": _normalize_optional_bool(raw_event.get("replay_safe")),
        "security_decision": _clean_optional_text(raw_event.get("security_decision")),
        "policy_source": _clean_optional_text(raw_event.get("policy_source")),
        "approval_wait_started_at": _clean_optional_text(raw_event.get("approval_wait_started_at")),
        "approval_wait_expires_at": _clean_optional_text(raw_event.get("approval_wait_expires_at")),
        "approval_wait_timeout_sec": _normalize_optional_number(
            raw_event.get("approval_wait_timeout_sec")
        ),
        "approval_wait_elapsed_sec": _normalize_optional_number(
            raw_event.get("approval_wait_elapsed_sec")
        ),
        "prompt_preview": _clean_optional_text(raw_event.get("prompt_preview")),
        "system_prompt": _clean_optional_text(raw_event.get("system_prompt")),
        "user_prompt": _clean_optional_text(raw_event.get("user_prompt")),
        "tool_call_id": _clean_optional_text(raw_event.get("tool_call_id")),
        "tool_name": _clean_optional_text(raw_event.get("tool_name")),
        "arguments_preview": _clean_optional_text(raw_event.get("arguments_preview")),
        "message_id": _clean_optional_text(raw_event.get("message_id")),
        "delivery_mode": _clean_optional_text(raw_event.get("delivery_mode")),
        "delivery_status": _clean_optional_text(raw_event.get("delivery_status")),
        "delivery_reason": _clean_optional_text(
            raw_event.get("delivery_reason") or raw_event.get("reason")
        ),
        "interrupt": _normalize_optional_bool(raw_event.get("interrupt")),
        "cancel_current_tool": _normalize_optional_bool(raw_event.get("cancel_current_tool")),
        "metadata": _filtered_metadata(raw_event),
        "created_at": created_at,
    }

    if raw_type == "role_error":
        status = normalized["status"]
        normalized["status"] = status or (
            "blocked" if normalized["blocker_type"] or normalized["approval_ids"] else "failed"
        )
    elif event_type == "subagent_started" and normalized["status"] is None:
        normalized["status"] = "running"
    elif (
        event_type
        in {
            "subagent_progress",
            "subagent_thinking",
            "subagent_prompt",
            "subagent_tool_call",
            "subagent_message_accepted",
            "subagent_message_queued",
            "subagent_interrupted",
            "subagent_tool_cancel_requested",
        }
        and normalized["status"] is None
    ):
        normalized["status"] = "running"
    elif event_type == "subagent_message_applied" and normalized["status"] is None:
        normalized["status"] = "running"
        normalized["delivery_status"] = normalized["delivery_status"] or "applied"
    elif event_type == "subagent_message_deferred" and normalized["status"] is None:
        normalized["status"] = "running"
        normalized["delivery_status"] = normalized["delivery_status"] or "deferred"
    elif event_type == "subagent_message_cancelled" and normalized["status"] is None:
        normalized["status"] = "cancelled"
        normalized["delivery_status"] = normalized["delivery_status"] or "cancelled"
    elif event_type == "subagent_tool_cancelled" and normalized["status"] is None:
        normalized["status"] = "cancelled"
    elif event_type == "subagent_tool_cancel_deferred" and normalized["status"] is None:
        normalized["status"] = "running"
    elif event_type == "subagent_completed" and normalized["status"] is None:
        normalized["status"] = "completed"

    normalized.update(
        _source_contract_fields(
            raw_event=raw_event,
            normalized=normalized,
            parent_type=parent_type,
            parent_id=parent_id,
            event_type=event_type,
        )
    )

    projected = {key: value for key, value in normalized.items() if value is not None}
    redacted = redact_for_persistence(projected)
    return redacted if isinstance(redacted, dict) else {}


def _source_contract_fields(
    *,
    raw_event: Mapping[str, Any],
    normalized: Mapping[str, Any],
    parent_type: str,
    parent_id: str,
    event_type: str,
) -> dict[str, str]:
    event_id = _clean_optional_text(
        raw_event.get("event_id")
        or raw_event.get("eventId")
    ) or _derive_event_id(
        raw_event=raw_event,
        normalized=normalized,
        parent_type=parent_type,
        parent_id=parent_id,
        event_type=event_type,
    )
    dedupe_key = _clean_optional_text(
        raw_event.get("dedupe_key") or raw_event.get("dedupeKey")
    ) or event_id
    visibility = _clean_optional_text(raw_event.get("visibility")) or _derive_visibility()
    durability = _clean_optional_text(raw_event.get("durability")) or _derive_durability()
    projection_lane = _clean_optional_text(
        raw_event.get("projection_lane") or raw_event.get("projectionLane")
    ) or _derive_projection_lane(event_type=event_type, normalized=normalized)

    return {
        "event_id": event_id,
        "dedupe_key": dedupe_key,
        "visibility": visibility,
        "durability": durability,
        "projection_lane": projection_lane,
    }


def _derive_visibility() -> str:
    return "visible"


def _derive_durability() -> str:
    # Execution transcript rows are persisted for replay, but they still belong to the
    # transient execution surface unless a producer explicitly upgrades them.
    return "transient"


def _derive_event_id(
    *,
    raw_event: Mapping[str, Any],
    normalized: Mapping[str, Any],
    parent_type: str,
    parent_id: str,
    event_type: str,
) -> str:
    identity_parts = [
        parent_type,
        parent_id,
        event_type,
        _clean_optional_text(normalized.get("subagent_id")) or "",
        _clean_optional_text(normalized.get("role_id")) or "",
        _clean_optional_text(normalized.get("status")) or "",
        _clean_optional_text(normalized.get("message_id")) or "",
        _clean_optional_text(normalized.get("tool_call_id")) or "",
        _clean_optional_text(normalized.get("tool_name")) or "",
        _clean_optional_text(normalized.get("blocker_type")) or "",
        ",".join(_normalize_string_list(normalized.get("approval_ids")) or []),
        ",".join(_normalize_string_list(normalized.get("tool_names")) or []),
        _clean_optional_text(normalized.get("recommended_action")) or "",
        _clean_optional_text(raw_event.get("seq")) or "",
        _clean_optional_text(raw_event.get("sequence")) or "",
        _clean_optional_text(raw_event.get("event_seq")) or "",
        _clean_optional_text(raw_event.get("created_at")) or "",
        _clean_optional_text(raw_event.get("timestamp")) or "",
        _clean_optional_text(raw_event.get("stage")) or "",
        _clean_optional_text(normalized.get("summary")) or "",
        _clean_optional_text(normalized.get("content")) or "",
        _clean_optional_text(normalized.get("reason")) or "",
    ]
    digest = sha1("\x1f".join(identity_parts).encode("utf-8")).hexdigest()[:16]
    return f"evt_{digest}"


def _derive_projection_lane(
    *,
    event_type: str,
    normalized: Mapping[str, Any],
) -> str:
    if event_type in _GOAL_SURFACE_LIFECYCLE_EVENT_TYPES:
        return "goal_surface"
    if event_type == "subagent_tool_result":
        status = (_clean_optional_text(normalized.get("status")) or "").lower()
        if status in _GOAL_SURFACE_TOOL_RESULT_STATUSES:
            return "goal_surface"
    return "subagent_detail"


def _fallback_subagent_id(*, parent_id: str, role_id: str | None, stage: str) -> str:
    safe_role = _slugify(role_id or "subagent")
    safe_stage = _slugify(stage or "event")
    digest = sha1(f"{parent_id}:{role_id or ''}:{stage}".encode("utf-8")).hexdigest()[:8]
    if role_id:
        return f"{parent_id}:{safe_role}:{safe_stage}"
    return f"{parent_id}:{safe_stage}:{digest}"


def _filtered_metadata(raw_event: Mapping[str, Any]) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    raw_metadata = raw_event.get("metadata")
    if isinstance(raw_metadata, Mapping):
        for key, value in raw_metadata.items():
            if _is_sensitive_key(key):
                continue
            metadata[str(key)] = value
    for key, value in raw_event.items():
        if key in _SAFE_EVENT_FIELDS or _is_sensitive_key(key):
            continue
        metadata[str(key)] = value
    return metadata


def _is_sensitive_key(key: Any) -> bool:
    normalized = str(key or "").strip().lower()
    if not normalized:
        return False
    return any(part in normalized for part in _SENSITIVE_KEY_PARTS)


def _normalize_string_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        return None
    normalized = [text for item in items if (text := _clean_optional_text(item))]
    return normalized or None


def _normalize_pending_approvals(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, list):
        return None
    normalized_items: list[dict[str, Any]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        metadata = item.get("metadata") if isinstance(item.get("metadata"), Mapping) else {}
        normalized = {
            "approval_id": _clean_optional_text(item.get("approval_id") or metadata.get("approval_id")),
            "tool_name": _clean_optional_text(item.get("tool_name") or metadata.get("tool_name")),
            "reason": _clean_optional_text(item.get("reason") or metadata.get("reason")),
            "status": _clean_optional_text(item.get("status") or metadata.get("status")),
            "request_id": _clean_optional_text(item.get("request_id")),
            "task_key": _clean_optional_text(item.get("task_key")),
            "stage": _clean_optional_text(item.get("stage")),
            "role_id": _clean_optional_text(item.get("role_id")),
            "source": _clean_optional_text(item.get("source")),
            "created_at": _clean_optional_text(
                item.get("created_at")
                or item.get("requested_at")
                or item.get("submitted_at")
                or metadata.get("created_at")
            ),
            "approval_kind": _clean_optional_text(
                item.get("approval_kind") or metadata.get("approval_kind")
            ),
            "approval_scope": _clean_optional_text(
                item.get("approval_scope") or metadata.get("approval_scope")
            ),
            "replay_safe": _normalize_optional_bool(
                item.get("replay_safe")
                if item.get("replay_safe") is not None
                else metadata.get("replay_safe")
            ),
            "security_decision": _clean_optional_text(
                item.get("security_decision") or metadata.get("security_decision")
            ),
            "policy_source": _clean_optional_text(
                item.get("policy_source") or metadata.get("policy_source")
            ),
            "allowed_decisions": _normalize_string_list(
                item.get("allowed_decisions") or metadata.get("allowed_decisions")
            ),
            "approval_wait_started_at": _clean_optional_text(
                item.get("approval_wait_started_at")
                or item.get("waiting_since")
                or item.get("pending_since")
                or metadata.get("approval_wait_started_at")
            ),
            "approval_wait_expires_at": _clean_optional_text(
                item.get("approval_wait_expires_at")
                or item.get("expires_at")
                or metadata.get("approval_wait_expires_at")
            ),
            "approval_wait_timeout_sec": _normalize_optional_number(
                item.get("approval_wait_timeout_sec")
                if item.get("approval_wait_timeout_sec") is not None
                else item.get("timeout_sec")
                if item.get("timeout_sec") is not None
                else metadata.get("approval_wait_timeout_sec")
            ),
            "approval_wait_elapsed_sec": _normalize_optional_number(
                item.get("approval_wait_elapsed_sec")
                if item.get("approval_wait_elapsed_sec") is not None
                else metadata.get("approval_wait_elapsed_sec")
            ),
        }
        filtered = {key: field for key, field in normalized.items() if field is not None}
        if filtered:
            normalized_items.append(filtered)
    return normalized_items or None


def _normalize_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _normalize_optional_number(value: Any) -> int | float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    return None


def _clean_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _slugify(value: str) -> str:
    pieces: list[str] = []
    pending_dash = False
    for char in value.lower():
        if char.isalnum():
            if pending_dash and pieces:
                pieces.append("-")
            pieces.append(char)
            pending_dash = False
        else:
            pending_dash = True
    return "".join(pieces) or "item"
