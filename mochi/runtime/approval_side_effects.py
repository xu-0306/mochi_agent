"""Durable claim/lease operations for approval side-effect outboxes."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from mochi.runtime.models import ApprovalRulePersistenceProjection
from mochi.runtime.security_audit import redact_for_persistence


def claim_side_effect(
    db_path: str | Path,
    *,
    lease_owner: str,
    lease_seconds: int = 30,
) -> dict[str, Any] | None:
    now = _now_iso()
    lease_expires_at = (
        datetime.now(UTC) + timedelta(seconds=max(1, lease_seconds))
    ).isoformat()
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT * FROM approval_side_effects
            WHERE status='pending'
               OR (status='retrying' AND (lease_expires_at IS NULL OR lease_expires_at<=?))
            ORDER BY created_at, side_effect_id
            LIMIT 1
            """,
            (now,),
        ).fetchone()
        if row is None:
            conn.rollback()
            return None
        cursor = conn.execute(
            """
            UPDATE approval_side_effects
            SET status='retrying',attempts=attempts+1,lease_owner=?,
                lease_expires_at=?,updated_at=?
            WHERE side_effect_id=?
              AND (status='pending'
                   OR (status='retrying' AND (lease_expires_at IS NULL OR lease_expires_at<=?)))
            """,
            (lease_owner, lease_expires_at, now, row["side_effect_id"], now),
        )
        if cursor.rowcount != 1:
            conn.rollback()
            return None
        conn.commit()
    return get_side_effect(db_path, str(row["side_effect_id"]), include_payload=True)


def get_side_effect(
    db_path: str | Path,
    side_effect_id: str,
    *,
    include_payload: bool = False,
) -> dict[str, Any] | None:
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        row = conn.execute(
            "SELECT * FROM approval_side_effects WHERE side_effect_id=?",
            (side_effect_id,),
        ).fetchone()
    return _decode(row, include_payload=include_payload)


def list_side_effects(
    db_path: str | Path,
    *,
    approval_id: str | None = None,
    include_payload: bool = False,
) -> list[dict[str, Any]]:
    with _connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        if approval_id is None:
            rows = conn.execute(
                "SELECT * FROM approval_side_effects ORDER BY created_at,side_effect_id"
            ).fetchall()
        else:
            rows = conn.execute(
                """
                SELECT * FROM approval_side_effects
                WHERE approval_id=? ORDER BY created_at,side_effect_id
                """,
                (approval_id,),
            ).fetchall()
    return [
        item
        for row in rows
        if (item := _decode(row, include_payload=include_payload)) is not None
    ]


def mark_side_effect_delivered(
    db_path: str | Path,
    side_effect_id: str,
    *,
    lease_owner: str,
) -> bool:
    now = _now_iso()
    return _transition(
        db_path,
        side_effect_id,
        lease_owner=lease_owner,
        status="delivered",
        last_error=None,
        delivered_at=now,
    )


def mark_side_effect_retry(
    db_path: str | Path,
    side_effect_id: str,
    *,
    lease_owner: str,
    error: str,
    retry_after_seconds: int = 1,
) -> bool:
    return _transition(
        db_path,
        side_effect_id,
        lease_owner=lease_owner,
        status="retrying",
        last_error=error,
        delivered_at=None,
        retry_at=(
            datetime.now(UTC) + timedelta(seconds=max(1, retry_after_seconds))
        ).isoformat(),
    )


def mark_side_effect_failed(
    db_path: str | Path,
    side_effect_id: str,
    *,
    lease_owner: str,
    error: str,
) -> bool:
    return _transition(
        db_path,
        side_effect_id,
        lease_owner=lease_owner,
        status="failed",
        last_error=error,
        delivered_at=None,
        retry_at=None,
    )


def _transition(
    db_path: str | Path,
    side_effect_id: str,
    *,
    lease_owner: str,
    status: str,
    last_error: str | None,
    delivered_at: str | None,
    retry_at: str | None = None,
) -> bool:
    now = _now_iso()
    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            UPDATE approval_side_effects
            SET status=?,lease_owner=NULL,lease_expires_at=?,last_error=?,
                delivered_at=COALESCE(?,delivered_at),updated_at=?
            WHERE side_effect_id=? AND status='retrying' AND lease_owner=?
            """,
            (
                status,
                retry_at,
                last_error,
                delivered_at,
                now,
                side_effect_id,
                lease_owner,
            ),
        )
        conn.commit()
        return cursor.rowcount == 1


def public_side_effect_projection(item: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(item, dict):
        return ApprovalRulePersistenceProjection().model_dump()
    projected_error = redact_for_persistence({"path": item.get("last_error")})
    redacted_error = (
        projected_error.get("path") if isinstance(projected_error, dict) else None
    )
    raw_status = item.get("status") or "pending"
    status = (
        raw_status
        if raw_status in {"pending", "retrying", "delivered", "failed"}
        else "failed"
    )
    return ApprovalRulePersistenceProjection(
        rule_persistence_status=status,
        side_effect_id=(
            item.get("side_effect_id")
            if isinstance(item.get("side_effect_id"), str)
            else None
        ),
        rule_persistence_error=(
            redacted_error if isinstance(redacted_error, str) else None
        ),
    ).model_dump()


def _connect(db_path: str | Path) -> sqlite3.Connection:
    conn = sqlite3.connect(Path(db_path), timeout=5.0)
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _decode(row: sqlite3.Row | None, *, include_payload: bool) -> dict[str, Any] | None:
    if row is None:
        return None
    item = dict(row)
    raw_payload = item.pop("payload_json", None)
    if include_payload:
        try:
            item["payload"] = json.loads(raw_payload or "{}")
        except (json.JSONDecodeError, TypeError):
            # Preserve the claimed row so the worker can transition malformed
            # durable payloads to a terminal failure instead of crashing its loop.
            item["payload"] = None
    return item


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


__all__ = [
    "claim_side_effect",
    "get_side_effect",
    "list_side_effects",
    "mark_side_effect_delivered",
    "mark_side_effect_failed",
    "mark_side_effect_retry",
    "public_side_effect_projection",
]
