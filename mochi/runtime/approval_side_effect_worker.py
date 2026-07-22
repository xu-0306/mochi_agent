"""Restart-safe delivery worker for approval side effects."""

from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import ValidationError

from mochi.config.manager import ConfigRevisionConflict, load_config_snapshot, save_config
from mochi.config.schema import CommandRuleConfig
from mochi.runtime.approval_side_effects import (
    claim_side_effect,
    mark_side_effect_delivered,
    mark_side_effect_failed,
    mark_side_effect_retry,
)
from mochi.runtime.security_audit import redact_for_persistence


class ApprovalSideEffectWorker:
    """Claim and deliver config side effects without replaying execution."""

    def __init__(self, database_paths: list[str | Path]) -> None:
        self._database_paths = tuple(dict.fromkeys(Path(path) for path in database_paths))
        self._lease_owner = f"approval-side-effect-worker:{uuid4()}"

    async def deliver_available(self, *, max_items: int = 32) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for _ in range(max(1, max_items)):
            claimed: tuple[Path, dict[str, Any]] | None = None
            for database_path in self._database_paths:
                try:
                    item = await asyncio.to_thread(
                        claim_side_effect,
                        database_path,
                        lease_owner=self._lease_owner,
                    )
                except (OSError, sqlite3.Error):
                    continue
                if item is not None:
                    claimed = (database_path, item)
                    break
            if claimed is None:
                break
            results.append(await self._deliver(*claimed))
        return results

    async def _deliver(
        self,
        database_path: Path,
        item: dict[str, Any],
    ) -> dict[str, Any]:
        side_effect_id = str(item.get("side_effect_id") or "")
        attempts = int(item.get("attempts") or 0)
        try:
            if item.get("kind") != "save_command_rule":
                raise ValueError("Unsupported approval side effect kind.")
            payload = item.get("payload")
            if not isinstance(payload, dict):
                raise ValueError("Command rule payload is missing.")
            rule = CommandRuleConfig.model_validate(payload)
            raw_target_path = str(item.get("target_config_path") or "").strip()
            if not raw_target_path:
                raise ValueError("Target config path is missing.")
            target_path = Path(raw_target_path).expanduser()

            for _ in range(4):
                snapshot = load_config_snapshot(target_path)
                security = snapshot.config.security
                delivered_ids = list(security.applied_rule_side_effect_ids)
                if side_effect_id in delivered_ids:
                    break
                rules = list(security.command_rules)
                if rule not in rules:
                    rules.append(rule)
                delivered_ids.append(side_effect_id)
                updated_security = security.model_copy(
                    update={
                        "command_rules": rules,
                        "applied_rule_side_effect_ids": delivered_ids[-4096:],
                    }
                )
                updated_config = snapshot.config.model_copy(
                    update={"security": updated_security}
                )
                try:
                    save_config(
                        updated_config,
                        target_path,
                        expected_revision=snapshot.revision,
                    )
                    break
                except ConfigRevisionConflict:
                    continue
            else:
                raise ConfigRevisionConflict(
                    expected_revision=snapshot.revision,
                    current_revision=load_config_snapshot(target_path).revision,
                    path=target_path,
                )

            delivered = await asyncio.to_thread(
                mark_side_effect_delivered,
                database_path,
                side_effect_id,
                lease_owner=self._lease_owner,
            )
            return {
                "side_effect_id": side_effect_id,
                "approval_id": item.get("approval_id"),
                "status": "delivered" if delivered else "retrying",
                "error": None,
            }
        except (ValueError, ValidationError) as exc:
            error = _bounded_error(exc)
            await asyncio.to_thread(
                mark_side_effect_failed,
                database_path,
                side_effect_id,
                lease_owner=self._lease_owner,
                error=error,
            )
            return {
                "side_effect_id": side_effect_id,
                "approval_id": item.get("approval_id"),
                "status": "failed",
                "error": error,
            }
        except (ConfigRevisionConflict, OSError) as exc:
            error = _bounded_error(exc)
            marker = mark_side_effect_failed if attempts >= 5 else mark_side_effect_retry
            marker_kwargs: dict[str, Any] = {
                "lease_owner": self._lease_owner,
                "error": error,
            }
            if marker is mark_side_effect_retry:
                marker_kwargs["retry_after_seconds"] = min(60, 2 ** max(0, attempts - 1))
            await asyncio.to_thread(marker, database_path, side_effect_id, **marker_kwargs)
            return {
                "side_effect_id": side_effect_id,
                "approval_id": item.get("approval_id"),
                "status": "failed" if attempts >= 5 else "retrying",
                "error": error,
            }


def _bounded_error(exc: BaseException) -> str:
    if isinstance(exc, ValidationError):
        return "Command rule payload is invalid."
    if isinstance(exc, ConfigRevisionConflict):
        return "Config changed while saving the command rule."
    if isinstance(exc, OSError):
        return f"Config persistence failed temporarily ({exc.__class__.__name__})."
    text = str(exc).strip() or exc.__class__.__name__
    redacted = redact_for_persistence(text[:500])
    return redacted if isinstance(redacted, str) else exc.__class__.__name__


__all__ = ["ApprovalSideEffectWorker"]
