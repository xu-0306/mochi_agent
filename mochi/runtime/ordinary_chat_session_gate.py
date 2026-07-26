"""Fail-closed session and policy gate for ordinary-Chat approvals."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from mochi.config.schema import SecurityConfig
from mochi.security.policy import EffectivePolicyResolver
from mochi.sessions.store import SessionStore, StrictSessionSnapshotError


class OrdinaryChatSessionGateError(ValueError):
    """A durable Chat session cannot safely authorize an approval operation."""

    def __init__(self, reason: str) -> None:
        self.reason = reason
        super().__init__(reason)


class OrdinaryChatSessionGate:
    """Derive approval policy from one validated, immutable session snapshot.

    The gate intentionally has no client-provided policy input.  Callers only
    receive an effective policy after the durable session identity and its
    creation record are validated from the same strict snapshot.
    """

    def __init__(
        self,
        *,
        session_store: SessionStore,
        security: SecurityConfig,
        policy_resolver: EffectivePolicyResolver | None = None,
    ) -> None:
        self._session_store = session_store
        self._security = security
        self._policy_resolver = policy_resolver or EffectivePolicyResolver()

    async def effective_policy(self, session_id: str | None) -> dict[str, Any]:
        if not isinstance(session_id, str) or not session_id.strip():
            raise OrdinaryChatSessionGateError("invalid")
        normalized_session_id = session_id.strip()
        try:
            snapshot = await self._session_store.load_strict_snapshot(normalized_session_id)
        except (StrictSessionSnapshotError, TypeError, ValueError):
            raise OrdinaryChatSessionGateError("invalid") from None
        if not snapshot.exists:
            raise OrdinaryChatSessionGateError("missing")
        events = snapshot.events
        if not _has_matching_creation_event(events, session_id=normalized_session_id):
            raise OrdinaryChatSessionGateError("invalid")
        try:
            override = _session_security_override(events)
        except ValueError:
            raise OrdinaryChatSessionGateError("invalid") from None
        return self._policy_resolver.resolve(
            self._security,
            session_overrides=override,
        ).to_dict()


def _has_matching_creation_event(
    events: Sequence[Mapping[str, Any]],
    *,
    session_id: str,
) -> bool:
    return any(
        event.get("type") == "session_meta"
        and event.get("event") == "created"
        and event.get("session_id") == session_id
        for event in events
    )


def _session_security_override(
    events: Sequence[Mapping[str, Any]],
) -> dict[str, object] | None:
    for event in reversed(events):
        if event.get("type") != "session_meta":
            continue
        if event.get("event") not in {"created", "security_override_updated"}:
            continue
        if "security_override" not in event:
            return None
        override = event.get("security_override")
        if not isinstance(override, Mapping):
            raise ValueError("security override must be an object")
        if set(override) != {"autonomy_mode"}:
            raise ValueError("security override has unsupported fields")
        autonomy_mode = override.get("autonomy_mode")
        if autonomy_mode in {"strict", "trusted_workspace", "auto_review", "high_autonomy"}:
            return {"autonomy_mode": str(autonomy_mode)}
        raise ValueError("security override autonomy_mode is invalid")
    return None


__all__ = ["OrdinaryChatSessionGate", "OrdinaryChatSessionGateError"]
