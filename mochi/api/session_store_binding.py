"""One route-level binding policy for durable session storage."""

from __future__ import annotations

from typing import Any

from mochi.sessions.store import (
    SessionStore,
    ToolWorkflowPublicationGate,
    ensure_sessions_dir_unchanged,
)


def resolve_route_session_store(app: Any, config: Any) -> SessionStore:
    """Return the Engine store first, preserving its live rollout boundary."""

    engine = getattr(app.state, "engine", None)
    engine_store = getattr(engine, "_session_store", None)
    if isinstance(engine_store, SessionStore):
        ensure_sessions_dir_unchanged(config.sessions_dir, engine_store.sessions_dir)
        app.state.session_store = engine_store
        return engine_store

    existing = getattr(app.state, "session_store", None)
    if isinstance(existing, SessionStore):
        ensure_sessions_dir_unchanged(config.sessions_dir, existing.sessions_dir)
        return existing

    observability_enabled = bool(
        getattr(getattr(config, "agent", None), "tool_observability_v1", False)
    )
    engine_gate = getattr(engine, "tool_workflow_publication_gate", None)
    gate = (
        engine_gate
        if isinstance(engine_gate, ToolWorkflowPublicationGate)
        else ToolWorkflowPublicationGate(observability_enabled)
    )
    store = SessionStore(
        config.sessions_dir,
        tool_observability_v1=observability_enabled,
        tool_workflow_publication_gate=gate,
    )
    app.state.session_store = store
    return store
