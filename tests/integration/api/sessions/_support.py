from __future__ import annotations

from mochi.api.server import create_app
from mochi.config.schema import MochiConfig
from mochi.sessions.store import SessionStore


def _create_test_app(*, config: MochiConfig, session_store: SessionStore | None = None):
    app = create_app()
    app.state.config_factory = lambda: config
    if session_store is not None:
        app.state.session_store = session_store
    return app
