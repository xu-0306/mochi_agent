from __future__ import annotations

import pytest

from tests.browser._fixture_server import BrowserFixtureServer, start_fixture_server


@pytest.fixture
def browser_fixture_server() -> BrowserFixtureServer:
    """Expose the deterministic HTTP fixture to subsequent browser packages."""

    server = start_fixture_server()
    try:
        yield server
    finally:
        server.stop()
