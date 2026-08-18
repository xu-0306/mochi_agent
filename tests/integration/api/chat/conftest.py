"""Shared pytest boundary for Chat API integration tests."""

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def isolate_chat_api_storage(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep relative Mochi defaults inside the current test's temp directory."""

    monkeypatch.chdir(tmp_path)
