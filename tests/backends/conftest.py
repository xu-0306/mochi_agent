"""Shared backend fixtures."""

from __future__ import annotations

import pytest

from mochi.backends.ollama import OllamaBackend


@pytest.fixture
def backend() -> OllamaBackend:
    """建立 OllamaBackend 測試實例。"""
    return OllamaBackend(model="llama3.2", base_url="http://localhost:11434")