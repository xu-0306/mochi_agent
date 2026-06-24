"""pytest 設定與共用 fixtures。"""

from __future__ import annotations

import pytest

from mochi.config.schema import MochiConfig


@pytest.fixture
def default_config() -> MochiConfig:
    """回傳預設設定實例。"""
    return MochiConfig()


@pytest.fixture
def ollama_config() -> MochiConfig:
    """回傳指向本地 Ollama 的設定實例。"""
    cfg = MochiConfig()
    cfg.model = "ollama:llama3.2"
    cfg.ollama.base_url = "http://localhost:11434"
    return cfg
