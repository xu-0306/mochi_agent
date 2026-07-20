"""pytest 設定與共用 fixtures。"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from mochi.config.schema import MochiConfig

_SLOW_TEST_NAMES = frozenset(
    {
        "test_openai_codex_refresh_access_token_times_out_on_live_file_lock",
        "test_web_search_metadata_distinguishes_missing_key_and_request_failure",
        "test_agent_run_events_stream_returns_sse_frames",
        "test_engine_persists_and_restores_session_history",
        "test_engine_compaction_does_not_pollute_canonical_restore",
        "test_registry_factory_caches_registries_per_workspace",
        "test_engine_preview_and_chat_invoke_share_classifier_first_tool_intent_contract",
        "test_engine_invoke_exposes_tool_exposure_metadata_from_final_plan",
        "test_engine_chinese_workspace_prompt_exposes_workspace_read_baseline",
        "test_engine_apply_config_refreshes_active_gguf_runtime_root",
    }
)


def pytest_configure(config: pytest.Config) -> None:
    """Keep pytest per-run temp root inside the workspace by default."""
    if getattr(config.option, "basetemp", None):
        return

    repo_root = Path(__file__).resolve().parents[1]
    base_root = repo_root / ".tmp" / "pytest-runs"
    base_root.mkdir(parents=True, exist_ok=True)
    run_id = f"{datetime.now(UTC):%Y%m%d-%H%M%S}-{os.getpid()}"
    config.option.basetemp = str(base_root / run_id)


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Classify tests from their structural path and measured baseline cost."""
    for item in items:
        path = item.nodeid.split("::", 1)[0].replace("\\", "/").lower()
        path_parts = set(path.split("/"))

        if "integration" in path_parts:
            item.add_marker("integration")
        if "security" in path_parts:
            item.add_marker("security")
        if "windows" in path:
            item.add_marker("windows")
        if "posix" in path:
            item.add_marker("posix")
        if item.name in _SLOW_TEST_NAMES:
            item.add_marker("slow")


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
