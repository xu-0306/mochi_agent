from __future__ import annotations

from collections.abc import AsyncIterator

from mochi.agents.tool_exposure import ToolExposurePlanner
from mochi.backends.base import BaseLLMBackend
from mochi.backends.types import GenerationResult, Message, ModelInfo, StreamChunk, ToolSchema


def _tool_capabilities(*tool_names: str) -> dict[str, dict]:
    capabilities: dict[str, dict] = {}
    for tool_name in tool_names:
        if tool_name == "web_search":
            capabilities[tool_name] = {
                "domains": ["web"],
                "retrieval_modes": ["search"],
                "preference_tags": ["open_web", "source_discovery"],
                "read_only": True,
                "open_world": True,
            }
        elif tool_name == "web_fetch":
            capabilities[tool_name] = {
                "domains": ["web"],
                "retrieval_modes": ["fetch"],
                "preference_tags": ["open_web", "source_reading"],
                "read_only": True,
                "open_world": True,
            }
        elif tool_name == "arxiv_search":
            capabilities[tool_name] = {
                "domains": ["literature"],
                "retrieval_modes": ["search"],
                "preference_tags": ["scholarly_index", "paper_metadata", "recent_papers"],
                "read_only": True,
                "open_world": True,
            }
        elif tool_name == "semantic_scholar_search":
            capabilities[tool_name] = {
                "domains": ["literature"],
                "retrieval_modes": ["search"],
                "preference_tags": ["scholarly_index", "paper_metadata", "citations", "recent_papers"],
                "read_only": True,
                "open_world": True,
            }
        elif tool_name == "crossref_search":
            capabilities[tool_name] = {
                "domains": ["literature"],
                "retrieval_modes": ["search"],
                "preference_tags": ["citation_lookup", "doi_lookup", "bibliographic_metadata"],
                "read_only": True,
                "open_world": True,
            }
        elif tool_name == "pubmed_search":
            capabilities[tool_name] = {
                "domains": ["literature"],
                "retrieval_modes": ["search"],
                "preference_tags": ["scholarly_index", "paper_metadata", "biomedical"],
                "read_only": True,
                "open_world": True,
            }
    return capabilities


class _FakeBackend(BaseLLMBackend):
    def __init__(self, backend_type: str = "openai_compat", metadata: dict | None = None) -> None:
        self._backend_type = backend_type
        self._metadata = metadata or {}

    async def generate(
        self,
        messages: list[Message],
        tools: list[ToolSchema] | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        top_p: float = 1.0,
        min_p: float = 0.0,
        top_k: int = 0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        repeat_penalty: float = 1.0,
        stream: bool = False,
    ) -> GenerationResult | AsyncIterator[StreamChunk]:
        del messages, tools, temperature, max_tokens, top_p, min_p, top_k
        del frequency_penalty, presence_penalty, repeat_penalty, stream
        return GenerationResult(content="")

    def supports_tool_calling(self) -> bool:
        return True

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(name="fake", backend_type=self._backend_type, metadata=dict(self._metadata))

    async def health_check(self) -> bool:
        return True


def test_tool_exposure_strict_mode_filters_risky_tools() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": ["file_read", "exec_command", "execute_code", "file_write"],
        }
    )
    plan = planner.plan(
        message="run command in project",
        available_tool_names=["file_read", "exec_command", "execute_code", "file_write"],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="strict",
    )
    assert plan.limit == 4
    assert "exec_command" not in plan.tool_names
    assert "execute_code" not in plan.tool_names
    assert plan.tool_names == ["file_read", "file_write"]


def test_tool_exposure_auto_review_limits_risky_count() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": ["file_read", "exec_command", "execute_code", "file_write", "process_stop"],
        }
    )
    plan = planner.plan(
        message="debug and run code",
        available_tool_names=["file_read", "exec_command", "execute_code", "file_write", "process_stop"],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
    )
    risky = {
        "exec_command",
        "execute_code",
        "file_write",
        "file_edit",
        "write_stdin",
        "kill_session",
        "process_stop",
        "mcp_call",
    }
    risky_selected = [name for name in plan.tool_names if name in risky]
    assert plan.limit == 8
    assert len(risky_selected) == 3
    assert plan.tool_names == ["file_read", "file_write", "exec_command", "execute_code"]


def test_tool_exposure_contextual_exec_session_tools_require_session_context() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": ["file_read", "exec_command", "read_session", "write_stdin", "kill_session"],
        }
    )
    base_tools = ["file_read", "exec_command", "read_session", "write_stdin", "kill_session"]

    no_session = planner.plan(
        message="run tests and inspect output",
        available_tool_names=base_tools,
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
    )
    with_session = planner.plan(
        message="use session_id and read_session to poll background command output",
        available_tool_names=base_tools,
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
    )

    assert "read_session" not in no_session.tool_names
    assert "write_stdin" not in no_session.tool_names
    assert "read_session" in with_session.tool_names


def test_tool_exposure_ignores_stale_preferred_tool_names() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "web": ["web_search"],
        }
    )
    plan = planner.plan(
        message="latest weather in Taipei",
        available_tool_names=["web_search"],
        backend=_FakeBackend(),
        session_bound_workspace=False,
        autonomy_mode="auto_review",
        preferred_tool_names=["file_read"],
    )
    assert plan.tool_names == ["web_search"]


def test_tool_exposure_uses_web_tools_for_weather_in_workspace() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "web": ["web_search", "web_fetch", "get_current_time", "calculator"],
            "workspace": ["file_read", "grep_search", "file_write"],
        }
    )
    plan = planner.plan(
        message="latest weather in Taichung",
        available_tool_names=[
            "file_read",
            "grep_search",
            "file_write",
            "web_search",
            "web_fetch",
            "get_current_time",
            "calculator",
        ],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
    )

    assert plan.matched_groups == ["web"]
    assert {"file_read", "grep_search", "web_search", "web_fetch", "get_current_time"} <= set(
        plan.tool_names
    )


def test_tool_exposure_co_exposes_literature_and_web_for_multilingual_paper_queries() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "web": ["web_search", "web_fetch"],
            "literature": [
                "arxiv_search",
                "semantic_scholar_search",
                "crossref_search",
                "pubmed_search",
            ],
            "workspace": ["file_read", "grep_search"],
        }
    )
    available_tools = [
        "file_read",
        "grep_search",
        "arxiv_search",
        "semantic_scholar_search",
        "crossref_search",
        "pubmed_search",
        "web_search",
        "web_fetch",
    ]
    plan = planner.plan(
        message="\u5e6b\u6211\u627e BERT \u8fd1\u5e7e\u5e74\u7684\u8ad6\u6587",
        available_tool_names=available_tools,
        backend=_FakeBackend(),
        session_bound_workspace=False,
        autonomy_mode="auto_review",
        tool_capabilities=_tool_capabilities(*available_tools),
    )

    assert plan.matched_groups == ["literature", "web"]
    assert set(plan.tool_names[:4]) == {
        "arxiv_search",
        "semantic_scholar_search",
        "crossref_search",
        "pubmed_search",
    }
    assert plan.tool_names.index("web_search") > 3
    assert plan.tool_names.index("web_fetch") > plan.tool_names.index("web_search")


def test_tool_exposure_prefers_crossref_for_indirect_doi_queries() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "web": ["web_search", "web_fetch"],
            "literature": ["arxiv_search", "semantic_scholar_search", "crossref_search", "pubmed_search"],
        }
    )
    available_tools = [
        "arxiv_search",
        "semantic_scholar_search",
        "crossref_search",
        "pubmed_search",
        "web_search",
        "web_fetch",
    ]
    plan = planner.plan(
        message="find metadata and references for 10.1038/nature12373",
        available_tool_names=available_tools,
        backend=_FakeBackend(),
        session_bound_workspace=False,
        autonomy_mode="auto_review",
        tool_capabilities=_tool_capabilities(*available_tools),
    )

    assert plan.tool_names[0] == "crossref_search"
    assert "web_search" in plan.tool_names
    assert "web_fetch" in plan.tool_names


def test_tool_exposure_prefers_pubmed_for_biomedical_paper_queries() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "web": ["web_search", "web_fetch"],
            "literature": ["arxiv_search", "semantic_scholar_search", "crossref_search", "pubmed_search"],
        }
    )
    available_tools = [
        "arxiv_search",
        "semantic_scholar_search",
        "crossref_search",
        "pubmed_search",
        "web_search",
        "web_fetch",
    ]
    plan = planner.plan(
        message="find recent biomedical transformer papers",
        available_tool_names=available_tools,
        backend=_FakeBackend(),
        session_bound_workspace=False,
        autonomy_mode="auto_review",
        tool_capabilities=_tool_capabilities(*available_tools),
    )

    assert plan.tool_names[0] == "pubmed_search"
    assert plan.tool_names.index("pubmed_search") < plan.tool_names.index("web_search")


def test_tool_exposure_includes_workspace_search_tools_for_find_queries() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": ["glob_search", "grep_search", "file_read", "file_write"],
        }
    )
    plan = planner.plan(
        message="find matching files and search for TODO in the repo",
        available_tool_names=["glob_search", "grep_search", "file_read", "file_write"],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
    )
    assert "glob_search" in plan.tool_names
    assert "grep_search" in plan.tool_names


def test_tool_exposure_includes_workspace_baseline_for_chinese_workspace_prompt() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": ["glob_search", "grep_search", "file_read", "file_write"],
        }
    )
    message = "請檢查目前工作區，找出包含 TODO 的地方，並查看相關內容"

    plan = planner.plan(
        message=message,
        available_tool_names=["glob_search", "grep_search", "file_read", "file_write"],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
    )

    assert "read" not in message.lower()
    assert "file" not in message.lower()
    assert {"file_read", "glob_search", "grep_search"} <= set(plan.tool_names)


def test_tool_exposure_includes_specialized_workspace_readers_for_chinese_prompt() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": ["file_read", "glob_search", "grep_search", "pdf_read", "csv_read", "docx_read", "notebook_read"],
        }
    )

    plan = planner.plan(
        message="請整理這個工作區的附件與文件內容，並核對資料表和筆記本輸出",
        available_tool_names=[
            "file_read",
            "glob_search",
            "grep_search",
            "pdf_read",
            "csv_read",
            "docx_read",
            "notebook_read",
        ],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
    )

    assert {"file_read", "glob_search", "grep_search", "pdf_read", "csv_read", "docx_read", "notebook_read"} <= set(
        plan.tool_names
    )


def test_tool_exposure_keeps_repo_queries_on_workspace_tools_without_open_world_leakage() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": ["glob_search", "grep_search", "file_read", "file_write"],
            "web": ["web_search", "web_fetch"],
            "literature": ["arxiv_search", "semantic_scholar_search"],
        }
    )
    available_tools = [
        "glob_search",
        "grep_search",
        "file_read",
        "file_write",
        "arxiv_search",
        "semantic_scholar_search",
        "web_search",
        "web_fetch",
    ]
    plan = planner.plan(
        message="find matching files and search for TODO in the repo",
        available_tool_names=available_tools,
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
        tool_capabilities=_tool_capabilities(*available_tools),
    )

    assert plan.matched_groups == ["workspace"]
    assert "glob_search" in plan.tool_names
    assert "grep_search" in plan.tool_names
    assert "web_search" not in plan.tool_names
    assert "arxiv_search" not in plan.tool_names


def test_tool_exposure_prioritizes_tool_search_for_tool_selection_queries() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": [
                "glob_search",
                "grep_search",
                "file_read",
                "file_write",
                "exec_command",
                "execute_code",
                "tool_search",
            ],
        }
    )
    plan = planner.plan(
        message="which tool should I use to search repo files and inspect a specific web page?",
        available_tool_names=[
            "glob_search",
            "grep_search",
            "file_read",
            "file_write",
            "exec_command",
            "execute_code",
            "tool_search",
            "memory_search",
            "web_search",
        ],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
    )
    assert "tool_search" in plan.tool_names
    assert plan.tool_names.index("tool_search") < plan.tool_names.index("memory_search")


def test_tool_exposure_prioritizes_specialized_readers_for_matching_file_types() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": [
                "glob_search",
                "grep_search",
                "file_read",
                "pdf_read",
                "csv_read",
                "notebook_read",
                "file_write",
                "exec_command",
                "execute_code",
            ],
        }
    )
    plan = planner.plan(
        message="read a pdf report, inspect a csv export, and review notebook outputs in the repo",
        available_tool_names=[
            "glob_search",
            "grep_search",
            "file_read",
            "file_write",
            "exec_command",
            "execute_code",
            "memory_search",
            "web_search",
            "pdf_read",
            "csv_read",
            "notebook_read",
        ],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
    )
    assert "pdf_read" in plan.tool_names
    assert "csv_read" in plan.tool_names
    assert "notebook_read" in plan.tool_names
    assert plan.tool_names.index("pdf_read") < plan.tool_names.index("file_write")


def test_tool_exposure_attached_workspace_reads_skip_risky_tools() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": [
                "exec_command",
                "file_read",
                "docx_read",
                "pdf_read",
                "file_write",
                "file_edit",
                "execute_code",
            ],
        }
    )
    plan = planner.plan(
        message=(
            "請幫我統整這份檔案重點。\n"
            "Attached workspace files:\n"
            "- .mochi/workspace/browser-imports/report.docx (report.docx)\n"
            "Use the appropriate file-reading tools if you need to inspect them before answering."
        ),
        available_tool_names=[
            "exec_command",
            "file_read",
            "docx_read",
            "pdf_read",
            "file_write",
            "file_edit",
            "execute_code",
        ],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="trusted_workspace",
    )
    assert {"file_read", "pdf_read", "docx_read"} <= set(plan.tool_names)
    assert not {"exec_command", "file_write", "file_edit", "execute_code"} & set(plan.tool_names)


def test_tool_exposure_attachment_bias_works_with_engine_structured_attachment_header() -> None:
    from mochi.agents.engine import AgentEngine
    from mochi.backends.types import AttachmentRef
    from mochi.config.schema import MochiConfig

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": ".",
            "sessions_dir": "./.tmp-test-sessions",
            "memory": {"db_path": "./.tmp-test-memory.db", "fts_top_k": 3},
        }
    )
    engine = AgentEngine(config)
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": [
                "exec_command",
                "file_read",
                "docx_read",
                "pdf_read",
                "file_write",
                "file_edit",
                "execute_code",
            ],
        }
    )
    planner_message = engine._build_tool_planner_message(  # noqa: SLF001
        "請先檢查附件內容再回答",
        [
            AttachmentRef(
                name="report.docx",
                path=".mochi/workspace/browser-imports/report.docx",
                source="workspace_file",
            )
        ],
    )

    plan = planner.plan(
        message=planner_message,
        available_tool_names=[
            "exec_command",
            "file_read",
            "docx_read",
            "pdf_read",
            "file_write",
            "file_edit",
            "execute_code",
        ],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="trusted_workspace",
    )

    assert "Structured attachments:" in planner_message
    assert {"file_read", "pdf_read", "docx_read"} <= set(plan.tool_names)
    assert not {"exec_command", "file_write", "file_edit", "execute_code"} & set(plan.tool_names)


def test_tool_exposure_file_browse_requests_skip_exec() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": [
                "glob_search",
                "grep_search",
                "file_read",
                "pdf_read",
                "exec_command",
            ],
        }
    )
    plan = planner.plan(
        message="browse the repo, find matching files, search for TODO, and inspect a pdf in the workspace",
        available_tool_names=[
            "glob_search",
            "grep_search",
            "file_read",
            "pdf_read",
            "exec_command",
        ],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
    )

    assert "glob_search" in plan.tool_names
    assert "grep_search" in plan.tool_names
    assert "pdf_read" in plan.tool_names
    assert "exec_command" not in plan.tool_names


def test_tool_exposure_disabled_mode_returns_empty_plan() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": ["file_read", "exec_command"],
        }
    )
    plan = planner.plan(
        message="debug the repo",
        available_tool_names=["file_read", "exec_command"],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
        tool_mode="disabled",
    )
    assert plan.tool_names == []
    assert plan.limit == 0


def test_tool_exposure_blocks_tools_when_backend_marks_tool_calling_unavailable() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "web": ["web_search", "web_fetch"],
            "workspace": ["file_read", "exec_command"],
        }
    )
    plan = planner.plan(
        message="latest weather in Taichung",
        available_tool_names=["web_search", "web_fetch", "file_read", "exec_command"],
        backend=_FakeBackend(
            metadata={
                "tool_call_mode": "unavailable",
                "tool_calling_blocked": True,
            }
        ),
        session_bound_workspace=False,
        autonomy_mode="auto_review",
    )
    assert plan.tool_names == []
    assert plan.limit == 0
