from __future__ import annotations

from mochi.agents.prompt_builder import PromptBuilder
from mochi.agents.tool_exposure import ToolExposurePlanner
from mochi.config.schema import MochiConfig
from mochi.memory.store import MemoryStore
from mochi.tools.registry import ToolRegistry
from mochi.tools.registry_factory import ToolRegistryFactory
from mochi.tools.tool_search import ToolSearchTool
from tests.unit.tool_exposure._support import (
    _DummyTool,
    _FakeBackend,
    _tool_capabilities,
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
    assert {"glob_search", "grep_search", "file_read", "file_write"} <= set(plan.tool_names)
    assert "web_search" not in plan.tool_names
    assert "web_fetch" not in plan.tool_names
    assert "arxiv_search" not in plan.tool_names
    assert "semantic_scholar_search" not in plan.tool_names



def test_tool_exposure_uses_group_signals_for_ranking_without_hiding_other_grouped_tools() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": ["file_read", "glob_search", "grep_search"],
            "web": ["web_search", "web_fetch", "get_current_time"],
            "literature": ["arxiv_search"],
        }
    )
    available_tools = [
        "file_read",
        "glob_search",
        "grep_search",
        "web_search",
        "web_fetch",
        "get_current_time",
        "arxiv_search",
    ]
    plan = planner.plan(
        message="latest weather in Taipei",
        available_tool_names=available_tools,
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
        tool_capabilities=_tool_capabilities(*available_tools),
    )

    assert {"web_search", "web_fetch", "get_current_time", "arxiv_search"} <= set(plan.tool_names)
    assert not {"file_read", "glob_search", "grep_search"} & set(plan.tool_names)
    assert plan.tool_names.index("web_fetch") < plan.tool_names.index("arxiv_search")



def test_tool_exposure_non_workspace_attachments_do_not_bias_workspace_readers(tmp_path) -> None:
    from mochi.agents.engine import AgentEngine
    from mochi.backends.types import AttachmentRef
    from mochi.config.schema import MochiConfig

    config = MochiConfig.model_validate(
        {
            "model": "ollama:test",
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db"), "fts_top_k": 3},
        }
    )
    engine = AgentEngine(config)
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": ["file_read", "pdf_read", "docx_read"],
            "web": ["web_search"],
        }
    )
    planner_message = engine._build_tool_planner_message(  # noqa: SLF001
        "Summarize the attached image.",
        [
            AttachmentRef(
                name="error.png",
                path="error.png",
                source="image",
            )
        ],
    )

    plan = planner.plan(
        message=planner_message,
        user_intent_message="Summarize the attached image.",
        available_tool_names=["file_read", "pdf_read", "docx_read", "web_search"],
        backend=_FakeBackend(),
        session_bound_workspace=False,
        autonomy_mode="auto_review",
        attachment_count=1,
        workspace_attachment_count=0,
    )

    assert "Structured attachments:" in planner_message
    assert "file_read" not in plan.tool_names



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



def test_tool_exposure_large_tool_sets_keep_workspace_baseline_and_tool_search() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": [
                "file_read",
                "glob_search",
                "grep_search",
                "csv_read",
                "pdf_read",
                "docx_read",
                "notebook_read",
                "file_write",
                "exec_command",
                "tool_search",
            ],
            "web": ["web_search", "web_fetch"],
        }
    )
    available_tools = [
        "file_read",
        "glob_search",
        "grep_search",
        "csv_read",
        "pdf_read",
        "docx_read",
        "notebook_read",
        "file_write",
        "exec_command",
        "tool_search",
        "web_search",
        "web_fetch",
        "memory_search",
    ]
    plan = planner.plan(
        message="inspect the repo structure and review local files",
        available_tool_names=available_tools,
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
    )

    assert {
        "file_read",
        "glob_search",
        "grep_search",
        "csv_read",
        "pdf_read",
        "docx_read",
        "notebook_read",
    } <= set(plan.tool_names)
    assert "tool_search" in plan.tool_names



def test_tool_exposure_truncation_below_previous_large_threshold_still_exposes_tool_search() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": [
                "file_read",
                "glob_search",
                "grep_search",
                "csv_read",
                "pdf_read",
                "docx_read",
                "notebook_read",
                "tool_search",
                "memory_search",
            ],
        }
    )
    available_tools = [
        "file_read",
        "glob_search",
        "grep_search",
        "csv_read",
        "pdf_read",
        "docx_read",
        "notebook_read",
        "tool_search",
        "memory_search",
    ]
    plan = planner.plan(
        message="inspect repo files and summarize matching documents",
        available_tool_names=available_tools,
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
    )

    assert len(available_tools) == 9
    assert plan.limit == 8
    assert "tool_search" in plan.tool_names
    assert "memory_search" not in plan.tool_names
    assert "memory_search" in plan.discoverable_tool_names



def test_tool_exposure_large_tool_sets_keep_literature_ranking_and_add_tool_search() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": ["file_read", "grep_search"],
            "web": ["web_search", "web_fetch"],
            "literature": [
                "arxiv_search",
                "semantic_scholar_search",
                "crossref_search",
                "pubmed_search",
            ],
        }
    )
    available_tools = [
        "file_read",
        "grep_search",
        "tool_search",
        "arxiv_search",
        "semantic_scholar_search",
        "crossref_search",
        "pubmed_search",
        "web_search",
        "web_fetch",
        "memory_search",
        "calculator",
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
    assert "tool_search" in plan.tool_names
    assert plan.tool_names.index("web_search") > plan.tool_names.index("pubmed_search")



async def test_tool_search_registry_view_only_returns_exposed_tools() -> None:
    registry = ToolRegistry(discover_builtin=False)
    registry.register(_DummyTool("visible_reader", "Read visible repo files", search_hint="read repo files"))
    registry.register(
        _DummyTool("hidden_mutation", "Dangerous hidden mutation tool", search_hint="mutate hidden files")
    )
    registry.register(ToolSearchTool(catalog_provider=registry.list_tools))

    view = registry.create_view(["visible_reader", "tool_search"])
    result = await view.execute("tool_search", {"query": "hidden mutation", "top_k": 10})

    assert result.error is None
    assert isinstance(result.output, list)
    names = {item["name"] for item in result.output if isinstance(item, dict) and "name" in item}
    assert names <= {"visible_reader", "tool_search"}
    assert "hidden_mutation" not in names



async def test_tool_search_registry_view_can_discover_turn_allowed_hidden_tools_only() -> None:
    registry = ToolRegistry(discover_builtin=False)
    registry.register(_DummyTool("visible_reader", "Read visible repo files", search_hint="read repo files"))
    registry.register(
        _DummyTool("hidden_allowed_reader", "Specialized allowed reader", search_hint="read notebook output")
    )
    registry.register(
        _DummyTool("blocked_hidden_mutation", "Dangerous hidden mutation tool", search_hint="mutate hidden files")
    )
    registry.register(ToolSearchTool(catalog_provider=registry.list_tools))

    view = registry.create_view(
        ["visible_reader", "tool_search"],
        tool_search_catalog_names=["visible_reader", "hidden_allowed_reader", "tool_search"],
    )
    result = await view.execute("tool_search", {"query": "read notebook output", "top_k": 10})

    assert result.error is None
    assert isinstance(result.output, list)
    names = {item["name"] for item in result.output if isinstance(item, dict) and "name" in item}
    assert "hidden_allowed_reader" in names
    assert "blocked_hidden_mutation" not in names



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



def test_tool_exposure_workspace_group_lists_and_materializes_repo_navigation_tools(tmp_path) -> None:
    config = MochiConfig.model_validate(
        {
            "workspace_dir": str(tmp_path),
            "sessions_dir": str(tmp_path / "sessions"),
            "memory": {"db_path": str(tmp_path / "memory.db")},
        }
    )
    factory = ToolRegistryFactory(
        config,
        memory_store=MemoryStore(db_path=tmp_path / "memory.db"),
    )
    registry = factory.create_registry(str(tmp_path))

    assert "repo_map" in factory.tool_groups["workspace"]
    assert "read_symbol" in factory.tool_groups["workspace"]
    assert {"file_read", "glob_search", "grep_search"} <= set(factory.tool_groups["workspace"])
    assert "discourse_topic_collect" in factory.tool_groups["collector"]
    assert registry.get("repo_map") is not None
    assert registry.get("read_symbol") is not None
    assert registry.get("discourse_topic_collect") is not None



def test_tool_exposure_repo_orientation_queries_surface_repo_map_without_dropping_baseline() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": [
                "file_read",
                "glob_search",
                "grep_search",
                "repo_map",
                "read_symbol",
                "file_write",
            ],
        }
    )
    available_tools = [
        "file_read",
        "glob_search",
        "grep_search",
        "repo_map",
        "read_symbol",
        "file_write",
    ]
    plan = planner.plan(
        message="use a repo map to orient me in this larger repo before opening concrete files",
        available_tool_names=available_tools,
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
        tool_capabilities=_tool_capabilities(*available_tools),
    )

    assert {"file_read", "glob_search", "grep_search", "repo_map"} <= set(plan.tool_names)
    assert plan.tool_names.index("repo_map") < plan.tool_names.index("file_write")



def test_tool_exposure_symbol_lookup_queries_surface_read_symbol_without_dropping_file_read() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": [
                "file_read",
                "glob_search",
                "grep_search",
                "read_symbol",
                "file_write",
            ],
        }
    )
    available_tools = [
        "file_read",
        "glob_search",
        "grep_search",
        "read_symbol",
        "file_write",
    ]
    plan = planner.plan(
        message="inspect the Router class definition in the workspace",
        available_tool_names=available_tools,
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
        tool_capabilities=_tool_capabilities(*available_tools),
    )

    assert {"file_read", "read_symbol"} <= set(plan.tool_names)
    assert plan.tool_names.index("read_symbol") < plan.tool_names.index("file_write")



def test_prompt_builder_guides_workspace_reads_and_tool_search() -> None:
    prompt = PromptBuilder(base_system_prompt="").build_system_prompt(task_workspace_dir="workspace")

    assert "Use these core workspace read tools directly when visible." in prompt
    assert "Use `repo_map` to orient in larger repos when needed" in prompt
    assert "`read_symbol` for targeted symbol inspection" in prompt
    assert "Continue using the normal read tools for concrete file content." in prompt
    assert "use `tool_search` to discover the right tool instead of guessing" in prompt
