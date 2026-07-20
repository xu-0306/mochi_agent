from __future__ import annotations

import pytest

from mochi.agents.tool_exposure import ToolExposurePlanner
from tests.unit.tool_exposure._support import (
    _FakeBackend,
    _tool_capabilities,
)


def test_tool_exposure_uses_routed_open_world_intent_for_chinese_weather_in_workspace() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": ["file_read", "glob_search", "grep_search", "file_write"],
            "web": ["web_search", "web_fetch", "get_current_time"],
        }
    )

    plan = planner.plan(
        message="\u5e6b\u6211\u67e5\u8a62\u53f0\u4e2d\u660e\u5929\u5929\u6c23",
        user_intent_message="\u5e6b\u6211\u67e5\u8a62\u53f0\u4e2d\u660e\u5929\u5929\u6c23",
        available_tool_names=[
            "file_read",
            "glob_search",
            "grep_search",
            "file_write",
            "web_search",
            "web_fetch",
            "get_current_time",
        ],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
        tool_capabilities=_tool_capabilities("web_search", "web_fetch"),
        routed_intent="open_world_lookup",
        intent_confidence=0.88,
        intent_source="fallback_keyword",
        intent_rationale="Matched open-world weather language.",
    )

    assert plan.matched_groups == ["web"]
    assert {"web_search", "web_fetch", "get_current_time"} <= set(plan.tool_names)
    assert plan.exposure_metadata()["intent_route"]["intent"] == "open_world_lookup"



def test_tool_exposure_uses_routed_literature_intent_for_chinese_research_in_workspace() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": ["file_read", "glob_search", "grep_search"],
            "web": ["web_search", "web_fetch", "get_current_time"],
            "literature": ["arxiv_search", "semantic_scholar_search", "crossref_search", "pubmed_search"],
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
        "semantic_scholar_search",
        "crossref_search",
        "pubmed_search",
    ]

    plan = planner.plan(
        message="\u5e6b\u6211\u7814\u7a76 ESG \u548c LLM \u7684\u95dc\u4fc2",
        user_intent_message="\u5e6b\u6211\u7814\u7a76 ESG \u548c LLM \u7684\u95dc\u4fc2",
        available_tool_names=available_tools,
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
        tool_capabilities=_tool_capabilities(*available_tools),
        routed_intent="literature_research",
        intent_confidence=0.9,
        intent_source="fallback_keyword",
        intent_rationale="Matched literature research language.",
    )

    assert plan.matched_groups == ["literature", "web"]
    assert {"web_search", "web_fetch"} <= set(plan.tool_names)
    assert (
        {"arxiv_search", "semantic_scholar_search", "crossref_search", "pubmed_search"} & set(plan.tool_names)
    )
    assert not {"file_read", "glob_search", "grep_search"} & set(plan.tool_names)



def test_tool_exposure_literature_request_does_not_add_workspace_baseline() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": ["file_read", "glob_search", "grep_search", "csv_read", "pdf_read"],
            "web": ["web_search", "web_fetch", "get_current_time"],
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
        "get_current_time",
        "file_read",
        "glob_search",
        "grep_search",
        "csv_read",
        "pdf_read",
        "tool_search",
    ]

    plan = planner.plan(
        message="幫我查詢小型多模態模型醫療影像微調的相關論文",
        user_intent_message="幫我查詢小型多模態模型醫療影像微調的相關論文",
        available_tool_names=available_tools,
        backend=_FakeBackend(backend_type="ollama"),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
        tool_capabilities=_tool_capabilities(*available_tools),
        routed_intent="literature_research",
        intent_confidence=0.88,
        intent_source="fallback_keyword",
        intent_rationale="Matched literature research language.",
    )

    assert plan.matched_groups == ["literature", "web"]
    assert {"arxiv_search", "semantic_scholar_search", "crossref_search", "pubmed_search"} & set(
        plan.tool_names
    )
    assert {"web_search", "web_fetch"} <= set(plan.tool_names)
    assert "tool_search" in plan.tool_names
    assert not {"file_read", "glob_search", "grep_search", "csv_read", "pdf_read"} & set(
        plan.tool_names
    )



def test_tool_exposure_workspace_readonly_baseline_keeps_tool_result_read() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": ["file_read", "tool_result_read", "glob_search", "grep_search", "file_write"],
        }
    )

    plan = planner.plan(
        message="inspect the repo and continue reading prior tool output if needed",
        user_intent_message="inspect the repo and continue reading prior tool output if needed",
        available_tool_names=["file_read", "tool_result_read", "glob_search", "grep_search", "file_write"],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
        tool_capabilities=_tool_capabilities(),
        routed_intent="workspace_read",
        intent_confidence=0.91,
        intent_source="classifier",
        intent_rationale="Explicit workspace inspection.",
    )

    assert {"file_read", "tool_result_read", "glob_search", "grep_search"} <= set(plan.tool_names)
    assert "file_write" not in plan.tool_names
    assert "file_write" in plan.discoverable_tool_names





def test_tool_exposure_routed_workspace_write_exposes_write_tools_without_session_binding() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": [
                "repo_map",
                "read_symbol",
                "glob_search",
                "grep_search",
                "file_read",
                "tool_search",
                "file_write",
                "file_edit",
                "apply_patch",
            ],
        }
    )

    plan = planner.plan(
        message="create report.md",
        user_intent_message="create report.md",
        available_tool_names=[
            "repo_map",
            "read_symbol",
            "glob_search",
            "grep_search",
            "file_read",
            "tool_search",
            "file_write",
            "file_edit",
            "apply_patch",
        ],
        backend=_FakeBackend(),
        session_bound_workspace=False,
        autonomy_mode="trusted_workspace",
        routed_intent="workspace_write",
        intent_confidence=0.99,
        intent_source="classifier",
        intent_rationale="The user is asking to create a workspace file.",
    )

    assert {"file_write", "file_edit", "apply_patch"} <= set(plan.tool_names)
    assert {"file_write", "file_edit", "apply_patch"} <= set(plan.discoverable_tool_names)



def test_tool_exposure_routed_workspace_write_exposes_write_tools_from_intent_not_phrase() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": [
                "repo_map",
                "read_symbol",
                "glob_search",
                "grep_search",
                "file_read",
                "tool_search",
                "file_write",
                "file_edit",
                "apply_patch",
            ],
        }
    )

    plan = planner.plan(
        message="produce an artifact",
        user_intent_message="produce an artifact",
        available_tool_names=[
            "repo_map",
            "read_symbol",
            "glob_search",
            "grep_search",
            "file_read",
            "tool_search",
            "file_write",
            "file_edit",
            "apply_patch",
        ],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="trusted_workspace",
        routed_intent="workspace_write",
        intent_confidence=0.93,
        intent_source="classifier",
        intent_rationale="The user asked for a workspace artifact to be saved.",
    )

    assert {"file_write", "file_edit", "apply_patch"} <= set(plan.tool_names)
    assert plan.exposure_metadata()["intent_route"]["source"] == "classifier"


def test_tool_exposure_workspace_code_creation_save_exposes_write_tools() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": [
                "file_read",
                "glob_search",
                "grep_search",
                "file_write",
                "file_edit",
                "apply_patch",
            ],
        }
    )

    plan = planner.plan(
        message="create a training script and save it in the workspace",
        user_intent_message="create a training script and save it in the workspace",
        available_tool_names=[
            "file_read",
            "glob_search",
            "grep_search",
            "file_write",
            "file_edit",
            "apply_patch",
        ],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="strict",
        routed_intent="workspace_write",
        intent_confidence=0.92,
        intent_source="classifier",
        intent_rationale="Explicit code creation and save request.",
    )

    assert {"file_write", "file_edit", "apply_patch"} <= set(plan.tool_names)
    assert {"file_write", "apply_patch"} <= set(plan.discoverable_tool_names)



def test_tool_exposure_workspace_greeting_keeps_write_tools_discoverable() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": ["file_read", "glob_search", "grep_search", "file_write", "apply_patch"],
        }
    )

    plan = planner.plan(
        message="hello",
        user_intent_message="hello",
        available_tool_names=["file_read", "glob_search", "grep_search", "file_write", "apply_patch"],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="strict",
    )

    assert {"file_write", "apply_patch"} <= set(plan.discoverable_tool_names)



def test_tool_exposure_routed_workspace_read_stays_workspace_focused() -> None:
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
        user_intent_message="find matching files and search for TODO in the repo",
        available_tool_names=available_tools,
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
        tool_capabilities=_tool_capabilities(*available_tools),
        routed_intent="workspace_read",
        intent_confidence=0.92,
        intent_source="classifier",
        intent_rationale="Repo inspection intent is explicit.",
    )

    assert plan.matched_groups == ["workspace"]
    assert {"glob_search", "grep_search", "file_read"} <= set(plan.tool_names)
    assert "file_write" not in plan.tool_names
    assert "file_write" in plan.discoverable_tool_names
    assert "web_search" not in plan.tool_names
    assert "semantic_scholar_search" not in plan.tool_names
    assert plan.exposure_metadata()["intent_route"]["intent"] == "workspace_read"



def test_tool_exposure_workspace_read_inspect_repository_hides_write_tools() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": [
                "file_read",
                "glob_search",
                "grep_search",
                "file_write",
                "file_edit",
                "apply_patch",
            ],
        }
    )
    write_tools = {"file_write", "file_edit", "apply_patch"}
    plan = planner.plan(
        message="inspect the repository",
        user_intent_message="inspect the repository",
        available_tool_names=[
            "file_read",
            "glob_search",
            "grep_search",
            "file_write",
            "file_edit",
            "apply_patch",
        ],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="trusted_workspace",
        tool_capabilities=_tool_capabilities(
            "file_read",
            "glob_search",
            "grep_search",
            "file_write",
            "file_edit",
            "apply_patch",
        ),
        routed_intent="workspace_read",
        intent_confidence=0.95,
        intent_source="classifier",
        intent_rationale="The user asks for repository inspection only.",
    )

    assert not write_tools & set(plan.tool_names)
    assert write_tools <= set(plan.discoverable_tool_names)



@pytest.mark.parametrize(
    "message",
    [
        "Update me on the latest weather in Taipei",
        "請更新台北最新天氣",
        "请更新台北最新天气",
    ],
)
def test_tool_exposure_open_world_weather_hides_write_tools(message: str) -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "web": ["web_search", "web_fetch"],
            "workspace": ["file_read", "file_write", "file_edit", "apply_patch"],
        }
    )
    write_tools = {"file_write", "file_edit", "apply_patch"}
    available_tools = [
        "web_search",
        "web_fetch",
        "file_read",
        "file_write",
        "file_edit",
        "apply_patch",
    ]
    plan = planner.plan(
        message=message,
        user_intent_message=message,
        available_tool_names=available_tools,
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="trusted_workspace",
        tool_capabilities=_tool_capabilities(*available_tools),
        routed_intent="open_world_lookup",
        intent_confidence=0.95,
        intent_source="classifier",
        intent_rationale="The user asks for current weather information.",
    )

    assert {"web_search", "web_fetch"} <= set(plan.tool_names)
    assert not write_tools & set(plan.tool_names)
    assert write_tools <= set(plan.discoverable_tool_names)



def test_tool_exposure_normalizes_legacy_workspace_read_alias_in_metadata() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": ["glob_search", "grep_search", "file_read", "file_write"],
            "web": ["web_search", "web_fetch"],
        }
    )

    plan = planner.plan(
        message="find matching files and search for TODO in the repo",
        user_intent_message="find matching files and search for TODO in the repo",
        available_tool_names=[
            "glob_search",
            "grep_search",
            "file_read",
            "file_write",
            "web_search",
            "web_fetch",
        ],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
        tool_capabilities=_tool_capabilities(
            "glob_search",
            "grep_search",
            "file_read",
            "file_write",
            "web_search",
            "web_fetch",
        ),
        routed_intent="workspace_inspection",
        intent_confidence=0.92,
        intent_source="classifier",
        intent_rationale="Legacy alias should normalize to workspace_read.",
    )

    assert plan.exposure_metadata()["intent_route"]["intent"] == "workspace_read"



def test_tool_exposure_routed_tool_discovery_prefers_tool_search() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": ["file_read", "glob_search", "grep_search"],
            "web": ["web_search", "web_fetch"],
        }
    )
    available_tools = [
        "file_read",
        "glob_search",
        "grep_search",
        "tool_search",
        "web_search",
        "web_fetch",
        "memory_search",
    ]

    plan = planner.plan(
        message="tell me which tool I should use to inspect notebook outputs",
        user_intent_message="tell me which tool I should use to inspect notebook outputs",
        available_tool_names=available_tools,
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
        routed_intent="tool_discovery",
        intent_confidence=0.87,
        intent_source="fallback_keyword",
        intent_rationale="Matched tool-discovery language.",
    )

    assert "tool_search" in plan.tool_names
    assert plan.exposure_metadata()["intent_route"]["source"] == "fallback_keyword"



def test_tool_exposure_tool_discovery_keeps_write_tools_discoverable_not_callable() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": ["tool_search", "file_read", "file_write"],
        }
    )

    plan = planner.plan(
        message="你有沒有保存檔案的工具？",
        user_intent_message="你有沒有保存檔案的工具？",
        available_tool_names=["tool_search", "file_read", "file_write"],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="trusted_workspace",
        routed_intent="tool_discovery",
        intent_confidence=0.9,
        intent_source="classifier",
        intent_rationale="The user is asking whether a save-file tool exists.",
    )

    assert "tool_search" in plan.tool_names
    assert "file_write" not in plan.tool_names
    assert "file_write" in plan.discoverable_tool_names



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



def test_tool_exposure_includes_delegate_subagent_for_explicit_subagent_request() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": ["file_read", "delegate_subagent_task", "glob_search", "grep_search"],
            "web": ["web_search", "web_fetch"],
        }
    )

    plan = planner.plan(
        message="Use two subagents to research A and B independently, then compare them.",
        user_intent_message="Use two subagents to research A and B independently, then compare them.",
        available_tool_names=[
            "file_read",
            "glob_search",
            "grep_search",
            "delegate_subagent_task",
            "web_search",
            "web_fetch",
        ],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
        tool_capabilities=_tool_capabilities("web_search", "web_fetch"),
        routed_intent="workspace_read",
        intent_confidence=0.78,
        intent_source="fallback_keyword",
        intent_rationale="Explicit subagent delegation request.",
    )

    assert "delegate_subagent_task" in plan.tool_names



def test_tool_exposure_includes_delegate_subagent_for_explicit_chinese_request() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": ["file_read", "delegate_subagent_task", "glob_search", "grep_search"],
            "web": ["web_search", "web_fetch"],
        }
    )

    message = (
        "\u8acb\u958b\u5169\u500b\u5b50\u4ee3\u7406\u5206\u982d\u7814\u7a76 A \u8ddf B\uff0c"
        "\u6700\u5f8c\u4ea4\u7d66\u4e3b\u4ee3\u7406\u6bd4\u8f03\u5dee\u7570\u3002"
    )
    plan = planner.plan(
        message=message,
        user_intent_message=message,
        available_tool_names=[
            "file_read",
            "glob_search",
            "grep_search",
            "delegate_subagent_task",
            "web_search",
            "web_fetch",
        ],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
        tool_capabilities=_tool_capabilities("web_search", "web_fetch"),
        routed_intent="workspace_read",
        intent_confidence=0.8,
        intent_source="fallback_keyword",
        intent_rationale="Explicit Chinese subagent delegation request.",
    )

    assert "delegate_subagent_task" in plan.tool_names



def test_tool_exposure_skips_delegate_subagent_for_trivial_chat() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": ["file_read", "delegate_subagent_task", "glob_search", "grep_search"],
            "web": ["web_search", "web_fetch", "get_current_time"],
        }
    )

    plan = planner.plan(
        message="What time is it in Taipei?",
        user_intent_message="What time is it in Taipei?",
        available_tool_names=[
            "file_read",
            "glob_search",
            "grep_search",
            "delegate_subagent_task",
            "web_search",
            "web_fetch",
            "get_current_time",
        ],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
        tool_capabilities=_tool_capabilities("web_search", "web_fetch"),
        routed_intent="open_world_lookup",
        intent_confidence=0.91,
        intent_source="fallback_keyword",
        intent_rationale="Simple current-time lookup.",
    )

    assert "delegate_subagent_task" not in plan.tool_names



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
    assert {"web_search", "web_fetch", "get_current_time"} <= set(plan.tool_names)
    assert not {"file_read", "grep_search"} & set(plan.tool_names)



def test_tool_exposure_keeps_weather_tools_without_workspace_baseline() -> None:
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
            ],
            "web": ["web_search", "web_fetch", "get_current_time"],
        }
    )
    plan = planner.plan(
        message="latest weather in Taichung",
        available_tool_names=[
            "file_read",
            "glob_search",
            "grep_search",
            "csv_read",
            "pdf_read",
            "docx_read",
            "notebook_read",
            "file_write",
            "web_search",
            "web_fetch",
            "get_current_time",
        ],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
    )

    assert {"web_search", "web_fetch", "get_current_time"} <= set(plan.tool_names)
    assert not {
        "file_read",
        "glob_search",
        "grep_search",
        "csv_read",
        "pdf_read",
        "docx_read",
        "notebook_read",
    } & set(plan.tool_names)



def test_tool_exposure_surfaces_discourse_collector_for_direct_topic_collection_requests() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "web": ["web_search", "web_fetch", "discourse_topic_collect"],
            "workspace": ["file_read", "grep_search"],
        }
    )
    available_tools = [
        "file_read",
        "grep_search",
        "web_search",
        "web_fetch",
        "discourse_topic_collect",
    ]
    plan = planner.plan(
        message="Collect https://forum.example/t/api-examples/274354 into dataset records with shard progress.",
        available_tool_names=available_tools,
        backend=_FakeBackend(),
        session_bound_workspace=False,
        autonomy_mode="auto_review",
        tool_capabilities={
            "web_search": _tool_capabilities("web_search")["web_search"],
            "web_fetch": {
                "domains": ["web"],
                "retrieval_modes": ["fetch"],
                "preference_tags": ["open_web", "source_reading"],
                "read_only": True,
                "open_world": True,
            },
            "discourse_topic_collect": {
                "domains": ["web"],
                "retrieval_modes": ["fetch"],
                "preference_tags": [
                    "open_web",
                    "structured_source_api",
                    "forum_thread",
                    "dataset_collection",
                    "source_capture",
                ],
                "read_only": True,
                "open_world": True,
            },
        },
    )

    assert "discourse_topic_collect" in plan.tool_names
    assert plan.tool_names.index("discourse_topic_collect") < plan.tool_names.index("web_fetch")



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



def test_tool_exposure_keeps_web_focus_under_web_heuristics() -> None:
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
            ],
            "web": ["web_search", "web_fetch", "get_current_time"],
        }
    )
    plan = planner.plan(
        message="latest weather in Taichung",
        available_tool_names=[
            "file_read",
            "glob_search",
            "grep_search",
            "csv_read",
            "pdf_read",
            "docx_read",
            "notebook_read",
            "file_write",
            "web_search",
            "web_fetch",
            "get_current_time",
        ],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
    )

    assert {"web_search", "web_fetch", "get_current_time"} <= set(plan.tool_names)
    assert not {
        "file_read",
        "glob_search",
        "grep_search",
        "csv_read",
        "pdf_read",
        "docx_read",
        "notebook_read",
    } & set(plan.tool_names)



def test_tool_exposure_keeps_general_web_tools_for_chinese_weather_queries_in_workspace() -> None:
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
            ],
            "web": ["web_search", "web_fetch", "get_current_time"],
        }
    )
    plan = planner.plan(
        message="幫我查詢台中明天天氣",
        user_intent_message="幫我查詢台中明天天氣",
        available_tool_names=[
            "file_read",
            "glob_search",
            "grep_search",
            "csv_read",
            "pdf_read",
            "docx_read",
            "notebook_read",
            "file_write",
            "web_search",
            "web_fetch",
            "get_current_time",
        ],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
        tool_capabilities=_tool_capabilities("web_search", "web_fetch"),
    )

    assert {"web_search", "web_fetch", "get_current_time"} <= set(plan.tool_names)
    assert {
        "file_read",
        "glob_search",
        "grep_search",
        "csv_read",
        "pdf_read",
        "docx_read",
        "notebook_read",
    } <= set(plan.tool_names)



def test_tool_exposure_keeps_open_world_research_tools_for_chinese_research_queries_in_workspace() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": ["file_read", "glob_search", "grep_search", "pdf_read", "docx_read", "notebook_read"],
            "web": ["web_search", "web_fetch", "get_current_time"],
            "literature": ["arxiv_search", "semantic_scholar_search", "crossref_search", "pubmed_search"],
        }
    )
    available_tools = [
        "file_read",
        "glob_search",
        "grep_search",
        "pdf_read",
        "docx_read",
        "notebook_read",
        "web_search",
        "web_fetch",
        "get_current_time",
        "arxiv_search",
        "semantic_scholar_search",
        "crossref_search",
        "pubmed_search",
    ]
    plan = planner.plan(
        message="幫我查詢 ESG 相關 LLM 微調資訊",
        user_intent_message="幫我查詢 ESG 相關 LLM 微調資訊",
        available_tool_names=available_tools,
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="auto_review",
        tool_capabilities=_tool_capabilities(
            "web_search",
            "web_fetch",
            "arxiv_search",
            "semantic_scholar_search",
            "crossref_search",
            "pubmed_search",
        ),
    )

    assert {"web_search", "web_fetch", "get_current_time"} <= set(plan.tool_names)
    assert (
        {"arxiv_search", "semantic_scholar_search", "crossref_search", "pubmed_search"} & set(plan.tool_names)
    )
