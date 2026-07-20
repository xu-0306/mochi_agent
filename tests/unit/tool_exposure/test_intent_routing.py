from __future__ import annotations

from mochi.agents.tool_exposure import ToolExposurePlanner
from mochi.agents.tool_intent_router import (
    ToolIntentRoute,
    ToolIntentRouter,
    parse_tool_intent_classifier_result,
)
from tests.unit.tool_exposure._support import (
    _FakeBackend,
    _FakeToolIntentClassifier,
    _tool_capabilities,
)


async def test_tool_intent_router_prefers_high_confidence_classifier_route() -> None:
    router = ToolIntentRouter()

    route = await router.route(
        user_message="find matching files and search for TODO in the repo",
        session_bound_workspace=True,
        classifier=_FakeToolIntentClassifier(
            ToolIntentRoute(
                intent="workspace_read",
                confidence=0.94,
                source="classifier",
                rationale="Repo inspection intent is explicit.",
            )
        ),
    )

    assert route.intent == "workspace_read"
    assert route.source == "classifier"
    assert route.confidence == 0.94



async def test_tool_intent_router_low_confidence_classifier_falls_back_to_chinese_weather_route() -> None:
    router = ToolIntentRouter()

    route = await router.route(
        user_message="\u5e6b\u6211\u67e5\u8a62\u53f0\u4e2d\u660e\u5929\u5929\u6c23",
        session_bound_workspace=True,
        classifier=_FakeToolIntentClassifier(
            ToolIntentRoute(
                intent="workspace_read",
                confidence=0.22,
                source="classifier",
                rationale="Uncertain workspace classification.",
            )
        ),
    )

    assert route.intent == "open_world_lookup"
    assert route.source == "fallback_keyword"
    assert "weather" in route.rationale.lower() or "\u5929\u6c23" in route.rationale



def test_parse_tool_intent_classifier_result_normalizes_legacy_workspace_aliases() -> None:
    read_route = parse_tool_intent_classifier_result(
        '{"intent":"workspace_inspection","confidence":0.91,"rationale":"legacy read alias"}'
    )
    write_route = parse_tool_intent_classifier_result(
        '{"intent":"workspace_mutation","confidence":0.89,"rationale":"legacy write alias"}'
    )

    assert read_route.intent == "workspace_read"
    assert write_route.intent == "workspace_write"



def test_tool_exposure_keeps_literature_tools_for_ollama_prompt_guided_rejection() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": ["file_read"],
            "web": ["web_search", "web_fetch"],
            "literature": ["arxiv_search", "semantic_scholar_search"],
            "tool_discovery": ["tool_search"],
        }
    )
    available_tools = [
        "arxiv_search",
        "semantic_scholar_search",
        "web_search",
        "web_fetch",
        "file_read",
        "tool_search",
    ]
    plan = planner.plan(
        message="\u5e6b\u6211\u67e5\u8a62\u5929\u6c23\u9810\u6e2c\u76f8\u95dc\u6a21\u578b\u5fae\u8abf\u7684\u6587\u737b",
        user_intent_message="\u5e6b\u6211\u67e5\u8a62\u5929\u6c23\u9810\u6e2c\u76f8\u95dc\u6a21\u578b\u5fae\u8abf\u7684\u6587\u737b",
        available_tool_names=available_tools,
        backend=_FakeBackend(
            backend_type="ollama",
            metadata={
                "tool_call_mode": "unavailable",
                "tool_calling_protocol": "prompt_guided",
                "native_tool_calling_status": "simulated_protocol_rejected",
            },
        ),
        session_bound_workspace=True,
        preferred_tool_names=[],
        tool_capabilities=_tool_capabilities(*available_tools),
        routed_intent="literature_research",
        intent_confidence=0.88,
        intent_source="fallback_keyword",
        intent_rationale="Matched literature research language.",
    )

    assert "arxiv_search" in plan.tool_names
    assert "semantic_scholar_search" in plan.tool_names
    assert "web_search" in plan.tool_names
    assert plan.tool_names
    metadata = plan.exposure_metadata()
    assert metadata["diagnostics"]["available_tool_count"] == len(available_tools)
    assert metadata["diagnostics"]["backend"]["backend_type"] == "ollama"
    assert metadata["diagnostics"]["backend"]["metadata"]["tool_calling_protocol"] == "prompt_guided"



async def test_tool_intent_router_routes_tool_discovery_queries() -> None:
    route = await ToolIntentRouter().route(
        user_message="which tool should I use to inspect notebook outputs?",
        session_bound_workspace=True,
    )

    assert route.intent == "tool_discovery"
    assert route.source == "fallback_keyword"



async def test_tool_intent_router_fallback_keeps_generic_code_switching_query_out_of_workspace() -> None:
    route = await ToolIntentRouter().route(
        user_message="explain code switching in multilingual LLMs",
        session_bound_workspace=True,
    )

    assert route.intent != "workspace_read"



async def test_tool_intent_router_fallback_keeps_generic_rewrite_change_modify_queries_out_of_workspace_write() -> None:
    router = ToolIntentRouter()

    prompts = (
        "rewrite code switching explanation for beginners",
        "change project management explanation to be shorter",
        "modify history source criticism summary",
    )

    for prompt in prompts:
        route = await router.route(
            user_message=prompt,
            session_bound_workspace=True,
        )

        assert route.intent != "workspace_write"



async def test_tool_intent_router_fallback_routes_explicit_repo_query_to_workspace_read() -> None:
    route = await ToolIntentRouter().route(
        user_message="find matching files and search for TODO in the repo",
        session_bound_workspace=True,
    )

    assert route.intent == "workspace_read"
    assert route.source == "fallback_keyword"



async def test_tool_intent_router_fallback_routes_explicit_local_mutation_queries_to_workspace_write() -> None:
    router = ToolIntentRouter()

    prompts = (
        (
            "rewrite foo.py to remove TODO",
            {"attachment_count": 0, "workspace_attachment_count": 0},
        ),
        (
            "modify the workspace file report.md",
            {"attachment_count": 0, "workspace_attachment_count": 0},
        ),
        (
            "update this attached workspace file",
            {"attachment_count": 1, "workspace_attachment_count": 1},
        ),
    )

    for prompt, attachment_counts in prompts:
        route = await router.route(
            user_message=prompt,
            session_bound_workspace=True,
            **attachment_counts,
        )

        assert route.intent == "workspace_write"
        assert route.source == "fallback_keyword"
