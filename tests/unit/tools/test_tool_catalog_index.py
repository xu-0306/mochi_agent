from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from mochi.tools.base import BaseTool, ToolResult
from mochi.tools.tool_catalog_index import ToolCatalogIndex


@dataclass
class _FakeTool(BaseTool):
    tool_name: str
    tool_description: str
    hint: str | None = None
    schema: dict[str, Any] | None = None
    capabilities: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return self.tool_name

    @property
    def description(self) -> str:
        return self.tool_description

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self.schema or {"type": "object", "properties": {}}

    @property
    def search_hint(self) -> str | None:
        return self.hint

    @property
    def tool_capabilities(self) -> dict[str, Any]:
        return self.capabilities or super().tool_capabilities

    async def execute(self, **kwargs: Any) -> ToolResult:
        del kwargs
        return ToolResult(output="unused")


def test_exact_select_returns_named_tool_with_rank_score_and_fingerprint() -> None:
    index = ToolCatalogIndex.from_tools(
        [
            _FakeTool("file_read", "Read a workspace file."),
            _FakeTool("file_write", "Write a workspace file."),
        ]
    )

    matches = index.search("select:file_write", top_k=10)

    assert len(matches) == 1
    assert matches[0].name == "file_write"
    assert matches[0].rank == 1
    assert matches[0].score == 1000.0
    assert matches[0].catalog_fingerprint == index.catalog_fingerprint


def test_multilingual_metadata_match_scores_positive() -> None:
    index = ToolCatalogIndex.from_tools(
        [
            _FakeTool(
                "file_write",
                "Write text into a workspace file.",
                hint="可用來寫入檔案、保存內容與更新程式。",
                schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Destination file path."},
                        "content": {"type": "string", "description": "UTF-8 text content."},
                    },
                },
                capabilities={"domains": ["workspace_write", "artifact"]},
            ),
            _FakeTool("web_search", "Search the web."),
        ]
    )

    matches = index.search("寫入 檔案 保存", top_k=5)

    assert matches
    assert matches[0].name == "file_write"
    assert matches[0].score > 0.0


def test_bounded_inflection_normalization_finds_saving_files() -> None:
    index = ToolCatalogIndex.from_tools(
        [
            _FakeTool("file_read", "Read a workspace file."),
            _FakeTool("file_write", "Write text into a workspace file."),
        ]
    )

    matches = index.search("saving files")

    assert matches
    file_write = next(match for match in matches if match.name == "file_write")
    assert file_write.score > 0.0
    assert index.search("astronomy nebula") == []


def test_zero_score_tools_are_filtered_and_ties_are_deterministic() -> None:
    index = ToolCatalogIndex.from_tools(
        [
            _FakeTool("alpha_reader", "Inspect project state.", hint="Read project notes."),
            _FakeTool("zeta_reader", "Inspect project state.", hint="Read project notes."),
            _FakeTool("web_search", "Search public websites."),
        ]
    )

    matches = index.search("inspect project", top_k=10)

    assert [item.name for item in matches] == ["alpha_reader", "zeta_reader"]
    assert all(item.score > 0.0 for item in matches)
    assert index.search("totally unrelated phrase", top_k=10) == []


def test_top_k_uses_default_and_hard_max() -> None:
    tools = [
        _FakeTool(f"tool_{index}", "Project helper tool.", hint="Project helper tool.")
        for index in range(5)
    ]
    index = ToolCatalogIndex.from_tools(tools, default_top_k=2, max_top_k=3)

    default_matches = index.search("project helper")
    hard_capped_matches = index.search("project helper", top_k=99)

    assert len(default_matches) == 2
    assert len(hard_capped_matches) == 3
