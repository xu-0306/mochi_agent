from __future__ import annotations

from pathlib import Path

import pytest

from mochi.tools.repo_map import ReadSymbolTool, RepoMapTool


def _file_map(entries: list[dict[str, object]]) -> dict[str, dict[str, object]]:
    return {
        str(entry["path"]): entry
        for entry in entries
        if isinstance(entry, dict) and "path" in entry
    }


@pytest.mark.asyncio
async def test_repo_map_extracts_python_symbols_and_skips_noisy_directories(tmp_path: Path) -> None:
    sample = tmp_path / "src" / "sample.py"
    sample.parent.mkdir(parents=True)
    sample.write_text(
        "class Greeter:\n"
        "    def greet(self) -> str:\n"
        "        return 'hi'\n"
        "\n"
        "def helper(value: str) -> str:\n"
        "    return value.upper()\n",
        encoding="utf-8",
    )
    ignored = tmp_path / ".git" / "ignored.py"
    ignored.parent.mkdir(parents=True)
    ignored.write_text("def hidden() -> None:\n    pass\n", encoding="utf-8")

    tool = RepoMapTool(workspace_dir=tmp_path)
    result = await tool.execute()

    assert result.error is None
    output = result.output
    assert isinstance(output, dict)
    files = _file_map(output["files"])
    assert "src/sample.py" in files
    assert ".git/ignored.py" not in files

    entry = files["src/sample.py"]
    assert entry["language"] == "python"
    assert entry["kind"] == "source"
    assert entry["symbols"] == [
        {"name": "Greeter", "kind": "class", "start_line": 1, "end_line": 3},
        {"name": "helper", "kind": "function", "start_line": 5, "end_line": 6},
    ]


@pytest.mark.asyncio
async def test_repo_map_extracts_js_ts_style_declarations(tmp_path: Path) -> None:
    sample = tmp_path / "web" / "app.ts"
    sample.parent.mkdir(parents=True)
    sample.write_text(
        "export class Client {\n"
        "  run() {}\n"
        "}\n"
        "const boot = () => {\n"
        "  return true;\n"
        "}\n"
        "export interface Config {\n"
        "  mode: string;\n"
        "}\n",
        encoding="utf-8",
    )

    tool = RepoMapTool(workspace_dir=tmp_path)
    result = await tool.execute(path="web")

    assert result.error is None
    output = result.output
    assert isinstance(output, dict)
    files = _file_map(output["files"])
    entry = files["web/app.ts"]
    assert entry["language"] == "typescript"
    assert entry["kind"] == "source"
    assert entry["symbols"] == [
        {"name": "Client", "kind": "class", "start_line": 1, "end_line": 3},
        {"name": "boot", "kind": "function", "start_line": 4, "end_line": 6},
        {"name": "Config", "kind": "interface", "start_line": 7, "end_line": 9},
    ]


@pytest.mark.asyncio
async def test_repo_map_js_ts_symbol_ranges_do_not_swallow_unrelated_top_level_lines(tmp_path: Path) -> None:
    sample = tmp_path / "web" / "app.ts"
    sample.parent.mkdir(parents=True)
    sample.write_text(
        "export function alpha() {\n"
        "  return 1;\n"
        "}\n"
        "const VERSION = 1;\n"
        "export function beta() {\n"
        "  return 2;\n"
        "}\n",
        encoding="utf-8",
    )

    tool = RepoMapTool(workspace_dir=tmp_path)
    result = await tool.execute(path="web")

    assert result.error is None
    output = result.output
    assert isinstance(output, dict)
    files = _file_map(output["files"])
    entry = files["web/app.ts"]
    assert entry["symbols"] == [
        {"name": "alpha", "kind": "function", "start_line": 1, "end_line": 3},
        {"name": "beta", "kind": "function", "start_line": 5, "end_line": 7},
    ]


@pytest.mark.asyncio
async def test_read_symbol_returns_python_symbol_block(tmp_path: Path) -> None:
    sample = tmp_path / "src" / "sample.py"
    sample.parent.mkdir(parents=True)
    sample.write_text(
        "class Greeter:\n"
        "    def greet(self) -> str:\n"
        "        return 'hi'\n"
        "\n"
        "def helper(value: str) -> str:\n"
        "    return value.upper()\n",
        encoding="utf-8",
    )

    tool = ReadSymbolTool(workspace_dir=tmp_path)
    result = await tool.execute(path="src/sample.py", symbol="helper")

    assert result.error is None
    assert result.output == "5: def helper(value: str) -> str:\n6:     return value.upper()"
    assert result.metadata["path"] == "src/sample.py"
    assert result.metadata["symbol"] == "helper"
    assert result.metadata["symbol_kind"] == "function"
    assert result.metadata["start_line"] == 5
    assert result.metadata["symbol_end_line"] == 6
    assert result.metadata["truncated"] is False


@pytest.mark.asyncio
async def test_read_symbol_returns_regex_based_symbol_block(tmp_path: Path) -> None:
    sample = tmp_path / "web" / "app.ts"
    sample.parent.mkdir(parents=True)
    sample.write_text(
        "export class Client {\n"
        "  run() {}\n"
        "}\n"
        "const boot = () => {\n"
        "  return true;\n"
        "}\n"
        "export interface Config {\n"
        "  mode: string;\n"
        "}\n",
        encoding="utf-8",
    )

    tool = ReadSymbolTool(workspace_dir=tmp_path)
    result = await tool.execute(path="web/app.ts", symbol="boot")

    assert result.error is None
    assert result.output == "4: const boot = () => {\n5:   return true;\n6: }"
    assert result.metadata["path"] == "web/app.ts"
    assert result.metadata["symbol"] == "boot"
    assert result.metadata["symbol_kind"] == "function"
    assert result.metadata["start_line"] == 4
    assert result.metadata["symbol_end_line"] == 6
    assert result.metadata["truncated"] is False


@pytest.mark.asyncio
async def test_read_symbol_js_ts_block_excludes_unrelated_top_level_lines(tmp_path: Path) -> None:
    sample = tmp_path / "web" / "app.ts"
    sample.parent.mkdir(parents=True)
    sample.write_text(
        "export function alpha() {\n"
        "  return 1;\n"
        "}\n"
        "const VERSION = 1;\n"
        "export function beta() {\n"
        "  return 2;\n"
        "}\n",
        encoding="utf-8",
    )

    tool = ReadSymbolTool(workspace_dir=tmp_path)
    result = await tool.execute(path="web/app.ts", symbol="alpha")

    assert result.error is None
    assert result.output == "1: export function alpha() {\n2:   return 1;\n3: }"
    assert result.metadata["path"] == "web/app.ts"
    assert result.metadata["symbol"] == "alpha"
    assert result.metadata["symbol_end_line"] == 3
    assert result.metadata["truncated"] is False


@pytest.mark.asyncio
async def test_read_symbol_fails_safely_for_missing_and_out_of_scope_paths(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.py"
    outside.write_text("def outside() -> None:\n    pass\n", encoding="utf-8")

    tool = ReadSymbolTool(workspace_dir=tmp_path)

    missing = await tool.execute(path="missing.py", symbol="helper")
    blocked = await tool.execute(path=str(outside), symbol="outside")

    assert missing.error == f"File not found: {tmp_path / 'missing.py'}"
    assert blocked.error == f"Path '{outside}' is outside workspace '{tmp_path.resolve(strict=False)}'."


@pytest.mark.asyncio
async def test_repo_navigation_tools_block_explicit_reads_from_ignored_directories(tmp_path: Path) -> None:
    ignored = tmp_path / ".git" / "ignored.py"
    ignored.parent.mkdir(parents=True)
    ignored.write_text("def hidden() -> None:\n    pass\n", encoding="utf-8")

    repo_map_tool = RepoMapTool(workspace_dir=tmp_path)
    read_symbol_tool = ReadSymbolTool(workspace_dir=tmp_path)

    repo_map_result = await repo_map_tool.execute(path=".git")
    read_symbol_result = await read_symbol_tool.execute(path=".git/ignored.py", symbol="hidden")

    assert repo_map_result.error == (
        "Path '.git' is inside ignored directory '.git' and is not available for repo navigation."
    )
    assert read_symbol_result.error == (
        "Path '.git/ignored.py' is inside ignored directory '.git' and is not available for symbol reads."
    )
