from __future__ import annotations

from pathlib import Path

import pytest

from mochi.tools.base import ToolExecutionContext, ToolResult
from mochi.tools.file_ops import FileReadTool


def test_file_read_format_result_preserves_small_json_looking_text(tmp_path: Path) -> None:
    tool = FileReadTool(workspace_dir=tmp_path)

    rendered = tool.format_result_for_model(
        ToolResult(output='{"name": "mochi"}'),
        max_chars=512,
    )

    assert rendered == '{"name": "mochi"}'


@pytest.mark.asyncio
async def test_file_read_degrades_large_files_to_guided_chunking_and_allows_bounded_reads(
    tmp_path: Path,
) -> None:
    target = tmp_path / "large.log"
    target.write_text("".join(f"line {idx}\n" for idx in range(1, 401)), encoding="utf-8")

    tool = FileReadTool(workspace_dir=tmp_path, max_read_bytes=128)

    guided = await tool.execute(path="large.log")
    assert guided.error is None
    assert 'file_read(path="large.log", offset=1, limit=200, line_numbers=True)' in str(guided.output)
    assert guided.metadata["path"] == str(target.resolve(strict=False))
    assert guided.metadata["size_bytes"] > 128
    assert guided.metadata["partial"] is True

    chunk = await tool.execute(path="large.log", offset=2, limit=3, line_numbers=True)
    assert chunk.error is None
    assert chunk.output == "2: line 2\n3: line 3\n4: line 4"
    assert chunk.metadata["partial"] is True
    assert chunk.metadata["start_line"] == 2
    assert chunk.metadata["end_line"] == 4


@pytest.mark.asyncio
async def test_file_read_resolves_tool_result_references_via_virtual_path(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "tool-results"
    artifact_dir.mkdir()
    artifact_path = artifact_dir / "file_read-abc123.txt"
    artifact_path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    tool = FileReadTool(workspace_dir=tmp_path)
    context = ToolExecutionContext(
        workspace_dir=str(tmp_path),
        tool_result_store_dir=str(artifact_dir),
        tool_result_references={
            "file_read-abc123": {
                "reference_id": "file_read-abc123",
                "artifact_path": str(artifact_path),
                "tool_name": "file_read",
                "encoding": "utf-8",
            }
        },
    )

    result = await tool.execute(
        path="tool-result://file_read-abc123",
        offset=2,
        limit=1,
        line_numbers=True,
        context=context,
    )

    assert result.error is None
    assert result.output == "2: beta"
    assert result.metadata["path"] == "tool-result://file_read-abc123"
    assert result.metadata["artifact_path"] == str(artifact_path)
    assert result.metadata["reference_id"] == "file_read-abc123"
    assert result.metadata["tool_name"] == "file_read"
