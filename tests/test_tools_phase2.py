from __future__ import annotations

from pathlib import Path

import pytest

from mochi.tools.base import ToolExecutionContext, ToolResult
from mochi.tools.file_ops import FileReadTool
from mochi.tools.transport_guard import ToolResultTransportGuard


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


@pytest.mark.asyncio
async def test_file_read_tool_result_continues_from_original_source_after_overflow(
    tmp_path: Path,
) -> None:
    target = tmp_path / "large.txt"
    target.write_text("".join(f"line {idx}\n" for idx in range(1, 121)), encoding="utf-8")

    context = ToolExecutionContext(
        workspace_dir=str(tmp_path),
        session_id="session-continuation",
        tool_result_store_dir=str(tmp_path / "tool-results"),
    )
    tool = FileReadTool(workspace_dir=tmp_path)
    guard = ToolResultTransportGuard(preview_chars=120)

    first_chunk = await tool.execute(
        path="large.txt",
        offset=1,
        limit=60,
        line_numbers=True,
        context=context,
    )
    assert first_chunk.error is None

    outcome = guard.guard(
        tool_name="file_read",
        result=first_chunk,
        formatted_content=tool.format_result_for_model(first_chunk, max_chars=220),
        context=context,
        max_chars=220,
        backend_name="openai_compat",
        api_mode="responses",
    )

    reference_id = outcome.diagnostics["reference_id"]
    assert reference_id
    assert context.tool_result_references[reference_id]["source_path"] == str(target.resolve(strict=False))

    continued = await tool.execute(
        path=f"tool-result://{reference_id}",
        offset=61,
        limit=3,
        line_numbers=True,
        context=context,
    )

    assert continued.error is None
    assert continued.output == "61: line 61\n62: line 62\n63: line 63"
    assert continued.metadata["path"] == f"tool-result://{reference_id}"
    assert continued.metadata["source_path"] == str(target.resolve(strict=False))


@pytest.mark.asyncio
async def test_file_read_tool_result_falls_back_to_artifact_when_source_missing(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "deleted-source.txt"
    source_path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    tool = FileReadTool(workspace_dir=tmp_path)
    context = ToolExecutionContext(
        workspace_dir=str(tmp_path),
        session_id="session-fallback-artifact",
        tool_result_store_dir=str(tmp_path / "tool-results"),
    )
    guard = ToolResultTransportGuard(preview_chars=120)

    first_chunk = await tool.execute(
        path="deleted-source.txt",
        offset=1,
        limit=3,
        line_numbers=True,
        context=context,
    )
    assert first_chunk.error is None

    outcome = guard.guard(
        tool_name="file_read",
        result=first_chunk,
        formatted_content=tool.format_result_for_model(first_chunk, max_chars=10),
        context=context,
        max_chars=10,
        backend_name="openai_compat",
        api_mode="responses",
    )
    reference_id = outcome.diagnostics["reference_id"]
    assert reference_id

    artifact_path = Path(context.tool_result_references[reference_id]["artifact_path"])
    assert artifact_path.is_file()
    source_path.unlink()

    result = await tool.execute(
        path=f"tool-result://{reference_id}",
        offset=2,
        limit=1,
        line_numbers=True,
        context=context,
    )

    assert result.error is None
    assert result.output == "2: 2: beta"
    assert result.metadata["source_path"] == str(source_path)
    assert result.metadata["artifact_path"] == str(artifact_path)
    assert result.metadata["encoding"] == "utf-8"
    assert result.metadata["path"] == f"tool-result://{reference_id}"


@pytest.mark.asyncio
async def test_file_read_tool_result_preserves_original_encoding_for_continuation(
    tmp_path: Path,
) -> None:
    target = tmp_path / "large-utf16.txt"
    target.write_text("".join(f"line {idx}\n" for idx in range(1, 121)), encoding="utf-16")

    context = ToolExecutionContext(
        workspace_dir=str(tmp_path),
        session_id="session-continuation-utf16",
        tool_result_store_dir=str(tmp_path / "tool-results"),
    )
    tool = FileReadTool(workspace_dir=tmp_path)
    guard = ToolResultTransportGuard(preview_chars=120)

    first_chunk = await tool.execute(
        path="large-utf16.txt",
        encoding="utf-16",
        offset=1,
        limit=60,
        line_numbers=True,
        context=context,
    )
    assert first_chunk.error is None
    assert first_chunk.metadata["encoding"] == "utf-16"

    outcome = guard.guard(
        tool_name="file_read",
        result=first_chunk,
        formatted_content=tool.format_result_for_model(first_chunk, max_chars=220),
        context=context,
        max_chars=220,
        backend_name="openai_compat",
        api_mode="responses",
    )

    reference_id = outcome.diagnostics["reference_id"]
    assert reference_id
    assert context.tool_result_references[reference_id]["encoding"] == "utf-16"

    continued = await tool.execute(
        path=f"tool-result://{reference_id}",
        offset=61,
        limit=3,
        line_numbers=True,
        context=context,
    )

    assert continued.error is None
    assert continued.output == "61: line 61\n62: line 62\n63: line 63"
    assert continued.metadata["encoding"] == "utf-16"


@pytest.mark.asyncio
async def test_file_read_tool_result_falls_back_to_artifact_with_artifact_encoding(
    tmp_path: Path,
) -> None:
    target = tmp_path / "deleted-utf16.txt"
    target.write_text("alpha\nbeta\ngamma\n", encoding="utf-16")

    context = ToolExecutionContext(
        workspace_dir=str(tmp_path),
        session_id="session-fallback-utf16",
        tool_result_store_dir=str(tmp_path / "tool-results"),
    )
    tool = FileReadTool(workspace_dir=tmp_path)
    guard = ToolResultTransportGuard(preview_chars=120)

    first_chunk = await tool.execute(
        path="deleted-utf16.txt",
        encoding="utf-16",
        offset=1,
        limit=3,
        line_numbers=True,
        context=context,
    )
    assert first_chunk.error is None

    outcome = guard.guard(
        tool_name="file_read",
        result=first_chunk,
        formatted_content=tool.format_result_for_model(first_chunk, max_chars=10),
        context=context,
        max_chars=10,
        backend_name="openai_compat",
        api_mode="responses",
    )
    reference_id = outcome.diagnostics["reference_id"]
    assert reference_id

    artifact_path = Path(context.tool_result_references[reference_id]["artifact_path"])
    assert artifact_path.is_file()
    target.unlink()

    continued = await tool.execute(
        path=f"tool-result://{reference_id}",
        offset=2,
        limit=1,
        line_numbers=True,
        context=context,
    )

    assert continued.error is None
    assert continued.output == "2: 2: beta"
    assert continued.metadata["encoding"] == "utf-8"
    assert continued.metadata["artifact_path"] == str(artifact_path)
