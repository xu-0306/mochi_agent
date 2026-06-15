from __future__ import annotations

from pathlib import Path

from mochi.tools.base import ToolExecutionContext, ToolResult
from mochi.tools.transport_guard import ToolResultTransportGuard


def test_transport_guard_preserves_small_plain_text(tmp_path: Path) -> None:
    context = ToolExecutionContext(
        session_id="session-1",
        tool_result_store_dir=str(tmp_path),
    )
    guard = ToolResultTransportGuard()

    outcome = guard.guard(
        tool_name="echo",
        result=ToolResult(output="hello"),
        formatted_content="Tool echo result:\nhello",
        context=context,
        max_chars=512,
        backend_name="openai_compat",
        api_mode="responses",
    )

    assert outcome.content == "Tool echo result:\nhello"
    assert outcome.diagnostics["summary_applied"] is False
    assert outcome.diagnostics["overflow_persisted"] is False


def test_transport_guard_preserves_small_file_read_text_for_model(tmp_path: Path) -> None:
    context = ToolExecutionContext(
        session_id="session-file-read",
        tool_result_store_dir=str(tmp_path),
    )
    guard = ToolResultTransportGuard()

    outcome = guard.guard(
        tool_name="file_read",
        result=ToolResult(
            output="1: alpha\n2: beta",
            metadata={"path": str(tmp_path / "sample.txt"), "line_numbers": True},
        ),
        formatted_content="1: alpha\n2: beta",
        context=context,
        max_chars=512,
        backend_name="openai_compat",
        api_mode="responses",
    )

    assert outcome.content == "1: alpha\n2: beta"
    assert not outcome.content.startswith("Tool file_read result:")
    assert outcome.diagnostics["summary_applied"] is False
    assert outcome.diagnostics["overflow_persisted"] is False


def test_transport_guard_preserves_small_json_looking_file_read_text(tmp_path: Path) -> None:
    context = ToolExecutionContext(
        session_id="session-file-read-json",
        tool_result_store_dir=str(tmp_path),
    )
    guard = ToolResultTransportGuard()

    outcome = guard.guard(
        tool_name="file_read",
        result=ToolResult(
            output='{"name": "mochi"}',
            metadata={"path": str(tmp_path / "sample.json"), "line_numbers": False},
        ),
        formatted_content='{"name": "mochi"}',
        context=context,
        max_chars=512,
        backend_name="openai_compat",
        api_mode="responses",
    )

    assert outcome.content == '{"name": "mochi"}'
    assert outcome.diagnostics["summary_applied"] is False
    assert outcome.diagnostics["overflow_persisted"] is False


def test_transport_guard_summarizes_structured_payload_to_backend_safe_text(
    tmp_path: Path,
) -> None:
    context = ToolExecutionContext(
        session_id="session-1",
        tool_result_store_dir=str(tmp_path),
    )
    guard = ToolResultTransportGuard()

    outcome = guard.guard(
        tool_name="structured_tool",
        result=ToolResult(output={"items": [{"title": "Mochi", "url": "https://example.com"}]}),
        formatted_content=(
            '{"ok": true, "output": {"items": [{"title": "Mochi", '
            '"url": "https://example.com"}]}}'
        ),
        context=context,
        max_chars=512,
        backend_name="openai_compat",
        api_mode="responses",
    )

    assert outcome.content.startswith("Tool structured_tool result:")
    assert "https://example.com" in outcome.content
    assert not outcome.content.lstrip().startswith("{")
    assert outcome.diagnostics["summary_applied"] is True
    assert "structured_payload" in outcome.diagnostics["risk_flags"]


def test_transport_guard_preserves_web_search_json_evidence_under_limit(
    tmp_path: Path,
) -> None:
    context = ToolExecutionContext(
        session_id="session-1",
        tool_result_store_dir=str(tmp_path),
    )
    guard = ToolResultTransportGuard()
    formatted_content = (
        '{"ok": true, "output": {"results": ['
        '{"title": "臺中市 - 縣市預報", '
        '"url": "https://www.cwa.gov.tw/V8/C/W/County/County.html?CID=66", '
        '"snippet": "今日白天多雲午後短暫雷陣雨"}], '
        '"citations": ["[1] 臺中市 - 縣市預報 - https://www.cwa.gov.tw/V8/C/W/County/County.html?CID=66 - 今日白天多雲午後短暫雷陣雨"]}}'
    )

    outcome = guard.guard(
        tool_name="web_search",
        result=ToolResult(
            output={
                "results": [
                    {
                        "title": "臺中市 - 縣市預報",
                        "url": "https://www.cwa.gov.tw/V8/C/W/County/County.html?CID=66",
                        "snippet": "今日白天多雲午後短暫雷陣雨",
                    }
                ],
            }
        ),
        formatted_content=formatted_content,
        context=context,
        max_chars=1024,
        backend_name="openai_compat",
        api_mode="chat_completions",
    )

    assert outcome.content == formatted_content
    assert "今日白天多雲午後短暫雷陣雨" in outcome.content
    assert outcome.diagnostics["summary_applied"] is False
    assert outcome.diagnostics["transport_type"] == "web_evidence_json"
    assert "structured_payload" in outcome.diagnostics["risk_flags"]


def test_transport_guard_persists_large_payload_and_returns_preview_reference(
    tmp_path: Path,
) -> None:
    context = ToolExecutionContext(
        session_id="session-1",
        tool_result_store_dir=str(tmp_path),
    )
    guard = ToolResultTransportGuard()

    outcome = guard.guard(
        tool_name="large_text_tool",
        result=ToolResult(output="A" * 12000),
        formatted_content='{"ok": true, "output": "' + ("A" * 800) + '"}',
        context=context,
        max_chars=400,
        backend_name="openai_compat",
        api_mode="responses",
    )

    assert "Reference:" in outcome.content
    assert outcome.diagnostics["overflow_persisted"] is True
    assert outcome.diagnostics["reference_id"]
    assert outcome.diagnostics["reference_id"] in context.tool_result_references
    persisted_path = Path(
        context.tool_result_references[outcome.diagnostics["reference_id"]]["artifact_path"]
    )
    assert persisted_path.is_file()


def test_transport_guard_persists_large_file_read_with_resumable_followup_contract(
    tmp_path: Path,
) -> None:
    context = ToolExecutionContext(
        session_id="session-large-file-read",
        tool_result_store_dir=str(tmp_path),
    )
    guard = ToolResultTransportGuard(preview_chars=120)

    outcome = guard.guard(
        tool_name="file_read",
        result=ToolResult(
            output="\n".join(f"{idx}: line {idx}" for idx in range(1, 401)),
            metadata={"path": str(tmp_path / "huge.log"), "line_numbers": True},
        ),
        formatted_content="\n".join(f"{idx}: line {idx}" for idx in range(1, 401)),
        context=context,
        max_chars=220,
        backend_name="openai_compat",
        api_mode="responses",
    )

    reference_id = outcome.diagnostics["reference_id"]
    assert outcome.diagnostics["summary_applied"] is True
    assert outcome.diagnostics["overflow_persisted"] is True
    assert reference_id
    assert f'tool-result://{reference_id}' in outcome.content
    assert 'file_read(path="tool-result://' in outcome.content
    assert "offset=1, limit=200, line_numbers=True" in outcome.content

    reference = context.tool_result_references[reference_id]
    assert reference["reference_id"] == reference_id
    assert reference["tool_name"] == "file_read"
    assert reference["encoding"] == "utf-8"
    assert reference["source_path"] == str(tmp_path / "huge.log")
    assert Path(reference["artifact_path"]).is_file()


def test_transport_guard_persists_file_read_when_formatted_text_exceeds_backend_cap(
    tmp_path: Path,
) -> None:
    context = ToolExecutionContext(
        session_id="session-file-read-early-persist",
        tool_result_store_dir=str(tmp_path),
    )
    guard = ToolResultTransportGuard(preview_chars=480, persistence_multiplier=4)
    text = "\n".join(f"{idx}: line {idx}" for idx in range(1, 45))

    outcome = guard.guard(
        tool_name="file_read",
        result=ToolResult(
            output=text,
            metadata={"path": str(tmp_path / "medium.log"), "line_numbers": True},
        ),
        formatted_content=text,
        context=context,
        max_chars=220,
        backend_name="openai_compat",
        api_mode="responses",
    )

    reference_id = outcome.diagnostics["reference_id"]
    assert len(text) > 220
    assert outcome.diagnostics["overflow_persisted"] is True
    assert reference_id
    assert f'tool-result://{reference_id}' in outcome.content
