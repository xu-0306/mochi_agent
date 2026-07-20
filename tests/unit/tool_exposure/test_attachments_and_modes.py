from __future__ import annotations

from mochi.agents.tool_exposure import ToolExposurePlanner
from mochi.config.schema import MochiConfig
from tests.unit.tool_exposure._support import (
    _FakeBackend,
)


def test_tool_exposure_attached_workspace_reads_skip_execution_but_keep_write_capability() -> None:
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
    assert {"file_write", "file_edit"} <= set(plan.discoverable_tool_names)



def test_tool_exposure_attachment_bias_works_with_engine_structured_attachment_header(tmp_path) -> None:
    from mochi.agents.engine import AgentEngine
    from mochi.backends.types import AttachmentRef

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
        user_intent_message="請先檢查附件內容再回答",
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
        workspace_attachment_count=1,
    )

    assert "Structured attachments:" in planner_message
    assert {"file_read", "pdf_read", "docx_read"} <= set(plan.tool_names)
    assert not {"exec_command", "file_write", "file_edit", "execute_code"} & set(plan.tool_names)
    assert {"file_write", "file_edit"} <= set(plan.discoverable_tool_names)



def test_tool_exposure_attached_docx_edit_intent_keeps_write_tools_available() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": [
                "file_read",
                "docx_read",
                "file_write",
                "file_edit",
                "apply_patch",
            ],
        }
    )

    plan = planner.plan(
        message="Update the attached docx and save the revised version in the workspace.",
        user_intent_message="Update the attached docx and save the revised version in the workspace.",
        available_tool_names=["file_read", "docx_read", "file_write", "file_edit", "apply_patch"],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="trusted_workspace",
        attachment_count=1,
        workspace_attachment_count=1,
    )

    assert {"file_read", "docx_read", "file_write", "file_edit", "apply_patch"} <= set(plan.tool_names)



def test_tool_exposure_write_summary_of_attached_pdf_stays_read_only() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": [
                "file_read",
                "pdf_read",
                "file_write",
                "file_edit",
                "apply_patch",
            ],
        }
    )

    plan = planner.plan(
        message="Write a summary of the attached PDF.",
        user_intent_message="Write a summary of the attached PDF.",
        available_tool_names=["file_read", "pdf_read", "file_write", "file_edit", "apply_patch"],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="trusted_workspace",
        attachment_count=1,
        workspace_attachment_count=1,
    )

    assert {"file_read", "pdf_read"} <= set(plan.tool_names)
    assert not {"file_write", "file_edit", "apply_patch"} & set(plan.tool_names)
    assert {"file_write", "file_edit", "apply_patch"} <= set(plan.discoverable_tool_names)



def test_tool_exposure_attached_json_processing_keeps_non_mutating_execution_tools() -> None:
    planner = ToolExposurePlanner(
        tool_groups={
            "workspace": [
                "file_read",
                "exec_command",
                "execute_code",
                "file_write",
                "file_edit",
            ],
        }
    )

    plan = planner.plan(
        message=(
            "Read the attached JSON file, translate the values into Traditional Chinese, "
            "and return a JSON array in the final answer without writing any files."
        ),
        user_intent_message=(
            "Read the attached JSON file, translate the values into Traditional Chinese, "
            "and return a JSON array in the final answer without writing any files."
        ),
        available_tool_names=["file_read", "exec_command", "execute_code", "file_write", "file_edit"],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="trusted_workspace",
        attachment_count=1,
        workspace_attachment_count=1,
    )

    assert {"file_read", "exec_command", "execute_code"} <= set(plan.tool_names)
    assert not {"file_write", "file_edit"} & set(plan.tool_names)
    assert {"file_write", "file_edit"} <= set(plan.discoverable_tool_names)



def test_tool_exposure_attachment_filename_with_update_keyword_stays_read_only(tmp_path) -> None:
    from mochi.agents.engine import AgentEngine
    from mochi.backends.types import AttachmentRef

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
            "workspace": [
                "file_read",
                "pdf_read",
                "file_write",
                "file_edit",
                "apply_patch",
            ],
        }
    )
    planner_message = engine._build_tool_planner_message(  # noqa: SLF001
        "Summarize the attachment.",
        [
            AttachmentRef(
                name="updated-report.pdf",
                path=".mochi/workspace/browser-imports/updated-report.pdf",
                source="workspace_file",
            )
        ],
    )

    plan = planner.plan(
        message=planner_message,
        user_intent_message="Summarize the attachment.",
        available_tool_names=["file_read", "pdf_read", "file_write", "file_edit", "apply_patch"],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="trusted_workspace",
        attachment_count=1,
        workspace_attachment_count=1,
    )

    assert {"file_read", "pdf_read"} <= set(plan.tool_names)
    assert not {"file_write", "file_edit", "apply_patch"} & set(plan.tool_names)
    assert {"file_write", "file_edit", "apply_patch"} <= set(plan.discoverable_tool_names)



def test_tool_exposure_attachment_filename_with_update_keyword_stays_read_only_without_user_intent(tmp_path) -> None:
    from mochi.agents.engine import AgentEngine
    from mochi.backends.types import AttachmentRef

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
            "workspace": [
                "file_read",
                "pdf_read",
                "file_write",
                "file_edit",
                "apply_patch",
            ],
        }
    )
    planner_message = engine._build_tool_planner_message(  # noqa: SLF001
        "Summarize the attachment.",
        [
            AttachmentRef(
                name="updated-report.pdf",
                path=".mochi/workspace/browser-imports/updated-report.pdf",
                source="workspace_file",
            )
        ],
    )

    plan = planner.plan(
        message=planner_message,
        available_tool_names=["file_read", "pdf_read", "file_write", "file_edit", "apply_patch"],
        backend=_FakeBackend(),
        session_bound_workspace=True,
        autonomy_mode="trusted_workspace",
        attachment_count=1,
        workspace_attachment_count=1,
    )

    assert {"file_read", "pdf_read"} <= set(plan.tool_names)
    assert not {"file_write", "file_edit", "apply_patch"} & set(plan.tool_names)
    assert {"file_write", "file_edit", "apply_patch"} <= set(plan.discoverable_tool_names)



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
