"""Runtime API tests grouped by ownership."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from mochi.agents.events import (
    FinalAnswerEvent,
    ToolCallRequestEvent,
    ToolCallResultEvent,
)
from mochi.api.server import create_app
from mochi.config.schema import MochiConfig
from mochi.runtime.approvals import APPROVAL_OWNER_TASK_ID_KEY

from ._support import (
    _CONTROLLED_SMOKE_COMMAND_RULE,
    _AgentRunModelBackedEngine,
    _RuntimeFakeEngine,
    _RuntimeFileMutationFakeEngine,
    _wait_until,
)

_TWO_FILE_PATCH = "\n".join(
    [
        "*** Begin Patch",
        "*** Update File: alpha.py",
        "@@",
        "-print('alpha')",
        "+print('alpha selected')",
        "*** Update File: beta.py",
        "@@",
        "-print('beta')",
        "+print('beta must remain unchanged')",
        "*** End Patch",
    ]
)


class _RuntimeSubsetReplayFakeEngine:
    """Emit a two-file approval for runtime's authoritative replay path."""

    def __init__(self) -> None:
        self._run_count = 0

    async def chat(
        self,
        message: str,
        session_id: str | None = None,
        inference_overrides: dict[str, object] | None = None,
        project_id: str | None = None,
        workspace_dir: str | None = None,
        task_workspace_dir: str | None = None,
        permission_policy: dict[str, object] | None = None,
    ) -> object:
        _ = (message, session_id, inference_overrides, project_id, workspace_dir)
        self._run_count += 1
        if self._run_count == 1:
            yield ToolCallRequestEvent(
                call_id="call-subset-patch-1",
                tool_name="apply_patch",
                arguments={"patch": _TWO_FILE_PATCH},
            )
            yield ToolCallResultEvent(
                call_id="call-subset-patch-1",
                tool_name="apply_patch",
                result=None,
                error="Patch application requires approval.",
                metadata={
                    "requires_approval": True,
                    "approval_kind": "apply_patch",
                    "approval_scope": "workspace",
                    "replay_safe": True,
                },
            )
            return

        yield FinalAnswerEvent(content="subset patch applied", trajectory_id="traj-subset")


def test_task_and_approval_flow_with_resume(tmp_path: Path) -> None:
    app = create_app()
    engine = _RuntimeFakeEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {"sessions_dir": str(tmp_path / "sessions")}
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/tasks",
            json={
                "input_message": "run something",
                "session_id": "runtime-s1",
                "workspace_dir": str(tmp_path / "project-workspace"),
            },
        )
        assert create_response.status_code == 200
        created = create_response.json()
        task_id = created["task_id"]
        assert created["project_workspace_dir"] == str(tmp_path / "project-workspace")
        expected_task_workspace = (
            tmp_path / "sessions" / "runtime-tasks" / task_id / "workspace"
        ).resolve()
        assert created["task_workspace_dir"] == str(expected_task_workspace)
        assert expected_task_workspace.is_dir()

        task_payload = _wait_until(client, task_id, {"awaiting_approval"})
        assert task_payload["project_workspace_dir"] == str(tmp_path / "project-workspace")
        assert task_payload["task_workspace_dir"] == str(expected_task_workspace)
        assert task_payload["pending_approval"] is not None
        assert task_payload["events"][0]["type"] == "thinking"
        assert task_payload["events"][1]["type"] == "tool_call_request"
        assert task_payload["events"][2]["type"] == "tool_call_result"

        list_response = client.get("/v1/tasks")
        assert list_response.status_code == 200
        listed = list_response.json()
        assert len(listed) == 1
        assert listed[0]["project_workspace_dir"] == str(tmp_path / "project-workspace")
        assert listed[0]["task_workspace_dir"] == str(expected_task_workspace)

        approvals_response = client.get("/v1/approvals?status=pending")
        assert approvals_response.status_code == 200
        approvals = approvals_response.json()
        assert len(approvals) == 1
        assert approvals[0]["approval_kind"] == "exec"
        assert approvals[0]["approval_scope"] == "dangerous_command"
        assert approvals[0]["requires_approval"] is True
        assert approvals[0]["replay_safe"] is False
        assert approvals[0]["security_decision"] == "require_approval"
        assert approvals[0]["policy_source"] == "static_policy"
        assert approvals[0]["policy_reason"] == "Command requires approval by policy."
        approval_id = approvals[0]["approval_id"]

        resolve_response = client.post(
            f"/v1/approvals/{approval_id}/resolve",
            json={"decision": "approve_once", "reason": "allowed"},
        )
        assert resolve_response.status_code == 200, resolve_response.json()
        resolved = resolve_response.json()
        assert resolved["reason"] == "allowed"
        assert resolved["approval_kind"] == "exec"
        assert resolved["security_decision"] == "require_approval"

        done_payload = _wait_until(client, task_id, {"succeeded"})
        assert done_payload["events"][-1]["type"] == "final_answer"

    assert engine.permission_policy_calls[0] == {
        "autonomy_mode": "trusted_workspace",
        "require_approval_for_file_write": False,
        "require_approval_for_exec": True,
        "file_read_scope": "workspace",
        "file_write_scope": "workspace",
        APPROVAL_OWNER_TASK_ID_KEY: task_id,
    }
    assert engine.permission_policy_calls[1] == {
        "autonomy_mode": "trusted_workspace",
        "require_approval_for_file_write": False,
        "require_approval_for_exec": True,
        "file_read_scope": "workspace",
        "file_write_scope": "workspace",
        APPROVAL_OWNER_TASK_ID_KEY: task_id,
        "approved_tool_calls": [
            {
                "tool_name": "exec_command",
                "arguments": {"command": "dir", "shell": "cmd"},
            }
        ]
    }
    assert engine.task_workspace_calls == [
        str(expected_task_workspace),
        str(expected_task_workspace),
    ]

@pytest.mark.parametrize("change_contract_mode", ["observe", "enforce"])
def test_file_mutation_approval_binds_edited_patch_preview(
    tmp_path: Path,
    change_contract_mode: str,
) -> None:
    app = create_app()
    engine = _RuntimeFileMutationFakeEngine()
    sessions_dir = tmp_path / "sessions"
    project_workspace = tmp_path / "project-workspace"
    project_workspace.mkdir(parents=True)
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "sessions_dir": str(sessions_dir),
            "workspace_dir": str(project_workspace),
            "security": {"change_contract_mode": change_contract_mode},
        }
    )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/tasks",
            json={
                "input_message": "edit notes",
                "session_id": "runtime-s1",
                "workspace_dir": str(project_workspace),
            },
        )
        assert create_response.status_code == 200
        task_id = create_response.json()["task_id"]
        task_workspace = sessions_dir / "runtime-tasks" / task_id / "workspace"
        task_workspace.mkdir(parents=True, exist_ok=True)
        (task_workspace / "notes.py").write_text("print('alpha')\n", encoding="utf-8")

        waiting = _wait_until(client, task_id, {"awaiting_approval"})
        approval_id = waiting["pending_approval"]["id"]

        approvals_response = client.get("/v1/approvals?status=pending")
        assert approvals_response.status_code == 200
        approval = next(item for item in approvals_response.json() if item["approval_id"] == approval_id)
        assert approval["tool_name"] == "apply_patch"
        assert approval["approval_kind"] == "apply_patch"
        assert approval["allowed_decisions"] == ["approve_once", "reject"]
        assert approval["change_count"] == 1
        assert approval["file_changes"][0]["relative_path"] == "notes.py"
        assert len(approval["request_digest"]) == 64

        edited_patch = "\n".join(
            [
                "*** Begin Patch",
                "*** Update File: notes.py",
                "@@",
                "-print('alpha')",
                "+print('gamma')",
                "*** End Patch",
            ]
        )
        preview_response = client.post(
            "/v1/workspace/patch/preview",
            json={
                "approval_id": approval_id,
                "patch": edited_patch,
            },
        )
        assert preview_response.status_code == 200
        preview_payload = preview_response.json()
        assert preview_payload["valid"] is True
        assert preview_payload["workspace_dir"] == str(task_workspace.resolve())
        assert preview_payload["file_changes"][0]["relative_path"] == "notes.py"
        assert preview_payload["change_contract_mode"] == change_contract_mode
        assert len(preview_payload["request_digest"]) == 64

        repeated_preview = client.post(
            "/v1/workspace/patch/preview",
            json={"approval_id": approval_id, "patch": edited_patch},
        )
        assert repeated_preview.status_code == 200
        repeated_payload = repeated_preview.json()
        assert repeated_payload["change_set_id"] == preview_payload["change_set_id"]
        assert (
            repeated_payload["replacement_approval_id"]
            == preview_payload["replacement_approval_id"]
        )

        if change_contract_mode == "observe":
            assert preview_payload["replacement_approval_id"] is None
            assert preview_payload["would_reject_edited_patch"] is True
            approval_to_resolve = approval_id
            replay_override = {
                "tool_name": "apply_patch",
                "arguments": {"patch": edited_patch},
            }
        else:
            approval_to_resolve = preview_payload["replacement_approval_id"]
            assert approval_to_resolve != approval_id
            assert preview_payload["approval_state"] == "replacement_pending"
            replay_override = None
            old_response = client.post(
                f"/v1/approvals/{approval_id}/resolve",
                json={"decision": "approve_once"},
            )
            assert old_response.status_code == 409
            assert (task_workspace / "notes.py").read_text(encoding="utf-8") == "print('alpha')\n"

        resolve_response = client.post(
            f"/v1/approvals/{approval_to_resolve}/resolve",
            json={
                "decision": "approve_and_save_rule",
                "reason": "apply edited patch",
                **({"replay_override": replay_override} if replay_override else {}),
            },
        )
        assert resolve_response.status_code == 200, resolve_response.json()
        resolved = resolve_response.json()
        assert resolved["status"] == "consumed", resolved
        assert resolved["decision"] == "approve_once"
        if change_contract_mode == "observe":
            assert resolved["would_reject_edited_patch"] is True

        done_payload = _wait_until(client, task_id, {"succeeded"})
        assert done_payload["status"] == "succeeded"
        assert done_payload["final_answer"] == "Applied approved file change to notes.py."

        if change_contract_mode == "enforce":
            result_event = next(
                event
                for event in reversed(done_payload["events"])
                if event.get("type") == "tool_call_result"
            )
            assert "original_content" not in result_event["metadata"]
            protected_change = result_event["metadata"]["file_changes"][0]
            assert "original_content" not in protected_change
            assert protected_change["undo_status"] == "retained"
            assert protected_change["undo_available"] is True
            assert protected_change["undo_entry_ids"] == [
                protected_change["entry_id"]
            ]
            undo_response = client.post(
                f"/v1/changes/{protected_change['change_set_id']}/undo",
                json={
                    "entry_ids": protected_change["undo_entry_ids"],
                    "request_digest": protected_change["request_digest"],
                },
            )
            assert undo_response.status_code == 200, undo_response.json()

    expected_content = (
        "print('alpha')\n"
        if change_contract_mode == "enforce"
        else "print('gamma')\n"
    )
    assert (task_workspace / "notes.py").read_text(encoding="utf-8") == expected_content
    assert len(engine.permission_policy_calls) == 1


def test_enforced_file_approval_rejects_stale_manifest_without_writing(
    tmp_path: Path,
) -> None:
    app = create_app()
    engine = _RuntimeFileMutationFakeEngine()
    sessions_dir = tmp_path / "sessions"
    project_workspace = tmp_path / "project-workspace"
    project_workspace.mkdir(parents=True)
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "sessions_dir": str(sessions_dir),
            "workspace_dir": str(project_workspace),
            "security": {"change_contract_mode": "enforce"},
        }
    )

    with TestClient(app) as client:
        created = client.post(
            "/v1/tasks",
            json={
                "input_message": "edit notes",
                "session_id": "runtime-s1",
                "workspace_dir": str(project_workspace),
            },
        ).json()
        task_id = created["task_id"]
        task_workspace = sessions_dir / "runtime-tasks" / task_id / "workspace"
        task_workspace.mkdir(parents=True, exist_ok=True)
        target = task_workspace / "notes.py"
        target.write_text("print('alpha')\n", encoding="utf-8")
        waiting = _wait_until(client, task_id, {"awaiting_approval"})
        approval_id = waiting["pending_approval"]["id"]
        target.write_text("print('external')\n", encoding="utf-8")

        response = client.post(
            f"/v1/approvals/{approval_id}/resolve",
            json={"decision": "approve_once"},
        )

    assert response.status_code == 409
    assert response.json()["detail"] == "change_set_conflicted"
    assert target.read_text(encoding="utf-8") == "print('external')\n"


def test_subset_preview_replaces_parent_and_replays_only_selected_entries(
    tmp_path: Path,
) -> None:
    app = create_app()
    engine = _RuntimeSubsetReplayFakeEngine()
    sessions_dir = tmp_path / "sessions"
    project_workspace = tmp_path / "project-workspace"
    project_workspace.mkdir()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "sessions_dir": str(sessions_dir),
            "workspace_dir": str(project_workspace),
            "security": {"change_contract_mode": "enforce"},
        }
    )

    with TestClient(app) as client:
        created = client.post(
            "/v1/tasks",
            json={
                "input_message": "edit two files selectively",
                "session_id": "runtime-subset",
                "workspace_dir": str(project_workspace),
            },
        )
        assert created.status_code == 200
        task_id = created.json()["task_id"]
        task_workspace = sessions_dir / "runtime-tasks" / task_id / "workspace"
        task_workspace.mkdir(parents=True, exist_ok=True)
        alpha = task_workspace / "alpha.py"
        beta = task_workspace / "beta.py"
        alpha.write_text("print('alpha')\n", encoding="utf-8")
        beta.write_text("print('beta')\n", encoding="utf-8")

        waiting = _wait_until(client, task_id, {"awaiting_approval"})
        parent_approval_id = waiting["pending_approval"]["id"]
        approvals = client.get("/v1/approvals?status=pending").json()
        parent = next(
            item
            for item in approvals
            if item["approval_id"] == parent_approval_id
        )
        entries = parent["file_changes"]
        assert len(entries) == 2
        alpha_entry = next(
            entry for entry in entries if entry["relative_path"] == "alpha.py"
        )
        beta_entry = next(
            entry for entry in entries if entry["relative_path"] == "beta.py"
        )
        alpha_entry_id = alpha_entry["entry_id"]
        beta_entry_id = beta_entry["entry_id"]

        missing_fields = client.post("/v1/workspace/patch/subset-preview", json={})
        empty_selection = client.post(
            "/v1/workspace/patch/subset-preview",
            json={"approval_id": parent_approval_id, "selected_entry_ids": []},
        )
        duplicate_selection = client.post(
            "/v1/workspace/patch/subset-preview",
            json={
                "approval_id": parent_approval_id,
                "selected_entry_ids": [alpha_entry_id, alpha_entry_id],
            },
        )
        unknown_selection = client.post(
            "/v1/workspace/patch/subset-preview",
            json={
                "approval_id": parent_approval_id,
                "selected_entry_ids": ["unknown-entry"],
            },
        )
        forbidden_client_projection = client.post(
            "/v1/workspace/patch/subset-preview",
            json={
                "approval_id": parent_approval_id,
                "selected_entry_ids": [alpha_entry_id],
                "patch": _TWO_FILE_PATCH,
            },
        )
        assert missing_fields.status_code == 422
        assert empty_selection.status_code == 422
        assert duplicate_selection.status_code == 422
        assert unknown_selection.status_code == 422
        assert forbidden_client_projection.status_code == 422

        preview = client.post(
            "/v1/workspace/patch/subset-preview",
            json={
                "approval_id": parent_approval_id,
                "selected_entry_ids": [alpha_entry_id],
            },
        )
        assert preview.status_code == 200, preview.json()
        payload = preview.json()
        assert payload["type"] == "workspace_patch_subset_preview"
        assert payload["valid"] is True
        assert payload["parent_request_digest"] == parent["request_digest"]
        assert payload["request_digest"] != parent["request_digest"]
        assert payload["selected_entry_ids"] == [alpha_entry_id]
        assert payload["approval_state"] == "replacement_pending"
        assert payload["replacement_approval_id"] != parent_approval_id
        assert len(payload["file_changes"]) == 1
        assert payload["file_changes"][0]["relative_path"] == "alpha.py"
        assert payload["file_changes"][0]["entry_id"] != alpha_entry_id

        retry = client.post(
            "/v1/workspace/patch/subset-preview",
            json={
                "approval_id": parent_approval_id,
                "selected_entry_ids": [alpha_entry_id],
            },
        )
        conflicting_retry = client.post(
            "/v1/workspace/patch/subset-preview",
            json={
                "approval_id": parent_approval_id,
                "selected_entry_ids": [beta_entry_id],
            },
        )
        stale_parent_resolution = client.post(
            f"/v1/approvals/{parent_approval_id}/resolve",
            json={"decision": "approve_once"},
        )
        assert retry.status_code == 200, retry.json()
        assert retry.json()["replacement_approval_id"] == payload["replacement_approval_id"]
        assert retry.json()["request_digest"] == payload["request_digest"]
        assert conflicting_retry.status_code == 409
        assert stale_parent_resolution.status_code == 409

        resolved = client.post(
            f"/v1/approvals/{payload['replacement_approval_id']}/resolve",
            json={"decision": "approve_once"},
        )
        assert resolved.status_code == 200, resolved.json()
        done = _wait_until(client, task_id, {"succeeded"})
        assert done["final_answer"] == "Applied approved file change to alpha.py."

    assert alpha.read_text(encoding="utf-8") == "print('alpha selected')\n"
    assert beta.read_text(encoding="utf-8") == "print('beta')\n"


def test_subset_preview_conflicts_stale_parent_and_missing_approval(tmp_path: Path) -> None:
    app = create_app()
    engine = _RuntimeSubsetReplayFakeEngine()
    sessions_dir = tmp_path / "sessions"
    project_workspace = tmp_path / "project-workspace"
    project_workspace.mkdir()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "sessions_dir": str(sessions_dir),
            "workspace_dir": str(project_workspace),
            "security": {"change_contract_mode": "enforce"},
        }
    )

    with TestClient(app) as client:
        missing = client.post(
            "/v1/workspace/patch/subset-preview",
            json={"approval_id": "missing", "selected_entry_ids": ["entry"]},
        )
        assert missing.status_code == 404

        created = client.post(
            "/v1/tasks",
            json={
                "input_message": "edit two files selectively",
                "session_id": "runtime-stale-subset",
                "workspace_dir": str(project_workspace),
            },
        )
        assert created.status_code == 200
        task_id = created.json()["task_id"]
        task_workspace = sessions_dir / "runtime-tasks" / task_id / "workspace"
        task_workspace.mkdir(parents=True, exist_ok=True)
        alpha = task_workspace / "alpha.py"
        alpha.write_text("print('alpha')\n", encoding="utf-8")
        (task_workspace / "beta.py").write_text("print('beta')\n", encoding="utf-8")

        waiting = _wait_until(client, task_id, {"awaiting_approval"})
        parent_approval_id = waiting["pending_approval"]["id"]
        parent = next(
            item
            for item in client.get("/v1/approvals?status=pending").json()
            if item["approval_id"] == parent_approval_id
        )
        alpha_entry_id = next(
            entry["entry_id"]
            for entry in parent["file_changes"]
            if entry["relative_path"] == "alpha.py"
        )
        alpha.write_text("print('changed outside approval')\n", encoding="utf-8")

        stale_preview = client.post(
            "/v1/workspace/patch/subset-preview",
            json={
                "approval_id": parent_approval_id,
                "selected_entry_ids": [alpha_entry_id],
            },
        )
        stale_resolution = client.post(
            f"/v1/approvals/{parent_approval_id}/resolve",
            json={"decision": "approve_once"},
        )

    assert stale_preview.status_code == 409
    assert stale_resolution.status_code == 409
    assert alpha.read_text(encoding="utf-8") == "print('changed outside approval')\n"
def test_controlled_subagent_execution_task_runs_in_task_workspace(tmp_path: Path) -> None:
    app = create_app()
    engine = _AgentRunModelBackedEngine()
    app.state.engine_factory = lambda: engine
    app.state.config_factory = lambda: MochiConfig.model_validate(
        {
            "sessions_dir": str(tmp_path / "sessions"),
                "security": {
                    "require_approval_for_exec": False,
                    "command_rules": [_CONTROLLED_SMOKE_COMMAND_RULE],
                },
            }
        )

    with TestClient(app) as client:
        create_response = client.post(
            "/v1/tasks",
            json={
                "input_message": "Run a controlled smoke command.",
                "task_type": "controlled_subagent_execution",
                "workspace_dir": str(tmp_path / "project-workspace"),
                "metadata": {
                    "protocol_config": {
                        "max_execution_requests": 1,
                        "max_commands_per_request": 1,
                        "default_timeout_sec": 30,
                        "background_allowed": False,
                    },
                    "selected_models_roles": {
                        "by_role": {
                            "planner": "controlled-planner-model",
                            "executor": "controlled-executor-model",
                            "controller": "controlled-controller-model",
                            "evaluator": "controlled-evaluator-model",
                            "judge": "controlled-judge-model",
                            "verifier": "controlled-verifier-model",
                        }
                    },
                },
            },
        )
        assert create_response.status_code == 200
        created = create_response.json()
        assert created["task_type"] == "controlled_subagent_execution"
        task_id = created["task_id"]
        expected_task_workspace = (
            tmp_path / "sessions" / "runtime-tasks" / task_id / "workspace"
        ).resolve()
        assert created["task_workspace_dir"] == str(expected_task_workspace)
        assert expected_task_workspace.is_dir()

        done_payload = _wait_until(client, task_id, {"succeeded"})
        assert done_payload["task_type"] == "controlled_subagent_execution"
        artifact_events = [
            event for event in done_payload["events"] if event.get("type") == "artifact"
        ]
        assert any(
            event.get("payload", {}).get("name") == "controlled_execution_runtime"
            for event in artifact_events
        )
        assert "Execution finished" in str(done_payload["final_answer"])
