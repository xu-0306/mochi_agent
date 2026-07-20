from __future__ import annotations

import pytest

from mochi.agents.multi_agent.orchestrator import MultiAgentOrchestrator, MultiAgentRunRequest
from mochi.runtime.approvals import InMemoryApprovalStore

from ._support import _ConfiguredModelEngineStub


@pytest.mark.asyncio
async def test_orchestrator_controlled_execution_approves_and_executes(tmp_path) -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(
        engine=engine,
        controlled_exec_require_approval=False,
    )

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Run a bounded smoke command.",
            protocol={"protocol": "controlled_subagent_execution"},
            metadata={
                "task_workspace_dir": str(tmp_path),
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
        )
    )

    runtime = result.artifacts["controlled_execution_runtime"]
    assert runtime["execution_boundary"] == "subagents_propose_controller_approves_shared_runtime_executes"
    assert result.artifacts["execution_requests"]["parse_diagnostics"]["status"] == "parsed"
    assert result.artifacts["controller_decisions"]["items"][0]["status"] == "approved"
    execution_result = result.artifacts["execution_results"]["items"][0]
    assert execution_result["status"] != "skipped"
    assert execution_result["command"] == "echo controlled-ok"
    if execution_result["status"] == "completed":
        assert "controlled-ok" in str(execution_result["stdout"])
    profiles = [call["execution_profile"] for call in engine.invoke_calls]
    assert "subagent_execution_request" in profiles
    assert "controller_exec" in profiles




@pytest.mark.asyncio
async def test_orchestrator_controlled_execution_approval_pending_returns_waiting_approval_state(tmp_path) -> None:
    engine = _ConfiguredModelEngineStub()
    approval_store = InMemoryApprovalStore()
    orchestrator = MultiAgentOrchestrator(
        engine=engine,
        exec_approval_store=approval_store,
        controlled_exec_require_approval=True,
    )

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Run a bounded smoke command after approval.",
            protocol={"protocol": "controlled_subagent_execution"},
            metadata={
                "task_workspace_dir": str(tmp_path),
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
        )
    )

    assert result.state == "awaiting_approval"
    assert result.artifacts["execution_results"]["items"][0]["status"] == "approval_pending"
    assert result.artifacts["controlled_execution_runtime"]["approval_pending_count"] == 1

    approval_state = result.metadata["approval_state"]
    recovery_state = result.metadata["recovery_state"]
    assert approval_state["status"] == "awaiting_approval"
    assert approval_state["pending_count"] == 1
    assert recovery_state["status"] == "awaiting_approval"
    assert recovery_state["action"] == "await_approval"
    assert recovery_state["stage"] == "controlled_execution_exec:req-1"
    assert recovery_state["approval_state"]["approval_ids"] == approval_state["approval_ids"]

    pending = approval_state["pending_approvals"][0]
    approval_id = pending["approval_id"]
    assert pending["source"] == "controlled_execution"
    assert pending["request_id"] == "req-1"
    assert pending["task_key"] == "controlled_execution_exec:req-1"
    assert pending["tool_name"] == "exec_command"
    approval_request = approval_store.get(approval_id)
    assert approval_request is not None
    assert pending["reason"] == approval_request.reason

    resume_snapshot = recovery_state["resume_payload"]["role_task_snapshot"]
    assert "controlled_execution_exec:req-1" not in resume_snapshot["tasks"]
    assert "controlled_execution_evaluator" not in resume_snapshot["tasks"]
    assert "evaluator" not in resume_snapshot["roles"]
    assert resume_snapshot["tasks"]["controlled_execution_controller:req-1"]["status"] == "completed"
    assert result.events[-1].payload["current_state"] == "awaiting_approval"
    assert result.events[-1].payload["reason"] == "approval_pending"




@pytest.mark.asyncio
async def test_orchestrator_controlled_execution_rejects_without_running(tmp_path) -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(
        engine=engine,
        controlled_exec_require_approval=False,
    )

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Review but do not run unnecessary command.",
            protocol={"protocol": "controlled_subagent_execution"},
            metadata={
                "task_workspace_dir": str(tmp_path),
                "selected_models_roles": {
                    "by_role": {
                        "planner": "controlled-planner-model",
                        "executor": "controlled-executor-model",
                        "controller": "controlled-reject-controller-model",
                        "evaluator": "controlled-evaluator-model",
                        "judge": "controlled-judge-model",
                        "verifier": "controlled-verifier-model",
                    }
                },
            },
        )
    )

    assert result.artifacts["controller_decisions"]["items"][0]["status"] == "rejected"
    assert result.artifacts["execution_results"]["items"][0]["status"] == "skipped"




@pytest.mark.asyncio
async def test_orchestrator_debate_can_attach_controlled_execution_capability(tmp_path) -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(
        engine=engine,
        controlled_exec_require_approval=False,
    )

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Debate the best deployment path and validate a bounded command if needed.",
            protocol={"protocol": "multi_agent_debate", "rounds": 2},
            metadata={
                "task_workspace_dir": str(tmp_path),
                "execution_policy": {
                    "mode": "controlled",
                    "max_execution_requests": 1,
                    "max_commands_per_request": 1,
                    "default_timeout_sec": 30,
                    "background_allowed": False,
                },
                "selected_models_roles": {
                    "by_role": {
                        "debater_a": "debater-a-model",
                        "debater_b": "debater-b-model",
                        "judge": "judge-model",
                        "verifier": "verifier-model",
                        "planner": "controlled-planner-model",
                        "executor": "controlled-executor-model",
                        "controller": "controlled-controller-model",
                        "evaluator": "controlled-evaluator-model",
                    }
                },
            },
        )
    )

    assert result.protocol == "multi_agent_debate"
    assert result.selected_candidate_id == "debater_b"
    runtime = result.artifacts["controlled_execution_runtime"]
    assert runtime["primary_workflow"] is False
    assert runtime["execution_boundary"] == "subagents_propose_controller_approves_shared_runtime_executes"
    profiles = [call["execution_profile"] for call in engine.invoke_calls]
    assert "controller_exec" in profiles
    assert any(
        "Controlled execution context summary" in str(call["message"])
        for call in engine.invoke_calls
        if call["execution_profile"] in {"subagent_readonly", "judge", "verifier"}
    )
    assert result.artifacts["controlled_execution_runtime"]["executed_count"] == 1
