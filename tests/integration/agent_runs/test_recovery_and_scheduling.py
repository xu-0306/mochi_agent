from __future__ import annotations

import pytest

from mochi.agents.multi_agent.orchestrator import (
    MultiAgentOrchestrator,
    MultiAgentRunRequest,
    RunPolicyStop,
)

from ._support import _ConfiguredModelEngineStub, _TeacherHeartbeatTimeoutEngine


@pytest.mark.asyncio
async def test_orchestrator_resume_continues_from_protocol_checkpoint_without_rerunning_protocol() -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(engine=engine)
    original = orchestrator._raise_if_run_policy_exhausted

    def _pause_after_protocol(**kwargs: object) -> None:
        original(**kwargs)
        if kwargs.get("stage") == "protocol_completed":
            raise RunPolicyStop(
                status="awaiting_resources",
                action="pause",
                reason="pause after protocol for resume test",
                stage="protocol_completed",
                checkpoint=orchestrator._latest_checkpoint,
            )

    orchestrator._raise_if_run_policy_exhausted = _pause_after_protocol  # type: ignore[method-assign]

    paused = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Summarize the deployment risks.",
            protocol={"protocol": "teacher_student_distill"},
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                        "judge": "judge-model",
                        "verifier": "verifier-model",
                    }
                },
                "summary": {
                    "evidence_packets": [
                        {
                            "evidence_id": "src-1",
                            "title": "Deployment note",
                            "content": "Only the student summary matches the approved deployment note.",
                        }
                    ]
                },
            },
        )
    )

    assert paused.state == "awaiting_resources"
    assert paused.metadata["recovery_state"]["resume_executor"] == "continue_from_checkpoint"

    resumed = await MultiAgentOrchestrator(engine=engine).run(
        MultiAgentRunRequest(
            run_id=paused.run_id,
            task_input="Summarize the deployment risks.",
            protocol={"protocol": "teacher_student_distill"},
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                        "judge": "judge-model",
                        "verifier": "verifier-model",
                    }
                },
                "summary": {
                    "evidence_packets": [
                        {
                            "evidence_id": "src-1",
                            "title": "Deployment note",
                            "content": "Only the student summary matches the approved deployment note.",
                        }
                    ],
                    "recovery_state": paused.metadata["recovery_state"],
                },
            },
        )
    )

    assert resumed.state == "succeeded"
    assert resumed.selected_candidate_id == "student"
    invoked_models = [call["inference_overrides"]["model"] for call in engine.invoke_calls]
    assert invoked_models.count("teacher-model") == 1
    assert invoked_models.count("student-model") == 1
    assert invoked_models[-1] == "judge-model"




@pytest.mark.asyncio
async def test_orchestrator_resume_reassigns_missing_role_and_exposes_role_task_snapshot() -> None:
    engine = _ConfiguredModelEngineStub()

    result = await MultiAgentOrchestrator(engine=engine).run(
        MultiAgentRunRequest(
            task_input="Summarize the deployment risks.",
            protocol={"protocol": "teacher_student_distill"},
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "judge": "judge-model",
                    }
                },
                "summary": {
                    "recovery_state": {
                        "resume_payload": {
                            "executor": "continue_from_checkpoint",
                            "stage": "student_generation",
                            "checkpoint": {
                                "checkpoint_index": 2,
                                "stage": "student_generation",
                            },
                            "role_task_snapshot": {
                                "roles": {
                                    "teacher": {
                                        "role_id": "teacher",
                                        "status": "completed",
                                        "stage": "teacher_generation",
                                        "assigned_model_id": "teacher-model",
                                        "original_model_id": "teacher-model",
                                        "candidate": {
                                            "candidate_id": "teacher",
                                            "role_id": "teacher",
                                            "content": "Teacher evidence-backed draft.",
                                            "metadata": {"model_id": "teacher-model"},
                                        },
                                    },
                                    "student": {
                                        "role_id": "student",
                                        "status": "running",
                                        "stage": "student_generation",
                                        "assigned_model_id": "student-model",
                                        "original_model_id": "student-model",
                                    },
                                }
                            },
                        }
                    }
                },
            },
        )
    )

    assert result.state == "succeeded"
    snapshot = result.artifacts["role_task_snapshot"]
    assert snapshot["roles"]["teacher"]["status"] == "completed"
    assert snapshot["roles"]["teacher"]["candidate"]["candidate_id"] == "teacher"
    assert snapshot["resume_plan"]["assignments"]["student"]["assigned_model_id"] == "teacher-model"
    assert snapshot["resume_plan"]["assignments"]["student"]["assignment_source"] == "reassigned_to_available_model"
    assert [call["inference_overrides"]["model"] for call in engine.invoke_calls] == [
        "teacher-model",
        "judge-model",
    ]




@pytest.mark.asyncio
async def test_orchestrator_resume_uses_same_task_snapshot_method_for_debate_protocol() -> None:
    engine = _ConfiguredModelEngineStub()

    result = await MultiAgentOrchestrator(engine=engine).run(
        MultiAgentRunRequest(
            task_input="Compare both deployment proposals.",
            protocol={"protocol": "multi_agent_debate", "rounds": 1},
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "debater_a": "debater-a-model",
                        "judge": "judge-model",
                    }
                },
                "summary": {
                    "recovery_state": {
                        "resume_payload": {
                            "executor": "continue_from_checkpoint",
                            "stage": "debate_round_1:debater_b",
                            "checkpoint": {
                                "checkpoint_index": 2,
                                "stage": "debate_round_1:debater_b",
                            },
                            "role_task_snapshot": {
                                "roles": {
                                    "debater_a": {
                                        "role_id": "debater_a",
                                        "status": "completed",
                                        "stage": "debate_round_1:debater_a",
                                        "assigned_model_id": "debater-a-model",
                                        "candidate": {
                                            "candidate_id": "debater_a",
                                            "role_id": "debater_a",
                                            "content": "Argument A prefers the first route.",
                                            "metadata": {"model_id": "debater-a-model"},
                                        },
                                    },
                                    "debater_b": {
                                        "role_id": "debater_b",
                                        "status": "running",
                                        "stage": "debate_round_1:debater_b",
                                        "assigned_model_id": "debater-b-model",
                                        "original_model_id": "debater-b-model",
                                    },
                                },
                                "tasks": {
                                    "debate_round_1:debater_a": {
                                        "task_key": "debate_round_1:debater_a",
                                        "role_id": "debater_a",
                                        "status": "completed",
                                        "stage": "debate_round_1:debater_a",
                                        "assigned_model_id": "debater-a-model",
                                        "candidate": {
                                            "candidate_id": "debater_a",
                                            "role_id": "debater_a",
                                            "content": "Argument A prefers the first route.",
                                            "metadata": {"model_id": "debater-a-model"},
                                        },
                                    }
                                },
                            },
                        }
                    }
                },
            },
        )
    )

    assert result.state == "succeeded"
    snapshot = result.artifacts["role_task_snapshot"]
    assert snapshot["tasks"]["debate_round_1:debater_a"]["status"] == "completed"
    assert snapshot["resume_plan"]["assignments"]["debater_b"]["assigned_model_id"] == "debater-a-model"
    assert snapshot["resume_plan"]["assignments"]["debater_b"]["assignment_source"] == "reassigned_to_available_model"
    assert [call["inference_overrides"]["model"] for call in engine.invoke_calls] == [
        "debater-a-model",
        "judge-model",
    ]




@pytest.mark.asyncio
async def test_orchestrator_resume_reuses_research_tasks_via_shared_task_snapshot() -> None:
    engine = _ConfiguredModelEngineStub()

    result = await MultiAgentOrchestrator(engine=engine).run(
        MultiAgentRunRequest(
            task_input="Compare both deployment proposals.",
            protocol={"protocol": "multi_agent_debate", "rounds": 1},
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "debater_a": "debater-a-model",
                        "debater_b": "debater-b-model",
                        "judge": "judge-model",
                        "verifier": "verifier-model",
                        "planner": "planner-model",
                        "local_worker": "local-worker-model",
                        "synthesizer": "synth-model",
                    }
                },
                "evaluation_policy": {
                    "research": {
                        "enabled": True,
                        "preset": "smart_judge_research_debate",
                        "output_targets": ["research_brief"],
                        "source_mode": "hybrid",
                        "citation_policy": "claim_level_required",
                        "local_worker_count": 2,
                        "max_research_queries": 4,
                        "max_sources_per_query": 3,
                        "debate_rounds": 1,
                    }
                },
                "summary": {
                    "evidence_queries": ["approved deployment note"],
                    "recovery_state": {
                        "resume_payload": {
                            "executor": "continue_from_checkpoint",
                            "stage": "research_synthesis",
                            "checkpoint": {
                                "checkpoint_index": 3,
                                "stage": "research_synthesis",
                            },
                            "metadata_state": {
                                "research_runtime": {
                                    "research_plan": {
                                        "subquestions": ["What does the source say?"],
                                        "evidence_queries": ["approved deployment note"],
                                    },
                                    "source_quality_table": {
                                        "claims": [],
                                        "sources": [],
                                    },
                                    "worker_outputs": {
                                        "worker_count": 2,
                                        "workers": [
                                            {"worker_id": "worker-1", "summary": "verified evidence"},
                                            {"worker_id": "worker-2", "summary": "contradiction note"},
                                        ],
                                    },
                                }
                            },
                            "evidence_packets": [
                                {
                                    "evidence_id": "src-1",
                                    "title": "Deployment note",
                                    "content": "Approved route is supported.",
                                }
                            ],
                            "role_task_snapshot": {
                                "roles": {
                                    "research_planner": {
                                        "role_id": "research_planner",
                                        "status": "completed",
                                        "stage": "research_planning",
                                    },
                                    "research_worker": {
                                        "role_id": "research_worker",
                                        "status": "completed",
                                        "stage": "research_workers",
                                    },
                                    "research_synthesizer": {
                                        "role_id": "research_synthesizer",
                                        "status": "completed",
                                        "stage": "research_synthesis",
                                    },
                                },
                                "tasks": {
                                    "research_planning": {
                                        "task_key": "research_planning",
                                        "role_id": "research_planner",
                                        "status": "completed",
                                        "stage": "research_planning",
                                        "result_summary": {
                                            "research_plan": {
                                                "subquestions": ["What does the source say?"],
                                                "evidence_queries": ["approved deployment note"],
                                            }
                                        },
                                    },
                                    "research_workers": {
                                        "task_key": "research_workers",
                                        "role_id": "research_worker",
                                        "status": "completed",
                                        "stage": "research_workers",
                                        "result_summary": {
                                            "worker_outputs": {
                                                "worker_count": 2,
                                                "workers": [
                                                    {"worker_id": "worker-1", "summary": "verified evidence"},
                                                    {"worker_id": "worker-2", "summary": "contradiction note"},
                                                ],
                                            }
                                        },
                                    },
                                    "research_synthesis": {
                                        "task_key": "research_synthesis",
                                        "role_id": "research_synthesizer",
                                        "status": "completed",
                                        "stage": "research_synthesis",
                                        "result_summary": {
                                            "research_brief": {
                                                "markdown": "# Research Brief\n\nReused brief.",
                                                "summary": "Reused brief.",
                                            }
                                        },
                                    },
                                },
                            },
                        }
                    },
                },
            },
        )
    )

    assert result.state == "succeeded"
    assert result.artifacts["research_brief"]["summary"] == "Reused brief."
    models = [call["inference_overrides"]["model"] for call in engine.invoke_calls]
    assert "planner-model" not in models
    assert "local-worker-model" not in models
    assert "synth-model" not in models




@pytest.mark.asyncio
async def test_orchestrator_resume_reuses_controlled_execution_tasks_via_shared_snapshot() -> None:
    engine = _ConfiguredModelEngineStub()

    result = await MultiAgentOrchestrator(
        engine=engine,
        controlled_exec_require_approval=False,
    ).run(
        MultiAgentRunRequest(
            task_input="Resume the bounded controlled execution workflow.",
            protocol={"protocol": "controlled_subagent_execution"},
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "planner": "controlled-evaluator-model",
                        "judge": "controlled-judge-model",
                        "verifier": "controlled-verifier-model",
                    }
                },
                "summary": {
                    "recovery_state": {
                        "resume_payload": {
                            "executor": "continue_from_checkpoint",
                            "stage": "controlled_execution_evaluator",
                            "checkpoint": {
                                "checkpoint_index": 4,
                                "stage": "controlled_execution_evaluator",
                            },
                            "role_task_snapshot": {
                                "roles": {
                                    "planner": {
                                        "role_id": "planner",
                                        "status": "completed",
                                        "stage": "controlled_execution_planner",
                                        "assigned_model_id": "controlled-planner-model",
                                        "candidate": {
                                            "candidate_id": "planner",
                                            "role_id": "planner",
                                            "content": "Plan: run one safe command and inspect the output.",
                                            "metadata": {"model_id": "controlled-planner-model"},
                                        },
                                    },
                                    "executor": {
                                        "role_id": "executor",
                                        "status": "completed",
                                        "stage": "controlled_execution_executor",
                                        "assigned_model_id": "controlled-executor-model",
                                        "candidate": {
                                            "candidate_id": "executor",
                                            "role_id": "executor",
                                            "content": (
                                                '{"execution_requests":[{"request_id":"req-1",'
                                                '"command":"echo controlled-ok","shell":"powershell",'
                                                '"timeout":30,"background":true,"rationale":"smoke test",'
                                                '"expected_artifacts":["stdout"],'
                                                '"success_metric":"stdout contains controlled-ok"}]}'
                                            ),
                                            "metadata": {"model_id": "controlled-executor-model"},
                                        },
                                    },
                                    "controller": {
                                        "role_id": "controller",
                                        "status": "completed",
                                        "stage": "controlled_execution_controller:req-1",
                                        "assigned_model_id": "controlled-controller-model",
                                    },
                                    "evaluator": {
                                        "role_id": "evaluator",
                                        "status": "running",
                                        "stage": "controlled_execution_evaluator",
                                        "assigned_model_id": "controlled-evaluator-model",
                                        "original_model_id": "controlled-evaluator-model",
                                    },
                                },
                                "tasks": {
                                    "controlled_execution_planner": {
                                        "task_key": "controlled_execution_planner",
                                        "role_id": "planner",
                                        "status": "completed",
                                        "stage": "controlled_execution_planner",
                                        "assigned_model_id": "controlled-planner-model",
                                        "result_summary": {
                                            "planner_output": {
                                                "candidate_id": "planner",
                                                "role_id": "planner",
                                                "content": "Plan: run one safe command and inspect the output.",
                                                "metadata": {"model_id": "controlled-planner-model"},
                                            }
                                        },
                                    },
                                    "controlled_execution_executor": {
                                        "task_key": "controlled_execution_executor",
                                        "role_id": "executor",
                                        "status": "completed",
                                        "stage": "controlled_execution_executor",
                                        "assigned_model_id": "controlled-executor-model",
                                        "result_summary": {
                                            "executor_output": {
                                                "candidate_id": "executor",
                                                "role_id": "executor",
                                                "content": (
                                                    '{"execution_requests":[{"request_id":"req-1",'
                                                    '"command":"echo controlled-ok","shell":"powershell",'
                                                    '"timeout":30,"background":true,"rationale":"smoke test",'
                                                    '"expected_artifacts":["stdout"],'
                                                    '"success_metric":"stdout contains controlled-ok"}]}'
                                                ),
                                                "metadata": {"model_id": "controlled-executor-model"},
                                            },
                                            "execution_requests": [
                                                {
                                                    "request_id": "req-1",
                                                    "command": "echo controlled-ok",
                                                    "shell": "powershell",
                                                    "timeout": 30,
                                                    "background": True,
                                                    "rationale": "smoke test",
                                                    "expected_artifacts": ["stdout"],
                                                    "success_metric": "stdout contains controlled-ok",
                                                }
                                            ],
                                            "request_parse_diagnostics": {
                                                "status": "parsed",
                                                "parsed_request_count": 1,
                                                "max_execution_requests": 1,
                                                "reason": None,
                                            },
                                        },
                                    },
                                    "controlled_execution_controller:req-1": {
                                        "task_key": "controlled_execution_controller:req-1",
                                        "role_id": "controller",
                                        "status": "completed",
                                        "stage": "controlled_execution_controller:req-1",
                                        "assigned_model_id": "controlled-controller-model",
                                        "result_summary": {
                                            "controller_decision": {
                                                "decision_id": "controller-decision-1",
                                                "request_id": "req-1",
                                                "status": "approved",
                                                "reason": "bounded smoke command",
                                                "command": "echo controlled-ok",
                                                "shell": "powershell",
                                                "timeout": 30,
                                                "background": True,
                                                "raw_content": (
                                                    '{"status":"approved","reason":"bounded smoke command",'
                                                    '"command":"echo controlled-ok","shell":"powershell",'
                                                    '"timeout":30,"background":true}'
                                                ),
                                                "diagnostics": {},
                                            }
                                        },
                                    },
                                    "controlled_execution_exec:req-1": {
                                        "task_key": "controlled_execution_exec:req-1",
                                        "role_id": "controller",
                                        "status": "completed",
                                        "stage": "controlled_execution_exec:req-1",
                                        "assigned_model_id": "controlled-controller-model",
                                        "result_summary": {
                                            "execution_result": {
                                                "request_id": "req-1",
                                                "status": "completed",
                                                "command": "echo controlled-ok",
                                                "shell": "powershell",
                                                "workdir": None,
                                                "timeout": 30,
                                                "background": True,
                                                "log_path": "H:/tmp/req-1/session.log",
                                                "session_log_path": "H:/tmp/req-1/session.log",
                                                "checkpoint_dir": "H:/tmp/req-1/checkpoints",
                                                "root_dir": "H:/tmp/req-1",
                                                "manifest_path": "H:/tmp/req-1/manifest.json",
                                                "stdout_log_path": "H:/tmp/req-1/stdout.log",
                                                "stderr_log_path": "H:/tmp/req-1/stderr.log",
                                                "stdout": "controlled-ok",
                                                "stderr": "",
                                                "error": None,
                                                "metadata": {
                                                    "session_id": "detached-req-1",
                                                    "status": "completed",
                                                    "detached": True,
                                                    "recovery_supported": True,
                                                    "detached_layout": {
                                                        "root_dir": "H:/tmp/req-1",
                                                        "log_path": "H:/tmp/req-1/session.log",
                                                        "session_log_path": "H:/tmp/req-1/session.log",
                                                        "checkpoint_dir": "H:/tmp/req-1/checkpoints",
                                                        "manifest_path": "H:/tmp/req-1/manifest.json",
                                                        "stdout_log_path": "H:/tmp/req-1/stdout.log",
                                                        "stderr_log_path": "H:/tmp/req-1/stderr.log",
                                                        "runtime_state_root": "H:/_python/agent_mochi/.mochi/exec-runtime",
                                                    },
                                                },
                                            }
                                        },
                                    },
                                },
                            },
                        }
                    }
                },
            },
        )
    )

    assert result.state == "succeeded"
    assert result.selected_candidate_id == "evaluator"
    assert result.artifacts["execution_requests"]["items"][0]["request_id"] == "req-1"
    assert result.artifacts["controller_decisions"]["items"][0]["status"] == "approved"
    assert result.artifacts["execution_results"]["items"][0]["metadata"]["session_id"] == "detached-req-1"
    assert result.artifacts["detached_exec_jobs"]["count"] == 1
    assert result.artifacts["controlled_execution_runtime"]["detached_exec_job_count"] == 1

    snapshot = result.artifacts["role_task_snapshot"]
    assert snapshot["tasks"]["controlled_execution_planner"]["status"] == "completed"
    assert snapshot["tasks"]["controlled_execution_executor"]["status"] == "completed"
    assert snapshot["tasks"]["controlled_execution_controller:req-1"]["status"] == "completed"
    assert snapshot["tasks"]["controlled_execution_exec:req-1"]["status"] == "completed"
    assert snapshot["tasks"]["controlled_execution_evaluator"]["status"] == "completed"
    assert snapshot["resume_plan"]["assignments"]["evaluator"]["assigned_model_id"] == "controlled-evaluator-model"
    assert snapshot["resume_plan"]["assignments"]["evaluator"]["assignment_source"] == "reassigned_to_available_model"

    models = [call["inference_overrides"]["model"] for call in engine.invoke_calls]
    assert "controlled-planner-model" not in models
    assert "controlled-executor-model" not in models
    assert "controlled-controller-model" not in models
    assert models.count("controlled-evaluator-model") == 1




@pytest.mark.asyncio
async def test_orchestrator_resume_reuses_saved_verification_and_evaluation_state() -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(engine=engine)
    original = orchestrator._raise_if_run_policy_exhausted

    def _pause_after_evaluation(**kwargs: object) -> None:
        original(**kwargs)
        if kwargs.get("stage") == "evaluation_completed":
            raise RunPolicyStop(
                status="awaiting_resources",
                action="pause",
                reason="pause after evaluation for resume test",
                stage="evaluation_completed",
                checkpoint=orchestrator._latest_checkpoint,
            )

    orchestrator._raise_if_run_policy_exhausted = _pause_after_evaluation  # type: ignore[method-assign]

    paused = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Summarize the deployment risks.",
            protocol={"protocol": "teacher_student_distill"},
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                        "judge": "judge-model",
                        "verifier": "verifier-model",
                    }
                },
                "summary": {
                    "evidence_packets": [
                        {
                            "evidence_id": "src-1",
                            "title": "Deployment note",
                            "content": "Only the student summary matches the approved deployment note.",
                        }
                    ]
                },
            },
        )
    )

    assert paused.state == "awaiting_resources"
    assert paused.metadata["recovery_state"]["resume_payload"]["evaluation"]["selected_candidate_id"] == "student"

    resumed = await MultiAgentOrchestrator(engine=engine).run(
        MultiAgentRunRequest(
            run_id=paused.run_id,
            task_input="Summarize the deployment risks.",
            protocol={"protocol": "teacher_student_distill"},
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                        "judge": "judge-model",
                        "verifier": "verifier-model",
                    }
                },
                "summary": {
                    "evidence_packets": [
                        {
                            "evidence_id": "src-1",
                            "title": "Deployment note",
                            "content": "Only the student summary matches the approved deployment note.",
                        }
                    ],
                    "recovery_state": paused.metadata["recovery_state"],
                },
            },
        )
    )

    assert resumed.state == "succeeded"
    assert resumed.selected_candidate_id == "student"
    assert resumed.artifacts["verification"]["verified_candidate_ids"] == ["student"]
    assert [call["inference_overrides"]["model"] for call in engine.invoke_calls] == [
        "teacher-model",
        "student-model",
        "verifier-model",
        "judge-model",
    ]




@pytest.mark.asyncio
async def test_orchestrator_resume_falls_back_to_restart_when_checkpoint_payload_is_incomplete() -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(engine=engine)

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Summarize the deployment risks.",
            protocol={"protocol": "teacher_student_distill"},
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                        "judge": "judge-model",
                    }
                },
                "summary": {
                    "recovery_state": {
                        "resume_payload": {
                            "executor": "continue_from_checkpoint",
                            "stage": "protocol_completed",
                            "checkpoint": {
                                "checkpoint_index": 3,
                                "stage": "protocol_completed",
                            },
                            "candidates": [],
                        }
                    }
                },
            },
        )
    )

    assert result.state == "succeeded"
    assert [call["inference_overrides"]["model"] for call in engine.invoke_calls] == [
        "teacher-model",
        "student-model",
        "judge-model",
    ]




@pytest.mark.asyncio
async def test_orchestrator_marks_run_stalled_on_heartbeat_timeout() -> None:
    engine = _TeacherHeartbeatTimeoutEngine()
    orchestrator = MultiAgentOrchestrator(engine=engine)

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Summarize the deployment note.",
            protocol={"protocol": "teacher_student_distill"},
            run_policy={
                "heartbeat_timeout_sec": 1,
                "max_subagent_failures_per_role": 0,
                "on_subagent_disconnect": "pause",
            },
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                    }
                }
            },
        )
    )

    assert result.state == "stalled"
    assert result.metadata["recovery_state"]["status"] == "stalled"
    assert result.metadata["recovery_state"]["action"] == "pause"
    snapshot = result.artifacts["subagent_health_snapshot"]
    assert snapshot["failure_counts"]["teacher"] == 1
