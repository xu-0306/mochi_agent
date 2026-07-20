from __future__ import annotations

import pytest

from mochi.agents.multi_agent.orchestrator import MultiAgentOrchestrator, MultiAgentRunRequest

from ._support import (
    _ConfiguredModelEngineStub,
    _DebaterBFailureEngine,
    _GenerateOnlyEngineStub,
)


@pytest.mark.asyncio
async def test_orchestrator_smoke_emits_state_and_guidance_events() -> None:
    orchestrator = MultiAgentOrchestrator()
    request = MultiAgentRunRequest(
        task_input="Summarize this note",
        protocol={"protocol": "teacher_student_distill"},
        guidance_messages=["Prefer concise answers."],
    )

    result = await orchestrator.run(request)
    event_types = [event.type for event in result.events]

    assert result.state == "succeeded"
    assert event_types[0] == "state_changed"
    assert "guidance" in event_types
    assert "role_output" in event_types
    assert "evaluation" in event_types
    assert event_types[-1] == "state_changed"




@pytest.mark.asyncio
async def test_orchestrator_finalize_partial_when_goal_checkpoint_step_cadence_reached() -> None:
    engine = _ConfiguredModelEngineStub()

    result = await MultiAgentOrchestrator(engine=engine).run(
        MultiAgentRunRequest(
            task_input="Summarize this deployment note.",
            protocol={"protocol": "teacher_student_distill"},
            run_policy={
                "checkpoint_interval_steps": 2,
                "goal_checkpoint_cadence_owned": True,
                "goal_checkpoint_cadence_effective_steps": 2,
                "on_budget_exhausted": "finalize_partial",
            },
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                        "judge": "judge-model",
                    }
                }
            },
        )
    )

    assert result.state == "partial"
    recovery_state = dict(result.metadata["recovery_state"])
    assert recovery_state["status"] == "partial"
    assert recovery_state["action"] == "finalize_partial"
    assert recovery_state["stage"] == "protocol_completed"
    assert recovery_state["checkpoint"]["checkpoint_index"] == 2
    assert recovery_state["reason"] == "Run reached checkpoint_interval_steps=2 at checkpoint_index=2."
    assert [call["inference_overrides"]["model"] for call in engine.invoke_calls] == [
        "teacher-model",
        "student-model",
    ]


@pytest.mark.asyncio
async def test_orchestrator_uses_configured_models_for_teacher_student_protocol() -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(engine=engine)

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Summarize the deployment risks.",
            protocol={"protocol": "teacher_student_distill"},
            guidance_messages=["Ground every claim."],
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                        "judge": "judge-model",
                    },
                    "subagents": [
                        {"role": "teacher", "model_id": "teacher-model"},
                        {"role": "student", "model_id": "student-model"},
                        {"role": "judge", "model_id": "judge-model"},
                    ],
                }
            },
        )
    )

    assert result.selected_candidate_id == "student"
    assert result.artifacts["final_answer"] == "Student concise final answer."
    assert [call["inference_overrides"]["model"] for call in engine.invoke_calls] == [
        "teacher-model",
        "student-model",
        "judge-model",
    ]
    assert engine.calls == []
    assert engine.invoke_calls[0]["execution_profile"] == "subagent_readonly"
    assert engine.invoke_calls[0]["tool_mode"] == "auto"
    assert "Role identity: Teacher" in str(engine.invoke_calls[0]["system_prompt_addendum"])
    assert "Teacher evidence-backed draft." in str(engine.invoke_calls[1]["message"])
    teacher_event_types = [
        event.type
        for event in result.events
        if event.payload.get("role_id") == "teacher"
    ]
    assert "role_started" in teacher_event_types
    assert "role_completed" in teacher_event_types
    assert "role_output" in teacher_event_types
    diagnostics = result.candidates[0].metadata["diagnostics"]
    assert diagnostics["execution_profile"] == "subagent_readonly"
    assert diagnostics["tool_mode"] == "auto"
    assert diagnostics["exposed_tools"] == ["file_read"]
    runtime = result.artifacts["subagent_runtime"]
    assert runtime["execution_boundary"] == "multi_agent_subagents_research_only_main_agent_executes_code_and_training"
    assert runtime["tool_event_count"] == 2
    assert runtime["approval_pending_count"] == 0
    assert runtime["risky_tool_event_count"] == 0
    assert runtime["invocations"][0]["tool_events"][0]["tool_name"] == "file_read"




@pytest.mark.asyncio
async def test_orchestrator_passes_goal_capability_tool_allowlist_to_engine() -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(engine=engine)

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Summarize the deployment risks.",
            protocol={"protocol": "teacher_student_distill"},
            metadata={
                "goal_capability_policy": {
                    "allowed_tools": [" file_read ", "web_search", "file_read", ""],
                },
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                        "judge": "judge-model",
                    }
                },
            },
        )
    )

    assert result.state == "succeeded"
    assert len(engine.invoke_calls) == 3
    assert all(
        call["tool_allowlist"] == ["file_read", "web_search"]
        for call in engine.invoke_calls
    )
    assert all(call["tool_denylist"] is None for call in engine.invoke_calls)




@pytest.mark.asyncio
async def test_orchestrator_passes_goal_operator_tool_denylist_to_engine() -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(engine=engine)

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Summarize the deployment risks.",
            protocol={"protocol": "teacher_student_distill"},
            metadata={
                "goal_operator_controls": {
                    "tool_denylist": [" web_fetch ", "web_search", "web_fetch", ""],
                },
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                        "judge": "judge-model",
                    }
                },
            },
        )
    )

    assert result.state == "succeeded"
    assert len(engine.invoke_calls) == 3
    assert all(
        call["tool_denylist"] == ["web_fetch", "web_search"]
        for call in engine.invoke_calls
    )
    assert all(call["tool_allowlist"] is None for call in engine.invoke_calls)




@pytest.mark.asyncio
async def test_orchestrator_derives_blocked_web_domain_policy_for_engine_and_evidence() -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(engine=engine)

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Summarize the deployment risks.",
            protocol={"protocol": "teacher_student_distill"},
            metadata={
                "permission_policy": {
                    "blocked_web_domains": ["preexisting.test"],
                },
                "goal_operator_controls": {
                    "blocked_domains": [
                        " Example.com ",
                        "blocked.example.org",
                        "example.com",
                    ],
                },
                "selected_models_roles": {
                    "by_role": {
                        "teacher": "teacher-model",
                        "student": "student-model",
                        "judge": "judge-model",
                        "verifier": "verifier-model",
                    }
                },
                "summary": {
                    "evidence_queries": ["approved deployment note"],
                },
            },
        )
    )

    assert result.state == "succeeded"
    assert len(engine.invoke_calls) == 4
    assert all(
        call["permission_policy"] == {
            "blocked_web_domains": [
                "preexisting.test",
                "example.com",
                "blocked.example.org",
            ],
        }
        for call in engine.invoke_calls
    )
    assert len(engine.evidence_calls) == 1
    assert engine.evidence_calls[0]["metadata"]["permission_policy"] == {
        "blocked_web_domains": [
            "preexisting.test",
            "example.com",
            "blocked.example.org",
        ],
    }




@pytest.mark.asyncio
async def test_orchestrator_role_candidate_fallback_records_diagnostics() -> None:
    engine = _GenerateOnlyEngineStub()
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
                }
            },
        )
    )

    assert result.selected_candidate_id == "student"
    assert engine.invoke_calls == []
    assert [call["model_id"] for call in engine.calls] == [
        "teacher-model",
        "student-model",
        "judge-model",
    ]
    teacher_diagnostics = result.candidates[0].metadata["diagnostics"]
    assert teacher_diagnostics["fallback_reason"] == "invoke_unavailable_used_generate_with_configured_model"




@pytest.mark.asyncio
async def test_orchestrator_uses_configured_models_for_dr_zero_protocol() -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(engine=engine)

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Self-evolve deployment search tasks.",
            protocol={
                "protocol": "dr_zero_self_evolve",
                "proposal_sample_size": 2,
                "solver_rollouts_per_task": 1,
            },
            guidance_messages=["Prefer verifiable tasks."],
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "proposer": "proposer-model",
                        "solver": "solver-model",
                        "verifier": "verifier-model",
                        "judge": "judge-model",
                    }
                },
                "summary": {
                    "evidence_packets": [
                        {
                            "evidence_id": "src-1",
                            "title": "Deployment note",
                            "content": "Evidence supports solver.",
                        }
                    ]
                },
            },
        )
    )

    assert result.protocol == "dr_zero_self_evolve"
    assert result.selected_candidate_id == "solver_1_1"
    assert result.artifacts["synthetic_tasks"]["parse_diagnostics"]["status"] == "parsed"
    assert len(result.artifacts["synthetic_tasks"]["tasks"]) == 2
    assert result.artifacts["solver_rollouts"]["rollout_count"] == 2
    assert result.artifacts["reward_summary"]["status"] == "pending_verification"
    assert result.artifacts["curriculum_state"]["iterations_configured"] == 1
    assert result.artifacts["drzero_iteration_summary"]["proposal_parse_status"] == "parsed"
    assert [call["inference_overrides"]["model"] for call in engine.invoke_calls] == [
        "proposer-model",
        "solver-model",
        "solver-model",
        "verifier-model",
        "judge-model",
    ]
    assert all(candidate.role_id == "solver" for candidate in result.candidates)
    assert result.candidates[0].metadata["synthetic_task_id"] == "task-1"




@pytest.mark.asyncio
async def test_orchestrator_role_guidance_stays_isolated_across_debate_roles() -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(engine=engine)
    directive = "Verifier-only directive."

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Assess the deployment summary for evidence-backed release guidance.",
            protocol={"protocol": "multi_agent_debate"},
            guidance_messages=["Keep responses concise."],
            role_guidance_messages={"verifier": [directive]},
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
                        "local_worker_count": 1,
                        "max_research_queries": 2,
                        "max_sources_per_query": 2,
                        "debate_rounds": 1,
                    }
                },
                "summary": {
                    "evidence_queries": ["approved deployment note"],
                    "evidence_packets": [
                        {
                            "evidence_id": "src-1",
                            "title": "Deployment note",
                            "content": "Approved route is supported.",
                        }
                    ],
                },
            },
        )
    )

    assert result.state == "succeeded"
    verifier_prompts = [
        str(call["message"])
        for call in engine.invoke_calls
        if call["inference_overrides"]["model"] == "verifier-model"
    ]
    assert any(directive in prompt for prompt in verifier_prompts)
    for model_id in {"planner-model", "debater-a-model", "debater-b-model", "judge-model"}:
        assert all(directive not in str(call["message"]) for call in engine.invoke_calls if call["inference_overrides"]["model"] == model_id)




@pytest.mark.asyncio
async def test_orchestrator_role_guidance_stays_isolated_across_dr_zero_solver_roles() -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(engine=engine)
    directive = "Verifier-only directive."

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Self-evolve deployment search tasks.",
            protocol={"protocol": "dr_zero_self_evolve", "proposal_sample_size": 1, "solver_rollouts_per_task": 1},
            guidance_messages=["Prefer verifiable tasks."],
            role_guidance_messages={"verifier": [directive]},
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "proposer": "proposer-model",
                        "solver": "solver-model",
                        "verifier": "verifier-model",
                        "judge": "judge-model",
                    }
                },
                "summary": {
                    "evidence_packets": [
                        {
                            "evidence_id": "src-1",
                            "title": "Deployment note",
                            "content": "Evidence supports solver.",
                        }
                    ]
                },
            },
        )
    )

    assert result.state == "succeeded"
    verifier_prompts = [
        str(call["message"])
        for call in engine.invoke_calls
        if call["inference_overrides"]["model"] == "verifier-model"
    ]
    assert any(directive in prompt for prompt in verifier_prompts)
    for model_id in {"proposer-model", "solver-model", "judge-model"}:
        assert all(directive not in str(call["message"]) for call in engine.invoke_calls if call["inference_overrides"]["model"] == model_id)




@pytest.mark.asyncio
async def test_orchestrator_dr_zero_falls_back_when_proposer_json_is_unparseable() -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(engine=engine)

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Create a fallback synthetic task.",
            protocol={"protocol": "dr_zero_self_evolve"},
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "proposer": "teacher-model",
                        "solver": "solver-model",
                        "judge": "judge-model",
                    }
                }
            },
        )
    )

    assert result.protocol == "dr_zero_self_evolve"
    assert result.artifacts["synthetic_tasks"]["parse_diagnostics"]["status"] == "fallback"
    assert result.artifacts["synthetic_tasks"]["tasks"][0]["metadata"]["fallback"] is True
    assert result.candidates[0].metadata["synthetic_task_id"] == "task-1"




@pytest.mark.asyncio
async def test_orchestrator_prefers_verified_candidate_and_emits_verification_outputs() -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(engine=engine)

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Summarize the deployment risks.",
            protocol={"protocol": "teacher_student_distill"},
            guidance_messages=["Ground every claim."],
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

    assert result.selected_candidate_id == "student"
    assert result.artifacts["verification"]["verified_candidate_ids"] == ["student"]
    assert result.artifacts["verification"]["failed_candidate_ids"] == ["teacher"]
    assert any(event.type == "verification" for event in result.events)
    scores_by_candidate = {
        score["candidate_id"]: score
        for score in result.evaluation["scores"]
    }
    assert scores_by_candidate["teacher"]["evidence_gate"]["status"] == "failed"
    assert scores_by_candidate["student"]["evidence_gate"]["status"] == "verified"
    assert [call["inference_overrides"]["model"] for call in engine.invoke_calls] == [
        "teacher-model",
        "student-model",
        "verifier-model",
        "judge-model",
    ]
    assert engine.calls == []




@pytest.mark.asyncio
async def test_orchestrator_collects_evidence_queries_before_verification() -> None:
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
                        "verifier": "verifier-model",
                    }
                },
                "summary": {
                    "evidence_queries": ["approved deployment note"],
                },
            },
        )
    )

    assert result.selected_candidate_id == "student"
    assert len(engine.evidence_calls) == 1
    assert engine.evidence_calls[0]["queries"] == ["approved deployment note"]
    assert engine.evidence_calls[0]["metadata"]["selected_models_roles"] == result.metadata["selected_models_roles"]
    assert result.artifacts["evidence_collection"]["collected_packet_count"] == 1
    assert result.artifacts["evidence_collection"]["mode"] == "hybrid"
    assert result.artifacts["verification"]["evidence_packet_count"] == 1
    artifact_events = [event for event in result.events if event.type == "artifact"]
    assert any(event.payload["name"] == "evidence_summary" for event in artifact_events)




@pytest.mark.asyncio
async def test_orchestrator_does_not_collect_evidence_when_policy_disables_it() -> None:
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
                        "verifier": "verifier-model",
                    }
                },
                "summary": {
                    "evidence_queries": ["approved deployment note"],
                },
                "evaluation_policy": {
                    "evidence_collection": {
                        "enabled": False,
                    }
                },
            },
        )
    )

    assert result.selected_candidate_id == "student"
    assert engine.evidence_calls == []
    assert result.artifacts["evidence_collection"]["status"] == "disabled"




@pytest.mark.asyncio
async def test_orchestrator_uses_configured_models_for_multi_agent_debate_protocol() -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(engine=engine)

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Choose the safer migration strategy.",
            protocol={"protocol": "multi_agent_debate"},
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "debater_a": "debater-a-model",
                        "debater_b": "debater-b-model",
                        "judge": "judge-model",
                    }
                }
            },
        )
    )

    assert result.selected_candidate_id == "student" or result.selected_candidate_id == "debater_b"
    assert [call["inference_overrides"]["model"] for call in engine.invoke_calls] == [
        "debater-a-model",
        "debater-b-model",
        "debater-a-model",
        "debater-b-model",
        "judge-model",
    ]
    assert engine.calls == []
    debate_events = [event for event in result.events if event.type == "role_output"]
    assert len(debate_events) == 4
    assert result.artifacts["debate_state"]["claim_cards"]
    assert result.artifacts["debate_context_snapshot"]["snapshots"]




@pytest.mark.asyncio
async def test_orchestrator_builds_research_debate_artifacts_and_honors_output_modes() -> None:
    engine = _ConfiguredModelEngineStub()
    orchestrator = MultiAgentOrchestrator(engine=engine)

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Assess the deployment summary for evidence-backed release guidance.",
            protocol={"protocol": "multi_agent_debate"},
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
                        "debate_rounds": 2,
                    }
                },
                "summary": {
                    "evidence_queries": ["approved deployment note"],
                },
            },
        )
    )

    assert "research_plan" in result.artifacts
    assert "source_quality_table" in result.artifacts
    assert "claim_evidence_map" in result.artifacts
    assert "research_brief" in result.artifacts
    assert result.artifacts["research_plan"]["worker_outputs"]["worker_count"] == 2
    assert result.artifacts["claim_evidence_map"]["claims"]
    assert result.artifacts["debate_state"]["claim_cards"][0]["support_status"] in {
        "supported",
        "contested",
        "needs_evidence",
        "refuted",
    }




@pytest.mark.asyncio
async def test_orchestrator_degrades_when_one_debate_role_fails() -> None:
    engine = _DebaterBFailureEngine()
    orchestrator = MultiAgentOrchestrator(engine=engine)

    result = await orchestrator.run(
        MultiAgentRunRequest(
            task_input="Compare two deployment approaches.",
            protocol={"protocol": "multi_agent_debate", "rounds": 2},
            run_policy={
                "max_subagent_failures_per_role": 0,
                "on_subagent_disconnect": "retry_then_degrade",
            },
            metadata={
                "selected_models_roles": {
                    "by_role": {
                        "debater_a": "debater-a-model",
                        "debater_b": "debater-b-model",
                        "judge": "judge-model",
                        "verifier": "verifier-model",
                    }
                }
            },
        )
    )

    assert result.state == "succeeded"
    assert result.metadata["degraded"] is True
    assert [candidate.role_id for candidate in result.candidates] == ["debater_a"]
    snapshot = result.artifacts["subagent_health_snapshot"]
    assert snapshot["degraded"] is True
    assert snapshot["degraded_role_ids"] == ["debater_b"]
    assert snapshot["failure_counts"]["debater_b"] == 1
