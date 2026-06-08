"""Multi-agent orchestration scaffolding exports。"""

from __future__ import annotations

from mochi.agents.multi_agent.evaluator import (
    CandidateScore,
    EvidenceGateResult,
    LLMFirstScoringPolicy,
    resolve_evidence_gate_status,
)
from mochi.agents.multi_agent.execution_coordinator import SubagentExecutionCoordinator
from mochi.agents.multi_agent.execution_policy import (
    LEGACY_CONTROLLED_SUBAGENT_PROTOCOL,
    SubagentExecutionPolicy,
    execution_policy_to_dict,
    parse_subagent_execution_policy,
    run_uses_controlled_execution,
)
from mochi.agents.multi_agent.orchestrator import (
    BoundedRunStateMachine,
    CandidateOutput,
    MultiAgentOrchestrator,
    MultiAgentRunEvent,
    MultiAgentRunRequest,
    MultiAgentRunResult,
    RunState,
    RunStateTransitionError,
)
from mochi.agents.multi_agent.protocols import (
    ControlledSubagentExecutionProtocol,
    DrZeroSelfEvolveProtocol,
    MultiAgentDebateProtocol,
    ProtocolConfig,
    ProtocolName,
    TeacherStudentDistillProtocol,
    parse_protocol_config,
)
from mochi.agents.multi_agent.roles import (
    AgentRoleProfile,
    build_controlled_execution_roles,
    build_dr_zero_roles,
    build_multi_agent_debate_roles,
    build_teacher_student_roles,
)

__all__ = [
    "AgentRoleProfile",
    "BoundedRunStateMachine",
    "CandidateOutput",
    "CandidateScore",
    "ControlledSubagentExecutionProtocol",
    "LEGACY_CONTROLLED_SUBAGENT_PROTOCOL",
    "EvidenceGateResult",
    "DrZeroSelfEvolveProtocol",
    "LLMFirstScoringPolicy",
    "MultiAgentDebateProtocol",
    "MultiAgentOrchestrator",
    "MultiAgentRunEvent",
    "MultiAgentRunRequest",
    "MultiAgentRunResult",
    "ProtocolConfig",
    "ProtocolName",
    "RunState",
    "RunStateTransitionError",
    "SubagentExecutionCoordinator",
    "SubagentExecutionPolicy",
    "TeacherStudentDistillProtocol",
    "build_dr_zero_roles",
    "build_controlled_execution_roles",
    "build_multi_agent_debate_roles",
    "build_teacher_student_roles",
    "execution_policy_to_dict",
    "parse_protocol_config",
    "parse_subagent_execution_policy",
    "resolve_evidence_gate_status",
    "run_uses_controlled_execution",
]
