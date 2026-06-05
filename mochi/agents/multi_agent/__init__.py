"""Multi-agent orchestration scaffolding exports。"""

from __future__ import annotations

from mochi.agents.multi_agent.evaluator import (
    CandidateScore,
    EvidenceGateResult,
    LLMFirstScoringPolicy,
    resolve_evidence_gate_status,
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
    MultiAgentDebateProtocol,
    ProtocolConfig,
    ProtocolName,
    TeacherStudentDistillProtocol,
    parse_protocol_config,
)
from mochi.agents.multi_agent.roles import (
    AgentRoleProfile,
    build_multi_agent_debate_roles,
    build_teacher_student_roles,
)

__all__ = [
    "AgentRoleProfile",
    "BoundedRunStateMachine",
    "CandidateOutput",
    "CandidateScore",
    "EvidenceGateResult",
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
    "TeacherStudentDistillProtocol",
    "build_multi_agent_debate_roles",
    "build_teacher_student_roles",
    "parse_protocol_config",
    "resolve_evidence_gate_status",
]
