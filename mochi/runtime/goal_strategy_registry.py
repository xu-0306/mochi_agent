"""Read-only registry of Goal execution strategies."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from threading import RLock
from typing import Iterator, Literal


GoalStrategyKind = Literal["protocol", "workflow_template", "execution_strategy"]
GoalStrategyExecutionTopology = Literal["single_agent", "multi_agent"]

DEFAULT_GOAL_STRATEGY_ID = "autonomous_single_agent"


@dataclass(frozen=True, slots=True)
class GoalStrategyRegistryEntryData:
    """Backend-owned description of a strategy selectable under a Goal."""

    id: str
    name: str
    display_name: str
    description: str
    when_to_use: str
    when_not_to_use: str
    execution_topology: GoalStrategyExecutionTopology
    kind: GoalStrategyKind = "execution_strategy"
    protocol_id: str | None = None
    required_capabilities: tuple[str, ...] = ()
    approval_profile: str = "standard_goal_policy"
    control_scope: str = "goal"
    interrupt_policy: str = "Can pause, resume, cancel, and steer through the Goal runtime."
    resume_policy: str = "Resume from the persisted Goal attempt and latest runtime checkpoint when available."
    event_contract: str = "Emits Goal and AgentRun events suitable for durable status and execution projection."
    success_signals: tuple[str, ...] = ()
    failure_modes: tuple[str, ...] = ()
    fallback_strategy_ids: tuple[str, ...] = ()
    requires_confirmation: bool = False
    is_default: bool = False
    available: bool = True
    availability_reason: str | None = None
    deprecated: bool = False
    override_label: str | None = None
    selection_guidance: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


_PRODUCTION_ENTRIES: tuple[GoalStrategyRegistryEntryData, ...] = (
    GoalStrategyRegistryEntryData(
        id="autonomous_single_agent",
        name="Autonomous single agent",
        display_name="Autonomous single agent",
        description=(
            "The ordinary Goal strategy: one autonomous worker owns the task, uses available tools, "
            "and keeps the durable Goal state updated until completion or a real blocker."
        ),
        when_to_use=(
            "Use for most Goals where a single capable agent can plan, execute, ask for approvals, "
            "and report progress without a specialized multi-agent protocol."
        ),
        when_not_to_use=(
            "Do not use when the user explicitly asks for structured debate, independent competing "
            "answers, or another specialized protocol."
        ),
        execution_topology="single_agent",
        protocol_id="autonomous_single_agent",
        required_capabilities=("tool_use", "checkpointing", "goal_resume"),
        success_signals=("Goal status becomes completed.", "Final answer or artifact is attached to the Goal."),
        failure_modes=("Tool approval blocked.", "Selected model role is unavailable.", "Runtime checkpoint expires."),
        fallback_strategy_ids=(),
        is_default=True,
        available=True,
        availability_reason=None,
        override_label="Single agent",
        selection_guidance="Select as the safe default when no stronger strategy is justified.",
    ),
    GoalStrategyRegistryEntryData(
        id="multi_agent_debate",
        name="Multi-agent debate",
        display_name="Multi-agent debate",
        description=(
            "Runs competing debater roles and a judge role so the Goal can compare alternatives before "
            "settling on an answer or recommendation."
        ),
        when_to_use=(
            "Use when the user explicitly wants tradeoff analysis, adversarial comparison, judgment "
            "between plausible options, or a debate-style review."
        ),
        when_not_to_use=(
            "Do not use for ordinary execution, simple research, or tasks where extra agent roles would "
            "add ceremony without improving the result."
        ),
        execution_topology="multi_agent",
        kind="protocol",
        protocol_id="multi_agent_debate",
        required_capabilities=("multi_agent_orchestration", "judge_role"),
        approval_profile="standard_goal_policy",
        interrupt_policy="Can pause, resume, cancel, and steer the debate through the Goal runtime.",
        resume_policy="Resume the active debate attempt and preserve role outputs when available.",
        event_contract="Emits debater, judge, and Goal lifecycle events for execution projection.",
        success_signals=("Judge selects or synthesizes a final answer.", "Competing positions are recorded."),
        failure_modes=("Role execution failed.", "Judge output is missing.", "Tool approval blocked."),
        fallback_strategy_ids=("autonomous_single_agent",),
        available=True,
        availability_reason=None,
        override_label="Debate",
        selection_guidance="Select only for explicit or semantically strong comparison/debate needs.",
    ),
    GoalStrategyRegistryEntryData(
        id="teacher_student_distill",
        name="Teacher-student distillation",
        display_name="Teacher-student distillation",
        description=(
            "A specialized protocol where a teacher role produces guidance and a student role distills "
            "or applies it. It is supported for explicit legacy and specialized workflows."
        ),
        when_to_use=(
            "Use only when the user explicitly selects distillation or the task truly needs teacher/student "
            "generation and compression."
        ),
        when_not_to_use=(
            "Do not use as a hidden fallback for ordinary Goals; default Goals should use autonomous_single_agent."
        ),
        execution_topology="multi_agent",
        kind="protocol",
        protocol_id="teacher_student_distill",
        required_capabilities=("multi_agent_orchestration", "teacher_role", "student_role"),
        approval_profile="standard_goal_policy",
        interrupt_policy="Can pause, resume, cancel, and steer through the Goal runtime.",
        resume_policy="Resume the distillation attempt when persisted role outputs are available.",
        event_contract="Emits teacher, student, and Goal lifecycle events for execution projection.",
        success_signals=("Student output is produced.", "Distilled final answer is attached to the Goal."),
        failure_modes=("Teacher output is missing.", "Student output is missing.", "Tool approval blocked."),
        fallback_strategy_ids=("autonomous_single_agent",),
        requires_confirmation=True,
        available=True,
        availability_reason="Available for explicit legacy and specialized distillation workflows; not a default.",
        override_label="Distill",
        selection_guidance="Specialized, non-default strategy. Never select as a safe default.",
    ),
)

_test_entries: tuple[GoalStrategyRegistryEntryData, ...] = ()
_test_entries_lock = RLock()


def list_goal_strategy_entries() -> tuple[GoalStrategyRegistryEntryData, ...]:
    """Return deterministic registry entries, including any test-only injected entries."""

    with _test_entries_lock:
        return (*_PRODUCTION_ENTRIES, *_test_entries)


def get_goal_strategy_entry(strategy_id: str) -> GoalStrategyRegistryEntryData | None:
    normalized_id = str(strategy_id or "").strip()
    if not normalized_id:
        return None
    for entry in list_goal_strategy_entries():
        if entry.id == normalized_id:
            return entry
    return None


def default_goal_strategy_entry() -> GoalStrategyRegistryEntryData:
    entry = get_goal_strategy_entry(DEFAULT_GOAL_STRATEGY_ID)
    if entry is None:
        raise RuntimeError(f"Default Goal strategy is not registered: {DEFAULT_GOAL_STRATEGY_ID}")
    return entry


@contextmanager
def registered_goal_strategy_entries_for_test(
    entries: list[GoalStrategyRegistryEntryData] | tuple[GoalStrategyRegistryEntryData, ...],
) -> Iterator[None]:
    """Temporarily append registry entries for tests without route-specific code."""

    global _test_entries
    with _test_entries_lock:
        previous_entries = _test_entries
        _test_entries = (*_test_entries, *tuple(entries))
    try:
        yield
    finally:
        with _test_entries_lock:
            _test_entries = previous_entries
