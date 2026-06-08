"""Shared execution-capability models for multi-agent workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping

LEGACY_CONTROLLED_SUBAGENT_PROTOCOL = "controlled_subagent_execution"


@dataclass(frozen=True)
class SubagentExecutionPolicy:
    """Execution-capability policy shared across multi-agent workflows."""

    mode: Literal["disabled", "research_only", "controlled"] = "disabled"
    allowed_roles: tuple[str, ...] = ()
    max_execution_requests: int = 5
    max_commands_per_request: int = 1
    default_timeout_sec: int = 300
    background_allowed: bool = True
    workspace_mode: Literal["task_sandbox"] = "task_sandbox"
    planner_role_id: str = "planner"
    executor_role_id: str = "executor"
    controller_role_id: str = "controller"
    evaluator_role_id: str = "evaluator"


def parse_subagent_execution_policy(
    payload: Mapping[str, Any] | None,
    *,
    legacy_protocol: str | None = None,
) -> SubagentExecutionPolicy:
    """Parse execution-capability metadata with legacy protocol fallback."""

    if payload is None:
        if legacy_protocol == LEGACY_CONTROLLED_SUBAGENT_PROTOCOL:
            payload = {"mode": "controlled"}
        else:
            return SubagentExecutionPolicy()

    mode = str(payload.get("mode") or "").strip().lower()
    if not mode:
        mode = "controlled" if legacy_protocol == LEGACY_CONTROLLED_SUBAGENT_PROTOCOL else "disabled"
    if mode not in {"disabled", "research_only", "controlled"}:
        mode = "disabled"

    planner_role_id = _clean_role_id(payload.get("planner_role_id"), default="planner")
    executor_role_id = _clean_role_id(payload.get("executor_role_id"), default="executor")
    controller_role_id = _clean_role_id(payload.get("controller_role_id"), default="controller")
    evaluator_role_id = _clean_role_id(payload.get("evaluator_role_id"), default="evaluator")

    allowed_roles = _normalize_allowed_roles(
        payload.get("allowed_roles"),
        defaults=(planner_role_id, executor_role_id, controller_role_id, evaluator_role_id),
        mode=mode,
    )

    return SubagentExecutionPolicy(
        mode=mode,  # type: ignore[arg-type]
        allowed_roles=allowed_roles,
        max_execution_requests=_bounded_int(payload.get("max_execution_requests"), default=5, minimum=1, maximum=20),
        max_commands_per_request=_bounded_int(
            payload.get("max_commands_per_request"),
            default=1,
            minimum=1,
            maximum=5,
        ),
        default_timeout_sec=_bounded_int(
            payload.get("default_timeout_sec"),
            default=300,
            minimum=1,
            maximum=86_400,
        ),
        background_allowed=bool(payload.get("background_allowed", True)),
        workspace_mode="task_sandbox",
        planner_role_id=planner_role_id,
        executor_role_id=executor_role_id,
        controller_role_id=controller_role_id,
        evaluator_role_id=evaluator_role_id,
    )


def execution_policy_to_dict(policy: SubagentExecutionPolicy) -> dict[str, Any]:
    """Serialize the shared execution policy."""

    payload = asdict(policy)
    payload["allowed_roles"] = list(policy.allowed_roles)
    return payload


def run_uses_controlled_execution(
    protocol: str,
    *,
    metadata: Mapping[str, Any] | None = None,
    artifacts: Mapping[str, Any] | None = None,
) -> bool:
    """Determine whether a run should be treated as controlled execution."""

    if protocol == LEGACY_CONTROLLED_SUBAGENT_PROTOCOL:
        return True
    runtime = artifacts.get("controlled_execution_runtime") if isinstance(artifacts, Mapping) else None
    if isinstance(runtime, Mapping):
        if runtime.get("primary_workflow") is True:
            return True
        if metadata is not None:
            policy = metadata.get("execution_policy")
            if isinstance(policy, Mapping) and str(policy.get("mode") or "").strip().lower() == "controlled":
                return bool(runtime.get("primary_workflow", False))
    if metadata is not None:
        policy = metadata.get("execution_policy")
        if isinstance(policy, Mapping) and str(policy.get("mode") or "").strip().lower() == "controlled":
            return False
    return False


def _bounded_int(value: Any, *, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        parsed = default
    return min(max(parsed, minimum), maximum)


def _clean_role_id(value: Any, *, default: str) -> str:
    text = str(value or "").strip()
    return text or default


def _normalize_allowed_roles(
    value: Any,
    *,
    defaults: tuple[str, str, str, str],
    mode: str,
) -> tuple[str, ...]:
    if isinstance(value, list):
        roles = tuple(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))
        if roles:
            return roles
    if mode == "disabled":
        return ()
    return tuple(role for role in defaults if role)
