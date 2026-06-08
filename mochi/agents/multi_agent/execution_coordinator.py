"""Shared controlled-execution coordinator for multi-agent workflows."""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable, Mapping

from mochi.agents.multi_agent.execution_policy import SubagentExecutionPolicy
from mochi.agents.multi_agent.roles import build_controlled_execution_roles
from mochi.agents.multi_agent.utils import parse_json_payload
from mochi.runtime.approvals import InMemoryApprovalStore
from mochi.runtime.exec_runtime import ExecRuntime
from mochi.tools.exec_command import ExecCommandTool

RoleGenerator = Callable[..., Awaitable[Any]]
TextInvoker = Callable[..., Awaitable[tuple[str, dict[str, Any]]]]
RoleModelResolver = Callable[..., str | None]


class SubagentExecutionCoordinator:
    """Run the shared controller-gated execution workflow."""

    def __init__(
        self,
        *,
        generate_role_candidate: RoleGenerator,
        invoke_text: TextInvoker,
        exec_runtime: ExecRuntime | None = None,
        exec_approval_store: InMemoryApprovalStore | None = None,
        require_approval: bool = True,
    ) -> None:
        self._generate_role_candidate = generate_role_candidate
        self._invoke_text = invoke_text
        self._exec_runtime = exec_runtime
        self._exec_approval_store = exec_approval_store
        self._require_approval = bool(require_approval)

    async def run(
        self,
        *,
        task_input: str,
        execution_policy: SubagentExecutionPolicy,
        guidance_messages: list[str],
        selected_models_roles: dict[str, str],
        ordered_models: list[str],
        metadata: dict[str, Any],
        emit: Any,
        resolve_model_id: RoleModelResolver,
        primary_workflow: bool = True,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        roles = build_controlled_execution_roles(
            planner_role_id=execution_policy.planner_role_id,
            executor_role_id=execution_policy.executor_role_id,
            controller_role_id=execution_policy.controller_role_id,
            evaluator_role_id=execution_policy.evaluator_role_id,
        )
        planner_role, executor_role, controller_role, evaluator_role = roles
        planner_model_id = resolve_model_id(
            role_id=planner_role.role_id,
            selected_models_roles=selected_models_roles,
            ordered_fallback=ordered_models,
            fallback_index=0,
        )
        executor_model_id = resolve_model_id(
            role_id=executor_role.role_id,
            selected_models_roles=selected_models_roles,
            ordered_fallback=ordered_models,
            fallback_index=1,
            default_model_id=planner_model_id,
        )
        controller_model_id = resolve_model_id(
            role_id=controller_role.role_id,
            selected_models_roles=selected_models_roles,
            ordered_fallback=ordered_models,
            fallback_index=2,
            default_model_id=planner_model_id,
        )
        evaluator_model_id = resolve_model_id(
            role_id=evaluator_role.role_id,
            selected_models_roles=selected_models_roles,
            ordered_fallback=ordered_models,
            fallback_index=3,
            default_model_id=controller_model_id,
        )
        if not planner_model_id or not executor_model_id or not controller_model_id or not evaluator_model_id:
            return [], {}

        planner_output = await self._generate_role_candidate(
            role_id=planner_role.role_id,
            role_title=planner_role.title,
            role_instruction=planner_role.instruction,
            model_id=planner_model_id,
            task_input=task_input,
            guidance_messages=guidance_messages,
            supporting_candidates=[],
        )
        emit(
            "role_output",
            {
                "role_id": planner_output.role_id,
                "content": planner_output.content,
                "round_index": 1,
                "candidate_id": planner_output.candidate_id,
                "model_id": planner_model_id,
            },
        )

        executor_prompt = _build_controlled_execution_request_prompt(
            task_input=task_input,
            execution_plan=planner_output.content,
            max_execution_requests=execution_policy.max_execution_requests,
            max_commands_per_request=execution_policy.max_commands_per_request,
            default_timeout_sec=execution_policy.default_timeout_sec,
            background_allowed=execution_policy.background_allowed,
            guidance_messages=guidance_messages,
        )
        executor_content, executor_diagnostics = await self._invoke_text(
            model_id=executor_model_id,
            system_prompt=executor_role.instruction,
            user_prompt=executor_prompt,
            temperature=0.1,
            max_tokens=1600,
            execution_profile="subagent_execution_request",
            tool_mode="auto",
            system_prompt_addendum=(
                f"Role identity: {executor_role.title} ({executor_role.role_id}).\n"
                f"Role instruction: {executor_role.instruction}"
            ),
            session_scope=f"role::{executor_role.role_id}",
        )
        executor_output = {
            "candidate_id": executor_role.role_id,
            "role_id": executor_role.role_id,
            "content": executor_content,
            "metadata": {"model_id": executor_model_id, "diagnostics": executor_diagnostics},
        }
        emit(
            "role_output",
            {
                "role_id": executor_output["role_id"],
                "content": executor_output["content"],
                "round_index": 1,
                "candidate_id": executor_output["candidate_id"],
                "model_id": executor_model_id,
            },
        )

        execution_requests, request_parse_diagnostics = _parse_controlled_execution_requests(
            executor_content,
            max_execution_requests=execution_policy.max_execution_requests,
            max_commands_per_request=execution_policy.max_commands_per_request,
            default_timeout_sec=execution_policy.default_timeout_sec,
            background_allowed=execution_policy.background_allowed,
        )
        controller_decisions: list[dict[str, Any]] = []
        execution_results: list[dict[str, Any]] = []
        task_workspace_dir = _metadata_string(metadata, "task_workspace_dir")
        workspace_dir = task_workspace_dir or _metadata_string(metadata, "workspace_dir")

        for request_index, execution_request in enumerate(execution_requests, start=1):
            controller_prompt = _build_controller_decision_prompt(
                task_input=task_input,
                execution_plan=planner_output.content,
                execution_request=execution_request,
            )
            decision_content, controller_diagnostics = await self._invoke_text(
                model_id=controller_model_id,
                system_prompt=controller_role.instruction,
                user_prompt=controller_prompt,
                temperature=0.0,
                max_tokens=900,
                execution_profile="controller_exec",
                tool_mode="auto",
                system_prompt_addendum=(
                    f"Role identity: {controller_role.title} ({controller_role.role_id}).\n"
                    f"Role instruction: {controller_role.instruction}"
                ),
                session_scope=f"role::{controller_role.role_id}::{request_index}",
            )
            decision = _parse_controller_decision(
                decision_content,
                execution_request=execution_request,
                diagnostics=controller_diagnostics,
                request_index=request_index,
            )
            controller_decisions.append(decision)
            if decision["status"] != "approved":
                execution_results.append(
                    {
                        "request_id": execution_request["request_id"],
                        "status": "skipped",
                        "reason": decision.get("reason") or "Controller did not approve execution.",
                    }
                )
                continue
            execution_results.append(
                await self._execute_controlled_command(
                    execution_request=execution_request,
                    controller_decision=decision,
                    workspace_dir=workspace_dir,
                    default_timeout_sec=execution_policy.default_timeout_sec,
                )
            )

        evaluator_prompt = _build_controlled_evaluator_prompt(
            task_input=task_input,
            execution_plan=planner_output.content,
            execution_requests=execution_requests,
            controller_decisions=controller_decisions,
            execution_results=execution_results,
        )
        evaluator_content, evaluator_diagnostics = await self._invoke_text(
            model_id=evaluator_model_id,
            system_prompt=evaluator_role.instruction,
            user_prompt=evaluator_prompt,
            temperature=0.1,
            max_tokens=1200,
            execution_profile="subagent_readonly",
            tool_mode="auto",
            system_prompt_addendum=(
                f"Role identity: {evaluator_role.title} ({evaluator_role.role_id}).\n"
                f"Role instruction: {evaluator_role.instruction}"
            ),
            session_scope=f"role::{evaluator_role.role_id}",
        )
        evaluator_output = {
            "candidate_id": evaluator_role.role_id,
            "role_id": evaluator_role.role_id,
            "content": evaluator_content,
            "metadata": {"model_id": evaluator_model_id, "diagnostics": evaluator_diagnostics},
        }
        emit(
            "role_output",
            {
                "role_id": evaluator_output["role_id"],
                "content": evaluator_output["content"],
                "round_index": 1,
                "candidate_id": evaluator_output["candidate_id"],
                "model_id": evaluator_model_id,
            },
        )
        evaluation_summary = _build_controlled_execution_summary(
            evaluator_output=evaluator_output,
            execution_results=execution_results,
            controller_decisions=controller_decisions,
        )
        protocol_artifacts = {
            "execution_plan": {
                "content": planner_output.content,
                "model_id": planner_model_id,
            },
            "execution_requests": {
                "items": execution_requests,
                "parse_diagnostics": request_parse_diagnostics,
            },
            "controller_decisions": {"items": controller_decisions},
            "execution_results": {"items": execution_results},
            "produced_artifacts": _collect_controlled_execution_artifacts(execution_results),
            "evaluation_summary": evaluation_summary,
            "controlled_execution_runtime": {
                "execution_boundary": "subagents_propose_controller_approves_shared_runtime_executes",
                "primary_workflow": primary_workflow,
                "workspace_mode": execution_policy.workspace_mode,
                "workspace_dir": workspace_dir,
                "request_count": len(execution_requests),
                "executed_count": sum(1 for item in execution_results if item.get("status") not in {"skipped"}),
                "approval_pending_count": sum(
                    1 for item in execution_results if item.get("status") == "approval_pending"
                ),
            },
        }
        return [
            {
                "candidate_id": planner_output.candidate_id,
                "role_id": planner_output.role_id,
                "content": planner_output.content,
                "metadata": dict(planner_output.metadata or {}),
            },
            executor_output,
            evaluator_output,
        ], protocol_artifacts

    async def _execute_controlled_command(
        self,
        *,
        execution_request: dict[str, Any],
        controller_decision: dict[str, Any],
        workspace_dir: str | None,
        default_timeout_sec: int,
    ) -> dict[str, Any]:
        command = str(controller_decision.get("command") or execution_request.get("command") or "").strip()
        if not command:
            return {
                "request_id": execution_request["request_id"],
                "status": "failed",
                "error": "Controller approved execution without a command.",
            }
        tool = ExecCommandTool(
            runtime=self._exec_runtime,
            approval_store=self._exec_approval_store,
            workspace_dir=workspace_dir,
            require_approval=self._require_approval,
            default_timeout_sec=default_timeout_sec,
        )
        result = await tool.execute(
            command=command,
            shell=str(controller_decision.get("shell") or execution_request.get("shell") or "powershell"),
            workdir=controller_decision.get("workdir") or execution_request.get("workdir"),
            timeout=controller_decision.get("timeout") or execution_request.get("timeout") or default_timeout_sec,
            background=bool(controller_decision.get("background", execution_request.get("background", False))),
        )
        metadata = dict(result.metadata or {})
        status = str(metadata.get("status") or "completed")
        if result.error and status != "approval_pending":
            status = "failed"
        return {
            "request_id": execution_request["request_id"],
            "status": status,
            "command": command,
            "shell": str(controller_decision.get("shell") or execution_request.get("shell") or "powershell"),
            "stdout": result.output.get("stdout") if isinstance(result.output, dict) else None,
            "stderr": result.output.get("stderr") if isinstance(result.output, dict) else None,
            "error": result.error,
            "metadata": metadata,
        }


def _build_controlled_execution_request_prompt(
    *,
    task_input: str,
    execution_plan: str,
    max_execution_requests: int,
    max_commands_per_request: int,
    default_timeout_sec: int,
    background_allowed: bool,
    guidance_messages: list[str],
) -> str:
    guidance = "\n".join(f"- {item}" for item in guidance_messages if item.strip())
    return "\n".join(
        [
            "Propose controller-reviewed execution requests for the task.",
            f"Task:\n{task_input.strip()}",
            f"Execution plan:\n{execution_plan.strip()}",
            f"Maximum requests: {max_execution_requests}",
            f"Maximum commands per request: {max_commands_per_request}",
            f"Default timeout seconds: {default_timeout_sec}",
            f"Background allowed: {str(background_allowed).lower()}",
            f"Guidance:\n{guidance or '- Keep execution minimal and reversible.'}",
            "",
            "Return JSON only with key `execution_requests`.",
            "Each item must include command, rationale, expected_artifacts, and success_metric.",
            "Optional fields: shell, workdir, timeout, background.",
        ]
    )


def _parse_controlled_execution_requests(
    content: str,
    *,
    max_execution_requests: int,
    max_commands_per_request: int,
    default_timeout_sec: int,
    background_allowed: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    parsed = parse_json_payload(content)
    raw_items: Any = None
    if isinstance(parsed, dict):
        raw_items = (
            parsed.get("execution_requests")
            or parsed.get("requests")
            or parsed.get("commands")
            or parsed.get("items")
        )
    elif isinstance(parsed, list):
        raw_items = parsed

    requests: list[dict[str, Any]] = []
    if isinstance(raw_items, list):
        for index, item in enumerate(raw_items[:max_execution_requests], start=1):
            normalized = _normalize_controlled_execution_request(
                item,
                index=index,
                max_commands_per_request=max_commands_per_request,
                default_timeout_sec=default_timeout_sec,
                background_allowed=background_allowed,
            )
            if normalized is not None:
                requests.append(normalized)

    status = "parsed" if requests else "fallback_no_execution"
    reason = None if requests else "Executor output did not contain parseable execution request JSON."
    return requests, {
        "status": status,
        "parsed_request_count": len(requests),
        "max_execution_requests": max_execution_requests,
        "reason": reason,
    }


def _normalize_controlled_execution_request(
    item: Any,
    *,
    index: int,
    max_commands_per_request: int,
    default_timeout_sec: int,
    background_allowed: bool,
) -> dict[str, Any] | None:
    if isinstance(item, str):
        payload: dict[str, Any] = {"command": item}
    elif isinstance(item, Mapping):
        payload = dict(item)
    else:
        return None

    command = str(payload.get("command") or "").strip()
    if not command:
        commands = payload.get("commands")
        if isinstance(commands, list) and commands:
            command = " && ".join(
                str(value).strip() for value in commands[:max_commands_per_request] if str(value).strip()
            )
    if not command:
        return None
    try:
        timeout = int(payload.get("timeout") or payload.get("timeout_sec") or default_timeout_sec)
    except (TypeError, ValueError):
        timeout = default_timeout_sec
    return {
        "request_id": str(payload.get("request_id") or payload.get("id") or f"exec-request-{index}"),
        "command": command,
        "shell": str(payload.get("shell") or "powershell"),
        "workdir": str(payload.get("workdir")).strip() if payload.get("workdir") else None,
        "timeout": max(1, min(timeout, 86_400)),
        "background": bool(payload.get("background", False)) if background_allowed else False,
        "rationale": str(payload.get("rationale") or "").strip(),
        "expected_artifacts": [
            str(value).strip()
            for value in payload.get("expected_artifacts", [])
            if str(value).strip()
        ]
        if isinstance(payload.get("expected_artifacts"), list)
        else [],
        "success_metric": str(payload.get("success_metric") or "").strip(),
    }


def _build_controller_decision_prompt(
    *,
    task_input: str,
    execution_plan: str,
    execution_request: dict[str, Any],
) -> str:
    return "\n".join(
        [
            "Review this execution request. Approve only if it is necessary, bounded, and fits the task.",
            f"Task:\n{task_input.strip()}",
            f"Execution plan:\n{execution_plan.strip()}",
            "Execution request JSON:",
            json.dumps(execution_request, ensure_ascii=False, indent=2),
            "",
            "Return JSON only with keys: status, reason, command, shell, workdir, timeout, background.",
            "status must be one of approved, rejected, rewrite_required.",
            "Use rewrite_required only when the command should not run as-is.",
        ]
    )


def _parse_controller_decision(
    content: str,
    *,
    execution_request: dict[str, Any],
    diagnostics: dict[str, Any],
    request_index: int,
) -> dict[str, Any]:
    parsed = parse_json_payload(content)
    payload = dict(parsed) if isinstance(parsed, Mapping) else {}
    status = str(payload.get("status") or payload.get("decision") or "rejected").strip().lower()
    if status not in {"approved", "rejected", "rewrite_required"}:
        status = "rejected"
    if status == "rewrite_required":
        status = "rejected"
    return {
        "decision_id": f"controller-decision-{request_index}",
        "request_id": execution_request["request_id"],
        "status": status,
        "reason": str(payload.get("reason") or payload.get("rationale") or "").strip(),
        "command": str(payload.get("command") or execution_request.get("command") or "").strip(),
        "shell": str(payload.get("shell") or execution_request.get("shell") or "powershell"),
        "workdir": payload.get("workdir")
        if isinstance(payload.get("workdir"), str)
        else execution_request.get("workdir"),
        "timeout": payload.get("timeout") or execution_request.get("timeout"),
        "background": bool(payload.get("background", execution_request.get("background", False))),
        "raw_content": content,
        "diagnostics": diagnostics,
    }


def _build_controlled_evaluator_prompt(
    *,
    task_input: str,
    execution_plan: str,
    execution_requests: list[dict[str, Any]],
    controller_decisions: list[dict[str, Any]],
    execution_results: list[dict[str, Any]],
) -> str:
    return "\n".join(
        [
            "Evaluate the controlled subagent execution trace.",
            f"Task:\n{task_input.strip()}",
            f"Execution plan:\n{execution_plan.strip()}",
            "Execution requests:",
            json.dumps(execution_requests, ensure_ascii=False, indent=2),
            "Controller decisions:",
            json.dumps(controller_decisions, ensure_ascii=False, indent=2),
            "Execution results:",
            json.dumps(execution_results, ensure_ascii=False, indent=2),
            "",
            "Summarize what worked, what failed, produced artifacts, metrics, and next steps.",
        ]
    )


def _build_controlled_execution_summary(
    *,
    evaluator_output: dict[str, Any],
    execution_results: list[dict[str, Any]],
    controller_decisions: list[dict[str, Any]],
) -> dict[str, Any]:
    status_counts: dict[str, int] = {}
    for result in execution_results:
        status = str(result.get("status") or "unknown")
        status_counts[status] = status_counts.get(status, 0) + 1
    approved_count = sum(1 for decision in controller_decisions if decision.get("status") == "approved")
    return {
        "status": "completed",
        "summary": str(evaluator_output.get("content") or ""),
        "approved_count": approved_count,
        "rejected_count": len(controller_decisions) - approved_count,
        "execution_status_counts": status_counts,
    }


def _collect_controlled_execution_artifacts(execution_results: list[dict[str, Any]]) -> dict[str, Any]:
    artifacts: list[dict[str, Any]] = []
    for result in execution_results:
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        session_id = metadata.get("session_id")
        if isinstance(session_id, str) and session_id:
            artifacts.append(
                {
                    "kind": "exec_session",
                    "session_id": session_id,
                    "request_id": result.get("request_id"),
                    "status": result.get("status"),
                }
            )
    return {"items": artifacts, "count": len(artifacts)}


def _metadata_string(metadata: Mapping[str, Any], key: str) -> str | None:
    value = metadata.get(key)
    if isinstance(value, str) and value.strip():
        return value.strip()
    summary = metadata.get("summary")
    if isinstance(summary, Mapping):
        nested = summary.get(key)
        if isinstance(nested, str) and nested.strip():
            return nested.strip()
    return None
