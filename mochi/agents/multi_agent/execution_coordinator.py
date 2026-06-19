"""Shared controlled-execution coordinator for multi-agent workflows."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
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


@dataclass(frozen=True)
class ControlledExecutionResumeHooks:
    """Optional shared task-state hooks supplied by the orchestrator."""

    get_task_summary: Callable[[str], Mapping[str, Any] | None] | None = None
    mark_task_running: Callable[..., None] | None = None
    mark_task_completed: Callable[..., None] | None = None
    mark_task_failed: Callable[..., None] | None = None
    mark_task_reused: Callable[..., None] | None = None


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
        command_rules: list[dict[str, Any]] | None = None,
        allowed_env_vars: list[str] | None = None,
        default_shell: str | None = None,
    ) -> None:
        self._generate_role_candidate = generate_role_candidate
        self._invoke_text = invoke_text
        self._exec_runtime = exec_runtime
        self._exec_approval_store = exec_approval_store
        self._require_approval = bool(require_approval)
        self._command_rules = [dict(rule) for rule in (command_rules or []) if isinstance(rule, dict)]
        self._allowed_env_vars = [str(item) for item in (allowed_env_vars or []) if str(item).strip()]
        self._default_shell = str(default_shell or "auto").strip() or "auto"

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
        resume_hooks: ControlledExecutionResumeHooks | None = None,
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
        planner_task_key = "controlled_execution_planner"
        executor_task_key = "controlled_execution_executor"
        evaluator_task_key = "controlled_execution_evaluator"

        planner_summary = _resume_task_summary(resume_hooks, planner_task_key)
        planner_output = _resume_candidate_payload(planner_summary, "planner_output")
        if planner_output is not None:
            _mark_task_reused(
                resume_hooks,
                task_key=planner_task_key,
                role_id=planner_role.role_id,
                stage=planner_task_key,
                candidate=planner_output,
            )
        else:
            if not planner_model_id:
                return [], {}
            generated_planner_output = await self._generate_role_candidate(
                role_id=planner_role.role_id,
                role_title=planner_role.title,
                role_instruction=planner_role.instruction,
                model_id=planner_model_id,
                task_input=task_input,
                guidance_messages=guidance_messages,
                supporting_candidates=[],
                stage=planner_task_key,
            )
            planner_output = _candidate_payload_from_output(generated_planner_output)
            emit(
                "role_output",
                {
                    "role_id": generated_planner_output.role_id,
                    "content": generated_planner_output.content,
                    "round_index": 1,
                    "candidate_id": generated_planner_output.candidate_id,
                    "model_id": planner_model_id,
                },
            )
            _mark_task_completed(
                resume_hooks,
                task_key=planner_task_key,
                role_id=planner_role.role_id,
                model_id=planner_model_id,
                stage=planner_task_key,
                candidate=planner_output,
                result_summary={"planner_output": planner_output},
            )
        if planner_output is None:
            return [], {}

        executor_prompt = _build_controlled_execution_request_prompt(
            task_input=task_input,
            execution_plan=str(planner_output.get("content") or ""),
            max_execution_requests=execution_policy.max_execution_requests,
            max_commands_per_request=execution_policy.max_commands_per_request,
            default_timeout_sec=execution_policy.default_timeout_sec,
            background_allowed=execution_policy.background_allowed,
            guidance_messages=guidance_messages,
        )
        executor_summary = _resume_task_summary(resume_hooks, executor_task_key)
        executor_output = _resume_candidate_payload(executor_summary, "executor_output")
        execution_requests = _resume_list_summary(executor_summary, "execution_requests")
        request_parse_diagnostics = _resume_dict_summary(executor_summary, "request_parse_diagnostics") or {}
        if executor_output is not None and execution_requests is not None:
            if not request_parse_diagnostics:
                request_parse_diagnostics = {
                    "status": "restored",
                    "parsed_request_count": len(execution_requests),
                    "max_execution_requests": execution_policy.max_execution_requests,
                    "reason": None,
                }
            _mark_task_reused(
                resume_hooks,
                task_key=executor_task_key,
                role_id=executor_role.role_id,
                stage=executor_task_key,
                candidate=executor_output,
            )
        else:
            if not executor_model_id:
                return [], {}
            _mark_task_running(
                resume_hooks,
                task_key=executor_task_key,
                role_id=executor_role.role_id,
                model_id=executor_model_id,
                stage=executor_task_key,
                depends_on=[planner_task_key],
            )
            try:
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
            except Exception as exc:
                _mark_task_failed(
                    resume_hooks,
                    task_key=executor_task_key,
                    role_id=executor_role.role_id,
                    model_id=executor_model_id,
                    stage=executor_task_key,
                    reason=str(exc),
                )
                raise
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
            _mark_task_completed(
                resume_hooks,
                task_key=executor_task_key,
                role_id=executor_role.role_id,
                model_id=executor_model_id,
                stage=executor_task_key,
                candidate=executor_output,
                result_summary={
                    "executor_output": executor_output,
                    "execution_requests": execution_requests,
                    "request_parse_diagnostics": request_parse_diagnostics,
                },
            )
        if executor_output is None or execution_requests is None:
            return [], {}
        controller_decisions: list[dict[str, Any]] = []
        execution_results: list[dict[str, Any]] = []
        task_workspace_dir = _metadata_string(metadata, "task_workspace_dir")
        workspace_dir = task_workspace_dir or _metadata_string(metadata, "workspace_dir")

        for request_index, execution_request in enumerate(execution_requests, start=1):
            request_id = str(execution_request.get("request_id") or f"exec-request-{request_index}")
            controller_task_key = f"controlled_execution_controller:{request_id}"
            exec_task_key = f"controlled_execution_exec:{request_id}"
            controller_prompt = _build_controller_decision_prompt(
                task_input=task_input,
                execution_plan=str(planner_output.get("content") or ""),
                execution_request=execution_request,
            )
            saved_controller_summary = _resume_task_summary(resume_hooks, controller_task_key)
            decision = _resume_dict_summary(saved_controller_summary, "controller_decision")
            if decision is not None:
                _mark_task_reused(
                    resume_hooks,
                    task_key=controller_task_key,
                    role_id=controller_role.role_id,
                    stage=controller_task_key,
                )
            else:
                if not controller_model_id:
                    return [], {}
                _mark_task_running(
                    resume_hooks,
                    task_key=controller_task_key,
                    role_id=controller_role.role_id,
                    model_id=controller_model_id,
                    stage=controller_task_key,
                    depends_on=[executor_task_key],
                )
                try:
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
                except Exception as exc:
                    _mark_task_failed(
                        resume_hooks,
                        task_key=controller_task_key,
                        role_id=controller_role.role_id,
                        model_id=controller_model_id,
                        stage=controller_task_key,
                        reason=str(exc),
                    )
                    raise
                decision = _parse_controller_decision(
                    decision_content,
                    execution_request=execution_request,
                    diagnostics=controller_diagnostics,
                    request_index=request_index,
                )
                _mark_task_completed(
                    resume_hooks,
                    task_key=controller_task_key,
                    role_id=controller_role.role_id,
                    model_id=controller_model_id,
                    stage=controller_task_key,
                    result_summary={"controller_decision": decision},
                )
            if decision is None:
                return [], {}
            controller_decisions.append(decision)
            saved_execution_summary = _resume_task_summary(resume_hooks, exec_task_key)
            execution_result = _resume_dict_summary(saved_execution_summary, "execution_result")
            if execution_result is not None:
                _mark_task_reused(
                    resume_hooks,
                    task_key=exec_task_key,
                    role_id=controller_role.role_id,
                    stage=exec_task_key,
                )
                execution_results.append(execution_result)
                continue
            if decision["status"] != "approved":
                execution_result = {
                    "request_id": execution_request["request_id"],
                    "status": "skipped",
                    "reason": decision.get("reason") or "Controller did not approve execution.",
                }
                _mark_task_completed(
                    resume_hooks,
                    task_key=exec_task_key,
                    role_id=controller_role.role_id,
                    model_id=controller_model_id,
                    stage=exec_task_key,
                    result_summary={"execution_result": execution_result},
                )
                execution_results.append(execution_result)
                continue
            _mark_task_running(
                resume_hooks,
                task_key=exec_task_key,
                role_id=controller_role.role_id,
                model_id=controller_model_id,
                stage=exec_task_key,
                depends_on=[controller_task_key],
            )
            try:
                execution_result = await self._execute_controlled_command(
                    execution_request=execution_request,
                    controller_decision=decision,
                    workspace_dir=workspace_dir,
                    default_timeout_sec=execution_policy.default_timeout_sec,
                )
            except Exception as exc:
                _mark_task_failed(
                    resume_hooks,
                    task_key=exec_task_key,
                    role_id=controller_role.role_id,
                    model_id=controller_model_id,
                    stage=exec_task_key,
                    reason=str(exc),
                )
                raise
            _mark_task_completed(
                resume_hooks,
                task_key=exec_task_key,
                role_id=controller_role.role_id,
                model_id=controller_model_id,
                stage=exec_task_key,
                result_summary={"execution_result": execution_result},
            )
            execution_results.append(execution_result)

        saved_evaluator_summary = _resume_task_summary(resume_hooks, evaluator_task_key)
        evaluator_output = _resume_candidate_payload(saved_evaluator_summary, "evaluator_output")
        evaluation_summary = _resume_dict_summary(saved_evaluator_summary, "evaluation_summary")
        if evaluator_output is not None and evaluation_summary is not None:
            _mark_task_reused(
                resume_hooks,
                task_key=evaluator_task_key,
                role_id=evaluator_role.role_id,
                stage=evaluator_task_key,
                candidate=evaluator_output,
            )
        else:
            if not evaluator_model_id:
                return [], {}
            evaluator_prompt = _build_controlled_evaluator_prompt(
                task_input=task_input,
                execution_plan=str(planner_output.get("content") or ""),
                execution_requests=execution_requests,
                controller_decisions=controller_decisions,
                execution_results=execution_results,
            )
            _mark_task_running(
                resume_hooks,
                task_key=evaluator_task_key,
                role_id=evaluator_role.role_id,
                model_id=evaluator_model_id,
                stage=evaluator_task_key,
                depends_on=[
                    *(f"controlled_execution_exec:{str(item.get('request_id') or '')}" for item in execution_requests),
                ],
            )
            try:
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
            except Exception as exc:
                _mark_task_failed(
                    resume_hooks,
                    task_key=evaluator_task_key,
                    role_id=evaluator_role.role_id,
                    model_id=evaluator_model_id,
                    stage=evaluator_task_key,
                    reason=str(exc),
                )
                raise
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
            _mark_task_completed(
                resume_hooks,
                task_key=evaluator_task_key,
                role_id=evaluator_role.role_id,
                model_id=evaluator_model_id,
                stage=evaluator_task_key,
                candidate=evaluator_output,
                result_summary={
                    "evaluator_output": evaluator_output,
                    "evaluation_summary": evaluation_summary,
                },
            )
        if evaluator_output is None or evaluation_summary is None:
            return [], {}
        protocol_artifacts = {
            "execution_plan": {
                "content": str(planner_output.get("content") or ""),
                "model_id": _candidate_model_id(planner_output),
            },
            "execution_requests": {
                "items": execution_requests,
                "parse_diagnostics": request_parse_diagnostics,
            },
            "controller_decisions": {"items": controller_decisions},
            "execution_results": {"items": execution_results},
            "detached_exec_jobs": _collect_detached_exec_jobs(execution_results),
            "produced_artifacts": _collect_controlled_execution_artifacts(execution_results),
            "evaluation_summary": evaluation_summary,
            "controlled_execution_runtime": {
                "execution_boundary": "subagents_propose_controller_approves_shared_runtime_executes",
                "primary_workflow": primary_workflow,
                "workspace_mode": execution_policy.workspace_mode,
                "workspace_dir": workspace_dir,
                "request_count": len(execution_requests),
                "executed_count": sum(1 for item in execution_results if item.get("status") not in {"skipped"}),
                "detached_exec_job_count": sum(
                    1 for item in execution_results if bool(item.get("background")) and _result_session_id(item)
                ),
                "approval_pending_count": sum(
                    1 for item in execution_results if item.get("status") == "approval_pending"
                ),
            },
        }
        return [
            planner_output,
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
        resolved_shell = str(
            controller_decision.get("shell")
            or execution_request.get("shell")
            or self._default_shell
            or "auto"
        )
        resolved_workdir = controller_decision.get("workdir") or execution_request.get("workdir")
        resolved_timeout = controller_decision.get("timeout") or execution_request.get("timeout") or default_timeout_sec
        resolved_background = bool(
            controller_decision.get("background", execution_request.get("background", False))
        )
        detached_layout = (
            _prepare_detached_exec_layout(
                workspace_dir=workspace_dir,
                request_id=str(execution_request["request_id"]),
            )
            if resolved_background
            else None
        )
        tool = ExecCommandTool(
            runtime=self._exec_runtime,
            approval_store=self._exec_approval_store,
            workspace_dir=workspace_dir,
            command_rules=self._command_rules,
            allowed_env_vars=self._allowed_env_vars,
            require_approval=self._require_approval,
            default_timeout_sec=default_timeout_sec,
        )
        result = await tool.execute(
            command=command,
            shell=resolved_shell,
            workdir=resolved_workdir,
            timeout=resolved_timeout,
            background=resolved_background,
            log_path=detached_layout["log_path"] if detached_layout is not None else None,
            checkpoint_dir=detached_layout["checkpoint_dir"] if detached_layout is not None else None,
            detached_layout=detached_layout,
        )
        metadata = dict(result.metadata or {})
        status = str(metadata.get("status") or "completed")
        if result.error and status != "approval_pending":
            status = "failed"
        return {
            "request_id": execution_request["request_id"],
            "status": status,
            "command": command,
            "shell": resolved_shell,
            "workdir": str(resolved_workdir).strip() if resolved_workdir else None,
            "timeout": resolved_timeout,
            "background": resolved_background,
            "log_path": metadata.get("log_path"),
            "session_log_path": metadata.get("session_log_path"),
            "checkpoint_dir": metadata.get("checkpoint_dir"),
            "root_dir": metadata.get("root_dir"),
            "manifest_path": metadata.get("manifest_path"),
            "stdout_log_path": metadata.get("stdout_log_path"),
            "stderr_log_path": metadata.get("stderr_log_path"),
            "stdout": result.output.get("stdout") if isinstance(result.output, dict) else None,
            "stderr": result.output.get("stderr") if isinstance(result.output, dict) else None,
            "error": result.error,
            "metadata": metadata,
        }


def _plain_data(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain_data(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_plain_data(item) for item in value]
    return value


def _candidate_payload_from_output(candidate: Any) -> dict[str, Any] | None:
    if isinstance(candidate, Mapping):
        payload = _plain_data(candidate)
    else:
        payload = {
            "candidate_id": getattr(candidate, "candidate_id", None),
            "role_id": getattr(candidate, "role_id", None),
            "content": getattr(candidate, "content", None),
            "metadata": _plain_data(getattr(candidate, "metadata", {}) or {}),
        }
    candidate_id = payload.get("candidate_id")
    role_id = payload.get("role_id")
    content = payload.get("content")
    if not all(isinstance(field, str) and field.strip() for field in (candidate_id, role_id, content)):
        return None
    payload["candidate_id"] = str(candidate_id)
    payload["role_id"] = str(role_id)
    payload["content"] = str(content)
    payload["metadata"] = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return payload


def _candidate_model_id(candidate: Mapping[str, Any] | None) -> str | None:
    metadata = candidate.get("metadata") if isinstance(candidate, Mapping) else None
    model_id = metadata.get("model_id") if isinstance(metadata, Mapping) else None
    if isinstance(model_id, str) and model_id.strip():
        return model_id.strip()
    return None


def _resume_task_summary(
    resume_hooks: ControlledExecutionResumeHooks | None,
    task_key: str,
) -> dict[str, Any] | None:
    getter = resume_hooks.get_task_summary if resume_hooks is not None else None
    if not callable(getter):
        return None
    summary = getter(task_key)
    if not isinstance(summary, Mapping):
        return None
    return _plain_data(summary)


def _resume_candidate_payload(summary: Mapping[str, Any] | None, key: str) -> dict[str, Any] | None:
    if not isinstance(summary, Mapping):
        return None
    return _candidate_payload_from_output(summary.get(key))


def _resume_dict_summary(summary: Mapping[str, Any] | None, key: str) -> dict[str, Any] | None:
    if not isinstance(summary, Mapping):
        return None
    value = summary.get(key)
    if not isinstance(value, Mapping):
        return None
    return _plain_data(value)


def _resume_list_summary(summary: Mapping[str, Any] | None, key: str) -> list[dict[str, Any]] | None:
    if not isinstance(summary, Mapping):
        return None
    value = summary.get(key)
    if not isinstance(value, list):
        return None
    payload = _plain_data(value)
    if not isinstance(payload, list):
        return None
    return [item for item in payload if isinstance(item, dict)]


def _mark_task_running(
    resume_hooks: ControlledExecutionResumeHooks | None,
    **kwargs: Any,
) -> None:
    marker = resume_hooks.mark_task_running if resume_hooks is not None else None
    if callable(marker):
        marker(**kwargs)


def _mark_task_completed(
    resume_hooks: ControlledExecutionResumeHooks | None,
    **kwargs: Any,
) -> None:
    marker = resume_hooks.mark_task_completed if resume_hooks is not None else None
    if callable(marker):
        marker(**kwargs)


def _mark_task_failed(
    resume_hooks: ControlledExecutionResumeHooks | None,
    **kwargs: Any,
) -> None:
    marker = resume_hooks.mark_task_failed if resume_hooks is not None else None
    if callable(marker):
        marker(**kwargs)


def _mark_task_reused(
    resume_hooks: ControlledExecutionResumeHooks | None,
    **kwargs: Any,
) -> None:
    marker = resume_hooks.mark_task_reused if resume_hooks is not None else None
    if callable(marker):
        marker(**kwargs)


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


def _collect_detached_exec_jobs(execution_results: list[dict[str, Any]]) -> dict[str, Any]:
    jobs: list[dict[str, Any]] = []
    for result in execution_results:
        session_id = _result_session_id(result)
        if not session_id or not bool(result.get("background")):
            continue
        metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
        detached_layout = (
            metadata.get("detached_layout") if isinstance(metadata.get("detached_layout"), dict) else {}
        )
        root_dir = (
            str(detached_layout.get("root_dir"))
            if isinstance(detached_layout.get("root_dir"), str)
            else result.get("root_dir")
        )
        log_path = (
            str(detached_layout.get("log_path"))
            if isinstance(detached_layout.get("log_path"), str)
            else result.get("log_path")
        )
        checkpoint_dir = (
            str(detached_layout.get("checkpoint_dir"))
            if isinstance(detached_layout.get("checkpoint_dir"), str)
            else result.get("checkpoint_dir")
        )
        manifest_path = (
            str(detached_layout.get("manifest_path"))
            if isinstance(detached_layout.get("manifest_path"), str)
            else result.get("manifest_path")
        )
        stdout_log_path = (
            str(detached_layout.get("stdout_log_path"))
            if isinstance(detached_layout.get("stdout_log_path"), str)
            else result.get("stdout_log_path")
        )
        stderr_log_path = (
            str(detached_layout.get("stderr_log_path"))
            if isinstance(detached_layout.get("stderr_log_path"), str)
            else result.get("stderr_log_path")
        )
        jobs.append(
            {
                "session_id": session_id,
                "request_id": result.get("request_id"),
                "command": result.get("command"),
                "shell": result.get("shell"),
                "workdir": result.get("workdir"),
                "timeout": result.get("timeout"),
                "root_dir": root_dir,
                "log_path": log_path,
                "session_log_path": metadata.get("session_log_path") or log_path,
                "checkpoint_dir": checkpoint_dir,
                "manifest_path": manifest_path,
                "stdout_log_path": stdout_log_path,
                "stderr_log_path": stderr_log_path,
                "status": result.get("status"),
                "background": True,
                "approval_state": metadata.get("approval_state"),
                "pid": metadata.get("pid"),
                "detached": bool(metadata.get("detached", True)),
                "restored": bool(metadata.get("restored", False)),
                "supports_stdin": bool(metadata.get("supports_stdin", True)),
                "lease_owner": "runtime_service",
                "reattach_supported": True,
                "recoverable": bool(root_dir and checkpoint_dir and log_path),
                "recovery_supported": bool(metadata.get("recovery_supported", True)),
                "runtime_state_root": metadata.get("runtime_state_root"),
                "detached_layout": detached_layout or None,
            }
        )
    return {
        "items": jobs,
        "count": len(jobs),
        "reattachable_count": sum(1 for item in jobs if bool(item.get("reattach_supported"))),
        "recoverable_count": sum(1 for item in jobs if bool(item.get("recoverable"))),
    }


def _result_session_id(result: dict[str, Any]) -> str | None:
    metadata = result.get("metadata") if isinstance(result.get("metadata"), dict) else {}
    session_id = metadata.get("session_id")
    if isinstance(session_id, str) and session_id.strip():
        return session_id.strip()
    return None


def _prepare_detached_exec_layout(workspace_dir: str | None, request_id: str) -> dict[str, str] | None:
    if not workspace_dir or not str(workspace_dir).strip():
        return None
    root = (Path(workspace_dir).resolve() / ".mochi-detached-exec" / request_id).resolve()
    root.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    runtime_state_root = (Path(".mochi").resolve() / "exec-runtime").resolve()
    return {
        "root_dir": str(root),
        "log_path": str((root / "session.log").resolve()),
        "session_log_path": str((root / "session.log").resolve()),
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "manifest_path": str((root / "manifest.json").resolve()),
        "stdout_log_path": str((root / "stdout.log").resolve()),
        "stderr_log_path": str((root / "stderr.log").resolve()),
        "runtime_state_root": str(runtime_state_root),
    }


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
