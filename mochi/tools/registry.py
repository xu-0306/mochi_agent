"""Tool registry and discovery helpers."""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import sys
import weakref
from pathlib import Path
from typing import Any

try:
    from loguru import logger
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal test envs
    import logging

    logger = logging.getLogger(__name__)

from mochi.security import deny_security_decision
from mochi.sessions.timeline_coordinator import timeline_operation_metadata
from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult
from mochi.tools.tool_activate import ToolActivateTool
from mochi.tools.tool_search import ToolSearchTool

ToolFactory = Any
_MUTATION_TOOLS = frozenset({"file_write", "file_edit", "apply_patch"})
_EXECUTION_TOOLS = frozenset(
    {
        "exec_command",
        "execute_code",
        "execute_code_v2",
        "write_stdin",
        "kill_session",
        "process_stop",
    }
)
_RISKY_ACTIVATION_TOOLS = _MUTATION_TOOLS | _EXECUTION_TOOLS
_TIMELINE_LIFECYCLE_TOOLS = _MUTATION_TOOLS | _EXECUTION_TOOLS
_TIMELINE_UNSUPPORTED_SIDE_EFFECT_TOOLS = frozenset()
_TIMELINE_RESULT_DISPOSITIONS = frozenset({"succeeded", "failed", "unknown"})
_TIMELINE_APPROVAL_MODES = frozenset({"none", "continuable", "revocable"})
_MUTATION_CAPABILITIES = frozenset(
    {
        "workspace_write",
        "file_write",
        "file_edit",
        "apply_patch",
        "file_mutation",
        "filesystem_write",
        "mutation",
    }
)
_CONTRACT_ELIGIBILITY_FIELDS = (
    "mutation_requirement",
    "requested_operations",
    "required_capabilities",
)


def _normalized_capability_names(value: Any) -> set[str]:
    if not isinstance(value, (list, tuple, set, frozenset)):
        return set()
    return {
        str(item).strip().lower()
        for item in value
        if isinstance(item, str) and item.strip()
    }


def _resolve_mutation_activation_eligibility(
    policy: dict[str, Any],
) -> tuple[bool, str, str | None]:
    """Resolve mutation eligibility exclusively from intent-contract fields."""
    has_contract_fields = any(key in policy for key in _CONTRACT_ELIGIBILITY_FIELDS)
    if has_contract_fields:
        mutation_requirement = str(policy.get("mutation_requirement") or "").strip().lower()
        if mutation_requirement == "forbidden":
            return False, "intent_contract", "mutation_forbidden_by_contract"

        required_capabilities = _normalized_capability_names(
            policy.get("required_capabilities")
        )
        requested_operations = _normalized_capability_names(
            policy.get("requested_operations")
        )
        if required_capabilities & _MUTATION_CAPABILITIES:
            return True, "intent_contract.required_capabilities", None
        if requested_operations & _MUTATION_CAPABILITIES:
            return True, "intent_contract.requested_operations", None
        if mutation_requirement == "required":
            return True, "intent_contract.mutation_requirement", None
        return False, "intent_contract", "contract_disallows_mutation_activation"

    return False, "intent_contract", "missing_intent_contract_eligibility"


def _resolve_mutation_call_approval_hint(
    *,
    tool: BaseTool,
    context: ToolExecutionContext,
) -> tuple[bool, str]:
    """Describe the downstream call policy without authorizing activation."""
    value = context.permission_policy.get("require_approval_for_file_write")
    if isinstance(value, bool):
        return value, "execution_context"
    return bool(getattr(tool, "requires_approval", False)), "cached_tool_fallback"


def _timeline_requires_approval(*, tool: BaseTool, context: ToolExecutionContext) -> bool:
    """Predict only the stable policy approvals before a timeline precommit."""
    policy = context.permission_policy
    if tool.name in _MUTATION_TOOLS:
        value = policy.get("require_approval_for_file_write")
        return bool(value) if isinstance(value, bool) else bool(tool.requires_approval)
    if tool.name in {"exec_command", "execute_code", "execute_code_v2"}:
        value = policy.get("require_approval_for_exec")
        return bool(value) if isinstance(value, bool) else bool(tool.requires_approval)
    return bool(tool.requires_approval)


def _timeline_approval_mode(tool: BaseTool) -> str:
    """Read the tool's explicit ordinary-Chat approval capability contract."""
    value = str(getattr(tool, "timeline_approval_mode", "none") or "").strip().lower()
    if value not in _TIMELINE_APPROVAL_MODES:
        raise RuntimeError(
            f"tool '{tool.name}' declares an unsupported timeline approval mode: {value!r}"
        )
    return value


def _timeline_lifecycle(context: ToolExecutionContext | None) -> Any | None:
    if context is None or not isinstance(context.state, dict):
        return None
    return context.state.get("timeline_tool_lifecycle")


def _timeline_result_disposition(result: ToolResult) -> str:
    value = result.metadata.get("timeline_result_disposition")
    if value in _TIMELINE_RESULT_DISPOSITIONS:
        return str(value)
    return "unknown" if result.error is not None else "succeeded"


class ToolRegistry:
    """Registry for built-in, discovered, and factory-backed tools."""

    def __init__(
        self,
        extra_dirs: list[str] | None = None,
        discover_builtin: bool = True,
    ) -> None:
        self._tools: dict[str, BaseTool] = {}
        self._factories: dict[str, ToolFactory] = {}
        self._activation_source: ToolRegistry | None = None
        self._activation_discoverable_names: set[str] = set()
        self._activation_callable_names: set[str] | None = None
        self._activation_callable_order: list[str] = []
        self._activation_schema_limit: int | None = None
        if discover_builtin:
            package_dir = Path(__file__).resolve().parent
            self._discover(package_dir)
            self._discover(package_dir / "custom")
        if extra_dirs:
            for directory in extra_dirs:
                self._discover(Path(directory))

    def register(self, tool: BaseTool) -> None:
        self._register_tool(tool, log_registration=True)

    def _register_tool(
        self,
        tool: BaseTool,
        *,
        log_registration: bool = False,
    ) -> None:
        self._tools[tool.name] = tool
        if log_registration:
            logger.debug("Registered tool: {}", tool.name)

    def register_factory(self, name: str, factory: ToolFactory) -> None:
        self._factories[name] = factory
        logger.debug("Registered tool factory: {}", name)

    def get(self, name: str) -> BaseTool | None:
        tool = self._tools.get(name)
        if tool is not None:
            return tool

        factory = self._factories.get(name)
        if factory is None:
            return None

        try:
            instance = factory()
        except TypeError:
            instance = factory(name)
        if not isinstance(instance, BaseTool):
            raise TypeError(f"Factory for tool '{name}' did not return BaseTool.")
        self._tools[name] = instance
        return instance

    def list_tools(self) -> list[BaseTool]:
        for name in list(self._factories):
            self.get(name)
        return list(self._tools.values())

    def get_schemas(self) -> list[dict[str, Any]]:
        return [tool.to_schema_dict() for tool in self.list_tools()]

    def get_schemas_for_names(self, tool_names: list[str]) -> list[dict[str, Any]]:
        schemas: list[dict[str, Any]] = []
        for name in tool_names:
            tool = self.get(name)
            if tool is not None:
                schemas.append(tool.to_schema_dict())
        return schemas

    def create_view(
        self,
        tool_names: list[str],
        *,
        tool_search_catalog_names: list[str] | None = None,
        schema_limit: int | None = None,
    ) -> ToolRegistry:
        """Create a shallow registry view containing only the selected tools."""
        registry = ToolRegistry(discover_builtin=False)
        callable_names = set(tool_names)
        scoped_catalog_names = list(tool_search_catalog_names or tool_names)
        registry._activation_source = self
        registry._activation_discoverable_names = set(scoped_catalog_names)
        registry._activation_callable_names = callable_names
        registry._activation_callable_order = list(dict.fromkeys(tool_names))
        registry._activation_schema_limit = (
            max(0, schema_limit) if schema_limit is not None else None
        )
        for name in tool_names:
            tool = self.get(name)
            if tool is not None:
                if isinstance(tool, ToolSearchTool):
                    registry._register_tool(
                        tool.scoped_to_catalog(
                            lambda: [
                                candidate
                                for candidate_name in scoped_catalog_names
                                if (candidate := self.get(candidate_name)) is not None
                            ],
                            callable_name_provider=lambda: set(callable_names),
                        )
                    )
                    continue
                registry._register_tool(tool)
        deferred_names = registry._activation_discoverable_names - callable_names
        if deferred_names:
            registry_ref = weakref.ref(registry)

            def request_activation(
                tool_name: str,
                context: ToolExecutionContext | None,
            ) -> ToolResult:
                active_registry = registry_ref()
                if active_registry is None:
                    return ToolResult(
                        error="tool_activation_denied: registry view is no longer available.",
                        metadata={
                            "runtime_category": "tool_activation",
                            "error_type": "tool_activation_denied",
                            "requested_tool": tool_name,
                            "reason": "registry_view_unavailable",
                        },
                        retryable=False,
                    )
                return active_registry.request_tool_activation(
                    tool_name,
                    context=context,
                )

            registry._register_tool(
                ToolActivateTool(request_activation=request_activation)
            )
            callable_names.add("tool_activate")
            registry._activation_callable_order.append("tool_activate")
        return registry

    def request_tool_activation(
        self,
        tool_name: str,
        *,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        """Request promotion of one discoverable tool into this view only."""
        requested_tool = str(tool_name or "").strip()
        policy: dict[str, Any] = {}
        if context is not None:
            candidate_policy = context.state.get("tool_activation_policy")
            if isinstance(candidate_policy, dict):
                policy = candidate_policy

        callable_names = self._activation_callable_names
        if callable_names is None:
            callable_names = {tool.name for tool in self.list_tools()}
        if requested_tool in callable_names:
            return ToolResult(
                metadata={
                    "status": "tool_already_callable",
                    "requested_tool": requested_tool,
                    "callable_this_turn": True,
                }
            )

        activation_signature = (
            self._activation_request_signature(
                requested_tool,
                context=context,
                discoverable_tool_names=self._activation_discoverable_names,
                view_identity=id(self),
                source_identity=id(self._activation_source or self),
            )
            if context is not None
            else None
        )
        denied_signatures = (
            context.state.get("denied_tool_activation_signatures")
            if context is not None
            else None
        )
        if (
            activation_signature is not None
            and isinstance(denied_signatures, set)
            and activation_signature in denied_signatures
        ):
            return self._activation_replay_denied(requested_tool)

        source = self._activation_source or self
        context_discoverable_names = policy.get("discoverable_tool_names")
        if (
            isinstance(context_discoverable_names, list)
            and bool(context_discoverable_names)
            and requested_tool not in context_discoverable_names
        ):
            return self._activation_denied(
                requested_tool=requested_tool,
                reason="not_discoverable",
                context=context,
            )
        if requested_tool not in self._activation_discoverable_names:
            return self._activation_denied(
                requested_tool=requested_tool,
                reason="not_discoverable",
                context=context,
            )
        tool = source.get(requested_tool)
        if tool is None:
            return self._activation_denied(
                requested_tool=requested_tool,
                reason="tool_not_found",
                context=context,
            )
        if context is None:
            return self._activation_denied(
                requested_tool=requested_tool,
                reason="missing_execution_context",
                context=context,
            )

        activation_diagnostics: dict[str, Any] = {
            "eligibility_source": "catalog_and_activation_policy",
            "activation_authorizes_tool_call": False,
        }
        mutation_activation_eligible = True
        mutation_denial_reason: str | None = None
        if requested_tool in _MUTATION_TOOLS:
            (
                mutation_activation_eligible,
                eligibility_source,
                mutation_denial_reason,
            ) = (
                _resolve_mutation_activation_eligibility(policy)
            )
            activation_diagnostics.update(
                {
                    "mutation_eligibility_source": eligibility_source,
                    "mutation_eligibility": (
                        "eligible" if mutation_activation_eligible else "forbidden"
                    ),
                }
            )

        execution_profile = str(policy.get("execution_profile") or "chat").strip().lower()
        readonly_profiles = {
            "subagent_readonly",
            "subagent_execution_request",
            "subagent_research",
            "judge",
            "verifier",
            "controller_exec",
        }
        if requested_tool in _RISKY_ACTIVATION_TOOLS and execution_profile in readonly_profiles:
            return self._activation_denied(
                requested_tool=requested_tool,
                reason="execution_profile_disallows_activation",
                context=context,
                diagnostics=activation_diagnostics,
            )

        tool_mode = str(policy.get("tool_mode") or "auto").strip().lower()
        if tool_mode == "disabled":
            return self._activation_denied(
                requested_tool=requested_tool,
                reason="tool_mode_disabled",
                context=context,
                diagnostics=activation_diagnostics,
            )

        allowlist = policy.get("tool_allowlist")
        if isinstance(allowlist, list) and requested_tool not in allowlist:
            return self._activation_denied(
                requested_tool=requested_tool,
                reason="allowlist_excluded",
                context=context,
                diagnostics=activation_diagnostics,
            )
        denylist = policy.get("tool_denylist")
        if isinstance(denylist, list) and requested_tool in denylist:
            return self._activation_denied(
                requested_tool=requested_tool,
                reason="denylist_blocked",
                context=context,
                diagnostics=activation_diagnostics,
            )

        allowed_tool_names = policy.get("activation_allowed_tool_names")
        capability_eligible = (
            policy.get("capability_enforcement_mode") == "enforce"
            and isinstance(allowed_tool_names, list)
            and requested_tool in allowed_tool_names
        )
        activation_diagnostics.update(
            {
                "capability_enforcement_mode": "enforce",
                "capability_plan_eligibility": (
                    "eligible" if capability_eligible else "ineligible"
                ),
                "eligibility_source": "capability_plan.eligible_tools",
            }
        )
        if not capability_eligible:
            return self._activation_denied(
                requested_tool=requested_tool,
                reason="contract_capability_mismatch",
                context=context,
                diagnostics=activation_diagnostics,
            )

        if not mutation_activation_eligible:
            return self._activation_denied(
                requested_tool=requested_tool,
                reason=(
                    mutation_denial_reason
                    or "contract_disallows_mutation_activation"
                ),
                context=context,
                diagnostics=activation_diagnostics,
            )

        if requested_tool in _MUTATION_TOOLS:
            approval_required, approval_policy_source = (
                _resolve_mutation_call_approval_hint(tool=tool, context=context)
            )
            activation_diagnostics.update(
                {
                    "authorization_required_for_call": approval_required,
                    "authorization_policy_source": approval_policy_source,
                    "authorization_state": "deferred_to_tool_call",
                }
            )
            workspace_dir = str(context.workspace_dir or "").strip()
            tool_workspace = getattr(tool, "_workspace_dir", None)
            workspace_mismatch = False
            if workspace_dir and tool_workspace is not None and not context.task_sandbox_dir:
                try:
                    workspace_mismatch = (
                        Path(workspace_dir).expanduser().resolve()
                        != Path(str(tool_workspace)).expanduser().resolve()
                    )
                except OSError:
                    workspace_mismatch = True
            if not workspace_dir or workspace_mismatch:
                return self._activation_denied(
                    requested_tool=requested_tool,
                    reason="workspace_security_rejected",
                    context=context,
                    diagnostics=activation_diagnostics,
                )

        callable_names.add(requested_tool)
        self._activation_callable_names = callable_names
        self._register_tool(tool)
        if requested_tool not in self._activation_callable_order:
            self._activation_callable_order.append(requested_tool)
        evicted_tools, activation_broker_retained = (
            self._reconcile_activated_schema_budget(
                requested_tool=requested_tool,
            )
        )
        return ToolResult(
            metadata={
                "status": "tool_activated",
                "requested_tool": requested_tool,
                "callable_this_turn": True,
                "activation_scope": "current_registry_view",
                "activation_schema_limit": self._activation_schema_limit,
                "activation_schema_count": len(self._tools),
                "activation_schema_evicted_tools": evicted_tools,
                "activation_broker_retained": activation_broker_retained,
                **activation_diagnostics,
            }
        )

    def _reconcile_activated_schema_budget(
        self,
        *,
        requested_tool: str,
    ) -> tuple[list[str], bool]:
        """Keep an activated tool visible without exceeding the view schema cap."""

        callable_names = self._activation_callable_names
        if callable_names is None:
            callable_names = {tool.name for tool in self.list_tools()}
            self._activation_callable_names = callable_names

        deferred_names = self._activation_discoverable_names - callable_names
        if not deferred_names:
            self._tools.pop("tool_activate", None)
            callable_names.discard("tool_activate")
            self._activation_callable_order = [
                name
                for name in self._activation_callable_order
                if name != "tool_activate"
            ]
            return [], False

        schema_limit = self._activation_schema_limit
        if schema_limit is None or len(self._tools) <= schema_limit:
            return [], "tool_activate" in self._tools

        protected = {requested_tool, "tool_activate"}
        preferred_evictions = [
            name
            for name in reversed(self._activation_callable_order)
            if name in self._tools
            and name not in protected
            and name not in {"tool_search", "tool_result_read"}
        ]
        fallback_evictions = [
            name
            for name in ("tool_result_read", "tool_search")
            if name in self._tools and name not in protected
        ]
        evicted: list[str] = []
        for name in [*preferred_evictions, *fallback_evictions]:
            if len(self._tools) <= schema_limit:
                break
            self._tools.pop(name, None)
            callable_names.discard(name)
            self._activation_discoverable_names.add(name)
            self._activation_callable_order = [
                candidate
                for candidate in self._activation_callable_order
                if candidate != name
            ]
            evicted.append(name)

        if len(self._tools) > schema_limit:
            self._tools.pop("tool_activate", None)
            callable_names.discard("tool_activate")
            self._activation_callable_order = [
                name
                for name in self._activation_callable_order
                if name != "tool_activate"
            ]
        return evicted, "tool_activate" in self._tools

    def _activation_denied(
        self,
        *,
        requested_tool: str,
        reason: str,
        context: ToolExecutionContext | None,
        diagnostics: dict[str, Any] | None = None,
    ) -> ToolResult:
        if context is not None:
            self._remember_denied_activation(context, requested_tool)
        return ToolResult(
            error=f"tool_activation_denied: {reason}",
            metadata={
                "runtime_category": "tool_activation",
                "error_type": "tool_activation_denied",
                "requested_tool": requested_tool,
                "reason": reason,
                "recoverability": "requires_policy_change_or_replanning",
                "retryable": False,
                **(diagnostics or {}),
            },
            retryable=False,
            suggestion="Change the activation context or choose a callable tool; do not replay unchanged.",
        )

    def _remember_denied_activation(self, context: ToolExecutionContext, tool_name: str) -> None:
        denied = context.state.setdefault("denied_tool_activation_names", set())
        if not isinstance(denied, set):
            denied = set()
            context.state["denied_tool_activation_names"] = denied
        denied.add(tool_name)
        signatures = context.state.setdefault("denied_tool_activation_signatures", set())
        if not isinstance(signatures, set):
            signatures = set()
            context.state["denied_tool_activation_signatures"] = signatures
        signatures.add(
            self._activation_request_signature(
                tool_name,
                context=context,
                discoverable_tool_names=self._activation_discoverable_names,
                view_identity=id(self),
                source_identity=id(self._activation_source or self),
            )
        )

    @staticmethod
    def _activation_request_signature(
        tool_name: str,
        *,
        context: ToolExecutionContext,
        discoverable_tool_names: set[str] | None = None,
        view_identity: int | None = None,
        source_identity: int | None = None,
    ) -> str:
        policy = context.state.get("tool_activation_policy")
        policy = policy if isinstance(policy, dict) else {}
        permission_policy = context.permission_policy
        path_signatures: dict[str, str | None] = {}
        for key in ("workspace_dir", "task_sandbox_dir"):
            value = getattr(context, key, None)
            if not value:
                path_signatures[key] = None
                continue
            try:
                path_signatures[key] = str(Path(str(value)).expanduser().resolve())
            except OSError:
                path_signatures[key] = str(value)
        return ToolRegistry._tool_call_signature(
            name=f"activation:{tool_name}",
            args={
                "mutation_requirement": policy.get("mutation_requirement"),
                "requested_operations": policy.get("requested_operations"),
                "required_capabilities": policy.get("required_capabilities"),
                "capability_enforcement_mode": policy.get(
                    "capability_enforcement_mode"
                ),
                "activation_allowed_tool_names": policy.get(
                    "activation_allowed_tool_names"
                ),
                "execution_profile": policy.get("execution_profile"),
                "tool_mode": policy.get("tool_mode"),
                "discoverable_tool_names": policy.get("discoverable_tool_names"),
                "view_discoverable_tool_names": sorted(discoverable_tool_names or set()),
                "view_identity": view_identity,
                "source_identity": source_identity,
                "tool_allowlist": policy.get("tool_allowlist"),
                "tool_denylist": policy.get("tool_denylist"),
                "require_approval_for_file_write": permission_policy.get(
                    "require_approval_for_file_write"
                ),
                "approved_activation_tools": permission_policy.get(
                    "approved_activation_tools"
                ),
                "approved_tool_calls": permission_policy.get("approved_tool_calls"),
                "workspace_dir": path_signatures["workspace_dir"],
                "task_sandbox_dir": path_signatures["task_sandbox_dir"],
            },
        )

    @staticmethod
    def _activation_replay_denied(tool_name: str) -> ToolResult:
        return ToolResult(
            error=(
                f"tool_activation_denied: the exact activation request for '{tool_name}' "
                "was already denied and cannot be replayed unchanged."
            ),
            metadata={
                "runtime_category": "tool_activation",
                "error_type": "tool_activation_denied",
                "requested_tool": tool_name,
                "reason": "activation_denied_replay",
                "recoverability": "requires_policy_change_or_replanning",
                "replay_safe": False,
            },
            retryable=False,
            suggestion="Change the policy context or obtain approval before retrying.",
        )

    async def execute(
        self,
        name: str,
        args: dict[str, Any],
        *,
        context: ToolExecutionContext | None = None,
    ) -> ToolResult:
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"Tool '{name}' not found in registry.")

        execution_args = dict(args)
        timeline_binding: dict[str, str] | None = None
        lifecycle = _timeline_lifecycle(context)
        try:
            denied_result = self._check_denied_tool_call(name=name, args=execution_args, context=context)
            if denied_result is not None:
                return denied_result

            validation_error = tool.validate_input(execution_args, context)
            if validation_error is not None:
                return validation_error

            permission_error = tool.check_permissions(execution_args, context)
            if permission_error is not None:
                return permission_error

            if lifecycle is not None and tool.name in _TIMELINE_UNSUPPORTED_SIDE_EFFECT_TOOLS:
                if context is None:
                    raise RuntimeError("timeline lifecycle requires a tool execution context")
                await lifecycle.block_unstarted_turn()
                return ToolResult(
                    error=(
                        "timeline_unsupported_side_effect: this ordinary Chat timeline "
                        "only supports file_write, file_edit, and apply_patch."
                    ),
                    metadata={
                        "status": "timeline_unsupported_side_effect",
                        "tool_name": name,
                        "timeline_fail_closed": True,
                        "timeline_unstarted_blocked": True,
                    },
                    retryable=False,
                )

            if lifecycle is not None and tool.name in _TIMELINE_LIFECYCLE_TOOLS:
                if context is None:
                    raise RuntimeError("timeline lifecycle requires a tool execution context")
                approval_mode = _timeline_approval_mode(tool)
                if (
                    tool.is_read_only
                    or not tool.supports_timeline_side_effect_boundary
                    or (
                        approval_mode == "revocable"
                        and not tool.supports_timeline_approval_revocation
                    )
                ):
                    await lifecycle.block_unstarted_turn()
                    return ToolResult(
                        error=(
                            "timeline_boundary_aware_tool_required: this ordinary Chat "
                            "timeline only dispatches first-party tools that record their "
                            "effect boundary before acting."
                        ),
                        metadata={
                            "status": "timeline_boundary_aware_tool_required",
                            "tool_name": name,
                            "timeline_fail_closed": True,
                            "timeline_unstarted_blocked": True,
                        },
                        retryable=False,
                    )
                if (
                    _timeline_requires_approval(tool=tool, context=context)
                    and approval_mode != "continuable"
                ):
                    await lifecycle.block_unstarted_turn()
                    return ToolResult(
                        error=(
                            "timeline_approval_capability_missing: this ordinary Chat tool "
                            "cannot persist an exact approval continuation."
                        ),
                        metadata={
                            "status": "timeline_approval_capability_missing",
                            "tool_name": name,
                            "requires_approval": True,
                            "timeline_approval_mode": approval_mode,
                            "timeline_fail_closed": True,
                            "timeline_unstarted_blocked": True,
                        },
                        retryable=False,
                    )
                call_id = str(context.state.get("timeline_tool_call_id") or "").strip()
                if not call_id:
                    await lifecycle.block_unstarted_turn()
                    return ToolResult(
                        error="timeline_call_id_missing: mutation cannot be bound to a durable tool call.",
                        metadata={
                            "status": "timeline_binding_failed",
                            "tool_name": name,
                            "timeline_fail_closed": True,
                            "timeline_unstarted_blocked": True,
                        },
                        retryable=False,
                    )
                operation_id, arguments_digest = await lifecycle.precommit_mutation(
                    tool_name=name,
                    arguments=execution_args,
                    call_id=call_id,
                )
                timeline_binding = {
                    "operation_id": operation_id,
                    "arguments_digest": arguments_digest,
                    "call_id": call_id,
                    "tool_name": name,
                    "arguments": dict(execution_args),
                }
                context.state["timeline_pending_operation"] = timeline_binding
                context.state.pop("timeline_operation_started", None)

            execute_signature = inspect.signature(tool.execute)
            accepts_keyword_context = any(
                parameter.kind is inspect.Parameter.VAR_KEYWORD
                for parameter in execute_signature.parameters.values()
            )
            if (
                context is not None
                and "approved" in execute_signature.parameters
                and self._is_auto_approved_call(name=name, args=execution_args, context=context)
            ):
                execution_args["approved"] = True
            if "context" in execute_signature.parameters or accepts_keyword_context:
                result = await tool.execute(**execution_args, context=context)
            else:
                result = await tool.execute(**execution_args)
            if timeline_binding is not None:
                approval_id = (
                    str(result.metadata.get("approval_id") or "").strip()
                    if result.metadata.get("status") == "approval_pending"
                    else ""
                )
                approval_mode = _timeline_approval_mode(tool)
                continuation_pending = False
                if approval_id:
                    metadata = dict(result.metadata or {})
                    binding_matches = (
                        metadata.get("operation_id") == timeline_binding["operation_id"]
                        and metadata.get("arguments_digest")
                        == timeline_binding["arguments_digest"]
                    )
                    if binding_matches:
                        try:
                            binding_matches = bool(
                                tool.validates_timeline_approval_binding(
                                    approval_id,
                                    operation_id=timeline_binding["operation_id"],
                                    arguments_digest=timeline_binding["arguments_digest"],
                                    call_id=timeline_binding["call_id"],
                                )
                            )
                        except Exception as exc:
                            logger.warning(
                                "Tool '%s' could not validate pending timeline approval '%s': %s",
                                name,
                                approval_id,
                                exc,
                            )
                            binding_matches = False
                    if approval_mode == "continuable" and binding_matches:
                        # The durable approval is the interrupt. Keep the
                        # precommitted/no-effect descriptor instead of
                        # superseding a valid continuation.
                        await lifecycle.block_unstarted_turn()
                        result.metadata = {
                            **metadata,
                            **timeline_operation_metadata(context),
                            "timeline_approval_pending": True,
                            "timeline_approval_mode": approval_mode,
                        }
                        continuation_pending = True
                    else:
                        invalidated = False
                        if tool.supports_timeline_approval_revocation:
                            try:
                                invalidated = await asyncio.to_thread(
                                    tool.revoke_timeline_approval,
                                    approval_id,
                                    reason=(
                                        "ordinary Chat rejected an approval that did not "
                                        "match its exact precommitted operation"
                                    ),
                                )
                            except Exception as exc:
                                logger.warning(
                                    "Tool '%s' could not revoke post-precommit approval '%s': %s",
                                    name,
                                    approval_id,
                                    exc,
                                )
                        if not invalidated:
                            # A tool that breaks its declared approval contract
                            # may already have crossed an undocumented boundary.
                            # Quarantine rather than replay its exact call.
                            await lifecycle.mark_mutation_started(
                                operation_id=timeline_binding["operation_id"]
                            )
                            context.state["timeline_operation_started"] = timeline_binding[
                                "operation_id"
                            ]
                            return ToolResult(
                                error=(
                                    "timeline_approval_contract_failed: a post-precommit "
                                    "approval could not be durably reconciled."
                                ),
                                metadata={
                                    **metadata,
                                    **timeline_operation_metadata(context),
                                    "timeline_effect_started": True,
                                    "timeline_result_disposition": "unknown",
                                    "timeline_result_unknown": True,
                                    "timeline_approval_mode": approval_mode,
                                },
                                retryable=False,
                            )
                        result = ToolResult(
                            error=(
                                "timeline_approval_invalidated: a non-continuable approval "
                                "was superseded before the effect boundary."
                            ),
                            metadata={
                                **metadata,
                                "status": "timeline_approval_invalidated",
                                "timeline_approval_invalidated": True,
                                "timeline_pre_effect_abandoned": True,
                                "timeline_approval_mode": approval_mode,
                            },
                            retryable=False,
                        )
                if not continuation_pending:
                    started = (
                        context is not None
                        and context.state.get("timeline_operation_started")
                        == timeline_binding["operation_id"]
                    )
                    if not started:
                        result = ToolResult(
                            error=(
                                "timeline_effect_boundary_not_reached: mutation was stopped before "
                                "a durable effect boundary."
                            ),
                            metadata={
                                **dict(result.metadata or {}),
                                "status": "timeline_pre_effect_blocked",
                                "timeline_fail_closed": True,
                                "timeline_pre_effect_abandoned": True,
                            },
                            retryable=False,
                        )
                    else:
                        disposition = _timeline_result_disposition(result)
                        result.metadata = {
                            **dict(result.metadata or {}),
                            "timeline_effect_started": True,
                            "timeline_result_disposition": disposition,
                            **({"timeline_result_unknown": True} if disposition == "unknown" else {}),
                        }
                result.metadata = {**dict(result.metadata or {}), **timeline_operation_metadata(context)}
            return result
        except Exception as exc:
            logger.warning("Tool '%s' execution error: %s", name, exc)
            result = ToolResult(error=str(exc))
            if timeline_binding is not None:
                started = (
                    context is not None
                    and context.state.get("timeline_operation_started")
                    == timeline_binding["operation_id"]
                )
                if not started:
                    result.metadata = {
                        **timeline_operation_metadata(context),
                        "timeline_fail_closed": True,
                        "timeline_pre_effect_abandoned": True,
                    }
                else:
                    result.metadata = {
                        **timeline_operation_metadata(context),
                        "timeline_effect_started": True,
                        "timeline_result_unknown": True,
                    }
            elif lifecycle is not None and tool.name in _TIMELINE_LIFECYCLE_TOOLS:
                result = ToolResult(
                    error=f"timeline_lifecycle_failed: {exc}",
                    metadata={
                        "status": "timeline_lifecycle_failed",
                        "tool_name": name,
                        "timeline_fail_closed": True,
                        "timeline_pre_effect_failure": True,
                    },
                    retryable=False,
                )
            return result
        finally:
            if timeline_binding is not None and context is not None:
                context.state.pop("timeline_pending_operation", None)
                context.state.pop("timeline_operation_started", None)

    @staticmethod
    def _tool_call_signature(*, name: str, args: dict[str, Any]) -> str:
        return json.dumps(
            {"tool_name": name, "arguments": args},
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )

    def _check_denied_tool_call(
        self,
        *,
        name: str,
        args: dict[str, Any],
        context: ToolExecutionContext | None,
    ) -> ToolResult | None:
        if context is None:
            return None

        signature = self._tool_call_signature(name=name, args=args)
        denied_signatures = context.state.setdefault("denied_tool_call_signatures", set())
        if not isinstance(denied_signatures, set):
            denied_signatures = set()
            context.state["denied_tool_call_signatures"] = denied_signatures

        denied_calls = context.permission_policy.get("denied_tool_calls")
        if isinstance(denied_calls, list):
            for candidate in denied_calls:
                if not isinstance(candidate, dict):
                    continue
                candidate_name = candidate.get("tool_name")
                candidate_args = candidate.get("arguments")
                if candidate_name != name or not isinstance(candidate_args, dict):
                    continue
                denied_signatures.add(
                    self._tool_call_signature(name=str(candidate_name), args=candidate_args)
                )

        if signature not in denied_signatures:
            return None

        decision = deny_security_decision(
            reason="This exact tool call was already denied and cannot be retried unchanged.",
            approval_scope="task_resume",
            replay_safe=False,
            policy_source="approval_memory",
        )
        metadata = decision.to_metadata()
        metadata.update(
            {
                "runtime_category": "permission",
                "error_type": "tool_denied",
                "recoverability": "requires_changed_tool_call",
                "status": "tool_denied",
                "code": "tool_denied",
                "tool_name": name,
                "arguments": args,
                "requires_approval": False,
            }
        )
        return ToolResult(
            error="tool_denied: This exact tool call was already denied and cannot be retried unchanged.",
            metadata=metadata,
            retryable=False,
            suggestion="Change the arguments or choose a different approach before retrying.",
        )

    @staticmethod
    def _is_auto_approved_call(
        *,
        name: str,
        args: dict[str, Any],
        context: ToolExecutionContext,
    ) -> bool:
        approved_calls = context.permission_policy.get("approved_tool_calls")
        if not isinstance(approved_calls, list):
            return False
        for candidate in approved_calls:
            if not isinstance(candidate, dict):
                continue
            if candidate.get("tool_name") != name:
                continue
            candidate_args = candidate.get("arguments")
            if isinstance(candidate_args, dict) and candidate_args == args:
                return True
        return False

    def format_result_for_model(
        self,
        name: str,
        result: ToolResult,
        *,
        max_chars: int = 2000,
    ) -> str:
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"Tool '{name}' not found in registry.")
        return tool.format_result_for_model(result, max_chars=max_chars)

    def summarize_result_for_ui(self, name: str, result: ToolResult) -> str:
        tool = self.get(name)
        if tool is None:
            raise KeyError(f"Tool '{name}' not found in registry.")
        return tool.summarize_result_for_ui(result)

    def _discover(self, directory: Path) -> None:
        if not directory.is_dir():
            logger.debug("Tool discovery directory not found: {}", directory)
            return

        for py_file in directory.rglob("*.py"):
            if py_file.name.startswith("_"):
                continue
            module_name = f"_mochi_tool_{py_file.stem}_{id(py_file)}"
            try:
                spec = importlib.util.spec_from_file_location(module_name, py_file)
                if spec is None or spec.loader is None:
                    continue
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)  # type: ignore[union-attr]

                build_tool = getattr(module, "build_tool", None)
                if callable(build_tool):
                    instance = build_tool()
                    if isinstance(instance, BaseTool):
                        self.register(instance)

                manifest = getattr(module, "TOOL_FACTORIES", None)
                if isinstance(manifest, dict):
                    for name, factory in manifest.items():
                        if callable(factory):
                            self.register_factory(str(name), factory)

                for _, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, BaseTool)
                        and obj is not BaseTool
                        and not inspect.isabstract(obj)
                    ):
                        try:
                            instance = obj()
                        except TypeError:
                            continue
                        self.register(instance)
            except Exception as exc:
                logger.warning("Failed to load tool from {}: {}", py_file, exc)
