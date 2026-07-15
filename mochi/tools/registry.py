"""Tool registry and discovery helpers."""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any

try:
    from loguru import logger
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal test envs
    import logging

    logger = logging.getLogger(__name__)

from mochi.security import deny_security_decision, require_approval_decision
from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult
from mochi.tools.tool_search import ToolSearchTool

ToolFactory = Any


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
    ) -> ToolRegistry:
        """Create a shallow registry view containing only the selected tools."""
        registry = ToolRegistry(discover_builtin=False)
        callable_names = set(tool_names)
        scoped_catalog_names = list(tool_search_catalog_names or tool_names)
        registry._activation_source = self
        registry._activation_discoverable_names = set(scoped_catalog_names)
        registry._activation_callable_names = callable_names
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

        routed_intent = str(policy.get("routed_intent") or "").strip().lower()
        mutation_tools = {"file_write", "file_edit", "apply_patch"}
        if requested_tool in mutation_tools and routed_intent != "workspace_write":
            return self._activation_denied(
                requested_tool=requested_tool,
                reason="routed_intent_disallows_activation",
                context=context,
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
        if requested_tool in mutation_tools and execution_profile in readonly_profiles:
            return self._activation_denied(
                requested_tool=requested_tool,
                reason="execution_profile_disallows_activation",
                context=context,
            )

        tool_mode = str(policy.get("tool_mode") or "auto").strip().lower()
        if tool_mode == "disabled":
            return self._activation_denied(
                requested_tool=requested_tool,
                reason="tool_mode_disabled",
                context=context,
            )

        allowlist = policy.get("tool_allowlist")
        if isinstance(allowlist, list) and requested_tool not in allowlist:
            return self._activation_denied(
                requested_tool=requested_tool,
                reason="allowlist_excluded",
                context=context,
            )
        denylist = policy.get("tool_denylist")
        if isinstance(denylist, list) and requested_tool in denylist:
            return self._activation_denied(
                requested_tool=requested_tool,
                reason="denylist_blocked",
                context=context,
            )

        if requested_tool in mutation_tools:
            permission_policy = context.permission_policy
            approval_required = bool(
                permission_policy.get("require_approval_for_file_write")
                or getattr(tool, "requires_approval", False)
            )
            approved_tools = permission_policy.get("approved_activation_tools")
            approved = isinstance(approved_tools, list) and requested_tool in approved_tools
            approved_tool_calls = permission_policy.get("approved_tool_calls")
            if isinstance(approved_tool_calls, list):
                approved = approved or any(
                    isinstance(candidate, dict)
                    and candidate.get("tool_name") == requested_tool
                    for candidate in approved_tool_calls
                )
            if approval_required and not approved:
                approval_kind = {
                    "file_write": "file_write",
                    "file_edit": "file_edit",
                    "apply_patch": "apply_patch",
                }.get(requested_tool, "other")
                decision = require_approval_decision(
                    reason="Activation requires approval before a workspace mutation tool can be promoted.",
                    approval_kind=approval_kind,
                    approval_scope="workspace",
                    policy_source="tool_activation_policy",
                )
                metadata = decision.to_metadata()
                metadata.update(
                    {
                        "runtime_category": "tool_activation",
                        "error_type": "tool_activation_denied",
                        "requested_tool": requested_tool,
                        "reason": "approval_required",
                        "recoverability": "requires_approval",
                    }
                )
                self._remember_denied_activation(context, requested_tool)
                return ToolResult(
                    error="tool_activation_denied: approval is required before activation.",
                    metadata=metadata,
                    retryable=False,
                    suggestion="Request or obtain approval, then retry with the same task context.",
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
                )

        callable_names.add(requested_tool)
        self._activation_callable_names = callable_names
        self._register_tool(tool)
        return ToolResult(
            metadata={
                "status": "tool_activated",
                "requested_tool": requested_tool,
                "callable_this_turn": True,
                "activation_scope": "current_registry_view",
            }
        )

    def _activation_denied(
        self,
        *,
        requested_tool: str,
        reason: str,
        context: ToolExecutionContext | None,
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
                "routed_intent": policy.get("routed_intent"),
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

            execute_signature = inspect.signature(tool.execute)
            if (
                context is not None
                and "approved" in execute_signature.parameters
                and self._is_auto_approved_call(name=name, args=execution_args, context=context)
            ):
                execution_args["approved"] = True
            if "context" in execute_signature.parameters:
                return await tool.execute(**execution_args, context=context)
            return await tool.execute(**execution_args)
        except Exception as exc:
            logger.warning("Tool '{}' execution error: {}", name, exc)
            return ToolResult(error=str(exc))

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
