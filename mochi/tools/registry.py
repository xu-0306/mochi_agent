"""Tool registry and discovery helpers."""

from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any

try:
    from loguru import logger
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal test envs
    import logging

    logger = logging.getLogger(__name__)

from mochi.tools.base import BaseTool, ToolExecutionContext, ToolResult

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
        if discover_builtin:
            package_dir = Path(__file__).resolve().parent
            self._discover(package_dir)
            self._discover(package_dir / "custom")
        if extra_dirs:
            for directory in extra_dirs:
                self._discover(Path(directory))

    def register(self, tool: BaseTool) -> None:
        self._tools[tool.name] = tool
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

    def create_view(self, tool_names: list[str]) -> ToolRegistry:
        """Create a shallow registry view containing only the selected tools."""
        registry = ToolRegistry(discover_builtin=False)
        for name in tool_names:
            tool = self.get(name)
            if tool is not None:
                registry.register(tool)
        return registry

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
