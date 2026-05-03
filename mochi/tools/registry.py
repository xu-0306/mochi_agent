# Inspired by hermes-agent/tools/registry.py design pattern
"""工具自動發現注冊表。"""

from __future__ import annotations

import importlib
import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any

from loguru import logger

from mochi.tools.base import BaseTool, ToolResult


class ToolRegistry:
    """工具注冊表 — 自動掃描目錄並注冊 BaseTool 子類。

    放入 tools/ 或 tools/custom/ 目錄的 Python 檔案，
    只要 export 繼承 BaseTool 的類，就會被自動發現並注冊。
    """

    def __init__(
        self,
        extra_dirs: list[str] | None = None,
        discover_builtin: bool = True,
    ) -> None:
        """初始化注冊表並執行自動發現。

        Args:
            extra_dirs: 額外掃描目錄列表（絕對路徑或相對路徑）。
            discover_builtin: 是否自動掃描 mochi.tools 內建工具與 custom 目錄。
        """
        self._tools: dict[str, BaseTool] = {}
        if discover_builtin:
            package_dir = Path(__file__).resolve().parent
            self._discover(package_dir)
            self._discover(package_dir / "custom")
        if extra_dirs:
            for d in extra_dirs:
                self._discover(Path(d))

    def register(self, tool: BaseTool) -> None:
        """手動注冊工具。

        Args:
            tool: BaseTool 實例。
        """
        self._tools[tool.name] = tool
        logger.debug(f"Registered tool: {tool.name}")

    def get(self, name: str) -> BaseTool | None:
        """依名稱取得工具實例。

        Args:
            name: 工具名稱。
        """
        return self._tools.get(name)

    def list_tools(self) -> list[BaseTool]:
        """回傳所有已注冊的工具列表。"""
        return list(self._tools.values())

    def get_schemas(self) -> list[dict[str, Any]]:
        """回傳所有工具的 OpenAI function calling schema 列表。"""
        return [t.to_schema_dict() for t in self._tools.values()]

    async def execute(self, name: str, args: dict[str, Any]) -> ToolResult:
        """執行指定工具。

        Args:
            name: 工具名稱。
            args: 工具參數字典。

        Returns:
            ToolResult 執行結果。

        Raises:
            KeyError: 若工具不存在。
        """
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"Tool '{name}' not found in registry.")
        try:
            return await tool.execute(**args)
        except Exception as exc:
            logger.warning(f"Tool '{name}' execution error: {exc}")
            return ToolResult(error=str(exc))

    def _discover(self, directory: Path) -> None:
        """遞迴掃描目錄，自動 import 並注冊 BaseTool 子類。

        Args:
            directory: 掃描目錄路徑。
        """
        if not directory.is_dir():
            logger.debug(f"Tool discovery: directory not found: {directory}")
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

                for _name, obj in inspect.getmembers(module, inspect.isclass):
                    if (
                        issubclass(obj, BaseTool)
                        and obj is not BaseTool
                        and not inspect.isabstract(obj)
                    ):
                        instance = obj()
                        self.register(instance)
            except Exception as exc:
                logger.warning(f"Failed to load tool from {py_file}: {exc}")
