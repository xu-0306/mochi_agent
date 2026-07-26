"""Route-independent tool policy and schema-budget planning."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from mochi.backends.base import BaseLLMBackend


@dataclass(frozen=True)
class ToolExposurePlan:
    """Policy-bounded tool names for one turn."""

    tool_names: list[str]
    matched_groups: list[str]
    limit: int
    discoverable_tool_names: list[str] = field(default_factory=list)
    workspace_bound: bool = False
    attachment_count: int = 0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def exposure_metadata(self) -> dict[str, Any]:
        payload = {
            "exposed_tools": list(self.tool_names),
            "workspace_bound": self.workspace_bound,
            "attachment_count": self.attachment_count,
        }
        if self.diagnostics:
            payload["diagnostics"] = dict(self.diagnostics)
        return payload


class ToolExposurePlanner:
    """Build the non-semantic policy envelope used by capability planning."""

    _STRICT_BLOCKED_TOOLS = frozenset(
        {
            "exec_command",
            "execute_code",
            "execute_code_v2",
            "write_stdin",
            "kill_session",
            "process_poll",
            "process_stop",
            "mcp_call",
        }
    )
    _AUTONOMY_SCHEMA_LIMITS = {
        "strict": 4,
        "trusted_workspace": 6,
        "auto_review": 8,
        "high_autonomy": 10,
    }
    _POLICY_BASELINE_TOOLS = ("tool_search",)

    def __init__(self, *, tool_groups: dict[str, list[str]]) -> None:
        # Keep constructor compatibility while semantic grouping is owned by
        # TurnIntentContract and CapabilityPlan.
        del tool_groups

    def plan_contract_baseline(
        self,
        *,
        available_tool_names: list[str],
        backend: BaseLLMBackend,
        session_bound_workspace: bool,
        autonomy_mode: str | None = None,
        attachment_count: int = 0,
        tool_mode: Literal["disabled", "auto", "required"] = "auto",
    ) -> ToolExposurePlan:
        """Build a baseline that cannot observe message text or inferred intent."""

        normalized_attachment_count = max(0, attachment_count)
        if tool_mode == "disabled":
            return self._disabled_plan(
                reason="tool_mode_disabled",
                available_tool_count=len(available_tool_names),
                session_bound_workspace=session_bound_workspace,
                attachment_count=normalized_attachment_count,
                tool_mode=tool_mode,
            )

        backend_info = backend.get_model_info()
        metadata = backend_info.metadata if isinstance(backend_info.metadata, dict) else {}
        prompt_guided_ollama = (
            backend_info.backend_type == "ollama"
            and (metadata.get("tool_calling_protocol") or metadata.get("tool_calling_style"))
            == "prompt_guided"
        )
        if metadata.get("tool_calling_blocked") is True or (
            metadata.get("tool_call_mode") == "unavailable" and not prompt_guided_ollama
        ):
            reason = (
                "backend_tool_calling_blocked"
                if metadata.get("tool_calling_blocked") is True
                else "backend_tool_calling_unavailable"
            )
            return self._disabled_plan(
                reason=reason,
                available_tool_count=len(available_tool_names),
                session_bound_workspace=session_bound_workspace,
                attachment_count=normalized_attachment_count,
                tool_mode=tool_mode,
                backend=self._backend_diagnostics(backend_info, metadata),
            )

        base_limit = 6 if backend_info.backend_type in {"gguf", "safetensors"} else 10
        effective_mode = autonomy_mode or "trusted_workspace"
        limit = min(
            base_limit,
            self._AUTONOMY_SCHEMA_LIMITS.get(effective_mode, base_limit),
        )
        discoverable_tool_names = [
            name
            for name in dict.fromkeys(available_tool_names)
            if not (
                effective_mode == "strict" and name in self._STRICT_BLOCKED_TOOLS
            )
        ]
        baseline_tool_names = [
            name
            for name in self._POLICY_BASELINE_TOOLS
            if name in discoverable_tool_names
        ]
        return ToolExposurePlan(
            tool_names=baseline_tool_names,
            matched_groups=[],
            limit=limit,
            discoverable_tool_names=discoverable_tool_names,
            workspace_bound=session_bound_workspace,
            attachment_count=normalized_attachment_count,
            diagnostics={
                "stage": "contract_policy_baseline",
                "available_tool_count": len(available_tool_names),
                "policy_eligible_tool_count": len(discoverable_tool_names),
                "schema_budget": {"limit": limit},
                "autonomy_mode": effective_mode,
                "tool_mode": tool_mode,
                "baseline_policy_tools": list(baseline_tool_names),
            },
        )

    @staticmethod
    def _disabled_plan(
        *,
        reason: str,
        available_tool_count: int,
        session_bound_workspace: bool,
        attachment_count: int,
        tool_mode: str,
        backend: dict[str, Any] | None = None,
    ) -> ToolExposurePlan:
        diagnostics: dict[str, Any] = {
            "stage": "contract_policy_baseline",
            "disable_reason": reason,
            "available_tool_count": available_tool_count,
            "tool_mode": tool_mode,
        }
        if backend is not None:
            diagnostics["backend"] = backend
        return ToolExposurePlan(
            tool_names=[],
            matched_groups=[],
            limit=0,
            discoverable_tool_names=[],
            workspace_bound=session_bound_workspace,
            attachment_count=attachment_count,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _backend_diagnostics(backend_info: Any, metadata: dict[str, Any]) -> dict[str, Any]:
        keys = (
            "tool_call_mode",
            "tool_calling_protocol",
            "tool_calling_style",
            "tool_calling_blocked",
            "native_tool_calling_status",
            "fallback_validation_status",
        )
        return {
            "backend_type": getattr(backend_info, "backend_type", None),
            "provider": getattr(backend_info, "provider", None),
            "model": getattr(backend_info, "name", None),
            "metadata": {key: metadata.get(key) for key in keys if key in metadata},
        }
