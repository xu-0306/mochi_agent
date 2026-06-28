"""Shared backend tool-calling state transitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ToolCallMode = Literal["native", "simulated_fallback", "unavailable"]
FallbackValidationStatus = Literal["not_attempted", "validated", "rejected"]


@dataclass
class ToolCallingState:
    """Mutable backend-local state for tool-calling availability."""

    active_mode: ToolCallMode = "native"
    native_status: str = "unknown"
    fallback_validation_status: FallbackValidationStatus = "not_attempted"

    def enter_simulated(self, status: str) -> bool:
        changed = (
            self.active_mode != "simulated_fallback"
            or self.native_status != status
            or self.fallback_validation_status != "not_attempted"
        )
        self.active_mode = "simulated_fallback"
        self.native_status = status
        self.fallback_validation_status = "not_attempted"
        return changed

    def validate_simulated(self) -> bool:
        changed = self.fallback_validation_status != "validated"
        self.fallback_validation_status = "validated"
        return changed

    def recover_native(self, status: str) -> bool:
        changed = (
            self.active_mode != "native"
            or self.native_status != status
            or self.fallback_validation_status != "not_attempted"
        )
        self.active_mode = "native"
        self.native_status = status
        self.fallback_validation_status = "not_attempted"
        return changed

    def mark_unavailable(self, status: str) -> bool:
        changed = (
            self.active_mode != "unavailable"
            or self.native_status != status
            or self.fallback_validation_status != "rejected"
        )
        self.active_mode = "unavailable"
        self.native_status = status
        self.fallback_validation_status = "rejected"
        return changed

    def supports_tool_calling(self) -> bool:
        return self.active_mode != "unavailable"
