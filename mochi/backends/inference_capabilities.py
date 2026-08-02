"""Provider-aware inference capability resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from mochi.backends.types import ModelInfo

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh", "max"]

ALL_INFERENCE_PARAMETERS: tuple[str, ...] = (
    "system_prompt",
    "temperature",
    "max_tokens",
    "top_p",
    "min_p",
    "top_k",
    "frequency_penalty",
    "presence_penalty",
    "repeat_penalty",
    "reasoning_effort",
)
_SYSTEM_PROMPT_AND_REASONING_ONLY: tuple[str, ...] = ("system_prompt", "reasoning_effort")
_ANTHROPIC_COMPAT_PARAMETERS: tuple[str, ...] = (
    "system_prompt",
    "temperature",
    "max_tokens",
    "top_p",
)
_LOW_MEDIUM_HIGH: tuple[ReasoningEffort, ...] = ("low", "medium", "high")
_EFFORT_ORDER: tuple[ReasoningEffort, ...] = (
    "none",
    "minimal",
    "low",
    "medium",
    "high",
    "xhigh",
    "max",
)
_NONZERO_EFFORT_ORDER: tuple[ReasoningEffort, ...] = _EFFORT_ORDER[1:]


@dataclass(frozen=True)
class InferenceCapabilities:
    """Resolved capability set for one active model."""

    provider: str | None
    supported_inference_parameters: tuple[str, ...]
    supported_reasoning_efforts: tuple[ReasoningEffort, ...] = ()
    policy_label: str | None = None
    policy_message: str | None = None
    capability_source: str = "unknown"
    capability_status: str = "unknown"
    capability_checked_at: str | None = None

    @property
    def supports_reasoning_effort(self) -> bool:
        return len(self.supported_reasoning_efforts) > 0

    def to_metadata(self) -> dict[str, Any]:
        return {
            "supported_inference_parameters": list(self.supported_inference_parameters),
            "supported_reasoning_efforts": list(self.supported_reasoning_efforts),
            "supports_reasoning_effort": self.supports_reasoning_effort,
            "inference_policy_label": self.policy_label,
            "inference_policy_message": self.policy_message,
            "capability_source": self.capability_source,
            "capability_status": self.capability_status,
            "capability_checked_at": self.capability_checked_at,
        }


def resolve_model_inference_capabilities(model_info: ModelInfo) -> InferenceCapabilities:
    """Resolve supported inference controls for the current model/provider pair."""

    metadata = model_info.metadata if isinstance(model_info.metadata, dict) else {}
    provider = model_info.provider or _string_or_none(metadata.get("provider"))
    backend_type = (model_info.backend_type or "").strip().lower()

    if backend_type == "openai_compat":
        return _resolve_openai_compat_capabilities(
            provider=provider,
            model_name=model_info.name,
            api_mode=_string_or_none(metadata.get("api_mode")),
            metadata=metadata,
        )

    if backend_type == "ollama":
        efforts = _LOW_MEDIUM_HIGH if metadata.get("supports_reasoning_effort") is True else ()
        return InferenceCapabilities(
            provider=provider or "ollama",
            supported_inference_parameters=ALL_INFERENCE_PARAMETERS,
            supported_reasoning_efforts=efforts,
        )

    if backend_type in {"gguf", "safetensors"}:
        return InferenceCapabilities(
            provider=provider or "local",
            supported_inference_parameters=ALL_INFERENCE_PARAMETERS,
            supported_reasoning_efforts=(),
        )

    if metadata.get("supports_reasoning_effort") is True:
        return InferenceCapabilities(
            provider=provider,
            supported_inference_parameters=ALL_INFERENCE_PARAMETERS,
            supported_reasoning_efforts=_LOW_MEDIUM_HIGH,
        )

    return InferenceCapabilities(
        provider=provider,
        supported_inference_parameters=ALL_INFERENCE_PARAMETERS,
        supported_reasoning_efforts=(),
    )


def sanitize_inference_params_for_capabilities(
    params: dict[str, Any] | None,
    capabilities: InferenceCapabilities,
) -> dict[str, Any]:
    """Drop unsupported inference overrides while preserving supported values."""

    if not params:
        return {}

    allowed = set(capabilities.supported_inference_parameters)
    sanitized: dict[str, Any] = {
        key: value
        for key, value in params.items()
        if key in allowed and value is not None
    }

    effort = params.get("reasoning_effort")
    if effort in capabilities.supported_reasoning_efforts:
        sanitized["reasoning_effort"] = cast(ReasoningEffort, effort)
    else:
        sanitized.pop("reasoning_effort", None)

    return sanitized


def select_lowest_reasoning_effort(
    capabilities: InferenceCapabilities,
    *,
    include_off: bool = False,
) -> ReasoningEffort | None:
    """Choose the cheapest supported effort, without conflating ``none`` and work.

    ``none`` means explicitly disable reasoning.  Background tasks normally want
    the least non-zero effort instead, so callers must opt in to ``include_off``.
    """

    order = _EFFORT_ORDER if include_off else _NONZERO_EFFORT_ORDER
    for effort in order:
        if effort in capabilities.supported_reasoning_efforts:
            return effort
    return None


def parse_model_capability_metadata(metadata: Any) -> tuple[ReasoningEffort, ...]:
    """Extract explicit effort enums from a model metadata extension.

    OpenAI's standard Models response has no such field.  In particular,
    ``supported_parameters: {reasoning_effort: true}`` is deliberately not
    treated as an enum declaration.
    """

    if not isinstance(metadata, dict):
        return ()
    candidates: list[Any] = [metadata.get("supported_reasoning_efforts")]
    capabilities = metadata.get("capabilities")
    if isinstance(capabilities, dict):
        effort = capabilities.get("effort")
        if isinstance(effort, dict):
            candidates.append(effort.get("supported"))
            # Anthropic exposes each level as either a bare boolean or an
            # object such as ``{"supported": true}``.
            enabled = [
                name
                for name in _EFFORT_ORDER
                if _metadata_capability_enabled(effort.get(name))
            ]
            if enabled:
                candidates.append(enabled)
    for candidate in candidates:
        if isinstance(candidate, (list, tuple, set)):
            declared = {value for value in candidate if value in _EFFORT_ORDER}
            return tuple(effort for effort in _EFFORT_ORDER if effort in declared)
    return ()


def _metadata_capability_enabled(value: Any) -> bool:
    return value is True or (
        isinstance(value, dict) and value.get("supported") is True
    )


def _resolve_openai_compat_capabilities(
    *,
    provider: str | None,
    model_name: str,
    api_mode: str | None,
    metadata: dict[str, Any],
) -> InferenceCapabilities:
    normalized_provider = (provider or "openai_compat").strip().lower()
    normalized_model = model_name.strip().lower()
    normalized_api_mode = (api_mode or "chat_completions").strip().lower()

    declared_efforts = parse_model_capability_metadata(
        metadata.get("capability_metadata", metadata)
    )
    metadata_source = _string_or_none(metadata.get("capability_source"))
    metadata_status = _string_or_none(metadata.get("capability_status"))
    checked_at = _string_or_none(metadata.get("capability_checked_at"))
    # This backend speaks OpenAI-compatible Chat/Responses only.  Anthropic's
    # native ``capabilities.effort`` uses ``output_config.effort``; retain the
    # metadata for diagnostics but never pretend this transport can send it.
    if normalized_provider == "anthropic":
        return InferenceCapabilities(
            provider="anthropic",
            supported_inference_parameters=_ANTHROPIC_COMPAT_PARAMETERS,
            supported_reasoning_efforts=(),
            policy_label="Anthropic compatibility",
            policy_message="Anthropic native effort metadata requires output_config.effort, which the OpenAI-compatible backend does not implement.",
            capability_source=metadata_source or "provider_policy",
            capability_status=metadata_status or "unavailable",
            capability_checked_at=checked_at,
        )

    if declared_efforts:
        return InferenceCapabilities(
            provider=normalized_provider,
            supported_inference_parameters=_SYSTEM_PROMPT_AND_REASONING_ONLY,
            supported_reasoning_efforts=declared_efforts,
            policy_label="Endpoint capability metadata",
            policy_message="Reasoning effort levels were declared by this endpoint's model metadata.",
            capability_source=metadata_source or "endpoint_metadata",
            capability_status=metadata_status or "resolved",
            capability_checked_at=checked_at,
        )

    if normalized_provider == "gemini":
        efforts: tuple[ReasoningEffort, ...]
        if "gemini-2.5" in normalized_model and "pro" not in normalized_model:
            efforts = ("none", "minimal", "low", "medium", "high")
        else:
            efforts = ("minimal", "low", "medium", "high")
        return InferenceCapabilities(
            provider="gemini",
            supported_inference_parameters=_SYSTEM_PROMPT_AND_REASONING_ONLY,
            supported_reasoning_efforts=efforts,
            policy_label="Gemini reasoning controls",
            policy_message="Gemini OpenAI-compatible models use provider-managed thinking controls. Other sampling overrides are disabled on chat.",
        )

    if normalized_provider == "openai_compat" and normalized_model.startswith("gpt-5"):
        if normalized_model.startswith(("gpt-5.6", "gpt-5.5", "gpt-5.4")):
            efforts = ("none", "low", "medium", "high", "xhigh", "max") if normalized_model.startswith("gpt-5.6") else ("none", "low", "medium", "high", "xhigh")
        elif normalized_model.startswith("gpt-5.2"):
            efforts = ("none", "low", "medium", "high", "xhigh")
        elif normalized_model.startswith("gpt-5.1"):
            efforts = ("none", "low", "medium", "high")
        elif normalized_model.startswith("gpt-5-pro"):
            efforts = ("high",)
        else:
            efforts = ("minimal", "low", "medium", "high")
        return InferenceCapabilities(
            provider="openai_compat",
            supported_inference_parameters=_SYSTEM_PROMPT_AND_REASONING_ONLY,
            supported_reasoning_efforts=efforts,
            policy_label="GPT-5 inference policy",
            policy_message="GPT-5-family API models should be controlled with the system prompt and reasoning effort only.",
            capability_source="registry",
            capability_status=metadata_status or "resolved",
            capability_checked_at=checked_at,
        )

    default_efforts: tuple[ReasoningEffort, ...] = ()
    if normalized_api_mode == "responses":
        default_efforts = _LOW_MEDIUM_HIGH
    return InferenceCapabilities(
        provider=normalized_provider,
        supported_inference_parameters=ALL_INFERENCE_PARAMETERS,
        supported_reasoning_efforts=default_efforts,
        capability_source=metadata_source or "unknown",
        capability_status=metadata_status or "unavailable",
        capability_checked_at=checked_at,
    )


def _string_or_none(value: Any) -> str | None:
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed or None
    return None
