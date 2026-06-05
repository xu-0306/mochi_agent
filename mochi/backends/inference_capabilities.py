"""Provider-aware inference capability resolution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

from mochi.backends.types import ModelInfo

ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]

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


@dataclass(frozen=True)
class InferenceCapabilities:
    """Resolved capability set for one active model."""

    provider: str | None
    supported_inference_parameters: tuple[str, ...]
    supported_reasoning_efforts: tuple[ReasoningEffort, ...] = ()
    policy_label: str | None = None
    policy_message: str | None = None

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


def _resolve_openai_compat_capabilities(
    *,
    provider: str | None,
    model_name: str,
    api_mode: str | None,
) -> InferenceCapabilities:
    normalized_provider = (provider or "openai_compat").strip().lower()
    normalized_model = model_name.strip().lower()
    normalized_api_mode = (api_mode or "chat_completions").strip().lower()

    if normalized_provider == "anthropic":
        return InferenceCapabilities(
            provider="anthropic",
            supported_inference_parameters=_ANTHROPIC_COMPAT_PARAMETERS,
            supported_reasoning_efforts=(),
            policy_label="Anthropic compatibility",
            policy_message="Anthropic OpenAI-compatible endpoints ignore reasoning effort and honor only a subset of sampling controls.",
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
        if normalized_model.startswith("gpt-5.2"):
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
        )

    default_efforts: tuple[ReasoningEffort, ...] = ()
    if normalized_api_mode == "responses":
        default_efforts = _LOW_MEDIUM_HIGH
    return InferenceCapabilities(
        provider=normalized_provider,
        supported_inference_parameters=ALL_INFERENCE_PARAMETERS,
        supported_reasoning_efforts=default_efforts,
    )


def _string_or_none(value: Any) -> str | None:
    if isinstance(value, str):
        trimmed = value.strip()
        return trimmed or None
    return None
