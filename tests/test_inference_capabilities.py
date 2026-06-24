from __future__ import annotations

from mochi.backends.inference_capabilities import (
    resolve_model_inference_capabilities,
    sanitize_inference_params_for_capabilities,
)
from mochi.backends.types import ModelInfo


def _model_info(
    *,
    name: str,
    provider: str | None,
    backend_type: str = "openai_compat",
    api_mode: str = "responses",
) -> ModelInfo:
    return ModelInfo(
        name=name,
        provider=provider,
        backend_type=backend_type,
        supports_tool_calling=True,
        metadata={"api_mode": api_mode},
    )


def test_openai_gpt52_capabilities_are_system_prompt_plus_reasoning_only() -> None:
    info = _model_info(name="gpt-5.2", provider="openai_compat")

    capabilities = resolve_model_inference_capabilities(info)

    assert capabilities.provider == "openai_compat"
    assert capabilities.supported_inference_parameters == ("system_prompt", "reasoning_effort")
    assert capabilities.supported_reasoning_efforts == ("none", "low", "medium", "high", "xhigh")


def test_openai_gpt5_capabilities_keep_minimal_without_none_or_xhigh() -> None:
    info = _model_info(name="gpt-5", provider="openai_compat")

    capabilities = resolve_model_inference_capabilities(info)

    assert capabilities.supported_reasoning_efforts == ("minimal", "low", "medium", "high")


def test_gemini_compat_capabilities_advertise_reasoning_subset() -> None:
    info = _model_info(name="gemini-3.5-flash", provider="gemini")

    capabilities = resolve_model_inference_capabilities(info)

    assert capabilities.supported_inference_parameters == ("system_prompt", "reasoning_effort")
    assert capabilities.supported_reasoning_efforts == ("minimal", "low", "medium", "high")


def test_anthropic_compat_capabilities_do_not_advertise_reasoning_effort() -> None:
    info = _model_info(name="claude-sonnet-4-6", provider="anthropic")

    capabilities = resolve_model_inference_capabilities(info)

    assert capabilities.supported_inference_parameters == (
        "system_prompt",
        "temperature",
        "max_tokens",
        "top_p",
    )
    assert capabilities.supported_reasoning_efforts == ()


def test_sanitize_inference_params_drops_unsupported_gpt5_family_overrides() -> None:
    info = _model_info(name="gpt-5.2", provider="openai_compat")
    capabilities = resolve_model_inference_capabilities(info)

    sanitized = sanitize_inference_params_for_capabilities(
        {
            "system_prompt": "You are Mochi.",
            "temperature": 0.1,
            "max_tokens": 2048,
            "top_p": 0.5,
            "min_p": 0.1,
            "top_k": 20,
            "frequency_penalty": 0.2,
            "presence_penalty": 0.3,
            "repeat_penalty": 1.1,
            "reasoning_effort": "xhigh",
        },
        capabilities,
    )

    assert sanitized == {
        "system_prompt": "You are Mochi.",
        "reasoning_effort": "xhigh",
    }


def test_sanitize_inference_params_drops_unsupported_reasoning_value() -> None:
    info = _model_info(name="gpt-5", provider="openai_compat")
    capabilities = resolve_model_inference_capabilities(info)

    sanitized = sanitize_inference_params_for_capabilities(
        {
            "system_prompt": "You are Mochi.",
            "reasoning_effort": "xhigh",
        },
        capabilities,
    )

    assert sanitized == {"system_prompt": "You are Mochi."}
