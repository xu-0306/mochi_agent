from mochi.backends.inference_capabilities import (
    InferenceCapabilities,
    parse_model_capability_metadata,
    resolve_model_inference_capabilities,
    select_lowest_reasoning_effort,
)
from mochi.backends.types import ModelInfo


def test_gpt_family_effort_registry() -> None:
    cases = {
        "gpt-5": ("minimal", "low", "medium", "high"),
        "gpt-5.1": ("none", "low", "medium", "high"),
        "gpt-5.2": ("none", "low", "medium", "high", "xhigh"),
        "gpt-5.4": ("none", "low", "medium", "high", "xhigh"),
        "gpt-5.6-luna": ("none", "low", "medium", "high", "xhigh", "max"),
    }
    for model, expected in cases.items():
        capabilities = resolve_model_inference_capabilities(
            ModelInfo(name=model, provider="openai_compat", backend_type="openai_compat")
        )
        assert capabilities.supported_reasoning_efforts == expected


def test_anthropic_style_metadata_is_authoritative() -> None:
    metadata = {
        "capabilities": {
            "effort": {
                "low": {"supported": True},
                "medium": {"supported": False},
                "high": True,
                "max": {"supported": True},
            }
        }
    }
    assert parse_model_capability_metadata(metadata) == ("low", "high", "max")
    capabilities = resolve_model_inference_capabilities(
        ModelInfo(
            name="claude-compatible",
            provider="anthropic",
            backend_type="openai_compat",
            metadata={"capability_metadata": metadata, "capability_source": "endpoint_metadata"},
        )
    )
    assert capabilities.supported_reasoning_efforts == ()
    assert capabilities.capability_source == "endpoint_metadata"


def test_supported_parameters_boolean_does_not_invent_effort_levels() -> None:
    metadata = {"supported_parameters": {"reasoning_effort": True}}
    assert parse_model_capability_metadata(metadata) == ()
    capabilities = resolve_model_inference_capabilities(
        ModelInfo(
            name="unlisted-model",
            provider="proxy",
            backend_type="openai_compat",
            metadata={"capability_metadata": metadata},
        )
    )
    assert capabilities.supported_reasoning_efforts == ()


def test_standard_openai_model_metadata_falls_back_to_gpt_registry() -> None:
    capabilities = resolve_model_inference_capabilities(
        ModelInfo(
            name="gpt-5.4",
            provider="openai_compat",
            backend_type="openai_compat",
            metadata={
                "capability_metadata": {
                    "id": "gpt-5.4",
                    "object": "model",
                    "created": 0,
                    "owned_by": "openai",
                },
                "capability_source": "standard_models_payload",
                "capability_status": "unavailable",
            },
        )
    )
    assert capabilities.supported_reasoning_efforts == (
        "none",
        "low",
        "medium",
        "high",
        "xhigh",
    )
    assert capabilities.capability_source == "registry"


def test_lowest_nonzero_effort_never_confuses_off_with_lowest_cost() -> None:
    capabilities = InferenceCapabilities(
        provider="test",
        supported_inference_parameters=("reasoning_effort",),
        supported_reasoning_efforts=("none", "low", "high"),
    )
    assert select_lowest_reasoning_effort(capabilities) == "low"
    assert select_lowest_reasoning_effort(capabilities, include_off=True) == "none"
