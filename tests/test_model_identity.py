from mochi.config.identity import model_target_alias, model_target_id
from mochi.config.schema import ConfiguredModelConfig


def _model(base_url: str) -> ConfiguredModelConfig:
    return ConfiguredModelConfig(
        id="gpt-5.4 (openai_compat)",
        provider="openai_compat",
        model="gpt-5.4",
        model_spec=base_url,
        base_url=base_url,
        label="gpt-5.4 (openai_compat)",
        backend_type="openai_compat",
    )


def test_target_id_is_not_a_display_label_and_distinguishes_endpoints() -> None:
    first = _model("https://one.example/v1")
    second = _model("https://two.example/v1")

    assert first.target_id is not None
    assert second.target_id is not None
    assert first.target_id != second.target_id
    assert first.target_id != first.label
    assert model_target_id(
        provider=first.provider,
        model=first.model,
        model_spec=first.model_spec,
        base_url=first.base_url,
        backend_type=first.backend_type,
    ) == first.target_id


def test_readable_target_alias_is_stable_for_legacy_clients() -> None:
    alias = model_target_alias(
        provider="openai_compat",
        model="gpt-5.4",
        model_spec="https://one.example/v1/",
        base_url="https://one.example/v1/",
    )
    assert alias.startswith("target:openai_compat:")
    assert "gpt-5.4 (openai_compat)" not in alias


def test_readable_target_alias_distinguishes_oauth_profiles() -> None:
    first = model_target_alias(
        provider="openai_codex",
        model="gpt-5.4",
        model_spec="https://chatgpt.com/backend-api",
        base_url="https://chatgpt.com/backend-api",
        auth_profile_id="profile-a",
    )
    second = model_target_alias(
        provider="openai_codex",
        model="gpt-5.4",
        model_spec="https://chatgpt.com/backend-api",
        base_url="https://chatgpt.com/backend-api",
        auth_profile_id="profile-b",
    )
    assert first != second


def test_endpoint_identity_drops_embedded_credentials() -> None:
    alias = model_target_alias(
        provider="openai_compat",
        model="gpt-5.4",
        model_spec="https://user:password@example.test/v1?api_key=query-secret&region=tw#fragment",
        base_url="https://user:password@example.test/v1?api_key=query-secret&region=tw#fragment",
    )
    assert "password" not in alias
    assert "query-secret" not in alias
    assert "fragment" not in alias
    assert "region%3Dtw" in alias
