"""Shared web-search provider metadata and normalization helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mochi.config.schema import ToolsConfig


@dataclass(frozen=True)
class WebSearchProviderSpec:
    """Canonical provider metadata used across config, API, and UI."""

    canonical_name: str
    ui_label: str
    aliases: tuple[str, ...] = ()
    no_key_supported: bool = False
    key_config_field: str | None = None
    configured_flag_field: str | None = None
    key_configured_flag_field: str | None = None
    base_url_config_field: str | None = None


WEB_SEARCH_PROVIDER_SPECS: tuple[WebSearchProviderSpec, ...] = (
    WebSearchProviderSpec(
        canonical_name="tavily",
        ui_label="Tavily",
        key_config_field="web_search_tavily_api_key",
        configured_flag_field="web_search_tavily_api_key_configured",
    ),
    WebSearchProviderSpec(
        canonical_name="serper",
        ui_label="Serper",
        key_config_field="web_search_serper_api_key",
        configured_flag_field="web_search_serper_api_key_configured",
    ),
    WebSearchProviderSpec(
        canonical_name="jina",
        ui_label="Jina",
        no_key_supported=True,
        key_config_field="web_search_jina_api_key",
        configured_flag_field="web_search_jina_configured",
        key_configured_flag_field="web_search_jina_api_key_configured",
    ),
    WebSearchProviderSpec(
        canonical_name="exa",
        ui_label="Exa",
        key_config_field="web_search_exa_api_key",
        configured_flag_field="web_search_exa_api_key_configured",
    ),
    WebSearchProviderSpec(
        canonical_name="brave",
        ui_label="Brave",
        key_config_field="web_search_brave_api_key",
        configured_flag_field="web_search_brave_api_key_configured",
    ),
    WebSearchProviderSpec(
        canonical_name="searxng",
        ui_label="SearXNG",
        base_url_config_field="web_search_searxng_base_url",
        configured_flag_field="web_search_searxng_configured",
    ),
    WebSearchProviderSpec(
        canonical_name="duckduckgo_html",
        ui_label="DuckDuckGo HTML",
        aliases=("duckduckgo", "ddg"),
        no_key_supported=True,
        configured_flag_field="web_search_duckduckgo_html_configured",
    ),
)

_SPECS_BY_NAME = {
    name: spec
    for spec in WEB_SEARCH_PROVIDER_SPECS
    for name in (spec.canonical_name, *spec.aliases)
}


def iter_web_search_provider_specs() -> tuple[WebSearchProviderSpec, ...]:
    return WEB_SEARCH_PROVIDER_SPECS


def normalize_web_search_provider(name: str) -> str:
    """Normalize aliases such as duckduckgo -> duckduckgo_html."""
    normalized = name.strip().lower().replace("-", "_")
    spec = _SPECS_BY_NAME.get(normalized)
    if spec is None:
        return normalized
    return spec.canonical_name


def get_web_search_provider_spec(name: str) -> WebSearchProviderSpec | None:
    """Return canonical provider metadata for a name or alias."""
    normalized = normalize_web_search_provider(name)
    return _SPECS_BY_NAME.get(normalized)


def supported_web_search_provider_names(*, include_aliases: bool = True) -> list[str]:
    """Return supported provider names for validation and UI."""
    names = [spec.canonical_name for spec in WEB_SEARCH_PROVIDER_SPECS]
    if include_aliases:
        for spec in WEB_SEARCH_PROVIDER_SPECS:
            names.extend(spec.aliases)
    return names


def provider_supports_key_management(name: str) -> bool:
    spec = get_web_search_provider_spec(name)
    return bool(spec and spec.key_config_field)


def provider_key_config_field(name: str) -> str | None:
    spec = get_web_search_provider_spec(name)
    return spec.key_config_field if spec else None


def provider_is_configured(*, name: str, tools: ToolsConfig) -> bool:
    """Whether the provider is usable under the current config."""
    spec = get_web_search_provider_spec(name)
    if spec is None:
        return False
    if spec.base_url_config_field:
        value = getattr(tools, spec.base_url_config_field, None)
        return isinstance(value, str) and bool(value.strip())
    if spec.key_config_field:
        value = getattr(tools, spec.key_config_field, None)
        if value is not None:
            return True
    return spec.no_key_supported


def provider_key_is_configured(*, name: str, tools: ToolsConfig) -> bool:
    """Whether a provider-specific secret is configured."""
    spec = get_web_search_provider_spec(name)
    if spec is None or not spec.key_config_field:
        return False
    return getattr(tools, spec.key_config_field, None) is not None


def build_web_search_provider_status_payload(tools: ToolsConfig) -> dict[str, bool]:
    """Build the safe settings payload describing provider availability."""
    payload: dict[str, bool] = {}
    for spec in WEB_SEARCH_PROVIDER_SPECS:
        if spec.configured_flag_field:
            payload[spec.configured_flag_field] = provider_is_configured(
                name=spec.canonical_name,
                tools=tools,
            )
        if spec.key_configured_flag_field:
            payload[spec.key_configured_flag_field] = provider_key_is_configured(
                name=spec.canonical_name,
                tools=tools,
            )
    return payload
