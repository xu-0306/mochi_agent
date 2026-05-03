"""Bounded model management API routes."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, SecretStr

from mochi.api.server import _get_config, _get_or_create_engine, _maybe_await
from mochi.config.manager import save_config
from mochi.config.schema import MochiConfig

router = APIRouter(prefix="/v1")

ModelProvider = Literal["ollama", "openai_compat", "gemini", "anthropic"]

_REMOTE_PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "openai_compat": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-3-flash-preview",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-sonnet-4-6",
    },
}

_SUPPORTED_MODEL_SPEC_FORMATS: list[dict[str, str]] = [
    {
        "type": "ollama",
        "pattern": "ollama:<model>",
        "description": "Use an Ollama-served model by name.",
    },
    {
        "type": "gguf",
        "pattern": "/path/to/model.gguf",
        "description": "Use a local llama.cpp GGUF model file.",
    },
    {
        "type": "safetensors",
        "pattern": "/path/to/model_dir/",
        "description": "Use a local HuggingFace model directory.",
    },
    {
        "type": "openai_compat",
        "pattern": "https://host/v1",
        "description": "Use an OpenAI-compatible API base URL.",
    },
]


class ModelsResponse(BaseModel):
    """`GET /v1/models` response payload。"""

    type: str = "models_status"
    configured_model: str
    supported_model_spec_formats: list[dict[str, str]]
    active_model: dict[str, Any] | None = None
    configured_remote_provider: str | None = None


class SwitchModelRequest(BaseModel):
    """`POST /v1/models/switch` request payload。"""

    model: str = Field(min_length=1)


class SwitchModelResponse(BaseModel):
    """`POST /v1/models/switch` response payload。"""

    type: str = "model_switch"
    active_model: dict[str, Any]


class ConfigureModelRequest(BaseModel):
    """`POST /v1/models/configure` request payload。"""

    provider: ModelProvider
    model: str = Field(min_length=1)
    base_url: str | None = None
    api_key: str | None = None
    persist: bool = True


class ConfigureModelResponse(BaseModel):
    """`POST /v1/models/configure` response payload。"""

    type: str = "model_configure"
    provider: str
    active_model: dict[str, Any]
    api_key_configured: bool = False
    persisted: bool = False
    config_path: str | None = None


class OllamaModelsResponse(BaseModel):
    """`GET /v1/models/ollama` response payload。"""

    type: str = "ollama_models"
    base_url: str
    models: list[str]


@router.get("/models", response_model=ModelsResponse)
async def get_models(request: Request) -> ModelsResponse:
    """回傳 configured model、支援格式與目前活躍模型資訊。"""
    config = await _get_config(request.app)
    active_model = await _load_active_model_info(request)

    return ModelsResponse(
        configured_model=str(getattr(config, "model", "")),
        supported_model_spec_formats=list(_SUPPORTED_MODEL_SPEC_FORMATS),
        active_model=active_model,
        configured_remote_provider=getattr(config.openai_compat, "provider", None),
    )


@router.post("/models/switch", response_model=SwitchModelResponse)
async def switch_model(request: Request, payload: SwitchModelRequest) -> SwitchModelResponse:
    """切換活躍模型。"""
    engine = await _get_or_create_engine(request.app)
    model_info = await _maybe_await(engine.switch_model(payload.model))
    return SwitchModelResponse(active_model=_serialize_model_info(model_info))


@router.post("/models/configure", response_model=ConfigureModelResponse)
async def configure_model(
    request: Request,
    payload: ConfigureModelRequest,
) -> ConfigureModelResponse:
    """用 WebGUI 表單設定並切換 runtime 模型後端，不回傳 API key。"""
    engine = await _get_or_create_engine(request.app)
    config = await _get_config(request.app)
    if payload.provider == "ollama":
        normalized_model = payload.model.strip()
        normalized_base_url = (payload.base_url or config.ollama.base_url).strip().rstrip("/")
        switch_ollama = getattr(engine, "switch_ollama_backend", None)
        if callable(switch_ollama):
            model_info = await _maybe_await(
                switch_ollama(model=normalized_model, base_url=normalized_base_url)
            )
        else:
            model_info = await _maybe_await(engine.switch_model(f"ollama:{normalized_model}"))
        updated = config.model_copy(deep=True)
        updated.model = f"ollama:{normalized_model}"
        updated.ollama.base_url = normalized_base_url
        request.app.state.config = updated
        persisted_path = _persist_config_if_enabled(request, updated, payload.persist)
        return ConfigureModelResponse(
            provider=payload.provider,
            active_model=_serialize_model_info(model_info),
            api_key_configured=False,
            persisted=persisted_path is not None,
            config_path=str(persisted_path) if persisted_path is not None else None,
        )

    provider_defaults = _REMOTE_PROVIDER_DEFAULTS[payload.provider]
    normalized_base_url = (payload.base_url or provider_defaults["base_url"]).strip().rstrip("/")
    normalized_model = payload.model.strip() or provider_defaults["model"]
    same_saved_remote = (
        config.openai_compat.provider == payload.provider
        and config.openai_compat.base_url.rstrip("/") == normalized_base_url
    )
    existing_key = (
        config.openai_compat.api_key.get_secret_value()
        if same_saved_remote and config.openai_compat.api_key is not None
        else ""
    )
    effective_api_key = payload.api_key or existing_key
    switch_openai = getattr(engine, "switch_openai_compat_backend", None)
    if callable(switch_openai):
        model_info = await _maybe_await(
            switch_openai(
                base_url=normalized_base_url,
                model=normalized_model,
                api_key=effective_api_key,
                provider=payload.provider,
            )
        )
    else:
        model_info = await _maybe_await(engine.switch_model(normalized_base_url))

    updated = config.model_copy(deep=True)
    updated.model = normalized_base_url
    updated.openai_compat.base_url = normalized_base_url
    updated.openai_compat.model = normalized_model
    updated.openai_compat.provider = payload.provider
    if effective_api_key:
        updated.openai_compat.api_key = SecretStr(effective_api_key)
    request.app.state.config = updated
    persisted_path = _persist_config_if_enabled(request, updated, payload.persist)

    return ConfigureModelResponse(
        provider=payload.provider,
        active_model=_serialize_model_info(model_info),
        api_key_configured=bool(effective_api_key),
        persisted=persisted_path is not None,
        config_path=str(persisted_path) if persisted_path is not None else None,
    )


@router.get("/models/ollama", response_model=OllamaModelsResponse)
async def list_ollama_models(
    base_url: str = Query(default="http://localhost:11434", min_length=1),
) -> OllamaModelsResponse:
    """讀取 Ollama `/api/tags`，供 WebGUI 將模型欄位切換成下拉選單。"""
    normalized_base_url = base_url.strip().rstrip("/")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{normalized_base_url}/api/tags")
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Ollama API is not reachable at {normalized_base_url}: {exc}",
        ) from exc

    payload = response.json()
    raw_models = payload.get("models", [])
    models = [
        item.get("name", "")
        for item in raw_models
        if isinstance(item, dict) and isinstance(item.get("name"), str) and item.get("name")
    ]
    return OllamaModelsResponse(base_url=normalized_base_url, models=sorted(models))


async def _load_active_model_info(request: Request) -> dict[str, Any] | None:
    """盡量從 engine 取得活躍模型資訊；取不到則回傳 None。"""
    try:
        engine = await _get_or_create_engine(request.app)
    except Exception:
        return None

    get_model_info = getattr(engine, "get_model_info", None)
    if callable(get_model_info):
        try:
            info = await _maybe_await(get_model_info())
        except Exception:
            return None
        return _serialize_model_info(info)

    return None


def _serialize_model_info(info: Any) -> dict[str, Any]:
    """將 ModelInfo-like 物件轉成 JSON-safe dict。"""
    if is_dataclass(info):
        return jsonable_encoder(asdict(info))
    if hasattr(info, "model_dump"):
        return jsonable_encoder(info.model_dump())
    if isinstance(info, dict):
        return jsonable_encoder(info)
    return jsonable_encoder(
        {
            "name": getattr(info, "name", ""),
            "backend_type": getattr(info, "backend_type", ""),
            "context_length": getattr(info, "context_length", None),
            "supports_tool_calling": getattr(info, "supports_tool_calling", None),
            "metadata": getattr(info, "metadata", {}),
        }
    )


def _persist_config_if_enabled(
    request: Request,
    config: MochiConfig,
    persist: bool,
) -> Path | None:
    if not persist:
        return None
    config_path = getattr(request.app.state, "config_path", None)
    if config_path is None and getattr(request.app.state, "config_factory", None) is not None:
        return None
    return save_config(config, config_path)
