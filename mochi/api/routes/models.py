"""Bounded model management API routes."""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict, is_dataclass
import re
from pathlib import Path
from typing import Any, Literal

import httpx
from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.encoders import jsonable_encoder
try:
    from loguru import logger
except ModuleNotFoundError:  # pragma: no cover - fallback for minimal test envs
    import logging

    logger = logging.getLogger(__name__)
from pydantic import BaseModel, Field, SecretStr

from mochi.api.server import (
    _call_with_supported_kwargs,
    _get_config,
    _get_or_create_engine,
    _maybe_await,
)
from mochi.auth.openai_codex import (
    OPENAI_CODEX_DEFAULT_BASE_URL,
    OpenAICodexAuthService,
    normalize_openai_codex_base_url,
)
from mochi.backends.local_models import (
    BaseLocalModelConverter,
    LlamaCppLocalModelConverter,
    LocalModelConversionError,
    LocalModelConvertRequest as BackendLocalModelConvertRequest,
    ManagedLlamaCppInstallError,
    _detect_hardware_summary,
    discover_hf_quantization_capabilities,
    discover_local_models,
    get_managed_llama_cpp_runtime_status,
    install_managed_llama_cpp_runtime,
    prepare_managed_llama_cpp_install_plan,
)
from mochi.backends.vllm_runtime import ManagedVLLMRuntimeManager
from mochi.backends.vllm_utils import (
    configured_vllm_launch_mode as shared_configured_vllm_launch_mode,
    ensure_local_path_allowed,
    is_hf_safetensors_dir,
    is_http_endpoint as shared_is_http_endpoint,
    is_local_path_candidate as shared_is_local_path_candidate,
    is_managed_vllm_configured_model as shared_is_managed_vllm_configured_model,
    is_possible_managed_vllm_target as shared_is_possible_managed_vllm_target,
    managed_vllm_base_url as shared_managed_vllm_base_url,
    normalize_vllm_managed_model_spec as shared_normalize_vllm_managed_model_spec,
    resolve_vllm_managed_model_spec as shared_resolve_vllm_managed_model_spec,
)
from mochi.config.manager import save_config
from mochi.config.schema import ConfiguredModelConfig, MochiConfig
from mochi.diagnostics.fallbacks import append_fallback_diagnostic

router = APIRouter(prefix="/v1")

ModelProvider = Literal[
    "ollama",
    "openai_compat",
    "openai_codex",
    "gemini",
    "anthropic",
    "vllm",
    "sglang",
    "tensorrt_llm",
    "local",
]
_OPENAI_COMPAT_EXTERNAL_PROVIDERS = {
    "openai_compat",
    "gemini",
    "anthropic",
    "vllm",
    "sglang",
    "tensorrt_llm",
}

_REMOTE_PROVIDER_DEFAULTS: dict[str, dict[str, str]] = {
    "openai_compat": {
        "base_url": "https://api.openai.com/v1",
        "model": "gpt-4o-mini",
    },
    "openai_codex": {
        "base_url": OPENAI_CODEX_DEFAULT_BASE_URL,
        "model": "gpt-5.4",
    },
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-3-flash-preview",
    },
    "anthropic": {
        "base_url": "https://api.anthropic.com/v1",
        "model": "claude-sonnet-4-6",
    },
    "vllm": {
        "base_url": "http://localhost:8000/v1",
        "model": "model",
    },
    "sglang": {
        "base_url": "http://localhost:30000/v1",
        "model": "default",
    },
    "tensorrt_llm": {
        "base_url": "http://localhost:8000/v1",
        "model": "model",
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

_WSL_MOUNT_PATH_RE = re.compile(r"^/mnt/([A-Za-z])(?:/(.*))?$")
_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^([A-Za-z]):[\\/]*(.*)$")


class ModelsResponse(BaseModel):
    """`GET /v1/models` response payload??"""

    type: str = "models_status"
    configured_model: str
    supported_model_spec_formats: list[dict[str, str]]
    active_model: dict[str, Any] | None = None
    available_models: list[dict[str, Any]] = Field(default_factory=list)
    configured_remote_provider: str | None = None


class SwitchModelRequest(BaseModel):
    """`POST /v1/models/switch` request payload??"""

    model: str = Field(min_length=1)


class SwitchModelResponse(BaseModel):
    """`POST /v1/models/switch` response payload??"""

    type: str = "model_switch"
    active_model: dict[str, Any]


class ConfigureModelRequest(BaseModel):
    """`POST /v1/models/configure` request payload??"""

    provider: ModelProvider
    model: str = Field(min_length=1)
    base_url: str | None = None
    api_key: str | None = None
    auth_profile_id: str | None = None
    persist: bool = True


class ConfigureModelResponse(BaseModel):
    """`POST /v1/models/configure` response payload??"""

    type: str = "model_configure"
    provider: str
    active_model: dict[str, Any]
    available_models: list[dict[str, Any]] = Field(default_factory=list)
    api_key_configured: bool = False
    persisted: bool = False
    config_path: str | None = None


class UpdateConfiguredModelPayload(BaseModel):
    """`PATCH /v1/models/configured/{model_id}` request payload??"""

    provider: ModelProvider | None = None
    model: str | None = None
    model_spec: str | None = None
    base_url: str | None = None
    api_key: str | None = None
    auth_profile_id: str | None = None
    persist: bool = True


class UpdateConfiguredModelResponse(BaseModel):
    """`PATCH /v1/models/configured/{model_id}` response payload??"""

    type: str = "model_entry_update"
    updated_model: dict[str, Any]
    available_models: list[dict[str, Any]] = Field(default_factory=list)
    configured_model: str
    api_key_configured: bool
    persisted: bool = False
    config_path: str | None = None


class DeleteConfiguredModelPayload(BaseModel):
    """`DELETE /v1/models/configured/{model_id}` request payload??"""

    persist: bool = True


class DeleteConfiguredModelResponse(BaseModel):
    """`DELETE /v1/models/configured/{model_id}` response payload??"""

    type: str = "model_entry_delete"
    deleted_model_id: str
    available_models: list[dict[str, Any]] = Field(default_factory=list)
    configured_model: str
    diagnostics: list[dict[str, Any]] = Field(default_factory=list)
    persisted: bool = False
    config_path: str | None = None


class OllamaModelsResponse(BaseModel):
    """`GET /v1/models/ollama` response payload??"""

    type: str = "ollama_models"
    base_url: str
    models: list[str]


class LocalModelsResponse(BaseModel):
    """`GET /v1/models/local` response payload??"""

    type: str = "local_models"
    root: str
    models: list[dict[str, Any]]
    warnings: list[str] = Field(default_factory=list)


class LocalModelQuantizationCapabilitiesResponse(BaseModel):
    """`GET /v1/models/local/capabilities` response payload??"""

    type: str = "local_model_quantization_capabilities"
    model_spec: str
    model_dir: str
    model_family: str | None = None
    formats: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    hardware: dict[str, Any] | None = None


class LocalModelConvertPayload(BaseModel):
    """`POST /v1/models/local/convert` request payload??"""

    source_model_dir: str = Field(min_length=1)
    target_format: str = Field(min_length=1)
    quantization: str = Field(min_length=1)
    persist: bool = True


class LocalModelConvertResponse(BaseModel):
    """`POST /v1/models/local/convert` response payload??"""

    type: str = "local_model_convert"
    provider: str = "local"
    target_format: str
    quantization: str
    source_model_dir: str
    output_model_path: str
    saved_as_model: dict[str, Any] | None = None
    available_models: list[dict[str, Any]] | None = None
    active_model: dict[str, Any] | None = None
    converted: bool
    persisted: bool
    config_path: str | None = None
    warnings: list[str] = Field(default_factory=list)
    message: str


class LocalModelRuntimeStatusResponse(BaseModel):
    """`GET /v1/models/local/runtime` response payload??"""

    type: str = "local_model_runtime_status"
    runtime: str = "llama.cpp"
    readiness: str
    installed: bool
    source: str
    root_dir: str | None = None
    install_dir: str | None = None
    python_executable: str
    version: str | None = None
    platform: str | None = None
    binary_asset: str | None = None
    convert_script: str | None = None
    quantize_binary: str | None = None
    missing_components: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    hardware: dict[str, Any] | None = None


class LocalModelRuntimeInstallPayload(BaseModel):
    """`POST /v1/models/local/runtime/install` request payload??"""

    action: Literal["prepare_managed", "register_existing_path"] = "prepare_managed"
    existing_path: str | None = None
    persist: bool = True


class LocalModelRuntimeInstallResponse(BaseModel):
    """`POST /v1/models/local/runtime/install` response payload??"""

    type: str = "local_model_runtime_install"
    runtime: str = "llama.cpp"
    action: str
    state: str
    source: str
    install_dir: str | None = None
    root_dir: str | None = None
    version: str | None = None
    platform: str | None = None
    binary_asset: str | None = None
    persisted: bool
    config_path: str | None = None
    runtime_status: LocalModelRuntimeStatusResponse
    warnings: list[str] = Field(default_factory=list)
    message: str


class LocalActiveModelRuntimeStatusResponse(BaseModel):
    """`GET /v1/models/local/active-runtime` response payload??"""

    type: str = "local_active_model_runtime_status"
    has_active_local_model: bool
    model_spec: str | None = None
    backend_type: str | None = None
    loaded: bool = False
    idle_unloaded: bool = False
    can_unload: bool = False


class LocalActiveModelRuntimeUnloadResponse(BaseModel):
    """`POST /v1/models/local/active-runtime/unload` response payload??"""

    type: str = "local_active_model_runtime_unload"
    unloaded: bool
    active_runtime: LocalActiveModelRuntimeStatusResponse


class VLLMRuntimeStatusResponse(BaseModel):
    """`GET /v1/models/vllm/runtime` response payload??"""

    type: str = "vllm_runtime_status"
    runtime: str = "vllm"
    state: str
    running: bool
    launch_mode: str | None = None
    active_model_id: str | None = None
    active_model_spec: str | None = None
    base_url: str | None = None
    backend_type: str = "openai_compat"
    message: str | None = None


class VLLMRuntimeStartPayload(BaseModel):
    """`POST /v1/models/vllm/runtime/start` request payload??"""

    model_id: str = Field(min_length=1)


class VLLMRuntimeControlResponse(BaseModel):
    """`POST /v1/models/vllm/runtime/{start|stop}` response payload??"""

    type: str = "vllm_runtime_control"
    action: Literal["start", "stop"]
    runtime_status: VLLMRuntimeStatusResponse


class ToolCallingProbeResponse(BaseModel):
    """`POST /v1/models/probe-tool-calling` response payload."""

    type: str = "tool_calling_probe"
    active_model: dict[str, Any] | None = None
    probe: dict[str, Any] | None = None


@router.get("/models", response_model=ModelsResponse)
async def get_models(request: Request) -> ModelsResponse:
    """?豯止齒 configured model?蹓澗??皝僱???獢???????????"""
    config = await _get_config(request.app)
    active_model = await _load_active_model_info(
        request,
        configured_model=_find_active_configured_model(config),
    )

    return ModelsResponse(
        configured_model=str(getattr(config, "model", "")),
        supported_model_spec_formats=list(_SUPPORTED_MODEL_SPEC_FORMATS),
        active_model=active_model,
        available_models=_serialize_configured_models(config),
        configured_remote_provider=_active_remote_provider_from_config(config),
    )


@router.post("/models/probe-tool-calling", response_model=ToolCallingProbeResponse)
async def probe_tool_calling(request: Request) -> ToolCallingProbeResponse:
    """Run an on-demand native tool-calling probe for the active backend when supported."""
    config = await _get_config(request.app)
    engine = await _get_or_create_engine(request.app)
    probe_method = getattr(engine, "probe_active_tool_calling", None)
    probe_result = await _maybe_await(probe_method()) if callable(probe_method) else None
    active_model = await _load_active_model_info(
        request,
        configured_model=_find_active_configured_model(config),
    )
    return ToolCallingProbeResponse(
        active_model=active_model,
        probe=probe_result,
    )


@router.get("/models/local", response_model=LocalModelsResponse)
async def list_local_models(
    request: Request,
    root: str = Query(min_length=1),
    max_depth: int | None = Query(default=None, ge=0, le=32),
) -> LocalModelsResponse:
    """?????秘???謕???GUF ??HuggingFace safetensors ?獢????"""
    config = await _get_config(request.app)
    local_cfg = config.local_models
    depth = max_depth if max_depth is not None else local_cfg.scan_max_depth
    max_entries = local_cfg.scan_max_entries

    try:
        normalized_root = _normalize_local_root(root, config)
        result = discover_local_models(
            normalized_root,
            max_depth=depth,
            max_entries=max_entries,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (NotADirectoryError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    models = [
        {
            "id": candidate.model_spec,
            "provider": "local",
            "model": candidate.model,
            "model_spec": candidate.model_spec,
            "label": candidate.model,
            "backend_type": candidate.backend_type,
            "metadata": candidate.metadata,
        }
        for candidate in result.models
    ]

    return LocalModelsResponse(
        root=result.root,
        models=models,
        warnings=result.warnings,
    )


@router.get(
    "/models/local/capabilities",
    response_model=LocalModelQuantizationCapabilitiesResponse,
)
async def get_local_model_quantization_capabilities(
    request: Request,
    model_spec: str = Query(min_length=1),
    include_hardware: bool = Query(default=True),
) -> LocalModelQuantizationCapabilitiesResponse:
    """?嚗貉??秘 HF ??????鞈???瘣菟??????capability??????改????"""
    config = await _get_config(request.app)
    normalized_model_spec = _normalize_local_capability_target(model_spec, config)
    model_path = Path(normalized_model_spec)

    if not model_path.is_dir():
        raise HTTPException(
            status_code=400,
            detail="Quantization capability probe currently supports local HuggingFace model directories only.",
        )

    try:
        result = discover_hf_quantization_capabilities(
            normalized_model_spec,
            include_hardware=include_hardware,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc
    except (NotADirectoryError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    return LocalModelQuantizationCapabilitiesResponse(
        model_spec=normalized_model_spec,
        model_dir=result.model_dir,
        model_family=result.model_family,
        formats=[_serialize_capability_format(item) for item in result.formats],
        warnings=list(result.warnings),
        hardware=_serialize_hardware_summary(result.hardware),
    )


@router.get("/models/local/runtime", response_model=LocalModelRuntimeStatusResponse)
async def get_local_model_runtime_status(request: Request) -> LocalModelRuntimeStatusResponse:
    """?豯止齒 llama.cpp ?改? runtime readiness??"""
    config = await _get_config(request.app)
    status = _discover_local_runtime_status(request, config)
    return _serialize_local_runtime_status(status)


@router.get("/models/local/active-runtime", response_model=LocalActiveModelRuntimeStatusResponse)
async def get_active_local_model_runtime_status(
    request: Request,
) -> LocalActiveModelRuntimeStatusResponse:
    """?豯止齒?獢? active ??秘?????????鈭????"""
    engine = await _get_or_create_engine(request.app)
    active_model = await _active_model_info(engine)
    return _serialize_active_local_model_runtime(active_model)


@router.post(
    "/models/local/active-runtime/unload",
    response_model=LocalActiveModelRuntimeUnloadResponse,
)
async def unload_active_local_model_runtime(
    request: Request,
) -> LocalActiveModelRuntimeUnloadResponse:
    """????鞎??獢? active ??秧??啾???????"""
    engine = await _get_or_create_engine(request.app)
    unload = getattr(engine, "unload_active_local_model", None)
    if not callable(unload):
        current = await _active_model_info(engine)
        return LocalActiveModelRuntimeUnloadResponse(
            unloaded=False,
            active_runtime=_serialize_active_local_model_runtime(current),
        )

    result = await _maybe_await(unload())
    active_runtime = _serialize_active_local_model_runtime(
        result if result is not None else await _active_model_info(engine)
    )
    return LocalActiveModelRuntimeUnloadResponse(
        unloaded=result is not None,
        active_runtime=active_runtime,
    )


@router.get("/models/vllm/runtime", response_model=VLLMRuntimeStatusResponse)
async def get_vllm_runtime_status(request: Request) -> VLLMRuntimeStatusResponse:
    """?豯止齒 managed vLLM runtime ?????"""
    config = await _get_config(request.app)
    manager = _get_or_create_vllm_runtime_manager(request)
    status = await _query_vllm_runtime_status(manager=manager, config=config)
    return _serialize_vllm_runtime_status(status)


@router.post("/models/vllm/runtime/start", response_model=VLLMRuntimeControlResponse)
async def start_vllm_runtime(
    request: Request,
    payload: VLLMRuntimeStartPayload,
) -> VLLMRuntimeControlResponse:
    """?賹? managed vLLM runtime??謘?? active instance???"""
    config = await _get_config(request.app)
    model_entry = _find_configured_model(config, payload.model_id.strip())
    if model_entry is None:
        raise HTTPException(status_code=404, detail=f"Configured model was not found: {payload.model_id}")

    managed_model_spec = _resolve_vllm_managed_model_spec(model_entry, config)
    manager = _get_or_create_vllm_runtime_manager(request)
    status = await _start_managed_vllm_runtime(
        manager=manager,
        model_id=model_entry.id,
        model_spec=managed_model_spec,
        base_url=_managed_vllm_base_url(model_entry.base_url),
        config=config,
    )
    return VLLMRuntimeControlResponse(
        action="start",
        runtime_status=_serialize_vllm_runtime_status(status),
    )


@router.post("/models/vllm/runtime/stop", response_model=VLLMRuntimeControlResponse)
async def stop_vllm_runtime(request: Request) -> VLLMRuntimeControlResponse:
    """?謚怨翰 managed vLLM runtime??謘?? active instance???"""
    config = await _get_config(request.app)
    manager = _get_or_create_vllm_runtime_manager(request)
    await _stop_managed_vllm_runtime(manager=manager)
    status = await _query_vllm_runtime_status(manager=manager, config=config)
    return VLLMRuntimeControlResponse(
        action="stop",
        runtime_status=_serialize_vllm_runtime_status(status),
    )


@router.post("/models/local/runtime/install", response_model=LocalModelRuntimeInstallResponse)
async def install_local_model_runtime(
    request: Request,
    payload: LocalModelRuntimeInstallPayload,
) -> LocalModelRuntimeInstallResponse:
    """managed llama.cpp installer/register surface??"""
    config = await _get_config(request.app)
    updated = config.model_copy(deep=True)
    warnings: list[str] = []
    message = ""
    action = payload.action

    if payload.action == "register_existing_path":
        if not payload.existing_path or not payload.existing_path.strip():
            raise HTTPException(
                status_code=400,
                detail="existing_path is required for action=register_existing_path.",
            )
        root_dir = Path(payload.existing_path.strip()).expanduser().resolve(strict=False)
        _ensure_local_path_allowed(root_dir, config)
        if not root_dir.exists():
            raise HTTPException(status_code=404, detail=f"Local runtime path does not exist: {root_dir}")
        if not root_dir.is_dir():
            raise HTTPException(status_code=400, detail="Local runtime path must be a directory.")
        if root_dir.is_symlink():
            raise HTTPException(status_code=400, detail="Symlink runtime paths are not supported.")

        updated.local_models.llama_cpp.source = "existing_path"
        updated.local_models.llama_cpp.root_dir = root_dir
        message = "Registered existing llama.cpp runtime path."
    else:
        effective_version = updated.local_models.llama_cpp.version
        plan = prepare_managed_llama_cpp_install_plan(
            managed_root=_managed_runtime_base_dir(updated),
            version=effective_version,
            network_available=True,
        )
        try:
            install_result = await install_managed_llama_cpp_runtime(
                managed_root=_managed_runtime_base_dir(updated),
                version=plan.version,
                python_executable=updated.local_models.llama_cpp.python_executable,
            )
        except ManagedLlamaCppInstallError as exc:
            raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc
        updated.local_models.llama_cpp.source = "managed"
        updated.local_models.llama_cpp.root_dir = Path(install_result.root_dir)
        updated.local_models.llama_cpp.version = install_result.version
        updated.local_models.llama_cpp.python_executable = install_result.python_executable
        warnings.extend(plan.warnings)
        warnings.extend(install_result.warnings)
        message = install_result.message

    request.app.state.config = updated
    setattr(request.app.state, "local_model_converter", None)
    engine = await _get_or_create_engine(request.app)
    apply_config = getattr(engine, "apply_config", None)
    if callable(apply_config):
        await _maybe_await(apply_config(updated, reload_voice=False))
    persisted_path = _persist_config_if_enabled(request, updated, payload.persist)
    runtime_status = _serialize_local_runtime_status(_discover_local_runtime_status(request, updated))

    return LocalModelRuntimeInstallResponse(
        action=action,
        state=runtime_status.readiness,
        source=runtime_status.source,
        install_dir=runtime_status.install_dir,
        root_dir=runtime_status.root_dir,
        version=runtime_status.version,
        platform=runtime_status.platform,
        binary_asset=runtime_status.binary_asset,
        persisted=persisted_path is not None,
        config_path=str(persisted_path) if persisted_path is not None else None,
        runtime_status=runtime_status,
        warnings=warnings,
        message=message,
    )


@router.post("/models/local/convert", response_model=LocalModelConvertResponse)
async def convert_local_model(
    request: Request,
    payload: LocalModelConvertPayload,
) -> LocalModelConvertResponse:
    """?????秘???改??hase 1?城? GGUF??ounded placeholder runtime???"""
    config = await _get_config(request.app)
    source_path = Path(payload.source_model_dir.strip()).expanduser().resolve(strict=False)
    if not source_path.is_absolute():
        source_path = source_path.resolve(strict=False)
    _ensure_local_path_allowed(source_path, config)

    converter = await _get_local_model_converter(request)
    try:
        async with _local_model_conversion_guard(
            request,
            source_model_dir=source_path,
            target_format=payload.target_format,
            quantization=payload.quantization,
        ):
            converted = await _maybe_await(
                converter.convert(
                    BackendLocalModelConvertRequest(
                        source_model_dir=str(source_path),
                        target_format=payload.target_format,
                        quantization=payload.quantization,
                        persist=payload.persist,
                    )
                )
            )
    except LocalModelConversionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=str(exc)) from exc

    saved_as_model: dict[str, Any] | None = None
    available_models: list[dict[str, Any]] | None = None
    active_model: dict[str, Any] | None = None
    persisted = False
    persisted_path: Path | None = None

    if converted.converted and payload.persist:
        output_path = str(Path(converted.output_model_path).expanduser().resolve(strict=False))
        normalized_target_format = converted.target_format.strip().lower()
        if normalized_target_format == "gguf":
            backend_type = "gguf"
        else:
            backend_type = "safetensors"
        updated = config.model_copy(deep=True)
        updated.model = output_path
        saved_entry = ConfiguredModelConfig(
            id=output_path,
            provider="local",
            model=Path(output_path).name,
            model_spec=output_path,
            label=Path(output_path).name,
            backend_type=backend_type,
        )
        updated.model_setup.configured_models = _upsert_configured_model(
            updated.model_setup.configured_models,
            saved_entry,
        )
        request.app.state.config = updated
        persisted_path = _persist_config_if_enabled(request, updated, True)
        saved_as_model = saved_entry.model_dump(exclude_none=True)
        available_models = _serialize_configured_models(updated)
        active_model = saved_as_model
        persisted = True

    return LocalModelConvertResponse(
        provider="local",
        target_format=converted.target_format,
        quantization=converted.quantization,
        source_model_dir=converted.source_model_dir,
        output_model_path=str(Path(converted.output_model_path).expanduser().resolve(strict=False)),
        saved_as_model=saved_as_model,
        available_models=available_models,
        active_model=active_model,
        converted=converted.converted,
        persisted=persisted,
        config_path=str(persisted_path) if persisted_path is not None else None,
        warnings=[],
        message=converted.message,
    )


def _get_local_model_conversion_registry(request: Request) -> tuple[asyncio.Lock, set[str]]:
    """?謘? app scoped ??秧??唾??????registry??"""
    registry_lock = getattr(request.app.state, "local_model_conversion_registry_lock", None)
    if not isinstance(registry_lock, asyncio.Lock):
        registry_lock = asyncio.Lock()
        setattr(request.app.state, "local_model_conversion_registry_lock", registry_lock)

    in_progress = getattr(request.app.state, "local_model_conversion_in_progress", None)
    if not isinstance(in_progress, set):
        in_progress = set()
        setattr(request.app.state, "local_model_conversion_in_progress", in_progress)

    return registry_lock, in_progress


@asynccontextmanager
async def _local_model_conversion_guard(
    request: Request,
    *,
    source_model_dir: Path,
    target_format: str,
    quantization: str,
):
    """???????????????湛??謜??頦???寞?????澗???鞎???"""
    lock_key = str(source_model_dir.expanduser().resolve(strict=False))
    registry_lock, in_progress = _get_local_model_conversion_registry(request)

    async with registry_lock:
        if lock_key in in_progress:
            raise HTTPException(
                status_code=409,
                detail=(
                    "A local model conversion is already in progress for this source model. "
                    "Wait for the existing conversion to finish before starting another one."
                ),
            )
        in_progress.add(lock_key)
        logger.info(
            "Accepted local model conversion request: source={} target={} quantization={}",
            lock_key,
            target_format,
            quantization,
        )
    try:
        yield
    finally:
        async with registry_lock:
            in_progress.discard(lock_key)
            logger.info(
                "Released local model conversion slot: source={} target={} quantization={}",
                lock_key,
                target_format,
                quantization,
            )


@router.post("/models/switch", response_model=SwitchModelResponse)
async def switch_model(request: Request, payload: SwitchModelRequest) -> SwitchModelResponse:
    """???????????"""
    model_info = await switch_model_runtime(request, payload.model)
    config = await _get_config(request.app)
    return SwitchModelResponse(
        active_model=_serialize_active_model_info(
            model_info,
            configured_model=_find_active_configured_model(config),
        )
    )


@router.patch("/models/configured/{model_id:path}", response_model=UpdateConfiguredModelResponse)
async def update_configured_model(
    request: Request,
    model_id: str,
    payload: UpdateConfiguredModelPayload,
) -> UpdateConfiguredModelResponse:
    """?箏?拇????殉朵?????畾??????API key ??????"""
    config = await _get_config(request.app)
    target_id = model_id.strip()
    if not target_id:
        raise HTTPException(status_code=400, detail="model_id is required.")

    existing = _find_configured_model(config, target_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Configured model was not found: {target_id}")

    updated = config.model_copy(deep=True)
    models = list(updated.model_setup.configured_models)
    target_index = next(
        (index for index, item in enumerate(models) if item.id == existing.id),
        None,
    )
    if target_index is None:
        raise HTTPException(status_code=404, detail=f"Configured model was not found: {target_id}")

    next_provider = payload.provider or existing.provider
    if next_provider not in {"ollama", "openai_codex", "local", *_OPENAI_COMPAT_EXTERNAL_PROVIDERS}:
        raise HTTPException(status_code=400, detail=f"Unsupported provider '{next_provider}'.")

    next_model = (payload.model if payload.model is not None else existing.model).strip()
    if not next_model:
        raise HTTPException(status_code=400, detail="model is required.")

    raw_model_spec = payload.model_spec if payload.model_spec is not None else existing.model_spec
    next_launch_mode: Literal["external", "managed"] | None = None
    next_base_url: str | None
    next_model_spec: str

    if next_provider == "local":
        if not raw_model_spec or not raw_model_spec.strip():
            raise HTTPException(status_code=400, detail="model_spec is required for provider=local.")
        next_model_spec = _normalize_local_model_spec(raw_model_spec, updated)
        next_model = Path(next_model_spec).name
        next_base_url = None
    elif next_provider == "ollama":
        provided_base_url = payload.base_url if payload.base_url is not None else existing.base_url
        normalized_base_url = (provided_base_url or updated.ollama.base_url).strip().rstrip("/")
        if not normalized_base_url:
            normalized_base_url = updated.ollama.base_url.rstrip("/")
        next_base_url = normalized_base_url
        next_model_spec = f"ollama:{next_model}"
    elif next_provider == "vllm":
        existing_mode = _configured_vllm_launch_mode(existing) if existing.provider == "vllm" else None
        use_managed_mode = False
        if payload.model_spec is not None and payload.model_spec.strip():
            use_managed_mode = not _is_http_endpoint(payload.model_spec)
        elif existing_mode is not None:
            use_managed_mode = existing_mode == "managed"
        elif raw_model_spec and raw_model_spec.strip():
            use_managed_mode = not _is_http_endpoint(raw_model_spec)

        if use_managed_mode:
            managed_model_source = (
                payload.model_spec
                if payload.model_spec is not None and payload.model_spec.strip()
                else payload.model
                if payload.model is not None and payload.model.strip()
                else raw_model_spec
            )
            if not managed_model_source or not managed_model_source.strip():
                raise HTTPException(status_code=400, detail="model_spec is required for managed provider=vllm.")
            next_model_spec = _normalize_vllm_managed_model_spec(managed_model_source, updated)
            next_model = next_model_spec
            provided_base_url = payload.base_url if payload.base_url is not None else existing.base_url
            normalized_base_url = _managed_vllm_base_url(provided_base_url)
            if not normalized_base_url.startswith(("http://", "https://")):
                raise HTTPException(
                    status_code=400,
                    detail="Remote provider base_url must start with http:// or https://.",
                )
            next_base_url = normalized_base_url
            next_launch_mode = "managed"
        else:
            provider_defaults = _REMOTE_PROVIDER_DEFAULTS[next_provider]
            provided_base_url = payload.base_url if payload.base_url is not None else (
                existing.base_url or existing.model_spec
            )
            normalized_base_url = (provided_base_url or provider_defaults["base_url"]).strip().rstrip("/")
            if not normalized_base_url.startswith(("http://", "https://")):
                raise HTTPException(
                    status_code=400,
                    detail="Remote provider base_url must start with http:// or https://.",
                )
            next_base_url = normalized_base_url
            next_model_spec = normalized_base_url
            next_launch_mode = "external"
    else:
        provider_defaults = _REMOTE_PROVIDER_DEFAULTS[next_provider]
        provided_base_url = payload.base_url if payload.base_url is not None else (
            existing.base_url or existing.model_spec
        )
        try:
            normalized_base_url = (
                normalize_openai_codex_base_url(provided_base_url)
                if next_provider == "openai_codex"
                else (provided_base_url or provider_defaults["base_url"]).strip().rstrip("/")
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        if not normalized_base_url.startswith(("http://", "https://")):
            raise HTTPException(
                status_code=400,
                detail="Remote provider base_url must start with http:// or https://.",
            )
        next_base_url = normalized_base_url
        next_model_spec = normalized_base_url

    backend_type = (
        "ollama"
        if next_provider == "ollama"
        else "openai_codex"
        if next_provider == "openai_codex"
        else "openai_compat"
        if next_provider in _OPENAI_COMPAT_EXTERNAL_PROVIDERS
        else "gguf"
        if next_model_spec.lower().endswith(".gguf")
        else "safetensors"
    )
    same_target_as_existing = (
        next_provider == existing.provider
        and next_model == existing.model
        and next_model_spec == existing.model_spec
        and (next_base_url or "") == (existing.base_url or "")
    )
    next_id = existing.id if same_target_as_existing else (
        next_model_spec
        if next_provider in {"ollama", "local"}
        else f"{next_provider}:{next_base_url}:{next_model}"
    )
    next_label = next_model if next_provider in {"ollama", "local"} else f"{next_model} ({next_provider})"
    next_auth_profile_id = (
        payload.auth_profile_id
        if "auth_profile_id" in payload.model_fields_set
        else existing.auth_profile_id
    )
    if next_provider == "openai_codex":
        auth_service = _openai_codex_auth_service(updated)
        next_auth_profile_id = auth_service.resolve_profile_id(next_auth_profile_id)
        if next_auth_profile_id is None:
            raise HTTPException(
                status_code=400,
                detail="No OpenAI Codex auth profile is available. Import Codex CLI login first.",
            )
    next_auth_mode = (
        "oauth"
        if next_provider == "openai_codex"
        else "api_key"
        if next_provider in _OPENAI_COMPAT_EXTERNAL_PROVIDERS
        else None
    )

    replacement = ConfiguredModelConfig(
        id=next_id,
        provider=next_provider,
        model=next_model,
        model_spec=next_model_spec,
        base_url=next_base_url,
        label=next_label,
        backend_type=backend_type,
        launch_mode=next_launch_mode,
        auth_profile_id=next_auth_profile_id,
        auth_mode=next_auth_mode,
    )

    deduped_models = [
        item
        for idx, item in enumerate(models)
        if idx != target_index and not _configured_models_equivalent(item, replacement)
    ]
    deduped_models.insert(target_index, replacement)
    updated.model_setup.configured_models = deduped_models

    is_active_entry = _is_configured_model_active(config, existing)
    if is_active_entry:
        updated = _apply_configured_model_to_config(updated, replacement)

    api_key_changed = "api_key" in payload.model_fields_set
    api_key_configured = False
    if next_provider == "openai_codex":
        updated.openai_codex.base_url = (next_base_url or next_model_spec).rstrip("/")
        updated.openai_codex.model = next_model
        updated.openai_codex.auth_profile_id = next_auth_profile_id
        api_key_configured = False
    elif next_provider in _OPENAI_COMPAT_EXTERNAL_PROVIDERS:
        updated.openai_compat.provider = next_provider
        updated.openai_compat.base_url = (next_base_url or next_model_spec).rstrip("/")
        updated.openai_compat.model = next_model

        incoming_api_key = (payload.api_key or "").strip() if api_key_changed else None
        if incoming_api_key is not None:
            if incoming_api_key:
                updated.openai_compat.api_key = SecretStr(incoming_api_key)
                api_key_configured = True
            else:
                updated.openai_compat.api_key = None
                api_key_configured = False
        else:
            api_key_configured = (
                updated.openai_compat.api_key is not None
                and updated.openai_compat.api_key.get_secret_value().strip() != ""
            )
    else:
        api_key_configured = False

    request.app.state.config = updated
    persisted_path = _persist_config_if_enabled(request, updated, payload.persist)
    return UpdateConfiguredModelResponse(
        updated_model=replacement.model_dump(exclude_none=True),
        available_models=_dump_saved_configured_models(updated),
        configured_model=str(updated.model),
        api_key_configured=api_key_configured,
        persisted=persisted_path is not None,
        config_path=str(persisted_path) if persisted_path is not None else None,
    )


@router.delete("/models/configured/{model_id:path}", response_model=DeleteConfiguredModelResponse)
async def delete_configured_model(
    request: Request,
    model_id: str,
    payload: DeleteConfiguredModelPayload | None = None,
) -> DeleteConfiguredModelResponse:
    """??畸?????殉朵??????朝?"""
    config = await _get_config(request.app)
    diagnostics: list[dict[str, Any]] = []
    target_id = model_id.strip()
    if not target_id:
        raise HTTPException(status_code=400, detail="model_id is required.")

    existing = _find_configured_model(config, target_id)
    if existing is None:
        raise HTTPException(status_code=404, detail=f"Configured model was not found: {target_id}")

    updated = config.model_copy(deep=True)
    remaining = [
        item
        for item in updated.model_setup.configured_models
        if not _configured_models_equivalent(item, existing)
    ]
    updated.model_setup.configured_models = remaining

    if _is_configured_model_active(config, existing):
        if remaining:
            fallback = remaining[0]
            updated = _apply_configured_model_to_config(updated, fallback)
            append_fallback_diagnostic(
                diagnostics,
                category="model_selection",
                name="active_configured_model_deleted",
                reason="deleted_active_model_switched_to_saved_model",
                kind="fallback",
                severity="warning",
                from_state=existing.id,
                to_state=fallback.id,
                metadata={
                    "from_model_spec": existing.model_spec,
                    "to_model_spec": fallback.model_spec,
                },
            )
        else:
            updated.model = updated.model_setup.default_model_spec
            append_fallback_diagnostic(
                diagnostics,
                category="model_selection",
                name="active_configured_model_deleted",
                reason="deleted_active_model_switched_to_default_model",
                kind="fallback",
                severity="warning",
                from_state=existing.id,
                to_state=updated.model_setup.default_model_spec,
                metadata={"from_model_spec": existing.model_spec},
            )

    request.app.state.config = updated
    engine = await _get_or_create_engine(request.app)
    apply_config = getattr(engine, "apply_config", None)
    if callable(apply_config):
        await _maybe_await(apply_config(updated, reload_voice=False))

    should_persist = True if payload is None else payload.persist
    persisted_path = _persist_config_if_enabled(request, updated, should_persist)
    return DeleteConfiguredModelResponse(
        deleted_model_id=existing.id,
        available_models=_dump_saved_configured_models(updated),
        configured_model=str(updated.model),
        diagnostics=diagnostics,
        persisted=persisted_path is not None,
        config_path=str(persisted_path) if persisted_path is not None else None,
    )


async def switch_model_runtime(request: Request, model_id: str) -> Any:
    """?????id/spec ??? runtime?? `/v1/models/switch` ??chat route ?璇???"""
    engine = await _get_or_create_engine(request.app)
    config = await _get_config(request.app)
    model_entry = _find_configured_model(config, model_id)
    if model_entry is not None:
        effective_entry = model_entry

        if _is_configured_model_active(config, model_entry):
            current = await _active_model_info(engine)
            if _is_managed_vllm_configured_model(effective_entry):
                model_info = await _switch_configured_model(request, engine, config, effective_entry)
                request.app.state.config = _apply_configured_model_to_config(config, effective_entry)
                return model_info
            if model_entry.provider != "local":
                return current
            active_model_name = _model_info_name(current)
            if active_model_name and _local_model_specs_equivalent(active_model_name, model_entry.model_spec):
                return current
            # local model ??config ??????殉次蹌?active?? runtime ??????鈭????????????            model_info = await _switch_configured_model(request, engine, config, model_entry)
            return model_info
        model_info = await _switch_configured_model(request, engine, config, effective_entry)
        updated = _apply_configured_model_to_config(config, effective_entry)
        request.app.state.config = updated
        return model_info

    if _is_active_model_id(config, model_id):
        return await _active_model_info(engine)

    try:
        model_info = await _maybe_await(engine.switch_model(model_id))
    except (RuntimeError, ValueError) as exc:
        raise _translate_model_switch_error(exc) from exc
    updated = config.model_copy(deep=True)
    updated.model = model_id
    request.app.state.config = updated
    return model_info


@router.post("/models/configure", response_model=ConfigureModelResponse)
async def configure_model(
    request: Request,
    payload: ConfigureModelRequest,
) -> ConfigureModelResponse:
    """??WebGUI ?萄謘???????runtime ???綽?Ｗ????豯止齒 API key??"""
    engine = await _get_or_create_engine(request.app)
    config = await _get_config(request.app)

    if payload.provider == "local":
        normalized_model_spec = _normalize_local_model_spec(payload.model, config)
        try:
            model_info = await _maybe_await(engine.switch_model(normalized_model_spec))
        except (RuntimeError, ValueError) as exc:
            raise _translate_model_switch_error(exc) from exc
        updated = config.model_copy(deep=True)
        updated.model = normalized_model_spec
        updated.model_setup.configured_models = _upsert_configured_model(
            updated.model_setup.configured_models,
            _configured_model_from_parts(
                provider=payload.provider,
                model=Path(normalized_model_spec).name,
                model_spec=normalized_model_spec,
                base_url=None,
                active_model=model_info,
            ),
        )
        request.app.state.config = updated
        persisted_path = _persist_config_if_enabled(request, updated, payload.persist)
        return ConfigureModelResponse(
            provider=payload.provider,
            active_model=_serialize_active_model_info(
                model_info,
                configured_model=_find_active_configured_model(updated),
            ),
            available_models=_serialize_configured_models(updated),
            api_key_configured=False,
            persisted=persisted_path is not None,
            config_path=str(persisted_path) if persisted_path is not None else None,
        )

    if payload.provider == "ollama":
        normalized_model = payload.model.strip()
        normalized_base_url = (payload.base_url or config.ollama.base_url).strip().rstrip("/")
        switch_ollama = getattr(engine, "switch_ollama_backend", None)
        try:
            if callable(switch_ollama):
                model_info = await _maybe_await(
                    switch_ollama(model=normalized_model, base_url=normalized_base_url)
                )
            else:
                model_info = await _maybe_await(engine.switch_model(f"ollama:{normalized_model}"))
        except (RuntimeError, ValueError) as exc:
            raise _translate_model_switch_error(exc) from exc
        updated = config.model_copy(deep=True)
        updated.model = f"ollama:{normalized_model}"
        updated.ollama.base_url = normalized_base_url
        updated.model_setup.configured_models = _upsert_configured_model(
            updated.model_setup.configured_models,
            _configured_model_from_parts(
                provider=payload.provider,
                model=normalized_model,
                model_spec=updated.model,
                base_url=normalized_base_url,
                active_model=model_info,
            ),
        )
        request.app.state.config = updated
        persisted_path = _persist_config_if_enabled(request, updated, payload.persist)
        return ConfigureModelResponse(
            provider=payload.provider,
            active_model=_serialize_active_model_info(
                model_info,
                configured_model=_find_active_configured_model(updated),
            ),
            available_models=_serialize_configured_models(updated),
            api_key_configured=False,
            persisted=persisted_path is not None,
            config_path=str(persisted_path) if persisted_path is not None else None,
        )

    if payload.provider == "openai_codex":
        provider_defaults = _REMOTE_PROVIDER_DEFAULTS[payload.provider]
        try:
            normalized_base_url = normalize_openai_codex_base_url(
                payload.base_url or provider_defaults["base_url"]
            )
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        normalized_model = payload.model.strip() or provider_defaults["model"]
        auth_service = _openai_codex_auth_service(config)
        auth_profile_id = auth_service.resolve_profile_id(
            payload.auth_profile_id or config.openai_codex.auth_profile_id
        )
        if auth_profile_id is None:
            raise HTTPException(
                status_code=400,
                detail="No OpenAI Codex auth profile is available. Import Codex CLI login first.",
            )
        switch_openai_codex = getattr(engine, "switch_openai_codex_backend", None)
        try:
            if callable(switch_openai_codex):
                model_info = await _maybe_await(
                    switch_openai_codex(
                        base_url=normalized_base_url,
                        model=normalized_model,
                        auth_profile_id=auth_profile_id,
                    )
                )
            else:
                model_info = await _maybe_await(engine.switch_model(normalized_base_url))
        except (RuntimeError, ValueError) as exc:
            raise _translate_model_switch_error(exc) from exc

        updated = config.model_copy(deep=True)
        updated.model = normalized_base_url
        updated.openai_codex.base_url = normalized_base_url
        updated.openai_codex.model = normalized_model
        updated.openai_codex.auth_profile_id = auth_profile_id
        updated.model_setup.configured_models = _upsert_configured_model(
            updated.model_setup.configured_models,
            _configured_model_from_parts(
                provider=payload.provider,
                model=normalized_model,
                model_spec=normalized_base_url,
                base_url=normalized_base_url,
                active_model=model_info,
                auth_profile_id=auth_profile_id,
                auth_mode="oauth",
            ),
        )
        request.app.state.config = updated
        persisted_path = _persist_config_if_enabled(request, updated, payload.persist)
        return ConfigureModelResponse(
            provider=payload.provider,
            active_model=_serialize_active_model_info(
                model_info,
                configured_model=_find_active_configured_model(updated),
            ),
            available_models=_serialize_configured_models(updated),
            api_key_configured=False,
            persisted=persisted_path is not None,
            config_path=str(persisted_path) if persisted_path is not None else None,
        )

    if payload.provider == "vllm" and _should_use_managed_vllm_mode(payload):
        managed_model_spec = _normalize_vllm_managed_model_spec(payload.model, config)
        manager = _get_or_create_vllm_runtime_manager(request)
        status = await _start_managed_vllm_runtime(
            manager=manager,
            model_id=None,
            model_spec=managed_model_spec,
            base_url=_managed_vllm_base_url(payload.base_url),
            config=config,
        )
        normalized_base_url = _managed_vllm_base_url(_vllm_runtime_base_url_from_status(status))

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
        try:
            if callable(switch_openai):
                model_info = await _maybe_await(
                    switch_openai(
                        base_url=normalized_base_url,
                        model=managed_model_spec,
                        api_key=effective_api_key,
                        provider=payload.provider,
                    )
                )
            else:
                model_info = await _maybe_await(engine.switch_model(normalized_base_url))
        except (RuntimeError, ValueError) as exc:
            raise _translate_model_switch_error(exc) from exc

        updated = config.model_copy(deep=True)
        updated.model = normalized_base_url
        updated.openai_compat.base_url = normalized_base_url
        updated.openai_compat.model = managed_model_spec
        updated.openai_compat.provider = payload.provider
        if effective_api_key:
            updated.openai_compat.api_key = SecretStr(effective_api_key)
        updated.model_setup.configured_models = _upsert_configured_model(
            updated.model_setup.configured_models,
            _configured_model_from_parts(
                provider=payload.provider,
                model=managed_model_spec,
                model_spec=managed_model_spec,
                base_url=normalized_base_url,
                active_model=model_info,
            ),
        )
        request.app.state.config = updated
        persisted_path = _persist_config_if_enabled(request, updated, payload.persist)
        return ConfigureModelResponse(
            provider=payload.provider,
            active_model=_serialize_active_model_info(
                model_info,
                configured_model=_find_active_configured_model(updated),
            ),
            available_models=_serialize_configured_models(updated),
            api_key_configured=bool(effective_api_key),
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
    try:
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
    except (RuntimeError, ValueError) as exc:
        raise _translate_model_switch_error(exc) from exc

    updated = config.model_copy(deep=True)
    updated.model = normalized_base_url
    updated.openai_compat.base_url = normalized_base_url
    updated.openai_compat.model = normalized_model
    updated.openai_compat.provider = payload.provider
    if effective_api_key:
        updated.openai_compat.api_key = SecretStr(effective_api_key)
    updated.model_setup.configured_models = _upsert_configured_model(
        updated.model_setup.configured_models,
        _configured_model_from_parts(
            provider=payload.provider,
            model=normalized_model,
            model_spec=normalized_base_url,
            base_url=normalized_base_url,
            active_model=model_info,
        ),
    )
    request.app.state.config = updated
    persisted_path = _persist_config_if_enabled(request, updated, payload.persist)

    return ConfigureModelResponse(
        provider=payload.provider,
        active_model=_serialize_active_model_info(
            model_info,
            configured_model=_find_active_configured_model(updated),
        ),
        available_models=_serialize_configured_models(updated),
        api_key_configured=bool(effective_api_key),
        persisted=persisted_path is not None,
        config_path=str(persisted_path) if persisted_path is not None else None,
    )


@router.get("/models/ollama", response_model=OllamaModelsResponse)
async def list_ollama_models(
    base_url: str = Query(default="http://localhost:11434", min_length=1),
) -> OllamaModelsResponse:
    """???Ollama `/api/tags`?? WebGUI ??????選???謜?????閰制???"""
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


async def _load_active_model_info(
    request: Request,
    *,
    configured_model: ConfiguredModelConfig | None = None,
) -> dict[str, Any] | None:
    """?????engine ?謘????????????????????None??"""
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
        return _serialize_active_model_info(info, configured_model=configured_model)

    return None


def _openai_codex_auth_service(config: MochiConfig) -> OpenAICodexAuthService:
    return OpenAICodexAuthService(config.workspace_dir)


def _is_openai_codex_remote_config(config: MochiConfig) -> bool:
    if not config.model.startswith(("http://", "https://")):
        return False
    try:
        normalized_base_url = normalize_openai_codex_base_url(config.openai_codex.base_url)
    except ValueError:
        return False
    if config.model.rstrip("/") != normalized_base_url:
        return False
    auth_service = _openai_codex_auth_service(config)
    return auth_service.resolve_profile_id(config.openai_codex.auth_profile_id) is not None


def _active_remote_provider_from_config(config: MochiConfig) -> str | None:
    if not config.model.startswith(("http://", "https://")):
        return None
    if _is_openai_codex_remote_config(config):
        return "openai_codex"
    return config.openai_compat.provider


def _configured_model_from_parts(
    *,
    provider: ModelProvider,
    model: str,
    model_spec: str,
    base_url: str | None,
    active_model: Any,
    launch_mode: Literal["external", "managed"] | None = None,
    auth_profile_id: str | None = None,
    auth_mode: Literal["none", "api_key", "oauth"] | None = None,
) -> ConfiguredModelConfig:
    """????賹??謜? runtime ?桀???梁???豯??賹?????獢???朝?"""
    backend_type = _model_info_backend_type(active_model) or (
        "ollama"
        if provider == "ollama"
        else (
            "gguf"
            if model_spec.lower().endswith(".gguf")
            else "safetensors"
        )
        if provider == "local"
        else ("openai_codex" if provider == "openai_codex" else "openai_compat")
    )
    model_id = (
        model_spec
        if provider in {"ollama", "local"}
        else f"{provider}:{(base_url or model_spec)}:{model}"
    )
    label = model if provider in {"ollama", "local"} else f"{model} ({provider})"
    resolved_launch_mode = launch_mode
    if provider == "vllm" and resolved_launch_mode is None:
        resolved_launch_mode = "external" if _is_http_endpoint(model_spec) else "managed"
    if provider in {"sglang", "tensorrt_llm"} and resolved_launch_mode is None:
        resolved_launch_mode = "external"
    return ConfiguredModelConfig(
        id=model_id,
        provider=provider,
        model=model,
        model_spec=model_spec,
        base_url=base_url,
        label=label,
        backend_type=backend_type,
        launch_mode=resolved_launch_mode,
        auth_profile_id=auth_profile_id,
        auth_mode=auth_mode,
    )


def _model_info_backend_type(info: Any) -> str | None:
    """??ModelInfo-like ??麾?謘? backend_type??"""
    if isinstance(info, dict):
        value = info.get("backend_type")
        return value if isinstance(value, str) and value else None
    value = getattr(info, "backend_type", None)
    return value if isinstance(value, str) and value else None


def _model_info_name(info: Any) -> str | None:
    """??ModelInfo-like ??麾?謘????????"""
    if isinstance(info, dict):
        value = info.get("name")
        return value if isinstance(value, str) and value else None
    value = getattr(info, "name", None)
    return value if isinstance(value, str) and value else None


def _upsert_configured_model(
    models: list[ConfiguredModelConfig],
    model: ConfiguredModelConfig,
) -> list[ConfiguredModelConfig]:
    """?皝????謘?????擗??賹澈?堊垓?????????????"""
    next_models = [
        item
        for item in models
        if not _configured_models_equivalent(item, model)
    ]
    return [model, *next_models]


def _find_configured_model(
    config: MochiConfig,
    model_id: str,
) -> ConfiguredModelConfig | None:
    """??id ?謘踐???model_spec ???頨急謍梯????"""
    for model in config.model_setup.configured_models:
        if _configured_model_matches_identifier(model, model_id):
            return model
        if model.provider == "ollama" and model.model == model_id:
            return model
    return None


def _find_active_configured_model(config: MochiConfig) -> ConfiguredModelConfig | None:
    """?豯止齒?獢? runtime config ?????configured model??"""
    for model in config.model_setup.configured_models:
        if _is_configured_model_active(config, model):
            return model
    return None


def _is_active_model_id(config: MochiConfig, model_id: str) -> bool:
    """??? request model id ??秋?????獢? config.model??"""
    if model_id == config.model:
        return True
    return config.model.startswith("ollama:") and model_id == config.model.removeprefix("ollama:")


def _is_configured_model_active(config: MochiConfig, model: ConfiguredModelConfig) -> bool:
    """????踐??????????澗??銵甇?????runtime config??"""
    if model.provider == "ollama":
        return (
            config.model == model.model_spec
            and config.ollama.base_url.rstrip("/") == (model.base_url or config.ollama.base_url).rstrip("/")
        )
    if model.provider == "local":
        return _local_model_specs_equivalent(config.model, model.model_spec)
    if model.provider == "openai_codex":
        expected_base_url = (model.base_url or model.model_spec).rstrip("/")
        return (
            config.model.rstrip("/") == expected_base_url
            and config.openai_codex.base_url.rstrip("/") == expected_base_url
            and config.openai_codex.model == model.model
            and config.openai_codex.auth_profile_id == model.auth_profile_id
        )
    expected_base_url = (model.base_url or model.model_spec).rstrip("/")
    expected_model_spec = model.model_spec
    if _is_managed_vllm_configured_model(model):
        expected_model_spec = _managed_vllm_base_url(model.base_url)
        expected_base_url = _managed_vllm_base_url(model.base_url).rstrip("/")
    return (
        config.model.rstrip("/") == expected_model_spec.rstrip("/")
        and config.openai_compat.provider == model.provider
        and config.openai_compat.model == model.model
        and config.openai_compat.base_url.rstrip("/") == expected_base_url
    )


async def _active_model_info(engine: Any) -> Any:
    """?謘??獢? active model info????sync/async engine stub??"""
    get_model_info = getattr(engine, "get_model_info", None)
    if callable(get_model_info):
        return await _maybe_await(get_model_info())
    raise RuntimeError("Engine does not provide get_model_info().")


async def _switch_configured_model(
    request: Request,
    engine: Any,
    config: MochiConfig,
    model: ConfiguredModelConfig,
) -> Any:
    """??甇豲????????獢???runtime backend??"""
    if model.provider == "ollama":
        switch_ollama = getattr(engine, "switch_ollama_backend", None)
        if callable(switch_ollama):
            return await _maybe_await(
                switch_ollama(model=model.model, base_url=model.base_url or config.ollama.base_url)
            )
        return await _maybe_await(engine.switch_model(model.model_spec))

    if model.provider == "local":
        return await _maybe_await(engine.switch_model(model.model_spec))

    if model.provider == "openai_codex":
        switch_openai_codex = getattr(engine, "switch_openai_codex_backend", None)
        if callable(switch_openai_codex):
            return await _maybe_await(
                switch_openai_codex(
                    base_url=model.base_url or model.model_spec,
                    model=model.model,
                    auth_profile_id=model.auth_profile_id or config.openai_codex.auth_profile_id,
                )
            )
        return await _maybe_await(engine.switch_model(model.base_url or model.model_spec))

    effective_model = model.model
    effective_base_url = model.base_url or model.model_spec
    if _is_managed_vllm_configured_model(model):
        managed_model_spec = _resolve_vllm_managed_model_spec(model, config)
        status = await _start_managed_vllm_runtime(
            manager=_get_or_create_vllm_runtime_manager(request),
            model_id=model.id,
            model_spec=managed_model_spec,
            base_url=_managed_vllm_base_url(model.base_url),
            config=config,
        )
        effective_model = managed_model_spec
        effective_base_url = _managed_vllm_base_url(_vllm_runtime_base_url_from_status(status))

    switch_openai = getattr(engine, "switch_openai_compat_backend", None)
    if callable(switch_openai):
        existing_key = (
            config.openai_compat.api_key.get_secret_value()
            if config.openai_compat.provider == model.provider
            and config.openai_compat.base_url.rstrip("/") == _managed_vllm_base_url(effective_base_url).rstrip("/")
            and config.openai_compat.api_key is not None
            else ""
        )
        return await _maybe_await(
            switch_openai(
                base_url=effective_base_url,
                model=effective_model,
                api_key=existing_key,
                provider=model.provider,
            )
        )
    return await _maybe_await(engine.switch_model(effective_base_url))


def _apply_configured_model_to_config(
    config: MochiConfig,
    model: ConfiguredModelConfig,
) -> MochiConfig:
    """?甇?????鞊???駁???runtime config??"""
    updated = config.model_copy(deep=True)
    updated.model = model.model_spec
    if model.provider == "ollama":
        if model.base_url:
            updated.ollama.base_url = model.base_url.rstrip("/")
        return updated
    if model.provider == "local":
        return updated
    if model.provider == "openai_codex":
        updated.model = (model.base_url or model.model_spec).rstrip("/")
        updated.openai_codex.base_url = (model.base_url or model.model_spec).rstrip("/")
        updated.openai_codex.model = model.model
        updated.openai_codex.auth_profile_id = model.auth_profile_id
        return updated

    if _is_managed_vllm_configured_model(model):
        updated.model = _managed_vllm_base_url(model.base_url)
    else:
        updated.model = model.model_spec

    updated.openai_compat.provider = model.provider
    updated.openai_compat.base_url = _managed_vllm_base_url(model.base_url or model.model_spec)
    updated.openai_compat.model = model.model
    return updated


def _serialize_configured_models(config: MochiConfig) -> list[dict[str, Any]]:
    """?豯止齒??? UI ????閰制??輯撒??蹓?????secret ????????朝?"""
    models = list(config.model_setup.configured_models)
    if not models or _find_active_configured_model(config) is None:
        models = _upsert_configured_model(
            models,
            _configured_model_from_config(config),
        )
    return [model.model_dump(exclude_none=True) for model in models]


def _dump_saved_configured_models(config: MochiConfig) -> list[dict[str, Any]]:
    """?豯止齒???踐????config.model_setup ???????畾???? runtime fallback??"""
    return [model.model_dump(exclude_none=True) for model in config.model_setup.configured_models]


def _configured_model_from_config(config: MochiConfig) -> ConfiguredModelConfig:
    """?綜等謑????? config.model ?高?暸??????綽謆撖??獢???朝?"""
    if config.model.startswith("ollama:"):
        model = config.model.removeprefix("ollama:")
        return ConfiguredModelConfig(
            id=config.model,
            provider="ollama",
            model=model,
            model_spec=config.model,
            base_url=config.ollama.base_url,
            label=model,
            backend_type="ollama",
        )
    if config.model.startswith(("http://", "https://")):
        provider = _active_remote_provider_from_config(config) or config.openai_compat.provider
        if provider == "openai_codex":
            return ConfiguredModelConfig(
                id=f"openai_codex:{config.openai_codex.base_url.rstrip('/')}:{config.openai_codex.model}",
                provider="openai_codex",
                model=config.openai_codex.model,
                model_spec=config.openai_codex.base_url.rstrip("/"),
                base_url=config.openai_codex.base_url.rstrip("/"),
                label=f"{config.openai_codex.model} (openai_codex)",
                backend_type="openai_codex",
                auth_profile_id=config.openai_codex.auth_profile_id,
                auth_mode="oauth",
            )
        return ConfiguredModelConfig(
            id=f"{provider}:{config.openai_compat.base_url.rstrip('/')}:{config.openai_compat.model}",
            provider=provider,
            model=config.openai_compat.model,
            model_spec=config.openai_compat.base_url.rstrip("/"),
            base_url=config.openai_compat.base_url.rstrip("/"),
            label=f"{config.openai_compat.model} ({provider})",
            backend_type="openai_compat",
            auth_mode="api_key",
        )
    backend_type = (
        "gguf"
        if config.model.lower().endswith(".gguf")
        else "safetensors"
    )
    return ConfiguredModelConfig(
        id=config.model,
        provider="local",
        model=config.model,
        model_spec=config.model,
        label=config.model,
        backend_type=backend_type,
    )


def _serialize_model_info(info: Any) -> dict[str, Any]:
    """??ModelInfo-like ??麾?改? JSON-safe dict??"""
    if is_dataclass(info):
        payload = asdict(info)
        return jsonable_encoder({key: value for key, value in payload.items() if value is not None})
    if hasattr(info, "model_dump"):
        payload = info.model_dump()
        return jsonable_encoder({key: value for key, value in payload.items() if value is not None})
    if isinstance(info, dict):
        return jsonable_encoder({key: value for key, value in info.items() if value is not None})
    return jsonable_encoder(
        {
            "name": getattr(info, "name", ""),
            "backend_type": getattr(info, "backend_type", ""),
            "context_length": getattr(info, "context_length", None),
            "supports_tool_calling": getattr(info, "supports_tool_calling", None),
            "metadata": getattr(info, "metadata", {}),
        }
    )


def _serialize_active_model_info(
    info: Any,
    *,
    configured_model: ConfiguredModelConfig | None = None,
) -> dict[str, Any]:
    """Attach configured model identity to active backend snapshots when available."""
    payload = _serialize_model_info(info)
    if configured_model is None:
        return payload

    payload["id"] = configured_model.id
    payload["provider"] = configured_model.provider
    payload["model_spec"] = configured_model.model_spec
    payload["base_url"] = configured_model.base_url
    payload["auth_profile_id"] = configured_model.auth_profile_id
    payload["auth_mode"] = configured_model.auth_mode
    payload["label"] = configured_model.label
    payload.setdefault("name", configured_model.model)
    payload.setdefault("backend_type", configured_model.backend_type)
    return payload


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


def _should_use_managed_vllm_mode(payload: ConfigureModelRequest) -> bool:
    if payload.provider != "vllm":
        return False
    if payload.base_url is not None and payload.base_url.strip():
        return False
    return shared_is_possible_managed_vllm_target(payload.model.strip())


def _is_managed_vllm_configured_model(model: ConfiguredModelConfig) -> bool:
    return model.provider == "vllm" and shared_is_managed_vllm_configured_model(model)


def _configured_vllm_launch_mode(model: ConfiguredModelConfig) -> str | None:
    return shared_configured_vllm_launch_mode(model)


def _resolve_vllm_managed_model_spec(model: ConfiguredModelConfig, config: MochiConfig) -> str:
    return shared_resolve_vllm_managed_model_spec(
        model,
        config,
        error_factory=lambda detail, status_code: HTTPException(
            status_code=status_code or 400,
            detail=detail,
        ),
    )


def _normalize_vllm_managed_model_spec(model_spec: str, config: MochiConfig) -> str:
    return shared_normalize_vllm_managed_model_spec(
        model_spec,
        config,
        error_factory=lambda detail, status_code: HTTPException(
            status_code=status_code or 400,
            detail=detail,
        ),
    )


def _is_http_endpoint(value: str) -> bool:
    return shared_is_http_endpoint(value)


def _is_local_path_candidate(value: str) -> bool:
    return shared_is_local_path_candidate(value)


def _managed_vllm_base_url(base_url: str | None) -> str:
    return shared_managed_vllm_base_url(base_url)


def _vllm_runtime_base_url_from_status(status: dict[str, Any]) -> str | None:
    value = status.get("base_url")
    return value if isinstance(value, str) and value.strip() else None


def _get_or_create_vllm_runtime_manager(request: Request) -> Any:
    manager = getattr(request.app.state, "vllm_runtime_manager", None)
    if manager is not None:
        return manager
    manager = ManagedVLLMRuntimeManager()
    request.app.state.vllm_runtime_manager = manager
    return manager


async def _query_vllm_runtime_status(*, manager: Any, config: MochiConfig) -> dict[str, Any]:
    status = getattr(manager, "status", None)
    if not callable(status):
        return {
            "state": "stopped",
            "running": False,
            "launch_mode": "managed",
            "active_model_id": None,
            "active_model_spec": None,
            "base_url": _managed_vllm_base_url(config.openai_compat.base_url),
            "message": "vLLM runtime manager is unavailable.",
        }
    payload = await _maybe_await(_call_with_supported_kwargs(status, config=config))
    if isinstance(payload, dict):
        launch_mode_value = payload.get("launch_mode")
        return {
            "state": str(payload.get("state", "running" if payload.get("running") else "stopped")),
            "running": bool(payload.get("running", payload.get("state") == "running")),
            "launch_mode": (
                launch_mode_value.strip().lower()
                if isinstance(launch_mode_value, str) and launch_mode_value.strip()
                else "managed"
            ),
            "active_model_id": payload.get("active_model_id"),
            "active_model_spec": payload.get("active_model_spec"),
            "base_url": _managed_vllm_base_url(
                payload.get("base_url")
                if isinstance(payload.get("base_url"), str)
                else config.openai_compat.base_url
            ),
            "message": payload.get("message"),
        }
    return {
        "state": "stopped",
        "running": False,
        "launch_mode": "managed",
        "active_model_id": None,
        "active_model_spec": None,
        "base_url": _managed_vllm_base_url(config.openai_compat.base_url),
        "message": "vLLM runtime manager returned invalid status payload.",
    }


async def _start_managed_vllm_runtime(
    *,
    manager: Any,
    model_id: str | None,
    model_spec: str,
    base_url: str,
    config: MochiConfig | None = None,
) -> dict[str, Any]:
    start = getattr(manager, "start", None)
    if not callable(start):
        raise HTTPException(status_code=503, detail="vLLM runtime manager does not support start().")
    try:
        payload = await _maybe_await(
            _call_with_supported_kwargs(
                start,
                model_id=model_id,
                model_spec=model_spec,
                base_url=base_url,
                launch_mode="managed",
                config=config,
            )
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if isinstance(payload, dict):
        launch_mode_value = payload.get("launch_mode")
        return {
            "state": str(payload.get("state", "running")),
            "running": bool(payload.get("running", True)),
            "launch_mode": (
                launch_mode_value.strip().lower()
                if isinstance(launch_mode_value, str) and launch_mode_value.strip()
                else "managed"
            ),
            "active_model_id": payload.get("active_model_id") or model_id,
            "active_model_spec": payload.get("active_model_spec") or model_spec,
            "base_url": _managed_vllm_base_url(
                payload.get("base_url") if isinstance(payload.get("base_url"), str) else base_url
            ),
            "message": payload.get("message"),
        }
    return {
        "state": "running",
        "running": True,
        "launch_mode": "managed",
        "active_model_id": model_id,
        "active_model_spec": model_spec,
        "base_url": _managed_vllm_base_url(base_url),
        "message": "Managed vLLM runtime started.",
    }


async def _stop_managed_vllm_runtime(*, manager: Any) -> None:
    stop = getattr(manager, "stop", None)
    if not callable(stop):
        raise HTTPException(status_code=503, detail="vLLM runtime manager does not support stop().")
    try:
        await _maybe_await(_call_with_supported_kwargs(stop))
    except HTTPException:
        raise
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def _serialize_vllm_runtime_status(payload: dict[str, Any]) -> VLLMRuntimeStatusResponse:
    return VLLMRuntimeStatusResponse(
        state=str(payload.get("state", "stopped")),
        running=bool(payload.get("running", False)),
        launch_mode=payload.get("launch_mode"),
        active_model_id=payload.get("active_model_id"),
        active_model_spec=payload.get("active_model_spec"),
        base_url=payload.get("base_url"),
        message=payload.get("message"),
    )


def _normalize_local_root(root: str, config: MochiConfig) -> str:
    """?????local ????撖抆????????allowlist ?踐????"""
    normalized = Path(root.strip()).expanduser().resolve(strict=False)
    if not normalized.is_absolute():
        normalized = normalized.resolve(strict=False)
    _ensure_local_path_allowed(normalized, config)
    return str(normalized)


def _normalize_local_model_spec(model_spec: str, config: MochiConfig) -> str:
    """?????local model path??謅??蝞?????荒????"""
    normalized_path = Path(model_spec.strip()).expanduser().resolve(strict=False)
    if not normalized_path.is_absolute():
        normalized_path = normalized_path.resolve(strict=False)
    _ensure_local_path_allowed(normalized_path, config)

    if not normalized_path.exists():
        raise HTTPException(status_code=404, detail=f"Local model path does not exist: {normalized_path}")
    if normalized_path.is_symlink():
        raise HTTPException(status_code=400, detail="Symlink model paths are not supported.")

    if normalized_path.is_file():
        if normalized_path.suffix.lower() != ".gguf":
            raise HTTPException(
                status_code=400,
                detail="Local model file must use .gguf extension.",
            )
        return str(normalized_path)

    if normalized_path.is_dir():
        if not _is_hf_safetensors_dir(normalized_path):
            raise HTTPException(
                status_code=400,
                detail=(
                    "Local model directory is not a valid HuggingFace safetensors directory. "
                    "Expected config.json, tokenizer files, and safetensors weights."
                ),
            )
        return str(normalized_path)

    raise HTTPException(status_code=400, detail="Local model path must be a file or directory.")


def _configured_model_matches_identifier(
    model: ConfiguredModelConfig,
    model_id: str,
) -> bool:
    """???????梱???秋??撖? API ??喉?朱?瞏汕?"""
    if model.id == model_id or model.model_spec == model_id:
        return True
    if model.provider != "local":
        return False
    return (
        _local_model_specs_equivalent(model.id, model_id)
        or _local_model_specs_equivalent(model.model_spec, model_id)
    )


def _configured_models_equivalent(
    left: ConfiguredModelConfig,
    right: ConfiguredModelConfig,
) -> bool:
    """??????configured model ??秋??????????????"""
    if left.provider != right.provider:
        return False
    if left.provider == "local":
        return _local_model_specs_equivalent(left.model_spec, right.model_spec)
    return (
        left.id == right.id
        or (
            left.model == right.model
            and left.model_spec == right.model_spec
            and (left.base_url or "") == (right.base_url or "")
            and (left.auth_profile_id or "") == (right.auth_profile_id or "")
        )
    )


def _local_model_specs_equivalent(left: str, right: str) -> bool:
    """?????秘??????秋??蝞????Windows/WSL ?????"""
    left_key = _local_model_compare_key(left)
    right_key = _local_model_compare_key(right)
    return bool(left_key) and left_key == right_key


def _local_model_compare_key(value: str) -> str:
    """?謓??唾?????綜????伍???????key??"""
    raw = value.strip()
    if not raw:
        return ""

    windows_match = _WINDOWS_ABSOLUTE_PATH_RE.match(raw)
    if windows_match:
        drive = windows_match.group(1).lower()
        remainder = _normalize_windows_relative_path(windows_match.group(2), lower_case=True)
        return f"windows-drive:{drive}/{remainder}" if remainder else f"windows-drive:{drive}"

    posix_like = raw.replace("\\", "/")
    wsl_match = _WSL_MOUNT_PATH_RE.match(posix_like)
    if wsl_match:
        drive = wsl_match.group(1).lower()
        remainder = _normalize_windows_relative_path(wsl_match.group(2) or "", lower_case=True)
        return f"windows-drive:{drive}/{remainder}" if remainder else f"windows-drive:{drive}"

    if raw.startswith("~"):
        expanded = str(Path(raw).expanduser())
        normalized = expanded.replace("\\", "/")
    else:
        normalized = posix_like
    normalized = re.sub(r"/+", "/", normalized)
    if len(normalized) > 1:
        normalized = normalized.rstrip("/")
    return normalized


def _normalize_windows_relative_path(value: str, *, lower_case: bool) -> str:
    """?????Windows/WSL drive ????閰????曇??"""
    normalized = re.sub(r"/+", "/", value.replace("\\", "/").strip("/"))
    return normalized.casefold() if lower_case else normalized


def _normalize_local_capability_target(model_spec: str, config: MochiConfig) -> str:
    """?????local capability target path???????HF ?荒???踐????"""
    normalized_path = Path(model_spec.strip()).expanduser().resolve(strict=False)
    if not normalized_path.is_absolute():
        normalized_path = normalized_path.resolve(strict=False)
    _ensure_local_path_allowed(normalized_path, config)

    if not normalized_path.exists():
        raise HTTPException(status_code=404, detail=f"Local model path does not exist: {normalized_path}")
    if normalized_path.is_symlink():
        raise HTTPException(status_code=400, detail="Symlink model paths are not supported.")
    if not (normalized_path.is_file() or normalized_path.is_dir()):
        raise HTTPException(status_code=400, detail="Local model path must be a file or directory.")

    return str(normalized_path)


def _ensure_local_path_allowed(path: Path, config: MochiConfig) -> None:
    """??local allowlist ?殉朱謓?蹇???? path ?對??選?謘?roots ?????"""
    ensure_local_path_allowed(
        path,
        config,
        error_factory=lambda detail, status_code: HTTPException(
            status_code=status_code or 400,
            detail=detail,
        ),
    )


def _translate_model_switch_error(exc: RuntimeError | ValueError) -> HTTPException:
    """??runtime model switch ?剜???改蹌剛??? API ??芰???"""
    detail = str(exc).strip() or "Model backend is unavailable."
    if isinstance(exc, ValueError):
        return HTTPException(status_code=400, detail=detail)
    return HTTPException(status_code=503, detail=detail)


async def _get_local_model_converter(request: Request) -> BaseLocalModelConverter:
    """?謘???秘???改??????? app.state ??舫?????"""
    injected_converter = getattr(request.app.state, "local_model_converter", None)
    if injected_converter is not None:
        return injected_converter

    factory = getattr(request.app.state, "local_model_converter_factory", None)
    if callable(factory):
        created = await _maybe_await(factory())
        if created is not None:
            request.app.state.local_model_converter = created
            return created

    config = await _get_config(request.app)
    llama_cpp = config.local_models.llama_cpp
    converter = LlamaCppLocalModelConverter(
        toolchain=None,
        env=None,
        cwd=Path.cwd(),
    )
    if llama_cpp.root_dir is not None or llama_cpp.python_executable or llama_cpp.version:
        from mochi.backends.local_models import discover_llama_cpp_toolchain

        toolchain = discover_llama_cpp_toolchain(
            managed_root=_managed_runtime_base_dir(config),
            preferred_root=llama_cpp.root_dir,
            preferred_python=llama_cpp.python_executable,
            preferred_version=llama_cpp.version,
            preferred_source=llama_cpp.source,
        )
        converter = LlamaCppLocalModelConverter(toolchain=toolchain)
    request.app.state.local_model_converter = converter
    return converter


def _managed_runtime_base_dir(config: MochiConfig) -> Path:
    """managed runtime metadata ????賃??制璆?怏?"""
    workspace = Path(config.workspace_dir).expanduser().resolve(strict=False)
    return workspace


def _discover_local_runtime_status(request: Request, config: MochiConfig) -> Any:
    """?????llama.cpp runtime discovery??"""
    llama_cpp = config.local_models.llama_cpp
    return get_managed_llama_cpp_runtime_status(
        cwd=Path.cwd(),
        managed_root=_managed_runtime_base_dir(config),
        preferred_root=llama_cpp.root_dir,
        preferred_python=llama_cpp.python_executable,
        preferred_version=llama_cpp.version,
        preferred_source=llama_cpp.source,
    )


def _serialize_local_runtime_status(status: Any) -> LocalModelRuntimeStatusResponse:
    """?制???runtime status??"""
    return LocalModelRuntimeStatusResponse(
        readiness=status.state,
        installed=status.installed,
        source=status.source,
        root_dir=status.root_dir,
        install_dir=status.install_dir,
        python_executable=status.python_executable,
        version=status.version,
        platform=status.platform,
        binary_asset=status.binary_asset,
        convert_script=status.convert_script,
        quantize_binary=status.quantize_binary,
        missing_components=list(status.missing_components),
        warnings=list(status.warnings),
        actions=list(status.actions),
        hardware=_serialize_hardware_summary(_detect_hardware_summary()),
    )


def _serialize_active_local_model_runtime(info: Any) -> LocalActiveModelRuntimeStatusResponse:
    """?制??謘橫???active ??秧??啾????????鈭????"""
    backend_type = _model_info_backend_type(info)
    model_name = _model_info_name(info)
    metadata = info.get("metadata", {}) if isinstance(info, dict) else getattr(info, "metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}

    is_local = backend_type in {"gguf", "safetensors"}
    loaded = bool(metadata.get("loaded")) if is_local else False
    idle_unloaded = bool(metadata.get("idle_unloaded")) if is_local else False

    return LocalActiveModelRuntimeStatusResponse(
        has_active_local_model=is_local,
        model_spec=model_name if is_local else None,
        backend_type=backend_type if is_local else None,
        loaded=loaded,
        idle_unloaded=idle_unloaded,
        can_unload=is_local,
    )


def _serialize_capability_format(item: Any) -> dict[str, Any]:
    """?制??謖??謘踐僱???謜??謕?"""
    options = [
        {
            "id": option.id,
            "name": option.name,
            "bits": option.bits,
            "description": option.description,
        }
        for option in getattr(item, "quantization_options", [])
    ]
    return {
        "format_id": item.format_id,
        "format_name": item.format_name,
        "supported": item.supported,
        "priority": item.priority,
        "reason": item.reason,
        "warnings": list(item.warnings),
        "quantization_options": options,
        "suggested_default_quantization": item.suggested_default_quantization,
    }


def _serialize_hardware_summary(summary: Any) -> dict[str, Any] | None:
    """?制??謘撾脫????秋撕??"""
    if summary is None:
        return None
    return {
        "provider": summary.provider,
        "cuda_available": summary.cuda_available,
        "gpu_count": summary.gpu_count,
        "gpu_vendor": getattr(summary, "gpu_vendor", None),
        "primary_gpu_name": summary.primary_gpu_name,
        "total_vram_gb": summary.total_vram_gb,
        "recommended_runtime_backend": getattr(summary, "recommended_runtime_backend", None),
        "recommended_runtime_label": getattr(summary, "recommended_runtime_label", None),
        "warnings": list(summary.warnings),
    }


def _is_hf_safetensors_dir(path: Path) -> bool:
    """?踐????秋???HuggingFace safetensors ?獢???"""
    return is_hf_safetensors_dir(path)
