"""vLLM managed runtime shared helpers."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from mochi.config.schema import ConfiguredModelConfig, MochiConfig

DEFAULT_MANAGED_VLLM_BASE_URL = "http://localhost:8000/v1"

_WINDOWS_ABSOLUTE_PATH_RE = re.compile(r"^([A-Za-z]):[\\/]*(.*)$")
_HF_REPO_ID_RE = re.compile(
    r"^[A-Za-z0-9][A-Za-z0-9._-]*(/[A-Za-z0-9][A-Za-z0-9._-]*)?(?:@[A-Za-z0-9._-]+)?$"
)

ManagedVLLMErrorFactory = Callable[[str, int | None], Exception]


def managed_vllm_base_url(base_url: str | None) -> str:
    """回傳 managed vLLM OpenAI-compatible base URL。"""
    return (base_url or DEFAULT_MANAGED_VLLM_BASE_URL).strip().rstrip("/")


def is_http_endpoint(value: str) -> bool:
    """判斷字串是否為 HTTP(S) endpoint。"""
    return value.strip().startswith(("http://", "https://"))


def is_local_path_candidate(value: str) -> bool:
    """判斷字串是否可能是本地模型路徑。"""
    raw = value.strip()
    if not raw:
        return False
    if raw.startswith(("~", "./", "../", "/", "\\")):
        return True
    if raw.endswith(("/", "\\")):
        return True
    if _WINDOWS_ABSOLUTE_PATH_RE.match(raw):
        return True
    return raw.startswith("file:")


def configured_vllm_launch_mode(model: ConfiguredModelConfig) -> str | None:
    """解析 configured model 的 vLLM launch mode。"""
    if model.provider != "vllm":
        return None
    explicit_mode = getattr(model, "launch_mode", None)
    if isinstance(explicit_mode, str) and explicit_mode.strip():
        normalized_mode = explicit_mode.strip().lower()
        if normalized_mode == "remote":
            return "external"
        if normalized_mode in {"external", "managed"}:
            return normalized_mode
    if is_http_endpoint(model.model_spec):
        return "external"
    return "managed"


def is_managed_vllm_configured_model(model: ConfiguredModelConfig) -> bool:
    """判斷 configured model 是否為 managed vLLM。"""
    return configured_vllm_launch_mode(model) == "managed"


def is_possible_managed_vllm_target(model_spec: str) -> bool:
    """判斷 configure payload 是否像是 managed vLLM target。"""
    raw = model_spec.strip()
    if not raw or is_http_endpoint(raw):
        return False
    if raw.lower().endswith(".gguf"):
        return True
    if is_local_path_candidate(raw):
        return True
    return "/" in raw


def resolve_vllm_managed_model_spec(
    model: ConfiguredModelConfig,
    config: MochiConfig,
    *,
    error_factory: ManagedVLLMErrorFactory,
) -> str:
    """將 configured vLLM model 解析為 managed runtime 可接受的 model spec。"""
    if model.provider != "vllm":
        raise error_factory("Only provider=vllm supports managed vLLM runtime.", 400)
    if configured_vllm_launch_mode(model) != "managed":
        raise error_factory("Managed vLLM runtime requires launch_mode=managed.", 400)
    raw_spec = model.model_spec if not is_http_endpoint(model.model_spec) else model.model
    return normalize_vllm_managed_model_spec(raw_spec, config, error_factory=error_factory)


def normalize_vllm_managed_model_spec(
    model_spec: str,
    config: MochiConfig,
    *,
    error_factory: ManagedVLLMErrorFactory,
) -> str:
    """驗證並正規化 managed vLLM model spec。"""
    raw = model_spec.strip()
    if not raw:
        raise error_factory("Managed vLLM model spec is required.", 400)
    if raw.lower().endswith(".gguf"):
        raise error_factory("Managed vLLM mode does not support .gguf model specs.", 400)
    if is_http_endpoint(raw):
        raise error_factory(
            "Managed vLLM model spec must be a HuggingFace repo id or local safetensors directory.",
            400,
        )

    if is_local_path_candidate(raw):
        normalized_path = Path(raw).expanduser().resolve(strict=False)
        if not normalized_path.is_absolute():
            normalized_path = normalized_path.resolve(strict=False)
        ensure_local_path_allowed(normalized_path, config, error_factory=error_factory)

        if not normalized_path.exists():
            raise error_factory(f"Local model path does not exist: {normalized_path}", 404)
        if normalized_path.is_symlink():
            raise error_factory("Symlink model paths are not supported.", 400)
        if not normalized_path.is_dir():
            raise error_factory("Managed vLLM local model path must be a directory.", 400)
        if not is_hf_safetensors_dir(normalized_path):
            raise error_factory(
                (
                    "Managed vLLM local directory is not a valid HuggingFace safetensors directory. "
                    "Expected config.json, tokenizer files, and safetensors weights."
                ),
                400,
            )
        return str(normalized_path)

    if _HF_REPO_ID_RE.match(raw):
        return raw

    raise error_factory(
        "Managed vLLM model spec must be a HuggingFace repo id or local safetensors directory.",
        400,
    )


def ensure_local_path_allowed(
    path: Path,
    config: MochiConfig,
    *,
    error_factory: ManagedVLLMErrorFactory,
) -> None:
    """驗證本地模型路徑是否落在允許的 roots 內。"""
    roots = [Path(root).expanduser().resolve(strict=False) for root in config.local_models.roots]
    if not roots:
        return
    if any(path.is_relative_to(root) for root in roots):
        return
    raise error_factory(f"Path is outside configured local model roots: {path}", 403)


def is_hf_safetensors_dir(path: Path) -> bool:
    """判斷目錄是否為 HuggingFace safetensors 模型目錄。"""
    if not path.is_dir():
        return False
    if not (path / "config.json").is_file():
        return False
    has_tokenizer = any(
        (path / name).is_file()
        for name in ("tokenizer.json", "tokenizer.model", "tokenizer_config.json")
    )
    if not has_tokenizer:
        return False
    if (path / "model.safetensors.index.json").is_file():
        return True
    return any(
        child.is_file() and child.suffix.lower() == ".safetensors"
        for child in path.iterdir()
        if not child.is_symlink()
    )
